"""Build each task's verifier image: the upstream image plus test.sh's installs, inert.

Run on a build host with Docker, network and our registry at `--registry` (never on a
gateway). Per task:

1. a container of the upstream image (by digest), Docker's default runtime, with network;
2. `tests/` staged at `/tests` with every `test_*.py` stubbed (one passing test: the
   tests' own side effects, servers, scratch files, never reach the overlay), `test.sh` run
   as Harbor runs it (root, from the image's WORKDIR): its installs (curl, uv, Python,
   pytest and the test dependencies, clones, downloads) land;
3. `docker diff`: every added or changed path, less `/tests`, `/logs`, `/tmp`, caches and
   logs, and the WORKDIR (what the tests ran on, except the few downloads `test.sh` puts
   there, `WORKDIR_KEEP`), tarred to `overlay.tar.gz`, split in parts of at most
   `images.PART_BYTES` (a registry layer each: Hippius times out on 3 GB layers);
4. `FROM <upstream>@<digest>` + one `COPY` per part into `/opt/tb-verifier/` + a file in
   `/tmp` (`images.TMP_KEEP`), pushed to the registry as `images.image_ref(task)`; its
   digest goes to the lock (`--write-lock`).

Output: one JSON line per task (digest, overlay bytes and parts, paths, seconds).

    uv run python scripts/build_verifier_images.py --registry 127.0.0.1:5001 --jobs 6 \
        --out build.jsonl
    uv run python scripts/build_verifier_images.py --write-lock --from build.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tempfile
import time
from pathlib import Path

from reliquary_terminal_bench import images, source

PACKAGE = Path(images.__file__).resolve().parent
EXCLUDED_PREFIXES = ("/tests", "/logs", "/tmp", "/proc", "/sys", "/dev", "/run",
                     "/var/cache/apt", "/var/lib/apt/lists", "/var/log", "/root/.cache/pip",
                     images.VERIFIER_DIR)
WORKDIR_KEEP = {
    # test.sh downloads these into its working directory before the tests.
    "fix-ocaml-gc": ("ocaml-original",),
    "sam-cell-seg": ("mobile_sam.pt",),
}
TEST_TIMEOUT_S = 3600
STUB = b"def test_reliquary_stub():\n    pass\n"


def stubbed_tests(task_id: str) -> bytes:
    """`tests/` with every `test_*.py` replaced by one passing test: test.sh still installs
    everything it names, the tests never run (their side effects stay out of the overlay)."""
    import tarfile

    src = tarfile.open(fileobj=io.BytesIO(source.tests_archive(task_id)))
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            name = member.name.rsplit("/", 1)[-1]
            if member.isfile() and name.startswith("test_") and name.endswith(".py"):
                data = STUB
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    return out.getvalue()


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def overlay_paths(task_id: str, diff: str) -> list[str]:
    """`docker diff` output -> the overlay's paths (relative, `./...`), in order."""
    workdir = source.workdir(task_id)
    keep = tuple(f"{workdir.rstrip('/')}/{name}" for name in WORKDIR_KEEP.get(task_id, ()))
    out = []
    for line in diff.splitlines():
        kind, _, path = line.partition(" ")
        if kind not in ("A", "C") or not path.startswith("/"):
            continue
        if any(_under(path, p) for p in EXCLUDED_PREFIXES):
            continue
        if _under(path, workdir) and not any(_under(path, k) for k in keep):
            continue
        out.append("." + path)
    return out


async def sh(*argv: str, stdin: bytes | None = None, timeout: float = 1800) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, b"", b"timeout"
    return process.returncode, out, err


async def build(args, task_id: str) -> dict:
    toml = source.task_toml(task_id)
    upstream = images.upstream()[toml["environment"]["docker_image"]]
    workdir = source.workdir(task_id)
    name = f"tbb-{task_id[:40]}"
    row: dict = {"task_id": task_id, "upstream": upstream}
    started = time.monotonic()
    await sh(args.docker, "rm", "-f", name)
    try:
        code, _, err = await sh(args.docker, "run", "-d", "--name", name, "-w", workdir,
                                "--entrypoint", "sleep", upstream, "infinity")
        if code:
            raise RuntimeError(f"run: {err.decode()[-300:]}")
        code, _, err = await sh(args.docker, "exec", "-i", name, "sh", "-c",
                                "mkdir -p /tests /logs/verifier && tar --no-same-owner -xzf - -C /tests",
                                stdin=stubbed_tests(task_id))
        if code:
            raise RuntimeError(f"stage: {err.decode()[-300:]}")
        clock = time.monotonic()
        code, out, err = await sh(args.docker, "exec", "-w", workdir, "-e", "HOME=/root",
                                  "-e", "DEBIAN_FRONTEND=noninteractive", name,
                                  "bash", "-c", "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
                                  timeout=TEST_TIMEOUT_S)
        row["test_sh_s"] = round(time.monotonic() - clock, 1)
        row["test_sh_exit"] = code
        _, reward, _ = await sh(args.docker, "exec", name, "cat", "/logs/verifier/reward.txt")
        row["stub_reward"] = reward.decode().strip() or None
        code, diff, err = await sh(args.docker, "diff", name)
        if code:
            raise RuntimeError(f"diff: {err.decode()[-300:]}")
        paths = overlay_paths(task_id, diff.decode())
        row["overlay_paths"] = len(paths)
        listing = "".join(p + "\0" for p in paths).encode()
        code, _, err = await sh(args.docker, "exec", "-i", name, "sh", "-c",
                                "cat > /tmp/.tb-overlay.list", stdin=listing)
        if code:
            raise RuntimeError(f"list: {err.decode()[-300:]}")
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            code, overlay, err = await sh(
                args.docker, "exec", name, "tar", "--null", "--no-recursion", "-C", "/",
                "-czf", "-", "-T", "/tmp/.tb-overlay.list")
            if code not in (0, 1):
                raise RuntimeError(f"tar: {err.decode()[-300:]}")
            parts = [overlay[i:i + images.PART_BYTES]
                     for i in range(0, len(overlay), images.PART_BYTES)] or [b""]
            names = [f"overlay.tar.gz.a{chr(ord('a') + i)}" for i in range(len(parts))]
            for part_name, part in zip(names, parts):
                (context / part_name).write_bytes(part)
            row["overlay_bytes"] = len(overlay)
            row["overlay_parts"] = len(parts)
            manifest = {"task_id": task_id, "upstream": upstream, "recipe": images.RECIPE,
                        "dataset": f"{source.DATASET}@{source.DIGEST}"}
            (context / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
            (context / "keep").write_text("")
            (context / "Dockerfile").write_text(
                f"FROM {upstream}\n"
                + "".join(f"COPY {n} {images.VERIFIER_DIR}/\n" for n in names)
                + f"COPY manifest.json {images.VERIFIER_DIR}/\n"
                + f"COPY keep {images.TMP_KEEP}\n")
            tag = f"{args.registry}/{images.image_ref(task_id)}"
            code, _, err = await sh(args.docker, "build", "-q", "-t", tag, str(context))
            if code:
                raise RuntimeError(f"build: {err.decode()[-300:]}")
        code, out, err = await sh(args.docker, "push", tag, timeout=3600)
        if code:
            raise RuntimeError(f"push: {err.decode()[-300:]}")
        digest = [w for w in out.decode().split() if w.startswith("sha256:")]
        if not digest:
            raise RuntimeError(f"push printed no digest: {out.decode()[-300:]}")
        row["image"] = f"{images.REPOSITORY}@{digest[-1]}"
        row["local_tag"] = tag
    except Exception as exc:  # noqa: BLE001 - recorded per task, the run goes on
        row["error"] = f"{type(exc).__name__}: {str(exc)[-500:]}"
    finally:
        await sh(args.docker, "rm", "-f", name)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


async def run_all(args) -> list[dict]:
    limit = asyncio.Semaphore(args.jobs)
    out = open(args.out, "a") if args.out else None

    async def one(task_id):
        async with limit:
            row = await build(args, task_id)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            if out is not None:
                out.write(line + "\n")
                out.flush()
            return row

    try:
        return await asyncio.gather(*(one(t) for t in (args.tasks or source.task_ids())))
    finally:
        if out is not None:
            out.close()


def write_lock(rows: list[dict]) -> None:
    pins = {images.image_ref(r["task_id"]): r["image"] for r in rows if r.get("image")}
    path = PACKAGE / "sandbox-images.lock.json"
    path.write_text(json.dumps({"images": dict(sorted(pins.items())), "version": 1},
                               indent=1, sort_keys=True) + "\n")
    print(f"lock: {len(pins)} images", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--registry", default="127.0.0.1:5001")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--out", help="append JSON lines here")
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--from", dest="source", help="lock from an earlier --out file")
    args = parser.parse_args()
    if args.source:
        rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line]
    else:
        rows = asyncio.run(run_all(args))
    if args.write_lock:
        latest = {r["task_id"]: r for r in rows if r.get("image")}
        write_lock(list(latest.values()))
    failed = sorted(r["task_id"] for r in rows if r.get("error"))
    print(json.dumps({"built": len(rows) - len(failed), "failed": failed}), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
