"""Task images for signed-episode sandbox hosts: list, pull, check, approve, build.

Image lists come from each env package's image lock (`sandbox-images.lock.json`, tag ->
`repo@sha256:<digest>`) and its own image listing, never from here:

    python -m reliquary_swe.images --split r2e --num-tasks 300 > r2e.tags
    python scripts/sandbox_images.py from-lock \
        environments/code/reliquary_swe/reliquary_swe/sandbox-images.lock.json --tags r2e.tags > r2e.json

`from-lock` prints `{"images": [digests]}` for the tags (or digests) listed one per line
in `--tags` (every image the lock pins when `--tags` is absent) and refuses, exit 2, a tag
the lock does not pin: a gateway serves only pinned images. reliquary-terminal's lock
(`environments/code/reliquary_terminal/reliquary_terminal/sandbox-images.lock.json`) comes
with its base image in a registry (docs/tmax.md); then:

    python scripts/sandbox_images.py from-lock \
        environments/code/reliquary_terminal/reliquary_terminal/sandbox-images.lock.json > terminal.json

Then, on the sandbox host (the gateway never pulls; spec §7):

    python scripts/sandbox_images.py pull r2e.json terminal.json --docker-host unix:///run/docker.sock
    python scripts/sandbox_images.py check r2e.json --runtime runsc --require git
    python scripts/sandbox_images.py check terminal.json --runtime runsc
    python scripts/sandbox_images.py approve r2e.json terminal.json   # -> the gateway setting

`check` records each image's last result in `--record` (default
`sandbox-images.checked.json` in the current directory); `approve` refuses unless every
image of its manifests has a recorded last check that passed.

SWE-smith, R2E and polyglot images are upstream images pinned by digest: once the
harness runs on the miner, nothing installs at run time, so nothing needs baking. The one
image we build is tmax's shared base. The validated base image goes to a registry as
is, never rebuilt (docs/tmax.md): a rebuild is a new image and needs a new validation.
`build-tmax` exists for that case:

    python scripts/sandbox_images.py build-tmax --registry <REGISTRY> --push

without --push it only prints the command. It runs the terminal package's `scripts/tmax_validate.py build --repository
<REGISTRY>/reliquary-tmax-base` (docs/tmax.md) and prints the pushed digest. Run Docker
commands only on a container host.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
ROOT = Path(__file__).resolve().parent.parent
TERMINAL = ROOT / "environments" / "code" / "reliquary_terminal"
BASE_TOOLS = ("bash", "sleep")
Run = Callable[..., Any]


def load_images(paths: Sequence[Path]) -> list[str]:
    images: list[str] = []
    for path in paths:
        for image in json.loads(Path(path).read_text())["images"]:
            if not DIGEST.fullmatch(image):
                raise ValueError(f"{path}: {image!r} is not pinned by digest")
            images.append(image)
    return list(dict.fromkeys(images))


def from_lock(lock_path: Path, tags: Sequence[str] | None) -> dict[str, list[str]]:
    """The digests the lock pins for `tags` (tags or digests it pins); all of them when
    `tags` is None. A tag the lock does not pin raises KeyError naming it."""
    images = json.loads(Path(lock_path).read_text())["images"]
    digests = set(images.values())
    if tags is None:
        return {"images": sorted(digests)}
    missing = [t for t in tags if t not in images and t not in digests]
    if missing:
        raise KeyError(", ".join(missing))
    return {"images": sorted({images.get(t, t) for t in tags})}


def _read_tags(path: Path) -> list[str]:
    """One tag or digest per line; blank lines ignored."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _docker(host: str | None) -> list[str]:
    return ["docker", "-H", host] if host else ["docker"]


def pull(images: Sequence[str], *, docker: list[str], jobs: int, run: Run = subprocess.run) -> list[str]:
    """Pull what is missing; returns the images that could not be pulled."""

    def one(image: str) -> str | None:
        if run([*docker, "image", "inspect", image], capture_output=True, text=True).returncode == 0:
            return None
        done = run([*docker, "pull", image], capture_output=True, text=True)
        return image if done.returncode != 0 else None

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        return [image for image in pool.map(one, images) if image is not None]


def check(images: Sequence[str], *, docker: list[str], runtime: str, require: Sequence[str],
          run: Run = subprocess.run) -> list[dict[str, Any]]:
    """Each image starts under `runtime` with no network and has what the episode
    tools and the env's hooks run: /bin/bash, sleep (the box's PID 1), and `require`."""
    probe = " ; ".join(
        ["m=", 'test -x /bin/bash || m="$m bash"']
        + [f'command -v {tool} >/dev/null 2>&1 || m="$m {tool}"' for tool in (*BASE_TOOLS[1:], *require)]
        + ['[ -z "$m" ] || { echo "missing:$m"; exit 3; }'])
    results = []
    for image in images:
        done = run([*docker, "run", "--rm", "--network", "none", "--runtime", runtime,
                    "--entrypoint", "/bin/sh", image, "-c", probe],
                   capture_output=True, text=True, timeout=600)
        results.append({"image": image, "ok": done.returncode == 0,
                        "detail": (done.stdout + done.stderr).strip()[-300:]})
    return results


DEFAULT_RECORD = Path("sandbox-images.checked.json")


def load_record(path: Path) -> dict[str, dict[str, Any]]:
    """image -> its last check result, or {} when nothing was recorded."""
    try:
        return dict(json.loads(Path(path).read_text())["images"])
    except FileNotFoundError:
        return {}


def record_checks(path: Path, results: Sequence[dict[str, Any]], runtime: str,
                  require: Sequence[str]) -> None:
    """Replace each checked image's entry; other images keep theirs."""
    images = load_record(path)
    for result in results:
        images[result["image"]] = {**result, "runtime": runtime, "require": list(require)}
    Path(path).write_text(json.dumps({"images": images}, indent=1, sort_keys=True) + "\n")


def unapproved(images: Sequence[str], record: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The images whose last recorded check is missing or did not pass."""
    return [image for image in images if not (record.get(image) or {}).get("ok")]


def approve_line(images: Sequence[str]) -> str:
    return "RELIQUARY_SANDBOX_EPISODE_IMAGES=" + json.dumps(sorted(set(images)))


def build_tmax_command(registry: str, package: Path = TERMINAL) -> list[str]:
    return ["uv", "run", "--project", str(package), "python",
            str(Path(package) / "scripts" / "tmax_validate.py"), "build",
            "--repository", f"{registry.rstrip('/')}/reliquary-tmax-base"]


def main(argv: list[str] | None = None, run: Run = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pull", "check", "approve"):
        sub = commands.add_parser(name)
        sub.add_argument("manifests", nargs="+", type=Path)
        if name != "approve":
            sub.add_argument("--docker-host")
        if name != "pull":
            sub.add_argument("--record", type=Path, default=DEFAULT_RECORD,
                             help="each image's last check result (check writes, approve reads)")
    commands.choices["pull"].add_argument("--jobs", type=int, default=3)
    commands.choices["check"].add_argument("--runtime", default="runsc")
    commands.choices["check"].add_argument("--require", action="append", default=[])
    listed = commands.add_parser("from-lock", help="the digests an env's image lock pins")
    listed.add_argument("lock", type=Path)
    listed.add_argument("--tags", type=Path,
                        help="one tag or digest per line (default: every image the lock pins)")
    build = commands.add_parser("build-tmax")
    build.add_argument("--registry", required=True)
    build.add_argument("--terminal-package", type=Path, default=TERMINAL)
    build.add_argument("--push", action="store_true",
                       help="actually build and push; without it the command is only printed")
    args = parser.parse_args(argv)
    if args.command == "from-lock":
        try:
            result = from_lock(args.lock, None if args.tags is None else _read_tags(args.tags))
        except KeyError as missing:
            print(f"not pinned by the lock: {missing.args[0]}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=1))
        return 0
    if args.command == "build-tmax":
        command = build_tmax_command(args.registry, args.terminal_package)
        if not args.push:
            print("dry run (add --push to build and push):", " ".join(command))
            return 0
        return run(command).returncode
    images = load_images(args.manifests)
    if args.command == "approve":
        refused = unapproved(images, load_record(args.record))
        if refused:
            print(f"not approved: no passing last check in {args.record} for "
                  f"{len(refused)} image(s); run check first:", file=sys.stderr)
            for image in refused:
                print(f"  {image}", file=sys.stderr)
            return 1
        print(approve_line(images))
        return 0
    docker = _docker(args.docker_host)
    if args.command == "pull":
        failed = pull(images, docker=docker, jobs=args.jobs, run=run)
        print(f"{len(images) - len(failed)}/{len(images)} present", file=sys.stderr)
        for image in failed:
            print(f"FAILED {image}")
        return 1 if failed else 0
    results = check(images, docker=docker, runtime=args.runtime, require=args.require, run=run)
    record_checks(args.record, results, args.runtime, args.require)
    for result in results:
        print(json.dumps(result))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
