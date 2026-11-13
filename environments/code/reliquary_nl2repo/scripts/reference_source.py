"""Recover each task's reference source from its image's build layers (gold-check host).

NL2Repo-Bench publishes no reference solution. Most images were built by copying the
upstream project into a directory (`COPY ./<project> /<dir>`), installing it, testing it,
then deleting the directory: the files are still in the copy's layer. Per image this
script runs `docker save` (the image must be pulled), replays the layers in order with
their whiteouts, and records every top-level directory outside the system tree that, in
some layer, holds Python files. The candidate with the most `.py` files, as it stood at
its fullest (just before the layer that deleted it, for most), is the reference:
written as `<out>/<task>.tar.gz` (regular files, paths relative to the directory).

Multi-stage builds that copied only site-packages into the final image (ftfy, structlog,
tqdm, ...) and images built from the tests alone hold no reference; they are reported
and the gold check cannot verify them. Output: one JSON line per task. Never
redistribute what it writes: it is upstream source under each project's license.

    python scripts/reference_source.py --out /root/refs [--tasks tinydb ...]
"""

from __future__ import annotations

import argparse
import io
import json
import posixpath
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "reliquary_nl2repo"
SYSTEM = frozenset("bin boot dev etc home lib lib32 lib64 libx32 media mnt opt proc root run "
                   "sbin srv sys tmp usr var workspace logs".split())


def lock() -> dict[str, str]:
    return json.loads((PACKAGE / "sandbox-images.lock.json").read_text())["images"]


def task_ids() -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_source", PACKAGE / "_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted({rel.split("/", 1)[0] for rel in module.FILES})


def layers(saved: Path) -> list[Path]:
    manifest = json.loads((saved / "manifest.json").read_text())[0]
    return [saved / layer for layer in manifest["Layers"]]


def replay(layer_paths: list[Path]) -> tuple[dict[str, dict[str, bytes]], dict[str, int]]:
    """Candidate roots -> {relative path: bytes} as they stood at their fullest."""
    state: dict[str, dict[str, bytes]] = {}
    best: dict[str, dict[str, bytes]] = {}
    for layer in layer_paths:
        with tarfile.open(layer, mode="r:*") as tar:
            members = tar.getmembers()
            for member in members:
                name = posixpath.normpath(member.name.lstrip("./"))
                parts = name.split("/")
                top, base = parts[0], parts[-1]
                if top in SYSTEM or top.startswith("."):
                    continue
                if base == ".wh..wh..opq":
                    rel = "/".join(parts[1:-1])
                    files = state.get(top, {})
                    for key in [k for k in files if not rel or k.startswith(rel + "/")]:
                        del files[key]
                    continue
                if base.startswith(".wh."):
                    target = "/".join(parts[:-1] + [base[4:]])
                    if target == top:
                        state.pop(top, None)
                        continue
                    rel = target[len(top) + 1:]
                    files = state.get(top, {})
                    for key in [k for k in files if k == rel or k.startswith(rel + "/")]:
                        del files[key]
                    continue
                if len(parts) < 2 or not member.isreg():
                    continue
                handle = tar.extractfile(member)
                state.setdefault(top, {})["/".join(parts[1:])] = handle.read() if handle else b""
        for top, files in state.items():
            if any(k.endswith(".py") for k in files) and len(files) >= len(best.get(top, {})):
                best[top] = dict(files)
    counts = {top: sum(k.endswith(".py") for k in files) for top, files in best.items()}
    return best, counts


def extract(task: str, ref: str, out: Path) -> dict:
    row: dict = {"task_id": task, "image": ref}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        saved = Path(tmp)
        save = subprocess.run(f"docker save {ref} | tar -x -C {tmp}", shell=True,
                              capture_output=True, text=True)
        if save.returncode:
            row["error"] = f"docker save: {save.stderr[-300:]}"
            return row
        best, counts = replay(layers(saved))
    row["candidates"] = counts
    if not best:
        row["reference"] = None
        return row
    top = max(counts, key=lambda t: (counts[t], len(best[t])))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel, data in sorted(best[top].items()):
            info = tarfile.TarInfo(rel)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    (out / f"{task}.tar.gz").write_bytes(buffer.getvalue())
    row.update(reference=f"/{top}", files=len(best[top]), py_files=counts[top],
               bytes=len(buffer.getvalue()))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pins = lock()
    for task in task_ids():
        if args.tasks and task not in args.tasks:
            continue
        ref = pins[f"ghcr.io/multimodal-art-projection/nl2repobench/{task.lower()}:1.0"]
        print(json.dumps(extract(task, ref, args.out), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
