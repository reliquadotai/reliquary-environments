"""Resolve MiMo polyglot images to registry digests for signed-episode sandboxes.

The verifiers taskset provisions polyglot boxes by tag; a sandbox task must name its
image by digest (reliquary_swe.sandbox.image_of). Same registry call as
pin_swesmith_digests.py (HEAD on the manifest, nothing pulled). Merges into
polyglot_digests.json: `--num-tasks N` pins the split's first N tasks, `--instance`
adds named ones (the goldens).

    uv run python scripts/pin_polyglot_digests.py --num-tasks 300 \\
        --instance format-code-task-002703 --instance format-code-task-001647
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pin_swesmith_digests import digest, pull_token  # noqa: E402

from reliquary_swe import corpus  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reliquary_swe" / "polyglot_digests.json"
WORKERS = 8
# Docker Hub's anonymous pull tokens live 300 s; a long run outlives one.
TOKEN_TTL = 240.0
_token: tuple[float, str] = (0.0, "")
_lock = threading.Lock()


def token() -> str:
    global _token
    with _lock:
        if time.monotonic() - _token[0] > TOKEN_TTL:
            _token = (time.monotonic(), pull_token(corpus.POLYGLOT_IMAGE_REPOSITORY))
        return _token[1]


def images(rows, num_tasks: int, instances: list[str]) -> list[str]:
    named = set(instances)
    chosen = list(rows[:num_tasks]) + [row for row in rows if row.instance_id in named]
    return list(dict.fromkeys(row.image for row in chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--instance", action="append", default=[])
    args = parser.parse_args()
    wanted = images(corpus.load_polyglot_rows(None), args.num_tasks, args.instance)
    pinned = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [image for image in wanted if image not in pinned]
    with ThreadPoolExecutor(WORKERS) as pool:
        for image, pin in zip(todo, pool.map(lambda image: digest(image, token()), todo),
                              strict=True):
            pinned[image] = pin
    OUT.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    print(f"{len(todo)} newly pinned, {len(pinned)} in {OUT.name}")


if __name__ == "__main__":
    main()
