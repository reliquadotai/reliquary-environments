"""Write reliquary_swe/sandbox-images.lock.json: the image lock a signed-episode sandbox
reads before it serves a task (every image pinned by digest).

The lock is the union of the three digest files the corpora already pin
(`swesmith_digests.json`, `r2e_digests.json`, `polyglot_digests.json`, tag -> digest), so
it covers every image the `train`, `r2e` and `polyglot` splits can name. Re-run after any
pin script changes one of them:

    uv run python scripts/pin_sandbox_lock.py           # rewrite the lock
    uv run python scripts/pin_sandbox_lock.py --check   # exit 1 when it is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "reliquary_swe"
SOURCES = ("swesmith_digests.json", "r2e_digests.json", "polyglot_digests.json")
LOCK_NAME = "sandbox-images.lock.json"


def build_lock(package: Path = PACKAGE) -> dict:
    images: dict[str, str] = {}
    for name in SOURCES:
        for tag, digest in json.loads((package / name).read_text()).items():
            if images.get(tag, digest) != digest:
                raise ValueError(f"{tag} is pinned to two digests ({images[tag]}, {digest})")
            images[tag] = digest
    return {"version": 1, "images": dict(sorted(images.items()))}


def render(lock: dict) -> str:
    return json.dumps(lock, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = render(build_lock())
    path = PACKAGE / LOCK_NAME
    if argv == ["--check"]:
        return 0 if path.is_file() and path.read_text() == text else 1
    if argv:
        print("usage: pin_sandbox_lock.py [--check]", file=sys.stderr)
        return 2
    path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
