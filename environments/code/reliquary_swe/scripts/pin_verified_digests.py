"""Resolve SWE-bench Verified's images (split `eval`) to their registry digests, once.

One image per instance, 500 in all, each in its own Docker Hub repository
(`swebench/sweb.eval.x86_64.<instance>:latest`, `swe_adapter.image_for`). Same registry
call as `pin_polyglot_digests.pin`: HEAD on the tag's manifest for its digest (an OCI
index for every Verified image), then a GET of that index, refusing an image that lists
no linux/amd64 manifest. The pin is the index digest, what `docker pull` records in
`RepoDigests`, as for the other corpora.

The eval split is held out: a gateway serves it only under a split named `eval`, and only
when its operator lists one. Locking these images adds nothing a training split can name,
and a gateway pre-pulls only the images of the splits and ranges it serves.

    uv run python scripts/pin_verified_digests.py           # pin missing images
    uv run python scripts/pin_verified_digests.py --verify  # re-resolve every pin, exit 1 on a moved tag
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pin_polyglot_digests import pin  # noqa: E402
from pin_swesmith_digests import pull_token  # noqa: E402

from reliquary_swe import corpus, swe_adapter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reliquary_swe" / "verified_digests.json"
WORKERS = 8
# Docker Hub's anonymous pull tokens live 300 s.
TOKEN_TTL = 240.0

_tokens: dict[str, tuple[float, str]] = {}
_lock = threading.Lock()


def _token(name: str) -> str:
    with _lock:
        issued, token = _tokens.get(name, (0.0, ""))
        if time.monotonic() - issued > TOKEN_TTL:
            token = pull_token(name)
            _tokens[name] = (time.monotonic(), token)
        return token


def images() -> list[str]:
    """The image of every row of the eval split, in the split's order."""
    return list(dict.fromkeys(swe_adapter.image_for(row) for row in corpus.load_rows("eval")))


def _pin(image: str, attempts: int = 5) -> str:
    """`pin`, retried on a transient network or registry error (500 repositories, 1,000
    requests: a dropped connection is expected over a full run)."""
    for attempt in range(attempts):
        try:
            return pin(image, _token(image.partition(":")[0]), check_single=True)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="re-resolve every pinned image and compare, writing nothing")
    args = parser.parse_args()
    wanted = images()
    pinned = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = wanted if args.verify else [image for image in wanted if image not in pinned]
    with ThreadPoolExecutor(WORKERS) as pool:
        resolved = dict(zip(todo, pool.map(_pin, todo), strict=True))
    if args.verify:
        moved = sorted(image for image in wanted if pinned.get(image) != resolved[image])
        for image in moved:
            print(f"{image}: pinned {pinned.get(image)}, registry {resolved[image]}")
        print(f"{len(wanted) - len(moved)}/{len(wanted)} pins match the registry")
        return 1 if moved else 0
    pinned.update(resolved)
    OUT.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    print(f"{len(todo)} newly pinned, {len(pinned)} in {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
