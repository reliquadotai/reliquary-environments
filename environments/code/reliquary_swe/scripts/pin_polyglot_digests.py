"""Resolve MiMo polyglot images to registry digests for signed-episode sandboxes.

The verifiers taskset provisions polyglot boxes by tag; a sandbox serves an image only
when the package's `sandbox-images.lock.json` pins it, and `scripts/pin_sandbox_lock.py`
builds that lock from this file. Same registry call as pin_swesmith_digests.py (HEAD on the
manifest, nothing pulled). Merges into polyglot_digests.json: `--num-tasks N` pins the
split's first N tasks (the whole split is 2,698), `--instance` adds named ones (the
goldens) and fails on a name the corpus does not have.

Platform: a sandbox host runs linux/amd64. A multi-platform index must list a
linux/amd64 manifest (always checked: one manifest GET). A single-platform manifest
(what MiMo publishes) carries its platform in its config blob, which takes a manifest
GET, and anonymous Docker Hub allows 100 of those per hour (HEAD is free): so the
named instances and `--check-sample K` images spread over the run are checked, not
every one.

    uv run python scripts/pin_polyglot_digests.py --num-tasks 2698 \\
        --instance format-code-task-002703 --instance format-code-task-001647
"""

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pin_swesmith_digests import ACCEPT, TIMEOUT, pull_token  # noqa: E402

from reliquary_swe import corpus  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reliquary_swe" / "polyglot_digests.json"
REGISTRY = "https://registry-1.docker.io/v2"
INDEX_TYPES = frozenset({"application/vnd.oci.image.index.v1+json",
                         "application/vnd.docker.distribution.manifest.list.v2+json"})
PLATFORM = ("linux", "amd64")
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


def fetch(url: str, bearer: str, method: str = "GET") -> tuple[dict[str, str], bytes]:
    request = urllib.request.Request(
        url, method=method, headers={"Authorization": f"Bearer {bearer}", "Accept": ACCEPT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return dict(response.headers), response.read()


def _header(headers: dict[str, str], name: str) -> str:
    return next((value for key, value in headers.items() if key.lower() == name), "")


def pin(image: str, bearer: str, *, check_single: bool, fetch=fetch) -> str:
    """`name:tag` as `name@sha256:...`, refusing an image with no linux/amd64 manifest."""
    name, _, tag = image.partition(":")
    headers, _ = fetch(f"{REGISTRY}/{name}/manifests/{tag or 'latest'}", bearer, "HEAD")
    digest = _header(headers, "docker-content-digest")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RuntimeError(f"{image}: registry returned no Docker-Content-Digest ({digest!r})")
    media = _header(headers, "content-type").split(";")[0].strip()
    by_digest = f"{REGISTRY}/{name}/manifests/{digest}"
    if media in INDEX_TYPES:
        manifests = json.loads(fetch(by_digest, bearer)[1]).get("manifests", [])
        found = [(m.get("platform", {}).get("os"), m.get("platform", {}).get("architecture"))
                 for m in manifests]
    elif check_single:
        config = json.loads(fetch(by_digest, bearer)[1])["config"]["digest"]
        blob = json.loads(fetch(f"{REGISTRY}/{name}/blobs/{config}", bearer)[1])
        found = [(blob.get("os"), blob.get("architecture"))]
    else:
        found = [PLATFORM]
    if PLATFORM not in found:
        raise RuntimeError(f"{image}: no linux/amd64 manifest (found {found})")
    return f"{name}@{digest}"


def images(rows, num_tasks: int, instances: list[str]) -> list[str]:
    named = set(instances)
    unknown = sorted(named - {row.instance_id for row in rows})
    if unknown:
        raise ValueError(f"not in the polyglot corpus: {', '.join(unknown)}")
    chosen = list(rows[:num_tasks]) + [row for row in rows if row.instance_id in named]
    return list(dict.fromkeys(row.image for row in chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--check-sample", type=int, default=20,
                        help="single-platform images whose platform is checked (rate-limited)")
    args = parser.parse_args()
    rows = corpus.load_polyglot_rows(None)
    wanted = images(rows, args.num_tasks, args.instance)
    pinned = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [image for image in wanted if image not in pinned]
    checked = set(images(rows, 0, args.instance))
    if todo and args.check_sample > 0:
        checked.update(todo[:: max(1, len(todo) // args.check_sample)][: args.check_sample])
    with ThreadPoolExecutor(WORKERS) as pool:
        pins = pool.map(lambda image: pin(image, token(), check_single=image in checked), todo)
        for image, digest in zip(todo, pins, strict=True):
            pinned[image] = digest
    OUT.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    print(f"{len(todo)} newly pinned ({len(checked & set(todo))} platform-checked), "
          f"{len(pinned)} in {OUT.name}")


if __name__ == "__main__":
    main()
