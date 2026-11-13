"""Pin every task image by its registry digest in `sandbox-images.lock.json`, once.

A signed-episode sandbox serves a task only when the package's lock pins its
`TaskData.image` (env norm, rule 7). SWE-bench Pro V2's 642 images are public on GHCR
(`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`). Per image: an anonymous
pull token, a HEAD on the tag's manifest (the digest `docker pull` records), then the
manifest and its config blob, refusing an image that is not linux/amd64. Nothing is
pulled; the compressed size of the layers is summed for disk planning.

    python scripts/pin_image_lock.py            # write the lock
    python scripts/pin_image_lock.py --verify   # re-resolve every pin, exit 1 on a moved tag
    python scripts/pin_image_lock.py --list     # print the pinned digests, one per line
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "reliquary_swebench_pro"
LOCK = PACKAGE / "sandbox-images.lock.json"
REGISTRY = "ghcr.io"
ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
))
INDEX_TYPES = frozenset({"application/vnd.oci.image.index.v1+json",
                         "application/vnd.docker.distribution.manifest.list.v2+json"})
PLATFORM = ("linux", "amd64")
TIMEOUT = 30
WORKERS = 8


def images() -> list[str]:
    """Every task's `[environment].docker_image`, in task order, deduplicated."""
    found = []
    for toml in sorted((PACKAGE / "tasks").glob("*/task.toml")):
        found.append(tomllib.loads(toml.read_text())["environment"]["docker_image"])
    return list(dict.fromkeys(found))


def _split(image: str) -> tuple[str, str]:
    host, _, rest = image.partition("/")
    if host != REGISTRY:
        raise ValueError(f"{image}: not on {REGISTRY}")
    name, _, tag = rest.rpartition(":")
    return name, tag


def token(name: str) -> str:
    query = urllib.parse.urlencode({"scope": f"repository:{name}:pull", "service": REGISTRY})
    with urllib.request.urlopen(f"https://{REGISTRY}/token?{query}", timeout=TIMEOUT) as r:
        return json.loads(r.read())["token"]


def fetch(url: str, bearer: str, method: str = "GET") -> tuple[dict[str, str], bytes]:
    request = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {bearer}", "Accept": ACCEPT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return {k.lower(): v for k, v in response.headers.items()}, response.read()


def pin(image: str, bearer: str) -> tuple[str, int]:
    """(`repo@sha256:...`, compressed bytes of the linux/amd64 image's layers)."""
    name, tag = _split(image)
    base = f"https://{REGISTRY}/v2/{name}"
    headers, _ = fetch(f"{base}/manifests/{tag}", bearer, "HEAD")
    digest = headers.get("docker-content-digest", "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RuntimeError(f"{image}: no Docker-Content-Digest ({digest!r})")
    media = headers.get("content-type", "").split(";")[0].strip()
    manifest = json.loads(fetch(f"{base}/manifests/{digest}", bearer)[1])
    if media in INDEX_TYPES:
        chosen = [m for m in manifest.get("manifests", [])
                  if (m.get("platform", {}).get("os"), m.get("platform", {}).get("architecture"))
                  == PLATFORM]
        if not chosen:
            raise RuntimeError(f"{image}: the index lists no linux/amd64 manifest")
        manifest = json.loads(fetch(f"{base}/manifests/{chosen[0]['digest']}", bearer)[1])
    else:
        config = json.loads(fetch(f"{base}/blobs/{manifest['config']['digest']}", bearer)[1])
        if (config.get("os"), config.get("architecture")) != PLATFORM:
            raise RuntimeError(f"{image}: {config.get('os')}/{config.get('architecture')}, "
                               "not linux/amd64")
    size = sum(int(layer.get("size", 0)) for layer in manifest.get("layers", []))
    return f"{REGISTRY}/{name}@{digest}", size


def _pin(image: str, attempts: int = 5) -> tuple[str, int]:
    for attempt in range(attempts):
        try:
            return pin(image, token(_split(image)[0]))
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
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    wanted = images()
    if args.list:
        pinned = json.loads(LOCK.read_text())["images"]
        print("\n".join(pinned[image] for image in wanted))
        return 0
    with ThreadPoolExecutor(WORKERS) as pool:
        resolved = dict(zip(wanted, pool.map(_pin, wanted), strict=True))
    total = sum(size for _, size in resolved.values())
    largest = max(resolved.items(), key=lambda kv: kv[1][1])
    print(f"{len(resolved)} images, {total / 1e9:.1f} GB compressed, largest "
          f"{largest[1][1] / 1e9:.2f} GB ({largest[0]})", file=sys.stderr)
    if args.verify:
        pinned = json.loads(LOCK.read_text())["images"]
        moved = sorted(image for image in wanted if pinned.get(image) != resolved[image][0])
        for image in moved:
            print(f"{image}: pinned {pinned.get(image)}, registry {resolved[image][0]}")
        print(f"{len(wanted) - len(moved)}/{len(wanted)} pins match the registry")
        return 1 if moved else 0
    document = {"version": 1, "images": {image: resolved[image][0] for image in sorted(resolved)}}
    LOCK.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"{len(resolved)} images pinned in {LOCK.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
