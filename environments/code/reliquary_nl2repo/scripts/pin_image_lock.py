"""Pin every task image by its registry digest in `sandbox-images.lock.json`, once.

A signed-episode sandbox serves a task only when the package's lock pins its
`TaskData.image` (env norm, rule 7). NL2Repo-Bench's 104 images are public on GHCR
(`ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0`, one per task, no shared
base). Per image: an anonymous pull token, a HEAD on the tag's manifest (the digest
`docker pull` records), then the manifest and its config blob, refusing an image that is
not linux/amd64. Nothing is pulled; the compressed size of the layers is summed.

    python scripts/pin_image_lock.py            # write the lock
    python scripts/pin_image_lock.py --verify   # re-resolve every pin, exit 1 on a moved tag
    python scripts/pin_image_lock.py --list     # print the pinned digests, one per line
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "reliquary_nl2repo"
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
TIMEOUT = 30


def images() -> list[str]:
    """Every task's image tag, in task order (from the pinned task list)."""
    spec = importlib.util.spec_from_file_location("_source", PACKAGE / "_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = sorted({rel.split("/", 1)[0] for rel in module.FILES})
    return [f"{REGISTRY}/multimodal-art-projection/nl2repobench/{t.lower()}:1.0" for t in tasks]


def _split(image: str) -> tuple[str, str]:
    host, _, rest = image.partition("/")
    if host != REGISTRY:
        raise ValueError(f"{image}: not on {REGISTRY}")
    name, _, tag = rest.rpartition(":")
    return name, tag


def _get(url: str, token: str, method: str = "GET", accept: str = ACCEPT):
    request = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": accept})
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def token(name: str) -> str:
    query = urllib.parse.urlencode({"scope": f"repository:{name}:pull", "service": REGISTRY})
    with urllib.request.urlopen(f"https://{REGISTRY}/token?{query}", timeout=TIMEOUT) as r:
        return json.load(r)["token"]


def resolve(image: str) -> tuple[str, int]:
    """(`<repo>@sha256:...`, compressed bytes) for `image`, checked linux/amd64."""
    name, tag = _split(image)
    bearer = token(name)
    base = f"https://{REGISTRY}/v2/{name}"
    with _get(f"{base}/manifests/{tag}", bearer, "HEAD") as r:
        digest = r.headers["Docker-Content-Digest"]
    with _get(f"{base}/manifests/{digest}", bearer) as r:
        manifest = json.load(r)
    if manifest.get("mediaType") in INDEX_TYPES:
        amd64 = [m for m in manifest["manifests"]
                 if m.get("platform", {}).get("architecture") == "amd64"
                 and m.get("platform", {}).get("os") == "linux"]
        if not amd64:
            raise ValueError(f"{image}: no linux/amd64 manifest")
        with _get(f"{base}/manifests/{amd64[0]['digest']}", bearer) as r:
            manifest = json.load(r)
    with _get(f"{base}/blobs/{manifest['config']['digest']}", bearer, accept="*/*") as r:
        config = json.load(r)
    if (config.get("os"), config.get("architecture")) != ("linux", "amd64"):
        raise ValueError(f"{image}: {config.get('os')}/{config.get('architecture')}")
    size = sum(layer["size"] for layer in manifest["layers"])
    return f"{REGISTRY}/{name}@{digest}", size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for ref in sorted(set(json.loads(LOCK.read_text())["images"].values())):
            print(ref)
        return 0
    tags = images()
    with ThreadPoolExecutor(8) as pool:
        resolved = dict(zip(tags, pool.map(resolve, tags)))
    if args.verify:
        pinned = json.loads(LOCK.read_text())["images"]
        moved = sorted(t for t in tags if pinned.get(t) != resolved[t][0])
        for tag in moved:
            print(f"moved: {tag} {pinned.get(tag)} -> {resolved[tag][0]}", file=sys.stderr)
        return 1 if moved or set(pinned) != set(tags) else 0
    lock = {"version": 1, "images": {tag: ref for tag, (ref, _) in resolved.items()}}
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    total = sum(size for _, size in resolved.values())
    print(f"{len(tags)} images, {total / 1e9:.1f} GB compressed (layers not deduplicated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
