"""Resolve each of the top SWE-smith images to its registry digest, once.

Asks the Docker Hub registry HTTP API directly (HEAD on the manifest, reading
`Docker-Content-Digest`), so neither `docker buildx` nor a pulled image is
needed. The digest is that of the tag's manifest list/index, which is what
`docker pull <image>` records in `RepoDigests`.
"""

import json
import urllib.request
from pathlib import Path

from reliquary_swe import corpus

OUT = Path(__file__).resolve().parent.parent / "reliquary_swe" / "swesmith_digests.json"
ACCEPT = ",".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def digest(image: str) -> str:
    name, _, tag = image.partition(":")
    tag = tag or "latest"
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{name}:pull"
    with urllib.request.urlopen(url) as r:
        token = json.load(r)["token"]
    req = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{name}/manifests/{tag}",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
    )
    with urllib.request.urlopen(req) as r:
        return f"{name}@{r.headers['Docker-Content-Digest']}"


def main():
    images = corpus.swesmith_image_rank()[: corpus.DEFAULT_SWESMITH_IMAGES]
    OUT.write_text(json.dumps({image: digest(image) for image in images}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
