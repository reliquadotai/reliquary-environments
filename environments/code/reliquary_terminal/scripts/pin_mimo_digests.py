"""Resolve MiMo-V2.6's terminal images to registry digests for signed-episode sandboxes.

The verifiers taskset provisions them by tag (`xiaomimimo/mimo-v2.6-rl-oss:<tag>`).
Signed-episode sandboxes serve only the `tmax` split (docs/env-norm.md), so no package code
reads `mimo_digests.json` today; it records the digests the MiMo rows were checked with.
HEAD on the manifest, nothing pulled; the same registry call as reliquary-swe's pin
scripts.

    uv run python scripts/pin_mimo_digests.py
"""

import json
import urllib.request
from pathlib import Path

from reliquary_terminal.taskset import TRAIN_IMAGE_REPOSITORY, load_train_rows

OUT = Path(__file__).resolve().parent.parent / "reliquary_terminal" / "mimo_digests.json"
ACCEPT = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
TIMEOUT = 30


def pull_token(name: str) -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{name}:pull"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)["token"]


def digest(image: str, token: str) -> str:
    name, _, tag = image.partition(":")
    request = urllib.request.Request(f"https://registry-1.docker.io/v2/{name}/manifests/{tag}",
                                     method="HEAD",
                                     headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        pinned = response.headers.get("Docker-Content-Digest")
    if not pinned or not pinned.startswith("sha256:"):
        raise RuntimeError(f"{image}: registry returned no Docker-Content-Digest ({pinned!r})")
    return f"{name}@{pinned}"


def images() -> list[str]:
    tags = (row["docker_image"].split(":", 1)[0] for row in load_train_rows())
    return list(dict.fromkeys(f"{TRAIN_IMAGE_REPOSITORY}:{tag}" for tag in tags))


def main() -> None:
    token = pull_token(TRAIN_IMAGE_REPOSITORY)
    pinned = {image: digest(image, token) for image in images()}
    OUT.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    print(f"{len(pinned)} pinned in {OUT.name}")


if __name__ == "__main__":
    main()
