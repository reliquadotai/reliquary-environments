"""The images a task runs on.

Upstream publishes one prebuilt image per task (`task.toml`'s `docker_image`,
`alexgshaw/<task>:<date>`, 20251031 for 79, 2026-04 rebuilds for 10), pinned here by registry digest (`upstream_images.json`,
read from Docker Hub without pulling). A box runs our verifier image instead: that
upstream image by digest plus inert files, `VERIFIER_DIR/overlay.tar.gz.<part>`, every file
`tests/test.sh`'s own installs add or change (curl, uv, Python, pytest and the test
dependencies, a few clones and downloads), recorded by running `test.sh` with network on
the untouched image (`scripts/build_verifier_images.py`). `setup` deletes it from the
agent's box; grading unpacks it so `test.sh` runs offline.

`sandbox-images.lock.json` maps each task's `image_ref` to the verifier image's digest;
a task whose image the lock does not pin is refused by the sandbox. Both images live in
our registry (Hippius-backed, docker.io paths kept), the verifier images there only.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = "reliquadotai/terminal-bench-2-1"
"""Our verifier images: one repository, tagged by task id."""
VERIFIER_DIR = "/opt/tb-verifier"
RECIPE = 2
"""`scripts/build_verifier_images.py`'s recipe; a change is a new image for every task.
2: the tests stubbed while recording (no test side effect in the overlay), the overlay
split in parts of at most `PART_BYTES` (one registry layer each), `/tmp` kept non-empty."""
PART_BYTES = 900 * 1024 * 1024
TMP_KEEP = "/tmp/.reliquary-tb-keep"
"""A file in the image's `/tmp`: gVisor mounts a tmpfs over an empty `/tmp`, Docker's runc
does not, and a test renaming a file into `/tmp` then fails with EXDEV."""


def image_ref(task_id: str) -> str:
    return f"{REPOSITORY}:{task_id}-r{RECIPE}"


@functools.cache
def upstream() -> dict[str, str]:
    """`alexgshaw/<task>:<tag>` -> `alexgshaw/<task>@sha256:...`."""
    return dict(json.loads((PACKAGE / "upstream_images.json").read_text())["images"])


@functools.cache
def lock() -> dict[str, str]:
    """`image_ref(task)` -> `reliquadotai/terminal-bench-2-1@sha256:...`."""
    return dict(json.loads((PACKAGE / "sandbox-images.lock.json").read_text())["images"])


__all__ = ["REPOSITORY", "RECIPE", "VERIFIER_DIR", "image_ref", "lock", "upstream"]
