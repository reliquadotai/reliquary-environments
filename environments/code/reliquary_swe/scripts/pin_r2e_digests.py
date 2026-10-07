"""Resolve R2E-Gym-Subset's images to their registry digests, once.

Same registry call as `pin_swesmith_digests.py` (HEAD on the manifest, read
`Docker-Content-Digest`; nothing is pulled), over the rows of the pinned
R2E revision -- one image per row, 4,578 in all, under 10 Docker Hub
repositories (`namanjain12/<repo>_final`). One pull token per repository,
a few requests in flight at once. Merges into `r2e_digests.json` rather than
overwriting it, so a partial run (`--num-tasks N`, the first N tasks of the
split's own order) only ever adds pins.
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

OUT = Path(__file__).resolve().parent.parent / "reliquary_swe" / "r2e_digests.json"
WORKERS = 8
# Docker Hub's anonymous pull tokens live 300 s; a full run takes longer.
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


def images(num_tasks: int | None) -> list[str]:
    """The `docker_image` tags of the first `num_tasks` tasks in the split's
    own order (`corpus.load_r2e_rows`'s), all of them for `None`."""
    dataset = corpus._r2e_dataset()
    by_id = {
        corpus._r2e_instance_id(repo, commit): image
        for repo, commit, image in zip(
            dataset["repo_name"], dataset["commit_hash"], dataset["docker_image"], strict=True
        )
    }
    ordered = sorted(by_id, key=corpus._r2e_order_key)
    if num_tasks is not None:
        ordered = ordered[:num_tasks]
    return [by_id[instance_id] for instance_id in ordered]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args()
    wanted = images(args.num_tasks)
    pinned = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [image for image in wanted if image not in pinned]
    with ThreadPoolExecutor(WORKERS) as pool:
        resolved = pool.map(lambda image: digest(image, _token(image.partition(":")[0])), todo)
        for image, pin in zip(todo, resolved, strict=True):
            pinned[image] = pin
    OUT.write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    print(f"{len(todo)} newly pinned, {len(pinned)} in {OUT.name}")


if __name__ == "__main__":
    main()
