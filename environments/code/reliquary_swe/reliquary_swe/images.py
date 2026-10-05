"""Print every image a taskset configuration needs, one per line.

Run before training, on the GPU server, so the first rollouts do not stall
pulling gigabytes each -- and so the disk budget is checked before a run
starts rather than discovered mid-run:

    python -m reliquary_swe.images --split polyglot --num-tasks 300 \\
        | xargs -P 4 -n 1 docker pull -q

Same selection arguments as `SweTasksetConfig`, read through the taskset
itself, so the list cannot drift from what a run with the same arguments
actually provisions.
"""

from __future__ import annotations

import argparse

import verifiers.v1 as vf

from reliquary_swe import corpus


def images(
    split: str,
    num_images: int = corpus.DEFAULT_SWESMITH_IMAGES,
    max_test_count: int | None = None,
    num_tasks: int | None = None,
) -> list[str]:
    config = vf.taskset_config_type("reliquary-swe")(
        id="reliquary-swe",
        split=split,
        num_images=num_images,
        max_test_count=max_test_count,
        num_tasks=num_tasks,
    )
    return list(dict.fromkeys(task.data.image for task in vf.load_taskset(config)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=["eval", "train", "polyglot", "r2e"])
    parser.add_argument("--num-images", type=int, default=corpus.DEFAULT_SWESMITH_IMAGES)
    parser.add_argument("--max-test-count", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args()
    for image in images(args.split, args.num_images, args.max_test_count, args.num_tasks):
        print(image)


if __name__ == "__main__":
    main()
