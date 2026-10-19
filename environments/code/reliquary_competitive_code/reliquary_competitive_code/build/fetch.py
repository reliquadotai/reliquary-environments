"""Download the pinned upstream data a build reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from reliquary_competitive_code.build.filters import LCB_REPOSITORY, LCB_REVISION
from reliquary_competitive_code.sources import deepcoder, openr1_codeforces


def fetch(out: Path, v11: bool = False) -> None:
    snapshot_download(
        deepcoder.REPOSITORY, repo_type="dataset", revision=deepcoder.REVISION,
        allow_patterns=[f"{config}/train-*" for config in deepcoder.CONFIGS],
        local_dir=Path(out) / "deepcoder",
    )
    snapshot_download(
        LCB_REPOSITORY, repo_type="dataset", revision=LCB_REVISION,
        allow_patterns=["test*.jsonl"], local_dir=Path(out) / "lcb",
    )
    if v11:
        # open-r1's statements and official tests (2.8 GB), read twice by
        # `build.extend`; its generated tests and Code-Contests-Plus are
        # streamed by `build.extend fetch`, one file at a time.
        snapshot_download(
            openr1_codeforces.REPOSITORY, repo_type="dataset", revision=openr1_codeforces.REVISION,
            allow_patterns=[openr1_codeforces.DATA_PATTERN], local_dir=Path(out) / "openr1",
        )
        snapshot_download(  # accepted Python references (5.6 GB, read by row group)
            openr1_codeforces.SUBMISSIONS_REPOSITORY, repo_type="dataset",
            revision=openr1_codeforces.SUBMISSIONS_REVISION,
            allow_patterns=[openr1_codeforces.SUBMISSIONS_PATTERN], local_dir=Path(out) / "openr1-submissions",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v11", action="store_true", help="also open-r1/codeforces data/ for build.extend")
    arguments = parser.parse_args()
    fetch(arguments.out, arguments.v11)
