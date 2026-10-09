"""Download the pinned upstream data a build reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from reliquary_competitive_code.build.filters import LCB_REPOSITORY, LCB_REVISION
from reliquary_competitive_code.sources import deepcoder


def fetch(out: Path) -> None:
    snapshot_download(
        deepcoder.REPOSITORY, repo_type="dataset", revision=deepcoder.REVISION,
        allow_patterns=[f"{config}/train-*" for config in deepcoder.CONFIGS],
        local_dir=Path(out) / "deepcoder",
    )
    snapshot_download(
        LCB_REPOSITORY, repo_type="dataset", revision=LCB_REVISION,
        allow_patterns=["test*.jsonl"], local_dir=Path(out) / "lcb",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    fetch(parser.parse_args().out)
