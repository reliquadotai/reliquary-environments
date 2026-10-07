"""Recompute the `tmax` split's selection manifest and base-image files.

Inputs, all pinned: TMax-15K's `tasks.zip` (`tmax.SOURCE_*`), Ubuntu 22.04's
package index at `tmax_select.APT_SNAPSHOT`, Terminal-Bench 2.0 and 2.1 at
the digests in `tmax_select.TB_DATASETS`, and optionally the box phase's
results (one JSON per task, written by `scripts/tmax_validate.py`). Outputs:
`reliquary_terminal/tmax_manifest.json` and `reliquary_terminal/tmax_base/`.
No container is involved; the network is used only to fetch the pinned
inputs the first time.

    uv run python scripts/tmax_manifest.py [--source tasks.zip] [--validation DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reliquary_terminal import tmax, tmax_select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="tasks.zip or its unpacked tree (default: download the pinned zip)")
    parser.add_argument("--validation", type=Path, help="directory of the box phase's per-task JSON results")
    parser.add_argument(
        "--base-image",
        help="REPOSITORY@sha256:... of the pushed base image, when the box phase ran on a local "
        "image id (sha256:...). Pass --validated-id too: the local id the results name.",
    )
    parser.add_argument("--validated-id", help="the local image id (sha256:...) the validation ran on")
    parser.add_argument("--out", type=Path, default=tmax_select.MANIFEST)
    parser.add_argument("--base-dir", type=Path, default=tmax_select.BASE_DIR)
    args = parser.parse_args()

    started = time.monotonic()
    source = tmax.Source(args.source or tmax.download_source())
    index = tmax_select.AptIndex.download()
    contamination = tmax_select.Contamination.from_dirs(tmax_select.terminal_bench_dirs())
    validation = None
    if args.validation:
        validation = {}
        for path in sorted(args.validation.glob("*.json")):
            result = json.loads(path.read_text())
            validation[result["task_id"]] = result
    manifest, base = tmax_select.build_manifest(source, index, contamination, validation)
    if args.base_image:
        manifest = tmax_select.repin_base_image(manifest, args.base_image, args.validated_id)
    args.out.write_text(tmax_select.dump_manifest(manifest))
    args.base_dir.mkdir(parents=True, exist_ok=True)
    for name, content in base.items():
        (args.base_dir / name).write_text(content)
    json.dump(manifest["counts"], sys.stdout, indent=1)
    print(f"\n{time.monotonic() - started:.0f} s; wrote {args.out} and {args.base_dir}/")


if __name__ == "__main__":
    main()
