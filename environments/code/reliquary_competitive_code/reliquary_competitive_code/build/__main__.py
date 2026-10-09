"""Build the curated dataset: python -m reliquary_competitive_code.build ..."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from reliquary_competitive_code.build.filters import HeldOutIndex, load_lcb_statements
from reliquary_competitive_code.build.io import write_dataset
from reliquary_competitive_code.build.pipeline import build
from reliquary_competitive_code.sources import deepcoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepcoder", type=Path, required=True)
    parser.add_argument("--lcb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N source rows, for a smoke run")
    args = parser.parse_args()
    rows = deepcoder.rows(args.deepcoder)
    if args.limit is not None:
        rows = itertools.islice(rows, args.limit)
    curated, report = build(rows, HeldOutIndex(load_lcb_statements(args.lcb)), workers=args.workers)
    print(json.dumps(report, sort_keys=True))
    overloaded = report.get("dropped_harness_overload", 0)
    if overloaded:
        # A curated dataset must not depend on how loaded the build box was.
        raise SystemExit(
            f"{overloaded} problem(s) dropped as harness_overload: the build box "
            "starved the reference runs; rerun on a quieter box (fewer --workers). "
            "No dataset written."
        )
    files = write_dataset(curated, args.out)
    (args.out / "build_report.json").write_text(
        json.dumps({"report": report, "files": files}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
