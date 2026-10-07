"""Sources in, curated problems out, with a count for every reason a row left."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from reliquary_competitive_code.build.dedup import deduplicate
from reliquary_competitive_code.build.filters import (
    CONTAMINATION_THRESHOLD,
    HeldOutIndex,
    example_disagrees,
    is_after_cutoff,
    is_multi_answer,
)
from reliquary_competitive_code.build.validate import Curated, curate
from reliquary_competitive_code.layout import SPLITS
from reliquary_competitive_code.sources.common import SourceRow, problem_id


def build(
    rows: Iterable[SourceRow], held_out: HeldOutIndex, *, workers: int
) -> tuple[list[Curated], dict[str, int]]:
    report: Counter[str] = Counter({"dropped_harness_overload": 0, "dropped_example_mismatch": 0})
    passed = []
    disagreeing = set()
    for row in rows:
        report["loaded"] += 1
        if is_after_cutoff(row):
            report["dropped_date"] += 1
        elif held_out.overlap(row.statement) >= CONTAMINATION_THRESHOLD:
            report["dropped_contaminated"] += 1
        elif is_multi_answer(row):
            report["dropped_multi_answer"] += 1
        else:
            if example_disagrees(row):
                disagreeing.add(problem_id(row.statement))
            passed.append(row)
    # Every copy goes: a copy whose tests skip the example is the same problem.
    kept = [row for row in passed if problem_id(row.statement) not in disagreeing]
    report["dropped_example_mismatch"] = len(passed) - len(kept)
    merged = deduplicate(kept)
    report["after_dedup"] = len(merged)
    candidates = []
    for row in merged:
        if row.references:
            candidates.append(row)
        else:
            report["dropped_no_reference"] += 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(curate, candidates))
    curated = sorted((o for o in outcomes if isinstance(o, Curated)), key=lambda c: c.problem_id)
    for outcome in outcomes:
        if not isinstance(outcome, Curated):
            report[f"dropped_{outcome.reason}"] += 1
    report["curated"] = len(curated)
    for split in SPLITS:
        report[f"split_{split}"] = sum(c.split == split for c in curated)
    return curated, dict(report)
