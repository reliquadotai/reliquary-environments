"""The pinned OpenCodeInstruct corpus, read one row-group at a time.

A curated subset of nvidia/OpenCodeInstruct: rows whose per-test cases are
structured and executable. Row order is the corpus identity, so index `i`
here is index `i` in Reliquary core at the same revision.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from reliquary_code.virtual_parquet import VirtualParquetDataset

OCI_REPO = "R0mAI/opencodeinstruct-curated"
OCI_REVISION = "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"

COLUMNS = ["input", "structured_cases"]


@lru_cache(maxsize=1)
def load_corpus() -> VirtualParquetDataset:
    return VirtualParquetDataset(OCI_REPO, OCI_REVISION, columns=COLUMNS)


def corpus_length() -> int:
    return len(load_corpus())


def get_problem(index: int) -> dict[str, Any]:
    row = load_corpus().get_row(int(index))
    # `structured_cases` is a JSON-encoded string column in the parquet
    # file, not a nested list column: every consumer (extraction.py's
    # `entry_function_name`/`contract_instruction`, taskset.py's `_reward`)
    # expects a `list[dict]`, so decode it here rather than at each call
    # site. `list(...)` on the raw string silently iterated characters
    # instead of cases -- caught generating this package's goldens against
    # the real corpus, since every unit test up to this point injected
    # already-decoded fixture rows.
    raw_cases = row["structured_cases"]
    return {
        "input": str(row["input"]),
        "structured_cases": json.loads(raw_cases) if raw_cases else [],
    }
