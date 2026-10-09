"""The published layout of the curated dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reliquary_competitive_code.build.validate import Curated
from reliquary_competitive_code.layout import (  # noqa: F401  (re-exported)
    REFERENCES_FILE,
    ROW_GROUP_SIZE,
    SPLITS,
    problems_file,
)

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(curated: Sequence[Curated], out: Path) -> dict[str, str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(curated, key=lambda c: c.problem_id)
    digests = {}
    for split in SPLITS:
        rows = [c for c in ordered if c.split == split]
        table = pa.table({
            "problem_id": [c.problem_id for c in rows],
            "statement": [c.statement for c in rows],
            "tests_json": [json.dumps([[t.stdin, t.stdout] for t in c.tests], ensure_ascii=False) for c in rows],
            "time_limit_s": pa.array([c.time_limit_s for c in rows], type=pa.float64()),
            "origin_json": [json.dumps(c.origin, sort_keys=True) for c in rows],
        })
        path = out / problems_file(split)
        pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
        digests[path.name] = _sha256(path)
    references = pa.table({
        "problem_id": [c.problem_id for c in ordered],
        "code": [c.reference for c in ordered],
    })
    path = out / REFERENCES_FILE
    pq.write_table(references, path, compression="zstd")
    digests[path.name] = _sha256(path)
    return digests
