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


def append_dataset(base: Path, appended: Sequence[Curated], out: Path) -> dict[str, str]:
    """v1.1: the published dataset at `base`, with `appended` added after the
    last train row, all for SFT.

    Every base row keeps its index and its bytes of content: eval and
    qualification are copied as they are (same digests), train is rewritten
    with its rows unchanged and the new ones after them, sorted by problem id;
    the references gain the new problems ("" for one shipped on upstream
    tests). A new problem must not share an id with any base problem.
    """
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    known = set()
    for split in SPLITS:
        known |= set(pq.read_table(base / problems_file(split), columns=["problem_id"]).column(0).to_pylist())
    ordered = sorted(appended, key=lambda c: c.problem_id)
    ids = [c.problem_id for c in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("appended problems repeat an id")
    if clash := known & set(ids):
        raise ValueError(f"{len(clash)} appended problem(s) already in the base, e.g. {sorted(clash)[0]}")
    digests = {}
    for split in SPLITS:
        source, path = base / problems_file(split), out / problems_file(split)
        if split != "train":
            path.write_bytes(source.read_bytes())
            digests[path.name] = _sha256(path)
            continue
        old = pq.read_table(source)
        new = pa.table({
            "problem_id": ids,
            "statement": [c.statement for c in ordered],
            "tests_json": [json.dumps([[t.stdin, t.stdout] for t in c.tests], ensure_ascii=False) for c in ordered],
            "time_limit_s": pa.array([c.time_limit_s for c in ordered], type=pa.float64()),
            "origin_json": [json.dumps(c.origin, sort_keys=True) for c in ordered],
        }).cast(old.schema)
        pq.write_table(pa.concat_tables([old, new]), path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
        digests[path.name] = _sha256(path)
    old_refs = pq.read_table(base / REFERENCES_FILE)
    new_refs = pa.table({"problem_id": ids, "code": [c.reference for c in ordered]}).cast(old_refs.schema)
    path = out / REFERENCES_FILE
    pq.write_table(pa.concat_tables([old_refs, new_refs]), path, compression="zstd")
    digests[path.name] = _sha256(path)
    return digests
