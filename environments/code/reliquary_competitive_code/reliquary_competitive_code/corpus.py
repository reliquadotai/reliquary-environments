"""The curated dataset this environment serves, pinned by revision and digest.

Built offline by `reliquary_competitive_code.build` and published to the Hub;
production only reads it. Statements and limits load eagerly, tests lazily by
row group, so a validator holds one row group of tests at a time rather than
the whole split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.build.io import REFERENCES_FILE, problems_file
from reliquary_competitive_code.build.validate import SPLITS
from reliquary_competitive_code.judge import TestCase


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repository: str
    revision: str
    files: dict[str, str]


PINNED: DatasetPin | None = None


@dataclass(frozen=True, slots=True)
class Problem:
    problem_id: str
    statement: str
    time_limit_s: float
    origin: dict


class Corpus:
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._problems: dict[str, list[Problem]] = {}
        self._row_groups: dict[str, list[int]] = {}
        for split in SPLITS:
            path = self._directory / problems_file(split)
            table = pq.read_table(path, columns=["problem_id", "statement", "time_limit_s", "origin_json"])
            self._problems[split] = [
                Problem(r["problem_id"], r["statement"], float(r["time_limit_s"]), json.loads(r["origin_json"]))
                for r in table.to_pylist()
            ]
            metadata = pq.ParquetFile(path).metadata
            starts, total = [], 0
            for group in range(metadata.num_row_groups):
                starts.append(total)
                total += metadata.row_group(group).num_rows
            self._row_groups[split] = starts
        self._references: dict[str, str] | None = None

    @classmethod
    def pinned(cls) -> Corpus:
        if PINNED is None:
            raise RuntimeError("no curated dataset is pinned in reliquary_competitive_code.corpus")
        from huggingface_hub import hf_hub_download

        directory = None
        for name, digest in PINNED.files.items():
            path = Path(hf_hub_download(PINNED.repository, name, repo_type="dataset", revision=PINNED.revision))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"{name}: downloaded {actual}, pinned {digest}")
            directory = path.parent
        return cls(directory)

    def problems(self, split: str) -> list[Problem]:
        return self._problems[split]

    def tests(self, split: str, index: int) -> tuple[TestCase, ...]:
        starts = self._row_groups[split]
        group = max(g for g, start in enumerate(starts) if start <= index)
        table = pq.ParquetFile(self._directory / problems_file(split)).read_row_group(group, columns=["tests_json"])
        raw = table.column("tests_json")[index - starts[group]].as_py()
        return tuple(TestCase(a, b) for a, b in json.loads(raw))

    def reference(self, problem_id: str) -> str:
        if self._references is None:
            table = pq.read_table(self._directory / REFERENCES_FILE)
            self._references = dict(zip(table.column("problem_id").to_pylist(), table.column("code").to_pylist()))
        return self._references[problem_id]


@lru_cache(maxsize=1)
def pinned_corpus() -> Corpus:
    return Corpus.pinned()
