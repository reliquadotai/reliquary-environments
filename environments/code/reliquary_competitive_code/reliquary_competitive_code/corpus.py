"""The curated dataset this environment serves, pinned by revision and digest.

Built offline by `reliquary_competitive_code.build` and published to the Hub;
production only reads it. Statements and limits load eagerly, tests lazily by
row group and a few decoded row groups are kept (most recently used), so a
validator holds a handful of 64-problem groups rather than the whole split.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.layout import REFERENCES_FILE, SPLITS, problems_file


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repository: str
    revision: str
    files: dict[str, str]


PINNED: DatasetPin | None = None
_CACHED_ROW_GROUPS = 4


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
        self._cache: OrderedDict[tuple[str, int], list[str]] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def pinned(cls) -> Corpus:
        if PINNED is None:
            raise RuntimeError("no curated dataset is pinned in reliquary_competitive_code.corpus")
        expected = {problems_file(s) for s in SPLITS} | {REFERENCES_FILE}
        if set(PINNED.files) != expected:
            raise RuntimeError(f"the pin must name exactly {sorted(expected)}, not {sorted(PINNED.files)}")
        from huggingface_hub import hf_hub_download

        directory = None
        for name, digest in PINNED.files.items():
            path = Path(hf_hub_download(PINNED.repository, name, repo_type="dataset", revision=PINNED.revision))
            actual = _sha256(path)
            if actual != digest:
                raise RuntimeError(f"{name}: downloaded {actual}, pinned {digest}")
            directory = path.parent
        return cls(directory)

    def problems(self, split: str) -> list[Problem]:
        return self._problems[split]

    def tests(self, split: str, index: int) -> tuple[TestCase, ...]:
        starts = self._row_groups[split]
        group = max(g for g, start in enumerate(starts) if start <= index)
        raw = self._row_group(split, group)[index - starts[group]]
        return tuple(TestCase(a, b) for a, b in json.loads(raw))

    def _row_group(self, split: str, group: int) -> list[str]:
        key = (split, group)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        table = pq.ParquetFile(self._directory / problems_file(split)).read_row_group(group, columns=["tests_json"])
        decoded = table.column("tests_json").to_pylist()
        with self._lock:
            self._cache[key] = decoded
            self._cache.move_to_end(key)
            while len(self._cache) > _CACHED_ROW_GROUPS:
                self._cache.popitem(last=False)
        return decoded

    def reference(self, problem_id: str) -> str:
        if self._references is None:
            table = pq.read_table(self._directory / REFERENCES_FILE)
            self._references = dict(zip(table.column("problem_id").to_pylist(), table.column("code").to_pylist()))
        return self._references[problem_id]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def pinned_corpus() -> Corpus:
    return Corpus.pinned()
