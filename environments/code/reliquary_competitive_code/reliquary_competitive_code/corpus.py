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
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.layout import REFERENCES_FILE, SPLITS, USES, problems_file, use_of_row


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repository: str
    revision: str
    files: dict[str, str]


# ReliquaryForge/competitive-code-curated, built 2026-10-09 from
# DeepCoder-Preview-Dataset @177913a7 (see the dataset's build_report.json).
PINNED: DatasetPin | None = DatasetPin(
    repository="ReliquaryForge/competitive-code-curated",
    revision="4ae529419ce2a1d2cc0d56fd1b061fc10af3322a",
    files={
        "problems-train.parquet": "c80d22ad0bc7054cc1ae5c0492bcc10fd0d96d3fe06f64a18cb413dd84f3a4b7",
        "problems-eval.parquet": "c6188cbb32d4eeae233470ac7061511b4e89953508da6506686845d1331e85f2",
        "problems-qualification.parquet": "a6b4edd4534d299772870970c9de856e2f690ea5504f6112ead0505d1a2deb6e",
        "references.parquet": "f86db0b03e4dc8bda2fa74f691be5f7046830e68e26f3382dee60cc0b0a50a27",
    },
)
# v1.1 row groups hold up to 64 problems x 4 MiB of tests: keep two decoded.
_CACHED_ROW_GROUPS = 2


@dataclass(frozen=True, slots=True)
class Problem:
    problem_id: str
    statement: str
    time_limit_s: float
    origin: dict

    @property
    def use(self) -> str:
        return use_of_row(self.problem_id, self.origin)


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

    def use_ranges(self, use: str) -> tuple[range, ...]:
        """The indices of train that serve `use` ("sft" or "rl"), as maximal
        contiguous ranges in index order: v1 is SFT then RL, sorted by
        problem id; v1.1 appends a second SFT range after them."""
        if use not in USES:
            raise ValueError(f"use must be one of {USES}")
        ranges, start = [], None
        uses = [p.use for p in self._problems["train"]]
        for index, current in enumerate([*uses, None]):
            if current == use and start is None:
                start = index
            elif current != use and start is not None:
                ranges.append(range(start, index))
                start = None
        return tuple(ranges)

    def use_range(self, use: str) -> range:
        """The one contiguous range of train that serves `use`; raises when
        there are several (use `use_ranges`)."""
        ranges = self.use_ranges(use)
        if len(ranges) > 1:
            raise RuntimeError(f"{use} is served by {len(ranges)} ranges of train: use use_ranges")
        if ranges:
            return ranges[0]
        uses = [p.use for p in self._problems["train"]]
        start = len(uses) if use == "rl" else 0
        return range(start, start)

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


_pinned: Corpus | None = None
_pinned_lock = threading.Lock()


def pinned_corpus() -> Corpus:
    """The pinned corpus, downloaded and verified once per process, even when
    several threads ask for it first at the same time."""
    global _pinned
    if _pinned is None:
        with _pinned_lock:
            if _pinned is None:
                _pinned = Corpus.pinned()
    return _pinned
