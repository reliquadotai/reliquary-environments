"""The prompt corpus: blocks, thinking modes, splits, and the segments jobs own.

The corpus is a dataset on the Hugging Face Hub, `REPOSITORY` at `REVISION`,
one gzip file per block under `data/`; the wheel ships only its pins. A file is
downloaded the first time a row of its block is asked for, and refused unless
its sha256 is the one `_pins.py` records (the Hub's own LFS digest for it).
`RELIQUARY_GENERAL_DATA` may name a local directory holding the same files,
for an offline host or a build; the digests are checked all the same.

A row is one prompt served in one thinking mode, so a prompt meant to be
answered both ways appears twice, once in each mode's run. Within a block the
file holds every direct row, then every thinking row; within each run the
single-turn rows (one user message, no tools, no system message) come first,
then the rest, each part shuffled by key. Within a split the blocks follow
`BLOCKS`. A corpus job has one renderer and one contiguous range of rows, so
the train split is a sequence of segments, one per (block, mode), and a job is
one segment or a slice of one; `Segment.single_turn` rows from its start are
the part a one-turn renderer can serve.

Each row names its split, chosen from a hash of the prompt's key (both modes of
one prompt, and both halves of a clarification pair, share a split). The
digests, row counts and single-turn counts are pinned in `_pins.py`, written by
the build together with the files: a rebuilt corpus that drops or reorders
rows fails loudly instead of silently moving every job's range.

Blocks are read lazily and kept as raw lines; a row is parsed when asked for.
A job touching one segment never downloads or decompresses the other blocks.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from reliquary_general._pins import BLOCK_ROWS, BLOCK_SINGLE_TURN, FILE_SHA256

# The published corpus (public, 2026-10-10). Set by hand once the files are
# uploaded; the digests in `_pins.py` are the build's and match what the
# revision holds.
REPOSITORY = "ReliquaryForge/general-prompts-curated"
REVISION: str | None = "02b3ef7acc83c034e971e8f77905a7b1dcb80ca2"
LOCAL_DATA_ENV = "RELIQUARY_GENERAL_DATA"

BLOCKS = (
    "chat",
    "multiturn",
    "ifeval",
    "structured",
    "safety",
    "clarification",
    "identity",
    "tools_call",
    "tools_pivot",
)
MODES = ("direct", "thinking")
SPLITS = ("train", "eval", "qualification")
# The two renderers a Reliquary corpus job can name, by mode.
RENDERERS = {"direct": "chat-template-v1", "thinking": "chat-template-thinking-v1"}
# Blocks with no programmatic grader: their jobs keep every completion.
UNGRADED_BLOCKS = frozenset({"chat", "multiturn", "safety"})

# Suggested generation budget per segment, in new tokens. Thinking budgets for
# instructions and structured outputs are deliberately short: on IFBench the
# distilled SFT v2 left 83 of 300 answers empty, ruminating over counting
# constraints until the budget ran out; a cap turns that into a truncation the
# export drops rather than a habit the student learns.
MAX_NEW_TOKENS = {
    ("chat", "direct"): 4096, ("chat", "thinking"): 16384,
    ("multiturn", "direct"): 4096, ("multiturn", "thinking"): 16384,
    ("ifeval", "direct"): 4096, ("ifeval", "thinking"): 8192,
    ("structured", "direct"): 4096, ("structured", "thinking"): 8192,
    ("safety", "direct"): 2048, ("safety", "thinking"): 4096,
    ("clarification", "direct"): 2048, ("clarification", "thinking"): 4096,
    ("identity", "direct"): 1024, ("identity", "thinking"): 2048,
    ("tools_call", "direct"): 2048, ("tools_call", "thinking"): 8192,
    ("tools_pivot", "direct"): 2048, ("tools_pivot", "thinking"): 8192,
}


def data_file(block: str) -> str:
    """The block's path in the dataset repository (and in a local copy)."""
    return f"data/{block}.jsonl.gz"


_download_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(block: str) -> bytes:
    """The block's gzip file, local or downloaded, checked against its pin."""
    name = data_file(block)
    local = os.environ.get(LOCAL_DATA_ENV)
    if local:
        path = Path(local) / name
    else:
        if REVISION is None:
            raise RuntimeError(
                f"no corpus revision is pinned in reliquary_general.corpus; set {LOCAL_DATA_ENV}"
            )
        from huggingface_hub import hf_hub_download

        with _download_lock:
            path = Path(hf_hub_download(REPOSITORY, name, repo_type="dataset", revision=REVISION))
    actual = _sha256(path)
    if actual != FILE_SHA256[block]:
        raise RuntimeError(f"{name}: sha256 is {actual}, pinned {FILE_SHA256[block]}")
    return path.read_bytes()


@dataclass(frozen=True, slots=True)
class Row:
    """One prompt in one thinking mode."""

    key: str
    split: str
    block: str
    mode: str
    source: str
    single_turn: bool
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] | None
    check: dict[str, Any] | None
    origin: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous run of one split's rows sharing a block and a mode."""

    block: str
    mode: str
    start: int
    count: int
    single_turn: int

    @property
    def name(self) -> str:
        return f"{self.block}/{self.mode}"

    @property
    def renderer(self) -> str:
        return RENDERERS[self.mode]

    @property
    def graded(self) -> bool:
        return self.block not in UNGRADED_BLOCKS

    @property
    def max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS[(self.block, self.mode)]

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def single_turn_stop(self) -> int:
        """Rows [start, single_turn_stop) are one user message, no tools, no system."""
        return self.start + self.single_turn


_HEAD = b'{"split":"'


def _head(line: bytes, block: str) -> tuple[str, str, bool]:
    """split, mode and single_turn, read from a row's first bytes.

    Rows are written `{"split":...,"mode":...,"single_turn":...,` first, so a
    block is laid out without parsing its rows; all three are checked again
    when a row is parsed.
    """
    try:
        if not line.startswith(_HEAD):
            raise ValueError
        end = line.index(b'"', 10)
        split = line[10:end].decode()
        if line[end : end + 10] != b'","mode":"':
            raise ValueError
        stop = line.index(b'"', end + 10)
        mode = line[end + 10 : stop].decode()
        rest = line[stop : stop + 24]
        if rest.startswith(b'","single_turn":true,'):
            single = True
        elif rest.startswith(b'","single_turn":false,'):
            single = False
        else:
            raise ValueError
    except ValueError:
        raise RuntimeError(f"{block}: malformed row") from None
    if split not in SPLITS or mode not in MODES:
        raise RuntimeError(f"{block}: row with split {split!r} and mode {mode!r}")
    return split, mode, single


@lru_cache(maxsize=None)
def _block_lines(block: str) -> dict[str, tuple[bytes, ...]]:
    """A block's raw lines by split, checked against the pinned digest and counts."""
    body = gzip.decompress(_fetch(block))
    by_split: dict[str, list[bytes]] = {split: [] for split in SPLITS}
    counts: dict[tuple[str, str], int] = {}
    singles: dict[tuple[str, str], int] = {}
    thinking_seen = False
    for line in body.split(b"\n"):
        if not line:
            continue
        split, mode, single = _head(line, block)
        key = (split, mode)
        seen = counts.get(key, 0)
        if mode == "thinking":
            thinking_seen = True
        elif thinking_seen:
            raise RuntimeError(f"{block}: a direct row after the thinking run")
        if single:
            if singles.get(key, 0) != seen:
                raise RuntimeError(f"{block}/{split}/{mode}: a single-turn row after the multi-turn part")
            singles[key] = seen + 1
        by_split[split].append(line)
        counts[key] = seen + 1
    for split in SPLITS:
        for mode in MODES:
            expected = BLOCK_ROWS[block][split][mode]
            if counts.get((split, mode), 0) != expected:
                raise RuntimeError(
                    f"{block}/{split}/{mode}: {counts.get((split, mode), 0)} rows, expected {expected}"
                )
            expected = BLOCK_SINGLE_TURN[block][split][mode]
            if singles.get((split, mode), 0) != expected:
                raise RuntimeError(
                    f"{block}/{split}/{mode}: {singles.get((split, mode), 0)} single-turn rows, expected {expected}"
                )
    return {split: tuple(lines) for split, lines in by_split.items()}


def _parse(line: bytes, block: str, split: str) -> Row:
    record = json.loads(line)
    if record["block"] != block or record["split"] != split or record["mode"] not in MODES:
        raise RuntimeError(f"row {record.get('key')} is filed under the wrong block, split or mode")
    messages = tuple(record["messages"])
    if not messages or messages[-1]["role"] not in ("user", "tool"):
        raise RuntimeError(f"row {record['key']} does not end on a user or tool turn")
    tools = tuple(record["tools"]) if record["tools"] else None
    single = len(messages) == 1 and messages[0]["role"] == "user" and tools is None
    if record["single_turn"] is not single:
        raise RuntimeError(f"row {record['key']} is filed as single_turn={record['single_turn']}, it is not")
    return Row(
        key=record["key"],
        split=split,
        block=block,
        mode=record["mode"],
        source=record["source"],
        single_turn=single,
        messages=messages,
        tools=tools,
        check=record["check"],
        origin=record["origin"],
    )


@lru_cache(maxsize=None)
def segments(split: str = "train") -> tuple[Segment, ...]:
    """The split's rows as (block, mode) runs, in serving order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    out: list[Segment] = []
    start = 0
    for block in BLOCKS:
        for mode in MODES:
            count = BLOCK_ROWS[block][split][mode]
            if count:
                out.append(Segment(block, mode, start, count, BLOCK_SINGLE_TURN[block][split][mode]))
                start += count
    return tuple(out)


def segment(name: str, split: str = "train") -> Segment:
    for candidate in segments(split):
        if candidate.name == name:
            return candidate
    raise KeyError(name)


def length(split: str = "train") -> int:
    found = segments(split)
    return found[-1].stop if found else 0


@lru_cache(maxsize=None)
def _starts(split: str) -> tuple[int, ...]:
    return tuple(s.start for s in segments(split))


def locate(index: int, split: str = "train") -> tuple[Segment, int]:
    """The segment an index falls in, and the index within it."""
    if not 0 <= index < length(split):
        raise IndexError(f"{split} has {length(split)} rows, index {index} is outside")
    found = segments(split)[bisect.bisect_right(_starts(split), index) - 1]
    return found, index - found.start


@lru_cache(maxsize=4096)
def row(index: int, split: str = "train") -> Row:
    found, offset = locate(index, split)
    lines = _block_lines(found.block)[split]
    # Within a block and split the direct rows come first, then the thinking rows.
    before = BLOCK_ROWS[found.block][split]["direct"] if found.mode == "thinking" else 0
    parsed = _parse(lines[before + offset], found.block, split)
    if parsed.mode != found.mode:
        raise RuntimeError(f"row {parsed.key} is out of its mode's run")
    return parsed


__all__ = [
    "BLOCKS",
    "LOCAL_DATA_ENV",
    "MAX_NEW_TOKENS",
    "MODES",
    "RENDERERS",
    "REPOSITORY",
    "REVISION",
    "SPLITS",
    "UNGRADED_BLOCKS",
    "Row",
    "Segment",
    "data_file",
    "length",
    "locate",
    "row",
    "segment",
    "segments",
]
