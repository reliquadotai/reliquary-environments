"""The pinned corpus: pins, segments, single-turn parts, task shape, and the goldens."""

from __future__ import annotations

import gzip
import hashlib
import importlib.resources
import json

import pytest

from reliquary_general import GENERATION_SYSTEM, GeneralPromptsEnvironment, Ungraded
from reliquary_general._pins import BLOCK_ROWS, BLOCK_SINGLE_TURN, FILE_SHA256
from reliquary_general.corpus import (
    BLOCKS,
    LOCAL_DATA_ENV,
    MAX_NEW_TOKENS,
    MODES,
    RENDERERS,
    REPOSITORY,
    REVISION,
    SPLITS,
    UNGRADED_BLOCKS,
    _fetch,
    data_file,
    length,
    locate,
    row,
    segment,
    segments,
)
from reliquary_general.environment import _build_task

ROOT = importlib.resources.files("reliquary_general")
GOLDENS = [json.loads(line) for line in ROOT.joinpath("goldens/reference.jsonl").read_text().splitlines()]


def test_every_block_and_mode_has_a_budget():
    assert set(MAX_NEW_TOKENS) == {(b, m) for b in BLOCKS for m in MODES}


@pytest.mark.parametrize("split", SPLITS)
def test_segments_tile_the_split_in_block_order(split):
    found = segments(split)
    position = 0
    for s in found:
        assert s.start == position
        position = s.stop
    assert position == length(split) == len(GeneralPromptsEnvironment(split))
    order = [BLOCKS.index(s.block) * 2 + MODES.index(s.mode) for s in found]
    assert order == sorted(order)


def test_train_has_both_modes_for_every_block():
    names = {s.name for s in segments("train")}
    assert names == {f"{b}/{m}" for b in BLOCKS for m in MODES}


def test_train_dominates_the_corpus():
    train = length("train")
    held = length("eval") + length("qualification")
    assert held / (train + held) < 0.06


@pytest.mark.parametrize("block", BLOCKS)
def test_first_and_last_row_of_each_train_segment(block):
    for mode in MODES:
        s = segment(f"{block}/{mode}")
        for index in (s.start, s.stop - 1):
            found = row(index)
            assert (found.block, found.mode) == (block, mode)
            assert locate(index) == (s, index - s.start)
            task = _build_task(index, "train")
            metadata = task["metadata"]
            assert metadata["renderer"] == RENDERERS[mode]
            assert metadata["segment"] == s.name
            assert metadata["system"] == GENERATION_SYSTEM
            assert metadata["system_scope"] == "generation-only"
            assert metadata["graded"] == (block not in UNGRADED_BLOCKS)
            assert metadata["messages"][-1]["role"] in ("user", "tool")
            users = [m["content"] for m in metadata["messages"] if m["role"] == "user"]
            assert task["prompt"] == users[-1] and task["prompt"].strip()
            if block.startswith("tools"):
                assert metadata["tools"]
            else:
                assert metadata["tools"] is None


def test_both_modes_of_a_prompt_share_a_split():
    keys: dict[str, set[str]] = {}
    for split in SPLITS:
        for s in segments(split):
            if s.block != "identity":
                continue
            for index in range(s.start, s.stop):
                keys.setdefault(row(index, split).key, set()).add(split)
    assert all(len(splits) == 1 for splits in keys.values())


def test_task_identity_follows_the_prompt_and_mode():
    s = segment("identity/direct")
    a = _build_task(s.start, "train")
    assert a["id"] == _build_task(s.start, "train")["id"]
    assert a["id"] != _build_task(s.start + 1, "train")["id"]


@pytest.mark.parametrize("block", sorted(UNGRADED_BLOCKS))
def test_ungraded_blocks_refuse_to_grade(block):
    s = segment(f"{block}/direct")
    with pytest.raises(Ungraded):
        GeneralPromptsEnvironment().grade(s.start, "Any answer at all.")


def test_pins_cover_every_block():
    assert set(BLOCK_ROWS) == set(BLOCK_SINGLE_TURN) == set(FILE_SHA256) == set(BLOCKS)


def test_the_corpus_is_pinned_to_a_hub_revision():
    assert REPOSITORY == "ReliquaryForge/general-prompts-curated"
    assert REVISION is not None and len(REVISION) == 40


# Blocks whose every row is one user turn, and blocks with none.
ALL_SINGLE = {"chat", "safety", "identity", "clarification"}
NO_SINGLE = {"multiturn", "tools_call", "tools_pivot"}


@pytest.mark.parametrize("split", SPLITS)
def test_single_turn_rows_open_each_segment(split):
    for s in segments(split):
        assert 0 <= s.single_turn <= s.count
        if s.block in ALL_SINGLE:
            assert s.single_turn == s.count, s.name
        if s.block in NO_SINGLE:
            assert s.single_turn == 0, s.name
        probes = {s.start, s.single_turn_stop - 1, s.single_turn_stop, s.stop - 1}
        for index in sorted(i for i in probes if s.start <= i < s.stop):
            task = _build_task(index, split)
            single = index < s.single_turn_stop
            assert task["metadata"]["single_turn"] is single, (s.name, index)
            if single:
                assert task["metadata"]["tools"] is None
                assert [m["role"] for m in task["metadata"]["messages"]] == ["user"]
                assert task["metadata"]["messages"][0]["content"] == task["prompt"]


def test_ifeval_and_structured_mix_both_parts():
    for name in ("ifeval/direct", "ifeval/thinking", "structured/direct", "structured/thinking"):
        s = segment(name)
        assert 0 < s.single_turn < s.count, name


def test_a_block_whose_file_does_not_match_its_pin_is_refused(tmp_path, monkeypatch):
    path = tmp_path / data_file("identity")
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b'{"split":"train"}\n'))
    monkeypatch.setenv(LOCAL_DATA_ENV, str(tmp_path))
    with pytest.raises(RuntimeError, match="sha256"):
        _fetch("identity")


@pytest.mark.parametrize("golden", GOLDENS, ids=[g["block"] for g in GOLDENS])
def test_goldens_replay(golden):
    environment = GeneralPromptsEnvironment(golden["split"])
    task = environment.task(golden["index"])
    assert task["id"] == golden["task_id"]
    digest = hashlib.sha256(json.dumps(task["metadata"]["messages"], sort_keys=True).encode()).hexdigest()
    assert digest == golden["prompt_sha256"]
    if not golden["graded"]:
        with pytest.raises(Ungraded):
            environment.grade(golden["index"], "anything")
        return
    assert environment.grade(golden["index"], golden["completion"])["reward"] == 1.0
    assert environment.grade(golden["index"], golden["thinking_completion"])["reward"] == 1.0
    assert environment.grade(golden["index"], golden["wrong_completion"])["reward"] == 0.0
    assert environment.grade(golden["index"], golden["completion"])["state_digest"] == golden["state_digest"]


def test_artifact_pins_every_shipped_file():
    artifact = json.loads(ROOT.joinpath("artifact.json").read_text())
    for name, digest in artifact["files"].items():
        assert hashlib.sha256(ROOT.parent.joinpath(name).read_bytes()).hexdigest() == digest, name
    shipped = {
        str(p.relative_to(ROOT.parent))
        for p in ROOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "artifact.json"
    }
    assert shipped == set(artifact["files"])
    # The corpus is the pinned Hub dataset, not part of the wheel.
    assert not any(name.endswith(".jsonl.gz") for name in shipped)
