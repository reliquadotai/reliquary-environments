"""The gold-check runner's offline parts (no Docker): the episodes it plays, the gate it
writes."""

import importlib.util
import json
from pathlib import Path

import pytest

from reliquary_swebench_pro import gate, source

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gold_check.py"
GOLDEN = "instance_flipt-io__flipt-406f9396ad65696d58865b3a6283109cd4eaf40e"


@pytest.fixture
def gold_check():
    spec = importlib.util.spec_from_file_location("gold_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    defaults = dict(tasks=[], conformance=False, noop=False, budget=700.0, runtime="runsc",
                    shard=None, out=None)
    return type("Args", (), {**defaults, **overrides})()


def flat(groups):
    return [case for group in groups for case in group]


def test_one_reference_episode_per_task(gold_check):
    played = flat(gold_check.episodes(args()))
    assert [t.key for t, *_ in played] == list(source.task_ids())
    assert all(expect == 1.0 and name.startswith("reference_") for _, name, _, expect in played)
    assert len(flat(gold_check.episodes(args(tasks=[GOLDEN])))) == 1
    both = gold_check.episodes(args(tasks=[GOLDEN], noop=True))
    assert len(both) == 1
    assert [(name.split("_")[0], expect) for _, name, _, expect in both[0]] == [
        ("reference", 1.0), ("noop", 0.0)]
    conformance = {name.split("_")[0] for _, name, _, _ in flat(gold_check.episodes(args(conformance=True)))}
    assert conformance > {"reference"}


def test_shards_are_contiguous_and_cover_every_task(gold_check):
    shards = [[t.key for t, *_ in flat(gold_check.episodes(args(shard=f"{k}/3")))]
              for k in range(3)]
    assert sum(shards, []) == list(source.task_ids())


def test_a_rerun_skips_the_cases_already_played(gold_check, tmp_path):
    out = tmp_path / "gold.jsonl"
    ids = source.task_ids()
    out.write_text(json.dumps({"task_id": ids[0], "case": "reference_0"}) + "\n"
                   + json.dumps({"task_id": ids[1], "case": "reference_1", "error": "x"}) + "\n")
    left = [t.key for t, *_ in flat(gold_check.episodes(args(tasks=list(ids[:3]), out=str(out))))]
    assert left == [ids[1], ids[2]]


def test_every_task_image_is_pinned_for_the_boxes(gold_check):
    assert {t.data.image for t, *_ in flat(gold_check.episodes(args()))} <= set(gold_check.LOCK)


def test_the_gate_keeps_passing_tasks_within_budget(gold_check, tmp_path, monkeypatch):
    copy = tmp_path / "gate.py"
    copy.write_text(Path(gate.__file__).read_text())
    monkeypatch.setattr(gate, "__file__", str(copy))
    ids = source.task_ids()
    rows = [
        {"task_id": ids[0], "case": "reference_0", "reward": 1.0, "grading_s": 100.0},
        {"task_id": ids[1], "case": "reference_1", "reward": 1.0, "grading_s": 900.0},
        {"task_id": ids[2], "case": "reference_2", "reward": 0.0, "grading_s": 50.0},
        {"task_id": ids[3], "case": "reference_3", "reward": 1.0, "grading_s": 10.0, "error": "x"},
        {"task_id": ids[4], "case": "reference_4", "reward": 1.0, "grading_s": 10.0},
        {"task_id": ids[4], "case": "noop_4", "reward": 1.0, "grading_s": 10.0},
        {"task_id": ids[0], "case": "noop_0", "reward": 0.0, "grading_s": 10.0},
    ]
    gold_check.write_gate(rows, args())
    namespace: dict = {}
    exec(compile(copy.read_text(), str(copy), "exec"), namespace)
    assert namespace["VERIFIED"] == frozenset({ids[0]})
    assert namespace["GOLD_CHECK"]["verified"] == f"1/{len(ids)}"
    assert namespace["EXCLUDED"] == gate.EXCLUDED
    assert copy.read_text().startswith(Path(gate.__file__).read_text().split("VERIFIED:")[0])
    json.dumps(namespace["GOLD_CHECK"])
