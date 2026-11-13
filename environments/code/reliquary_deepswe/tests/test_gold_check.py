"""The gold-check runner's offline parts (no Docker): the episodes it plays, the gate it
writes."""

import importlib.util
import json
from pathlib import Path

import pytest

from reliquary_deepswe import gate, source

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gold_check.py"


@pytest.fixture
def gold_check():
    spec = importlib.util.spec_from_file_location("gold_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    defaults = dict(tasks=[], conformance=False, budget=700.0, runtime="runsc")
    return type("Args", (), {**defaults, **overrides})()


def test_one_reference_episode_per_task(gold_check):
    played = gold_check.episodes(args())
    assert [t.key for t, *_ in played] == list(source.task_ids())
    assert all(expect == 1.0 and name.startswith("reference_") for _, name, _, expect in played)
    assert len(gold_check.episodes(args(tasks=["etree-xml-diff-patch"]))) == 1
    assert {name.split("_")[0] for _, name, _, _ in gold_check.episodes(args(conformance=True))} > {"reference"}


def test_every_task_image_is_pinned_for_the_boxes(gold_check):
    assert {t.data.image for t, *_ in gold_check.episodes(args())} <= set(gold_check.LOCK)


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
    ]
    gold_check.write_gate(rows, args())
    namespace: dict = {}
    exec(compile(copy.read_text(), str(copy), "exec"), namespace)
    assert namespace["VERIFIED"] == frozenset({ids[0]})
    assert namespace["GOLD_CHECK"]["verified"] == f"1/{len(ids)}"
    assert namespace["EXCLUDED"] == gate.EXCLUDED
    assert copy.read_text().startswith(Path(gate.__file__).read_text().split("VERIFIED:")[0])
    json.dumps(namespace["GOLD_CHECK"])
