"""Splits, order and the gate."""

import pytest

from reliquary_nl2repo import conformance, gate, source, taskset
from reliquary_nl2repo.taskset import NL2RepoConfig, NL2RepoTaskset


def make(split, **kw):
    return NL2RepoTaskset(NL2RepoConfig(split=split, id="reliquary-nl2repo", **kw))


def test_eval_all_serves_every_task_in_order():
    tasks = make("eval_all")
    assert len(tasks) == 104
    loaded = list(tasks.load())
    assert [t.key for t in loaded] == list(source.task_ids())
    assert [t.data.idx for t in loaded] == list(range(104))
    assert tasks.task_at(7).hash == loaded[7].hash


def test_a_subset_keeps_the_split_indices():
    loaded = list(make("eval_all", tasks=["six", "tinydb"]).load())
    ids = source.task_ids()
    assert [(t.key, t.data.idx) for t in loaded] == [("six", ids.index("six")),
                                                     ("tinydb", ids.index("tinydb"))]


def test_the_split_is_required_and_known():
    with pytest.raises(ValueError, match="required"):
        len(NL2RepoTaskset(NL2RepoConfig(id="x")))
    with pytest.raises(ValueError, match="unknown split"):
        taskset.split_ids("train")
    with pytest.raises(IndexError):
        make("eval_all").task_at(104)


def test_eval_follows_the_gate(monkeypatch):
    taskset.split_ids.cache_clear()
    monkeypatch.setattr(gate, "VERIFIED", None)
    with pytest.raises(ValueError, match="not gated"):
        taskset.split_ids("eval")
    taskset.split_ids.cache_clear()
    monkeypatch.setattr(gate, "VERIFIED", frozenset({"six", "tinydb", "retrying"}))
    monkeypatch.setattr(gate, "EXCLUDED", {"retrying": "test"})
    assert taskset.split_ids("eval") == ("six", "tinydb")
    taskset.split_ids.cache_clear()
    monkeypatch.setattr(gate, "VERIFIED", frozenset({"nope"}))
    with pytest.raises(ValueError, match="unknown"):
        taskset.split_ids("eval")
    taskset.split_ids.cache_clear()


def test_the_gate_names_known_tasks():
    if gate.VERIFIED is not None:
        assert gate.VERIFIED <= set(source.task_ids()) and gate.GOLD_CHECK


def test_conformance_has_no_reference_and_attacks_the_goldens():
    assert conformance.reference_calls("eval_all", 0) is None
    cases = conformance.conformance_cases("eval_all")
    assert {c["index"] for c in cases} == {source.task_ids().index(g) for g in conformance.GOLDENS}
    assert all(c["expect"] == 0.0 for c in cases) and len(cases) == 8 * len(conformance.GOLDENS)
    assert conformance.conformance_cases("nope") == []


def test_upload_calls_rebuild_the_bytes(tmp_path):
    import base64, subprocess
    data = bytes(range(256)) * 600
    target = tmp_path / "out.bin"
    calls = conformance.upload_calls(data, str(target))
    assert len(calls) > 2
    script = "\n".join(args["command"] for _, args in calls).replace(conformance.STAGE,
                                                                     str(tmp_path / "s.b64"))
    (tmp_path / "s.sh").write_text(script)
    subprocess.run(["bash", str(tmp_path / "s.sh")], check=True)
    assert target.read_bytes() == data
