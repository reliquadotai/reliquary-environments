"""Splits, order, the gate, and what each task declares."""

import pytest

from reliquary_terminal_bench import gate, images, source, taskset
from reliquary_terminal_bench.taskset import TerminalBenchConfig, TerminalBenchTaskset


def tasks(split, **kw):
    return TerminalBenchTaskset(TerminalBenchConfig(split=split, id="reliquary-terminal-bench", **kw))


def test_the_split_is_required():
    with pytest.raises(ValueError, match="split is required"):
        len(TerminalBenchTaskset(TerminalBenchConfig(id="reliquary-terminal-bench")))


def test_eval_all_is_every_task_in_order():
    ts = tasks("eval_all")
    assert len(ts) == 89
    loaded = list(ts.load())
    assert [t.key for t in loaded] == list(source.task_ids())
    assert all(ts.task_at(i).key == t.key for i, t in enumerate(loaded))
    with pytest.raises(IndexError):
        ts.task_at(89)


@pytest.mark.parametrize("split", ["eval", "eval_offline"])
def test_a_gated_split_refuses_until_gated(monkeypatch, split):
    monkeypatch.setattr(gate, taskset.GATES[split], None)
    taskset.split_ids.cache_clear()
    try:
        with pytest.raises(ValueError, match="not gated"):
            len(tasks(split))
    finally:
        taskset.split_ids.cache_clear()


def test_eval_is_the_verified_tasks_less_the_excluded(monkeypatch):
    excluded = next(iter(gate.EXCLUDED))
    monkeypatch.setattr(gate, "VERIFIED", frozenset({"regex-log", "fix-git", excluded}))
    taskset.split_ids.cache_clear()
    try:
        assert [t.key for t in tasks("eval").load()] == ["fix-git", "regex-log"]
    finally:
        taskset.split_ids.cache_clear()


def test_excluded_tasks_exist():
    assert set(gate.EXCLUDED) <= set(source.task_ids())
    assert len(gate.EXCLUDED) == 10


def test_task_data_is_harbors_box_on_our_image():
    task = tasks("eval_all", tasks=["fix-git"]).task_at(source.task_ids().index("fix-git"))
    data = task.data
    assert data.image == images.image_ref("fix-git")
    assert data.upstream_image == "alexgshaw/fix-git:20260403"
    assert data.upstream_image in images.upstream()
    assert data.workdir == "/app/personal-site"
    assert data.network_allow == []
    assert (data.resources.cpu, data.resources.memory, data.resources.disk) == (1.0, 2.0, 10.0)
    assert data.timeout.agent == float(source.task_toml("fix-git")["agent"]["timeout_sec"])
    assert data.timeout.scoring == 810.0
    assert [a.source for a in data.artifacts] == ["/logs/artifacts"]
    assert data.prompt == source.instruction("fix-git")


def test_the_task_hash_does_not_depend_on_the_machine():
    a = tasks("eval_all").task_at(0)
    b = tasks("eval_all").task_at(0)
    assert a.hash == b.hash
    assert "task_dir" not in a.data.model_dump() or not a.data.task_dir


def test_every_upstream_image_is_pinned():
    pinned = images.upstream()
    for task_id in source.task_ids():
        image = source.task_toml(task_id)["environment"]["docker_image"]
        assert pinned[image].startswith(image.rsplit(":", 1)[0] + "@sha256:")


def test_the_gate_partitions_the_89_tasks():
    verified = gate.VERIFIED if gate.VERIFIED is not None else gate.VERIFIED_OFFLINE
    excluded, unverified = set(gate.EXCLUDED), set(gate.NOT_VERIFIED)
    assert not (verified & excluded or verified & unverified or excluded & unverified)
    assert verified | excluded | unverified == set(source.task_ids())


def test_offline_is_a_subset_of_online():
    assert len(gate.VERIFIED_OFFLINE) == 49 and len(gate.VERIFIED) == 65
    taskset.split_ids.cache_clear()
    assert len(tasks("eval_offline")) == 49
    if gate.VERIFIED is not None:
        assert gate.VERIFIED_OFFLINE <= gate.VERIFIED


def test_every_task_image_is_locked():
    locked = images.lock()
    assert set(locked) == {images.image_ref(t) for t in source.task_ids()}
    assert all(v.startswith(images.REPOSITORY + "@sha256:") for v in locked.values())
