"""The taskset as a signed-episode sandbox reads it (docs/env-norm.md)."""

import json
import pickle
import re
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.utils.loaders import taskset_class, taskset_config_type

from reliquary_deepswe import gate, source, taskset
from reliquary_deepswe.task import PATCH_PATH, DeepSWETask
from reliquary_deepswe.taskset import DeepSWEConfig, DeepSWETaskset

LOCK = Path(taskset.__file__).with_name("sandbox-images.lock.json")
DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


def eval_all() -> DeepSWETaskset:
    return DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe"))


def test_verifiers_finds_the_taskset_and_its_config():
    assert taskset_class("reliquary-deepswe") is DeepSWETaskset
    assert taskset_config_type("reliquary-deepswe") is DeepSWEConfig


def test_split_is_required():
    with pytest.raises(ValueError, match="split is required"):
        len(DeepSWETaskset(DeepSWEConfig(id="reliquary-deepswe")))


def test_eval_refuses_until_gated():
    assert gate.VERIFIED is None
    with pytest.raises(ValueError, match="not gated yet"):
        len(DeepSWETaskset(DeepSWEConfig(split="eval", id="reliquary-deepswe")))


def test_eval_serves_the_verified_tasks_less_the_excluded(monkeypatch):
    ids = source.task_ids()
    monkeypatch.setattr(gate, "VERIFIED", frozenset(ids[:10]))
    monkeypatch.setattr(gate, "EXCLUDED", {ids[3]: "test"})
    taskset.split_ids.cache_clear()
    try:
        served = DeepSWETaskset(DeepSWEConfig(split="eval", id="reliquary-deepswe"))
        assert len(served) == 9
        assert [served.task_at(i).key for i in range(9)] == [t for t in ids[:10] if t != ids[3]]
        assert served.task_at(0).data.split == "eval"
        monkeypatch.setattr(gate, "VERIFIED", frozenset({"no-such-task"}))
        taskset.split_ids.cache_clear()
        with pytest.raises(ValueError, match="unknown tasks"):
            len(served)
    finally:
        taskset.split_ids.cache_clear()


def test_eval_all_is_every_task_in_order_and_task_at_matches_load():
    tasks = eval_all()
    assert len(tasks) == 113
    loaded = list(tasks.load())
    assert [t.key for t in loaded] == list(source.task_ids())
    for index in (0, 56, 112):
        assert tasks.task_at(index).hash == loaded[index].hash
    with pytest.raises(IndexError):
        tasks.task_at(113)


def test_task_filter_keeps_split_indices():
    ids = source.task_ids()
    picked = DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe",
                                          tasks=[ids[5], ids[7]]))
    assert [t.data.idx for t in picked.load()] == [5, 7]


def test_task_data_follows_task_toml():
    for task in eval_all().load():
        data, toml = task.data, source.task_toml(task.key)
        assert data.prompt == source.instruction(task.key)
        assert data.image == toml["environment"]["docker_image"]
        assert data.workdir == "/app" and data.network_allow == []
        assert data.base_commit == toml["metadata"]["base_commit_hash"]
        assert data.language == toml["metadata"]["language"]
        assert data.verifier is None and data.task_dir == "" and not data.collect
        assert [a.source for a in data.artifacts] == [PATCH_PATH]
        assert data.resources == vf.TaskResources(cpu=2.0, memory=8.0, disk=20.0)


def test_timeouts_fit_the_bridge():
    """Setup in [35, 600] s; each grading step (max of finalize and scoring) in
    [70, 810] s for a task with grading_setup; the agent's own 5400 s."""
    timeout = eval_all().task_at(0).data.timeout
    assert 35 <= timeout.setup <= 600
    assert max(timeout.finalize, timeout.scoring) == 810
    assert timeout.agent == 5400


def test_hash_is_machine_independent_and_binds_the_staged_tests():
    task = eval_all().task_at(0)
    dumped = json.dumps(task.data.model_dump(mode="json"))
    assert "/home/" not in dumped and str(source.TASKS) not in dumped
    assert task.data.tests_sha256 in dumped
    assert len({t.hash for t in eval_all().load()}) == 113


def test_tasks_pickle_and_carry_one_reward():
    task = eval_all().task_at(0)
    clone = pickle.loads(pickle.dumps(task))
    assert isinstance(clone, DeepSWETask) and clone.hash == task.hash
    assert [fn.__name__ for fn in task.hooks("reward")] == ["solved"]
    assert task.runtime_env() == {}
    assert callable(task.grading_setup)


def test_the_lock_pins_every_task_image_by_digest():
    document = json.loads(LOCK.read_text())
    assert set(document) == {"version", "images"} and document["version"] == 1
    images = document["images"]
    assert set(images) == {t.data.image for t in eval_all().load()}
    for tag, pinned in images.items():
        assert DIGEST.fullmatch(pinned), tag
        assert pinned.split("@")[0] == tag.rsplit(":", 1)[0]
    assert LOCK.read_text() == json.dumps(document, indent=1, sort_keys=True) + "\n"
