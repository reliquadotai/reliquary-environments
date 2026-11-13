"""The vendored tasks are upstream's at the pinned commit, bound to the package's `.py`
files through `_source.FILES`."""

import collections
import hashlib
import json
import re

import pytest

from reliquary_deepswe import _source, source

LANGUAGES = {"typescript": 35, "go": 34, "python": 34, "javascript": 5, "rust": 5}
TASK_FILES = {"task.toml", "instruction.md", "pre_artifacts.sh", "environment/Dockerfile",
              "tests/Dockerfile", "tests/test.sh", "tests/test.patch", "tests/grader.py",
              "tests/config.json", "solution/solve.sh", "solution/solution.patch"}


def test_pinned_commit():
    assert source.COMMIT == "8cae5984d5dd0ee37445beff0e928dc10c331116"
    assert source.UPSTREAM == "https://github.com/datacurve-ai/deep-swe"


def test_113_tasks_by_language():
    ids = source.task_ids()
    assert len(ids) == 113 and list(ids) == sorted(ids)
    counts = collections.Counter(source.task_toml(t)["metadata"]["language"] for t in ids)
    assert counts == LANGUAGES


def test_every_vendored_file_matches_its_digest_and_nothing_else_is_vendored():
    on_disk = {p.relative_to(source.TASKS).as_posix()
               for p in source.TASKS.rglob("*") if p.is_file()}
    assert on_disk == set(_source.FILES)
    for rel, digest in _source.FILES.items():
        assert hashlib.sha256((source.TASKS / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.parametrize("task_id", source.task_ids())
def test_each_task_has_upstreams_files(task_id):
    assert set(source.files_of(task_id)) == TASK_FILES


def test_a_changed_file_is_refused(monkeypatch):
    task_id = source.task_ids()[0]
    monkeypatch.setitem(_source.FILES, f"{task_id}/instruction.md", "0" * 64)
    source.read.cache_clear()
    try:
        with pytest.raises(source.TaskFileMismatch):
            source.read(task_id, "instruction.md")
    finally:
        source.read.cache_clear()


def test_an_unlisted_file_is_refused():
    with pytest.raises(source.TaskFileMismatch):
        source.read(source.task_ids()[0], "tests/../../_source.py")


def test_every_task_declares_the_same_offline_box_and_a_separate_verifier():
    for task_id in source.task_ids():
        toml = source.task_toml(task_id)
        env, verifier = toml["environment"], toml["verifier"]
        assert (env["cpus"], env["memory_mb"], env["storage_mb"]) == (2, 8192, 20480)
        assert env["allow_internet"] is False
        assert verifier["environment"]["allow_internet"] is False
        assert verifier["environment_mode"] == "separate"
        assert toml["agent"]["timeout_sec"] == 5400.0
        assert toml["artifacts"] == ["/logs/artifacts/model.patch"]
        meta = toml["metadata"]
        assert meta["task_id"] == task_id
        assert env["docker_image"] == ("public.ecr.aws/d3j8x8q7/swe-bench-202605:"
                                       f"{meta['ext_id']}-v1.1")
        assert re.fullmatch(r"[0-9a-f]{7,40}", meta["base_commit_hash"])
        config = json.loads(source.read(task_id, "tests/config.json"))
        assert config["base_commit"] == meta["base_commit_hash"]
        assert config["f2p_node_ids"], task_id


def test_the_verifier_image_is_the_task_image_plus_tests():
    """Upstream's verifier image is `FROM <task image>` plus `COPY tests/`: a pristine
    box of the task image with /tests staged is that image."""
    for task_id in source.task_ids():
        dockerfile = source.read(task_id, "tests/Dockerfile").decode()
        image = source.task_toml(task_id)["environment"]["docker_image"]
        lines = [line for line in dockerfile.splitlines() if line and not line.startswith("#")]
        assert lines[0] == f"FROM {image}"
        assert all(line.startswith(("COPY ", "RUN chmod +x /tests/test.sh")) for line in lines[1:])


def test_upstreams_grader_and_runner_frame_are_shared_by_every_task():
    graders = {source.read(t, "tests/grader.py") for t in source.task_ids()}
    assert len(graders) == 1
    frames = {source.read(t, "tests/test.sh").decode().split("# >>> RUN TESTS")[0]
              for t in source.task_ids()}
    assert len(frames) == 1
