"""The vendored tasks are upstream's at the pinned commit, bound to the package's `.py`
files through `_source.FILES`."""

import collections
import hashlib

import pytest

from reliquary_swebench_pro import _source, scoring, source

LANGUAGES = {"go": 256, "python": 237, "js": 145, "ts": 4}
TASK_FILES = {"task.toml", "instruction.md", "instance.json", "environment/Dockerfile",
              "solution/gold_patch.diff", "solution/solve.sh", "tests/config.json",
              "tests/parser.py", "tests/run_script.sh", "tests/test.sh",
              "tests/test_patch.patch"}


def test_pins():
    assert source.COMMIT == "66f92766bba642462d4bbe5479e83f91f9211862"
    assert source.UPSTREAM == "https://github.com/scaleapi/SWE-bench_Pro-os"
    assert source.DATASET_REVISION == "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"


def test_642_tasks_by_language():
    ids = source.task_ids()
    assert len(ids) == 642 and list(ids) == sorted(ids)
    counts = collections.Counter(source.instance(t)["repo_language"] for t in ids)
    assert counts == LANGUAGES
    assert sum(source.instance(t)["hard"] for t in ids) == 51


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


def test_every_task_declares_the_same_box_and_its_dataset_row():
    for task_id in source.task_ids():
        toml, row = source.task_toml(task_id), source.instance(task_id)
        env = toml["environment"]
        assert (env["cpus"], env["memory_mb"], env["storage_mb"]) == (1, 4096, 10240)
        assert toml["agent"]["network_mode"] == "no-network"
        assert toml["agent"]["timeout_sec"] == 3000.0
        assert env["docker_image"] == row["docker_image"] == (
            f"ghcr.io/scaleapi/swe-bench_pro-v2:{task_id}")
        assert row["instance_id"] == task_id and len(row["base_commit"]) == 40
        assert source.read(task_id, "environment/Dockerfile").decode().split()[:2] == [
            "FROM", env["docker_image"]]
        config = source.config(task_id)
        assert set(config) == {"fail_to_pass", "pass_to_pass", "selected_test_files_to_run"}
        assert scoring.test_names(config["fail_to_pass"]), task_id


def test_the_runner_frame_is_upstreams():
    """Every test.sh resets test code, applies the test patch, runs run_script.sh and
    parser.py, and keeps the raw logs where scoring reads them."""
    for task_id in source.task_ids():
        script = source.read(task_id, "tests/test.sh").decode()
        assert "git apply --verbose /tests/test_patch.patch" in script
        assert "python3 /tests/parser.py" in script
        assert 'cp "$STDOUT_LOG" /logs/verifier/run-script-stdout.txt' in script
        assert 'cp "$STDERR_LOG" /logs/verifier/run-script-stderr.txt' in script
