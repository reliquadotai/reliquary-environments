"""What a grading box receives as /tests."""

import gzip
import io
import json
import tarfile

import pytest

from reliquary_deepswe import overlay, source

ALL = source.task_ids()


@pytest.mark.parametrize("task_id", ALL)
def test_every_task_stages(task_id):
    staged = overlay.staged_tests(task_id)
    assert {"test.sh", "test.patch", "config.json", "grader.py", "upstream_grader.py",
            "metadata.json"} <= set(staged)
    assert "Dockerfile" not in staged
    assert staged["grader.py"] == overlay.VERIFIER.read_bytes()
    assert staged["upstream_grader.py"] == source.read(task_id, "tests/grader.py")
    assert json.loads(staged["metadata.json"])["task_id"] == task_id
    assert json.loads(staged["config.json"]) == overlay.effective_config(task_id)


def test_only_the_corrected_files_differ_from_upstream():
    for task_id in ALL:
        staged = overlay.staged_tests(task_id)
        assert staged["test.patch"] == source.read(task_id, "tests/test.patch")
        if task_id != overlay.KCP:
            assert staged["config.json"] == source.read(task_id, "tests/config.json")
        language = source.task_toml(task_id)["metadata"]["language"]
        if language != "go" and task_id != overlay.EICRUD:
            assert staged["test.sh"] == source.read(task_id, "tests/test.sh")


def test_kcp_excludes_the_preemption_test_from_execution_and_scoring():
    script = overlay.staged_tests(overlay.KCP)["test.sh"].decode()
    assert '-run "^TestMux" -skip "^TestMuxPriorityPreemption$"' in script
    config = overlay.effective_config(overlay.KCP)
    original = json.loads(source.read(overlay.KCP, "tests/config.json"))
    assert len(config["f2p_node_ids"]) == len(original["f2p_node_ids"]) - 1 == 29
    assert not any(i.endswith("TestMuxPriorityPreemption")
                   for i in config["f2p_node_ids"] + config["p2p_node_ids"])


def test_eicrud_raises_the_open_file_limit_first():
    script = overlay.staged_tests(overlay.EICRUD)["test.sh"].decode()
    assert "set -uo pipefail\nulimit -n 64000 || exit $?\n" in script


def test_go_runners_keep_build_events_in_the_run_log():
    for task_id in ALL:
        if source.task_toml(task_id)["metadata"]["language"] != "go":
            continue
        script = overlay.staged_tests(task_id)["test.sh"].decode()
        code = "\n".join(line for line in script.splitlines()
                         if not line.lstrip().startswith("#"))
        assert code.count(overlay._GO_FILTER) >= 1, task_id
        assert overlay._GO_FILTER_THEN_TEE.search(code) is None, task_id
        assert code.count('| tee -a "$RUN_LOG"') >= code.count(overlay._GO_FILTER)


def test_a_correction_that_no_longer_matches_is_refused():
    with pytest.raises(overlay.OverlayError):
        overlay.corrected_test_sh(overlay.KCP, "go", "go test ./...\n")


def test_the_archive_is_deterministic_and_holds_the_staged_files():
    task_id = ALL[0]
    archive = overlay.tests_archive(task_id)
    overlay.tests_archive.cache_clear()
    assert overlay.tests_archive(task_id) == archive
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive))) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert set(members) == set(overlay.staged_tests(task_id))
        assert members["test.sh"].mode == 0o755
        assert all(m.mtime == 0 and m.uid == 0 and m.isfile() for m in members.values())
        assert tar.extractfile("config.json").read() == overlay.staged_tests(task_id)["config.json"]


def test_tests_digest_is_per_task():
    digests = {overlay.tests_sha256(t) for t in ALL}
    assert len(digests) == len(ALL)
