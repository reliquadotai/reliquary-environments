"""The vendored tasks are the pinned dataset's, bound to the package's `.py` files."""

import hashlib

import pytest

from reliquary_terminal_bench import _source, source


def test_pinned_dataset():
    assert source.DATASET == "terminal-bench/terminal-bench-2-1"
    assert source.DIGEST == ("sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177a"
                             "d3a0699a")


def test_89_tasks_sorted():
    ids = source.task_ids()
    assert len(ids) == 89 and list(ids) == sorted(ids)


def test_every_vendored_file_matches_its_digest_and_nothing_else_is_vendored():
    on_disk = {p.relative_to(source.TASKS).as_posix()
               for p in source.TASKS.rglob("*") if p.is_file()}
    gzipped = {rel + ".gz" for rel in _source.GZIPPED}
    assert on_disk == (set(_source.FILES) - set(_source.GZIPPED)) | gzipped
    for rel, digest in _source.FILES.items():
        task_id, name = rel.split("/", 1)
        assert hashlib.sha256(source.read(task_id, name)).hexdigest() == digest, rel


@pytest.mark.parametrize("task_id", source.task_ids())
def test_each_task_has_harbors_files(task_id):
    files = set(source.files_of(task_id))
    assert {"task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh",
            "solution/solve.sh"} <= files


def test_workdirs_follow_the_dockerfiles():
    assert source.workdir("fix-git") == "/app/personal-site"
    assert source.workdir("sanitize-git-repo") == "/app/dclm"
    assert source.workdir("prove-plus-comm") == "/workspace"
    others = {source.workdir(t) for t in source.task_ids()
              if t not in ("fix-git", "sanitize-git-repo", "prove-plus-comm")}
    assert others == {"/app"}


def test_archives_are_deterministic_and_hold_the_tests():
    import io
    import tarfile

    first = source.tests_archive("regex-log")
    source.tests_archive.cache_clear()
    assert source.tests_archive("regex-log") == first
    names = tarfile.open(fileobj=io.BytesIO(first)).getnames()
    assert set(names) == {rel[len("tests/"):] for rel in source.files_of("regex-log", "tests/")}


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


def test_a_gzipped_file_is_staged_as_upstream_ships_it():
    import io
    import tarfile

    archive = tarfile.open(fileobj=io.BytesIO(source.tests_archive("sanitize-git-repo")))
    data = archive.extractfile("test_outputs.py").read()
    digest = _source.FILES["sanitize-git-repo/tests/test_outputs.py"]
    assert hashlib.sha256(data).hexdigest() == digest
