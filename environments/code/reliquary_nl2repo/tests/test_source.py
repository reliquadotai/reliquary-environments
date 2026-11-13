"""The tasks are upstream's at the pinned commit, bound to the package through
`_source.FILES` (fetched by scripts/vendor_tasks.py, never committed)."""

import hashlib
import json

import pytest

from reliquary_nl2repo import _source, source

FILES = {"start.md", "test_case_count.txt", "test_commands.json", "test_files.json"}


def test_pinned_commit():
    assert source.COMMIT == "781a1da1ee41fb8edb0bed22f586d69111610edf"
    assert source.UPSTREAM == "https://github.com/multimodal-art-projection/NL2RepoBench"


def test_104_tasks_with_four_files_each():
    ids = source.task_ids()
    assert len(ids) == 104 and list(ids) == sorted(ids)
    for task_id in ids:
        assert {k.split("/", 1)[1] for k in _source.FILES if k.startswith(task_id + "/")} == FILES


def test_every_file_on_disk_matches_its_digest_and_nothing_else_is_there():
    on_disk = {p.relative_to(source.TASKS).as_posix()
               for p in source.TASKS.rglob("*") if p.is_file()}
    assert on_disk == set(_source.FILES)
    for rel, digest in _source.FILES.items():
        assert hashlib.sha256((source.TASKS / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.parametrize("task_id", source.task_ids())
def test_each_task_parses(task_id):
    assert source.test_commands(task_id) and source.test_case_count(task_id) > 0
    assert all(not p.startswith("/") and ".." not in p.split("/")
               for p in source.test_files(task_id))
    assert source.spec(task_id).strip()
    assert source.image(task_id) == source.image(task_id).lower()


def test_upstream_totals():
    assert sum(source.test_case_count(t) for t in source.task_ids()) == 25640


def test_a_changed_file_is_refused(monkeypatch):
    task_id = source.task_ids()[0]
    monkeypatch.setitem(_source.FILES, f"{task_id}/start.md", "0" * 64)
    source.read.cache_clear()
    try:
        with pytest.raises(source.TaskFileMismatch):
            source.read(task_id, "start.md")
    finally:
        source.read.cache_clear()


def test_an_unlisted_file_is_refused():
    with pytest.raises(source.TaskFileMismatch):
        source.read(source.task_ids()[0], "../_source.py")


def test_a_missing_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "TASKS", tmp_path)
    source.read.cache_clear()
    try:
        with pytest.raises(source.TaskFileMismatch, match="vendor_tasks"):
            source.read(source.task_ids()[0], "start.md")
    finally:
        source.read.cache_clear()


def test_the_image_lock_pins_every_task_by_digest():
    lock = json.loads((source.TASKS.parent / "sandbox-images.lock.json").read_text())
    assert lock["version"] == 1
    assert set(lock["images"]) == {source.image(t) for t in source.task_ids()}
    for tag, ref in lock["images"].items():
        assert ref.startswith(tag.rsplit(":", 1)[0] + "@sha256:") and len(ref.split("@")[1]) == 71
