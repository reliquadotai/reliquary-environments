"""The workspace archive rebuilt in the scoring process, and the JUnit count."""

import io
import tarfile

import pytest

from reliquary_nl2repo import submission


def archive(entries, mode="w:gz"):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode=mode) as tar:
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, data
                tar.addfile(info)
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, data
                tar.addfile(info)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)
    return out.getvalue()


def names(clean):
    with tarfile.open(fileobj=io.BytesIO(clean)) as tar:
        members = tar.getmembers()
    assert all(m.isreg() and m.uid == 0 and m.mode in (0o644, 0o755) for m in members)
    return sorted(m.name for m in members)


def test_source_files_are_kept():
    clean, report = submission.sanitize(archive([
        ("./pkg/__init__.py", "file", b"x = 1\n"), ("pkg", "dir", None),
        ("src/other/core.py", "file", b""), ("README.md", "file", b"r")]), ["tests"])
    assert names(clean) == ["README.md", "pkg/__init__.py", "src/other/core.py"]
    assert report == {"kept": 3, "dropped": {}}


@pytest.mark.parametrize("name, reason", [
    ("tests/test_a.py", "test_path"), ("tests/data/x.json", "test_path"),
    ("utils/testutils.py", "test_path"),
    ("setup.py", "package_file"), ("sub/pyproject.toml", "package_file"),
    ("pytest.ini", "package_file"), (".pytest.ini", "package_file"), ("tox.ini", "package_file"),
    ("conftest.py", "runner_hook"), ("pkg/conftest.py", "runner_hook"),
    ("sitecustomize.py", "runner_hook"), ("src/usercustomize.py", "runner_hook"),
    ("zz.pth", "runner_hook"),
    ("pytest.py", "shadows_runner"), ("_pytest/__init__.py", "shadows_runner"),
    ("src/pluggy/__init__.py", "shadows_runner"), ("pytest_zz.py", "shadows_runner"),
    ("py.py", "shadows_runner"),
    ("pkg/test_mine.py", "test_file"), ("mine_test.py", "test_file"),
    (".git/config", "toolchain_dir"), ("pkg/__pycache__/a.pyc", "toolchain_dir"),
    (".venv/lib/x.py", "toolchain_dir"), ("pkg.egg-info/PKG-INFO", "toolchain_dir"),
    ("build/lib/pkg/a.py", "toolchain_dir"), ("dist/x.whl", "toolchain_dir"),
])
def test_drop_reasons(name, reason):
    assert submission.drop_reason(name, ("tests", "utils/testutils.py")) == reason


def test_a_package_named_like_a_toolchain_dir_below_the_top_is_kept():
    assert submission.drop_reason("pkg/build/a.py", ()) is None
    assert submission.drop_reason("pkg/testing.py", ()) is None


@pytest.mark.parametrize("entry, reason", [
    (("../escape.py", "file", b"x"), "unsafe_path"),
    (("/etc/passwd", "file", b"x"), "unsafe_path"),
    (("a/../../b.py", "file", b"x"), "unsafe_path"),
    (("link.py", "symlink", "/etc/hostname"), "not_a_regular_file"),
    (("hard.py", "hardlink", "pkg/a.py"), "not_a_regular_file"),
    (("pipe", "fifo", None), "not_a_regular_file"),
])
def test_unsafe_members_are_dropped(entry, reason):
    clean, report = submission.sanitize(archive([("pkg/a.py", "file", b"a"), entry]), [])
    assert names(clean) == ["pkg/a.py"]
    assert report["dropped"] == {reason: 1}


def test_executable_bits_survive_and_nothing_else():
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo("run.sh")
        info.size, info.mode, info.uid = 2, 0o4777, 1000
        tar.addfile(info, io.BytesIO(b"#!"))
    clean, _ = submission.sanitize(out.getvalue(), [])
    with tarfile.open(fileobj=io.BytesIO(clean)) as tar:
        member = tar.getmember("run.sh")
    assert member.mode == 0o755 and member.uid == 0


def test_uncompressed_and_empty_archives_are_read():
    assert names(submission.sanitize(archive([("a.py", "file", b"")], "w"), [])[0]) == ["a.py"]
    assert names(submission.sanitize(submission.empty_archive(), [])[0]) == []


@pytest.mark.parametrize("raw", [b"", b"not a tar", b"\x1f\x8b\x08garbage"])
def test_unreadable_archives_are_refused(raw):
    with pytest.raises(submission.ArchiveRefused):
        submission.sanitize(raw, [])


def test_an_archive_too_large_unpacked_is_refused(monkeypatch):
    monkeypatch.setattr(submission, "MAX_UNPACKED_BYTES", 10)
    with pytest.raises(submission.ArchiveRefused, match="unpacked"):
        submission.sanitize(archive([("a.py", "file", b"x" * 11)]), [])


def test_too_many_members_are_refused(monkeypatch):
    monkeypatch.setattr(submission, "MAX_MEMBERS", 2)
    with pytest.raises(submission.ArchiveRefused, match="members"):
        submission.sanitize(archive([(f"{i}.py", "file", b"") for i in range(3)]), [])


JUNIT = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="5">
<testcase classname="t" name="a"/>
<testcase classname="t" name="b"><failure message="x"/></testcase>
<testcase classname="t" name="c"><error message="x"/></testcase>
<testcase classname="t" name="d"><skipped message="x"/></testcase>
<testcase classname="t" name="e"><system-out>ok</system-out></testcase>
</testsuite></testsuites>"""


def test_count_passed():
    assert submission.count_passed(JUNIT) == (2, 5)
    assert submission.count_passed(b'<testsuite><testcase name="a"/></testsuite>') == (1, 1)


@pytest.mark.parametrize("raw", [
    b'<!DOCTYPE x [<!ENTITY a "b">]><testsuite/>',
    b"<html/>",
    b"<testsuite><testcase",
])
def test_count_passed_refuses_what_is_not_a_plain_report(raw):
    with pytest.raises(Exception):
        submission.count_passed(raw)
