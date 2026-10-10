"""TMax grading-box user separation (env norm: the agent's code must not touch the
grader's verdict). The final-state test runs as an unprivileged uid; root writes the
verdict once every process of that uid is gone. What needs root (setuid, chown, kill) is
checked live on a sandbox; here, everything around it."""

import ast
import os
import stat
from pathlib import Path

import pytest

from reliquary_terminal import tmax, tmax_box


def mode_of(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_grading_script_never_runs_pytest_itself():
    script = tmax.TEST_SH
    assert "/usr/bin/python3 -I /tests/tmax_box.py relay" in script
    assert script.rstrip().endswith("exec /usr/bin/python3 -I /tests/tmax_box.py run-tests")
    assert "pytest" not in script
    assert script.index("echo 0 >") < script.index("tmax_box.py relay")


def test_hand_over_gives_every_entry_to_the_uid_and_never_follows_links(tmp_path):
    home = tmp_path / "home" / "user"
    (home / "sub").mkdir(parents=True)
    secret = home / "sub" / "secret.txt"
    secret.write_text("x")
    secret.chmod(0)
    outside = tmp_path / "outside.txt"
    outside.write_text("y")
    outside.chmod(0o040)
    (home / "link").symlink_to(outside)
    owned = []
    tmax_box.hand_over(["/app", "/home/user"], 61000, root=str(tmp_path),
                       lchown=lambda path, uid, gid: owned.append((path, uid, gid)))
    names = {Path(path).relative_to(tmp_path).as_posix() for path, _, _ in owned}
    assert names == {"home/user", "home/user/sub", "home/user/sub/secret.txt",
                     "home/user/link"}
    assert all(uid == gid == 61000 for _, uid, gid in owned)
    assert mode_of(secret) == 0o400  # readable, and no more
    assert mode_of(outside) == 0o040  # never through the link


def test_hand_over_opens_closed_directories_and_adds_only_read_to_files(tmp_path):
    app = tmp_path / "app"
    closed = app / "closed"
    closed.mkdir(parents=True)
    (closed / "tool").write_text("#!/bin/sh\n")
    (closed / "tool").chmod(0o050)
    (closed / "data").write_text("d")
    (closed / "data").chmod(0o004)
    (app / "suid").write_text("s")
    (app / "suid").chmod(0o4755)
    shut = app / "shut"
    shut.mkdir()
    (shut / "inner").write_text("i")
    shut.chmod(0)
    closed.chmod(0o600)  # no x: nothing under it can be reached
    owned = []
    tmax_box.hand_over(["/app"], 61000, root=str(tmp_path),
                       lchown=lambda path, uid, gid: owned.append(path))
    assert str(closed / "tool") in owned and str(shut / "inner") in owned
    assert mode_of(closed) == 0o700
    assert mode_of(shut) == 0o500  # u+rx: listed and entered, still not writable
    assert mode_of(closed / "tool") == 0o450  # u+r only: no x, no w added
    assert mode_of(closed / "data") == 0o404
    assert mode_of(app / "suid") == 0o755  # no set-id bit survives the hand-over


def test_hand_over_keeps_the_modes_an_agent_set_that_reading_allows(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    modes = {"ro": 0o400, "rx": 0o500, "shared": 0o444, "private": 0o600, "none": 0o000,
             "wo": 0o200, "suid_ro": 0o4500, "sgid": 0o2644}
    for name, mode in modes.items():
        (app / name).write_text(name)
        (app / name).chmod(mode)
    sticky = app / "sticky"
    sticky.mkdir()
    sticky.chmod(0o2555)  # a set-gid directory with no w for its owner: kept as it is
    tmax_box.hand_over(["/app"], 61000, root=str(tmp_path), lchown=lambda *a: None)
    assert {name: mode_of(app / name) for name in modes} == {
        "ro": 0o400, "rx": 0o500, "shared": 0o444, "private": 0o600, "none": 0o400,
        "wo": 0o600, "suid_ro": 0o500, "sgid": 0o644}
    assert mode_of(sticky) == 0o2555


@pytest.mark.parametrize("mode, directory, handed", [
    (0o400, False, 0o400), (0o000, False, 0o400), (0o200, False, 0o600),
    (0o050, False, 0o450), (0o4755, False, 0o755), (0o6711, False, 0o711),
    (0o600, True, 0o700), (0o000, True, 0o500), (0o555, True, 0o555),
    (0o2750, True, 0o2750), (0o100644, False, 0o644), (0o040300, True, 0o700),
])
def test_handed_mode_adds_only_what_reading_needs(mode, directory, handed):
    assert tmax_box.handed_mode(mode, directory) == handed


def test_hand_over_leaves_a_file_shared_with_the_rest_of_the_box_alone(tmp_path):
    (tmp_path / "app").mkdir()
    system = tmp_path / "etc-passwd"
    system.write_text("root:x:0:0")
    system.chmod(0o644)
    os.link(system, tmp_path / "app" / "passwd")
    os.link(tmp_path / "app" / "passwd", tmp_path / "app" / "again")
    (tmp_path / "app" / "a").write_text("a")
    os.link(tmp_path / "app" / "a", tmp_path / "app" / "b")  # both links inside: handed over
    owned = []
    tmax_box.hand_over(["/app"], 61000, root=str(tmp_path),
                       lchown=lambda path, uid, gid: owned.append(Path(path).name))
    assert sorted(owned) == ["a", "app", "b"]
    assert mode_of(system) == 0o644


def test_hand_over_skips_missing_roots_and_does_not_enter_a_linked_root(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "f").write_text("f")
    (tmp_path / "app").symlink_to(elsewhere)
    owned = []
    tmax_box.hand_over(["/app", "/home/user"], 61000, root=str(tmp_path),
                       lchown=lambda path, uid, gid: owned.append(path))
    assert owned == [str(tmp_path / "app")]


def test_processes_of_reads_real_and_effective_uids_and_skips_zombies(tmp_path):
    for pid, uids, state in (("123", "61000\t61000\t61000\t61000", "S (sleeping)"),
                             ("124", "0\t0\t0\t0", "R (running)"),
                             ("125", "0\t61000\t0\t0", "S (sleeping)"),
                             ("126", "61000\t61000\t61000\t61000", "Z (zombie)"),
                             ("127", "610000\t610000\t610000\t610000", "S (sleeping)")):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "status").write_text(f"Name:\tx\nState:\t{state}\nUid:\t{uids}\n")
    (tmp_path / "128").mkdir()  # exited between the listing and the read
    (tmp_path / "self").mkdir()
    assert tmax_box.processes_of(61000, proc=str(tmp_path)) == [123, 125]


def test_reap_with_nothing_left_never_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(tmax_box.os, "fork", lambda: pytest.fail("forked"))
    assert tmax_box.reap(61000, proc=str(tmp_path)) is True


def test_reap_names_the_processes_that_survived(tmp_path, monkeypatch, capsys):
    (tmp_path / "4321").mkdir()
    (tmp_path / "4321" / "status").write_text("State:\tS (sleeping)\nUid:\t61000\t61000\t0\t0\n")
    monkeypatch.setattr(tmax_box.os, "fork", lambda: 99999)
    monkeypatch.setattr(tmax_box.os, "waitpid", lambda pid, flags: (pid, 0))
    monkeypatch.setattr(tmax_box.time, "sleep", lambda s: None)
    assert tmax_box.reap(61000, proc=str(tmp_path), rounds=2) is False
    assert "4321" in capsys.readouterr().err


@pytest.mark.skipif(os.geteuid() == 0, reason="as root the drop succeeds")
def test_a_failed_drop_never_runs_the_command(tmp_path):
    mark = tmp_path / "ran"
    rc = tmax_box.run_as(61000, ["/bin/sh", "-c", f"touch {mark}"], {"PATH": "/usr/bin:/bin"},
                         cwd=str(tmp_path))
    assert rc == 127 and not mark.exists()


def test_the_test_command_runs_with_no_new_privs(tmp_path, monkeypatch):
    monkeypatch.setattr(tmax_box, "_drop", lambda uid: None)
    mark = tmp_path / "status"
    rc = tmax_box.run_as(61000, ["/bin/sh", "-c", f"grep NoNewPrivs /proc/self/status > {mark}"],
                         {"PATH": "/usr/bin:/bin"}, cwd=str(tmp_path))
    assert rc == 0 and mark.read_text().split() == ["NoNewPrivs:", "1"]


def test_a_failed_no_new_privs_never_runs_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(tmax_box, "_drop", lambda uid: None)

    def refuse():
        raise OSError("prctl refused")

    monkeypatch.setattr(tmax_box, "_no_new_privs", refuse)
    mark = tmp_path / "ran"
    rc = tmax_box.run_as(61000, ["/bin/sh", "-c", f"touch {mark}"], {"PATH": "/usr/bin:/bin"},
                         cwd=str(tmp_path))
    assert rc == 127 and not mark.exists()


def test_no_new_privs_raises_when_prctl_refuses(monkeypatch):
    class Libc:
        def prctl(self, *args):
            assert args == (38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
            return -1

    monkeypatch.setattr(tmax_box.ctypes, "CDLL", lambda *a, **k: Libc())
    with pytest.raises(OSError):
        tmax_box._no_new_privs()


def report_in(tmp_path, content=b'{"results": {"tests": []}}'):
    out = tmp_path / "out"
    out.mkdir()
    (out / "ctrf.json").write_bytes(content)
    return str(out / "ctrf.json"), tmp_path / "logs" / "verifier"


ME = os.geteuid()  # owns what these tests write: it stands for the test uid in `finish`


def test_one_needs_a_clean_exit_a_complete_reap_and_a_report(tmp_path):
    report, verifier = report_in(tmp_path)
    assert tmax_box.finish(0, report, str(verifier), True, ME) == 0
    assert (verifier / "reward.txt").read_text() == "1\n"
    assert (verifier / "ctrf.json").read_bytes() == b'{"results": {"tests": []}}'
    assert tmax_box.finish(0, report, str(verifier), False, ME) == 1
    assert (verifier / "reward.txt").read_text() == "0\n"
    assert not (verifier / "ctrf.json").exists()
    assert tmax_box.finish(2, report, str(verifier), True, ME) == 2
    assert (verifier / "reward.txt").read_text() == "0\n"
    os.unlink(report)
    assert tmax_box.finish(0, report, str(verifier), True, ME) == 0
    assert (verifier / "reward.txt").read_text() == "0\n"


def test_a_report_the_test_uid_does_not_own_or_that_has_other_links_is_never_read(tmp_path):
    report, verifier = report_in(tmp_path)
    assert tmax_box.finish(0, report, str(verifier), True, ME + 1) == 0
    assert (verifier / "reward.txt").read_text() == "0\n"
    assert not (verifier / "ctrf.json").exists()
    os.link(report, tmp_path / "elsewhere.json")  # a second name the test uid could rewrite
    assert tmax_box.finish(0, report, str(verifier), True, ME) == 0
    assert (verifier / "reward.txt").read_text() == "0\n"


def test_a_report_that_is_a_link_a_fifo_or_too_large_is_never_read(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "elsewhere.json").write_text("{}")
    (out / "ctrf.json").symlink_to(tmp_path / "elsewhere.json")
    verifier = tmp_path / "logs" / "verifier"
    tmax_box.finish(0, str(out / "ctrf.json"), str(verifier), True, ME)
    assert not (verifier / "ctrf.json").exists()
    assert (verifier / "reward.txt").read_text() == "0\n"
    (tmp_path / "b").mkdir()
    big, verifier2 = report_in(tmp_path / "b", b"x" * (tmax_box.MAX_CTRF_BYTES + 1))
    tmax_box.finish(0, big, str(verifier2), True, ME)
    assert (verifier2 / "reward.txt").read_text() == "0\n"
    os.mkfifo(tmp_path / "fifo")  # no writer: a blocking open would hang the grader
    tmax_box.finish(0, str(tmp_path / "fifo"), str(verifier2), True, ME)
    assert (verifier2 / "reward.txt").read_text() == "0\n"


def test_finish_clears_what_it_did_not_write_and_replaces_a_verifier_link(tmp_path):
    report, verifier = report_in(tmp_path)
    verifier.parent.mkdir(parents=True)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    verifier.symlink_to(decoy)
    tmax_box.finish(1, report, str(verifier), True, ME)
    assert verifier.is_dir() and not verifier.is_symlink()
    assert not list(decoy.iterdir())
    (verifier / "reward.json").write_text('{"reward": 1}')
    (verifier / "ctrf.json").write_text("{}")
    tmax_box.finish(1, str(tmp_path / "no-report"), str(verifier), True, ME)
    assert sorted(p.name for p in verifier.iterdir()) == ["reward.txt"]
    assert mode_of(verifier) == 0o755
    tmax_box.finish(1, report, str(verifier), True, ME)  # a failing run's report is kept
    assert sorted(p.name for p in verifier.iterdir()) == ["ctrf.json", "reward.txt"]


def test_the_test_command_is_isolated_and_names_the_report():
    argv = tmax_box.test_argv("/tmp/r/ctrf.json")
    assert argv[:4] == ["/usr/bin/python3", "-I", "-m", "pytest"]
    assert argv[argv.index("--ctrf") + 1] == "/tmp/r/ctrf.json"
    assert "--confcutdir=/tests" in argv and argv[-1] == "/tests/" + tmax.FINAL_TEST
    assert tmax_box.FINAL_TEST == tmax.FINAL_TEST


def test_the_test_environment_is_the_tasks_with_a_home_outside_the_roots():
    env = tmax_box.test_env({"PATH": "/usr/bin", "HOME": "/root", "X": "1"}, "/tmp/t/home")
    assert env == {"PATH": "/usr/bin", "X": "1", "HOME": "/tmp/t/home", "PYTHONNOUSERSITE": "1"}


def _box(tmp_path, tests_mode=0o700, file_mode=0o600):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / tmax.FINAL_TEST).write_text("def test_x(): pass\n")
    (tests / tmax.FINAL_TEST).chmod(file_mode)
    tests.chmod(tests_mode)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "proc").mkdir()
    (tmp_path / "app").mkdir()
    yama = tmp_path / "proc/sys/kernel/yama"
    yama.mkdir(parents=True)
    (yama / "ptrace_scope").write_text("1\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs").chmod(0o755)
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/passwd").write_text("root:x:0:0:root:/root:/bin/bash\n")
    (tmp_path / "etc/group").write_text("root:x:0:\n")
    return tests


def _never_runs(monkeypatch):
    monkeypatch.setattr(tmax_box, "hand_over", lambda *a: pytest.fail("handed over"))
    monkeypatch.setattr(tmax_box, "run_as", lambda *a, **k: pytest.fail("ran"))


def _refused(tmp_path, capsys, why):
    assert tmax_box.run_tests(uid=61000, root=str(tmp_path)) == 1
    assert (tmp_path / "logs/verifier/reward.txt").read_text() == "0\n"
    assert why in capsys.readouterr().err


def test_run_tests_hands_over_runs_reaps_then_scores(tmp_path, monkeypatch):
    tests = _box(tmp_path, tests_mode=0o777)
    seen = {}
    monkeypatch.setattr(tmax_box.os, "chown", lambda *a: None)
    monkeypatch.setattr(tmax_box, "hand_over",
                        lambda roots, uid, root: seen.setdefault("roots", (list(roots), uid)))

    def run_as(uid, argv, env, cwd):
        seen.update(uid=uid, argv=argv, env=env, cwd=cwd)
        assert (tmp_path / "logs/verifier/reward.txt").read_text() == "0\n"
        Path(argv[argv.index("--ctrf") + 1]).write_text('{"results": {}}')
        return 0

    report = tmax_box._report

    def report_owned_by_me(path, owner):
        seen["owner"] = owner
        return report(path, os.geteuid())

    monkeypatch.setattr(tmax_box, "run_as", run_as)
    monkeypatch.setattr(tmax_box, "reap", lambda uid, proc: seen.setdefault("reaped", proc))
    monkeypatch.setattr(tmax_box, "_report", report_owned_by_me)
    monkeypatch.setenv("HOME", "/root")
    assert tmax_box.run_tests(uid=61000, root=str(tmp_path)) == 0
    assert (tmp_path / "logs/verifier/reward.txt").read_text() == "1\n"
    assert (tmp_path / "logs/verifier/ctrf.json").read_text() == '{"results": {}}'
    assert seen["roots"] == (list(tmax_box.ROOTS), 61000)
    assert seen["uid"] == 61000 and seen["cwd"] == str(tests)
    assert seen["reaped"] == str(tmp_path / "proc")
    home = seen["env"]["HOME"]
    assert home.startswith(str(tmp_path / "tmp") + "/") and not tmax._under_roots(home)
    assert seen["env"]["PYTHONNOUSERSITE"] == "1"
    assert seen["owner"] == 61000
    assert (tmp_path / "etc/passwd").read_text().splitlines()[1:] == [
        f"reliquary-test:x:61000:61000::{home}:/usr/sbin/nologin"]
    assert (tmp_path / "etc/group").read_text().splitlines()[1:] == ["reliquary-test:x:61000:"]
    # /tests is readable by the test uid and writable by nobody but its owner.
    assert mode_of(tests) == 0o755 and mode_of(tests / tmax.FINAL_TEST) == 0o644
    assert not list((tmp_path / "tmp").iterdir())  # the run's directory is cleaned up


def test_run_tests_refuses_tests_the_test_uid_owns(tmp_path, monkeypatch, capsys):
    _box(tmp_path)
    _never_runs(monkeypatch)
    assert tmax_box.run_tests(uid=os.geteuid(), root=str(tmp_path)) == 1
    assert (tmp_path / "logs/verifier/reward.txt").read_text() == "0\n"
    assert "owned by the test uid" in capsys.readouterr().err


@pytest.mark.parametrize("scope", [None, "0\n", "x\n"])
def test_run_tests_refuses_a_box_where_the_test_could_ptrace_pytest(tmp_path, monkeypatch,
                                                                     capsys, scope):
    _box(tmp_path)
    path = tmp_path / "proc/sys/kernel/yama/ptrace_scope"
    if scope is None:
        path.unlink()
    else:
        path.write_text(scope)
    _never_runs(monkeypatch)
    _refused(tmp_path, capsys, "ptrace_scope")


@pytest.mark.parametrize("how", ["writable", "link"])
def test_run_tests_refuses_logs_the_test_uid_could_write(tmp_path, monkeypatch, capsys, how):
    _box(tmp_path)
    if how == "writable":
        (tmp_path / "logs").chmod(0o777)
    else:
        (tmp_path / "logs").rmdir()
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "logs").symlink_to(tmp_path / "elsewhere")
    _never_runs(monkeypatch)
    assert tmax_box.run_tests(uid=61000, root=str(tmp_path)) == 1
    assert "/logs" in capsys.readouterr().err


def test_run_tests_refuses_when_the_test_uid_already_runs_something(tmp_path, monkeypatch,
                                                                    capsys):
    _box(tmp_path)
    (tmp_path / "proc/777").mkdir()
    (tmp_path / "proc/777/status").write_text("State:\tS (sleeping)\nUid:\t61000\t61000\t0\t0\n")
    _never_runs(monkeypatch)
    _refused(tmp_path, capsys, "777")


def test_the_test_account_is_added_once(tmp_path):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/passwd").write_text("root:x:0:0:root:/root:/bin/bash")  # no final newline
    tmax_box.ensure_account(61000, "/tmp/h", root=str(tmp_path))
    tmax_box.ensure_account(61000, "/tmp/other", root=str(tmp_path))
    assert (tmp_path / "etc/passwd").read_text() == (
        "root:x:0:0:root:/root:/bin/bash\n"
        "reliquary-test:x:61000:61000::/tmp/h:/usr/sbin/nologin\n")
    assert (tmp_path / "etc/group").read_text() == "reliquary-test:x:61000:\n"
    (tmp_path / "etc/passwd").write_text("someone:x:61000:61000::/x:/bin/sh")  # no final newline
    tmax_box.ensure_account(61000, "/tmp/h", root=str(tmp_path))
    assert (tmp_path / "etc/passwd").read_text() == "someone:x:61000:61000::/x:/bin/sh"


def test_run_tests_scores_zero_when_a_process_survives(tmp_path, monkeypatch):
    _box(tmp_path)
    monkeypatch.setattr(tmax_box.os, "chown", lambda *a: None)
    monkeypatch.setattr(tmax_box, "hand_over", lambda *a: None)

    def run_as(uid, argv, env, cwd):
        Path(argv[argv.index("--ctrf") + 1]).write_text("{}")
        return 0

    monkeypatch.setattr(tmax_box, "run_as", run_as)
    monkeypatch.setattr(tmax_box, "reap", lambda uid, proc: False)
    assert tmax_box.run_tests(uid=61000, root=str(tmp_path)) == 1
    assert (tmp_path / "logs/verifier/reward.txt").read_text() == "0\n"


def test_the_helper_stays_python_3_10_and_stdlib():
    source = Path(tmax_box.__file__).read_text()
    tree = ast.parse(source, feature_version=(3, 10))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"__future__", "ctypes", "hashlib", "json", "os", "shutil", "signal",
                        "stat", "sys", "tempfile", "time"}
    assert os.access(tmax_box.__file__, os.R_OK)
