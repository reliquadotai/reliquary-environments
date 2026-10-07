"""R2E grading's pure pieces -- the log parser and the reward rule -- with no
container. `tests/test_r2e_goldens.py` checks them against real images."""

from __future__ import annotations

import json
from types import SimpleNamespace

from reliquary_swe import grading

# Shaped like real output: tornado's own runner on the golden image (stdout,
# captured on the box), and pytest's `-rA` summary.
TORNADO_LOG = """\
AttributeError: 'HTTPRequest' object has no attribute 'partition'

==================== short test summary info ====================
PASSED r2e_tests.test_1::WebSocketTest::test_websocket_callbacks
ERROR r2e_tests.test_1::WebSocketTest::test_websocket_headers
=================== 1 error, 1 passed in 0.03s ===================
"""

PYTEST_LOG = """\
PASSED r2e_tests/test_1.py::TestX::test_before_the_summary
=========================== short test summary info ============================
PASSED r2e_tests/test_1.py::TestX::test_ok
FAILED r2e_tests/test_1.py::TestX::test_bad - AssertionError: assert 1 == 2
ERROR r2e_tests/test_1.py::test_fixture - RuntimeError: boom
PASSED r2e_tests/test_1.py::test_param[a - b]
========================= 1 failed, 2 passed, 1 error in 0.50s =========================
"""


def test_parse_reads_only_the_short_test_summary():
    assert grading.parse_log_pytest(PYTEST_LOG) == {
        "TestX.test_ok": "PASSED",
        "TestX.test_bad": "FAILED",
        "test_fixture": "ERROR",
        # R2E keeps " - " in a PASSED name; the reward rule splits it off.
        "test_param[a - b]": "PASSED",
    }


def test_parse_reads_tornados_own_runner():
    assert grading.parse_log_pytest(TORNADO_LOG) == {
        "WebSocketTest.test_websocket_callbacks": "PASSED",
        "WebSocketTest.test_websocket_headers": "ERROR",
    }


def test_parse_without_a_summary_is_empty():
    assert grading.parse_log_pytest("Traceback (most recent call last):\n") == {}
    assert grading.parse_log_pytest("") == {}


def test_a_collection_error_parses_to_the_empty_name():
    log = "=== short test summary info ===\nERROR r2e_tests/test_1.py - ImportError: x\n"
    assert grading.parse_log_pytest(log) == {"": "ERROR"}


def _expected(mapping: dict[str, str]) -> str:
    return json.dumps(mapping)


def test_reward_is_one_only_for_the_exact_verdict_map():
    expected = _expected({"TestX.test_ok": "PASSED", "TestX.test_bad": "FAILED"})
    assert grading.r2e_reward({"TestX.test_ok": "PASSED", "TestX.test_bad": "FAILED"}, expected) == 1.0
    # A test expected to fail must fail: "all green" is not the target.
    assert grading.r2e_reward({"TestX.test_ok": "PASSED", "TestX.test_bad": "PASSED"}, expected) == 0.0
    assert grading.r2e_reward({"TestX.test_ok": "PASSED"}, expected) == 0.0
    assert (
        grading.r2e_reward(
            {"TestX.test_ok": "PASSED", "TestX.test_bad": "FAILED", "TestX.extra": "PASSED"},
            expected,
        )
        == 0.0
    )


def test_reward_for_an_empty_parse_is_zero_even_against_an_empty_map():
    # R2E's own rule (same size, every key matches) would pay 0 == 0.
    assert grading.r2e_reward({}, _expected({})) == 0.0
    assert grading.r2e_reward({}, _expected({"t": "PASSED"})) == 0.0


def test_a_collection_error_cannot_stand_in_for_a_test():
    # R2E's own comparison skips an empty name, so on a task with ONE
    # expected test (46 rows in the pinned revision) a patch that breaks the
    # test module's import -- parsed as {"": "ERROR"} -- would pay 1.0.
    assert grading.r2e_reward({"": "ERROR"}, _expected({"test_fix": "PASSED"})) == 0.0


def test_an_expected_collection_error_must_still_be_reproduced():
    # One row (orange3 f813020a9c0a) expects {"": "ERROR"} among 27 entries.
    expected = _expected({"": "ERROR", "T.test_a": "PASSED"})
    assert grading.r2e_reward({"": "ERROR", "T.test_a": "PASSED"}, expected) == 1.0
    assert grading.r2e_reward({"T.test_a": "PASSED", "T.test_b": "PASSED"}, expected) == 0.0


def test_reward_normalises_names_as_r2e_does():
    # Keys are cut at " - " on both sides, and ANSI bold is stripped from
    # expected names (369 rows store them that way).
    expected = _expected({"\x1b[1mtest_param[a - b]\x1b[0m": "PASSED"})
    assert grading.r2e_reward({"test_param[a - b]": "PASSED"}, expected) == 1.0


def test_restore_strategy_is_the_r2e_strategy_for_an_r2e_row():
    data = SimpleNamespace(split="r2e", test_patch="")
    assert grading._restore_strategy_for(data) is grading._restore_r2e


def test_a_status_word_inside_a_failure_message_does_not_pass():
    # The exploit upstream's `"PASSED" in line` allows: one
    # `raise RuntimeError("PASSED")` on the buggy path turns a failing test
    # into a pass. Only the line's first token is the status.
    log = (
        "=== short test summary info ===\n"
        "FAILED r2e_tests/test_1.py::TestX::test_fix - RuntimeError: PASSED\n"
        "ERROR r2e_tests/test_1.py::TestX::test_other - PASSED\n"
        "FAILED r2e_tests/test_1.py::TestX::test_third - ERROR\n"
    )
    parsed = grading.parse_log_pytest(log)
    assert parsed == {
        "TestX.test_fix": "FAILED",
        "TestX.test_other": "ERROR",
        "TestX.test_third": "FAILED",
    }
    expected = _expected(
        {"TestX.test_fix": "PASSED", "TestX.test_other": "PASSED", "TestX.test_third": "PASSED"}
    )
    assert grading.r2e_reward(parsed, expected) == 0.0


def test_lines_not_starting_with_a_status_are_ignored():
    log = (
        "=== short test summary info ===\n"
        "  PASSED r2e_tests/test_1.py::TestX::test_indented\n"
        "something PASSED r2e_tests/test_1.py::TestX::test_buried\n"
        "PASSED: r2e_tests/test_1.py::TestX::test_colon\n"
        "PASSED r2e_tests/test_1.py::TestX::test_real\n"
        "=== 1 passed in 0.1s ===\n"
    )
    assert grading.parse_log_pytest(log) == {"TestX.test_real": "PASSED"}


def test_numstat_paths_reads_git_s_own_parse_of_the_patch():
    # `git apply --numstat -z`: "added\tdeleted\tpath\0", and for a rename
    # "added\tdeleted\t\0old\0new\0". Paths arrive unquoted, whatever
    # characters they hold.
    out = "1\t0\ta.py\x002\t1\t\x00old/b.py\x00new/b.py\x00-\t-\tweird\tname.pth\x00"
    assert grading._numstat_paths(out) == ["a.py", "old/b.py", "new/b.py", "weird\tname.pth"]


def test_patch_paths_outside_the_tracked_tree_are_forbidden():
    untracked = [".venv/", "coverage.egg-info/", "install.sh", "coverage/tracer.so"]
    forbidden = grading._forbidden_patch_paths(
        [
            "coverage/debug.py",  # tracked source: fine
            "reproduce_issue.py",  # new root file: fine (deleted later)
            "coverage/new_module.py",  # new file in a tracked dir: fine
            ".venv/lib/python3.9/site-packages/zz.pth",
            ".venv",
            "coverage.egg-info/entry_points.txt",
            "install.sh",
            "coverage/tracer.so",
            "build/ignored.py",
        ],
        untracked,
        ignored=["build/ignored.py"],
    )
    assert forbidden == [
        ".venv/lib/python3.9/site-packages/zz.pth",
        ".venv",
        "coverage.egg-info/entry_points.txt",
        "install.sh",
        "coverage/tracer.so",
        "build/ignored.py",
    ]


class _FakeRuntime:
    """Answers every command from `answers` (first matching substring of the
    joined argv), else exit 0 with a SHA-shaped stdout; records each one."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.commands: list[str] = []

    async def run(self, argv, env):
        command = " ".join(argv)
        self.commands.append(command)
        for needle, result in self.answers.items():
            if needle in command:
                return result
        return SimpleNamespace(exit_code=0, stdout="0" * 40 + "\n", stderr="")

    async def write(self, path, data):
        self.commands.append(f"write {path}")


def _r2e_data():
    from reliquary_swe.taskset import SweData

    return SweData(
        idx=0,
        name="r2e__x__0",
        prompt="p",
        instance_id="r2e__x__0",
        repo="x",
        base_commit="HEAD",
        version="",
        fail_to_pass=(),
        pass_to_pass=(),
        gold_patch="",
        test_patch="",
        split="r2e",
        expected_output_json=json.dumps({"T.test_a": "PASSED"}),
    )


async def test_a_failed_restoration_scores_zero_and_runs_nothing(monkeypatch):
    async def restore_fails(runtime, data):
        return False

    monkeypatch.setattr(grading, "_restore_r2e", restore_fails)
    runtime = _FakeRuntime()
    report = await grading.grade(runtime, _r2e_data(), "")
    assert report.reward == 0.0
    assert report.restored is False
    assert not any("run_tests.sh" in c and "bash" in c for c in runtime.commands)


async def test_check_ignore_failing_is_a_violation_not_an_infra_error():
    # A path past a tracked symlink makes `git check-ignore` exit 128
    # ("beyond a symbolic link"). Raising would retry provisioning and drop
    # the episode; the patch is the cause, so it scores 0 instead.
    runtime = _FakeRuntime(
        {
            "--numstat": SimpleNamespace(exit_code=0, stdout="1\t0\tlink/x.py\0", stderr=""),
            "ls-files": SimpleNamespace(exit_code=0, stdout="", stderr=""),
            "check-ignore": SimpleNamespace(
                exit_code=128, stdout="", stderr="fatal: pathspec is beyond a symbolic link"
            ),
        }
    )
    assert await grading._patch_violations(runtime, _r2e_data())
    report = await grading.grade(
        runtime, _r2e_data(), "diff --git a/link/x.py b/link/x.py\n"
    )
    assert report.reward == 0.0
    assert report.applied is False
    assert not any(c.startswith("git apply -v") for c in runtime.commands)
