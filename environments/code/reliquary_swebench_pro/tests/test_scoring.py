"""The verdict, from logs, with the task's own vendored parser; the patch refusals."""

import json

import pytest

from reliquary_swebench_pro import scoring, source

QUTE = "instance_qutebrowser__qutebrowser-77c3557995704a683cdb67e2a3055f7547fa22c3-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"


def pytest_log(statuses: dict[str, str]) -> bytes:
    return "".join(f"{name} {status} [ 50%]\n" for name, status in statuses.items()).encode()


def required(task_id):
    config = source.config(task_id)
    return (scoring.test_names(config["fail_to_pass"]), scoring.test_names(config["pass_to_pass"]))


def test_every_tasks_parser_loads_and_grades_empty_logs_0():
    """Some parsers list expected tests as FAILED on empty logs, one a summary entry that
    no task requires: no required test passes."""
    for task_id in source.task_ids():
        assessment = scoring.assess(task_id, b"", b"")
        assert assessment["reward"] == 0.0 and assessment["f2p_passed"] == 0, task_id


def test_upstream_lists_parse_as_json_or_python_literals():
    assert scoring.test_names('["a", "b"]') == {"a", "b"}
    assert scoring.test_names("['a', 'b']") == {"a", "b"}
    assert scoring.test_names(["a"]) == {"a"} and scoring.test_names("nonsense") == set()


def test_every_required_test_passing_is_1():
    f2p, p2p = required(QUTE)
    assessment = scoring.assess(QUTE, pytest_log({n: "PASSED" for n in f2p | p2p}), b"")
    assert assessment["reward"] == 1.0 and assessment["evaluation_valid"] == 1.0
    assert assessment["f2p_passed"] == len(f2p) and assessment["p2p_passed"] == len(p2p)


def test_one_failing_or_missing_required_test_is_0():
    f2p, p2p = required(QUTE)
    one = sorted(f2p)[0]
    failing = scoring.assess(QUTE, pytest_log({n: "FAILED" if n == one else "PASSED"
                                               for n in f2p | p2p}), b"")
    assert failing["reward"] == 0.0 and failing["failed"] == [one]
    missing = scoring.assess(QUTE, pytest_log({n: "PASSED" for n in (f2p | p2p) - {one}}), b"")
    assert missing["reward"] == 0.0 and missing["missing"] == [one]


def test_the_tasks_own_export_decides_a_repeated_name():
    """qutebrowser's export keeps the last status of a repeated name, as in the box."""
    f2p, p2p = required(QUTE)
    one = sorted(f2p)[0]
    log = pytest_log({n: "PASSED" for n in f2p | p2p}) + pytest_log({one: "FAILED"})
    assert scoring.assess(QUTE, log, b"")["reward"] == 0.0
    log = pytest_log({one: "FAILED"}) + pytest_log({n: "PASSED" for n in f2p | p2p})
    assert scoring.assess(QUTE, log, b"")["reward"] == 1.0


def test_missing_logs_are_0():
    assessment = scoring.assess(QUTE, None, b"")
    assert assessment["reward"] == 0.0 and assessment["evaluation_valid"] == 0.0


def test_the_parser_runs_from_the_pinned_bytes_not_the_box(monkeypatch):
    scoring.parser_module.cache_clear()
    monkeypatch.setitem(source._source.FILES, f"{QUTE}/tests/parser.py", "0" * 64)
    source.read.cache_clear()
    try:
        with pytest.raises(source.TaskFileMismatch):
            scoring.parse(QUTE, b"", b"")
    finally:
        source.read.cache_clear()
        scoring.parser_module.cache_clear()


@pytest.mark.parametrize("patch, why", [
    (b"diff --git a/x b/x\nnew file mode 120000\n", "symlink_or_submodule"),
    (b"diff --git a/sub b/sub\nnew file mode 160000\n", "symlink_or_submodule"),
    (b"diff --git a/node_modules/m/i.js b/node_modules/m/i.js\n", "refused_directory"),
    (b"diff --git a/src/.venv/x.pth b/src/.venv/x.pth\n", "refused_directory"),
    (b"--- a/../etc/passwd\n+++ b/../etc/passwd\n", "path_outside_repository"),
    (b"diff --git a/a b/a\nrename from a\nrename to .git/hooks/pre-commit\n", "refused_directory"),
])
def test_refused_patch_shapes(patch, why):
    assert scoring.patch_refusal(patch).startswith(why)


def test_an_ordinary_patch_is_not_refused():
    for task_id in source.task_ids():
        assert scoring.patch_refusal(source.gold_patch(task_id)) is None, task_id


def test_summary_truncates():
    summary = scoring.summary({"failed": ["x" * 500] * 50, "missing": ["y"] * 100})
    assert len(summary["failed"]) == 20 and len(summary["failed"][0]) == 300
    assert len(summary["missing"]) == 40
    json.dumps(summary)
