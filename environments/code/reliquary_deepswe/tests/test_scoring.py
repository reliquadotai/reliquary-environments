"""The verdict computed outside the box, from report bytes."""

import importlib.util
import json
import sys

import pytest

from reliquary_deepswe import overlay, scoring, source


def upstream_grader():
    path = source.TASKS / source.task_ids()[0] / "tests" / "grader.py"
    spec = importlib.util.spec_from_file_location("deepswe_upstream_grader", path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ in tasks/
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def ctrf(tests):
    return json.dumps({"results": {"tool": {"name": "t"}, "tests": tests}}).encode()


CTRF_TESTS = [
    {"name": "test_a", "suite": ["pkg/a"], "status": "passed"},
    {"name": "test_b", "suite": "pkg/b", "status": "failed", "message": "boom"},
    {"name": "test_c", "status": "skipped"},
    {"name": "test_b", "suite": "pkg/b", "status": "passed"},
    {"name": " test_d ", "status": "pending", "trace": "t"},
    {"name": "", "status": "passed"},
    "junk",
]
JUNIT = b"""<?xml version="1.0"?>
<testsuites><testsuite name="s">
<testcase classname="m.C" name="test_ok"/>
<testcase classname="m.C" name="test_bad"><failure message="no">trace</failure></testcase>
<testcase classname="" name="test_skip"><skipped/></testcase>
<testcase classname="m.C" name="test_err"><error message="e"/></testcase>
</testsuite></testsuites>"""


@pytest.mark.parametrize("node_id", ["name", "suite.name"])
def test_ctrf_parsing_matches_upstream(tmp_path, node_id):
    raw = ctrf(CTRF_TESTS)
    path = tmp_path / "r.json"
    path.write_bytes(raw)
    cfg = {"node_id": node_id}
    assert scoring.parse_ctrf(raw, cfg) == upstream_grader().parse_ctrf(path, cfg)


def test_junit_parsing_matches_upstream(tmp_path):
    path = tmp_path / "r.xml"
    path.write_bytes(JUNIT)
    assert scoring.parse_junit(JUNIT, {}) == upstream_grader().parse_junit(path, {})


@pytest.mark.parametrize("raw", [b"", b"{}", b'{"results": {"tests": {}}}', b"not json"])
def test_a_malformed_ctrf_report_raises(raw):
    with pytest.raises((ValueError, KeyError, TypeError)):
        scoring.parse_ctrf(raw, {})


def test_a_junit_report_with_entities_is_refused():
    with pytest.raises(ValueError):
        scoring.parse_junit(b'<!DOCTYPE x [<!ENTITY a "aaaa">]><testsuite/>', {})


CONFIG = {
    "f2p_node_ids": ["test_new", "test_new2"],
    "p2p_node_ids": ["test_old"],
    "grade": {"format": "ctrf", "node_id": "name", "tool_label": "t",
              "reports": ["/logs/verifier/base-ctrf.json", "/logs/verifier/new-ctrf.json"]},
}


def reports(base, new):
    return {"/logs/verifier/base-ctrf.json": base, "/logs/verifier/new-ctrf.json": new}


def test_all_required_tests_passing_is_1():
    a = scoring.assess_reports(CONFIG, "javascript", reports(
        ctrf([{"name": "test_old", "status": "passed"}]),
        ctrf([{"name": "test_new", "status": "passed"}, {"name": "test_new2", "status": "passed"}])),
        None)
    assert a["scores"]["reward"] == 1 and a["scores"]["f2p"] == 1


@pytest.mark.parametrize("base, new", [
    # a p2p failure
    (ctrf([{"name": "test_old", "status": "failed"}]),
     ctrf([{"name": "test_new", "status": "passed"}, {"name": "test_new2", "status": "passed"}])),
    # a missing f2p test
    (ctrf([{"name": "test_old", "status": "passed"}]),
     ctrf([{"name": "test_new", "status": "passed"}])),
    # a missing report
    (None, ctrf([{"name": "test_new", "status": "passed"}, {"name": "test_new2", "status": "passed"}])),
    # a skipped f2p test
    (ctrf([{"name": "test_old", "status": "passed"}]),
     ctrf([{"name": "test_new", "status": "passed"}, {"name": "test_new2", "status": "skipped"}])),
    # no report at all
    (None, None),
])
def test_anything_less_is_0(base, new):
    a = scoring.assess_reports(CONFIG, "javascript", reports(base, new), None)
    assert a["scores"]["reward"] == 0


def test_a_complete_passing_python_family_matches_missing_parameter_ids():
    config = {**CONFIG, "f2p_node_ids": ["t.py::test_p[1]", "t.py::test_p[2]"]}
    a = scoring.assess_reports(config, "python", reports(
        ctrf([{"name": "test_old", "status": "passed"}]),
        ctrf([{"name": "t.py::test_p[a]", "status": "passed"},
              {"name": "t.py::test_p[b]", "status": "passed"}])), None)
    assert a["scores"]["reward"] == 1 and a["scores"]["family_matches"] == 2


def test_go_timeouts_named_by_the_panic_fail_their_tests():
    config = {**CONFIG, "f2p_node_ids": ["example.com/p.TestA"], "p2p_node_ids": []}
    log = "\n".join(json.dumps({"Action": "output", "Package": "example.com/p", "Output": line})
                    for line in ["panic: test timed out after 2m0s\n", "\trunning tests:\n",
                                 "\t\tTestA (2m0s)\n"])
    a = scoring.assess_reports(config, "go", reports(ctrf([]), ctrf([])), log)
    row = next(r for r in a["tests"] if r["id"] == "example.com/p.TestA")
    assert row["status"] == "failed" and "timed out" in row["message"]
    assert scoring.retry_eligible(a) is False  # the f2p test itself failed


def test_retry_only_when_every_f2p_passed_and_every_failure_is_a_timeout():
    good = ctrf([{"name": "test_new", "status": "passed"}, {"name": "test_new2", "status": "passed"}])
    timed_out = ctrf([{"name": "test_old", "status": "failed", "message": "TimeoutError: x"}])
    asserted = ctrf([{"name": "test_old", "status": "failed", "message": "AssertionError"}])
    assert scoring.retry_eligible(scoring.assess_reports(CONFIG, "python", reports(timed_out, good), None))
    assert not scoring.retry_eligible(scoring.assess_reports(CONFIG, "python", reports(asserted, good), None))
    assert not scoring.retry_eligible(scoring.assess_reports(
        CONFIG, "python", reports(ctrf([{"name": "test_old", "status": "passed"}]), good), None))


def test_every_task_config_assesses_to_0_on_empty_reports():
    for task_id in source.task_ids():
        config = overlay.effective_config(task_id)
        language = source.task_toml(task_id)["metadata"]["language"]
        a = scoring.assess_reports(config, language, {}, "")
        assert a["scores"]["reward"] == 0, task_id


def test_report_locations_follow_the_frames_tuck():
    assert scoring.report_locations("/logs/verifier/new-ctrf.json") == (
        "/logs/verifier/new-ctrf.json", "/logs/verifier/reports/new-ctrf.json")


def diff(path, mode="100644"):
    return (f"diff --git a/{path} b/{path}\nnew file mode {mode}\nindex 0000000..e69de29\n"
            f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n").encode()


@pytest.mark.parametrize("patch, refused", [
    (diff("src/lib.py"), None),
    (diff("src/vendor/x.go"), None),
    (diff("link", "120000"), "symlink_or_submodule"),
    (diff("sub", "160000"), "symlink_or_submodule"),
    (diff("node_modules/jest/index.js"), "refused_directory"),
    (diff("packages/a/node_modules/x.js"), "refused_directory"),
    (diff(".venv/lib/python3.12/site-packages/zz.pth"), "refused_directory"),
    (diff(".gocache/00/abc-d"), "refused_directory"),
    (diff("target/debug/deps/x"), "refused_directory"),
    (diff("src/__pycache__/m.cpython-312.pyc"), "refused_directory"),
    (b"diff --git a/a b/b\nrename from a\nrename to ../../etc/x\n", "path_outside_repository"),
])
def test_patch_refusals(patch, refused):
    why = scoring.patch_refusal(patch)
    assert (why is None) if refused is None else why.startswith(refused)


def test_no_gold_patch_is_refused():
    for task_id in source.task_ids():
        assert scoring.patch_refusal(source.gold_patch(task_id)) is None, task_id
