"""The conformance cases and references, as plain data."""

import base64

from reliquary_deepswe import conformance, source
from reliquary_deepswe.taskset import split_ids


def commands(calls):
    assert all(tool == "bash" and set(args) == {"command"} for tool, args in calls)
    return [args["command"] for _, args in calls]


def test_every_task_has_a_reference_that_uploads_its_gold_patch_and_commits():
    ids = split_ids("eval_all")
    for index, task_id in enumerate(ids):
        lines = commands(conformance.reference_calls("eval_all", index))
        chunks = [c.split(" ", 2)[2].split(" >> ")[0] for c in lines if c.startswith("printf %s")]
        assert base64.b64decode("".join(chunks)) == source.gold_patch(task_id)
        assert all(len(c) < conformance.CHUNK + 200 for c in lines)
        assert "git apply --whitespace=nowarn" in lines[-1] and "commit -q --no-verify" in lines[-1]
    assert conformance.reference_calls("eval_all", len(ids)) is None


def test_eval_cases_use_eval_indices_and_unknown_splits_have_none():
    eval_ids = split_ids("eval")
    cases = conformance.conformance_cases("eval")
    assert {c["index"] for c in cases} == {eval_ids.index(t) for t in conformance.GOLDENS}
    assert conformance.reference_calls("eval", len(eval_ids)) is None
    assert conformance.conformance_cases("train") == []


def test_cases_on_each_golden():
    cases = conformance.conformance_cases("eval_all")
    ids = split_ids("eval_all")
    assert {c["index"] for c in cases} == {ids.index(t) for t in conformance.GOLDENS}
    assert len({c["name"] for c in cases}) == len(cases)
    references = [c for c in cases if c["name"].startswith("reference_")]
    assert len(references) == len(conformance.GOLDENS)
    assert all(c["expect"] == 1.0 for c in references)
    graded_1 = {c["name"].rsplit("_", 1)[0] for c in cases
                if c["expect"] == 1.0 and c not in references}
    assert graded_1 == {"gold_uncommitted", "gold_with_node_modules_file"}
    assert all(c["expect"] in (0.0, 1.0) for c in cases)
    languages = {source.task_toml(t)["metadata"]["language"] for t in conformance.GOLDENS}
    assert languages == {"python", "go"}
    assert any(c["name"].startswith("conftest_forcing_passes_") for c in cases)
    for case in cases:
        commands(case["calls"])
