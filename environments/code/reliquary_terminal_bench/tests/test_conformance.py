"""References and cases are plain data the sandbox can play through `bash`."""

import base64
import io
import tarfile

from reliquary_terminal_bench import conformance, gate, source
from reliquary_terminal_bench.conformance import CHUNK


def decoded_upload(calls):
    chunks = [c[1]["command"].split(" ", 2)[2].split(" >> ")[0] for c in calls
              if c[1]["command"].startswith("printf %s ")]
    return base64.b64decode("".join(chunks))


def test_a_reference_uploads_the_solution_and_runs_solve_sh():
    calls = conformance.solve_calls("fix-git")
    assert all(c[0] == "bash" for c in calls)
    archive = decoded_upload(calls)
    names = tarfile.open(fileobj=io.BytesIO(archive)).getnames()
    assert "solve.sh" in names
    assert calls[-1][1]["command"].endswith("cd /app/personal-site && bash /solution/solve.sh")
    assert all(len(c[1]["command"]) <= CHUNK + 64 for c in calls)


def test_reference_calls_by_index():
    ids = source.task_ids()
    assert conformance.reference_calls("eval_all", ids.index("regex-log")) == \
        conformance.solve_calls("regex-log")
    assert conformance.reference_calls("eval_all", len(ids)) is None


def test_goldens_name_files_their_tests_read():
    for task_id, output in conformance.GOLDENS.items():
        assert task_id not in gate.EXCLUDED
        tests = b"".join(source.read(task_id, r) for r in source.files_of(task_id, "tests/"))
        assert output.encode() in tests


def test_cases_cover_reference_noop_mtime_deletion_and_plant():
    cases = conformance.conformance_cases("eval_all")
    kinds = {c["name"][len(t) + 1:] for c in cases for t in conformance.GOLDENS
             if c["name"].startswith(t + "-")}
    assert kinds == {"reference", "noop", "old-mtimes", "output-deleted", "planted-state"}
    assert len(cases) == 5 * len(conformance.GOLDENS)
    for case in cases:
        assert case["expect"] in (0.0, 1.0) and case["calls"]
