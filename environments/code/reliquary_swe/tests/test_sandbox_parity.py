"""The spec §6 gate for reliquary-swe: on a local signed-episode gateway (gVisor, network
none, xfs quota), the reference fix scores 1, an untouched repository 0, and the container
goldens' tamper patches 0, the same verdicts tests/test_*_goldens.py assert. Two more
agent behaviours only a live box shows: a runner outside the tracked tree, force-added,
committed and edited behind a moved base ref, scores 0 unapplied; a deleted workdir
scores 0, never `aborted` or `extract_failed`.

Opt-in, sandbox test host only:
    RUN_SANDBOX_PARITY=1                         enable
    SANDBOX_PARITY_IMAGES=<manifest.json>        images approved on the gateway (pulled)
    DOCKER_HOST, EPISODE_LIVE_DOCKER_PIDFILE     the isolated xfs daemon (test-xfs-docker.sh env)
    SANDBOX_PARITY_BUSYBOX=/usr/bin/busybox      the static helper busybox
    SANDBOX_PARITY_METRICS=<file>                append one JSON line per episode

Each metrics line carries the case, the expected and observed verdicts, and the seconds
the signed records attest: `prepare_s` (token issued -> record 0: box creation + the env's
prepare) and `grade_s` (last call record -> final: process stop, extract, pristine box,
grading). Written before any assertion, so a differing verdict is still recorded.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.sandbox_live,
    pytest.mark.skipif(os.getenv("RUN_SANDBOX_PARITY") != "1",
                       reason="set RUN_SANDBOX_PARITY=1 on the sandbox test host"),
]

from test_polyglot_goldens import SHAPE_ONE  # noqa: E402
from test_r2e_goldens import COVERAGEPY, _new_file  # noqa: E402
from test_swesmith_goldens import (  # noqa: E402
    _TAMPERED_TEST_UTILS,
    FORCE_PASS_PATCH,
    SWESMITH_GOLDEN,
)

from reliquary_swe import corpus, sandbox  # noqa: E402

CHUNK = 60_000
GIT = "git -c user.name=a -c user.email=a@a"


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    lg = pytest.importorskip("reliquary_sandbox_service.episodes.local_gateway")
    busybox = Path(os.getenv("SANDBOX_PARITY_BUSYBOX", "/usr/bin/busybox"))
    images = json.loads(Path(os.environ["SANDBOX_PARITY_IMAGES"]).read_text())["images"]
    with lg.local_gateway(
            tmp_path_factory.mktemp("gateway"),
            envs={"reliquary-swe": "reliquary_swe.sandbox:sandbox_task"}, images=images,
            episode_helper_busybox=busybox,
            episode_helper_busybox_sha256=hashlib.sha256(busybox.read_bytes()).hexdigest(),
            episode_docker_pidfile=Path(os.getenv("EPISODE_LIVE_DOCKER_PIDFILE",
                                                  "/var/run/docker.pid")),
            episode_nano_cpus=1_000_000_000, episode_capacity=1) as gw:
        yield gw


def bash(command):
    return ("bash", {"command": command})


def write_file(path, data: bytes):
    calls = [bash(f"rm -f {path}")]
    for start in range(0, len(data), CHUNK):
        chunk = base64.b64encode(data[start:start + CHUNK]).decode()
        calls.append(bash(f"printf %s {chunk} | base64 -d >> {path}"))
    return calls


def apply(patch: str):
    return [*write_file("/tmp/agent.patch", patch.encode()), bash("git apply /tmp/agent.patch")]


def _seconds(run):
    records = run.transcript["records"]
    issued = run.transcript["token"]["claims"]["issued_at"]
    opened = records[0]["body"]["at"]
    last_call = records[-2]["body"]["at"]  # record 0 when the agent made no call
    return opened - issued, run.final["at"] - last_call


async def episode(gateway, split, index, calls, *, case, expected, exits_zero=True):
    from reliquary_sandbox_service.episodes.local_gateway import run_scripted_episode

    started = time.monotonic()
    started_at = time.time()
    run = await run_scripted_episode(gateway, env="reliquary-swe", split=split, index=index,
                                     calls=calls)
    prepare_s, grade_s = _seconds(run)
    facts = run.final["grading"] or {}
    line = {"env": "reliquary-swe", "split": split, "index": index, "case": case,
            "expected": expected, "status": run.final["status"], "reward": run.final["reward"],
            "reason": run.final["reason"], "prepare_s": prepare_s, "grade_s": grade_s,
            "seconds": round(time.monotonic() - started, 1), "started_at": round(started_at, 1),
            "ended_at": round(time.time(), 1), "cpu_total_ms": run.final["cpu_total_ms"],
            "calls": len(run.records), "applied": facts.get("applied"),
            "restored": facts.get("restored"), "grader": facts.get("_grader"),
            "test_output_tail": (facts.get("test_output_tail") or "")[-300:]}
    print("PARITY", json.dumps(line))
    if os.getenv("SANDBOX_PARITY_METRICS"):
        with open(os.environ["SANDBOX_PARITY_METRICS"], "a") as out:
            out.write(json.dumps(line) + "\n")
    assert run.verification.ok, run.verification.reasons
    if exits_zero:
        assert all(record["exit_code"] == 0 for record in run.records), run.records
    assert run.final["status"] == "graded", run.final
    assert run.final["reward"] == expected, run.final
    if expected == 0.0:
        # A zero from the package's own grading, never a sandbox verdict (`_grader`).
        assert "_grader" not in (run.final["grading"] or {}), run.final
    return run


# -- SWE-smith (oauthlib) -------------------------------------------------------

@pytest.fixture(scope="module")
def smith():
    return sandbox.sandbox_index("train", SWESMITH_GOLDEN)


async def test_swesmith_reference_scores_one(gateway, smith):
    run = await episode(gateway, "train", smith,
                        apply(corpus.swesmith_row(SWESMITH_GOLDEN).gold_patch),
                        case="reference", expected=1.0)
    assert run.state is not None and b"diff --git" in run.state


async def test_swesmith_committed_reference_scores_one(gateway, smith):
    calls = [*apply(corpus.swesmith_row(SWESMITH_GOLDEN).gold_patch),
             bash(f"{GIT} commit -qam fix")]
    await episode(gateway, "train", smith, calls, case="committed reference", expected=1.0)


async def test_swesmith_untouched_scores_zero(gateway, smith):
    await episode(gateway, "train", smith, [], case="untouched", expected=0.0)


async def test_swesmith_forced_pass_conftest_scores_zero(gateway, smith):
    run = await episode(gateway, "train", smith, apply(FORCE_PASS_PATCH),
                        case="forced-pass conftest", expected=0.0)
    assert run.final["grading"]["applied"] is True and run.final["grading"]["restored"] is True


async def test_swesmith_overwritten_test_file_scores_zero(gateway, smith):
    calls = write_file("tests/oauth2/rfc6749/test_utils.py", _TAMPERED_TEST_UTILS.encode())
    run = await episode(gateway, "train", smith, calls, case="overwritten test file",
                        expected=0.0)
    assert run.final["grading"]["restored"] is True


# The image's editable install puts /testbed on sys.path, so its ignored
# `oauthlib.egg-info/` is the distribution pytest reads entry points from: a `[pytest11]`
# entry there loads a plugin that forces every outcome to "passed". The agent force-adds
# the ignored directory, commits it, moves the base ref onto that commit, then edits it and
# force-adds the entry point, so its capture carries the ignored paths.
_FORCE_PLUGIN = FORCE_PASS_PATCH.split("+++ b/conftest.py\n", 1)[1].split("\n", 1)[1]
_FORCE_PLUGIN = "".join(line[1:] + "\n" for line in _FORCE_PLUGIN.splitlines())


async def test_swesmith_committed_ignored_runner_behind_a_moved_base_scores_zero(gateway, smith):
    calls = [bash(f"{GIT} add -f oauthlib.egg-info && {GIT} commit -qm meta"
                  " && git update-ref refs/reliquary/base HEAD"),
             *write_file("oauthlib/zz_force.py", _FORCE_PLUGIN.encode()),
             bash("printf '[pytest11]\\nzz = oauthlib.zz_force\\n'"
                  " > oauthlib.egg-info/entry_points.txt"
                  " && echo zz >> oauthlib.egg-info/top_level.txt"
                  " && git add -f oauthlib.egg-info/entry_points.txt")]
    run = await episode(gateway, "train", smith, calls,
                        case="ignored runner committed, base moved", expected=0.0)
    grading = run.final["grading"]
    assert grading["applied"] is False, grading
    assert "outside the tracked tree" in grading["test_output_tail"]
    assert "oauthlib.egg-info/" in grading["test_output_tail"]
    assert b"oauthlib.egg-info/top_level.txt" in run.state


async def test_swesmith_deleted_workdir_scores_zero(gateway, smith):
    run = await episode(gateway, "train", smith, [bash("cd / && rm -rf /testbed")],
                        case="rm -rf workdir", expected=0.0, exits_zero=False)
    assert run.state == b"" or run.state is None


# -- R2E (coveragepy) -----------------------------------------------------------

@pytest.fixture(scope="module")
def r2e():
    row = corpus.r2e_row(COVERAGEPY)
    return row, sandbox.sandbox_index("r2e", row.instance_id)


async def test_r2e_reference_scores_one(gateway, r2e):
    row, index = r2e
    await episode(gateway, "r2e", index, apply(row.gold_patch), case="reference", expected=1.0)


async def test_r2e_untouched_scores_zero(gateway, r2e):
    await episode(gateway, "r2e", r2e[1], [], case="untouched", expected=0.0)


async def test_r2e_forged_runner_and_tests_score_zero(gateway, r2e):
    row, index = r2e
    names = json.loads(row.expected_output_json)
    summary = "\n".join(["echo '=== short test summary info ==='"] + [
        f"echo '{status} r2e_tests/test_1.py::{name.replace('.', '::')}'"
        for name, status in names.items()])
    forged = _new_file("run_tests.sh", summary) + _new_file(
        "r2e_tests/test_1.py", "def test_nothing():\n    pass\n")
    await episode(gateway, "r2e", index, apply(forged), case="forged runner + tests",
                  expected=0.0)


async def test_r2e_root_conftest_forcing_passes_scores_zero(gateway, r2e):
    conftest = ("import pytest\n\n@pytest.hookimpl(hookwrapper=True)\n"
                "def pytest_runtest_makereport(item, call):\n    outcome = yield\n"
                "    outcome.get_result().outcome = 'passed'\n")
    await episode(gateway, "r2e", r2e[1], apply(_new_file("conftest.py", conftest)),
                  case="root conftest", expected=0.0)


async def test_r2e_root_module_shadowing_pytest_scores_zero(gateway, r2e):
    row, index = r2e
    names = json.loads(row.expected_output_json)
    shim = "\n".join(["import sys", "print('=== short test summary info ===')"] + [
        f"print('PASSED r2e_tests/test_1.py::{name.replace('.', '::')}')" for name in names]
        + ["sys.exit(0)"])
    await episode(gateway, "r2e", index, apply(_new_file("pytest.py", shim)),
                  case="root pytest.py shadow", expected=0.0)


async def test_r2e_committed_venv_startup_file_behind_a_moved_base_scores_zero(gateway, r2e):
    # run_tests.sh runs .venv/bin/python, which executes every site-packages *.pth at
    # start-up: the ignored .venv is the runner. Force-added, committed, base moved, then
    # rewritten to print the expected summary and exit before pytest runs.
    row, index = r2e
    names = json.loads(row.expected_output_json)
    lines = ["=== short test summary info ==="] + [
        f"PASSED r2e_tests/test_1.py::{name.replace('.', '::')}" for name in names]
    payload = json.dumps("\n".join(lines) + "\n")
    pth = f"import os; os.write(1, {payload}.encode()); os._exit(0)\n"
    target = ".venv/lib/python3.7/site-packages/distutils-precedence.pth"
    calls = [bash(f"test -f {target} && {GIT} add -f {target} && {GIT} commit -qm venv"
                  " && git update-ref refs/reliquary/base HEAD"),
             *write_file(target, pth.encode())]
    run = await episode(gateway, "r2e", index, calls,
                        case="ignored .venv runner committed, base moved", expected=0.0)
    grading = run.final["grading"]
    assert grading["applied"] is False, grading
    assert target in grading["test_output_tail"]


async def test_r2e_deleted_workdir_scores_zero(gateway, r2e):
    run = await episode(gateway, "r2e", r2e[1], [bash("cd / && rm -rf /testbed")],
                        case="rm -rf workdir", expected=0.0, exits_zero=False)
    assert run.state == b"" or run.state is None


# -- polyglot (Go, shape one) ---------------------------------------------------
# No tamper golden for an ignored runner here: this image has no untracked or ignored
# path at its base (`git status --ignored` is empty, checked on the image).

@pytest.fixture(scope="module")
def polyglot():
    index = sandbox.sandbox_index("polyglot", SHAPE_ONE)
    found = sandbox.declaration("polyglot", index)
    script = ('cd "$1" && ref=$(git for-each-ref --format="%(refname)" refs/remotes '
              '| grep -v HEAD | head -n 1) && git diff --binary HEAD "$ref"')
    reference = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", found.image,
         "-c", script, "sh", found.workdir], capture_output=True, check=True, timeout=900).stdout
    return index, found, reference


async def test_polyglot_reference_scores_one(gateway, polyglot):
    index, _, reference = polyglot
    calls = [*write_file("/tmp/agent.patch", reference), bash("git apply /tmp/agent.patch")]
    await episode(gateway, "polyglot", index, calls, case="reference", expected=1.0)


async def test_polyglot_untouched_scores_zero(gateway, polyglot):
    await episode(gateway, "polyglot", polyglot[0], [], case="untouched", expected=0.0)


async def test_polyglot_rewritten_test_script_scores_zero(gateway, polyglot):
    forged = ("diff --git a/mimo_test_command.sh b/mimo_test_command.sh\nnew file mode 100755\n"
              "--- /dev/null\n+++ b/mimo_test_command.sh\n@@ -0,0 +1,2 @@\n"
              "+#!/usr/bin/env bash\n+exit 0\n")
    await episode(gateway, "polyglot", polyglot[0], apply(forged),
                  case="rewritten test script", expected=0.0)


async def test_polyglot_guessed_hidden_test_file_scores_zero(gateway, polyglot):
    from reliquary_swe import grading

    index, found, _ = polyglot
    hidden = next(p for p in grading._paths_touched_by(found.data.test_patch)
                  if p.endswith("_test.go"))
    fake = (f"diff --git a/{hidden} b/{hidden}\nnew file mode 100644\n--- /dev/null\n"
            f"+++ b/{hidden}\n@@ -0,0 +1,5 @@\n+package spew_test\n+\n+import \"testing\"\n+\n"
            "+func TestNothing(t *testing.T) {}\n")
    await episode(gateway, "polyglot", index, apply(fake), case="guessed hidden test file",
                  expected=0.0)


async def test_polyglot_deleted_workdir_scores_zero(gateway, polyglot):
    run = await episode(gateway, "polyglot", polyglot[0],
                        [bash("cd / && rm -rf /workspace/repo")],
                        case="rm -rf workdir", expected=0.0, exits_zero=False)
    assert run.state == b"" or run.state is None
