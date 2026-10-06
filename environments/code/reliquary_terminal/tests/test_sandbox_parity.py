"""The spec §6 gate for reliquary-terminal (MiMo `train`) on a live signed-episode
gateway (gVisor, network none, xfs quota): reference 1, untouched 0, the planted-conftest
tamper 0, as tests/test_train_goldens.py asserts; plus the agent removing or redirecting
/app (graded 0, never aborted), a forged reward.json that never substitutes, symlinks that
resolve into the grader's files or a kernel tree (graded 0, `link_into_grader`), a row that
runs agent code inside pytest (refused at resolve), and -- on a served subprocess row whose
tests import a mutable module -- a respawning process left in the grading box
(`grading_box_failed`) and a background process that races `stop_processes` to rewrite the
reward (the documented residual; the outcome is recorded, not asserted either way).

Opt-in, sandbox test host only -- same variables as reliquary-swe's tests/test_sandbox_parity.py:
    RUN_SANDBOX_PARITY=1                          enable
    SANDBOX_PARITY_IMAGES=<manifest.json>         images approved on the gateway (pulled)
    DOCKER_HOST, EPISODE_LIVE_DOCKER_PIDFILE      the isolated xfs daemon (test-xfs-docker.sh env)
    SANDBOX_PARITY_BUSYBOX=/usr/bin/busybox       the static helper busybox
    SANDBOX_PARITY_METRICS=<file>                 append one JSON survey line per episode

Each survey line carries the case, the expected and observed verdicts, `state_bytes`, and
the seconds the signed records attest: `prepare_s` (token issued -> record 0: box creation
plus the env's prepare) and `grade_s` (last call record -> final: process stop, extract, the
pristine box, grading). Written before any assertion, so a differing verdict is still
recorded.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.sandbox_live,
    pytest.mark.skipif(os.getenv("RUN_SANDBOX_PARITY") != "1",
                       reason="set RUN_SANDBOX_PARITY=1 on the sandbox test host"),
]

from reliquary_terminal import sandbox  # noqa: E402
from reliquary_terminal.taskset import load_train_rows  # noqa: E402

GOLDEN = "candidate-0036-software-data-engineering"
# A served subprocess row whose tests run /app/run_metrics.py, which imports the row's
# mutable modules (vendor/project/metrics/*/*.py): editing one injects code into the
# grading box's pytest subprocess without touching a manifest-protected file.
SUBPROCESS_ROW = "candidate-1165-ml-evaluation"
MUTABLE_IMPORTED = "vendor/project/metrics/f1/f1.py"
REFERENCE = Path(__file__).parent / "reference_solutions" / f"{GOLDEN}.sh"
CHUNK = 60_000


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    lg = pytest.importorskip("reliquary_sandbox_service.episodes.local_gateway")
    busybox = Path(os.getenv("SANDBOX_PARITY_BUSYBOX", "/usr/bin/busybox"))
    images = json.loads(Path(os.environ["SANDBOX_PARITY_IMAGES"]).read_text())["images"]
    with lg.local_gateway(
            tmp_path_factory.mktemp("gateway"),
            envs={"reliquary-terminal": "reliquary_terminal.sandbox:sandbox_task"}, images=images,
            episode_helper_busybox=busybox,
            episode_helper_busybox_sha256=hashlib.sha256(busybox.read_bytes()).hexdigest(),
            episode_docker_pidfile=Path(os.getenv("EPISODE_LIVE_DOCKER_PIDFILE",
                                                  "/var/run/docker.pid")),
            episode_nano_cpus=1_000_000_000, episode_capacity=1) as gw:
        yield gw


def _index(instance_id):
    return next(i for i, row in enumerate(load_train_rows()) if row["instance_id"] == instance_id)


@pytest.fixture(scope="module")
def golden():
    return _index(GOLDEN)


def bash(command):
    return ("bash", {"command": command})


def write_file(path, data: bytes):
    """Write `data` to `path` in the box, chunked under the argument limit."""
    calls = [bash(f"rm -f {path}")]
    for start in range(0, len(data), CHUNK):
        chunk = base64.b64encode(data[start:start + CHUNK]).decode()
        calls.append(bash(f"printf %s {chunk} | base64 -d >> {path}"))
    return calls


def solve():
    encoded = base64.b64encode(REFERENCE.read_bytes()).decode()
    return [bash(f"printf %s {encoded} | base64 -d > /tmp/solve.sh"), bash("sh /tmp/solve.sh")]


def _seconds(run):
    records = run.transcript["records"]
    issued = run.transcript["token"]["claims"]["issued_at"]
    opened = records[0]["body"]["at"]
    last_call = records[-2]["body"]["at"]  # record 0 when the agent made no call
    return opened - issued, run.final["at"] - last_call


async def episode(gateway, index, calls, *, case, expected=None, exits_zero=True):
    from reliquary_sandbox_service.episodes.local_gateway import run_scripted_episode

    started = time.monotonic()
    run = await run_scripted_episode(gateway, env="reliquary-terminal", split="train",
                                     index=index, calls=calls)
    prepare_s, grade_s = _seconds(run)
    facts = run.final["grading"] or {}
    inner = facts.get("grading") if isinstance(facts.get("grading"), dict) else {}
    line = {"env": "reliquary-terminal", "split": "train", "index": index, "case": case,
            "expected": expected, "status": run.final["status"], "reward": run.final["reward"],
            "reason": run.final["reason"], "prepare_s": prepare_s, "grade_s": grade_s,
            "seconds": round(time.monotonic() - started, 1),
            "cpu_total_ms": run.final["cpu_total_ms"],
            "state_bytes": (len(run.state) if run.state is not None else None),
            "calls": len(run.records), "grader": facts.get("_grader"),
            "stray_stopped": facts.get("stray_stopped"),
            "link_into_grader": facts.get("link_into_grader"),
            "exit_code": inner.get("exit_code"),
            "ctrf_counts": (inner.get("ctrf") or {}).get("counts") if inner.get("ctrf") else None,
            "ctrf_incomplete": inner.get("ctrf_incomplete"),
            "reward_without_success": inner.get("reward_without_success"),
            "output_tail": (inner.get("output_tail") or "")[-300:]}
    print("PARITY", json.dumps(line))
    if os.getenv("SANDBOX_PARITY_METRICS"):
        with open(os.environ["SANDBOX_PARITY_METRICS"], "a") as out:
            out.write(json.dumps(line) + "\n")
    assert run.verification.ok, run.verification.reasons
    if exits_zero:
        assert all(record["exit_code"] == 0 for record in run.records), run.records
    assert run.final["status"] == "graded", run.final
    if expected is not None:
        assert run.final["reward"] == expected, run.final
    return run


# -- the spec gate (brief) ------------------------------------------------------

async def test_mimo_reference_scores_one(gateway, golden):
    run = await episode(gateway, golden, solve(), case="reference", expected=1.0)
    assert run.final["grading"]["grading"]["exit_code"] == 0


async def test_mimo_untouched_scores_zero(gateway, golden):
    run = await episode(gateway, golden, [], case="untouched", expected=0.0)
    assert run.final["grading"]["grading"]["ctrf"]["counts"].get("failed", 0) > 0


async def test_mimo_planted_conftest_zeroes_a_correct_fix(gateway, golden):
    run = await episode(gateway, golden, [*solve(), bash("echo '# planted' > /app/conftest.py")],
                        case="planted conftest", expected=0.0)
    assert "anti_hack_guard" in run.final["grading"]["grading"]["output_tail"]


async def test_a_deleted_app_scores_zero_not_aborted(gateway, golden):
    run = await episode(gateway, golden, [*solve(), bash("cd / && rm -rf /app"),
                                          bash("echo bash-still-runs")],
                        case="deleted /app", expected=0.0, exits_zero=False)
    # The runner cd's into the workdir itself: once /app is gone, a later bash call is the
    # agent's own `cd` error (exit 1), reported back -- the session stays alive and keeps
    # answering, so the episode is graded 0, never aborted.
    last = run.records[-1]
    assert last["exit_code"] == 1 and "No such file or directory" in last["output"], last
    assert "_grader" not in run.final["grading"]


async def test_an_app_redirected_by_a_symlink_scores_zero_not_aborted(gateway, golden):
    run = await episode(gateway, golden, [*solve(), bash("cd / && rm -rf /app && ln -s /etc /app")],
                        case="symlinked /app", expected=0.0)
    assert "_grader" not in run.final["grading"]


# -- more review goldens --------------------------------------------------------

async def test_a_forged_reward_json_never_substitutes(gateway, golden):
    # reward.json is never read (TerminalTask._reward_json is None); a planted one in the
    # agent's /app cannot turn a failing run into a pass.
    run = await episode(gateway, golden,
                        [bash("printf '{\"reward\": 1}' > /app/reward.json")],
                        case="forged reward.json", expected=0.0)
    assert run.final["grading"]["grading"]["ctrf"]["counts"].get("failed", 0) > 0


async def test_a_link_into_the_graders_tests_is_graded_zero(gateway, golden):
    run = await episode(gateway, golden, [*solve(), bash("ln -s /tests /app/evil")],
                        case="link into /tests", expected=0.0)
    assert "/app/evil" in run.final["grading"]["link_into_grader"]


async def test_a_proc_self_root_link_is_graded_zero(gateway, golden):
    run = await episode(gateway, golden,
                        [*solve(), bash("ln -s /proc/self/root/tests /app/evil")],
                        case="link via /proc/self/root", expected=0.0)
    assert "/app/evil" in run.final["grading"]["link_into_grader"]


def test_an_excluded_in_process_row_is_refused_at_resolve():
    # No episode opens: resolve raises for a row whose tests run agent code in-process.
    rows = load_train_rows()
    excluded = next(i for i, row in enumerate(rows)
                    if "in_process_agent_code" in (sandbox._refusal(row) or ""))
    with pytest.raises(ValueError, match="in_process_agent_code"):
        sandbox.sandbox_task("train", excluded)


# -- the subprocess residual, on a served row that imports a mutable module -----
# `python3 /app/run_metrics.py` (run by the row's pytest) imports MUTABLE_IMPORTED. The
# agent may edit that module (the guard allows mutable files), so code it writes runs in the
# grading box's pytest subprocess -- the shape of every served subprocess row.

_STRAY = (
    "import os\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    os.execvp('sleep', ['sleep', '300'])\n"  # a detached process left in the box
)
_BACKGROUND_FORGE = (
    "import os\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    import time, json\n"
    "    time.sleep({delay})\n"
    "    try:\n"
    "        os.makedirs('/logs/verifier', exist_ok=True)\n"
    "        report = {{'results': {{'tests': [{{'name': 't', 'status': 'passed'}}]}}}}\n"
    "        open('/logs/verifier/ctrf.json', 'w').write(json.dumps(report))\n"
    "        open('/logs/verifier/reward.txt', 'w').write('1\\n')\n"
    "    except Exception:\n"
    "        pass\n"
    "    os._exit(0)\n"
)


def _prepend_to_mutable(payload: str):
    index = _index(SUBPROCESS_ROW)
    return index, [*write_file("/tmp/inject.py", payload.encode()),
                   bash(f"cat /app/{MUTABLE_IMPORTED} >> /tmp/inject.py "
                        f"&& cp /tmp/inject.py /app/{MUTABLE_IMPORTED}")]


async def test_a_stray_process_in_the_grading_box_is_stopped_and_graded_zero(gateway):
    # A detached process left in the grading box is SIGKILLed before the reward is read
    # (`stop_processes`, the moment test.sh returns): graded 0, never aborted, `stray_stopped`.
    # The respawning branch -- a fork bomb the kill rounds cannot drain -> `grading_box_failed`
    # -- is covered by reliquary-sandbox's own suite (test_runtime_stop_processes.py and
    # test_episode_grader.py); it is not run live, to keep an exponential fork bomb off a box
    # that shares a TMax validation.
    index, calls = _prepend_to_mutable(_STRAY)
    run = await episode(gateway, index, calls, case="stray stopped", expected=0.0)
    assert run.final["grading"]["stray_stopped"] is True
    assert "_grader" not in run.final["grading"]


async def test_a_background_reward_rewrite_races_stop_processes(gateway):
    # The residual: a detached child that rewrites reward.txt + a passing report after
    # test.sh. Record which way the race goes; do not assert its direction.
    index, calls = _prepend_to_mutable(_BACKGROUND_FORGE.format(delay="1.5"))
    run = await episode(gateway, index, calls, case="background reward rewrite", exits_zero=False)
    assert run.final["status"] == "graded"
    assert run.final["reward"] in (0.0, 1.0), run.final
