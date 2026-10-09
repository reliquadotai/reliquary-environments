import pytest
import verifiers.v1 as vf

from reliquary_swe.taskset import BASE_REF, SweTask

docker = pytest.mark.docker


def _first_task() -> SweTask:
    config = vf.taskset_config_type("reliquary-swe")
    return next(iter(vf.load_taskset(config(id="reliquary-swe", split="eval")).head(1)))


def _trace(task: SweTask) -> vf.Trace:
    # `vf.Trace()` validates `task`/`agent` as required fields on this pinned
    # commit -- there is no bare constructor. This is the minimal shape
    # verifiers' own test suite uses (`tests/v1/test_trace.py`); `setup` and
    # `finalize` never read `task`/`agent` themselves, only `trace.info`, so
    # its content doesn't matter here beyond validating.
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
    )


def test_task_key_is_the_instance_id_not_a_content_hash():
    # Instance ids are durable across dataset revisions; content hashes are
    # not, and a run cannot be compared with an earlier one if its keys moved.
    task = _first_task()
    assert task.key == task.data.instance_id


def test_every_task_refuses_the_network():
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe", split="eval")).head(5):
        assert task.data.network_allow == []


def test_every_task_names_an_image_and_a_workdir():
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe", split="eval")).head(5):
        assert task.data.image
        assert task.data.workdir


def test_every_phase_has_a_deadline():
    # IMPORTANT 1: setup/agent/finalize used to default to None (no limit) --
    # only scoring was ever bounded. A policy that runs the repository's own
    # tests, the most natural thing a repair agent does, could otherwise hold
    # a rollout slot for hours; nothing here should be able to do that
    # unboundedly again.
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe", split="eval")).head(5):
        assert task.data.timeout.setup is not None
        assert task.data.timeout.agent is not None
        assert task.data.timeout.finalize is not None
        assert task.data.timeout.scoring is not None


def test_the_taskset_resolves_to_sweenv_by_default():
    # `SweEnv` grades in a second, isolated box; `SingleAgentEnv` (the
    # fallback for a taskset that exports no `Env`) would instead default to
    # grading in the agent's own box -- silently defeating the whole design.
    # No container needed: this is package wiring (`reliquary_swe.__all__`),
    # not a rollout.
    from reliquary_swe.env import SweEnv

    assert vf.environment_class("reliquary-swe") is SweEnv


@docker
async def test_setup_leaves_the_repository_at_the_base_commit(runtime):
    task = _first_task()
    await task.setup(_trace(task), runtime)
    head = await runtime.run(["git", "rev-parse", "HEAD"], {})
    assert head.stdout.strip() == task.data.base_commit


@docker
async def test_setup_removes_history_after_the_base_commit(runtime):
    # The published fix lives in a later commit, reachable only through a
    # surviving ref -- a branch, a tag, a remote-tracking ref. Checking
    # `base_commit..HEAD` alone -- this test's original form -- is
    # tautological once HEAD is known (by the previous test) to equal
    # base_commit: that range is empty by definition regardless of what
    # refs or history still exist, so it cannot detect the leak it is named
    # for even if `_STRIP_AND_GC`'s ref-deletion/reflog-expire/gc lines were
    # deleted outright. This checks the two properties that actually
    # matter instead: no ref survives cleanup, and no commit past
    # base_commit remains reachable through any ref that does.
    task = _first_task()
    await task.setup(_trace(task), runtime)
    # The one ref setup leaves is its own record of the base (BASE_REF -> HEAD).
    refs = await runtime.run(["git", "for-each-ref", "--format=%(refname)"], {})
    assert refs.stdout.split() == [BASE_REF]
    reachable = await runtime.run(
        [
            "sh",
            "-c",
            f"git log --oneline --all --not {task.data.base_commit} 2>/dev/null | wc -l",
        ],
        {},
    )
    assert reachable.stdout.strip() == "0"


@docker
async def test_the_container_cannot_reach_the_network(runtime):
    task = _first_task()
    await task.setup(_trace(task), runtime)
    result = await runtime.run(
        ["sh", "-c", "curl -s -m 5 https://raw.githubusercontent.com || echo BLOCKED"],
        {},
    )
    assert "BLOCKED" in result.stdout


@docker
async def test_finalize_captures_an_edit_the_agent_made(runtime):
    task = _first_task()
    trace = _trace(task)
    await task.setup(trace, runtime)
    await runtime.run(
        ["sh", "-c", "echo '# reliquary marker' >> $(git -C /testbed ls-files | head -1)"],
        {},
    )
    await task.finalize(trace, runtime)
    assert "reliquary marker" in trace.info["patch"]


@docker
async def test_finalize_does_not_credit_files_the_image_shipped(runtime):
    # Untracked files present before the agent ran must stay out of the patch,
    # or `git apply` fails in a fresh container of that same image -- which is
    # exactly what the grading box is.
    task = _first_task()
    trace = _trace(task)
    await runtime.run(["sh", "-c", "echo shipped > /testbed/shipped.txt"], {})
    await task.setup(trace, runtime)
    await task.finalize(trace, runtime)
    assert "shipped.txt" not in trace.info["patch"]


@docker
async def test_the_patch_is_written_where_artifact_collection_finds_it(runtime):
    from reliquary_swe.taskset import PATCH_PATH

    task = _first_task()
    trace = _trace(task)
    await task.setup(trace, runtime)
    await task.finalize(trace, runtime)
    assert (await runtime.run(["ls", PATCH_PATH], {})).exit_code == 0
