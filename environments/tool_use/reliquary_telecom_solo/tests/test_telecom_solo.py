import ast
import asyncio
import hashlib
import importlib.resources
import io
import json
import re
import tokenize
import tomllib
from pathlib import Path

import pytest

import reliquary_telecom_solo as package
from reliquary_telecom_solo import corpus
from reliquary_telecom_solo._tau2.data_model import TelecomDB
from reliquary_telecom_solo._tau2.environment import TelecomSoloWorld
from reliquary_telecom_solo._tau2.user_data_model import TelecomUserDB
from reliquary_telecom_solo.environment import (
    ENVIRONMENT,
    MAX_ERRORS,
    MAX_TURNS,
    STOP_TOKEN,
    STOP_TOOL,
    TelecomSoloEnvironment,
    build_world,
)
from reliquary_telecom_solo.grading import grade, transcript_digest

try:
    import verifiers.v1 as vf
except ModuleNotFoundError:  # pragma: no cover - exercised by the packaged wheel
    vf = None

# Verifiers is the training-side surface. Replay, grading and the corpus do not
# import it, and these tests say so by still running when it is absent.
needs_verifiers = pytest.mark.skipif(vf is None, reason="verifiers is not installed")

FAMILIES = {"mms_issue": 1984, "mobile_data_issue": 254, "service_issue": 47}
# 2,253 tickets are scored on assertions alone; the other 32 also require the
# agent to hand over to a human.
ACTION_SCORED = 32
PACKAGE_ROOT = Path(package.__file__).parent


def _goldens() -> list[dict]:
    return [
        json.loads(line)
        for line in importlib.resources.files("reliquary_telecom_solo")
        .joinpath("goldens/reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _trace(task, actions: list[dict]):
    """A trace shaped like one the harness would hand back."""
    environment = TelecomSoloEnvironment(task.data.split)
    state = environment.reset(task.data.idx or 0)["state"]
    nodes = []
    parent = None
    for position, action in enumerate(actions):
        call_id = f"call-{position}"
        nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.AssistantMessage(
                    tool_calls=[
                        vf.ToolCall(
                            id=call_id,
                            name=action["tool"],
                            arguments=json.dumps(action["arguments"]),
                        )
                    ]
                ),
                sampled=True,
            )
        )
        parent = len(nodes) - 1
        result = environment.call(state, action["tool"], action["arguments"])
        nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.ToolMessage(
                    tool_call_id=call_id,
                    name=action["tool"],
                    content=result["content"],
                ),
            )
        )
        parent = len(nodes) - 1
    from reliquary_telecom_solo.taskset import TelecomSoloState

    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        state=TelecomSoloState(index=task.data.idx or 0, split=task.data.split),
    )


def test_the_corpus_matches_its_pin() -> None:
    """The digest covers the tickets, not just the file.

    A corpus that arrived differently is a different environment: the same
    index would address a different ticket, and every measurement taken against
    it would be about something else. The starting databases and the policy are
    pinned for the same reason — a ticket is a problem only in the world it
    arrives in.
    """
    import gzip

    body = gzip.decompress(
        importlib.resources.files("reliquary_telecom_solo")
        .joinpath(corpus.TASKS_FILE)
        .read_bytes()
    )
    assert hashlib.sha256(body).hexdigest() == corpus.TASKS_SHA256
    tasks = corpus.tasks()
    assert len(tasks) == corpus.TASK_COUNT == 2285
    assert len({task.key for task in tasks}) == len(tasks)

    for name, expected in (
        (corpus.DB_FILE, corpus.DB_SHA256),
        (corpus.USER_DB_FILE, corpus.USER_DB_SHA256),
        (corpus.MAIN_POLICY_FILE, corpus.MAIN_POLICY_SHA256),
        (corpus.WORKFLOW_POLICY_FILE, corpus.WORKFLOW_POLICY_SHA256),
        (corpus.TOOL_SCHEMAS_FILE, corpus.TOOL_SCHEMAS_SHA256),
    ):
        raw = (
            importlib.resources.files("reliquary_telecom_solo")
            .joinpath(name)
            .read_bytes()
        )
        assert hashlib.sha256(raw).hexdigest() == expected, name


def test_every_task_carries_a_ticket() -> None:
    """Solo mode has no conversation, so the ticket is the whole problem statement.

    Telecom is the only τ² domain where every task has one, which is why it is
    the only one that can be ported this way.
    """
    for task in corpus.tasks():
        assert task.ticket.strip()
        assert task.assertions, task.key
        assert set(task.reward_basis) <= set(corpus.REWARD_TYPES)

    basis = [task.reward_basis for task in corpus.tasks()]
    assert sum(1 for entry in basis if "ACTION" in entry) == ACTION_SCORED
    assert all("ENV_ASSERTION" in entry for entry in basis)


def test_the_families_are_what_the_manifest_says() -> None:
    counts: dict[str, int] = {}
    for task in corpus.tasks():
        counts[task.family] = counts.get(task.family, 0) + 1
    assert counts == FAMILIES

    manifest = tomllib.loads((Path(__file__).parents[1] / "environment.toml").read_text())
    assert sorted(manifest["task_families"]) == sorted(FAMILIES)
    assert manifest["execution"]["max_turns"] == MAX_TURNS
    assert manifest["data"]["rows"] == corpus.TASK_COUNT
    assert manifest["data"]["sha256"] == corpus.TASKS_SHA256
    assert manifest["compatibility_entrypoint"].endswith(":TelecomSoloEnvironment")


def test_splits_are_disjoint_and_cover_the_corpus() -> None:
    pools = {split: corpus.rows(split) for split in corpus.SPLITS}
    keys = {split: {task.key for task in tasks} for split, tasks in pools.items()}
    assert sum(len(pool) for pool in pools.values()) == corpus.TASK_COUNT
    assert set.union(*keys.values()) == {task.key for task in corpus.tasks()}
    for left in corpus.SPLITS:
        for right in corpus.SPLITS:
            if left != right:
                assert not keys[left] & keys[right]


def test_splits_preserve_the_family_mix() -> None:
    """Drawn inside each family, so the mix is exact rather than probable.

    A hash over the whole corpus would be unbiased in expectation and wrong in
    fact: 10% of the 47 service tickets is 4.7, and where that lands decides
    whether the eval split holds two of them or nine.
    """
    whole = {
        family: 100 * count / corpus.TASK_COUNT for family, count in FAMILIES.items()
    }
    for split in corpus.SPLITS:
        pool = corpus.rows(split)
        for family, expected in whole.items():
            share = 100 * sum(1 for task in pool if task.family == family) / len(pool)
            assert abs(share - expected) < 1.0, (split, family, share, expected)


def test_the_agent_cannot_call_its_own_marker() -> None:
    """The predicates that score the episode are not tools, and cannot become tools.

    This is the one invariant that would quietly turn the reward into a
    formality: an agent that could call `assert_can_send_mms` could read its
    own mark, and one that could call `unseat_sim_card` or
    `suspend_line_for_overdue_bill` could set the scenario it is being marked
    against — or undo it.
    """
    db, user_db = corpus.databases()
    world = TelecomSoloWorld(
        TelecomDB.model_validate(db), TelecomUserDB.model_validate(user_db)
    )
    offered = {schema["function"]["name"] for schema in corpus.tool_schemas()}

    hidden = {
        name
        for toolkit in (world.tools, world.user_tools)
        for name in dir(toolkit)
        if name.startswith("assert_")
    }
    assert len(hidden) >= 12
    assert not hidden & offered
    for name in hidden:
        assert not world.has_tool(name)
        with pytest.raises(ValueError):
            world.call(name, {})

    # Every predicate any task names is one of the hidden ones.
    named = {assertion.func_name for task in corpus.tasks() for assertion in task.assertions}
    assert named <= hidden
    assert named == {
        "assert_can_send_mms",
        "assert_data_refueling_amount",
        "assert_internet_speed",
        "assert_mobile_data_status",
        "assert_no_overdue_bill",
        "assert_service_status",
    }

    # The helpers that break something are equally out of reach. Three of the
    # setup calls are ordinary tools — turning roaming on is both how a
    # scenario is arranged and something a support agent may legitimately do —
    # and the rest, every `break_*`, `unseat_sim_card`, `lock_sim_card` and
    # `set_data_usage` among them, exist only for the setup.
    scenario = {call.func_name for task in corpus.tasks() for call in task.setup}
    assert scenario & offered == {
        "disable_roaming",
        "enable_roaming",
        "set_network_mode_preference",
    }
    assert len(scenario - offered) == 17
    # Upstream grants the agent `grant_app_permission` and keeps
    # `remove_app_permission` for itself, which is the asymmetry the MMS
    # tickets turn on: the agent can restore a permission the setup took away
    # and cannot take one away to make a ticket look solved.
    assert "grant_app_permission" in offered
    assert "remove_app_permission" not in offered


def test_assertions_are_predicates() -> None:
    """Every assertion returns a bool on the world its own task starts in.

    Upstream raises when one returns something else, which would cost a whole
    window over one rollout; this package scores it zero instead. The test is
    what says that path is unreachable rather than merely forgiving.
    """
    seen: set[tuple[str, str]] = set()
    for task in corpus.tasks():
        shapes = {(a.env_type, a.func_name) for a in task.assertions}
        if shapes <= seen:
            continue
        seen |= shapes
        world = build_world(task)
        for assertion in task.assertions:
            result = world.run_function(
                assertion.env_type, assertion.func_name, assertion.arguments
            )
            assert isinstance(result, bool), (task.key, assertion.func_name)


def test_tool_schemas_match_the_toolkits() -> None:
    """The frozen schema is a copy of the live signatures, and stays one.

    The schemas are generated once with upstream's own reflection code and
    shipped as data, so that a docstring parser is not a runtime dependency and
    a library upgrade cannot silently rewrite the agent's prompt. What the
    freezing costs is the chance of drift, and this is what pays it back.
    """
    import inspect

    db, user_db = corpus.databases()
    world = TelecomSoloWorld(
        TelecomDB.model_validate(db), TelecomUserDB.model_validate(user_db)
    )
    schemas = {schema["function"]["name"]: schema["function"] for schema in corpus.tool_schemas()}
    live = set(world.tools._tool_names()) | set(world.user_tools._tool_names())
    assert set(schemas) == live | {STOP_TOOL}
    assert len(schemas) == corpus.TOOL_COUNT

    for name in live:
        signature = inspect.signature(getattr(world.toolkit_of(name), name))
        declared = set(schemas[name]["parameters"].get("properties", {}))
        assert declared == set(signature.parameters), name
        required = set(schemas[name]["parameters"].get("required", []))
        assert required == {
            parameter
            for parameter, value in signature.parameters.items()
            if value.default is inspect.Parameter.empty
        }, name
        assert schemas[name]["description"].strip()


def test_nothing_reachable_is_non_deterministic() -> None:
    """Grep the package for the things that make a replay disagree with itself.

    This is the regression guard that matters most: every other test here would
    still pass on a package that had picked up a clock or a random source, and
    the failure it would cause — a validator computing a different reward, or
    the same reward over a different transcript — surfaces as an accusation of
    cheating rather than as a bug.

    Comments and docstrings are stripped first, so the prose in this package
    that names `uuid.uuid4` in order to explain why it is gone does not trip
    the check it is describing.
    """
    banned = re.compile(
        r"\b(?:"
        r"time\s*\.\s*(?:time|monotonic|perf_counter)"
        r"|datetime\s*\.\s*(?:now|utcnow|today)"
        r"|date\s*\.\s*today"
        r"|uuid"
        r"|random"
        r"|requests"
        r"|httpx"
        r"|socket"
        r"|urllib"
        r"|getenv"
        r"|environ"
        r")\b"
    )
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert len(sources) >= 8
    for path in sources:
        code = _code_without_prose(path)
        hits = sorted(set(banned.findall(code)) | {m.group(0) for m in banned.finditer(code)})
        assert not hits, f"{path.name}: {hits}"


def _code_without_prose(path: Path) -> str:
    """The module's source with comments and docstrings blanked out.

    String literals that are not docstrings are kept: `__import__("uuid")`
    would be exactly the kind of thing worth catching, and blanking every
    string would hide it.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add((first.value.lineno, first.value.col_offset))

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and token.start in docstrings:
            continue
        kept.append(token.string)
    return "\n".join(kept)


def test_a_replayed_episode_repeats_itself() -> None:
    """Same actions, same reward — and same transcript, which is the harder half.

    A reward that repeats proves the world ended in the same place. A
    transcript that repeats proves nothing the agent read along the way was
    invented, and that is what a network paying for tokens has to be able to
    check.
    """
    for golden in _goldens():
        environment = TelecomSoloEnvironment(golden["split"])
        first = environment.replay(golden["index"], golden["actions"])
        second = environment.replay(golden["index"], golden["actions"])
        assert first["transcript_digest"] == second["transcript_digest"]
        assert first["reward"]["state_digest"] == second["reward"]["state_digest"]
        assert first["reward"]["reward"] == second["reward"]["reward"]
        assert first["events"] == second["events"]

        assert first["task"]["id"] == golden["task_id"]
        assert first["reward"]["reward"] == golden["reward"]
        assert first["termination_reason"] == golden["termination_reason"]
        assert first["transcript_digest"] == golden["transcript_digest"]
        assert first["reward"]["state_digest"] == golden["state_digest"]


def test_a_draft_bill_is_named_after_the_bill() -> None:
    """The one upstream defect this port fixes, and the shape of the fix.

    `_apply_one_time_charge` built its draft-bill id from `uuid.uuid4()`. No
    assertion reads a bill id, so the reward never noticed — but the id is
    handed back to the agent by `get_bills_for_customer`, so the transcript
    could not repeat. The replacement is a hash of the customer and the billing
    period, which is unique because a customer has one draft bill per period.
    """
    environment = TelecomSoloEnvironment("train")
    # C1002 has no draft bill in the shipped database, so refuelling one of
    # their lines takes the branch that creates one. C1001, whom every
    # reference solution refuels, already has one and never does.
    actions = [
        {
            "tool": "refuel_data",
            "arguments": {"customer_id": "C1002", "line_id": "L1004", "gb_amount": 1.0},
        },
        {"tool": "get_bills_for_customer", "arguments": {"customer_id": "C1002"}},
    ]
    first = environment.replay(0, actions)
    second = environment.replay(1, actions)
    bills = json.loads(first["events"][1]["content"])
    drafts = [bill for bill in bills if bill["status"] == "Draft"]
    assert len(drafts) == 1
    assert re.fullmatch(r"B[0-9a-f]{8}", drafts[0]["bill_id"])
    assert drafts[0]["period_start"] == "2025-03-01"
    # Same customer and period in a different task: same id.
    assert drafts[0]["bill_id"] in second["events"][1]["content"]


def test_the_clock_is_frozen() -> None:
    """Every telecom tool believes it is 2025-02-25, and that is what makes
    a bill period computed today the same one computed next month."""
    from reliquary_telecom_solo._tau2.toolkit import get_now, get_today

    assert get_today().isoformat() == "2025-02-25"
    assert get_now().isoformat() == "2025-02-25T12:08:00"

    environment = TelecomSoloEnvironment("train")
    episode = environment.replay(
        0,
        [
            {
                "tool": "get_data_usage",
                "arguments": {"customer_id": "C1001", "line_id": "L1001"},
            }
        ],
    )
    assert json.loads(episode["events"][0]["content"])["cycle_end_date"] == "2025-02-28"


def test_a_reference_solution_scores_and_a_truncated_one_does_not() -> None:
    """Both halves, over the whole corpus of the qualification split.

    The second half is not a formality. For the 32 action-scored tickets the
    assertion already holds when the episode starts — the SIM is locked and the
    ticket cannot be solved — so without the action component an agent that did
    nothing would collect the reward.
    """
    environment = TelecomSoloEnvironment("qualification")
    for index in range(len(environment)):
        actions = environment.reference_actions(index)
        assert environment.replay(index, actions)["reward"]["reward"] == 1.0
        assert environment.replay(index, [])["reward"]["reward"] == 0.0
        assert (
            environment.replay(index, [{"tool": STOP_TOOL, "arguments": {}}])["reward"][
                "reward"
            ]
            == 0.0
        )


def test_the_reference_solution_scores_on_every_split() -> None:
    """The same, sampled across train and eval, which are too large to sweep here."""
    for split in ("train", "eval"):
        environment = TelecomSoloEnvironment(split)
        for index in range(0, len(environment), 37):
            assert (
                environment.replay(index, environment.reference_actions(index))[
                    "reward"
                ]["reward"]
                == 1.0
            )


def test_an_episode_ends_when_the_agent_says_so() -> None:
    environment = TelecomSoloEnvironment("train")
    episode = environment.replay(
        0,
        [
            {"tool": "check_status_bar", "arguments": {}},
            {"tool": STOP_TOOL, "arguments": {}},
            {"tool": "check_status_bar", "arguments": {}},
        ],
    )
    assert episode["termination_reason"] == "finished"
    assert len(episode["events"]) == 2
    assert episode["events"][1]["content"] == STOP_TOKEN


def test_a_failing_tool_call_is_reported_and_bounded() -> None:
    """Errors are ordinary here, so one must not end the episode — and ten must.

    Looking a customer up by a number that is not theirs raises, and looking is
    how the agent finds out. Upstream's orchestrator tolerates the same ten.
    """
    environment = TelecomSoloEnvironment("train")
    episode = environment.replay(
        0,
        [
            {"tool": "get_customer_by_phone", "arguments": {"phone_number": "000"}},
            {"tool": "check_status_bar", "arguments": {}},
            {"tool": STOP_TOOL, "arguments": {}},
        ],
    )
    assert episode["events"][0]["content"].startswith("Error: ")
    assert episode["termination_reason"] == "finished"

    flood = [
        {"tool": "get_customer_by_phone", "arguments": {"phone_number": "000"}}
    ] * (MAX_ERRORS + 4)
    assert environment.replay(0, flood)["termination_reason"] == "too_many_errors"

    unknown = environment.replay(0, [{"tool": "assert_can_send_mms", "arguments": {}}])
    assert unknown["events"][0]["content"] == "Error: Tool 'assert_can_send_mms' not found."


def test_an_episode_stops_at_the_turn_limit() -> None:
    environment = TelecomSoloEnvironment("train")
    episode = environment.replay(
        0, [{"tool": "check_status_bar", "arguments": {}}] * (MAX_TURNS + 5)
    )
    assert episode["termination_reason"] == "turn_limit"
    assert len(episode["actions"]) == MAX_TURNS


def test_a_malformed_action_ends_the_episode() -> None:
    environment = TelecomSoloEnvironment("train")
    for action in ({"final": "done"}, {"tool": 7, "arguments": {}}, {"tool": "x"}):
        episode = environment.replay(0, [action])
        assert episode["termination_reason"] == "invalid_action"
        assert episode["reward"]["reward"] == 0.0


def test_the_prompt_carries_the_ticket_and_the_solo_policy() -> None:
    """The prompt is part of the environment, not of whichever harness runs it."""
    environment = TelecomSoloEnvironment("train")
    task = environment.task(0)
    prompt = task["prompt"]
    assert environment._task(0).ticket in prompt
    assert "<main_policy>" in prompt and "<tech_support_policy>" in prompt
    assert f"`{STOP_TOOL}`" in prompt
    assert "You cannot communicate with the user" in prompt
    # The solo policies, not their dual-control siblings: nothing here should
    # be telling the agent to ask a user to do something.
    assert "Ask the user" not in prompt
    assert task["metadata"]["split"] == "train"
    assert len(task["tools"]) == corpus.TOOL_COUNT


def test_the_task_identity_follows_the_ticket() -> None:
    """A task is named after the ticket it grades, whatever index it sits at.

    Named by a digest of the ticket's key rather than the key itself: upstream
    keys list every injected fault and run to 215 characters, past the 128 a
    replay consumer accepts. The readable key travels in the metadata.
    """
    import hashlib

    tasks = {
        split: [TelecomSoloEnvironment(split).task(i) for i in range(5)]
        for split in corpus.SPLITS
    }
    for split, issued in tasks.items():
        expected = [task.key for task in corpus.rows(split)[:5]]
        assert [task["metadata"]["key"] for task in issued] == expected
        assert [task["id"] for task in issued] == [
            hashlib.sha256(key.encode("utf-8")).hexdigest() for key in expected
        ]
    ids = {task["id"] for issued in tasks.values() for task in issued}
    assert len(ids) == 15
    assert all(len(task_id) <= 128 for task_id in ids)


def test_every_task_id_fits_a_replay_consumer() -> None:
    """Over half of upstream's keys exceed 128 characters, and the failure is
    per task rather than at load — so the check is over the whole corpus, not a
    sample, or it would pass on the short tickets and miss the long ones."""
    for split in corpus.SPLITS:
        environment = TelecomSoloEnvironment(split)
        for index in range(len(environment)):
            assert 1 <= len(environment.task(index)["id"]) <= 128


def test_state_is_not_shared_between_episodes() -> None:
    """Every replay starts in the world the ticket found, not in the last one's.

    The starting databases are module-level and cached, so this is the test
    that says the cached copy is never the one that gets mutated.
    """
    environment = TelecomSoloEnvironment("train")
    wrecking = [
        {"tool": "toggle_airplane_mode", "arguments": {}},
        {"tool": "suspend_line", "arguments": {"customer_id": "C1001", "line_id": "L1001", "reason": "x"}},
        {"tool": "check_status_bar", "arguments": {}},
    ]
    first = environment.replay(0, wrecking)
    second = environment.replay(0, wrecking)
    assert first["events"] == second["events"]
    assert first["reward"]["state_digest"] == second["reward"]["state_digest"]

    # And a reference solution run afterwards still scores the point.
    assert (
        environment.replay(0, environment.reference_actions(0))["reward"]["reward"]
        == 1.0
    )
    assert first["reward"]["state_digest"] != environment.replay(0, [])["reward"][
        "state_digest"
    ]


def test_grading_is_a_product_of_its_components() -> None:
    """Binary, and one failed predicate is enough to take the whole reward."""
    environment = TelecomSoloEnvironment("train")
    index = next(
        i
        for i in range(len(environment))
        if "ACTION" in environment._task(i).reward_basis
    )
    task = environment._task(index)
    world = build_world(task)
    assertions_only = grade(task, world, [])
    assert assertions_only["reward_breakdown"]["env_assertions"] == 1.0
    assert assertions_only["reward_breakdown"]["actions"] == 0.0
    assert assertions_only["reward"] == 0.0

    with_transfer = grade(
        task, world, [{"tool": "transfer_to_human_agents", "arguments": {"summary": "x"}}]
    )
    assert with_transfer["reward"] == 1.0
    assert {check["passed"] for check in with_transfer["checks"]} == {True}


def test_transcript_digest_sees_the_observations() -> None:
    events = [{"role": "tool", "name": "a", "content": "1"}]
    assert transcript_digest(events) != transcript_digest(
        [{"role": "tool", "name": "a", "content": "2"}]
    )
    assert transcript_digest(events) == transcript_digest(list(events))


@needs_verifiers
def test_the_taskset_loads_and_validates() -> None:
    exported = [
        getattr(package, name)
        for name in package.__all__
        if isinstance(getattr(package, name), type)
        and issubclass(getattr(package, name), vf.Taskset)
    ]
    assert exported == [package.TelecomSoloTaskset]
    config_type = vf.taskset_config_type("reliquary-telecom-solo")
    taskset = vf.load_taskset(config_type(id="reliquary-telecom-solo"))
    tasks = [task for _, task in zip(range(4), taskset)]
    assert [task.key for task in tasks] == [
        row.key for row in corpus.rows("train")[:4]
    ]
    for task in tasks:
        assert task.data.split == "train"
        assert task.data.assertions >= 1
        assert asyncio.run(task.validate(None)) is True


@needs_verifiers
def test_both_surfaces_advertise_the_same_tools() -> None:
    """Same names, same parameters, same descriptions.

    They are two ways of reaching one implementation, so a difference here
    would mean the agent that trained and the agent that is verified were read
    two different manuals.
    """
    import inspect

    from verifiers.v1.utils.decorators import discover_decorated

    from reliquary_telecom_solo.taskset import TelecomSoloToolset

    toolset = TelecomSoloToolset(vf.ToolsetConfig())
    methods = {fn.__name__: fn for fn in discover_decorated(toolset, "tool")}
    frozen = {
        schema["function"]["name"]: schema["function"]
        for schema in corpus.tool_schemas()
    }
    assert set(methods) == set(frozen)
    for name, method in methods.items():
        assert (method.__doc__ or "").strip() == frozen[name]["description"], name
        assert set(inspect.signature(method).parameters) == set(
            frozen[name]["parameters"].get("properties", {})
        ), name


@needs_verifiers
def test_both_surfaces_agree_on_the_reward() -> None:
    """The training surface and the replay surface, on the same actions.

    They cannot disagree by construction — the toolset replays through the same
    environment — and the test is here because "by construction" is a claim
    about code that someone will edit.
    """
    config_type = vf.taskset_config_type("reliquary-telecom-solo")
    for split in corpus.SPLITS:
        taskset = vf.load_taskset(config_type(id="reliquary-telecom-solo", split=split))
        task = next(iter(taskset))
        environment = TelecomSoloEnvironment(split)
        for actions in (
            environment.reference_actions(0),
            environment.reference_actions(0)[:-2],
            [{"tool": "check_status_bar", "arguments": {}}],
        ):
            replayed = environment.replay(0, actions)["reward"]["reward"]
            scored = asyncio.run(task.ticket_resolved(_trace(task, actions)))
            assert scored == replayed, (split, actions)


@needs_verifiers
def test_the_toolset_reads_what_replay_reads() -> None:
    """The observations the training surface hands back are the replayed ones."""
    from reliquary_telecom_solo.taskset import TelecomSoloState, TelecomSoloToolset

    environment = TelecomSoloEnvironment("train")
    actions = environment.reference_actions(0)
    toolset = TelecomSoloToolset(vf.ToolsetConfig())
    toolset._inert_state = TelecomSoloState(index=0, split="train")
    observed = [
        toolset._run(action["tool"], **action["arguments"]) for action in actions
    ]
    assert observed == [
        event["content"] for event in environment.replay(0, actions)["events"]
    ]


@needs_verifiers
def test_the_toolset_ends_where_replay_ends() -> None:
    """A call accepted after `done` would be one the agent read and the reward
    never saw, so the toolset refuses it exactly where replay stops."""
    from reliquary_telecom_solo.taskset import TelecomSoloState, TelecomSoloToolset

    toolset = TelecomSoloToolset(vf.ToolsetConfig())
    toolset._inert_state = TelecomSoloState(index=0, split="train")
    assert toolset.check_status_bar().startswith("Status Bar:")
    assert toolset.done() == STOP_TOKEN
    assert toolset.check_status_bar() == "Error: this episode has already ended"
    assert [call["tool"] for call in toolset.state.calls] == [
        "check_status_bar",
        STOP_TOOL,
    ]


@needs_verifiers
def test_a_trace_that_talks_instead_of_calling_scores_nothing() -> None:
    """There is nobody to talk to, so prose is not an action this can price."""
    config_type = vf.taskset_config_type("reliquary-telecom-solo")
    task = next(iter(vf.load_taskset(config_type(id="reliquary-telecom-solo"))))
    trace = _trace(task, [{"tool": "check_status_bar", "arguments": {}}])
    trace.nodes.append(
        vf.MessageNode(
            parent=len(trace.nodes) - 1,
            message=vf.AssistantMessage(content="I think the SIM is fine."),
            sampled=True,
        )
    )
    assert asyncio.run(task.ticket_resolved(trace)) == 0.0


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_telecom_solo")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["environment"] == ENVIRONMENT
    assert artifact["entrypoints"]["taskset"] == (
        "reliquary_telecom_solo:TelecomSoloTaskset"
    )
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        == artifact["source_manifest_sha256"]
    )
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected
