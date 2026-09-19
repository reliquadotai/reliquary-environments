import asyncio
import gzip
import hashlib
import importlib.resources
import json
import tomllib
from pathlib import Path

import pytest

import reliquary_dapo_math as package
from reliquary_dapo_math import corpus
from reliquary_dapo_math.environment import (
    ENVIRONMENT,
    DapoMathEnvironment,
    _build_task,
)
from reliquary_dapo_math.grading import (
    ANSWER_INSTRUCTION,
    GRADER_VERSION,
    SPEC_SCHEMA,
    grade,
    last_boxed_span,
    parse_integer,
    verifier_spec,
)

try:
    import verifiers.v1 as vf
except ModuleNotFoundError:  # pragma: no cover - exercised by the packaged wheel
    vf = None

# Verifiers is the training-side surface. Replay, grading and the corpus do not
# import it, and these tests say so by still running when it is absent.
needs_verifiers = pytest.mark.skipif(vf is None, reason="verifiers is not installed")

SPLIT_SIZES = {"train": 13931, "eval": 1645, "qualification": 1595}
# DAPO's own answer contract, which the build strips off both ends.
UPSTREAM_PREAMBLE = "Solve the following math problem step by step."
UPSTREAM_POSTAMBLE = "Remember to put your answer"


def _trace(task, completion: str):
    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            vf.MessageNode(
                parent=None,
                message=vf.AssistantMessage(content=completion),
                sampled=True,
            )
        ],
        state=vf.State(),
    )


def _goldens() -> list[dict]:
    return [
        json.loads(line)
        for line in importlib.resources.files("reliquary_dapo_math")
        .joinpath("goldens/reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _manifest() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "environment.toml").read_text())


def test_the_corpus_matches_its_pin() -> None:
    """The digest covers the problems, not just the file.

    A corpus that arrived differently is a different environment: the same
    index would address a different problem, and every measurement taken
    against it would be about something else.
    """
    body = importlib.resources.files("reliquary_dapo_math").joinpath(
        corpus.CORPUS_FILE
    )
    assert (
        hashlib.sha256(gzip.decompress(body.read_bytes())).hexdigest()
        == corpus.CORPUS_SHA256
    )
    problems = corpus.load()
    assert len(problems) == corpus.VIRTUAL_LENGTH
    assert len({problem.key for problem in problems}) == len(problems)
    for problem in problems:
        assert isinstance(problem.answer, int)
        assert abs(problem.answer) <= corpus.MAX_ANSWER_MAGNITUDE


def test_no_two_tasks_share_a_prompt() -> None:
    """The reason this environment ships 17,171 rows and not 1,791,700.

    Upstream repeats each problem a hundred times or more, and an environment
    indexing those rows would serve one problem under a hundred indices — paid
    for a hundred times where the cooldown keys on the index, and killing
    ninety-nine indices in a hundred where it keys on the content. Whitespace
    is included the other way round: two prompts that differ only in layout are
    the same problem, and neither may appear twice.
    """
    prompts = [problem.problem + ANSWER_INSTRUCTION for problem in corpus.load()]
    assert len(set(prompts)) == corpus.VIRTUAL_LENGTH
    assert len({" ".join(prompt.split()) for prompt in prompts}) == (
        corpus.VIRTUAL_LENGTH
    )
    identities = {
        DapoMathEnvironment(split).task(index)["id"]
        for split, size in SPLIT_SIZES.items()
        for index in range(size)
    }
    assert len(identities) == corpus.VIRTUAL_LENGTH


def test_the_corpus_counts_what_the_build_dropped() -> None:
    """Every problem upstream holds, minus the ones that cannot be graded.

    Twelve problems whose copies disagree about the answer and five whose
    answer is beyond what a double carries exactly. Pinning the counts is what
    makes a rebuild that silently drops more of the corpus fail here rather
    than quietly become a smaller environment.
    """
    assert corpus.VIRTUAL_LENGTH == (
        corpus.DISTINCT_PROBLEMS
        - corpus.AMBIGUOUS_PROBLEMS
        - corpus.OVERSIZED_PROBLEMS
    )
    assert corpus.UPSTREAM_ROWS // corpus.DISTINCT_PROBLEMS == 104


def test_the_prompt_carries_this_repositorys_answer_contract() -> None:
    """DAPO's own contract is stripped, and ours is appended in its place.

    Keeping the `Answer:` line would measure two changes at once: a corpus the
    policy has not seen and a convention it was not trained on.
    """
    environment = DapoMathEnvironment("eval")
    for index in range(200):
        prompt = environment.task(index)["prompt"]
        assert prompt.endswith(ANSWER_INSTRUCTION)
        assert UPSTREAM_PREAMBLE not in prompt
        assert UPSTREAM_POSTAMBLE not in prompt
        assert prompt.strip()


def test_splits_are_disjoint_and_alike() -> None:
    """Hashing the problem's key must not sort difficulty into one split.

    The share of each split is a property of the hash and the mix inside it a
    property of the corpus, so both are measured rather than assumed. Answer
    magnitude and the share of non-English statements stand in for difficulty:
    neither is what the hash sees.
    """
    pools = {split: corpus.problems(split) for split in corpus.SPLITS}
    assert {split: len(pool) for split, pool in pools.items()} == SPLIT_SIZES
    keys = {split: {problem.key for problem in pool} for split, pool in pools.items()}
    assert sum(len(pool) for pool in keys.values()) == corpus.VIRTUAL_LENGTH
    assert not set.intersection(*keys.values())

    identities = {
        split: {DapoMathEnvironment(split).task(index)["id"] for index in range(400)}
        for split in corpus.SPLITS
    }
    assert not set.intersection(*identities.values())

    everything = corpus.load()
    for measure in (
        lambda problem: abs(problem.answer) < 100,
        lambda problem: not problem.problem.isascii(),
    ):
        overall = sum(measure(problem) for problem in everything) / len(everything)
        for split, pool in pools.items():
            share = sum(measure(problem) for problem in pool) / len(pool)
            assert abs(share - overall) < 0.03, split


def test_the_reference_goldens_hold() -> None:
    """One task per split, and four completions that freeze the grader.

    The accepting pair is not enough on its own: a grader that read any boxed
    number as correct would pass it. The wrong answer and the unboxed one are
    what make the accepting half mean something.
    """
    goldens = _goldens()
    assert len(goldens) == len(corpus.SPLITS)
    assert {golden["split"] for golden in goldens} == set(corpus.SPLITS)
    for golden in goldens:
        environment = DapoMathEnvironment(golden["split"])
        task = environment.task(golden["index"])
        assert task["id"] == golden["task_id"]
        assert task["metadata"]["corpus_problem"] == golden["corpus_problem"]
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        assert environment.reference_completion(golden["index"]) == (
            golden["completion"]
        )
        for field in ("completion", "narrated_completion"):
            good = environment.grade(golden["index"], golden[field])
            assert good["reward"] == 1.0
            assert good["success"] is True
            assert good["state_digest"] == golden["state_digest"]
        for field in ("wrong_completion", "unboxed_completion"):
            bad = environment.grade(golden["index"], golden[field])
            assert bad["reward"] == 0.0
            assert bad["state_digest"] != golden["state_digest"]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("\\boxed{42}", 42),
        ("\\fbox{42}", 42),
        ("\\boxed{ 42 }", 42),
        ("\\boxed{+42}", 42),
        ("\\boxed{-42}", -42),
        ("\\boxed{$42$}", 42),
        # Thousands separators, in both the plain and the LaTeX spelling: a
        # separator is typography, and the value either side of it is the same.
        ("\\boxed{1,234}", 1234),
        ("\\boxed{1{,}234}", 1234),
        ("\\boxed{12\\,345}", 12345),
        ("first \\boxed{7}, then \\boxed{9}", 9),
        # Nothing else is normalised, on purpose: every further rule is a rule
        # that can make two different values equal.
        ("\\boxed{42.0}", None),
        ("\\boxed{\\text{42}}", None),
        ("\\boxed{\\frac{1}{2}}", None),
        ("\\boxed{forty-two}", None),
        ("\\boxed{1,23}", None),
        ("\\boxed{42} and \\boxed{x}", None),
    ],
)
def test_the_box_is_read_the_way_the_contract_states(
    completion: str, expected: int | None
) -> None:
    span = last_boxed_span(completion)
    assert span is not None
    assert parse_integer(span) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "   ",
        "The answer is 42.",
        "\\boxed 42",
        # What a completion cut at the token budget looks like: the box was
        # opened and the rollout ended before it closed.
        "The final step gives \\boxed{42",
    ],
)
def test_an_answer_outside_the_box_scores_nothing(completion: str) -> None:
    """No box, no answer.

    A number picked out of the prose would reward whichever number the
    reasoning happened to end on, which on a wrong derivation is a wrong
    answer that happens to be the last thing written.
    """
    assert last_boxed_span(completion) is None
    assert grade(verifier_spec(42), completion) == 0.0


def test_the_comparison_is_exact() -> None:
    """Both sides are integers by construction, so nothing is near enough."""
    spec = verifier_spec(1729)
    assert grade(spec, "\\boxed{1729}") == 1.0
    assert grade(spec, "\\boxed{1,729}") == 1.0
    for wrong in ("\\boxed{1728}", "\\boxed{1730}", "\\boxed{17290}", "\\boxed{-1729}"):
        assert grade(spec, wrong) == 0.0


@pytest.mark.parametrize(
    "spec",
    [
        "not json at all",
        "[]",
        json.dumps({"schema": "something/else", "answer": 42}),
        json.dumps(
            {"schema": SPEC_SCHEMA, "grader_version": "older", "answer": 42}
        ),
        json.dumps({"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION}),
        json.dumps(
            {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": "42"}
        ),
        json.dumps(
            {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": True}
        ),
    ],
)
def test_the_spec_channel_fails_closed(spec: str) -> None:
    """A spec that is not this grader's grades nothing, rather than everything.

    The case worth stating is the one stripped of its answer: there is then
    nothing to be wrong about, and paying for that would pay for removing the
    task.
    """
    assert grade(spec, "\\boxed{42}") == 0.0


def test_grading_the_same_answer_twice_agrees() -> None:
    """Nothing in the grader consults anything outside the completion."""
    environment = DapoMathEnvironment("eval")
    for index in range(200):
        spec = _build_task(index, "eval")["private"]["verifier_spec"]
        assert spec == _build_task(index, "eval")["private"]["verifier_spec"]
        reference = environment.reference_completion(index)
        assert grade(spec, reference) == 1.0
        assert environment.grade(index, reference) == DapoMathEnvironment("eval").grade(
            index, reference
        )


def test_the_compatibility_surface_is_deterministic() -> None:
    environment = DapoMathEnvironment("eval")
    assert environment.task(7) == DapoMathEnvironment("eval").task(7)
    assert environment.max_turns == 1
    assert environment.validator_authoritative_reward is True
    assert environment.name == ENVIRONMENT
    assert len(environment) == SPLIT_SIZES["eval"]
    # Indices wrap: a taskset may be asked for more tasks than the split holds.
    assert environment.task(0)["id"] == environment.task(len(environment))["id"]
    replay = environment.replay(3, environment.reference_completion(3))
    assert replay["reward"]["reward"] == 1.0
    with pytest.raises(ValueError):
        DapoMathEnvironment("nope")


def test_the_declared_contract_matches_the_surface() -> None:
    """`boxed-answer`, because the answer channel is a box rather than JSON.

    The logic sibling's `answer-json` promises a fenced JSON object and the
    instruction-following sibling's `checked-answer` promises no reference
    completion at all. This environment has a reference completion and reads a
    `\\boxed{}` span, so it claims neither name.
    """
    artifact = json.loads(
        importlib.resources.files("reliquary_dapo_math")
        .joinpath("artifact.json")
        .read_text()
    )
    assert artifact["contract"] == "reliquary/boxed-answer/v1"
    assert artifact["environment"] == ENVIRONMENT
    surface = {"task", "grade", "replay", "reference_completion"}
    assert surface <= set(dir(DapoMathEnvironment))
    assert not {"reset", "step"} & set(dir(DapoMathEnvironment))


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_dapo_math")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == (
        "reliquary_dapo_math:DapoMathTaskset"
    )
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        == artifact["source_manifest_sha256"]
    )
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected


def test_the_manifest_declares_what_the_corpus_holds() -> None:
    manifest = _manifest()
    assert manifest["data"]["sha256"] == corpus.CORPUS_SHA256
    assert manifest["data"]["rows"] == corpus.UPSTREAM_ROWS
    assert manifest["data"]["problems"] == corpus.DISTINCT_PROBLEMS
    assert manifest["data"]["virtual_length"] == corpus.VIRTUAL_LENGTH
    assert manifest["data"]["license"] == "apache-2.0"
    assert manifest["data"]["redistributable"] is True
    assert manifest["execution"]["max_turns"] == 1
    assert manifest["execution"]["network"] is False
    assert manifest["reward"]["components"] == ["exact_answer"]
    assert manifest["task_families"] == ["dapo_math"]
    assert manifest["entrypoint"] == "reliquary_dapo_math:DapoMathTaskset"
    assert manifest["compatibility_entrypoint"] == (
        "reliquary_dapo_math:DapoMathEnvironment"
    )


def test_the_manifest_declares_the_budget_the_task_needs() -> None:
    """The measurement that justifies it, and the run that has to honour it.

    An environment measured under a budget it cannot fit is measured wrong, so
    the number lives beside the task rather than in whichever config a run
    happened to use — and the shipped run has to agree with it, in both phases.
    """
    policy = _manifest()["policy"]
    assert policy["reasoning"] == "thinking"
    assert policy["reasoning_rationale"].strip()
    assert policy["max_new_tokens"] == 32768
    assert policy["max_new_tokens_rationale"].strip()

    run = tomllib.loads(
        (Path(__file__).parents[1] / "examples/prime_rl/rl.toml").read_text()
    )
    for phase in ("train", "eval"):
        assert run["orchestrator"][phase]["sampling"]["max_completion_tokens"] == (
            policy["max_new_tokens"]
        )
    assert run["seq_len"] > policy["max_new_tokens"]
    assert run["inference"]["vllm"]["max_model_len"] > policy["max_new_tokens"]
    assert run["orchestrator"]["renderer"]["enable_thinking"] is True


def test_prime_rl_reference_config_and_pins() -> None:
    """The training lane must name this taskset and the repository's pins."""
    environment_root = Path(__file__).parents[1]
    config = tomllib.loads((environment_root / "examples/prime_rl/rl.toml").read_text())
    compatibility = tomllib.loads(
        (environment_root.parents[2] / "compatibility.toml").read_text()
    )

    assert compatibility["prime_rl"]["version"] == "0.9.0"
    project = tomllib.loads((environment_root / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == ["verifiers==0.3.1"]
    assert (
        project["tool"]["uv"]["sources"]["verifiers"]["rev"]
        == compatibility["verifiers"]["source_commit"]
    )
    assert config["model"]["name"] == (
        compatibility["models"]["qwen3_4b_instruct_2507"]["id"]
    )
    assert config["orchestrator"]["algo"]["type"] == "grpo"
    assert config["orchestrator"]["renderer"]["name"]
    for phase, split in (("train", "train"), ("eval", "eval")):
        source = config["orchestrator"][phase]["source"][0]
        assert source["env"]["taskset"] == {
            "id": "reliquary-dapo-math",
            "split": split,
        }
        assert source["env"]["agent"]["max_turns"] == 1
        assert source["env"]["agent"]["runtime"] == {"type": "docker", "allow": []}
    # Single turn with no tools: a tool-call parser here would be cargo.
    assert "tool_call_parser" not in config["inference"]["vllm"]


@needs_verifiers
def test_verifiers_load_and_wire_replay() -> None:
    exported = [
        item
        for name in package.__all__
        if isinstance((item := getattr(package, name)), type)
        and issubclass(item, vf.Taskset)
    ]
    assert exported == [package.DapoMathTaskset]
    config_type = vf.taskset_config_type("reliquary-dapo-math")
    taskset = vf.load_taskset(config_type(id="reliquary-dapo-math"))
    tasks = list(taskset.head(3))
    assert len(tasks) == 3
    environment = DapoMathEnvironment()
    for position, task in enumerate(tasks):
        assert task.key == environment.task(position)["id"]
        assert asyncio.run(task.validate(None)) is True
        for completion, expected in (
            (environment.reference_completion(position), 1.0),
            ("no box anywhere in this answer", 0.0),
            ("", 0.0),
        ):
            trace = _trace(task, completion)
            assert asyncio.run(task.exact_answer(trace)) == expected
            wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
            assert asyncio.run(task.exact_answer(wire)) == expected


@needs_verifiers
def test_both_surfaces_agree_on_the_same_answer() -> None:
    """A reward earned in training must replay to the same number."""
    config_type = vf.taskset_config_type("reliquary-dapo-math")
    for golden in _goldens():
        taskset = vf.load_taskset(
            config_type(id="reliquary-dapo-math", split=golden["split"])
        )
        task = next(
            item
            for item in taskset.head(golden["index"] + 1)
            if item.data.idx == golden["index"]
        )
        environment = DapoMathEnvironment(golden["split"])
        assert task.key == environment.task(golden["index"])["id"]
        assert task.data.split == golden["split"]
        assert task.data.corpus_problem == golden["corpus_problem"]
        for field in (
            "completion",
            "narrated_completion",
            "wrong_completion",
            "unboxed_completion",
        ):
            trace = _trace(task, golden[field])
            asyncio.run(task.score(trace))
            assert trace.reward == (
                environment.grade(golden["index"], golden[field])["reward"]
            )
            assert set(trace.rewards) == {"exact_answer"}
            assert trace.rewards["exact_answer"].weight == 1.0


@needs_verifiers
def test_a_supplied_answer_is_not_a_sampled_one() -> None:
    config_type = vf.taskset_config_type("reliquary-dapo-math")
    taskset = vf.load_taskset(config_type(id="reliquary-dapo-math", split="qualification"))
    task = next(iter(taskset))
    environment = DapoMathEnvironment("qualification")
    trace = _trace(task, environment.reference_completion(0))
    trace.nodes[0].sampled = False
    asyncio.run(task.score(trace))
    assert trace.reward == 0.0
