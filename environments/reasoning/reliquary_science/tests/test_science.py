import asyncio
import hashlib
import importlib.resources
import json
import tomllib
from pathlib import Path

import pytest

import reliquary_science as package
from reliquary_science import corpus
from reliquary_science.environment import (
    ENVIRONMENT,
    ScienceEnvironment,
    _build_task,
    unit_instruction,
)
from reliquary_science.grading import (
    ANSWER_INSTRUCTION,
    GRADER_VERSION,
    SPEC_SCHEMA,
    grade,
    last_boxed_span,
    parse_number,
    split_number,
    verifier_spec,
)

try:
    import verifiers.v1 as vf
except ModuleNotFoundError:  # pragma: no cover - exercised by the packaged wheel
    vf = None

# Verifiers is the training-side surface. Replay, grading and the corpus do not
# import it, and these tests say so by still running when it is absent.
needs_verifiers = pytest.mark.skipif(vf is None, reason="verifiers is not installed")

SPLIT_SIZES = {"train": 10871, "eval": 1319, "qualification": 1357}


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
        for line in importlib.resources.files("reliquary_science")
        .joinpath("goldens/reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _manifest() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "environment.toml").read_text())


def test_the_corpus_matches_its_pin() -> None:
    """The digest covers the derived problems, not the fetched file.

    A corpus derived differently is a different environment: the same index
    would address a different problem, and every measurement taken against it
    would be about something else.
    """
    problems = corpus.load()
    assert hashlib.sha256(corpus.corpus_body(problems)).hexdigest() == (
        corpus.CORPUS_SHA256
    )
    assert len(problems) == corpus.VIRTUAL_LENGTH
    assert len({problem.key for problem in problems}) == len(problems)
    assert [problem.key for problem in problems] == sorted(
        problem.key for problem in problems
    )


def test_every_reference_is_one_number_read_the_policys_way() -> None:
    """The corpus keeps a problem only if the grader could grade its answer.

    The reference is read by the same parser that reads the policy's box, so a
    problem whose reference that parser cannot state as one number is not one
    this environment can pay for honestly — and it is not in the corpus.
    """
    for problem in corpus.load():
        assert isinstance(problem.answer, float)
        assert grade(verifier_spec(problem.answer), f"\\boxed{{{problem.answer!r}}}") == 1.0
        # The unit is what the prompt names, so it must read as one.
        assert problem.unit == " ".join(problem.unit.split())


def test_the_corpus_counts_what_the_build_dropped() -> None:
    """Half the source has no numeric reference, and 79 more point at a figure.

    Pinning the counts is what makes a rebuild that silently drops more of the
    corpus fail here rather than quietly become a smaller environment.
    """
    rows = corpus.read_source(corpus._fetch())
    derived, counts = corpus.derive(rows)
    assert counts == {
        "rows": corpus.UPSTREAM_ROWS,
        "numeric": corpus.NUMERIC_ANSWERS,
        "figures": corpus.FIGURE_PROBLEMS,
        "biology": corpus.BIOLOGY_PROBLEMS,
        "conflicting": 0,
        "kept": corpus.VIRTUAL_LENGTH,
    }
    assert derived == corpus.load()
    assert not any(problem.domain == "biology" for problem in derived)


def test_no_two_tasks_share_a_prompt() -> None:
    prompts = [
        ScienceEnvironment(split).task(index)["prompt"]
        for split, size in SPLIT_SIZES.items()
        for index in range(size)
    ]
    assert len(set(prompts)) == corpus.VIRTUAL_LENGTH
    assert len({" ".join(prompt.split()) for prompt in prompts}) == corpus.VIRTUAL_LENGTH


def test_the_prompt_carries_the_unit_and_the_answer_contract() -> None:
    """The unit the reference is in is stated, because it is not converted."""
    environment = ScienceEnvironment("eval")
    with_unit = 0
    for index in range(400):
        task = _build_task(index, "eval")
        problem = corpus.problems("eval")[index]
        assert task["prompt"].endswith(ANSWER_INSTRUCTION)
        assert task["prompt"].startswith(problem.problem)
        assert unit_instruction(problem.unit) in task["prompt"]
        with_unit += bool(problem.unit)
        assert environment.task(index)["prompt"] == task["prompt"]
    assert with_unit > 50
    assert unit_instruction("") == ""
    assert unit_instruction("%") == "\n\nGive the final answer as a percentage."
    assert unit_instruction("°") == "\n\nGive the final answer in degrees."
    assert unit_instruction("kJ/mol") == "\n\nGive the final answer in kJ/mol."


def test_splits_are_disjoint_and_alike() -> None:
    """Hashing the problem's key must not sort the domains into one split."""
    pools = {split: corpus.problems(split) for split in corpus.SPLITS}
    assert {split: len(pool) for split, pool in pools.items()} == SPLIT_SIZES
    keys = {split: {problem.key for problem in pool} for split, pool in pools.items()}
    assert sum(len(pool) for pool in keys.values()) == corpus.VIRTUAL_LENGTH
    assert not set.intersection(*keys.values())

    everything = corpus.load()
    for measure in (
        lambda problem: problem.domain == "physics",
        lambda problem: problem.domain == "chemistry",
        lambda problem: bool(problem.unit),
    ):
        overall = sum(measure(problem) for problem in everything) / len(everything)
        for split, pool in pools.items():
            share = sum(measure(problem) for problem in pool) / len(pool)
            assert abs(share - overall) < 0.03, split


def test_the_reference_goldens_hold() -> None:
    """One task per split, and four completions that freeze the grader."""
    goldens = _goldens()
    assert len(goldens) == len(corpus.SPLITS)
    assert {golden["split"] for golden in goldens} == set(corpus.SPLITS)
    for golden in goldens:
        environment = ScienceEnvironment(golden["split"])
        task = environment.task(golden["index"])
        assert task["id"] == golden["task_id"]
        assert task["metadata"]["corpus_problem"] == golden["corpus_problem"]
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        assert environment.reference_completion(golden["index"]) == golden["completion"]
        for field in ("completion", "narrated_completion"):
            good = environment.grade(golden["index"], golden[field])
            assert good["reward"] == 1.0, field
            assert good["state_digest"] == golden["state_digest"]
        for field in ("wrong_completion", "unboxed_completion"):
            bad = environment.grade(golden["index"], golden[field])
            assert bad["reward"] == 0.0, field
            assert bad["state_digest"] != golden["state_digest"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", (42.0, "")),
        ("-50.72", (-50.72, "")),
        ("−3.5", (-3.5, "")),
        ("1,234.5", (1234.5, "")),
        ("1{,}000", (1000.0, "")),
        ("1.2e3", (1200.0, "")),
        ("1.41 \\times 10^{19}", (1.41e19, "")),
        ("7 \\cdot 10^{-6}", (7e-06, "")),
        ("10^{-3}", (0.001, "")),
        ("\\frac{1}{2}", (0.5, "")),
        ("3/4", (0.75, "")),
        ("88.24\\%", (88.24, "%")),
        ("75.0^\\circ", (75.0, "°")),
        ("9.31 \\times 10^{-8} \\, \\text{N}", (9.31e-08, "N")),
        ("-3.2 \\times 10^{-4}\\,\\mathrm{s^{-1}}", (-0.00032, "s^{-1}")),
        ("96.7 kJ mol^{-1}", (96.7, "kJ mol^{-1}")),
        ("4\\,\\mu m", (4.0, "µm")),
        ("3.2 s⁻¹", (3.2, "s^{-1}")),
        ("1.5 × 10⁻⁴ M", (0.00015, "M")),
        ("12 molecules", (12.0, "molecules")),
        ("$2.88 \\times 10^{-6} \\text{ M}$", (2.88e-06, "M")),
        # One relation naming a variable is read past.
        ("v_2 = 5 m/s", (5.0, "m/s")),
        ("\\Delta H = -92.4 \\text{ kJ}", (-92.4, "kJ")),
        ("\\approx 0.45", (0.45, "")),
        # Anything that is not one number is refused, not read in part.
        ("2 or 3", None),
        ("x = 3, y = 4", None),
        ("105-115 min", None),
        ("2.4 \\text{ volts to } 2.6 \\text{ volts}", None),
        ("-13.6 eV, 13.6 eV", None),
        ("5 \\pm 0.1", None),
        ("5\\pi", None),
        ("3\\sqrt{2}", None),
        ("5^2", None),
        ("E = mc^2 = 9", None),
        ("5 \\sin \\theta - 3 \\cos \\theta = 3", None),
        ("Glucose and galactose", None),
        # What follows the number must be a unit, not a name or a symbol.
        ("2-butyne", None),
        ("24-hour urinary free cortisol", None),
        ("3 k_B T", None),
        ("4 kT", None),
        ("2 mv^2", None),
        ("1 Z^2 eV", None),
        ("5 Rahul", None),
        ("5^{2}", None),
        ("5²", None),
        ("\\frac{kq}{r^2}", None),
        ("\\frac{1}{0}", None),
        # An exponent past any magnitude is refused before it is computed.
        ("10^{999999999}", None),
        ("1e999999999", None),
        ("2 \\times 10^{-999999999}", None),
    ],
)
def test_a_box_states_one_number_or_none(text: str, expected) -> None:
    assert split_number(text) == expected


def test_a_reference_is_read_without_a_relation() -> None:
    """An equation as the reference is not a number to compare against."""
    assert split_number("v = 5", relation=False) is None
    assert split_number("5 m/s", relation=False) == (5.0, "m/s")


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "The answer is 109.",
        "\\boxed 109",
        # What a completion cut at the token budget looks like.
        "The final step gives \\boxed{109",
    ],
)
def test_an_answer_outside_the_box_scores_nothing(completion: str) -> None:
    assert grade(verifier_spec(109.0), completion) == 0.0


def test_the_comparison_tolerates_two_percent_and_no_more() -> None:
    """A measurement's rounding passes and a different number does not."""
    spec = verifier_spec(109.0)
    for right in ("108.7", "109.5 hp", "1.09 \\times 10^{2}", "111.1", "106.9"):
        assert grade(spec, f"\\boxed{{{right}}}") == 1.0, right
    for wrong in ("111.3", "106.7", "1090", "-109", "0.109"):
        assert grade(spec, f"\\boxed{{{wrong}}}") == 0.0, wrong
    zero = verifier_spec(0.0)
    assert grade(zero, "\\boxed{0}") == 1.0
    assert grade(zero, "\\boxed{0.001}") == 0.0


def test_the_last_box_is_the_answer() -> None:
    assert last_boxed_span("first \\boxed{7}, then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert parse_number(last_boxed_span("\\fbox{12}")) == 12.0


@pytest.mark.parametrize(
    "spec",
    [
        "not json at all",
        "[]",
        json.dumps({"schema": "something/else", "answer": 42}),
        json.dumps({"schema": SPEC_SCHEMA, "grader_version": "older", "answer": 42}),
        json.dumps({"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION}),
        json.dumps({"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": "42"}),
        json.dumps({"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": True}),
    ],
)
def test_the_spec_channel_fails_closed(spec: str) -> None:
    assert grade(spec, "\\boxed{42}") == 0.0


def test_the_compatibility_surface_is_deterministic() -> None:
    environment = ScienceEnvironment("eval")
    assert environment.task(7) == ScienceEnvironment("eval").task(7)
    assert environment.max_turns == 1
    assert environment.validator_authoritative_reward is True
    assert environment.name == ENVIRONMENT
    assert len(environment) == SPLIT_SIZES["eval"]
    assert environment.task(0)["id"] == environment.task(len(environment))["id"]
    for index in range(200):
        assert environment.grade(index, environment.reference_completion(index))["reward"] == 1.0
    assert environment.replay(3, environment.reference_completion(3))["reward"]["reward"] == 1.0
    with pytest.raises(ValueError):
        ScienceEnvironment("nope")


def test_the_declared_contract_matches_the_surface() -> None:
    artifact = json.loads(
        importlib.resources.files("reliquary_science").joinpath("artifact.json").read_text()
    )
    assert artifact["contract"] == "reliquary/boxed-answer/v1"
    assert artifact["environment"] == ENVIRONMENT
    surface = {"task", "grade", "replay", "reference_completion"}
    assert surface <= set(dir(ScienceEnvironment))
    assert not {"reset", "step"} & set(dir(ScienceEnvironment))


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_science")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == "reliquary_science:ScienceTaskset"
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
    assert manifest["data"]["numeric"] == corpus.NUMERIC_ANSWERS
    assert manifest["data"]["virtual_length"] == corpus.VIRTUAL_LENGTH
    assert manifest["data"]["source_revision"] == corpus.SOURCE_REVISION
    assert manifest["data"]["source_sha256"] == corpus.SOURCE_SHA256
    assert manifest["data"]["redistributable"] is False
    assert manifest["execution"]["max_turns"] == 1
    assert manifest["execution"]["network"] is False
    assert manifest["reward"]["components"] == ["numeric_answer"]
    assert manifest["entrypoint"] == "reliquary_science:ScienceTaskset"
    assert manifest["compatibility_entrypoint"] == "reliquary_science:ScienceEnvironment"


def test_the_manifest_declares_the_budget_and_the_run_honours_it() -> None:
    policy = _manifest()["policy"]
    assert policy["reasoning"] == "thinking"
    assert policy["max_new_tokens"] == 32768
    run = tomllib.loads(
        (Path(__file__).parents[1] / "examples/prime_rl/rl.toml").read_text()
    )
    for phase in ("train", "eval"):
        assert run["orchestrator"][phase]["sampling"]["max_completion_tokens"] == (
            policy["max_new_tokens"]
        )
        source = run["orchestrator"][phase]["source"][0]
        assert source["env"]["taskset"]["id"] == "reliquary-science"
    assert run["seq_len"] > policy["max_new_tokens"]
    assert run["orchestrator"]["renderer"]["enable_thinking"] is True


@needs_verifiers
def test_verifiers_load_and_wire_replay() -> None:
    config_type = vf.taskset_config_type("reliquary-science")
    taskset = vf.load_taskset(config_type(id="reliquary-science"))
    tasks = list(taskset.head(3))
    assert len(tasks) == 3
    environment = ScienceEnvironment()
    for position, task in enumerate(tasks):
        assert task.key == environment.task(position)["id"]
        assert asyncio.run(task.validate(None)) is True
        for completion, expected in (
            (environment.reference_completion(position), 1.0),
            ("no box anywhere in this answer", 0.0),
        ):
            trace = _trace(task, completion)
            assert asyncio.run(task.numeric_answer(trace)) == expected
            wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
            assert asyncio.run(task.numeric_answer(wire)) == expected
    assert isinstance(package.ScienceTaskset, type)


@needs_verifiers
def test_both_surfaces_agree_on_the_same_answer() -> None:
    """A reward earned in training must replay to the same number."""
    config_type = vf.taskset_config_type("reliquary-science")
    for golden in _goldens():
        taskset = vf.load_taskset(config_type(id="reliquary-science", split=golden["split"]))
        task = next(
            item for item in taskset.head(golden["index"] + 1) if item.data.idx == golden["index"]
        )
        environment = ScienceEnvironment(golden["split"])
        for field in ("completion", "narrated_completion", "wrong_completion", "unboxed_completion"):
            trace = _trace(task, golden[field])
            asyncio.run(task.score(trace))
            assert trace.reward == environment.grade(golden["index"], golden[field])["reward"]
            assert set(trace.rewards) == {"numeric_answer"}
