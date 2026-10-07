import asyncio
import gzip
import hashlib
import importlib.resources
import json
import time
import tomllib
from pathlib import Path

import pytest

import reliquary_hard_math as package
from reliquary_hard_math import corpus
from reliquary_hard_math.environment import ENVIRONMENT, HardMathEnvironment, _build_task
from reliquary_hard_math.grading import (
    ANSWER_INSTRUCTION,
    GRADER_VERSION,
    SPEC_SCHEMA,
    answer_region,
    equivalent,
    grade,
    last_boxed_span,
    parse_answer,
    reference_completion,
    verifier_spec,
    wrong_answer,
)

try:
    import verifiers.v1 as vf
except ModuleNotFoundError:  # pragma: no cover - exercised by the packaged wheel
    vf = None

# Verifiers is the training-side surface. Replay, grading and the corpus do not
# import it, and these tests say so by still running when it is absent.
needs_verifiers = pytest.mark.skipif(vf is None, reason="verifiers is not installed")

SPLIT_SIZES = {"train": 23227, "eval": 754, "qualification": 474}
# Train-split index ranges of each tier, which a job taking a prefix relies on.
TRAIN_TIERS = {"T1": (0, 1430), "T2": (1430, 15749), "T3": (15749, 23227)}


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
        for line in importlib.resources.files("reliquary_hard_math")
        .joinpath("goldens/reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def _manifest() -> dict:
    return tomllib.loads((Path(__file__).parents[1] / "environment.toml").read_text())


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

TRUE_POSITIVES = [
    # Numbers in every spelling.
    ("5", "5"), ("+5", "5"), ("05", "5"), ("5.0", "5"), ("-0", "0"),
    ("2.5", "5/2"), ("\\frac{5}{2}", "5/2"), ("\\dfrac52", "\\frac{5}{2}"),
    ("\\tfrac{5}{2}", "2.5"), ("1 / 2", "\\frac{1}{2}"), ("\\frac{-1}{2}", "-\\frac12"),
    ("1,000", "1000"), ("1{,}000", "1000"), ("2\\,449\\,999", "2449999"),
    ("2^{10}", "1024"), ("2^{2011}", "2^{2011}"),
    # Surds and pi, by value.
    ("\\frac{1}{\\sqrt2}", "\\frac{\\sqrt{2}}{2}"), ("\\sqrt{0.5}", "\\frac{\\sqrt2}{2}"),
    ("3\\sqrt{3}", "\\sqrt{27}"), ("\\sqrt[3]{27}", "3"), ("\\sqrt[3]{-8}", "-2"),
    ("\\sqrt{16}", "4"), ("2\\pi", "\\pi \\cdot 2"), ("\\frac{\\pi}{4}", "0.25\\pi"),
    ("24\\sqrt\\frac{3}{5}", "24\\sqrt{\\frac{3}{5}}"), ("√2", "\\sqrt{2}"), ("π", "\\pi"),
    # Expressions in letters.
    ("2+n", "n+2"), ("x^2-1", "(x-1)(x+1)"), ("\\frac{a}{c+b}", "\\frac{a}{b+c}"),
    ("(-1)^{n}", "(-1)^n"), ("n2^{n-1}", "n \\cdot 2^{n-1}"), ("F_{n + 1}", "F_{n+1}"),
    ("a_{x}", "a_x"), ("\\frac{1-\\sqrt{1-4a}}{2}", "\\dfrac{1-\\sqrt{1-4a}}{2}"),
    # Decoration around the value.
    ("x = 5", "5"), ("x = \\frac{1}{2}", "0.5"), ("5 \\text{ cm}", "5"), ("90^\\circ", "90"),
    ("90°", "90"), ("50\\%", "50"), ("\\left(\\frac{1}{2}\\right)", "\\frac12"), ("$7$", "7"),
    # Structures.
    ("(0, 1)", "(0,1)"), ("[-1, 2)", "[-1,2)"), ("(1,0),(0,1)", "(0,1), (1,0)"),
    ("\\{(1,0),(0,1)\\}", "(0,1), (1,0)"), ("1:2", "1 : 2"), ("x = (1, 0, 2)", "(1, 0, 2)"),
]

FALSE_POSITIVES = [
    # The one the brief names, and its relatives.
    ("5", "5/2"), ("5", "\\frac{5}{2}"), ("2", "\\frac52"), ("2\\frac12", "\\frac52"),
    # Roundings, at any length.
    ("0.333", "\\frac13"), ("0.3333333333333333333333333333", "\\frac13"),
    ("0.7071", "\\frac{\\sqrt{2}}{2}"), ("3.14159", "\\pi"), ("1.4142135623730951", "\\sqrt2"),
    ("10^{-50}", "0"),
    # Neighbours at magnitudes a float cannot tell apart.
    ("2^{2011}+1", "2^{2011}"), ("9" * 60, "9" * 59 + "8"),
    ("2^{2011}+\\sqrt2-\\sqrt2+1", "2^{2011}"),
    # Letters.
    ("n+3", "n+2"), ("k+2", "n+2"), ("2^n", "n^2"), ("x^2+1", "(x-1)(x+1)"),
    ("\\frac{a}{b}+c", "\\frac{a}{b+c}"), ("-1^n", "(-1)^n"), ("F_{n+2}", "F_{n+1}"),
    # Structure is not normalised away.
    ("[0,1]", "(0,1)"), ("(1,0)", "(0,1)"), ("(0,1)", "(0,1), (1,0)"),
    ("(1,0),(1,0)", "(0,1), (1,0)"), ("2:4", "1:2"), ("1:2", "\\frac12"), ("5,2", "(5,2)"),
    # Ambiguous spellings are refused, not read one way.
    ("2 3", "6"), ("23", "2 3"), ("2^10", "1024"), ("5\\text{ or }7", "5"),
    ("5 \\text{ or } 7", "35"), ("\\sin x", "\\sin x"), ("\\infty", "\\infty"),
    ("abc", "abc"), ("x \\le 2", "x \\le 2"), ("\\{1, 2\\}", "\\{1, 2\\}"), ("1, 2", "1, 2"),
    # Not an answer.
    ("", "5"), ("\\frac{1}{0}", "\\frac{1}{0}"), ("\\sqrt{-1}", "\\sqrt{-1}"),
]


@pytest.mark.parametrize(("candidate", "reference"), TRUE_POSITIVES)
def test_equivalent_spellings_are_accepted(candidate: str, reference: str) -> None:
    assert equivalent(candidate, reference)


@pytest.mark.parametrize(("candidate", "reference"), FALSE_POSITIVES)
def test_different_or_ambiguous_answers_are_refused(candidate: str, reference: str) -> None:
    assert not equivalent(candidate, reference)


@pytest.mark.parametrize(
    "candidate",
    [
        "(" * 190 + "1" + ")" * 190,
        "1+" * 190 + "1",
        "\\sqrt{" * 60 + "2" + "}" * 60,
        "(1+\\sqrt2)^{100000}",
        "\\pi^{100000}",
        "(2^{99999})^{99999}",
        "(a+b+c+d)^{99999}",
        "2^{2^{2^{65536}}}",
        "9^{9^{9^{9}}}",
        "10^{2900}\\pi",
    ],
)
def test_adversarial_answers_are_bounded(candidate: str) -> None:
    start = time.perf_counter()
    equivalent(candidate, "1")
    equivalent(candidate, candidate)
    assert time.perf_counter() - start < 2.0


def test_the_answer_is_read_after_the_reasoning() -> None:
    spec = verifier_spec("7")
    assert grade(spec, "<think>maybe \\boxed{7}</think>\n\\boxed{8}") == 0.0
    assert grade(spec, "<think>maybe \\boxed{8}</think>\n\\boxed{7}") == 1.0
    # A template that opens the block in the prompt: only the close is seen.
    assert grade(spec, "first \\boxed{8}\n</think>\n\nSo \\boxed{7}.") == 1.0
    assert grade(spec, "first \\boxed{7}\n</think>\n\nSo \\boxed{8}.") == 0.0
    # Opened and never closed: all of it is reasoning.
    assert grade(spec, "<think>so it is \\boxed{7}") == 0.0
    # No tag at all is read whole (direct mode, or a stripped trace).
    assert grade(spec, "so it is \\boxed{7}") == 1.0
    assert answer_region("a</think>b</think>c") == "c"


def test_the_last_box_counts_and_must_be_closed() -> None:
    spec = verifier_spec("\\frac{1}{2}")
    assert grade(spec, "\\boxed{3} then \\boxed{\\frac{1}{2}}") == 1.0
    assert grade(spec, "\\boxed{\\frac{1}{2}} then \\boxed{3}") == 0.0
    assert grade(spec, "\\fbox{0.5}") == 1.0
    assert grade(spec, "\\boxed{\\frac{1}{2}") == 0.0
    assert grade(spec, "the answer is 1/2") == 0.0
    assert last_boxed_span("x \\boxed{\\frac{1}{2}} y") == "\\frac{1}{2}"


def test_the_spec_is_checked_before_the_answer() -> None:
    completion = "\\boxed{5}"
    assert grade(verifier_spec("5"), completion) == 1.0
    assert grade("not json", completion) == 0.0
    assert grade(json.dumps([1]), completion) == 0.0
    for change in ({"schema": "other"}, {"grader_version": "other"}, {"answer": ""},
                   {"answer": 5}):
        spec = {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": "5"}
        spec.update(change)
        assert grade(json.dumps(spec), completion) == 0.0
    assert grade(verifier_spec("5"), None) == 0.0


def test_the_wrong_answer_has_the_reference_shape_and_is_wrong() -> None:
    for reference in ("5", "\\frac{\\sqrt{3}}{2}", "n+2", "(1, 0, 2)", "(0,1), (1,0)", "3:4"):
        wrong = wrong_answer(reference)
        assert parse_answer(wrong).shape == parse_answer(reference).shape
        assert grade(verifier_spec(reference), reference_completion(wrong)) == 0.0


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


def test_the_corpus_matches_its_pin() -> None:
    problems = corpus.load()
    assert len(problems) == corpus.VIRTUAL_LENGTH == 24455
    body = gzip.decompress(
        importlib.resources.files("reliquary_hard_math").joinpath(corpus.CORPUS_FILE).read_bytes()
    )
    assert hashlib.sha256(body).hexdigest() == corpus.CORPUS_SHA256


def test_the_corpus_counts_what_the_build_dropped() -> None:
    assert corpus.SOURCE_ROWS - corpus.UNGRADABLE_REFERENCES == corpus.VIRTUAL_LENGTH
    tiers = {tier: 0 for tier in corpus.TIER_COUNTS}
    for problem in corpus.load():
        tiers[problem.tier] += 1
    assert tiers == corpus.TIER_COUNTS


def test_every_reference_scores_itself() -> None:
    """The build's own rule, on the shipped corpus: 24,455 of 24,455."""
    failures = [
        problem.key
        for problem in corpus.load()
        if grade(verifier_spec(problem.answer), reference_completion(problem.answer)) != 1.0
    ]
    assert failures == []


def test_every_record_names_its_source() -> None:
    assert {problem.source for problem in corpus.load()} == {
        "nvidia/Nemotron-Math-v2@8e793210:aops"
    }


def test_no_two_tasks_share_a_prompt() -> None:
    seen: set[str] = set()
    for split in corpus.SPLITS:
        environment = HardMathEnvironment(split)
        for index in range(len(environment)):
            prompt = " ".join(environment.task(index)["prompt"].split())
            assert prompt not in seen
            seen.add(prompt)
    assert len(seen) == corpus.VIRTUAL_LENGTH


def test_the_prompt_carries_the_answer_contract() -> None:
    task = HardMathEnvironment().task(0)
    assert task["prompt"].endswith(ANSWER_INSTRUCTION)
    assert task["metadata"]["tier"] == "T1"
    assert "answer" not in json.dumps(task["metadata"])


def test_splits_are_disjoint_sized_and_ordered_hardest_first() -> None:
    keys = {}
    for split, size in SPLIT_SIZES.items():
        pool = corpus.problems(split)
        assert len(pool) == size
        keys[split] = {problem.key for problem in pool}
    assert not keys["train"] & keys["eval"]
    assert not keys["train"] & keys["qualification"]
    assert not keys["eval"] & keys["qualification"]
    train = corpus.problems("train")
    for tier, (start, end) in TRAIN_TIERS.items():
        assert {problem.tier for problem in train[start:end]} == {tier}
    for split in corpus.SPLITS:
        tiers = [problem.tier for problem in corpus.problems(split)]
        assert tiers == sorted(tiers)


def test_the_task_identity_follows_the_problem() -> None:
    environment = HardMathEnvironment("eval")
    assert environment.task(0)["id"] == environment.task(len(environment))["id"]
    assert len({environment.task(index)["id"] for index in range(len(environment))}) == len(
        environment
    )


# --------------------------------------------------------------------------
# The surfaces
# --------------------------------------------------------------------------


def test_the_reference_goldens_hold() -> None:
    goldens = _goldens()
    assert {golden["split"] for golden in goldens} == set(corpus.SPLITS)
    for golden in goldens:
        environment = HardMathEnvironment(golden["split"])
        task = _build_task(golden["index"], golden["split"])
        assert task["id"] == golden["task_id"]
        assert task["private"]["reference_answer"] == golden["answer"]
        assert hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest() == golden["prompt_sha256"]
        for field, expected in (
            ("completion", 1.0),
            ("narrated_completion", 1.0),
            ("wrong_completion", 0.0),
            ("unboxed_completion", 0.0),
        ):
            result = environment.grade(golden["index"], golden[field])
            assert result["reward"] == expected, field
            assert result["success"] is (expected == 1.0)
        assert environment.grade(golden["index"], golden["completion"])["state_digest"] == (
            golden["state_digest"]
        )


def test_the_replay_surface() -> None:
    environment = HardMathEnvironment("eval")
    assert environment.name == ENVIRONMENT
    for index in range(100):
        assert environment.grade(index, environment.reference_completion(index))["reward"] == 1.0
    assert environment.replay(3, environment.reference_completion(3))["reward"]["reward"] == 1.0
    with pytest.raises(ValueError):
        HardMathEnvironment("nope")


def test_the_declared_contract_matches_the_surface() -> None:
    artifact = json.loads(
        importlib.resources.files("reliquary_hard_math").joinpath("artifact.json").read_text()
    )
    assert artifact["contract"] == "reliquary/boxed-answer/v1"
    assert artifact["environment"] == ENVIRONMENT
    surface = {"task", "grade", "replay", "reference_completion"}
    assert surface <= set(dir(HardMathEnvironment))
    assert not {"reset", "step"} & set(dir(HardMathEnvironment))


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_hard_math")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"] == {
        "taskset": "reliquary_hard_math:HardMathTaskset",
        "replay": "reliquary_hard_math:HardMathEnvironment",
    }
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == (
        artifact["source_manifest_sha256"]
    )
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected
    shipped = {
        path.relative_to(Path(str(root))).as_posix()
        for path in Path(str(package_root)).rglob("*")
        if path.is_file() and path.name != "artifact.json" and "__pycache__" not in path.parts
    }
    assert shipped == set(artifact["files"])


def test_the_manifest_declares_what_the_corpus_holds() -> None:
    manifest = _manifest()
    assert manifest["data"]["sha256"] == corpus.CORPUS_SHA256
    assert manifest["data"]["rows"] == corpus.SOURCE_ROWS
    assert manifest["data"]["ungradable"] == corpus.UNGRADABLE_REFERENCES
    assert manifest["data"]["virtual_length"] == corpus.VIRTUAL_LENGTH
    assert manifest["data"]["redistributable"] is True
    assert manifest["data"]["license"] == "cc-by-4.0"
    assert manifest["execution"]["max_turns"] == 1
    assert manifest["execution"]["network"] is False
    assert manifest["reward"]["components"] == ["equivalent_answer"]
    assert manifest["entrypoint"] == "reliquary_hard_math:HardMathTaskset"
    assert manifest["compatibility_entrypoint"] == "reliquary_hard_math:HardMathEnvironment"


def test_the_manifest_declares_the_budget_and_the_run_honours_it() -> None:
    policy = _manifest()["policy"]
    assert policy["reasoning"] == "thinking"
    assert policy["max_new_tokens"] == 32768
    run = tomllib.loads((Path(__file__).parents[1] / "examples/prime_rl/rl.toml").read_text())
    for phase in ("train", "eval"):
        assert run["orchestrator"][phase]["sampling"]["max_completion_tokens"] == (
            policy["max_new_tokens"]
        )
    assert run["orchestrator"]["renderer"]["enable_thinking"] is True


def test_grading_does_not_import_verifiers() -> None:
    import subprocess
    import sys

    code = (
        "import sys, reliquary_hard_math, reliquary_hard_math.grading;"
        "from reliquary_hard_math import HardMathEnvironment;"
        "HardMathEnvironment().grade(0, '\\\\boxed{1}');"
        "assert 'verifiers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@needs_verifiers
def test_verifiers_load_and_wire_replay() -> None:
    config_type = vf.taskset_config_type("reliquary-hard-math")
    taskset = vf.load_taskset(config_type(id="reliquary-hard-math"))
    tasks = list(taskset.head(3))
    assert len(tasks) == 3
    environment = HardMathEnvironment()
    for position, task in enumerate(tasks):
        assert task.key == environment.task(position)["id"]
        assert asyncio.run(task.validate(None)) is True
        for completion, expected in (
            (environment.reference_completion(position), 1.0),
            ("no box anywhere in this answer", 0.0),
        ):
            trace = _trace(task, completion)
            assert asyncio.run(task.equivalent_answer(trace)) == expected
            wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
            assert asyncio.run(task.equivalent_answer(wire)) == expected
    assert isinstance(package.HardMathTaskset, type)


@needs_verifiers
def test_both_surfaces_agree_on_the_same_answer() -> None:
    """A reward earned in training must replay to the same number."""
    config_type = vf.taskset_config_type("reliquary-hard-math")
    for golden in _goldens():
        taskset = vf.load_taskset(config_type(id="reliquary-hard-math", split=golden["split"]))
        task = next(
            item for item in taskset.head(golden["index"] + 1) if item.data.idx == golden["index"]
        )
        environment = HardMathEnvironment(golden["split"])
        for field in ("completion", "narrated_completion", "wrong_completion", "unboxed_completion"):
            trace = _trace(task, golden[field])
            asyncio.run(task.score(trace))
            assert trace.reward == environment.grade(golden["index"], golden[field])["reward"]
            assert set(trace.rewards) == {"equivalent_answer"}
