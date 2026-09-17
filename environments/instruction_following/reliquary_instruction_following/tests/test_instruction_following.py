import asyncio
import collections
import gzip
import hashlib
import importlib.resources
import json
import sys
import tomllib
from pathlib import Path

import pytest

import reliquary_instruction_following as package
from reliquary_instruction_following import corpus
from reliquary_instruction_following._ifeval import instructions_registry
from reliquary_instruction_following._ifeval.instructions_util import (
    count_words,
    word_tokenize,
)
from reliquary_instruction_following.environment import (
    ENVIRONMENT,
    InstructionFollowingEnvironment,
    _build_task,
)
from reliquary_instruction_following.grading import (
    SPEC_SCHEMA,
    builds,
    follows,
    grade,
)

try:
    import verifiers.v1 as vf
except ModuleNotFoundError:  # pragma: no cover - exercised by the packaged wheel
    vf = None

# Verifiers is the training-side surface. Replay, grading and the corpus do not
# import it, and these tests say so by still running when it is absent.
needs_verifiers = pytest.mark.skipif(vf is None, reason="verifiers is not installed")

FAMILIES = 14
INSTRUCTION_IDS = 44
# The shipped file, and what survives `corpus._is_stable`.
CONSTRAINT_COUNTS = {1: 2626, 2: 18737, 3: 15254}


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
        for line in importlib.resources.files("reliquary_instruction_following")
        .joinpath("goldens/reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def test_the_corpus_matches_its_pin() -> None:
    """The digest covers the rows, not just the file.

    A corpus that arrived differently is a different environment: the same
    index would address a different prompt, and every measurement taken
    against it would be about something else.
    """
    body = importlib.resources.files("reliquary_instruction_following").joinpath(
        corpus.CORPUS_FILE
    )
    assert (
        hashlib.sha256(gzip.decompress(body.read_bytes())).hexdigest()
        == corpus.CORPUS_SHA256
    )
    rows = corpus.load()
    assert len(rows) == corpus.CORPUS_ROWS - corpus.UNSTABLE_ROWS
    assert len({row.key for row in rows}) == len(rows)
    counts = collections.Counter(len(row.constraints) for row in rows)
    assert dict(counts) == CONSTRAINT_COUNTS
    assert len({c.instruction_id for row in rows for c in row.constraints}) == (
        INSTRUCTION_IDS
    )


def test_the_corpus_is_disjoint_from_the_excluded_verifiers() -> None:
    """Nothing gradable here consults `langdetect` or the Punkt download."""
    named = {
        constraint.instruction_id
        for row in corpus.load()
        for constraint in row.constraints
    }
    assert named & corpus.EXCLUDED_INSTRUCTION_IDS == set()
    assert corpus.EXCLUDED_INSTRUCTION_IDS == {
        "change_case:english_capital",
        "change_case:english_lowercase",
        "language:response_language",
        "length_constraints:number_sentences",
    }


def test_an_excluded_verifier_stops_the_load() -> None:
    """The filter is enforced, not inherited from how the file was built."""
    row = json.dumps(
        {
            "id": 1,
            "prompt": "write something",
            "instruction_id_list": ["language:response_language"],
            "kwargs": [{"language": "en"}],
        }
    )
    with pytest.raises(ValueError, match="excluded"):
        corpus._parse(row.encode("utf-8"))


def test_every_instruction_id_resolves_in_the_registry() -> None:
    named = {
        constraint.instruction_id
        for row in corpus.load()
        for constraint in row.constraints
    }
    assert named <= set(instructions_registry.INSTRUCTION_DICT)


def test_splits_are_disjoint_and_unbiased() -> None:
    """Hashing the row key must not sort constraint families into splits.

    The share of each split is a property of the hash, and the mix inside it a
    property of the corpus, so both are measured rather than assumed.
    """
    rows = {split: corpus.rows(split) for split in corpus.SPLITS}
    keys = {split: {row.key for row in pool} for split, pool in rows.items()}
    assert sum(len(pool) for pool in keys.values()) == len(corpus.load())
    assert not set.intersection(*keys.values())

    identities = {
        split: {
            InstructionFollowingEnvironment(split).task(index)["id"]
            for index in range(400)
        }
        for split in corpus.SPLITS
    }
    assert not set.intersection(*identities.values())

    overall = collections.Counter(
        constraint.instruction_id
        for row in corpus.load()
        for constraint in row.constraints
    )
    total = sum(overall.values())
    for split, pool in rows.items():
        counts = collections.Counter(
            constraint.instruction_id for row in pool for constraint in row.constraints
        )
        assert len(counts) == INSTRUCTION_IDS, split
        share = sum(counts.values())
        for instruction_id, count in overall.items():
            assert abs(counts[instruction_id] / share - count / total) < 0.01, (
                split,
                instruction_id,
            )


def test_the_reference_goldens_hold() -> None:
    """Hand-written answers, because no generator can write one.

    Each golden carries an answer that satisfies its constraints and one that
    breaks a single constraint. The pair is what freezes the checker: an
    accepting half alone would pass on a grader that accepted anything.
    """
    rows = _goldens()
    assert len(rows) == len(corpus.SPLITS)
    for golden in rows:
        environment = InstructionFollowingEnvironment(golden["split"])
        task = environment.task(golden["index"])
        assert task["id"] == golden["task_id"]
        assert task["metadata"]["instruction_ids"] == golden["instruction_ids"]
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        good = environment.grade(golden["index"], golden["answer"])
        assert good["reward"] == 1.0
        assert good["success"] is True
        assert good["state_digest"] == golden["state_digest"]
        bad = environment.grade(golden["index"], golden["broken_answer"])
        assert bad["reward"] == 0.0
        assert bad["state_digest"] != golden["state_digest"]


@pytest.mark.parametrize("answer", ["", "   ", "\n\n", "\t\n "])
def test_an_empty_answer_scores_zero(answer: str) -> None:
    """The guard, and the reason it exists.

    Several checkers are satisfied vacuously by an empty string — "at most 3
    lowercase words" is true of one — so the task below is followed, constraint
    by constraint, by an answer that says nothing at all.
    """
    environment = InstructionFollowingEnvironment()
    vacuous = None
    for index in range(len(environment)):
        spec = json.loads(_build_task(index, "train")["private"]["verifier_spec"])
        if all(
            follows(c["id"], c["kwargs"], spec["prompt"], "") for c in spec["constraints"]
        ):
            vacuous = index
            break
    assert vacuous is not None
    assert environment.grade(vacuous, answer)["reward"] == 0.0
    assert environment.grade(vacuous, answer)["success"] is False


@pytest.mark.parametrize(
    "spec",
    [
        "not json at all",
        "[]",
        json.dumps({"schema": "something/else", "constraints": []}),
        json.dumps({"schema": SPEC_SCHEMA, "checker_version": "older", "constraints": []}),
        json.dumps({"schema": SPEC_SCHEMA, "checker_version": "ifevalg-checker-v1"}),
    ],
)
def test_the_spec_channel_fails_closed(spec: str) -> None:
    """A spec that is not this checker's grades nothing, rather than everything.

    The last case is the one worth stating: a spec stripped of its constraints
    has nothing to fail, and paying for that would pay for removing the task.
    """
    assert grade(spec, "an answer that would satisfy nothing in particular") == 0.0


def test_a_verifier_that_does_not_resolve_fails_validation() -> None:
    assert not builds(
        json.dumps(
            {
                "schema": SPEC_SCHEMA,
                "checker_version": "ifevalg-checker-v1",
                "prompt": "write something",
                "constraints": [{"id": "no_such:verifier", "kwargs": {}}],
            }
        )
    )


def test_grading_the_same_answer_twice_agrees() -> None:
    """Within a process and between two builds of the same constraint.

    A checker that fills an argument in for itself would pass the first half
    and fail the second, which is what `corpus._is_stable` exists to keep out
    of the corpus.
    """
    environment = InstructionFollowingEnvironment("eval")
    golden = next(row for row in _goldens() if row["split"] == "eval")
    first = environment.grade(golden["index"], golden["answer"])
    second = InstructionFollowingEnvironment("eval").grade(
        golden["index"], golden["answer"]
    )
    assert first == second

    for index in range(200):
        spec = _build_task(index, "eval")["private"]["verifier_spec"]
        assert spec == _build_task(index, "eval")["private"]["verifier_spec"]
        answer = "Point one is fine. Point two is also fine."
        assert grade(spec, answer) == grade(spec, answer)


def test_a_band_selects_by_constraint_count() -> None:
    """The difficulty dial, and that it moves nothing else."""
    full = InstructionFollowingEnvironment("eval")
    hard = InstructionFollowingEnvironment("eval", (3,))
    assert 0 < len(hard) < len(full)
    keys = {row.key for row in corpus.rows("eval")}
    for index in range(50):
        task = hard.task(index)
        assert task["metadata"]["constraints"] == 3
        assert task["metadata"]["corpus_row"] in keys
    with pytest.raises(ValueError):
        InstructionFollowingEnvironment("eval", (4,))
    with pytest.raises(ValueError):
        InstructionFollowingEnvironment("nope")


def test_the_vendored_checkers_need_nothing_downloaded() -> None:
    """No `nltk`, no `langdetect`, no data fetched the first time it grades.

    Both would make the reward depend on a download and on which release
    answered it; `langdetect` samples, so two participants can disagree about
    the same text.
    """
    assert not {"nltk", "langdetect", "absl", "immutabledict"} & set(sys.modules)
    ifeval = importlib.resources.files("reliquary_instruction_following").joinpath(
        "_ifeval"
    )
    for name in ("instructions.py", "instructions_util.py", "instructions_registry.py"):
        source = ifeval.joinpath(name).read_text(encoding="utf-8")
        assert "import nltk" not in source
        assert "\nimport langdetect" not in source
        assert "from absl" not in source
    assert ifeval.joinpath("LICENSE").read_text().lstrip().startswith("Apache License")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world. Goodbye world.", ["Hello", "world", ".", "Goodbye", "world", "."]),
        (
            "It's a dog-eat-dog world, isn't it?",
            ["It", "'s", "a", "dog-eat-dog", "world", ",", "is", "n't", "it", "?"],
        ),
        (
            'She said "stop", then left.',
            ["She", "said", "``", "stop", "''", ",", "then", "left", "."],
        ),
        # The known divergence: NLTK's Punkt model knows `Dr.` is an
        # abbreviation and keeps the period on the word. A regex cannot.
        (
            "Dr. Smith went to Washington.",
            ["Dr", ".", "Smith", "went", "to", "Washington", "."],
        ),
    ],
)
def test_the_pinned_tokenizer_splits_where_nltk_splits(
    text: str, expected: list[str]
) -> None:
    assert word_tokenize(text) == expected


def test_words_are_counted_as_the_pattern_says() -> None:
    """`\\w+`, the same pattern NLTK's `RegexpTokenizer` was given.

    A hyphenated word counts as its parts and a clitic as its own word, which
    is what the word-count constraints have always measured.
    """
    assert count_words("It's a dog-eat-dog world") == 7
    assert count_words("") == 0


def test_the_compatibility_surface_is_deterministic() -> None:
    environment = InstructionFollowingEnvironment("eval")
    assert environment.task(7) == InstructionFollowingEnvironment("eval").task(7)
    assert environment.max_turns == 1
    assert environment.validator_authoritative_reward is True
    assert environment.name == ENVIRONMENT
    # Indices wrap: a taskset may be asked for more tasks than the split holds.
    assert environment.task(0)["id"] == environment.task(len(environment))["id"]
    golden = next(row for row in _goldens() if row["split"] == "eval")
    replay = environment.replay(golden["index"], golden["answer"])
    assert replay["reward"]["reward"] == 1.0


def test_the_declared_contract_matches_the_surface() -> None:
    """`checked-answer`, not `answer-json`: there is no gold completion.

    Writing a text that satisfies the constraints is the task itself. The
    sibling contract promises a `reference_completion`, and naming it here
    would claim a surface this environment cannot have, so the name differs
    and this test is what keeps it honest.
    """
    artifact = json.loads(
        importlib.resources.files("reliquary_instruction_following")
        .joinpath("artifact.json")
        .read_text()
    )
    assert artifact["contract"] == "reliquary/checked-answer/v1"
    assert artifact["environment"] == ENVIRONMENT
    surface = {"task", "grade", "replay"}
    assert surface <= set(dir(InstructionFollowingEnvironment))
    assert not {"reference_completion", "reset", "step"} & set(
        dir(InstructionFollowingEnvironment)
    )


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_instruction_following")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == (
        "reliquary_instruction_following:InstructionFollowingTaskset"
    )
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        == artifact["source_manifest_sha256"]
    )
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected


def test_the_manifest_declares_what_the_corpus_holds() -> None:
    manifest = tomllib.loads(
        (Path(__file__).parents[1] / "environment.toml").read_text()
    )
    families = {
        constraint.family for row in corpus.load() for constraint in row.constraints
    }
    assert set(manifest["task_families"]) == families
    assert len(families) == FAMILIES
    assert manifest["execution"]["max_turns"] == 1
    assert manifest["execution"]["network"] is False
    assert manifest["data"]["sha256"] == corpus.CORPUS_SHA256
    assert manifest["data"]["rows"] == corpus.CORPUS_ROWS
    assert manifest["data"]["license"] == "odc-by"
    assert manifest["entrypoint"] == (
        "reliquary_instruction_following:InstructionFollowingTaskset"
    )
    assert manifest["compatibility_entrypoint"] == (
        "reliquary_instruction_following:InstructionFollowingEnvironment"
    )


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
    for phase, split in (("train", "train"), ("eval", "eval")):
        source = config["orchestrator"][phase]["source"][0]
        assert source["env"]["taskset"] == {
            "id": "reliquary-instruction-following",
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
    assert exported == [package.InstructionFollowingTaskset]
    config_type = vf.taskset_config_type("reliquary-instruction-following")
    taskset = vf.load_taskset(config_type(id="reliquary-instruction-following"))
    tasks = list(taskset.head(3))
    assert len(tasks) == 3
    environment = InstructionFollowingEnvironment()
    for position, task in enumerate(tasks):
        assert task.key == environment.task(position)["id"]
        assert asyncio.run(task.validate(None)) is True
        for completion, expected in (("", 0.0), ("this ignores the instructions", 0.0)):
            trace = _trace(task, completion)
            assert asyncio.run(task.followed_instructions(trace)) == expected
            wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
            assert asyncio.run(task.followed_instructions(wire)) == expected


@needs_verifiers
def test_both_surfaces_agree_on_the_same_answer() -> None:
    """A reward earned in training must replay to the same number."""
    config_type = vf.taskset_config_type("reliquary-instruction-following")
    for golden in _goldens():
        taskset = vf.load_taskset(
            config_type(id="reliquary-instruction-following", split=golden["split"])
        )
        task = next(
            item for item in taskset.head(golden["index"] + 1)
            if item.data.idx == golden["index"]
        )
        environment = InstructionFollowingEnvironment(golden["split"])
        assert task.key == environment.task(golden["index"])["id"]
        assert task.data.constraints == golden["constraints"]
        for answer in (golden["answer"], golden["broken_answer"], ""):
            trace = _trace(task, answer)
            asyncio.run(task.score(trace))
            assert trace.reward == environment.grade(golden["index"], answer)["reward"]
            assert set(trace.rewards) == {"followed_instructions"}
            assert trace.rewards["followed_instructions"].weight == 1.0


@needs_verifiers
def test_a_taskset_carries_its_split_and_band_to_the_task() -> None:
    config_type = vf.taskset_config_type("reliquary-instruction-following")
    taskset = vf.load_taskset(
        config_type(
            id="reliquary-instruction-following",
            split="qualification",
            constraint_counts=(1,),
        )
    )
    task = next(iter(taskset))
    assert task.data.split == "qualification"
    assert task.data.constraints == 1
    assert task.key == (
        InstructionFollowingEnvironment("qualification", (1,)).task(0)["id"]
    )
    assert task.data.verifier_spec == (
        _build_task(0, "qualification", (1,))["private"]["verifier_spec"]
    )
    # A supplied few-shot answer is not a sampled model response.
    golden = next(row for row in _goldens() if row["split"] == "train")
    trained = next(
        iter(vf.load_taskset(config_type(id="reliquary-instruction-following")))
    )
    trace = _trace(trained, golden["answer"])
    trace.nodes[0].sampled = False
    asyncio.run(trained.score(trace))
    assert trace.reward == 0.0
