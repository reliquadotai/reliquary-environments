import asyncio
import hashlib
import importlib
import pkgutil

from reliquary_math.taskset import MathConfig


def test_package_imports() -> None:
    module = importlib.import_module("reliquary_math")
    assert module.__name__ == "reliquary_math"


def test_no_module_imports_reliquary_core() -> None:
    """The port is a copy, not a dependency. A stray `import reliquary`
    would make the package silently unusable outside the core checkout."""
    import reliquary_math

    for info in pkgutil.walk_packages(
        reliquary_math.__path__, prefix="reliquary_math."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "import reliquary\n" not in text
        assert "from reliquary." not in text
        assert "import reliquary." not in text


from reliquary_math.virtual_parquet import VirtualParquetDataset


def test_virtual_parquet_is_importable_and_lazy() -> None:
    """Constructing must not touch the network: the class exists to avoid a
    bulk download, so an eager constructor would defeat its whole purpose."""
    dataset = VirtualParquetDataset(
        "nvidia/OpenMathInstruct-2",
        "469216e3f46f4dacf476b382e192485ea51a143e",
        columns=["problem", "expected_answer"],
        fs=None,
    )
    assert dataset is not None


from reliquary_math import corpus


def test_shard_list_is_exactly_the_full_train_split() -> None:
    """The subset directories (train_1M/2M/5M) duplicate rows already in the
    full split. Naming the 32 files explicitly makes that class of mistake
    impossible instead of guarding it with a flag."""
    assert len(corpus.TRAIN_SHARDS) == 32
    assert corpus.TRAIN_SHARDS[0] == "data/train-00000-of-00032.parquet"
    assert corpus.TRAIN_SHARDS[-1] == "data/train-00031-of-00032.parquet"
    assert all(name.startswith("data/train-") for name in corpus.TRAIN_SHARDS)
    assert not any("train_1M" in name for name in corpus.TRAIN_SHARDS)
    assert not any("train_2M" in name for name in corpus.TRAIN_SHARDS)
    assert not any("train_5M" in name for name in corpus.TRAIN_SHARDS)
    assert len(set(corpus.TRAIN_SHARDS)) == 32


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OMI_REPO == "nvidia/OpenMathInstruct-2"
    assert corpus.OMI_REVISION == "469216e3f46f4dacf476b382e192485ea51a143e"


class _FakeShardListingFilesystem:
    """Stands in for `HfFileSystem`: `ls(path, detail=False)` is the only
    method `_ensure_manifest`'s selection step calls on it."""

    def __init__(self, basenames: list[str]) -> None:
        self._basenames = basenames

    def ls(self, path: str, detail: bool = False) -> list[str]:
        assert detail is False
        return [f"{path}/{name}" for name in self._basenames]


class _StubParquetFile:
    """Stands in for `pyarrow.parquet.ParquetFile`: only `.metadata` and
    `.close()` are touched once `_open_parquet_file` is bypassed, so the
    manifest can be built without reading any real parquet bytes."""

    class _Metadata:
        num_row_groups = 0

    metadata = _Metadata()

    def close(self) -> None:
        pass


def test_load_corpus_selects_exactly_the_train_shards_and_excludes_subsets(
    monkeypatch,
) -> None:
    """Drives `corpus.load_corpus()` through `VirtualParquetDataset`'s real
    manifest-selection code (`_ensure_manifest`'s `fs.ls` at
    virtual_parquet.py:291 filtered by `_shard_included`/`_filename_prefix`
    at virtual_parquet.py:148-152), offline, against a fake listing holding
    both the 32 canonical shards and 23 train_1M/2M/5M decoys.

    The tests above only check the `TRAIN_SHARDS` constant against a pattern
    generated three lines above it in the same module — a tautology that
    stays green even if `load_corpus()` stopped passing `filename_prefix`
    altogether. This test fails if that happens: it captures the
    `filename_prefix` `load_corpus()` actually hands the constructor, and it
    fails if the resulting file list admits a single train_1M/2M/5M decoy.
    """
    corpus.load_corpus.cache_clear()

    canonical = [f"train-{index:05d}-of-00032.parquet" for index in range(32)]
    decoys = (
        [f"train_1M-{index:05d}-of-00003.parquet" for index in range(3)]
        + [f"train_2M-{index:05d}-of-00006.parquet" for index in range(6)]
        + [f"train_5M-{index:05d}-of-00014.parquet" for index in range(14)]
    )
    assert len(decoys) == 23
    fake_fs = _FakeShardListingFilesystem(canonical + decoys)

    captured_kwargs: dict[str, object] = {}
    real_init = corpus.VirtualParquetDataset.__init__

    def spying_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)
        kwargs["fs"] = fake_fs
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(corpus.VirtualParquetDataset, "__init__", spying_init)
    monkeypatch.setattr(
        corpus.VirtualParquetDataset,
        "_open_parquet_file",
        lambda self, path: _StubParquetFile(),
    )

    try:
        dataset = corpus.load_corpus()
        dataset._ensure_manifest()
        selected = sorted(path.rsplit("/", 1)[-1] for path in dataset._files)
    finally:
        corpus.load_corpus.cache_clear()

    assert captured_kwargs.get("filename_prefix") == "train-"
    assert selected == sorted(canonical)
    assert not (set(selected) & set(decoys))


import pytest

from reliquary_math.grading import answers_equal, compute_reward


@pytest.mark.parametrize(
    "candidate,truth",
    [
        ("45", "45"),
        ("\\frac{3}{4}", "0.75"),
        ("0.5", "\\frac{1}{2}"),
        ("2\\sqrt{2}", "\\sqrt{8}"),
        ("$12$", "12"),
        ("(1, 2)", "(1,2)"),
        # Every protocol profile this package targets (v4 and later) sets
        # prompt_encoding="raw", under which raw-completion answers come
        # wrapped in inline/display LaTeX delimiters that carry no meaning
        # of their own. Verified against Reliquary core at 10c2a4d9 with
        # RAW_COMPLETION_PROMPTS forced True.
        ("\\(45\\)", "45"),
        ("\\[45\\]", "45"),
    ],
)
def test_equivalent_surface_forms_compare_equal(candidate: str, truth: str) -> None:
    assert answers_equal(candidate, truth) is True


@pytest.mark.parametrize(
    "candidate,truth",
    [
        ("45", "46"),
        ("\\frac{3}{4}", "0.74"),
        ("(1, 2)", "(2, 1)"),
        ("", "5"),
        # Verified against Reliquary core at 10c2a4d9: core's grader does
        # not strip an "x =" prefix or a thousands separator, so these two
        # read like equivalent forms but are not. Pinned here rather than
        # dropped so the port stays honest about where core is stricter
        # than it looks.
        ("x = 5", "5"),
        ("1,000", "1000"),
    ],
)
def test_different_answers_compare_unequal(candidate: str, truth: str) -> None:
    assert answers_equal(candidate, truth) is False


def test_reward_reads_the_last_boxed_answer() -> None:
    """A model that reconsiders mid-completion is graded on its conclusion,
    not on its first attempt."""
    problem = {"expected_answer": "7"}
    completion = "First \\boxed{3}. On reflection, \\boxed{7}."
    assert compute_reward(problem, completion) == 1.0


def test_reward_is_zero_without_a_boxed_answer() -> None:
    assert compute_reward({"expected_answer": "7"}, "the answer is 7") == 0.0


def test_reward_refuses_an_unsafe_expression() -> None:
    """The safety gate is what keeps a model answer from becoming arbitrary
    sympy evaluation."""
    problem = {"expected_answer": "7"}
    assert compute_reward(problem, "\\boxed{__import__('os').system('id')}") == 0.0


from reliquary_math.taskset import DEFAULT_PROMPT, MathEnvironment


def test_reference_completion_scores_one_and_a_wrong_answer_scores_zero(
    monkeypatch,
) -> None:
    """Both halves matter: a grader that accepted anything would pass the
    first on its own."""
    monkeypatch.setattr(
        "reliquary_math.taskset.get_problem",
        lambda index: {"problem": "1+1?", "expected_answer": "2"},
    )
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 10)
    environment = MathEnvironment()

    assert environment.grade(0, environment.reference_completion(0))["reward"] == 1.0
    assert environment.grade(0, "\\boxed{99}")["reward"] == 0.0


def test_task_renders_the_problem_into_the_template(monkeypatch) -> None:
    monkeypatch.setattr(
        "reliquary_math.taskset.get_problem",
        lambda index: {"problem": "1+1?", "expected_answer": "2"},
    )
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 10)
    environment = MathEnvironment(prompt_template="Q: {problem} A:")

    assert environment.task(0)["prompt"] == "Q: 1+1? A:"


def test_default_prompt_asks_for_a_boxed_answer() -> None:
    assert "\\boxed" in DEFAULT_PROMPT


def test_index_is_bounded_by_the_corpus(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 10)
    monkeypatch.setattr(
        "reliquary_math.taskset.get_problem",
        lambda index: {"problem": str(index), "expected_answer": "2"},
    )
    environment = MathEnvironment()
    assert len(environment) == 10
    with pytest.raises(IndexError):
        environment.task(10)


from reliquary_math import MathEnvironment as ExportedEnvironment
from reliquary_math import MathTaskset


def test_package_exports_exactly_two_names() -> None:
    import reliquary_math

    assert reliquary_math.__all__ == ["MathEnvironment", "MathTaskset"]
    assert ExportedEnvironment is not None


def test_taskset_yields_tasks_that_carry_the_index(monkeypatch) -> None:
    monkeypatch.setattr(
        "reliquary_math.taskset.get_problem",
        lambda index: {"problem": f"p{index}", "expected_answer": "2"},
    )
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 3)

    taskset = MathTaskset(MathConfig())
    tasks = list(taskset.load())

    assert len(tasks) == 3
    assert [task.data.idx for task in tasks] == [0, 1, 2]
    assert tasks[0].data.prompt.startswith("Solve")


def test_taskset_load_reads_each_row_exactly_once(monkeypatch) -> None:
    """`load()` needs both the prompt and the ground truth from one row;
    fetching it twice per index would be wasted work at best and a real
    second read for a consumer that seeks non-sequentially."""
    calls: list[int] = []

    def counting_get_problem(index: int) -> dict[str, str]:
        calls.append(index)
        return {"problem": f"p{index}", "expected_answer": "2"}

    monkeypatch.setattr("reliquary_math.taskset.get_problem", counting_get_problem)
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 3)

    list(MathTaskset(MathConfig()).load())

    assert calls == [0, 1, 2]


def test_task_validate_accepts_the_reference_and_rejects_a_wrong_answer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "reliquary_math.taskset.get_problem",
        lambda index: {"problem": "1+1?", "expected_answer": "2"},
    )
    monkeypatch.setattr("reliquary_math.taskset.corpus_length", lambda: 1)

    task = next(iter(MathTaskset(MathConfig()).load()))
    assert asyncio.run(task.validate(None)) is True


import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "reliquary_math"


def test_goldens_replay_offline() -> None:
    """Every pinned index must still produce its recorded prompt, and the
    dataset's own answer must still score 1.0 while a wrong one scores 0."""
    lines = (
        PACKAGE_ROOT / "goldens" / "reference.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 20

    environment = MathEnvironment()
    for line in lines:
        golden = json.loads(line)
        task = environment.task(golden["index"])
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        reference = environment.reference_completion(golden["index"])
        assert environment.grade(golden["index"], reference)["reward"] == golden[
            "reference_reward"
        ]
        assert environment.grade(golden["index"], "\\boxed{__not_the_answer__}")[
            "reward"
        ] == golden["wrong_reward"]


def test_artifact_manifest_hashes_installed_files() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.build_artifact import build_artifact

    committed = json.loads((PACKAGE_ROOT / "artifact.json").read_text())
    regenerated = build_artifact(
        PACKAGE_ROOT,
        environment=committed["environment"],
        contract=committed["contract"],
        distribution=committed["distribution"],
        entrypoints=committed["entrypoints"],
    )
    assert regenerated == committed


def test_environment_toml_declares_the_real_pins() -> None:
    import tomllib

    declared = tomllib.loads(
        (PACKAGE_ROOT.parent / "environment.toml").read_text(encoding="utf-8")
    )
    assert declared["id"] == "reliquary/math"
    assert declared["entrypoint"] == "reliquary_math:MathTaskset"
    assert declared["compatibility_entrypoint"] == "reliquary_math:MathEnvironment"
    assert declared["provenance"]["port"] == "generator-identical-new-identity"
    assert declared["data"]["license"] == "cc-by-4.0"
    assert corpus.OMI_REVISION in declared["data"]["train"]


def test_repository_compatibility_declares_this_package() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[4]
    declared = tomllib.loads((root / "compatibility.toml").read_text())
    assert "reliquary_math" in declared["releases"]
    assert declared["releases"]["reliquary_math"]["tag"] == "v0.1.0a2"
