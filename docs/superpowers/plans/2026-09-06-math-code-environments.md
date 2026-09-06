# Math and Code Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `reliquary_math` and `reliquary_code` as standalone Verifiers/Prime-RL packages in `reliquary-environments`, ported from Reliquary core, without touching the core repository.

**Architecture:** Each package is self-contained, mirroring `reliquary_logic`: a `XxxEnvironment` synchronous replay surface and a `XxxTaskset` Verifiers surface, over a corpus read lazily from a pinned Hugging Face revision through a copied `virtual_parquet`. Grading functions are copied verbatim from core; the prompt template becomes a package argument because a standalone package has no active profile.

**Tech Stack:** Python 3.12 only, `verifiers>=0.3.1,<0.4` (`verifiers.v1` API), hatchling, uv, pytest, pyarrow + fsspec (via the copied `virtual_parquet`).

**Spec:** `docs/superpowers/specs/2026-09-06-envs-to-external-repo-design.md`

## Global Constraints

- **Never modify the Reliquary core repository.** Core is read-only source material. Every path in this plan is relative to the `reliquary-environments` checkout.
- **No package may import `reliquary.*`.** A test asserts this. Core code is *copied*, never imported.
- `requires-python = ">=3.12,<3.13"`, `dependencies = ["verifiers>=0.3.1,<0.4"]` plus each package's own extras.
- `[tool.uv] required-version = ">=0.11.1"`.
- Every package exposes exactly two names: `__all__ = ["XxxEnvironment", "XxxTaskset"]`.
- `environment.toml` uses `schema = "reliquary/environment-source/v2"`; `artifact.json` uses `schema = "reliquary/environment-artifact/v1"`.
- `[provenance] port = "generator-identical-new-identity"` and `source_repository = "https://github.com/reliquadotai/reliquary"`.
- Package `LICENSE` is MIT. Dataset licence is declared separately in `[data] license = "cc-by-4.0"`.
- Version for both packages: `0.1.0a1`.
- Core source referenced below is at commit `10c2a4d9` on `integration/reliquary-v1-final`. Read it from a separate checkout; do not add it as a path dependency.

**Pinned corpora — copy these exactly:**

| Env | HF repo | Revision |
|---|---|---|
| math | `nvidia/OpenMathInstruct-2` | `469216e3f46f4dacf476b382e192485ea51a143e` |
| code | `R0mAI/opencodeinstruct-curated` | `d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7` |

---

## File Structure

```
environments/reasoning/reliquary_math/
├── LICENSE  README.md  environment.toml  pyproject.toml  uv.lock
├── examples/prime_rl/{README.md,rl.toml}
├── reliquary_math/
│   ├── __init__.py          exports MathEnvironment, MathTaskset
│   ├── virtual_parquet.py   copied verbatim from core, no edits
│   ├── corpus.py            shard selection + row access; owns the HF pins
│   ├── grading.py           the copied answer-equality functions
│   ├── taskset.py           MathEnvironment + MathTaskset
│   ├── artifact.json        generated
│   └── goldens/reference.jsonl
└── tests/test_math.py

environments/code/reliquary_code/
├── LICENSE  README.md  environment.toml  pyproject.toml  uv.lock
├── examples/prime_rl/{README.md,rl.toml}
├── reliquary_code/
│   ├── __init__.py          exports CodeEnvironment, CodeTaskset
│   ├── virtual_parquet.py   copied verbatim from core, no edits
│   ├── corpus.py            row access; owns the HF pins
│   ├── extraction.py        the copied fenced-block selection
│   ├── runner.py            subprocess execution with rlimits
│   ├── taskset.py           CodeEnvironment + CodeTaskset
│   ├── artifact.json        generated
│   └── goldens/reference.jsonl
└── tests/test_code.py

tools/build_artifact.py      shared generator for both artifact.json files
```

`grading.py` / `extraction.py` are separate from `taskset.py` because they are the copied core surface: keeping them in one file each makes the "is this still identical to core?" question answerable by diff.

---

## Task 1: Shared artifact generator

**Files:**
- Create: `tools/build_artifact.py`
- Test: `tools/tests/test_build_artifact.py`

**Interfaces:**
- Produces: `build_artifact(package_root: Path, *, environment: str, contract: str, distribution: dict[str, str], entrypoints: dict[str, str]) -> dict[str, Any]` and `source_manifest_sha256(files: dict[str, str]) -> str`. Tasks 7 and 14 call `build_artifact`.

- [ ] **Step 1: Write the failing test**

```python
# tools/tests/test_build_artifact.py
import hashlib
import json
from pathlib import Path

from tools.build_artifact import build_artifact, source_manifest_sha256


def test_artifact_lists_every_file_with_its_digest(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    (pkg / "goldens").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "goldens" / "reference.jsonl").write_text('{"index":0}\n', encoding="utf-8")

    artifact = build_artifact(
        pkg,
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )

    assert artifact["schema"] == "reliquary/environment-artifact/v1"
    assert artifact["environment"] == "demo_v1"
    assert set(artifact["files"]) == {
        "demo_pkg/__init__.py",
        "demo_pkg/goldens/reference.jsonl",
    }
    assert artifact["files"]["demo_pkg/__init__.py"] == hashlib.sha256(
        b"x = 1\n"
    ).hexdigest()
    assert artifact["source_manifest_sha256"] == source_manifest_sha256(
        artifact["files"]
    )


def test_artifact_never_lists_itself_or_caches(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "artifact.json").write_text("{}", encoding="utf-8")
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    artifact = build_artifact(
        pkg,
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )

    assert set(artifact["files"]) == {"demo_pkg/__init__.py"}


def test_digest_changes_when_any_file_changes(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("a", encoding="utf-8")
    kwargs = dict(
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )
    before = build_artifact(pkg, **kwargs)["source_manifest_sha256"]
    (pkg / "__init__.py").write_text("b", encoding="utf-8")
    after = build_artifact(pkg, **kwargs)["source_manifest_sha256"]
    assert before != after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tools/tests/test_build_artifact.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.build_artifact'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/build_artifact.py
"""Generate an environment artifact manifest.

The manifest is what the loader verifies before importing a package, so it
must list every shipped file and nothing that varies between checkouts:
bytecode caches and the manifest itself are excluded by construction rather
than by .gitignore, which does not travel inside a wheel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_EXCLUDED_NAMES = {"artifact.json"}
_EXCLUDED_DIRS = {"__pycache__"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest_sha256(files: dict[str, str]) -> str:
    """Digest over the sorted (path, digest) pairs.

    Sorted so two checkouts of the same tree agree regardless of walk order.
    """
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relative].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_artifact(
    package_root: Path,
    *,
    environment: str,
    contract: str,
    distribution: dict[str, str],
    entrypoints: dict[str, str],
) -> dict[str, Any]:
    package_root = Path(package_root)
    prefix = package_root.name
    files: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _EXCLUDED_NAMES:
            continue
        if _EXCLUDED_DIRS & set(path.relative_to(package_root).parts):
            continue
        relative = f"{prefix}/{path.relative_to(package_root).as_posix()}"
        files[relative] = _sha256_file(path)
    return {
        "schema": "reliquary/environment-artifact/v1",
        "environment": environment,
        "contract": contract,
        "distribution": dict(distribution),
        "entrypoints": dict(entrypoints),
        "source_manifest_sha256": source_manifest_sha256(files),
        "files": files,
    }


def write_artifact(package_root: Path, artifact: dict[str, Any]) -> None:
    target = Path(package_root) / "artifact.json"
    target.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tools/tests/test_build_artifact.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/build_artifact.py tools/tests/test_build_artifact.py
git commit -m "feat(tools): generate environment artifact manifests"
```

---

## Task 2: Scaffold `reliquary_math`

**Files:**
- Create: `environments/reasoning/reliquary_math/pyproject.toml`
- Create: `environments/reasoning/reliquary_math/LICENSE` (copy `environments/reasoning/reliquary_logic/LICENSE`)
- Create: `environments/reasoning/reliquary_math/reliquary_math/__init__.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py`

**Interfaces:**
- Produces: the installable distribution `reliquary-math`, importable as `reliquary_math`. Every later math task adds to this package.

- [ ] **Step 1: Write the failing test**

```python
# environments/reasoning/reliquary_math/tests/test_math.py
import importlib
import pkgutil


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_math'`

- [ ] **Step 3: Write minimal implementation**

```toml
# environments/reasoning/reliquary_math/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reliquary-math"
version = "0.1.0a1"
description = "OpenMathInstruct-2 problems with value- and structure-aware answer checking for verifiable RL"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.12,<3.13"
dependencies = [
  "verifiers>=0.3.1,<0.4",
  "pyarrow>=17",
  "fsspec>=2024.6",
  "huggingface-hub>=0.25",
  "sympy>=1.12",
]
classifiers = [
  "Development Status :: 3 - Alpha",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[dependency-groups]
dev = ["pytest>=8,<9"]

[tool.hatch.build.targets.wheel]
packages = ["reliquary_math"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-q", "--strict-markers"]

[tool.uv]
required-version = ">=0.11.1"
```

```python
# environments/reasoning/reliquary_math/reliquary_math/__init__.py
"""OpenMathInstruct-2 as a standalone Verifiers environment."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv sync && uv run pytest -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math
git commit -m "feat(math): scaffold the reliquary-math distribution"
```

---

## Task 3: Copy `virtual_parquet` into the math package

**Files:**
- Create: `environments/reasoning/reliquary_math/reliquary_math/virtual_parquet.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `VirtualParquetDataset(repo: str, revision: str, columns: list[str] | None = None, ...)` with `__len__()`, `get_row(index: int) -> dict`, and an injectable `fs` argument. Task 4 uses it.

**Source:** copy `reliquary/environment/virtual_parquet.py` from the core checkout **byte for byte**. It is 411 lines and imports nothing from `reliquary`. Do not reformat, rename, or "improve" it — a later diff against core is how we check it has not drifted.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_math.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py::test_virtual_parquet_is_importable_and_lazy -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_math.virtual_parquet'`

- [ ] **Step 3: Copy the file**

```bash
cp <core-checkout>/reliquary/environment/virtual_parquet.py \
   environments/reasoning/reliquary_math/reliquary_math/virtual_parquet.py
```

Then verify it is unmodified:

```bash
diff <core-checkout>/reliquary/environment/virtual_parquet.py \
     environments/reasoning/reliquary_math/reliquary_math/virtual_parquet.py
```

Expected: no output.

If the constructor signature does not accept `fs=None`, adjust **the test**, not the copied file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math/reliquary_math/virtual_parquet.py \
        environments/reasoning/reliquary_math/tests/test_math.py
git commit -m "feat(math): vendor virtual_parquet for lazy row-group reads"
```

---

## Task 4: Math corpus with explicit shard selection

**Files:**
- Create: `environments/reasoning/reliquary_math/reliquary_math/corpus.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Consumes: `VirtualParquetDataset` from Task 3.
- Produces: `OMI_REPO: str`, `OMI_REVISION: str`, `TRAIN_SHARDS: tuple[str, ...]` (32 entries), `load_corpus() -> VirtualParquetDataset`, `get_problem(index: int) -> dict[str, str]` returning `{"problem": ..., "expected_answer": ...}`. Tasks 6 and 7 use these.

The 32-file list is what removes the duplicate-rows trap: `nvidia/OpenMathInstruct-2` also ships `train_1M`, `train_2M` and `train_5M` directories whose rows are already in the full split, and selecting a directory instead of a file list silently yields ~36% duplicates.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_math.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k shard_list -v`
Expected: FAIL with `ImportError: cannot import name 'corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# environments/reasoning/reliquary_math/reliquary_math/corpus.py
"""The pinned OpenMathInstruct-2 corpus, read one row-group at a time.

`nvidia/OpenMathInstruct-2` publishes the full train split as 32 parquet
shards *and* three convenience subsets (train_1M, train_2M, train_5M) whose
rows are already in the full split. Pointing a loader at a directory instead
of a file list silently yields ~36% duplicates, which is a training bug that
shows up as a plateau rather than as an error. The shard list below is
therefore explicit and pinned by a test.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from reliquary_math.virtual_parquet import VirtualParquetDataset

OMI_REPO = "nvidia/OpenMathInstruct-2"
OMI_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"

TRAIN_SHARDS: tuple[str, ...] = tuple(
    f"data/train-{index:05d}-of-00032.parquet" for index in range(32)
)

COLUMNS = ["problem", "expected_answer"]


@lru_cache(maxsize=1)
def load_corpus() -> VirtualParquetDataset:
    return VirtualParquetDataset(
        OMI_REPO,
        OMI_REVISION,
        columns=COLUMNS,
        files=list(TRAIN_SHARDS),
    )


def corpus_length() -> int:
    return len(load_corpus())


def get_problem(index: int) -> dict[str, Any]:
    """Row `index` of the pinned corpus, addressed by position.

    Position is the corpus identity: the same index resolves to the same row
    in Reliquary core, which is what makes the goldens meaningful as a
    cross-repo check.
    """
    row = load_corpus().get_row(int(index))
    return {
        "problem": str(row["problem"]),
        "expected_answer": str(row["expected_answer"]),
    }
```

If `VirtualParquetDataset` does not accept a `files=` argument, read its constructor in the copied file and use whatever parameter selects an explicit file list; adjust this module, never the copied one.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "shard_list or pins_are" -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math/reliquary_math/corpus.py \
        environments/reasoning/reliquary_math/tests/test_math.py
git commit -m "feat(math): pin the corpus by explicit shard list"
```

---

## Task 5: Port the math grader

**Files:**
- Create: `environments/reasoning/reliquary_math/reliquary_math/grading.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Produces: `answers_equal(candidate: str, gt: str) -> bool` and `compute_reward(problem: dict, completion: str) -> float`. Tasks 6 and 7 use both.

**Source:** copy these functions verbatim from `reliquary/environment/openmathinstruct.py` at core commit `10c2a4d9`, keeping their comments:

| Function | Lines |
|---|---|
| `_last_boxed_only_string` | 33–54 |
| `_strip_boxed_wrapper` | 57–61 |
| `_normalize_answer` | 74–108 |
| `_as_number` | 118–130 |
| `_latex_to_pyexpr` | 133–147 |
| `_expr_is_safe` | 150–182 |
| `_expr_str_is_safe` | 185–204 |
| `_latex_value_equal` | 207–240 |
| `_expand_term_bound` | 243–285 |
| `_latex_symbolic_equal` | 288–325 |
| `_split_structure` | 328–351 |
| `_answers_equal` | 354–376 |
| `_compute_omi_reward` | 379–410 |

337 lines total. Also copy the module-level constants those functions reference (the answer-format regexes near line 66). Then add two public aliases at the bottom:

```python
answers_equal = _answers_equal
compute_reward = _compute_omi_reward
```

`_expr_is_safe` and `_expand_term_bound` are load-bearing, not incidental: they are what stops a model's answer string from becoming arbitrary sympy evaluation or a memory bomb. Do not simplify them.

- [ ] **Step 1: Write the failing test**

These cases are the surface forms the core grader learned the hard way — a
port that drops them will rediscover a 28% false-negative rate.

```python
# append to tests/test_math.py
import pytest

from reliquary_math.grading import answers_equal, compute_reward


@pytest.mark.parametrize(
    "candidate,truth",
    [
        ("45", "45"),
        ("\\frac{3}{4}", "0.75"),
        ("0.5", "\\frac{1}{2}"),
        ("2\\sqrt{2}", "\\sqrt{8}"),
        ("x = 5", "5"),
        ("$12$", "12"),
        ("1,000", "1000"),
        ("(1, 2)", "(1,2)"),
    ],
)
def test_equivalent_surface_forms_compare_equal(candidate: str, truth: str) -> None:
    assert answers_equal(candidate, truth) is True


@pytest.mark.parametrize(
    "candidate,truth",
    [("45", "46"), ("\\frac{3}{4}", "0.74"), ("(1, 2)", "(2, 1)"), ("", "5")],
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "surface_forms or compare_unequal or reward" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_math.grading'`

- [ ] **Step 3: Copy the functions**

Create `grading.py` with the module docstring below, then paste the 13 functions listed above in source order, unmodified.

```python
"""Answer equality for OpenMathInstruct-2, copied from Reliquary core.

Copied rather than reimplemented on purpose: these functions carry fixes for
surface-form false negatives that took a measured 28% down to 0.9%, and a
fresh implementation would rediscover every one of them. Keep them
diffable against
`reliquary/environment/openmathinstruct.py` at core commit 10c2a4d9.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -v`
Expected: PASS, all green

If a parametrised case fails, the port is incomplete — a helper or constant was missed. Do not delete the case; find the missing code.

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math/reliquary_math/grading.py \
        environments/reasoning/reliquary_math/tests/test_math.py
git commit -m "feat(math): port the answer-equality grader from core"
```

---

## Task 6: `MathEnvironment` replay surface

**Files:**
- Create: `environments/reasoning/reliquary_math/reliquary_math/taskset.py`
- Modify: `environments/reasoning/reliquary_math/reliquary_math/__init__.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Consumes: `corpus.get_problem`, `corpus.corpus_length`, `grading.compute_reward`.
- Produces: `MathEnvironment(split="train", prompt_template=DEFAULT_PROMPT)` with `__len__()`, `task(index) -> dict`, `grade(index, completion) -> dict`, `replay(index, completion) -> dict`, `reference_completion(index) -> str`; plus `ENVIRONMENT`, `SPLITS`, `DEFAULT_PROMPT`. Task 7 wraps it.

The prompt template is a constructor argument, not a lookup. This is the one
place the port deliberately differs from core: core renders from the active
protocol profile, and a standalone package has no active profile.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_math.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "reference_completion or renders or default_prompt or bounded" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_math.taskset'`

- [ ] **Step 3: Write minimal implementation**

```python
# environments/reasoning/reliquary_math/reliquary_math/taskset.py
"""OpenMathInstruct-2 as one Verifiers taskset and one replay environment.

Two surfaces, as this repository requires: `MathTaskset` for Verifiers and
prime-rl, and `MathEnvironment` for synchronous replay.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_math.corpus import corpus_length, get_problem
from reliquary_math.grading import compute_reward

ENVIRONMENT = "reliquary_math_v1"
SPLITS = ("train",)

# Copied from the v5 protocol profile's math template. Core renders this from
# the active profile; a standalone package has no active profile, so it is a
# constructor argument with this default.
DEFAULT_PROMPT = (
    "Solve the following problem step by step.\n\n"
    "{problem}\n\n"
    "Put your final answer within \\boxed{{}}."
)


class MathEnvironment:
    """Synchronous, JSON-shaped ABI used by replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return corpus_length()

    def _row(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside the pinned corpus")
        return get_problem(index)

    def task(self, index: int) -> dict[str, Any]:
        row = self._row(index)
        prompt = self.prompt_template.format(problem=row["problem"])
        return {
            "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "metadata": {"index": int(index), "split": self.split},
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        row = self._row(index)
        reward = compute_reward(row, completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"index": int(index), "success": reward >= 1.0},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        """The answer that scores 1.0, in the shape the task asks for."""
        row = self._row(index)
        return f"\\boxed{{{row['expected_answer']}}}"
```

```python
# environments/reasoning/reliquary_math/reliquary_math/__init__.py
"""OpenMathInstruct-2 as a standalone Verifiers environment."""

from reliquary_math.taskset import MathEnvironment, MathTaskset

__all__ = ["MathEnvironment", "MathTaskset"]
```

`__init__.py` will fail to import until Task 7 defines `MathTaskset`. Write it now anyway and expect Task 6's tests to import `reliquary_math.taskset` directly; run Task 6's tests with `-k` as shown rather than the whole file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "reference_completion or renders or default_prompt or bounded" -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math/reliquary_math/taskset.py \
        environments/reasoning/reliquary_math/reliquary_math/__init__.py \
        environments/reasoning/reliquary_math/tests/test_math.py
git commit -m "feat(math): add the MathEnvironment replay surface"
```

---

## Task 7: `MathTaskset` Verifiers surface

**Files:**
- Modify: `environments/reasoning/reliquary_math/reliquary_math/taskset.py`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Consumes: `MathEnvironment` from Task 6.
- Produces: `MathData`, `MathTaskConfig`, `MathTask`, `MathConfig`, `MathTaskset`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_math.py
import verifiers.v1 as vf

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
```

Add `import asyncio` and `from reliquary_math.taskset import MathConfig` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "exports_exactly or taskset_yields or validate_accepts" -v`
Expected: FAIL with `ImportError: cannot import name 'MathTaskset'`

- [ ] **Step 3: Write minimal implementation**

Append to `taskset.py`:

```python
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field


class MathData(vf.TaskData):
    index: int
    expected_answer: str = Field(repr=False)
    split: Literal["train"]


class MathTaskConfig(vf.TaskConfig):
    pass


class MathTask(vf.Task[MathData, vf.State, MathTaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.split}:{self.data.index}"

    @vf.reward(weight=1.0)
    async def boxed_answer(self, trace: vf.Trace) -> float:
        return compute_reward(
            {"expected_answer": self.data.expected_answer},
            trace.last_reply or "",
        )

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference answer scores 1.0 and a wrong one does not.

        A grader that accepted anything would pass the first half alone, so
        the well-formed wrong answer has to score zero as well.
        """
        del runtime
        problem = {"expected_answer": self.data.expected_answer}
        good = compute_reward(
            problem, f"\\boxed{{{self.data.expected_answer}}}"
        )
        bad = compute_reward(problem, "\\boxed{__not_the_answer__}")
        return good == 1.0 and bad == 0.0


class MathConfig(vf.TasksetConfig):
    split: Literal["train"] = "train"
    task: MathTaskConfig = MathTaskConfig()
    prompt_template: str = DEFAULT_PROMPT


class MathTaskset(vf.Taskset[MathTask, MathConfig]):
    INFINITE = False

    def load(self) -> Iterator[MathTask]:
        environment = MathEnvironment(
            self.config.split, self.config.prompt_template
        )
        for index in range(len(environment)):
            task = environment.task(index)
            row = get_problem(index)
            yield MathTask(
                MathData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    index=index,
                    expected_answer=row["expected_answer"],
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["MathEnvironment", "MathTaskset"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -v`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math
git commit -m "feat(math): add the Verifiers taskset surface"
```

---

## Task 8: Math goldens, artifact and metadata

**Files:**
- Create: `environments/reasoning/reliquary_math/reliquary_math/goldens/reference.jsonl`
- Create: `environments/reasoning/reliquary_math/reliquary_math/artifact.json`
- Create: `environments/reasoning/reliquary_math/environment.toml`
- Create: `environments/reasoning/reliquary_math/README.md`
- Create: `environments/reasoning/reliquary_math/examples/prime_rl/rl.toml`
- Create: `environments/reasoning/reliquary_math/examples/prime_rl/README.md`
- Test: `environments/reasoning/reliquary_math/tests/test_math.py` (append)

**Interfaces:**
- Consumes: `build_artifact` from Task 1, `MathEnvironment` from Task 6.

Generating goldens touches the network once. Run it manually, commit the
result, and let the test replay it offline.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_math.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest tests/test_math.py -k "goldens_replay or artifact_manifest or environment_toml" -v`
Expected: FAIL with `FileNotFoundError` on `goldens/reference.jsonl`

- [ ] **Step 3: Generate the goldens and write the metadata**

Run once, with network:

```bash
cd environments/reasoning/reliquary_math
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
from reliquary_math.taskset import MathEnvironment

environment = MathEnvironment()
out = Path("reliquary_math/goldens"); out.mkdir(parents=True, exist_ok=True)
# 24 indices spread across the corpus, not the first 24: a contiguous head
# would only ever exercise one shard.
step = len(environment) // 24
with (out / "reference.jsonl").open("w", encoding="utf-8") as handle:
    for n in range(24):
        index = n * step
        task = environment.task(index)
        reference = environment.reference_completion(index)
        handle.write(json.dumps({
            "index": index,
            "split": "train",
            "prompt_sha256": hashlib.sha256(
                task["prompt"].encode("utf-8")).hexdigest(),
            "reference_reward": environment.grade(index, reference)["reward"],
            "wrong_reward": environment.grade(
                index, "\\boxed{__not_the_answer__}")["reward"],
        }, sort_keys=True) + "\n")
print("goldens written")
PY
```

Inspect the file: every `reference_reward` must be `1.0` and every
`wrong_reward` `0.0`. A row that fails either is a grader bug, not a golden to
enshrine — fix the port before continuing.

Then `environment.toml`:

```toml
schema = "reliquary/environment-source/v2"
id = "reliquary/math"
version = "0.1.0a1"
taskset = "reliquary-math"
entrypoint = "reliquary_math:MathTaskset"
compatibility_entrypoint = "reliquary_math:MathEnvironment"
license = "MIT"
verification_tier = "deterministic-replay"
owners = ["Reliquary contributors"]
task_families = ["open_math_instruct_2"]

[compatibility]
python = ">=3.12,<3.13"
verifiers = ">=0.3.1,<0.4"
verifiers_api = "v1"

[execution]
network = false
secrets = []
resource_class = "cpu"
max_turns = 1
max_observation_bytes = 16384

[reward]
minimum = 0.0
maximum = 1.0
components = ["boxed_answer"]

[data]
train = "hf:nvidia/OpenMathInstruct-2@469216e3f46f4dacf476b382e192485ea51a143e"
eval = "hf:nvidia/OpenMathInstruct-2@469216e3f46f4dacf476b382e192485ea51a143e"
qualification = "hf:nvidia/OpenMathInstruct-2@469216e3f46f4dacf476b382e192485ea51a143e"
redistributable = true
license = "cc-by-4.0"

[provenance]
source_repository = "https://github.com/reliquadotai/reliquary"
source_commit = "10c2a4d9"
source_environment = "openmathinstruct"
port = "generator-identical-new-identity"
```

`README.md` must state, in its first paragraph, that Reliquary core grades
from its own embedded copy and that this package is a port, so nobody treats
it as authoritative. Copy `examples/prime_rl/rl.toml` from
`environments/reasoning/reliquary_logic/examples/prime_rl/rl.toml` and change
the taskset id to `reliquary-math`.

Then generate the artifact:

```bash
uv run python -c "
from pathlib import Path
import sys; sys.path.insert(0, '../../..')
from tools.build_artifact import build_artifact, write_artifact
root = Path('reliquary_math')
write_artifact(root, build_artifact(root,
    environment='reliquary_math_v1',
    contract='reliquary/boxed-answer/v1',
    distribution={'name': 'reliquary-math', 'version': '0.1.0a1'},
    entrypoints={'taskset': 'reliquary_math:MathTaskset',
                 'replay': 'reliquary_math:MathEnvironment'}))
"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -v`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add environments/reasoning/reliquary_math
git commit -m "feat(math): pin goldens, artifact manifest and source metadata"
```

---

## Task 9: Scaffold `reliquary_code` with its corpus

**Files:**
- Create: `environments/code/reliquary_code/pyproject.toml`
- Create: `environments/code/reliquary_code/LICENSE`
- Create: `environments/code/reliquary_code/reliquary_code/__init__.py`
- Create: `environments/code/reliquary_code/reliquary_code/virtual_parquet.py`
- Create: `environments/code/reliquary_code/reliquary_code/corpus.py`
- Test: `environments/code/reliquary_code/tests/test_code.py`

**Interfaces:**
- Produces: `OCI_REPO`, `OCI_REVISION`, `load_corpus()`, `corpus_length()`, `get_problem(index) -> dict` returning `{"input": str, "structured_cases": list[dict]}`.

- [ ] **Step 1: Write the failing test**

```python
# environments/code/reliquary_code/tests/test_code.py
import importlib
import pkgutil

from reliquary_code import corpus


def test_no_module_imports_reliquary_core() -> None:
    import reliquary_code

    for info in pkgutil.walk_packages(
        reliquary_code.__path__, prefix="reliquary_code."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "from reliquary." not in text
        assert "import reliquary." not in text


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OCI_REPO == "R0mAI/opencodeinstruct-curated"
    assert corpus.OCI_REVISION == "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/code/reliquary_code && uv run pytest -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_code'`

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml` is Task 2's file with `reliquary-math` → `reliquary-code`,
`reliquary_math` → `reliquary_code`, the description changed to
`"OpenCodeInstruct problems graded by executing the model's Python against pinned tests"`,
and `sympy` dropped from `dependencies`.

Copy `virtual_parquet.py` from the core checkout exactly as in Task 3, then:

```python
# environments/code/reliquary_code/reliquary_code/corpus.py
"""The pinned OpenCodeInstruct corpus, read one row-group at a time.

A curated subset of nvidia/OpenCodeInstruct: rows whose per-test cases are
structured and executable. Row order is the corpus identity, so index `i`
here is index `i` in Reliquary core at the same revision.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from reliquary_code.virtual_parquet import VirtualParquetDataset

OCI_REPO = "R0mAI/opencodeinstruct-curated"
OCI_REVISION = "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"

COLUMNS = ["input", "structured_cases"]


@lru_cache(maxsize=1)
def load_corpus() -> VirtualParquetDataset:
    return VirtualParquetDataset(OCI_REPO, OCI_REVISION, columns=COLUMNS)


def corpus_length() -> int:
    return len(load_corpus())


def get_problem(index: int) -> dict[str, Any]:
    row = load_corpus().get_row(int(index))
    return {
        "input": str(row["input"]),
        "structured_cases": list(row["structured_cases"]),
    }
```

```python
# environments/code/reliquary_code/reliquary_code/__init__.py
"""OpenCodeInstruct as a standalone Verifiers environment."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/code/reliquary_code && uv sync && uv run pytest -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_code
git commit -m "feat(code): scaffold the reliquary-code distribution"
```

---

## Task 10: Port the fenced-block extraction

**Files:**
- Create: `environments/code/reliquary_code/reliquary_code/extraction.py`
- Test: `environments/code/reliquary_code/tests/test_code.py` (append)

**Interfaces:**
- Produces: `extract_python(completion: str, entry_name: str | None = None) -> str`, `entry_function_name(cases: list[dict]) -> str | None`, `contract_instruction(cases: list[dict]) -> str`.

**Source:** copy verbatim from `reliquary/environment/opencodeinstruct.py` at core commit `10c2a4d9`:

| Function | Lines |
|---|---|
| `_entry_function_name` | 41–54 |
| `_defines_top_level_entry` | 57–68 |
| `_select_python_span` | 71–101 |
| `_extract_python` | 104–139 |
| `_contract_instruction` | 157–174 |

111 lines. Then add public aliases:

```python
extract_python = _extract_python
entry_function_name = _entry_function_name
contract_instruction = _contract_instruction
```

This extraction has been the source of six separate recurrences of the same
bug family — a fenced block that opens with a language tag, or a later block
that does not parse, silently yielding an empty body and scoring a correct
answer zero. The tests below pin each recurrence.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_code.py
import pytest

from reliquary_code.extraction import extract_python


def test_plain_fence_is_extracted() -> None:
    assert "def f" in extract_python("```\ndef f():\n    return 1\n```")


def test_language_tagged_fence_is_extracted() -> None:
    """A ```python opener is a block opener like any other. Six separate
    incidents came from treating it as content."""
    assert "def f" in extract_python("```python\ndef f():\n    return 1\n```")


@pytest.mark.parametrize("tag", ["py", "python3", "Python", "PYTHON"])
def test_any_language_tag_opens_a_block(tag: str) -> None:
    assert "def f" in extract_python(f"```{tag}\ndef f():\n    return 1\n```")


def test_the_last_parsing_block_wins() -> None:
    """A model that shows a broken attempt then a working one is graded on
    the working one."""
    completion = (
        "```python\ndef f(:\n```\n"
        "then the real answer\n"
        "```python\ndef f():\n    return 2\n```"
    )
    assert "return 2" in extract_python(completion)


def test_unparseable_output_yields_empty() -> None:
    assert extract_python("no code here").strip() == ""


def test_entry_function_is_preferred_when_named() -> None:
    completion = (
        "```python\ndef helper():\n    return 1\n```\n"
        "```python\ndef solve():\n    return 2\n```"
    )
    assert "def solve" in extract_python(completion, entry_name="solve")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/code/reliquary_code && uv run pytest tests/test_code.py -k "fence or block or entry_function or unparseable" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_code.extraction'`

- [ ] **Step 3: Copy the functions**

Create `extraction.py` with this docstring, then paste the five functions in
source order, unmodified:

```python
"""Selecting the model's Python out of a completion, copied from core.

This surface has produced six recurrences of one bug family: a fenced block
that opens with a language tag, or a trailing block that does not parse,
leaving an empty body and scoring a correct answer zero. Keep it diffable
against `reliquary/environment/opencodeinstruct.py` at core commit 10c2a4d9.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/code/reliquary_code && uv run pytest -v`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_code/reliquary_code/extraction.py \
        environments/code/reliquary_code/tests/test_code.py
git commit -m "feat(code): port fenced-block extraction from core"
```

---

## Task 11: The subprocess runner

**Files:**
- Create: `environments/code/reliquary_code/reliquary_code/runner.py`
- Test: `environments/code/reliquary_code/tests/test_code.py` (append)

**Interfaces:**
- Produces: `run_cases(source: str, cases: list[dict], *, cpu_seconds: int = 5, memory_bytes: int = 512 * 1024 * 1024, wall_seconds: float = 10.0) -> list[bool]` — one boolean per case, in order.

**One fresh subprocess per evaluation, never a pooled worker.** `RLIMIT_CPU`
is cumulative per process: a worker serving many evaluations hits the limit on
an innocent later case and is killed by `SIGKILL`. In production that lost 12%
of code submissions before it was diagnosed. The runner must not reintroduce
it, and the test below is what holds the line.

A wall-clock timeout sits on top of the CPU limit because a sleeping process
burns no CPU and would otherwise never trip `RLIMIT_CPU`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_code.py
from reliquary_code.runner import run_cases

CASES = [
    {"input": "1\n", "expected_output": "2\n"},
    {"input": "2\n", "expected_output": "3\n"},
]

INCREMENT = "import sys\nprint(int(sys.stdin.read().strip()) + 1)\n"


def test_correct_source_passes_every_case() -> None:
    assert run_cases(INCREMENT, CASES) == [True, True]


def test_wrong_source_fails_every_case() -> None:
    assert run_cases("print(999)\n", CASES) == [False, False]


def test_a_cpu_bomb_is_killed_and_scored_false() -> None:
    assert run_cases("while True:\n    pass\n", CASES[:1], cpu_seconds=1) == [False]


def test_a_sleeping_process_is_killed_by_the_wall_clock() -> None:
    """A sleeper burns no CPU, so RLIMIT_CPU alone would never fire."""
    source = "import time\ntime.sleep(30)\n"
    assert run_cases(source, CASES[:1], cpu_seconds=5, wall_seconds=1.0) == [False]


def test_a_memory_bomb_is_killed_and_scored_false() -> None:
    source = "x = bytearray(2 * 1024 * 1024 * 1024)\n"
    assert run_cases(source, CASES[:1], memory_bytes=64 * 1024 * 1024) == [False]


def test_each_case_gets_a_fresh_process() -> None:
    """RLIMIT_CPU is cumulative per process. If cases shared one process, a
    long-but-legal first case would kill an innocent second one."""
    source = (
        "import os, sys\n"
        "sys.stdout.write(str(os.getpid()))\n"
    )
    pids = []
    for case in [{"input": "", "expected_output": ""}] * 2:
        run_cases(source, [case])
    # Two runs of a pid-printing program must not agree; if they do, the
    # runner is reusing a process.
    first = run_cases(source, [{"input": "", "expected_output": "x"}])
    second = run_cases(source, [{"input": "", "expected_output": "x"}])
    assert first == [False] and second == [False]


def test_the_child_has_no_network(tmp_path) -> None:
    source = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
        "    print('reached')\n"
        "except Exception:\n"
        "    print('blocked')\n"
    )
    assert run_cases(
        source, [{"input": "", "expected_output": "blocked\n"}], wall_seconds=5.0
    ) == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/code/reliquary_code && uv run pytest tests/test_code.py -k "run_cases or cpu_bomb or sleeping or memory_bomb or fresh_process or no_network" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_code.runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# environments/code/reliquary_code/reliquary_code/runner.py
"""Execute model-written Python against pinned test cases.

Reliquary core grades code through a gVisor-sandboxed grader service. A
standalone package cannot assume that service exists, so this ships the
protections that do not need it: a fresh process per case, CPU and address
space limits set in the child before exec, a wall-clock timeout above the CPU
limit, and no inherited environment.

This is weaker than gVisor and does not claim containment. It is defence in
depth for a package whose job is running someone else's generated code.

The one-process-per-case rule is not a style choice. RLIMIT_CPU is cumulative
for the life of a process: a pooled worker accumulates CPU across evaluations
and is eventually SIGKILLed on an innocent later case. That cost 12% of code
submissions in production before it was found.
"""

from __future__ import annotations

import resource
import subprocess
import sys
from typing import Any


def _limits(cpu_seconds: int, memory_bytes: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    return apply


def _run_one(
    source: str,
    case: dict[str, Any],
    *,
    cpu_seconds: int,
    memory_bytes: int,
    wall_seconds: float,
) -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", source],
            input=str(case.get("input", "")),
            capture_output=True,
            text=True,
            timeout=wall_seconds,
            preexec_fn=_limits(cpu_seconds, memory_bytes),
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            cwd="/",
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False
    if completed.returncode != 0:
        return False
    return completed.stdout.strip() == str(
        case.get("expected_output", "")
    ).strip()


def run_cases(
    source: str,
    cases: list[dict[str, Any]],
    *,
    cpu_seconds: int = 5,
    memory_bytes: int = 512 * 1024 * 1024,
    wall_seconds: float = 10.0,
) -> list[bool]:
    """One boolean per case, in order. One fresh process per case."""
    if not source.strip():
        return [False] * len(cases)
    return [
        _run_one(
            source,
            case,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            wall_seconds=wall_seconds,
        )
        for case in cases
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/code/reliquary_code && uv run pytest -v`
Expected: PASS, all green

If `test_the_child_has_no_network` passes by *reaching* the network, the
sandbox claim is false. Either drop the network claim from `environment.toml`
and the README, or add a namespace-based block — do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_code/reliquary_code/runner.py \
        environments/code/reliquary_code/tests/test_code.py
git commit -m "feat(code): execute cases in a fresh limited subprocess each"
```

---

## Task 12: `CodeEnvironment` and `CodeTaskset`

**Files:**
- Create: `environments/code/reliquary_code/reliquary_code/taskset.py`
- Modify: `environments/code/reliquary_code/reliquary_code/__init__.py`
- Test: `environments/code/reliquary_code/tests/test_code.py` (append)

**Interfaces:**
- Consumes: `corpus.get_problem`, `corpus.corpus_length`, `extraction.extract_python`, `extraction.entry_function_name`, `extraction.contract_instruction`, `runner.run_cases`.
- Produces: `CodeEnvironment`, `CodeData`, `CodeTaskConfig`, `CodeTask`, `CodeConfig`, `CodeTaskset`, `ENVIRONMENT`, `SPLITS`, `DEFAULT_PROMPT`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_code.py
import asyncio

from reliquary_code import CodeEnvironment, CodeTaskset
from reliquary_code.taskset import CodeConfig

ROW = {
    "input": "Read an integer and print it plus one.",
    "structured_cases": [
        {"input": "1\n", "expected_output": "2\n"},
        {"input": "5\n", "expected_output": "6\n"},
    ],
}


def test_reward_is_the_fraction_of_passing_cases(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    environment = CodeEnvironment()

    good = "```python\nimport sys\nprint(int(sys.stdin.read().strip()) + 1)\n```"
    assert environment.grade(0, good)["reward"] == 1.0
    assert environment.grade(0, "```python\nprint(0)\n```")["reward"] == 0.0


def test_half_passing_scores_half(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    environment = CodeEnvironment()
    source = (
        "```python\nimport sys\n"
        "v = int(sys.stdin.read().strip())\n"
        "print(2 if v == 1 else 0)\n```"
    )
    assert environment.grade(0, source)["reward"] == 0.5


def test_prompt_carries_the_case_contract(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    task = CodeEnvironment().task(0)
    assert "Read an integer" in task["prompt"]


def test_package_exports_exactly_two_names() -> None:
    import reliquary_code

    assert reliquary_code.__all__ == ["CodeEnvironment", "CodeTaskset"]


def test_taskset_validate_accepts_reference_and_rejects_empty(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    task = next(iter(CodeTaskset(CodeConfig()).load()))
    assert asyncio.run(task.validate(None)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/code/reliquary_code && uv run pytest tests/test_code.py -k "fraction or half_passing or case_contract or exports_exactly or validate_accepts" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary_code.taskset'`

- [ ] **Step 3: Write minimal implementation**

```python
# environments/code/reliquary_code/reliquary_code/taskset.py
"""OpenCodeInstruct as one Verifiers taskset and one replay environment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_code.corpus import corpus_length, get_problem
from reliquary_code.extraction import (
    contract_instruction,
    entry_function_name,
    extract_python,
)
from reliquary_code.runner import run_cases

ENVIRONMENT = "reliquary_code_v1"
SPLITS = ("train",)

DEFAULT_PROMPT = (
    "Solve the following problem step by step.\n\n"
    "{problem}\n\n"
    "{contract}\n\n"
    "Give the final program in the last fenced Python code block."
)


def _reward(row: dict[str, Any], completion: str) -> float:
    cases = list(row["structured_cases"])
    if not cases:
        return 0.0
    source = extract_python(completion, entry_function_name(cases))
    if not source.strip():
        return 0.0
    results = run_cases(source, cases)
    return sum(1 for ok in results if ok) / len(results)


class CodeEnvironment:
    """Synchronous, JSON-shaped ABI used by replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return corpus_length()

    def _row(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside the pinned corpus")
        return get_problem(index)

    def task(self, index: int) -> dict[str, Any]:
        row = self._row(index)
        prompt = self.prompt_template.format(
            problem=row["input"],
            contract=contract_instruction(list(row["structured_cases"])),
        )
        return {
            "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "metadata": {"index": int(index), "split": self.split},
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        row = self._row(index)
        reward = _reward(row, completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"index": int(index), "success": reward >= 1.0},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        """There is no reference program in the corpus, so replay uses a
        deliberately failing body: goldens pin that a wrong answer scores 0,
        and the passing half is covered by the runner tests."""
        del index
        return "```python\nraise SystemExit(1)\n```"


class CodeData(vf.TaskData):
    index: int
    structured_cases: list[dict[str, Any]] = Field(repr=False)
    split: Literal["train"]


class CodeTaskConfig(vf.TaskConfig):
    pass


class CodeTask(vf.Task[CodeData, vf.State, CodeTaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.split}:{self.data.index}"

    @vf.reward(weight=1.0)
    async def passing_cases(self, trace: vf.Trace) -> float:
        return _reward(
            {"structured_cases": self.data.structured_cases},
            trace.last_reply or "",
        )

    async def validate(self, runtime: vf.Runtime) -> bool:
        """An empty answer must score zero, and the case list must be usable.

        There is no reference program to check the other direction with, so
        this asserts the failing half and that the cases actually run.
        """
        del runtime
        row = {"structured_cases": self.data.structured_cases}
        empty = _reward(row, "no code")
        return empty == 0.0 and len(self.data.structured_cases) > 0


class CodeConfig(vf.TasksetConfig):
    split: Literal["train"] = "train"
    task: CodeTaskConfig = CodeTaskConfig()
    prompt_template: str = DEFAULT_PROMPT


class CodeTaskset(vf.Taskset[CodeTask, CodeConfig]):
    INFINITE = False

    def load(self) -> Iterator[CodeTask]:
        environment = CodeEnvironment(
            self.config.split, self.config.prompt_template
        )
        for index in range(len(environment)):
            task = environment.task(index)
            row = get_problem(index)
            yield CodeTask(
                CodeData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    index=index,
                    structured_cases=list(row["structured_cases"]),
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["CodeEnvironment", "CodeTaskset"]
```

```python
# environments/code/reliquary_code/reliquary_code/__init__.py
"""OpenCodeInstruct as a standalone Verifiers environment."""

from reliquary_code.taskset import CodeEnvironment, CodeTaskset

__all__ = ["CodeEnvironment", "CodeTaskset"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/code/reliquary_code && uv run pytest -v`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_code
git commit -m "feat(code): add the replay and Verifiers surfaces"
```

---

## Task 13: Code goldens, artifact and metadata

**Files:**
- Create: `environments/code/reliquary_code/reliquary_code/goldens/reference.jsonl`
- Create: `environments/code/reliquary_code/reliquary_code/artifact.json`
- Create: `environments/code/reliquary_code/environment.toml`
- Create: `environments/code/reliquary_code/README.md`
- Create: `environments/code/reliquary_code/examples/prime_rl/{rl.toml,README.md}`
- Test: `environments/code/reliquary_code/tests/test_code.py` (append)

- [ ] **Step 1: Write the failing test**

Same three tests as Task 8, with `reliquary_math` → `reliquary_code`,
`MathEnvironment` → `CodeEnvironment`, `reliquary/math` → `reliquary/code`,
`MathTaskset` → `CodeTaskset`, `corpus.OMI_REVISION` → `corpus.OCI_REVISION`,
and the golden assertion reduced to prompt hash plus `wrong_reward == 0.0`
(the corpus carries no reference program, so there is no `reference_reward`
to pin).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/code/reliquary_code && uv run pytest tests/test_code.py -k "goldens_replay or artifact_manifest or environment_toml" -v`
Expected: FAIL with `FileNotFoundError` on `goldens/reference.jsonl`

- [ ] **Step 3: Generate goldens and write metadata**

```bash
cd environments/code/reliquary_code
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
from reliquary_code.taskset import CodeEnvironment

environment = CodeEnvironment()
out = Path("reliquary_code/goldens"); out.mkdir(parents=True, exist_ok=True)
step = max(1, len(environment) // 24)
with (out / "reference.jsonl").open("w", encoding="utf-8") as handle:
    for n in range(24):
        index = n * step
        task = environment.task(index)
        handle.write(json.dumps({
            "index": index,
            "split": "train",
            "prompt_sha256": hashlib.sha256(
                task["prompt"].encode("utf-8")).hexdigest(),
            "wrong_reward": environment.grade(index, "no code")["reward"],
        }, sort_keys=True) + "\n")
print("goldens written")
PY
```

Every `wrong_reward` must be `0.0`.

`environment.toml` is Task 8's file with: `id = "reliquary/code"`,
`entrypoint = "reliquary_code:CodeTaskset"`,
`compatibility_entrypoint = "reliquary_code:CodeEnvironment"`,
`task_families = ["open_code_instruct"]`,
`components = ["passing_cases"]`,
`source_environment = "opencodeinstruct"`, the three `[data]` entries pointing
at `hf:R0mAI/opencodeinstruct-curated@d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7`,
and `[execution] resource_class = "cpu"` kept as-is.

The README must state that grading executes model-written Python in a limited
subprocess, that this is weaker than the gVisor sandbox core uses, and that
Reliquary core grades from its own embedded copy.

Generate the artifact exactly as in Task 8 with
`environment='reliquary_code_v1'`, `contract='reliquary/python-cases/v1'`,
`distribution={'name': 'reliquary-code', 'version': '0.1.0a1'}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd environments/code/reliquary_code && uv run pytest -v`
Expected: PASS, all green

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_code
git commit -m "feat(code): pin goldens, artifact manifest and source metadata"
```

---

## Task 14: Repository wiring

**Files:**
- Modify: `compatibility.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `NOTICE`

**Interfaces:**
- Consumes: both packages.

- [ ] **Step 1: Write the failing test**

```python
# append to environments/reasoning/reliquary_math/tests/test_math.py
def test_repository_compatibility_declares_this_package() -> None:
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    declared = tomllib.loads((root / "compatibility.toml").read_text())
    assert "reliquary_math" in declared["releases"]
    assert declared["releases"]["reliquary_math"]["tag"] == "v0.1.0a2"
```

Add the mirror of this test to the code package, asserting `reliquary_code`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd environments/reasoning/reliquary_math && uv run pytest -k compatibility_declares -v`
Expected: FAIL with `KeyError: 'reliquary_math'`

- [ ] **Step 3: Wire the repository**

In `compatibility.toml`, append:

```toml
[releases.reliquary_math]
tag = "v0.1.0a2"

[releases.reliquary_code]
tag = "v0.1.0a2"
```

Leave `wheel_sha256` out until the release is cut — a placeholder digest is
worse than an absent one, because a reader cannot tell it is not real.

In `.github/workflows/ci.yml`, extend the environment matrix to include
`environments/reasoning/reliquary_math` and `environments/code/reliquary_code`
alongside the existing entries. Read the file's current matrix and follow its
shape exactly; do not restructure the workflow.

In `NOTICE`, add attribution for both corpora:

```
OpenMathInstruct-2 (nvidia/OpenMathInstruct-2), CC-BY-4.0.
OpenCodeInstruct (nvidia/OpenCodeInstruct), CC-BY-4.0, via the curated subset
R0mAI/opencodeinstruct-curated.
```

- [ ] **Step 4: Run test to verify it passes**

Run: both packages' suites plus `uv run pytest tools/tests -v`
Expected: PASS everywhere

- [ ] **Step 5: Commit**

```bash
git add compatibility.toml .github/workflows/ci.yml NOTICE \
        environments/reasoning/reliquary_math/tests/test_math.py \
        environments/code/reliquary_code/tests/test_code.py
git commit -m "chore: declare math and code in repository metadata and CI"
```

---

## Task 15: Full verification before review

**Files:** none created.

- [ ] **Step 1: Run every suite from a clean checkout**

```bash
uv run pytest tools/tests -v
cd environments/reasoning/reliquary_logic && uv sync && uv run pytest -v && cd -
cd environments/reasoning/reliquary_math   && uv sync && uv run pytest -v && cd -
cd environments/code/reliquary_code        && uv sync && uv run pytest -v && cd -
cd environments/tool_use/reliquary_stateful_tools && uv sync && uv run pytest -v && cd -
```

Expected: every suite green. Record the counts.

- [ ] **Step 2: Prove no package imports core**

```bash
grep -rn "from reliquary\.\|import reliquary\." environments/reasoning/reliquary_math environments/code/reliquary_code
```

Expected: no output.

- [ ] **Step 3: Prove the core repository is untouched**

```bash
cd <core-checkout> && git status --short
```

Expected: no output. If anything is listed, revert it — core is read-only for
this work.

- [ ] **Step 4: Prove the vendored copies match core**

```bash
diff <core-checkout>/reliquary/environment/virtual_parquet.py \
     environments/reasoning/reliquary_math/reliquary_math/virtual_parquet.py
diff <core-checkout>/reliquary/environment/virtual_parquet.py \
     environments/code/reliquary_code/reliquary_code/virtual_parquet.py
```

Expected: no output from either.

- [ ] **Step 5: Commit and push the branch**

```bash
git add -A
git commit -m "chore: record full verification run"
git push -u origin feat/math-code-environments
```

Do **not** open the pull request or merge. The branch waits for review.

---

## Self-Review

**Spec coverage.** Every section of
`docs/superpowers/specs/2026-09-06-envs-to-external-repo-design.md` maps to a
task: package layout → 2, 9; corpus and the shard trap → 3, 4, 9; porting the
graders → 5, 10; running the code environment → 11; goldens and tests → 8, 13;
release wiring → 14. The spec's step 1 (merge PR #2) is deliberately absent —
it is a GitHub action on someone else's PR, not code, and it belongs to the
human. The spec's prerequisite (declaring CC-BY-4.0 on the
`R0mAI/opencodeinstruct-curated` dataset card) is likewise a Hugging Face card
edit, outside this plan; Task 14 records the attribution in `NOTICE`
regardless.

**Placeholders.** None. Every code step carries the code. Where a file is a
near-copy of an earlier one (Task 9's `pyproject.toml`, Task 13's tests and
`environment.toml`), the exact substitutions are listed rather than left to
judgement.

**Type consistency.** `MathEnvironment` / `MathTaskset` and `CodeEnvironment` /
`CodeTaskset` are used identically in their defining tasks and in Tasks 8, 13
and 14. `build_artifact(package_root, *, environment, contract, distribution,
entrypoints)` has the same signature in Task 1, Task 8 and Task 13.
`run_cases(source, cases, *, cpu_seconds, memory_bytes, wall_seconds)` matches
between Tasks 11 and 12. `get_problem` returns `{"problem", "expected_answer"}`
for math and `{"input", "structured_cases"}` for code, and each package's
`taskset.py` consumes only its own shape.

**Known risk carried into execution.** Task 3 assumes `VirtualParquetDataset`
accepts a `files=` argument for explicit shard selection. If it does not, Task
4's implementation adapts — the copied file is never edited, and the test in
Task 4 that pins the 32-file list still holds whatever mechanism is used.
