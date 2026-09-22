# reliquary-swe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `reliquary-swe`, a containerised software-engineering environment where a policy repairs a real repository through a shell and is rewarded only if its patch makes the repository's own tests pass in a container it never touched.

**Architecture:** A Verifiers v1 taskset plus a small `Env`. The agent works in a Docker container with no network, driven by a harness selected from the ones `verifiers` already ships. At the end of the episode the agent's `git diff` is captured into the artifacts directory. A custom `Env.finalize` then provisions a **second, pristine container** from the same image, restores that diff into it, applies the instance's own test patch, runs the fail-to-pass and pass-to-pass test sets, and records the reward onto the solver's trace.

**Tech Stack:** Python 3.12, `verifiers` v1 (pinned commit), `swebench` (image keys and per-repo log parsing only), Docker, `uv`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-22-reliquary-swe-env-design.md`

## What we are not writing, and why

The spec's section 3 lists the machinery `verifiers` already provides. Re-reading it before each task will save real work. Three points bear repeating because they shrank this plan substantially after a first draft assumed otherwise:

- **No harness.** `verifiers.v1.harnesses` ships `bash`, `mini_swe_agent`, `claude_code`, `codex`, `terminus_2` and more. The harness is chosen by configuration (`--env.agent.harness.*`). Writing a bash toolset would be redundant *and* would forfeit the cross-harness training the spec's section 5 wants.
- **No patch plumbing.** `vf.capture_patch`, `vf.resolve_head` and `vf.snapshot_untracked` exist and are documented for exactly this: their own docstrings describe "tasks graded in a second sandbox". They already solve the traps — untracked files the image shipped, a stale `index.lock`, a patch cap, and telling a dead sandbox apart from a failing git.
- **No second-box scaffolding.** `verifiers.v1.tasksets.harbor.env.HarborEnv` is a working separate-box grader. Read it before Task 5; our `SweEnv` is the same shape with a different verifier.

## Scope of this plan

Delivers the environment end-to-end on **SWE-bench Verified** (500 instances) only.

The spec also names SWE-smith as the training corpus. That is deliberately not here: once the loop works, adding a corpus is a change to `corpus.py` and a second data pin, not a change to the task, the grader or the env. Doing it now means pulling a far larger image set before anything is proven to run. Follow-on plan.

## Global Constraints

Copied verbatim from the spec and from this repository's existing packages. Every task's requirements implicitly include this section.

- **All repository-bound text is English.** Code, comments, docstrings, README, commit messages.
- **Python:** `>=3.12,<3.13`.
- **verifiers:** `>=0.3.1,<0.4`, API `v1`, pinned commit `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf` — `scripts/check_verifiers_pin.py` enforces it.
- **`environment.toml` declares no `compatibility_entrypoint`.** Verifiers-only, a conscious deviation from the repository README, recorded in the spec.
- **`network_allow = []` on every task.** An empty concrete list replaces the default `["*"]` wildcard.
- **Never grade in the agent's container.** Any change that reads a test result out of the agent's box is a defect however convenient.
- **An unreachable grading box is never reward 0.** Infrastructure failure must raise, not score. A zero that means "our Docker daemon died" teaches the policy something false.
- **Branch:** `feat/reliquary-swe`. Never commit to `main`, never `git push` without asking.
- **Commands:** `uv sync --locked`, then `uv run --no-sync pytest`.

## File Structure

All paths relative to `environments/code/reliquary_swe/`.

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, pins, taskset registration. |
| `environment.toml` | Declared contract: tier, resource class, network, budgets, `[policy]`. |
| `README.md` | What it is, and why grading happens in a second container. |
| `reliquary_swe/__init__.py` | Public exports: `SweTaskset`, `SweTask`, `SweEnv`. |
| `reliquary_swe/corpus.py` | Dataset rows only. Never imports Docker or a verifiers runtime. |
| `reliquary_swe/swe_adapter.py` | The only module importing `swebench`. Image key and log parsing, behind names we own. |
| `reliquary_swe/taskset.py` | `SweData`, `SweTask`, `SweTaskset`. Setup and patch capture. |
| `reliquary_swe/grading.py` | Inside the grading box: apply, restore tests, run, parse, score. Takes a `Runtime`, provisions nothing. |
| `reliquary_swe/env.py` | `SweEnv`. Provisions the grading box, restores artifacts, calls `grading`, records the reward. |
| `examples/prime_rl/` | Runnable example; CI checks it matches the declared reasoning mode. |
| `tests/test_corpus.py` | Corpus shape and stable keys. No Docker. |
| `tests/test_adapter.py` | Log parsing against recorded fixtures. No Docker. |
| `tests/test_episode.py` | Container preparation, isolation, patch capture. Docker-marked. |
| `tests/test_goldens.py` | The four goldens. Docker-marked. |

`grading.py` receives a live `Runtime` and never creates one. `env.py` owns provisioning. Keeping that seam means the grader can be tested against any box, and the retry policy lives in one place.

---

### Task 1: Package skeleton and corpus

**Files:**
- Create: `pyproject.toml`, `reliquary_swe/__init__.py`, `reliquary_swe/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `corpus.SweRow`, a frozen slotted dataclass with fields `instance_id: str`, `repo: str`, `base_commit: str`, `problem_statement: str`, `fail_to_pass: tuple[str, ...]`, `pass_to_pass: tuple[str, ...]`, `gold_patch: str`, `test_patch: str`; `corpus.load_rows(split: str = "eval") -> tuple[SweRow, ...]`; `corpus.SPLITS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
from reliquary_swe import corpus


def test_verified_split_has_the_published_instance_count():
    assert len(corpus.load_rows("eval")) == 500


def test_rows_carry_everything_grading_needs():
    row = corpus.load_rows("eval")[0]
    assert row.instance_id
    assert row.repo
    assert len(row.base_commit) == 40
    assert row.problem_statement.strip()
    # An instance with no fail-to-pass test cannot express failure, so it
    # cannot express success either.
    assert row.fail_to_pass
    assert row.gold_patch.strip()
    # The test patch is what the grading box applies to restore the tests the
    # agent may have edited. Without it there is nothing to grade against.
    assert row.test_patch.strip()


def test_instance_ids_are_unique_and_usable_as_task_keys():
    ids = [row.instance_id for row in corpus.load_rows("eval")]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run --no-sync pytest tests/test_corpus.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'reliquary_swe'`.

- [ ] **Step 3: Write `pyproject.toml`**

Copy the shape of `environments/code/reliquary_code/pyproject.toml` and change the identity. Add `datasets` and `swebench` to dependencies; pin `verifiers` exactly as the siblings do.

- [ ] **Step 4: Write `corpus.py`**

```python
"""SWE-bench rows, and nothing that needs a container.

Deliberately free of Docker and of verifiers runtimes: corpus questions — how
many instances, which repositories, what a task id looks like — must be
answerable on a machine that cannot host images.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass

from datasets import load_dataset

SPLITS = ("eval",)

# Pinned by revision, as every other environment here pins its data. An
# unpinned corpus makes two runs incomparable for a reason that never shows up
# in the metrics. The SHA is filled in by Step 6.
_SOURCES = {
    "eval": ("princeton-nlp/SWE-bench_Verified", "test", ""),
}


@dataclass(frozen=True, slots=True)
class SweRow:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    test_patch: str


def _tests(raw: object) -> tuple[str, ...]:
    """SWE-bench stores the two test lists as JSON-encoded strings."""
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(raw or ())


@functools.lru_cache(maxsize=None)
def load_rows(split: str = "eval") -> tuple[SweRow, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}; expected one of {SPLITS}")
    name, hf_split, revision = _SOURCES[split]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    return tuple(
        SweRow(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            fail_to_pass=_tests(row["FAIL_TO_PASS"]),
            pass_to_pass=_tests(row["PASS_TO_PASS"]),
            gold_patch=row["patch"],
            test_patch=row["test_patch"],
        )
        for row in dataset
    )
```

Write `__init__.py` exporting `corpus`.

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `uv run --no-sync pytest tests/test_corpus.py -v`
Expected: 3 passed.

If the count assertion fails, do **not** edit the assertion to match what loaded. A different count means a different dataset revision, which is exactly what the test exists to catch.

- [ ] **Step 6: Pin the dataset revision**

```bash
uv run --no-sync python -c "
from huggingface_hub import dataset_info
print(dataset_info('princeton-nlp/SWE-bench_Verified').sha)"
```

Put the printed SHA into `_SOURCES` in place of the empty string. Re-run the tests.

- [ ] **Step 7: Commit**

```bash
git add environments/code/reliquary_swe
git commit -m "feat(swe): load SWE-bench Verified rows behind a container-free corpus module"
```

---

### Task 2: The swebench adapter

Per-repository test output has no generic shape — each repository uses a different framework and prints a different format. `swebench` carries a parser per repository and the naming scheme for the prebuilt images. This task wraps both behind names we own, so no other module imports `swebench` and a version bump touches one file.

**Files:**
- Create: `reliquary_swe/swe_adapter.py`, `tests/fixtures/`
- Test: `tests/test_adapter.py`

**Interfaces:**
- Consumes: `corpus.SweRow`.
- Produces: `swe_adapter.image_for(row: SweRow) -> str`; `swe_adapter.test_command(row: SweRow, tests: tuple[str, ...]) -> list[str]`; `swe_adapter.parse_results(row: SweRow, stdout: str) -> dict[str, str]` mapping a test name to one of `"PASSED"`, `"FAILED"`, `"ERROR"`, `"SKIPPED"`.

- [ ] **Step 1: Resolve the exact swebench symbols**

`swebench`'s internal module paths move between versions, and this plan does not name them from memory. Resolve them against the installed version:

```bash
uv run --no-sync python -c "
import swebench, pathlib
print('version:', getattr(swebench, '__version__', 'unknown'))
root = pathlib.Path(swebench.__file__).parent
for p in sorted(root.rglob('*.py')):
    text = p.read_text(errors='replace')
    hits = [k for k in ('MAP_REPO_VERSION_TO_SPECS', 'MAP_REPO_TO_PARSER',
                        'get_eval_report', 'make_test_spec', 'instance_image_key',
                        'get_test_directives') if k in text]
    if hits:
        print(p.relative_to(root), hits)
"
```

Record the resolved import paths as a comment at the top of `swe_adapter.py`. They are the only coupling this package has to `swebench` internals.

- [ ] **Step 2: Record parser fixtures**

Capture two real test-run outputs — one where a known test passes, one where a known test fails — from one repository in the corpus. Save as `tests/fixtures/pass.txt` and `tests/fixtures/fail.txt`. Note the instance id and the exact test name for each; Step 3's test must name them as literals.

- [ ] **Step 3: Write the failing test**

Create `tests/test_adapter.py`. Replace `INSTANCE`, `PASSING_TEST` and `FAILING_TEST` with the literals recorded in Step 2 — the committed test names them directly, it does not read them from a variable.

```python
from pathlib import Path

from reliquary_swe import corpus, swe_adapter

FIXTURES = Path(__file__).parent / "fixtures"

INSTANCE = "<instance id recorded in Step 2>"
PASSING_TEST = "<test name that passes in pass.txt>"
FAILING_TEST = "<test name that fails in fail.txt>"


def _row(instance_id: str) -> corpus.SweRow:
    for row in corpus.load_rows("eval"):
        if row.instance_id == instance_id:
            return row
    raise AssertionError(f"{instance_id} is not in the corpus")


def test_image_reference_is_stable():
    row = _row(INSTANCE)
    assert swe_adapter.image_for(row) == swe_adapter.image_for(row)
    assert swe_adapter.image_for(row).strip()


def test_parses_a_passing_run():
    results = swe_adapter.parse_results(_row(INSTANCE), (FIXTURES / "pass.txt").read_text())
    assert results[PASSING_TEST] == "PASSED"


def test_parses_a_failing_run():
    results = swe_adapter.parse_results(_row(INSTANCE), (FIXTURES / "fail.txt").read_text())
    assert results[FAILING_TEST] == "FAILED"


def test_unparseable_output_yields_no_results_rather_than_raising():
    assert swe_adapter.parse_results(_row(INSTANCE), "") == {}


def test_the_test_command_names_the_tests_it_was_given():
    argv = swe_adapter.test_command(_row(INSTANCE), (PASSING_TEST,))
    assert any(PASSING_TEST.split("::")[0] in part for part in argv)
```

- [ ] **Step 4: Run it to make sure it fails**

Run: `uv run --no-sync pytest tests/test_adapter.py -v`
Expected: FAIL, `ImportError: cannot import name 'swe_adapter'`.

- [ ] **Step 5: Write `swe_adapter.py`**

Use the imports resolved in Step 1. The module's shape:

```python
"""The only module here that imports `swebench`.

Per-repository test output has no generic shape: each repository uses a
different framework and prints a different format. `swebench` already carries a
parser per repository and the naming scheme for the prebuilt evaluation images.
Wrapping both here means a version bump touches one file, and the rest of this
package depends on names we control.

Resolved swebench imports for the pinned version (see the implementation plan,
Task 2 Step 1, for how these were found):
    <record them here>
"""

from __future__ import annotations

from reliquary_swe.corpus import SweRow


def image_for(row: SweRow) -> str:
    """The prebuilt evaluation image holding this instance's repository."""


def test_command(row: SweRow, tests: tuple[str, ...]) -> list[str]:
    """An argv running exactly `tests` inside the instance's image."""


def parse_results(row: SweRow, stdout: str) -> dict[str, str]:
    """Map each reported test name to PASSED / FAILED / ERROR / SKIPPED.

    Never raises. Unparseable output returns an empty map, which grading reads
    as "no test reported a pass" — the correct reward for a run whose output we
    cannot trust.
    """
```

Fill each body from the resolved symbols. No function may be left without an implementation.

- [ ] **Step 6: Run the tests and make sure they pass**

Run: `uv run --no-sync pytest tests/test_adapter.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add environments/code/reliquary_swe
git commit -m "feat(swe): wrap swebench image keys and log parsing behind an owned interface"
```

---

### Task 3: Tasks, and a container prepared against leaks

**Files:**
- Create: `reliquary_swe/taskset.py`
- Modify: `reliquary_swe/__init__.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Consumes: `corpus.load_rows`, `swe_adapter.image_for`.
- Produces: `taskset.SweData` extending `vf.TaskData` with `instance_id: str`, `repo: str`, `base_commit: str`, `fail_to_pass: tuple[str, ...]`, `pass_to_pass: tuple[str, ...]`, `gold_patch: str`, `test_patch: str`, `split: str`; `taskset.SweTask` with `NEEDS_CONTAINER = True`, a `key` property returning `instance_id`, `setup`, and `finalize`; `taskset.SweTaskset`; `taskset.WORKDIR`; `taskset.PATCH_PATH`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_episode.py`:

```python
import pytest
import verifiers.v1 as vf

from reliquary_swe.taskset import SweTask

docker = pytest.mark.docker


def _first_task() -> SweTask:
    config = vf.taskset_config_type("reliquary-swe")
    return next(iter(vf.load_taskset(config(id="reliquary-swe")).head(1)))


def test_task_key_is_the_instance_id_not_a_content_hash():
    # Instance ids are durable across dataset revisions; content hashes are
    # not, and a run cannot be compared with an earlier one if its keys moved.
    task = _first_task()
    assert task.key == task.data.instance_id


def test_every_task_refuses_the_network():
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe")).head(5):
        assert task.data.network_allow == []


def test_every_task_names_an_image_and_a_workdir():
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe")).head(5):
        assert task.data.image
        assert task.data.workdir


@docker
async def test_setup_leaves_the_repository_at_the_base_commit(runtime):
    task = _first_task()
    await task.setup(vf.Trace(), runtime)
    head = await runtime.run(["git", "rev-parse", "HEAD"], {})
    assert head.stdout.strip() == task.data.base_commit


@docker
async def test_setup_removes_history_after_the_base_commit(runtime):
    # The published fix lives in a later commit. Leaving it reachable turns
    # the repair task into a lookup.
    task = _first_task()
    await task.setup(vf.Trace(), runtime)
    later = await runtime.run(
        ["sh", "-c", f"git log --oneline {task.data.base_commit}..HEAD 2>/dev/null | wc -l"],
        {},
    )
    assert later.stdout.strip() == "0"


@docker
async def test_the_container_cannot_reach_the_network(runtime):
    task = _first_task()
    await task.setup(vf.Trace(), runtime)
    result = await runtime.run(
        ["sh", "-c", "curl -s -m 5 https://raw.githubusercontent.com || echo BLOCKED"],
        {},
    )
    assert "BLOCKED" in result.stdout
```

- [ ] **Step 2: Run the non-Docker tests to make sure they fail**

Run: `uv run --no-sync pytest tests/test_episode.py -v -m "not docker"`
Expected: FAIL, `ModuleNotFoundError: No module named 'reliquary_swe.taskset'`.

- [ ] **Step 3: Write `taskset.py`**

```python
"""SWE-bench as one Verifiers taskset.

The agent gets a repository at its base commit, a bug report, and whatever
shell its harness provides. It gets nothing else: no network, no history past
the base commit, no build output that could name the fix.

This module deliberately defines no tools. The harness supplies them, and
`verifiers` already ships several (`bash`, `mini_swe_agent`, `codex`,
`claude_code`, `terminus_2`). Training across more than one is how task-solving
strategy stops being welded to a single harness's quirks.
"""

from __future__ import annotations

from collections.abc import Iterator

import verifiers.v1 as vf

from reliquary_swe import corpus, swe_adapter

WORKDIR = "/testbed"
PATCH_PATH = f"{vf.ARTIFACTS_DIR}/patch.diff"

PROMPT = """\
You are working in a checked-out repository at {workdir}.

{problem_statement}

Fix the issue by editing the repository's source. Do not edit the tests.
"""

# Everything an agent could read the published fix out of. Build logs and
# verifier output are produced while the image is built with the reference
# patch applied, so they can name the very lines under test. Third-party
# dependencies are kept: the repository must still build offline.
_CLEANUP = " ; ".join(
    [
        "set -e",
        'git -C /testbed reset --hard "$BASE_COMMIT"',
        'git -C /testbed checkout -q "$BASE_COMMIT"',
        # Drop every ref that could reach a later commit, then expire the
        # reflog so neither `git log --all` nor `git fsck` finds one.
        "git -C /testbed for-each-ref --format='%(refname)' "
        "| xargs -r -n1 git -C /testbed update-ref -d",
        "git -C /testbed reflog expire --expire=now --all",
        "git -C /testbed gc --prune=now --quiet",
        "rm -rf /testbed/.git/logs",
        "rm -f /testbed/*.orig /testbed/*.rej",
        "rm -f /*.patch /home/*.patch /tmp/*.patch /root/*.patch",
        "find /testbed -name '__pycache__' -type d -prune -exec rm -rf {} + || true",
        "rm -rf /root/.cache/pip /tmp/build",
    ]
)


class SweData(vf.TaskData):
    instance_id: str
    repo: str
    base_commit: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    test_patch: str
    split: str


class SweTask(vf.Task[SweData]):
    NEEDS_CONTAINER = True

    # Keyed by id(runtime); set in setup, read in finalize. Host memory only:
    # a base commit kept inside the box is one the agent can rewrite.
    _heads: dict[int, str] = {}
    _untracked: dict[int, list[str]] = {}

    @property
    def key(self) -> str:
        return self.data.instance_id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        result = await runtime.run(
            ["sh", "-c", _CLEANUP], {"BASE_COMMIT": self.data.base_commit}
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id} "
                f"(exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        self._heads[id(runtime)] = await vf.resolve_head(runtime)
        self._untracked[id(runtime)] = await vf.snapshot_untracked(runtime)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Snapshot the agent's diff while its box is still alive.

        Diffed against the SHA recorded at setup rather than bare HEAD, so
        commits the agent made are inside it. Edits to test files are captured
        on purpose: grading discards them, and keeping them in the trace is
        what makes reward hacking visible afterwards.
        """
        await vf.capture_patch(
            trace,
            runtime,
            base_commit=self._heads.pop(id(runtime), ""),
            ignore=self._untracked.pop(id(runtime), []),
            write_path=PATCH_PATH,
        )


class SweTaskset(vf.Taskset[SweTask, vf.TasksetConfig]):
    def load(self) -> Iterator[SweTask]:
        for index, row in enumerate(corpus.load_rows(self.config.split)):
            yield SweTask(
                SweData(
                    idx=index,
                    name=row.instance_id,
                    prompt=PROMPT.format(
                        workdir=WORKDIR, problem_statement=row.problem_statement
                    ),
                    image=swe_adapter.image_for(row),
                    workdir=WORKDIR,
                    network_allow=[],
                    instance_id=row.instance_id,
                    repo=row.repo,
                    base_commit=row.base_commit,
                    fail_to_pass=row.fail_to_pass,
                    pass_to_pass=row.pass_to_pass,
                    gold_patch=row.gold_patch,
                    test_patch=row.test_patch,
                    split=self.config.split,
                ),
                self.config.task,
            )
```

Confirm `vf.Taskset`'s config exposes `split`; if it does not, define a `SweTasksetConfig(vf.TasksetConfig)` carrying `split: str = "eval"` and parameterise `SweTaskset` with it, following whichever sibling package already does this.

- [ ] **Step 4: Run the non-Docker tests**

Run: `uv run --no-sync pytest tests/test_episode.py -v -m "not docker"`
Expected: 3 passed.

- [ ] **Step 5: Run the Docker tests**

Run: `uv run --no-sync pytest tests/test_episode.py -v -m docker`
Expected: 3 passed.

Watch the network test. If `curl` reaches the internet, the allowlist is not being applied and every anti-leak claim in the spec is false. Stop and fix it; do not mark it xfail.

- [ ] **Step 6: Add the patch-capture tests**

Append to `tests/test_episode.py`:

```python
@docker
async def test_finalize_captures_an_edit_the_agent_made(runtime):
    task = _first_task()
    trace = vf.Trace()
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
    # or `git apply` fails in a fresh container of that same image — which is
    # exactly what the grading box is.
    task = _first_task()
    trace = vf.Trace()
    await runtime.run(["sh", "-c", "echo shipped > /testbed/shipped.txt"], {})
    await task.setup(trace, runtime)
    await task.finalize(trace, runtime)
    assert "shipped.txt" not in trace.info["patch"]


@docker
async def test_the_patch_is_written_where_artifact_collection_finds_it(runtime):
    from reliquary_swe.taskset import PATCH_PATH

    task = _first_task()
    trace = vf.Trace()
    await task.setup(trace, runtime)
    await task.finalize(trace, runtime)
    assert (await runtime.run(["ls", PATCH_PATH], {})).exit_code == 0
```

Run: `uv run --no-sync pytest tests/test_episode.py -v -m docker`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add environments/code/reliquary_swe
git commit -m "feat(swe): prepare an isolated container and capture the agent's diff"
```

---

### Task 4: The grader, against a live box

`grading.py` is given a running `Runtime` and never creates one. That seam is what lets the grader be tested directly, and keeps provisioning and retry policy in Task 5's `env.py` alone.

**Files:**
- Create: `reliquary_swe/grading.py`
- Test: `tests/test_goldens.py`

**Interfaces:**
- Consumes: `swe_adapter.test_command`, `swe_adapter.parse_results`, `taskset.SweData`.
- Produces: `grading.Report`, a frozen slotted dataclass with `reward: float`, `applied: bool`, `fail_to_pass_passed: int`, `pass_to_pass_passed: int`, `results: dict[str, str]`; and `async grading.grade(runtime: vf.Runtime, data: SweData, patch: str) -> Report`.

- [ ] **Step 1: Write the failing goldens**

Create `tests/test_goldens.py`. Replace `GOLDEN` with one literal instance id from the corpus — pick a small, fast repository so the goldens stay quick.

```python
import pytest
import verifiers.v1 as vf

from reliquary_swe import grading

docker = pytest.mark.docker

GOLDEN = "<instance id chosen for the goldens>"

# A patch that tampers with collection rather than fixing anything. It applies
# cleanly, which is the point: the zero must come from the tampering being
# discarded, not from `git apply` refusing it.
TEST_EDITING_PATCH = """\
diff --git a/conftest.py b/conftest.py
new file mode 100644
--- /dev/null
+++ b/conftest.py
@@ -0,0 +1,3 @@
+def pytest_collection_modifyitems(items):
+    for item in items:
+        item.add_marker("skip")
"""


def _data(instance_id: str):
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe")):
        if task.data.instance_id == instance_id:
            return task.data
    raise AssertionError(f"{instance_id} is not in the corpus")


@docker
async def test_the_reference_patch_scores_one(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, data.gold_patch)
    assert report.applied
    assert report.reward == 1.0


@docker
async def test_an_empty_patch_scores_zero(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    assert report.reward == 0.0


@docker
async def test_a_patch_that_edits_the_tests_scores_zero(grading_runtime):
    # The golden that justifies the whole design. `applied` must be True:
    # otherwise the zero proves only that git rejected the patch, and the test
    # is worthless.
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, TEST_EDITING_PATCH)
    assert report.applied is True
    assert report.reward == 0.0


@docker
async def test_fail_to_pass_really_fails_before_any_patch(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    for name in data.fail_to_pass:
        assert report.results.get(name) != "PASSED"


@docker
async def test_pass_to_pass_really_passes_before_any_patch(grading_runtime):
    # If these do not pass on an untouched checkout, the instance is broken or
    # flaky and no reward computed from it means anything.
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    for name in data.pass_to_pass:
        assert report.results.get(name) == "PASSED"
```

Add a `grading_runtime` fixture in `tests/conftest.py` that provisions a Docker runtime from the golden instance's image and tears it down after each test, so every test gets a pristine box.

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run --no-sync pytest tests/test_goldens.py -v -m docker`
Expected: FAIL, `ImportError: cannot import name 'grading'`.

- [ ] **Step 3: Write `grading.py`**

```python
"""Score a patch inside a box the agent never touched.

Order matters and is the whole design. The agent's patch is applied first, then
the instance's own test patch is applied on top of a checkout of the test files
as they exist at the base commit. So a `conftest.py` that skips everything, a
monkeypatched framework, or an edited test file is overwritten before a single
test runs — it can never be rewarded, only recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import verifiers.v1 as vf

from reliquary_swe import swe_adapter
from reliquary_swe.corpus import SweRow

_DIFF_TARGET = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Report:
    reward: float
    applied: bool
    fail_to_pass_passed: int
    pass_to_pass_passed: int
    results: dict[str, str] = field(default_factory=dict)


def _paths_touched_by(patch: str) -> list[str]:
    """The files a unified diff writes to."""
    return [match.group(1) for match in _DIFF_TARGET.finditer(patch)]


def _row_for(data) -> SweRow:
    return SweRow(
        instance_id=data.instance_id,
        repo=data.repo,
        base_commit=data.base_commit,
        problem_statement="",
        fail_to_pass=data.fail_to_pass,
        pass_to_pass=data.pass_to_pass,
        gold_patch=data.gold_patch,
        test_patch=data.test_patch,
    )


async def grade(runtime: vf.Runtime, data, patch: str) -> Report:
    """Apply `patch` in `runtime`, restore the tests, run them, and score.

    `runtime` must be a freshly provisioned box from this instance's image.
    Passing the agent's own box would defeat the entire point.
    """
    await runtime.run(["git", "-C", "/testbed", "checkout", "-q", data.base_commit], {})

    applied = True
    if patch.strip():
        await runtime.write("/tmp/agent.diff", patch.encode())
        result = await runtime.run(
            ["git", "-C", "/testbed", "apply", "-v", "/tmp/agent.diff"], {}
        )
        applied = result.exit_code == 0
        if not applied:
            # A patch that will not apply changed nothing, which is reward 0 —
            # a real outcome, not an infrastructure failure.
            return Report(0.0, False, 0, 0, {})

    # Restore only the files the test patch touches, then apply it. Source
    # files the agent edited are untouched by this; test files it edited are
    # discarded.
    test_paths = _paths_touched_by(data.test_patch)
    if test_paths:
        await runtime.run(
            ["git", "-C", "/testbed", "checkout", data.base_commit, "--", *test_paths],
            {},
        )
    await runtime.write("/tmp/tests.diff", data.test_patch.encode())
    await runtime.run(["git", "-C", "/testbed", "apply", "-v", "/tmp/tests.diff"], {})

    names = tuple(data.fail_to_pass) + tuple(data.pass_to_pass)
    run = await runtime.run(swe_adapter.test_command(_row_for(data), names), {})
    results = swe_adapter.parse_results(_row_for(data), run.stdout or "")

    f2p = sum(1 for name in data.fail_to_pass if results.get(name) == "PASSED")
    p2p = sum(1 for name in data.pass_to_pass if results.get(name) == "PASSED")
    resolved = f2p == len(data.fail_to_pass) and p2p == len(data.pass_to_pass)
    return Report(1.0 if resolved else 0.0, applied, f2p, p2p, results)
```

- [ ] **Step 4: Run the goldens and make sure they pass**

Run: `uv run --no-sync pytest tests/test_goldens.py -v -m docker`
Expected: 5 passed.

If `test_the_reference_patch_scores_one` fails, the most likely cause is the test-restoration step reverting more than the test files. Check `_paths_touched_by` against the instance's `test_patch` before suspecting anything else.

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_swe
git commit -m "feat(swe): score a patch against restored tests in a given box"
```

---

### Task 5: The env that provisions the grading box

Modelled on `verifiers.v1.tasksets.harbor.env.HarborEnv`. Read it before writing this.

**Files:**
- Create: `reliquary_swe/env.py`
- Modify: `reliquary_swe/__init__.py`, `pyproject.toml` (register the env)
- Test: `tests/test_goldens.py`

**Interfaces:**
- Consumes: `grading.grade`, `taskset.SweTask`, `taskset.PATCH_PATH`.
- Produces: `env.SweEnv`, a `vf.Env[SweEnvConfig]` whose `finalize` records the reward `"patch_passes_tests"` on the solver's trace; `env.SweEnvConfig` with `agent: vf.AgentConfig`, `grading_runtime: RuntimeConfig | None`, `grading_retries: int = 2`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_goldens.py`:

```python
@docker
async def test_a_gold_rollout_scores_one_end_to_end():
    # The whole loop: agent box, patch capture, a second box, grading.
    config = vf.taskset_config_type("reliquary-swe")
    task = next(iter(vf.load_taskset(config(id="reliquary-swe")).head(1)))
    episode = await run_gold_episode(task)  # helper in tests/conftest.py
    assert episode.traces[0].rewards["patch_passes_tests"] == 1.0


@docker
async def test_an_unreachable_grading_box_raises_rather_than_scoring_zero():
    # A zero that means "our Docker daemon died" teaches the policy something
    # false. Infrastructure failure must fail the episode instead.
    from reliquary_swe.env import SweEnv, SweEnvConfig

    broken = SweEnvConfig(grading_runtime=unreachable_runtime_config())
    with pytest.raises(Exception):
        await SweEnv(broken).finalize(task_with_patch(), episode_with_patch())
```

Write `run_gold_episode`, `unreachable_runtime_config` and `task_with_patch` / `episode_with_patch` as helpers in `tests/conftest.py`. `run_gold_episode` applies the instance's `gold_patch` into the agent box and runs the normal finalize path, so it exercises capture and grading without needing a model.

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run --no-sync pytest tests/test_goldens.py -v -m docker -k end_to_end`
Expected: FAIL, `ImportError: cannot import name 'env'`.

- [ ] **Step 3: Write `env.py`**

```python
"""Grade each rollout in a container the agent never entered.

The same shape as `verifiers.v1.tasksets.harbor.env.HarborEnv`: the solve runs
in the agent's box, then `finalize` provisions a fresh box from the task's own
image, restores the collected patch, grades there, and records the reward onto
the solver's trace.

Infrastructure failures retry and then raise. They never score. A grading box
that could not be reached must not read as reward 0.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

from reliquary_swe import grading
from reliquary_swe.taskset import SweTask

logger = logging.getLogger(__name__)


class SweEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The policy under training; pin `--env.agent.harness.*` to choose which
    harness drives it. Varying it across runs is the point."""
    grading_runtime: RuntimeConfig | None = None
    """Where grading runs. None derives it from the agent's runtime policy."""
    grading_retries: int = Field(2, ge=0)
    """Extra attempts at provisioning-and-grading before the episode fails.
    Grading is deterministic; what these retry is the infrastructure."""


class SweEnv(vf.Env[SweEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, SweTask):
            raise TypeError(f"the swe env runs swe tasks; got {type(task).__name__}")
        await agents.agent.run(task)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if not isinstance(task, SweTask):
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        patch = solution.info.get("patch", "")
        report = await self._grade(task, patch)
        solution.record_reward("patch_passes_tests", report.reward)
        solution.info["swe_report"] = {
            "applied": report.applied,
            "fail_to_pass_passed": report.fail_to_pass_passed,
            "fail_to_pass_total": len(task.data.fail_to_pass),
            "pass_to_pass_passed": report.pass_to_pass_passed,
            "pass_to_pass_total": len(task.data.pass_to_pass),
        }

    async def _grade(self, task: SweTask, patch: str) -> grading.Report:
        base = (
            self.config.grading_runtime
            if self.config.grading_runtime is not None
            else self.config.agent.runtime
        )
        config = resolve_runtime_config(base, task)
        last: Exception | None = None
        for attempt in range(self.config.grading_retries + 1):
            if attempt:
                delay = backoff(attempt - 1)
                logger.warning(
                    "swe grading attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self.config.grading_retries + 1,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                async with AsyncExitStack() as boxes:
                    async with asyncio.timeout(task.data.timeout.scoring):
                        box = await boxes.enter_async_context(
                            provision_runtime(config, env=task.runtime_env())
                        )
                        await box.prepare_setup()
                        await box.prepare_execution([])
                        return await grading.grade(box, task.data, patch)
            except Exception as e:  # noqa: BLE001 - each attempt's failure is retried
                last = e
        assert last is not None
        raise last
```

Register `SweEnv` as the package's env the way the harbor taskset exports `HarborEnv`; check how `[tool.verifiers]` in a sibling `pyproject.toml` declares it.

- [ ] **Step 4: Run the goldens and make sure they pass**

Run: `uv run --no-sync pytest tests/test_goldens.py -v -m docker`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add environments/code/reliquary_swe
git commit -m "feat(swe): grade each rollout in a freshly provisioned container"
```

---

### Task 6: Declared contract, example, and CI

**Files:**
- Create: `environment.toml`, `README.md`, `examples/prime_rl/`
- Modify: repository-root `README.md`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: everything above.
- Produces: a package passing the repository's four CI gates.

- [ ] **Step 1: Write `environment.toml`**

```toml
schema = "reliquary/environment-source/v2"
id = "reliquary/swe"
version = "0.1.0a1"
taskset = "reliquary-swe"
entrypoint = "reliquary_swe:SweTaskset"
license = "MIT"
verification_tier = "container-reverified"
owners = ["Reliquary contributors"]
task_families = ["swe_bench_verified"]

# No `compatibility_entrypoint`. This environment is Verifiers-only: a
# container cannot be replayed deterministically, so a replay surface would be
# a promise the implementation cannot keep. Recorded as a decision in
# docs/superpowers/specs/2026-09-22-reliquary-swe-env-design.md.

[compatibility]
python = ">=3.12,<3.13"
verifiers = ">=0.3.1,<0.4"
verifiers_api = "v1"

[execution]
network = false
secrets = []
resource_class = "container"
# Provisional. Unlike this repository's other budgets, no measurement stands
# behind it yet; the spec lists it as an open question and the first pilot is
# expected to replace it.
max_turns = 40
max_observation_bytes = 16384

[reward]
minimum = 0.0
maximum = 1.0
components = ["patch_passes_tests"]

[policy]
reasoning = "thinking"
reasoning_rationale = """
A repair starts with diagnosis: the agent must read code it has never seen and
locate a fault from a bug report before its first edit, and with a shell as its
only tool there is nowhere else for that reasoning to happen. This is an
argument, not a measurement, and the other environments here earned theirs by
measurement — it must be rewritten with a measured band after the first pilot.
"""
```

- [ ] **Step 2: Write the example and the READMEs**

Copy `environments/code/reliquary_code/examples/prime_rl/` as the starting shape. The example must request `enable_thinking = true` to match the declared mode — `scripts/check_example_matches_policy.py` compares them and fails on drift. Name the renderer explicitly; auto-resolution silently falls back to a renderer with no tool support.

The package README explains what the environment is and, in its own section, why grading happens in a second container. Add a row to the repository-root README table marking this environment Verifiers-only.

- [ ] **Step 3: Run the CI gates locally**

```bash
uv run --no-sync python ../../../scripts/check_verifiers_pin.py
uv run --no-sync python ../../../scripts/check_reasoning_mode.py
uv run --no-sync python ../../../scripts/check_example_matches_policy.py
uv run --no-sync pytest
```

Expected: all four succeed.

- [ ] **Step 4: Run the gold validation under Docker**

```bash
uv run --no-sync validate reliquary-swe -n 3 --only-gold true \
  --runtime.type docker --runtime.allow '[]' --rich false
```

Expected: three gold rollouts, each scoring 1.0. This is the environment exercised exactly as CI exercises its siblings; if it passes, the loop works end to end.

- [ ] **Step 5: Add the package to CI**

Add a job to `.github/workflows/ci.yml` mirroring the existing per-package jobs, including the `validate` step from Step 4.

- [ ] **Step 6: Commit**

```bash
git add environments/code/reliquary_swe README.md .github/workflows/ci.yml
git commit -m "feat(swe): declare the environment contract and wire it into CI"
```

---

## What this plan deliberately does not do

Listed so a reviewer does not read them as oversights. Each is tracked in the spec.

- **Supervision quality is not measured.** The false-positive and false-negative rates of these test suites as graders are unknown. Establishing them — repeated execution to find flaky suites, and auditing rollouts whose reward disagrees with an independent reading of the patch — is required before a real training run, and is its own piece of work.
- **The turn and token budgets are not derived.** `max_turns = 40` is a starting point, not a measurement.
- **SWE-smith is not wired in.** See Scope.
- **Only one harness is exercised.** The taskset is harness-agnostic by construction, but this plan validates it against one. Training across several is a follow-on.
- **No shared container base package is extracted.** That decision belongs to the second containerised environment.
