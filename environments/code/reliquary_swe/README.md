# reliquary-swe

SWE-bench Verified repair tasks: a policy is given a real repository at a
real commit, a bug report, and a shell, and is rewarded for producing a
patch that makes the repository's own tests pass.

This module defines no tools of its own. A harness — `bash`,
`mini_swe_agent`, or any other `verifiers.v1.harnesses` entry — supplies the
shell; this package supplies the container, the prepared repository, and the
reward. That is deliberate: training across more than one harness is how
task-solving strategy stops being a property of one harness's quirks, and it
costs a configuration change here, not a code change.

## Isolation: grade the patch, not the container

When the episode ends, the agent's container yields its `git diff` as a
captured artifact. Grading then runs in a **second, freshly provisioned
container**, started from the task's pristine image, where that diff is
applied and the instance's tests run. The agent's container is never the
grading container.

This removes an entire class of reward hacks in one move: editing the tests,
monkeypatching `pytest`, leaving a permissive `conftest.py` behind, mutating
installed packages — none of it survives a transfer where only the diff
travels. `reliquary_swe/grading.py` additionally restores every path a
patch confined to test or configuration files could use to control what "the
tests" report (the instance's own `test_patch`, the `conftest.py` hierarchy
above it, the test runner's own entry point, repo-root pytest/tox
configuration) before running anything — see that module's docstring for
exactly what this does and does not guarantee; a source-only patch that
forges its own test outcome (e.g. monkeypatching `pytest`'s own reporting at
import time) is a residual this scheme cannot close, because restoring
source is the one thing a grader must never do.

Reward is binary and comes from exactly one component:
`patch_passes_tests` — every FAIL_TO_PASS and every PASS_TO_PASS test must
pass, or the reward is `0.0`. A fractional reward would pay for a
half-repair.

Network is closed the same way on both containers: `network_allow = []` on
every task, enforced by the Docker runtime's egress proxy — not by this
package, which does not implement network containment itself. `setup()`
additionally truncates the repository's git history at the base commit and
clears caches and build logs that could otherwise name the published fix.

## Verifiers-only

This package declares **no `compatibility_entrypoint`** in `environment.toml`
and ships no synchronous JSON replay surface. A container cannot be replayed
deterministically, so shipping that surface would be a promise this package
cannot keep. This is a conscious deviation from this repository's own
convention (every other package here exposes both a native `Taskset` and a
compatibility surface) — see the design spec's section 2
(`docs/superpowers/specs/2026-09-22-reliquary-swe-env-design.md`) and the
repository-root README's package table, which marks this environment
Verifiers-only for the same reason.

## How the policy is prompted

`environment.toml` declares `[policy] reasoning = "thinking"`. The rationale
is an **argument, not a measurement**: a repair starts with diagnosis — the
agent must read code it has never seen and locate a fault from a bug report
before its first edit — and with a shell as its only tool there is nowhere
else for that reasoning to happen. Every other environment in this
repository earned its reasoning mode by measurement; this one has not yet
had a pilot to measure it against, and the rationale must be rewritten with
a measured band once one exists.

`max_turns = 40` is likewise **provisional and carries no measurement** —
copied from `reliquary-telecom-solo`'s measured turn budget as a starting
point, not derived from anything observed here. The design spec's section 10
lists both the turn budget and the token budget as open questions to be
settled by a pilot's observed distribution rather than by opinion.

## Grading timeout

Every task's `timeout.scoring` is set to **1800 seconds (30 minutes)** in
`reliquary_swe/taskset.py`, which points here for the full reasoning. Without
this, `env.py`'s `_grade` wraps provisioning-through-grading in
`asyncio.timeout(task.data.timeout.scoring)`, and a `None` timeout is
unbounded: a box that is reachable but *hangs* — a wedged daemon, a stuck
exec, a test process that loops instead of finishing — would neither raise
nor score, which defeats the same "infrastructure failure must raise, never
score" guarantee `env.py`'s retry-then-raise exists to give from the other
side.

Measured on the container host: `astropy__astropy-12907` (15 tests) 2.3s;
`sympy-14248` (435 tests) 99s (~0.23s/test, the slowest per-test rate
measured); `django__django-10097` — whose `test_patch` names no test files,
so `get_test_directives` returns none and django's own runner falls back to
its *entire* suite, 12,311 tests — ~226s of test execution, ~225s wall
clock including checkout and teardown. Across the full 500-instance corpus,
test count is median 52.0, mean 123.3, p90 250, max 2,488 — re-measured
directly from the loaded corpus rather than taken on trust (`p90` by sorting
all 500 counts and indexing `round(0.9 * 499)`; a different interpolation
method gives 254, the same ballpark, and nothing here turns on the
difference). The max is `matplotlib__matplotlib-25122`, which could not be
graded end to end on the measurement box: its image would not pull there, a
Docker daemon subordinate-UID limit unrelated to timing. Projecting the
slowest measured per-test rate onto that instance's 2,488 tests gives ~570s
— above django's measured worst case despite fewer tests, because per-test
cost varies by well over 10x across repository families and nothing
measured here bounds it from above for every family. 1800s gives roughly 8x
headroom over the worst wall-clock time actually observed and roughly 3x
headroom over the pessimistic projection for the corpus's most test-heavy,
unmeasured instance.

## Reward-hacking mitigation

Three layers, cheapest first: an empty network allowlist closes package
installation, raw file fetches, and upstream clones outright; `setup()`
removes git history past the base commit and build artifacts that could name
the fix; grading restores test and test-infrastructure paths before running
anything (see Isolation, above). What none of this measures yet — the
false-positive and false-negative rate of these test suites as graders, and
whether a suite is flaky — is tracked as an open item in the design spec's
section 10, and is required before a real training run, not before shipping
this package.

## Corpus

`princeton-nlp/SWE-bench_Verified`, pinned at revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, 500 human-validated instances.
This is an **evaluation set and must never be trained on** — see
`environment.toml`'s `[data]` table, which declares only `eval`. Training is
meant to come from SWE-smith, which is not wired in: its rows carry no
`version`, `base_commit`, or `test_patch`, which changes the task and the
grader's test-restoration strategy, not only the corpus loader (design
spec's section 8, "a second corpus is a profile, not a data pin").

The SWE-bench project itself (the harness and dataset-construction code) is
MIT-licensed; the dataset's own Hugging Face card declares no license tag.
Each row's actual content — a repository diff and the text of the issue
that motivated it — is drawn from the underlying open-source project
(astropy, django, sympy, sphinx, ...) under that project's own license.

## Test and load

```bash
uv sync --locked
uv run pytest
```

Unit tests (`tests/test_corpus.py`, most of `tests/test_adapter.py`) need no
Docker. Container tests are marked `@pytest.mark.docker` and need a real
Docker daemon with the instances' SWE-bench images reachable — they are the
only tests that exercise the isolation guarantees above end to end, and
several of this package's most serious defects (silent-zero rewards from a
wrong test invocation, a false full reward from an unrestored
`conftest.py`) were only ever visible inside a real container, never in a
mocked one. Four of them are additionally marked `@pytest.mark.slow`:
`tests/test_adapter.py::test_grading_django_end_to_end_is_not_actually_broken`
and two `tests/test_goldens.py` negative controls each run django's own
*entire* 12,311-test suite (its `test_patch` touches only `.txt` fixtures, so
a genuine, restored run falls back to everything, ~226s each), and
`tests/test_adapter.py::test_grading_a_large_instance_end_to_end_gets_a_real_result`
runs xarray's 1,818 FAIL_TO_PASS + PASS_TO_PASS tests (~300s). Together they
add roughly 15 minutes of test execution and their own multi-gigabyte images,
so CI's every-push job deselects them (`-m "not slow"`) to stay within a
hosted runner's disk and time budget — see `.github/workflows/ci.yml`'s
`reliquary-swe` job for exactly what that excludes and why. All four remain
part of the routine suite run against a dedicated container host (measured
there at 789s / 63 passed for the full, undeselected suite), and nothing
about the CI exclusion is silent: the marker exists precisely so it is a
stated decision rather than a test that quietly stopped running.

```bash
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-swe'); print(next(iter(vf.load_taskset(c(id='reliquary-swe')))).key)"
```

The package exports `SweTaskset`, `SweEnv`, and `SweEnvConfig` for Verifiers.
It imports no Reliquary code.
