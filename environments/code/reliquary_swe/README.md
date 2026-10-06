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
above it and, at every level of that same walk, pytest's own five config
filenames — not only the repo root, since pytest's own config search finds a
nested one first — and the test runner's own entry point) before running
anything — see that module's docstring for exactly what this does and does
not guarantee; a patch confined entirely to source that forges its own test
outcome (e.g. monkeypatching `pytest`'s own reporting at import time) is a
residual this scheme cannot close, because restoring source is the one thing
a grader must never do.

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

## Timeouts

Every phase of a rollout used to default to no limit at all except scoring:
`TimeoutConfig`'s and `TaskTimeout`'s own fields are all `None` by default,
`agent.py` resolves the agent phase as "task's timeout, else no limit", and
`rollout.py` enters `asyncio.timeout_at(None)` when that resolves to `None` —
which never raises. This is the one environment in this repository that
hands a policy a general shell in a repository whose own suite can take
minutes, and `max_turns` bounds turns, not wall clock: running the tests is
the most natural thing a repair agent does, and unlike a wedged daemon this
is *policy-triggerable* — a rollout slot held for hours, on purpose or not.
`reliquary_swe/taskset.py` now sets all four `TaskTimeout` fields; three of
the four are sized past a real measurement, not guessed, reasoned about
below. `setup` is the exception: see its own paragraph.

**`setup`** (900s) covers `SweTask.setup`'s cleanup script — `git reset`,
history truncation, `git gc --prune=now` — *and* the harness's own setup:
`rollout.py` computes one setup-stage deadline and wraps both `task.setup`
and `harness.setup(runtime)` in it, so the two share this single budget.
Only the first term is measured. On the container host, across the
corpus's largest families by instance count and by one clear outlier in
`.git` size (astropy, django, matplotlib, sympy, scikit-learn, sphinx):
matplotlib's `.git` is the heaviest sampled at 296 MB (the next largest,
django's, is 78 MB), and its full cleanup script — `git gc` included —
runs in 3.6s. The second term, harness setup, is unmeasured and is the one
that dominates: for the `bash` harness this repository's example pins
(`examples/prime_rl/rl.toml`), setup is a genuine per-rollout network
install inside the fresh container — `pip install -q -U --user uv`
(falling back to `apt-get install curl` plus a curl-fetched installer),
then `uv sync --script`, which fetches a managed CPython and the script's
dependencies. Egress is still open at this point (it closes only after
setup), and there is no cross-rollout cache — the uv interpreter cache
lives on the per-rollout `Runtime`. 900s is sized by analogy with the
other three phases below, not derived from a measurement of that install;
an honest "unmeasured, sized by analogy" beats a fabricated derivation.
`--env.agent.timeout.setup` overrides the task's value at run time, so a
pilot that hits this ceiling does not need a code change.

**`agent`** (3600s) bounds the policy-triggerable phase Important 1 is
actually about. The slowest real full-suite run measured
(`django__django-10097`, whose `test_patch` names no test files, so it falls
back to its entire 12,311-test suite) took ~226s of test execution
(12,311 tests) by the django runner's own report, and a separately
hand-timed wall clock of ~3m45s (225s) including migrations and teardown —
the same figures `tests/test_adapter.py`'s own docstring records by hand.
These are two independent measurements of about the same run, not a
precise execution/wall-clock split, so the 1s gap between them is
measurement rounding, not a claim that the wall clock is smaller than the
execution it contains. The worst *projected* instance
(`matplotlib__matplotlib-25122`, unrunnable on the measurement box — a
Docker daemon subordinate-UID limit, not a timing fact) comes to ~570s by
projecting the slowest measured per-test rate onto its 2,488 tests. An hour
gives roughly 16x headroom over the worst wall clock actually observed and
roughly 6x over the pessimistic projection — room for several such runs plus
real editing, while still turning "holds the slot for hours" into a bounded,
finite failure. This is unmeasured against a real trajectory distribution,
like `max_turns` and the per-turn token budget below it — revisit once a
pilot exists.

**`finalize`** (900s) covers `SweTask.finalize`'s `git add -A` + `git diff
--cached --binary` against whatever the agent's box holds by the time it
runs. Measured on the container host: a single 573 MB novel file costs
18.5s (`add`) + 16.7s (`diff --binary`) = 35.2s combined, roughly 60s/GB.
900s covers on the order of 15 GB of agent-authored content — far past any
legitimate source edit — while still bounding a disk-filling pathology
(container disk quotas are advisory on Docker, per `TaskResources.disk`'s
own docstring, not enforced) to a fixed ceiling instead of an unbounded
hang.

**`scoring`** (1800s, unchanged) bounds `env.py`'s `_grade`, which wraps
provisioning-through-grading in `asyncio.timeout(task.data.timeout.scoring)`.
A `None` timeout here is unbounded: a box that is reachable but *hangs* — a
wedged daemon, a stuck exec, a test process that loops instead of finishing —
would neither raise nor score, which defeats the same "infrastructure
failure must raise, never score" guarantee `env.py`'s retry-then-raise
exists to give from the other side.

Measured on the container host: `astropy__astropy-12907` (15 tests) 2.3s;
`sympy-14248` (435 tests) 99s (~0.23s/test, the slowest per-test rate
measured); `django__django-10097`'s ~226s/~3m45s full-suite run, above.
Across the full 500-instance corpus, test count is median 52.0, mean 123.3,
p90 250, max 2,488 — re-measured directly from the loaded corpus rather than
taken on trust (`p90` by sorting all 500 counts and indexing
`round(0.9 * 499)`; a different interpolation method gives 254, the same
ballpark, and nothing here turns on the difference). The max is
`matplotlib__matplotlib-25122`, unrunnable on the measurement box for the
same subordinate-UID reason as above. 1800s gives roughly 8x headroom over
the worst wall-clock time actually observed and roughly 3x headroom over the
pessimistic ~570s projection for the corpus's most test-heavy, unmeasured
instance.

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

**Monitoring, not mitigation.** All nine implementation defects recorded in
this project's build history (conda not on `PATH`, the 128 KiB argv cap,
django's runner writing to stderr, sphinx's own invocation missing `-rA`,
among others) presented as the exact same shape: the test command ran but
produced nothing `parse_results` could read, and `results == {}` scored
identically to "the tests ran and genuinely failed" — invisible until a
human noticed a zero on real data, one at a time, on separate nights.
`grading.Report` now also carries `results_parsed` (how many test names were
actually readable) and `test_command_exit_code`, both surfaced onto
`solution.info["swe_report"]`. Neither is a reward component, and this
cannot become an unconditional raise — a patch with a genuine syntax error
legitimately produces no parseable output. The actual monitoring canary is
aggregate, not per-instance: `pass_to_pass_passed == 0` across an entire
batch, because a real adapter break zeroes p2p for every instance in the
batch at once, and a patch that is merely bad does not. That shape would have
caught four of the nine defects (the adapter-level ones) in one batch,
instead of four separate nights.

## Corpus

**Evaluation**: `princeton-nlp/SWE-bench_Verified`, pinned at revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, 500 human-validated instances.
This is an **evaluation set and must never be trained on** — `--taskset.split`
defaults to `"eval"`, and nothing in this package's own code path ever passes
`"train"` for it.

**Training**: SWE-smith (`SWE-bench/SWE-smith`, pinned at revision
`ea6d7173829c7ec8fa16c22055699ff2e9188091`, 59,136 rows across 222 images —
`--taskset.split train`). Its schema carries no `version`, `base_commit` or
`test_patch`, which changes the task and the grader's test-restoration
strategy, not only the corpus loader (design spec's section 8, "a second
corpus is a profile, not a data pin"):

- **The image is given, not derived.** SWE-smith's own `image_name` column
  is the actual, pullable reference; a from-scratch derivation
  (`swesmith.profiles.RepoProfile.image_name`) computes a *different*, wrong
  Docker Hub namespace (checked directly — see the implementation report).
- **No `version`.** `reliquary_swe/swesmith_adapter.py` — the only module
  importing `swesmith`, same pattern as `swe_adapter.py` and `swebench` —
  resolves the test command and log parser through
  `swesmith.profiles.registry` instead, keyed on `repo` alone. Not every
  SWE-smith repository is Python (92 of the pinned corpus's 222 images are
  Go, PHP, Java or Rust); this module only knows how to run Python ones, and
  `SweTaskset.load()` fails loudly at load time (not mid-rollout) if a
  configured `num_images` reaches one — checked and not triggered below 31.
- **No `base_commit`.** Each SWE-smith instance is its own git branch,
  committed on top of the image's clean `main`: "Bug Patch" (the dataset's
  `patch` column, source only), then "Remove F2P Tests" (deletes the
  fail-to-pass tests' own files). `~1` from the branch tip — bug present,
  every test file still pristine — is the state a rollout starts from, and
  `SweRow.base_commit` carries it as a git revision expression
  (`f"origin/{instance_id}~1"`) rather than a literal SHA. `setup()`'s
  existing ref-stripping cleanup consumes either shape identically, and
  closes the reachable-by-ref version of the obvious reward hack here (the
  exact fix sitting in `origin/main`, in the very same container). It is
  **not** enough on its own: `origin/main` and the "Bug Patch" commit's own
  parent are the same object — `~1` is a *child* of the pristine tree, so
  after checkout the box holds exactly two commits with the fix one parent
  away (`git diff HEAD HEAD^` is the exact gold patch, `git show
  HEAD^:<path>` is the fixed file, both offline, with zero refs needed).
  Ref-stripping cannot touch an ancestor of HEAD — `git gc` will not drop
  one. `taskset._TRAIN_GUARD_AND_REROOT` closes it for real: assert the
  branch is shaped as documented (a "Bug Patch" commit message on
  `base_commit`, so a mis-shaped branch raises instead of silently paying),
  then re-root HEAD onto a fresh, parentless commit with the identical
  tree, *before* the existing reflog-expire+`gc` runs, so the old chain
  becomes genuinely unreachable and gets pruned. Verified directly: after
  this, `git rev-list --count HEAD` is 1 and a full
  `git cat-file --batch-all-objects --batch` scan finds none of the gold
  patch's added lines anywhere in the object store. The grading box is
  unaffected (a fresh checkout of `origin/<instance>~1`, never put through
  this cleanup, needs that ref to resolve `base_commit` at all).
- **No `test_patch`, and `patch`'s own direction is inverted.** SWE-smith's
  `patch` column is the diff that *introduces* its bug, not the fix —
  confirmed three ways (diff content, a real container run, and SWE-smith's
  own harness applying "gold" predictions with `git apply --reverse`,
  commented "fix = revert"). `corpus._reverse_unified_diff` normalizes this
  once at load time, so `SweRow.gold_patch` means the same thing for every
  row regardless of source corpus. Restoration has no `test_patch` to
  reapply because SWE-smith never touches tests at all; `grading.py`'s
  `_restore_from_pristine_image` — the second strategy
  `_restore_strategy_for` was always meant to grow (spec section 8) — checks
  out the instance's own fail-to-pass/pass-to-pass test files, plus the same
  `conftest.py`/pytest-config ancestor walk the SWE-bench Verified path
  uses, from that same base-commit expression.

**Corpus selection is two declared parameters, never auto-detected**
(`SweTasksetConfig.num_images`, default 20; `.max_test_count`, default
`None` — uncapped) — see `corpus.load_swesmith_rows`. `num_images` takes
the top N images by task count (ties broken by image name), because a size
sized from local free disk would let two machines disagree on what task
#400 is. `max_test_count`, when set, caps fail-to-pass + pass-to-pass test
count per instance, because that count is heavily right-skewed (p50=415,
p95=5060, max=22,028 across the full corpus) and, for the minority of
repositories whose `swesmith` profile narrows its test command to exactly
those tests (`min_testing=True` — pandas, sqlglot, sunpy, conan, sqlfluff
among the top 20 images), that skew is a real per-instance cost
difference. Both parameters participate in task identity: two configs
differing in either one are different task sets, never a shared index with
a subset relationship.

Real container measurements (seven gradings across the size range, the
implementation report has the transcripts) found the *actual* cost far
below a naive per-test estimate — even the single most expensive instance in
the whole corpus (pandas, 22,028 fail+pass-to-pass entries) graded in 40
seconds — so `max_test_count` defaults to `None`: discarding real training
rows by default to bound a cost this package's own measurement shows is not
a problem here would be the wrong trade. `corpus.DEFAULT_SWESMITH_MAX_TEST_
COUNT` (5060, the measured p95) is kept as a named value to pass explicitly
if a repository this package has not measured ever turns out as
per-test-expensive as sympy did (the source of the original, much higher
estimate). For the majority of repositories (profile `min_testing=False`,
the default), the cap would not bound cost at all even if set: their test
command always runs the whole suite regardless of any one instance's
counts, so cost there is a fixed per-repository quantity — measured
directly at 3-34 seconds for four such repositories in the top 20 images.

**Zero repository overlap with the evaluation set**, checked at every image
count from 5 through the full 222, not only the shipped default of 20 — see
`tests/test_swesmith_adapter.py::test_no_repo_overlap_between_the_two_corpora`.

The SWE-bench project itself (the harness and dataset-construction code) is
MIT-licensed; SWE-smith's own Hugging Face card declares `license: mit`
too. Neither dataset's card license extends to a row's actual content — a
repository diff and the text of the issue or bug that motivated it — which
is drawn from the underlying open-source project (astropy, django, sympy,
pandas, oauthlib, ...) under that project's own license.

## Polyglot corpus

**Training, second corpus**: `XiaomiMiMo/MiMo-V2.6-RL-oss`, config `code`,
pinned at revision `639865fd3374018d6cb29b9fb82dd531406fcf5f` — 2,698 tasks,
`--taskset.split polyglot`. By the language of their hidden tests: 1,154
Python, 701 Go, 348 JavaScript, 138 TypeScript, 24 TSX, then Ruby, PHP,
Java, C++ and Rust. It differs from both corpora above in what it publishes:

- **No fix, no test lists.** Each row is a feature request or bug report plus
  a `test_patch` that adds *hidden* tests and `mimo_test_command.sh`. The
  verdict is that script's exit status — 0 pays, anything else does not.
  `SweRow.gold_patch`, `fail_to_pass` and `pass_to_pass` stay empty rather
  than guessed.
- **One image per task**, `xiaomimimo/mimo-v2.6-rl-oss:<instance_id>`, with
  the repository at the `/testbed` or `/workspace/repo` the row names. On the
  first 1,000 tags Docker Hub lists anonymously, the median is 3.1 GB
  compressed (0.3–10.7 GB), about 2.6 TB for those 1,000 alone. There is no
  `num_images` knob to lean on here: pulling the whole corpus is a disk
  decision to take before pointing a run at it, not after.
- **Two image shapes**, both checked on a real image. Either HEAD is a
  parentless "task base" commit with the requested behaviour cut out of the
  tree — and the complete upstream, implementation *and* tests, still parked
  under `origin/<branch>` (`git diff HEAD origin/master` would have handed
  the agent the answer) — or HEAD is upstream's own tip with nothing after
  it. The ref stripping `setup()` and `grade()` already run removes the
  first shape's parked upstream, verified by a full object-store scan; no
  re-root is needed, because nothing leaks through HEAD's ancestry in either
  shape. Two fixes to the shared cleanup came out of this corpus:
  `git checkout --detach` (its `base_commit` is the symbolic `HEAD`, which
  the strip otherwise leaves attached to a deleted branch) and deleting
  broken symbolic refs directly (`for-each-ref` skips them, and `gc` then
  dies on them).
- **A wider restoration walk.** An exit status is only as honest as the
  runner that produced it, so grading puts back, at the root and above every
  file `test_patch` touches, the runner configuration of every ecosystem in
  the corpus — `package.json`, jest/vitest/mocha configs, `go.mod`,
  `Cargo.toml`, `pom.xml`, `Gemfile`, `phpunit.xml`, `Makefile`, and
  pytest's own — alongside `test_patch`'s own paths (see
  `grading._POLYGLOT_RUNNER_FILES`). It can revert a legitimate edit to one
  of those files; the trade is the one `_TEST_CONFIG_FILES` already made.
  The source-monkeypatch residual named under "Reward-hacking mitigation"
  applies here too, and is wider: a Go `init()` that exits 0 needs no test
  framework internals at all.

**The reference fix is in the image, not the dataset.** For a shape-one
image, the diff from HEAD to its parked upstream is a fix the hidden tests
accept: `tests/test_polyglot_goldens.py` recovers it in a separate box and
scores it 1.0 through the whole loop. A shape-two image has no such
reference; only its negative control (an empty patch scores 0) is checked.
Whether every task is solvable at all is therefore unmeasured, and so is the
share of each shape across the corpus — two images were inspected, one of
each.

**Repository overlap with the evaluation set**: none found, but only by
heuristic — rows carry no repository name. No `test_patch` path matches the
layout of any of SWE-bench Verified's 12 repositories; the two partial
matches (`testing/test_runner.py`, `tests/test_requests.py`) are xdoctest
and Starlette. Problem statements that *mention* those projects (Django
plugins, geopandas, pygmt) are downstream code, not the projects.

## R2E corpus

**Training, third corpus**: `R2E-Gym/R2E-Gym-Subset`, pinned at revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5` — 4,578 tasks, `--taskset.split
r2e`, the set behind DeepSWE's SWE-bench Verified gains. Every task is a real
commit of one of 10 Python repositories: pandas 1,444, numpy 781, pillow 620,
orange3 482, aiohttp 299, tornado 261, scrapy 215, pyramid 189, datalad 179,
coveragepy 108. None is one of SWE-bench Verified's 12
(`tests/test_r2e_corpus.py` checks it against the `eval` split itself).
Apache-2.0 per the dataset card.

- **One image per task**, `namanjain12/<repo>_final:<fix commit>`, with the
  repository at `/testbed`, HEAD detached at the pre-fix commit and its own
  `.venv`. All 4,578 are pinned by registry digest in
  `reliquary_swe/r2e_digests.json` (`scripts/pin_r2e_digests.py`, which
  asks the registry and pulls nothing; it merges, so a re-run only adds).
  `--taskset.num_tasks N` is the first N tasks of a fixed order — ascending
  sha256 of the instance id — not the dataset's own, which is grouped by
  repository (its first 482 rows are all orange3): the first 100 tasks span
  9 repositories, the first 500 all 10.
- **The image's working tree is the base, not its HEAD.** On the pandas
  golden the image builder staged edits to three files, edited `setup.cfg`
  and deleted `pyproject.toml`; restoring HEAD's `setup.cfg` brings back an
  `addopts = --strict-data-files` the hidden tests cannot run under, and
  every rollout, gold included, scores 0. Setup and grading therefore commit
  the image's own state on top of HEAD and take that commit as the base
  (`taskset._R2E_CHECKOUT`).
- **The full upstream history is in every image** — all branches, the fix
  commit included (`git cat-file -t <fix>` answers `commit` on all three
  goldens' images). The shared ref strip and `gc` remove it from the agent's
  box and from the grading box alike; `tests/test_r2e_goldens.py` asserts
  the fix commit is unreachable after setup on all five golden images.
  Setup and grading also refuse a repository with object alternates, linked
  worktrees, a shallow file or grafts (`taskset._R2E_LEAK_GUARD`), which
  the strip-and-gc reasoning does not cover; none of the five has any.
- **The hidden tests are in the image too** — `/r2e_tests` and
  `/testbed/run_tests.sh` (`python -m pytest -rA r2e_tests`, or tornado's
  own `r2e_tests/tornado_unittest_runner.py`, which prints the same summary
  format). Setup deletes both from the agent's box. Grading, in a fresh box,
  moves both out of the repository before the patch is applied, then puts
  them back as R2E's own runtime does (`r2e_tests` as a symlink at
  `/testbed/r2e_tests`, `run_tests.sh` at the root), deletes every root
  entry the patch added and restores the root `conftest.py` and pytest
  configs (`grading._restore_r2e`). Two of those vectors were measured on
  the coveragepy golden before being closed: a root `conftest.py` forcing
  "passed" and a root `pytest.py` shadowing pytest (`python -m` puts the
  root first on `sys.path`) each make the hidden tests report whatever the
  patch wants. Before the patch is applied, grading also refuses (0.0, not
  applied) any patch touching `.venv/`, a path present but untracked at the
  base (a `<project>.egg-info`, build output), or a path git ignores at the
  base: those already exist in the image, so deleting new root entries never
  reaches them, and a `.venv/lib/python3.X/site-packages/*.pth` runs at
  interpreter start-up -- measured on coveragepy, one printed a forged
  summary and exited 0. The agent's own capture never produces such a path
  unless it un-ignored one on purpose. The source-monkeypatch residual
  named under "Reward-hacking mitigation" remains.
- **Reward: the exact verdict map.** `run_tests.sh` runs under a 300 s
  timeout (R2E's own default; 0-3 s measured), only once the hidden tests
  and runner are confirmed in place (otherwise 0.0, nothing run). Its
  stdout is parsed by a port of R2E's `parse_log_pytest` that reads a
  status only from a line's first token -- upstream's `"PASSED" in line`
  lets `FAILED ...::test_fix - RuntimeError: PASSED` count as a pass --
  and the result must equal the row's
  `expected_output_json` after R2E's own key normalisation — FAILED and
  ERROR entries included (2,204 rows expect at least one). Two departures
  from R2E's `_calculate_reward_r2e`, both stricter: an empty parse never
  pays, and a collection error (parsed as the empty name, which upstream
  skips) cannot stand in for a missing test — upstream pays that on any of
  the 46 single-test tasks (`grading.r2e_reward`).
- **The gold patch is rebuilt**, from each fixed file's full pre- and
  post-fix contents, test files excluded (`corpus.is_r2e_test_file`: a
  `tests`/`test`/`r2e_tests` directory, case-sensitive, so pillow's
  `Tests/` helpers count as source; or `test_*.py`/`*_test.py`). It is
  never shown to the agent or graded against; the goldens use it. Checked
  with `git apply` against every row of the pinned revision.

**Goldens** (`tests/test_r2e_goldens.py`, real images, all marked `slow`:
CI's every-push job deselects them, run them on a container host):
coveragepy `c1bfa735`, tornado `b5ec807e`, pandas `fadb72cf`, numpy
`ebe2cfb6` and orange3 `c0174f90` each score 0 with an empty patch and 1
with the gold patch through the whole loop -- numpy's and orange3's expected
maps hold FAILED and ERROR entries, so their gold run also checks the
first-token parser against R2E's own maps -- with the fix commit
unreachable after setup; four tamper controls (a forged `run_tests.sh` and
`r2e_tests/`, a root `conftest.py`, a root `pytest.py`, a `.pth` shipped
into `.venv`) score 0 on coveragepy. Whether every task is solvable in this
harness is unmeasured beyond those five.

## Test and load

```bash
uv sync --locked
uv run pytest
```

Unit tests (`tests/test_corpus.py`, most of `tests/test_adapter.py`) need no
Docker. On a container host shared with production work,
`RELIQUARY_SWE_TEST_CPUS` and `RELIQUARY_SWE_TEST_MEMORY_GB` cap every box
the tests start (`--cpus`/`--memory`; unset, no cap). Container tests are marked `@pytest.mark.docker` and need a real
Docker daemon with the instances' SWE-bench images reachable — `uv run
pytest` stays runnable without one: `tests/conftest.py`'s own
`pytest_collection_modifyitems` skips every `@docker` test (a real `docker
version` round trip, not just the CLI's presence) rather than letting them
error for want of a runtime fixture. They are the
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
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-swe'); print(next(iter(vf.load_taskset(c(id='reliquary-swe', split='eval')))).key)"
```

The package exports `SweTaskset`, `SweEnv`, and `SweEnvConfig` for Verifiers.
It imports no Reliquary code.

## Signed sandboxes

`reliquary_swe.sandbox:sandbox_task` serves this package's tasks on a signed-episode
sandbox gateway (`episode_envs = {"reliquary-swe": "reliquary_swe.sandbox:sandbox_task"}`).

- Served splits: `train` and `train:<n>` (SWE-smith), `r2e`, `polyglot`. `eval` (SWE-bench
  Verified) is refused. Polyglot rows without a pinned digest are refused.
- State handed to the grader: the diff against `refs/reliquary/base` (so a committed fix
  counts), published to the miner as its `final_diff`. Grading runs in a pristine box and,
  on every split, refuses a patch that touches an untracked or ignored path or `.venv/`.
- Limits: 4 GiB, 10 GiB disk, 1024 pids, 3600 s, 600 s per call, grading 810 s. No network.
- `reliquary-sandbox` is imported lazily and is not a dependency of this package.

How hooks fail, what is refused and the measured parity table: `docs/sandbox-tasks.md`
(repository root).
