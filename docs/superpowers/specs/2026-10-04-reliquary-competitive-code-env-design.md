# reliquary-competitive-code: competitive programming graded on stdin/stdout

Date: 2026-10-04
Status: implemented as environments/code/reliquary_competitive_code (Tasks 1-9); Task 10 (build/publish/pin) pending.

## 1. Why this environment

Teutonic's RL mix needs a single-turn code environment that the model does not
already solve. `reliquary-code` (OpenCodeInstruct, function-call cases) no longer
is one:

- On Teutonic SFT v2 at 8,192 tokens it succeeds 70 % of the time with a
  best-of-16 margin of only x1.3 (measured 2026-09-18). Most groups are unanimous.
- The corpus SFT jobs `corpus-code-v1` and `corpus-code-v2` distil rows
  0-200,000 of the same dataset. RL on those rows would re-saturate further.

Every open recipe that reports code RL on a model of this size uses competitive
programming with hidden stdin/stdout tests (DeepCoder, Nemotron-Cascade,
INTELLECT-3 `i3-code`, Skywork-OR1). See the research report
`Catalyst/reports/Environnements RL pour Teutonic.md`.

## 2. The shape: one verifier, one curated dataset, N sources

An environment here is a package that pins four things together: the data, the
prompt, the verifier and the policy (`reasoning`, `max_new_tokens`). The data is
pinned, not plugged in at runtime, because the validator replays and every miner
must see the same problems. Adding a source therefore means publishing a new
curated revision and a new package version.

Inside the package the three concerns are kept apart, so that sources can be
added without touching the verifier and the verifier can be reused without any
source:

| Path | Responsibility | Knows about sources? |
| --- | --- | --- |
| `reliquary_competitive_code/judge/` | Run Python on a stdin, compare its stdout to the expected one. Pure comparator + a local subprocess runner for development and qualification. | No |
| `reliquary_competitive_code/sources/` | One adapter per upstream dataset (`deepcoder.py`, `ccplus.py`, ...). Each yields rows in the common schema of section 4. | Its own only |
| `reliquary_competitive_code/build/` | Assemble sources, deduplicate, decontaminate, validate with a reference solution, cap tests, split, publish. Runs offline on a build box. | Through the adapters |
| `reliquary_competitive_code/taskset.py` | Read the pinned curated dataset, render the prompt, call `judge/`. | No |

Production never runs `build/`. It reads one pinned dataset:
`hf:R0mAI/competitive-code-curated@<sha>`.

## 3. What was measured (2026-10-04, local, no GPU)

DeepCoder-Preview-Dataset @`177913a7`, licence MIT, train configs only:

| config | rows | stdin/stdout | function-call | statements matching "print any"-style regex | tests per problem (p10 / p50) |
| --- | --- | --- | --- | --- | --- |
| taco | 7,436 | 6,387 | 1,049 | 597 | 65 / 103 |
| primeintellect | 16,252 | 14,998 | 1,254 | 1,834 | 18 / 101 |
| lcbv5 (train, contests 2023-05 to 2024-07) | 599 | 322 | 277 | 10 | 12 / 15 |

- The 21,707 stdin rows are only **11,287 distinct statements** (normalised hash):
  4,463 appear in both taco and primeintellect, and primeintellect repeats itself.
- 554 stdin rows carry more than 1 MB of test data.
- `lcbv5/test` (279) and `codeforces/test` (408) are excluded outright: their dates
  fall inside the held-out LiveCodeBench window.

CodeContests+ @`96c85054`, licence CC-BY-4.0, `default` config (no tests):

- 11,690 problems: Codeforces 7,566, AIZU 2,036, AtCoder 1,322, CodeChef 766.
- Codeforces contest ids go up to 1623 (2021). **Nothing is near the LCB window.**
- **Every problem carries a per-problem testlib checker**, so "has a checker" does
  not identify the multiple-answer problems.
- **No correct submission is in Python** (C++/Java only), so CC+ alone gives no
  Python reference to validate tests or calibrate time limits.
- 10,671 distinct statements, **6,854 shared with DeepCoder**; the union is 17,276.
- Pre-generated tests: `1x` averages 25 tests per problem and weighs 46 GB.

## 4. The curated dataset

### Row schema (what every adapter emits)

```
problem_id     sha256 of the normalised statement, 16 hex (identity, stable across sources)
statement      text shown to the model
tests          list of {stdin, stdout}, hidden from the model
time_limit_s   per test, Python, calibrated (below)
origin         {source, upstream_id, contest_id?, date?}
reference      a Python solution known to pass every kept test (published in references.parquet: the solutions are public upstream)
```

### Build pipeline (offline, on a build box with disk; never on the local VPS)

1. **Load** each source through its adapter. Keep stdin/stdout tests only.
2. **Exclude by date**: any known date at or after 2024-08-01.
3. **Decontaminate**: 13-gram overlap of the statement against LiveCodeBench v5 and
   v6 statements. taco and primeintellect have no dates, so this is the only guard
   for them. Any hit drops the problem.
4. **Deduplicate**: exact normalised hash, then MinHash (Jaccard >= 0.8) for
   near-duplicates. When a problem exists in several sources, keep one row and
   take the strongest tests (CC+ when present).
5. **Exclude multiple-answer problems (v1)**: the regex above, plus, when two or
   more reference solutions exist, any test on which they disagree token-wise.
   They need the testlib checkers, which are v2 (section 8).
6. **Validate with a reference**: run a Python reference through `judge/` on every
   test. Any failing test drops the **problem**, not the test: a test the reference
   fails is either wrong or impossible in our time limit, and keeping the problem
   with fewer tests silently weakens it.
7. **Calibrate time limits**: `time_limit_s = clamp(4 x reference CPU time on the
   slowest test, 1 s, 4 s)`. Problems whose reference needs more than 2 s on any
   test are dropped, which keeps the margin at 2x or more (they would grade on
   hardware speed, not correctness).
8. **Cap tests by cost, not count**: keep all tests while the reference's total
   CPU time stays under 8 s; otherwise keep the boundary tests (smallest, largest)
   and a fixed hash-chosen sample up to that budget. Drop problems left with fewer
   than 5 tests.
9. **Split by hash** of `problem_id`: train 95 %, eval 2.5 %, qualification 2.5 %.
10. **Publish** to `R0mAI/competitive-code-curated`, record the sha and per-stage
    drop counts in the README (as `reliquary-science` did).

### Sources, in order

- **DeepCoder (v1)**: taco, primeintellect, lcbv5 train. References come from the
  rows' own `solutions` (Python).
- **CodeContests+ (v1.1)**: blocked on Python references. Plan: for the 6,854
  shared problems, use CC+'s tests with DeepCoder's Python reference; for the
  ~3,800 CC+-only problems, use a teacher (Qwen3.8-27B) solution that passes every
  test as the reference. CC+ tests are read on the build box from the `1x` config.

Expected size after all filters: **8,000-14,000 problems**. An estimate; the build
records the real number.

## 5. The verifier

### Execution semantics (identical in the dev runner and in production)

- The submission is the last fenced Python block of the completion.
- It runs as a script: `__name__ == "__main__"`, stdin = the test input, stdout
  captured (capped at 4 x the expected output size + 64 KB; more is a failure).
- Allowed imports: the current grader allow-list plus `fractions`, `random`,
  `string`, `itertools`, `threading` (recursion via a large-stack thread is common),
  and a **`sys` shim** exposing only `stdin`, `stdout`, `setrecursionlimit`,
  `maxsize` and `exit`. `input()` and `print()` work. No `os`, no `io` file access.
  As implemented, the allow-list is `ALLOWED_IMPORT_ROOTS` in `judge/guest.py`,
  and the prompt lists it, generated from that set.
- Per test: CPU time limit `time_limit_s`; wall clock = 2 x CPU limit as backstop.

### Comparator (pure function, the single source of truth)

- Split both outputs on whitespace; same number of tokens required.
- Token equal as strings, or both parse as floats and agree within 1e-6 absolute
  or relative.
- Nothing else is normalised, with one open exception: the case of `yes`/`no`
  tokens (section 9, question 1).

### Reward

- **Binary**: 1.0 if every test passes, else 0.0. DeepCoder reports that partial
  credit was reward-hacked. Reward lattice `binary-v1`.
- **Stop at the first failing test.** It changes nothing to a binary reward and
  bounds the cost of wrong submissions, which are most of them.
- Statuses kept for diagnostics: `ok`, `wrong_answer`, `timeout`, `runtime_error`,
  `forbidden_import`, `output_limit`, `no_code`. `harness_overload` (the wall
  clock fired before the program used its CPU limit) is not a verdict: grading
  raises rather than return a reward.

### Where it runs

- **Dev and qualification**: `judge/runner.py`, one subprocess per test, limits
  set by the child on itself (the `reliquary-code` lesson: no `preexec_fn`).
- **Production**: the core gVisor grader. The comparator is imported from this
  package by the core, so there is exactly one comparator. The execution semantics
  are duplicated (subprocess here, warm worker there) and kept equal by shared
  goldens (section 7).

## 6. Core integration (Catalyst)

- **Grader worker**: a new request kind `stdin` carrying `{code, stdin,
  time_limit_s}` and returning `{stdout, status}`. As today, the worker never sees
  the expected output; the trusted server compares. RLIMIT_CPU is cumulative per
  warm worker, so the worker's lifetime budget (`worker_lifetime_cpu_budget_seconds`)
  must be raised or workers recycled per submission; measured, not guessed.
  `judge/guest.py`'s contract: one run per fresh process (or fork), never
  concurrent; module state (e.g. monkeypatched math, random seed, daemon
  threads) is not isolated between runs in one process. The core imports
  `extract_program` and `outputs_match` from this package.
- **Registry**: an `EnvironmentSpec` `reliquary_competitive_code_v1`,
  `admission_resource_class="sandbox"`, `final_answer_policy="fenced_python"`,
  `reward_lattice_policy="binary-v1"`, `attainable_rewards=(0.0, 1.0)`,
  `validator_authoritative_reward=True`, bound by the artifact manifest digest
  like `reliquary_science_v1`.
- **Profile**: declared in the dormant Teutonic profile, with its cooldown
  (one pass of the train split) and its budget.

Replay note: a time limit is hardware-dependent. Rewards are validator-
authoritative, so miners and validator never need to agree. The calibration
margin (x4 the reference) keeps a single validator stable; a second validator on
slower CPUs would need the same calibration check.

## 7. Testing

- **Comparator unit tests**: tokens, floats, trailing whitespace, extra lines,
  empty output.
- **Discrimination goldens** (the failure that matters on this project is a reward
  0 that is really a broken harness): for a fixed set of ~30 problems, the
  reference scores 1.0, a wrong-answer variant 0.0, an infinite loop `timeout`,
  `import os` `forbidden_import`, a `sys.stdin.readline` solution 1.0, a
  recursion-heavy solution using the thread trick 1.0.
- **Parity goldens**: the same set run through the core gVisor grader must give
  identical verdicts. CI on the env repo runs the subprocess runner; the Catalyst
  side runs the gVisor path.
- **Build report**: per-stage drop counts and the reference pass rate (100 % by
  construction) committed with each curated revision.

## 8. Not in this version

- **testlib checkers** for multiple-answer problems (v2). They are trusted code,
  so they can run outside gVisor, but they need g++ and testlib in the grader
  image. Worth it only if v1 lacks volume or the excluded share is large.
- More sources (rStar-Coder, 418K, CC BY 4.0) once the pipeline is proven.
- Languages other than Python.

## 9. Open questions

1. **Case of YES/NO tokens.** Codeforces accepts any case for YES/NO answers; a
   strict comparator would mark correct solutions wrong. Proposal: case-insensitive
   only for the tokens `yes`/`no`.
2. **`max_new_tokens` and reasoning.** Competitive problems need longer reasoning
   than OpenCodeInstruct. To measure at 8k / 16k / 32k on Teutonic, as was done for
   DAPO; `reasoning = "thinking"` (the extractor takes the last fenced block, so
   deliberation cannot cost reward).
3. **Grader capacity.** Cost per rollout is bounded by the 8 s reference budget x4
   for correct submissions. To measure on the grader host before setting the
   environment's share.
