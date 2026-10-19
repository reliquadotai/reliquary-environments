# reliquary-competitive-code

Competitive-programming problems graded by running the policy's Python program
on hidden stdin/stdout tests. Design: `docs/superpowers/specs/2026-10-04-reliquary-competitive-code-env-design.md`.

## What it is

One turn, no tools. The task is a competitive-programming statement plus a
fixed contract (a complete Python 3 program reading stdin, printing stdout, in
a single fenced block at the end). The reward is 1.0 when that program passes
every hidden test of the problem within its time limit, and 0.0 otherwise;
token-wise output comparison, so trailing whitespace does not matter.

## Layers

1. `build/` and `sources/` (offline): fetch upstream problems, filter, dedup,
   validate the reference solution against its own tests, split by problem id
   and write a curated parquet dataset.
2. The published dataset: `problems-<split>.parquet` (statements, limits,
   tests by row group) and `references.parquet`, pinned in `corpus.PINNED` by
   repository, revision and file digests.
3. Served side: `corpus.py` (lazy by row group), `environment.py` (the
   synchronous replay ABI) and `taskset.py` (the Verifiers taskset
   `reliquary-competitive-code`), over the `judge/` that runs the program.

## Building the dataset

    uv run python -m reliquary_competitive_code.build.fetch --out DIR
    uv run python -m reliquary_competitive_code.build --deepcoder DIR/deepcoder --lcb DIR/lcb --out dataset/

`build.fetch` downloads the pinned upstream data into `DIR/deepcoder` and
`DIR/lcb`; `build` curates it and writes the dataset (`--workers`, `--limit`
for a smoke run; see `--help`). The build is not streaming: use a box with at
least 16 GB of RAM.

## Published revision

`ReliquaryForge/competitive-code-curated@1f6e4f12` (public), pinned in
`corpus.PINNED` with the sha256 of its four files. Built 2026-10-09 from
DeepCoder-Preview-Dataset @`177913a7` (taco + primeintellect) on 32 cores in
1 h 10 min, harness_overload 0:

| step | count |
| --- | --- |
| loaded | 21,294 |
| dropped: several answers (statement) | 2,180 |
| dropped: tests contradict a statement example | 388 |
| dropped: overlaps LiveCodeBench | 37 |
| problems after dedup | 8,509 |
| dropped: no passing Python 3 reference | 1,150 |
| dropped: reference too slow | 46 |
| dropped: references disagree | 35 |
| dropped: too few tests | 5 |
| curated | 7,273: train 6,899, eval 166, qualification 208 |

Spot checks: ten curated references replayed by `judge` all pass; of sixty
sampled rejections, most `no passing reference` are primeintellect problems
whose upstream solutions are Python 2 (`print x`, `raw_input`), 1,143 of the
3,100 primeintellect problems. Three of the 276 problems of the 300-row smoke
are absent because dedup merged them with their primeintellect copies.

## v1.1 sources: new problems appended for SFT

`python -m reliquary_competitive_code.build.extend` (stages `select`, `fetch`,
`curate`, `write`; see the module docstring) adds problems to the published v1
dataset without moving any of it: eval and qualification are copied byte for
byte, train keeps its 6,899 rows at their indices and the new problems follow,
sorted by id, every one for SFT (`origin.use = "sft"`). Train is then v1 SFT
`[0, 4181)`, v1 RL `[4181, 6899)` and the appended SFT `[6899, 6899 + n)`,
`Corpus.use_ranges("sft")` giving both SFT ranges.

Sources:

- open-r1/codeforces @`fbe3f6e9` (CC-BY-4.0), Python submissions from
  open-r1/codeforces-submissions @`07e7bf31`: rating 1200 to 2400, contest
  before 2024-08-01, stdin/stdout, not interactive, no checker; official plus
  generated tests.
- ByteDance-Seed/Code-Contests-Plus `ccplus_1x` @`96c85054` (CC-BY-4.0):
  testlib stock checkers our comparison reproduces (`ncmp`, `wcmp`, `lcmp`,
  huge integers, `rcmp6`, and the case-insensitive YES/NO checker, which our
  comparison already matches: kept with no grader change, each expected
  output checked to hold only yes/no tokens); `rcmp4`, `rcmp9` and custom
  checkers dropped. Its Codeforces problems take open-r1's date and rating.
- The v1 rejects with a Python 2 reference (DeepCoder), converted by 2to3
  without `fix_input` plus a literal-reading `_py2_input` (`sources/py2.py`).
  Python 2 submissions of the two sources above are converted the same way
  and tried after the Python 3 ones.

Each new problem carries `origin.reference_kind`:

| kind | meaning | time limit |
| --- | --- | --- |
| `python_reference` | a Python 3 reference passes every kept test (v1's check) | max(v1 rule, upstream x factor) |
| `py2to3` | the passing reference is a 2to3 conversion | as above (v1 rule alone for DeepCoder rows, which have no upstream limit) |
| `upstream_tests_only` | no Python reference passed; the tests are the judge's or upstream-validated (open-r1 official/generated; CC+ with true positive rate 1.0) | upstream x factor |

The factor (`--python-time-factor`, default 4) turns a C++-calibrated limit
into a Python one, between 1 s and `--max-time-limit` (10 s). Measured on the
24 sampled problems with a passing Python reference (accepted submissions,
CPU on our runner over the upstream limit): median 0.05, p90 0.75, p95 1.08,
max 1.17. v1 gives a reference 4x its own CPU; 4x the upstream limit gives a
teacher program the same headroom over the slowest accepted Python. An
`upstream_tests_only` problem has no reference in `references.parquet` (""),
`reference_completion` returns None, and the miners' teacher answers are what
gets graded at export: a problem no answer passes yields nothing. A Python
reference that runs to a wrong answer drops the problem even when the tests
are upstream's (an accepted submission failing the official tests means the
tests or the comparison are not what we think), a 2to3 conversion excepted.

Dedup and decontamination of new problems: exact and MinHash (as v1) against
the 21,294 DeepCoder rows and across new sources; then a 13-gram containment
guard (`--containment`, default 0.3) against that pool and every new problem
kept before, which drops reformatted copies MinHash misses; then 13-gram
overlap >= 0.10 with any LiveCodeBench statement (every version, test.jsonl to
test6.jsonl) or any v1 eval or qualification statement.

Test budget (`--test-bytes`, default 4 MiB of stdin plus expected stdout per
problem, the same value for `fetch` and `curate`; `--max-tests`, default 32):
open-r1's generated tests reach the constraints' maximum sizes (often 1 to 2
MB each), so a small budget leaves fewer than 5 tests. On the 50 sampled
open-r1 problems, those keeping 5 tests or more: 31 at 1 MiB, 34 at 2, 36 at
4, 39 at 6, 41 at 8 (Code-Contests-Plus: 48 to 49 throughout). The count cap
bounds grading time where no reference CPU budget applies. Measured on the
sample: median 29 KB, mean 0.95 MB of tests per new problem, so a decoded
row group of 64 new problems is about 60 MiB on average, 256 MiB at most.

## SFT and RL shares of train

A train problem is for SFT when its id falls in the first 60% of the 64-bit id
space (`layout.use_of`), otherwise for RL. Train is sorted by id, so each use
is one contiguous range, declared in `environment.toml` and served by
`Corpus.use_range`: SFT `[0, 4181)`, RL `[4181, 6899)`. A task's metadata
carries its `use`. The rule depends only on the problem, so a rebuild keeps
every problem on its side.

## Grading rules

- The program is the last fenced block tagged `python`/`py`/`python3` after
  the last `</think>` (an untagged block only when there is no tagged one).
- Output is compared token by token, with two relaxations: `yes`/`no` in any
  case, and decimals within 1e-6 (an integer answer accepts only an exactly
  equal decimal, `2.000000` for `2`).
- Importable modules: `abc`, `array`, `bisect`, `cmath`, `collections`, `copy`,
  `dataclasses`, `datetime`, `decimal`, `enum`, `fractions`, `functools`,
  `heapq`, `itertools`, `math`, `operator`, `queue`, `random`, `re`,
  `statistics`, `string`, `sys` (a reduced shim), `threading`, `time`,
  `typing` (and `__future__`). The prompt states the same list.
- A program that sleeps, deadlocks or blocks scores 0 (`timeout`); only one
  starved on the host's run queue past its limit is `harness_overload`, which
  gives no verdict (grading raises).
- Deterministic grading: the module-level `random` is seeded with a fixed
  value (`guest.RANDOM_SEED`) before every run and the runner pins
  `PYTHONHASHSEED=0`, so the same program prints the same output on every
  grading.

## Known limits

- Problems with several valid answers are excluded: the comparison is exact
  on tokens, there is no checker. Two detectors: the statement says so ("print
  any", "if there are several answers" without a tie-break), counted as
  `dropped_multi_answer`; or a test that feeds one of the statement's example
  inputs expects another output than the statement shows (Codeforces 1433D),
  counted as `dropped_example_mismatch`, and every copy of that problem goes.
  A multi-answer problem whose tests agree with its examples and whose
  statement does not say so still slips through. The second detector also
  drops a few gradable problems whose statement is corrupt (numbers written as
  words, "Ten" for 10, in machine-translated statements).
- A program that reseeds `random` from entropy (`random.seed()`,
  `random.Random()`, a seed from `time`) is not deterministic.
- Python only.
- Deduplication across sources is MinHash on statements; about 161 residual
  near-duplicate pairs remain across splits.
- The local runner is a process with resource limits, not a sandbox; run it
  where an untrusted program may be executed.
