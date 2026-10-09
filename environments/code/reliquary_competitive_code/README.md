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
