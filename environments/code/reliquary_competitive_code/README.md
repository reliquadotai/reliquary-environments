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

    uv run python -m reliquary_competitive_code.build.fetch --out raw/
    uv run python -m reliquary_competitive_code.build --deepcoder raw/... --lcb raw/... --out dataset/

`build.fetch` downloads the upstream data; `build` curates it and writes the
dataset (`--workers`, `--limit` for a smoke run; see `--help`).

## Published revision

Drop counts: filled in with the published revision.

## Known limits

- Problems with several valid answers are excluded: the comparison is exact
  on tokens, there is no checker.
- Python only.
- The local runner is a process with resource limits, not a sandbox; run it
  where an untrusted program may be executed.
