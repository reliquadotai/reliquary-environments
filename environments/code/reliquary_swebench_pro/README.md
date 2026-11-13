# reliquary-swebench-pro

[SWE-bench Pro](https://scale.com/leaderboard/swe_bench_pro_public) V2 (Scale AI), the public
set: 642 issue and feature tasks from eleven production repositories in Go (256), Python
(237), JavaScript (145) and TypeScript (4), each in its own prebuilt image, graded by the
repository's own tests in a pristine box. **Evaluation only**: never trained on.
**Verifiers-only**, like `reliquary-swe` and `reliquary-deepswe`: served on signed-episode
sandboxes through the generic verifiers bridge (`docs/env-norm.md`).

V2 (2026-09-22) is the dataset's default config and supersedes the original 731-task
public set: 89 tasks dropped, none added, images sanitised (no future commits), tests and
parsers revised. Upstream's gate graded the gold patch 1 and the empty patch 0 on all 642,
with network for the verifier.

## Provenance and license

| What | Where | Pin |
| --- | --- | --- |
| Tasks (`reliquary_swebench_pro/tasks/<id>/`, Harbor format, vendored unchanged) | [scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) `v2/tasks` | `66f92766bba642462d4bbe5479e83f91f9211862` (tag `v2.0.0`), each file checked against upstream's `v2/SHA256SUMS` |
| `instance.json` per task (repo, base commit, language, `hard`) | [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro), default config | `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50` |
| Images (`sandbox-images.lock.json`) | `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` | 642 digests, 1,051 GB compressed (544 GB of distinct layers) |

The package is Apache-2.0 (`LICENSE`); Scale's harness and task packaging are MIT, and each
task's patches and problem statement stay under their repository's own license (`NOTICE`).
`reliquary_swebench_pro/_source.py` records the sha256 of every vendored file: a bridged
env's identity digests the package's `.py` files only, and through that table it binds the
task files too; a task whose files do not match is refused at load. Re-vendor with
`scripts/vendor_tasks.py <checkout at the pin> <parquet at the revision>`.

## Splits

| Split | Tasks | For |
| --- | --- | --- |
| `eval` | the gold-checked tasks (`reliquary_swebench_pro/gate.py`) | what a gateway serves and a Catalyst eval set freezes |
| `eval_all` | all 642 | the gold check and conformance runs |

`eval` refuses to load until the gold check has recorded its verified list. Task order is
the `instance_id`'s, sorted; `task_at(i)` is the `i`-th task `load()` yields.

## The task (env norm)

- **Box**: the task's image, `/app` at the base commit, 1 CPU, 4 GB memory, 10 GB disk
  (upstream's), no network. Agent timeout 3,000 s (upstream's). Tools: `bash` and `edit`.
- **Prompt**: `instruction.md`, unchanged (problem statement, requirements, new
  interfaces).
- **`setup`**: checks `/app` is at the base commit and records the files the image ships
  untracked, so the capture leaves them out.
- **`finalize`**: everything the agent changed against the base, committed or not
  (Scale's locked protocol: `git add -A; git diff --cached`), to
  `/logs/artifacts/model.patch`, the only declared artifact. Captured as
  `reliquary-deepswe` does: the agent's work tree staged over the base tree in a scratch
  repository whose only link to the agent's is its object directory, so no hook, config,
  attribute driver, fsmonitor or redirected `.git` the agent left runs; toolchain
  directories (`node_modules`, `.venv`, `.gocache`, ...) and the image's untracked files
  are never captured; a planted patch or link is removed first. Over 16 MB, no patch.
- **`grading_setup`**: checks the pristine grading box's `/app` is at the base commit.
- **`solved`** (reward `solved`, 0 or 1), in that box with the patch restored:
  1. an empty patch is 0 without running anything; a patch with a symlink or submodule,
     a path outside the repository, or touching a toolchain directory is 0 and never
     applied;
  2. the patch is applied as Scale's `patch_replay.py` does: `git apply --verbose`, else
     `git apply --3way`, else `patch --fuzz=3 -p1` (with `--batch`); none applying is 0;
  3. upstream's `tests/` is staged as `/tests` and the task's `test.sh` runs unchanged:
     it resets every test file to the base, applies the test patch, runs the task's
     `run_script.sh` on the selected tests and keeps the raw logs;
  4. the verdict is computed **in the scoring process**: the logs are read out of the box
     (32 MB caps), the task's own `parser.py` runs on them there (its pinned bytes, its own
     `export_to_json`, so the `output.json` is the box's), and upstream's rule applies: 1
     iff every fail-to-pass and pass-to-pass test is PASSED. The box's own `reward.txt`
     and `output.json` are never read: the candidate's code ran there as root.
- **Timeouts**: setup 120 s, finalize 300 s, scoring 810 s (the most a gateway's
  verification window admits; upstream's verifier allows 3,000 s).
- **Residual**, as for every repository-test grader: the candidate's code runs during its
  tests and can print output that parses as passing. Upstream's rule also ignores a
  FAILED entry for a test reported PASSED elsewhere, and 313 tasks have no pass-to-pass
  tests (Scale issue #125): counted (`required_also_failed`), not scored.

## Gate: the gold check

Upstream's verifier has network; ours does not, and grading must fit 810 s. A task is
served only when its gold patch grades 1 and a no-op patch (one unused new file, so every
test runs on the base code) grades 0, offline, through this package's own hooks, grading
within 700 s:

```bash
uv run python scripts/gold_check.py --runtime runsc --jobs 7 --noop --prune --shard 0/5 --out gold-0.jsonl
uv run python scripts/gold_check.py --write-gate --from gold-0.jsonl,gold-1.jsonl,...   # rewrites gate.py
uv run python scripts/gold_check.py --runtime runsc --conformance                       # attack cases
```

Run on a gVisor host shaped like a gateway (a test VM, never a production one). Each task's
image is pulled by digest and, with `--prune`, removed after its cases.

## Conformance (`conformance_cases`, `reference_calls`)

Every task has a reference (gold applied and committed). On the goldens (qutebrowser
`77c35579`, Python; flipt `406f9396`, Go: the smallest images of their language), expected
1: the gold applied but not committed, and the gold with a force-added `node_modules/` file
(never captured); expected 0: the gold planted in `/logs/artifacts`; planted by git
settings run during a capture; the gold committed with a symlink; and on the Python golden
a root `conftest.py` forcing passes (reset by `test.sh` as test code). `eval` has none
before the gate.

## Images and disk

642 images, 1,051 GB compressed, 544 GB of distinct layers (median 1.13 GB, largest
6.52 GB, ProtonMail): a gateway serving the whole split pre-pulls all of them
(`scripts/pin_image_lock.py --list` prints the digests). `--verify` re-resolves every tag
and exits 1 when one moved.

## Development

```bash
uv sync --locked
uv run pytest          # no Docker, no network
uv build
```
