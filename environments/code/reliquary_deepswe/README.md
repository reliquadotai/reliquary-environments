# reliquary-deepswe

[DeepSWE v1.1](https://deepswe.datacurve.ai/) (Datacurve): 113 original, long-horizon
software-engineering tasks in TypeScript (35), Go (34), Python (34), JavaScript (5) and
Rust (5), each in its own prebuilt image, graded by the task's own tests in a pristine
box. **Evaluation only**: never trained on. **Verifiers-only**, like `reliquary-swe` and
`reliquary-terminal`: a container cannot be replayed deterministically. Served on
signed-episode sandboxes through the generic verifiers bridge (`docs/env-norm.md`).

## Provenance and license

| What | Where | Pin |
| --- | --- | --- |
| Tasks (`reliquary_deepswe/tasks/`, vendored unchanged) | [datacurve-ai/deep-swe](https://github.com/datacurve-ai/deep-swe) | `8cae5984d5dd0ee37445beff0e928dc10c331116` ("DeepSWE V1.1") |
| License of the task content | same repository, `LICENSE` (Apache-2.0) | added at `f2975d3457b8add60d4ee6e8a65ad3aaffe86fd2`, whose `tasks/` is byte-identical to the pin |
| Per-repository licenses | `reliquary_deepswe/PROVENANCE.md` | upstream `6db64a40f3318d8659238ff34a8cc4b491c49205` |
| Shared verifier (`reliquary_deepswe/verifier.py`) and runner corrections (`overlay.py`) | [PrimeIntellect-ai/research-environments](https://github.com/PrimeIntellect-ai/research-environments) `environments/swe/pi_deepswe` (Apache-2.0) | `b41d8759070715dc102000e06fba0d73ebce9955`, verifier version 17 |
| Images (`sandbox-images.lock.json`) | `public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1` | 113 tags, 106 distinct digests (`scripts/pin_image_lock.py`) |

The package is Apache-2.0 (`LICENSE`, `NOTICE`). Datacurve's license covers its task
specifications, verifiers and curation; the repository code each task patches stays under
its own project's license (all permissive, `PROVENANCE.md`).

`reliquary_deepswe/_source.py` records the sha256 of every vendored file. A bridged env's
identity digests the package's `.py` files only; through that table it binds the task
files too, and a task whose files do not match is refused at load. Re-vendor with
`scripts/vendor_tasks.py <checkout at the pin>`.

## Splits

| Split | Tasks | For |
| --- | --- | --- |
| `eval` | 110 gold-checked tasks (`reliquary_deepswe/gate.py`) | what a gateway serves and a Catalyst eval set freezes |
| `eval_all` | all 113 | the gold check and conformance runs |

`eval` is the gold check's verified list less the excluded tasks (`gate.VERIFIED`,
`gate.EXCLUDED`, a `.py` file so the served set is part of the package identity); without a
recorded list it refuses to load. Task
order is the task id's, sorted; `task_at(i)` is the `i`-th task `load()` yields.

## The task (env norm)

- **Box**: the task's image, `/app` at the base commit, 2 CPUs, 8 GB memory, 20 GB disk,
  no network (all 113 declare `allow_internet = false` for the agent and the verifier).
  Agent timeout 5,400 s (upstream's at the pin; the bridge's `wall_s`). Tools: `bash` and
  `edit`.
- **Prompt**: `instruction.md`, unchanged. It ends with "work on this in a new branch
  from main and commit everything when you are done".
- **`setup`**: checks `/app` is at the base commit (the image is the task's; ours if not)
  and records the files the image ships untracked, so the capture leaves them out.
- **`finalize`**: **everything the agent changed against the base commit, committed or
  not**: its work tree staged over the base tree in a scratch repository (`git add -A`),
  then `git diff --cached --binary <base>`, to `/logs/artifacts/model.patch` (the only
  declared artifact). **A deviation from upstream DeepSWE**, which grades only committed
  work (`git diff <base> HEAD`): uncommitted work counts here, so scores are comparable
  across our harnesses whether or not they commit. The scratch repository's only link to
  the agent's is the object directory (read as an alternate) and the work tree, so no
  hook, config, attribute driver, fsmonitor or redirected `.git` the agent left runs;
  in-tree `.gitignore` files apply, toolchain directories (`node_modules`, `.venv`,
  `.gocache`, `target/`, ...) and the image's untracked files are never captured; a
  planted patch, or a link at `/logs` or `/logs/artifacts`, is removed first. Over
  16 MB, no patch.
- **`grading_setup`**: checks the pristine grading box's `/app` is at the base commit.
- **`solved`** (reward `solved`, 0 or 1), in that box with the patch restored:
  1. an empty patch is 0 without running anything (the fail-to-pass tests fail on the
     base by construction); a patch with a symlink or submodule, or touching a
     toolchain directory (`node_modules`, `.venv`, `site-packages`, `.gocache`, Cargo's
     `target/`, `.git`, ...), is 0 and never applied;
  2. `/tests` is staged (`overlay`): upstream's tests, PI-DeepSWE's shared verifier as
     `grader.py` with upstream's as `upstream_grader.py`, and the runner corrections;
  3. the task's `test.sh` runs: `grader.py prepare` restores test files, fixtures and
     runner hooks the patch touched (and Rust test items inside source files), applies the
     patch then the official test patch, applies the pinned per-task fixture corrections;
     the suites run and write their reports;
  4. the verdict is computed **in the scoring process** from the report files read out of
     the box (`scoring`), with PI-DeepSWE's rule: 1 iff every fail-to-pass test passes
     and no pass-to-pass test fails; a missing test fails; a Python parameter or Go
     loopback-port family may match only when complete and passing. The box's own
     `reward.json` is never read: the candidate's code ran there as root.
  5. A run whose every fail-to-pass test passed and whose only failures are explicit
     timeouts runs twice more when both retries fit in 700 s from the reward's start, and
     recovers only if both pass (PI-DeepSWE's rule; skipped retries are counted in
     `retry_skipped_for_budget`).

  Grading environment: `GOCACHE=/app/.gocache`, `GOFLAGS=-count=1`, `GOPROXY=off` (a
  module fetch fails at once). Go's toolchain and checksum settings stay at their defaults:
  several images pin a newer Go than they ship, and the cached toolchain is used offline.
- **Timeouts**: setup 120 s, finalize 300 s, scoring 810 s: each grading step gets 810 s,
  the most a gateway's verification window admits.
- **Residual**, as for every repository-test grader: the candidate's code runs during its
  tests and can write a report claiming every test passed. Test files it changed are
  restored first; nothing closes the in-process case.

### PI-DeepSWE's corrections, ported

All of v17's are kept: in `verifier.prepare` (in the box), the Pest, SQLfmt, Testem,
Adaptix, Vulture, Mashumaro, Meriyah, Obsidian Linter, Ink, Eicrud (TypeScript rebuild of
`shared/`) and Onedump fixture fixes, each failing visibly when its anchor is not found;
in `overlay` (the staged runner): `kcp-go-multiplexed-kcp-streams` without
`TestMuxPriorityPreemption` (29 fail-to-pass tests remain), Eicrud's `ulimit -n 64000`,
and Go build events kept in `run.log`. Not ported: the Go warm-up (`go mod download all`
in trusted setup, which needs network): a Go task whose image lacks module data fails
offline, and the gold check leaves it out.

## Gate: the gold check

```bash
uv run python scripts/gold_check.py --runtime runsc --jobs 8 --out gold.jsonl
uv run python scripts/gold_check.py --write-gate --from gold.jsonl   # rewrites gate.py
uv run python scripts/gold_check.py --runtime runsc --conformance    # attack cases
```

Run on a gVisor host with the images pulled by digest (a test VM shaped like a gateway,
never a production one). Per task it plays an episode through this package's own hooks:
agent box, `setup`, the reference (`conformance.reference_calls`: the gold patch applied
and committed), `finalize`, then a fresh grading box, `grading_setup`, the patch restored,
`solved`, timed. A task is verified when its gold grades 1 within `--budget` (700 s).
`scripts/static_budget.py` lists the 17 tasks whose runners allow more than 810 s
(per-suite ceilings, 900-2,700 s, all Go but one), the ones to watch: `abs-stepped-slices`,
`anko-default-function-arguments`, `anko-typed-variable-bindings`,
`dasel-html-document-format`, `geo-shapeindex-serialization`,
`goreleaser-retry-publish-auditing`, `helm-array-merge-strategies`,
`kgateway-consistent-hash-policy`, `onedump-dump-encryption-pipeline`,
`python-statemachine-state-data-scoping`, `scc-bounded-memory-spilling`,
`tengo-callable-instance-isolation`, `tengo-destructuring-bindings`,
`termenv-preserve-ansi-resets`, `updo-policy-alerting`, `wazero-multi-module-snapshots`,
`ytt-jsonpath-query-api`.

### Results, 2026-11-13

Targon cpu-xlarge-amd VM (32 vCPU), gVisor `release-20260928.0` with the gateways' runsc
arguments, 12 episodes at a time, no network in any box, images by digest. Each task played
twice: the gold patch (expected 1) and a no-op, one new file no test reads (expected 0, the
suites run on the base state).

- **Served: 110/113.** Gold 1 and no-op 0 within 700 s for 111; one of them is excluded on
  timing. Grading time of the served golds: median 50 s, p90 221 s, slowest 656 s
  (`kgateway-consistent-hash-policy`; then `arcane-drift-detection-baselines` 522 s,
  `skrub-duration-encoding` 513 s, `oxvg-structural-selector-preservation` 390 s).
- **Excluded** (`gate.EXCLUDED`): `expr-try-catch-errors` (a task defect under the shared
  verifier: the gold changes `vm.NewProgram`'s signature and updates `vm/vm_test.go`, which
  the verifier restores as a test file, so 125 pass-to-pass tests no longer compile);
  `numba-stencil-boundary-modes` (gold grading over 810 s; the no-op alone 763 s);
  `prometheus-transactional-reload-status` (685-799 s, over 700 s).
- **No task needs network.** The first pass set `GOSUMDB=off` and `GOTOOLCHAIN=local`; five
  Go tasks (`arcane`, `goreleaser`, `kgateway`, `ytt`, `expr`) pin a newer Go than their
  image ships and failed in 3 s. Their toolchain and its checksum entries are in the
  image's module cache, which Go uses offline with only `GOPROXY=off`: the grading
  environment now sets just that, and all 34 Go tasks were re-run (33 gold 1).
- Conformance cases on both goldens: 13/13 as expected.

Raw results are kept out of the repository; `GOLD_CHECK.results_sha256` digests them.

## Conformance (`conformance_cases`, `reference_calls`)

Every task has a reference (gold applied and committed). On the goldens
(`sqlite-utils-safe-import-checkpoints`, Python; `etree-xml-diff-patch`, Go: the smallest
images; they must be in the verified list): expected 1, the gold applied but not
committed, and the gold with a force-added `node_modules/` file (never captured);
expected 0, the gold planted in `/logs/artifacts`; planted by git settings run during a
capture; the gold committed with a symlink; and on the Python golden a root
`conftest.py` forcing passes. `eval` has none before the
gate.

## Images and disk

113 images, 106 distinct, **111 GB compressed** (0.76-2.55 GB each): a gateway serving
the whole split pre-pulls all of them (`scripts/pin_image_lock.py --list` prints the
digests). `--verify` re-resolves every tag and exits 1 when one moved.

## Development

```bash
uv sync --locked
uv run pytest          # no Docker, no network
uv build
```
