# Prime-RL v0.9.0

This is the first pinned Prime-RL training lane for `reliquary-swe`. It uses
Prime-RL's native Verifiers environment boundary, Qwen3 renderer, GRPO,
zero-signal group rejection, held-out online evaluation, and native
checkpoint/run evidence. Reliquary does not wrap or fork the trainer.

The taskset is multi-turn and tool-using: `reliquary_swe` defines no tools of
its own (see the package README), so this config drives the policy through
Verifiers' `bash` harness — a minimal coding-agent loop offering `bash` and
`edit` as OpenAI-style function-calling tools inside the task's container.
Training across a different harness (`mini_swe_agent`, `claude_code`, ...) is
a configuration change to `env.agent.harness.id`, not a code change.

The reference topology needs two CUDA GPUs: one trainer and one inference
worker. A shared single-GPU trainer/inference process is not supported by the
Prime-RL v0.9.0 launcher and is not claimed here.

## This example does not train on SWE-bench Verified for real

`reliquary-swe`'s `eval` split is SWE-bench Verified, pinned in
`environment.toml`. It is an evaluation set and must never be trained on
(design spec section 8, `docs/superpowers/specs/2026-09-22-reliquary-swe-env-design.md`).
Training comes from SWE-smith instead (`env.taskset.split = "train"`,
`num_images`/`max_test_count` choosing its size and cost bound — see the
package README's Corpus section). This file still ships with no real
`[[orchestrator.train.source]]`, on purpose: wiring one up is a decision
for whoever configures a real run, not something this mechanical-smoke
example should default to.

A loud comment is not a control: an example is a template people copy, and
nothing in a comment stops `cp rl.toml x.toml && rl @ x.toml --max-steps
5000` from training on it for real. So `rl.toml` ships with **no
`[[orchestrator.train.source]]` at all**. Prime-RL's `TrainConfig.source`
defaults to an empty list — pydantic raises nothing on its own — but
`TrainSource.__init__` does: `if not self.envs: raise ValueError("TrainSource
needs at least one train env")`, before any rollout runs. Verified directly
against the exact pinned commit (`ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`):
`TrainConfig` has no non-empty-sources validator (unlike `EvalConfig`, which
has one), so the *only* thing standing between a bare copy of this file and
training on SWE-bench Verified is that guard. A missing key that trips a
real exception is a control; a comment on a key that is present is not.

The smoke commands below instead supply a source from a second, separate
file, `smoke-only-eval-split-source.toml`, via Prime-RL's own nested
`@`-file CLI syntax: `--orchestrator.train @
smoke-only-eval-split-source.toml` loads that file and merges it under
`orchestrator.train`, adding `source` without touching `rl.toml`'s own
`filter_zero_advantages`/`sampling` (verified against the exact pinned
`pydantic-config` commit `65b15dffba82d4be19efdaf8b2b9705cc1756be8`'s
`_process_args`/`_deep_merge`, and against these two actual files, by
running that library's merge directly and inspecting the result). That
file's own header repeats the warning, so a reader who opens only it still
sees why it exists and why it is not `rl.toml` itself.

This is a mechanical-loop check only — config resolution, container
provisioning, the `bash` harness's tool calls, reward computation, a
checkpoint write, run with `--dry-run` and a handful of `--max-steps`. This
README stops there deliberately. It gives no full-run command, unlike this
repository's other Prime-RL examples, and none should be added — to either
file — until a real training corpus exists.

## Install the exact stack

Run from a clean working directory on the GPU server. Git LFS and Docker must
be installed. Docker enforces the taskset's `network = false` policy from
`environment.toml`; Verifiers correctly refuses to run that policy with its
unsandboxed subprocess runtime. This matters doubly here: grading itself
provisions a *second* container per rollout (see the package README's
Isolation section), so the GPU server's Docker daemon needs enough disk for
the SWE-bench Verified instance images this taskset's tasks name — sized in
gigabytes per repository family, not megabytes. Set `RELIQUARY_ENVS_PATH` to
a checkout of this repository (`reliquary-environments`) before running the
block below.

```bash
git clone --branch v0.9.0 --depth 1 \
  https://github.com/PrimeIntellect-ai/prime-rl.git
git -C prime-rl -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive --depth 1
test "$(git -C prime-rl rev-parse HEAD)" = "ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1"
test "$(git -C prime-rl/deps/verifiers rev-parse HEAD)" = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
test "$(git -C prime-rl/deps/renderers rev-parse HEAD)" = "cb8243913702367878427c7a7094b350ea1a8e20"
test "$(git -C prime-rl/deps/pydantic-config rev-parse HEAD)" = "65b15dffba82d4be19efdaf8b2b9705cc1756be8"
test "$(git -C prime-rl/deps/prime-envs rev-parse HEAD)" = "26dafdc9582576975ec576f893be7319028daf51"
uv sync --project prime-rl --frozen --package prime-rl --extra gpu --no-dev
uv pip install --python prime-rl/.venv/bin/python --no-deps \
  -e "$RELIQUARY_ENVS_PATH/environments/code/reliquary_swe"
```

`reliquary-swe` has never had a tagged release, so there is no wheel to fetch
from a GitHub release yet — installing from a release-wheel URL is not
possible. The command above installs the package directly from its
directory in this repository instead, the same source `uv sync --locked` in
the package's own README installs.

Use the environment's package after `uv sync`; a later sync can remove it
because it is not part of Prime-RL's lock. The commands below call the
virtual-environment binaries directly for the same reason.

Download the exact model snapshot. Prime-RL v0.9.0 accepts a model name or
path but has no revision field, so the local snapshot path is the immutable
boundary:

```bash
MODEL_SNAPSHOT_PATH="$(prime-rl/.venv/bin/hf download \
  Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554)"
```

## Validate, then smoke test

Set `RELIQUARY_ENVS_PATH` to this repository checkout.

```bash
PRIME_CONFIG_PATH="$RELIQUARY_ENVS_PATH/environments/code/reliquary_swe/examples/prime_rl/rl.toml"
SMOKE_SOURCE_PATH="$RELIQUARY_ENVS_PATH/environments/code/reliquary_swe/examples/prime_rl/smoke-only-eval-split-source.toml"

prime-rl/.venv/bin/python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-swe'); print(next(iter(vf.load_taskset(c(id='reliquary-swe', split='eval')))).key)"

prime-rl/.venv/bin/rl @ "$PRIME_CONFIG_PATH" \
  --orchestrator.train @ "$SMOKE_SOURCE_PATH" \
  --model.name "$MODEL_SNAPSHOT_PATH" --dry-run True \
  --output-dir outputs --run.name reliquary-swe-dry-run

prime-rl/.venv/bin/rl @ "$PRIME_CONFIG_PATH" \
  --orchestrator.train @ "$SMOKE_SOURCE_PATH" \
  --model.name "$MODEL_SNAPSHOT_PATH" --max-steps 5 \
  --output-dir outputs --run.name reliquary-swe-smoke
```

Running `rl @ "$PRIME_CONFIG_PATH"` without the `--orchestrator.train @
...` override — including a naive copy of this file — fails fast with
`TrainSource needs at least one train env`, by design (see above).

The five-step smoke must produce well-formed `bash`/`edit` tool calls, a
captured patch per rollout, non-zero mixed-outcome groups, a checkpoint, and
clean train/eval traces. Keep Prime-RL's resolved configs, metrics, traces,
and checkpoint manifests as the evidence bundle; do not invent a parallel
logging format. There is no further "then train" step here — see "This
example does not train on SWE-bench Verified for real", above.

## Scientific boundary

The config carries the reusable INTELLECT-3 ideas: group-relative repeated
sampling, exact verifiable rewards, rejection of groups with no learning
signal, bounded off-policy lag, evaluation before training, and periodic
checkpoints. It is deliberately sized for a small mechanical proof, not
copied from INTELLECT-3's 512-H200 GLM run — and, per the section above, it
is not sized or scoped for a real training claim at all yet.

Prime-RL v0.9.0 uses its current DPPO+KL trainer loss with GRPO advantages.
It is therefore accurate to report **Prime-RL v0.9.0 GRPO**, not an exact
reproduction of INTELLECT-3's report-era IcePop loss.

Every number this config carries that is not pinned by measurement is named
as such, in this file and in `rl.toml` itself: the per-turn token budget and
the turn budget. The training corpus is not a number to caveat — it is
absent from `rl.toml` outright, and `smoke-only-eval-split-source.toml`
supplies `eval` for the mechanical smoke commands only, never for a real
run, pending SWE-smith. None of this should be read as derived; see
`environments/code/reliquary_swe/README.md` and the design spec's section
10 for what is actually settled and what is still open.

Primary references: the [INTELLECT-3 technical report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf), [Prime-RL v0.9.0](https://github.com/PrimeIntellect-ai/prime-rl/tree/v0.9.0), and its [multi-turn training design](https://github.com/PrimeIntellect-ai/prime-rl/blob/v0.9.0/docs/algorithms.md#multi-turn-trajectories).
