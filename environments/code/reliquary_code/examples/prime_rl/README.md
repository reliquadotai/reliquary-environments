# Prime-RL v0.9.0

This is the first pinned Prime-RL training lane for `reliquary-code`. It uses
Prime-RL's native Verifiers environment boundary, Qwen3 renderer, GRPO,
zero-signal group rejection, held-out online evaluation, and native
checkpoint/run evidence. Reliquary does not wrap or fork the trainer.

The taskset is single-turn, tool-free text completion: the model emits a
fenced Python program in one completion, and `reliquary_code`'s own runner
executes that program — in a fresh, resource-limited subprocess per test
case — against each problem's pinned cases (`reliquary_code/runner.py`).
Grading runs code, but the *model* is never given a tool to call mid-rollout,
so the run still uses Prime-RL's plain `null` chat harness rather than a
tool-calling one. `CodeTaskset` exposes one split, `train`, covering the
full pinned OpenCodeInstruct-curated corpus; `orchestrator.eval` reads the
same split and holds itself to `num_examples = 48` per evaluation, so
held-out scoring comes from sampling, not from a second split id.

The reference topology needs two CUDA GPUs: one trainer and one inference
worker. A shared single-GPU trainer/inference process is not supported by the
Prime-RL v0.9.0 launcher and is not claimed here.

## Install the exact stack

Run from a clean working directory on the GPU server. Git LFS and Docker must
be installed. Docker enforces the taskset's `network = false` policy from
`environment.toml`; Verifiers correctly refuses to run that policy with its
unsandboxed subprocess runtime. This matters more here than for a
text-only taskset: `reliquary_code`'s own subprocess runner does **not**
block network access on its own (see the package README's Isolation
section), so Docker's enforcement is the only thing actually closing that
gap in this reference run.

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
  "https://github.com/reliquadotai/reliquary-environments/releases/download/v0.1.0a1/reliquary_code-0.1.0a1-py3-none-any.whl#sha256=15233377bedf462d4521f8580afca431a8466c82f17884cb23bf35e7addb3569"
```

Use the environment's release wheel after `uv sync`; a later sync can remove
it because it is not part of Prime-RL's lock. The commands below call the
virtual-environment binaries directly for the same reason.

Download the exact model snapshot. Prime-RL v0.9.0 accepts a model name or
path but has no revision field, so the local snapshot path is the immutable
boundary:

```bash
MODEL_SNAPSHOT_PATH="$(prime-rl/.venv/bin/hf download \
  Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554)"
```

## Validate, smoke test, then train

Set `RELIQUARY_ENVS_PATH` to this repository checkout.

```bash
PRIME_CONFIG_PATH="$RELIQUARY_ENVS_PATH/environments/code/reliquary_code/examples/prime_rl/rl.toml"

prime-rl/.venv/bin/python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-code'); print(next(iter(vf.load_taskset(c(id='reliquary-code', split='train')))).key)"

prime-rl/.venv/bin/rl @ "$PRIME_CONFIG_PATH" \
  --model.name "$MODEL_SNAPSHOT_PATH" --dry-run True \
  --output-dir outputs --run.name reliquary-code-dry-run

prime-rl/.venv/bin/rl @ "$PRIME_CONFIG_PATH" \
  --model.name "$MODEL_SNAPSHOT_PATH" --max-steps 5 \
  --output-dir outputs --run.name reliquary-code-smoke

prime-rl/.venv/bin/rl @ "$PRIME_CONFIG_PATH" \
  --model.name "$MODEL_SNAPSHOT_PATH" \
  --output-dir outputs --run.name reliquary-code-qwen3-4b-run-1
```

The five-step smoke must produce well-formed fenced-Python completions,
non-zero mixed-outcome groups, a checkpoint, and clean train/eval traces
before the 100-step run. Keep Prime-RL's resolved configs, metrics, traces,
and checkpoint manifests as the evidence bundle; do not invent a parallel
logging format.

## Scientific boundary

The config carries the reusable INTELLECT-3 ideas: group-relative repeated
sampling, exact verifiable rewards, rejection of groups with no learning
signal, bounded off-policy lag, evaluation before training and every 15
steps, and periodic checkpoints. The 32-prompt LoRA run is deliberately sized
for a small proof, not copied from INTELLECT-3's 512-H200 GLM run.

Prime-RL v0.9.0 uses its current DPPO+KL trainer loss with GRPO advantages.
It is therefore accurate to report **Prime-RL v0.9.0 GRPO**, not an exact
reproduction of INTELLECT-3's report-era IcePop loss. Muon, 65K context,
difficulty-pool sampling, and SFT warm-up are excluded until measurements
justify them.

For a public skill-improvement claim, freeze the config and eval indices, run
at least three independent runs, select a checkpoint using only `eval`, then
evaluate `qualification` once. Report Avg@16 and per-family means; pass@16
alone can overstate the gain. Compare the base snapshot and selected
checkpoint with the same renderer, sampling values, limits, and task indices.
`reliquary_code`'s runner and extractor are a port of Reliquary core's
embedded copy (`reliquary/environment/opencodeinstruct.py`), running behind
Docker rather than core's gVisor sandbox; a discrepancy against a core run is
a signal to re-check the port, not to average it away.

Primary references: the [INTELLECT-3 technical report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf), [Prime-RL v0.9.0](https://github.com/PrimeIntellect-ai/prime-rl/tree/v0.9.0), and its [multi-turn training design](https://github.com/PrimeIntellect-ai/prime-rl/blob/v0.9.0/docs/algorithms.md#multi-turn-trajectories).
