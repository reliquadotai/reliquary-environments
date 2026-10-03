# Held-out benchmarks

Public benchmarks that no Reliquary run trains on, for measuring whether a trained
checkpoint learned a skill or learned its corpus. A score on an environment the
policy trained on says the second; a score here can say the first.

Nothing in this directory is an environment. Every benchmark is Prime Intellect's own
native Verifiers v1 taskset from
[research-environments](https://github.com/PrimeIntellect-ai/research-environments),
taken unmodified at commit `f2e0d709` and run on the same Verifiers source this
repository pins (`b2e4e815`). What lives here is the pin, one config per benchmark,
one replacement reward, and a runner.

## What each benchmark is held out from

Teutonic trains on four families: code (OpenCodeInstruct), instruction following
(`reliquary-instruction-following`), DAPO maths and telecom tool use (τ²). Each
benchmark below sits next to one of them without sharing its corpus or its checker.

| Benchmark | Tasks | Draws | Held out from | What a gain there means |
| --- | --- | --- | --- | --- |
| `ifbench` | 300 | 1 | instruction following | The policy follows constraints it never saw. Our environment grades with IFEval's verifiers; IFBench's 58 constraints were written not to overlap them, which is why IFEval itself cannot be used. |
| `bfcl-v3` | 4,441 | 1 | telecom tool use | Tool calling transfers to schemas other than the telecom toolkit, single- and multi-turn. |
| `aime26` | 30 | 8 | DAPO maths | Competition maths on problems set in February 2026. |
| `aime25` | 30 | 8 | DAPO maths | The same, on problems older than the base model. Reported for comparison with published model cards. |
| `livecodebench` | 1,055 | 1 | code | Competitive programming against hidden tests, outside OpenCodeInstruct's distribution. |
| `gpqa` | 198 | 4 | none | A guard: graduate science, which RL on our families should not cost. |
| `mmlu-pro` | 12,032 | 1 | none | A guard over 14 subjects, for the same reason. |

AIME is drawn eight times because thirty problems sampled once cannot separate two
checkpoints: one problem is 3.3 points.

## Run

Serve the checkpoint behind an OpenAI-compatible endpoint, then:

```bash
uv sync --locked
uv run python run.py --model teutonic --base-url http://127.0.0.1:8000/v1
uv run python run.py --model teutonic --base-url http://127.0.0.1:8000/v1 --only ifbench bfcl-v3
```

Every episode runs in Docker. The runner writes `outputs/summary.json` and prints the
mean reward per benchmark; each run's traces are under `outputs/<benchmark>/`.
`--num-tasks 4` makes a smoke run; it takes tasks in order, so on BFCL the first few are
all `irrelevance` tasks, which a model that never calls a tool passes. Sampling defaults to temperature 0.6 and top-p 0.95,
the setting Qwen publishes its thinking-mode numbers with; whatever is chosen, compare
two checkpoints only at the same values.

Run the starting checkpoint through the same suite before the trained one. Every
benchmark here may have been seen in part by the base model's pretraining, but it was
seen equally by both, so the difference between the two is what training did.

## What the runner overrides, and why

Upstream defaults are built for Prime's platform. Each one would have changed the
measurement or sent it somewhere:

- **`--push` is on by default** and uploads the finished run to the Prime platform.
  The runner always passes `--no-push`.
- **The default runtime is a hosted sandbox VM per episode** (`prime`). The runner
  writes a Docker runtime with every destination denied into the config the run
  reads; the CLI cannot spell an empty allow list. The subprocess runtime is not an
  option: these tasks declare a network policy, which it refuses.
- **The default harness is `bash`**, which hands the model shell and edit tools. On
  BFCL a stray `bash` call scores as a wrong function call. Every config sets the
  `null` harness: the model sees the task's own tools and nothing else.
- **GPQA falls back to an LLM judge** whenever its letter extractor finds nothing,
  and that judge defaults to a hosted model. `gpqa_letter.py` replaces the task's
  `correct` reward under the same name: the upstream extractor still decides, and an
  answer it cannot read scores 0.
- **A request without tools leaves the field out.** verifiers' chat mediation turns the
  null harness's `tools: null` into `tools: []`, which vLLM refuses with a 400 on every
  request. `wirefix/sitecustomize.py`, put first on `PYTHONPATH` by the runner (so the
  processes verifiers spawns carry it), drops the empty array.
- **IFBench runs in `strict` mode**, upstream's default being `loose`: a response
  passes only if every instruction checks out as written. `--env.taskset.task.mode
  loose` in `configs/ifbench.toml` gives the published loose figure.

## Before the first run

- **GPQA is gated on Hugging Face.** Accept the terms on
  [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) with the account
  whose token the box uses (`HF_TOKEN` or `hf auth login`); without it the benchmark
  fails at load.
- **LiveCodeBench downloads 4.5 GB** (all six release files) on first load, and keeps
  the official window of contests from 2024-08-01 to 2025-05-01. That window predates
  the base model, so treat it as a comparison between checkpoints, not as unseen code.
- `aime25`, `aime26`, `ifbench`, `mmlu-pro` and `bfcl-v3` were loaded on the pinned
  stack (30, 30, 300, 12,032 and 4,441 tasks). `gpqa` and `livecodebench` resolve
  their configs but were not loaded here, the first for want of a token and the
  second for want of disk.

## Test

```bash
uv run pytest
```

The tests check that every config resolves against the installed tasksets, that the
runner can never omit `--no-push` or the closed Docker runtime, and that the GPQA
reward scores the right letter 1 and anything else 0. They need neither a model nor a
network.
