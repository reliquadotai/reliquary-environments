# reliquary-terminal-bench

[Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1) (89 Harbor
tasks: build, debug, recover, configure, in each task's own prebuilt image), graded by
each task's own `tests/test.sh`, **offline**, in a pristine box that receives the agent's
filesystem; the agent's box may reach an allowlist through the gateway's egress proxy (as
upstream lets every task reach the internet). **Evaluation only**: never trained on. **Verifiers-only**, like
`reliquary-terminal` and `reliquary-deepswe`: served on signed-episode sandboxes through
the generic verifiers bridge (`docs/env-norm.md`).

## Provenance

| What | Where | Pin |
| --- | --- | --- |
| Tasks (`reliquary_terminal_bench/tasks/`: `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/`, `solution/`, unchanged) | Harbor registry `terminal-bench/terminal-bench-2-1` ([repo](https://github.com/harbor-framework/terminal-bench-2-1), Apache-2.0) | `sha256:7d7bdc1c…699a`, the leaderboard's pin |
| Upstream images (`upstream_images.json`) | `alexgshaw/<task>:<date>` (79 at 20251031, 10 rebuilt 2026-04) | registry digests, read without pulling |
| Verifier images (`sandbox-images.lock.json`) | ours: `reliquadotai/terminal-bench-2-1:<task>-r2`, in our Hippius registry only | `scripts/build_verifier_images.py` |

`_source.py` records the sha256 of every vendored file; a task whose files do not match
is refused at load. Re-vendor with `scripts/vendor_tasks.py <Harbor cache dir>`.

## Splits

| Split | Tasks | For |
| --- | --- | --- |
| `eval` | 65: gold-checked with the agent's box online through controlled egress, grading offline (`gate.VERIFIED`) | what a gateway with egress serves and a Catalyst eval set freezes |
| `eval_offline` | 49: gold-checked with both boxes offline (`gate.VERIFIED_OFFLINE`) | a gateway without egress |
| `eval_all` | all 89 | the gold check |

## How a task is graded without network

Upstream grades in the agent's own box: `test.sh` installs curl (apt), uv (astral.sh) and
the test dependencies (`uvx -w pytest==… -w numpy==…`, PyPI, PyTorch's index, a few git
clones and downloads), then runs the final-state tests. Our boxes have no network, the
agent's processes are stopped before grading, and grading happens in a pristine box. So:

1. **Verifier image** (`images.py`, `scripts/build_verifier_images.py`): the upstream
   image by digest plus inert files, `/opt/tb-verifier/overlay.tar.gz.<part>` (900 MB parts,
   one layer each) — every path `test.sh` adds or changes when run with network on the
   untouched image, its `test_*.py` stubbed so no test side effect is kept (`docker diff`,
   less `/tests`, `/logs`, `/tmp`, apt lists and caches, and the WORKDIR, except the two
   downloads `test.sh` puts there: `fix-ocaml-gc`'s OCaml clone, `sam-cell-seg`'s
   weights). The image also keeps a file in `/tmp`: gVisor mounts a tmpfs over an empty
   `/tmp` (Docker's runc does not), and a test renaming into `/tmp` then fails (EXDEV).
2. **`setup`** (agent's box) removes `/opt/tb-verifier`: the agent works in the upstream
   image as published. It marks the time work starts.
3. **`finalize`** (agent's box, processes stopped) captures every file, link and directory
   whose inode changed since the mark (`find -cnewer`: ctime, so an mtime restored by
   `tar`, `cp -p` or `touch` cannot hide a change) as a tar with owners and modes, and the
   list of every path (deletions). Both go to `/logs/artifacts` (the only artifact root).
   Pruned: `/proc`, `/sys`, `/dev`, `/run`, `/logs`, `/tests`. Capped at 512 MB
   compressed (`task.MAX_STATE_BYTES`).
4. **`grading_setup`** (pristine box) lists its paths.
5. **`solved`**: unpack the verifier overlay, delete what the agent deleted, unpack the
   agent's files over `/` as root, stage `tests/` at `/tests`, run `bash /tests/test.sh`
   from the image's WORKDIR with `UV_OFFLINE=1`. Its `apt-get`/`curl` calls fail at once
   and change nothing; its tools are already installed. Reward: `reward.json` or
   `reward.txt`, as Harbor reads it. `info.terminal_bench` carries the CTRF counts and the
   output tail.

The WORKDIR is the image's own (`source.workdir`, from the Dockerfile): verifiers'
default `/app` would make three references score 0 (`fix-git`, `sanitize-git-repo`,
`prove-plus-comm`).

**Trust** is Harbor's: the agent's files land as root over the whole filesystem, as in
upstream's same-box grading, so the reward is as tamper-proof as upstream's, not more.
The forced-draw audit is what binds the actions to the model. Do not train on this env.

## What is served: 65 of 89 (49 without egress)

Gold checks of 2026-11-13 on a Targon CPU VM, runsc (`gate.GOLD_CHECK`, `GOLD_CHECK_OFFLINE`):
reference 1 and no-op 0 for **65** tasks with the agent's box online through reliquary-sandbox's
egress proxy (profile `packages` + `gate.EGRESS_EXTRA_HOSTS`, apt snapshot 20261111, PyPI/uv
date pins 2026-11-11), grading offline: the `eval` split. **49** of them pass with both boxes
offline: `eval_offline`. Slowest verified grading 372 s, largest verified state 474 MB
(`train-fasttext`). Conformance cases on the goldens: 10/10. The 24 others, each with its reason
in `gate.py`:

- **10 excluded** (`gate.EXCLUDED`): graded against a process the agent leaves running
  (servers, daemons, VMs).
- **4 over the 512 MB state cap** with the agent's installs (`caffe-cifar-10`, `gcode-to-text`,
  `mteb-leaderboard`, `pytorch-model-cli`).
- **5 references fail with egress** (hosts beyond the allowlist, or the task):
  `build-cython-ext`, `compile-compcert`, `mcmc-sampling-stan`, `protein-assembly`,
  `rstan-to-pystan`.
- **2 need network inside the tests** (`filter-js-from-html`, `reshard-c4-data`): grading is
  offline.
- **3 grade over the 810 s step** on 1 CPU under gVisor (`fix-ocaml-gc`, `query-optimize`,
  `schemelike-metacircular-eval`).
- No task needs a GPU or a compose sidecar (all 89 are single-container, `gpus = 0`).

A score on `eval` is **TB 2.1, offline subset (65/89)** (on `eval_offline`: 49/89), never "TB 2.1".

## Box and budgets

Resources and the agent timeout are `task.toml`'s, unchanged (the leaderboard forbids
changing them): 1-4 CPUs, 2-8 GB, 10 GB disk; agent 360-12,000 s (the bridge's `wall_s`).
Setup 120 s, finalize 600 s, scoring 810 s (the bridge's most). Tools: `bash`.

## Gate: the gold check

```bash
# a build host with Docker, network and our registry on 127.0.0.1:5001
uv run python scripts/build_verifier_images.py --jobs 6 --out build.jsonl
uv run python scripts/build_verifier_images.py --write-lock --from build.jsonl
# gVisor, the verifier images local; offline, then with the agent on the egress network
uv run python scripts/gold_check.py --jobs 6 --noop --out offline.jsonl
uv run python scripts/gold_check.py --write-gate eval_offline --from offline.jsonl
uv run python scripts/gold_check.py --jobs 6 --noop --egress-network rq-egress \
    --egress-env "$(cat egress-env.json)" --out egress.jsonl   # proxy vars + date pins
uv run python scripts/gold_check.py --write-gate eval --from egress.jsonl
```

Per task: agent box (setup, `solution/solve.sh` offline within the agent timeout, every
process killed, finalize), then a fresh grading box (grading_setup, artifacts restored,
`solved`, timed). Verified = reference 1 within 700 s of grading and no-op 0. `--egress-network`
puts only the agent's box on the gateway's egress network, as the sandbox does for
`network.agent`.

## Gateway settings

A gateway serving `eval` needs reliquary-sandbox with controlled egress (7743b57 or later,
`docs/egress.md`) and these bridged-env options:

```json
{"tools": ["bash"], "splits": {"eval": {"split": "eval"}},
 "defaults": {"per_call_timeout_s": 600, "max_artifact_bytes": 570425344},
 "network": {"profile": "packages", "agent": true, "grading": false}}
```

`max_artifact_bytes` is `task.MAX_ARTIFACT_BYTES` (the 512 MB state plus the path list; the
bridge's default 32 MiB is below what references leave). The egress host setup
(`scripts/egress-host-setup.sh <venv> 20261111T000000Z`, the gold check's snapshot) with
`gate.EGRESS_EXTRA_HOSTS` added to the `packages` profile, and
`EPISODE_EGRESS_ENVS={"reliquary-terminal-bench": ["packages"]}`. A gateway without egress
serves `eval_offline` with no `network` option. Verifier images pre-pulled by digest from our
registry (`scripts/sandbox_images.py from-lock`).

## Development

```bash
uv sync --locked
uv run pytest          # no Docker, no network
uv build
```
