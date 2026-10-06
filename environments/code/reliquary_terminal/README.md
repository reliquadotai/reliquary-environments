# reliquary-terminal

Terminal tasks: a policy gets a container, an instruction and a shell, and
is rewarded when the task's own tests pass. Two splits, graded differently
on purpose (design spec `docs/superpowers/specs/2026-09-23-reliquary-terminal-env-design.md`,
section 4, option C):

| Split | Corpus | Graded | Why |
| --- | --- | --- | --- |
| `eval` | Terminal-Bench 2.1, 89 tasks, pinned by digest | in the agent's own box, as it ships | the number stays comparable to published results |
| `train` | MiMo-V2.6-RL-oss's 64 Terminal-Bench-format tasks | in a fresh box that receives only the agent's `/app` | a training reward must not be reachable by editing the grader |

`--taskset.split` has no default, so a training source cannot fall back to
the evaluation set by omission.

Like `reliquary-swe`, this package defines no tools of its own: the harness
supplies the shell. It builds on `verifiers`' own Harbor integration
(`HarborTask`, `HarborEnv`) rather than reimplementing it, and adds three
things a 2026-10-02 qualification run showed were missing (below): a
per-command timeout, per-test grading detail, and removal of the containers
an interrupted run leaves behind.

## The shell: `bash` with a per-command timeout

The package's default harness, `reliquary-terminal`, is `verifiers`' `bash`
harness with one function replaced. Upstream runs every command with a fixed
3,600 s timeout that kills `bash` alone; in the qualification run one `grep`
over a whole tree held its rollout for 1,489 s. Here a command that outlives
`--env.agent.harness.command-timeout` (default 180 s) has its process group
killed, and the agent gets its output so far followed by
`[command timed out after 180 s; its process group was killed. ...]`; only the
last MiB of each output stream is kept. 180 s is 18x the
longest of the 2,120 terminal commands in that run's traces (10 s;
`environment.toml`, `command_timeout_seconds`). `--env.agent.harness.id bash`
still selects the upstream harness, without the timeout.

## Grading detail

Every task of both splits runs pytest with `--ctrf /logs/verifier/ctrf.json`
before writing 1 or 0 to `reward.txt`. The reward is still `reward.txt`; the
trace that carries it also gets `info["grading"]` -- `test.sh`'s exit code,
the tail of its output (where `anti_hack_guard.py` explains a rejection),
and each test's name, status and failure message -- and the metrics
`tests_total`, `tests_passed`, `tests_failed`. A report already in the box
is deleted before `test.sh` runs, so a missing one reads as missing.

## Containers left behind

`verifiers` removes a rollout's container on a clean exit and on a first
Ctrl-C or SIGTERM, but not when the process is SIGKILLed -- which its own
worker pool does to a worker still tearing down 10 s after an interrupt --
and its containers (named after the rollout's trace id) carry no label. So every process that sets up
a box for this package records the container's name in a ledger under
`~/.cache/reliquary-terminal/containers/`, and a detached guardian removes
them once that process is dead, however it died. Ledgers whose guardian died
too are reaped when the environment next starts on the host, or by hand:

```bash
uv run python -m reliquary_terminal.containers list   # ledgers and their containers
uv run python -m reliquary_terminal.containers reap   # remove those of dead processes
```

Only names shaped like verifiers' boxes (32 hex, or `vf-` + 12 hex), from
ledgers this user wrote on this host and in this pid namespace, are ever
removed. A process that cannot read its own start time or pid namespace
(no `/proc`) records nothing and starts no guardian: "cannot tell" always
reads as alive.

## `eval`: Terminal-Bench 2.1

`terminal-bench/terminal-bench-2-1@sha256:7d7bdc1c…` — revision 6 on the
Harbor registry when pinned, downloaded by the Harbor CLI with no
credentials. One correction, carried because it is not optional:

**The image's own `WORKDIR`.** `verifiers`' `DockerConfig` defaults
`workdir` to `/app`, which overrides the image's `WORKDIR` whenever
`task.toml` declares none — and none of the 89 do. Three tasks work
elsewhere (`fix-git` in `/app/personal-site`, `sanitize-git-repo` in
`/app/dclm`, `prove-plus-comm` in `/workspace`), and for those even the
reference solution scores 0. `image_workdir` reads the last `WORKDIR` of the
task's own Dockerfile instead. `tests/test_eval_goldens.py` pins `fix-git`'s
reference solution at 1.0; with the default `/app` it was observed to score
0.0.

Everything else is as shipped, including same-box grading and open network
access for all 89 tasks — deliberately, so the score means what a published
Terminal-Bench 2.1 score means.

## `train`: MiMo-V2.6's terminal tasks

The `terminal_bench` rows of `XiaomiMiMo/MiMo-V2.6-RL-oss`'s `general`
config, pinned at `639865fd…`. Categories: 18 software, 18 ML, 15 security,
4 operations, 3 each science, media and hardware — nearly all repairs in a
vendored real project (Snakemake, Bandit, Volatility3, Transformers, ...).
One image per task, `xiaomimimo/mimo-v2.6-rl-oss:general-agent-env-<n>`,
about 0.3 GB compressed each.

These tasks were written for separate grading, and this split grades them
that way:

- **The tests are not in the image.** Each row carries its `tests/`
  base64-encoded; `materialize_tests` writes them to a local cache, and they
  reach only the grading box.
- **Only `/app` travels.** The agent's `/app` (about 2 MB on the image
  checked, far under `verifiers`' 32 MB artifact cap) is collected when it
  finishes and restored into a fresh box of the same image.
- **The tests guard the tree they receive.** Each task ships
  `anti_hack_guard.py`, which runs first and scores 0 if a pytest or
  interpreter hook the pristine tree never had appears under `/app`, or a
  protected file's hash changed. `tests/test_train_goldens.py` checks this
  with a correct fix plus a planted `conftest.py`: 0, with the plant
  confirmed present in the artifact.
- **No network**, in either box, as every row declares.

One `verifiers` behaviour this split works around: `restore()` deletes each
artifact root before writing the archive with `docker exec --workdir
<workdir>`, so a grading box whose workdir *is* the artifact root (`/app`)
fails every restore. The grading box therefore works in `/`; the tests
address `/app` and `/tests` by absolute path only.

**No reference solutions are published** for these tasks.
`tests/reference_solutions/` holds one, written for this package, for
`candidate-0036` (Snakemake's wildcard-expanded output flags); it scores 1.0
through the whole separate-box path. Whether the other 63 are solvable is
unmeasured, and so are the turn and token budgets — as for `reliquary-swe`,
that is what a first pilot measures.

## Test

```bash
uv sync --locked
uv run pytest
```

Container tests are marked `@pytest.mark.docker` and run only with
`RELIQUARY_DOCKER_TESTS=1` and a reachable Docker daemon. They pull two images: `alexgshaw/fix-git` and
`general-agent-env-1`.

## Train

See `examples/prime_rl/`: `rl.toml` evaluates on Terminal-Bench 2.1 and
carries no train source; `train-sources.toml` supplies the `train` split.
Sixty-four tasks is small — meant as one component of a mix (e.g. with
`reliquary-swe`'s sources, weighted by `ratio`), not a run on its own.

## Signed sandboxes

`reliquary_terminal.sandbox:sandbox_task` serves the `train` split on a signed-episode
sandbox gateway (`episode_envs = {"reliquary-terminal": "reliquary_terminal.sandbox:sandbox_task"}`).

- Served: `train` rows only, 17 of the 64. Refused: `eval` (graded in the agent's box by
  design), rows that need the network, three rows that put `/app` on `PYTHONPATH`, and rows
  whose tests run agent code inside pytest (`in_process_agent_code`). tmax is not served yet.
- State handed to the grader: the artifact roots that exist (`/app`), through the sandbox's
  `archive`, framed with the list of present roots; a deleted root is graded, not an error.
  The grading box restores them once, then stages `/tests`.
- Grading refuses links into `/tests`, `/logs`, `/proc`, `/dev` and `/sys`, ignores
  `reward.json`, requires a complete CTRF report for a reward of 1, and stops stray processes
  after `test.sh`. Residual: a subprocess left by a served row can race that stop; only separate
  uids close it.
- Needs reliquary-sandbox at `e0217aa` or later (`stop_processes`); import fails otherwise.

Failure contract, limits and the parity table: `docs/sandbox-tasks.md` (repository root).
