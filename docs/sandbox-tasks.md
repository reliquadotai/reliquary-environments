# Running env packages on signed-episode sandboxes

This page is for authors of env packages and for the people who qualify them. It describes what `reliquary_swe.sandbox` and `reliquary_terminal.sandbox` do today. The sandbox side (gateway, records, grader) lives in `reliquary-sandbox`, a private repository; its docs are `docs/deployment.md` and `docs/episodes-client.md` there, and the hook contract is the module docstring of `src/reliquary_sandbox/episode_task.py`.

## 1. What a sandbox task is

- The entry point is `sandbox_task(split, index) -> SandboxTask`. A gateway lists it as `episode_envs = {"<env>": "<module>:sandbox_task"}`: `reliquary-swe` is `reliquary_swe.sandbox:sandbox_task`, `reliquary-terminal` is `reliquary_terminal.sandbox:sandbox_task`.
- The gateway imports env packages in its own process, next to its signing key. Only pinned wheels (sha256 checked) are installed there, and a package change is a deployment, done with no live episode (a restarted episode whose task fingerprint changed ends `aborted`, reason `task_changed`).
- `reliquary-sandbox` is imported lazily, inside `sandbox_task` and the hooks (`grade`). It is private, so these public packages never declare it as a dependency. Tests that need it use `pytest.importorskip`.
- Each package also exports `sandbox_prompt(split, index)` (the first user message), `declaration(split, index)` (the task without importing the sandbox), and `python -m <package>.sandbox images`.

Source: `environments/code/reliquary_swe/reliquary_swe/sandbox.py`, `environments/code/reliquary_terminal/reliquary_terminal/sandbox.py`.

## 2. The three hooks and the failure contract

`prepare` runs in the agent's box before record 0. `extract` runs in the agent's box at finish and returns the state bytes. `grade` runs in a pristine box of the same image with that state and returns a `GradeResult(reward, facts)`.

The contract below is copied from `reliquary_sandbox/episode_task.py`:

| Outcome | When | Whose |
| --- | --- | --- |
| `OPEN_FAILED` | `prepare` raises. No record 0 is written and the session token is burnt. | Ours |
| `ABORTED` | Only infrastructure: a sandbox `InfraFault` at any point of `extract` or `grade` (even one the hook caught or re-raised as another exception), every pristine-box attempt failing to start, a failure of the sandbox's own code before the hook runs, or a hook calling a runtime member the sandbox does not offer (`prepare_uv_script`, `run_uv_script`, `open_process`, `run_background`). Void; the task may be re-run. | Ours |
| `BOX_FAILED` | The agent's box failed under `extract` because of what the agent did (killed its processes, exhausted its pids). Not paid. | Agent's |
| `GRADED 0.0` | Everything else, with a sandbox fact under the reserved `_grader` key: an exception raised by `extract` or `grade` (`extract_failed`, `grading_failed`, logged and counted per env), a step past `grading_timeout_s` (`extract_timeout`, `grading_timeout`), output or state over its bound (`state_too_large`, `grading_output_overflow`), a non-regular file or refused archive member (`state_unreadable`), a `BoxFault` in the pristine box (`grading_box_failed`), a `grade` result that is not a `GradeResult` with a finite reward in [0, 1] and canonical-JSON facts without `_grader` (`invalid_grade_result`). Facts over 64 KiB canonical are replaced by `{"facts_truncated": true, "_grader": {...}}`. | Agent's outcome |

The rule for env code: return agent-attributable outcomes, and raise only for "the pristine image is not what the corpus says". An exception is graded 0.0 `extract_failed` or `grading_failed` and counted per env by the gateway, so a real env bug shows up as a counter, not as voided episodes. Watch that counter.

Examples from this repo:

- SWE `extract`: a git refusal in the agent's box, or a box whose git no longer answers because the agent wrecked its workdir, returns an empty diff. An untracked-list file the agent deleted, replaced or inflated is an empty list. The step deadline (`TimeoutError`) and real box faults are never swallowed.
- SWE `grade`: a patch touching a path outside the tracked tree returns reward 0 with the refusal in `test_output_tail`.
- Terminal `extract`: a root the agent deleted (or a dangling link) is listed as absent, graded and never an error.
- Terminal `grade`: links that resolve into the grader return `GradeResult(0.0, {"link_into_grader": [...]})`; a reward that is not a finite number in [0, 1] returns `GradeResult(0.0, {..., "reward_out_of_range": ...})`; the state header is written by `extract`, so a header `decode_state` rejects is our bug and raises (graded `grading_failed`, counted).

Source: `reliquary-sandbox` `src/reliquary_sandbox/episode_task.py` and `docs/deployment.md` ("How an episode ends"), plus the two `sandbox.py` files.

## 3. Moving state

- SWE: the state is a diff (`vf.capture_patch`). Terminal: `runtime.archive(roots, max_bytes)` in `extract`, `runtime.restore_archive(data, roots)` in `grade`. Never use the image's `tar`, or `git` run as root on state the agent controlled.
- `archive` is built by the sandbox's own busybox: symlinks are stored and never followed, member names are `<i>/<path>` relative to root `i`, and a manifest comes first. A special file or unreadable file makes it raise (graded `state_unreadable`).
- `restore_archive` vets and rebuilds the bytes before extraction: `..` in names, members under a symlink member, FIFOs and devices are refused; owners and setuid/setgid bits are dropped; each root is cleared first. Symlink targets are resolved against the archive's own links (at most 40 hops) and must end inside the roots. Absolute targets that end outside the roots are kept as inert links.
- Symlinks between roots must therefore be absolute: a relative walk that leaves the roots is refused.
- Restore each root once, into the pristine box, after the task's own grading-box setup and before `/tests` are staged (this is the order of verifiers' `HarborEnv._grade`). A second restore into a root, or into a root nested in it, aborts the episode as an integration bug.
- `stop_processes`: `await runtime.stop_processes()` SIGKILLs every process in the box except PID 1, the entrypoint and the helper's own chain (bounded rounds, as root, from `/`). It exists so that a result agent-influenced code could still rewrite is read only after nothing can run. Terminal grading calls it the moment `test.sh` returns, before reading `reward.txt` and the CTRF report. Processes that keep respawning are a `BoxFault` (graded 0, `grading_box_failed`); a failure on our side is `aborted`. The sandbox runs the same helper on the agent's box before `extract`. Terminal's sandbox module requires a sandbox at or after `e0217aa` and refuses to import against an older one.
- `max_state_bytes`: the bound the sandbox enforces on the state it reads. SWE sets 8 MiB (`capture_patch` caps a patch at 2 MB, and its UTF-8 re-encoding can grow it 3x). Terminal sets 32 MiB of archive plus 64 KiB of header. Over the bound the episode is graded 0.0 `state_too_large`. Measured honest terminal state: 110 KB to 7.6 MB.
- Terminal state framing: `reliquary-terminal-state/1\n` + `{"roots": [present roots]}\n` + the archive. A root listed absent is removed in the grading box (verifiers' `required=False`).
- SWE state: the diff against `refs/reliquary/base`, a ref `prepare` writes after the cleanup, never bare `HEAD`, so an agent that runs `git commit -am fix` still has its fix in the diff. The image's untracked files are recorded in `.git/reliquary-untracked`. Both live in the agent's box and the agent can rewrite them, which is why grading, not the capture, is what makes the diff safe (section 4).

Source: `reliquary-sandbox` `src/reliquary_sandbox_service/episodes/state_archive.py` and `episode_task.py`; the two `sandbox.py` files.

## 4. What each env refuses and enforces

### reliquary-swe

- Splits: `train` and `train:<num_images>` (SWE-smith), `r2e`, `polyglot`. `eval` (SWE-bench Verified) is never served.
- Every split refuses a patch that touches a path that is untracked or ignored at the base, or lies under `.venv/` (`grading._forbidden_patch_paths`). The diff is the agent's to shape: its git config (`diff.external`) and base ref are in its box, so any path can appear. The refusal runs in the pristine box before the patch is applied and returns reward 0 with the paths in `test_output_tail`. This used to be R2E only; it now covers every split.
- The diff is taken against `refs/reliquary/base`. The image's untracked files are left out of it.
- Polyglot images are pinned in `reliquary_swe/polyglot_digests.json`: all 2,698 rows. A row whose image has no pin is refused at resolve, by every entry point (`row_for`, `declaration`, `sandbox_task`, `sandbox_images`).
- SWE-smith rows are resolved one at a time (`corpus.swesmith_row_at`, `corpus.swesmith_order`), because building all 18,546 rows keeps about 4.1 GB resident.

### reliquary-terminal

- Splits: `train` only (MiMo's 64 Terminal-Bench-format rows). `eval` (Terminal-Bench 2.1) grades in the agent's own box by design and is never served. tmax is not served yet (see section 9).
- Refused rows: rows that need the network (`allow_internet`); three rows whose `test.sh` puts `/app` code on `PYTHONPATH` (`UNSERVED`: the grader would import the agent's code); and 45 rows tagged `in_process_agent_code` (their tests run agent code inside pytest's own process, which can write a complete passing report and `os._exit(0)`). 17 of the 64 rows are served.
- CTRF completeness: in the grading box a nonzero reward needs the CTRF report to show no failed test, at least one passed, and every `test_*` the row's test files collect among the passed ones; otherwise reward 0, recorded as `ctrf_incomplete`. A nonzero `reward.txt` next to a failing `test.sh` exit is 0 (`reward_without_success`).
- `reward.json` is ignored for every split. No MiMo `test.sh` writes it, so one in the box was planted.
- Links refused: after the restore, any link that resolves into `/tests` or `/logs` (the grader's files), or into `/proc`, `/dev` or `/sys` (kernel trees whose links lead anywhere, e.g. `/proc/self/root`), or that loops, grades 0 with `link_into_grader`. Other absolute links stay (venvs need them). The image's own links are not followed: the image is pristine. The 17 served images contain no symlinks under `/app`.
- Environment values must be literals: a `${VAR}` template would be resolved against the gateway's own environment. The verifier's image must be the agent's image, with the task's env and no healthcheck.

### Known residuals

State them plainly; do not let a caller assume otherwise.

- The subprocess race (terminal). The 15 served rows whose tests run the agent's programs as subprocesses can leave a process behind that races `stop_processes`. Until it is killed, it can rewrite `reward.txt` and the report, and so forge a verdict. In the live goldens the defender won at a 1.5 s child delay every time; a faster child could win. Production container grading has the same exposure. The real fix is separate uids for the verdict writer and the agent's code. That needs a sandbox change (a second user in the grading box) and is a user decision; it is not done.
- Single-platform polyglot images were checked for `linux/amd64` by sample only. MiMo images are single-platform OCI manifests; the platform is in the config blob, and reading it costs one manifest GET against an anonymous Docker Hub budget of 100 per hour. 22 of the 2,698 were checked (both goldens included), all amd64. An unchecked non-amd64 image would fail at pull or start on the host, not silently pass.
- Only 17 of 64 MiMo rows are served until the uid separation exists.
- The `grading_box_failed` path for a fork bomb was not run live on the shared test box; it is covered by the sandbox's own suite.

## 5. Limits and timeouts

`TaskLimits` fields: `cpus`, `memory_bytes`, `disk_bytes`, `pids`, `wall_s`, `per_call_timeout_s`, `max_calls` (`None` means no requirement).

- The gateway enforces `memory_bytes`, `disk_bytes` and `pids`: a session token whose budgets are below them is refused with `422` before the token is spent, so an honest task never fails for want of memory. `cpus` sizes the box, capped by the machine's `episode_nano_cpus`. `wall_s`, `per_call_timeout_s` and `max_calls` are the validator's to choose and are never refused; the env's values are what the validator copies by default.
- `grading_timeout_s <= 810`. Plan 1's verification window requires `2 * grading_timeout_s + 180 <= GRADING_GRACE_S = 1800`, and a task that does not fit is refused at open with `422`. SWE grades at 810 s (its previous 1800 s scoring timeout no longer fits); terminal at 600 s.
- `prepare_timeout_s`: 600 s for both packages. A preparation that outlasts the token fails the open.
- Measured against these: prepare at most 20 s, grading at most 48 s, peak box memory at most 270 MB (section 9).

Per-split mapping:

| Env / split | cpus | memory | disk | pids | wall_s | per call | grading_timeout_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swe `train`, `r2e`, `polyglot` | not set | 4 GiB (provisional) | 10 GiB | 1024 | 3600 | 600 s | 810 |
| terminal `train` (MiMo row) | the row's `cpus` | the row's `memory_mb` | the row's `storage_mb` | 1024 | the row's `agent_timeout_sec` | 180 s | 600 |
| terminal tmax | planned: 2 | planned: 4 GiB | n/a | n/a | n/a | planned: 180 s | not served yet |

## 6. Prompts and tools

- `sandbox_prompt(split, index)` is the first user message exactly, and it never travels through the sandbox. The miner side renders no restricted-network notice, because the subprocess runtime verifiers uses does not add one.
- The tools `bash` and `edit` belong to the sandbox (`reliquary-tools/1`), and their implementations live in the gateway, versioned in record 0. SWE offers `("bash", "edit")`; terminal offers `("bash",)`. Record 0 signs the sorted offered set (`tools`), and a miner may call only those (`offered`): the router refuses anything else locally with `not_offered`, and the verifier refuses a call record whose tool is not in record 0's `tools` (`tool_not_offered`).
- What the model sees for a call is `reliquary_sandbox.observation.render_observation` of the signed record, never the env's own text. See `docs/episodes-client.md` in `reliquary-sandbox`.

## 7. Images

- Every `SandboxTask.image` is pinned by digest (`repo@sha256:<64 hex>`), must equal the session token's image, and is pre-pulled and approved on the host. The controller never pulls.
- `python -m reliquary_swe.sandbox images --split r2e --num-tasks 300` and `python -m reliquary_terminal.sandbox images --split train` print a JSON manifest of the images a split needs (they load the rows, so they also warm the row caches). The terminal list leaves out refused rows: never served, never pulled.
- `scripts/sandbox_images.py pull | check | approve | build-tmax --registry <REGISTRY>`: `pull` fetches what is missing, `check` runs each image under the host runtime and verifies `/bin/bash` and `sleep`, plus whatever `--require` names (`git` for SWE), `approve` prints the `RELIQUARY_SANDBOX_EPISODE_IMAGES` value for the gateway. `build-tmax` only prints its command unless `--push` is also given.
- Nothing is baked except the tmax base. The harness runs on the miner, so nothing installs at run time: SWE-smith, R2E, polyglot and MiMo boxes use their upstream images pinned by digest, and SWE and terminal grading already run with the network cut. tmax's install half is already in its shared base image (`reliquary_terminal/tmax_base/Dockerfile`).
- The registry for the tmax base is a parameter until one is chosen.

## 8. Network

None, ever. Every box runs with `--network none` under gVisor. Rows that need it are refused (`allow_internet`), and environment values that are `${VAR}` templates are refused, because they would resolve against the gateway's own environment. No env hook may need the network.

## 9. Testing, and enabling an env

### Unit tests

- `tests/sandbox_fakes.py` (in each package) has a `ScriptedRuntime`: no box, no Docker. Hooks run against a scripted runtime and the asserts check the argv and files they used.
- `reliquary_sandbox_service.episodes.local_gateway` runs the real app (routes, manager, grader, signing) on 127.0.0.1 under uvicorn with fresh keys. `run_scripted_episode` plays a list of `(tool, arguments)` calls and returns the records, the final, the state, the transcript and the verification result:

```python
from reliquary_sandbox_service.episodes.local_gateway import local_gateway, run_scripted_episode

with local_gateway(tmp_path, envs={"reliquary-swe": "reliquary_swe.sandbox:sandbox_task"},
                   images=images, episode_helper_busybox=busybox,
                   episode_helper_busybox_sha256=sha256) as gateway:
    run = await run_scripted_episode(
        gateway, env="reliquary-swe", split="train", index=0,
        calls=[("bash", {"command": "git apply /tmp/agent.patch"})])
    assert run.verification.ok and run.final["reward"] == 1.0
```

Never run `local_gateway` on a production host. Tests that reach it are marked `sandbox_live` and skip unless opted in.

### The parity goldens and their variables

The file `tests/test_sandbox_parity.py` of each package drives real episodes on a local gateway, over the isolated xfs daemon of the test box, and assert the verdicts the existing container goldens assert. Opt in:

- `RUN_SANDBOX_PARITY=1`;
- `SANDBOX_PARITY_IMAGES=<manifest.json>` (images pulled and approved);
- `DOCKER_HOST`, `EPISODE_LIVE_DOCKER_PIDFILE` (the isolated daemon, from `scripts/test-xfs-docker.sh env`);
- `SANDBOX_PARITY_BUSYBOX=/usr/bin/busybox`;
- `SANDBOX_PARITY_METRICS=<file>` (one JSON line per episode, written before any assertion so a differing verdict is still recorded).

### Test-box rules (verbatim)

The test box is `root@<sandbox-test-host>` (2 vCPU / 7 GB / about 46 GB free).

- never touch the main Docker daemon's config;
- never attach to, send keys to, or kill tmux session `tmaxval` (a TMax validation runs there);
- use Docker only through `SBX/scripts/test-xfs-docker.sh` (second daemon, private bridge `rqxfs0`);
- run `/opt/rq-plan2/guard.sh` before and after every live step, and stop on any failure it reports (`scripts/test-box-guard.sh` is its source);
- keep images small and remove them when a task is done;
- one live test process at a time.

Two more that the live runs taught:

- Run every Python step on a shared host under a memory cap (`systemd-run --scope -p MemoryMax=3G -p MemorySwapMax=0`). An uncapped script that built the R2E arrow cache reached 6.7 GB and was killed by the kernel's global OOM killer.
- Never configure a second dockerd with `"bridge": "none"`: moby deletes the host's `docker0` at start, which cut the main daemon's container networking on 2026-10-06. `test-xfs-docker.sh` uses its own bridge `rqxfs0` and tears itself down if a pre-existing link changes.

### The VPS rule

Never run Docker on the VPS: production (Wardian) runs there, and the packages' `conftest.py` probes `docker version` and would run `@docker` tests. On the VPS run only the files a step names, always with `-m "not docker and not slow and not sandbox_live"`, never a whole envs suite, and sequentially (22 GB, no swap).

### The gate: when an env is enabled

An env is enabled on the sandbox only when its goldens give the same verdicts as current grading: reference 1, untouched 0, every tamper patch 0. All verdicts below match the container goldens. Measured on the test box (2 vCPU, 1 CPU per box, gVisor, xfs quota); `prepare` runs from token issue to record 0, `grade` from the last call to the final record, and CPU is the final record's `cpu_total_ms` (agent plus grading box). Raw lines: `environments/code/reliquary_swe/tests/sandbox-parity-2026-10.jsonl` and `environments/code/reliquary_terminal/tests/sandbox-parity-2026-10.jsonl`.

reliquary-swe (19 of 19 match):

| split | case | expected | got | prepare s | grade s | s / episode | CPU s / episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train (oauthlib) | reference | 1 | 1 | 18 | 44 | 83.6 | 25.6 |
| train | committed reference (`git commit -am`) | 1 | 1 | 19 | 41 | 61.6 | 25.4 |
| train | untouched | 0 | 0 | 18 | 46 | 63.4 | 25.5 |
| train | forced-pass conftest | 0 | 0 | 18 | 47 | 65.9 | 26.4 |
| train | overwritten test file | 0 | 0 | 18 | 48 | 67.8 | 26.3 |
| train | ignored runner committed behind a moved base | 0 | 0 | 19 | 20 | 40.4 | 15.5 |
| train | `rm -rf` workdir | 0 | 0 | 20 | 46 | 66.7 | 25.9 |
| r2e (coveragepy) | reference | 1 | 1 | 5 | 8 | 14.3 | 4.9 |
| r2e | untouched | 0 | 0 | 5 | 8 | 12.4 | 4.5 |
| r2e | forged runner and tests | 0 | 0 | 4 | 7 | 12.8 | 4.8 |
| r2e | root conftest | 0 | 0 | 5 | 8 | 14.9 | 5.1 |
| r2e | root `pytest.py` shadow | 0 | 0 | 5 | 8 | 14.7 | 5.0 |
| r2e | ignored `.venv` startup file committed behind a moved base | 0 | 0 | 6 | 6 | 13.0 | 3.9 |
| r2e | `rm -rf` workdir | 0 | 0 | 5 | 7 | 11.5 | 4.6 |
| polyglot (Go) | reference | 1 | 1 | 1 | 11 | 14.5 | 7.5 |
| polyglot | untouched | 0 | 0 | 1 | 8 | 8.6 | 5.8 |
| polyglot | rewritten test script | 0 | 0 | 1 | 8 | 10.9 | 6.3 |
| polyglot | guessed hidden test file | 0 | 0 | 1 | 8 | 10.7 | 6.2 |
| polyglot | `rm -rf` workdir | 0 | 0 | 1 | 7 | 9.3 | 5.9 |

The two "ignored runner" cases end with `applied=False` and the forbidden-path refusal in the test output: grading refused the patch. The three `rm -rf` cases end `graded` with an empty state and no `_grader` fact: never `extract_failed`, `aborted` or `box_failed`. Polyglot has no tamper golden of that kind: at that image's base nothing is untracked or ignored, so there is no runner outside the tracked tree to attack.

reliquary-terminal (10 episodes recorded, 11 of 11 tests pass; the eleventh asserts a refusal at resolve and runs no episode):

| split | case | expected | got | prepare s | grade s | s / episode | CPU s / episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train (index 0) | reference | 1 | 1 | 1 | 3 | 5.4 | 2.4 |
| train | untouched | 0 | 0 | 0 | 4 | 4.3 | 1.8 |
| train | planted conftest | 0 | 0 | 0 | 2 | 4.3 | 1.4 |
| train | deleted `/app` | 0 | 0 | 1 | 3 | 5.6 | 2.4 |
| train | symlinked `/app` (`ln -s /etc /app`) | 0 | 0 | 1 | 2 | 4.2 | 1.4 |
| train | forged `reward.json` | 0 | 0 | 1 | 4 | 4.7 | 1.9 |
| train | link into `/tests` | 0 | 0 | 0 | 1 | 3.4 | 1.0 |
| train | link via `/proc/self/root` | 0 | 0 | 1 | 1 | 3.2 | 1.0 |
| train (index 20, subprocess row) | stray process stopped | 0 | 0 | 1 | 32 | 33.7 | 25.7 |
| train (index 20) | background `reward.txt` rewrite | 0 (defender won at 1.5 s) | 0 | 1 | 32 | 34.5 | 25.2 |
| train | in-process row | refused at resolve | refused | n/a | n/a | n/a | n/a |

Against the budgets: prepare at most 20 s (600 s), grading at most 48 s (810 s), peak box memory at most 270 MB.

Not enabled, and why:

- reliquary-terminal `eval` and reliquary-swe `eval`: graded in the agent's own box or held out by design; never served.
- reliquary-terminal tmax: not served. Its task (the tmax branch merge, the finished validation and a chosen registry for the base image are its preconditions) is gated and not implemented in `reliquary_terminal.sandbox`.
- 47 of the 64 MiMo rows: three need `/app` on `PYTHONPATH`, 45 run agent code in pytest's process; none is served until verdict writer and agent code run under separate uids.
- Not run live: the fork-bomb path to `grading_box_failed` (sandbox suite only), and the largest SWE-smith and R2E repositories (the goldens use the smallest to medium images; a `git gc` on pandas or numpy could take far longer than the 20 s measured).
