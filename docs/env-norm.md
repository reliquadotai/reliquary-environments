# The env norm: env packages on signed-episode sandboxes

This page is for authors of env packages and for the people who qualify them. A
signed-episode sandbox serves a standard `verifiers` v1 Taskset through a generic bridge:
the env package holds no sandbox code and imports nothing from Reliquary. `reliquary-swe`
and `reliquary-terminal` (0.3.0) are served that way; their former per-env sandbox
adapters are gone.

## 1. The norm, as an env author applies it

1. **One contract in the Task.** `Task.setup` prepares the agent's box, `Task.finalize`
   leaves the final state in the box as declared artifacts (box commands and file writes
   only; it parses nothing), and a task-level `@vf.reward` grades. An optional
   `Task.grading_setup(runtime)` prepares the pristine grading box before the agent's
   artifacts are restored there. No `Env` subclass is needed for grading (the Harbor
   pattern of `verifiers.v1.tasksets.harbor`).
2. **Grade in a pristine box.** The sandbox starts a fresh box from the same image, runs
   `grading_setup`, restores only the declared artifacts (plus verifiers' implicit
   `/logs/artifacts`), and calls the rewards there. A reward never reads anything the agent
   could write other than those artifacts.
3. **No trust in agent state.** Whatever grading needs (base commit, original file list,
   protected inputs) is recomputed from the image inside the grading box, never read from
   the agent's box and never kept in host memory. One `Task` instance serves concurrent
   episodes: it keeps no per-episode state.
4. **Minimal, explicit artifacts.** SWE: a git patch against the base, no binary hunk, no
   symlink, no submodule. Terminal: only the declared roots.
5. **Hardening in the env's own grading.** Each known attack (forged report, untracked
   file, symlink, process left running, reward-file tampering) is a test of the env and,
   where it needs a real box, a declared conformance case (section 5).
6. **Deterministic scoring.** No network in any box, no judge, one reward source (an exit
   status or a strictly parsed report); two gradings of the same artifacts give the same
   reward.
7. **Pinned images.** The package ships `sandbox-images.lock.json`:
   `{"version": 1, "images": {"<TaskData.image>": "<repo>@sha256:<digest>"}}`. A task whose
   image the lock does not pin is refused.
8. **Strict versions.** Any change to grading or to an image is a new package version. The
   sandbox names an env by its installed distribution and a digest of its installed files,
   plus the bridge version; validators allow-list that identity. The task's content hash
   (`Task.hash`) is signed in the episode's first record and recomputed by validators, so
   it must not depend on the machine (paths, caches).
9. **Explicit splits.** The served split names are the taskset's own split values. `eval`
   is never served.
10. **The conformance suite is the gate.** The sandbox runs the env through the bridge on
    gVisor: references grade 1, the empty artifact grades 0, every declared attack grades 0,
    and repeated gradings agree. Passing it is what plugs an env in.

Bridge budgets an env must fit: setup at most 600 s, each grading step (finalize, scoring)
between 35 s and 810 s (at least 70 s for a task with `grading_setup`, which has its own
earlier deadline inside the step). `TaskData.resources` gives cpu, memory and disk; pids and the
per-call timeout come from the gateway's defaults for the env.

### What happens when a hook fails

Every hook runs in the sandbox's unprivileged scoring process, never in the gateway.

| Hook | Outcome when it fails | Whose |
| --- | --- | --- |
| `setup` (agent's box) | the session open fails; no episode exists | ours |
| `finalize` (agent's box) | an exception: graded 0 (`extract_failed`, counted per env); past its deadline: graded 0 (`extract_timeout`) | the agent's outcome |
| `grading_setup` (pristine box, nothing of the agent in it yet) | the episode is aborted (void, may be re-run) | ours |
| a reward (pristine box, artifacts restored) | an exception: graded 0 (`grading_failed`, logged and counted per env); past its deadline: graded 0 (`grading_timeout`) | the agent's outcome |

A real box or runtime fault keeps the sandbox's own label at any step. The rule for env
code: an exception about us or the image belongs in `setup` or `grading_setup`; anything
the agent's state can cause must happen after the restore, where it grades 0. Otherwise an
agent could void its own failing episodes. Watch the per-env counters: a real env bug
shows up there, not as voided episodes.

## 2. reliquary-swe

- Splits served: `train` (SWE-smith, the default 20 images), `r2e`, `polyglot`, whole
  corpora. `eval` (SWE-bench Verified) is never served. The taskset loads one task by index
  (`SweTaskset.task_at`, equal to the index-th task `load()` yields; `len()` is the split's
  size): SWE-smith's whole split weighs about 4 GB as rows.
- `setup` runs the cleanup (checkout, SWE-smith's guard and re-root, R2E's leak guard, ref
  stripping and gc, R2E's hidden-test removal), then records in the box the base the
  agent's diff is taken against (`refs/reliquary/base`) and the image's untracked files
  (`.git/reliquary-untracked`). An image whose untouched box does not capture as an empty
  patch fails setup (ours).
- `finalize` removes `/logs/artifacts/patch.diff` first (a planted patch never travels),
  then writes the agent's diff there with `vf.capture_patch`, run in a scratch repository
  of ours so no git setting or hook the agent left runs. The agent can move the base ref or
  edit the untracked list: that only widens or narrows its own diff, which grading never
  trusts.
- `grading_setup` is `grading.prepare_box`: the pristine-box preparation (checkout, history
  strip, R2E's hidden tests set aside). A `grading.PristineBoxError` there aborts the
  episode.
- The reward, `patch_passes_tests`, reads the patch byte for byte (missing: graded as an
  empty patch, 0; over 2 MB or not a regular file: 0) and runs `grading.grade_prepared`:
  a patch with a binary hunk, a symlink or a submodule
  (`grading.patch_shape_violations`), or one touching a path untracked or ignored at the
  base or under `.venv/`, grades 0 and is never applied; otherwise test files are restored
  from the image and the tests decide.
- Timeouts: setup 600 s, agent 3600 s, finalize 600 s, scoring 810 s. Resources: 4 GB
  memory (provisional), 10 GB disk. Recommended gateway options: tools `bash` and `edit`,
  per-call timeout 600 s, splits `train`, `r2e`, `polyglot` each mapped to itself.
- Images: `reliquary_swe/sandbox-images.lock.json`, built by `scripts/pin_sandbox_lock.py`
  from the per-corpus digest files (`swesmith_digests.json`, `r2e_digests.json`,
  `polyglot_digests.json`).
- prime-rl's `SweEnv` keeps grading in a box of its own (`task.graded_elsewhere()`); the
  reward records nothing on any runtime but a sandbox's grading box.
- Known residual, the same as in container grading: a patch confined to source can
  monkeypatch pytest's reporting at import time so every test reports PASSED. Only an
  injected canary check would close it (not built).

## 3. reliquary-terminal

- Split served: `tmax` (TMax-15K's kept tasks, all RL). `eval` (Terminal-Bench 2.1, graded
  in the agent's own box by design) is never served; MiMo's `train` rows are not served on
  sandboxes. Tasks excluded statically by the selection manifest (`tmax_select`) are never
  served: `in_process_agent_code`, `environment_in_artifact_roots`,
  `mode_changed_by_hand_over` among them (`environments/code/reliquary_terminal/docs/tmax.md`).
- Artifacts: the roots `/app` and `/home/user`, archived and restored by the sandbox
  itself (symlinks stored, never followed; `..`, FIFOs and devices refused; owners and
  set-id bits dropped). A root the agent deleted is graded, never an error. `finalize`
  collects nothing.
- `grading_setup` runs the task's setup in the grade role in the pristine box: the task's
  data is regenerated there and its protected inputs are stashed outside both roots, as
  verifiers' own `HarborEnv` does for its verifier box.
- The reward, `solved`, stages the tests and runs `test.sh` in that box, every root command
  with a fixed PATH, shells by absolute path, HOME outside the roots and
  `PYTHONNOUSERSITE=1`. `test.sh` puts the stashed inputs back, hands the restored roots to
  the test uid 61000 (never following a link), runs the final-state test as that uid under
  `python3 -I` with no_new_privs, kills every process of that uid, and only then writes
  the verdict as root: 1 only for a clean exit, every process gone and a report. A box where
  the test uid could ptrace pytest, or write `/logs`, is refused (0).
- The task hash leaves out where the task's files were written (`task_dir` is replaced by
  a digest of its content), so a miner and a gateway with caches in different places
  compute the same one.
- Resources: 2 CPUs, 4 GB; scoring 600 s. Recommended gateway options: tool `bash` only,
  split `tmax` mapped to itself.
- Image: one shared base image, served by digest from the package's
  `sandbox-images.lock.json` once that image is in a registry (`environments/code/reliquary_terminal/docs/tmax.md`, "What is
  left"). The validated image is pushed as is; a rebuild is a new image and needs a new
  validation.

## 4. Conformance cases and reference solutions

Each package exports two functions of plain data, read by the sandbox's conformance suite
and reference sweep (neither imports a sandbox):

- `conformance_cases(split) -> list[dict]`: each case has a `name`, the task `index`, the
  `expect`ed reward and the `calls` (`(tool, arguments)` pairs) the suite plays as the
  agent; a case may also list paths that must be `absent` from the grading box after
  scoring.
- `reference_calls(split, index) -> list | None`: the calls that solve task `index`, or
  None when it has no reference.

reliquary-swe, on each golden of `train`, `r2e` and `polyglot`: the reference applies the
gold patch through `bash` in base64 chunks (expect 1). Attacks, each expect 0: the gold
patch plus a forced `.venv` path, plus a symlink, plus a binary file; the gold patch
planted in `/logs/artifacts` with the sources untouched, beside a stale `index.lock`, by
git settings that would run during the capture, or with the repository gone; a root
`conftest.py` forcing every test to pass; on `train`, the fail-to-pass tests overridden to
pass in their own files. A gold patch the norm refuses is no reference (8 R2E rows add a
symlink). Polyglot's references are shipped under `reliquary_swe/references/`.

reliquary-terminal, on `tmax` only (task 0 and the first task whose final test runs
programs): the reference replays the task's recorded solution (expect 1). Canaries, each
expect 0 with their markers `absent`: a `conftest.py` in `/home/user` and `/app`, a
`usercustomize.py` in the user site, a `sitecustomize.py` in `/app`, `bash` and `python3`
in `/home/user/.local/bin`; each file only creates a marker under `/tmp` if executed, so a
marker in the grading box means the grader ran the agent's file.

Indices are those of a split served with the taskset's defaults; a split served otherwise
names other tasks under these indices, and its reference case fails.

## 5. Image lists

A sandbox host pre-pulls, checks and approves every image it may run; the gateway never
pulls. Lists come from each package's lock:

```bash
python -m reliquary_swe.images --split r2e --num-tasks 300 > r2e.tags
python scripts/sandbox_images.py from-lock \
    environments/code/reliquary_swe/reliquary_swe/sandbox-images.lock.json --tags r2e.tags > r2e.json
python scripts/sandbox_images.py pull r2e.json --docker-host unix:///run/docker.sock
python scripts/sandbox_images.py check r2e.json --runtime runsc --require git
python scripts/sandbox_images.py approve r2e.json   # -> the gateway's image setting
```

`from-lock` prints `{"images": [digests]}` for the tags (or digests) listed one per line,
every image the lock pins without `--tags`, and refuses (exit 2) a tag the lock does not
pin. `check` runs each image under the host runtime with no network and verifies
`/bin/bash`, `sleep` and whatever `--require` names; `approve` prints the gateway setting
only when every image has a recorded passing check.

## 6. Moving a host to this version

Hosts and jobs pin an envs commit. Removing the adapters changes nothing for a host still
on an older commit (production corpus executors call `grading.grade` and the corpus
modules, which keep their signatures). Move every pin at once, between jobs, with no live
episode: the gateway's env packages, the validators' allow-listed identities and the
corpus executors. A gateway on 0.3.0 lists the envs as `verifiers:reliquary-swe==0.3.0`
and `verifiers:reliquary-terminal==0.3.0`, never as `module:callable` entries. A validator
places no session on a gateway whose env identity it does not allow-list, so a half-moved
fleet refuses sessions rather than grading them two ways.
