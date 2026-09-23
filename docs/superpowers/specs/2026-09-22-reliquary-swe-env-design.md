# reliquary-swe: a containerised software-engineering environment

Date: 2026-09-22
Status: design, approved in principle; not yet implemented

## 1. Goal

Add `reliquary-swe` to this repository: an environment where a policy is given a
real repository at a real commit, a bug report, and a shell, and is rewarded for
producing a patch that makes the repository's own tests pass.

It is the first of four "frontier" environments we intend to ship (software
engineering, terminal, cybersecurity, knowledge work). It goes first because its
corpus is free and abundant, and because its grader is the most objective of the
four: tests pass or they do not, with no judge model in the loop.

## 2. Non-goals

- **Not Reliquary consensus.** This environment targets Verifiers and prime-rl
  only. It does not ship the synchronous JSON replay surface the other packages
  provide, and it does not participate in subnet scoring. Deterministic replay
  of a container is not achievable, and nothing here should pretend otherwise.

  This is a conscious deviation from the repository README, which states that
  every package exposes both surfaces. No CI check enforces the second surface,
  so the deviation is possible; it is recorded here so that it reads as a
  decision rather than an omission. `environment.toml` therefore declares no
  `compatibility_entrypoint`, and the README table entry must say plainly that
  this environment is Verifiers-only.
- **Not the other three environments.** Terminal, cyber and knowledge-work
  environments are deliberately out of scope. They are expected to reuse what
  this one establishes, but no shared abstraction is extracted until a second
  consumer exists.
- **Not a new container backend.** See section 3.

## 3. What already exists, and must not be rewritten

Most of the machinery a containerised environment needs is already inside the
`verifiers` version this repository pins. Re-implementing any of it would be
waste:

- `verifiers.v1.runtimes.docker.DockerRuntime` — starts a container, runs argv
  in it, reads and writes files, tears it down.
- `verifiers.v1.runtimes.docker.egress` — an egress proxy enforcing an
  allowlist by SNI. This is the network isolation the anti-leak protocol needs;
  we configure it, we do not build it.
- `RuntimeConfig` selects among docker, subprocess, modal and prime backends.
  Where containers physically run is therefore a configuration choice, not an
  architectural one. Self-hosted Docker is the default; a hosted backend
  remains available without touching environment code.
- `TaskData` already carries `image`, `workdir`, `network_allow`,
  `network_block`, `artifacts`, `timeout` and `resources`.
- `Task` already exposes the lifecycle we need: `setup(trace, runtime)`,
  `finalize(trace, runtime)`, `validate(runtime)`, `score(trace, runtime)`,
  plus `NEEDS_CONTAINER` and a `key` override for durable dataset identifiers.
- `TaskData.artifacts` collects paths from one runtime and restores them at the
  same locations in another. This is the supported mechanism for grading in a
  different container from the one the agent worked in, and section 6 depends
  on it.

What we write is the corpus, the task, and the grader.

## 4. Package layout

`environments/code/reliquary_swe/`, following the shape of `reliquary_code` and
`reliquary_telecom_solo`:

| Module | Responsibility |
| --- | --- |
| `corpus.py` | Rows only: `instance_id`, repo, base commit, problem statement, fail-to-pass and pass-to-pass test lists, reference patch, test patch. No execution. |
| `swe_adapter.py` | The only module importing `swebench`: image keys and per-repository log parsing, behind names we own. |
| `taskset.py` | `SweData`, `SweTask`, `SweTaskset`. Container preparation and patch capture. |
| `grading.py` | Given a live container: apply the patch, restore the tests, run the two test sets, parse, score. Provisions nothing. |
| `env.py` | Provisions the grading container, grades in it, records the reward on the solver's trace. |
| `environment.toml` | Declared contract: tier, resource class, network policy, budgets. |
| `examples/` | A runnable prime-rl example. Required: CI checks that it asks for the reasoning mode the environment declared. |
| `tests/` | Goldens and unit tests (section 9). |

Each module must be readable on its own. `corpus.py` in particular must not
import anything that starts a container, so that corpus questions can be
answered without Docker present.

## 5. The harness

**We write no harness.** `verifiers.v1.harnesses` already ships `bash`,
`mini_swe_agent`, `claude_code`, `codex`, `terminus_2` and others, and the
harness is chosen by configuration rather than by the environment.

This corrects an earlier draft of this section, which specified a two-tool
harness of our own. Writing one would have been redundant, and worse, it would
have welded the environment to a single interaction mechanism — the opposite of
what we want, since training across several harnesses is what stops task-solving
strategy from becoming a property of one harness's quirks.

The environment is therefore harness-agnostic by construction: it supplies the
container, the prepared repository and the reward, and says nothing about how
the policy reaches a shell. A first pilot validates against one harness;
training across several costs a configuration change, not a code change.

One consequence is worth stating, because it is the reason production harnesses
are a poor default for training: they wrap the agent loop in safeguards and
instruction prompts that fall outside the reward signal, which makes credit
assignment unreliable and leaves reward-unmeasured requirements to be ignored.
A minimal harness such as `bash` or `mini_swe_agent` is the better baseline.

## 6. The central decision: grade the patch, not the container

When the episode ends, `finalize()` extracts `git diff` from the agent's
container and declares it as an artifact. Grading then happens in a **fresh
container started from the task's pristine image**, where the diff is applied
and the two test sets are run.

The agent's container is never the grading container.

This removes an entire class of reward hacks in one move — editing the tests,
monkeypatching `pytest`, leaving a permissive `conftest.py` or a
`sitecustomize.py` behind, mutating installed packages. None of it survives a
transfer where only the diff travels.

It also makes the reward reproducible. A rollout can be re-graded months later
from its stored diff, with no surviving container, which is what makes it
possible to audit supervision after the fact rather than only in the moment.

The cost is a second container per rollout. That container is short-lived: it
applies a patch and runs tests, with no agent in it. To be precise about what is
shared and what is not: every grading runs in its own fresh container, always.
What is reused across rollouts of the same repository is the *image*, which is
pulled and cached once. Reusing a grading container between two rollouts would
reintroduce exactly the contamination this section exists to prevent.

## 7. Reward-hacking mitigation

The absence of adversarial miners does not make this optional. The policy
itself reward-hacks: the failure mode is documented in the literature as
retrieving the published fix instead of deriving it — installing a newer
release of the target package and reading its source, fetching an upstream file
over HTTP, cloning the upstream repository, or looking up the issue's linked
commit. Each of those satisfies a test-based reward without the model having
solved anything, and RL then reinforces it.

Three layers, in order of how much they cost us:

1. **Network isolation.** `network_allow = []` on every task. An empty concrete
   allowlist replaces the default wildcard, so nothing is reachable. This alone
   closes package installation, raw file fetches and upstream clones.
2. **Environment preparation in `setup()`.** Truncate git history at and
   including the base commit, removing later commits and their refs; remove
   build logs, verifier output, residual patches and generated bytecode;
   clear caches that may hold solution text. Third-party dependencies are
   preserved, since the repository must still build offline.
3. **Audit, not assumption.** Section 10 lists this as an open item rather than
   a solved one: we do not yet know our leak rate, and claiming a number before
   measuring it would be exactly the kind of false assurance this repository's
   other environments avoid.

## 8. Corpus

Two sources, pinned by revision as every other environment here pins its data:

- **Evaluation** — `princeton-nlp/SWE-bench_Verified`, 500 human-validated
  instances. Chosen because it is the number a reader can compare against
  published results.
- **Training** — SWE-smith. Its instances are packaged into a small number of
  Docker images (one per repository rather than one per task), which is what
  makes the disk footprint tractable; a task-per-image corpus of comparable
  size would not fit on a box we are willing to pay for.

Evaluation and training corpora must not overlap by repository. Contamination
across that boundary would make the evaluation number meaningless, and we have
been bitten by a contaminated held-out set before. SWE-bench Verified is an
evaluation set and must never be trained on.

### Correction: a second corpus is a profile, not a data pin

An earlier draft of this section claimed that adding a corpus is a change to
`corpus.py` and a second data pin, touching neither the task nor the grader.
That was wrong, and inspecting SWE-smith's actual schema is what showed it.
SWE-smith rows carry `instance_id`, `patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`,
`image_name`, `repo` and `problem_statement` — and nothing else:

- **The image is given, not derived.** `image_name` names it outright
  (`jyangballin/swesmith.x86_64.<repo>.<commit>`), so `SweRow` should carry an
  image field and `image_for`'s swebench derivation becomes the Verified
  fallback rather than the rule. This is simpler, not harder.
- **There is no `version`,** so the `MAP_REPO_VERSION_TO_SPECS` lookup that
  resolves a test command cannot apply. A second resolution path is needed.
- **There is no `base_commit`.** The image already sits at the state the agent
  starts from, so `setup()`'s checkout has nothing to check out.
- **There is no `test_patch`, and this one reaches section 6.** Grading defeats
  test tampering by restoring the tests from the instance's own test patch
  before running anything. SWE-smith injects its bug into the *source*, leaving
  the tests already present and pristine in the image, so there is no patch to
  re-apply. The property still holds — a freshly started container has pristine
  tests by construction — but it must be obtained a different way.

The consequence for implementation: **grading must treat test restoration as a
seam, not a hardcoded step.** One strategy re-applies a test patch; the other
restores the test paths from the pristine image. Both answer the same question —
"make the tests be what they should be, whatever the agent did" — and the
grader should ask that question rather than prescribe one answer.

## 9. Testing

Goldens, in the style used elsewhere in this repository:

1. The reference patch scores 1.0.
2. An empty patch scores 0.0.
3. **A patch that edits the tests scores 0.0.** This is the golden that proves
   section 6 works. It is the one test that must never be allowed to rot.
4. Fail-to-pass tests fail and pass-to-pass tests pass *before* the reference
   patch is applied, and both pass after.

Unit tests cover corpus loading and diff extraction without Docker. Container
tests are marked and skipped when Docker is absent, so the suite stays runnable
on a machine that cannot host images.

### CI gates this package must satisfy

CI already runs every environment through the same four gates, and a new
package is not done until it passes them:

- `check_verifiers_pin.py` — the package resolves the pinned verifiers commit.
- `check_reasoning_mode.py` — `environment.toml` declares a `[policy]`
  reasoning mode *and* a rationale. A value without a rationale is refused.
- `check_example_matches_policy.py` — the shipped example asks for the mode the
  environment declared. The two drift silently otherwise, and the run then
  trains a mode the environment did not ask for.
- `validate <taskset> -n 3 --only-gold true --runtime.type docker
  --runtime.allow '[]'` — the environment runs its gold solutions under the
  Docker runtime with an empty egress allowlist.

That last gate is worth noting: the Docker runtime and the empty allowlist are
already exercised paths in this repository's CI, not new ground. It also maps
cleanly onto this environment, where `--only-gold` means the reference patch,
which is golden 1 of section 9.

On the reasoning mode, the starting declaration is `thinking`, on the grounds
that a repair requires reading and diagnosis before the first edit, and that an
agentic loop with a single shell tool gives the policy nowhere else to do that
reasoning. This rationale is an argument, not a measurement, and the other
environments here earned theirs by measurement. It must be confirmed against a
pilot before release, and the rationale text rewritten to state the number.

## 10. Open questions, to be settled by measurement rather than opinion

These are deliberately unresolved. This repository's environments declare their
budgets with a measured rationale, and inventing numbers now would break that.

- **Turn budget.** Starting point 40 turns, to be re-derived from the observed
  distribution of a pilot. The telecom environment set 40 from a measured
  reference-solution length; we have no equivalent measurement yet.
- **Token budget.** Unset until measured. A larger budget is not neutral and
  can be actively worse: `reliquary-code` measured a *narrower* trainable band
  at 24,576 tokens than at 8,192.
- **Reward shape.** Binary (all fail-to-pass tests pass) is the recommendation,
  on the grounds that a fractional reward pays for a half-repair. To be
  confirmed against the observed distribution of partial passes.
- **Supervision quality.** We do not yet know the false-positive and
  false-negative rates of these test suites as graders. Establishing them —
  by repeated execution for flakiness, and by auditing rollouts whose observed
  reward disagrees with an independent reading of the patch — is required
  before this environment is used for a real run, and is tracked separately
  from shipping the package.

## 11. Risks

- **Disk.** Image storage is the binding constraint, not CPU. The corpus choice
  in section 8 is driven by it.
- **Flaky suites.** A test that fails intermittently injects noise directly into
  the gradient. Repeated-execution screening is the mitigation, and it belongs
  to the supervision-quality work item above.
- **Container leakage under concurrency.** Containers that are not torn down
  accumulate and eventually exhaust the host. `DockerRuntime` owns teardown, and
  this is already structural rather than something our `finalize()` must be
  careful about: verified against `verifiers.v1.rollout.Rollout.close()`, task
  finalize (`rollout.py:474`) and runtime teardown (`rollout.py:526-531`) run
  as separately-guarded stages inside one `try`/`finally`, so a `finalize()`
  that raises — `capture_patch` does exactly this by design, via
  `SandboxError`, on a dead box — still reaches teardown. Not a risk this
  package needs to defend against.
- **Scope creep toward the other three environments.** The moment a second
  environment is written, the pressure to extract a shared base package will be
  real and should be revisited then — not now.
