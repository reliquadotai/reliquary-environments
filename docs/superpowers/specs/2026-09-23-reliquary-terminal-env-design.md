# reliquary-terminal: a Terminal-Bench environment

Date: 2026-09-23
Status: implemented as `environments/code/reliquary_terminal`. Section 4 decided 2026-09-28: option C -- `eval` = Terminal-Bench 2.1 graded as shipped, `train` = MiMo-V2.6-RL-oss's 64 terminal tasks graded in a separate box.

## 1. What the spike established

Not inference — measured on the container host, against the pinned `verifiers`
commit and `harbor` 0.21.0.

- `HarborConfig(dataset="terminal-bench/terminal-bench-2-1")` downloads all 89
  Terminal-Bench 2.1 tasks in five to seven seconds, with **no credentials, no
  account and no payment**.
- A real task ran end to end through `HarborTaskset` and `HarborTask`. The
  reference solution for `openssl-selfsigned-cert` scored **1.0**; the same task
  with no solution scored **0.0**. The harness discriminates.
- The corpus is 89 pullable images, about 18.2 GB compressed across the 85 that
  could be measured, so 25-40 GB on disk for the full set.

That is enough to say the mechanism works. What follows is what it costs.

## 2. What we would write

Almost nothing, which is the point. `verifiers` already ships the taskset, the
Harbor integration, the separate-box grading pattern and every harness. A
`reliquary-terminal` package is:

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Metadata, the `harbor` dependency, the pinned `verifiers`. |
| `environment.toml` | The declared contract: tier, resource class, network, budgets, `[policy]`. |
| `reliquary_terminal/taskset.py` | A thin subclass of `HarborTaskset` pinning the dataset, applying our task-level policy, and carrying the workdir correction of section 3. |
| `README.md` | What it is, and what we changed from stock Terminal-Bench and why. |
| `examples/prime_rl/` | A runnable example. CI checks it matches the declared reasoning mode. |
| `tests/` | The discrimination goldens of section 5. |

No corpus module: Harbor owns the corpus. No adapter: there is no second library
to wrap. No grader: the task's own `tests/` are the grader.

## 3. The bug we must carry a fix for

`verifiers`' `DockerConfig` defaults `workdir="/app"`, and that default
**silently overrides a task image's own baked-in `WORKDIR`** whenever
`task.toml` does not declare one. None of the 89 tasks declare one.

Confirmed on `fix-git`: the **reference** solution scored 0, exit 128, "not a
git repository", until the container's workdir was forced to
`/app/personal-site` — after which it scored 1.0. Three of the 89 tasks are
affected (`fix-git`, `sanitize-git-repo`, `prove-plus-comm`).

This is the tenth instance on this project of the only failure shape that
matters here: **reward 0 that looks like a bad policy and is actually a broken
harness**. It was found in the first hour of looking at a new environment, by
checking that a known-good solution scores 1.0 rather than accepting that a
rollout scored 0.

Whatever else this design becomes, the package must read each task's real
`WORKDIR` rather than inherit the default, and a golden must pin one of the
three affected tasks at 1.0 for its reference solution.

## 4. The decision that is not mine

**Terminal-Bench grades every task in the same container the agent worked in.**
Zero of the 89 tasks declare `[verifier].environment_mode = "separate"`.

`reliquary-swe` is built around the opposite discipline, and section 6 of its
design says why: only the diff travels, grading happens in a pristine container,
so a patch that tampers with the test apparatus cannot be rewarded — not because
we detect it, but because it cannot survive the transfer. That environment spent
a night closing four separate routes through that property, and one of them
earned a **false reward of 1.0** before it was closed.

Same-box grading gives that property up. An agent with a shell and a verifier in
the same container can edit the verifier.

The trade, stated plainly:

**Option A — take the dataset as it ships.** Same-box grading. Our numbers are
directly comparable to every published Terminal-Bench 2.1 result, which is the
entire reason to use a public benchmark. We inherit its integrity properties,
including the ones we would not have chosen. For *evaluation* this is almost
certainly right.

**Option B — impose separate-box grading.** Harbor supports it; the tasks do not
use it. We would collect the agent's box as artifacts, provision a fresh one
from the task image, restore the artifacts, and run `tests/` there. That is
exactly what `HarborEnv.finalize` already does for tasks that ask for it, so the
machinery exists — what does not exist is any guarantee that a Terminal-Bench
task's verifier still works when its filesystem arrives as artifacts rather than
as the live box it was written against. Some will not. For *training*, where a
false 1.0 is poison rather than an embarrassment, this is the safer discipline.

**Option C — both, by configuration.** Same-box for the evaluation split, so the
published number means what it says; separate-box for the training split, where
integrity beats comparability. This is more work than either alone and it means
maintaining two grading paths, but it is the only option that does not force a
choice between a comparable number and a trustworthy gradient.

**My recommendation is C, arrived at reluctantly** — it is the most work, and I
would normally argue against building two paths before one is proven. What
changes it here is that the two uses genuinely want different things, and we
already know from `reliquary-swe` that a reward-hacking route in a training
environment is not hypothetical: one was found, open, earning full marks, in an
environment designed from the start to prevent exactly that.

But this is a decision about what the environment is *for*, and that is not a
decision an implementer should make at five in the morning.

## 5. Testing, whichever option is chosen

The goldens follow `reliquary-swe`'s, because the failure shape is identical:

1. A reference solution scores **1.0** — on one of the three workdir-affected
   tasks, so section 3's correction is pinned by a test rather than a comment.
2. An empty solution scores **0.0**.
3. Under Option B or C, a solution that tampers with the task's own `tests/`
   scores **0.0** *with the tamper confirmed applied* — otherwise the golden
   proves only that the tamper failed to land.

And the standing rule this project earned the hard way: for every claim about
reward, a test that runs in a real container and **whose failure has been
observed at least once**. Nine defects on `reliquary-swe` and one on this spike
all produced reward 0 rather than an error. A test that has never been seen to
fail is a test nobody has checked.

## 6. What this design does not answer

- Whether Terminal-Bench task verifiers survive being run against restored
  artifacts rather than a live box. Option B and C both depend on it and the
  spike did not test it. It is the first thing to measure if either is chosen.
- The turn and token budgets, which must be measured rather than guessed, as
  `reliquary-swe`'s remain.
- Whether the 89 tasks carry enough signal for RL at all — the same
  qualification question that is still open for SWE, and for the same reason:
  an environment that runs is not an environment that teaches.
