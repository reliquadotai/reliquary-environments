# Prime-RL v0.9.0

A pinned Prime-RL training lane for `reliquary-logic`, using Prime-RL's native
Verifiers boundary, Qwen3 renderer, GRPO, zero-signal group rejection, and
held-out online evaluation. Reliquary does not wrap or fork the trainer.

The stack pins, the install commands, and the model-snapshot step are
identical to the
[`reliquary-stateful-tools` lane](../../../../tool_use/reliquary_stateful_tools/examples/prime_rl/README.md);
follow that README and substitute this environment. Two CUDA GPUs are needed:
one trainer, one inference worker.

Docker is still required even though this environment runs no code. Its tasks
declare a deny-all network policy, and Verifiers correctly refuses to honour a
policy under its unsandboxed subprocess runtime.

## What differs from the tool-use lane

Single turn, no tools, so `max_turns = 1` and there is no `tool_call_parser`.
The completion budget is larger instead: every family is asked to reason step
by step, which takes the median generated length from 15 tokens to 217.

## Where the signal is

Measured on Qwen3-4B-Base over 64 prompts × 16 rollouts, 73.2% of groups carry
a usable spread of outcomes and no group is solved by all sixteen rollouts.
Six of the twelve families are answered correctly in ten tokens and contribute
little; `numbrix`, `math_path`, `cryptarithm`, and the two `dyck_language`
families carry most of it. If a run flattens, check the per-family band before
touching the loss.

`filter_zero_advantages` and the `zero_signal` gate exist for the same reason:
an all-correct or all-wrong group teaches nothing and should not spend a step.

## Reporting

Freeze the config and eval indices, run at least three seeds, select a
checkpoint on `eval` alone, then evaluate `qualification` once. Report Avg@16
and per-family means. pass@16 alone overstates the gain on an environment
where reliability, not capability, is what moves.
