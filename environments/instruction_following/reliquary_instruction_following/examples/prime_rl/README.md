# Prime-RL v0.9.0

A pinned Prime-RL training lane for `reliquary-instruction-following`, using
Prime-RL's native Verifiers boundary, Qwen3 renderer, GRPO, zero-signal group
rejection, and held-out online evaluation. Reliquary does not wrap or fork the
trainer.

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
The reply is graded exactly as it arrives, so the completion budget is a
correctness parameter rather than a cost one: a truncated answer does not miss
narrowly, it breaks every constraint that speaks about how the text ends.

## Where the signal is

The difficulty dial is the number of constraints, and it is on the taskset:
`env.taskset.constraint_counts = [3]` trains on the three-constraint band
alone, `[1]` on the easiest. Neither is the default, because the band that
pays is a property of the policy being trained — measure the per-count reward
first, then narrow. A pool the policy always or never satisfies teaches
nothing, which is what `filter_zero_advantages` and the `zero_signal` gate
exist to stop paying for.

## Reporting

Freeze the config and eval indices, run at least three seeds, select a
checkpoint on `eval` alone, then evaluate `qualification` once. Report Avg@16
overall and per constraint count. An improvement that is entirely in the
one-constraint band is a formatting habit, not instruction following.
