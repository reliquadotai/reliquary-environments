# Prime-RL v0.9.0

A pinned Prime-RL training lane for `reliquary-dapo-math`, using Prime-RL's
native Verifiers boundary, Qwen3 renderer, GRPO, zero-signal group rejection,
and held-out online evaluation. Reliquary does not wrap or fork the trainer.

The stack pins, the install commands, and the model-snapshot step are
identical to the
[`reliquary-stateful-tools` lane](../../../../tool_use/reliquary_stateful_tools/examples/prime_rl/README.md);
follow that README and substitute this environment. Two CUDA GPUs are needed:
one trainer, one inference worker.

Docker is still required even though this environment runs no code. Its tasks
declare a deny-all network policy, and Verifiers correctly refuses to honour a
policy under its unsandboxed subprocess runtime.

## What differs from the other lanes

Single turn, no tools, so `max_turns = 1` and there is no `tool_call_parser`.
What differs from every sibling is the window: `max_completion_tokens` is
24,576 rather than the 1,024 a formatting task needs, and `seq_len` and
`max_model_len` are sized to hold it. That is not generosity. Measured at 16
rollouts per prompt, an 8,192-token ceiling scores this environment at 16.6%
with 45.8% of groups failing unanimously; 24,576 scores it at 28.9% with 22.9%
failing unanimously. Cutting the budget does not make the environment cheaper,
it makes it look harder than it is and starves the very groups that would have
taught something.

A run that cannot afford the window should reduce `batch_size` or
`group_size`, not the completion budget.

## Where the signal is

75.0% of groups carry a usable spread of outcomes at the declared budget, and
best-of-16 reaches 2.7 times what one rollout scores — both measured on the
policy this lane trains. That headroom is the environment's reason to exist:
`openmathinstruct`, measured the same way on the same model, bands at 39.6%
with a headroom of ×1.2, which is most of a corpus the policy has already
learnt what it is going to learn from.

`filter_zero_advantages` and the `zero_signal` gate exist for the same reason:
an all-correct or all-wrong group teaches nothing and should not spend a step.

## Reporting

Freeze the config and eval indices, run at least three seeds, select a
checkpoint on `eval` alone, then evaluate `qualification` once. Report Avg@16,
and report the truncation rate beside it — a gain that comes from answers that
stopped being cut off is a gain in length discipline, not in mathematics.
