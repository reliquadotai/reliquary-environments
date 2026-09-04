# reliquary-logic

Twelve families of procedurally generated logic puzzle — boolean evaluation,
bracket balancing, cryptarithms, constraint grids, spatial and temporal
reasoning — each graded by comparing typed JSON against a reference answer.
Single turn, no tools, no state, no network.

The answer channel is what makes the reward exact. `52` has one spelling, so
a correct answer cannot be marked wrong for its formatting and a wrong one
cannot be argued into acceptance. That removes the grader ambiguity that
free-text marking suffers from, which is the failure mode that cost a sibling
lane 28% of its correct answers before it was fixed.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-logic'); print(next(iter(vf.load_taskset(c(id='reliquary-logic')))).key)"
```

To evaluate a model, bound the infinite procedural taskset:

```bash
uv run eval reliquary-logic -n 24 --env.agent.max-turns 1
```

## Prompting

Every family is asked to reason step by step before answering, and the
grader reads the last fenced JSON block that parses.

That default is a deliberate over-application. Measured on Qwen3-4B-Base over
64 prompts × 16 rollouts, the reasoning template is worth +10.9 points of
in-band groups on `numbrix` and +9.4 on `dyck_language_errors`, costs
`math_path` 14.1, and changes nothing on the six families that already answer
correctly in ten tokens — where it multiplies generated tokens by fifteen.
Restricting it to the families it helps buys the same band far more cheaply,
and `LogicConfig.reasoning_families` does exactly that. It is not the default
because the band measures the signal available now and says nothing about
whether reinforcing reasoning broadly pays over a training run.

Reasoning brings the model's own code fences with it. An odd number of fence
markers shifts the pairing so the answer block is consumed as an earlier
block's terminator, and the completion ends in a perfectly good answer the
grader never sees. Reading the last block that *parses*, rather than the last
block, recovered 31.7% of previously unparsable completions on 3,072 real
rollouts — 2.90% of all rollouts were correct answers scored zero. The
brace-balanced fallback runs only when no fenced block parses, and every
bound (depth, size, duplicate keys, floats, non-finite constants) still
applies to whichever block wins.

## Relationship to the core environment

The generator is vendored unchanged from `reliquarylogic_v1`: same seeding,
same puzzles, verified task-for-task. What is new here is the package
identity, the disjoint train/eval/qualification splits, and the Verifiers
surface. Measurements taken against the core environment therefore carry
over.

The package exports `LogicTaskset` for Verifiers and `prime-rl`, and
`LogicEnvironment` for synchronous Reliquary-compatible replay. It imports no
Reliquary code.

`LogicEnvironment` implements `reliquary/answer-json/v1` — `task`, `grade`,
`replay`, `reference_completion` — and deliberately not the episode contract
its tool-use sibling implements: one turn means there is no `reset` and no
`step` to expose.
