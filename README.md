# Reliquary Environments

Standalone, independently installable environments for verifiable agentic RL.

Each package has no dependency on Reliquary core and exposes:

- a native Verifiers v1 `Taskset` for evaluation and `prime-rl`;
- a synchronous JSON compatibility surface for Reliquary replay.

| Package | What it is |
| --- | --- |
| [`reliquary-stateful-tools`](environments/tool_use/reliquary_stateful_tools) | A deterministic multi-turn CRM world with exact state-based rewards. |
| [`reliquary-logic`](environments/reasoning/reliquary_logic) | Twelve families of procedurally generated logic puzzle, graded on typed JSON rather than free text. |
| [`reliquary-telecom-solo`](environments/tool_use/reliquary_telecom_solo) | τ²-bench's telecom domain in solo mode: 2,285 support tickets scored by predicates over the device and the carrier's records. |

```bash
cd environments/<area>/<package>
uv sync --locked
uv run pytest
uv build
```

## How the policy is prompted

Every environment declares, in its `environment.toml`, whether its task is meant to be
answered after deliberation or directly:

```toml
[policy]
reasoning = "thinking"
reasoning_rationale = """…"""
```

The environment knows this; the harness running it does not. Left to a harness-wide
default, a format environment ends up rewarding a long chain of thought that talks itself
into satisfying a constraint, and a maths environment ends up answering from the hip.
A rationale is required alongside the value so that whoever changes it knows what it was
weighed against, and CI refuses an environment that declares neither.

Declaring is not enforcing: a Verifiers run carries it as
`env.agent.sampling.chat_template_kwargs.enable_thinking`, and a harness with its own
prompt encoding applies it there. What this repository guarantees is that the intent
travels with the task rather than with the run.

The embedded `reliquary_stateful_tools_v1` and `reliquarylogic_v1` implementations remain in the core repository for historical replay. This repository versions forward; it does not move or rewrite those consensus artifacts. `reliquary-logic` vendors its generator unchanged, so its puzzles are task-for-task identical to the core environment's.

The pinned [Prime-RL v0.9.0 training lane](environments/tool_use/reliquary_stateful_tools/examples/prime_rl) uses the package directly through Verifiers. It remains non-production until GPU qualification is complete.

See [Prime / Verifiers compatibility](docs/prime-verifiers-compatibility.md) for
the exact tested API and wheel pins, installed-package commands, supported
directions and onboarding steps for another environment. Native Verifiers
support does not automatically qualify every Prime environment for Reliquary
consensus.
