# Reliquary Environments

Standalone, independently installable environments for verifiable agentic RL.

Each package has no dependency on Reliquary core and exposes:

- a native Verifiers v1 `Taskset` for evaluation and `prime-rl`;
- a synchronous JSON compatibility surface for Reliquary replay.

| Package | What it is |
| --- | --- |
| [`reliquary-stateful-tools`](environments/tool_use/reliquary_stateful_tools) | A deterministic multi-turn CRM world with exact state-based rewards. |
| [`reliquary-logic`](environments/reasoning/reliquary_logic) | Twelve families of procedurally generated logic puzzle, graded on typed JSON rather than free text. |

```bash
cd environments/<area>/<package>
uv sync --locked
uv run pytest
uv build
```

The embedded `reliquary_stateful_tools_v1` and `reliquarylogic_v1` implementations remain in the core repository for historical replay. This repository versions forward; it does not move or rewrite those consensus artifacts. `reliquary-logic` vendors its generator unchanged, so its puzzles are task-for-task identical to the core environment's.

The pinned [Prime-RL v0.9.0 training lane](environments/tool_use/reliquary_stateful_tools/examples/prime_rl) uses the package directly through Verifiers. It remains non-production until GPU qualification is complete.
