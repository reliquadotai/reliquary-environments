# Reliquary Environments

Standalone, independently installable environments for verifiable agentic RL.

The first package is [`reliquary-stateful-tools`](environments/tool_use/reliquary_stateful_tools): a deterministic multi-turn CRM world with exact state-based rewards. It has no dependency on Reliquary core and exposes:

- a native Verifiers v1 `Taskset` for evaluation and `prime-rl`;
- a synchronous JSON compatibility surface for Reliquary replay.

```bash
cd environments/tool_use/reliquary_stateful_tools
uv sync --locked
uv run pytest
uv build
```

The embedded `reliquary_stateful_tools_v1` implementation remains in the core repository for historical replay. This repository versions forward; it does not move or rewrite that consensus artifact.

The pinned [Prime-RL v0.9.0 training lane](environments/tool_use/reliquary_stateful_tools/examples/prime_rl) uses the package directly through Verifiers. It remains non-production until GPU qualification is complete.
