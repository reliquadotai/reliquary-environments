# reliquary-stateful-tools

A deterministic, procedural CRM environment for multi-turn tool use. Tasks cover shipping-address updates, exact refunds, and support notes. Success requires the exact intended final world state; collateral mutations receive zero reward.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-stateful-tools'); print(next(iter(vf.load_taskset(c(id='reliquary-stateful-tools')))).key)"
```

For a model evaluation, use the tool-only `null` harness and bound the infinite procedural taskset:

```bash
uv run eval reliquary-stateful-tools -n 12 --env.agent.harness.id null --env.agent.runtime.type docker --env.agent.runtime.allow '[]' --env.agent.max-turns 8
```

The package exports `StatefulToolsTaskset` for Verifiers and `StatefulToolsEnvironment` for synchronous Reliquary-compatible replay. It performs no network access and imports no Reliquary code.
