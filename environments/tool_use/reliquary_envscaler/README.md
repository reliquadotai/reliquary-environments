# reliquary-envscaler

EnvScaler's tool-interactive worlds, scored by their own checklists. 191
synthesised worlds, 2,550 scenarios, tool inventories that differ per world —
15 tools on the first scenario alone — and rewards that are a fraction of the
checks the agent is asked to flip.

## Why the tools are declared differently

The sibling packages list a fixed set of `@vf.tool` methods because their
worlds share one roster. EnvScaler's roster belongs to the task, not to the
environment, so `EnvScalerToolset.register` builds it from the world instead
of from decorators.

That works because Verifiers gives each rollout its own tool server and runs
`setup_task` before `register` (`verifiers/v1/mcp/server.py`):

```python
await self.setup()
await self._setup_task_from_channel(...)   # setup_task(task)
mcp = FastMCP(...)
self.register(mcp)                          # after
```

The world itself is a live Python object — the release ships each world as
class source, `exec`'d at reset — so it is held beside the state rather than
in it. `EnvScalerState` carries a key, never the object.

## Reward

Upstream averages every check. That keeps the reward continuous, which is what
a variance gate needs, but pays for the **15.5% already true at reset**: a
no-op agent scores 0.166 and the usable range shrinks. Here the denominator is
restricted to the checks that start false. Measured on Qwen3-4B, groups
clearing `SIGMA_MIN` go from 2.1% to 10.4% with no change to the gate.
`RELIQUARY_ENVSCALER_REWARD=upstream` restores their definition so the two are
comparable on one generation.

Checklists have a **median of 14 items**, so flipping one is worth 0.07 and
group sigma sits around 0.05. A gate calibrated for near-binary rewards will
discard almost every group; that is a fact about the gate, not the model.

## Data

60 MB, too large for a wheel, so `corpus.py` names the upstream datasets and
checks what it gets: 191 worlds and 2,550 scenarios, or it refuses. Set
`RELIQUARY_ENVSCALER_DATA` to a directory to use a local copy.

Two fields arrive as JSON *strings* inside the JSON — `tools` on a world,
`init_config` on a scenario — and are decoded on load. That belongs to the
package because it is what the release means, not a preprocessing convenience.

**Only the RL scenarios are included.** The 4,684 SFT ones ship no
`checklist_with_func`, so no answer to them can be scored, and an environment
that cannot grade is not an environment. That costs 89 of the 191 worlds.

## Splits

The first 250 scenarios are the `eval` split and nothing else may use them;
`train` starts after. Frozen, because the band is measured on these worlds and
a teacher must never be handed a task the evaluation will ask about.

## Use

```bash
uv sync --locked
uv run pytest
uv run eval reliquary-envscaler -n 8 -m <model> --client.base-url <endpoint>
```

The package exports `EnvScalerTaskset` for Verifiers and
`EnvScalerToolsEnvironment` for synchronous replay. It imports no Reliquary
core; the episode contract is vendored in `episode.py`.

## Caveat

Worlds and checkers are dataset-carried Python, executed at reset and at
grading. Run it sandboxed. The environment's own `step` is forgiving by
design — an unreadable turn or an unknown tool is an observation, not a
termination — because upstream's is, and `RELIQUARY_ENVSCALER_TOOL_ERRORS`
bounds it.
