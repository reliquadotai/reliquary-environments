# Prime / Verifiers compatibility

Reliquary Logic and Stateful Tools are native Verifiers v1 Tasksets. They load
and score through the upstream API without importing Reliquary core, and have
reference configurations for unmodified Prime-RL. Bringing a Prime environment
into Reliquary consensus is a separate, explicit integration.

## Supported directions

| Direction | Qualified surface | Boundary |
| --- | --- | --- |
| Logic → Verifiers | `reliquary_logic:LogicTaskset`; single-turn typed JSON answers; three splits, twelve families | Native load, validation, `Task.score` and Trace/WireTrace scoring. |
| Stateful Tools → Verifiers | `reliquary_stateful_tools:StatefulToolsTaskset`; seven deterministic CRM tools, three task families | Native load/tool declarations, validation and authoritative action replay scoring. |
| Both → Prime-RL | Native tasksets, pinned v0.9.0 configs, GRPO advantages and upstream trainer | CPU config dry-runs passed. The documented two-GPU training/quality run is a separate qualification. |
| Logic → Reliquary | `reliquary/answer-json/v1`, `LogicEnvironment`, external ID `reliquary_logic_v2` | Registered in the explicit Math+Code+Logic V1 profile; artifact and release pin required. |
| Stateful Tools → Reliquary | `reliquary/episode-json/v1`, `StatefulToolsEnvironment`, external ID `reliquary_stateful_tools_v2` | Registered replay/conformance fixture; not a lane in the Math+Code+Logic profile. |
| Another Prime taskset → Reliquary | A reviewed implementation of one of those explicit replay contracts | Installation on the Prime Hub alone does not register or activate it in Reliquary. |
| Every arbitrary Prime ENV → Reliquary | **Not supported as a blanket claim** | Arbitrary tools, judges, mutable services, images, agent control flow and reward conventions do not become deterministic consensus contracts automatically. |

The native API is `import verifiers.v1 as vf`, with `Task`, `Taskset`, `Trace`
and `WireTrace`. It is distinct from the legacy v0
`verifiers.load_environment`/`SingleTurnEnv`/`Rubric` interface. Upstream retains
a legacy bridge, but that bridge is not our Reliquary replay adapter. See the
[upstream API overview](https://github.com/PrimeIntellect-ai/verifiers/blob/b2e4e8157783b2c0dffc7821044c87f29f1c3ccf/docs/overview.md)
and [native taskset contract](https://github.com/PrimeIntellect-ai/verifiers/blob/b2e4e8157783b2c0dffc7821044c87f29f1c3ccf/docs/v1/tasksets.md).

## Exact stack

The machine-readable source of truth is [`compatibility.toml`](../compatibility.toml).

| Component | Tested pin |
| --- | --- |
| Python | 3.12 |
| Verifiers package version | 0.3.1 |
| Verifiers **source** | `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf` |
| Prime-RL v0.9.0 | `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1` |
| Logic wheel 0.1.0a1 | SHA256 `d12e4258fa9b190f29a33a055cee11ae632d813aed5f51b28a20a3727212b527` |
| Stateful Tools wheel 0.1.0a1 | SHA256 `f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155` |

Dependency version ranges in package/source metadata are not evidence that every
matching upstream version is qualified. A Git build can share a version number
with a registry release. `scripts/check_verifiers_pin.py` checks installed
provenance as well as the version before the native qualification jobs run.
Prime-RL also pins its renderer, config library and prime-envs submodules; use
the explicit submodule checks in the reference lane instead of updating them.

## Run the installed packages through native Verifiers

From this repository root, use the existing locked CPU environment and install
the reviewed wheels. No Reliquary core checkout is needed:

```sh
uv sync --locked --project environments/reasoning/reliquary_logic
ENV_PYTHON="$PWD/environments/reasoning/reliquary_logic/.venv/bin/python"
ENV_BIN="$PWD/environments/reasoning/reliquary_logic/.venv/bin"
uv pip install --python "$ENV_PYTHON" --no-deps --reinstall \
  'https://github.com/reliquadotai/reliquary-environments/releases/download/logic-v0.1.0a1/reliquary_logic-0.1.0a1-py3-none-any.whl#sha256=d12e4258fa9b190f29a33a055cee11ae632d813aed5f51b28a20a3727212b527' \
  'https://github.com/reliquadotai/reliquary-environments/releases/download/v0.1.0a1/reliquary_stateful_tools-0.1.0a1-py3-none-any.whl#sha256=f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155'
"$ENV_PYTHON" scripts/check_verifiers_pin.py
"$ENV_PYTHON" -m pytest \
  environments/reasoning/reliquary_logic/tests \
  environments/tool_use/reliquary_stateful_tools/tests --import-mode=importlib
```

Use these explicit binaries after installing the wheels: another `uv sync` or
`uv run` without `--no-sync` may reinstall the editable checkout or change an
explicitly installed dependency to the project's lock entry.

Docker must be available for native runtime validation. Both tasksets declare
`network_allow=[]`; the unsandboxed subprocess runtime cannot enforce that
policy. Keep the deny-all policy and choose Docker:

```sh
"$ENV_BIN/validate" reliquary-logic -n 3 --only-gold true \
  --runtime.type docker --runtime.allow '[]' --rich false
"$ENV_BIN/validate" reliquary-stateful-tools -n 3 --only-gold true \
  --runtime.type docker --runtime.allow '[]' --rich false
```

For a real model eval, supply an actual model endpoint through the upstream CLI
and use `--env.agent.runtime.type docker --env.agent.runtime.allow '[]'`.
Logic uses `--env.agent.harness.id null --env.agent.max-turns 1`; Stateful Tools
uses the tool-capable harness and turn limits in its reference TOML. Always set
`-n`: these tasksets are infinite. A successful `validate --only-gold` checks the
reference checker; it does not execute model inference or prove policy quality.

For Prime-RL, follow the
[exact stack/model-snapshot commands](../environments/tool_use/reliquary_stateful_tools/examples/prime_rl/README.md),
then choose either existing `examples/prime_rl/rl.toml`. The Logic lane needs its
Logic wheel installed too; substituting a config path does not install a package.
Use the pinned local model snapshot, run `rl @ <config> --dry-run True`, and only
then the explicitly scheduled two-GPU smoke. A dry-run does not run an optimizer.
[Prime-RL documents native Verifiers integration](https://github.com/PrimeIntellect-ai/prime-rl/blob/ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1/README.md).

## Onboard another Prime environment into Reliquary

1. **Freeze the candidate.** Record package/source commit, wheel SHA256, license,
   dataset revisions/generator, parser, reward and model/tool dependencies.
   Export the native v1 Taskset through the package's `__all__` and demonstrate
   `vf.taskset_config_type` / `vf.load_taskset` on the pinned upstream API.
2. **Choose the existing replay contract.** A deterministic single-turn textual
   answer can implement `task`, `grade`, `replay`, `reference_completion`
   (`answer-json/v1`). Deterministic sequential tool tasks additionally need
   explicit reset/step/state/replay semantics (`episode-json/v1`). Implement the
   narrow package surface around the original task/checker; do not replace its
   reward with an unrelated approximation merely to fit an adapter.
3. **Establish authoritative replay.** Pin task IDs/splits, all randomness,
   tool schemas and observation/action budgets. Replay must reproduce the final
   reward from canonical task inputs and admitted actions; caller-supplied
   rewards or final state are not authority. Freeze goldens and check correct,
   wrong, malformed, duplicate-key, oversize and interrupted trajectories.
4. **Test both boundaries.** Execute native `Task.score`, serialize/deserialize
   the native trace, then recover actions and replay through the Reliquary ABI.
   Compare reward and state digests. Declare the treatment of branches, sampled
   vs prompt-supplied messages, tool-call IDs and final responses. General
   parallel calls, mixed text/tool turns or multi-agent flows need a separate
   explicit mapping; the sequential adapter does not silently flatten them.
5. **Register the immutable artifact in core.** Add the artifact manifest/file
   hashes and exact registry entry. Use the existing hash-before-import loader,
   install the wheel in the candidate image and qualify that installed image.
   A package installation or registry entry alone does not alter an active
   protocol profile.
6. **Select and qualify a new profile.** Declare prompt/checker contract, caps,
   groups and reward allocation. Exercise assembly/codec/trainer quotas and
   genuine GPU proof/capacity for every added lane. Resource/network isolation
   must match the environment's actual needs. Activate only through the reviewed
   profile/checkpoint/run transition; preserve historical replay artifacts.

The standard adapter does not qualify external LLM judges, mutable web APIs,
browser/GUI worlds, unrecorded nondeterminism, multimodal token/proof formats or
arbitrary multi-agent control flow. Such environments may run perfectly well in
Verifiers while requiring additional Reliquary contracts. Upstream explicitly
separates [environment/agent control flow](https://github.com/PrimeIntellect-ai/verifiers/blob/b2e4e8157783b2c0dffc7821044c87f29f1c3ccf/docs/v1/env.md)
from [harness/runtime execution](https://github.com/PrimeIntellect-ai/verifiers/blob/b2e4e8157783b2c0dffc7821044c87f29f1c3ccf/docs/v1/harnesses.md).

These gates are the onboarding process, not evidence that every Prime catalog
entry has already passed them. The CI results cover the named released packages
and exact pins above.
