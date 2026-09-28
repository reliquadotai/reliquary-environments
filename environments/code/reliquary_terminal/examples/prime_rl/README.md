# Prime-RL v0.9.0

Same stack, pins and install steps as `reliquary-swe`'s example
(`environments/code/reliquary_swe/examples/prime_rl/README.md`); install
this package instead, with its `harbor` dependency:

```bash
uv pip install --python prime-rl/.venv/bin/python \
  -e "$RELIQUARY_ENVS_PATH/environments/code/reliquary_terminal"
```

`rl.toml` evaluates on Terminal-Bench 2.1 (`split = "eval"`) and ships with
no train source, so a bare copy fails fast. The corpus comes from
`train-sources.toml`:

```bash
EXAMPLES="$RELIQUARY_ENVS_PATH/environments/code/reliquary_terminal/examples/prime_rl"
prime-rl/.venv/bin/rl @ "$EXAMPLES/rl.toml" \
  --orchestrator.train @ "$EXAMPLES/train-sources.toml" \
  --model.name "$MODEL_SNAPSHOT_PATH" --max-steps 5 \
  --output-dir outputs --run.name reliquary-terminal-smoke
```

Pre-pull the 64 training images (~20 GB compressed) and Terminal-Bench's
89 (~18 GB compressed) before a real run. Turn and token budgets are
unmeasured; start with a few steps.
