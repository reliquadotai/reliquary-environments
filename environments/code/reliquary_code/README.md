# reliquary-code

Reliquary core grades every production rollout from its own embedded copy of
this grader (`reliquary/environment/opencodeinstruct.py`). This package is a
**port** of that grader — generator-identical logic under a new, standalone
identity — not the authoritative implementation. If core and this package
ever disagree, core is right; open an issue and update the goldens here.

OpenCodeInstruct problems graded by executing the model's Python against
pinned test cases. Reward is the fraction of a problem's cases the model's
extracted code passes.

## Isolation

Grading runs model-written Python in a fresh, resource-limited subprocess per
test case (`reliquary_code/runner.py`): CPU time, address space, process
count and file size are capped, and a wall-clock timeout backstops the CPU
limit for processes that sleep instead of spending CPU. Each case gets its
own subprocess — never a pooled worker — because `RLIMIT_CPU` is cumulative
for the life of a process, and a shared worker would eventually be killed on
an innocent case after a costly-but-legal earlier one.

This is weaker than the gVisor-sandboxed grader service Reliquary core uses,
and it does **not** block network access: a plain `subprocess` with a
scrubbed environment can still open sockets if the host machine can reach
the network. `environment.toml` declares `network = false`, but that
declaration is enforced by the embedding harness, not by this runner —
exactly as `network_allow=[]` on core's `TaskData` is enforced by core's
harness rather than by the environment itself. Reliquary core grades from
its own embedded copy of this logic behind the real sandbox; do not treat
this package's subprocess limits as a network or filesystem containment
boundary.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-code'); print(next(iter(vf.load_taskset(c(id='reliquary-code')))).key)"
```

The package exports `CodeTaskset` for Verifiers and `CodeEnvironment` for
synchronous, Reliquary-compatible replay. It performs no network access at
grading time (the corpus is read once, ahead of time, through a pinned
row-group-streaming loader) and imports no Reliquary code.

## Provenance

The task corpus is `R0mAI/opencodeinstruct-curated`, pinned at revision
`d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7`, distributed under the dataset's
own `cc-by-4.0` licence — separate from this package's MIT licence. This
curated subset carries no reference program to check a submission against,
only per-problem structured test cases, so unlike `reliquary-math` there is
no reward this package can pin at `1.0`. Twenty four goldens in
`reliquary_code/goldens/reference.jsonl` instead pin specific corpus
indices, their rendered prompt hash, and the reward a well-formed non-answer
must keep scoring (`0.0`), so a change to the extractor, the runner, or the
loader that silently drifts is caught by `uv run pytest` offline, without
touching the network again.
