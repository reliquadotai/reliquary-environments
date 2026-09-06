# reliquary-code

OpenCodeInstruct problems graded by executing the model's Python against pinned tests.

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
```
