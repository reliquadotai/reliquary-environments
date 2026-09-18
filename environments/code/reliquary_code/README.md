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
test case (`reliquary_code/runner.py`): CPU time, address space and file
size are capped, and a wall-clock timeout backstops the CPU limit for
processes that sleep instead of spending CPU. Each case gets its own
subprocess — never a pooled worker — because `RLIMIT_CPU` is cumulative for
the life of a process, and a shared worker would eventually be killed on an
innocent case after a costly-but-legal earlier one.

The limits are applied by the child to itself, as the first statements of its
own program, and not by a `preexec_fn`. The earlier version used one and stated
the assumption that justified it: that callers are single-threaded batch or
test code. A validator grades concurrently, so that assumption is not one this
package is entitled to make, and it fails badly when it is wrong. `preexec_fn`
runs in the forked child between fork and exec; a fork carries only the calling
thread, so a lock another thread held at fork time arrives locked with no owner
to release it. The child blocks on it forever, never reaches exec, never writes
the error pipe that `Popen.__init__` is reading — and the parent hangs in the
constructor, before `communicate`'s timeout exists to save it. There is no
timeout on that path. A twelve-thread probe hit it after about an hour of
grading and stayed stuck until it was killed.

Two consequences of moving the limits, both conservative and both deliberate:
`RLIMIT_CPU` is cumulative from process start, so the interpreter's own startup
now counts against the submission's budget, and `RLIMIT_AS` lands after the
interpreter has allocated, so what remains is slightly under the nominal cap.
A submission is judged a little more harshly than before, never more leniently.

Process count is **not** capped. `RLIMIT_NPROC` was tried and removed: on
Linux it is scoped to the real UID rather than to this process's subtree, so
on a box where that UID already owns hundreds of processes a legitimate
`subprocess`- or `multiprocessing`-using submission hits `EAGAIN` on its
first fork and is scored False for reasons that have nothing to do with its
own behaviour. Containment of runaway forks instead comes from
`start_new_session=True` plus `os.killpg` on the process group at timeout,
which reaches forked descendants directly.

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

## How the policy is prompted

`environment.toml` declares `reasoning = "thinking"` and a budget of **8,192**
tokens. Both are measured, not chosen.

Reasoning is safe here because the extractor was repaired so that it is. Under
"last fenced block wins", a rollout that closed with a usage demo or a test
block had *that* span executed instead of its implementation — 13.1% of code
rollouts at the v5 cutover — and a correct solution scored zero. Group-relative
advantage turned those zeros into "never open a second block", which the policy
generalised into "never reason". The extractor now selects the last block that
*defines* the entry function, so deliberating costs tokens and cannot cost
reward.

The budget is the more surprising number, because it goes the other way from
the maths sibling. Measured on the policy this environment trains, 16 rollouts
per prompt, 96 prompts strided across the whole corpus:

| budget | mean success | too easy (n/n) | band (σ ≥ 0.24) | in-band failures |
| --- | --- | --- | --- | --- |
| **8,192** | 70.1% | 21.9% | **45.8%** | 49.3% capacity, 50.7% truncation |
| 24,576 | 73.2% | 24.0% | 42.7% | 82.3% capacity, 17.7% truncation |

Raising the budget is not neutral here, it is worse. The extra tokens do fix
truncation — in-band truncation falls from 50.7% to 17.7% — but the groups that
stop being truncated do not become harder, they become unanimous, and the
too-easy share rises. The budget was never the constraint. A Python function
that passes its cases is short; `reliquary-dapo-math` needs 24,576 for the
opposite and equally measured reason.

One methodological note, because it changed the answer. A first pass strided by
977 over 96 prompts, which covers the first 93,792 rows of 2,481,806 — 3.8% of
the corpus, and still a corner of it. It read the band at 36.5% with a
best-of-16 headroom of ×1.1, i.e. "exhausted, not worth training on". Across
the whole corpus the same measurement reads 45.8% and ×1.3. The stride is now
derived from the corpus length, which is what this package's own goldens do.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-code'); print(next(iter(vf.load_taskset(c(id='reliquary-code')))).key)"
```

The package exports `CodeTaskset` for Verifiers and `CodeEnvironment` for
synchronous, Reliquary-compatible replay. It imports no Reliquary code. Row
data is fetched lazily: `task()` and `grade()` both resolve through
`get_problem`, which reads one row-group at a time from the pinned corpus
(`reliquary_code/virtual_parquet.py`) and caches it. The first touch of any
row still not in that cache — or in Hugging Face's local disk cache — issues
an HTTP range read, so grading can hit the network unless the rows involved
were already read.

## Provenance

The task corpus is `R0mAI/opencodeinstruct-curated`, pinned at revision
`d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7`, distributed under the dataset's
own `cc-by-4.0` licence — separate from this package's MIT licence. This
curated subset carries no reference program to check a submission against,
only per-problem structured test cases, so unlike `reliquary-math` there is
no reward this package can pin at `1.0` from a golden alone (see
`test_correct_completion_scores_one_on_a_real_corpus_row` in `test_code.py`
for a synthetic correct solution instead). Twenty four goldens in
`reliquary_code/goldens/reference.jsonl` instead pin specific corpus
indices, their rendered prompt hash, and the reward a well-formed non-answer
must keep scoring (`0.0`), so a change to the extractor, the runner, or the
loader that silently drifts is caught by `uv run pytest`'s
`tests/test_code.py::test_goldens_replay_against_pinned_corpus`. That test
is not offline: it fetches each golden index's row through the same
`get_problem` path grading uses, so it needs network access (or an
already-warm Hugging Face cache) the same way grading does.
