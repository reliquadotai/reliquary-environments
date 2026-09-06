# reliquary-math

Reliquary core grades every production rollout from its own embedded copy of
this grader (`reliquary/environment/openmathinstruct.py`). This package is a
**port** of that grader — generator-identical logic under a new, standalone
identity — not the authoritative implementation. If core and this package
ever disagree, core is right; open an issue and update the goldens here.

OpenMathInstruct-2 problems with value- and structure-aware answer checking
for verifiable RL. Answers are compared symbolically (via `sympy`) and by
normalized surface form, so equivalent forms such as `\frac{3}{4}` and `0.75`
compare equal without either side changing the model's `\boxed{}` habit.

## Test and load

```bash
uv sync --locked
uv run pytest
uv run python -c "import verifiers.v1 as vf; c=vf.taskset_config_type('reliquary-math'); print(next(iter(vf.load_taskset(c(id='reliquary-math')))).key)"
```

The package exports `MathTaskset` for Verifiers and `MathEnvironment` for
synchronous, Reliquary-compatible replay. It imports no Reliquary code. Row
data is fetched lazily: `task()` and `grade()` both resolve through
`get_problem`, which reads one row-group at a time from the pinned corpus
(`reliquary_math/virtual_parquet.py`) and caches it. The first touch of any
row still not in that cache — or in Hugging Face's local disk cache — issues
an HTTP range read, so grading can hit the network unless the rows involved
were already read.

## Provenance

The task corpus is `nvidia/OpenMathInstruct-2`, pinned at revision
`469216e3f46f4dacf476b382e192485ea51a143e`, distributed under the dataset's
own `cc-by-4.0` licence — separate from this package's MIT licence. Twenty
four goldens in `reliquary_math/goldens/reference.jsonl` pin specific corpus
indices, their rendered prompt hash, and the reward the dataset's own
reference answer must keep scoring (`1.0`, against `0.0` for a well-formed
wrong answer), so a change to the grader or the loader that silently drifts
is caught by `uv run pytest`'s
`tests/test_math.py::test_goldens_replay_against_pinned_corpus`. That test
is not offline: it fetches each golden index's row through the same
`get_problem` path graded rollouts use, so it needs network access (or an
already-warm Hugging Face cache) the same way grading does.
