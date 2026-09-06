# Publishing math, code and logic to `reliquary-environments` — 2026-09-06

## Goal

Give `math`, `code` and `logic` a second, standalone home in
[reliquary-environments](https://github.com/reliquadotai/reliquary-environments),
following the shape `reliquary_stateful_tools` and `reliquary_logic` already
established. This is a **publication, not a migration**: the embedded copies in
Reliquary core stay exactly where they are and keep grading production.

Nothing here changes a protocol contract, a profile, or a reward. A validator
that never installs these packages is byte-identical before and after.

## Why this is cheap

Every expensive property of an environment move comes from *removing* the core
copy, and we are not removing it:

| Cost usually attached to a move | Why it does not apply |
|---|---|
| Generation-contract digest changes | `environment_manifest_sha256` in `profiles.py` is unchanged; no profile references the new packages |
| Loader must accept `single_turn` | We are not loading these externally; `registry.py` gains no entry |
| Archived-window replay breaks | The core graders that produced the archives are untouched |
| Coordinated miner release | `PROTOCOL_GENERATION_CONTRACT` is unchanged, so no `request_profile_mismatch` |

The one real constraint is the reverse: the published packages must not become
a second source of truth people assume is authoritative. `[provenance]` carries
`port = "generator-identical-new-identity"`, the same marker `reliquary_logic`
already uses, and each README states plainly that Reliquary core grades from
its own copy.

## Scope

**In:**
- `reasoning/reliquary_math` — new package
- `code/reliquary_code` — new package
- `reasoning/reliquary_logic` — already written, sitting in PR #2; merge it
- one release, `v0.1.0a2`, carrying all three

**Out:**
- `envscaler` — its corpus is not vendored, the exact bytes the 2026-09-02
  measurements ran against (`env_meta.json` `d2c0010f…`, `rl_scen.json`
  `5977bda0…`) are not recoverable locally, and upstream has moved. Publishing
  an environment whose corpus cannot be pinned defeats the `artifact.json` +
  goldens contract. It returns when a pinned fork exists.
- `reliquaryverifiable_v1`, and the three `*_tools_v1` episode environments —
  not requested.
- Any change to `reliquary/environment/` in core.
- Any change to the external loader (`agentic/external.py`), which still
  refuses non-episode distributions. That extension belongs to a future
  decision to *replace* the embedded copies, not to this one.

## Package layout

Both new packages follow `reliquary_logic` exactly:

```
environments/<family>/<name>/
├── LICENSE                       MIT (the port); dataset licence declared separately
├── README.md                     what it is, how to train it, that core grades its own copy
├── environment.toml              schema reliquary/environment-source/v2
├── pyproject.toml                hatchling, verifiers>=0.3.1,<0.4, python >=3.12,<3.13
├── uv.lock
├── examples/prime_rl/
│   ├── README.md
│   └── rl.toml
├── <name>/
│   ├── __init__.py               exports Taskset + Environment
│   ├── artifact.json             schema reliquary/environment-artifact/v1
│   ├── goldens/reference.jsonl
│   └── <modules>.py
└── tests/test_<name>.py
```

Families: math goes to `reasoning/` beside logic; code gets a new `code/`
directory. `environment.toml` ids are `reliquary/math` and `reliquary/code`.

`artifact.json` lists every package file with its SHA-256 plus a
`source_manifest_sha256`, exactly as logic does. It is generated, never
hand-edited, and a test regenerates and compares it.

## Corpus

Both environments are corpus-backed, unlike logic which is procedural. Neither
needs a new dataset published, and neither can vendor its corpus.

The measurements that settle it:

| | Size |
|---|---|
| `nvidia/OpenMathInstruct-2`, all files | 12.63 GB |
| — its 32 `train-*` shards alone | 7.58 GB |
| `R0mAI/opencodeinstruct-curated`, single parquet | 1.39 GB |
| `reliquary/environment/virtual_parquet.py` | 411 lines, zero `reliquary` imports |

**Vendoring the corpus in git is out.** 1.39 GB for code alone, against
GitHub's hard 100 MB per-file limit. The data stays on Hugging Face.

**Publishing a new curated dataset is also unnecessary**, which is the
correction this section exists to record. `VirtualParquetDataset` reads only
the row-groups actually touched, over HTTP range requests, so a 12 GB dataset
costs no bulk download and roughly no RAM. It is 411 lines of stdlib with no
dependency on anything else in core.

So `virtual_parquet` **is** ported — copied into each package, not left behind.
The earlier draft of this spec said the opposite, on the reasoning that a
standalone trainer wants the ordinary `datasets.load_dataset` path. That was
wrong: the ordinary path costs a 7.58 GB download for math and reintroduces the
shard trap below.

| Env | Coordinate | Revision |
|---|---|---|
| math | `nvidia/OpenMathInstruct-2` | `469216e3f46f4dacf476b382e192485ea51a143e` |
| code | `R0mAI/opencodeinstruct-curated` | `d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7` |

**The shard trap, stated so the port does not reintroduce it.**
`nvidia/OpenMathInstruct-2` carries `train_1M`, `train_2M` and `train_5M`
subsets alongside the full split, duplicating rows already present. Selecting
the wrong `data_dir` silently yields ~36% duplicates. Core guards this with
`OMI_TRAIN_SHARDS_ONLY`; the package selects its 32 `train-*` files explicitly
and a test asserts the file list, which removes the trap by construction rather
than by flag.

Porting `virtual_parquet` has a second benefit worth naming: index addressing
becomes identical to core, so a golden pinned at index *i* means the same row
in both repos. That makes the goldens genuinely useful as cross-checks and
leaves the door open to a faithful-copy path later without redoing the corpus
layer.

The cost is 411 duplicated lines per package. That is the right trade here:
the repo's shape is one self-contained package per directory, each with its own
`pyproject.toml` and `uv.lock`, and a shared distribution would couple the
packages to each other for no gain at two consumers.

**Prerequisite — declare the code subset's licence.**
`R0mAI/opencodeinstruct-curated` is public but its dataset card declares no
licence and carries no licence tag. It derives from `nvidia/OpenCodeInstruct`
(CC-BY-4.0), so it should declare CC-BY-4.0 with attribution before a public
package depends on it. This is a card edit, not a publication. Math points
straight at `nvidia/OpenMathInstruct-2`, which already declares CC-BY-4.0.

With that done, `[data] redistributable = true` and `license = "cc-by-4.0"`
are accurate. Attribution goes in `NOTICE` at the repo root, which already
exists, and in each README.

**Consequence to state in both READMEs:** row order is the corpus identity for
both. Prompt index *i* in the package resolves to the same row as index *i* in
core only while the pinned revision matches. The goldens pin that.

## Porting the graders

The core environments expose four methods: `__len__`, `get_problem(index)`,
`compute_reward(problem, completion)`, `source_health()`. The port keeps the
grading functions and drops the core plumbing.

**What is copied verbatim** — these are the environment, and copying them is
what makes the port faithful enough to be useful:

- math: `_last_boxed_only_string`, `_strip_boxed_wrapper`, `_normalize_answer`,
  `_as_number`, `_latex_to_pyexpr`, `_expr_is_safe`, `_expr_str_is_safe`,
  `_latex_value_equal`, `_expand_term_bound`, `_latex_symbolic_equal`,
  `_split_structure`, `_answers_equal`, `_compute_omi_reward`
- code: `_entry_function_name`, `_defines_top_level_entry`,
  `_select_python_span`, `_extract_python`, `_contract_instruction`

`_expr_is_safe` and `_expand_term_bound` are load-bearing, not incidental: they
are what stops a model's answer string from becoming arbitrary sympy evaluation
or a memory bomb. They carry their core comments across unchanged.

**What is replaced:**

| Core dependency | Replacement |
|---|---|
| `profiles.render_active_prompt` | a `prompt_template` argument on the taskset, defaulting to the current v5 template copied into the package |
| `constants.RAW_COMPLETION_PROMPTS`, `MATH_ANSWER_FORMAT`, `OMI_TRAIN_SHARDS_ONLY` | package-level defaults, overridable through the constructor |
| `constants.GRADER_EVAL_TIMEOUT_SECONDS`, `PROTOCOL_VERSION` | package defaults |
| `environment.virtual_parquet` | the same file, copied into the package (see Corpus) |
| `environment.grader_client.GraderClient` | the local runner below |

The prompt template moving from a core lookup to a package argument is the one
place the port is deliberately *not* identical to core: core renders from the
active profile, and a standalone package has no active profile.

## Running the code environment

Core grades code through `GraderClient`, a separate gVisor-sandboxed service.
A standalone package cannot assume that service exists, so it ships its own
runner with the protections that do not need gVisor:

- one **fresh subprocess per evaluation** — never a pooled worker. `RLIMIT_CPU`
  is cumulative per process, and a worker serving many evaluations hits the
  limit on an innocent later case and dies. That cost 12% of code submissions
  in production before it was found; the package must not reintroduce it.
- `RLIMIT_CPU` and `RLIMIT_AS` set in the child before `exec`
- wall-clock timeout on top of the CPU limit, because a sleeping process burns
  no CPU
- no network: the child gets a closed environment, and `[execution] network =
  false` declares it
- stdout/stderr captured and truncated to `max_observation_bytes`

This is weaker than gVisor and the README says so in those words. It is
defence in depth for a package whose whole job is executing model-written
Python on someone else's machine, not a claim of containment.

## Goldens and tests

Each package ships `goldens/reference.jsonl`, one line per pinned index, in the
shape logic already uses, adapted to single-turn:

```json
{"index": 41, "split": "train", "prompt_sha256": "...", "expected_answer_sha256": "...", "reference_reward": 1.0, "wrong_reward": 0.0}
```

`reference_reward` and `wrong_reward` are the two facts that make a golden
worth having: the dataset's own reference answer must score 1.0, and a
deliberately wrong answer must score 0.0. An environment that fails either is
broken whatever else it does.

Tests per package:

1. **goldens replay** — every pinned index reproduces its `prompt_sha256`, and
   the reference and wrong answers reproduce their rewards
2. **artifact integrity** — regenerating `artifact.json` yields the committed file
3. **grader unit tests** — the surface-form cases the core grader learned the
   hard way: LaTeX equivalence, `\boxed{}` extraction, fenced-block selection
   with a language tag, the last parsing fenced answer
4. **code runner** — timeout fires, memory limit fires, network is absent, one
   process per evaluation

Point 3 matters more than it looks. The core graders carry fixes for six
recurrences of the same fenced-extraction bug family and a 28%→0.9%
false-negative reduction on math surface forms. A port that silently drops
those regression cases will rediscover them.

## Release

After all three packages are in `main`:

- tag `v0.1.0a2`, with a wheel + sdist + `SHA256SUMS` + release-lock per package,
  matching how `v0.1.0a1` shipped
- `compatibility.toml` gains a `[releases.*]` block per package with its wheel
  SHA-256, and `[reliquary] source_commit` moves to the core commit the ports
  were taken from
- `docs/standalone-environment-qualification.md` in core is **not** touched:
  it documents the loader path for `reliquary_stateful_tools_v2`, and none of
  these three packages is loadable by that path

## Order

0. **Prerequisite** — declare CC-BY-4.0 on the `R0mAI/opencodeinstruct-curated`
   dataset card. A card edit; blocks nothing structurally but should land before
   the code package is public.
1. Merge PR #2 — logic is already written and `MERGEABLE`
2. `reasoning/reliquary_math`
3. `code/reliquary_code`
4. Release `v0.1.0a2` and update `compatibility.toml`

Steps 1, 2 and 3 are mutually independent and can land in any order or in
parallel PRs. Nothing waits on a dataset being published.

## Success criteria

- `uv run pytest` green inside each package directory, with no Reliquary core
  on the path — the packages must not import `reliquary.*`
- each package trains a smoke run under the pinned Prime-RL from
  `examples/prime_rl/rl.toml`
- core's test suite is untouched and unchanged: same count, same result
- no diff in `reliquary/environment/`, `reliquary/protocol/profiles.py`, or
  `reliquary/constants.py`

## Explicit non-goals

- replacing the embedded copies
- extending the external loader to `single_turn`
- giving `openmathinstruct` or `opencodeinstruct` a core manifest
- reproducing core's grading bit-for-bit; `port = "generator-identical-new-identity"`
  is the honest label, and the prompt-template change alone breaks bit-identity
- publishing any new Hugging Face dataset; both packages read the revisions
  core already pins
