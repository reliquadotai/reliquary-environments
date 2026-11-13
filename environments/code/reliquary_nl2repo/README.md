# reliquary-nl2repo

[NL2Repo-Bench](https://github.com/multimodal-art-projection/NL2RepoBench)
([arXiv 2512.12730](https://arxiv.org/abs/2512.12730)): 104 Python projects to build from
a natural-language specification alone (`start.md`, 15-393 KB), each graded by the
upstream project's own test suite, hidden in the task's image. **Evaluation only**: never
trained on. **Verifiers-only**, like `reliquary-swe` and `reliquary-deepswe`: a container
cannot be replayed deterministically. Served on signed-episode sandboxes through the
generic verifiers bridge (`docs/env-norm.md`).

## Provenance and license

| What | Where | Pin |
| --- | --- | --- |
| Tasks (`start.md`, `test_case_count.txt`, `test_commands.json`, `test_files.json`) | [multimodal-art-projection/NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench) `test_files/` | `781a1da1ee41fb8edb0bed22f586d69111610edf` |
| Images (`sandbox-images.lock.json`) | `ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0` | 104 tags, 104 digests, 55 GB compressed (`scripts/pin_image_lock.py`) |

**Upstream publishes no license.** The task files are therefore never committed here:
`reliquary_nl2repo/_source.py` pins the sha256 of each of the 416 files, and
`scripts/vendor_tasks.py` fetches them from a clean checkout at the pin into
`reliquary_nl2repo/tasks/` (gitignored), refusing any byte that differs. The wheel carries
them (hatch `artifacts`): whoever builds and distributes it must have the right to.
`environment.toml` says `redistributable = false`. This package's own code is Apache-2.0.

## Splits

| Split | Tasks | For |
| --- | --- | --- |
| `eval` | the gold-checked tasks (`reliquary_nl2repo/gate.py`) | what a gateway serves and a Catalyst eval set freezes |
| `eval_all` | all 104 | the gold check and conformance runs |

`eval` refuses to load until the gold check has recorded its verified list. Task order is
the task id's (upstream's directory name), sorted; `task_at(i)` is the `i`-th task
`load()` yields.

## The task (env norm)

- **Box**: the task's image, workdir `/workspace`, 4 CPUs, 8 GB memory, 20 GB disk (Prime's
  adaptation's sizes), no network. Agent timeout 7,200 s: upstream bounds nothing
  (OpenHands with no iteration limit); the order's turn and token budget binds first.
  Tools: `bash` and `edit`.
- **Prompt**: upstream's instruction, unchanged ("According to the start.md in the
  workspace, implement the entire project ..."); the specification is `start.md` in the
  workspace, bound into the task hash by its digest (`spec_sha256`).
- **`setup`**: the images carry the reference implementation in several places, found by
  scanning all 104 (`reliquary_nl2repo/hygiene.py`): 21 hold a distribution installed
  from the project's directory (7 non-editable: the reference is in site-packages), 24 a
  `setup.py develop` link, four a source copy outside `/workspace` (asteval, jinja,
  mootdx, paillier), one a build tree in `/tmp`, most a pip cache. `setup` removes the
  copies and caches, uninstalls every `dir_info` distribution (PEP 610) and develop link
  with the box's own Python (3.7-3.13), empties `/tmp` and `/workspace` (hidden tests,
  upstream packaging), then writes `start.md`. A failure is ours: the open fails.
- **`finalize`**: `/workspace` as `/logs/artifacts/workspace.tar.gz`: regular files only,
  `.git`, `__pycache__`, virtualenvs, `node_modules`, caches, `*.egg-info` and top-level
  `build/`, `dist/` left out, at most 24 MiB (the bridge carries 32). The artifact
  directory is emptied first, so nothing planted there travels.
- **`grading_setup`**: the same hygiene in the pristine box, `/workspace` kept; a box
  without test files is ours.
- **`pass_rate`** (reward, in [0, 1]), in that box with the archive restored:
  1. the archive is read into the scoring process and rebuilt (`submission.sanitize`):
     regular files under relative paths only; dropped are upstream's packaging files
     (its `remove_package_files`), the task's `test_files.json` paths, toolchain
     directories, and what could change how the tests run rather than what they test:
     every `conftest.py`, `sitecustomize.py`, `usercustomize.py`, `*.pth`, pytest config,
     module shadowing pytest or a plugin (the workspace is on `PYTHONPATH`), and file
     pytest would collect (`test_*.py`, `*_test.py`: several upstream commands run a bare
     `pytest` over the workspace). Unreadable: 0;
  2. the clean archive is extracted over `/workspace` with `--skip-old-files`: the
     image's hidden tests and packaging always win;
  3. upstream's `test_commands.json` run in order in bash (upstream runs them without a
     shell, which breaks seven tasks' commands; `set X=Y` becomes an export, as in
     Prime's adaptation), with `PYTHONPATH=/workspace:/workspace/src`, pip offline
     (`PIP_NO_INDEX`, `--no-build-isolation --no-deps`: where the image lacks the build
     backend the install fails and `PYTHONPATH` carries the tests), each pytest run
     writing a JUnit report, all within 660 s;
  4. reward = min(passed / `test_case_count`, 1), upstream's per-task metric; passed test
     cases are counted from the JUnit reports in the scoring process (no DTD, size-capped),
     never from the box's say-so. Upstream's stdout count is kept as a metric
     (`summary_passed`).
- **Timeouts**: setup 300 s, finalize 300 s, scoring 810 s.
- **Residual**, as for every repository-test grader: the candidate's code runs during the
  tests and can rewrite the reports or patch pytest at import time.

## Gate: the gold check

NL2Repo-Bench publishes no reference solution. The images were built from the reference,
and most still hold it in a build layer: `scripts/reference_source.py` replays each
image's layers (`docker save`) and recovers the source directory as it stood before its
deletion. Run on a gVisor host with the images pulled by digest (never a production
gateway); the references stay on that host (upstream source, never redistributed).

```bash
uv run python scripts/reference_source.py --out /root/refs
uv run python scripts/gold_check.py --refs /root/refs --jobs 4 --out gold.jsonl
uv run python scripts/gold_check.py --write-gate --from gold.jsonl   # rewrites gate.py
uv run python scripts/gold_check.py --conformance                    # attack cases
```

Per task, two episodes through this package's own hooks: the reference unpacked into the
emptied workspace, and the workspace left empty; after `setup`, a leak probe tries to
import the reference's top-level modules. A task is verified when its reference grades 1,
its empty workspace 0, both within `--budget` (700 s), and nothing leaked.

## Conformance (`conformance_cases`, `reference_calls`)

No reference ships (`reference_calls` is None). On the goldens (`parse`, `retrying`;
they must be in the verified list), every case is expected to grade 0: the empty
workspace, a root `conftest.py` forcing passes, the same as `sitecustomize.py` and as a
`.pth`, a `pytest` package shadowing the runner, pytest config loading a forcing plugin,
symlinks, and an archive planted at the artifact path.

## Images and disk

104 images, one per task, 55 GB compressed (about 28 GB of distinct layers), 0.08-9.4 GB
each (`pytorch-grad-cam` the largest). `scripts/pin_image_lock.py --list` prints the
digests a gateway pre-pulls; `--verify` re-resolves every tag and exits 1 when one moved.

## Development

```bash
python scripts/vendor_tasks.py <NL2RepoBench checkout at the pin>   # once
uv sync --locked
uv run pytest          # no Docker, no network
uv build
```
