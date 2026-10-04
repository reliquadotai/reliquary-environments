# reliquary-competitive-code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new environment package, `reliquary-competitive-code`, that serves competitive-programming problems and grades a Python program by running it on hidden stdin/stdout tests, built from a curated, pinned dataset assembled from DeepCoder.

**Architecture:** One package, three separated layers: `judge/` (run + compare, knows no dataset), `sources/` (one adapter per upstream dataset, emitting a common `SourceRow`), `build/` (offline: filter, decontaminate, deduplicate, validate with a reference, calibrate, cap, split, write parquet). The served side (`corpus.py`, `environment.py`, `taskset.py`) reads the published curated dataset only. The Catalyst core integration (gVisor `stdin` mode, registry entry) is a separate, later plan.

**Tech Stack:** Python 3.12, uv, hatchling 1.27.0, pyarrow, huggingface-hub, numpy (build only), verifiers @ `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`, pytest 8.

**Spec:** `docs/superpowers/specs/2026-10-04-reliquary-competitive-code-env-design.md`

## Global Constraints

- Package directory: `environments/code/reliquary_competitive_code/`, import name `reliquary_competitive_code`, distribution `reliquary-competitive-code`, version `0.1.0a1`.
- Environment name `reliquary_competitive_code_v1`; task family `competitive_programming_stdio_v1`; judge version `stdio-tokens-v1`.
- `requires-python = ">=3.12,<3.13"`; `verifiers==0.3.1` sourced from git rev `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`; `pyarrow>=17`; `huggingface-hub>=0.25`; numpy only in the `build` dependency group.
- Reward is binary: 1.0 iff every kept test passes. Stop at the first failing test.
- Comparator: whitespace tokens; equal strings; `yes`/`no` case-insensitive; float tolerance 1e-6 (relative or absolute) only when the expected token is a decimal (contains `.`, `e` or `E`); integers exact.
- Time limit per test: `clamp(4 x reference slowest CPU, 1 s, 4 s)`; problems whose reference needs more than 2 s on a test are dropped (keeps the margin >= 2x).
- Test cap per problem: reference CPU total <= 8 s and test bytes <= 1 MiB; at least 5 tests kept.
- Held-out cut-off: contest date >= `2024-08-01` excluded; LCB decontamination against LiveCodeBench `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505` problems dated >= `2024-08-01`, 13-gram overlap >= 10 % drops the problem.
- DeepCoder pin: `agentica-org/DeepCoder-Preview-Dataset@177913a7bd43791646ef6a43645caa3c871ab3db`, configs `taco` and `primeintellect` train only. Never `lcbv5/test` or `codeforces/test`. `lcbv5/train` is skipped in v1 (no reference solutions).
- Splits by hash of `problem_id` with salt `reliquary_competitive_code_v1`: train 95 %, eval 2.5 %, qualification 2.5 %.
- Never run the full build or heavy test suites on the local VPS beyond the package's unit tests (22 GB RAM shared with production). Data lives in `/home/ubuntu/cc-data` (on disk, not `/tmp`).
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. Fast-input idioms (`input = sys.stdin.buffer.readline`, `sys.stdin.read().split()`) must score 1.0 — test in Task 3.
2. Deep recursion through `threading.stack_size(...)` + `threading.Thread(target=main)` must score 1.0 — test in Task 3.
3. A program that prints forever must end as `output_limit` (bounded memory), not hang or exhaust RAM — test in Task 3.
4. A completion truncated inside its fence, with no code, or whose last block is ```` ```cpp ```` must grade 0 with status `no_code`, never crash — tests in Tasks 1 and 8.
5. `exit()` / `sys.exit(0)` after printing is a normal end (reward possible); `sys.exit(1)` is `runtime_error` — test in Task 3.

---

## File Structure

```
environments/code/reliquary_competitive_code/
  pyproject.toml, environment.toml, README.md, LICENSE, .gitignore, uv.lock
  reliquary_competitive_code/
    __init__.py            lazy Verifiers surface (as reliquary_science)
    extraction.py          extract_program(completion) -> str | None
    judge/__init__.py      re-exports TestCase, RunResult, Verdict, judge, run_test, outputs_match
    judge/compare.py       outputs_match(expected, actual) -> bool
    judge/guest.py         self-contained: run(code, stdin_text, output_cap) -> dict   (stdlib only)
    judge/runner.py        subprocess per test: run_test(...), judge(...)
    sources/__init__.py
    sources/common.py      SourceRow, normalise(), problem_id()
    sources/deepcoder.py   rows(root) -> Iterator[SourceRow], convert(...)
    build/__init__.py
    build/filters.py       is_after_cutoff, is_multi_answer, HeldOutIndex, load_lcb_statements
    build/dedup.py         deduplicate(rows) -> list[SourceRow]
    build/validate.py      Curated, Rejection, curate(row), split_of(problem_id)
    build/pipeline.py      build(rows, held_out, workers) -> (list[Curated], dict)
    build/io.py            write_dataset(curated, out_dir)
    build/fetch.py         download pinned DeepCoder + LCB
    build/__main__.py      CLI
    corpus.py              Corpus, Problem, DatasetPin, PINNED, pinned_corpus()
    environment.py         CompetitiveCodeEnvironment, render_prompt, INSTRUCTION
    taskset.py             CompetitiveCodeTaskset (Verifiers)
    goldens/reference.jsonl, artifact.json     (Task 10)
  scripts/write_goldens.py, scripts/verify_wheel.py
  examples/prime_rl/rl.toml
  tests/test_extraction.py test_compare.py test_runner.py test_sources.py
        test_filters.py test_dedup.py test_validate.py test_pipeline.py test_environment.py
```

All commands below run from `environments/code/reliquary_competitive_code/` inside the worktree `~/worktrees/env-competitive-code` unless stated.

---

### Task 1: Package scaffold and program extraction

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE` (copy of `../../reasoning/reliquary_science/LICENSE`), `README.md` (one-paragraph stub, finished in Task 9), `reliquary_competitive_code/__init__.py`, `reliquary_competitive_code/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Produces: `extract_program(completion: str) -> str | None`

- [ ] **Step 1: Create the scaffold**

`pyproject.toml`:
```toml
[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[project]
name = "reliquary-competitive-code"
version = "0.1.0a1"
description = "Competitive programming graded on hidden stdin/stdout tests for verifiable RL"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.12,<3.13"
dependencies = ["verifiers==0.3.1", "pyarrow>=17", "huggingface-hub>=0.25"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[dependency-groups]
dev = ["pytest>=8,<9"]
build = ["numpy>=1.26"]

[tool.hatch.build.targets.wheel]
packages = ["reliquary_competitive_code"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-q", "--strict-markers"]

[tool.uv]
required-version = ">=0.11.1"
default-groups = ["dev", "build"]

[tool.uv.sources]
verifiers = { git = "https://github.com/PrimeIntellect-ai/verifiers.git", rev = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf" }
```

`.gitignore`: copy `../../reasoning/reliquary_science/.gitignore` verbatim.

`README.md`:
```markdown
# reliquary-competitive-code

Competitive-programming problems graded by running the policy's Python program
on hidden stdin/stdout tests. Design: `docs/superpowers/specs/2026-10-04-reliquary-competitive-code-env-design.md`.
```

`reliquary_competitive_code/__init__.py`:
```python
"""Competitive programming graded on hidden stdin/stdout tests."""

__all__ = ["CompetitiveCodeTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Replay, grading and the build need none of Verifiers, and a validator that
    only has to reproduce a reward should not have to install it to do so.
    """
    if name == "CompetitiveCodeTaskset":
        from reliquary_competitive_code.taskset import CompetitiveCodeTaskset

        return CompetitiveCodeTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 2: Write the failing test** — `tests/test_extraction.py`:
```python
from reliquary_competitive_code.extraction import extract_program


def test_takes_the_last_python_block() -> None:
    completion = (
        "Idea:\n```python\nprint(1)\n```\nBetter:\n```python\nprint(2)\n```\n"
    )
    assert extract_program(completion) == "print(2)\n"


def test_untagged_and_py_tags_count_as_python() -> None:
    assert extract_program("```\nprint(3)\n```") == "print(3)\n"
    assert extract_program("```py\nprint(4)\n```") == "print(4)\n"
    assert extract_program("```Python3\nprint(5)\n```") == "print(5)\n"


def test_a_trailing_non_python_block_does_not_hide_the_program() -> None:
    completion = "```python\nprint(6)\n```\nSample output:\n```text\n6\n```"
    assert extract_program(completion) == "print(6)\n"


def test_no_program_cases_return_none() -> None:
    assert extract_program("") is None
    assert extract_program("no code at all") is None
    assert extract_program("```cpp\nint main(){}\n```") is None
    assert extract_program("```python\nprint(1)\n") is None  # truncated in the fence
    assert extract_program("```python\n   \n```") is None
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv lock && uv sync && uv run pytest tests/test_extraction.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'reliquary_competitive_code.extraction'`

- [ ] **Step 4: Implement** — `reliquary_competitive_code/extraction.py`:
```python
"""Which part of a completion is the program.

The last fenced block tagged as Python, or untagged, is the submission. A block
in another language after it (sample output, a C++ sketch) does not hide it,
and a block left open by truncation is not a program: it scores as no code
rather than running half a solution.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = frozenset({"", "python", "python3", "py"})


def extract_program(completion: str) -> str | None:
    programs = [
        body
        for tag, body in _FENCE.findall(completion or "")
        if tag.strip().lower() in _PYTHON_TAGS
    ]
    if not programs or not programs[-1].strip():
        return None
    return programs[-1]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): scaffold the package and extract the program

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Output comparator

**Files:**
- Create: `reliquary_competitive_code/judge/__init__.py` (empty for now, filled in Task 3), `reliquary_competitive_code/judge/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces: `outputs_match(expected: str, actual: str) -> bool`, constant `FLOAT_TOLERANCE = 1e-6`

- [ ] **Step 1: Write the failing test** — `tests/test_compare.py`:
```python
from reliquary_competitive_code.judge.compare import outputs_match


def test_whitespace_and_line_endings_do_not_matter() -> None:
    assert outputs_match("1 2\n3\n", "1  2 3")
    assert outputs_match("1\r\n2\r\n", "1\n2\n")
    assert outputs_match("", "   \n")


def test_token_count_and_values_matter() -> None:
    assert not outputs_match("1 2", "1 2 3")
    assert not outputs_match("1 2", "1 3")
    assert not outputs_match("abc", "abd")
    assert not outputs_match("1", "")


def test_yes_no_ignore_case_but_other_words_do_not() -> None:
    assert outputs_match("YES\nNO", "yes\nNo")
    assert not outputs_match("Alice", "alice")
    assert not outputs_match("YES", "YESS")


def test_decimals_have_a_tolerance_and_integers_do_not() -> None:
    assert outputs_match("0.3333333", "0.33333331")
    assert outputs_match("1e9", "1000000000.0000005")
    assert outputs_match("2.5", "2.5000000001")
    assert not outputs_match("2.5", "2.51")
    assert not outputs_match("10", "10.0000001")
    assert not outputs_match("123456789012345678901", "123456789012345678902")


def test_non_finite_tokens_compare_as_text() -> None:
    assert outputs_match("nan", "nan")
    assert not outputs_match("1.0", "nan")
    assert not outputs_match("inf", "1e400")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_compare.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/judge/compare.py`:
```python
"""Whether a program's output is the expected one.

The single source of truth for a verdict: the local runner and the core's
sandboxed grader both call this function on the captured output.

Token by token on whitespace, so line endings and spacing never decide a
verdict. Two relaxations, both what the original judges accept: `yes`/`no` in
any case, and a tolerance on decimals. An expected integer is compared
exactly, so a float that happens to be close to an integer answer is wrong.
"""

from __future__ import annotations

import math

FLOAT_TOLERANCE = 1e-6
_CASELESS = frozenset({"yes", "no"})


def outputs_match(expected: str, actual: str) -> bool:
    want, got = expected.split(), actual.split()
    return len(want) == len(got) and all(map(_tokens_match, want, got))


def _tokens_match(want: str, got: str) -> bool:
    if want == got:
        return True
    if want.lower() in _CASELESS:
        return want.lower() == got.lower()
    if not _is_decimal(want):
        return False
    a, b = _finite(want), _finite(got)
    return (
        a is not None
        and b is not None
        and math.isclose(a, b, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
    )


def _is_decimal(token: str) -> bool:
    return any(c in token for c in ".eE") and _finite(token) is not None


def _finite(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_compare.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): compare outputs token by token

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Guest program and subprocess runner

**Files:**
- Create: `reliquary_competitive_code/judge/guest.py`, `reliquary_competitive_code/judge/runner.py`
- Modify: `reliquary_competitive_code/judge/__init__.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `outputs_match` (Task 2)
- Produces:
  - `guest.run(code: str, stdin_text: str, output_cap: int) -> dict` with keys `status` (`ok|runtime_error|forbidden_import|output_limit`), `stdout` (str, `""` unless ok), `cpu_seconds` (float)
  - `runner.TestCase(stdin: str, stdout: str)` (frozen dataclass)
  - `runner.RunResult(status: str, stdout: str, cpu_seconds: float)`
  - `runner.Verdict(passed: bool, status: str, tests_run: int, max_cpu_seconds: float)`
  - `runner.output_cap(expected: str) -> int`
  - `runner.run_test(code: str, stdin: str, *, time_limit_s: float, output_cap: int) -> RunResult` (statuses add `timeout`)
  - `runner.judge(code: str | None, tests: Sequence[TestCase], *, time_limit_s: float) -> Verdict` (statuses add `wrong_answer`, `no_code`)
  - `judge/__init__.py` re-exports: `TestCase, RunResult, Verdict, judge, run_test, output_cap, outputs_match`

- [ ] **Step 1: Write the failing test** — `tests/test_runner.py`:
```python
import time

import pytest

from reliquary_competitive_code.judge import TestCase, judge, output_cap, run_test

ADD = TestCase("2 3\n", "5\n")


def _status(code: str, tests=(ADD,), limit: float = 2.0) -> str:
    return judge(code, list(tests), time_limit_s=limit).status


def test_a_correct_program_passes() -> None:
    verdict = judge("a, b = map(int, input().split())\nprint(a + b)\n", [ADD], time_limit_s=2.0)
    assert verdict.passed and verdict.status == "ok" and verdict.tests_run == 1


def test_fast_input_idioms_pass() -> None:
    assert _status("import sys\ninput = sys.stdin.buffer.readline\na, b = map(int, input().split())\nprint(a + b)\n") == "ok"
    assert _status("import sys\ndata = sys.stdin.read().split()\nprint(int(data[0]) + int(data[1]))\n") == "ok"
    assert _status("from sys import stdin\na, b = map(int, stdin.readline().split())\nprint(a + b)\n") == "ok"
    assert _status("import sys\na, b = map(int, sys.stdin.readline().split())\nsys.stdout.write(str(a + b) + '\\n')\n") == "ok"


def test_thread_based_deep_recursion_passes() -> None:
    code = (
        "import sys, threading\n"
        "sys.setrecursionlimit(1 << 20)\n"
        "threading.stack_size(1 << 26)\n"
        "def depth(n):\n"
        "    return 0 if n == 0 else 1 + depth(n - 1)\n"
        "def main():\n"
        "    a, b = map(int, input().split())\n"
        "    print(depth(100000) - 100000 + a + b)\n"
        "threading.Thread(target=main).start()\n"
    )
    assert _status(code) == "ok"


def test_exit_codes() -> None:
    assert _status("print(5)\nexit()\n") == "ok"
    assert _status("import sys\nprint(5)\nsys.exit(0)\n") == "ok"
    assert _status("import sys\nprint(5)\nsys.exit(1)\n") == "runtime_error"


def test_failures_have_their_own_status() -> None:
    assert _status("print(6)\n") == "wrong_answer"
    assert _status("raise ValueError\n") == "runtime_error"
    assert _status("this is not python\n") == "runtime_error"
    assert _status("import os\nprint(5)\n") == "forbidden_import"
    assert _status("__import__('subprocess')\nprint(5)\n") == "forbidden_import"
    assert _status("open('/etc/passwd')\nprint(5)\n") == "runtime_error"
    assert judge(None, [ADD], time_limit_s=2.0).status == "no_code"


def test_an_infinite_loop_times_out_quickly() -> None:
    start = time.monotonic()
    assert _status("while True:\n    pass\n", limit=1.0) == "timeout"
    assert time.monotonic() - start < 6.0


def test_printing_forever_hits_the_output_limit() -> None:
    assert _status("while True:\n    print('x' * 1000)\n") == "output_limit"


def test_judge_stops_at_the_first_failing_test() -> None:
    tests = [ADD, TestCase("1 1\n", "3\n"), TestCase("4 4\n", "8\n")]
    verdict = judge("a, b = map(int, input().split())\nprint(a + b)\n", tests, time_limit_s=2.0)
    assert not verdict.passed and verdict.status == "wrong_answer" and verdict.tests_run == 2


def test_run_test_reports_cpu_time() -> None:
    result = run_test("s = 0\nfor i in range(3_000_000):\n    s += i\nprint(s)\n", "", time_limit_s=4.0, output_cap=output_cap("1"))
    assert result.status == "ok"
    assert 0.05 < result.cpu_seconds < 4.0


def test_output_cap_scales_with_the_expected_output() -> None:
    assert output_cap("") == 64 * 1024
    assert output_cap("x" * 1000) == 4000 + 64 * 1024
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL, `ImportError: cannot import name 'TestCase'`

- [ ] **Step 3: Implement the guest** — `reliquary_competitive_code/judge/guest.py`:
```python
"""The program's side of the sandbox: one submission, one stdin, one stdout.

Standard library only, and no import from this package: the local runner ships
this file's text into a fresh interpreter, and the core's gVisor worker is
meant to call the same `run` in its warm process, so the two cannot drift.

The import gate and the reduced builtins are rules of the task, not a security
boundary. Containment is the sandbox's job (gVisor in production, a limited
subprocess locally). What this file guarantees is that the rules are the same
everywhere: the same modules, the same `sys`, the same verdict for an exit.
"""

from __future__ import annotations

import builtins
import io
import sys
import threading
import time
import types

ALLOWED_IMPORT_ROOTS = frozenset({
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "fractions", "functools", "heapq", "itertools", "math", "operator",
    "random", "re", "statistics", "string", "sys", "threading", "typing",
})
DENIED_BUILTINS = frozenset({
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help",
    "locals", "open", "vars",
})
GATE_MESSAGE = "is not available in the grader sandbox"


class OutputLimitExceeded(BaseException):
    """Raised from a write past the cap. A BaseException, so that a
    submission's `except Exception` cannot swallow it and keep printing."""


class _CappedBytes(io.BytesIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap

    def write(self, data) -> int:
        if self.tell() + len(data) > self._cap:
            raise OutputLimitExceeded()
        return super().write(data)


def _sys_shim(stdin, stdout) -> types.ModuleType:
    shim = types.ModuleType("sys")
    shim.stdin = stdin
    shim.stdout = stdout
    shim.stderr = io.StringIO()
    shim.argv = ["main.py"]
    shim.maxsize = sys.maxsize
    shim.version_info = sys.version_info
    shim.exit = sys.exit
    shim.setrecursionlimit = sys.setrecursionlimit
    shim.getrecursionlimit = sys.getrecursionlimit
    shim.set_int_max_str_digits = sys.set_int_max_str_digits
    shim.get_int_max_str_digits = sys.get_int_max_str_digits
    return shim


def _safe_builtins(shim: types.ModuleType) -> dict:
    real_import = builtins.__import__

    def gated_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if level != 0 or root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"module {name!r} {GATE_MESSAGE}")
        if root == "sys":
            return shim
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in builtins.__dict__.items() if k not in DENIED_BUILTINS}
    safe["__import__"] = gated_import
    # `exit` and `quit` come from `site`, which the local child runs without
    # (-S). Defined here so a script ending in `exit()` behaves the same in
    # every host.
    safe["exit"] = safe["quit"] = sys.exit
    return safe


def _join_threads() -> None:
    for thread in threading.enumerate():
        if thread is not threading.main_thread() and not thread.daemon:
            thread.join()


def run(code: str, stdin_text: str, output_cap: int) -> dict:
    """Run `code` as a script on `stdin_text`; return status, stdout, CPU time."""
    raw_out = _CappedBytes(output_cap)
    stdout = io.TextIOWrapper(raw_out, encoding="utf-8", write_through=True)
    stdin = io.TextIOWrapper(io.BytesIO(stdin_text.encode("utf-8")), encoding="utf-8")
    shim = _sys_shim(stdin, stdout)
    namespace = {"__name__": "__main__", "__builtins__": _safe_builtins(shim)}
    saved_stdin, saved_stdout = sys.stdin, sys.stdout
    saved_limit = sys.getrecursionlimit()
    sys.stdin, sys.stdout = stdin, stdout
    status = "ok"
    start = time.process_time()
    try:
        exec(compile(code, "<submission>", "exec"), namespace)
        _join_threads()
    except OutputLimitExceeded:
        status = "output_limit"
    except SystemExit as stop:
        if stop.code not in (None, 0):
            status = "runtime_error"
        else:
            _join_threads()
    except ImportError as error:
        status = "forbidden_import" if GATE_MESSAGE in str(error) else "runtime_error"
    except BaseException:
        status = "runtime_error"
    finally:
        cpu_seconds = time.process_time() - start
        try:
            stdout.flush()
        except OutputLimitExceeded:
            status = "output_limit"
        sys.stdin, sys.stdout = saved_stdin, saved_stdout
        sys.setrecursionlimit(saved_limit)
    text = raw_out.getvalue().decode("utf-8", errors="replace")
    return {
        "status": status,
        "stdout": text if status == "ok" else "",
        "cpu_seconds": cpu_seconds,
    }
```

- [ ] **Step 4: Implement the runner** — `reliquary_competitive_code/judge/runner.py`:
```python
"""Run a submission on tests, one fresh subprocess per test.

The child receives `guest.py`'s text plus a small driver, sets its own CPU,
address-space and file-size limits as its first statements (never a
`preexec_fn`: forking from a threaded caller deadlocks, see reliquary-code),
runs the submission through `guest.run`, and prints one JSON line. A wall
clock above the CPU limit and a process-group kill cover what sleeps or forks.

Not a sandbox: no network isolation, no filesystem isolation. Production runs
the same `guest.run` inside gVisor; this runner serves development, the build
and qualification.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reliquary_competitive_code.judge.compare import outputs_match

MEMORY_BYTES = 2 << 30
OUTPUT_CAP_SLACK = 64 * 1024
WALL_FACTOR = 2.0
WALL_SLACK_S = 1.0

_GUEST_SOURCE = Path(__file__).with_name("guest.py").read_text(encoding="utf-8")
_DRIVER = """

if __name__ == "__main__":
    import json as _json
    _request = _json.loads(sys.stdin.buffer.read().decode("utf-8"))
    _result = run(_request["code"], _request["stdin"], _request["output_cap"])
    sys.__stdout__.write("\\n" + _json.dumps(_result) + "\\n")
    sys.__stdout__.flush()
"""


@dataclass(frozen=True, slots=True)
class TestCase:
    stdin: str
    stdout: str


@dataclass(frozen=True, slots=True)
class RunResult:
    status: str
    stdout: str
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    status: str
    tests_run: int
    max_cpu_seconds: float


def output_cap(expected: str) -> int:
    return 4 * len(expected.encode("utf-8")) + OUTPUT_CAP_SLACK


def _program(cpu_seconds: int) -> str:
    return (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_BYTES}, {MEMORY_BYTES}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))\n"
        f"exec(compile({_GUEST_SOURCE + _DRIVER!r}, '<guest>', 'exec'), "
        "{'__name__': '__main__'})\n"
    )


def _kill_group(proc: subprocess.Popen) -> tuple[bytes, bytes]:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate()


def run_test(code: str, stdin: str, *, time_limit_s: float, output_cap: int) -> RunResult:
    request = json.dumps({"code": code, "stdin": stdin, "output_cap": output_cap})
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _program(math.ceil(time_limit_s) + 1)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            cwd="/",
            start_new_session=True,
        )
    except OSError:
        return RunResult("runtime_error", "", 0.0)
    timed_out = False
    try:
        out, _ = proc.communicate(
            input=request.encode("utf-8"),
            timeout=WALL_FACTOR * time_limit_s + WALL_SLACK_S,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        out, _ = _kill_group(proc)
    if timed_out or proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        return RunResult("timeout", "", time_limit_s)
    try:
        result = json.loads(out.decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)[-1])
        status, text, cpu = str(result["status"]), str(result["stdout"]), float(result["cpu_seconds"])
    except (ValueError, KeyError, TypeError, IndexError):
        return RunResult("runtime_error", "", 0.0)
    if cpu > time_limit_s:
        return RunResult("timeout", "", cpu)
    return RunResult(status, text, cpu)


def judge(code: str | None, tests: Sequence[TestCase], *, time_limit_s: float) -> Verdict:
    """Binary verdict over `tests`, stopping at the first failure."""
    if code is None:
        return Verdict(False, "no_code", 0, 0.0)
    slowest = 0.0
    for count, test in enumerate(tests, start=1):
        result = run_test(code, test.stdin, time_limit_s=time_limit_s, output_cap=output_cap(test.stdout))
        slowest = max(slowest, result.cpu_seconds)
        if result.status != "ok":
            return Verdict(False, result.status, count, slowest)
        if not outputs_match(test.stdout, result.stdout):
            return Verdict(False, "wrong_answer", count, slowest)
    return Verdict(True, "ok", len(tests), slowest)
```

`reliquary_competitive_code/judge/__init__.py`:
```python
"""Run a program on stdin/stdout tests and decide whether it passes."""

from reliquary_competitive_code.judge.compare import outputs_match
from reliquary_competitive_code.judge.runner import (
    RunResult,
    TestCase,
    Verdict,
    judge,
    output_cap,
    run_test,
)

__all__ = ["RunResult", "TestCase", "Verdict", "judge", "output_cap", "outputs_match", "run_test"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: 10 passed. If `test_thread_based_deep_recursion_passes` fails with `runtime_error`, check that `threading.stack_size(1 << 26)` fits under `RLIMIT_AS` (2 GiB) before changing anything else.

- [ ] **Step 6: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): run a program on stdin tests in a limited subprocess

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Source rows and the DeepCoder adapter

**Files:**
- Create: `reliquary_competitive_code/sources/__init__.py` (empty), `reliquary_competitive_code/sources/common.py`, `reliquary_competitive_code/sources/deepcoder.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `TestCase` (Task 3), `extract_program` (Task 1)
- Produces:
  - `SourceRow(source: str, upstream_id: str, statement: str, tests: tuple[TestCase, ...], references: tuple[str, ...], contest_date: str | None = None)` frozen dataclass
  - `normalise(statement: str) -> str`, `problem_id(statement: str) -> str` (16 hex)
  - `deepcoder.REPOSITORY`, `deepcoder.REVISION`, `deepcoder.CONFIGS = ("taco", "primeintellect")`
  - `deepcoder.convert(config: str, upstream_id: str, record: dict) -> SourceRow | None`
  - `deepcoder.rows(root: Path) -> Iterator[SourceRow]` reading `root/<config>/train-*.parquet`

- [ ] **Step 1: Write the failing test** — `tests/test_sources.py`:
```python
import json

import pyarrow as pa
import pyarrow.parquet as pq

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources import deepcoder
from reliquary_competitive_code.sources.common import normalise, problem_id

PI_STATEMENT = (
    "Solve the following coding problem using the programming language python:\n\n"
    "Add two numbers $a$ and $b$.\n\n"
    "The input will be stdin and you should print your solution to stdout\n\n\n"
    "Now solve the problem and return the code."
)


def test_normalise_ignores_formatting_and_latex() -> None:
    assert normalise("Add  $a$ and \\texttt{b}!") == normalise("add a and b")
    assert problem_id("Add a and b") == problem_id("ADD A, AND B.")
    assert len(problem_id("x")) == 16


def test_primeintellect_rows_lose_their_wrapper_and_fence() -> None:
    record = {
        "problem": PI_STATEMENT,
        "tests": json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}]),
        "solutions": ["```python\na, b = map(int, input().split())\nprint(a + b)\n```"],
    }
    row = deepcoder.convert("primeintellect", "pi#0", record)
    assert row.statement == "Add two numbers $a$ and $b$."
    assert row.tests == (TestCase("1 2\n", "3\n"),)
    assert row.references == ("a, b = map(int, input().split())\nprint(a + b)\n",)
    assert row.source == "deepcoder/primeintellect" and row.contest_date is None


def test_taco_rows_keep_their_code_and_skip_function_tests() -> None:
    stdin_record = {
        "problem": "Print the sum.",
        "tests": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}),
        "solutions": ["print(sum(map(int, input().split())))"],
    }
    row = deepcoder.convert("taco", "taco#0", stdin_record)
    assert row.tests == (TestCase("1 2\n", "3\n"),)
    assert row.references == ("print(sum(map(int, input().split())))",)

    function_record = dict(stdin_record, tests=json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[1]]}))
    assert deepcoder.convert("taco", "taco#1", function_record) is None


def test_malformed_tests_drop_the_row() -> None:
    base = {"problem": "p", "solutions": []}
    for tests in (
        "not json",
        json.dumps({"inputs": ["1"], "outputs": []}),
        json.dumps({"inputs": [["1"]], "outputs": [["1"]]}),
        json.dumps([{"type": "functional", "input": "1", "output": "1"}]),
        json.dumps([]),
    ):
        assert deepcoder.convert("taco", "x", dict(base, tests=tests)) is None


def test_rows_reads_both_configs(tmp_path) -> None:
    for config, record in (
        ("taco", {"problem": "Print the sum.", "tests": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}), "solutions": ["print(3)"]}),
        ("primeintellect", {"problem": PI_STATEMENT, "tests": json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}]), "solutions": ["```python\nprint(3)\n```"]}),
    ):
        (tmp_path / config).mkdir()
        pq.write_table(pa.Table.from_pylist([record]), tmp_path / config / "train-00000-of-00001.parquet")
    rows = list(deepcoder.rows(tmp_path))
    assert [row.source for row in rows] == ["deepcoder/taco", "deepcoder/primeintellect"]
    assert rows[0].upstream_id == "taco/train-00000-of-00001.parquet#0"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_sources.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/sources/common.py`:
```python
"""The row every source adapter emits, and the identity of a problem."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from reliquary_competitive_code.judge import TestCase

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class SourceRow:
    source: str
    upstream_id: str
    statement: str
    tests: tuple[TestCase, ...]
    references: tuple[str, ...]
    contest_date: str | None = None


def normalise(statement: str) -> str:
    """Lower-case words only: the same problem formatted twice compares equal."""
    text = _LATEX_COMMAND.sub(" ", statement.lower())
    return " ".join(_NON_WORD.sub(" ", text).split())


def problem_id(statement: str) -> str:
    return hashlib.sha256(normalise(statement).encode("utf-8")).hexdigest()[:16]
```

`reliquary_competitive_code/sources/deepcoder.py`:
```python
"""DeepCoder-Preview-Dataset: taco and primeintellect train configs.

Kept: problems whose every test is a stdin/stdout pair of strings. Dropped:
function-call tests (reliquary-code covers that contract), malformed tests.
`lcbv5/train` is not read in v1 because it carries no reference solution, and
`lcbv5/test` and `codeforces/test` are never read: their dates fall inside the
held-out LiveCodeBench window.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.extraction import extract_program
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

REPOSITORY = "agentica-org/DeepCoder-Preview-Dataset"
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
CONFIGS = ("taco", "primeintellect")

_PI_PREFIX = re.compile(r"^\s*Solve the following coding problem using the programming language python:\s*")
_PI_SUFFIX = re.compile(
    r"\s*The input will be stdin and you should print your solution to stdout\s*"
    r"Now solve the problem and return the code\.\s*$"
)


def rows(root: Path) -> Iterator[SourceRow]:
    for config in CONFIGS:
        for path in sorted((Path(root) / config).glob("train-*.parquet")):
            table = pq.read_table(path, columns=["problem", "tests", "solutions"])
            for offset, record in enumerate(table.to_pylist()):
                row = convert(config, f"{config}/{path.name}#{offset}", record)
                if row is not None:
                    yield row


def convert(config: str, upstream_id: str, record: dict) -> SourceRow | None:
    tests = _tests(record.get("tests"))
    if not tests:
        return None
    statement = record.get("problem") or ""
    if config == "primeintellect":
        statement = _PI_SUFFIX.sub("", _PI_PREFIX.sub("", statement))
    references = tuple(
        code
        for code in (_reference(config, solution) for solution in record.get("solutions") or ())
        if code
    )
    return SourceRow(
        source=f"deepcoder/{config}",
        upstream_id=upstream_id,
        statement=statement.strip(),
        tests=tests,
        references=references,
    )


def _tests(raw: object) -> tuple[TestCase, ...]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return ()
    if isinstance(data, dict):
        if data.get("fn_name"):
            return ()
        inputs, outputs = data.get("inputs") or [], data.get("outputs") or []
        if len(inputs) != len(outputs):
            return ()
        pairs = list(zip(inputs, outputs))
    elif isinstance(data, list):
        if any(not isinstance(item, dict) or item.get("type") != "stdin_stdout" for item in data):
            return ()
        pairs = [(item.get("input"), item.get("output")) for item in data]
    else:
        return ()
    if not pairs or any(not isinstance(a, str) or not isinstance(b, str) for a, b in pairs):
        return ()
    return tuple(TestCase(a, b) for a, b in pairs)


def _reference(config: str, solution: object) -> str | None:
    if not isinstance(solution, str) or not solution.strip():
        return None
    if config == "primeintellect":
        return extract_program(solution)
    return solution
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sources.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke-check on the real data** (read-only, light)

Run: `uv run python -c "from itertools import islice; from reliquary_competitive_code.sources import deepcoder; rs=list(deepcoder.rows('/home/ubuntu/cc-data/deepcoder')); print(len(rs), sum(bool(r.references) for r in rs))"`
Expected: about 21,400 rows (6,387 taco + 14,998 primeintellect stdin rows, minus malformed), most with references. Record the two numbers in the commit message.

- [ ] **Step 6: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): adapt DeepCoder taco and primeintellect to source rows

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Held-out filters (date, LiveCodeBench overlap, multiple answers)

**Files:**
- Create: `reliquary_competitive_code/build/__init__.py` (empty), `reliquary_competitive_code/build/filters.py`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `SourceRow`, `normalise` (Task 4)
- Produces:
  - `HELD_OUT_FROM = "2024-08-01"`, `CONTAMINATION_THRESHOLD = 0.10`, `NGRAM = 13`
  - `is_after_cutoff(row: SourceRow) -> bool`
  - `is_multi_answer(row: SourceRow) -> bool`
  - `class HeldOutIndex(statements: Iterable[str])` with `.overlap(statement: str) -> float`
  - `load_lcb_statements(root: Path) -> list[str]` (reads `root/test*.jsonl`, keeps `contest_date >= HELD_OUT_FROM`)
  - `LCB_REPOSITORY = "livecodebench/code_generation_lite"`, `LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"`

- [ ] **Step 1: Write the failing test** — `tests/test_filters.py`:
```python
import json

from reliquary_competitive_code.build.filters import (
    HeldOutIndex,
    is_after_cutoff,
    is_multi_answer,
    load_lcb_statements,
)
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

HELD = (
    "Takahashi has N cards numbered 1 to N arranged in a row and wants to "
    "remove exactly K of them so that the remaining sum is maximal modulo M"
)


def _row(statement: str, date: str | None = None) -> SourceRow:
    return SourceRow("s", "u", statement, (TestCase("", ""),), ("print()",), date)


def test_dates_on_or_after_the_cutoff_are_excluded() -> None:
    assert is_after_cutoff(_row("x", "2024-08-01"))
    assert is_after_cutoff(_row("x", "2025-01-04"))
    assert not is_after_cutoff(_row("x", "2024-07-31"))
    assert not is_after_cutoff(_row("x", None))


def test_multiple_answer_statements_are_detected() -> None:
    assert is_multi_answer(_row("If there are several answers, print any of them."))
    assert is_multi_answer(_row("Output any valid permutation."))
    assert not is_multi_answer(_row("Print the number of ways modulo 10^9+7."))


def test_overlap_with_held_out_statements() -> None:
    index = HeldOutIndex([HELD])
    assert index.overlap("Story. " + HELD + " Constraints follow.") > 0.5
    assert index.overlap("Print the sum of two integers a and b given on one line of input") == 0.0
    assert index.overlap("too short") == 0.0


def test_lcb_loader_keeps_only_held_out_dates(tmp_path) -> None:
    lines = [
        {"question_content": "old problem", "contest_date": "2024-07-31T00:00:00"},
        {"question_content": "new problem", "contest_date": "2024-08-01T00:00:00"},
    ]
    (tmp_path / "test5.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert load_lcb_statements(tmp_path) == ["new problem"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_filters.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/build/filters.py`:
```python
"""What must never be trained on, and what this version cannot grade.

Held out: LiveCodeBench problems dated from 2024-08-01, the window the held-out
benchmark measures. A source with dates is cut on its date; taco and
primeintellect carry none, so every statement is also compared, by 13-gram
overlap, with every held-out LiveCodeBench statement.

Not gradable in v1: problems that accept several outputs. The comparator has
one expected output per test; such a problem would mark valid answers wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from reliquary_competitive_code.sources.common import SourceRow, normalise

HELD_OUT_FROM = "2024-08-01"
CONTAMINATION_THRESHOLD = 0.10
NGRAM = 13
LCB_REPOSITORY = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

_MULTI_ANSWER = re.compile(
    r"(print|output)\s+any\b"
    r"|any\s+(of\s+them|valid|correct|one\s+of\s+them)"
    r"|if\s+there\s+are\s+(several|multiple|many)"
    r"|(multiple|several)\s+(possible|correct|valid)\s+(answers|solutions)",
    re.IGNORECASE,
)


def is_after_cutoff(row: SourceRow) -> bool:
    return row.contest_date is not None and row.contest_date[:10] >= HELD_OUT_FROM


def is_multi_answer(row: SourceRow) -> bool:
    return _MULTI_ANSWER.search(row.statement) is not None


def _ngrams(statement: str) -> set[tuple[str, ...]]:
    words = normalise(statement).split()
    return {tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


class HeldOutIndex:
    def __init__(self, statements: Iterable[str]) -> None:
        self._grams: set[tuple[str, ...]] = set()
        for statement in statements:
            self._grams |= _ngrams(statement)

    def overlap(self, statement: str) -> float:
        grams = _ngrams(statement)
        if not grams:
            return 0.0
        return len(grams & self._grams) / len(grams)


def load_lcb_statements(root: Path) -> list[str]:
    statements = []
    for path in sorted(Path(root).glob("test*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record.get("contest_date", ""))[:10] >= HELD_OUT_FROM:
                    statements.append(record["question_content"])
    return statements
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_filters.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): filter held-out dates, LiveCodeBench overlap and multi-answer problems

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Deduplication across sources

**Files:**
- Create: `reliquary_competitive_code/build/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `SourceRow`, `normalise`, `problem_id` (Task 4)
- Produces: `deduplicate(rows: Sequence[SourceRow]) -> list[SourceRow]`, constants `SHINGLE = 5`, `PERMUTATIONS = 64`, `BANDS = 16`, `THRESHOLD = 0.8`. A merged row keeps the statement, tests, source and upstream id of the member with the most tests, the union of all members' references (first-seen order, duplicates removed) and the earliest non-null contest date.

- [ ] **Step 1: Write the failing test** — `tests/test_dedup.py`:
```python
from reliquary_competitive_code.build.dedup import deduplicate
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

LONG = (
    "Vasya has an array of n integers and wants to split it into the minimum "
    "number of contiguous segments such that every segment has a sum that does "
    "not exceed s and every segment contains at most k elements print the answer"
)


def _row(statement, tests, refs, source="a", date=None) -> SourceRow:
    return SourceRow(source, f"{source}#{len(tests)}", statement, tuple(TestCase(str(i), str(i)) for i in range(tests)), refs, date)


def test_exact_duplicates_merge_and_keep_the_most_tests() -> None:
    merged = deduplicate([
        _row(LONG, 3, ("r1",), "taco"),
        _row(LONG.upper() + "!", 10, ("r2", "r1"), "primeintellect", "2021-01-01"),
    ])
    assert len(merged) == 1
    assert merged[0].source == "primeintellect" and len(merged[0].tests) == 10
    assert merged[0].references == ("r1", "r2")
    assert merged[0].contest_date == "2021-01-01"


def test_near_duplicates_merge() -> None:
    near = LONG.replace("print the answer", "output the answer")
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(near, 4, ("b",))])) == 1


def test_different_problems_stay_apart() -> None:
    other = "Petya walks along a grid of h rows and w columns collecting coins on every cell he visits and must reach the bottom right corner print the maximum number of coins"
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(other, 3, ("b",))])) == 2


def test_output_order_is_deterministic() -> None:
    rows = [_row(LONG, 3, ("a",)), _row("Print one integer the sum of a and b for every test case given in the input lines", 2, ("b",))]
    assert deduplicate(rows) == deduplicate(list(reversed(rows)))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/build/dedup.py`:
```python
"""One row per problem, across sources.

taco, primeintellect and CodeContests+ are largely the same Codeforces problems
formatted differently (measured: 21,707 DeepCoder stdin rows are 11,287
distinct statements). Exact duplicates share a normalised hash; near
duplicates share most 5-word shingles and are found by MinHash with LSH
banding, then confirmed at an estimated Jaccard of 0.8.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from reliquary_competitive_code.sources.common import SourceRow, normalise, problem_id

SHINGLE = 5
PERMUTATIONS = 64
BANDS = 16
THRESHOLD = 0.8
_PRIME = (1 << 31) - 1
_RNG = np.random.default_rng(20261004)
_A = _RNG.integers(1, _PRIME, PERMUTATIONS, dtype=np.uint64)
_B = _RNG.integers(0, _PRIME, PERMUTATIONS, dtype=np.uint64)


def _signature(statement: str) -> np.ndarray:
    words = normalise(statement).split()
    shingles = {" ".join(words[i : i + SHINGLE]) for i in range(max(1, len(words) - SHINGLE + 1))}
    hashes = np.array(
        [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") for s in shingles],
        dtype=np.uint64,
    )
    return ((_A[:, None] * hashes[None, :] + _B[:, None]) % _PRIME).min(axis=1)


def _merge(members: list[SourceRow]) -> SourceRow:
    lead = max(members, key=lambda row: (len(row.tests), row.source, row.upstream_id))
    references = tuple(dict.fromkeys(ref for row in members for ref in row.references))
    dates = sorted(row.contest_date for row in members if row.contest_date)
    return SourceRow(lead.source, lead.upstream_id, lead.statement, lead.tests, references, dates[0] if dates else None)


def deduplicate(rows: Sequence[SourceRow]) -> list[SourceRow]:
    exact: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        exact[problem_id(row.statement)].append(row)
    keys = sorted(exact)
    signatures = [_signature(exact[key][0].statement) for key in keys]

    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    width = PERMUTATIONS // BANDS
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, signature in enumerate(signatures):
            buckets[signature[band * width : (band + 1) * width].tobytes()].append(i)
        for members in buckets.values():
            for j in members[1:]:
                if float(np.mean(signatures[members[0]] == signatures[j])) >= THRESHOLD:
                    parent[find(j)] = find(members[0])

    clusters: dict[int, list[SourceRow]] = defaultdict(list)
    for i, key in enumerate(keys):
        clusters[find(i)].extend(exact[key])
    merged = [_merge(members) for members in clusters.values()]
    return sorted(merged, key=lambda row: problem_id(row.statement))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: 4 passed

- [ ] **Step 5: Measure on the real data** (light, about a minute)

Run: `uv run python -c "from reliquary_competitive_code.sources import deepcoder; from reliquary_competitive_code.build.dedup import deduplicate; rs=list(deepcoder.rows('/home/ubuntu/cc-data/deepcoder')); print(len(rs), len(deduplicate(rs)))"`
Expected: second number at or below 11,287 (the exact-hash count measured on the first 120 words). Put both numbers in the commit message.

- [ ] **Step 6: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): deduplicate problems across sources with MinHash

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Reference validation, time calibration, test capping, splits

**Files:**
- Create: `reliquary_competitive_code/build/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `run_test`, `output_cap`, `outputs_match`, `TestCase` (Task 3); `SourceRow`, `problem_id` (Task 4)
- Produces:
  - Constants: `VALIDATION_TIME_LIMIT_S = 10.0`, `TIME_FACTOR = 4.0`, `MIN_TIME_LIMIT_S = 1.0`, `MAX_TIME_LIMIT_S = 4.0`, `MAX_REFERENCE_TEST_S = 2.0`, `REFERENCE_CPU_BUDGET_S = 8.0`, `TEST_BYTES_BUDGET = 1 << 20`, `MIN_TESTS = 5`, `MAX_REFERENCES_TRIED = 3`, `SPLITS = ("train", "eval", "qualification")`
  - `Curated(problem_id: str, split: str, statement: str, tests: tuple[TestCase, ...], time_limit_s: float, reference: str, origin: dict)` frozen dataclass
  - `Rejection(problem_id: str, reason: str)` with reasons `no_passing_reference`, `references_disagree`, `reference_too_slow`, `too_few_tests`
  - `split_of(problem_id: str) -> str`
  - `time_limit(slowest_cpu_s: float) -> float`
  - `select_tests(tests: Sequence[TestCase], timings: Sequence[float]) -> tuple[int, ...]`
  - `curate(row: SourceRow) -> Curated | Rejection`

- [ ] **Step 1: Write the failing test** — `tests/test_validate.py`:
```python
from reliquary_competitive_code.build.validate import (
    MIN_TESTS,
    Curated,
    Rejection,
    curate,
    select_tests,
    split_of,
    time_limit,
)
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow, problem_id

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = tuple(TestCase(f"{i} {i}\n", f"{2 * i}\n") for i in range(6))


def _row(refs, tests=TESTS) -> SourceRow:
    return SourceRow("deepcoder/taco", "taco#0", "Print a plus b.", tests, tuple(refs))


def test_a_passing_reference_curates_the_problem() -> None:
    curated = curate(_row([SUM]))
    assert isinstance(curated, Curated)
    assert curated.problem_id == problem_id("Print a plus b.")
    assert curated.reference == SUM and curated.tests == TESTS
    assert curated.time_limit_s == 1.0
    assert curated.origin == {"source": "deepcoder/taco", "upstream_id": "taco#0"}
    assert curated.split == split_of(curated.problem_id)


def test_the_first_passing_reference_is_kept_after_a_crashing_one() -> None:
    curated = curate(_row(["raise SystemExit(3)", SUM]))
    assert isinstance(curated, Curated) and curated.reference == SUM


def test_no_passing_reference_rejects() -> None:
    assert curate(_row(["raise ValueError"])) == Rejection(problem_id("Print a plus b."), "no_passing_reference")
    assert curate(_row(["print(0)", "raise ValueError"])).reason == "no_passing_reference"


def test_a_wrong_answer_beside_a_passing_reference_rejects() -> None:
    assert curate(_row([SUM, "print(0)"])).reason == "references_disagree"


def test_too_few_tests_rejects() -> None:
    assert curate(_row([SUM], TESTS[: MIN_TESTS - 1])).reason == "too_few_tests"


def test_slow_references_reject() -> None:
    slow = "s = 0\nfor i in range(60_000_000):\n    s += i\n" + SUM
    assert curate(_row([slow])).reason == "reference_too_slow"


def test_time_limit_is_four_times_the_reference_within_bounds() -> None:
    assert time_limit(0.01) == 1.0
    assert time_limit(0.5) == 2.0
    assert time_limit(1.5) == 4.0


def test_select_tests_respects_the_cpu_budget_and_keeps_extremes() -> None:
    tests = [TestCase("x" * n, "1") for n in (5, 1, 9, 3)]
    assert select_tests(tests, [1.0, 1.0, 1.0, 1.0]) == (0, 1, 2, 3)
    kept = select_tests(tests, [3.0, 3.0, 3.0, 3.0])
    assert kept == (1, 2)  # smallest and largest input first, budget 8 s


def test_splits_are_stable_and_roughly_95_25_25() -> None:
    splits = [split_of(f"{i:016x}") for i in range(4000)]
    assert splits == [split_of(f"{i:016x}") for i in range(4000)]
    assert 0.93 < splits.count("train") / 4000 < 0.97
    assert splits.count("eval") > 50 and splits.count("qualification") > 50
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/build/validate.py`:
```python
"""Keep a problem only if a known solution passes every test we keep.

A test the reference fails is wrong or impossible under our limits; the
problem is dropped rather than the test, because a problem kept with fewer
tests is silently weaker. Two references that disagree (one passes, another
gives a different answer) usually mean a problem with several valid outputs,
which v1 cannot grade, so that drops it too.

The reference's own CPU time then sets the limit (x4, between 1 and 4 s) and
the test budget (8 s of reference CPU, 1 MiB of test data). Only CPU time is
measured, so the limit does not depend on how loaded the build box is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from reliquary_competitive_code.judge import TestCase, output_cap, outputs_match, run_test
from reliquary_competitive_code.sources.common import SourceRow, problem_id

VALIDATION_TIME_LIMIT_S = 10.0
TIME_FACTOR = 4.0
MIN_TIME_LIMIT_S = 1.0
MAX_TIME_LIMIT_S = 4.0
MAX_REFERENCE_TEST_S = 2.0
REFERENCE_CPU_BUDGET_S = 8.0
TEST_BYTES_BUDGET = 1 << 20
MIN_TESTS = 5
MAX_REFERENCES_TRIED = 3
SPLITS = ("train", "eval", "qualification")
_SPLIT_SALT = "reliquary_competitive_code_v1"


@dataclass(frozen=True, slots=True)
class Curated:
    problem_id: str
    split: str
    statement: str
    tests: tuple[TestCase, ...]
    time_limit_s: float
    reference: str
    origin: dict = field(hash=False, compare=True)


@dataclass(frozen=True, slots=True)
class Rejection:
    problem_id: str
    reason: str


def split_of(pid: str) -> str:
    bucket = int(hashlib.sha256(f"{_SPLIT_SALT}:{pid}".encode()).hexdigest(), 16) % 1000
    if bucket < 950:
        return "train"
    return "eval" if bucket < 975 else "qualification"


def time_limit(slowest_cpu_s: float) -> float:
    return min(MAX_TIME_LIMIT_S, max(MIN_TIME_LIMIT_S, round(TIME_FACTOR * slowest_cpu_s, 2)))


def select_tests(tests: Sequence[TestCase], timings: Sequence[float]) -> tuple[int, ...]:
    by_size = sorted(range(len(tests)), key=lambda i: (len(tests[i].stdin), i))
    rest = sorted(
        by_size[1:-1],
        key=lambda i: hashlib.sha256(f"{i}:{tests[i].stdin}".encode()).hexdigest(),
    )
    keep, cpu, size = [], 0.0, 0
    for i in dict.fromkeys([by_size[0], by_size[-1], *rest]):
        cost = len(tests[i].stdin.encode()) + len(tests[i].stdout.encode())
        if cpu + timings[i] > REFERENCE_CPU_BUDGET_S or size + cost > TEST_BYTES_BUDGET:
            continue
        keep.append(i)
        cpu += timings[i]
        size += cost
    return tuple(sorted(keep))


def _timings(code: str, tests: Sequence[TestCase]) -> list[float] | str:
    timings = []
    for test in tests:
        result = run_test(
            code, test.stdin,
            time_limit_s=VALIDATION_TIME_LIMIT_S, output_cap=output_cap(test.stdout),
        )
        if result.status != "ok":
            return result.status
        if not outputs_match(test.stdout, result.stdout):
            return "wrong_answer"
        timings.append(result.cpu_seconds)
    return timings


def curate(row: SourceRow) -> Curated | Rejection:
    pid = problem_id(row.statement)
    reference, timings, wrong = None, None, False
    for code in row.references[:MAX_REFERENCES_TRIED]:
        outcome = _timings(code, row.tests)
        if isinstance(outcome, list):
            if reference is None:
                reference, timings = code, outcome
        elif outcome == "wrong_answer":
            wrong = True
    if reference is None:
        # Disagreement only means something beside a reference that passes.
        return Rejection(pid, "no_passing_reference")
    if wrong:
        return Rejection(pid, "references_disagree")
    if max(timings) > MAX_REFERENCE_TEST_S:
        return Rejection(pid, "reference_too_slow")
    kept = select_tests(row.tests, timings)
    if len(kept) < MIN_TESTS:
        return Rejection(pid, "too_few_tests")
    return Curated(
        problem_id=pid,
        split=split_of(pid),
        statement=row.statement,
        tests=tuple(row.tests[i] for i in kept),
        time_limit_s=time_limit(max(timings[i] for i in kept)),
        reference=reference,
        origin={"source": row.source, "upstream_id": row.upstream_id},
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validate.py -v`
Expected: 9 passed (the slow-reference test takes a few seconds)

- [ ] **Step 5: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): validate with a reference, calibrate limits, cap tests, split

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Build pipeline, dataset writer, fetch and CLI

**Files:**
- Create: `reliquary_competitive_code/build/pipeline.py`, `reliquary_competitive_code/build/io.py`, `reliquary_competitive_code/build/fetch.py`, `reliquary_competitive_code/build/__main__.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 4-7
- Produces:
  - `pipeline.build(rows: Iterable[SourceRow], held_out: HeldOutIndex, *, workers: int) -> tuple[list[Curated], dict[str, int]]` — report keys: `loaded`, `dropped_date`, `dropped_contaminated`, `dropped_multi_answer`, `after_dedup`, `dropped_no_reference`, `dropped_<rejection reason>`, `curated`, `split_train`, `split_eval`, `split_qualification`
  - `io.write_dataset(curated: Sequence[Curated], out: Path) -> dict[str, str]` writing `problems-train.parquet`, `problems-eval.parquet`, `problems-qualification.parquet` (columns `problem_id`, `statement`, `tests_json`, `time_limit_s`, `origin_json`; row group size 64; rows sorted by `problem_id`) and `references.parquet` (`problem_id`, `code`); returns `{filename: sha256}`
  - `io.ROW_GROUP_SIZE = 64`, `io.problems_file(split: str) -> str`, `io.REFERENCES_FILE = "references.parquet"`
  - `tests_json` format: `json.dumps([[stdin, stdout], ...], ensure_ascii=False)`
  - CLI: `python -m reliquary_competitive_code.build --deepcoder DIR --lcb DIR --out DIR --workers N [--limit N]` writes the dataset plus `build_report.json` (`{"report": {...}, "files": {name: sha256}}`)
  - `python -m reliquary_competitive_code.build.fetch --out DIR` downloads DeepCoder (`taco/train-*`, `primeintellect/train-*`) to `DIR/deepcoder` and LCB `test*.jsonl` to `DIR/lcb`, at the pinned revisions

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline.py`:
```python
import json

import pyarrow.parquet as pq

from reliquary_competitive_code.build import io
from reliquary_competitive_code.build.filters import HeldOutIndex
from reliquary_competitive_code.build.pipeline import build
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = tuple(TestCase(f"{i} {i}\n", f"{2 * i}\n") for i in range(6))
HELD = "Takahashi has N cards numbered 1 to N arranged in a row and wants to remove exactly K of them so that the remaining sum is maximal modulo M"


def _rows():
    return [
        SourceRow("t", "1", "Print the sum of a and b given on a single line.", TESTS, (SUM,)),
        SourceRow("p", "2", "PRINT THE SUM OF A AND B, given on a single line!", TESTS, ()),
        SourceRow("t", "3", "Story. " + HELD, TESTS, (SUM,)),
        SourceRow("t", "4", "Print the sum. If there are several answers, print any of them.", TESTS, (SUM,)),
        SourceRow("t", "5", "Print a plus b, a newer problem.", TESTS, (SUM,), "2025-02-01"),
        SourceRow("t", "6", "Print the product of a and b given on a single line.", TESTS, ("print(0)",)),
    ]


def test_build_applies_every_stage_and_reports_it() -> None:
    curated, report = build(_rows(), HeldOutIndex([HELD]), workers=2)
    # Rows 1 and 2 are the same problem; both have 6 tests, so the lead is the
    # larger (source, upstream_id): ("t", "1"). Row 2 brings no reference and
    # is not dropped for it, because duplicates pool references first.
    assert [c.origin["upstream_id"] for c in curated] == ["1"]
    assert report["loaded"] == 6
    assert report["dropped_date"] == 1
    assert report["dropped_contaminated"] == 1
    assert report["dropped_multi_answer"] == 1
    assert report["after_dedup"] == 2
    assert report["dropped_no_passing_reference"] == 1
    assert report["curated"] == 1
    assert sum(report[f"split_{s}"] for s in ("train", "eval", "qualification")) == 1


def test_write_dataset_round_trips(tmp_path) -> None:
    curated, _ = build(_rows(), HeldOutIndex([]), workers=1)
    digests = io.write_dataset(curated, tmp_path)
    assert set(digests) == {io.problems_file(s) for s in ("train", "eval", "qualification")} | {io.REFERENCES_FILE}
    split = curated[0].split
    table = pq.read_table(tmp_path / io.problems_file(split))
    row = table.to_pylist()[0]
    assert row["problem_id"] == curated[0].problem_id
    assert [TestCase(a, b) for a, b in json.loads(row["tests_json"])] == list(curated[0].tests)
    assert json.loads(row["origin_json"]) == curated[0].origin
    refs = pq.read_table(tmp_path / io.REFERENCES_FILE).to_pylist()
    assert refs == [{"problem_id": curated[0].problem_id, "code": SUM}]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/build/pipeline.py`:
```python
"""Sources in, curated problems out, with a count for every reason a row left."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from reliquary_competitive_code.build.dedup import deduplicate
from reliquary_competitive_code.build.filters import (
    CONTAMINATION_THRESHOLD,
    HeldOutIndex,
    is_after_cutoff,
    is_multi_answer,
)
from reliquary_competitive_code.build.validate import SPLITS, Curated, curate
from reliquary_competitive_code.sources.common import SourceRow


def build(
    rows: Iterable[SourceRow], held_out: HeldOutIndex, *, workers: int
) -> tuple[list[Curated], dict[str, int]]:
    report: Counter[str] = Counter()
    kept = []
    for row in rows:
        report["loaded"] += 1
        if is_after_cutoff(row):
            report["dropped_date"] += 1
        elif held_out.overlap(row.statement) >= CONTAMINATION_THRESHOLD:
            report["dropped_contaminated"] += 1
        elif is_multi_answer(row):
            report["dropped_multi_answer"] += 1
        else:
            kept.append(row)
    merged = deduplicate(kept)
    report["after_dedup"] = len(merged)
    candidates = []
    for row in merged:
        if row.references:
            candidates.append(row)
        else:
            report["dropped_no_reference"] += 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(curate, candidates))
    curated = sorted((o for o in outcomes if isinstance(o, Curated)), key=lambda c: c.problem_id)
    for outcome in outcomes:
        if not isinstance(outcome, Curated):
            report[f"dropped_{outcome.reason}"] += 1
    report["curated"] = len(curated)
    for split in SPLITS:
        report[f"split_{split}"] = sum(c.split == split for c in curated)
    return curated, dict(report)
```

Rows without references are counted after deduplication on purpose: duplicates pool their references first, so a row with none is only dropped if no duplicate of it has one. In the test, `dropped_no_passing_reference == 1` comes from row `"6"`, whose only reference prints the wrong answer.

`reliquary_competitive_code/build/io.py`:
```python
"""The published layout of the curated dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reliquary_competitive_code.build.validate import SPLITS, Curated

ROW_GROUP_SIZE = 64
REFERENCES_FILE = "references.parquet"


def problems_file(split: str) -> str:
    return f"problems-{split}.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(curated: Sequence[Curated], out: Path) -> dict[str, str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(curated, key=lambda c: c.problem_id)
    digests = {}
    for split in SPLITS:
        rows = [c for c in ordered if c.split == split]
        table = pa.table({
            "problem_id": [c.problem_id for c in rows],
            "statement": [c.statement for c in rows],
            "tests_json": [json.dumps([[t.stdin, t.stdout] for t in c.tests], ensure_ascii=False) for c in rows],
            "time_limit_s": pa.array([c.time_limit_s for c in rows], type=pa.float64()),
            "origin_json": [json.dumps(c.origin, sort_keys=True) for c in rows],
        })
        path = out / problems_file(split)
        pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
        digests[path.name] = _sha256(path)
    references = pa.table({
        "problem_id": [c.problem_id for c in ordered],
        "code": [c.reference for c in ordered],
    })
    path = out / REFERENCES_FILE
    pq.write_table(references, path, compression="zstd")
    digests[path.name] = _sha256(path)
    return digests
```

`reliquary_competitive_code/build/fetch.py`:
```python
"""Download the pinned upstream data a build reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from reliquary_competitive_code.build.filters import LCB_REPOSITORY, LCB_REVISION
from reliquary_competitive_code.sources import deepcoder


def fetch(out: Path) -> None:
    snapshot_download(
        deepcoder.REPOSITORY, repo_type="dataset", revision=deepcoder.REVISION,
        allow_patterns=[f"{config}/train-*" for config in deepcoder.CONFIGS],
        local_dir=Path(out) / "deepcoder",
    )
    snapshot_download(
        LCB_REPOSITORY, repo_type="dataset", revision=LCB_REVISION,
        allow_patterns=["test*.jsonl"], local_dir=Path(out) / "lcb",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    fetch(parser.parse_args().out)
```

`reliquary_competitive_code/build/__main__.py`:
```python
"""Build the curated dataset: python -m reliquary_competitive_code.build ..."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from reliquary_competitive_code.build.filters import HeldOutIndex, load_lcb_statements
from reliquary_competitive_code.build.io import write_dataset
from reliquary_competitive_code.build.pipeline import build
from reliquary_competitive_code.sources import deepcoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepcoder", type=Path, required=True)
    parser.add_argument("--lcb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N source rows, for a smoke run")
    args = parser.parse_args()
    rows = deepcoder.rows(args.deepcoder)
    if args.limit is not None:
        rows = itertools.islice(rows, args.limit)
    curated, report = build(rows, HeldOutIndex(load_lcb_statements(args.lcb)), workers=args.workers)
    files = write_dataset(curated, args.out)
    (args.out / "build_report.json").write_text(
        json.dumps({"report": report, "files": files}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 2 passed

- [ ] **Step 5: Smoke run on real data** (small: 300 rows, 4 workers, local)

Run:
```bash
uv run python -m reliquary_competitive_code.build.fetch --out /home/ubuntu/cc-data/fetched
uv run python -m reliquary_competitive_code.build --deepcoder /home/ubuntu/cc-data/fetched/deepcoder --lcb /home/ubuntu/cc-data/fetched/lcb --out /home/ubuntu/cc-data/smoke --workers 4 --limit 300
```
Expected: a report where `curated` is a substantial share of `after_dedup`. If `dropped_no_passing_reference` exceeds half, inspect five rejected problems by hand (run their reference with `run_test` and look at the status) before going further: a harness bug here zeroes the whole environment. Put the report in the commit message.

- [ ] **Step 6: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): build, write and fetch the curated dataset

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Served side — corpus, environment, Verifiers taskset, packaging files

**Files:**
- Create: `reliquary_competitive_code/corpus.py`, `reliquary_competitive_code/environment.py`, `reliquary_competitive_code/taskset.py`, `environment.toml`, `examples/prime_rl/rl.toml`, `scripts/write_goldens.py`, `scripts/verify_wheel.py`
- Modify: `README.md`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: `write_dataset`, `problems_file`, `REFERENCES_FILE`, `ROW_GROUP_SIZE` (Task 8); `judge`, `TestCase` (Task 3); `extract_program` (Task 1); `SPLITS` (Task 7)
- Produces:
  - `corpus.DatasetPin(repository: str, revision: str, files: dict[str, str])`, `corpus.PINNED: DatasetPin | None = None` (set in Task 10)
  - `corpus.Problem(problem_id: str, statement: str, time_limit_s: float, origin: dict)`
  - `corpus.Corpus(directory: Path)` with `.problems(split) -> list[Problem]`, `.tests(split, index) -> tuple[TestCase, ...]`, `.reference(problem_id) -> str`; `Corpus.pinned() -> Corpus`; module function `pinned_corpus() -> Corpus` (cached)
  - `environment.ENVIRONMENT = "reliquary_competitive_code_v1"`, `TASK_FAMILY = "competitive_programming_stdio_v1"`, `JUDGE_VERSION = "stdio-tokens-v1"`, `INSTRUCTION`, `render_prompt(statement) -> str`
  - `environment.CompetitiveCodeEnvironment(split="train", corpus: Corpus | None = None)` with `__len__`, `task(index) -> {"id","prompt","metadata"}`, `grade(index, completion) -> {"reward","success","status","state_digest"}`, `replay(index, completion) -> {"reward": <grade dict>}`, `reference_completion(index) -> str`
  - `taskset.CompetitiveCodeTaskset` registered as `reliquary-competitive-code`

- [ ] **Step 1: Write the failing test** — `tests/test_environment.py`:
```python
import pytest

from reliquary_competitive_code.build.io import write_dataset
from reliquary_competitive_code.build.validate import Curated, split_of
from reliquary_competitive_code.corpus import Corpus
from reliquary_competitive_code.environment import (
    INSTRUCTION,
    CompetitiveCodeEnvironment,
    render_prompt,
)
from reliquary_competitive_code.judge import TestCase

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"


def _curated(n: int) -> list[Curated]:
    out = []
    for i in range(n):
        pid = f"{i:016x}"
        tests = tuple(TestCase(f"{i} {j}\n", f"{i + j}\n") for j in range(5))
        out.append(Curated(pid, split_of(pid), f"Problem {i}: print a plus b.", tests, 1.0, SUM, {"source": "t", "upstream_id": str(i)}))
    return out


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Corpus:
    directory = tmp_path_factory.mktemp("dataset")
    write_dataset(_curated(300), directory)
    return Corpus(directory)


def test_tasks_carry_the_statement_and_the_contract(corpus) -> None:
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    assert len(environment) == len(corpus.problems("train")) > 200
    task = environment.task(3)
    assert task["prompt"] == render_prompt(corpus.problems("train")[3].statement)
    assert task["prompt"].endswith(INSTRUCTION)
    assert task["metadata"]["problem_id"] == corpus.problems("train")[3].problem_id
    assert len(task["id"]) == 16


def test_grading_rewards_only_a_passing_program(corpus) -> None:
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    good = environment.grade(5, environment.reference_completion(5))
    assert good["reward"] == 1.0 and good["success"] and good["status"] == "ok"
    assert environment.grade(5, "```python\nprint(-1)\n```")["status"] == "wrong_answer"
    for completion in ("", "no code", "```cpp\nint main(){}\n```", "```python\nprint(1)\n"):
        graded = environment.grade(5, completion)
        assert graded["reward"] == 0.0 and graded["status"] == "no_code"
    assert environment.replay(5, environment.reference_completion(5))["reward"]["reward"] == 1.0


def test_state_digest_follows_the_verdict_not_the_text(corpus) -> None:
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    a = environment.grade(1, environment.reference_completion(1))["state_digest"]
    b = environment.grade(1, "Thinking...\n```python\n" + SUM + "```")["state_digest"]
    assert a == b


def test_tests_are_read_lazily_by_row_group(corpus) -> None:
    problems = corpus.problems("train")
    for index in (0, 63, 64, len(problems) - 1):
        tests = corpus.tests("train", index)
        assert len(tests) == 5
        assert tests[0].stdin.split()[0] == str(int(problems[index].problem_id, 16))


def test_the_pinned_corpus_refuses_to_load_without_a_pin(monkeypatch) -> None:
    from reliquary_competitive_code import corpus as module

    monkeypatch.setattr(module, "PINNED", None)
    with pytest.raises(RuntimeError, match="no curated dataset is pinned"):
        Corpus.pinned()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_environment.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `reliquary_competitive_code/corpus.py`:
```python
"""The curated dataset this environment serves, pinned by revision and digest.

Built offline by `reliquary_competitive_code.build` and published to the Hub;
production only reads it. Statements and limits load eagerly, tests lazily by
row group, so a validator holds one row group of tests at a time rather than
the whole split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.build.io import REFERENCES_FILE, problems_file
from reliquary_competitive_code.build.validate import SPLITS
from reliquary_competitive_code.judge import TestCase


@dataclass(frozen=True, slots=True)
class DatasetPin:
    repository: str
    revision: str
    files: dict[str, str]


PINNED: DatasetPin | None = None


@dataclass(frozen=True, slots=True)
class Problem:
    problem_id: str
    statement: str
    time_limit_s: float
    origin: dict


class Corpus:
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._problems: dict[str, list[Problem]] = {}
        self._row_groups: dict[str, list[int]] = {}
        for split in SPLITS:
            path = self._directory / problems_file(split)
            table = pq.read_table(path, columns=["problem_id", "statement", "time_limit_s", "origin_json"])
            self._problems[split] = [
                Problem(r["problem_id"], r["statement"], float(r["time_limit_s"]), json.loads(r["origin_json"]))
                for r in table.to_pylist()
            ]
            metadata = pq.ParquetFile(path).metadata
            starts, total = [], 0
            for group in range(metadata.num_row_groups):
                starts.append(total)
                total += metadata.row_group(group).num_rows
            self._row_groups[split] = starts
        self._references: dict[str, str] | None = None

    @classmethod
    def pinned(cls) -> Corpus:
        if PINNED is None:
            raise RuntimeError("no curated dataset is pinned in reliquary_competitive_code.corpus")
        from huggingface_hub import hf_hub_download

        directory = None
        for name, digest in PINNED.files.items():
            path = Path(hf_hub_download(PINNED.repository, name, repo_type="dataset", revision=PINNED.revision))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"{name}: downloaded {actual}, pinned {digest}")
            directory = path.parent
        return cls(directory)

    def problems(self, split: str) -> list[Problem]:
        return self._problems[split]

    def tests(self, split: str, index: int) -> tuple[TestCase, ...]:
        starts = self._row_groups[split]
        group = max(g for g, start in enumerate(starts) if start <= index)
        table = pq.ParquetFile(self._directory / problems_file(split)).read_row_group(group, columns=["tests_json"])
        raw = table.column("tests_json")[index - starts[group]].as_py()
        return tuple(TestCase(a, b) for a, b in json.loads(raw))

    def reference(self, problem_id: str) -> str:
        if self._references is None:
            table = pq.read_table(self._directory / REFERENCES_FILE)
            self._references = dict(zip(table.column("problem_id").to_pylist(), table.column("code").to_pylist()))
        return self._references[problem_id]


@lru_cache(maxsize=1)
def pinned_corpus() -> Corpus:
    return Corpus.pinned()
```

`reliquary_competitive_code/environment.py`:
```python
"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

The task is a competitive-programming statement and the contract below; the
submission is the last fenced Python block; the reward is 1.0 when it passes
every hidden test of the problem and 0.0 otherwise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_competitive_code.build.validate import SPLITS
from reliquary_competitive_code.corpus import Corpus, pinned_corpus
from reliquary_competitive_code.extraction import extract_program
from reliquary_competitive_code.judge import judge

ENVIRONMENT = "reliquary_competitive_code_v1"
TASK_FAMILY = "competitive_programming_stdio_v1"
JUDGE_VERSION = "stdio-tokens-v1"
INSTRUCTION = (
    "\n\nWrite a complete Python 3 program that reads the input from standard "
    "input and prints the answer to standard output. Use input() or sys.stdin; "
    "the os and io modules and file access are not available. Put the whole "
    "program in a single ```python code block at the end of your answer."
)


def render_prompt(statement: str) -> str:
    return statement.strip() + INSTRUCTION


class CompetitiveCodeEnvironment:
    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(self, split: str = "train", corpus: Corpus | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self._corpus = corpus if corpus is not None else pinned_corpus()

    def __len__(self) -> int:
        return len(self._corpus.problems(self.split))

    def _problem(self, index: int):
        problems = self._corpus.problems(self.split)
        return int(index) % len(problems), problems[int(index) % len(problems)]

    def _identity(self, problem_id: str) -> str:
        return hashlib.sha256(f"{ENVIRONMENT}:{JUDGE_VERSION}:{problem_id}".encode()).hexdigest()[:16]

    def task(self, index: int) -> dict[str, Any]:
        _, problem = self._problem(index)
        return {
            "id": self._identity(problem.problem_id),
            "prompt": render_prompt(problem.statement),
            "metadata": {
                "task_family": TASK_FAMILY,
                "problem_id": problem.problem_id,
                "judge_version": JUDGE_VERSION,
                "time_limit_s": problem.time_limit_s,
                "split": self.split,
            },
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        position, problem = self._problem(index)
        verdict = judge(
            extract_program(completion or ""),
            self._corpus.tests(self.split, position),
            time_limit_s=problem.time_limit_s,
        )
        reward = 1.0 if verdict.passed else 0.0
        return {
            "reward": reward,
            "success": verdict.passed,
            "status": verdict.status,
            # The digest carries the verdict, not the program: two correct
            # programs differ in every character.
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"id": self._identity(problem.problem_id), "success": verdict.passed},
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        _, problem = self._problem(index)
        return f"```python\n{self._corpus.reference(problem.problem_id)}\n```"


__all__ = ["ENVIRONMENT", "INSTRUCTION", "JUDGE_VERSION", "TASK_FAMILY", "CompetitiveCodeEnvironment", "render_prompt"]
```

`reliquary_competitive_code/taskset.py`:
```python
"""Competitive programming as a native Verifiers taskset. Single turn, no tools."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from reliquary_competitive_code.environment import CompetitiveCodeEnvironment


class CompetitiveCodeData(vf.TaskData):
    task_id: str
    problem_id: str
    judge_version: str
    time_limit_s: float
    split: Literal["train", "eval", "qualification"]


class CompetitiveCodeTaskConfig(vf.TaskConfig):
    pass


class CompetitiveCodeTask(vf.Task[CompetitiveCodeData, vf.State, CompetitiveCodeTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    def _environment(self) -> CompetitiveCodeEnvironment:
        return CompetitiveCodeEnvironment(self.data.split)

    @vf.reward(weight=1.0)
    async def passes_all_tests(self, trace: vf.Trace) -> float:
        graded = await asyncio.to_thread(self._environment().grade, self.data.idx, trace.last_reply or "")
        return graded["reward"]

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference passes and a program printing nothing useful does not."""
        del runtime
        environment = self._environment()
        good = await asyncio.to_thread(environment.grade, self.data.idx, environment.reference_completion(self.data.idx))
        bad = await asyncio.to_thread(environment.grade, self.data.idx, "```python\nprint('reliquary-wrong')\n```")
        return good["reward"] == 1.0 and bad["reward"] == 0.0


class CompetitiveCodeConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: CompetitiveCodeTaskConfig = CompetitiveCodeTaskConfig()


class CompetitiveCodeTaskset(vf.Taskset[CompetitiveCodeTask, CompetitiveCodeConfig]):
    def load(self) -> Iterator[CompetitiveCodeTask]:
        environment = CompetitiveCodeEnvironment(self.config.split)
        for index in range(len(environment)):
            task = environment.task(index)
            metadata = task["metadata"]
            yield CompetitiveCodeTask(
                CompetitiveCodeData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    problem_id=metadata["problem_id"],
                    judge_version=metadata["judge_version"],
                    time_limit_s=metadata["time_limit_s"],
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["CompetitiveCodeTaskset"]
```

No registration step is needed: Verifiers resolves the taskset id `reliquary-competitive-code` by importing the module `reliquary_competitive_code` (dashes to underscores, `verifiers/v1/utils/loaders.py::_import_plugin`) and taking the single `Taskset` subclass named in its `__all__`, which the lazy `__getattr__` of Task 1 provides.

`environment.toml` (the `[data]` table is completed in Task 10):
```toml
schema = "reliquary/environment-source/v2"
id = "reliquary/competitive-code"
version = "0.1.0a1"
taskset = "reliquary-competitive-code"
entrypoint = "reliquary_competitive_code:CompetitiveCodeTaskset"
compatibility_entrypoint = "reliquary_competitive_code.environment:CompetitiveCodeEnvironment"
license = "MIT"
verification_tier = "deterministic-replay"
owners = ["Reliquary contributors"]
task_families = ["competitive_programming_stdio"]

[compatibility]
python = ">=3.12,<3.13"
verifiers = ">=0.3.1,<0.4"
verifiers_api = "v1"

[execution]
network = false
secrets = []
resource_class = "cpu"
max_turns = 1
max_observation_bytes = 16384

[reward]
minimum = 0.0
maximum = 1.0
components = ["passes_all_tests"]

[provenance]
source_repository = "https://github.com/reliquadotai/reliquary-environments"
upstream = "https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset"
upstream_license = "MIT"
port = "curated-stdio-subset"

[policy]
reasoning = "thinking"
reasoning_rationale = """
A competitive problem is an algorithm to find before it is a program to type,
and only the last fenced Python block is run: the reasoning before it is never
executed and never graded, so deliberation costs tokens and cannot cost reward.
"""
max_new_tokens = 16384
max_new_tokens_rationale = """
Not yet measured on a policy. Twice the sibling reliquary-code's measured 8,192,
because these problems need an algorithm before the program; to be replaced by
a value measured at 8k, 16k and 32k on Teutonic, as reliquary-dapo-math's was.
"""
```

`examples/prime_rl/rl.toml`: copy `../../reasoning/reliquary_science/examples/prime_rl/rl.toml`, then replace every `reliquary-science` by `reliquary-competitive-code`, set `seq_len = 24576`, `max_model_len = 24576` and both `max_completion_tokens = 16384`, and rewrite the two comments that mention derivations and boxes to speak of a program and its fenced block.

`scripts/write_goldens.py` and `scripts/verify_wheel.py`: copy the science versions and adapt. The goldens are one task per split at indexes `{"train": 16, "eval": 5, "qualification": 11}`, each with fields `split`, `index`, `task_id`, `problem_id`, `prompt_sha256`, `completion` (the reference completion, expected 1.0), `wrong_completion` (`"```python\nprint('reliquary-wrong')\n```"`, expected 0.0), `no_code_completion` (`"I would use a segment tree."`, expected 0.0), `state_digest`. The artifact manifest uses `"environment": "reliquary_competitive_code_v1"`, `"contract": "reliquary/stdio-program/v1"`, distribution `reliquary-competitive-code` `0.1.0a1`, entrypoints `taskset: reliquary_competitive_code:CompetitiveCodeTaskset` and `replay: reliquary_competitive_code.environment:CompetitiveCodeEnvironment`, and excludes `build/` and `sources/` from nothing (every shipped file is pinned). `verify_wheel.py` checks the artifact digests, grades the three completions of each golden, and checks that the three split sizes equal the `[data]` sizes declared in `environment.toml`.

`README.md`: what the environment is, the three layers (section 2 of the spec), how to fetch and build (`build.fetch`, then `build`), the drop counts of the published revision (filled in Task 10), and the known limits (multi-answer problems excluded, Python only, the local runner is not a sandbox).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: every test of the package passes (Tasks 1-9).

- [ ] **Step 5: Commit**
```bash
git add environments/code/reliquary_competitive_code
git commit -m "feat(competitive-code): serve the curated dataset through the ABI and a Verifiers taskset

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Real build, publish, pin, goldens and CI (operator-gated)

This task needs two things a subagent does not have: a build box (8+ dedicated cores, 20 GB free disk; **not** the local VPS, **not** `sandbox-dev-01` without the user's say-so, since it is the agentic corpus executor) and a Hugging Face token with write access to `R0mAI`. Stop and ask the user for both before Step 1.

**Files:**
- Modify: `reliquary_competitive_code/corpus.py` (`PINNED`), `environment.toml` (`[data]`), `README.md`, `.github/workflows/ci.yml` (repo root)
- Create: `reliquary_competitive_code/goldens/reference.jsonl`, `reliquary_competitive_code/artifact.json` (both generated)

- [ ] **Step 1: Build on the box**
```bash
uv run python -m reliquary_competitive_code.build.fetch --out /opt/cc-build/fetched
uv run python -m reliquary_competitive_code.build --deepcoder /opt/cc-build/fetched/deepcoder --lcb /opt/cc-build/fetched/lcb --out /opt/cc-build/out --workers $(nproc)
cat /opt/cc-build/out/build_report.json
```
Expected: `curated` between 6,000 and 14,000. Below 6,000, stop and report the drop counts to the user before publishing.

- [ ] **Step 2: Spot-check five curated problems and five rejections** of each reason by running the stored reference with `run_test` and reading the statement. Write what you saw in the README section "Build of <revision>".

- [ ] **Step 3: Publish**
```bash
huggingface-cli upload R0mAI/competitive-code-curated /opt/cc-build/out . --repo-type dataset --commit-message "competitive-code-curated v1 from DeepCoder taco+primeintellect"
```
Record the commit sha it prints.

- [ ] **Step 4: Pin** — in `corpus.py` set:
```python
PINNED = DatasetPin(
    repository="R0mAI/competitive-code-curated",
    revision="<the sha printed by Step 3>",
    files={  # copied from build_report.json "files"
        "problems-train.parquet": "<sha256>",
        "problems-eval.parquet": "<sha256>",
        "problems-qualification.parquet": "<sha256>",
        "references.parquet": "<sha256>",
    },
)
```
and add to `environment.toml`:
```toml
[data]
train = "hf:R0mAI/competitive-code-curated@<sha>#problems-train.parquet"
eval = "hf:R0mAI/competitive-code-curated@<sha>#problems-eval.parquet"
qualification = "hf:R0mAI/competitive-code-curated@<sha>#problems-qualification.parquet"
redistributable = true
license = "MIT (DeepCoder-Preview-Dataset); problem statements inherit their original judges' terms"
rows = <after_dedup from the report>
virtual_length = <curated from the report>
```
(The angle-bracket values are the build's outputs, copied from `build_report.json` and the upload; nothing else is to be invented.)

- [ ] **Step 5: Goldens and artifact**

Run: `uv run python scripts/write_goldens.py && uv run pytest -v`
Expected: all pass.

- [ ] **Step 6: CI job** — add to `.github/workflows/ci.yml` a job `competitive-code` copied from the `science` job, with `working-directory: environments/code/reliquary_competitive_code`, the validate line `uv run --no-sync validate reliquary-competitive-code -n 3 --only-gold true --split eval --runtime.type docker --runtime.allow '[]' --rich false`, and the wheel name `reliquary_competitive_code-0.1.0a1-py3-none-any.whl`.

- [ ] **Step 7: Commit, push, open the PR**
```bash
git add -A environments/code/reliquary_competitive_code .github/workflows/ci.yml
git commit -m "feat(competitive-code): pin the curated dataset, goldens and CI

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin feat/reliquary-competitive-code
gh pr create --title "feat(env): reliquary-competitive-code, competitive programming on hidden stdin/stdout tests" --body "<summary + build report + spot checks>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## After this plan (not in it)

- Catalyst core plan: gVisor worker `stdin` request kind running `guest.run`, server-side `outputs_match`, `EnvironmentSpec` `reliquary_competitive_code_v1`, Teutonic profile entry, parity goldens through the real grader.
- Qualification on GPU: `max_new_tokens` at 8k/16k/32k on Teutonic, band and margin, grader cost per rollout.
- v1.1: CodeContests+ source adapter with teacher references.
