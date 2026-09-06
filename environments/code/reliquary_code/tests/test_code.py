import hashlib
import importlib
import json
import pkgutil
import time
from pathlib import Path

import pytest

from reliquary_code import corpus, taskset
from reliquary_code.extraction import extract_python
from reliquary_code.runner import run_cases


def test_no_module_imports_reliquary_core() -> None:
    """The port is a copy, not a dependency. A stray `import reliquary`
    would make the package silently unusable outside the core checkout."""
    import reliquary_code

    for info in pkgutil.walk_packages(
        reliquary_code.__path__, prefix="reliquary_code."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "import reliquary\n" not in text
        assert "from reliquary." not in text
        assert "import reliquary." not in text


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OCI_REPO == "R0mAI/opencodeinstruct-curated"
    assert corpus.OCI_REVISION == "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"


def test_malformed_structured_cases_yields_empty_list_not_a_crash() -> None:
    """Fidelity gap fix: core's `_row_cases` (opencodeinstruct.py) wraps the
    JSON decode in try/except and filters non-dicts. This port used to do a
    bare `json.loads(raw)`, so one malformed row would abort the whole
    `CodeTaskset.load()` instead of yielding an empty case list."""
    assert corpus._row_cases({"structured_cases": "not json"}) == []
    assert corpus._row_cases({"structured_cases": ""}) == []
    assert corpus._row_cases({"structured_cases": '{"not": "a list"}'}) == []
    assert corpus._row_cases({}) == []
    assert corpus._row_cases(
        {"structured_cases": '[1, "skip", {"a": 1}]'}
    ) == [{"a": 1}]


def test_plain_fence_is_extracted() -> None:
    assert "def f" in extract_python("```\ndef f():\n    return 1\n```")


def test_language_tagged_fence_is_extracted() -> None:
    """A ```python opener is a block opener like any other. Six separate
    incidents came from treating it as content."""
    assert "def f" in extract_python("```python\ndef f():\n    return 1\n```")


@pytest.mark.parametrize("tag", ["py", "python3"])
def test_any_language_tag_opens_a_block(tag: str) -> None:
    assert "def f" in extract_python(f"```{tag}\ndef f():\n    return 1\n```")


def test_uppercase_language_tag_is_not_recognised() -> None:
    """The brief for this test proposed parametrizing over ["py", "python3",
    "Python", "PYTHON"], asserting all four open a block. Verified directly
    against an unmodified core checkout (reliquary/environment/
    opencodeinstruct.py @ 10c2a4d9, calling `_select_python_span` with
    protocol_version=5): `_FENCE_RE`'s tag group `(?:python3?|py)?` has no
    `re.IGNORECASE`, so "Python" and "PYTHON" fail to open a block there too
    — this is core's actual behavior at the pinned commit, not a copying
    error in this port. Pinning the verified behavior here rather than the
    brief's assumed one, so a future core fix shows up as a visible,
    intentional diff instead of this port silently drifting from core."""
    assert extract_python("```Python\ndef f():\n    return 1\n```") == ""


def test_the_last_parsing_block_wins() -> None:
    """A model that shows a broken attempt then a working one is graded on
    the working one."""
    completion = (
        "```python\ndef f(:\n```\n"
        "then the real answer\n"
        "```python\ndef f():\n    return 2\n```"
    )
    assert "return 2" in extract_python(completion)


def test_unparseable_output_yields_empty() -> None:
    assert extract_python("no code here").strip() == ""


def test_entry_function_is_preferred_when_named() -> None:
    completion = (
        "```python\ndef helper():\n    return 1\n```\n"
        "```python\ndef solve():\n    return 2\n```"
    )
    assert "def solve" in extract_python(completion, entry_name="solve")


# ---------------------------------------------------------------------------
# Runner: the real OpenCodeInstruct corpus ships function-call cases, not
# stdin/stdout pairs — see reliquary_code/runner.py's module docstring for
# the defect this replaced. Every case below names an entrypoint and calls
# it with `args`/`kwargs`; grading compares the return value to `expected`.
# ---------------------------------------------------------------------------

CASES = [
    {
        "entry": {"kind": "function", "name": "increment"},
        "args": [1],
        "kwargs": {},
        "expected": 2,
        "compare": "exact",
    },
    {
        "entry": {"kind": "function", "name": "increment"},
        "args": [2],
        "kwargs": {},
        "expected": 3,
        "compare": "exact",
    },
]

INCREMENT = "def increment(n):\n    return n + 1\n"


def test_correct_source_passes_every_case() -> None:
    assert run_cases(INCREMENT, CASES) == [True, True]


def test_wrong_source_fails_every_case() -> None:
    wrong = "def increment(n):\n    return 999\n"
    assert run_cases(wrong, CASES) == [False, False]


def test_a_cpu_bomb_is_killed_and_scored_false() -> None:
    source = "def increment(n):\n    while True:\n        pass\n"
    assert run_cases(source, CASES[:1], cpu_seconds=1) == [False]


def test_a_sleeping_process_is_killed_by_the_wall_clock() -> None:
    """A sleeper burns no CPU, so RLIMIT_CPU alone would never fire.

    Case execution runs submitted code through `cases.evaluate_call`'s
    sandbox, which only allows importing a fixed safe module list (see
    `cases._ALLOWED_IMPORT_ROOTS`) — `time` is not on it, so a case cannot
    sleep without importing something the sandbox rejects, and there is no
    case-format way around that. This exercises the same wall-clock
    guarantee `run_cases` relies on at the layer that actually enforces it:
    `_spawn`, which is generic over what `source` contains and is not
    itself sandboxed (the sandbox is `cases.evaluate_call`'s doing, layered
    on top by the worker script `run_cases` hands to `_spawn`).
    """
    from reliquary_code.runner import _spawn

    source = "import time\ntime.sleep(30)\n"
    wall_seconds = 1.0
    started = time.monotonic()
    completed = _spawn(
        source, "", cpu_seconds=5, memory_bytes=512 * 1024 * 1024, wall_seconds=wall_seconds
    )
    elapsed = time.monotonic() - started
    assert completed.returncode != 0
    assert elapsed < wall_seconds + 4.0


def test_a_memory_bomb_is_killed_and_scored_false() -> None:
    source = "def increment(n):\n    return len(bytearray(2 * 1024 * 1024 * 1024))\n"
    assert run_cases(source, CASES[:1], memory_bytes=64 * 1024 * 1024) == [False]


def _iterations_for(seconds: float) -> int:
    """Calibrate a pure busy-loop iteration count against CPU time.

    `time` is not on `cases._ALLOWED_IMPORT_ROOTS`, so submitted code
    cannot self-time with `time.process_time()` the way the old
    stdin/stdout version of this test did. Calibrating in the test process
    (unsandboxed) and baking a literal iteration count into the submitted
    source keeps the test's original point — CPU-time-targeted, not
    wall-clock-targeted, so it is not flaky under a loaded machine — without
    needing the submitted code to time itself.
    """
    probe = 2_000_000
    start = time.process_time()
    total = 0
    for i in range(probe):
        total += i
    elapsed = time.process_time() - start or 1e-6
    rate = probe / elapsed
    return max(1, int(rate * seconds))


def test_each_case_gets_a_fresh_process() -> None:
    """RLIMIT_CPU is cumulative for the life of a process: if a worker
    served both cases, a legal-but-costly first case would leave a CPU debt
    that an innocent, equally legal second case inherits — and gets
    SIGKILLed for. That is the exact failure mode described in the runner's
    module docstring (12% of code submissions lost in production).

    The brief's version of this test collected a `pids` list it never
    asserted on; its only real assertions were two unrelated output
    mismatches ("blocked"/"reached" strings copied from the network test it
    also called), so it would pass against a pooled implementation and
    proved nothing about process freshness.

    This version proves freshness through the public `run_cases` API by
    reproducing the production incident directly: two cases, each alone
    burning ~0.6 CPU-seconds against a 1-CPU-second budget — individually
    legal, but 1.2s combined would blow a *shared* 1s budget. If each case
    gets its own fresh process (its own fresh RLIMIT_CPU accounting), both
    finish comfortably inside the budget and pass. If the two cases shared
    one process, the second case would inherit the first case's CPU debt and
    be SIGKILLed, turning `[True, True]` into `[True, False]`.
    """
    iterations = _iterations_for(0.6)
    source = (
        "def busy(n):\n"
        "    total = 0\n"
        f"    for i in range({iterations}):\n"
        "        total += i\n"
        "    return 'done'\n"
    )
    cases = [
        {
            "entry": {"kind": "function", "name": "busy"},
            "args": [0],
            "kwargs": {},
            "expected": "done",
            "compare": "exact",
        },
        {
            "entry": {"kind": "function", "name": "busy"},
            "args": [0],
            "kwargs": {},
            "expected": "done",
            "compare": "exact",
        },
    ]
    assert run_cases(source, cases, cpu_seconds=1) == [True, True]


def test_two_evaluations_get_different_process_ids() -> None:
    """Names the invariant directly: two independent evaluations of a
    PID-printing program must report different PIDs, because each gets its
    own fresh subprocess. `test_each_case_gets_a_fresh_process` above proves
    the *consequence* (a shared CPU budget would kill an innocent case);
    this proves the *invariant* itself, which is faster to read for anyone
    changing the runner later.

    `run_cases`'s public API only returns booleans, with nowhere to carry a
    PID back to the caller, so this goes one layer down to `_spawn` — the
    same per-case entry point `run_cases` calls once per case — to read the
    real child's stdout.
    """
    from reliquary_code.runner import _spawn

    source = "import os, sys\nsys.stdout.write(str(os.getpid()))\n"
    first = _spawn(
        source, "", cpu_seconds=5, memory_bytes=512 * 1024 * 1024, wall_seconds=5.0
    )
    second = _spawn(
        source, "", cpu_seconds=5, memory_bytes=512 * 1024 * 1024, wall_seconds=5.0
    )
    assert first.stdout.strip() != second.stdout.strip()


def _process_state(pid: int) -> str | None:
    """The Linux process state character for `pid`, or `None` if it is gone
    (already reaped)."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    return next(
        line.split()[1] for line in status.splitlines() if line.startswith("State:")
    )


def test_a_forked_grandchild_does_not_survive_the_wall_clock_kill() -> None:
    """A prior version of this runner relied on `RLIMIT_NPROC` for fork-bomb
    containment. That limit is scoped to the real UID on Linux, not to this
    process's subtree, so it did not actually bound what a case's own
    process tree could do — and independently broke legitimate
    `subprocess`/`multiprocessing` submissions on a busy host (see
    `runner._limits`). The real gap it was covering: `subprocess.run`'s
    default timeout handling kills only the immediate child, so a forked
    grandchild can outlive the wall-clock kill entirely.

    This drives `_spawn` directly with a source that forks a grandchild
    that sleeps far longer than `wall_seconds`, and checks two things after
    `_spawn` returns: that it returned promptly, and that the grandchild is
    dead.

    Both checks are needed. Checking only the end state has a blind spot: a
    regression that reverts `_kill_group` to a plain `proc.kill()` (killing
    only the immediate child) while keeping `start_new_session=True` was
    verified by hand to still pass an end-state-only version of this test.
    The grandchild inherits the stdout pipe fd; killing only the immediate
    child does not close the grandchild's copy of it, so `communicate()`
    blocks until the grandchild exits *on its own* — measured at 10.02s wall
    against a 1.0s `wall_seconds` in that reproduction. By the time `_spawn`
    finally returns, the grandchild has already exited normally, so an
    end-state check alone sees a clean state and reports the test as
    passing — a real regression in the wall-clock guarantee passes slowly
    instead of failing. Bounding elapsed time closes that gap.

    `os.kill(pid, 0)` alone can't tell "killed, not yet reaped" from "still
    alive and sleeping": a zombie still answers signal 0 (its PID slot
    persists until reaped). Reading the state character distinguishes
    actually-still-running (`S`/`R`) from killed (gone, `Z` zombie, or the
    transient `X`/`x` dead state Linux reports mid-teardown).
    """
    source = (
        "import os, sys, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "sys.stdout.write(str(child))\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    from reliquary_code.runner import _spawn

    wall_seconds = 1.0
    started = time.monotonic()
    completed = _spawn(
        source,
        "",
        cpu_seconds=5,
        memory_bytes=512 * 1024 * 1024,
        wall_seconds=wall_seconds,
    )
    elapsed = time.monotonic() - started

    # Not a strict multiple of wall_seconds: spawn/kill/reap overhead is a
    # roughly fixed cost, not proportional to the budget, so a fixed slack
    # on top of it is what stays non-flaky on a loaded machine. This bound
    # only needs to sit well below "the grandchild's own 30s sleep" (or the
    # 10.02s measured in the broken-kill reproduction above) to catch a
    # descendant that kept the pipe open — it is not a precision timing
    # assertion.
    assert elapsed < wall_seconds + 4.0, (
        f"_spawn took {elapsed:.2f}s against a {wall_seconds}s wall clock: "
        "a surviving descendant likely kept the output pipe open"
    )

    grandchild_pid = int(completed.stdout.strip())

    deadline = time.monotonic() + 3.0
    state = _process_state(grandchild_pid)
    while state in ("S", "R") and time.monotonic() < deadline:
        time.sleep(0.05)
        state = _process_state(grandchild_pid)

    assert state not in ("S", "R"), (
        f"grandchild {grandchild_pid} is still alive (state={state!r}) after "
        "the wall-clock kill: it escaped the timeout"
    )


def test_wrong_but_valid_completion_scores_zero_on_a_real_corpus_row() -> None:
    """Regression test for the empty-stdin defect this module used to have.

    `run_cases` used to grade `case["input"]` / `case["expected_output"]` —
    fields the real corpus never sets, since it ships function-call cases
    (see `corpus.get_problem`). Both `.get()` calls silently returned `""`,
    so a wrong answer's empty stdout compared equal to an empty expected
    string and every case passed: a deliberately wrong but syntactically
    valid completion scored 1.0 against a real corpus row. This is exactly
    that input — a completion that defines a callable but returns a
    constant unrelated to any of the row's real cases — replayed against
    real corpus row 0. It must score 0.0.
    """
    row = corpus.get_problem(0)
    cases = row["structured_cases"]
    assert cases, "corpus row 0 must actually carry cases for this test to mean anything"

    wrong_but_valid = (
        "def _wrong_answer(*args, **kwargs):\n"
        "    return '__reliquary_wrong_answer_sentinel__'\n"
    )
    results = run_cases(wrong_but_valid, cases)
    assert results == [False] * len(cases)
    assert sum(1 for ok in results if ok) / len(results) == 0.0


def test_correct_completion_scores_one_on_a_real_corpus_row() -> None:
    """The uncovered direction: the goldens pin that a wrong answer scores
    0 against every pinned index, but nothing pinned that a genuinely
    correct program scores 1.0 against a real corpus row. Row 0 is a
    balanced-brackets problem (see the wrong-answer test above); this is a
    real, correct solution to it, graded end-to-end through
    `CodeEnvironment` so extraction and case execution both run for real,
    not just `run_cases` in isolation."""
    completion = (
        "```python\n"
        "def is_balanced_brackets(expression):\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for ch in expression:\n"
        "        if ch in pairs.values():\n"
        "            stack.append(ch)\n"
        "        elif ch in pairs:\n"
        "            if not stack or stack.pop() != pairs[ch]:\n"
        "                return False\n"
        "    return not stack\n"
        "```"
    )
    assert CodeEnvironment().grade(0, completion)["reward"] == 1.0


import asyncio

from reliquary_code import CodeEnvironment, CodeTaskset
from reliquary_code.taskset import CodeConfig

ROW = {
    "input": "Add one to the given integer and return it.",
    "structured_cases": [
        {
            "entry": {"kind": "function", "name": "increment"},
            "args": [1],
            "kwargs": {},
            "expected": 2,
            "compare": "exact",
        },
        {
            "entry": {"kind": "function", "name": "increment"},
            "args": [5],
            "kwargs": {},
            "expected": 6,
            "compare": "exact",
        },
    ],
}


def test_reward_is_the_fraction_of_passing_cases(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    environment = CodeEnvironment()

    good = "```python\ndef increment(n):\n    return n + 1\n```"
    assert environment.grade(0, good)["reward"] == 1.0
    wrong = "```python\ndef increment(n):\n    return 0\n```"
    assert environment.grade(0, wrong)["reward"] == 0.0


def test_half_passing_scores_half(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    environment = CodeEnvironment()
    source = "```python\ndef increment(n):\n    return 2 if n == 1 else 0\n```"
    assert environment.grade(0, source)["reward"] == 0.5


def test_prompt_carries_the_case_contract(monkeypatch) -> None:
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    task = CodeEnvironment().task(0)
    assert "Add one to the given integer" in task["prompt"]


def test_package_exports_exactly_two_names() -> None:
    import reliquary_code

    assert reliquary_code.__all__ == ["CodeEnvironment", "CodeTaskset"]


def test_taskset_validate_true_when_cases_present_and_wrong_answer_scores_zero(
    monkeypatch,
) -> None:
    """There is no reference completion for code (see `CodeEnvironment.
    known_wrong_completion`), so `validate` has nothing to check the passing
    direction with. This pins what it does check: a well-formed wrong
    answer actually runs through `run_cases` and scores 0, on a non-empty
    case list."""
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: ROW)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    task = next(iter(CodeTaskset(CodeConfig()).load()))
    assert asyncio.run(task.validate(None)) is True


def test_taskset_validate_false_when_case_list_is_empty(monkeypatch) -> None:
    empty_row = {"input": ROW["input"], "structured_cases": []}
    monkeypatch.setattr("reliquary_code.taskset.get_problem", lambda index: empty_row)
    monkeypatch.setattr("reliquary_code.taskset.corpus_length", lambda: 1)
    task = next(iter(CodeTaskset(CodeConfig()).load()))
    assert asyncio.run(task.validate(None)) is False


PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "reliquary_code"

# Shared with `CodeTask.validate` (see taskset.py) so the goldens and the
# taskset's own self-check exercise the identical wrong-but-valid probe.
WRONG_BUT_VALID_COMPLETION = taskset.WRONG_BUT_VALID_COMPLETION


def test_goldens_replay_against_pinned_corpus() -> None:
    """Every pinned index must still produce its recorded prompt, and a
    wrong-but-valid completion must still score 0 by actually running
    through case execution. The corpus carries no reference program, so
    there is no reference reward to pin here — only the prompt hash and the
    wrong-answer floor.

    Not offline: `task()` and `grade()` both resolve through `get_problem`,
    which issues an HTTP range read for any row not already cached (see
    `VirtualParquetDataset.get_row`). Verified directly: with an unroutable
    `HF_ENDPOINT` and `HF_HUB_OFFLINE=1` this raises `PromptSourceUnavailable`
    rather than passing from a local cache alone."""
    lines = (
        PACKAGE_ROOT / "goldens" / "reference.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 20

    environment = CodeEnvironment()
    for line in lines:
        golden = json.loads(line)
        task = environment.task(golden["index"])
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        assert (
            environment.grade(golden["index"], WRONG_BUT_VALID_COMPLETION)["reward"]
            == golden["wrong_reward"]
        )


def test_artifact_manifest_hashes_installed_files() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from tools.build_artifact import build_artifact

    committed = json.loads((PACKAGE_ROOT / "artifact.json").read_text())
    regenerated = build_artifact(
        PACKAGE_ROOT,
        environment=committed["environment"],
        contract=committed["contract"],
        distribution=committed["distribution"],
        entrypoints=committed["entrypoints"],
    )
    assert regenerated == committed


def test_environment_toml_declares_the_real_pins() -> None:
    import tomllib

    declared = tomllib.loads(
        (PACKAGE_ROOT.parent / "environment.toml").read_text(encoding="utf-8")
    )
    assert declared["id"] == "reliquary/code"
    assert declared["entrypoint"] == "reliquary_code:CodeTaskset"
    assert declared["compatibility_entrypoint"] == "reliquary_code:CodeEnvironment"
    assert declared["provenance"]["port"] == "generator-identical-new-identity"
    assert declared["data"]["license"] == "cc-by-4.0"
    assert corpus.OCI_REVISION in declared["data"]["train"]


def test_repository_compatibility_declares_this_package() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[4]
    declared = tomllib.loads((root / "compatibility.toml").read_text())
    assert "reliquary_code" in declared["releases"]
    assert declared["releases"]["reliquary_code"]["tag"] == "v0.1.0a2"
