import importlib
import pkgutil
import time
from pathlib import Path

import pytest

from reliquary_code import corpus
from reliquary_code.extraction import extract_python
from reliquary_code.runner import run_cases


def test_no_module_imports_reliquary_core() -> None:
    import reliquary_code

    for info in pkgutil.walk_packages(
        reliquary_code.__path__, prefix="reliquary_code."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "from reliquary." not in text
        assert "import reliquary." not in text


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OCI_REPO == "R0mAI/opencodeinstruct-curated"
    assert corpus.OCI_REVISION == "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"


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


CASES = [
    {"input": "1\n", "expected_output": "2\n"},
    {"input": "2\n", "expected_output": "3\n"},
]

INCREMENT = "import sys\nprint(int(sys.stdin.read().strip()) + 1)\n"


def test_correct_source_passes_every_case() -> None:
    assert run_cases(INCREMENT, CASES) == [True, True]


def test_wrong_source_fails_every_case() -> None:
    assert run_cases("print(999)\n", CASES) == [False, False]


def test_a_cpu_bomb_is_killed_and_scored_false() -> None:
    assert run_cases("while True:\n    pass\n", CASES[:1], cpu_seconds=1) == [False]


def test_a_sleeping_process_is_killed_by_the_wall_clock() -> None:
    """A sleeper burns no CPU, so RLIMIT_CPU alone would never fire."""
    source = "import time\ntime.sleep(30)\n"
    assert run_cases(source, CASES[:1], cpu_seconds=5, wall_seconds=1.0) == [False]


def test_a_memory_bomb_is_killed_and_scored_false() -> None:
    source = "x = bytearray(2 * 1024 * 1024 * 1024)\n"
    assert run_cases(source, CASES[:1], memory_bytes=64 * 1024 * 1024) == [False]


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
    burning 0.6 CPU-seconds against a 1-CPU-second budget — individually
    legal, but 1.2s combined would blow a *shared* 1s budget. If each case
    gets its own fresh process (its own fresh RLIMIT_CPU accounting), both
    finish comfortably inside the budget and pass. If the two cases shared
    one process, the second case would inherit the first case's CPU debt and
    be SIGKILLed, turning `[True, True]` into `[True, False]`. Timing is
    measured with `time.process_time()` (CPU time, not wall clock) so the
    test is not flaky under a loaded machine.
    """
    source = (
        "import sys, time\n"
        "target = float(sys.stdin.read().strip())\n"
        "start = time.process_time()\n"
        "while time.process_time() - start < target:\n"
        "    pass\n"
        "print('done')\n"
    )
    cases = [
        {"input": "0.6", "expected_output": "done"},
        {"input": "0.6", "expected_output": "done"},
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

