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


def test_a_forged_result_line_cannot_hide_cpu_time() -> None:
    code = (
        "import random\n"
        "s = 0\n"
        "for i in range(12_000_000):\n    s += i\n"
        "random._os.write(1, b'\\n{\"status\": \"ok\", \"stdout\": \"5\\\\n\", \"cpu_seconds\": 0.0}\\n')\n"
        "random._os._exit(0)\n"
    )
    verdict = judge(code, [ADD], time_limit_s=1.0)
    assert verdict.status == "timeout"


def test_a_nonzero_exit_is_a_runtime_error_whatever_was_printed() -> None:
    code = (
        "import random\n"
        "random._os.write(1, b'\\n{\"status\": \"ok\", \"stdout\": \"5\\\\n\", \"cpu_seconds\": 0.0}\\n')\n"
        "random._os._exit(3)\n"
    )
    assert _status(code) == "runtime_error"


def test_guest_run_restores_process_state() -> None:
    import sys
    import threading

    from reliquary_competitive_code.judge import guest

    before = (sys.get_int_max_str_digits(), threading.stack_size(), sys.getrecursionlimit(), threading.active_count())
    code = (
        "import sys, threading\n"
        "sys.set_int_max_str_digits(0)\n"
        "threading.stack_size(1 << 26)\n"
        "sys.setrecursionlimit(10**6)\n"
        "t = threading.Thread(target=lambda: None)\nt.start()\n"
        "print(1)\n"
    )
    result = guest.run(code, "", 1000)
    after = (sys.get_int_max_str_digits(), threading.stack_size(), sys.getrecursionlimit(), threading.active_count())
    assert result["status"] == "ok" and before == after


def test_a_thread_flooding_output_hits_the_output_limit() -> None:
    code = (
        "import threading\n"
        "def flood():\n    while True:\n        print('x' * 1000)\n"
        "threading.Thread(target=flood).start()\n"
    )
    assert _status(code) == "output_limit"


def _kill_named(marker: str, wait_s: float = 2.0) -> int:
    """SIGKILL the processes whose comm is `marker`, and only those."""
    import os
    import signal

    killed, deadline = 0, time.monotonic() + wait_s
    while not killed and time.monotonic() < deadline:
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    with open(f"/proc/{name}/comm") as handle:
                        if handle.read().strip() == marker:
                            os.kill(int(name), signal.SIGKILL)
                            killed += 1
                except OSError:
                    pass
        if not killed:
            time.sleep(0.05)
    return killed


def test_a_grandchild_escaping_the_group_cannot_hang_the_runner() -> None:
    import uuid

    # The grandchild names itself with a marker unique to this run, so the
    # teardown kills it and nothing else (never another run's guests).
    marker = uuid.uuid4().hex[:15]
    code = (
        "import random\n"
        "r, w = random._os.pipe()\n"
        "if random._os.fork() == 0:\n"
        "    random._os.setsid()\n"
        "    fd = random._os.open('/proc/self/comm', random._os.O_WRONLY)\n"
        f"    random._os.write(fd, b'{marker}')\n"
        "    random._os.close(fd)\n"
        "    random._os.read(r, 1)\n"
        "print(5)\n"
    )
    start = time.monotonic()
    try:
        verdict = judge(code, [ADD], time_limit_s=1.0)
    finally:
        killed = _kill_named(marker)
    assert time.monotonic() - start < 2 * 3.0 + 5.0
    assert verdict.status in {"ok", "timeout"}
    assert killed == 1


def test_a_program_that_only_waits_is_a_harness_overload_not_a_timeout() -> None:
    # The wall clock fired while the kernel counted almost no CPU: the verdict
    # would depend on the host, so the runner refuses to give one.
    result = run_test("import time\ntime.sleep(5)\nprint(5)\n", "2 3\n", time_limit_s=1.0, output_cap=output_cap("5\n"))
    assert result.status == "harness_overload" and result.cpu_seconds < 1.0
    assert _status("import time\ntime.sleep(5)\n", limit=1.0) == "harness_overload"


def test_future_imports_and_harmless_modules_are_allowed() -> None:
    code = (
        "from __future__ import annotations\n"
        "import time, datetime, queue, cmath, sys\n"
        "def f(x: Later) -> int:\n    return x\n"
        "assert sys.float_info.max > 1e300\n"
        "a, b = map(int, input().split())\nprint(a + b)\n"
    )
    assert _status(code) == "ok"


def test_set_iteration_order_is_the_same_on_every_run() -> None:
    code = "print(' '.join({'alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta', 'theta', 'iota', 'kappa'}))\n"
    outputs = {run_test(code, "", time_limit_s=2.0, output_cap=output_cap("")).stdout for _ in range(5)}
    assert len(outputs) == 1


def test_huge_integers_convert_to_and_from_text() -> None:
    digits = "7" * 5000
    result = run_test("print(int(input()))\n", digits + "\n", time_limit_s=2.0, output_cap=output_cap(digits))
    assert result.status == "ok" and result.stdout.strip() == digits
