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
