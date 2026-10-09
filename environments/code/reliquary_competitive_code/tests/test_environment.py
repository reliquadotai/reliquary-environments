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


def test_an_empty_split_loads(tmp_path) -> None:
    pids = [c for c in _curated(300) if c.split == "train"]
    write_dataset(pids, tmp_path)
    loaded = Corpus(tmp_path)
    assert loaded.problems("eval") == [] and loaded.problems("qualification") == []
    assert len(loaded.problems("train")) == len(pids)
    with pytest.raises(ValueError, match="eval split is empty"):
        CompetitiveCodeEnvironment("eval", corpus=loaded).task(0)
    with pytest.raises(ValueError, match="eval split is empty"):
        CompetitiveCodeEnvironment("eval", corpus=loaded).grade(0, "")


def test_a_problem_without_tests_is_an_error_not_a_reward(corpus, monkeypatch) -> None:
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    monkeypatch.setattr(corpus, "tests", lambda split, index: ())
    with pytest.raises(RuntimeError, match="has no tests"):
        environment.grade(0, environment.reference_completion(0))


def test_row_groups_are_cached_and_bounded(corpus) -> None:
    corpus._cache.clear()
    for index in range(0, 300, 64):
        corpus.tests("train", index % len(corpus.problems("train")))
    first = corpus._row_group("train", 0)
    assert corpus._row_group("train", 0) is first
    assert len(corpus._cache) <= 4


def test_a_pin_that_misses_a_file_is_refused_before_any_download(monkeypatch) -> None:
    from reliquary_competitive_code import corpus as module

    pin = module.DatasetPin("o/r", "rev", {"references.parquet": "00"})
    monkeypatch.setattr(module, "PINNED", pin)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)  # any import would fail
    with pytest.raises(RuntimeError, match="the pin must name exactly"):
        Corpus.pinned()


def test_serving_does_not_import_the_build() -> None:
    import subprocess
    import sys

    code = (
        "import sys, reliquary_competitive_code.environment;"
        "bad = [m for m in sys.modules if m.startswith('reliquary_competitive_code.build')];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_harness_overload_is_never_a_reward(corpus, monkeypatch) -> None:
    from reliquary_competitive_code.judge import runner

    # Starved by the host: runnable far longer than the limit, never run.
    monkeypatch.setattr(runner, "_run_queue_wait_s", lambda pid: 60.0)
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    with pytest.raises(RuntimeError, match="harness_overload"):
        environment.grade(5, "```python\nimport time\ntime.sleep(5)\n```")


def test_an_idle_program_scores_zero(corpus) -> None:
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    graded = environment.grade(5, "```python\nimport time\ntime.sleep(5)\n```")
    assert graded["reward"] == 0.0 and graded["status"] == "timeout"


def test_the_instruction_lists_the_allowed_modules() -> None:
    from reliquary_competitive_code.judge.guest import ALLOWED_IMPORT_ROOTS

    for module in ALLOWED_IMPORT_ROOTS - {"__future__"}:
        assert f"{module}," in INSTRUCTION or f"{module}." in INSTRUCTION or f"{module};" in INSTRUCTION
    assert "__future__" not in INSTRUCTION
    assert "the os and io modules" in INSTRUCTION and "file access" in INSTRUCTION


def test_the_pinned_corpus_loads_once_under_concurrent_first_use(monkeypatch, corpus) -> None:
    import threading
    import time

    from reliquary_competitive_code import corpus as module

    calls = []

    def slow_pinned():
        calls.append(1)
        time.sleep(0.2)
        return corpus

    monkeypatch.setattr(module, "_pinned", None)
    monkeypatch.setattr(module.Corpus, "pinned", staticmethod(slow_pinned))
    results = []
    threads = [threading.Thread(target=lambda: results.append(module.pinned_corpus())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1 and all(r is corpus for r in results) and len(results) == 8


def test_train_splits_into_one_sft_range_then_one_rl_range(tmp_path) -> None:
    from reliquary_competitive_code.layout import use_of

    # Ids spread over the whole 64-bit space, so both uses are present.
    spread = [
        Curated(f"{i * (2**64 // 40):016x}", "train", f"Problem {i}.", (TestCase("1\n", "1\n"),) * 5, 1.0, SUM, {})
        for i in range(40)
    ]
    write_dataset(spread, tmp_path)
    corpus = Corpus(tmp_path)
    train = corpus.problems("train")
    sft, rl = corpus.use_range("sft"), corpus.use_range("rl")
    assert sft.start == 0 and sft.stop == rl.start and rl.stop == len(train)
    assert 0 < len(sft) and 0 < len(rl)
    assert all(use_of(train[i].problem_id) == "sft" for i in sft)
    assert all(use_of(train[i].problem_id) == "rl" for i in rl)
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    assert environment.task(sft.start)["metadata"]["use"] == "sft"
    assert environment.task(rl.start)["metadata"]["use"] == "rl"
    assert len(sft) == 25
    with pytest.raises(ValueError):
        corpus.use_range("train")


def test_the_sft_share_is_sixty_percent_of_the_id_space() -> None:
    from reliquary_competitive_code.layout import use_of

    assert use_of("0" * 16) == "sft" and use_of("f" * 16) == "rl"
    assert use_of("9999999999999998") == "sft" and use_of("999999999999999a") == "rl"


def test_admission_reward_cases_hand_over_the_tests_without_running_them(corpus, monkeypatch) -> None:
    import reliquary_competitive_code.environment as module
    from reliquary_competitive_code.judge import output_cap

    def refuse(*args, **kwargs):
        raise AssertionError("handing over the tests must not run a program")

    monkeypatch.setattr(module, "judge", refuse)
    environment = CompetitiveCodeEnvironment("train", corpus=corpus)
    position = 7 + len(environment)  # wraps like every other accessor
    problem = corpus.problems("train")[7]
    cases = environment.admission_reward_cases(position)
    assert cases == [
        {"stdin": t.stdin, "stdout": t.stdout, "time_limit_s": problem.time_limit_s,
         "output_cap": output_cap(t.stdout)}
        for t in corpus.tests("train", 7)
    ]
    cases[0]["stdout"] = "tampered"
    assert environment.admission_reward_cases(7)[0]["stdout"] != "tampered"
