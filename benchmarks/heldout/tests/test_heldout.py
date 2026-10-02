import asyncio
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import gpqa_letter  # noqa: E402
import run  # noqa: E402

BENCHMARKS = ["aime25", "aime26", "bfcl-v3", "gpqa", "ifbench", "livecodebench", "mmlu-pro"]


def runner_args(tmp_path: Path, *extra: str):
    parser_args = ["--model", "m", "--base-url", "http://127.0.0.1:9/v1", "--output-dir", str(tmp_path), *extra]
    captured = {}

    def fake_run(command, cwd):
        captured.setdefault("commands", []).append(command)
        return SimpleNamespace(returncode=0)

    return parser_args, captured, fake_run


def test_the_suite_is_the_seven_configs():
    assert run.BENCHMARKS == BENCHMARKS


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_every_config_names_its_taskset_and_the_null_harness(benchmark):
    config = tomllib.loads((HERE / "configs" / f"{benchmark}.toml").read_text())
    assert config["env"]["taskset"]["id"] == benchmark
    # The bash harness adds tools a BFCL episode scores as wrong calls, and gives
    # every other benchmark a shell it was not designed to have.
    assert config["env"]["agent"]["harness"]["id"] == "null"
    # The runtime is the runner's to write, so no config can open the network.
    assert "runtime" not in config["env"]["agent"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_every_config_resolves_against_the_installed_taskset(benchmark, tmp_path, monkeypatch):
    monkeypatch.setenv("HELDOUT_API_KEY", "EMPTY")
    args = run.argparse.Namespace(
        model="m",
        base_url="http://127.0.0.1:9/v1",
        api_key_var="HELDOUT_API_KEY",
        temperature=0.6,
        top_p=0.95,
        max_concurrent=4,
        num_tasks=None,
        output_dir=tmp_path,
    )
    command = run.eval_command(args, benchmark, run.write_config(args, benchmark)) + ["--dry-run", "true"]
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    resolved = json.loads((tmp_path / benchmark / "configs" / "resolved" / "eval.json").read_text())
    assert resolved["push"] is False
    assert resolved["env"]["agent"]["runtime"]["type"] == "docker"
    assert resolved["env"]["agent"]["runtime"]["allow"] == []
    assert resolved["env"]["agent"]["harness"]["id"] == "null"
    assert resolved["client"]["base_url"] == "http://127.0.0.1:9/v1"
    if benchmark == "gpqa":
        assert resolved["env"]["taskset"]["task"]["rewards"]["correct"]["fn"] == "gpqa_letter.py:correct"


def test_the_runner_never_uploads_and_always_closes_the_network(tmp_path, monkeypatch):
    argv, captured, fake_run = runner_args(tmp_path, "--num-tasks", "2")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run.main(argv) == 0
    assert len(captured["commands"]) == len(BENCHMARKS)
    for command in captured["commands"]:
        assert "--no-push" in command
        config = tomllib.loads(Path(command[2]).read_text())
        assert config["env"]["agent"]["runtime"] == {"type": "docker", "allow": []}


def test_summary_sums_weighted_rewards_and_counts_failed_episodes(tmp_path):
    def episode(ok, score):
        rewards = {"correct": {"score": score, "weight": 1.0}} if ok else {}
        return {"ok": ok, "traces": [{"rewards": rewards}]}

    run_dir = tmp_path / "aime26"
    run_dir.mkdir()
    lines = [episode(True, 1.0), episode(True, 0.0), episode(True, 1.0), episode(False, 0.0)]
    (run_dir / "traces.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert run.summarize(run_dir) == {"rollouts": 4, "reward": 2 / 3, "errors": 1}


def test_summary_of_a_run_that_wrote_nothing():
    assert run.summarize(Path("/nonexistent")) == {"rollouts": 0, "reward": None, "errors": None}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("The answer is (C).", 1.0),
        ("Answer: C", 1.0),
        ("The answer is (B).", 0.0),
        ("I cannot decide between the options.", 0.0),
        ("", 0.0),
    ],
)
def test_gpqa_scores_the_extracted_letter_without_a_judge(reply, expected):
    task = SimpleNamespace(answer="C")
    trace = SimpleNamespace(last_reply=reply)
    assert asyncio.run(gpqa_letter.correct(task, trace)) == expected
