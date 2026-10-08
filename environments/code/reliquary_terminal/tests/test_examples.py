"""Guards the shipped Prime-RL example. No Docker, no network."""

from __future__ import annotations

import tomllib
from pathlib import Path

_EXAMPLES = Path(__file__).parent.parent / "examples" / "prime_rl"


def test_rl_toml_ships_with_no_train_source():
    run = tomllib.loads((_EXAMPLES / "rl.toml").read_text())
    assert "source" not in run["orchestrator"]["train"]


def test_rl_toml_evaluates_on_terminal_bench():
    run = tomllib.loads((_EXAMPLES / "rl.toml").read_text())
    splits = {s["env"]["taskset"]["split"] for s in run["orchestrator"]["eval"]["source"]}
    assert splits == {"eval"}


def test_train_sources_never_name_the_evaluation_split():
    sources = tomllib.loads((_EXAMPLES / "train-sources.toml").read_text())["source"]
    assert sources
    # Never `eval` (held out), `tmax_sft` (the SFT corpus job's part) or
    # `tmax` (both parts).
    assert {s["env"]["taskset"]["split"] for s in sources} == {"train", "tmax_rl"}


def test_every_source_runs_the_harness_with_a_command_timeout():
    from reliquary_terminal.harness import DEFAULT_COMMAND_TIMEOUT_SECONDS

    descriptor = tomllib.loads((_EXAMPLES.parent.parent / "environment.toml").read_text())
    assert descriptor["execution"]["command_timeout_seconds"] == DEFAULT_COMMAND_TIMEOUT_SECONDS
    run = tomllib.loads((_EXAMPLES / "rl.toml").read_text())
    sources = run["orchestrator"]["eval"]["source"]
    sources += tomllib.loads((_EXAMPLES / "train-sources.toml").read_text())["source"]
    for source in sources:
        h = source["env"]["agent"]["harness"]
        assert h["id"] == "reliquary-terminal"
        assert h["command_timeout"] == DEFAULT_COMMAND_TIMEOUT_SECONDS
