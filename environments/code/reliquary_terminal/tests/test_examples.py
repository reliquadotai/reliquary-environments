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
    assert all(s["env"]["taskset"]["split"] == "train" for s in sources)
