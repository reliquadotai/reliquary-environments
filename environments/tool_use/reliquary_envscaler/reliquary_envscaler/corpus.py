"""Fetch and normalise the published EnvScaler release.

60 MB of worlds and scenarios, too large to ship in a wheel, so the package
names the upstream datasets and pins what it expects to find. Two fields
arrive as JSON *strings* inside the JSON — `tools` on a world and
`init_config` on a scenario — and the environment reads them as containers.
Decoding them belongs here rather than in a script beside the package: it is
not a preprocessing convenience, it is what the release means.

Only the RL scenarios are usable. The 4,684 SFT ones ship no
`checklist_with_func`, so nothing can score an answer to them, and an
environment that cannot grade is not an environment.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

WORLDS_REPO = "XXHStudyHard/EnvScaler-191-Env"
WORLDS_FILE = "191_env_metadata_processed.json"
SCENARIOS_REPO = "XXHStudyHard/EnvScaler-RL-Scenario"
SCENARIOS_FILE = "envscaler_rl_scenario_metadata.json"

# What a correct download contains. A short count is a truncated file, and a
# silently short corpus shifts every task index.
EXPECTED_WORLDS = 191
EXPECTED_SCENARIOS = 2550

DATA_ENV_VAR = "RELIQUARY_ENVSCALER_DATA"


def _decode(value: Any, expect: type) -> Any:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, expect):
        raise TypeError(f"expected {expect.__name__}, got {type(value).__name__}")
    return value


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def _locate(filename: str, repo: str) -> Path:
    """A local directory if one is configured, the Hub otherwise."""
    configured = os.environ.get(DATA_ENV_VAR)
    if configured:
        local = Path(configured) / filename
        if local.exists():
            return local
    return _download(repo, filename)


@lru_cache(maxsize=1)
def load() -> tuple[dict[str, dict], tuple[dict, ...]]:
    """`(worlds by env_id, scenarios)`, decoded and checked."""
    worlds = _read(_locate(WORLDS_FILE, WORLDS_REPO))
    if isinstance(worlds, dict):
        worlds = list(worlds.values())
    scenarios = _read(_locate(SCENARIOS_FILE, SCENARIOS_REPO))
    if isinstance(scenarios, dict):
        scenarios = list(scenarios.values())

    if len(worlds) != EXPECTED_WORLDS or len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError(
            f"EnvScaler release is {len(worlds)} worlds and {len(scenarios)} "
            f"scenarios, expected {EXPECTED_WORLDS} and {EXPECTED_SCENARIOS}"
        )

    for world in worlds:
        world["tools"] = _decode(world.get("tools", []), list)
    for scenario in scenarios:
        scenario["init_config"] = _decode(scenario.get("init_config", {}), dict)

    by_id = {world["env_id"]: world for world in worlds}
    # Order is the corpus identity: scenarios are addressed by position.
    usable = tuple(s for s in scenarios if s["env_id"] in by_id)
    return by_id, usable


__all__ = ["load", "DATA_ENV_VAR", "EXPECTED_WORLDS", "EXPECTED_SCENARIOS"]
