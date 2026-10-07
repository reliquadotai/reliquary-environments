"""Guards the shipped Prime-RL example, not just its README.

Needs no Docker and no network: this reads a checked-in file with `tomllib`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_RL_TOML = Path(__file__).parent.parent / "examples" / "prime_rl" / "rl.toml"


def test_rl_toml_ships_with_no_train_source():
    # SWE-bench Verified (this taskset's `eval` split) is an evaluation set
    # that must never be trained on (design spec section 8). The actual
    # guard against that lives in prime-rl -- `TrainSource.__init__` raises
    # with no train envs, verified against the exact pinned commit in
    # examples/prime_rl/README.md -- not in this repository, and nothing
    # here re-runs it. Without this test, a future edit re-adding
    # `[[orchestrator.train.source]]` to rl.toml would silently undo that
    # whole fix: `cp rl.toml x.toml && rl @ x.toml --max-steps 5000` would
    # train on an evaluation set for real, with zero coverage catching it.
    run = tomllib.loads(_RL_TOML.read_text())
    assert "source" not in run["orchestrator"]["train"]


_TRAIN_SOURCES = _RL_TOML.parent / "train-sources.toml"


def test_train_sources_never_name_the_evaluation_split():
    # The one file meant to be passed for a real run. Every source must
    # declare its split explicitly (the taskset has no default, on purpose)
    # and none may be SWE-bench Verified.
    sources = tomllib.loads(_TRAIN_SOURCES.read_text())["source"]
    assert sources
    for source in sources:
        assert source["env"]["taskset"]["split"] in ("train", "polyglot", "r2e"), source["name"]


def test_train_sources_bound_the_polyglot_disk_budget():
    # One image per polyglot or R2E task: an unbounded source would try to
    # pull the whole corpus, terabytes, on the first pass through it.
    for source in tomllib.loads(_TRAIN_SOURCES.read_text())["source"]:
        if source["env"]["taskset"]["split"] in ("polyglot", "r2e"):
            assert isinstance(source["env"]["taskset"].get("num_tasks"), int)
