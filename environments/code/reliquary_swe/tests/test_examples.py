"""Guards the shipped Prime-RL example, not just its README.

Needs no Docker and no network: this reads a checked-in file with `tomllib`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_RL_TOML = Path(__file__).parent.parent / "examples" / "prime_rl" / "rl.toml"


def test_rl_toml_ships_with_no_train_source():
    # SWE-bench Verified is this taskset's only split and an evaluation set
    # that must never be trained on (environment.toml's [data] table
    # declares only `eval`; design spec section 8). The actual guard against
    # that lives in prime-rl -- `TrainSource.__init__` raises with no train
    # envs, verified against the exact pinned commit in
    # examples/prime_rl/README.md -- not in this repository, and nothing
    # here re-runs it. Without this test, a future edit re-adding
    # `[[orchestrator.train.source]]` to rl.toml would silently undo that
    # whole fix: `cp rl.toml x.toml && rl @ x.toml --max-steps 5000` would
    # train on an evaluation set for real, with zero coverage catching it.
    run = tomllib.loads(_RL_TOML.read_text())
    assert "source" not in run["orchestrator"]["train"]
