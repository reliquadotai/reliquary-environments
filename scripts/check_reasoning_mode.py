#!/usr/bin/env python3
"""Check that this environment says how its policy should be prompted.

An environment knows whether its task rewards deliberation or directness; the
harness running it does not. Leaving the choice to a harness-wide default is
how a format environment ends up rewarding a long chain of thought that talks
itself into satisfying the constraint, and how a maths environment ends up
answering from the hip. Declaring it here makes the decision reviewable in the
same commit as the task, instead of living in whatever config a run happened
to use.

Run from an environment directory.
"""

import json
import sys
import tomllib
from pathlib import Path

MODES = ("thinking", "direct")


def released(descriptor: Path) -> bool:
    """Whether a reviewed release pin binds this descriptor's bytes.

    `build_logic_release.py` binds `environment.toml` into the published
    artifact, so adding a field to a released environment is an act of release
    rather than an ordinary commit. Those declare their mode when they are next
    cut, not by breaking their own pin here.
    """
    root = Path(__file__).resolve().parents[1]
    releases = tomllib.loads((root / "compatibility.toml").read_text()).get("releases", {})
    package = tomllib.loads(descriptor.read_text())["taskset"].replace("-", "_")
    return package in releases


def main() -> None:
    descriptor = Path("environment.toml")
    if not descriptor.is_file():
        raise SystemExit("run this from an environment directory")
    if released(descriptor):
        print(json.dumps({"reasoning": None, "released": True}))
        return
    policy = tomllib.loads(descriptor.read_text()).get("policy") or {}
    reasoning = policy.get("reasoning")
    if reasoning not in MODES:
        raise SystemExit(
            f"environment.toml must declare [policy] reasoning as one of {MODES}, "
            f"found {reasoning!r}"
        )
    if not (policy.get("reasoning_rationale") or "").strip():
        raise SystemExit(
            "[policy] reasoning_rationale must say why, so the next person to "
            "change it knows what it was weighed against"
        )
    print(json.dumps({"reasoning": reasoning}))


if __name__ == "__main__":
    sys.exit(main())
