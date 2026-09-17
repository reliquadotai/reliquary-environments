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


def main() -> None:
    descriptor = Path("environment.toml")
    if not descriptor.is_file():
        raise SystemExit("run this from an environment directory")
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
