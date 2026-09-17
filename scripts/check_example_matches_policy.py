#!/usr/bin/env python3
"""Check that the shipped run example prompts the policy the way the task asks.

import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent))

Two files declare the same intent: `environment.toml` says whether the task
rewards deliberation, and the run example has to ask the renderer for it. They
drift the moment one is edited alone, and the drift is silent — the run simply
trains a mode the environment did not ask for.

Skipped where no example is shipped. Run from an environment directory.
"""

import json
import sys
import tomllib
from pathlib import Path

from check_reasoning_mode import released


def main() -> None:
    descriptor = Path("environment.toml")
    if not descriptor.is_file():
        raise SystemExit("run this from an environment directory")
    if released(descriptor):
        # Its descriptor is bound by a reviewed release pin; it declares a mode
        # when it is next cut.
        print(json.dumps({"released": True}))
        return
    reasoning = tomllib.loads(descriptor.read_text())["policy"]["reasoning"]

    example = Path("examples/prime_rl/rl.toml")
    if not example.is_file():
        print(json.dumps({"example": None, "reasoning": reasoning}))
        return

    run = tomllib.loads(example.read_text())
    renderer = (run.get("orchestrator") or {}).get("renderer") or {}
    if not renderer.get("name"):
        raise SystemExit(
            "the example must name a renderer: auto-resolution falls back to one "
            "with no tool support for models outside the map"
        )
    wanted = reasoning == "thinking"
    if renderer.get("enable_thinking") != wanted:
        raise SystemExit(
            f"environment.toml asks for {reasoning!r} but the example sets "
            f"enable_thinking={renderer.get('enable_thinking')!r}"
        )
    print(json.dumps({"renderer": renderer["name"], "enable_thinking": wanted}))


if __name__ == "__main__":
    sys.exit(main())
