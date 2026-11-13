"""Write `artifact.json`, which pins the digest of every file the wheel ships.

Run after any change to the package's files (`environment.toml` is bound
through `source_manifest_sha256`):

    python scripts/write_artifact.py

Prints the `environment_manifest_sha256` a Reliquary registry spec pins. No
golden episodes are shipped: the tests pin a digest of the probe episodes
instead (`test_probe_episodes_match_their_pin`).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_automationbench"
sys.path.insert(0, str(ROOT))

from reliquary_automationbench.environment import ENVIRONMENT  # noqa: E402


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != "artifact.json"
    )
    manifest = {
        "schema": "reliquary/environment-artifact/v1",
        "environment": ENVIRONMENT,
        "contract": "reliquary/episode-json/v1",
        "distribution": {"name": "reliquary-automationbench", "version": version},
        "entrypoints": {
            "taskset": "reliquary_automationbench:AutomationBenchTaskset",
            "replay": "reliquary_automationbench:AutomationBenchEnvironment",
        },
        "source_manifest_sha256": hashlib.sha256((ROOT / "environment.toml").read_bytes()).hexdigest(),
        "files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (PACKAGE / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    print(f"{len(files)} files pinned")
    print(f"environment_manifest_sha256 {hashlib.sha256(canonical.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
