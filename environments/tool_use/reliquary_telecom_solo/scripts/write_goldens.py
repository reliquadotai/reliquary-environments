"""Write the packaged goldens and the artifact manifest that pins them.

Run after any change to the package's files: the goldens freeze one reference
episode per family in every split, plus the first action-scored ticket of a
split when its family pick was not one, and `artifact.json` pins the digest of
every file the wheel ships, `environment.toml` included through
`source_manifest_sha256`.

    uv run python scripts/write_goldens.py

The picks are the first row of each family in the split's own order, so the
choice is reproducible and moves only when the layout does.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from reliquary_telecom_solo.corpus import SPLITS, rows
from reliquary_telecom_solo.environment import ENVIRONMENT, TelecomSoloEnvironment

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_telecom_solo"
FAMILIES = ("mms_issue", "mobile_data_issue", "service_issue")


def _goldens() -> list[dict]:
    goldens = []
    for split in SPLITS:
        environment = TelecomSoloEnvironment(split)
        pool = rows(split)
        picks = [next(i for i, task in enumerate(pool) if task.family == family) for family in FAMILIES]
        scored = next((i for i, task in enumerate(pool) if "ACTION" in task.reward_basis), None)
        if scored is not None and scored not in picks:
            picks.append(scored)
        for index in picks:
            actions = environment.reference_actions(index)
            episode = environment.replay(index, actions)
            if episode["reward"]["reward"] != 1.0:
                raise RuntimeError(f"{split}[{index}]: the reference solution does not score")
            goldens.append(
                {
                    "actions": actions,
                    "family": pool[index].family,
                    "index": index,
                    "reward": episode["reward"]["reward"],
                    "reward_basis": list(pool[index].reward_basis),
                    "split": split,
                    "state_digest": episode["reward"]["state_digest"],
                    "task_id": episode["task"]["id"],
                    "termination_reason": episode["termination_reason"],
                    "transcript_digest": episode["transcript_digest"],
                }
            )
    return goldens


def main() -> None:
    goldens = _goldens()
    (PACKAGE / "goldens" / "reference.jsonl").write_text(
        "".join(json.dumps(golden, sort_keys=True) + "\n" for golden in goldens)
    )
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
        "distribution": {"name": "reliquary-telecom-solo", "version": version},
        "entrypoints": {
            "taskset": "reliquary_telecom_solo:TelecomSoloTaskset",
            "replay": "reliquary_telecom_solo:TelecomSoloEnvironment",
        },
        "source_manifest_sha256": hashlib.sha256((ROOT / "environment.toml").read_bytes()).hexdigest(),
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    (PACKAGE / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    print(f"{len(goldens)} goldens, {len(files)} files pinned")
    print(f"environment_manifest_sha256 {hashlib.sha256(canonical.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
