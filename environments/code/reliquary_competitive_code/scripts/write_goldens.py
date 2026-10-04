"""Write the packaged goldens and the artifact manifest that pins them.

Run after any change to the package's files, and only once `corpus.PINNED`
names the published dataset: the goldens freeze one task per split and three
completions per task, and `artifact.json` pins the digest of every file the
wheel ships, `environment.toml` included through `source_manifest_sha256`.

    uv run python scripts/write_goldens.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reliquary_competitive_code.layout import SPLITS
from reliquary_competitive_code.environment import ENVIRONMENT, CompetitiveCodeEnvironment

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_competitive_code"
# Strided across each split rather than the first task, so that the goldens do
# not all sit at index 0 where an off-by-one would go unseen.
GOLDEN_INDEX = {"train": 16, "eval": 5, "qualification": 11}
WRONG = "```python\nprint('reliquary-wrong')\n```"
NO_CODE = "I would use a segment tree."


def golden(split: str) -> dict[str, object]:
    index = GOLDEN_INDEX[split]
    environment = CompetitiveCodeEnvironment(split)
    task = environment.task(index)
    completion = environment.reference_completion(index)
    graded = environment.grade(index, completion)
    assert graded["reward"] == 1.0, f"{split}: the reference scored {graded}"
    assert environment.grade(index, WRONG)["reward"] == 0.0
    assert environment.grade(index, NO_CODE)["reward"] == 0.0
    return {
        "completion": completion,
        "index": index,
        "no_code_completion": NO_CODE,
        "problem_id": task["metadata"]["problem_id"],
        "prompt_sha256": hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest(),
        "split": split,
        "state_digest": graded["state_digest"],
        "task_id": task["id"],
        "wrong_completion": WRONG,
    }


def main() -> None:
    goldens = [golden(split) for split in SPLITS]
    (PACKAGE / "goldens").mkdir(exist_ok=True)
    (PACKAGE / "goldens/reference.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in goldens),
        encoding="utf-8",
    )
    # Every shipped file is pinned, `build/` and `sources/` included.
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file()
        and path.name != "artifact.json"
        and "__pycache__" not in path.parts
    }
    artifact = {
        "schema": "reliquary/environment-artifact/v1",
        "environment": ENVIRONMENT,
        "contract": "reliquary/stdio-program/v1",
        "distribution": {"name": "reliquary-competitive-code", "version": "0.1.0a1"},
        "entrypoints": {
            "taskset": "reliquary_competitive_code:CompetitiveCodeTaskset",
            "replay": "reliquary_competitive_code.environment:CompetitiveCodeEnvironment",
        },
        "source_manifest_sha256": hashlib.sha256(
            (ROOT / "environment.toml").read_bytes()
        ).hexdigest(),
        "files": files,
    }
    (PACKAGE / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"goldens": len(goldens), "files": len(files)}))


if __name__ == "__main__":
    main()
