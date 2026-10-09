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
    graded = _grade(environment, index, completion)
    assert graded["reward"] == 1.0, f"{split}: the reference scored {graded}"
    assert _grade(environment, index, WRONG)["reward"] == 0.0
    assert _grade(environment, index, NO_CODE)["reward"] == 0.0
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


# Discrimination goldens: ~30 real problems strided across the splits, each
# frozen with four completions and the status the judge must give them.
DISCRIMINATION_PER_SPLIT = {"train": 18, "eval": 6, "qualification": 6}
LOOP = "```python\nwhile True:\n    pass\n```"
FORBIDDEN = "```python\nimport os\nprint(os.getpid())\n```"


def _wrong(reference: str) -> str:
    """The reference with one extra token in its output. Printed first, so a
    reference that exits early cannot skip it; last only when the reference
    starts with a `__future__` import, which must stay first."""
    extra = "print('reliquary-wrong')"
    body = f"{reference}\n{extra}" if "__future__" in reference else f"{extra}\n{reference}"
    return f"```python\n{body}\n```"


def _grade(environment, index: int, completion: str) -> dict[str, object]:
    """Grade, retrying when a starved host gave no verdict (harness_overload
    raises; it never becomes a reward)."""
    for attempt in range(6):
        try:
            return environment.grade(index, completion)
        except RuntimeError as error:
            if "harness_overload" not in str(error) or attempt == 5:
                raise
    raise AssertionError("unreachable")


def discrimination(split: str, count: int) -> list[dict[str, object]]:
    environment = CompetitiveCodeEnvironment(split)
    stride = max(1, len(environment) // count)
    items = []
    for index in range(stride // 2, len(environment), stride):
        if len(items) == count:
            break
        task = environment.task(index)
        reference = environment._corpus.reference(task["metadata"]["problem_id"])
        cases = {
            "reference": (f"```python\n{reference}\n```", 1.0, "ok"),
            "wrong": (_wrong(reference), 0.0, "wrong_answer"),
            "loop": (LOOP, 0.0, "timeout"),
            "forbidden": (FORBIDDEN, 0.0, "forbidden_import"),
        }
        graded = {name: _grade(environment, index, completion) for name, (completion, _, _) in cases.items()}
        if any((graded[n]["reward"], graded[n]["status"]) != (r, st) for n, (_, r, st) in cases.items()):
            print(f"skip {split}#{index}: {[(n, g['status']) for n, g in graded.items()]}")
            continue
        items.append({
            "split": split,
            "index": index,
            "problem_id": task["metadata"]["problem_id"],
            "task_id": task["id"],
            "cases": [
                {"name": n, "completion": c, "reward": r, "status": st}
                for n, (c, r, st) in cases.items()
            ],
        })
    assert len(items) == count, f"{split}: {len(items)} discrimination goldens, wanted {count}"
    return items


def main() -> None:
    goldens = [golden(split) for split in SPLITS]
    (PACKAGE / "goldens").mkdir(exist_ok=True)
    (PACKAGE / "goldens/reference.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in goldens),
        encoding="utf-8",
    )
    discriminating = [
        item for split in SPLITS for item in discrimination(split, DISCRIMINATION_PER_SPLIT[split])
    ]
    (PACKAGE / "goldens/discrimination.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in discriminating),
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
    print(json.dumps({"goldens": len(goldens), "discrimination": len(discriminating), "files": len(files)}))


if __name__ == "__main__":
    main()
