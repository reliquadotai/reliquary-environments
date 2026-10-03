"""Write the packaged goldens and the artifact manifest that pins them.

Run after any change to the package's files: the goldens freeze one task per
split and four completions per task, and `artifact.json` pins the digest of
every file the wheel ships, `environment.toml` included through
`source_manifest_sha256`.

    uv run python scripts/write_goldens.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reliquary_science.corpus import SPLITS
from reliquary_science.environment import ENVIRONMENT, ScienceEnvironment, _build_task

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_science"
# Strided across each split rather than the first task, so that the goldens do
# not all sit at index 0 where an off-by-one would go unseen.
GOLDEN_INDEX = {"train": 16, "eval": 5, "qualification": 11}


def golden(split: str) -> dict[str, object]:
    index = GOLDEN_INDEX[split]
    environment = ScienceEnvironment(split)
    task = _build_task(index, split)
    answer = task["private"]["reference_answer"]
    completion = environment.reference_completion(index)
    # Rewritten the way a model writes it: an intermediate box first, then the
    # answer in scientific notation behind a relation, with a unit after it.
    mantissa, exponent = f"{answer:.6e}".split("e")
    narrated = (
        f"An intermediate quantity comes out as \\boxed{{{answer * 3!r}}}, which "
        "is not what the problem asks for. Finishing the derivation:\n\n"
        f"\\boxed{{x \\approx {mantissa} \\times 10^{{{int(exponent)}}}"
        " \\text{ units}}"
    )
    wrong = answer * 1.05 if answer else 1.0
    return {
        "answer": answer,
        "completion": completion,
        "corpus_problem": task["metadata"]["corpus_problem"],
        "index": index,
        "narrated_completion": narrated,
        "prompt_sha256": hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest(),
        "split": split,
        "state_digest": environment.grade(index, completion)["state_digest"],
        "task_id": task["id"],
        "unboxed_completion": f"The answer is {answer!r}.",
        "wrong_completion": f"\\boxed{{{wrong!r}}}",
    }


def main() -> None:
    goldens = [golden(split) for split in SPLITS]
    (PACKAGE / "goldens").mkdir(exist_ok=True)
    (PACKAGE / "goldens/reference.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in goldens),
        encoding="utf-8",
    )
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
        "contract": "reliquary/boxed-answer/v1",
        "distribution": {"name": "reliquary-science", "version": "0.1.0a1"},
        "entrypoints": {
            "taskset": "reliquary_science:ScienceTaskset",
            "replay": "reliquary_science:ScienceEnvironment",
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
