"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. A wheel that builds
but ships the corpus without its digest, or with a reference its own grader
does not recognise, is a wheel that would be discovered in a training run
rather than in CI.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from reliquary_hard_math import VIRTUAL_LENGTH, HardMathEnvironment
from reliquary_hard_math.corpus import SPLITS, load
from reliquary_hard_math.grading import ANSWER_INSTRUCTION, grade, verifier_spec

SPLIT_SIZES = {"train": 23227, "eval": 754, "qualification": 474}

root = importlib.resources.files("reliquary_hard_math")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

goldens = [
    json.loads(line)
    for line in root.joinpath("goldens/reference.jsonl").read_text().splitlines()
]
assert len(goldens) == 3, f"{len(goldens)} goldens, expected one per split"
for golden in goldens:
    environment = HardMathEnvironment(golden["split"])
    assert environment.task(golden["index"])["id"] == golden["task_id"]
    for field, expected in (
        ("completion", 1.0),
        ("narrated_completion", 1.0),
        ("wrong_completion", 0.0),
        ("unboxed_completion", 0.0),
    ):
        reward = environment.grade(golden["index"], golden[field])["reward"]
        assert reward == expected, f"{golden['split']}: {field} scored {reward}"

for split, size in SPLIT_SIZES.items():
    assert len(HardMathEnvironment(split)) == size, (split, len(HardMathEnvironment(split)))
assert sum(SPLIT_SIZES.values()) == VIRTUAL_LENGTH

# Every reference, on the installed wheel, scores 1.0 against itself.
for problem in load():
    assert grade(verifier_spec(problem.answer), f"\\boxed{{{problem.answer}}}") == 1.0, problem.key

prompts = {
    HardMathEnvironment(split).task(index)["prompt"]
    for split in SPLITS
    for index in range(len(HardMathEnvironment(split)))
}
assert len(prompts) == VIRTUAL_LENGTH, f"{len(prompts)} distinct prompts"
assert all(prompt.endswith(ANSWER_INSTRUCTION) for prompt in prompts)

print("wheel verified")
