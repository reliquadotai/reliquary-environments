"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. A wheel that builds
but ships the corpus without its digest, or that ships it with the duplicates
back in, is a wheel that would be discovered in a training run rather than in
CI.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from reliquary_dapo_math import VIRTUAL_LENGTH, DapoMathEnvironment
from reliquary_dapo_math.corpus import SPLITS
from reliquary_dapo_math.grading import ANSWER_INSTRUCTION

QUALIFICATION_TASKS = 1595

root = importlib.resources.files("reliquary_dapo_math")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

goldens = [
    json.loads(line)
    for line in root.joinpath("goldens/reference.jsonl").read_text().splitlines()
]
assert len(goldens) == 3, f"{len(goldens)} goldens, expected one per split"

# Each golden's index is relative to its own split, and the four completions
# are what freeze the grader: an accepting one alone would pass on a grader
# that read any boxed number as correct.
for golden in goldens:
    environment = DapoMathEnvironment(golden["split"])
    for field, expected in (
        ("completion", 1.0),
        ("narrated_completion", 1.0),
        ("wrong_completion", 0.0),
        ("unboxed_completion", 0.0),
    ):
        reward = environment.grade(golden["index"], golden[field])["reward"]
        assert reward == expected, f"{golden['split']}: {field} scored {reward}"

environment = DapoMathEnvironment("qualification")
assert len(environment) == QUALIFICATION_TASKS, len(environment)

# The duplication check, on the shipped wheel rather than on the source tree:
# one problem must not be reachable from two indices, in any split.
prompts = {
    DapoMathEnvironment(split).task(index)["prompt"]
    for split in SPLITS
    for index in range(len(DapoMathEnvironment(split)))
}
assert len(prompts) == VIRTUAL_LENGTH, f"{len(prompts)} distinct prompts"
assert all(prompt.endswith(ANSWER_INSTRUCTION) for prompt in prompts)

print("wheel verified")
