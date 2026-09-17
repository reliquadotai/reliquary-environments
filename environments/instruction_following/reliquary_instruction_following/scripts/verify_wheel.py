"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. A wheel that builds
but ships the corpus without its digests, or whose goldens no longer grade the
way they were frozen to, is a wheel that would be discovered in a training run
rather than in CI.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from reliquary_instruction_following import (
    EXCLUDED_INSTRUCTION_IDS,
    InstructionFollowingEnvironment,
)

QUALIFICATION_TASKS = 3594

root = importlib.resources.files("reliquary_instruction_following")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

goldens = [
    json.loads(line)
    for line in root.joinpath("goldens/reference.jsonl").read_text().splitlines()
]
assert len(goldens) == 3, f"{len(goldens)} goldens, expected one per split"

# Each golden's index is relative to its own split, and the pair of answers is
# what freezes the checker: an accepting answer alone would pass on a grader
# that accepted anything.
for golden in goldens:
    environment = InstructionFollowingEnvironment(golden["split"])
    good = environment.grade(golden["index"], golden["answer"])
    assert good["reward"] == 1.0, f"{golden['split']}: golden answer scored {good['reward']}"
    broken = environment.grade(golden["index"], golden["broken_answer"])
    assert broken["reward"] == 0.0, f"{golden['split']}: broken answer scored {broken['reward']}"
    empty = environment.grade(golden["index"], "   ")
    assert empty["reward"] == 0.0, f"{golden['split']}: empty answer scored {empty['reward']}"

environment = InstructionFollowingEnvironment("qualification")
assert len(environment) == QUALIFICATION_TASKS, len(environment)
for index in range(64):
    named = environment.task(index)["metadata"]["instruction_ids"]
    excluded = sorted(set(named) & EXCLUDED_INSTRUCTION_IDS)
    assert not excluded, f"task {index} names excluded verifiers: {excluded}"

print("wheel verified")
