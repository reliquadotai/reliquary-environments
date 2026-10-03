"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. The corpus is fetched
on first use, so this also checks that a fresh install derives the pinned
corpus from the pinned parquet.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from reliquary_science import VIRTUAL_LENGTH, ScienceEnvironment
from reliquary_science.corpus import SPLITS
from reliquary_science.grading import ANSWER_INSTRUCTION

QUALIFICATION_TASKS = 1357

root = importlib.resources.files("reliquary_science")

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
    environment = ScienceEnvironment(golden["split"])
    for field, expected in (
        ("completion", 1.0),
        ("narrated_completion", 1.0),
        ("wrong_completion", 0.0),
        ("unboxed_completion", 0.0),
    ):
        reward = environment.grade(golden["index"], golden[field])["reward"]
        assert reward == expected, f"{golden['split']}: {field} scored {reward}"

environment = ScienceEnvironment("qualification")
assert len(environment) == QUALIFICATION_TASKS, len(environment)

prompts = {
    ScienceEnvironment(split).task(index)["prompt"]
    for split in SPLITS
    for index in range(len(ScienceEnvironment(split)))
}
assert len(prompts) == VIRTUAL_LENGTH, f"{len(prompts)} distinct prompts"
assert all(prompt.endswith(ANSWER_INSTRUCTION) for prompt in prompts)

print("wheel verified")
