"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. The dataset is fetched
on first use, so this also checks that a fresh install serves the pinned
revision, and that its split sizes add up to the `[data].virtual_length` that
`environment.toml` declares.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import tomllib
from pathlib import Path

from reliquary_competitive_code.layout import SPLITS
from reliquary_competitive_code.environment import INSTRUCTION, CompetitiveCodeEnvironment

root = importlib.resources.files("reliquary_competitive_code")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

goldens = [
    json.loads(line)
    for line in root.joinpath("goldens/reference.jsonl").read_text().splitlines()
]
assert len(goldens) == len(SPLITS), f"{len(goldens)} goldens, expected one per split"

for golden in goldens:
    environment = CompetitiveCodeEnvironment(golden["split"])
    task = environment.task(golden["index"])
    assert task["id"] == golden["task_id"], golden["split"]
    prompt_digest = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
    assert prompt_digest == golden["prompt_sha256"], golden["split"]
    for field, expected in (
        ("completion", 1.0),
        ("wrong_completion", 0.0),
        ("no_code_completion", 0.0),
    ):
        reward = environment.grade(golden["index"], golden[field])["reward"]
        assert reward == expected, f"{golden['split']}: {field} scored {reward}"

# environment.toml sits beside the package in the source tree (the CI job runs
# this script by absolute path from the checkout) and is pinned by
# `source_manifest_sha256`; `[data].virtual_length` is the number of tasks served.
declared = Path(__file__).resolve().parents[1] / "environment.toml"
assert (
    hashlib.sha256(declared.read_bytes()).hexdigest()
    == artifact["source_manifest_sha256"]
), "environment.toml differs from the pinned manifest"
data = tomllib.loads(declared.read_text())["data"]
served = sum(len(CompetitiveCodeEnvironment(split)) for split in SPLITS)
assert served == data["virtual_length"], f"{served} tasks served, declared {data['virtual_length']}"

prompts = {
    CompetitiveCodeEnvironment(split).task(index)["prompt"]
    for split in SPLITS
    for index in range(len(CompetitiveCodeEnvironment(split)))
}
assert all(prompt.endswith(INSTRUCTION) for prompt in prompts)

print("wheel verified")
