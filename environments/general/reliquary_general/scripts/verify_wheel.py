"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them: every shipped file
matches its pin, no corpus file is shipped, every golden replays, every row of
every split loads from the pinned Hub dataset and its check builds, task
identities are unique, the segment table tiles each split with no gap, and
each segment's single-turn rows are exactly its leading part.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json

from reliquary_general import GeneralPromptsEnvironment, Ungraded
from reliquary_general.corpus import SPLITS, UNGRADED_BLOCKS, length, locate, row, segments
from reliquary_general.environment import _build_task
from reliquary_general.grading import builds

root = importlib.resources.files("reliquary_general")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

assert not any(name.endswith(".jsonl.gz") for name in artifact["files"]), "the wheel ships corpus files"

goldens = [json.loads(line) for line in root.joinpath("goldens/reference.jsonl").read_text().splitlines()]
for golden in goldens:
    environment = GeneralPromptsEnvironment(golden["split"])
    task = environment.task(golden["index"])
    assert task["id"] == golden["task_id"], golden["block"]
    if not golden["graded"]:
        try:
            environment.grade(golden["index"], "anything")
        except Ungraded:
            continue
        raise AssertionError(f"{golden['block']}: ungraded golden scored")
    for field, expected in (("completion", 1.0), ("thinking_completion", 1.0), ("wrong_completion", 0.0)):
        reward = environment.grade(golden["index"], golden[field])["reward"]
        assert reward == expected, f"{golden['block']}: {field} scored {reward}"

ids: set[str] = set()
for split in SPLITS:
    position = 0
    for segment in segments(split):
        assert segment.start == position, (split, segment.name)
        position = segment.stop
    assert position == length(split)
    for index in range(length(split)):
        found = row(index, split)
        task = _build_task(index, split)
        segment, _ = locate(index, split)
        assert found.single_turn == (index < segment.single_turn_stop), (split, index)
        assert task["metadata"]["single_turn"] == found.single_turn
        assert task["id"] not in ids, (split, index)
        ids.add(task["id"])
        assert (found.check is None) == (found.block in UNGRADED_BLOCKS), found.key
        assert builds(found.check, task["prompt"]), found.key

print(f"wheel verified: {len(ids)} rows, {len(goldens)} goldens")
