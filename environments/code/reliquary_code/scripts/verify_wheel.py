"""Check the built wheel from outside the source tree.

Run from a directory that is not the package's own, so the imports resolve to
what was installed rather than to the files beside them. A wheel that builds
but ships a stale extractor, or that reaches back into Reliquary core for its
grader, is a wheel that would be discovered in a training run rather than here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.resources
import json
import pkgutil

import reliquary_code
from reliquary_code import CodeEnvironment
from reliquary_code.corpus import OCI_REPO, OCI_REVISION
from reliquary_code.extraction import extract_python

root = importlib.resources.files("reliquary_code")

artifact = json.loads(root.joinpath("artifact.json").read_text())
for name, digest in artifact["files"].items():
    actual = hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: shipped {actual}, pinned {digest}"

# The port is a copy, not a dependency, and the wheel is where that stops being
# a claim about the source tree: an installed package that imports `reliquary.*`
# works in the core checkout and nowhere else.
for info in pkgutil.walk_packages(reliquary_code.__path__, prefix="reliquary_code."):
    source = importlib.util.find_spec(info.name).origin
    assert source is not None, info.name
    text = open(source, encoding="utf-8").read()
    assert "import reliquary\n" not in text, info.name
    assert "from reliquary." not in text, info.name
    assert "import reliquary." not in text, info.name

assert OCI_REPO == "R0mAI/opencodeinstruct-curated", OCI_REPO
assert OCI_REVISION == "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7", OCI_REVISION

# The graded span, checked on the installed extractor rather than on the source
# beside it. "Last block wins" is the behaviour this package was cut to replace:
# a rollout that reasons, implements, then demonstrates must still be graded on
# its implementation.
reasoned = (
    "Let me work through it.\n\n"
    "```python\ndef solve(n):\n    return n + 1\n```\n\n"
    "And here is how you would call it:\n\n"
    "```python\nprint(solve(1))\n```\n"
)
assert extract_python(reasoned, "solve") == "def solve(n):\n    return n + 1"
assert extract_python(reasoned) == "print(solve(1))"

# The goldens carry no network dependency of their own beyond the pinned
# corpus, so a wheel verified offline checks the surface and defers these.
environment = CodeEnvironment("train")
assert environment.max_turns == 1
assert environment.validator_authoritative_reward is True
surface = {"task", "grade", "replay", "known_wrong_completion"}
assert surface <= set(dir(CodeEnvironment)), sorted(surface - set(dir(CodeEnvironment)))

print("wheel verified")
