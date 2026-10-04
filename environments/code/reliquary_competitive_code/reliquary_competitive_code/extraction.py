"""Which part of a completion is the program.

The last fenced block tagged as Python, or untagged, is the submission. A block
in another language after it (sample output, a C++ sketch) does not hide it,
and a block left open by truncation is not a program: it scores as no code
rather than running half a solution.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = frozenset({"", "python", "python3", "py"})


def extract_program(completion: str) -> str | None:
    programs = [
        body
        for tag, body in _FENCE.findall(completion or "")
        if tag.strip().lower() in _PYTHON_TAGS
    ]
    if not programs or not programs[-1].strip():
        return None
    return programs[-1]
