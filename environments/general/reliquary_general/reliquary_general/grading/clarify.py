"""Clarification pairs: ask when a piece is missing, answer when it is not.

A heuristic, and labelled as one. Each pair is one real request seen twice: as
written, and with the piece the answer depends on cut out (the text to work on,
or the language to translate into). The cut version should be met with a short
question about that piece; the full version with the work itself, not with a
question. Both halves are graded so that the pair teaches when to ask, not to
ask.

`ask` passes when the answer is short (at most `ASK_MAX_CHARS`), contains a
question mark, and mentions the missing piece. `answer` passes when the answer
says something and does not ask for the material or the language.
"""

from __future__ import annotations

import re
from typing import Any

ASK_MAX_CHARS = 1500
SLOT_CUES = {
    "content": re.compile(
        r"\b(?:text|article|passage|content|document|essay|email|e-mail|paragraph|story|poem|letter|report|"
        r"message|review|speech|script|summary|abstract|post|lyrics|chapter|section|statement|description|"
        r"paste|share|provide|send|include|attach)\w*\b",
        re.I,
    ),
    "target_language": re.compile(r"\b(?:language|translat\w*|into\s+which|which\s+one)\b", re.I),
}
REQUEST = re.compile(
    r"\b(?:could|can|would)\s+you\s+(?:please\s+)?(?:provide|share|paste|send|specify|clarify|tell\s+me\s+which)\b|"
    r"\bplease\s+(?:provide|share|paste|send|specify)\b|\bwhich\s+language\b|\bwhat\s+language\b|"
    r"\b(?:you\s+)?(?:haven't|have\s+not|didn't|did\s+not)\s+(?:included|provided|shared|pasted|attached|specified)\b",
    re.I,
)


def grade(check: dict[str, Any], answer: str) -> float:
    if not answer.strip():
        return 0.0
    if check["expect"] == "ask":
        if len(answer) > ASK_MAX_CHARS or "?" not in answer:
            return 0.0
        return 1.0 if SLOT_CUES[check["slot"]].search(answer) else 0.0
    return 0.0 if REQUEST.search(answer) else 1.0


__all__ = ["ASK_MAX_CHARS", "grade"]
