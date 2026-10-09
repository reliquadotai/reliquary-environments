"""The part of a completion that is the answer.

A completion from the thinking renderer is the reasoning, `</think>`, then the
answer; from the direct renderer it is the answer alone, the template having
already closed an empty reasoning block in the prompt. Every grader here reads
only what follows the last `</think>`: a constraint such as "no commas" is a
claim about the reply the user sees, not about the scratchpad.

A completion that opened a reasoning block and never closed it has no answer:
it ran out of budget while thinking, and the export must not mistake its
reasoning for a reply.
"""

from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TURN_END = ("<|im_end|>", "<|endoftext|>")


def answer_text(completion: str | None) -> str:
    if not isinstance(completion, str):
        return ""
    text = completion.rstrip()
    changed = True
    while changed:
        changed = False
        for terminator in TURN_END:
            if text.endswith(terminator):
                text = text[: -len(terminator)].rstrip()
                changed = True
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    if THINK_OPEN in text:
        return ""
    return text.strip()


__all__ = ["THINK_CLOSE", "THINK_OPEN", "TURN_END", "answer_text"]
