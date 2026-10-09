"""Verifiable instructions, graded by the vendored IFEvalG checkers.

Same rule as `reliquary-instruction-following`: every constraint holds or the
answer scores zero, an empty answer scores zero before any checker runs (several
are satisfied vacuously by one), and a checker that trips over an answer has
failed to find what it looked for. A constraint that cannot be built at all is
a packaging fault, which `builds` lets the corpus loader refuse up front.
"""

from __future__ import annotations

from typing import Any

from reliquary_general._ifeval import instructions_registry

# Checkers whose verdict is not a fact two participants share: three consult
# `langdetect`, which samples, and one needs the Punkt model, a download.
NONDETERMINISTIC = frozenset(
    {
        "change_case:english_capital",
        "change_case:english_lowercase",
        "language:response_language",
        "length_constraints:number_sentences",
    }
)


def _instruction(instruction_id: str, kwargs: dict[str, Any], prompt: str) -> Any:
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    arguments = instruction.get_instruction_args()
    if arguments and "prompt" in arguments:
        instruction.build_description(prompt=prompt)
    return instruction


def builds(check: dict[str, Any], prompt: str) -> bool:
    try:
        constraints = check["constraints"]
        if not constraints:
            return False
        for constraint in constraints:
            if constraint["id"] in NONDETERMINISTIC:
                return False
            _instruction(constraint["id"], constraint["kwargs"], prompt)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return True


def grade(check: dict[str, Any], prompt: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    for constraint in check["constraints"]:
        instruction = _instruction(constraint["id"], constraint["kwargs"], prompt)
        try:
            if not instruction.check_following(answer):
                return 0.0
        except Exception:
            return 0.0
    return 1.0


__all__ = ["NONDETERMINISTIC", "builds", "grade"]
