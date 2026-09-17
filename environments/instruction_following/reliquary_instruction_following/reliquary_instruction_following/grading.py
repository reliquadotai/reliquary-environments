"""Whether an answer followed its instructions, and what that is worth.

One reward and no partial credit: every constraint holds, or the answer scores
zero. Two constraints out of three is not two thirds of an instruction-following
answer, it is an answer that ignored an instruction.

Each constraint is graded the way IFEval grades it — build the instruction from
the row's arguments, hand it the prompt if it asks for one, then check. The
order matters: several checkers read their arguments only at build time, so a
checker that was never built measures whatever it invented for itself.
"""

from __future__ import annotations

import json
from typing import Any

from reliquary_instruction_following._ifeval import instructions_registry

CHECKER_VERSION = "ifevalg-checker-v1"
SPEC_SCHEMA = "reliquary/instruction-following-verifier/v1"


def verifier_spec(prompt: str, constraints: Any) -> str:
    """Everything grading needs, as one string the task can carry."""
    return json.dumps(
        {
            "schema": SPEC_SCHEMA,
            "checker_version": CHECKER_VERSION,
            "prompt": prompt,
            "constraints": [
                {"id": constraint.instruction_id, "kwargs": constraint.kwargs}
                for constraint in constraints
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _instruction(instruction_id: str, kwargs: dict[str, Any], prompt: str) -> Any:
    """Build one verifier, IFEval's way.

    `None` arguments are dropped rather than passed: a checker reads an absent
    argument as "choose one for me", and passing `None` explicitly says the
    same thing while looking like a value.
    """
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(
        **{key: value for key, value in kwargs.items() if value is not None}
    )
    arguments = instruction.get_instruction_args()
    if arguments and "prompt" in arguments:
        instruction.build_description(prompt=prompt)
    return instruction


def follows(instruction_id: str, kwargs: dict[str, Any], prompt: str, answer: str) -> bool:
    """Whether `answer` satisfies one constraint."""
    instruction = _instruction(instruction_id, kwargs, prompt)
    try:
        return bool(instruction.check_following(answer))
    except Exception:
        # A checker that trips over an answer has failed to find the thing it
        # was looking for, so the answer scores zero. Raising here would cost
        # the whole window, and an answer can be arbitrary text. A spec that
        # cannot be built at all still raises, above: that is a packaging
        # fault, not an answer.
        return False


def builds(spec: str) -> bool:
    """Whether every constraint in `spec` resolves to a verifier and builds.

    A spec that cannot be built scores every answer zero, which looks from the
    outside exactly like a policy that follows no instruction.
    """
    try:
        parsed = json.loads(spec)
        for constraint in parsed["constraints"]:
            _instruction(
                constraint["id"], constraint["kwargs"], parsed.get("prompt") or ""
            )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def grade(spec: str, answer: str | None) -> float:
    """1.0 if every constraint in `spec` checks out, 0.0 otherwise."""
    try:
        parsed = json.loads(spec)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    if parsed.get("schema") != SPEC_SCHEMA:
        return 0.0
    if parsed.get("checker_version") != CHECKER_VERSION:
        return 0.0

    # An empty or whitespace-only answer scores zero before any checker sees
    # it: several are satisfied vacuously by one — "at most 3 lowercase words"
    # is true of "" — so without this guard the policy would learn to answer
    # nothing.
    if not isinstance(answer, str) or not answer.strip():
        return 0.0

    constraints = parsed.get("constraints") or []
    if not constraints:
        # A spec with nothing to check is a broken spec, not a free point.
        return 0.0

    prompt = parsed.get("prompt") or ""
    for constraint in constraints:
        if not follows(constraint["id"], constraint["kwargs"], prompt, answer):
            return 0.0
    return 1.0


__all__ = [
    "CHECKER_VERSION",
    "SPEC_SCHEMA",
    "builds",
    "follows",
    "grade",
    "verifier_spec",
]
