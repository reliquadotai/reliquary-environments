"""Selecting the model's Python out of a completion, copied from core.

This surface has produced six recurrences of one bug family: a fenced block
that opens with a language tag, or a trailing block that does not parse,
leaving an empty body and scoring a correct answer zero. Keep it diffable
against `reliquary/environment/opencodeinstruct.py` at core commit 10c2a4d9.
"""

from __future__ import annotations

import ast
import re

# Match fenced code blocks: ``` or ~~~ optionally followed by a language tag.
# Greedy match on the closing fence so the last block wins (model's final
# answer wins over earlier drafts).
_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)

# `_select_python_span` (below) originally resolved a missing
# `protocol_version` from `reliquary.constants.PROTOCOL_VERSION`, which is
# itself `ACTIVE_PROTOCOL_PROFILE.protocol_version` — a value selected at
# process start by the deployment's `RELIQUARY_PROTOCOL_PROFILE` environment
# variable, not a self-contained module-level constant. It is not copyable
# without either importing `reliquary.constants` (forbidden — this package
# may not import `reliquary.*`, and doing so would also drag in the whole
# protocol-profile module) or reproducing the live profile table here, which
# would drift out of sync with core silently. This is the same "call into
# another core module" case reliquary_math/grading.py already hit and
# reported rather than silently substituted for (see that task's report).
#
# The only thing `protocol_version` controls in `_select_python_span` is the
# `entry_rule = protocol_version >= 5` gate. Every profile still used for
# code grading in `reliquary/protocol/profiles.py` at this checkout
# (`qwen3-4b-base-dapo-reasoning-v5` through the v8-dev profiles) is >= 5, so
# this package pins the gate open at the threshold itself rather than assert
# a specific higher version number it has no way to keep verified against a
# moving deployment default. A caller that needs the pre-v5 legacy behavior
# (v2-v4, kept byte-exact in core as historical controls) can still pass
# `protocol_version=` explicitly to `extract_python`/`_select_python_span`.
_PROTOCOL_VERSION = 5


def _entry_function_name(cases: list[dict]) -> str | None:
    """The contract's graded entry function, or None when it isn't a function.

    Same source as ``_contract_instruction``: the cases carry the exact name the
    grader will call, so the extractor can pin the graded block to a definition
    rather than to a position. Method entries define no top-level ``def``, so
    they pin nothing.
    """
    for case in cases or ():
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            return str(name)
    return None


def _defines_top_level_entry(source: str, entry_name: str) -> bool:
    """Whether *source* defines the exact callable the grader will resolve."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_name
        for node in tree.body
    )


def _select_python_span(
    completion: str,
    entry_name: str | None = None,
    *,
    protocol_version: int | None = None,
) -> tuple[str, int, int] | None:
    """Return the exact fenced code span selected for execution.

    Offsets are completion-relative so semantic checks inspect the same bytes.
    v2-v4 retain their legacy last-fence/raw-completion behavior; v5 keeps its
    prompt and generation unchanged while selecting the intended fenced block.
    """

    if not completion:
        return None
    if protocol_version is None:
        protocol_version = _PROTOCOL_VERSION

    entry_rule = int(protocol_version) >= 5
    matches = list(_FENCE_RE.finditer(completion))
    if not matches:
        return None if entry_rule else (completion, 0, len(completion))
    if entry_name and entry_rule:
        for match in reversed(matches):
            body = match.group(2)
            if _defines_top_level_entry(body, entry_name):
                return body, match.start(2), match.end(2)
    match = matches[-1]
    return match.group(2), match.start(2), match.end(2)


def _extract_python(completion: str, entry_name: str | None = None) -> str:
    """Extract Python code from a model completion.

    Strategy: find all fenced code blocks (``` or ~~~ with optional
    'python' tag). From protocol v5 on, return the last block that *defines*
    ``entry_name``; otherwise return the last block.

    With no fence at all, v2-v4 return the raw completion and let exec reject
    obviously-non-code; from v5 the fenced block is the only answer channel, so
    nothing is graded. That fallback fired 762 times across 30 768 production
    rollouts without ever producing a positive reward — a rollout holding code
    always fences it — so it only ever ran ``exec`` on reasoning prose.

    Why the definition beats the position: "last block wins" assumed the
    model closes with its final implementation, which held for the v2-v4 chat
    model. Under the v5 reasoning prompt the model routinely closes with a usage
    demo, an expected-output listing, or a test block — 13.1% of code rollouts
    at the v5 cutover — and grading that span scores a correct answer zero.
    Because the group-relative advantage is what trains the policy, those zeros
    read as "never open a second block", which the model generalised into "never
    reason".

    The gate stops at v5 so v2-v4 stay byte-exact as historical controls: their
    archived runs must stay reproducible. That is the ONLY thing it guards.

    Changing the graded span is not wire-affecting for Code. This environment
    sets ``validator_authoritative_reward = True``, so the validator overwrites
    the miner's declared reward instead of comparing it (admission.py sets
    ``authoritative = True`` for opencodeinstruct; the 1e-6 ``reward_mismatch``
    branch is never reached). A miner on older code is not rejected — it merely
    pre-filters its own submissions against a stale local reward, so it may skip
    groups the validator would have paid. Math keeps the strict comparison, and
    is untouched by this function.
    """
    selected = _select_python_span(completion, entry_name=entry_name)
    return selected[0] if selected is not None else ""


def _contract_instruction(cases: list[dict]) -> str:
    """The grader calls a named function and checks its RETURN value, but the raw
    prompts are stdin/stdout-framed and rarely name the function. Append the exact
    contract (name + "return, don't print") derived from the cases so the model
    writes a callable returning function instead of guessing. Empty for non-
    function entries (nothing to pin)."""
    for case in cases:
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            nargs = len(case.get("args") or [])
            args = "argument" if nargs == 1 else "arguments"
            return (
                f"\n\nWrite your solution as a Python function named `{name}` that "
                f"takes {nargs} {args} and returns the result; do not read from "
                f"stdin or print."
            )
    return ""


extract_python = _extract_python
entry_function_name = _entry_function_name
contract_instruction = _contract_instruction
