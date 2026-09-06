"""OpenCodeInstruct as one Verifiers taskset and one replay environment.

Two surfaces, as this repository requires: `CodeTaskset` for Verifiers and
prime-rl, and `CodeEnvironment` for synchronous replay.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_code.corpus import corpus_length, get_problem
from reliquary_code.extraction import (
    contract_instruction,
    entry_function_name,
    extract_python,
)
from reliquary_code.runner import run_cases

ENVIRONMENT = "reliquary_code_v1"
SPLITS = ("train",)

DEFAULT_PROMPT = (
    "Solve the following problem step by step.\n\n"
    "{problem}\n\n"
    "{contract}\n\n"
    "Give the final program in the last fenced Python code block."
)

# A deliberately wrong but syntactically valid completion: it defines a
# callable (so extraction and case dispatch actually run it) but returns a
# constant no real corpus case expects. Used both by `CodeTask.validate`
# (to prove the failing half runs through `run_cases` for real, not the
# `if not source.strip()` short-circuit that a plain "no code" string hits
# before ever executing anything) and by the goldens test, so the two stay
# in lockstep.
WRONG_BUT_VALID_COMPLETION = (
    "```python\n"
    "def _wrong_answer(*args, **kwargs):\n"
    "    return '__reliquary_golden_wrong_answer__'\n"
    "```"
)


def _reward(row: dict[str, Any], completion: str) -> float:
    cases = list(row["structured_cases"])
    if not cases:
        return 0.0
    source = extract_python(completion, entry_function_name(cases))
    if not source.strip():
        return 0.0
    results = run_cases(source, cases)
    return sum(1 for ok in results if ok) / len(results)


class CodeEnvironment:
    """Synchronous, JSON-shaped ABI used by replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return corpus_length()

    def _row(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside the pinned corpus")
        return get_problem(index)

    def task(self, index: int) -> dict[str, Any]:
        row = self._row(index)
        prompt = self.prompt_template.format(
            problem=row["input"],
            contract=contract_instruction(list(row["structured_cases"])),
        )
        return {
            "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "metadata": {"index": int(index), "split": self.split},
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        row = self._row(index)
        reward = _reward(row, completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"index": int(index), "success": reward >= 1.0},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def known_wrong_completion(self, index: int) -> str:
        """There is no reference program in this corpus (unlike
        `reliquary_math`'s `reference_completion`, which returns an answer
        that scores 1.0): this is a deliberately failing probe, not a
        reference. Goldens pin that it scores 0; the passing half is
        covered by the runner tests and by the real-corpus-row correctness
        test in `test_code.py`."""
        del index
        return "```python\nraise SystemExit(1)\n```"


class CodeData(vf.TaskData):
    index: int
    structured_cases: list[dict[str, Any]] = Field(repr=False)
    split: Literal["train"]


class CodeTaskConfig(vf.TaskConfig):
    pass


class CodeTask(vf.Task[CodeData, vf.State, CodeTaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.split}:{self.data.index}"

    @vf.reward(weight=1.0)
    async def passing_cases(self, trace: vf.Trace) -> float:
        return _reward(
            {"structured_cases": self.data.structured_cases},
            trace.last_reply or "",
        )

    async def validate(self, runtime: vf.Runtime) -> bool:
        """A well-formed wrong answer must score zero, and the case list
        must be usable.

        There is no reference program to check the passing direction with
        (see `CodeEnvironment.known_wrong_completion`), so this asserts the
        failing half — using `WRONG_BUT_VALID_COMPLETION`, which extracts to
        real source and actually runs through `run_cases`, rather than a
        bare "no code" string that short-circuits at the
        `if not source.strip()` check in `_reward` before `run_cases` is
        ever called.
        """
        del runtime
        row = {"structured_cases": self.data.structured_cases}
        wrong = _reward(row, WRONG_BUT_VALID_COMPLETION)
        return wrong == 0.0 and len(self.data.structured_cases) > 0


class CodeConfig(vf.TasksetConfig):
    split: Literal["train"] = "train"
    task: CodeTaskConfig = CodeTaskConfig()
    prompt_template: str = DEFAULT_PROMPT


class CodeTaskset(vf.Taskset[CodeTask, CodeConfig]):
    INFINITE = False

    def load(self) -> Iterator[CodeTask]:
        environment = CodeEnvironment(
            self.config.split, self.config.prompt_template
        )
        for index in range(len(environment)):
            row = environment._row(index)
            prompt = environment.prompt_template.format(
                problem=row["input"],
                contract=contract_instruction(list(row["structured_cases"])),
            )
            yield CodeTask(
                CodeData(
                    idx=index,
                    prompt=prompt,
                    network_allow=[],
                    index=index,
                    structured_cases=list(row["structured_cases"]),
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["CodeEnvironment", "CodeTaskset"]
