"""GPQA scored on the extracted letter alone.

The upstream task falls back to an LLM judge whenever its letter extractor finds
no match, and that judge defaults to a hosted model. A judge's verdict is not a
fact two runs can both reproduce, and a checkpoint compared against its
predecessor would then differ by the judge's mood as well as by the policy. This
replaces the task's `correct` reward under the same name, so the upstream
extractor still decides and an answer it cannot read scores 0.

Plugged by `configs/gpqa.toml` through `env.taskset.task.rewards.correct.fn`.
"""

from gpqa.mcq import extract_mcq_answer


async def correct(task, trace) -> float:
    return 1.0 if extract_mcq_answer(trace.last_reply) == task.answer else 0.0
