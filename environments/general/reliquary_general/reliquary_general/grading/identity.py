"""Identity: the answer names Teutonic and credits no maker.

The model is called Teutonic and its creator is deliberately left unstated. The
teacher that writes the answers is a Qwen model, which left alone says so; the
generation system prompt tells it otherwise, and this check catches the answers
where the base model's own identity leaks through anyway.

Three rules, all on the answer text:

1. `Teutonic` appears as a word.
2. The teacher's lineage — Qwen, Alibaba, Tongyi — is never named, unless the
   question named it first (then a denial has to be able to say "I am not Qwen").
3. No organisation or model from `CLAIMS` is claimed as the maker or as the
   model itself ("I was trained by OpenAI", "I'm ChatGPT"), whether or not the
   question named it; a negated sentence ("I was not made by Google") is a
   denial, not a claim.
"""

from __future__ import annotations

import re
from typing import Any

NAME = re.compile(r"\bTeutonic\b")
LINEAGE = ("Qwen", "Alibaba", "Tongyi")
CLAIMS = (
    "Qwen", "Alibaba", "Tongyi", "OpenAI", "ChatGPT", "GPT-4o", "GPT-4", "GPT-3.5", "GPT", "Anthropic", "Claude",
    "Google", "DeepMind", "Gemini", "Gemma", "Bard", "Meta", "Llama", "Mistral", "DeepSeek", "Microsoft", "Copilot",
    "Phi", "NVIDIA", "Nemotron", "xAI", "Grok", "IBM", "Amazon", "Apple", "Baidu", "Hugging Face", "OLMo",
    "Allen Institute", "Ai2",
)
_ORG = "|".join(re.escape(name) for name in sorted(CLAIMS, key=len, reverse=True))
_MAKER = re.compile(
    rf"\b(?:made|built|created|developed|trained|designed|engineered|produced|programmed|owned|released)\s+"
    rf"(?:\w+\s+){{0,2}}?(?:by|at)\s+(?:the\s+)?(?:team\s+at\s+)?(?P<org>{_ORG})\b",
    re.I,
)
_SELF = re.compile(
    rf"\b(?:I\s+am|I'm|I\s+was|this\s+is)\s+(?:a\s+|an\s+|the\s+)?(?:version\s+of\s+|variant\s+of\s+)?"
    rf"(?P<org>{_ORG})(?:'s)?\b",
    re.I,
)
_BASED = re.compile(rf"\b(?:based\s+on|built\s+on|powered\s+by|fine-?tuned\s+from)\s+(?:the\s+)?(?P<org>{_ORG})\b", re.I)
_NEGATION = re.compile(r"\b(?:not|n't|never|no|neither|nor|rather\s+than|instead\s+of|unlike|nothing\s+to\s+do)\b", re.I)
_SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]?")


def mentioned(prompt: str) -> list[str]:
    return [name for name in LINEAGE if re.search(rf"\b{re.escape(name)}", prompt, re.I)]


def _claims(answer: str) -> bool:
    for sentence in _SENTENCE.findall(answer):
        for pattern in (_MAKER, _SELF, _BASED):
            for match in pattern.finditer(sentence):
                before = sentence[: match.start()] + match.group(0)
                if not _NEGATION.search(before):
                    return True
    return False


def grade(check: dict[str, Any], prompt: str, answer: str) -> float:
    if not answer.strip() or not NAME.search(answer):
        return 0.0
    allowed = {name.lower() for name in check.get("allow_mention") or mentioned(prompt)}
    for name in LINEAGE:
        if name.lower() not in allowed and re.search(rf"\b{re.escape(name)}", answer, re.I):
            return 0.0
    return 0.0 if _claims(answer) else 1.0


__all__ = ["CLAIMS", "LINEAGE", "grade", "mentioned"]
