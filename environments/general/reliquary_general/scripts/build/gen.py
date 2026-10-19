"""Home-made prompt generators: verifiable instructions, clarification pairs, identity.

Deterministic: every random draw is seeded from the prompt's key, so a rerun on
the same pools writes the same prompts. Imported by assemble.py.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # the package root
from reliquary_general._ifeval import instructions_registry as reg  # noqa: E402


def rng_for(*parts) -> random.Random:
    seed = int.from_bytes(hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest(), "big")
    return random.Random(seed)


def seed_module_random(*parts) -> None:
    """The IFEval builders draw from the module-level `random`."""
    random.seed(int.from_bytes(hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest(), "big"))


# ------------------------------------------------------------------ instructions

# Weighted toward constraints a person would plausibly ask for. Excluded:
#   - nondeterministic checkers (langdetect, Punkt): response_language, english_capital,
#     english_lowercase, number_sentences;
#   - checkers that need the prompt echoed (repeat_prompt, copy, copy_span_idx, copying_*);
#   - IFBench-train checkers whose idea reappears in the IFBench *test* set
#     (palindrome, keyword_specific_position, count_unique);
#   - degenerate shapes that make poor distillation targets (square_brackets,
#     bigram_wrapping, sentence_hyphens, no_adjacent_consecutive, lowercase_counting,
#     letter_counting, counting_composition, first/last_word_sent, punctuation_dot,
#     repeat_phrase, exclude_word_harder, count_increment_word whose recorded
#     arguments do not rebuild it).
CONSTRAINT_WEIGHTS = {
    "keywords:existence": 3,
    "keywords:frequency": 2,
    "keywords:forbidden_words": 3,
    "keywords:letter_frequency": 1,
    "length_constraints:number_paragraphs": 2,
    "length_constraints:number_words": 4,
    "length_constraints:nth_paragraph_first_word": 1,
    "detectable_content:number_placeholders": 2,
    "detectable_content:postscript": 3,
    "detectable_format:number_bullet_lists": 4,
    "detectable_format:constrained_response": 1,
    "detectable_format:number_highlighted_sections": 3,
    "detectable_format:multiple_sections": 2,
    "detectable_format:json_format": 1,
    "detectable_format:title": 4,
    "combination:two_responses": 1,
    "startend:end_checker": 3,
    "startend:quotation": 1,
    "change_case:capital_word_frequency": 2,
    "punctuation:no_comma": 3,
    "punctuation:punctuation_exclamation": 1,
    "first_word:first_word_answer": 2,
    "last_word:last_word_answer": 2,
    "keywords:word_once": 2,
    "keywords:word_count_different_numbers": 1,
    "paragraphs:paragraphs": 1,
    "paragraphs:paragraphs2": 1,
    "keywords:start_end": 1,
    "letters:letter_counting2": 1,
}
K_WEIGHTS = {1: 25, 2: 35, 3: 25, 4: 10, 5: 5}
STYLES = ("inline", "rules", "list")

_CONFLICTS = {k: set(v) for k, v in reg.INSTRUCTION_CONFLICTS.items()}


def _conflict(a: str, b: str) -> bool:
    return b in _CONFLICTS.get(a, set()) or a in _CONFLICTS.get(b, set())


def build(instruction_id: str, kwargs: dict | None, *seed):
    seed_module_random(*seed)
    ins = reg.INSTRUCTION_DICT[instruction_id](instruction_id)
    desc = ins.build_description(**(kwargs or {}))
    args = ins.get_instruction_args() or {}
    return desc, {k: v for k, v in args.items() if v is not None}


def draw_constraints(key: str):
    r = rng_for("if-k", key)
    k = r.choices(list(K_WEIGHTS), weights=list(K_WEIGHTS.values()))[0]
    ids = list(CONSTRAINT_WEIGHTS)
    chosen: list[str] = []
    tries = 0
    while len(chosen) < k and tries < 50:
        tries += 1
        c = r.choices(ids, weights=[CONSTRAINT_WEIGHTS[i] for i in ids])[0]
        if any(c == x or _conflict(c, x) for x in chosen):
            continue
        chosen.append(c)
    out = []
    for j, cid in enumerate(chosen):
        desc, args = build(cid, None, "if-build", key, j)
        # The stored arguments must rebuild the same instruction, or the grader
        # would check something other than what the prompt asked for.
        try:
            desc2, args2 = build(cid, args, "if-rebuild", key, j)
        except (AttributeError, TypeError, ValueError):
            return None
        if desc2 != desc or args2 != args:
            return None
        if cid == "length_constraints:nth_paragraph_first_word" and args["nth_paragraph"] > args["num_paragraphs"]:
            return None
        out.append({"id": cid, "kwargs": args, "description": desc})
    return out


def render_constraints(base: str, constraints, key: str) -> str:
    style = STYLES[rng_for("if-style", key).randrange(len(STYLES))]
    descs = [c["description"].strip() for c in constraints]
    if style == "inline":
        return base.rstrip() + "\n\n" + " ".join(descs)
    if style == "rules":
        return base.rstrip() + "\n\nPlease follow these rules when answering: " + " ".join(descs)
    return base.rstrip() + "\n\nRequirements:\n" + "\n".join(f"- {d}" for d in descs)


_CODEY = re.compile(r"```|\b(?:code|function|python|javascript|typescript|java|c\+\+|c#|sql|html|css|regex|script|"
                    r"program|bash|json|yaml|xml|csv|latex|excel|formula)\b", re.I)
_FORMATTED = re.compile(r"\b(?:\d+\s*(?:words?|paragraphs?|sentences?|bullet|lines?|points?)|table|bullet|"
                        r"list of|in one sentence|word count|format|outline|step[- ]by[- ]step|translate|rewrite|"
                        r"paraphrase|proofread|correct|fix|summari[sz]e|tl;?dr)\b", re.I)


_TRANSCRIPT = re.compile(r"\[/?INST\]|^\s*(?:User|Assistant|Human|AI|System)\s*:", re.I | re.M)


def if_base_ok(text: str) -> bool:
    """A plain request: no code, no format of its own to clash with, no pasted transcript."""
    n = len(text.split())
    return (5 <= n <= 250 and not _CODEY.search(text) and not _FORMATTED.search(text)
            and not _TRANSCRIPT.search(text))


# ---------------------------------------------------------------- clarification

_LANGS = ("english|spanish|french|german|italian|portuguese|chinese|mandarin|japanese|korean|russian|arabic|"
          "hindi|dutch|turkish|polish|swedish|vietnamese|indonesian|greek|hebrew|thai|ukrainian|czech|romanian|"
          "persian|farsi|urdu|bengali|tagalog|malay")
TRANSLATE = re.compile(rf"^(?P<pre>\s*(?:please\s+|can you\s+|could you\s+)?translate\b[^\n:]{{0,60}}?)"
                       rf"\s+(?:in|into|to)\s+(?P<lang>{_LANGS})\b(?P<post>[^\n]{{0,40}}?[:\n])", re.I)
CONTENT_INSTR = re.compile(
    r"^(?P<instr>[^\n]{8,240}?\b(?:the following|below|this|these|my)\s+"
    r"(?:text|article|passage|essay|email|e-mail|paragraph|paragraphs|story|poem|letter|report|abstract|"
    r"message|review|speech|script|document|summary|content|post|lyrics|chapter|section|statement|description)"
    r"[^\n:]{0,80}?)\s*(?::[ \t]*\n|\n[ \t]*\n)\s*(?P<content>.+)$", re.S | re.I)
# The same request with the material on the instruction's own line: the colon
# has to follow the noun that names the material (at most three words later),
# so a colon inside the instruction ("structured as 'Question: ...'") is not
# taken for the cut.
CONTENT_INLINE = re.compile(
    r"^(?P<instr>[^\n:]{8,240}?\b(?:the following|below|this|these|my)\s+"
    r"(?:text|article|passage|essay|email|e-mail|paragraph|paragraphs|story|poem|letter|report|abstract|"
    r"message|review|speech|script|document|summary|content|post|lyrics|chapter|section|statement|description)"
    r"(?:\s+[\w'-]+){0,3})\s*:[ \t]*(?P<content>\S.+)$", re.S | re.I)
CONTENT_VERBS = re.compile(r"\b(?:summari[sz]e|paraphrase|rewrite|proofread|edit|improve|shorten|simplify|"
                           r"analy[sz]e|critique|review|explain|translate|correct|polish|condense|expand|reword|"
                           r"make|give|write|check|grade|evaluate)\b", re.I)


def clarification(text: str):
    """(amputated prompt, slot) when `text` can lose a piece the answer depends on."""
    m = TRANSLATE.match(text)
    if m:
        rest = text[m.end():].strip()
        if len(rest.split()) >= 5:
            amputated = (m.group("pre").rstrip() + m.group("post") + text[m.end():]).strip()
            if amputated != text and not re.search(rf"\b(?:{_LANGS})\b", m.group("pre"), re.I):
                return amputated, "target_language"
    m = CONTENT_INSTR.match(text) or CONTENT_INLINE.match(text)
    if m and not _TRANSCRIPT.search(m.group("instr")) and CONTENT_VERBS.search(m.group("instr")) and len(m.group("content").split()) >= 60 \
            and len(m.group("instr").split()) <= 40:
        instr = m.group("instr").rstrip()
        return instr + ":", "content"
    return None


# --------------------------------------------------------------------- identity

MODELS = ("ChatGPT", "GPT-4", "GPT-4o", "Claude", "Gemini", "Llama", "Qwen", "Mistral", "DeepSeek", "Grok",
          "Copilot", "Nemotron", "Phi", "Gemma", "Bard")
ORGS = ("OpenAI", "Anthropic", "Google", "Meta", "Alibaba", "Mistral AI", "DeepSeek", "Microsoft", "NVIDIA",
        "xAI", "IBM", "Amazon", "Apple", "Baidu", "Hugging Face")
PLAIN = (
    "Who made you?", "Who created you?", "Who built you?", "Who trained you?", "Who developed you?",
    "What is your name?", "What's your name?", "What should I call you?", "Introduce yourself.",
    "Tell me about yourself.", "What are you?", "Which company is behind you?",
    "Which organization developed you?", "What model are you?", "What model am I talking to?",
    "Which language model powers you?", "What AI are you?", "Are you an AI or a human?", "Who owns you?",
    "Who is your creator?", "Who are your developers?", "Where do you come from?",
    "What is the name of this assistant?", "Do you have a name?", "What version are you?",
    "What are you called?", "Which lab made you?", "Who is responsible for building you?",
    "Can you tell me who designed you?", "What's the model behind this chat?",
    "Describe yourself in a few words.", "Who should I credit for your answers?",
    "What kind of assistant are you?", "Are you open source?", "Who maintains you?",
)
ENTITY = (
    "Are you {M}?", "Is this {M}?", "Am I talking to {M}?", "Are you based on {M}?",
    "Are you a version of {M}?", "Do you run on {M}?", "Are you {M} in disguise?", "Is {M} your real name?",
    "Were you made by {O}?", "Were you trained by {O}?", "Did {O} create you?", "Is {O} your developer?",
    "Are you {O}'s model?", "Do you belong to {O}?", "Did {O} build you?", "Are you a product of {O}?",
)
PREFIX = ("", "Hey, ", "Quick question: ", "Hi! ", "Just curious, ", "Honestly, ", "Be straight with me: ")
SUFFIX = ("", " Be honest.", " Answer in one sentence.", " I'm writing a report and need to know.",
          " My friend says otherwise.")


def _cap(prefix: str, q: str) -> str:
    if prefix and prefix.endswith(", "):
        q = q[0].lower() + q[1:] if not q.startswith(("I ", "OpenAI", "Anthropic", "Google", "Meta", "Alibaba",
                                                         "Mistral", "DeepSeek", "Microsoft", "NVIDIA", "IBM",
                                                         "Amazon", "Apple", "Baidu", "Hugging", "xAI")) else q
    return prefix + q


def identity_paraphrases(n: int) -> list[str]:
    pool = set()
    for q in PLAIN:
        for p in PREFIX:
            for s in SUFFIX:
                pool.add(_cap(p, q) + s)
    for q in ENTITY:
        for m in MODELS:
            for o in ORGS:
                base = q.format(M=m, O=o)
                for p in PREFIX[:4]:
                    for s in SUFFIX[:3]:
                        pool.add(_cap(p, base) + s)
    ordered = sorted(pool)
    rng_for("identity-paraphrases").shuffle(ordered)
    return ordered[:n]
