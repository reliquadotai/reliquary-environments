"""Shared helpers for the general-prompts build: hashing, n-grams, MinHash, filters.

Every hash here is deterministic (blake2b / crc32), never Python's salted `hash`,
so a rerun of the build reaches the same corpus byte for byte.
"""
from __future__ import annotations

import hashlib
import os
import json
import re
import zlib

import numpy as np

H = os.environ.get("GENERAL_PROMPTS_WORK", "/home/ubuntu/cc-data/general-prompts")

_TOK = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def norm_key(text: str) -> str:
    """Exact-duplicate key: lowercase alphanumeric token stream."""
    return hashlib.blake2b(" ".join(words(text)).encode(), digest_size=12).hexdigest()


def bucket(key: str, salt: str, mod: int = 10_000) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest(), "big") % mod


def ngram_hashes(toks: list[str], n: int) -> set[int]:
    if len(toks) < n:
        return {zlib.crc32(" ".join(toks).encode()) | (len(toks) << 32)} if toks else set()
    return {zlib.crc32(" ".join(toks[i:i + n]).encode()) | (n << 32) for i in range(len(toks) - n + 1)}


def user_text(messages) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


# ------------------------------------------------------------------ MinHash / LSH

_P = np.uint64((1 << 61) - 1)
_rng = np.random.default_rng(20261009)
NPERM = 36
BANDS, ROWS = 6, 6
_A = _rng.integers(1, 1 << 32, size=NPERM, dtype=np.uint64)
_B = _rng.integers(0, 1 << 32, size=NPERM, dtype=np.uint64)
_MASK = np.uint64(0xFFFFFFFF)


def minhash(text: str) -> np.ndarray | None:
    sh = ngram_hashes(words(text), 5)
    if not sh:
        return None
    x = np.fromiter(sh, dtype=np.uint64) & _MASK
    v = (_A[:, None] * x[None, :] + _B[:, None]) % _P
    return v.min(axis=1)


def band_keys(sig: np.ndarray) -> list[int]:
    return [int.from_bytes(hashlib.blake2b(bytes([b]) + sig[b * ROWS:(b + 1) * ROWS].tobytes(),
                                           digest_size=8).digest(), "big") for b in range(BANDS)]


class NearDup:
    """Greedy near-duplicate filter: the first document of a cluster wins."""

    def __init__(self, threshold: float = 0.7):
        self.t = threshold
        self.bands: dict[int, int] = {}
        self.sigs: list[np.ndarray] = []

    def seen(self, text: str) -> bool:
        sig = minhash(text)
        if sig is None:
            return False
        keys = band_keys(sig)
        for k in keys:
            j = self.bands.get(k)
            if j is not None and float((self.sigs[j] == sig).mean()) >= self.t:
                return True
        idx = len(self.sigs)
        self.sigs.append(sig)
        for k in keys:
            self.bands.setdefault(k, idx)
        return False


# ------------------------------------------------------------------------ filters

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w.])")
# Placeholders the upstream de-identifiers leave behind.
REDACTED = re.compile(r"\[(?:REDACTED|NAME|EMAIL|PHONE)[^\]]*\]|<\|?(?:PII|REDACTED)[^>]*>", re.I)


def has_pii(text: str) -> bool:
    return bool(EMAIL.search(text) or PHONE.search(text))


# A deliberately short lexicon: the sources are already moderated (WildChat by
# OpenAI Moderation + Detoxify, oasst2 by review); this catches what slipped
# through at the extreme end. The safety block is exempt.
_EXTREME = re.compile(
    r"\b(?:child\s*porn\w*|cp\s+links|loli(?:con)?|shota(?:con)?|pedo(?:phile|philia)?|underage\s+(?:sex|nude|porn)|"
    r"nigg(?:er|a)s?|fag(?:got)?s?|kikes?|spics?|chinks?|tranny|trannies|rape\s+(?:her|him|them)|incest\s+porn|"
    r"bestiality|zoophil\w*|beheading\s+video|how\s+to\s+make\s+(?:a\s+)?(?:bomb|meth|ricin|sarin|nerve\s+agent))\b",
    re.I)


def extreme(text: str) -> bool:
    return bool(_EXTREME.search(text))


_JUNK = re.compile(r"^\s*(?:continue|go on|more|next|ok|okay|thanks?|thank you|yes|no|hi|hello|hey|test)\W*\s*$", re.I)
MIDJOURNEY = re.compile(r"(?:midjourney|/imagine prompt|stable diffusion prompt|As a prompt generator for a generative AI)", re.I)


def junk(text: str) -> bool:
    return bool(_JUNK.match(text)) or bool(MIDJOURNEY.search(text))


def wc(text: str) -> int:
    return len(text.split())


def jdump(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=False)
