"""8-gram decontamination against the held-out evaluations.

A candidate text is contaminated by an eval item when it shares at least
`min(3, grams(item))` of that item's 8-grams: three shared 8-grams is ten or
more consecutive words in common, and an item shorter than that only counts
when it appears whole. 8-grams that occur in more than MAX_DF eval items are
boilerplate ("answer the following question with ...") and are dropped from
the index, as are 8-grams shared by most of a benchmark's own template.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

from common import H, ngram_hashes, words

N = 8
MAX_DF = 20


class EvalIndex:
    def __init__(self, kinds=("prompt", "tool_doc")):
        self.items: list[tuple[str, str]] = []  # (bench, id)
        self.size: list[int] = []
        self.idx: dict[int, int] = {}
        df: Counter = Counter()
        grams_by_item = []
        self.function_names: dict[str, set[str]] = defaultdict(set)
        for path in sorted(glob.glob(f"{H}/evals/*.jsonl")):
            for line in open(path):
                r = json.loads(line)
                if r["kind"] == "function_name":
                    self.function_names[r["bench"]].add(norm_fn(r["text"]))
                    continue
                if r["kind"] not in kinds:
                    continue
                g = ngram_hashes(words(r["text"]), N)
                if not g:
                    continue
                self.items.append((r["bench"], r["id"]))
                self.size.append(len(g))
                grams_by_item.append(g)
                df.update(g)
        self.boilerplate = sum(1 for c in df.values() if c > MAX_DF)
        for i, g in enumerate(grams_by_item):
            for h in g:
                if df[h] > MAX_DF:
                    continue
                prev = self.idx.get(h)
                # A gram in several items keeps the first; hits are counted per item.
                if prev is None:
                    self.idx[h] = i
        del grams_by_item, df

    def hits(self, text: str) -> list[tuple[str, str]]:
        """Eval items `text` is contaminated by."""
        per: Counter = Counter()
        for h in ngram_hashes(words(text), N):
            i = self.idx.get(h)
            if i is not None:
                per[i] += 1
        return [self.items[i] for i, c in per.items() if c >= min(3, self.size[i])]


def norm_fn(name: str) -> str:
    name = name.split(".")[-1]
    return re.sub(r"[^a-z0-9]", "", name.lower())
