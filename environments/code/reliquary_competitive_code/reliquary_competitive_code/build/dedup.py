"""One row per problem, across sources.

taco, primeintellect and CodeContests+ are largely the same Codeforces problems
formatted differently (measured: 21,294 DeepCoder stdin rows are 10,781
distinct statements). Exact duplicates share a normalised hash; near
duplicates share most 5-word shingles and are found by MinHash with LSH
banding, then confirmed at an estimated Jaccard of 0.8. MinHash cannot see
that "n <= 10" and "n <= 3000" differ, so a confirmed edge must also have the
same digits (all digit runs joined, so "10^5" and "10<sup>5</sup>" agree) and
must not pair an easy version with a hard one.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from reliquary_competitive_code.sources.common import SourceRow, normalise, problem_id

SHINGLE = 5
PERMUTATIONS = 64
BANDS = 16
THRESHOLD = 0.8
_NUMBER = re.compile(r"\d+")
_EASY = re.compile(r"\b(?:easy|easier) version\b")
_HARD = re.compile(r"\b(?:hard|harder) version\b")
_PRIME = (1 << 31) - 1
_RNG = np.random.default_rng(20261004)
_A = _RNG.integers(1, _PRIME, PERMUTATIONS, dtype=np.uint64)
_B = _RNG.integers(0, _PRIME, PERMUTATIONS, dtype=np.uint64)


def _signature(statement: str) -> np.ndarray:
    words = normalise(statement).split()
    shingles = {" ".join(words[i : i + SHINGLE]) for i in range(max(1, len(words) - SHINGLE + 1))}
    hashes = np.array(
        [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") for s in shingles],
        dtype=np.uint64,
    )
    return ((_A[:, None] * hashes[None, :] + _B[:, None]) % _PRIME).min(axis=1)


def _digits(text: str) -> str:
    """Digits joined, so "10^5", "10<sup>5</sup>" and "200 000" / "200000" agree."""
    return "".join(_NUMBER.findall(text))


def _compatible(left: str, right: str) -> bool:
    """Guards a MinHash edge: same numbers, and not an easy/hard version pair."""
    if _digits(left) != _digits(right):
        return False
    left_easy, left_hard = bool(_EASY.search(left)), bool(_HARD.search(left))
    right_easy, right_hard = bool(_EASY.search(right)), bool(_HARD.search(right))
    if left_easy and not left_hard and right_hard and not right_easy:
        return False
    return not (left_hard and not left_easy and right_easy and not right_hard)


def _merge(members: list[SourceRow]) -> SourceRow:
    """References come lead-first: its own exact group, then the other members'."""
    lead = max(members, key=lambda row: (len(row.tests), row.source, row.upstream_id))
    lead_id = problem_id(lead.statement)
    key = lambda row: (row.source, row.upstream_id)  # noqa: E731
    siblings = sorted((r for r in members if r is not lead and problem_id(r.statement) == lead_id), key=key)
    others = sorted((r for r in members if problem_id(r.statement) != lead_id), key=key)
    references = tuple(dict.fromkeys(ref for row in [lead, *siblings, *others] for ref in row.references))
    dates = sorted(row.contest_date for row in members if row.contest_date)
    return SourceRow(lead.source, lead.upstream_id, lead.statement, lead.tests, references, dates[0] if dates else None)


def deduplicate(rows: Sequence[SourceRow]) -> list[SourceRow]:
    exact: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        exact[problem_id(row.statement)].append(row)
    keys = sorted(exact)
    texts = [normalise(exact[key][0].statement) for key in keys]
    signatures = [_signature(exact[key][0].statement) for key in keys]

    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    width = PERMUTATIONS // BANDS
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, signature in enumerate(signatures):
            buckets[signature[band * width : (band + 1) * width].tobytes()].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if find(i) == find(j):
                        continue
                    if float(np.mean(signatures[i] == signatures[j])) >= THRESHOLD and _compatible(texts[i], texts[j]):
                        parent[find(j)] = find(i)

    clusters: dict[int, list[SourceRow]] = defaultdict(list)
    for i, key in enumerate(keys):
        clusters[find(i)].extend(exact[key])
    merged = [_merge(members) for members in clusters.values()]
    return sorted(merged, key=lambda row: problem_id(row.statement))
