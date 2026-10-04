"""One row per problem, across sources.

taco, primeintellect and CodeContests+ are largely the same Codeforces problems
formatted differently (measured: 21,294 DeepCoder stdin rows are 10,781
distinct statements). Exact duplicates share a normalised hash; near
duplicates share most 5-word shingles and are found by MinHash with LSH
banding, then confirmed at an estimated Jaccard of 0.8. MinHash cannot see
that "n <= 10" and "n <= 3000" differ, so a confirmed edge must also agree on
its number tokens and must not pair an easy version with a hard one.
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
_EASY = frozenset({"easy", "easier"})
_HARD = frozenset({"hard", "harder"})
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


def _version_words(text: str) -> frozenset[str]:
    return frozenset(word for word in text.split() if word in _EASY or word in _HARD)


def _compatible(left: str, right: str) -> bool:
    """Guards a MinHash edge: same numbers, and not two versions of one problem."""
    if set(_NUMBER.findall(left)) != set(_NUMBER.findall(right)):
        return False
    a, b = _version_words(left), _version_words(right)
    if (a & _EASY and b & _HARD) or (a & _HARD and b & _EASY):
        return False
    return not (a and b and a != b)


def _merge(members: list[SourceRow]) -> SourceRow:
    """References come lead-first: its own exact group, then the other members'."""
    lead = max(members, key=lambda row: (len(row.tests), row.source, row.upstream_id))
    lead_id = problem_id(lead.statement)
    ordered = [r for r in members if problem_id(r.statement) == lead_id] + [
        r for r in members if problem_id(r.statement) != lead_id
    ]
    references = tuple(dict.fromkeys(ref for row in ordered for ref in row.references))
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
