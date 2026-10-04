"""One row per problem, across sources.

taco, primeintellect and CodeContests+ are largely the same Codeforces problems
formatted differently (measured: 21,707 DeepCoder stdin rows are 11,287
distinct statements). Exact duplicates share a normalised hash; near
duplicates share most 5-word shingles and are found by MinHash with LSH
banding, then confirmed at an estimated Jaccard of 0.8.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from reliquary_competitive_code.sources.common import SourceRow, normalise, problem_id

SHINGLE = 5
PERMUTATIONS = 64
BANDS = 16
THRESHOLD = 0.8
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


def _merge(members: list[SourceRow]) -> SourceRow:
    lead = max(members, key=lambda row: (len(row.tests), row.source, row.upstream_id))
    references = tuple(dict.fromkeys(ref for row in members for ref in row.references))
    dates = sorted(row.contest_date for row in members if row.contest_date)
    return SourceRow(lead.source, lead.upstream_id, lead.statement, lead.tests, references, dates[0] if dates else None)


def deduplicate(rows: Sequence[SourceRow]) -> list[SourceRow]:
    exact: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        exact[problem_id(row.statement)].append(row)
    keys = sorted(exact)
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
            for j in members[1:]:
                if float(np.mean(signatures[members[0]] == signatures[j])) >= THRESHOLD:
                    parent[find(j)] = find(members[0])

    clusters: dict[int, list[SourceRow]] = defaultdict(list)
    for i, key in enumerate(keys):
        clusters[find(i)].extend(exact[key])
    merged = [_merge(members) for members in clusters.values()]
    return sorted(merged, key=lambda row: problem_id(row.statement))
