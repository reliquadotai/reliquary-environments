"""Rebuild the packaged corpus from the prepared parquet.

Not run by CI and not needed to use the environment: the corpus ships inside
the wheel and is pinned by its digest. It exists so that the derivation is a
program rather than a paragraph — which rows were dropped, and on what rule,
is answerable by running this again and comparing the digest it prints with
the one `corpus.py` holds.

    uv run --with pyarrow python scripts/build_corpus.py \
        --parquet hard_math_aops_v1.parquet \
        --output reliquary_hard_math/data/problems.jsonl.gz

The parquet is the prepared AoPS subset of `nvidia/Nemotron-Math-v2`
(revision 8e793210, CC BY 4.0), filtered upstream of this script to answers a
boxed-answer grader can read, deduplicated, and decontaminated against
DAPO-Math-17k and the public competition benchmarks; README.md says how. What
this script adds is the one test that belongs to the grader: every reference
must be read by `grading.py` and must score 1.0 against itself, boxed. A
reference that does not is dropped and counted, never patched.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from reliquary_hard_math.grading import gradable

# Hardest first (`T1` < `T2` < `T3`). The train split keeps this order, so a
# job taking a prefix of it takes the hardest problems; one covering it whole
# is unaffected.
TIER_NAMES = {"T1_hardest": "T1", "T2_hard": "T2", "T3_medium": "T3"}
SOURCE = "nvidia/Nemotron-Math-v2@8e793210:aops"
LICENSE = "cc-by-4.0"


def build(parquet: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        parquet, columns=["id", "question", "answer", "source", "tier", "license"]
    ).to_pylist()
    corpus: list[dict[str, object]] = []
    ungradable: list[str] = []
    for row in rows:
        if row["license"] != LICENSE or row["source"] != "nemotron-math-v2/aops":
            raise ValueError(f"{row['id']}: unexpected source or licence")
        if not gradable(row["answer"]):
            ungradable.append(row["id"])
            continue
        corpus.append(
            {
                "answer": row["answer"],
                "id": row["id"],
                "problem": row["question"].strip(),
                "source": SOURCE,
                "tier": TIER_NAMES[row["tier"]],
            }
        )
    corpus.sort(key=lambda record: (record["tier"], record["id"]))
    if len({record["id"] for record in corpus}) != len(corpus):
        raise ValueError("two problems share an identity")
    if len({record["problem"] for record in corpus}) != len(corpus):
        raise ValueError("two problems share a text")
    return corpus, {
        "rows": len(rows),
        "ungradable": len(ungradable),
        "kept": len(corpus),
        "tiers": dict(collections.Counter(record["tier"] for record in corpus)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    corpus, counts = build(arguments.parquet)
    body = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in corpus
    ).encode("utf-8")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # An empty name and no timestamp in the gzip header, so that the container
    # is a function of its contents and the wheel is reproducible.
    with arguments.output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        ) as handle:
            handle.write(body)

    counts["sha256"] = hashlib.sha256(body).hexdigest()
    counts["parquet_sha256"] = hashlib.sha256(arguments.parquet.read_bytes()).hexdigest()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
