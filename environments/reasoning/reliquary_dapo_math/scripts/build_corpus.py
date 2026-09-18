"""Rebuild the packaged corpus from the upstream parquet.

Not run by CI and not needed to use the environment: the corpus ships inside
the wheel and is pinned by its digest. It exists so that the derivation is a
program rather than a paragraph — which rows were dropped, and on what rule,
is answerable by running this again and comparing the digest it prints with
the one `corpus.py` holds.

    uv run --no-project --with pyarrow python scripts/build_corpus.py \
        --parquet dapo-math-17k.parquet \
        --output reliquary_dapo_math/data/problems.jsonl.gz

The parquet is `data/dapo-math-17k.parquet` from
https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

# DAPO wraps every problem in its own answer contract, at both ends. Both are
# stripped so that only the problem is left to carry: keeping the `Answer:`
# line would measure two changes at once, a corpus the policy has not seen and
# a convention it was not trained on.
PREAMBLE = re.compile(
    r"^Solve the following math problem step by step\..*?"
    r"\$Answer is the answer to the problem\.\s*",
    re.S,
)
POSTAMBLE = re.compile(r"\s*Remember to put your answer.*$", re.S)

# The largest integer a double carries without loss. A label above it cannot be
# told apart from its own float image, and five of them upstream plainly are
# one: 22099999999999998951424 is exactly what `float(2.21e22)` prints.
MAX_ANSWER_MAGNITUDE = (1 << 53) - 1


def problem_text(prompt: list[dict[str, str]]) -> str:
    """One row's problem, with DAPO's contract taken off both ends."""
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError("a row must carry exactly one user message")
    raw = prompt[0]["content"]
    stripped = PREAMBLE.sub("", raw)
    if stripped == raw:
        raise ValueError(f"row does not carry DAPO's preamble: {raw[:80]!r}")
    text = POSTAMBLE.sub("", stripped)
    if text == stripped:
        raise ValueError(f"row does not carry DAPO's postamble: {raw[-80:]!r}")
    return text.strip()


def identity(text: str) -> str:
    """A problem's name is a digest of the text the task will serve.

    Upstream's own `extra_info.index` is a uuid per row group rather than per
    problem — 628 problems carry two of them — so it would name the same
    problem twice and reintroduce exactly the duplication this build removes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build(parquet: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    import pyarrow.parquet as pq

    first: dict[str, str] = {}
    truths: dict[str, set[int]] = collections.defaultdict(set)
    rows = 0

    for batch in pq.ParquetFile(parquet).iter_batches(
        batch_size=20_000, columns=["prompt", "reward_model"]
    ):
        columns = batch.to_pydict()
        for prompt, reward_model in zip(
            columns["prompt"], columns["reward_model"], strict=True
        ):
            rows += 1
            text = problem_text(prompt)
            # Whitespace is the one difference that is not a difference: 67
            # problems appear twice under layouts that only a diff can tell
            # apart, and an environment indexing both would serve one problem
            # from two indices again, on a smaller scale.
            key = " ".join(text.split())
            first.setdefault(key, text)
            truths[key].add(int(str(reward_model["ground_truth"]).strip()))

    corpus: list[dict[str, object]] = []
    ambiguous = oversized = 0
    for key, text in first.items():
        if len(truths[key]) > 1:
            # Twelve problems carry two answers across their copies, and four
            # of those pair a real answer with -1. Which copy a grader happened
            # to read would decide the reward, so neither is kept.
            ambiguous += 1
            continue
        (answer,) = truths[key]
        if abs(answer) > MAX_ANSWER_MAGNITUDE:
            oversized += 1
            continue
        corpus.append({"answer": answer, "id": identity(text), "problem": text})

    # Sorted by identity rather than left in upstream's row order: the file is
    # then a function of the problems it holds and not of how they arrived.
    corpus.sort(key=lambda record: record["id"])
    if len({record["id"] for record in corpus}) != len(corpus):
        raise ValueError("two problems share an identity")
    return corpus, {
        "rows": rows,
        "distinct": len(first),
        "ambiguous": ambiguous,
        "oversized": oversized,
        "kept": len(corpus),
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
    # An empty name and no timestamp in the gzip header: left to itself gzip
    # stamps both, and a rebuild written to another path would then differ in
    # bytes from a corpus identical in every problem. The pinned digest covers
    # the body rather than the container, but a container that is a function of
    # its contents keeps the wheel reproducible too.
    with arguments.output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        ) as handle:
            handle.write(body)

    counts["sha256"] = hashlib.sha256(body).hexdigest()
    counts["parquet_sha256"] = hashlib.sha256(
        arguments.parquet.read_bytes()
    ).hexdigest()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
