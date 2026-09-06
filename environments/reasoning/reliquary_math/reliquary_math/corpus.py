"""The pinned OpenMathInstruct-2 corpus, read one row-group at a time.

`nvidia/OpenMathInstruct-2` publishes the full train split as 32 parquet
shards *and* three convenience subsets (train_1M, train_2M, train_5M) whose
rows are already in the full split. Pointing a loader at a directory instead
of a file list silently yields ~36% duplicates, which is a training bug that
shows up as a plateau rather than as an error. The shard list below is
therefore explicit and pinned by a test.
"""

from __future__ import annotations

from functools import lru_cache

from reliquary_math.virtual_parquet import VirtualParquetDataset

OMI_REPO = "nvidia/OpenMathInstruct-2"
OMI_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"

TRAIN_SHARDS: tuple[str, ...] = tuple(
    f"data/train-{index:05d}-of-00032.parquet" for index in range(32)
)

COLUMNS = ["problem", "expected_answer"]


@lru_cache(maxsize=1)
def load_corpus() -> VirtualParquetDataset:
    # VirtualParquetDataset has no `files=` parameter for an explicit shard
    # list (checked against the vendored constructor). It does provide
    # `filename_prefix`, a basename-prefix filter applied to its own remote
    # directory listing, which is the mechanism Reliquary core itself uses
    # (reliquary/environment/openmathinstruct.py, filename_prefix="train-")
    # to exclude train_1M/2M/5M from the manifest. TRAIN_SHARDS stays the
    # pinned, test-verified statement of what that filter must resolve to.
    return VirtualParquetDataset(
        OMI_REPO,
        OMI_REVISION,
        columns=COLUMNS,
        filename_prefix="train-",
    )


def corpus_length() -> int:
    return len(load_corpus())


def get_problem(index: int) -> dict[str, str]:
    """Row `index` of the pinned corpus, addressed by position.

    Position is the corpus identity: the same index resolves to the same row
    in Reliquary core, which is what makes the goldens meaningful as a
    cross-repo check.
    """
    row = load_corpus().get_row(int(index))
    return {
        "problem": str(row["problem"]),
        "expected_answer": str(row["expected_answer"]),
    }
