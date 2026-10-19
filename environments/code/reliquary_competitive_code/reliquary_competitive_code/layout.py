"""The names of the published dataset's files and splits, shared by the build
that writes them and the served side that reads them, so that serving never
imports the build."""

SPLITS = ("train", "eval", "qualification")
ROW_GROUP_SIZE = 64
REFERENCES_FILE = "references.parquet"


def problems_file(split: str) -> str:
    return f"problems-{split}.parquet"

# Train is shared between SFT and RL by problem identity: a problem is for SFT
# when its id (16 hex of a statement hash) falls in the first 60% of the id
# space. Train is written sorted by id, so each use is one contiguous range of
# it and a rebuild keeps every problem on its side.
USES = ("sft", "rl")
SFT_ID_BOUND = 2**64 * 6 // 10


def use_of(problem_id: str) -> str:
    return "sft" if int(problem_id, 16) < SFT_ID_BOUND else "rl"


# v1.1 appends its problems after v1's train, all for SFT, so that every v1
# index keeps naming the same problem: such a row says so in its origin
# (`"use": "sft"`). A row without that key is v1's and follows `use_of`. Train
# is then [v1 SFT][v1 RL][appended SFT], each use a few contiguous ranges.
def use_of_row(problem_id: str, origin: dict) -> str:
    use = origin.get("use")
    if use is None:
        return use_of(problem_id)
    if use not in USES:
        raise ValueError(f"unknown use {use!r} for {problem_id}")
    return use
