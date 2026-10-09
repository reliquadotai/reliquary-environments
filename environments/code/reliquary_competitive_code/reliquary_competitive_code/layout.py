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
