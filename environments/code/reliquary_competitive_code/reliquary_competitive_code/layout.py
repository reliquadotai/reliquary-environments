"""The names of the published dataset's files and splits, shared by the build
that writes them and the served side that reads them, so that serving never
imports the build."""

SPLITS = ("train", "eval", "qualification")
ROW_GROUP_SIZE = 64
REFERENCES_FILE = "references.parquet"


def problems_file(split: str) -> str:
    return f"problems-{split}.parquet"
