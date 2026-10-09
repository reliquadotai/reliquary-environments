"""Pack an assembled corpus for publication: gzip each block, pin what it holds.

    uv run python scripts/pack_corpus.py <assembled dir> <dataset dir>

`<assembled dir>` holds `<block>.jsonl` files from `scripts/build/assemble.py`.
Each is recompressed with a zero timestamp (the same rows give the same bytes)
into `<dataset dir>/data/<block>.jsonl.gz`, the layout of the Hub dataset, and
`reliquary_general/_pins.py` is rewritten with the sha256 of every gzip file
(what the Hub reports as its LFS digest), the row counts per split and mode,
and how many of those rows are single-turn. Upload `<dataset dir>`, set
`corpus.REVISION` to the commit it made, then run `scripts/write_goldens.py`
with `RELIQUARY_GENERAL_DATA` unset: the goldens are taken from the Hub.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

from reliquary_general.corpus import BLOCKS, MODES, SPLITS, data_file

ROOT = Path(__file__).resolve().parents[1]
HEAD = ["split", "mode", "single_turn", "key"]


def main(source: Path, target: Path) -> None:
    digests: dict[str, str] = {}
    rows: dict[str, dict[str, dict[str, int]]] = {}
    singles: dict[str, dict[str, dict[str, int]]] = {}
    for block in BLOCKS:
        body = (source / f"{block}.jsonl").read_bytes()
        counts = {split: {mode: 0 for mode in MODES} for split in SPLITS}
        single = {split: {mode: 0 for mode in MODES} for split in SPLITS}
        seen_thinking = False
        for line in body.split(b"\n"):
            if not line:
                continue
            record = json.loads(line)
            assert list(record)[:4] == HEAD, record["key"]
            split, mode = record["split"], record["mode"]
            if mode == "thinking":
                seen_thinking = True
            elif seen_thinking:
                raise SystemExit(f"{block}: a direct row after the thinking run")
            if record["single_turn"]:
                if single[split][mode] != counts[split][mode]:
                    raise SystemExit(f"{block}/{split}/{mode}: a single-turn row after the multi-turn part")
                single[split][mode] += 1
            counts[split][mode] += 1
        rows[block], singles[block] = counts, single
        path = target / data_file(block)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as f:
                f.write(body)
        digests[block] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(block, digests[block][:12], counts, "single-turn", single)
    pins = ROOT / "reliquary_general" / "_pins.py"
    pins.write_text(
        '"""Pinned by scripts/pack_corpus.py: the sha256 of every block\'s gzip file\n'
        "as published, its rows per split and mode, and how many of them open the\n"
        'run as single-turn rows. Written, not edited."""\n\n'
        f"FILE_SHA256 = {json.dumps(digests, indent=4)}\n\n"
        f"BLOCK_ROWS = {json.dumps(rows, indent=4)}\n\n"
        f"BLOCK_SINGLE_TURN = {json.dumps(singles, indent=4)}\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
