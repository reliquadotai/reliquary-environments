"""Freeze AutomationBench's tasks, as upstream builds them, into one data file.

Upstream builds its tasks in Python at load time (noise injection seeded by
`example_id`) and hands them to a Hugging Face `Dataset`, which normalises the
rows by adding every missing key as None. We capture the rows before that
step, so `info` is the dict upstream wrote, and ship them in the wheel: the
gateway, the miner and the validator then load the same rows offline, without
`datasets` and without running upstream's builders.

    python -I scripts/export_tasks.py <AutomationBench checkout> <out.jsonl.gz>

Run with `datasets` importable (upstream imports it at module level); the
checkout's commit is recorded in `environment.toml`.
"""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from pathlib import Path

DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr", "simple")


def main(upstream: Path, out: Path) -> None:
    sys.path.insert(0, str(upstream))
    import datasets

    captured: list[dict] = []
    original = datasets.Dataset.from_list

    def capture(rows, *args, **kwargs):
        captured.extend(json.loads(json.dumps(row)) for row in rows)
        return original(rows[:1], *args, **kwargs)

    datasets.Dataset.from_list = capture  # type: ignore[method-assign]
    lines = []
    for domain in DOMAINS:
        module = importlib.import_module(f"automationbench.domains.{domain}.tasks")
        loader = getattr(module, f"get_{domain}_dataset")
        captured.clear()
        loader()
        ids = set()
        for row in captured:
            info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
            if row["example_id"] in ids:
                raise SystemExit(f"{domain}: duplicate example_id {row['example_id']}")
            ids.add(row["example_id"])
            lines.append(
                json.dumps(
                    {
                        "domain": domain,
                        "example_id": row["example_id"],
                        "prompt": row["prompt"],
                        "info": info,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
        print(f"{domain}: {len(captured)}")
    body = ("\n".join(lines) + "\n").encode("utf-8")
    # mtime=0 so the archive bytes are a function of the rows alone.
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(body)
    import hashlib

    print(f"rows {len(lines)} sha256(body) {hashlib.sha256(body).hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
