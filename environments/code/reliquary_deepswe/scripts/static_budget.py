"""Flag tasks whose test runner allows more time than a grading step has, read statically.

A gateway grades within 810 s (restore and scoring start included). Upstream's runners
set their own per-suite ceilings (Go's `-timeout`, `timeout N`); a task whose ceilings add
up past the step can only finish in time if its suites end well before them. This reads
the ceilings, it measures nothing: `gold_check.py` measures. Prints one line per flagged
task, then a count.

    python scripts/static_budget.py [--step 810]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reliquary_deepswe import overlay, source  # noqa: E402

GO_TIMEOUT = re.compile(r"-timeout[ =](\d+)([smh])")
TIMEOUT_CMD = re.compile(r"(?:^|[\s;&|(])timeout\s+(?:-[a-zA-Z]+\s+\S+\s+)*(\d+)([smh]?)\s")
UNIT = {"s": 1, "m": 60, "h": 3600, "": 1}


def ceilings(script: str) -> list[int]:
    body = script.split("# >>> RUN TESTS", 1)[-1].split("# >>> END RUN TESTS", 1)[0]
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    found = [int(n) * UNIT[u] for n, u in GO_TIMEOUT.findall(code)]
    found += [int(n) * UNIT[u] for n, u in TIMEOUT_CMD.findall(code)]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=810.0)
    args = parser.parse_args()
    flagged = 0
    for task_id in source.task_ids():
        script = overlay.staged_tests(task_id)["test.sh"].decode()
        found = ceilings(script)
        if sum(found) > args.step:
            flagged += 1
            language = source.task_toml(task_id)["metadata"]["language"]
            print(f"{task_id:52} {language:10} ceilings {found} = {sum(found)} s")
    print(f"{flagged} of {len(source.task_ids())} tasks allow more than {args.step:g} s")


if __name__ == "__main__":
    main()
