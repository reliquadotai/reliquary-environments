#!/bin/sh
# A reference fix for MiMo-V2.6's candidate-0036, written for this package's
# goldens (upstream publishes none). Rule-level membership sets held the
# *unexpanded* IOFile objects, so a wildcard-expanded job output never
# matched them; the job now reads each flag off the expanded output itself.
set -eu
cd /app
python3 - <<'PY'
from pathlib import Path

src = Path("vendor/snakemake/src/snakemake")
rules = src / "rules.py"
text = rules.read_text()
for name in ("temp", "protected", "touch"):
    text = text.replace(f"        self.{name}_output = set()\n", "", 1)
    text = text.replace(
        f'            if is_flagged(item, "{name}"):\n'
        f"                if output:\n"
        f"                    self.{name}_output.add(_item)\n",
        "",
        1,
    )
rules.write_text(text)

jobs = src / "jobs.py"
text = jobs.read_text()
for name in ("temp", "protected", "touch"):
    text = text.replace(f"if f_ in self.rule.{name}_output:", f'if is_flagged(f_, "{name}"):', 1)
jobs.write_text(text)
PY
python3 workflow_probe.py
