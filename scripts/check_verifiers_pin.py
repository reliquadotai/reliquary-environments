#!/usr/bin/env python3
"""Check the actual installed native API, not only its reusable version number."""
import importlib.metadata
import json
from pathlib import Path
import tomllib


def main():
    expected = tomllib.loads((Path(__file__).resolve().parents[1] / "compatibility.toml").read_text())["verifiers"]
    distribution = importlib.metadata.distribution("verifiers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    actual = provenance.get("vcs_info", {}).get("commit_id")
    if distribution.version != expected["version"] or actual != expected["source_commit"]:
        raise SystemExit("native qualification requires the exact Verifiers Git pin in compatibility.toml")
    import verifiers.v1 as vf
    if not all(hasattr(vf, name) for name in ("Task", "Taskset", "Trace", "WireTrace", "load_taskset")):
        raise SystemExit("installed Verifiers lacks the qualified v1 API")
    print(json.dumps({"verifiers_version": distribution.version, "source_commit": actual, "api": "v1"}))


if __name__ == "__main__":
    main()
