#!/usr/bin/env python3
"""Build one clean source revision twice and emit an exact wheel release bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "environments/reasoning/reliquary_logic"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def build(output: Path) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("release requires a clean committed source tree")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text())
    compatibility = tomllib.loads((ROOT / "compatibility.toml").read_text())
    name = f'reliquary_logic-{project["project"]["version"]}-py3-none-any.whl'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("release output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="reliquary-logic-build-") as temporary:
        paths = [Path(temporary) / str(index) for index in range(2)]
        for path in paths:
            subprocess.run(["uv", "build", "--wheel", "--out-dir", str(path)], cwd=PACKAGE,
                           env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"}, check=True)
        first, second = (path / name for path in paths)
        if first.read_bytes() != second.read_bytes():
            raise ValueError("wheel builds are not byte-for-byte reproducible")
        digest = hashlib.sha256(first.read_bytes()).hexdigest()
        if digest != compatibility["releases"]["reliquary_logic"]["wheel_sha256"]:
            raise ValueError("built wheel differs from the reviewed release pin")
        with zipfile.ZipFile(first) as archive:
            artifact = json.loads(archive.read("reliquary_logic/artifact.json"))
            for path, expected in artifact["files"].items():
                if hashlib.sha256(archive.read(path)).hexdigest() != expected:
                    raise ValueError(f"artifact file digest mismatch: {path}")
        manifest_digest = hashlib.sha256(canonical(artifact)).hexdigest()
        if manifest_digest != compatibility["releases"]["reliquary_logic"]["artifact_sha256"]:
            raise ValueError("built artifact differs from the reviewed manifest pin")
        source_manifest_digest = hashlib.sha256((PACKAGE / "environment.toml").read_bytes()).hexdigest()
        if source_manifest_digest != artifact["source_manifest_sha256"]:
            raise ValueError("artifact source manifest binding is stale")
        provenance = {
            "schema": "reliquary/environment-release/v1", "source_commit": source_commit,
            "tag": compatibility["releases"]["reliquary_logic"]["tag"],
            "wheel": name, "wheel_sha256": digest, "artifact_sha256": manifest_digest,
            "source_manifest_sha256": source_manifest_digest,
            "verifiers_commit": compatibility["verifiers"]["source_commit"],
            "build_backend": project["build-system"]["requires"],
            "source_date_epoch": 315532800, "reproducible_builds": 2,
        }
        shutil.copyfile(first, output / name)
    (output / "release.json").write_text(json.dumps(provenance, indent=2) + "\n")
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                 for path in sorted(output.iterdir())]
    (output / "SHA256SUMS").write_text("".join(checksums))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output.resolve()), indent=2))
