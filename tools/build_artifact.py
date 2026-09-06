"""Generate an environment artifact manifest.

The manifest is what the loader verifies before importing a package, so it
must list every shipped file and nothing that varies between checkouts:
bytecode caches and the manifest itself are excluded by construction rather
than by .gitignore, which does not travel inside a wheel.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_EXCLUDED_NAMES = {"artifact.json"}
_EXCLUDED_DIRS = {"__pycache__"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest_sha256(files: dict[str, str]) -> str:
    """Digest over the sorted (path, digest) pairs.

    Sorted so two checkouts of the same tree agree regardless of walk order.
    """
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relative].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_artifact(
    package_root: Path,
    *,
    environment: str,
    contract: str,
    distribution: dict[str, str],
    entrypoints: dict[str, str],
) -> dict[str, Any]:
    package_root = Path(package_root)
    prefix = package_root.name
    files: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _EXCLUDED_NAMES:
            continue
        if _EXCLUDED_DIRS & set(path.relative_to(package_root).parts):
            continue
        relative = f"{prefix}/{path.relative_to(package_root).as_posix()}"
        files[relative] = _sha256_file(path)
    return {
        "schema": "reliquary/environment-artifact/v1",
        "environment": environment,
        "contract": contract,
        "distribution": dict(distribution),
        "entrypoints": dict(entrypoints),
        "source_manifest_sha256": source_manifest_sha256(files),
        "files": files,
    }


def write_artifact(package_root: Path, artifact: dict[str, Any]) -> None:
    target = Path(package_root) / "artifact.json"
    target.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
