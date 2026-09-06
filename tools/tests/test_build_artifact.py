import hashlib
import json
from pathlib import Path

from tools.build_artifact import build_artifact, source_manifest_sha256


def test_artifact_lists_every_file_with_its_digest(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    (pkg / "goldens").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "goldens" / "reference.jsonl").write_text('{"index":0}\n', encoding="utf-8")

    artifact = build_artifact(
        pkg,
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )

    assert artifact["schema"] == "reliquary/environment-artifact/v1"
    assert artifact["environment"] == "demo_v1"
    assert set(artifact["files"]) == {
        "demo_pkg/__init__.py",
        "demo_pkg/goldens/reference.jsonl",
    }
    assert artifact["files"]["demo_pkg/__init__.py"] == hashlib.sha256(
        b"x = 1\n"
    ).hexdigest()
    assert artifact["source_manifest_sha256"] == source_manifest_sha256(
        artifact["files"]
    )


def test_artifact_never_lists_itself_or_caches(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "artifact.json").write_text("{}", encoding="utf-8")
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    artifact = build_artifact(
        pkg,
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )

    assert set(artifact["files"]) == {"demo_pkg/__init__.py"}


def test_digest_changes_when_any_file_changes(tmp_path: Path) -> None:
    pkg = tmp_path / "demo_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("a", encoding="utf-8")
    kwargs = dict(
        environment="demo_v1",
        contract="reliquary/answer-json/v1",
        distribution={"name": "demo", "version": "0.1.0a1"},
        entrypoints={"taskset": "demo_pkg:T", "replay": "demo_pkg:E"},
    )
    before = build_artifact(pkg, **kwargs)["source_manifest_sha256"]
    (pkg / "__init__.py").write_text("b", encoding="utf-8")
    after = build_artifact(pkg, **kwargs)["source_manifest_sha256"]
    assert before != after
