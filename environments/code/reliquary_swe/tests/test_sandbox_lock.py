"""The image lock a sandbox reads: every image a served split can name,
tag -> repo@sha256 digest, generated from the pinned digest files."""

import importlib.resources
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pin_sandbox_lock  # noqa: E402

DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
PACKAGE = Path(__file__).resolve().parents[1] / "reliquary_swe"


def test_the_shipped_lock_is_the_generated_one():
    shipped = (PACKAGE / "sandbox-images.lock.json").read_text()
    assert shipped == pin_sandbox_lock.render(pin_sandbox_lock.build_lock())
    assert pin_sandbox_lock.main(["--check"]) == 0


def test_the_lock_has_the_shape_a_sandbox_accepts():
    document = json.loads((PACKAGE / "sandbox-images.lock.json").read_text())
    assert set(document) == {"version", "images"} and document["version"] == 1
    assert document["images"]
    for tag, digest in document["images"].items():
        assert tag and not any(c.isspace() for c in tag)
        assert DIGEST.fullmatch(digest)
        if DIGEST.fullmatch(tag):
            assert digest == tag


def test_every_pinned_digest_of_every_corpus_is_locked():
    images = json.loads((PACKAGE / "sandbox-images.lock.json").read_text())["images"]
    for name in pin_sandbox_lock.SOURCES:
        for tag, digest in json.loads((PACKAGE / name).read_text()).items():
            assert images[tag] == digest


def test_a_tag_pinned_twice_with_two_digests_is_refused(tmp_path):
    for name in pin_sandbox_lock.SOURCES:
        (tmp_path / name).write_text("{}")
    (tmp_path / "r2e_digests.json").write_text(json.dumps({"a:1": "a@sha256:" + "1" * 64}))
    (tmp_path / "polyglot_digests.json").write_text(json.dumps({"a:1": "a@sha256:" + "2" * 64}))
    try:
        pin_sandbox_lock.build_lock(tmp_path)
    except ValueError as exc:
        assert "a:1" in str(exc)
    else:
        raise AssertionError("a conflicting pin was accepted")


def test_the_lock_ships_inside_the_package():
    assert importlib.resources.files("reliquary_swe").joinpath(
        "sandbox-images.lock.json").is_file()
