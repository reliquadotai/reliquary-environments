import importlib.util
import json
import re
from importlib.resources import files
from pathlib import Path

import pytest

from reliquary_swe import corpus

DIGEST = re.compile(r"\A[^@\s]+@sha256:[0-9a-f]{64}\Z")


def test_every_default_image_has_a_pinned_digest():
    pinned = json.loads(files("reliquary_swe").joinpath("swesmith_digests.json").read_text())
    for image in corpus.swesmith_image_rank()[: corpus.DEFAULT_SWESMITH_IMAGES]:
        assert DIGEST.match(pinned[image]), image


def test_swesmith_rows_reference_images_by_digest():
    row = corpus.load_swesmith_rows(1)[0]
    assert DIGEST.match(row.image)


def test_num_images_beyond_the_pins_is_refused_before_loading(monkeypatch):
    def no_load(*_a, **_k):
        raise AssertionError("the dataset must not be loaded")

    monkeypatch.setattr(corpus, "load_dataset", no_load)
    limit = corpus.pinned_swesmith_images()
    assert limit == corpus.DEFAULT_SWESMITH_IMAGES
    with pytest.raises(ValueError, match=rf"num_images={limit + 1} exceeds the {limit} .*pin_swesmith_digests"):
        corpus.load_swesmith_rows(limit + 1)


def test_taskset_config_refuses_num_images_beyond_the_pins():
    from reliquary_swe.taskset import SweTasksetConfig

    assert SweTasksetConfig(id="reliquary-swe").num_images == corpus.DEFAULT_SWESMITH_IMAGES
    with pytest.raises(ValueError, match="pin_swesmith_digests"):
        SweTasksetConfig(id="reliquary-swe", num_images=corpus.pinned_swesmith_images() + 1)


def test_swesmith_row_outside_the_pinned_images_is_a_clear_error():
    unpinned = corpus.swesmith_image_rank()[corpus.pinned_swesmith_images()]
    name, hf_split, revision = corpus._SOURCES["train"]
    dataset = corpus.load_dataset(name, split=hf_split, revision=revision or None)
    for row in dataset:
        if row["image_name"] == unpinned and row["problem_statement"].strip():
            break
    with pytest.raises(ValueError, match="outside the 20 SWE-smith images pinned"):
        corpus.swesmith_row(row["instance_id"])


def _pin_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "pin_swesmith_digests.py"
    spec = importlib.util.spec_from_file_location("pin_swesmith_digests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, body=b"", headers=None):
        self._body, self.headers = body, headers or {}

    def read(self, *_):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_pin_script_refuses_a_missing_digest_and_sets_a_timeout(monkeypatch):
    script = _pin_script()
    timeouts = []

    def urlopen(req, timeout=None):
        timeouts.append(timeout)
        if isinstance(req, str):
            return _Response(b'{"token": "t"}')
        return _Response(headers={})

    monkeypatch.setattr(script.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="no Docker-Content-Digest"):
        script.digest("jyangballin/x:latest")
    assert timeouts and all(t == script.TIMEOUT for t in timeouts)
