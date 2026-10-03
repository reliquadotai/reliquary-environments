import json
import re
from importlib.resources import files

from reliquary_swe import corpus

DIGEST = re.compile(r"\A[^@\s]+@sha256:[0-9a-f]{64}\Z")


def test_every_default_image_has_a_pinned_digest():
    pinned = json.loads(files("reliquary_swe").joinpath("swesmith_digests.json").read_text())
    for image in corpus.swesmith_image_rank()[: corpus.DEFAULT_SWESMITH_IMAGES]:
        assert DIGEST.match(pinned[image]), image


def test_swesmith_rows_reference_images_by_digest():
    row = corpus.load_swesmith_rows(1)[0]
    assert DIGEST.match(row.image)
