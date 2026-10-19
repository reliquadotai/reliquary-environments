"""The image lock a sandbox reads (env norm, rule 7): the tmax base image, by digest."""

import importlib.resources
import json
import re

from reliquary_terminal import tmax_select

DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


def lock():
    path = importlib.resources.files("reliquary_terminal").joinpath("sandbox-images.lock.json")
    return json.loads(path.read_text())


def test_the_lock_pins_the_manifests_base_image():
    document = lock()
    assert set(document) == {"version", "images"} and document["version"] == 1
    base = tmax_select.load_manifest()["base_image"]
    assert DIGEST.fullmatch(base) and document["images"] == {base: base}


def test_the_manifest_keeps_the_validated_image_id():
    manifest = tmax_select.load_manifest()
    assert manifest["base_image_id"] == (
        "sha256:3d10148ef334ddea2ad5c7f1ea5d541fe8aa7877fe97a35bd6f706238e253f15")
