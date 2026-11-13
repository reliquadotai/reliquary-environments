"""The verifier overlay keeps test.sh's installs, never the tests' own scratch."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_verifier_images.py"
spec = importlib.util.spec_from_file_location("build_verifier_images", SCRIPT)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

DIFF = """C /root
A /root/.local/bin/uv
A /root/.cache/uv/archive-v0/x/pytest.py
A /root/.cache/pip/http/y
C /usr/bin
A /usr/bin/curl
C /var/lib/dpkg/status
A /var/lib/apt/lists/archive.ubuntu.com_InRelease
A /var/log/apt/history.log
A /tests/test_outputs.py
A /logs/verifier/ctrf.json
A /tmp/tmp123
A /app/.pytest_cache/v/cache
A /app/ocaml-original
A /app/ocaml-original/Makefile
A /app/ocaml/_build/x.o
D /etc/removed
"""


def test_installs_kept_scratch_dropped():
    assert build.overlay_paths("regex-log", DIFF) == [
        "./root", "./root/.local/bin/uv", "./root/.cache/uv/archive-v0/x/pytest.py",
        "./usr/bin", "./usr/bin/curl", "./var/lib/dpkg/status"]


def test_workdir_downloads_kept_where_test_sh_puts_them():
    kept = build.overlay_paths("fix-ocaml-gc", DIFF)
    assert "./app/ocaml-original" in kept and "./app/ocaml-original/Makefile" in kept
    assert "./app/ocaml/_build/x.o" not in kept


def test_tests_are_stubbed_while_recording():
    import io
    import tarfile

    from reliquary_terminal_bench import source

    stubbed = tarfile.open(fileobj=io.BytesIO(build.stubbed_tests("train-fasttext")))
    original = tarfile.open(fileobj=io.BytesIO(source.tests_archive("train-fasttext")))
    assert stubbed.getnames() == original.getnames()
    for member in stubbed.getmembers():
        data = stubbed.extractfile(member).read()
        if member.name.startswith("test_") and member.name.endswith(".py"):
            assert data == build.STUB
        else:
            assert data == original.extractfile(member.name).read()
