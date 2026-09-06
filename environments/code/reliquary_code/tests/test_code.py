import importlib
import pkgutil

import pytest

from reliquary_code import corpus
from reliquary_code.extraction import extract_python


def test_no_module_imports_reliquary_core() -> None:
    import reliquary_code

    for info in pkgutil.walk_packages(
        reliquary_code.__path__, prefix="reliquary_code."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "from reliquary." not in text
        assert "import reliquary." not in text


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OCI_REPO == "R0mAI/opencodeinstruct-curated"
    assert corpus.OCI_REVISION == "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"


def test_plain_fence_is_extracted() -> None:
    assert "def f" in extract_python("```\ndef f():\n    return 1\n```")


def test_language_tagged_fence_is_extracted() -> None:
    """A ```python opener is a block opener like any other. Six separate
    incidents came from treating it as content."""
    assert "def f" in extract_python("```python\ndef f():\n    return 1\n```")


@pytest.mark.parametrize("tag", ["py", "python3"])
def test_any_language_tag_opens_a_block(tag: str) -> None:
    assert "def f" in extract_python(f"```{tag}\ndef f():\n    return 1\n```")


def test_uppercase_language_tag_is_not_recognised() -> None:
    """The brief for this test proposed parametrizing over ["py", "python3",
    "Python", "PYTHON"], asserting all four open a block. Verified directly
    against an unmodified core checkout (reliquary/environment/
    opencodeinstruct.py @ 10c2a4d9, calling `_select_python_span` with
    protocol_version=5): `_FENCE_RE`'s tag group `(?:python3?|py)?` has no
    `re.IGNORECASE`, so "Python" and "PYTHON" fail to open a block there too
    — this is core's actual behavior at the pinned commit, not a copying
    error in this port. Pinning the verified behavior here rather than the
    brief's assumed one, so a future core fix shows up as a visible,
    intentional diff instead of this port silently drifting from core."""
    assert extract_python("```Python\ndef f():\n    return 1\n```") == ""


def test_the_last_parsing_block_wins() -> None:
    """A model that shows a broken attempt then a working one is graded on
    the working one."""
    completion = (
        "```python\ndef f(:\n```\n"
        "then the real answer\n"
        "```python\ndef f():\n    return 2\n```"
    )
    assert "return 2" in extract_python(completion)


def test_unparseable_output_yields_empty() -> None:
    assert extract_python("no code here").strip() == ""


def test_entry_function_is_preferred_when_named() -> None:
    completion = (
        "```python\ndef helper():\n    return 1\n```\n"
        "```python\ndef solve():\n    return 2\n```"
    )
    assert "def solve" in extract_python(completion, entry_name="solve")
