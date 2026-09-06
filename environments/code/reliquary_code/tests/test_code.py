import importlib
import pkgutil

from reliquary_code import corpus


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
