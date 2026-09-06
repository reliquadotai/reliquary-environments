import importlib
import pkgutil


def test_package_imports() -> None:
    module = importlib.import_module("reliquary_math")
    assert module.__name__ == "reliquary_math"


def test_no_module_imports_reliquary_core() -> None:
    """The port is a copy, not a dependency. A stray `import reliquary`
    would make the package silently unusable outside the core checkout."""
    import reliquary_math

    for info in pkgutil.walk_packages(
        reliquary_math.__path__, prefix="reliquary_math."
    ):
        source = importlib.util.find_spec(info.name).origin
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "import reliquary\n" not in text
        assert "from reliquary." not in text
        assert "import reliquary." not in text
