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


from reliquary_math.virtual_parquet import VirtualParquetDataset


def test_virtual_parquet_is_importable_and_lazy() -> None:
    """Constructing must not touch the network: the class exists to avoid a
    bulk download, so an eager constructor would defeat its whole purpose."""
    dataset = VirtualParquetDataset(
        "nvidia/OpenMathInstruct-2",
        "469216e3f46f4dacf476b382e192485ea51a143e",
        columns=["problem", "expected_answer"],
        fs=None,
    )
    assert dataset is not None
