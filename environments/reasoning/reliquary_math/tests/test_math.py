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


from reliquary_math import corpus


def test_shard_list_is_exactly_the_full_train_split() -> None:
    """The subset directories (train_1M/2M/5M) duplicate rows already in the
    full split. Naming the 32 files explicitly makes that class of mistake
    impossible instead of guarding it with a flag."""
    assert len(corpus.TRAIN_SHARDS) == 32
    assert corpus.TRAIN_SHARDS[0] == "data/train-00000-of-00032.parquet"
    assert corpus.TRAIN_SHARDS[-1] == "data/train-00031-of-00032.parquet"
    assert all(name.startswith("data/train-") for name in corpus.TRAIN_SHARDS)
    assert not any("train_1M" in name for name in corpus.TRAIN_SHARDS)
    assert not any("train_2M" in name for name in corpus.TRAIN_SHARDS)
    assert not any("train_5M" in name for name in corpus.TRAIN_SHARDS)
    assert len(set(corpus.TRAIN_SHARDS)) == 32


def test_pins_are_the_ones_core_uses() -> None:
    assert corpus.OMI_REPO == "nvidia/OpenMathInstruct-2"
    assert corpus.OMI_REVISION == "469216e3f46f4dacf476b382e192485ea51a143e"
