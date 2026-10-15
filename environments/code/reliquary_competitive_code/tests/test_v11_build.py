import json
from argparse import Namespace

import pyarrow.parquet as pq
import pytest

from reliquary_competitive_code.build import extend
from reliquary_competitive_code.build.dedup import ContainmentGuard
from reliquary_competitive_code.build.io import append_dataset, write_dataset
from reliquary_competitive_code.build.validate import MIN_TESTS, Curated, Policy, curate, split_of
from reliquary_competitive_code.corpus import Corpus
from reliquary_competitive_code.environment import CompetitiveCodeEnvironment
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.layout import SPLITS, problems_file
from reliquary_competitive_code.sources.common import PY2TO3, PYTHON_REFERENCE, UPSTREAM_TESTS_ONLY, SourceRow

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = tuple(TestCase(f"{i} {i}\n", f"{2 * i}\n") for i in range(8))
POLICY = Policy(test_bytes=1 << 20, max_tests=6, python_time_factor=3.0, max_time_limit_s=10.0)


def _row(refs=(), *, upstream=True, limit=2.0, converted=frozenset(), tests=TESTS) -> SourceRow:
    return SourceRow("open-r1/codeforces", "1/A", "Print a plus b.", tests, tuple(refs),
                     upstream_tests=upstream, time_limit_s=limit, converted=converted)


def test_upstream_tests_ship_without_a_reference() -> None:
    curated = curate(_row(), POLICY)
    assert isinstance(curated, Curated) and curated.reference_kind == UPSTREAM_TESTS_ONLY
    assert curated.reference == "" and len(curated.tests) == 6 and curated.time_limit_s == 6.0
    assert curate(_row(limit=5.0), POLICY).time_limit_s == 10.0  # the ceiling
    # A slow or crashing Python reference does not stop it either.
    assert curate(_row(["raise SystemExit(1)"]), POLICY).reference_kind == UPSTREAM_TESTS_ONLY
    # Not for v1, nor for tests the upstream does not vouch for.
    assert curate(_row()).reason == "no_passing_reference"
    assert curate(_row(upstream=False), POLICY).reason == "no_passing_reference"
    assert curate(_row(limit=None), POLICY).reason == "no_upstream_time_limit"


def test_an_accepted_submission_with_a_wrong_answer_drops_the_problem() -> None:
    assert curate(_row(["print(0)"]), POLICY).reason == "reference_wrong_answer"
    # ... unless 2to3 made it (integer division and the like).
    assert curate(_row(["print(0)"], converted=frozenset({"print(0)"})), POLICY).reference_kind == UPSTREAM_TESTS_ONLY


def test_a_python_reference_still_validates_and_sets_a_floor_from_upstream() -> None:
    curated = curate(_row([SUM]), POLICY)
    assert curated.reference_kind == PYTHON_REFERENCE and curated.reference == SUM
    assert curated.time_limit_s == 6.0 and len(curated.tests) == 6 and curated.reference_cpu_s is not None
    assert curate(_row([SUM], converted=frozenset({SUM})), POLICY).reference_kind == PY2TO3
    assert curate(_row([SUM, "print(0)"]), POLICY).reason == "references_disagree"
    assert curate(_row([SUM], tests=TESTS[: MIN_TESTS - 1]), POLICY).reason == "too_few_tests"


def test_the_byte_budget_bounds_the_kept_tests() -> None:
    big = tuple(TestCase(f"{i} {i}\n" + " " * 300, f"{2 * i}\n") for i in range(8))
    curated = curate(_row(tests=big), Policy(test_bytes=1200, python_time_factor=3.0))
    assert curated.reason == "too_few_tests"
    curated = curate(_row(tests=big), Policy(test_bytes=2000, python_time_factor=3.0))
    assert len(curated.tests) == 6 and sum(len(t.stdin) + len(t.stdout) for t in curated.tests) <= 2000


def test_containment_guard() -> None:
    legend = " ".join(f"word{i}" for i in range(40))
    guard = ContainmentGuard([legend], threshold=0.3)
    assert guard.contained(legend + " and then something else entirely new here")
    assert not guard.contained(" ".join(f"other{i}" for i in range(40)))
    assert guard.share("too short") == 0.0
    with pytest.raises(ValueError):
        ContainmentGuard([], threshold=0.0)


def _base(tmp_path, n=40):
    rows = [Curated(f"{i * (2**64 // n):016x}", "train", f"Problem {i}.", TESTS[:5], 1.0, SUM, {"source": "t", "upstream_id": str(i)})
            for i in range(n)]
    rows += [Curated(f"{i:015x}e", s, f"Held {s} {i}.", TESTS[:5], 1.0, SUM, {}) for i, s in enumerate(("eval", "qualification"))]
    write_dataset(rows, tmp_path / "base")
    return tmp_path / "base"


def _new(pid, kind=UPSTREAM_TESTS_ONLY):
    reference = "" if kind == UPSTREAM_TESTS_ONLY else SUM
    return Curated(pid, split_of(pid), f"New {pid}.", TESTS[:5], 6.0, reference,
                   {"source": "open-r1/codeforces", "upstream_id": pid, "use": "sft", "reference_kind": kind},
                   reference_kind=kind)


def test_append_keeps_every_base_index_and_adds_one_sft_range(tmp_path) -> None:
    base = _base(tmp_path)
    before = Corpus(base)
    appended = [_new("ffffffffffffff01"), _new("0000000000000a01", PYTHON_REFERENCE)]
    digests = append_dataset(base, appended, tmp_path / "out")
    after = Corpus(tmp_path / "out")
    for split in ("eval", "qualification"):
        assert (tmp_path / "out" / problems_file(split)).read_bytes() == (base / problems_file(split)).read_bytes()
    old, new = before.problems("train"), after.problems("train")
    assert new[: len(old)] == old
    for i in range(len(old)):
        assert after.tests("train", i) == before.tests("train", i)
    # Appended sorted by id, both for SFT, after v1's RL.
    assert [p.problem_id for p in new[len(old):]] == ["0000000000000a01", "ffffffffffffff01"]
    sft, rl = after.use_ranges("sft"), after.use_ranges("rl")
    assert sft == (before.use_range("sft"), range(len(old), len(old) + 2)) and rl == (before.use_range("rl"),)
    with pytest.raises(RuntimeError):
        after.use_range("sft")
    environment = CompetitiveCodeEnvironment("train", corpus=after)
    assert environment.task(len(old) + 1)["metadata"]["use"] == "sft"
    assert environment.reference_completion(len(old) + 1) is None
    assert environment.reference_completion(len(old)).startswith("```python")
    assert environment.task(0) == CompetitiveCodeEnvironment("train", corpus=before).task(0)
    assert set(digests) == {problems_file(s) for s in SPLITS} | {"references.parquet"}
    with pytest.raises(ValueError):
        append_dataset(base, [_new(old[3].problem_id)], tmp_path / "clash")


def test_spill_caps_and_reads_back(tmp_path) -> None:
    spill = extend.Spill(tmp_path, test_bytes=10)
    assert spill.write("p", "a", [TestCase("1\n", "2\n"), TestCase("x" * 20, "y"), TestCase("1\n", "2\n")]) == 2
    spill.write("p", "b", [TestCase("3\n", "4\n")])
    assert spill.read("p") == (TestCase("1\n", "2\n"), TestCase("3\n", "4\n"))


def _entry(group="ccplus", checker="ncmp", **changes):
    entry = {"pid": "p", "group": group, "source": "ccplus", "upstream_id": "AIZU/p1", "statement": "Print a plus b.",
             "contest_date": None, "time_limit_s": 1.0, "rating": None, "upstream_tests": True,
             "members": [{"source": "ccplus", "upstream_id": "AIZU/p1", "checker": checker}]}
    entry.update(changes)
    return entry


def test_rows_pool_python3_then_converted_python2() -> None:
    row = extend.row_of(_entry(), TESTS, {"py3": [SUM], "py2": ["print 1\n", SUM]})
    assert row.references == (SUM, "print(1)\n") and row.converted == frozenset({"print(1)\n"})
    row = extend.row_of(_entry(group="py2"), TESTS, {"py3": [], "py2": [], "pooled": ["print 3\n", "print(4)\n"]})
    assert row.references == ("print(3)\n", "print(4)\n") and row.converted == frozenset({"print(3)\n"})


def test_decide_applies_the_yes_no_and_example_rules() -> None:
    yes = tuple(TestCase(f"{i}\n", "YES\n") for i in range(6))
    row = SourceRow("ccplus", "x", "Say yes.", yes, (), upstream_tests=True, time_limit_s=1.0)
    assert isinstance(extend.decide(_entry(checker="nyesno"), row, POLICY), Curated)
    mixed = SourceRow("ccplus", "x", "Say yes.", (*yes, TestCase("9\n", "YES 1\n")), (), upstream_tests=True, time_limit_s=1.0)
    assert extend.decide(_entry(checker="nyesno"), mixed, POLICY) == "yesno_checker_other_tokens"
    statement = "Print a pair.\n\nInput\n4\n\nOutput\n1 3\n"
    row = SourceRow("ccplus", "x", statement, (TestCase("4\n", "2 2\n"), *TESTS), (), upstream_tests=True, time_limit_s=1.0)
    assert extend.decide(_entry(), row, POLICY) == "example_mismatch"


def test_sample_takes_whole_contests_and_dense_row_groups() -> None:
    entries = [{"pid": str(i), "group": "openr1", "members": [{"contest_id": str(i // 3)}]} for i in range(30)]
    entries += [{"pid": f"c{i}", "group": "ccplus", "members": [{"file": "f", "row_group": i % 4}]} for i in range(20)]
    entries += [{"pid": f"y{i}", "group": "py2", "members": [{}]} for i in range(10)]
    args = Namespace(sample_openr1=6, sample_ccplus=5, sample_py2=3, seed=1)
    out = extend._sample(entries, args)
    openr1 = [e for e in out if e["group"] == "openr1"]
    assert len(openr1) == 6 and len({e["members"][0]["contest_id"] for e in openr1}) == 2
    ccp = [e for e in out if e["group"] == "ccplus"]
    assert len(ccp) == 5 and len({e["members"][0]["row_group"] for e in ccp}) == 1
    assert sum(e["group"] == "py2" for e in out) == 3
    assert extend._sample(entries, Namespace(sample_openr1=None, sample_ccplus=None, sample_py2=None, seed=1)) == entries
