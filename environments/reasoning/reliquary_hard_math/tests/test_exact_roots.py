"""Exact rational roots retain integer precision with bounded work."""

import subprocess
import sys
from fractions import Fraction

import pytest

from reliquary_hard_math import grading


def test_exact_roots_and_their_neighbours_across_integer_sizes() -> None:
    # A child process bounds the check even if integer convergence regresses.
    code = """
from fractions import Fraction
from reliquary_hard_math.grading import _exact_root, _NotExact

for degree in (2, 3, 5, 7):
    for numerator in (2, 3**60, (1 << 260) + 1):
        value = Fraction(numerator, 5**60)
        assert _exact_root(value**degree, degree) == value
        for offset in (-1, 1):
            try:
                _exact_root(Fraction(numerator**degree + offset), degree)
            except _NotExact:
                pass
            else:
                raise AssertionError('a neighbouring integer is not an exact root')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


def test_exact_root_work_exhaustion_refuses_the_comparison(monkeypatch) -> None:
    monkeypatch.setattr(grading, "MAX_ROOT_ITERATIONS", 0)
    with pytest.raises(grading._Refused, match="root work limit"):
        grading._exact_root(Fraction(8), 3)
    assert not grading.equivalent(r"\sqrt[3]{8}", "2")
