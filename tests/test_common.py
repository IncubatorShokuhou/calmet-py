"""Tests for shared Fortran boundary conventions."""

from __future__ import annotations

import pytest

from pycalmet._common import (
    fortran_inclusive_slice,
)


def test_fortran_inclusive_slice_converts_and_validates_bounds() -> None:
    assert fortran_inclusive_slice((2, 4), 5) == slice(1, 4)
    assert fortran_inclusive_slice(None, 5) == slice(0, 5)
    with pytest.raises(ValueError, match=r"outside 1\.\.5"):
        fortran_inclusive_slice((0, 4), 5)
