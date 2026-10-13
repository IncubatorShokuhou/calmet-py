"""Tests for shared Fortran boundary conventions."""

from __future__ import annotations

import numpy as np
import pytest

from pycalmet._common import (
    as_real4,
    fortran_inclusive_slice,
    fortran_nint,
    km_to_m,
    m_to_km,
    missing_mask,
)


def test_fortran_inclusive_slice_converts_and_validates_bounds() -> None:
    assert fortran_inclusive_slice((2, 4), 5) == slice(1, 4)
    assert fortran_inclusive_slice(None, 5) == slice(0, 5)
    with pytest.raises(ValueError, match=r"outside 1\.\.5"):
        fortran_inclusive_slice((0, 4), 5)


def test_real4_nint_missing_and_unit_helpers() -> None:
    values = as_real4([-1.5, -1.49, 1.49, 1.5])
    assert values.dtype == np.float32
    np.testing.assert_array_equal(fortran_nint(values), [-2, -1, 1, 2])
    np.testing.assert_array_equal(missing_mask([9998.98, 9998.99], 9999.0), [False, True])
    np.testing.assert_allclose(m_to_km(km_to_m(np.array([1.25]))), [1.25])
