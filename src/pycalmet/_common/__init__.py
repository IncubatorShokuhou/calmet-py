"""Shared adapters for recurring Fortran-to-NumPy boundary rules."""

from __future__ import annotations

from .indexing import fortran_inclusive_slice
from .missing import missing_mask
from .precision import as_real4, fortran_nint
from .units import km_to_m, m_to_km

__all__ = [
    "as_real4",
    "fortran_inclusive_slice",
    "fortran_nint",
    "km_to_m",
    "m_to_km",
    "missing_mask",
]
