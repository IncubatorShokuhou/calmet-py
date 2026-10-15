"""Shared adapters for recurring Fortran-to-NumPy boundary rules."""

from __future__ import annotations

from typing import Final

from .indexing import fortran_inclusive_slice
from .precision import as_real4, fortran_nint

__all__: Final[list[str]] = [
    "as_real4",
    "fortran_inclusive_slice",
    "fortran_nint",
]
