"""Explicit default-REAL conversion and rounding rules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def as_real4(values: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Materialize values at a GNU Fortran default-REAL boundary."""
    return np.asarray(values, dtype=np.float32)


def fortran_nint(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Round to nearest integer with ties away from zero, like ``NINT``."""
    data = as_real4(values)
    half = np.float32(0.5)
    rounded = np.where(
        data >= np.float32(0.0),
        np.floor(np.asarray(data + half, dtype=np.float32)),
        np.ceil(np.asarray(data - half, dtype=np.float32)),
    )
    return np.asarray(rounded, dtype=np.int64)
