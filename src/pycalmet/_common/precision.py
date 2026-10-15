"""Explicit default-REAL conversion and rounding rules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def as_real4(values: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Materialize values at a traditional default-REAL file boundary.

    Args:
        values: Scalar or array-like values written to a legacy interchange
            format.

    Returns:
        NumPy array converted to 32-bit floating point.
    """
    return np.asarray(values, dtype=np.float32)


def fortran_nint(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Round legacy file fields to nearest integers with ties away from zero.

    Args:
        values: Scalar or array-like values interpreted at ``float32`` file
            precision.

    Returns:
        Signed 64-bit integer array using Fortran ``NINT`` tie behavior.

    Notes:
        This helper is limited to traditional CALWRF file serialization. Model
        calculations use ordinary NumPy operations and the configured model
        dtype directly.
    """
    data = as_real4(values)
    half = np.float32(0.5)
    rounded = np.where(
        data >= np.float32(0.0),
        np.floor(np.asarray(data + half, dtype=np.float32)),
        np.ceil(np.asarray(data - half, dtype=np.float32)),
    )
    return np.asarray(rounded, dtype=np.int64)
