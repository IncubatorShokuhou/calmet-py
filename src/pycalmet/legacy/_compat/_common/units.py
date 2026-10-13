"""Unit conversions shared by component boundaries."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

FloatArray = TypeVar("FloatArray", float, npt.NDArray[np.floating])


def km_to_m(values: FloatArray) -> FloatArray:
    """Convert kilometres to metres while preserving array shape."""
    return values * 1000.0


def m_to_km(values: FloatArray) -> FloatArray:
    """Convert metres to kilometres while preserving array shape."""
    return values / 1000.0
