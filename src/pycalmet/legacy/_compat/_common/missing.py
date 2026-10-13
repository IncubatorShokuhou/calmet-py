"""Missing-value helpers used at formatted-data boundaries."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def missing_mask(
    values: npt.ArrayLike,
    sentinel: float,
    *,
    tolerance: float = 0.01,
) -> npt.NDArray[np.bool_]:
    """Return the CALMET-style mask for values at a missing sentinel."""
    data = np.asarray(values)
    return np.asarray(data >= sentinel - tolerance, dtype=np.bool_)
