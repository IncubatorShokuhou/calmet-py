"""Cycle-neutral state types shared by CALMET meteorology adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


Float32Array: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SourceThermalState:
    """RDMM5 potential temperatures retained from the last daylight hour.

    Attributes:
        pt20: ``float32 (source_y, source_x)`` lower reference-level potential
            temperature in kelvins.
        pt30: ``float32 (source_y, source_x)`` upper reference-level potential
            temperature in kelvins.
    """

    pt20: Float32Array
    pt30: Float32Array


# Preserve the established public class location while allowing both the
# solver and MM4 adapter to import the implementation without a cycle.
SourceThermalState.__module__ = "pycalmet.calmet_solver"
