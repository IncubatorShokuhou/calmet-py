"""Public MAKEGEO component interface."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .config import MakegeoConfig


def run(config: MakegeoConfig) -> Path:
    """Generate a geophysical product.

    Args:
        config: Terrain and land-use inputs, target grid, surface-property
            tables, optional snow controls, and output settings.

    Returns:
        Path to the primary generated NetCDF or legacy geophysical product.

        A snow-processing run can create one dated product per day.  In that
        case this is the first dated product.

    Raises:
        FileNotFoundError: If a configured terrain, land-use, snow, or
            property-table input does not exist.
        ValueError: If an input product, target grid, surface-property table,
            category split, or snow-processing option is invalid.
    """
    from ._makegeo import run as _run

    return Path(_run(config).output)


__all__: Final[list[str]] = ["run"]
