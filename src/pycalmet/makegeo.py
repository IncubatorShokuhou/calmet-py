"""Public MAKEGEO component interface."""

from __future__ import annotations

from pathlib import Path

from .config import MakegeoConfig


def run(config: MakegeoConfig) -> Path:
    """Generate a geophysical product.

    Parameters
    ----------
    config
        Terrain and land-use inputs, physical-property tables, and output
        settings.

    Returns
    -------
    pathlib.Path
        Path to the primary generated product.

        A snow-processing run can create one dated product per day.  In that
        case this is the first dated product.
    """
    from ._makegeo import run as _run

    return Path(_run(config).output)


__all__ = ["run"]
