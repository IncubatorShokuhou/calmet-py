"""Public CALWRF component interface.

The numerical implementation remains in :mod:`pycalmet._calwrf` so existing
imports keep working.  This module is the stable component boundary used by
the package-level API and by applications that want to construct a
``CalwrfConfig`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .config import CalwrfConfig


def run(config: CalwrfConfig) -> Path:
    """Convert WRF-ARW NetCDF records into CALWRF meteorological data.

    Args:
        config: CALWRF configuration containing WRF input files, the inclusive
            hourly interval, optional one-based horizontal and vertical
            subsets, compression, and canonical or legacy output settings.

    Returns:
        Path to the canonical CALWRF NetCDF product.

    Raises:
        FileNotFoundError: If a configured WRF input does not exist.
        ValueError: If the configuration, source coverage, subdomain, or
            required WRF variables are invalid.
        OSError: If an input cannot be opened or an output cannot be written.

    Notes:
        This call writes ``config.output`` and can also write configured M3D,
        M2D, and listing products. Existing destinations may be replaced.
    """
    from ._calwrf import run as _run

    return _run(config)


__all__: Final[list[str]] = ["run"]
