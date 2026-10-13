"""Public CALWRF component interface.

The numerical implementation remains in :mod:`pycalmet._calwrf` so existing
imports keep working.  This module is the stable component boundary used by
the package-level API and by applications that want to construct a
``CalwrfConfig`` directly.
"""

from __future__ import annotations

from pathlib import Path

from .config import CalwrfConfig


def run(config: CalwrfConfig) -> Path:
    """Convert WRF-ARW NetCDF records into CALWRF meteorological data.

    Parameters
    ----------
    config
        Input files, inclusive time interval, optional one-based subdomain,
        and output settings.

    Returns
    -------
    pathlib.Path
        Path to the canonical NetCDF output.
    """
    from ._calwrf import run as _run

    return _run(config)


__all__ = ["run"]
