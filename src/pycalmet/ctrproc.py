"""Public land-use preprocessing component interface.

The legacy program is named CTGPROC.  ``ctrproc`` is the spelling requested
by the Python workflow API; :mod:`pycalmet.ctgproc` remains as a compatibility
module.
"""

from __future__ import annotations

from pathlib import Path

from .config import CtgprocConfig


def run(config: CtgprocConfig) -> Path:
    """Generate a gridded land-use product.

    Args:
        config: Land-use sources, target grid, category mappings, coastline
            controls, quality-assurance settings, and destination path.

    Returns:
        Path to the generated NetCDF or legacy CTGPROC land-use product.

    Raises:
        FileNotFoundError: If a configured input dataset does not exist.
        ValueError: If an input dataset, grid definition, category mapping, or
            processing option is invalid.
    """
    from ._ctgproc import run as _run

    return Path(_run(config).output)


__all__ = ["run"]
