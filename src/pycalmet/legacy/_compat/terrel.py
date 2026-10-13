"""Public TERREL component interface.

The implementation and its detailed QA result remain in
:mod:`pycalmet._terrel`.  The public workflow boundary returns the generated
file so every top-level stage has the same path-oriented contract.
"""

from __future__ import annotations

from pathlib import Path

from .config import TerrelConfig


def run(config: TerrelConfig) -> Path:
    """Generate a gridded terrain product.

    Parameters
    ----------
    config
        Terrain sources, target grid, processing options, and output path.

    Returns
    -------
    pathlib.Path
        Path to the generated terrain product.
    """
    from ._terrel import run as _run

    return Path(_run(config).output)


__all__ = ["run"]
