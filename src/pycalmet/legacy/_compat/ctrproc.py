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

    Parameters
    ----------
    config
        Land-use sources, target grid, category rules, and output path.

    Returns
    -------
    pathlib.Path
        Path to the generated land-use product.
    """
    from ._ctgproc import run as _run

    return Path(_run(config).output)


__all__ = ["run"]
