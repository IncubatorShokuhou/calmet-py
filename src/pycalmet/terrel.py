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

    Args:
        config: TERREL configuration containing terrain sources, target grid,
            averaging or peak processing, optional coastline/receptor/QA
            products, and the primary output path.

    Returns:
        Path to the generated NetCDF or legacy terrain product.

    Raises:
        FileNotFoundError: If a configured terrain source does not exist.
        ValueError: If a source, grid, sampling mode, or processing option is
            invalid.
        NotImplementedError: If an explicit terrain source type is unsupported.
        OSError: If an input cannot be read or an output cannot be written.

    Notes:
        The primary product is published atomically. The run can also write
        configured save, coastline, receptor, plot, and raw-echo files.
    """
    from ._terrel import run as _run

    return Path(_run(config).output)


__all__ = ["run"]
