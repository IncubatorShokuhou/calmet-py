"""Public CALMET component interface."""

from __future__ import annotations

from pathlib import Path

from .config import CalmetConfig


def run(config: CalmetConfig) -> Path:
    """Run the diagnostic meteorological model.

    Parameters
    ----------
    config
        Meteorological and geophysical inputs, model controls, target grid,
        and output settings.

    Returns
    -------
    pathlib.Path
        Path to the generated CALMET product.
    """
    from ._calmet import run as _run

    return Path(_run(config))


__all__ = ["run"]
