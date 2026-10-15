"""Public CALMET component interface."""

from __future__ import annotations

from pathlib import Path

from .config import CalmetConfig


def run(config: CalmetConfig) -> Path:
    """Run the diagnostic meteorological model.

    Args:
        config: Meteorological and geophysical inputs, model controls, target
            grid, and output settings.

    Returns:
        Path to the generated CALMET product selected by ``config``.

    Raises:
        TypeError: If an input does not match its configured data protocol.
        ValueError: If configuration values, input metadata, time coverage,
            grid geometry, or meteorological fields are invalid.
    """
    from ._calmet import run as _run

    return Path(_run(config))


__all__ = ["run"]
