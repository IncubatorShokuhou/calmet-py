"""Select a usable PROJ database before importing projection classes."""

from __future__ import annotations

import os
from pathlib import Path
import warnings


def configure_proj_data() -> None:
    """Honor Conda's PROJ data directory when pyproj's bundled copy is stale."""
    configured = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if configured is None or not (Path(configured) / "proj.db").is_file():
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pyproj unable to set PROJ database path.*",
            category=UserWarning,
            module=r"pyproj\.network",
        )
        from pyproj import datadir

    datadir.set_data_dir(configured)


__all__ = ["configure_proj_data"]
