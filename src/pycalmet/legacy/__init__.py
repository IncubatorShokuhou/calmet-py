"""Opt-in compatibility API for the original Fortran-matching implementation.

This namespace preserves the historical numerical path, including staged
single-precision arithmetic and optional M3D interchange. New applications
should use the top-level :mod:`pycalmet` API.
"""
from __future__ import annotations

from typing import Final

from ._compat.api import calmet, calwrf, ctgproc, ctrproc, makegeo, terrel
from ._compat.readlst import readlst
from ._compat.config import (
    CalmetConfig, CalwrfConfig, CtgprocConfig, CtgprocSource, GridConfig,
    MakegeoConfig, TerrelConfig, TerrelSource,
)

__all__: Final[list[str]] = [
    "calwrf", "terrel", "ctrproc", "ctgproc", "makegeo", "calmet", "readlst",
    "GridConfig", "CalwrfConfig", "TerrelConfig", "TerrelSource",
    "CtgprocConfig", "CtgprocSource", "MakegeoConfig", "CalmetConfig",
]
