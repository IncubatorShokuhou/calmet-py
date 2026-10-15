"""Compatibility import for the legacy CTGPROC spelling."""

from __future__ import annotations

from typing import Final

from .ctrproc import run

__all__: Final[list[str]] = ["run"]
