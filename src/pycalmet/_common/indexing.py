"""Index conversion helpers for legacy one-based control values."""

from __future__ import annotations


def fortran_inclusive_slice(
    bounds: tuple[int, int] | None,
    size: int,
    *,
    label: str = "Fortran",
) -> slice:
    """Convert an inclusive, one-based interval to a Python slice."""
    if size < 0:
        raise ValueError("slice size must be non-negative")
    if bounds is None:
        return slice(0, size)
    first, last = bounds
    if first < 1 or last < first or last > size:
        raise ValueError(f"{label} range {bounds!r} is outside 1..{size}")
    return slice(first - 1, last)
