"""Index conversion helpers for legacy one-based control values."""

from __future__ import annotations


def fortran_inclusive_slice(
    bounds: tuple[int, int] | None,
    size: int,
    *,
    label: str = "Fortran",
) -> slice:
    """Convert an inclusive one-based interval to a Python slice.

    Args:
        bounds: Inclusive ``(first, last)`` indices, or ``None`` for the full
            dimension.
        size: Length of the indexed dimension.
        label: Source-field name included in validation errors.

    Returns:
        Zero-based half-open slice selecting the same elements.

    Raises:
        ValueError: If ``size`` is negative or ``bounds`` falls outside the
            one-based dimension.
    """
    if size < 0:
        raise ValueError("slice size must be non-negative")
    if bounds is None:
        return slice(0, size)
    first, last = bounds
    if first < 1 or last < first or last > size:
        raise ValueError(f"{label} range {bounds!r} is outside 1..{size}")
    return slice(first - 1, last)
