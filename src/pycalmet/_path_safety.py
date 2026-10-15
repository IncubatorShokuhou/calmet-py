"""Internal safeguards against destructive input/output path aliasing."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Final, Sequence


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two path spellings identify the same filesystem entry.

    Args:
        first: First existing or prospective filesystem path.
        second: Second existing or prospective filesystem path.

    Returns:
        ``True`` when both paths identify the same entry after resolving
        symlinks and non-existent path components.
    """
    left = Path(first)
    right = Path(second)
    try:
        if left.samefile(right):
            return True
    except OSError:
        pass
    return left.resolve(strict=False) == right.resolve(strict=False)


def validate_calwrf_paths(
    wrf_files: Sequence[str | Path],
    output: str | Path,
    legacy_output: str | Path | None,
    surface_output: str | Path | None,
    listing_output: str | Path | None,
) -> None:
    """Reject CALWRF paths that could overwrite an input or another output.

    Args:
        wrf_files: Source WRF NetCDF paths that must remain unchanged.
        output: Primary CALWRF NetCDF output path.
        legacy_output: Optional traditional M3D output path.
        surface_output: Optional traditional M2D output path.
        listing_output: Optional human-readable listing path.

    Raises:
        ValueError: If two outputs alias one another or an output aliases a
            WRF input, including through a symlink or normalized path.
    """
    outputs = tuple(
        (name, path)
        for name, path in (
            ("output", output),
            ("legacy_output", legacy_output),
            ("surface_output", surface_output),
            ("listing_output", listing_output),
        )
        if path is not None
    )
    for (first_name, first_path), (second_name, second_path) in combinations(outputs, 2):
        if _same_path(first_path, second_path):
            raise ValueError(
                "CALWRF output paths collide: "
                f"{first_name} and {second_name} both refer to {first_path}"
            )
    for output_name, output_path in outputs:
        for index, input_path in enumerate(wrf_files):
            if _same_path(output_path, input_path):
                raise ValueError(
                    f"CALWRF {output_name} conflicts with wrf_files[{index}]: "
                    f"both refer to {output_path}"
                )


__all__: Final[list[str]] = ["validate_calwrf_paths"]
