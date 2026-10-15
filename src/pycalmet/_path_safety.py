"""Internal safeguards against destructive input/output path aliasing."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Return whether two path spellings identify the same filesystem entry."""
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
    """Reject CALWRF paths that could overwrite an input or another output."""
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


__all__ = ["validate_calwrf_paths"]
