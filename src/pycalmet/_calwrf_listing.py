"""Small human-readable completion report shared by both CALWRF engines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final, Sequence


def _format_time(value: datetime) -> str:
    """Format a listing timestamp with second precision.

    Args:
        value: Datetime to render. Any timezone information is preserved by
            :meth:`datetime.datetime.isoformat`.

    Returns:
        ISO 8601 text using a space between date and time and including whole
        seconds.
    """
    return value.isoformat(sep=" ", timespec="seconds")


def write_calwrf_listing(
    path: str | Path,
    *,
    title: str,
    wrf_files: Sequence[str | Path],
    requested_start: datetime,
    requested_end: datetime,
    output: str | Path,
    legacy_output: str | Path | None,
    surface_output: str | Path | None,
    written_times: Sequence[datetime],
    nx: int,
    ny: int,
    nz: int,
) -> Path:
    """Write a human-readable completion listing for a successful CALWRF run.

    Args:
        path: Destination UTF-8 text path. Missing parent directories are
            created and an existing file is replaced.
        title: User-supplied CALWRF run title.
        wrf_files: Ordered WRF NetCDF input paths.
        requested_start: Inclusive requested start datetime.
        requested_end: Inclusive requested end datetime.
        output: Canonical CALWRF NetCDF output path.
        legacy_output: Optional legacy M3D output path.
        surface_output: Optional legacy M2D surface output path.
        written_times: Chronological datetimes actually written. At least one
            time is required.
        nx: Output grid size along the west-east axis.
        ny: Output grid size along the south-north axis.
        nz: Number of output mass levels.

    Returns:
        Path to the completed listing file.

    Raises:
        IndexError: If ``written_times`` is empty.
        OSError: If the destination directory or listing cannot be written.
    """
    target: Path = Path(path)
    lines: list[str] = [
        "PYCALMET CALWRF RUN LISTING",
        f"Title: {title}",
        "Status: completed",
        "",
        f"WRF inputs ({len(wrf_files)}):",
        *(f"  - {item}" for item in wrf_files),
        "",
        f"Requested start: {_format_time(requested_start)}",
        f"Requested end:   {_format_time(requested_end)}",
        f"Time records written: {len(written_times)}",
        f"First time written: {_format_time(written_times[0])}",
        f"Last time written:  {_format_time(written_times[-1])}",
        f"Output grid (nx, ny, nz): {nx} x {ny} x {nz}",
        "",
        f"Main NetCDF output: {output}",
    ]
    if legacy_output is not None:
        lines.append(f"Legacy M3D output: {legacy_output}")
    if surface_output is not None:
        lines.append(f"Surface M2D output: {surface_output}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__: Final[list[str]] = ["write_calwrf_listing"]
