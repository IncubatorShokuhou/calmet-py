"""Shared paths and geophysical settings for the Zhejiang examples."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pycalmet


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
EXAMPLE: Final[Path] = ROOT / "examples" / "zhejiang"
DATA: Final[Path] = ROOT / "data"
SRTM_DIRECTORY: Final[Path] = DATA / "terrain" / "srtm3"
LANDUSE_FILE: Final[Path] = (
    DATA / "landuse" / "usgs-glcc" / "EAUSGS1.IMG"
)
OBSERVATION_DIRECTORY: Final[Path] = (
    DATA / "observations" / "zhejiang-20180318"
)
CALMET_OBSERVATION_DIRECTORY: Final[Path] = (
    OBSERVATION_DIRECTORY / "calmet"
)
SURFACE_OBSERVATIONS: Final[Path] = (
    CALMET_OBSERVATION_DIRECTORY / "SURF.DAT"
)
UPPER_AIR_OBSERVATIONS: Final[tuple[Path, ...]] = tuple(
    CALMET_OBSERVATION_DIRECTORY / f"UP_{station}.DAT"
    for station in (
        "CHM00058362",
        "CHM00058457",
        "CHM00058633",
        "CHM00058665",
    )
)
OUTPUT: Final[Path] = ROOT / "output" / "zhejiang"
WRF_FILES: Final[tuple[Path, ...]] = tuple(
    EXAMPLE / f"wrfout_d01_2018-03-18_{hour:02d}_00_00.nc"
    for hour in range(2)
)
SRTM_FILES: Final[tuple[Path, ...]] = tuple(
    SRTM_DIRECTORY / f"N{latitude:02d}E{longitude:03d}.hgt"
    for latitude in range(26, 32)
    for longitude in range(117, 124)
)
TERRAIN_SOURCES: Final[tuple[pycalmet.TerrelSource, ...]] = tuple(
    pycalmet.TerrelSource(
        path=path,
        input_type="SRTM3",
        input_datum="WGS-96",
    )
    for path in SRTM_FILES
)
GRID: Final[pycalmet.GridConfig] = pycalmet.GridConfig(
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
)


def observation_inputs() -> tuple[Path, tuple[Path, ...]]:
    """Locate the prepared Zhejiang surface and upper-air observations.

    The preparation utility writes geographic ``LL``/``WGS-84`` station
    metadata inside each CALMET text file.  The runtime therefore obtains
    station IDs, coordinates, elevations, and anemometer heights from the
    files instead of duplicating them in Python control records.

    Returns:
        Pair containing ``SURF.DAT`` and the four explicitly selected
        ``UP_*.DAT`` paths in stable station order.

    Raises:
        FileNotFoundError: If the surface file or any of the four selected
            upper-air files is absent. This usually means the local
            observation fetch/preparation workflow has not completed.
    """
    if not SURFACE_OBSERVATIONS.is_file():
        raise FileNotFoundError(
            "missing prepared Zhejiang surface observations: "
            f"{SURFACE_OBSERVATIONS}"
        )
    missing_upper_air: tuple[Path, ...] = tuple(
        path for path in UPPER_AIR_OBSERVATIONS if not path.is_file()
    )
    if missing_upper_air:
        raise FileNotFoundError(
            "missing prepared Zhejiang upper-air observation(s): "
            + ", ".join(str(path) for path in missing_upper_air)
        )
    return SURFACE_OBSERVATIONS, UPPER_AIR_OBSERVATIONS


def build_geophysical(workspace: Path) -> Path:
    """Build Zhejiang terrain, land-use, and GEO datasets in ``workspace``.

    Args:
        workspace: Existing temporary directory in which intermediate
            NetCDF files may be created.

    Returns:
        Path to the temporary ``geo.nc`` file consumed by CALMET.

    Raises:
        OSError: If an input cannot be read or an output cannot be written.
        ValueError: If the source data or target grid are invalid.
    """
    terrain: Path = pycalmet.terrel(
        inp=None,
        srtm_dir=SRTM_DIRECTORY,
        output=workspace / "terrain.nc",
        sampling="bilinear",
        terrain_sources=TERRAIN_SOURCES,
        grid=GRID,
    )
    landuse: Path = pycalmet.ctrproc(
        inp=None,
        landuse_files=(LANDUSE_FILE,),
        output=workspace / "landuse.nc",
        input_type="GLAZAS",
        input_datum="ESR-S",
        mesh_density=12,
        grid=GRID,
    )
    return pycalmet.makegeo(
        inp=None,
        terrain=terrain,
        landuse=landuse,
        output=workspace / "geo.nc",
        title="Zhejiang observation example",
        grid=GRID,
    )
