"""Run the Zhejiang example with each stage's keyword-parameter API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

import pycalmet


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
EXAMPLE: Final[Path] = ROOT / "examples" / "zhejiang"
SRTM_DIRECTORY: Final[Path] = ROOT / "terrain" / "srtm3"
LANDUSE_FILE: Final[Path] = EXAMPLE / "landuse" / "zhejiang_generic.landuse"
OUTPUT: Final[Path] = ROOT / "output" / "zhejiang"
WRF_FILES: Final[tuple[Path, ...]] = tuple(
    EXAMPLE / f"wrfout_d01_2018-03-18_{hour:02d}_00_00.nc"
    for hour in range(3)
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
WRF_START: Final[datetime] = datetime(2018, 3, 18, 0)
WRF_END: Final[datetime] = datetime(2018, 3, 18, 2)
CALMET_END: Final[datetime] = datetime(2018, 3, 18, 1)
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


def main() -> None:
    """Run all stages explicitly without reading traditional control files."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] CALWRF", flush=True)
    meteorology: Path = pycalmet.calwrf(
        inp=None,
        wrf_files=WRF_FILES,
        start=WRF_START,
        end=WRF_END,
        output=OUTPUT / "calwrf.nc",
        create_surface=False,
        title="Create CALWRF data for the Zhejiang example",
    )
    print(f"      -> {meteorology}", flush=True)

    print("[2/5] TERREL", flush=True)
    terrain: Path = pycalmet.terrel(
        inp=None,
        srtm_dir=SRTM_DIRECTORY,
        output=OUTPUT / "terrain.nc",
        sampling="bilinear",
        terrain_sources=TERRAIN_SOURCES,
        grid=GRID,
    )
    print(f"      -> {terrain}", flush=True)

    print("[3/5] CTGPROC", flush=True)
    landuse: Path = pycalmet.ctrproc(
        inp=None,
        landuse_files=(LANDUSE_FILE,),
        output=OUTPUT / "landuse.nc",
        input_type="GEN",
        mesh_density=12,
        grid=GRID,
    )
    print(f"      -> {landuse}", flush=True)

    print("[4/5] MAKEGEO", flush=True)
    geophysical: Path = pycalmet.makegeo(
        inp=None,
        terrain=terrain,
        landuse=landuse,
        output=OUTPUT / "geo.nc",
        title="Demo Application",
        grid=GRID,
    )
    print(f"      -> {geophysical}", flush=True)

    print("[5/5] CALMET", flush=True)
    result: Path = pycalmet.calmet(
        inp=None,
        meteorology=meteorology,
        geophysical=geophysical,
        output=OUTPUT / "calmet.nc",
        start=WRF_START,
        end=CALMET_END,
        grid=GRID,
        noobs=2,
        iprog=14,
        irhprog=1,
        itprog=2,
        coriolis=1.0e-4,
        maximum_mixing_height_m=2500.0,
        maximum_water_mixing_height_m=2500.0,
        precipitation_radius_km=50.0,
        mnmdav=10,
        wind_extrapolation=-1,
        precipitation_station_count=-1,
        include_w_temperature=False,
    )
    print(f"      -> {result}", flush=True)


if __name__ == "__main__":
    main()
