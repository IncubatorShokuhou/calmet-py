"""Run the Zhejiang pipeline with typed Python parameters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

import pycalmet


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
EXAMPLE: Final[Path] = ROOT / "examples" / "zhejiang"
DATA: Final[Path] = ROOT / "data"
SRTM_DIRECTORY: Final[Path] = DATA / "terrain" / "srtm3"
LANDUSE_FILE: Final[Path] = DATA / "landuse" / "usgs-glcc" / "EAUSGS1.IMG"
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
# CALWRF includes both boundary times; CALMET writes the interval [start, end).
RUN_START: Final[datetime] = datetime(2018, 3, 18, 0)
RUN_END: Final[datetime] = datetime(2018, 3, 18, 1)
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
    """Run all stages from keyword parameters and retain only CALMET output."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result: Path = pycalmet.run_pipeline(
        inp=None,
        calwrf_wrf_files=WRF_FILES,
        calwrf_start=RUN_START,
        calwrf_end=RUN_END,
        calwrf_create_surface=False,
        calwrf_title="Create CALWRF data for the Zhejiang example",
        terrel_srtm_dir=SRTM_DIRECTORY,
        terrel_sampling="bilinear",
        terrel_terrain_sources=TERRAIN_SOURCES,
        terrel_grid=GRID,
        ctgproc_landuse_files=(LANDUSE_FILE,),
        ctgproc_input_type="GLAZAS",
        ctgproc_input_datum="ESR-S",
        ctgproc_mesh_density=12,
        ctgproc_grid=GRID,
        makegeo_title="Demo Application",
        makegeo_grid=GRID,
        calmet_start=RUN_START,
        calmet_end=RUN_END,
        calmet_grid=GRID,
        calmet_noobs=2,
        calmet_iprog=14,
        calmet_irhprog=1,
        calmet_itprog=2,
        calmet_coriolis=1.0e-4,
        calmet_maximum_mixing_height_m=2500.0,
        calmet_maximum_water_mixing_height_m=2500.0,
        calmet_precipitation_radius_km=50.0,
        calmet_mnmdav=10,
        calmet_wind_extrapolation=-1,
        calmet_precipitation_station_count=-1,
        calmet_include_w_temperature=False,
        output=OUTPUT / "calmet.nc",
    )
    print(f"CALMET -> {result}", flush=True)


if __name__ == "__main__":
    main()
