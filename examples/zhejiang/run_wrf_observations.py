"""Run Zhejiang CALMET with WRF, surface, and upper-air observations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

import pycalmet

from zhejiang_settings import (
    GRID,
    LANDUSE_FILE,
    OUTPUT,
    SRTM_DIRECTORY,
    TERRAIN_SOURCES,
    WRF_FILES,
    observation_inputs,
)


# CALWRF includes both boundary times; CALMET writes the interval [start, end).
RUN_START: Final[datetime] = datetime(2018, 3, 18, 0)
RUN_END: Final[datetime] = datetime(2018, 3, 18, 1)


def main() -> None:
    """Fuse WRF with observations and retain only the CALMET output."""
    surface_data: Path
    upper_air_files: tuple[Path, ...]
    surface_data, upper_air_files = observation_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    result: Path = pycalmet.run_pipeline(
        inp=None,
        calwrf_wrf_files=WRF_FILES,
        calwrf_start=RUN_START,
        calwrf_end=RUN_END,
        calwrf_create_surface=False,
        calwrf_title="Create CALWRF data for the Zhejiang WRF + observations example",
        terrel_srtm_dir=SRTM_DIRECTORY,
        terrel_sampling="bilinear",
        terrel_terrain_sources=TERRAIN_SOURCES,
        terrel_grid=GRID,
        ctgproc_landuse_files=(LANDUSE_FILE,),
        ctgproc_input_type="GLAZAS",
        ctgproc_input_datum="ESR-S",
        ctgproc_mesh_density=12,
        ctgproc_grid=GRID,
        makegeo_title="Zhejiang WRF + observations example",
        makegeo_grid=GRID,
        calmet_start=RUN_START,
        calmet_end=RUN_END,
        calmet_grid=GRID,
        calmet_noobs=0,
        calmet_iprog=14,
        calmet_itprog=0,
        calmet_irhprog=0,
        calmet_cloud_method=4,
        calmet_surface_data=surface_data,
        calmet_upper_air_files=upper_air_files,
        calmet_overwater_flux_method=10,
        calmet_water_temperature_category_min=999,
        calmet_water_temperature_category_max=999,
        calmet_coriolis=1.0e-4,
        calmet_maximum_mixing_height_m=2500.0,
        calmet_maximum_water_mixing_height_m=2500.0,
        calmet_temperature_radius_km=20.0,
        calmet_precipitation_radius_km=50.0,
        calmet_mnmdav=10,
        calmet_wind_extrapolation=-1,
        calmet_upper_air_separation_cells=-1.0,
        calmet_varying_radius=True,
        calmet_stations_per_level=(5,) * 10,
        calmet_precipitation_station_count=-1,
        calmet_base_timezone_label="UTC+0000",
        output=OUTPUT / "calmet-wrf-observations.nc",
    )
    print(f"CALMET (WRF + SURF + UP) -> {result}", flush=True)


if __name__ == "__main__":
    main()
