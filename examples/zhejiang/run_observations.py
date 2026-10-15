"""Run Zhejiang CALMET from surface and upper-air observations only."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import pycalmet

from zhejiang_settings import (
    GRID,
    OUTPUT,
    build_geophysical,
    observation_inputs,
)


RUN_START: Final[datetime] = datetime(2018, 3, 18, 0)
RUN_END: Final[datetime] = datetime(2018, 3, 18, 1)


def main() -> None:
    """Run the observation-only case and retain only its CALMET output."""
    surface_data: Path
    upper_air_files: tuple[Path, ...]
    surface_data, upper_air_files = observation_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="pycalmet-zhejiang-observations-") as temporary:
        geophysical: Path = build_geophysical(Path(temporary))
        result: Path = pycalmet.calmet(
            inp=None,
            meteorology=None,
            geophysical=geophysical,
            output=OUTPUT / "calmet-observations.nc",
            start=RUN_START,
            end=RUN_END,
            grid=GRID,
            noobs=0,
            iprog=0,
            itprog=0,
            irhprog=0,
            cloud_method=1,
            surface_data=surface_data,
            upper_air_files=upper_air_files,
            precipitation_station_count=0,
            overwater_flux_method=10,
            water_temperature_category_min=999,
            water_temperature_category_max=999,
            coriolis=1.0e-4,
            maximum_mixing_height_m=2500.0,
            maximum_water_mixing_height_m=2500.0,
            precipitation_radius_km=50.0,
            temperature_radius_km=20.0,
            mnmdav=10,
            wind_extrapolation=-1,
            upper_air_separation_cells=-1.0,
            varying_radius=True,
            stations_per_level=(5,) * 10,
            base_timezone_label="UTC+0000",
        )
    print(f"CALMET (SURF + UP) -> {result}", flush=True)


if __name__ == "__main__":
    main()
