#!/usr/bin/env python3
"""Compare PyCALMET NetCDF fields with formatted CALWRF m3d samples."""

import argparse
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycalmet.legacy_m3d import selected_cells  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True, help="PyCALMET CALWRF NetCDF")
    p.add_argument("--fortran", required=True, help="Fortran CALWRF m3d")
    p.add_argument("--time", type=int, default=0)
    args = p.parse_args()

    with nc.Dataset(args.python) as ds:
        ny, nx = len(ds.dimensions["y"]), len(ds.dimensions["x"])
        points = {(args.time, 0, 0), (args.time, ny // 2, nx // 2),
                  (args.time, ny - 1, nx - 1)}
        legacy = selected_cells(args.fortran, points)
        failures = []
        for key, old in legacy.items():
            t, j, i = key
            # Legacy surface: PS(hPa), rain(cm), snow, SW, LW, T2, Q2(g/kg),
            # direction, speed, SST.
            new_surface = np.array([
                ds["surface_pressure"][t, j, i] / 100.0,
                ds["precipitation"][t, j, i] / 10.0,
                ds["snow_cover"][t, j, i], ds["shortwave_down"][t, j, i],
                ds["longwave_down"][t, j, i], ds["temperature_2m"][t, j, i],
                ds["qvapor_2m"][t, j, i] * 1000.0,
                ds["wind_direction_10m"][t, j, i], ds["wind_speed_10m"][t, j, i],
                ds["sst"][t, j, i],
            ])
            surface_tol = np.array([.051, .006, .01, .051, .051, .051, .006, .051, .051, .051])
            sd = np.abs(new_surface - old.surface)
            if np.any(sd > surface_tol):
                failures.append(f"{key} surface max normalized error={np.max(sd/surface_tol):.2f}")

            # Legacy levels: pressure rounded hPa, height m, temperature 0.1 K,
            # direction degree, speed 0.1 m/s, W 0.01 m/s, RH integer, Q g/kg.
            new_levels = np.column_stack([
                ds["pressure"][t, :, j, i], ds["height"][t, :, j, i],
                ds["temperature"][t, :, j, i], ds["wind_direction"][t, :, j, i],
                ds["wind_speed"][t, :, j, i], ds["vertical_velocity"][t, :, j, i],
                np.zeros(len(ds.dimensions["level"])), ds["qvapor"][t, :, j, i] * 1000.0,
            ])
            # RH is derived only while formatting m3d and is excluded here.
            cols = np.array([0, 1, 2, 3, 4, 5, 7])
            tol = np.array([.501, .501, .051, .501, .051, .006, .006])
            delta = np.abs(new_levels[:, cols] - old.levels[:, cols])
            if np.any(delta > tol):
                failures.append(f"{key} levels max normalized error={np.max(delta/tol):.2f}")
        if failures:
            raise SystemExit("CALWRF regression failed:\n" + "\n".join(failures))
        print(f"CALWRF regression passed for {len(points)} cells × {len(ds.dimensions['level'])} levels")


if __name__ == "__main__":
    main()

