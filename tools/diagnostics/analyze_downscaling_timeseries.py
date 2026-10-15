#!/usr/bin/env python3
"""Plot regional-mean WRF/CALMET evolution for all CALMET output hours."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Final, TypeAlias

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.typing as npt

from visualize_downscaling import read_calmet


FloatArray: TypeAlias = npt.NDArray[np.float64]
RowValue: TypeAlias = int | float | str
DIAGNOSTICS_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


def wrf_fields(
    ds: netCDF4.Dataset, hour: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    get = lambda name, t=hour: np.asarray(ds.variables[name][t], dtype=float)
    u, v = get("U10"), get("V10")
    t = get("T2") - 273.15
    q, p = get("Q2"), get("PSFC")
    e = q * p / (0.622 + q)
    es = 611.2 * np.exp(17.67 * t / (t + 243.5))
    rh = np.clip(100 * e / es, 0, 100)
    rain = get("RAINC") + get("RAINNC")
    if hour:
        rain -= get("RAINC", hour - 1) + get("RAINNC", hour - 1)
    return np.hypot(u, v), t, rh, get("PBLH"), np.maximum(rain, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wrf", required=True)
    p.add_argument("--calmet", required=True)
    p.add_argument(
        "--output",
        default=str(DIAGNOSTICS_ROOT / "results/figures/downscaling_timeseries.png"),
    )
    p.add_argument(
        "--csv",
        default=str(DIAGNOSTICS_ROOT / "results/figures/downscaling_timeseries.csv"),
    )
    args = p.parse_args()

    rows: list[dict[str, RowValue]] = []
    with netCDF4.Dataset(args.wrf) as ds:
        # CALMET writes one record for each modeled interval start.  A run
        # bounded by 00 and 12 therefore has records stamped 00..11, while
        # the driving WRF file also contains the final 12 UTC endpoint.
        nt = len(ds.dimensions["Time"]) - 1
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        for hour in range(nt):
            cal = read_calmet(args.calmet, hour)
            extent = (np.nanmin(cal["lon"]), np.nanmax(cal["lon"]),
                      np.nanmin(cal["lat"]), np.nanmax(cal["lat"]))
            mask = ((lon >= extent[0]) & (lon <= extent[1]) &
                    (lat >= extent[2]) & (lat <= extent[3]))
            w = wrf_fields(ds, hour)
            c = (np.hypot(cal["u"], cal["v"]),
                 np.asarray(cal["surface"]["TEMPK"],dtype=np.float64) - 273.15,
                 cal["surface"]["IRH"].astype(float),
                 cal["surface"]["ZI"], cal["surface"]["RMM"])
            stamp = netCDF4.chartostring(ds.variables["Times"][hour]).item()
            row: dict[str, RowValue] = {"hour": hour, "time": str(stamp)}
            for name, wa, ca in zip(("wind", "temperature", "rh", "mixing", "rain"), w, c):
                wrf_mean = float(np.nanmean(np.where(mask, wa, np.nan)))
                calmet_mean = float(np.nanmean(ca))
                row[f"wrf_{name}"] = wrf_mean
                row[f"calmet_{name}"] = calmet_mean
                row[f"delta_{name}"] = calmet_mean - wrf_mean
            rows.append(row)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    hours = np.array([int(r["hour"]) for r in rows])
    specs = (("wind", "Wind speed", "m/s"),
             ("temperature", "Temperature", "°C"),
             ("rh", "Relative humidity", "%"),
             ("mixing", "PBL / mixing height", "m"),
             ("rain", "Hourly precipitation", "mm/h"))
    fig, axes = plt.subplots(5, 2, figsize=(13, 17), constrained_layout=True)
    for row_idx, (key, title, unit) in enumerate(specs):
        wrf_values = np.array([float(r[f"wrf_{key}"]) for r in rows])
        calmet_values = np.array([float(r[f"calmet_{key}"]) for r in rows])
        delta = calmet_values - wrf_values
        axes[row_idx, 0].plot(hours, wrf_values, "o-", label="WRF 9 km")
        axes[row_idx, 0].plot(hours, calmet_values, "o-", label="CALMET 1 km")
        axes[row_idx, 0].set(title=title, ylabel=unit)
        axes[row_idx, 0].grid(alpha=.25); axes[row_idx, 0].legend()
        axes[row_idx, 1].axhline(0, color="black", lw=.8)
        axes[row_idx, 1].plot(hours, delta, "o-", color="tab:red")
        axes[row_idx, 1].set(title=f"CALMET − WRF: {title}", ylabel=unit)
        axes[row_idx, 1].grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel("Forecast hour from 2018-03-18 00 UTC")
    fig.suptitle("Zhejiang original-WRF downscaling: regional means", fontsize=15)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170); plt.close(fig)

    print(f"wrote={out}\nwrote={csv_path}")
    for key, _, unit in specs:
        d = np.array([float(r[f"delta_{key}"]) for r in rows])
        use = d[1:] if key == "rain" else d
        print(f"{key}: mean_delta={np.mean(use):.3f} {unit}, "
              f"range={np.min(use):.3f}..{np.max(use):.3f}")


if __name__ == "__main__":
    main()
