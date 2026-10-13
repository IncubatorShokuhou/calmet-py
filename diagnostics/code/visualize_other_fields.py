#!/usr/bin/env python3
"""Compare non-wind WRF and CALMET surface/boundary-layer fields."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from visualize_downscaling import read_calmet


DIAGNOSTICS_ROOT = Path(__file__).resolve().parents[1]


def read_wrf(path, hour):
    with netCDF4.Dataset(path) as ds:
        get = lambda name, t=hour: np.asarray(ds.variables[name][t], dtype=float)
        lat, lon = get("XLAT"), get("XLONG")
        t2 = get("T2") - 273.15
        q2, psfc = get("Q2"), get("PSFC")
        vapor_pressure = q2 * psfc / (0.622 + q2)
        saturation = 611.2 * np.exp(17.67 * t2 / (t2 + 243.5))
        rh2 = np.clip(100.0 * vapor_pressure / saturation, 0, 100)
        pblh = get("PBLH")
        rain_now = get("RAINC") + get("RAINNC")
        if hour:
            rain_prev = get("RAINC", hour - 1) + get("RAINNC", hour - 1)
            rain = np.maximum(rain_now - rain_prev, 0)
        else:
            rain = rain_now
        raw_time = ds.variables["Times"][hour]
        timestamp = netCDF4.chartostring(raw_time).item()
    return {"lat": lat, "lon": lon, "temperature": t2, "rh": rh2,
            "mixing": pblh, "rain": rain, "timestamp": timestamp}


def limits(a, b, symmetric=False):
    values = np.r_[a[np.isfinite(a)], b[np.isfinite(b)]]
    lo, hi = np.percentile(values, [1, 99])
    if symmetric:
        bound = max(abs(lo), abs(hi)); return -bound, bound
    return lo, hi


def summary(name, wrf, cal):
    print(f"{name}: WRF mean={np.nanmean(wrf):.3f}, "
          f"CALMET mean={np.nanmean(cal):.3f}, "
          f"WRF range={np.nanmin(wrf):.3f}..{np.nanmax(wrf):.3f}, "
          f"CALMET range={np.nanmin(cal):.3f}..{np.nanmax(cal):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wrf", required=True)
    p.add_argument("--calmet", required=True)
    p.add_argument("--hour", type=int, default=2)
    p.add_argument(
        "--output",
        default=str(DIAGNOSTICS_ROOT / "results/figures/other_fields_comparison.png"),
    )
    args = p.parse_args()

    wrf = read_wrf(args.wrf, args.hour)
    raw = read_calmet(args.calmet, args.hour)
    cal = {
        "lat": raw["lat"], "lon": raw["lon"],
        "temperature": raw["surface"]["TEMPK"] - 273.15,
        "rh": raw["surface"]["IRH"],
        "mixing": raw["surface"]["ZI"],
        "rain": raw["surface"]["RMM"],
    }
    extent = [float(np.nanmin(cal["lon"])), float(np.nanmax(cal["lon"])),
              float(np.nanmin(cal["lat"])), float(np.nanmax(cal["lat"]))]
    mask = ((wrf["lon"] >= extent[0]) & (wrf["lon"] <= extent[1]) &
            (wrf["lat"] >= extent[2]) & (wrf["lat"] <= extent[3]))

    fields = [
        ("temperature", "2 m / surface temperature", "Temperature (°C)", "coolwarm"),
        ("rh", "2 m / surface relative humidity", "Relative humidity (%)", "YlGnBu"),
        ("mixing", "PBL / mixing-layer height", "Height (m)", "viridis"),
        ("rain", "Hourly precipitation", "Precipitation (mm/h)", "Blues"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 19), constrained_layout=True)
    for row, (key, title, unit, cmap) in enumerate(fields):
        w = np.where(mask, wrf[key], np.nan)
        c = cal[key]
        if key == "rh":
            vmin, vmax = 0, 100
        elif key == "rain":
            vmax = max(float(np.nanpercentile(w, 99.5)),
                       float(np.nanpercentile(c, 99.5)), 0.1)
            vmin = 0
        else:
            vmin, vmax = limits(w, c)
        for col, (data, values, prefix) in enumerate(((wrf, w, "WRF"), (cal, c, "CALMET"))):
            ax = axes[row, col]
            mesh = ax.pcolormesh(data["lon"], data["lat"], values,
                                 shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                                 rasterized=True)
            ax.set(title=f"{prefix}: {title}", xlabel="Longitude", ylabel="Latitude",
                   xlim=extent[:2], ylim=extent[2:])
            fig.colorbar(mesh, ax=ax, label=unit)
        summary(key, w, c)

    fig.suptitle(f"Non-wind downscaling fields at {wrf['timestamp']} UTC", fontsize=16)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
