#!/usr/bin/env python3
"""Compare non-wind WRF and CALMET surface/boundary-layer fields."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.typing as npt

from visualize_downscaling import NumericArray, read_calmet


DIAGNOSTICS_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
FloatArray: TypeAlias = npt.NDArray[np.float64]
FieldName: TypeAlias = Literal["temperature", "rh", "mixing", "rain"]


class WrfSurfaceFields(TypedDict):
    """WRF surface fields used by the diagnostic figure."""

    lat: FloatArray
    lon: FloatArray
    temperature: FloatArray
    rh: FloatArray
    mixing: FloatArray
    rain: FloatArray
    timestamp: str


class PlotFields(TypedDict):
    """Common gridded fields plotted for WRF and CALMET."""

    lat: NumericArray
    lon: NumericArray
    temperature: NumericArray
    rh: NumericArray
    mixing: NumericArray
    rain: NumericArray


def read_wrf(path: str | Path, hour: int) -> WrfSurfaceFields:
    with netCDF4.Dataset(path) as ds:
        def get(name: str, t: int = hour) -> FloatArray:
            """Read one WRF time slice as float64.

            Args:
                name: NetCDF variable name.
                t: Time index to select.

            Returns:
                Selected two-dimensional float64 field.
            """
            return np.asarray(ds.variables[name][t], dtype=np.float64)

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
        raw_timestamp = netCDF4.chartostring(raw_time).item()
        timestamp = (
            raw_timestamp.decode("ascii")
            if isinstance(raw_timestamp, bytes)
            else str(raw_timestamp)
        )
    return {"lat": lat, "lon": lon, "temperature": t2, "rh": rh2,
            "mixing": pblh, "rain": rain, "timestamp": timestamp}


def limits(
    a: npt.ArrayLike, b: npt.ArrayLike, symmetric: bool = False
) -> tuple[float, float]:
    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    values = np.r_[a_values[np.isfinite(a_values)],
                   b_values[np.isfinite(b_values)]]
    lo, hi = np.percentile(values, [1, 99])
    if symmetric:
        bound = float(max(abs(lo), abs(hi))); return -bound, bound
    return float(lo), float(hi)


def summary(
    name: str, wrf: npt.ArrayLike, cal: npt.ArrayLike
) -> None:
    wrf_values = np.asarray(wrf, dtype=np.float64)
    cal_values = np.asarray(cal, dtype=np.float64)
    print(f"{name}: WRF mean={np.nanmean(wrf_values):.3f}, "
          f"CALMET mean={np.nanmean(cal_values):.3f}, "
          f"WRF range={np.nanmin(wrf_values):.3f}..{np.nanmax(wrf_values):.3f}, "
          f"CALMET range={np.nanmin(cal_values):.3f}..{np.nanmax(cal_values):.3f}")


def main() -> None:
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
    cal: PlotFields = {
        "lat": raw["lat"], "lon": raw["lon"],
        "temperature": np.asarray(raw["surface"]["TEMPK"],dtype=np.float64) - 273.15,
        "rh": np.asarray(raw["surface"]["IRH"],dtype=np.float64),
        "mixing": np.asarray(raw["surface"]["ZI"],dtype=np.float64),
        "rain": np.asarray(raw["surface"]["RMM"],dtype=np.float64),
    }
    extent = [float(np.nanmin(cal["lon"])), float(np.nanmax(cal["lon"])),
              float(np.nanmin(cal["lat"])), float(np.nanmax(cal["lat"]))]
    mask = ((wrf["lon"] >= extent[0]) & (wrf["lon"] <= extent[1]) &
            (wrf["lat"] >= extent[2]) & (wrf["lat"] <= extent[3]))

    fields: tuple[tuple[FieldName, str, str, str], ...] = (
        ("temperature", "2 m / surface temperature", "Temperature (°C)", "coolwarm"),
        ("rh", "2 m / surface relative humidity", "Relative humidity (%)", "YlGnBu"),
        ("mixing", "PBL / mixing-layer height", "Height (m)", "viridis"),
        ("rain", "Hourly precipitation", "Precipitation (mm/h)", "Blues"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(14, 19), constrained_layout=True)
    for row, (key, title, unit, cmap) in enumerate(fields):
        w = np.where(mask, np.asarray(wrf[key], dtype=np.float64), np.nan)
        c = np.asarray(cal[key], dtype=np.float64)
        vmin: float
        vmax: float
        if key == "rh":
            vmin, vmax = 0.0, 100.0
        elif key == "rain":
            vmax = max(float(np.nanpercentile(w, 99.5)),
                       float(np.nanpercentile(c, 99.5)), 0.1)
            vmin = 0.0
        else:
            vmin, vmax = limits(w, c)
        plot_fields = (
            (wrf["lon"], wrf["lat"], w, "WRF"),
            (cal["lon"], cal["lat"], c, "CALMET"),
        )
        for col, (longitude, latitude, values, prefix) in enumerate(plot_fields):
            ax = axes[row, col]
            mesh = ax.pcolormesh(longitude, latitude, values,
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
