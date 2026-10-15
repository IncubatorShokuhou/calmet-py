#!/usr/bin/env python3
"""Compare WRF and CALMET terrain/10 m wind fields for one output hour."""

from __future__ import annotations

import argparse
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer


NumericArray: TypeAlias = (
    npt.NDArray[np.float32]
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int32]
)
DIAGNOSTICS_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


class CalmetFields(TypedDict):
    """Fields decoded from one legacy CALMET output hour."""

    lon: NumericArray
    lat: NumericArray
    elevation: NumericArray
    u: NumericArray
    v: NumericArray
    timestamp: int | None
    height: float
    start: tuple[int, int, int, int, int]
    surface: dict[str, NumericArray]


class WrfFields(TypedDict):
    """WRF fields used by the downscaling comparison."""

    lon: NumericArray
    lat: NumericArray
    elevation: NumericArray
    u: NumericArray
    v: NumericArray
    timestamp: str


def records(path: str | Path) -> Iterator[bytes]:
    with open(path, "rb") as stream:
        while marker := stream.read(4):
            size = struct.unpack("<i", marker)[0]
            payload = stream.read(size)
            closing = struct.unpack("<i", stream.read(4))[0]
            if closing != size:
                raise ValueError(f"Bad Fortran record marker in {path}")
            yield payload


def read_calmet(path: str | Path, hour_index: int) -> CalmetFields:
    rec = records(path)
    first = next(rec)
    if not first.startswith(b"CALMET.DAT"):
        raise ValueError("Not a CALMET.DAT file")
    ncomment = struct.unpack("<i", next(rec))[0]
    for _ in range(ncomment):
        next(rec)
    header = next(rec)

    start = struct.unpack_from("<5i", header, 0)
    nx, ny, nz = struct.unpack_from("<3i", header, 56)
    dgrid, xorigin, yorigin = struct.unpack_from("<3f", header, 68)
    pmap = header[116:124].decode().strip()
    lat0, lon0, lat1, lat2 = struct.unpack_from("<4f", header, 160)

    zface_rec = next(rec)
    zfaces = np.frombuffer(zface_rec, dtype="<f4", offset=24, count=nz + 1).copy()
    next(rec)  # roughness
    next(rec)  # land use
    elev_rec = next(rec)
    next(rec)  # leaf area index

    def grid(
        payload: bytes, dtype: npt.DTypeLike = "<f4"
    ) -> NumericArray:
        values = np.frombuffer(payload, dtype=dtype, offset=24, count=nx * ny)
        return values.reshape((nx, ny), order="F").T.copy()

    elevation = grid(elev_rec)
    u: NumericArray | None = None
    v: NumericArray | None = None
    timestamp: int | None = None
    surface: dict[str, NumericArray] = {}
    records_per_hour = 2 * nz + 11
    for index in range((hour_index + 1) * records_per_hour):
        payload = next(rec)
        within = index % records_per_hour
        current_hour = index // records_per_hour
        label = payload[:8].decode().strip()
        if current_hour == hour_index and within == 0:
            u = grid(payload)
            timestamp = struct.unpack_from("<i", payload, 8)[0]
        elif current_hour == hour_index and within == 1:
            v = grid(payload)
        elif current_hour == hour_index and within >= 2 * nz:
            dtype = "<i4" if label in {"IRH", "IPCODE"} else "<f4"
            surface[label] = grid(payload, dtype=dtype)

    if u is None or v is None:
        raise IndexError("Requested CALMET hour is absent")

    if pmap != "LCC":
        raise ValueError(f"Unsupported CALMET projection: {pmap}")
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} "
        f"+lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x = xorigin + (np.arange(nx) + 0.5) * dgrid
    y = yorigin + (np.arange(ny) + 0.5) * dgrid
    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)
    return {
        "lon": lon, "lat": lat, "elevation": elevation,
        "u": u, "v": v, "timestamp": timestamp,
        "height": float(0.5 * (zfaces[0] + zfaces[1])), "start": start,
        "surface": surface,
    }


def read_wrf(path: str | Path, hour_index: int) -> WrfFields:
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables["XLAT"][hour_index])
        lon = np.asarray(ds.variables["XLONG"][hour_index])
        elevation = np.asarray(ds.variables["HGT"][hour_index])
        u = np.asarray(ds.variables["U10"][hour_index])
        v = np.asarray(ds.variables["V10"][hour_index])
        raw_time = ds.variables["Times"][hour_index]
        raw_timestamp = netCDF4.chartostring(raw_time).item()
        timestamp = (
            raw_timestamp.decode("ascii")
            if isinstance(raw_timestamp, bytes)
            else str(raw_timestamp)
        )
    return {"lon": lon, "lat": lat, "elevation": elevation,
            "u": u, "v": v, "timestamp": timestamp}


def stats(
    speed: npt.ArrayLike, spacing_km: float
) -> tuple[float, float, float, float]:
    values = np.asarray(speed, dtype=np.float64)
    valid = values[np.isfinite(values)]
    roughness = 0.5 * (
        np.nanmean(np.abs(np.diff(values, axis=0))) +
        np.nanmean(np.abs(np.diff(values, axis=1)))
    ) / spacing_km
    return (
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.percentile(valid, 95)),
        float(roughness),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrf", required=True)
    parser.add_argument("--calmet", required=True)
    parser.add_argument("--hour", type=int, default=2)
    parser.add_argument(
        "--output",
        default=str(DIAGNOSTICS_ROOT / "results/figures/downscaling_comparison.png"),
    )
    args = parser.parse_args()

    wrf = read_wrf(args.wrf, args.hour)
    cal = read_calmet(args.calmet, args.hour)
    ws_wrf = np.hypot(wrf["u"], wrf["v"])
    ws_cal = np.hypot(cal["u"], cal["v"])

    extent = [float(np.nanmin(cal["lon"])), float(np.nanmax(cal["lon"])),
              float(np.nanmin(cal["lat"])), float(np.nanmax(cal["lat"]))]
    wrf_mask = ((wrf["lon"] >= extent[0]) & (wrf["lon"] <= extent[1]) &
                (wrf["lat"] >= extent[2]) & (wrf["lat"] <= extent[3]))
    terrain_max = np.nanpercentile(np.r_[wrf["elevation"][wrf_mask],
                                         cal["elevation"].ravel()], 99)
    wind_max = np.nanpercentile(np.r_[ws_wrf[wrf_mask], ws_cal.ravel()], 98)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    terrain = [(wrf, "WRF terrain (9 km)"), (cal, "CALMET terrain (1 km)")]
    for ax, (data, title) in zip(axes[0], terrain):
        mesh = ax.pcolormesh(data["lon"], data["lat"], data["elevation"],
                             shading="auto", cmap="terrain", vmin=0,
                             vmax=terrain_max, rasterized=True)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude",
               xlim=extent[:2], ylim=extent[2:])
        fig.colorbar(mesh, ax=ax, label="Elevation (m)")

    wind = [(wrf, ws_wrf, "WRF 10 m wind", 5),
            (cal, ws_cal, f"CALMET {cal['height']:.0f} m wind", 24)]
    for ax, (data, speed, title, stride) in zip(axes[1], wind):
        mesh = ax.pcolormesh(data["lon"], data["lat"], speed, shading="auto",
                             cmap="turbo", vmin=0, vmax=wind_max, rasterized=True)
        sl = (slice(None, None, stride), slice(None, None, stride))
        ax.quiver(data["lon"][sl], data["lat"][sl], data["u"][sl], data["v"][sl],
                  color="black", alpha=0.65, scale=180, width=0.002)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude",
               xlim=extent[:2], ylim=extent[2:])
        fig.colorbar(mesh, ax=ax, label="Wind speed (m/s)")

    sw = stats(np.where(wrf_mask, ws_wrf, np.nan), 9.0)
    sc = stats(ws_cal, 1.0)
    fig.suptitle(
        f"Zhejiang downscaling at {wrf['timestamp']} UTC\n"
        f"WRF mean/std/P95={sw[0]:.2f}/{sw[1]:.2f}/{sw[2]:.2f} m/s, "
        f"CALMET={sc[0]:.2f}/{sc[1]:.2f}/{sc[2]:.2f} m/s",
        fontsize=15,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote={output}")
    print(f"wrf_mean={sw[0]:.4f} wrf_std={sw[1]:.4f} wrf_p95={sw[2]:.4f} wrf_gradient={sw[3]:.5f}")
    print(f"calmet_mean={sc[0]:.4f} calmet_std={sc[1]:.4f} calmet_p95={sc[2]:.4f} calmet_gradient={sc[3]:.5f}")
    print(f"extent={extent}")


if __name__ == "__main__":
    main()
