"""Terrain preprocessing for standard rasters and CALMET source formats.

Rasterio handles standard SRTM and GeoTIFF inputs.  Format-specific readers
remain for historical terrain products that GDAL cannot reliably identify.
"""

from __future__ import annotations

from dataclasses import dataclass,replace
from pathlib import Path
import re
import struct
import tempfile

import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.errors import RasterioIOError
import tifffile

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from .config import GridConfig, TerrelConfig, TerrelSource
from ._ctgproc import _coastline_types, _read_coastline_bln
from .datums import NIMA_DATABASE_DATE
from .gshhs import extract_gshhs
from .legacy_text import gfortran_real_pair
from .projections import (grid_crs, project_from_crs, project_geographic,
                          project_glaz_to_lcc,
                          transform_grid, unproject_to_crs)

_HGT_NAME = re.compile(r"^([NS])(\d{2})([EW])(\d{3})\.hgt$", re.IGNORECASE)
_GTOPO_NAME = re.compile(r"^([EW])(\d{3})([NS])(\d{2})(?:\..*)?$", re.IGNORECASE)
_3CD_NAME = re.compile(r"(?:^|.*?)(\d{2})(\d{3})\.3cd$", re.IGNORECASE)

COMPUTE_DTYPE = np.float32
COORD_DTYPE = np.float64
ACCUM_DTYPE = np.float64


@dataclass(frozen=True, slots=True)
class TerrainResult:
    output: Path
    tiles: int
    samples_inside_grid: int
    populated_cells: int
    minimum_m: float
    maximum_m: float
    average_hits: int = 0
    low_hit_threshold: int = 0
    high_hit_threshold: int = 0
    low_hit_cells: int = 0
    high_hit_cells: int = 0


def _hit_statistics(counts, threshold_percent):
    """Reproduce TERREL's integer QA summary for non-empty cells."""
    if not 0 <= threshold_percent <= 100:
        raise ValueError("TERREL ITHRES must be in the range 0..100")
    hits = np.asarray(counts, dtype=np.int64).ravel()
    populated = hits[hits > 0]
    average = int(populated.sum() // populated.size) if populated.size else 0
    low = int(np.floor(np.float64(0.01) * np.float64(average * threshold_percent)
                       + np.float64(0.5)))
    high = int(np.floor(np.float64(0.01) *
                        np.float64(average * (200 - threshold_percent))
                        + np.float64(0.5)))
    return (average, low, high, int(np.count_nonzero((hits > 0) & (hits < low))),
            int(np.count_nonzero(hits > high)))


@dataclass(frozen=True, slots=True)
class UsgslaSpec:
    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float
    latitude_origin: float
    longitude_origin: float


@dataclass(frozen=True, slots=True)
class DemHeader:
    codes: tuple[int, int, int, int]
    coordinate_unit: int
    elevation_unit: int
    corners: np.ndarray
    resolution: tuple[float, float, float]
    profile_count: tuple[int, int]
    horizontal_datum: int


@dataclass(slots=True)
class _ReceptorState:
    coordinates_km: np.ndarray
    heights_m: np.ndarray
    receptor_heights_m: np.ndarray
    interpolate: bool
    radius_km: float
    vertices_x: np.ndarray
    vertices_y: np.ndarray
    vertices_z: np.ndarray
    distances: np.ndarray

    def add(self, x_m, y_m, elevation_m) -> None:
        """Vectorised port of XYELEV, batched to bound temporary memory."""
        x=np.asarray(x_m,dtype=COORD_DTYPE).ravel()*.001
        y=np.asarray(y_m,dtype=COORD_DTYPE).ravel()*.001
        z=np.asarray(elevation_m,dtype=COMPUTE_DTYPE).ravel()
        valid = z >= COMPUTE_DTYPE(-32000.)
        x, y, z = x[valid], y[valid], z[valid]
        if not x.size:
            return
        if not self.interpolate:
            tree=cKDTree(np.column_stack((x,y)))
            groups=tree.query_ball_point(self.coordinates_km,self.radius_km)
            for receptor,indices in enumerate(groups):
                if indices:
                    self.heights_m[receptor]=max(
                        self.heights_m[receptor],np.max(z[indices]))
            return
        for start in range(0, self.coordinates_km.shape[0], 256):
            stop = min(start + 256, self.coordinates_km.shape[0])
            receptor_x = self.coordinates_km[start:stop, 0, None]
            receptor_y = self.coordinates_km[start:stop, 1, None]
            dx=np.asarray(x[None,:]-receptor_x,dtype=COORD_DTYPE)
            dy=np.asarray(y[None,:]-receptor_y,dtype=COORD_DTYPE)
            distance=np.hypot(dx,dy)
            quadrant = np.where(
                dx <= 0.0,
                np.where(dy <= 0.0, 0, 3),
                np.where(dy <= 0.0, 1, 2),
            )
            for corner in range(4):
                corner_distance = np.where(quadrant == corner, distance, np.inf)
                index = np.argmin(corner_distance, axis=1)
                nearest = corner_distance[np.arange(stop - start), index]
                replace = nearest < self.distances[corner, start:stop]
                if replace.any():
                    target = np.flatnonzero(replace) + start
                    source = index[replace]
                    self.vertices_x[corner, target] = x[source]
                    self.vertices_y[corner, target] = y[source]
                    self.vertices_z[corner, target] = z[source]
                    self.distances[corner, target] = nearest[replace]

    def finish(self) -> np.ndarray:
        if self.interpolate:
            for index, (x, y) in enumerate(self.coordinates_km):
                self.heights_m[index] = _interp2d(
                    self.vertices_x[:, index], self.vertices_y[:, index],
                    self.vertices_z[:, index], self.distances[:, index],
                    x, y, self.radius_km,
                )
        return self.heights_m


class _RawEchoWriter:
    """Bounded-memory RAWECHO stream with atomic publication."""

    def __init__(self,path,grid,datum):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        handle=tempfile.NamedTemporaryFile(
            dir=self.path.parent,prefix=self.path.name+".",delete=False,mode="w",
            encoding="ascii",newline="\n",
        )
        self.temporary=Path(handle.name); self.stream=handle
        _write_legacy_header(
            self.stream,grid,datum,dataset="RAWECHO.DAT",version="1.0"
        )

    def write(self,x_m,y_m,elevation_m,selected):
        x=np.asarray(x_m).ravel()[selected]*.001
        y=np.asarray(y_m).ravel()[selected]*.001
        z=np.asarray(elevation_m).ravel()[selected]
        for x_value,y_value,z_value in zip(x,y,z):
            self.stream.write(f"{x_value:11.3f}{y_value:11.3f}{z_value:12.3f}\n")

    def finish(self):
        if self.stream is None:
            return
        self.stream.close(); self.stream=None
        self.temporary.replace(self.path)

    def __del__(self):
        stream=getattr(self,"stream",None)
        if stream is not None:
            stream.close()
        temporary=getattr(self,"temporary",None)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _TerrainFilter:
    polygons: tuple
    coastal: bool
    noise_replacement: np.ndarray
    noise_minimum_m: np.ndarray
    missing_replacement: np.ndarray
    default_elevation_m: np.ndarray
    raw_echo: _RawEchoWriter | None = None

    def types(self, x_km, y_km):
        if not self.coastal:
            return np.ones(np.asarray(x_km).shape, dtype=np.int8)
        return _coastline_types(x_km, y_km, self.polygons)

    def extracted(self, x_m, y_m, elevation_m, domain):
        elevation=np.asarray(elevation_m,dtype=COMPUTE_DTYPE).copy()
        selected=np.flatnonzero(domain)
        if not selected.size:
            return elevation
        if self.raw_echo is not None:
            self.raw_echo.write(x_m,y_m,elevation_m,selected)
        kinds=self.types(
            np.asarray(x_m,dtype=COORD_DTYPE).ravel()[selected]*.001,
            np.asarray(y_m,dtype=COORD_DTYPE).ravel()[selected]*.001)
        values=elevation.ravel()[selected]
        defaults=self.default_elevation_m[kinds]
        missing_rule=self.missing_replacement[kinds]
        always=missing_rule==3
        values[always]=defaults[always]
        remaining=~always
        noise_rule=self.noise_replacement[kinds]
        noisy=(remaining&(noise_rule>0)&(values<self.noise_minimum_m[kinds])&
               (values>COMPUTE_DTYPE(-32000.)))
        values[noisy&(noise_rule==1)]=COMPUTE_DTYPE(-32001.)
        values[noisy&(noise_rule==2)]=self.noise_minimum_m[kinds][noisy&(noise_rule==2)]
        values[noisy&(noise_rule==3)]=defaults[noisy&(noise_rule==3)]
        missing=remaining&(values<COMPUTE_DTYPE(-32000.))
        values[missing&(missing_rule==2)]=defaults[missing&(missing_rule==2)]
        elevation.ravel()[selected]=values
        return elevation

    def output(self, values, kinds):
        result=np.asarray(values,dtype=COMPUTE_DTYPE).copy()
        kinds=np.asarray(kinds,dtype=np.int8)
        defaults=self.default_elevation_m[kinds]
        missing_rule=self.missing_replacement[kinds]
        result[missing_rule==3]=defaults[missing_rule==3]
        noise_rule=self.noise_replacement[kinds]
        noisy=((missing_rule!=3)&(noise_rule>0)&
               (result<self.noise_minimum_m[kinds])&
               (result>COMPUTE_DTYPE(-999.)))
        result[noisy&(noise_rule==1)]=COMPUTE_DTYPE(-999.)
        result[noisy&(noise_rule==2)]=self.noise_minimum_m[kinds][noisy&(noise_rule==2)]
        result[noisy&(noise_rule==3)]=defaults[noisy&(noise_rule==3)]
        missing=result<COMPUTE_DTYPE(-998.)
        replace=missing&(missing_rule!=0)
        result[replace]=defaults[replace]
        return result


def _read_receptors(config: TerrelConfig) -> _ReceptorState | None:
    if config.receptor_input is None:
        return None
    path = Path(config.receptor_input)
    try:
        records = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"invalid TERREL discrete receptor file: {path}") from error
    if records.shape[1] != config.receptor_columns:
        raise ValueError(
            f"TERREL receptor file has {records.shape[1]} columns; "
            f"NXYCOL={config.receptor_columns}"
        )
    count = records.shape[0]
    height=(records[:,2].astype(COMPUTE_DTYPE) if config.receptor_columns==4 else
            np.full(count,-999.,dtype=COMPUTE_DTYPE))
    receptor_height=(records[:,3].astype(COMPUTE_DTYPE)
                     if config.receptor_columns==4 else
                     np.zeros(count,dtype=COMPUTE_DTYPE))
    return _ReceptorState(
        records[:, :2].copy(), height, receptor_height,
        config.interpolate_receptors, config.receptor_radius_km,
        np.vstack((np.full(count,-1.e15,dtype=np.float64),
                   np.full(count,1.e15,dtype=np.float64),
                   np.full(count,1.e15,dtype=np.float64),
                   np.full(count,-1.e15,dtype=np.float64))),
        np.vstack((np.full(count,-1.e15,dtype=np.float64),
                   np.full(count,-1.e15,dtype=np.float64),
                   np.full(count,1.e15,dtype=np.float64),
                   np.full(count,1.e15,dtype=np.float64))),
        np.full((4,count),-999.,dtype=COMPUTE_DTYPE),
        np.full((4,count),1.e15,dtype=np.float64),
    )


def _terrain_filter(config: TerrelConfig) -> _TerrainFilter:
    coast_grid=config.grid
    if config.grid.grid_type==3:
        radial,_=_polar_boundaries(config.grid)
        extent=float(radial[-1])*.001
        coast_grid=replace(
            config.grid,nx=2,ny=2,dx_km=extent,dy_km=extent,
            x_origin_km=config.grid.x_origin_km-extent,
            y_origin_km=config.grid.y_origin_km-extent,
        )
    if config.gshhs_file is not None:
        if config.coastline_file is None:
            raise ValueError("TERREL GSHHS processing requires a COASTBLN output")
        extract_gshhs(
            config.gshhs_file,
            config.coastline_file,
            coast_grid,
            source_datum=config.coastline_source_datum,
            wdbii_datum=config.coastline_wdbii_datum,
        )
    polygons = (() if config.coastline_file is None else
                _read_coastline_bln(config.coastline_file, coast_grid, allow_empty=True))
    arrays = tuple(
        np.asarray(values, dtype=dtype)
        for values, dtype in (
            (config.noise_replacement, np.int8),
            (config.noise_minimum_m, COMPUTE_DTYPE),
            (config.missing_replacement, np.int8),
            (config.default_elevation_m, COMPUTE_DTYPE),
        )
    )
    if any(values.shape != (5,) for values in arrays):
        raise ValueError("TERREL terrain replacement controls must contain five types")
    raw_echo=(_RawEchoWriter(config.raw_echo_output,config.grid,config.datum)
              if config.raw_echo_output is not None else None)
    return _TerrainFilter(
        tuple(polygons),config.coastline_file is not None,*arrays,raw_echo
    )


def _domain_mask(x, y, grid):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    x_lower,x_upper,y_lower,y_upper=_grid_domain_metres(grid)
    return ((x>=x_lower)&(x<=x_upper)&(y>=y_lower)&(y<=y_upper))


def _grid_domain_metres(grid):
    """Return the projected domain bounds in metres."""
    x_origin=float(grid.x_origin_km); y_origin=float(grid.y_origin_km)
    if grid.grid_type==3:
        radial,_=_polar_boundaries_km(grid)
        extent=float(radial[-1])
        x_lower,x_upper=x_origin-extent,x_origin+extent
        y_lower,y_upper=y_origin-extent,y_origin+extent
    else:
        x_lower=x_origin; y_lower=y_origin
        x_upper=x_lower+grid.nx*float(grid.dx_km)
        y_upper=y_lower+grid.ny*float(grid.dy_km)
    return tuple(COORD_DTYPE(value*1000.) for value in
                 (x_lower,x_upper,y_lower,y_upper))


def _write_receptors(path: Path, receptors: _ReceptorState,
                     terrain_filter: _TerrainFilter) -> None:
    elevation = receptors.finish()
    kinds=terrain_filter.types(receptors.coordinates_km[:,0],receptors.coordinates_km[:,1])
    elevation=terrain_filter.output(elevation,kinds)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for (x, y), terrain, height in zip(
            receptors.coordinates_km, elevation, receptors.receptor_heights_m
        ):
            stream.write(f" {x:12.5f}{y:12.5f}{terrain:10.2f}{height:10.2f}\n")


def _save_header(grid: GridConfig, receptor_count: int) -> np.ndarray:
    x_lower,_,y_lower,_=_grid_domain_metres(grid)
    cell_metres=np.asarray(
        np.float64(grid.dx_km)*np.float64(1000.),dtype=np.float64)
    return np.asarray(
        [int(x_lower),int(y_lower),grid.utm_zone,int(cell_metres),
         grid.nx,grid.ny,receptor_count],
        dtype=np.int32,
    )


def _fortran_records(path: Path) -> tuple[str, list[bytes]]:
    """Read classic four-byte-marker sequential unformatted records."""
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"TERREL save file is truncated: {path}")
    endian = next(
        (candidate for candidate in ("<", ">")
         if struct.unpack_from(candidate + "i", raw, 0)[0] == 28),
        None,
    )
    if endian is None:
        raise ValueError(f"TERREL save file has an invalid first record: {path}")
    records = []
    offset = 0
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError(f"TERREL save file has a truncated record marker: {path}")
        length = struct.unpack_from(endian + "i", raw, offset)[0]
        end = offset + 4 + length
        if length < 0 or end + 4 > len(raw):
            raise ValueError(f"TERREL save file has an invalid record length: {path}")
        if struct.unpack_from(endian + "i", raw, end)[0] != length:
            raise ValueError(f"TERREL save file record markers disagree: {path}")
        records.append(raw[offset + 4:end])
        offset = end + 4
    return endian, records


def _load_save(path: Path, grid: GridConfig, sums, counts, maxima,
               receptors: _ReceptorState | None) -> None:
    endian, records = _fortran_records(path)
    receptor_count = 0 if receptors is None else receptors.coordinates_km.shape[0]
    expected_header = _save_header(grid, receptor_count)
    if len(records) < 4:
        raise ValueError(f"TERREL save file lacks cell arrays: {path}")
    header = np.frombuffer(records[0], dtype=endian + "i4")
    if header.size != 7 or not np.array_equal(header, expected_header):
        raise ValueError("TERREL save-file header does not match the current grid/receptors")
    cell_count = grid.nx * grid.ny

    def record(index, dtype, size, label):
        values = np.frombuffer(records[index], dtype=endian + dtype)
        if values.size != size:
            raise ValueError(f"TERREL save-file {label} record has the wrong size")
        return values

    sums[:] = record(1, "f4", cell_count, "SUM")
    counts[:] = record(2, "i4", cell_count, "KNT")
    # CELLOUT stores its initialization sentinel in untouched cells.  Use a
    # neutral accumulator internally so a later LPREV run can add data there.
    sums[counts == 0] = 0.0
    maxima[:] = record(3, "f4", cell_count, "HTMAX")
    expected_records = 9 if receptors is not None else 4
    if len(records) != expected_records:
        raise ValueError("TERREL save file has an unexpected number of records")
    if receptors is not None:
        receptors.heights_m[:] = record(4, "f4", receptor_count, "ELRECM")
        for target, index, label in (
            (receptors.vertices_x, 5, "XX"),
            (receptors.vertices_y, 6, "YY"),
            (receptors.vertices_z, 7, "ZZ"),
            (receptors.distances, 8, "DISTXY"),
        ):
            target[:] = record(index, "f4", 4 * receptor_count, label).reshape(
                receptor_count, 4
            ).T


def _write_save(path: Path, grid: GridConfig, sums, counts, maxima,
                receptors: _ReceptorState | None) -> None:
    """Write CELLOUT's little-endian sequential-unformatted layout."""
    receptor_count = 0 if receptors is None else receptors.coordinates_km.shape[0]
    arrays = [
        _save_header(grid, receptor_count).astype("<i4", copy=False),
        np.where(np.asarray(counts) == 0, np.float64(-999.0), sums).astype(
            "<f4", copy=False
        ),
        np.asarray(counts, dtype="<i4"),
        np.asarray(maxima, dtype="<f4"),
    ]
    if receptors is not None:
        arrays.append(np.asarray(receptors.heights_m, dtype="<f4"))
        arrays.extend(
            np.asarray(values.T, dtype="<f4").ravel()
            for values in (receptors.vertices_x, receptors.vertices_y,
                           receptors.vertices_z, receptors.distances)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for values in arrays:
            payload = values.tobytes(order="C")
            marker = struct.pack("<i", len(payload))
            stream.write(marker)
            stream.write(payload)
            stream.write(marker)


_USGSLA_SPECS = {
    "NA": UsgslaSpec(9223, 8996, -4487.0, -4515.0, 50.0, -100.0),
    "SA": UsgslaSpec(6000, 8000, -3000.0, -4899.0, -15.0, -60.0),
    "AP": UsgslaSpec(9300, 8000, -5000.0, -3944.891, -15.0, 135.0),
    "AF": UsgslaSpec(8350, 9276, -4458.0, -4795.0, 5.0, 20.0),
    "EU": UsgslaSpec(13000, 13000, -3000.0, -4999.0, 55.0, 20.0),
    "AS": UsgslaSpec(13000, 12000, -8000.0, -5499.0, 45.0, 100.0),
}


def _tile_origin(path: Path) -> tuple[int, int]:
    match = _HGT_NAME.match(path.name)
    if match is None:
        raise ValueError(f"not an SRTM HGT filename: {path.name}")
    ns, lat, ew, lon = match.groups()
    return int(lat) * (1 if ns.upper() == "N" else -1), int(lon) * (1 if ew.upper() == "E" else -1)


def _read_srtm_tile(path: Path) -> np.ndarray:
    """Read a standard SRTM HGT tile through GDAL's SRTMHGT driver."""
    try:
        with rasterio.open(path) as source:
            if source.driver != "SRTMHGT" or source.count != 1:
                raise ValueError(f"not a standard single-band SRTM HGT tile: {path}")
            if source.width != source.height or source.width < 2:
                raise ValueError(f"invalid square 16-bit HGT tile: {path}")
            if np.dtype(source.dtypes[0]).kind not in {"i", "u"}:
                raise ValueError(f"SRTM HGT elevations must be integers: {path}")
            return source.read(1)
    except RasterioIOError as error:
        raise ValueError(f"invalid SRTM HGT tile: {path}") from error


def _accumulate_tile(path, grid, sums, counts, input_datum='WGS-84', *, maxima=None,
                     peak=False, receptors=None, direct=False, terrain_filter=None) -> int:
    path = Path(path)
    lat_s, lon_w = _tile_origin(path)
    raw = _read_srtm_tile(path)
    side = raw.shape[0]
    corner_lon = np.array([lon_w, lon_w + 1, lon_w, lon_w + 1], dtype=COORD_DTYPE)
    corner_lat = np.array([lat_s + 1, lat_s + 1, lat_s, lat_s], dtype=COORD_DTYPE)
    cx, cy = (np.asarray(a, dtype=COORD_DTYPE) for a in
              project_geographic(corner_lon,corner_lat,grid,input_datum))
    xnw, xne, xsw, xse = cx
    ynw, yne, ysw, yse = cy
    denominator = float(side - 1)
    step = 1.0 / denominator
    along = np.arange(side - 1, dtype=COORD_DTYPE)
    inside_total = 0
    # TERREL skips the northern overlap row and eastern overlap column.
    for j in range(1, side):
        south = j * step
        xnear = xnw + (xsw - xnw) * south
        ynear = ynw + (ysw - ynw) * south
        xfar = xne + (xse - xne) * south
        yfar = yne + (yse - yne) * south
        delx = (xfar - xnear) * step
        dely = (yfar - ynear) * step
        interpolated_x = xnear + delx * along
        interpolated_y = ynear + dely * along
        if direct:
            longitude = lon_w + along / denominator
            latitude = np.full(longitude.shape, lat_s + 1.0 - float(south))
            x, y = project_geographic(longitude, latitude, grid, input_datum)
            x, y = np.asarray(x, dtype=COORD_DTYPE), np.asarray(y, dtype=COORD_DTYPE)
        else:
            x, y = interpolated_x, interpolated_y
        z = np.asarray(raw[j, : side - 1], dtype=COMPUTE_DTYPE)
        radius = (_peak_radius_from_points(interpolated_x, interpolated_y, grid)
                  if peak else None)
        inside_total += _accumulate_points(
            x, y, z, grid, sums, counts, maxima, radius, receptors, terrain_filter
        )
    return inside_total


def _gtopo_spec(path: Path) -> tuple[int, int, int, int, int]:
    """Return north, west, rows, columns and samples/degree for GTOPO30."""
    match = _GTOPO_NAME.match(path.name)
    if match is None:
        raise ValueError(f"not a GTOPO30 filename: {path.name}")
    ew, lon, ns, lat = match.groups()
    west = int(lon) * (1 if ew.upper() == "E" else -1)
    north = int(lat) * (1 if ns.upper() == "N" else -1)
    lon_degrees, lat_degrees = ((60, 30) if north == -60 else (40, 50))
    return north, west, lat_degrees * 120, lon_degrees * 120, 120


def _accumulate_gtopo30(path, grid, sums, counts, input_datum='WGS-84', *, spec=None,
                        maxima=None, peak=False, receptors=None, direct=False,
                        terrain_filter=None) -> int:
    """Accumulate a headerless, north-to-south GTOPO30 DEM.

    ``spec`` is an internal test seam for small synthetic rasters; production
    files derive the canonical 120 samples/degree dimensions from their name.
    """
    north, west, rows, columns, samples = _gtopo_spec(path) if spec is None else spec
    if path.stat().st_size != rows * columns * 2:
        raise ValueError(f"invalid GTOPO30 raster size: {path}")
    raw = np.memmap(path, dtype=">i2", mode="r", shape=(rows, columns))
    along = np.arange(samples, dtype=np.float64) / np.float64(samples)
    inside_total = 0
    for row0 in range(0, rows, samples):
        lat_n = north - row0 / samples
        for col0 in range(0, columns, samples):
            lon_w = west + col0 / samples
            corner_lon=np.array([lon_w,lon_w+1,lon_w,lon_w+1],dtype=np.float64)
            corner_lat=np.array([lat_n,lat_n,lat_n-1,lat_n-1],dtype=np.float64)
            cx,cy=(np.asarray(a,dtype=np.float64) for a in
                   project_geographic(corner_lon,corner_lat,grid,input_datum))
            xnw,xne,xsw,xse=cx; ynw,yne,ysw,yse=cy
            block=np.asarray(
                raw[row0:row0+samples,col0:col0+samples],dtype=COMPUTE_DTYPE)
            for j in range(block.shape[0]):
                south=np.float64(j)/np.float64(samples)
                fraction=along[:block.shape[1]]
                xnear=xnw+(xsw-xnw)*south; ynear=ynw+(ysw-ynw)*south
                xfar=xne+(xse-xne)*south; yfar=yne+(yse-yne)*south
                interpolated_x=xnear+(xfar-xnear)*fraction
                interpolated_y=ynear+(yfar-ynear)*fraction
                if direct:
                    longitude=lon_w+fraction.astype(np.float64)
                    latitude=np.full(longitude.shape,lat_n-float(south))
                    x,y=project_geographic(longitude,latitude,grid,input_datum)
                    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
                else:
                    x=interpolated_x; y=interpolated_y
                z=block[j]
                radius = (_peak_radius_from_points(interpolated_x,interpolated_y,grid)
                          if peak else None)
                inside_total += _accumulate_points(
                    x, y, z, grid, sums, counts, maxima, radius, receptors,
                    terrain_filter
                )
    return inside_total


def _read_generic(path: Path) -> tuple[int, int, int, np.ndarray]:
    """Read a TERREL Generic header and its three-column point records."""
    with path.open("r", encoding="ascii") as stream:
        header = stream.readline().split()
        if len(header) != 3:
            raise ValueError(f"invalid TERREL Generic header: {path}")
        try:
            coordinate_type, elevation_unit, utm_zone = map(int, header)
        except ValueError as error:
            raise ValueError(f"invalid TERREL Generic header: {path}") from error
        try:
            points = np.loadtxt(stream, dtype=np.float64, ndmin=2)
        except ValueError as error:
            raise ValueError(f"invalid TERREL Generic point record: {path}") from error
    if coordinate_type not in {1, 2, 3}:
        raise ValueError("Generic coordinate type must be 1 (UTM m), 2 (UTM km), or 3 (lat/lon)")
    if elevation_unit not in {1, 2}:
        raise ValueError("Generic elevation unit must be 1 (metre) or 2 (foot)")
    if coordinate_type in {1, 2} and not 1 <= utm_zone <= 60:
        raise ValueError("Generic UTM zone must be in the range 1..60")
    if points.size == 0 or points.shape[1] != 3:
        raise ValueError(f"TERREL Generic file must contain three-column point records: {path}")
    return coordinate_type, elevation_unit, utm_zone, points


def _add_cell_samples(flat, elevation, sums, counts, maxima):
    np.add.at(sums, flat, np.asarray(elevation, dtype=sums.dtype))
    counts += np.bincount(flat, minlength=counts.size).astype(counts.dtype, copy=False)
    if maxima is not None:
        np.maximum.at(maxima, flat, elevation)


def _peak_radius_from_points(x, y, grid) -> float | tuple[float, float]:
    """Return a source footprint radius in the grid's native index units.

    Polar accumulation works in projected metres.  Cartesian accumulation
    works in cell coordinates, so its x and y radii must be scaled by the
    corresponding cell spacing.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("TERREL peak processing needs at least two source points")
    radius_m = np.float64(0.5) * np.hypot(x[1] - x[0], y[1] - y[0])
    return _fixed_peak_radius(radius_m, grid)


def _fixed_peak_radius(radius_m, grid) -> float | tuple[float, float]:
    """Convert a projected-metre radius for polar or Cartesian accumulation."""
    radius_m = float(radius_m)
    if grid.grid_type == 3:
        return radius_m
    return (
        radius_m / (float(grid.dx_km) * 1000.0),
        radius_m / (float(grid.dy_km) * 1000.0),
    )


def _polar_boundaries_km(grid):
    rings = np.asarray(grid.ring_distances_km, dtype=np.float64)
    angles = np.asarray(grid.ray_angles_deg, dtype=np.float64)
    if rings.size != grid.nx or angles.size != grid.ny or rings.size < 2 or angles.size < 2:
        raise ValueError("polar grid requires ordered DISK and ANG values matching NRING/NRAYS")
    if np.any(np.diff(rings) <= 0.0) or np.any(np.diff(angles) <= 0.0) or angles[0] < 1.0:
        raise ValueError("polar DISK and ANG values must be strictly increasing; first ANG must be >= 1")
    radial = np.empty(rings.shape, dtype=np.float64)
    if grid.polar_processing == 2:
        radial[:-1] = rings[1:]
        radial[-1] = 2.0 * rings[-1] - rings[-2]
    elif grid.polar_processing == 1:
        radial[:-1] = 0.5 * (rings[:-1] + rings[1:])
        radial[-1] = 2.0 * rings[-1] - radial[-2]
    else:
        raise ValueError("polar IPROC must be 1 or 2")
    angular = np.empty(angles.shape, dtype=np.float64)
    angular[:-1] = 0.5 * (angles[:-1] + angles[1:]) - angles[0]
    angular[-1] = 0.5 * (angles[-1] + angles[0] + 360.0) - angles[0]
    return radial,angular


def _polar_boundaries(grid):
    radial,angular=_polar_boundaries_km(grid)
    return np.asarray(radial*np.float64(1000.),dtype=np.float64),angular


def _polar_cell_indices(dx,dy,radial_boundaries,angular_boundaries,
                        first_angle,radius=None):
    """Return polar ring and ray indices for projected offsets."""
    dx=np.asarray(dx,dtype=COORD_DTYPE); dy=np.asarray(dy,dtype=COORD_DTYPE)
    radial=np.asarray(radial_boundaries,dtype=COORD_DTYPE)
    angular=np.asarray(angular_boundaries,dtype=COORD_DTYPE)
    distance=np.hypot(dx,dy)
    radial_index=np.searchsorted(radial,distance,side="right")
    direction=np.degrees(np.arctan2(dx,dy))
    shifted=np.mod(direction-float(first_angle),360.)
    angular_index=np.searchsorted(angular,shifted,side="right")
    angular_index=np.where(angular_index>=angular.size,0,angular_index)
    if radius is None:
        return distance,shifted,radial_index,angular_index

    radius=np.broadcast_to(np.asarray(radius,dtype=COORD_DTYPE),distance.shape)
    peak_radial=radial_index.copy()
    in_grid=radial_index<radial.size
    boundary=radial[np.minimum(radial_index,radial.size-1)]
    outward=distance+radius>boundary
    peak_radial[in_grid&outward]=radial_index[in_grid&outward]+1
    previous_boundary=radial[np.maximum(radial_index-1,0)]
    inward=(in_grid&(radial_index>=1)&
            (distance-radius<previous_boundary))
    peak_radial[inward]=radial_index[inward]-1
    previous=np.where(angular_index==0,angular.size-1,angular_index-1)
    following=np.where(angular_index==angular.size-1,0,angular_index+1)
    up_edge=np.mod(angular[angular_index]-shifted,360.)
    down_edge=np.mod(shifted-angular[previous],360.)
    divisor=np.where(distance==0.,.001,distance)
    angular_radius=np.degrees(radius/divisor)
    peak_angular=angular_index.copy()
    peak_angular=np.where(up_edge<angular_radius,following,peak_angular)
    peak_angular=np.where((up_edge>=angular_radius)&(down_edge<angular_radius),
                          previous,peak_angular)
    return (distance,shifted,radial_index,angular_index,peak_radial,
            peak_angular,up_edge,angular_radius)


def _accumulate_polar_points(x, y, elevation, grid, sums, counts, maxima, radius_m):
    radial_boundaries, angular_boundaries = _polar_boundaries(grid)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=COMPUTE_DTYPE)
    x_origin=np.asarray(np.float64(grid.x_origin_km)*np.float64(1000.),dtype=np.float64)
    y_origin=np.asarray(np.float64(grid.y_origin_km)*np.float64(1000.),dtype=np.float64)
    dx = np.asarray(x-x_origin,dtype=np.float64)
    dy = np.asarray(y-y_origin,dtype=np.float64)
    indexed=_polar_cell_indices(
        dx,dy,radial_boundaries,angular_boundaries,grid.ray_angles_deg[0],radius_m)
    distance,shifted,radial_index,angular_index=indexed[:4]
    valid = (elevation >= -32000.0) & (radial_index < grid.nx)
    flat = angular_index[valid] * grid.nx + radial_index[valid]
    _add_cell_samples(flat, elevation[valid], sums, counts, maxima)
    if radius_m is not None:
        peak_radial,peak_angular=indexed[4:6]
        perturbed_valid=(valid&(peak_radial>=0)&(peak_radial<grid.nx)&
                         (peak_angular>=0)&(peak_angular<grid.ny))
        for candidate_r, candidate_a, changed in (
            (peak_radial, angular_index, peak_radial != radial_index),
            (radial_index, peak_angular, peak_angular != angular_index),
            (peak_radial, peak_angular,
             (peak_radial != radial_index) & (peak_angular != angular_index)),
        ):
            selected=(perturbed_valid&changed&(candidate_r>=0)&
                      (candidate_r<grid.nx))
            if selected.any():
                candidate_flat = candidate_a[selected] * grid.nx + candidate_r[selected]
                _add_cell_samples(candidate_flat, elevation[selected], sums, counts, maxima)
    return int(valid.sum())


def _accumulate_points(x, y, elevation, grid, sums, counts, maxima=None,
                       peak_radius_grid=None, receptors=None, terrain_filter=None) -> int:
    """Apply SAVEELEV's standard mean-grid path to projected metre points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=COMPUTE_DTYPE)
    domain=_domain_mask(x,y,grid)
    if terrain_filter is not None:
        elevation=terrain_filter.extracted(x,y,elevation,domain)
    if receptors is not None:
        receptors.add(x[domain], y[domain], elevation[domain])
    if grid.grid_type == 3:
        return _accumulate_polar_points(
            x, y, elevation, grid, sums, counts, maxima, peak_radius_grid
        )
    xllm = np.float64(np.float64(grid.x_origin_km) * np.float64(1000.0))
    yllm = np.float64(np.float64(grid.y_origin_km) * np.float64(1000.0))
    xscale = np.float64(np.float64(1.0) /
                        np.float64(np.float64(grid.dx_km) * np.float64(1000.0)))
    yscale = np.float64(np.float64(1.0) /
                        np.float64(np.float64(grid.dy_km) * np.float64(1000.0)))
    cell_x = np.asarray((x - xllm) * xscale, dtype=np.float64)
    cell_y = np.asarray((y - yllm) * yscale, dtype=np.float64)
    ix = np.trunc(cell_x).astype(np.int64)
    iy = np.trunc(cell_y).astype(np.int64)
    ix += cell_x >= np.float64(0.0)
    iy += cell_y >= np.float64(0.0)
    ix -= 1
    iy -= 1
    valid = (
        (elevation >= -32000.0)
        & (ix >= 0)
        & (ix < grid.nx)
        & (iy >= 0)
        & (iy < grid.ny)
    )
    if not valid.any():
        return 0
    flat = iy[valid] * grid.nx + ix[valid]
    _add_cell_samples(flat, elevation[valid], sums, counts, maxima)
    if peak_radius_grid is not None:
        radius_x, radius_y = (
            np.broadcast_to(np.asarray(value, dtype=np.float64), x.shape)
            for value in peak_radius_grid
        )
        peak_x = np.trunc(cell_x + radius_x).astype(np.int64)
        peak_y = np.trunc(cell_y + radius_y).astype(np.int64)
        peak_x = np.where(
            peak_x == ix, np.trunc(cell_x - radius_x).astype(np.int64), peak_x
        )
        peak_y = np.where(
            peak_y == iy, np.trunc(cell_y - radius_y).astype(np.int64), peak_y
        )
        for candidate_x, candidate_y, changed in (
            (peak_x, iy, peak_x != ix),
            (ix, peak_y, peak_y != iy),
            (peak_x, peak_y, (peak_x != ix) & (peak_y != iy)),
        ):
            selected = (valid & changed & (candidate_x >= 0) & (candidate_x < grid.nx) &
                        (candidate_y >= 0) & (candidate_y < grid.ny))
            if selected.any():
                candidate_flat = candidate_y[selected] * grid.nx + candidate_x[selected]
                _add_cell_samples(candidate_flat, elevation[selected], sums, counts, maxima)
    return int(valid.sum())


def _accumulate_generic(path, grid, sums, counts, input_datum="WGS-84", *, maxima=None,
                        peak=False, receptors=None, terrain_filter=None) -> int:
    """Accumulate Generic scattered terrain points using LOADGNR semantics."""
    coordinate_type, elevation_unit, utm_zone, points = _read_generic(Path(path))
    if coordinate_type == 3:
        latitude, longitude = points[:, 0], points[:, 1]
        x, y = project_geographic(longitude, latitude, grid, input_datum)
    else:
        x, y = points[:, 0].copy(), points[:, 1].copy()
        if coordinate_type == 2:
            x *= 1000.0
            y *= 1000.0
        source_grid = GridConfig(
            nx=1,
            ny=1,
            dx_km=1.0,
            dy_km=1.0,
            projection="UTM",
            utm_zone=utm_zone,
            hemisphere="N",
            datum=input_datum,
        )
        x, y = transform_grid(x, y, source_grid, grid)

    elevation = points[:, 2].copy()
    if elevation_unit == 2:
        elevation *= 0.3048
    radius = _peak_radius_from_points(x, y, grid) if peak else None
    return _accumulate_points(
        x, y, elevation, grid, sums, counts, maxima, radius, receptors, terrain_filter
    )


def _accumulate_nzgen(path, grid, sums, counts, input_datum="WGS-84", *, maxima=None,
                      peak=False, receptors=None, terrain_filter=None) -> int:
    """Accumulate New Zealand Generic ``id elevation lon lat`` records."""
    try:
        points = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"invalid TERREL NZGEN point record: {path}") from error
    if points.size == 0 or points.shape[1] != 4:
        raise ValueError("TERREL NZGEN files must contain four-column point records")
    elevation = points[:, 1]
    longitude = points[:, 2]
    latitude = points[:, 3]
    x, y = project_geographic(longitude, latitude, grid, input_datum)
    radius = _peak_radius_from_points(x, y, grid) if peak else None
    return _accumulate_points(
        x, y, elevation, grid, sums, counts, maxima, radius, receptors, terrain_filter
    )


def _usgsla_spec(path: Path) -> UsgslaSpec:
    name = path.name.upper()
    if name.startswith(("NA", "SA", "AP", "AF")):
        return _USGSLA_SPECS[name[:2]]
    if name.startswith("EA") and len(name) >= 7 and name[6] in {"E", "A"}:
        return _USGSLA_SPECS["EU" if name[6] == "E" else "AS"]
    raise ValueError(f"unrecognized USGSLA continent filename: {path.name}")


def _accumulate_usgsla(path, grid, sums, counts, input_datum="ESR-S", *, spec=None,
                       maxima=None, peak=False, receptors=None,
                       terrain_filter=None) -> int:
    """Accumulate a USGS Global Lambert Azimuthal two-byte DEM image."""
    path = Path(path)
    spec = _usgsla_spec(path) if spec is None else spec
    if path.stat().st_size != spec.nx * spec.ny * 2:
        raise ValueError(f"USGSLA raster must contain {spec.nx}x{spec.ny} two-byte values: {path}")
    source_grid = GridConfig(
        nx=1,
        ny=1,
        dx_km=1.0,
        dy_km=1.0,
        projection="LAZA",
        lat0=spec.latitude_origin,
        lon0=spec.longitude_origin,
        datum=input_datum,
    )
    source_crs = grid_crs(source_grid, units="km")
    corners_x = np.array([
        grid.x_origin_km,
        grid.x_origin_km + grid.nx * grid.dx_km,
        grid.x_origin_km + grid.nx * grid.dx_km,
        grid.x_origin_km,
    ])
    corners_y = np.array([
        grid.y_origin_km,
        grid.y_origin_km,
        grid.y_origin_km + grid.ny * grid.dy_km,
        grid.y_origin_km + grid.ny * grid.dy_km,
    ])
    px, py = unproject_to_crs(corners_x, corners_y, grid, source_crs, input_datum, units="km")
    i0 = int(np.floor(np.min(px) - spec.x_origin_km))
    i1 = int(np.ceil(np.max(px) - spec.x_origin_km))
    j0 = int(np.floor(np.min(py) - spec.y_origin_km))
    j1 = int(np.ceil(np.max(py) - spec.y_origin_km))
    buffer = max(abs(i1 - i0) // 10 + 1, abs(j1 - j0) // 10 + 1)
    i0, i1 = max(1, i0 - buffer), min(spec.nx, i1 + buffer)
    j0, j1 = max(1, j0 - buffer), min(spec.ny, j1 + buffer)
    if i0 > i1 or j0 > j1:
        return 0
    raw = np.memmap(path, dtype="u1", mode="r", shape=(spec.ny, spec.nx, 2))
    source_x = spec.x_origin_km + np.arange(i0, i1 + 1, dtype=np.float64)
    inside_total = 0
    for j in range(j0, j1 + 1):
        encoded = np.asarray(
            raw[spec.ny - j, i0 - 1:i1], dtype=COMPUTE_DTYPE)
        elevation = encoded[:, 1] + COMPUTE_DTYPE(255.0) * encoded[:, 0]
        source_y = np.full(source_x.shape, spec.y_origin_km + j)
        if grid.projection.upper() == 'LCC':
            x, y = project_glaz_to_lcc(
                source_x, source_y, source_grid, grid, units="m"
            )
        else:
            x, y = project_from_crs(
                source_x, source_y, source_crs, input_datum, grid, units="m"
            )
        radius = _fixed_peak_radius(np.sqrt(np.float64(2.0)) * 500.0, grid) if peak else None
        inside_total += _accumulate_points(
            x, y, elevation, grid, sums, counts, maxima, radius, receptors,
            terrain_filter
        )
    return inside_total


def _fixed_width_integers(lines, width: int, count: int, label: str) -> tuple[np.ndarray, int]:
    values = []
    line_index = 0
    while len(values) < count and line_index < len(lines):
        line = lines[line_index].rstrip("\r\n")
        for start in range(0, len(line), width):
            field = line[start:start + width]
            if field.strip():
                try:
                    values.append(int(field))
                except ValueError as error:
                    raise ValueError(f"invalid {label} fixed-width integer {field!r}") from error
        line_index += 1
    if len(values) != count:
        raise ValueError(f"truncated {label}: expected {count} integers, found {len(values)}")
    return np.asarray(values, dtype=np.int64), line_index


def _read_arm3(path: Path):
    lines = path.read_text(encoding="ascii").splitlines()
    header, consumed = _fixed_width_integers(lines, 10, 5, "ARM3 header")
    _, longitude_se, latitude_se, nx, ny = header
    if nx < 1 or ny < 1:
        raise ValueError("ARM3 panel dimensions must be positive")
    index, used = _fixed_width_integers(lines[consumed:], 20, 2 * nx * ny, "ARM3 index")
    present = int(np.count_nonzero(index[:nx * ny]))
    elevations, data_used = _fixed_width_integers(
        lines[consumed + used:], 4, present * 120 * 120, "ARM3 elevation data"
    )
    if any(line.strip() for line in lines[consumed + used + data_used:]):
        raise ValueError("ARM3 file has trailing elevation records")
    return int(longitude_se), int(latitude_se), int(nx), int(ny), index, elevations


def _accumulate_arm3(path, grid, sums, counts, input_datum="NAS-C", *, maxima=None,
                     peak=False, receptors=None, direct=False,
                     terrain_filter=None) -> int:
    """Accumulate ARM3 30-arc-second panels using legacy sheet interpolation."""
    longitude_se, latitude_se, nx, ny, index, elevations = _read_arm3(Path(path))
    offset = 0
    fractions = np.arange(120, dtype=np.float64) / np.float64(120.0)
    inside_total = 0
    for sheet_y in range(ny):
        for sheet_x in range(nx):
            if index[sheet_y * nx + sheet_x] == 0:
                continue
            block = elevations[offset:offset + 120 * 120].reshape(120, 120)
            offset += 120 * 120
            west_longitude = longitude_se + sheet_x
            south_latitude = latitude_se + sheet_y
            corner_lon = np.array(
                [-west_longitude, -(west_longitude + 1),
                 -west_longitude, -(west_longitude + 1)],
                dtype=np.float64,
            )
            corner_lat = np.array(
                [south_latitude, south_latitude,
                 south_latitude + 1, south_latitude + 1],
                dtype=np.float64,
            )
            cx, cy = (
                np.asarray(values, dtype=np.float64)
                for values in project_geographic(corner_lon, corner_lat, grid, input_datum)
            )
            xse, xsw, xne, xnw = cx
            yse, ysw, yne, ynw = cy
            for row, south in enumerate(fractions):
                xnear = xse + (xne - xse) * south
                ynear = yse + (yne - yse) * south
                xfar = xsw + (xnw - xsw) * south
                yfar = ysw + (ynw - ysw) * south
                interpolated_x = xnear + (xfar - xnear) * fractions
                interpolated_y = ynear + (yfar - ynear) * fractions
                if direct:
                    longitude = -west_longitude - fractions.astype(np.float64)
                    latitude = np.full(longitude.shape, south_latitude + float(south))
                    x, y = project_geographic(longitude, latitude, grid, input_datum)
                    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
                else:
                    x, y = interpolated_x, interpolated_y
                radius = (_peak_radius_from_points(interpolated_x,interpolated_y,grid)
                          if peak else None)
                inside_total += _accumulate_points(
                    x, y, block[row], grid, sums, counts, maxima, radius, receptors,
                    terrain_filter
                )
    return inside_total


def _3cd_origin(path: Path) -> tuple[int, int]:
    match = _3CD_NAME.match(path.name)
    if match is None:
        raise ValueError(f"not a 3CD terrain filename: {path.name}")
    latitude, west_longitude = match.groups()
    return int(latitude), int(west_longitude)


def _accumulate_3cd(path, grid, sums, counts, input_datum="WGS-72", *, samples=1201,
                    maxima=None, peak=False, receptors=None, direct=False,
                    terrain_filter=None) -> int:
    """Accumulate a transparent-access 3CD little-endian int16 raster."""
    path = Path(path)
    latitude_south, west_longitude_east_edge = _3cd_origin(path)
    if samples < 2 or path.stat().st_size != samples * samples * 2:
        raise ValueError(f"3CD raster must contain {samples}x{samples} int16 values: {path}")
    raw = np.memmap(path, dtype="<i2", mode="r", shape=(samples, samples))
    fraction = np.arange(samples, dtype=np.float64) / np.float64(samples - 1)
    corner_lon = np.array(
        [-(west_longitude_east_edge + 1), -west_longitude_east_edge,
         -(west_longitude_east_edge + 1), -west_longitude_east_edge],
        dtype=np.float64,
    )
    corner_lat = np.array(
        [latitude_south, latitude_south,
         latitude_south + 1, latitude_south + 1],
        dtype=np.float64,
    )
    cx, cy = (
        np.asarray(values, dtype=np.float64)
        for values in project_geographic(corner_lon, corner_lat, grid, input_datum)
    )
    xsw, xse, xnw, xne = cx
    ysw, yse, ynw, yne = cy
    inside_total = 0
    for strip, east in enumerate(fraction):
        xnear = xsw + (xse - xsw) * east
        ynear = ysw + (yse - ysw) * east
        xfar = xnw + (xne - xnw) * east
        yfar = ynw + (yne - ynw) * east
        interpolated_x = xnear + (xfar - xnear) * fraction
        interpolated_y = ynear + (yfar - ynear) * fraction
        if direct:
            longitude = np.full(
                fraction.shape, -(west_longitude_east_edge + 1) + float(east)
            )
            latitude = latitude_south + fraction.astype(np.float64)
            x, y = project_geographic(longitude, latitude, grid, input_datum)
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        else:
            x, y = interpolated_x, interpolated_y
        radius = (_peak_radius_from_points(interpolated_x,interpolated_y,grid)
                  if peak else None)
        inside_total += _accumulate_points(
            x, y, np.asarray(raw[strip]), grid, sums, counts, maxima, radius, receptors,
            terrain_filter
        )
    return inside_total


def _dmdf_point(record: str, second: bool = False) -> tuple[float, float, float]:
    record = record.ljust(80)
    fields = ((45, 49, 49, 55, 56, 60, 60, 66, 67, 75) if second else
              (14, 18, 18, 24, 25, 29, 29, 35, 36, 44))
    try:
        x_major = int(record[fields[0]:fields[1]])
        x_minor = int(record[fields[2]:fields[3]])
        y_major = int(record[fields[4]:fields[5]])
        y_minor = int(record[fields[6]:fields[7]])
        elevation = int(record[fields[8]:fields[9]])
    except ValueError as error:
        raise ValueError("invalid Canadian DMDF fixed-column point") from error
    return (
        x_major * 1000.0 + x_minor / 1000.0,
        y_major * 1000.0 + y_minor / 1000.0,
        elevation / 1000.0,
    )


def _read_dmdf(path: Path) -> np.ndarray:
    points = []
    previous_type = ""
    for line in path.read_text(encoding="ascii").splitlines():
        record = line.ljust(80)
        kind = record[:2]
        if kind == "99":
            break
        if kind == "07":
            previous_type = kind
            continue
        if kind == "06":
            previous_type = kind
            continue
        if kind in {"01", "02", "03"}:
            points.append(_dmdf_point(record))
            if kind in {"02", "03"}:
                points.append(_dmdf_point(record, second=True))
            previous_type = kind
            continue
        if kind == "00" and previous_type != "06":
            points.append(_dmdf_point(record))
            if record[45:55].strip():
                points.append(_dmdf_point(record, second=True))
    if not points:
        raise ValueError(f"Canadian DMDF file contains no terrain points: {path}")
    return np.asarray(points, dtype=np.float64)


def _accumulate_dmdf(path, grid, sums, counts, input_datum="NAS-C", *, maxima=None,
                     peak=False, receptors=None, terrain_filter=None) -> int:
    """Accumulate Canadian DMDF fixed-column UTM point records."""
    points = _read_dmdf(Path(path))
    x = points[:, 0].copy()
    y = points[:, 1]
    elevation = points[:, 2]
    zones = np.where(x > 1_000_000.0, 12, 11)
    x[zones == 12] -= 1_000_000.0
    projected_x = np.empty_like(x)
    projected_y = np.empty_like(y)
    for zone in (11, 12):
        selected = zones == zone
        if not selected.any():
            continue
        source_grid = GridConfig(
            nx=1,
            ny=1,
            dx_km=1.0,
            dy_km=1.0,
            projection="UTM",
            utm_zone=zone,
            hemisphere="N",
            datum=input_datum,
        )
        projected_x[selected], projected_y[selected] = transform_grid(
            x[selected], y[selected], source_grid, grid
        )
    radius = _fixed_peak_radius(np.sqrt(np.float64(2.0)) * 50.0, grid) if peak else None
    return _accumulate_points(
        projected_x, projected_y, elevation, grid, sums, counts, maxima, radius,
        receptors, terrain_filter
    )


def _dem_float(field: str) -> float:
    try:
        return float(field.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"invalid USGS DEM floating-point field {field!r}") from error


def _dem_records(path: Path) -> list[str]:
    raw = path.read_bytes()
    if b"\n" in raw or b"\r" in raw:
        records = raw.decode("ascii").splitlines()
    else:
        if len(raw) % 1024:
            raise ValueError("USGS DEM byte stream is not a multiple of 1024 bytes")
        records = [raw[start:start + 1024].decode("ascii")
                   for start in range(0, len(raw), 1024)]
    if not records:
        raise ValueError(f"empty USGS DEM file: {path}")
    return [record.ljust(1024) for record in records]


def _parse_dem_header(record: str) -> DemHeader:
    def integers(start, count):
        try:
            return tuple(int(record[start + 6 * i:start + 6 * (i + 1)]) for i in range(count))
        except ValueError as error:
            raise ValueError("invalid USGS DEM integer header field") from error

    codes = integers(144, 4)
    coordinate_unit, elevation_unit, _ = integers(528, 3)
    corners = np.array([_dem_float(record[546 + 24 * i:570 + 24 * i])
                        for i in range(8)], dtype=np.float64).reshape(4, 2)
    resolution = tuple(_dem_float(record[816 + 12 * i:828 + 12 * i]) for i in range(3))
    profiles = integers(852, 2)
    datum_field = record[890:892].strip()
    datum = int(datum_field) if datum_field else 0
    return DemHeader(codes, coordinate_unit, elevation_unit, corners,
                     resolution, profiles, datum)


def _parse_dem_profiles(records: list[str], expected: int):
    profiles = []
    record_index = 1
    for wanted in range(1, expected + 1):
        if record_index >= len(records):
            raise ValueError("truncated USGS DEM profile records")
        record = records[record_index]
        record_index += 1
        try:
            integers = [int(record[6 * i:6 * (i + 1)]) for i in range(4)]
        except ValueError as error:
            raise ValueError("invalid USGS DEM profile header") from error
        row, column, point_count, _ = integers
        if column != wanted:
            raise ValueError(f"USGS DEM profile index {column} does not match {wanted}")
        origin = (_dem_float(record[24:48]), _dem_float(record[48:72]))
        elevation_base = _dem_float(record[72:96])
        first = min(point_count, 146)
        values = [int(record[144 + 6 * i:150 + 6 * i]) for i in range(first)]
        remaining = point_count - first
        while remaining:
            if record_index >= len(records):
                raise ValueError("truncated USGS DEM profile elevations")
            continuation = records[record_index]
            record_index += 1
            take = min(remaining, 170)
            values.extend(int(continuation[6 * i:6 * (i + 1)]) for i in range(take))
            remaining -= take
        profiles.append((row, origin, elevation_base, np.asarray(values, dtype=np.int64)))
    if any(record.strip() for record in records[record_index:]):
        raise ValueError("USGS DEM file has trailing logical records")
    return profiles


def _accumulate_usgs_degree_dem(path, grid, sums, counts, input_datum="WGS-72", *,
                                maxima=None, peak=False, receptors=None,
                                direct=False, terrain_filter=None) -> int:
    """Accumulate USGS90/CDED geographic A/B-record DEM profiles."""
    records = _dem_records(Path(path))
    header = _parse_dem_header(records[0])
    if header.codes[2] != 0:
        raise ValueError("USGS90/CDED DEM must use geographic coordinates")
    profiles = _parse_dem_profiles(records, header.profile_count[1])
    datum_names = {1: "NAS-C", 2: "WGS-72", 3: "WGS-84", 4: "NAR-C"}
    source_datum = datum_names.get(header.horizontal_datum, input_datum)
    longitude = header.corners[:, 0] / 3600.0
    latitude = header.corners[:, 1] / 3600.0
    cx, cy = (
        np.asarray(values, dtype=np.float64)
        for values in project_geographic(longitude, latitude, grid, source_datum)
    )
    xsw, xnw, xne, xse = cx
    ysw, ynw, yne, yse = cy
    profile_step = header.resolution[0] / 3600.0
    scale = 0.3048 if header.elevation_unit == 1 else 1.0
    inside_total = 0
    for profile_index, (_, origin, base, raw) in enumerate(profiles):
        east = np.float64(profile_index * profile_step)
        xnear = xsw + (xse - xsw) * east
        ynear = ysw + (yse - ysw) * east
        xfar = xnw + (xne - xnw) * east
        yfar = ynw + (yne - ynw) * east
        along = np.arange(raw.size, dtype=np.float64) / np.float64(raw.size - 1)
        interpolated_x = xnear + (xfar - xnear) * along
        interpolated_y = ynear + (yfar - ynear) * along
        if direct:
            longitude = np.full(raw.shape, origin[0] / 3600.0)
            latitude = (origin[1] + header.resolution[1] * np.arange(raw.size)) / 3600.0
            x, y = project_geographic(longitude, latitude, grid, source_datum)
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        else:
            x, y = interpolated_x, interpolated_y
        valid = raw > -32000
        rounded = np.where(base + raw >= 0.0, np.floor(base + raw + 0.5),
                           np.ceil(base + raw - 0.5))
        elevation = np.where(
            valid,
            rounded.astype(np.int16).astype(COMPUTE_DTYPE) * COMPUTE_DTYPE(scale),
            raw,
        ).astype(COMPUTE_DTYPE, copy=False)
        radius = (_peak_radius_from_points(interpolated_x,interpolated_y,grid)
                  if peak else None)
        inside_total += _accumulate_points(
            x, y, elevation, grid, sums, counts, maxima, radius, receptors,
            terrain_filter
        )
    return inside_total


def _accumulate_usgs30(path, grid, sums, counts, input_datum="NAS-C", *, maxima=None,
                       peak=False, receptors=None, terrain_filter=None) -> int:
    """Accumulate USGS 7.5-minute UTM A/B-record DEM profiles."""
    records = _dem_records(Path(path))
    header = _parse_dem_header(records[0])
    if header.codes[2] != 1:
        raise ValueError("USGS30 DEM must use UTM coordinates")
    if not 1 <= header.codes[3] <= 60:
        raise ValueError("USGS30 DEM contains an invalid UTM zone")
    profiles = _parse_dem_profiles(records, header.profile_count[1])
    source_grid = GridConfig(
        nx=1,
        ny=1,
        dx_km=1.0,
        dy_km=1.0,
        projection="UTM",
        utm_zone=header.codes[3],
        hemisphere="N",
        datum=input_datum,
    )
    elevation_scale = header.resolution[2]
    if header.elevation_unit == 1:
        elevation_scale *= 0.3048
    inside_total = 0
    for _, origin, base, raw in profiles:
        source_x = np.full(raw.shape, origin[0], dtype=np.float64)
        source_y = origin[1] + header.resolution[1] * np.arange(raw.size)
        x, y = transform_grid(source_x, source_y, source_grid, grid)
        valid = raw > -32000
        rounded = np.where(base + raw >= 0.0, np.floor(base + raw + 0.5),
                           np.ceil(base + raw - 0.5))
        elevation = np.where(
            valid,
            rounded.astype(np.int16).astype(COMPUTE_DTYPE)
            * COMPUTE_DTYPE(elevation_scale),
            raw,
        ).astype(COMPUTE_DTYPE, copy=False)
        radius = _fixed_peak_radius(
            np.float64(0.5) * np.float64(header.resolution[1]), grid
        ) if peak else None
        inside_total += _accumulate_points(
            x, y, elevation, grid, sums, counts, maxima, radius, receptors,
            terrain_filter
        )
    return inside_total


def _geotiff_geometry(path: Path):
    """Read one standard GeoTIFF band, validity mask, and centre geometry."""
    path = Path(path)
    try:
        with rasterio.open(path) as source:
            if source.count != 1:
                raise ValueError("terrain GeoTIFF must contain one numeric two-dimensional band")
            if source.crs is None:
                raise ValueError(f"terrain GeoTIFF has no CRS: {path}")
            data = source.read(1)
            valid = source.read_masks(1) != 0
            source_crs = CRS.from_user_input(source.crs)
            transform = source.transform
            units = (source.units[0] or "").casefold()
    except RasterioIOError as error:
        raise ValueError(f"unable to read terrain GeoTIFF: {path}") from error
    if data.ndim != 2 or data.dtype.kind not in {"u", "i", "f"}:
        raise ValueError("terrain GeoTIFF must contain one numeric two-dimensional band")
    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError("rotated terrain GeoTIFF grids are not supported")
    dx, dy = float(transform.a), float(transform.e)
    if dx <= 0.0 or dy == 0.0:
        raise ValueError("terrain GeoTIFF pixel spacing is invalid")
    x0, y0 = transform @ (0.5, 0.5)
    elevation_scale = 0.3048 if units in {"foot", "feet", "ft", "us survey foot"} else 1.0

    # GDAL does not expose TIFF orientation 4 and returns its stored row order.
    # Keep this narrow metadata fallback for uncommon bottom-up TIFFs; all CRS,
    # affine, band, unit, and pixel reads above still go through Rasterio.
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        orientation = int(page.tags[274].value) if 274 in page.tags else 1
        if orientation not in {1, 4}:
            raise ValueError(f"unsupported terrain GeoTIFF orientation {orientation}")
        if orientation == 4:
            metadata = tiff.geotiff_metadata or {}
            scale = metadata.get("ModelPixelScale")
            tie = metadata.get("ModelTiepoint")
            if scale is None or tie is None or len(scale) < 2 or len(tie) < 6:
                raise ValueError(f"terrain GeoTIFF has invalid scale or tiepoint tags: {path}")
            raster_type = int(metadata.get("GTRasterTypeGeoKey", 1))
            half = 0.5 if raster_type == 1 else 0.0
            x0 = float(tie[3]) + (half - float(tie[0])) * dx
            pixel_height = abs(dy)
            first_y = float(tie[4]) + (half - float(tie[1])) * pixel_height
            y0 = first_y + (data.shape[0] - 1) * pixel_height
            dy = -pixel_height
            data = data[::-1]
            valid = valid[::-1]
    valid &= np.isfinite(data)
    return source_crs, float(x0), float(y0), dx, dy, elevation_scale, data, valid


def _accumulate_geotiff(path, grid, sums, counts, input_datum="WGS-84", *, maxima=None,
                        peak=False, receptors=None, terrain_filter=None) -> int:
    """Accumulate GeoTIFF pixel values following LOADTIFF/SAVEELEV."""
    source_crs, x0, y0, dx, dy, elevation_scale, data, valid_data = _geotiff_geometry(
        Path(path)
    )
    to_geographic = Transformer.from_crs(source_crs, source_crs.geodetic_crs, always_xy=True)
    source_x = x0 + dx * np.arange(data.shape[1], dtype=np.float64)
    radius = None
    if peak:
        corner_source_x = np.array([source_x[0], source_x[-1], source_x[-1], source_x[0]])
        last_y = y0 + dy * (data.shape[0] - 1)
        corner_source_y = np.array([y0, y0, last_y, last_y])
        corner_lon, corner_lat = to_geographic.transform(corner_source_x, corner_source_y)
        corner_x, corner_y = project_geographic(corner_lon, corner_lat, grid, input_datum)
        dx_m = min(abs(corner_x[1] - corner_x[0]),
                   abs(corner_x[2] - corner_x[3])) / data.shape[1]
        dy_m = min(abs(corner_y[3] - corner_y[0]),
                   abs(corner_y[2] - corner_y[1])) / data.shape[0]
        radius = _fixed_peak_radius(
            np.float64(0.5) * np.hypot(np.float64(dx_m), np.float64(dy_m)), grid
        )
    inside_total = 0
    for row, raw in enumerate(data):
        source_y = np.full(source_x.shape, y0 + dy * row)
        longitude, latitude = to_geographic.transform(source_x, source_y)
        x, y = project_geographic(longitude, latitude, grid, input_datum)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        elevation = np.asarray(raw, dtype=COMPUTE_DTYPE) * COMPUTE_DTYPE(elevation_scale)
        valid = valid_data[row]
        out_of_range = valid & ((elevation > 20000.0) | (elevation < -10000.0))
        elevation[out_of_range] = 0.0
        elevation[~valid] = COMPUTE_DTYPE(-32768.0)
        inside_total += _accumulate_points(
            x, y, elevation, grid, sums, counts, maxima, radius, receptors,
            terrain_filter
        )
    return inside_total


def _write_netcdf(path, terrain, counts, grid, datum, *, peak=False) -> None:
    with Dataset(path, "w", format="NETCDF4") as ds:
        polar = grid.grid_type == 3
        y_dimension, x_dimension = (("ray", "ring") if polar else ("y", "x"))
        ds.createDimension(y_dimension, grid.ny)
        ds.createDimension(x_dimension, grid.nx)
        ds.Conventions = "CF-1.8"
        ds.source = "PyCALMET TERREL terrain preprocessing"
        ds.projection, ds.datum = grid.projection.upper(), datum
        ds.lat_0, ds.lon_0, ds.lat_1, ds.lat_2 = grid.lat0, grid.lon0, grid.lat1, grid.lat2
        ds.ttm_scale = grid.ttm_scale
        ds.grid_nx,ds.grid_ny=grid.nx,grid.ny
        ds.grid_dx_km,ds.grid_dy_km=grid.dx_km,grid.dy_km
        ds.x_origin_km,ds.y_origin_km=grid.x_origin_km,grid.y_origin_km
        ds.false_easting_km,ds.false_northing_km=(
            grid.false_easting_km,grid.false_northing_km)
        if polar:
            xv = ds.createVariable("ring_distance", "f8", (x_dimension,))
            yv = ds.createVariable("ray_angle", "f8", (y_dimension,))
            xv.units, yv.units = "km", "degree_clockwise_from_north"
            xv[:] = grid.ring_distances_km
            yv[:] = grid.ray_angles_deg
        else:
            xv = ds.createVariable("x", "f8", (x_dimension,))
            yv = ds.createVariable("y", "f8", (y_dimension,))
            xv.units = yv.units = "km"
            xv[:] = grid.x_origin_km + (np.arange(grid.nx) + 0.5) * grid.dx_km
            yv[:] = grid.y_origin_km + (np.arange(grid.ny) + 0.5) * grid.dy_km
        tv = ds.createVariable("terrain_height", "f4", (y_dimension, x_dimension),
                               fill_value=COMPUTE_DTYPE(-999.0), zlib=True)
        tv.units = "m"
        tv.long_name = ("grid-cell peak terrain elevation above mean sea level" if peak else
                        "grid-cell mean terrain elevation above mean sea level")
        tv[:] = np.asarray(terrain,dtype=COMPUTE_DTYPE).reshape(grid.ny,grid.nx)
        ds.createVariable("sample_count", "i4", (y_dimension, x_dimension),
                          zlib=True)[:] = counts.reshape(grid.ny, grid.nx)


def _cell_type_coordinates(grid):
    """Return projected cell-centre coordinates for terrain classification."""
    x_origin=float(grid.x_origin_km); y_origin=float(grid.y_origin_km)
    if grid.grid_type==3:
        distances=np.asarray(grid.ring_distances_km,dtype=COORD_DTYPE)
        angles=np.radians(np.asarray(grid.ray_angles_deg,dtype=COORD_DTYPE))
        x=x_origin+np.sin(angles)[:,None]*distances[None,:]
        y=y_origin+np.cos(angles)[:,None]*distances[None,:]
        return x,y
    x=x_origin+(np.arange(grid.nx,dtype=COORD_DTYPE)+.5)*float(grid.dx_km)
    y=y_origin+(np.arange(grid.ny,dtype=COORD_DTYPE)+.5)*float(grid.dy_km)
    return np.meshgrid(x,y,indexing="xy")


def _cell_types(grid, terrain_filter):
    x,y=_cell_type_coordinates(grid)
    return terrain_filter.types(x,y)


def _cell_averages(sums, counts):
    """Finish high-precision terrain sums as the model's FP32 field."""
    totals=np.asarray(sums,dtype=ACCUM_DTYPE)
    hits=np.asarray(counts)
    denominator=np.asarray(hits,dtype=ACCUM_DTYPE)
    average=np.full(totals.shape,-999.,dtype=ACCUM_DTYPE)
    np.divide(totals,denominator,out=average,where=hits>0)
    return average.astype(COMPUTE_DTYPE)


def _fill_polar_angular_gaps(terrain, missing_threshold=-9999.0,
                             ray_angles_deg=None):
    """Fill each missing polar cell from the nearest populated ray.

    Rays are circular and form the first dimension.  Ties choose the lower
    original ray index, making the result independent of traversal order.
    """
    f32=COMPUTE_DTYPE
    values=np.asarray(terrain,dtype=f32).copy()
    if values.ndim!=2:
        raise ValueError("TERREL FILLANG terrain must be a two-dimensional array")
    missing_limit=f32(missing_threshold)
    missing=values<=missing_limit
    entirely_missing=np.all(missing,axis=0)
    if np.any(entirely_missing):
        rings=", ".join(str(index+1) for index in np.flatnonzero(entirely_missing))
        raise ValueError(f"TERREL FILLANG cannot fill entirely empty polar ring(s): {rings}")
    ray_count=values.shape[0]
    rays=np.arange(ray_count,dtype=np.int64)
    if ray_angles_deg is None:
        angles=np.linspace(0.,2.*np.pi,ray_count,endpoint=False,dtype=COORD_DTYPE)
    else:
        angles=np.radians(np.asarray(ray_angles_deg,dtype=COORD_DTYPE))
        if angles.shape!=(ray_count,):
            raise ValueError("TERREL polar ray angles must match the terrain ray dimension")
    ray_coordinates=np.column_stack((np.sin(angles),np.cos(angles)))
    for ring in range(values.shape[1]):
        missing_rays=rays[missing[:,ring]]
        if not missing_rays.size:
            continue
        populated_rays=rays[~missing[:,ring]]
        candidate_count=min(2,populated_rays.size)
        distance,index=cKDTree(ray_coordinates[populated_rays]).query(
            ray_coordinates[missing_rays],k=candidate_count)
        if candidate_count==1:
            nearest=np.full(missing_rays.shape,populated_rays[int(index[0])],dtype=np.int64)
        else:
            nearest=populated_rays[index[:,0]]
            tied=np.isclose(distance[:,0],distance[:,1],rtol=1e-12,atol=1e-12)
            if np.any(tied):
                nearest[tied]=np.minimum(populated_rays[index[tied,0]],
                                         populated_rays[index[tied,1]])
        values[missing_rays,ring]=values[nearest,ring]
    return values


def _finalize_terrain(sums,counts,maxima,grid,peak,terrain_filter):
    average=_cell_averages(sums,counts)
    raw=maxima.copy() if peak else average
    kinds=_cell_types(grid,terrain_filter).ravel()
    final=terrain_filter.output(raw,kinds)
    counts[(raw>-999.)&(final<=-999.)]=0
    final=final.reshape(grid.ny,grid.nx)
    if grid.grid_type==3:
        # Python represents HEIGHTS' internal -32000 void as -999.  FILLANG
        # runs before output and fills either representation identically.
        final=_fill_polar_angular_gaps(
            final,missing_threshold=-998.,ray_angles_deg=grid.ray_angles_deg)
    return final


def _write_surfer_grid(path,terrain,grid):
    """Write PLTOUT's Surfer ASCII DSAA grid for a Cartesian domain."""
    valid=np.asarray(terrain)[np.asarray(terrain)>=-998.]
    if not valid.size:
        z_low,z_high=9999.,-9999.
    else:
        z_low,z_high=float(np.min(valid)),float(np.max(valid))
    x_low=grid.x_origin_km+.5*grid.dx_km
    x_high=grid.x_origin_km+(grid.nx-.5)*grid.dx_km
    y_low=grid.y_origin_km+.5*grid.dy_km
    y_high=grid.y_origin_km+(grid.ny-.5)*grid.dy_km
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="ascii",newline="\n") as stream:
        stream.write("DSAA\n")
        stream.write(f" {grid.nx} {grid.ny}\n")
        stream.write(f" {x_low:g} {x_high:g}\n")
        stream.write(f" {y_low:g} {y_high:g}\n")
        stream.write(f" {z_low:g} {z_high:g}\n")
        for row in terrain:
            for start in range(0,grid.nx,10):
                stream.write("".join(f"{value:8.1f}" for value in row[start:start+10])+"\n")


def _write_plot_output(path,terrain,grid):
    if grid.grid_type!=3:
        _write_surfer_grid(path,terrain,grid)
        return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("",encoding="ascii")


def _interp2d(x_vertices, y_vertices, values, distances, x, y, distance_max):
    """Interpolate a value from the nearest point in each quadrant."""
    xo=np.asarray(x_vertices,dtype=COORD_DTYPE)
    yo=np.asarray(y_vertices,dtype=COORD_DTYPE)
    zo=np.asarray(values,dtype=COMPUTE_DTYPE)
    dist=np.asarray(distances,dtype=COORD_DTYPE)
    x=float(x); y=float(y); distance_max=float(distance_max)
    minimum = np.min(dist)
    if minimum<=1.e-3:
        return COMPUTE_DTYPE(zo[np.flatnonzero(dist==minimum)[0]])
    if np.min(zo)<-998. or np.max(dist)>distance_max:
        return COMPUTE_DTYPE(-999.)
    dx_point=x-xo; dy_point=y-yo
    angle_point=np.arctan2(dy_point,dx_point)
    angle_point[(dx_point==0.)&(dy_point==0.)]=0.
    minimum_y = np.min(yo)
    candidates = np.flatnonzero(yo == minimum_y)
    first = int(candidates[np.argmin(xo[candidates])])
    other = [index for index in range(4) if index != first]
    angles=[]
    for index in other:
        dy=yo[index]-yo[first]
        dx=xo[index]-xo[first]+1.e-30
        angle=np.arctan2(dy,dx)
        if angle< -np.pi/2.:
            angle+=2.*np.pi
        angles.append(angle)
    angles=np.asarray(angles,dtype=COORD_DTYPE)
    second = other[int(np.argmin(angles))]
    fourth = other[int(np.argmax(angles))]
    third = 6 - first - second - fourth
    order = [first, second, third, fourth, first]
    perpendicular = np.empty(4, dtype=COORD_DTYPE)
    edge_values = np.empty(4, dtype=COORD_DTYPE)
    for edge in range(4):
        start, end = order[edge], order[edge + 1]
        dx=xo[start]-xo[end]; dy=yo[start]-yo[end]
        side=np.hypot(dx,dy)
        angle=np.arctan2(dy,dx) if dx!=0. or dy!=0. else 0.
        relative=angle_point[start]-angle
        along=abs(dist[start]*np.cos(relative))
        perpendicular[edge]=abs(dist[start]*np.sin(relative))
        edge_values[edge]=(along*zo[end]+(side-along)*zo[start])/side
        if perpendicular[edge]<1.e-3:
            return COMPUTE_DTYPE(edge_values[edge])
    def opposite(first,third):
        numerator=(edge_values[first]*perpendicular[third]+
                   edge_values[third]*perpendicular[first])
        return numerator/(perpendicular[first]+perpendicular[third])
    first_estimate=opposite(0,2); second_estimate=opposite(1,3)
    p24=perpendicular[1]+perpendicular[3]
    p13=perpendicular[0]+perpendicular[2]
    numerator=first_estimate*p24+second_estimate*p13
    return COMPUTE_DTYPE(numerator/np.sum(perpendicular))


def _fill_void_cells(sums, counts, grid, radius_km, maxima=None) -> int:
    """Fill voids from the nearest original cell within a physical radius.

    Coordinates are expressed in kilometres, so rectangular cells are handled
    directly.  Every target is evaluated against the same immutable set of
    populated cells; newly filled values never affect later targets.  Exact
    nearest-neighbour ties are averaged (and peak ties take their maximum),
    which also removes traversal-order dependence.
    """
    radius_km = float(radius_km)
    if not np.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("CELLRADKM must be a positive finite distance")

    sums_grid = np.asarray(sums).reshape(grid.ny, grid.nx)
    counts_grid = np.asarray(counts).reshape(grid.ny, grid.nx)
    maxima_grid = None if maxima is None else np.asarray(maxima).reshape(grid.ny, grid.nx)
    populated = counts_grid > 0
    missing = ~populated
    if not np.any(missing) or not np.any(populated):
        return 0

    y_index, x_index = np.indices((grid.ny, grid.nx), dtype=np.float64)
    coordinates = np.column_stack((
        ((x_index + 0.5) * float(grid.dx_km)).ravel(),
        ((y_index + 0.5) * float(grid.dy_km)).ravel(),
    ))
    populated_flat = populated.ravel()
    missing_flat = missing.ravel()
    source_coordinates = coordinates[populated_flat]
    target_coordinates = coordinates[missing_flat]
    source_values = (
        sums_grid[populated].astype(ACCUM_DTYPE, copy=False)
        / counts_grid[populated].astype(ACCUM_DTYPE, copy=False)
    )

    tree = cKDTree(source_coordinates)
    distances, nearest = tree.query(
        target_coordinates, k=1, distance_upper_bound=radius_km
    )
    fillable = np.isfinite(distances)
    if not np.any(fillable):
        return 0

    target_flat_indices = np.flatnonzero(missing_flat)[fillable]
    filled_values = np.empty(np.count_nonzero(fillable), dtype=ACCUM_DTYPE)
    filled_maxima = (
        np.empty(np.count_nonzero(fillable), dtype=COMPUTE_DTYPE)
        if maxima_grid is not None else None
    )
    source_maxima = maxima_grid[populated] if maxima_grid is not None else None
    for output_index, (point, distance, nearest_index) in enumerate(zip(
        target_coordinates[fillable], distances[fillable], nearest[fillable]
    )):
        tolerance = max(1.0e-12, float(distance) * 1.0e-12)
        candidates = np.asarray(
            tree.query_ball_point(point, float(distance) + tolerance), dtype=np.int64
        )
        candidate_distances = np.linalg.norm(
            source_coordinates[candidates] - point, axis=1
        )
        tied = candidates[np.isclose(
            candidate_distances, distance, rtol=1.0e-12, atol=tolerance
        )]
        if not tied.size:
            tied = np.asarray([nearest_index], dtype=np.int64)
        filled_values[output_index] = np.mean(source_values[tied], dtype=ACCUM_DTYPE)
        if filled_maxima is not None:
            filled_maxima[output_index] = np.max(source_maxima[tied])

    sums_grid.ravel()[target_flat_indices] = filled_values
    counts_grid.ravel()[target_flat_indices] = 1
    if maxima_grid is not None:
        maxima_grid.ravel()[target_flat_indices] = filled_maxima
    return int(target_flat_indices.size)


def _write_legacy_header(stream, grid, datum, *, dataset="TERREL.DAT", version="2.0") -> None:
    """Write the common TERREL 2.0 header emitted by ``WRTHEAD``."""
    stream.write(f"{dataset:<16}{version:<16}"
                 f"{'Header structure with coordinate parameters':<64}\n")
    stream.write(f"{2:4d}\n")
    stream.write("Produced by PyCALMET TERREL NumPy implementation\n")
    stream.write("Internal Coordinate Transformations  ---  pyproj\n")
    projection = grid.projection.upper()
    stream.write(f"{projection:<8}\n")
    latitude=lambda value: f'{abs(value):g}{"N" if value>=0 else "S"}'
    longitude=lambda value: f'{abs(value):g}{"E" if value>=0 else "W"}'
    if projection == "UTM":
        stream.write(f"{grid.utm_zone:4d}{grid.hemisphere:<4}\n")
    elif projection == "LCC":
        fields = (latitude(grid.lat0), longitude(grid.lon0),
                  latitude(grid.lat1), latitude(grid.lat2))
        stream.write("".join(f"{value:<16}" for value in fields) + "\n")
    elif projection == "PS":
        fields = (latitude(grid.lat0),longitude(grid.lon0),latitude(grid.lat1))
        stream.write("".join(f"{value:<16}" for value in fields) + "\n")
    elif projection in {"EM", "LAZA", "TTM"}:
        fields = (latitude(grid.lat0),longitude(grid.lon0))
        stream.write("".join(f"{value:<16}" for value in fields) + "\n")
    if projection in {"LCC", "LAZA", "TTM"}:
        stream.write(gfortran_real_pair(grid.false_easting_km,grid.false_northing_km))
    stream.write(f"{datum:<8}{NIMA_DATABASE_DATE:<12}\n")
    x_lower,y_lower=grid.x_origin_km,grid.y_origin_km
    if grid.grid_type==3:
        radial,_=_polar_boundaries(grid)
        extent=float(radial[-1])*.001
        x_lower-=extent; y_lower-=extent
    stream.write(f"{grid.nx:8d}{grid.ny:8d}{x_lower:12.3f}"
                 f"{y_lower:12.3f}{grid.dx_km:12.3f}{grid.dy_km:12.3f}\n")
    stream.write("KM  M   \nW_E N_S \n")


def _write_legacy(path, terrain, grid, datum, model=1) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        _write_legacy_header(stream, grid, datum)
        if model == 3:
            stream.write("RE GRIDPOLR  XXXXX  STA\n")
            stream.write(f"                    ORIG{grid.x_origin_km * 1000.0:15.1f}"
                         f"{grid.y_origin_km * 1000.0:15.1f}\n")
            distances = np.asarray(grid.ring_distances_km) * 1000.0
            for start in range(0, distances.size, 5):
                stream.write("                    DIST" +
                             "".join(f"{value:8.1f}" for value in distances[start:start + 5]) + "\n")
            angles = np.asarray(grid.ray_angles_deg)
            for start in range(0, angles.size, 5):
                stream.write("                    DDIR" +
                             "".join(f"{value:6.0f}" for value in angles[start:start + 5]) + "\n")
            for ray, angle in enumerate(angles):
                for start in range(0, grid.nx, 5):
                    stream.write("                    ELEV" + f"{angle:6.0f}" +
                                 "".join(f"{value:8.1f}"
                                         for value in terrain[ray, start:start + 5]) + "\n")
            stream.write("RE GRIDPOLR  XXXXX  END\n")
        elif model == 4 and grid.grid_type == 3:
            for ray, angle in enumerate(grid.ray_angles_deg):
                radians = np.radians(angle)
                for ring, distance in enumerate(grid.ring_distances_km):
                    x = (grid.x_origin_km + distance * np.sin(radians)) * 1000.0
                    y = (grid.y_origin_km + distance * np.cos(radians)) * 1000.0
                    stream.write(f"RE DISCCART{x:10.1f}{y:10.1f}"
                                 f"{terrain[ray, ring]:8.1f}\n")
        elif model == 4:
            for j in range(grid.ny):
                y = (grid.y_origin_km + (j + 0.5) * grid.dy_km) * 1000.0
                for i in range(grid.nx):
                    x = (grid.x_origin_km + (i + 0.5) * grid.dx_km) * 1000.0
                    stream.write(f"RE DISCCART{x:10.1f}{y:10.1f}{terrain[j, i]:8.1f}\n")
        elif model == 1:
            # CALMET rows are north-to-south.
            for row in terrain[::-1]:
                for start in range(0, grid.nx, 10):
                    stream.write("".join(f"{value:8.1f}"
                                         for value in row[start:start + 10]) + "\n")
        elif model == 5:
            for row in terrain:
                for start in range(0, grid.nx, 10):
                    stream.write("".join(f"{value:8.1f}"
                                         for value in row[start:start + 10]) + "\n")
        elif model in {2, 6}:
            if model == 6:
                x_low = grid.x_origin_km + 0.5 * grid.dx_km
                y_low = grid.y_origin_km + 0.5 * grid.dy_km
                x_high = grid.x_origin_km + (grid.nx - 0.5) * grid.dx_km
                y_high = grid.y_origin_km + (grid.ny - 0.5) * grid.dy_km
                stream.write(f" {grid.nx} {grid.ny} {x_low:g} {y_low:g} "
                             f"{x_high:g} {y_high:g} {grid.dx_km:g}\n")
            for row in terrain:
                for start in range(0, grid.nx, 16):
                    stream.write("".join(f"{value:5.0f}"
                                         for value in row[start:start + 16]) + "\n")


def run(config: TerrelConfig) -> TerrainResult:
    if config.model not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("TERREL IMODEL must be in the range 1..6")
    processing_method=(config.grid.polar_processing if config.processing_method is None
                       else config.processing_method)
    if config.grid.grid_type == 3 and processing_method not in {1, 2}:
        raise ValueError("TERREL polar IPROC must be 1 or 2")
    if config.terrain_sources:
        sources=list(config.terrain_sources)
        missing=[str(item.path) for item in sources if not Path(item.path).is_file()]
        if missing: raise FileNotFoundError('TERREL control lists missing terrain files: '+', '.join(missing))
        if len({Path(item.path).resolve() for item in sources})!=len(sources):
            raise ValueError('TERREL control lists duplicate terrain files')
        return _run_sources(config,sources)
    root = Path(config.srtm_dir)
    available={path.name.casefold():path for path in root.iterdir() if _HGT_NAME.match(path.name)}
    if config.terrain_files:
        missing=[]; tiles=[]
        for item in config.terrain_files:
            name=Path(item).name.casefold(); tile=available.get(name)
            if tile is None: missing.append(Path(item).name)
            else: tiles.append(tile)
        if missing: raise FileNotFoundError('TERREL control lists missing HGT tiles: '+', '.join(missing))
        if len(set(tiles))!=len(tiles): raise ValueError('TERREL control lists duplicate HGT tiles')
    else:
        tiles=list(available.values())
    tiles=sorted(tiles,key=_tile_origin)
    if not tiles:
        raise FileNotFoundError(f"no [NS]dd[EW]ddd.hgt files in {root}")
    if config.sampling.lower() not in {"bilinear", "legacy", "msheet0", "direct", "msheet1"}:
        raise ValueError("sampling must select MSHEET=0 interpolation or MSHEET=1 direct projection")
    direct=config.sampling.lower() in {"direct","msheet1"}
    grid = (replace(config.grid, polar_processing=processing_method)
            if config.grid.grid_type == 3 else config.grid)
    projection_grid=replace(grid,datum=config.datum)
    sums = np.zeros(grid.nx * grid.ny, dtype=ACCUM_DTYPE)
    counts = np.zeros(grid.nx * grid.ny, dtype=np.int64)
    maxima = np.full(grid.nx * grid.ny, -999.0, dtype=COMPUTE_DTYPE)
    receptors = _read_receptors(config)
    terrain_filter=_terrain_filter(config)
    if config.previous_save is not None:
        _load_save(Path(config.previous_save), grid, sums, counts, maxima, receptors)
    peak = config.model in {3, 4}
    extracted = sum(_accumulate_tile(tile, projection_grid, sums, counts,
                                     maxima=maxima, peak=peak, receptors=receptors,
                                     direct=direct,terrain_filter=terrain_filter)
                    for tile in tiles)
    if config.void_fill:
        _fill_void_cells(sums, counts, grid, config.cell_radius_km, maxima)
    if config.save_output is not None:
        _write_save(Path(config.save_output), grid, sums, counts, maxima, receptors)
    terrain=_finalize_terrain(sums,counts,maxima,grid,peak,terrain_filter)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        if output.suffix.lower() in {".nc", ".nc4", ".cdf"}:
            _write_netcdf(temporary, terrain, counts, grid, config.datum, peak=peak)
        else:
            _write_legacy(temporary, terrain, grid, config.datum, config.model)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if terrain_filter.raw_echo is not None:
        terrain_filter.raw_echo.finish()
    if config.plot_output is not None:
        _write_plot_output(config.plot_output,terrain,grid)
    if receptors is not None:
        receptor_output = Path(config.receptor_output)
        receptor_output.parent.mkdir(parents=True, exist_ok=True)
        _write_receptors(receptor_output, receptors, terrain_filter)
    valid = terrain > -998.0
    statistics=_hit_statistics(counts,config.qa_hit_threshold_percent)
    return TerrainResult(output, len(tiles), extracted, int(valid.sum()),
                         float(terrain[valid].min()), float(terrain[valid].max()),
                         *statistics)


def _run_sources(config: TerrelConfig, sources: list[TerrelSource]) -> TerrainResult:
    if config.sampling.lower() not in {"bilinear", "legacy", "msheet0", "direct", "msheet1"}:
        raise ValueError("sampling must select MSHEET=0 interpolation or MSHEET=1 direct projection")
    direct=config.sampling.lower() in {"direct","msheet1"}
    processing_method=(config.grid.polar_processing if config.processing_method is None
                       else config.processing_method)
    grid=(replace(config.grid,polar_processing=processing_method)
          if config.grid.grid_type==3 else config.grid)
    projection_grid=replace(grid,datum=config.datum)
    sums=np.zeros(grid.nx*grid.ny,dtype=ACCUM_DTYPE)
    counts=np.zeros(grid.nx*grid.ny,dtype=np.int64)
    maxima=np.full(grid.nx*grid.ny,-999.,dtype=COMPUTE_DTYPE)
    receptors=_read_receptors(config)
    terrain_filter=_terrain_filter(config)
    if config.previous_save is not None:
        _load_save(Path(config.previous_save),grid,sums,counts,maxima,receptors)
    peak=config.model in {3,4}
    extracted=0
    for source in sources:
        kind=source.input_type.upper(); source_grid=replace(projection_grid)
        if kind in {'SRTM1','SRTM3'}:
            extracted+=_accumulate_tile(Path(source.path),source_grid,sums,counts,
                                        source.input_datum,maxima=maxima,peak=peak,
                                        receptors=receptors,direct=direct,
                                        terrain_filter=terrain_filter)
        elif kind=='GTOPO30':
            extracted+=_accumulate_gtopo30(Path(source.path),source_grid,sums,counts,
                                           source.input_datum,maxima=maxima,peak=peak,
                                           receptors=receptors,direct=direct,
                                           terrain_filter=terrain_filter)
        elif kind=='GEN':
            extracted+=_accumulate_generic(Path(source.path),source_grid,sums,counts,
                                           source.input_datum,maxima=maxima,peak=peak,
                                           receptors=receptors,terrain_filter=terrain_filter)
        elif kind=='NZGEN':
            extracted+=_accumulate_nzgen(Path(source.path),source_grid,sums,counts,
                                         source.input_datum,maxima=maxima,peak=peak,
                                         receptors=receptors,terrain_filter=terrain_filter)
        elif kind=='USGSLA':
            extracted+=_accumulate_usgsla(Path(source.path),source_grid,sums,counts,
                                          source.input_datum,maxima=maxima,peak=peak,
                                          receptors=receptors,terrain_filter=terrain_filter)
        elif kind=='ARM3':
            extracted+=_accumulate_arm3(Path(source.path),source_grid,sums,counts,
                                        source.input_datum,maxima=maxima,peak=peak,
                                        receptors=receptors,direct=direct,
                                        terrain_filter=terrain_filter)
        elif kind=='3CD':
            extracted+=_accumulate_3cd(Path(source.path),source_grid,sums,counts,
                                       source.input_datum,maxima=maxima,peak=peak,
                                       receptors=receptors,direct=direct,
                                       terrain_filter=terrain_filter)
        elif kind=='DMDF':
            extracted+=_accumulate_dmdf(Path(source.path),source_grid,sums,counts,
                                        source.input_datum,maxima=maxima,peak=peak,
                                        receptors=receptors,terrain_filter=terrain_filter)
        elif kind in {'USGS90','CDED'}:
            extracted+=_accumulate_usgs_degree_dem(Path(source.path),source_grid,sums,counts,
                                                   source.input_datum,maxima=maxima,peak=peak,
                                                   receptors=receptors,direct=direct,
                                                   terrain_filter=terrain_filter)
        elif kind=='USGS30':
            extracted+=_accumulate_usgs30(Path(source.path),source_grid,sums,counts,
                                          source.input_datum,maxima=maxima,peak=peak,
                                          receptors=receptors,terrain_filter=terrain_filter)
        elif kind=='GEOTIFF':
            extracted+=_accumulate_geotiff(Path(source.path),source_grid,sums,counts,
                                           source.input_datum,maxima=maxima,peak=peak,
                                           receptors=receptors,terrain_filter=terrain_filter)
        else:
            raise NotImplementedError(f"TERREL input type {kind} is not ported yet")
    if config.void_fill:
        _fill_void_cells(sums, counts, grid, config.cell_radius_km, maxima)
    if config.save_output is not None:
        _write_save(Path(config.save_output),grid,sums,counts,maxima,receptors)
    terrain=_finalize_terrain(sums,counts,maxima,grid,peak,terrain_filter)
    output=Path(config.output); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',delete=False) as tmp:
        temporary=Path(tmp.name)
    try:
        if output.suffix.lower() in {'.nc','.nc4','.cdf'}:
            _write_netcdf(temporary,terrain,counts,grid,config.datum,peak=peak)
        else: _write_legacy(temporary,terrain,grid,config.datum,config.model)
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    if terrain_filter.raw_echo is not None:
        terrain_filter.raw_echo.finish()
    if config.plot_output is not None:
        _write_plot_output(config.plot_output,terrain,grid)
    if receptors is not None:
        receptor_output=Path(config.receptor_output)
        receptor_output.parent.mkdir(parents=True,exist_ok=True)
        _write_receptors(receptor_output,receptors,terrain_filter)
    valid=terrain>-998.
    statistics=_hit_statistics(counts,config.qa_hit_threshold_percent)
    return TerrainResult(output,len(sources),extracted,int(valid.sum()),
                         float(terrain[valid].min()),float(terrain[valid].max()),
                         *statistics)
