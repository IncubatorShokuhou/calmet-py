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
    """Summary of a completed TERREL preprocessing run.

    Attributes:
        output: Path to the primary terrain product.
        tiles: Number of terrain source files processed.
        samples_inside_grid: Number of source elevation samples accepted
            inside the target domain.
        populated_cells: Number of target cells containing final terrain data.
        minimum_m: Minimum valid terrain elevation in metres.
        maximum_m: Maximum valid terrain elevation in metres.
        average_hits: Integer mean source-sample count among non-empty cells.
        low_hit_threshold: QA count below which a populated cell is sparse.
        high_hit_threshold: QA count above which a cell is unusually dense.
        low_hit_cells: Number of populated cells below ``low_hit_threshold``.
        high_hit_cells: Number of cells above ``high_hit_threshold``.
    """
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
    """Calculate TERREL integer hit-count QA statistics.

    Args:
        counts: Per-cell source-sample counts in any shape.
        threshold_percent: Symmetric QA percentage in the inclusive range
            0 through 100. The low threshold is this percentage of the mean;
            the high threshold is ``200 - threshold_percent`` percent.

    Returns:
        A tuple ``(average, low, high, low_cells, high_cells)``. All values are
        integers; the mean excludes empty cells and uses integer division.

    Raises:
        ValueError: If ``threshold_percent`` lies outside 0 through 100.
    """
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
    """Geometry of a headerless USGS Lambert azimuthal terrain raster.

    Attributes:
        nx: Number of source columns.
        ny: Number of source rows.
        x_origin_km: Source-grid western origin in kilometres.
        y_origin_km: Source-grid southern origin in kilometres.
        latitude_origin: Projection latitude of origin in decimal degrees.
        longitude_origin: Projection longitude of origin in decimal degrees.
    """
    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float
    latitude_origin: float
    longitude_origin: float


@dataclass(frozen=True, slots=True)
class DemHeader:
    """Parsed USGS one-degree DEM type-A header.

    Attributes:
        codes: Four format-identification integers from the type-A record.
        coordinate_unit: USGS code describing horizontal coordinate units.
        elevation_unit: USGS code describing elevation units.
        corners: Float64 ``(4, 2)`` array of source corner coordinates.
        resolution: Horizontal x/y and vertical sample resolutions in the
            source header's units.
        profile_count: Number of profile rows and columns declared by the DEM.
        horizontal_datum: USGS horizontal-datum code.
    """
    codes: tuple[int, int, int, int]
    coordinate_unit: int
    elevation_unit: int
    corners: np.ndarray
    resolution: tuple[float, float, float]
    profile_count: tuple[int, int]
    horizontal_datum: int


@dataclass(slots=True)
class _ReceptorState:
    """Mutable terrain estimates for discrete receptors.

    Attributes:
        coordinates_km: Float64 ``(n, 2)`` receptor x/y coordinates in km.
        heights_m: Float32 ``(n,)`` evolving terrain estimates in metres.
        receptor_heights_m: Float32 ``(n,)`` receptor heights above terrain in
            metres, copied through to the receptor output.
        interpolate: Whether to interpolate from one nearest sample in each
            quadrant instead of selecting the local maximum.
        radius_km: Maximum search or interpolation distance in kilometres.
        vertices_x: Float64 ``(4, n)`` x coordinates of interpolation samples
            in kilometres.
        vertices_y: Float64 ``(4, n)`` y coordinates of interpolation samples
            in kilometres.
        vertices_z: Float32 ``(4, n)`` sample elevations in metres.
        distances: Float64 ``(4, n)`` receptor-to-sample distances in km.
    """
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
        """Accumulate terrain samples for every discrete receptor.

        Args:
            x_m: Source x coordinates in metres, in any shape.
            y_m: Source y coordinates in metres with the same flattened length
                as ``x_m``.
            elevation_m: Source elevations in metres with the same flattened
                length. Values below -32000 are ignored as missing.

        Returns:
            None. The receptor height or four interpolation vertices are
            updated in place. Receptors are processed in batches to bound
            temporary memory.
        """
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
        """Finalize and return the receptor terrain elevations.

        Returns:
            The mutable float32 ``(n,)`` ``heights_m`` array in metres. In
            interpolation mode, each receptor is first interpolated from its
            four stored quadrant samples; unresolved values remain missing.
        """
        if self.interpolate:
            for index, (x, y) in enumerate(self.coordinates_km):
                self.heights_m[index] = _interp2d(
                    self.vertices_x[:, index], self.vertices_y[:, index],
                    self.vertices_z[:, index], self.distances[:, index],
                    x, y, self.radius_km,
                )
        return self.heights_m


class _RawEchoWriter:
    """Bounded-memory RAWECHO stream with atomic publication.

    Attributes:
        path: Final ``RAWECHO.DAT`` destination.
        temporary: Temporary file published to ``path`` only by ``finish``.
        stream: Open ASCII text stream, or ``None`` after successful finish.
    """

    def __init__(self,path,grid,datum):
        """Open a temporary raw-sample echo and write its legacy header.

        Args:
            path: Final ASCII output path.
            grid: Target grid metadata written to the TERREL-style header.
            datum: CALPUFF datum name written to the header.

        Raises:
            OSError: If the parent directory or temporary file cannot be
                created or the header cannot be written.
        """
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
        """Append selected projected terrain samples to the echo stream.

        Args:
            x_m: Source x coordinates in metres, in any shape.
            y_m: Source y coordinates in metres, matching ``x_m``.
            elevation_m: Source elevations in metres, matching ``x_m``.
            selected: Flat integer indices or a boolean mask selecting samples.

        Returns:
            None. Selected coordinates are written in kilometres and
            elevations in metres.

        Raises:
            OSError: If the temporary stream cannot be written.
        """
        x=np.asarray(x_m).ravel()[selected]*.001
        y=np.asarray(y_m).ravel()[selected]*.001
        z=np.asarray(elevation_m).ravel()[selected]
        for x_value,y_value,z_value in zip(x,y,z):
            self.stream.write(f"{x_value:11.3f}{y_value:11.3f}{z_value:12.3f}\n")

    def finish(self):
        """Close and atomically publish the completed raw echo.

        Returns:
            None. Calling the method after completion has no effect.

        Raises:
            OSError: If the stream cannot be closed or the temporary file
                cannot replace the final destination.
        """
        if self.stream is None:
            return
        self.stream.close(); self.stream=None
        self.temporary.replace(self.path)

    def __del__(self):
        """Close and discard an unpublished temporary raw echo.

        Returns:
            None. Cleanup errors can be reported by Python's finalizer
            machinery and do not publish the partial file.
        """
        stream=getattr(self,"stream",None)
        if stream is not None:
            stream.close()
        temporary=getattr(self,"temporary",None)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _TerrainFilter:
    """Coastal terrain classification and replacement rules.

    Attributes:
        polygons: Parsed coastline polygons used to classify terrain points.
        coastal: Whether coastline classification is enabled.
        noise_replacement: Int8 replacement codes indexed by terrain type.
        noise_minimum_m: Float32 minimum plausible elevations in metres by
            terrain type.
        missing_replacement: Int8 missing-value rules by terrain type.
        default_elevation_m: Float32 replacement elevations in metres by type.
        raw_echo: Optional writer receiving unfiltered in-domain samples.
    """
    polygons: tuple
    coastal: bool
    noise_replacement: np.ndarray
    noise_minimum_m: np.ndarray
    missing_replacement: np.ndarray
    default_elevation_m: np.ndarray
    raw_echo: _RawEchoWriter | None = None

    def types(self, x_km, y_km):
        """Classify projected points into TERREL coastline types.

        Args:
            x_km: Projected x coordinates in kilometres.
            y_km: Projected y coordinates in kilometres, broadcast-compatible
                with ``x_km``.

        Returns:
            An int8 array with the broadcast coordinate shape. Non-coastal
            processing returns type 1 for every point.
        """
        if not self.coastal:
            return np.ones(np.asarray(x_km).shape, dtype=np.int8)
        return _coastline_types(x_km, y_km, self.polygons)

    def extracted(self, x_m, y_m, elevation_m, domain):
        """Apply source-sample terrain replacement rules inside the domain.

        Args:
            x_m: Projected source x coordinates in metres.
            y_m: Projected source y coordinates in metres, matching ``x_m``.
            elevation_m: Source elevations in metres, matching ``x_m``.
            domain: Boolean mask marking samples inside the target domain.

        Returns:
            A float32 elevation array in metres with the shape of
            ``elevation_m``. Noise and missing rules affect in-domain samples;
            raw samples are optionally echoed before replacement.
        """
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
        """Apply final cell-level replacement rules to gridded terrain.

        Args:
            values: Final terrain elevations in metres in any grid shape.
            kinds: Integer coastline types with the same shape as ``values``.

        Returns:
            A float32 elevation array in metres with noise and missing-value
            policies applied independently to each terrain type.
        """
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
    """Read TERREL discrete receptors and initialize interpolation state.

    Args:
        config: TERREL configuration containing receptor input layout,
            interpolation mode, and search radius.

    Returns:
        Initialized receptor state, or ``None`` when no receptor input is
        configured. Coordinates are interpreted as target-grid kilometres and
        elevations as metres.

    Raises:
        OSError: If the receptor file cannot be read.
        ValueError: If the text is not numeric or its column count differs
            from ``config.receptor_columns``.
    """
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
    """Build coastline classification and terrain replacement state.

    Args:
        config: TERREL configuration containing the target grid, optional
            GSHHS/BLN paths, five terrain-type rule arrays, and raw echo path.

    Returns:
        An immutable terrain filter ready for sample and cell processing.

    Raises:
        ValueError: If GSHHS extraction lacks a BLN destination, a replacement
            array does not contain five values, or coastline data is invalid.
        OSError: If coastline or raw-echo files cannot be read or written.
    """
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
    """Test projected metre coordinates against the closed target domain.

    Args:
        x: Projected x coordinates in metres.
        y: Projected y coordinates in metres, broadcast-compatible with ``x``.
        grid: Cartesian or polar target grid definition.

    Returns:
        Boolean array with the broadcast coordinate shape. Boundary points are
        included.
    """
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    x_lower,x_upper,y_lower,y_upper=_grid_domain_metres(grid)
    return ((x>=x_lower)&(x<=x_upper)&(y>=y_lower)&(y<=y_upper))


def _grid_domain_metres(grid):
    """Return projected target-domain bounds in metres.

    Args:
        grid: Cartesian or polar target grid definition. A polar domain uses
            its outer radial boundary around the configured origin.

    Returns:
        Float64 tuple ``(xmin, xmax, ymin, ymax)`` in metres.
    """
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
    """Finalize and write TERREL discrete-receptor elevations.

    Args:
        path: Destination ASCII receptor path.
        receptors: Accumulated receptor interpolation state.
        terrain_filter: Cell-level coastline and replacement policy applied to
            final receptor terrain elevations.

    Returns:
        None. Each output row contains x/y in kilometres, terrain elevation in
        metres, and receptor height above ground in metres.

    Raises:
        OSError: If the output cannot be opened or written.
    """
    elevation = receptors.finish()
    kinds=terrain_filter.types(receptors.coordinates_km[:,0],receptors.coordinates_km[:,1])
    elevation=terrain_filter.output(elevation,kinds)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for (x, y), terrain, height in zip(
            receptors.coordinates_km, elevation, receptors.receptor_heights_m
        ):
            stream.write(f" {x:12.5f}{y:12.5f}{terrain:10.2f}{height:10.2f}\n")


def _save_header(grid: GridConfig, receptor_count: int) -> np.ndarray:
    """Build the seven-integer TERREL CELLOUT save-file header.

    Args:
        grid: Target grid definition.
        receptor_count: Number of discrete receptors stored after the cell
            arrays.

    Returns:
        Int32 array containing integer metre origins, UTM zone, x cell size in
        metres, grid dimensions, and receptor count.
    """
    x_lower,_,y_lower,_=_grid_domain_metres(grid)
    cell_metres=np.asarray(
        np.float64(grid.dx_km)*np.float64(1000.),dtype=np.float64)
    return np.asarray(
        [int(x_lower),int(y_lower),grid.utm_zone,int(cell_metres),
         grid.nx,grid.ny,receptor_count],
        dtype=np.int32,
    )


def _fortran_records(path: Path) -> tuple[str, list[bytes]]:
    """Read classic four-byte-marker sequential unformatted records.

    Args:
        path: TERREL save-file path.

    Returns:
        A pair ``(endian, records)`` where ``endian`` is ``'<'`` or ``'>'``
        and ``records`` contains marker-free record payloads as bytes.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is truncated or has invalid record markers.
    """
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
    """Restore CELLOUT terrain accumulators and receptor state.

    Args:
        path: Existing sequential-unformatted save-file path.
        grid: Current grid, which must match the stored header.
        sums: Writable flat float array receiving accumulated elevations in m.
        counts: Writable flat integer array receiving sample counts.
        maxima: Writable flat float array receiving maximum elevations in m.
        receptors: Optional current receptor state receiving stored arrays.

    Returns:
        None. All supplied arrays are updated in place.

    Raises:
        OSError: If ``path`` cannot be read.
        ValueError: If record markers, header, shapes, or record count do not
            match the current run.
    """
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
        """Decode and size-check one save-file numeric record.

        Args:
            index: Zero-based record index in the loaded payload list.
            dtype: NumPy dtype code without the file byte-order prefix.
            size: Required number of scalar values.
            label: Human-readable record label for validation errors.

        Returns:
            A read-only one-dimensional NumPy view with the requested dtype
            and the save file's byte order.

        Raises:
            IndexError: If ``index`` is absent.
            ValueError: If the decoded record has the wrong element count.
        """
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
    """Write TERREL CELLOUT in little-endian sequential-unformatted layout.

    Args:
        path: Save-file destination. Missing parent directories are created.
        grid: Target grid written to the compatibility header.
        sums: Flat accumulated elevation sums in metres.
        counts: Flat integer source-sample counts.
        maxima: Flat maximum elevations in metres.
        receptors: Optional receptor interpolation state to append.

    Returns:
        None. Numeric records use four-byte Fortran length markers.

    Raises:
        OSError: If the destination cannot be created or written.
    """
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
    """Decode an SRTM tile's southwest integer-degree corner from its name.

    Args:
        path: Path whose basename follows ``N00E000.hgt`` conventions.

    Returns:
        Tuple ``(south_latitude, west_longitude)`` in signed integer degrees.

    Raises:
        ValueError: If the basename is not a standard SRTM HGT name.
    """
    match = _HGT_NAME.match(path.name)
    if match is None:
        raise ValueError(f"not an SRTM HGT filename: {path.name}")
    ns, lat, ew, lon = match.groups()
    return int(lat) * (1 if ns.upper() == "N" else -1), int(lon) * (1 if ew.upper() == "E" else -1)


def _read_srtm_tile(path: Path) -> np.ndarray:
    """Read a standard SRTM HGT tile through GDAL's SRTMHGT driver.

    Args:
        path: Source ``.hgt`` path.

    Returns:
        Two-dimensional integer elevation array in metres, ordered from north
        to south as stored by SRTM. Its dimensions are equal and at least two.

    Raises:
        ValueError: If GDAL cannot identify a single-band square integer HGT
            raster or the file cannot be decoded.
    """
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
    """Accumulate one SRTM HGT tile into a TERREL target grid.

    Args:
        path: Standard one-degree SRTM HGT path.
        grid: Target grid definition.
        sums: Writable flat float64 elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of the SRTM longitude and latitude.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to extend each sample footprint into adjacent peak cells.
        receptors: Optional receptor state updated from in-domain samples.
        direct: If true, project every source sample; otherwise interpolate
            projected coordinates between the tile corners.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid elevation samples accumulated inside the target grid.

    Raises:
        ValueError: If the filename or HGT raster is invalid.
        OSError: If the source cannot be opened.
    """
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
    """Derive canonical GTOPO30 geometry from a tile filename.

    Args:
        path: GTOPO30 path whose basename begins with a longitude and latitude
            code such as ``W020N90``.

    Returns:
        Tuple ``(north, west, rows, columns, samples_per_degree)``. Geographic
        bounds are signed integer degrees and canonical resolution is 120
        samples per degree.

    Raises:
        ValueError: If the basename does not match a GTOPO30 region name.
    """
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

    Args:
        path: Headerless big-endian signed-int16 GTOPO30 raster path.
        grid: Target grid definition.
        sums: Writable flat float64 elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of the source geographic coordinates.
        spec: Optional ``(north, west, rows, columns, samples_per_degree)``
            override used for small synthetic rasters. Production geometry is
            derived from ``path``.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to extend sample footprints into adjacent peak cells.
        receptors: Optional receptor state updated from in-domain samples.
        direct: If true, project every sample rather than interpolating
            between projected one-degree block corners.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid elevation samples accumulated inside the target grid.

    Raises:
        ValueError: If the name or byte size is inconsistent with the geometry.
        OSError: If file metadata cannot be read.
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
    """Read a TERREL Generic header and its scattered terrain records.

    Args:
        path: ASCII Generic terrain path.

    Returns:
        Tuple ``(coordinate_type, elevation_unit, utm_zone, points)``. Points
        are a float64 ``(n, 3)`` array. Coordinate type 1 is UTM metres, 2 is
        UTM kilometres, and 3 is ``(latitude, longitude)`` degrees. Elevation
        unit 1 is metres and 2 is feet.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the header codes, UTM zone, or point records are invalid.
    """
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
    """Add already indexed elevation samples to flat cell accumulators.

    Args:
        flat: One-dimensional integer target-cell indices.
        elevation: Elevations in metres corresponding to ``flat``.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        maxima: Optional writable flat maximum-elevation array in metres.

    Returns:
        None. ``sums``, ``counts``, and optional ``maxima`` are updated in
        place, including repeated indices.
    """
    np.add.at(sums, flat, np.asarray(elevation, dtype=sums.dtype))
    counts += np.bincount(flat, minlength=counts.size).astype(counts.dtype, copy=False)
    if maxima is not None:
        np.maximum.at(maxima, flat, elevation)


def _peak_radius_from_points(x, y, grid) -> float | tuple[float, float]:
    """Infer a source footprint radius from the first two projected points.

    Args:
        x: Projected source x coordinates in metres.
        y: Projected source y coordinates in metres.
        grid: Target Cartesian or polar grid.

    Returns:
        Half the first-point spacing. A polar grid returns a scalar radius in
        metres. A Cartesian grid returns ``(x_radius, y_radius)`` in target
        cell-index units.

    Raises:
        ValueError: If fewer than two source points are supplied.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("TERREL peak processing needs at least two source points")
    radius_m = np.float64(0.5) * np.hypot(x[1] - x[0], y[1] - y[0])
    return _fixed_peak_radius(radius_m, grid)


def _fixed_peak_radius(radius_m, grid) -> float | tuple[float, float]:
    """Convert a projected footprint radius into accumulation coordinates.

    Args:
        radius_m: Non-negative source footprint radius in metres.
        grid: Target Cartesian or polar grid definition.

    Returns:
        Scalar metres for a polar grid, or ``(x_radius, y_radius)`` in
        Cartesian cell-index units.
    """
    radius_m = float(radius_m)
    if grid.grid_type == 3:
        return radius_m
    return (
        radius_m / (float(grid.dx_km) * 1000.0),
        radius_m / (float(grid.dy_km) * 1000.0),
    )


def _polar_boundaries_km(grid):
    """Calculate polar ring and relative ray boundaries.

    Args:
        grid: Polar grid whose ring centres are ``ring_distances_km`` and ray
            bearings are ``ray_angles_deg``.

    Returns:
        Pair ``(radial, angular)`` of float64 arrays. Radial outer boundaries
        are in kilometres; angular upper boundaries are degrees relative to
        the first ray.

    Raises:
        ValueError: If ring/ray counts, ordering, first angle, or
            ``polar_processing`` are invalid.
    """
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
    """Calculate polar cell boundaries in accumulation units.

    Args:
        grid: Valid polar grid definition.

    Returns:
        Pair ``(radial, angular)`` of float64 arrays. Radial boundaries are in
        metres and angular boundaries in degrees relative to the first ray.

    Raises:
        ValueError: If the polar grid definition is invalid.
    """
    radial,angular=_polar_boundaries_km(grid)
    return np.asarray(radial*np.float64(1000.),dtype=np.float64),angular


def _polar_cell_indices(dx,dy,radial_boundaries,angular_boundaries,
                        first_angle,radius=None):
    """Assign projected offsets to polar cells and optional peak neighbours.

    Args:
        dx: Eastward offsets from the polar origin in metres.
        dy: Northward offsets from the polar origin in metres.
        radial_boundaries: Ordered outer ring boundaries in metres.
        angular_boundaries: Ordered ray boundaries in degrees relative to the
            first ray.
        first_angle: Bearing of the first ray in degrees clockwise from north.
        radius: Optional source footprint radius in metres, scalar or
            broadcast-compatible with ``dx``.

    Returns:
        Without ``radius``, tuple ``(distance, shifted_angle, ring, ray)``.
        With ``radius``, four additional arrays describe a neighbouring peak
        ring, neighbouring ray, distance to the upper ray edge, and angular
        footprint radius. Distances are metres, angles are degrees, and cell
        indices are zero based.
    """
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
    """Accumulate projected samples into flat polar-grid cell arrays.

    Args:
        x: Projected source x coordinates in metres.
        y: Projected source y coordinates in metres, matching ``x``.
        elevation: Source elevations in metres, matching ``x``.
        grid: Valid polar target grid.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        maxima: Optional writable flat maximum-elevation array in metres.
        radius_m: Optional source footprint radius in metres used to populate
            neighbouring peak cells.

    Returns:
        Number of non-missing samples whose primary ring lies inside the grid.

    Raises:
        ValueError: If the polar grid definition is invalid.
    """
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
    """Apply TERREL SAVEELEV accumulation to projected terrain samples.

    Args:
        x: Projected source x coordinates in metres.
        y: Projected source y coordinates in metres, matching ``x``.
        elevation: Source elevations in metres, matching ``x``.
        grid: Cartesian or polar target grid.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak_radius_grid: Optional footprint radius. Cartesian grids expect an
            ``(x_radius, y_radius)`` pair in cell units; polar grids expect
            metres.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid samples assigned to a primary target cell. Optional
        peak-neighbour assignments do not increase this count.

    Raises:
        ValueError: If a polar grid definition is invalid or peak radius data
            cannot be broadcast to the sample shape.
    """
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
    """Accumulate TERREL Generic points using LOADGNR semantics.

    Args:
        path: ASCII Generic terrain file.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of source coordinates.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to infer and apply a source footprint radius.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid points accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If its header, points, projection, or peak spacing is
            invalid.
    """
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
    """Accumulate New Zealand Generic terrain point records.

    Args:
        path: ASCII file containing ``id elevation longitude latitude`` rows.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of source longitude and latitude.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to infer and apply a source footprint radius.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid points accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If records do not contain four numeric columns or peak
            processing lacks enough points.
    """
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
    """Select USGSLA continent geometry from a source basename.

    Args:
        path: USGS Global Lambert Azimuthal raster path.

    Returns:
        Canonical raster and projection geometry for the named continent.

    Raises:
        ValueError: If the basename does not encode a supported continent.
    """
    name = path.name.upper()
    if name.startswith(("NA", "SA", "AP", "AF")):
        return _USGSLA_SPECS[name[:2]]
    if name.startswith("EA") and len(name) >= 7 and name[6] in {"E", "A"}:
        return _USGSLA_SPECS["EU" if name[6] == "E" else "AS"]
    raise ValueError(f"unrecognized USGSLA continent filename: {path.name}")


def _accumulate_usgsla(path, grid, sums, counts, input_datum="ESR-S", *, spec=None,
                       maxima=None, peak=False, receptors=None,
                       terrain_filter=None) -> int:
    """Accumulate a USGS Global Lambert Azimuthal two-byte DEM image.

    Args:
        path: Headerless USGSLA raster path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of the source azimuthal projection.
        spec: Optional source geometry override for synthetic test rasters.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to use the one-kilometre cell diagonal as a footprint.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid samples accumulated inside the target grid.

    Raises:
        ValueError: If the filename or raster byte size is invalid.
        OSError: If file metadata cannot be read.
    """
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
    """Parse a requested number of fixed-width integers from text lines.

    Args:
        lines: Sequence of source text lines.
        width: Character width of each integer field.
        count: Exact number of integers required.
        label: Format label included in parse and truncation errors.

    Returns:
        Pair ``(values, consumed_lines)``. ``values`` is a one-dimensional
        int64 array of length ``count``.

    Raises:
        ValueError: If a non-blank field is not an integer or the lines end
            before ``count`` values are found.
    """
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
    """Read an ARM3 panel index and packed 30-arc-second elevations.

    Args:
        path: ASCII ARM3 terrain path.

    Returns:
        Tuple ``(longitude_se, latitude_se, nx, ny, index, elevations)``.
        Origins and panel dimensions are integers; ``index`` and
        ``elevations`` are int64 arrays, with elevations in metres.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If fixed-width fields, dimensions, counts, or trailing
            records are invalid.
    """
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
    """Accumulate ARM3 30-arc-second terrain panels.

    Args:
        path: ASCII ARM3 terrain path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of source longitude and latitude.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to extend source footprints into neighbouring cells.
        receptors: Optional receptor state updated from in-domain samples.
        direct: If true, project every source point; otherwise interpolate
            coordinates between projected one-degree panel corners.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid samples accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If the ARM3 layout is invalid.
    """
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
    """Decode a 3CD tile's latitude and west-longitude code.

    Args:
        path: Path whose basename ends in a five-digit ``.3cd`` tile code.

    Returns:
        Tuple ``(south_latitude, west_longitude_east_edge)`` in integer degrees.

    Raises:
        ValueError: If the basename is not a recognized 3CD tile name.
    """
    match = _3CD_NAME.match(path.name)
    if match is None:
        raise ValueError(f"not a 3CD terrain filename: {path.name}")
    latitude, west_longitude = match.groups()
    return int(latitude), int(west_longitude)


def _accumulate_3cd(path, grid, sums, counts, input_datum="WGS-72", *, samples=1201,
                    maxima=None, peak=False, receptors=None, direct=False,
                    terrain_filter=None) -> int:
    """Accumulate a transparent-access 3CD little-endian int16 raster.

    Args:
        path: Headerless one-degree 3CD raster path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of source longitude and latitude.
        samples: Number of rows and columns, including overlapping boundaries.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to extend source footprints into neighbouring cells.
        receptors: Optional receptor state updated from in-domain samples.
        direct: If true, project every source point; otherwise interpolate
            coordinates between projected tile corners.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid samples accumulated inside the target grid.

    Raises:
        ValueError: If the filename, sample count, or raster byte size is
            invalid.
        OSError: If file metadata cannot be read.
    """
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
    """Decode one point slot from a Canadian DMDF fixed-column record.

    Args:
        record: One DMDF logical record, padded to 80 characters as needed.
        second: Whether to decode the second point slot instead of the first.

    Returns:
        Tuple ``(utm_x, utm_y, elevation)`` in metres. A million-metre x
        prefix, when present, is retained for later UTM-zone selection.

    Raises:
        ValueError: If a required fixed-column field is not an integer.
    """
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
    """Read terrain points from a Canadian DMDF transfer file.

    Args:
        path: ASCII DMDF source path.

    Returns:
        Float64 ``(n, 3)`` array containing UTM x, UTM y, and elevation in
        metres.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a point record is malformed or no points are present.
    """
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
    """Accumulate Canadian DMDF fixed-column UTM point records.

    Args:
        path: ASCII DMDF source path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of source UTM zones 11 and 12.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to use the 100-m cell diagonal as a source footprint.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid points accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If DMDF records or coordinate transformations are invalid.
    """
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
    """Parse a USGS DEM floating-point field, including Fortran exponents.

    Args:
        field: Fixed-width numeric text using ``E`` or ``D`` exponent syntax.

    Returns:
        Parsed Python floating-point value.

    Raises:
        ValueError: If ``field`` is not a valid floating-point value.
    """
    try:
        return float(field.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"invalid USGS DEM floating-point field {field!r}") from error


def _dem_records(path: Path) -> list[str]:
    """Load USGS DEM logical records from line or fixed-block storage.

    Args:
        path: USGS DEM source path.

    Returns:
        ASCII records padded to at least 1024 characters. Line-oriented files
        retain their logical line count; binary-style streams are split into
        1024-byte records.

    Raises:
        OSError: If the source cannot be read.
        UnicodeDecodeError: If the source is not ASCII.
        ValueError: If a block stream is misaligned or the file is empty.
    """
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
    """Parse a USGS DEM type-A header record.

    Args:
        record: Type-A logical record padded to 1024 characters.

    Returns:
        Structured header containing projection codes, units, source corners,
        resolutions, profile counts, and horizontal datum code.

    Raises:
        ValueError: If a fixed-width integer or floating-point field is invalid.
    """
    def integers(start, count):
        """Decode consecutive six-character integers from the header.

        Args:
            start: Zero-based character offset of the first field.
            count: Number of consecutive fields to decode.

        Returns:
            Tuple of ``count`` Python integers.

        Raises:
            ValueError: If any selected field is not an integer.
        """
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
    """Parse USGS DEM type-B elevation profiles.

    Args:
        records: Logical records including the type-A header at index zero.
        expected: Exact number of sequential profile columns required.

    Returns:
        List of ``(row, origin, elevation_base, values)`` tuples. Origins and
        bases use source units; each ``values`` member is an int64 array of raw
        elevation offsets.

    Raises:
        ValueError: If profile headers, indices, elevations, record counts, or
            trailing logical records are invalid.
    """
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
    """Accumulate USGS90 or CDED geographic A/B-record DEM profiles.

    Args:
        path: USGS/CDED DEM source path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: Fallback CALPUFF datum when the DEM code is unrecognized.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to infer a source footprint from adjacent profile points.
        receptors: Optional receptor state updated from in-domain samples.
        direct: If true, project each geographic point; otherwise interpolate
            coordinates between projected DEM corners.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid elevations accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If the DEM records, coordinate system, or profiles are
            invalid.
    """
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
    """Accumulate USGS 7.5-minute UTM A/B-record DEM profiles.

    Args:
        path: USGS30 DEM source path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum of the source UTM coordinates.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to apply half the profile spacing as a source footprint.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid elevations accumulated inside the target grid.

    Raises:
        OSError: If the source cannot be read.
        ValueError: If the DEM is not UTM, its zone is invalid, or records are
            malformed.
    """
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
    """Read one terrain GeoTIFF band, mask, CRS, and centre geometry.

    Args:
        path: Single-band terrain GeoTIFF path.

    Returns:
        Tuple ``(crs, x0, y0, dx, dy, elevation_scale, data, valid)``. ``x0``
        and ``y0`` locate the first pixel centre in CRS units; ``dx`` and
        ``dy`` are signed pixel spacings; ``data`` is the native 2-D numeric
        band; and ``valid`` is its boolean mask. ``elevation_scale`` converts
        declared feet to metres.

    Raises:
        ValueError: If the raster is unreadable, lacks a CRS, has multiple or
            nonnumeric bands, uses rotation, invalid spacing, or unsupported
            TIFF orientation.
    """
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
    """Accumulate terrain GeoTIFF pixels following LOADTIFF/SAVEELEV.

    Args:
        path: Single-band georeferenced terrain raster path.
        grid: Target grid definition.
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        input_datum: CALPUFF datum used when projecting geographic coordinates.
        maxima: Optional writable flat maximum-elevation array in metres.
        peak: Whether to infer a source footprint from raster corner spacing.
        receptors: Optional receptor state updated from in-domain samples.
        terrain_filter: Optional coastline and replacement policy.

    Returns:
        Number of valid pixels accumulated inside the target grid.

    Raises:
        ValueError: If GeoTIFF geometry or coordinate transformation is invalid.
    """
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
    """Write the canonical CF-style TERREL NetCDF product.

    Args:
        path: Destination path, whose parent must already exist.
        terrain: Final ``(ny, nx)`` float terrain field in metres.
        counts: Flat or ``(ny, nx)`` integer source-sample counts.
        grid: Target Cartesian or polar grid metadata.
        datum: CALPUFF datum name stored in global metadata.
        peak: Whether ``terrain`` represents cell maxima instead of means.

    Returns:
        None. The dataset is closed before returning.

    Raises:
        OSError: If the NetCDF file cannot be created or written.
        RuntimeError: If NetCDF dimensions or variables cannot be created.
    """
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
    """Calculate projected cell-centre coordinates for classification.

    Args:
        grid: Cartesian or polar target grid.

    Returns:
        Pair ``(x, y)`` of float64 arrays with shape ``(ny, nx)`` in
        kilometres. Polar centres use configured ring distances and ray angles.
    """
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
    """Classify every target cell using the configured coastline polygons.

    Args:
        grid: Cartesian or polar target grid.
        terrain_filter: Classification and replacement policy.

    Returns:
        Int8 ``(ny, nx)`` array of TERREL terrain-type codes.
    """
    x,y=_cell_type_coordinates(grid)
    return terrain_filter.types(x,y)


def _cell_averages(sums, counts):
    """Finish high-precision terrain sums as a model float32 field.

    Args:
        sums: Elevation sums in metres, in any shape.
        counts: Integer sample counts with the same shape as ``sums``.

    Returns:
        Float32 mean elevations in metres with the input shape. Empty cells use
        the TERREL missing value -999.
    """
    totals=np.asarray(sums,dtype=ACCUM_DTYPE)
    hits=np.asarray(counts)
    denominator=np.asarray(hits,dtype=ACCUM_DTYPE)
    average=np.full(totals.shape,-999.,dtype=ACCUM_DTYPE)
    np.divide(totals,denominator,out=average,where=hits>0)
    return average.astype(COMPUTE_DTYPE)


def _fill_polar_angular_gaps(terrain, missing_threshold=-9999.0,
                             ray_angles_deg=None):
    """Fill each missing polar cell from the nearest populated ray.

    Args:
        terrain: Two-dimensional ``(ray, ring)`` terrain array in metres.
        missing_threshold: Values at or below this threshold are missing.
        ray_angles_deg: Optional ``(ray,)`` bearings in degrees clockwise from
            north. Uniformly spaced rays are assumed when omitted.

    Returns:
        Float32 ``(ray, ring)`` copy with angular gaps filled. Rays are
        circular; exact nearest-ray ties choose the lower original ray index.

    Raises:
        ValueError: If ``terrain`` is not two dimensional, an entire ring is
            missing, or ``ray_angles_deg`` has the wrong shape.
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
    """Convert accumulators into the filtered final terrain grid.

    Args:
        sums: Flat high-precision elevation sums in metres.
        counts: Writable flat integer sample counts.
        maxima: Flat float32 maximum elevations in metres.
        grid: Cartesian or polar target grid.
        peak: Whether to output ``maxima`` rather than cell means.
        terrain_filter: Cell classification and replacement policy.

    Returns:
        Float32 terrain array with shape ``(ny, nx)`` in metres. Missing polar
        ray cells are filled angularly.

    Raises:
        ValueError: If a polar ring is wholly missing or its ray geometry is
            inconsistent.
    """
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
    """Write PLTOUT's Surfer ASCII DSAA grid for a Cartesian domain.

    Args:
        path: Destination ASCII path. Missing parent directories are created.
        terrain: ``(ny, nx)`` terrain elevations in metres.
        grid: Cartesian target grid metadata in kilometres.

    Returns:
        None. Missing cells are included in the data but excluded from the
        reported z range.

    Raises:
        OSError: If the destination cannot be created or written.
    """
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
    """Write TERREL plot output for the target grid type.

    Args:
        path: Destination plot path.
        terrain: Final ``(ny, nx)`` terrain field in metres.
        grid: Target grid definition.

    Returns:
        None. Cartesian grids receive a Surfer DSAA product; polar grids
        receive the empty compatibility file produced by the current port.

    Raises:
        OSError: If the destination cannot be created or written.
    """
    if grid.grid_type!=3:
        _write_surfer_grid(path,terrain,grid)
        return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("",encoding="ascii")


def _interp2d(x_vertices, y_vertices, values, distances, x, y, distance_max):
    """Interpolate a receptor elevation from one point in each quadrant.

    Args:
        x_vertices: Four sample x coordinates in kilometres.
        y_vertices: Four sample y coordinates in kilometres.
        values: Four sample elevations in metres.
        distances: Four receptor-to-sample distances in kilometres.
        x: Receptor x coordinate in kilometres.
        y: Receptor y coordinate in kilometres.
        distance_max: Maximum acceptable vertex distance in kilometres.

    Returns:
        Float32 interpolated elevation in metres, or -999 when any quadrant is
        unresolved or outside ``distance_max``.
    """
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
        """Blend values from one pair of opposite polygon edges.

        Args:
            first: Index of the first edge in ``edge_values``.
            third: Index of the opposite edge.

        Returns:
            Distance-weighted elevation in metres at the receptor.
        """
        numerator=(edge_values[first]*perpendicular[third]+
                   edge_values[third]*perpendicular[first])
        return numerator/(perpendicular[first]+perpendicular[third])
    first_estimate=opposite(0,2); second_estimate=opposite(1,3)
    p24=perpendicular[1]+perpendicular[3]
    p13=perpendicular[0]+perpendicular[2]
    numerator=first_estimate*p24+second_estimate*p13
    return COMPUTE_DTYPE(numerator/np.sum(perpendicular))


def _fill_void_cells(sums, counts, grid, radius_km, maxima=None) -> int:
    """Fill void cells from nearest original cells within a physical radius.

    Args:
        sums: Writable flat elevation-sum array in metres.
        counts: Writable flat integer sample-count array.
        grid: Cartesian target grid whose cell spacings are in kilometres.
        radius_km: Positive finite maximum centre-to-centre fill distance in
            kilometres.
        maxima: Optional writable flat maximum-elevation array in metres.

    Returns:
        Number of previously empty cells filled. Newly filled cells receive a
        count of one. Exact nearest-neighbour ties average mean terrain and use
        the greatest tied peak.

    Raises:
        ValueError: If ``radius_km`` is not positive and finite.
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
    """Write the common legacy TERREL header emitted by ``WRTHEAD``.

    Args:
        stream: Open writable ASCII text stream.
        grid: Cartesian or polar target grid metadata.
        datum: CALPUFF datum name.
        dataset: Legacy dataset identifier occupying the first header field.
        version: Legacy file-format version text.

    Returns:
        None. Projection parameters, datum metadata, grid dimensions, origins,
        and units are appended to ``stream``.

    Raises:
        OSError: If the stream cannot be written.
        ValueError: If a polar grid definition is invalid.
    """
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
    """Write a legacy TERREL product for the selected CALPUFF model layout.

    Args:
        path: Destination ASCII path.
        terrain: Final ``(ny, nx)`` terrain field in metres.
        grid: Cartesian or polar target grid metadata.
        datum: CALPUFF datum name written to the shared header.
        model: TERREL ``IMODEL`` output layout from 1 through 6.

    Returns:
        None. Rows and receptor-style records follow the selected historical
        model format.

    Raises:
        OSError: If the destination cannot be created or written.
        ValueError: If required polar geometry is invalid.
    """
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
    """Generate a terrain product from SRTM tiles or configured source types.

    Args:
        config: TERREL configuration defining sources, target grid, averaging
            or peak output, optional void fill/receptors/coastline QA, and
            output paths.

    Returns:
        Summary containing the primary output path, processed source and
        sample counts, valid elevation range in metres, and hit-count QA.

    Raises:
        FileNotFoundError: If the source directory, selected tiles, or explicit
            terrain sources are missing.
        ValueError: If model, polar processing, sampling, grid, duplicate
            sources, input formats, or resulting terrain are invalid.
        NotImplementedError: If an explicit source has an unsupported type.
        OSError: If an input cannot be read or an output cannot be written.

    Notes:
        The primary output is published atomically. Optional save, raw echo,
        plot, coastline, and receptor products are also written as configured.
    """
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
    """Run TERREL for an explicit ordered list of heterogeneous sources.

    Args:
        config: TERREL configuration supplying the target grid and all
            processing and output options.
        sources: Existing terrain sources with per-file type, datum, and path.

    Returns:
        Summary containing the primary output path, source/sample counts,
        valid elevation range in metres, and hit-count QA.

    Raises:
        ValueError: If sampling, grid geometry, an input format, or resulting
            terrain is invalid.
        NotImplementedError: If a source ``input_type`` is not implemented.
        OSError: If a source cannot be read or an output cannot be written.

    Notes:
        The primary output is published atomically. Optional compatibility,
        QA, coastline, plot, and receptor outputs follow ``config``.
    """
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
