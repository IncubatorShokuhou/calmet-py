"""Land-use preprocessing for standard rasters and CALMET source formats.

Sample indexing and histogram binning follow :mod:`pycalmet.backend`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass,replace
from os import PathLike
from pathlib import Path
import re
import tempfile
from typing import Final, SupportsInt, TypeAlias, TypeVar, cast
import numpy as np
from numpy.typing import ArrayLike, NDArray
from netCDF4 import Dataset
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import RasterioIOError
from shapely import contains_xy,intersects_xy,make_valid
from shapely.geometry import Polygon

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS, Transformer
from .backend import using_jax
from .config import CtgprocConfig,CtgprocSource,GridConfig
from ._device_kernels import bincount_hits as _device_bincount_hits
from ._device_kernels import grid_indices as _device_grid_indices
from .ctg_data import MODIS_SPEC,NLCD92_SPECS,ModisSpec,Nlcd92Spec
from .datums import NIMA_DATABASE_DATE,normalize_datum
from .gshhs import extract_gshhs
from .legacy_text import gfortran_real_pair
from .projections import (geographic_crs,grid_crs,project_geographic,
                          project_from_crs,project_glaz_to_lcc,transform_grid,unproject_geographic,
                          unproject_to_crs)

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]
Int16Array: TypeAlias = NDArray[np.int16]
Int8Array: TypeAlias = NDArray[np.int8]
BoolArray: TypeAlias = NDArray[np.bool_]
RasterArray: TypeAlias = NDArray[np.generic]
PathInput: TypeAlias = str | PathLike[str]
SnowRecord: TypeAlias = tuple[int, int]
SnowGrouped: TypeAlias = dict[tuple[int, int], list[SnowRecord]]
SnowRow: TypeAlias = tuple[int, int, int, np.float32, int]
AccumulationResult: TypeAlias = tuple[Int64Array, int, int]
_ReaderResultT = TypeVar('_ReaderResultT')

USGS38: Final[Int16Array] = np.array([11,12,13,14,15,16,17,21,22,23,24,31,32,33,41,42,43,51,52,53,54,55,61,62,71,72,73,74,75,76,77,81,82,83,84,85,91,92], dtype=np.int16)
GLOBAL_MERGE_TYPES: Final[set[str]] = {'GLAZNA','GLAZSA','GLAZEU','GLAZAS','GLAZAF','GLAZAP',
                    'GLC2K','UMDGLC','MODIS'}
GLAZ_LIMITS: Final[dict[str, tuple[float, float, float, float]]] = {
    'GLAZNA':(90.,8.5,-20.,-180.),'GLAZSA':(13.,-60.,-32.,-100.),
    'GLAZEU':(90.,8.8,60.,-25.),'GLAZAS':(90.,-6.,200.,60.),
    'GLAZAF':(46.,-39.,60.,-30.),'GLAZAP':(22.,-51.,175.,88.),
}
GLAZ_TO_USGS: Final[Int16Array] = np.array([0,16,21,21,21,21,21,31,32,33,33,41,41,42,42,43,52,62,61,77,82,81,85,83,91], dtype=np.int16)
NZ_TO_USGS: Final[Int16Array] = np.array([
    0,55,0,0,0,0,0,0,0,0,16,14,0,0,0,0,0,0,0,0,21,22,0,0,0,0,0,0,0,0,
    43,43,0,0,0,0,0,0,0,0,32,0,0,61,0,0,0,0,0,0,52,0,0,0,0,0,0,0,0,0,
    62,62,0,0,0,0,0,0,0,0,75,72,74],dtype=np.int16)
NLCD92_TO_USGS: Final[Int16Array] = np.zeros(256,dtype=np.int16)
NLCD92_TO_USGS[[11,12,21,22,23,31,32,33,41,42,43,51,61,71,81,82,83,84,85,91,92]]=(
    52,91,11,11,13,74,75,76,41,42,43,32,22,31,21,21,21,21,21,61,62)
NLCD01_TO_USGS: Final[Int16Array] = np.zeros(256,dtype=np.int16)
NLCD01_TO_USGS[[11,12,21,22,23,24,31,32,41,42,43,51,52,71,72,73,74,81,82,
                91,92,93,94,95,96,97,98,99,100]]=(
    52,91,30,11,11,13,74,72,41,42,43,32,32,31,82,82,82,21,21,
    61,61,61,61,61,62,62,62,62,62)
GLC2K_TO_USGS: Final[Int16Array] = np.array([0,42,41,41,42,41,43,61,61,43,33,32,32,21,31,
                         62,21,22,22,77,55,91,11,55,0],dtype=np.int16)
UMDGLC_TO_USGS: Final[Int16Array] = np.array([55,42,42,41,41,43,32,33,32,32,31,21,21,16,77],
                         dtype=np.int16)
MODIS_TO_USGS: Final[Int16Array] = np.array([0,52,42,42,41,41,43,43,33,43,32,31,62,21,16,21,
                        91,77,52],dtype=np.int16)
CORINE_TO_USGS: Final[Int16Array] = np.array([0,16,16,13,14,14,14,75,75,75,22,22,24,24,21,22,
    22,22,21,21,21,21,43,41,42,43,31,82,32,81,72,74,83,74,91,61,61,62,
    62,72,51,52,54,54,55],dtype=np.int16)
CORINE_CLASSES: Final[NDArray[np.int32]] = np.array([111,112,121,122,123,124,131,132,133,141,142,211,
    212,213,221,222,223,231,241,242,243,244,311,312,313,321,322,323,324,
    331,332,333,334,335,411,412,421,422,423,511,512,521,522,523],dtype=np.int32)
CORINE_RGB: Final[NDArray[np.int32]] = np.array([230000077,255000000,204077242,204000000,230204204,
    230204230,166000204,166077000,255077255,255166255,255230255,255255168,
    255255000,230230000,230128000,242166077,230166000,230230077,255230166,
    255230077,230204077,242204166,128255000,166000,77255000,204242077,
    166255128,166230077,166242000,230230230,204204204,204255204,0,
    166230204,166166255,77077255,204204255,230230255,166166230,204242,
    128242230,255166,166255230,230242255],dtype=np.int32)
QA_FILENAMES: Final[tuple[str, ...]] = ('qamesh.dat','qactg.dat','qaglaz.dat','qanzgen.dat',
              'qanlcd.dat','qacorine.dat','qagcl2k.dat','qaumdglc.dat','qamodis.dat',
              'qageneric.dat')

COMPUTE_DTYPE: Final[type[np.float32]] = np.float32
COORD_DTYPE: Final[type[np.float64]] = np.float64

@dataclass(frozen=True,slots=True)
class GlazSpec:
    """Describe one raw global GLAZ equal-area raster.

    Attributes:
        nx: Number of raster columns.
        ny: Number of raster rows.
        x_origin_km: X coordinate of the first pixel centre, in kilometres.
        y_origin_km: Y coordinate of the first pixel centre, in kilometres.
        latitude_origin: Lambert azimuthal equal-area latitude of origin, in
            degrees north.
        longitude_origin: Lambert azimuthal equal-area central longitude, in
            degrees east.
    """

    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float
    latitude_origin: float
    longitude_origin: float

GLAZ_SPECS: Final[dict[str, GlazSpec]] = {
    'GLAZNA':GlazSpec(9223,8996,-4487.,-4515.,50.,-100.),
    'GLAZSA':GlazSpec(6000,8000,-3000.,-4899.,-15.,-60.),
    'GLAZEU':GlazSpec(13000,13000,-3000.,-4999.,55.,20.),
    'GLAZAS':GlazSpec(13000,12000,-8000.,-5499.,45.,100.),
    'GLAZAF':GlazSpec(8350,9276,-4458.,-4795.,5.,20.),
    'GLAZAP':GlazSpec(9300,8000,-5000.,-3944.891,-15.,135.),
}

@dataclass(frozen=True, slots=True)
class LanduseResult:
    """Summarize a completed CTGPROC land-use aggregation.

    Attributes:
        output: Generated land-use product path.
        source_cells_read: Number of source raster cells or point records read.
        samples_inside_grid: Number of mesh samples accepted by the target grid.
        populated_cells: Number of target cells with at least one sample.
        average_hits: Integer mean source-sample count per target cell.
        low_hit_threshold: QA hit-count threshold derived from the configured
            percentage of ``average_hits``.
        low_hit_cells: Number of target cells below the QA threshold.
    """

    output: Path
    source_cells_read: int
    samples_inside_grid: int
    populated_cells: int
    average_hits: int
    low_hit_threshold: int
    low_hit_cells: int


@dataclass(frozen=True,slots=True)
class CoastPolygon:
    """Store one projected COASTRD polygon.

    Attributes:
        kind: Nested coastline polygon type, from 1 through 4.
        x: One-dimensional ``float64`` x coordinates in target-grid kilometres.
        y: One-dimensional ``float64`` y coordinates in target-grid kilometres.
    """

    kind: int
    x: Float64Array
    y: Float64Array


def _grid_indices(
    x: ArrayLike, y: ArrayLike, grid: GridConfig
) -> tuple[Int64Array, Int64Array]:
    """Convert projected coordinates to zero-based target-cell indices.

    Args:
        x: Scalar or array-like x coordinate in grid kilometres.
        y: Scalar or array-like y coordinate in grid kilometres.
        grid: Target grid origin and cell spacing.

    Returns:
        Pair of ``int64`` arrays ``(ix, iy)`` with the respective input shapes.
        Index values are not clipped to the grid extent.
    """
    if using_jax():
        return _device_grid_indices(
            x, y, float(grid.x_origin_km), float(grid.y_origin_km),
            float(grid.dx_km), float(grid.dy_km),
        )
    x=np.asarray(x,dtype=COORD_DTYPE); y=np.asarray(y,dtype=COORD_DTYPE)
    delx=x-float(grid.x_origin_km); dely=y-float(grid.y_origin_km)
    qx=delx/float(grid.dx_km); qy=dely/float(grid.dy_km)
    ix=np.floor(qx).astype(np.int64)
    iy=np.floor(qy).astype(np.int64)
    return ix,iy


def _inside_grid(
    x: ArrayLike, y: ArrayLike, grid: GridConfig
) -> tuple[Int64Array, Int64Array, BoolArray]:
    """Locate projected samples and flag those inside the target grid.

    Args:
        x: Scalar or array-like x coordinate in grid kilometres.
        y: Scalar or array-like y coordinate in grid kilometres.
        grid: Target grid origin, dimensions, and spacing.

    Returns:
        Tuple ``(ix, iy, inside)`` of ``int64`` cell-index arrays and a Boolean
        mask. Arrays have the corresponding input shape.
    """
    ix,iy=_grid_indices(x,y,grid)
    return ix,iy,(ix>=0)&(ix<grid.nx)&(iy>=0)&(iy<grid.ny)


def _grid_centers_1d(origin: float, spacing: float, count: int) -> Float64Array:
    """Construct regularly spaced one-dimensional cell-centre coordinates.

    Args:
        origin: Coordinate of the outer grid edge, in caller-defined units.
        spacing: Signed cell spacing in the same units.
        count: Number of cells.

    Returns:
        One-dimensional ``float64`` array of length ``count``.
    """
    index=np.arange(int(count),dtype=COORD_DTYPE)+.5
    return float(origin)+float(spacing)*index


def _snow_grid_indices(
    longitude: ArrayLike, latitude: ArrayLike
) -> tuple[Int64Array, Int64Array]:
    """Map geographic coordinates to CTGPROC's fixed SNODAS grid.

    Args:
        longitude: Scalar or array-like longitude in degrees east on WGS 84.
        latitude: Scalar or array-like latitude in degrees north on WGS 84.

    Returns:
        Pair of one-based ``int64`` SNODAS column and row arrays.
    """
    longitude=np.asarray(longitude,dtype=COORD_DTYPE)
    latitude=np.asarray(latitude,dtype=COORD_DTYPE)
    lon_ratio=(longitude+124.729584)/8.33333377e-3
    lat_ratio=(latitude-52.8704185)/-8.33333377e-3
    isn=np.floor(lon_ratio+.5).astype(np.int64)+1
    jsn=np.floor(lat_ratio+.5).astype(np.int64)+1
    return isn,jsn


@dataclass(slots=True)
class SnowGridInfo:
    """Store the sparse CTGPROC-to-SNODAS sample mapping.

    Attributes:
        grid: Target CTGPROC grid used to flatten cell indices.
        hits: Hit counts keyed by ``(cell, land-use category, SNODAS cell)``.
        order: First-seen SNODAS cell order for each target-cell/category pair.
    """

    grid: GridConfig
    hits: dict[tuple[int, int, int], int]
    order: dict[tuple[int, int], list[int]]

    @classmethod
    def create(cls, grid: GridConfig) -> SnowGridInfo:
        """Create an empty sparse snow mapping.

        Args:
            grid: Target CTGPROC grid.

        Returns:
            Empty mapping associated with ``grid``.
        """
        return cls(grid, {}, {})

    def add_hits(
        self,
        cell: SupportsInt,
        category: SupportsInt,
        snow_grid: SupportsInt,
        count: SupportsInt = 1,
    ) -> None:
        """Accumulate one sparse snow-grid record without fixed array limits.

        Args:
            cell: Zero-based flattened target-grid cell index.
            category: Land-use category identifier.
            snow_grid: One-based flattened SNODAS grid index.
            count: Number of samples to add.

        Returns:
            None.
        """
        group=(int(cell),int(category)); snow_grid=int(snow_grid)
        grids=self.order.setdefault(group,[])
        if snow_grid not in grids:
            grids.append(snow_grid)
        key=(group[0],group[1],snow_grid)
        self.hits[key]=self.hits.get(key,0)+int(count)

    def merge(self, other: SnowGridInfo) -> None:
        """Merge sparse records from a continuation dataset.

        Args:
            other: Sparse mapping on the same target grid.

        Returns:
            None. This mapping is updated in place.
        """
        for (cell,category),grids in other.order.items():
            for snow_grid in grids:
                self.add_hits(cell,category,snow_grid,
                              other.hits[cell,category,snow_grid])

    def remap(self, mapping: Mapping[int, int]) -> SnowGridInfo:
        """Remap snow-record land-use categories like the main grid.

        Args:
            mapping: Source-to-output integer category mapping.

        Returns:
            New sparse mapping with hit counts merged under remapped categories.
        """
        remapped=type(self).create(self.grid)
        for (cell,category),grids in self.order.items():
            target=mapping.get(category,category)
            for snow_grid in grids:
                remapped.add_hits(cell,target,snow_grid,
                                  self.hits[cell,category,snow_grid])
        return remapped

    def add(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categories: Int16Array,
        good: BoolArray,
    ) -> None:
        """Add accepted projected samples to the sparse SNODAS mapping.

        Args:
            x: One-dimensional projected sample x coordinates in kilometres.
            y: One-dimensional projected sample y coordinates in kilometres.
            categories: Land-use category for each sample.
            good: Boolean mask selecting samples accepted by CTGPROC.

        Returns:
            None. Records are accumulated in place.
        """
        selected=np.flatnonzero(good)
        if selected.size==0:
            return
        x=np.asarray(x)[selected]; y=np.asarray(y)[selected]
        category=np.asarray(categories,dtype=np.int64)[selected]
        ix,iy=_grid_indices(x,y,self.grid)
        cell=iy*self.grid.nx+ix
        lon,lat=unproject_geographic(x,y,self.grid,'WGS-84',units='km')
        isn,jsn=_snow_grid_indices(lon,lat)
        snow=(jsn-1)*6935+isn
        # Preserve first-seen order so serialized records are deterministic.
        for raw_cell,raw_category,raw_snow in zip(cell,category,snow,strict=True):
            self.add_hits(raw_cell,raw_category,raw_snow)

    def records(self, categories: Int16Array) -> tuple[SnowGrouped, Int64Array]:
        """Encode sparse snow hits using the output category ordering.

        Args:
            categories: Ordered output land-use category identifiers.

        Returns:
            Pair ``(grouped, totals)``. ``grouped`` maps zero-based
            ``(target cell, one-based category position)`` keys to ordered
            ``(SNODAS cell, hit count)`` records. ``totals`` is an ``int64``
            vector with one total per flattened target cell.

        Raises:
            ValueError: If a recorded category is absent from ``categories``.
        """
        lookup={int(value):index+1 for index,value in enumerate(categories)}
        grouped: SnowGrouped = {}
        totals=np.zeros(self.grid.nx*self.grid.ny,dtype=np.int64)
        for (cell,category,snow),count in self.hits.items():
            if category not in lookup:
                raise ValueError(f'snow land-use category {category} is absent from output categories')
            totals[cell]+=count
        if self.order:
            for (cell,category),grids in self.order.items():
                land=lookup[category]
                grouped[cell,land]=[(snow,self.hits[cell,category,snow]) for snow in grids]
        else:
            # Compatibility for callers that populated ``hits`` directly.
            for (cell,category,snow),count in self.hits.items():
                grouped.setdefault((cell,lookup[category]),[]).append((snow,count))
        return grouped,totals


def _bin_samples(
    counts: Int64Array,
    columns: Mapping[int, int],
    x: ArrayLike,
    y: ArrayLike,
    categories: Int16Array,
    good: BoolArray,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> None:
    """Accumulate accepted source samples into target land-use counts.

    Args:
        counts: Mutable ``(category, flattened cell)`` integer count array.
        columns: Mapping from category identifiers to rows in ``counts``.
        x: Projected sample x coordinates in target-grid kilometres.
        y: Projected sample y coordinates in target-grid kilometres.
        categories: Category identifier for each sample.
        good: Boolean mask selecting samples to accumulate.
        config: Active CTGPROC configuration and target grid.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        None. ``counts`` and, when supplied, ``snow`` are updated in place.
    """
    if snow is not None:
        snow.add(x,y,categories,good)
    ix,iy=_grid_indices(np.asarray(x)[good],np.asarray(y)[good],config.grid)
    cells=iy*config.grid.nx+ix
    selected_categories=np.asarray(categories)[good]
    for category in np.unique(selected_categories):
        selected=selected_categories==category
        if using_jax():
            counts[columns[int(category)]] += _device_bincount_hits(
                cells[selected], int(counts.shape[1]),
            )
        else:
            counts[columns[int(category)]]+=np.bincount(
                cells[selected],minlength=counts.shape[1]).astype(counts.dtype,copy=False)


def _read_with_optional_snow(
    function: Callable[..., _ReaderResultT],
    *args: object,
    snow: SnowGridInfo | None = None,
    **kwargs: object,
) -> _ReaderResultT:
    """Call a source reader with snow bookkeeping only when enabled.

    Args:
        function: Reader callable accepting ``*args`` and, optionally, ``snow``.
        *args: Positional arguments forwarded to ``function``.
        snow: Optional sparse snow mapping.
        **kwargs: Keyword arguments forwarded to ``function``.

    Returns:
        Unmodified return value from ``function``.
    """
    if snow is None:
        return function(*args,**kwargs)
    return function(*args,snow=snow,**kwargs)


def _qa_directory(config: CtgprocConfig) -> Path:
    """Resolve the directory for CTGPROC sample-coordinate QA files.

    Args:
        config: CTGPROC output and QA-directory settings.

    Returns:
        Explicit QA directory, or the main output file's parent directory.
    """
    return (Path(config.qa_output_directory) if config.qa_output_directory is not None
            else Path(config.output).parent)


def _qa_write(
    config: CtgprocConfig, name: str, *columns: ArrayLike
) -> None:
    """Append projected sample columns to one optional QA text file.

    Args:
        config: CTGPROC QA enable flag and destination settings.
        name: QA filename relative to the resolved QA directory.
        *columns: Array-like equally sized columns written as rows.

    Returns:
        None. Disabled or empty QA output performs no I/O.

    Raises:
        ValueError: If the columns cannot be combined into equal-length rows.
        OSError: If the QA file cannot be opened or written.
    """
    if not config.qa_cell_output:
        return
    values=np.column_stack(tuple(np.asarray(column).ravel() for column in columns))
    if values.size==0:
        return
    with (_qa_directory(config)/name).open('ab') as stream:
        np.savetxt(stream,values,fmt='%.8g')


def _read_coastline_bln(
    path: PathInput, grid: GridConfig, *, allow_empty: bool = False
) -> tuple[CoastPolygon, ...]:
    """Read COASTRD's pre-extracted target-grid BLN polygons.

    Args:
        path: ASCII BLN coastline file.
        grid: Target grid used to validate polygon coverage; coordinates are
            interpreted in this grid's kilometres.
        allow_empty: Return an empty tuple instead of rejecting an empty file.

    Returns:
        Tuple of validated :class:`CoastPolygon` objects in file order.

    Raises:
        ValueError: If headers, points, polygon types, lengths, or grid bounds
            are invalid.
        OSError: If the file cannot be read.
    """
    source=Path(path)
    lines=source.read_text(encoding='ascii',errors='strict').splitlines()
    polygons: list[CoastPolygon] = []; index=0
    while index<len(lines):
        if not lines[index].strip():
            index+=1; continue
        header=re.fullmatch(r'\s*(\d+)\s*(?:,\s*|\s+)([-+]?\d+)\s*(?:,\s*|\s+)(.*?)\s*',
                            lines[index])
        index+=1
        if header is None: raise ValueError(f'invalid coastline BLN header at {source}:{index}')
        try:
            count=int(header.group(1))
            identifiers=[int(value) for value in re.findall(r'-?\d+',header.group(3))]
        except ValueError as error:
            raise ValueError(f'invalid coastline BLN header at {source}:{index}') from error
        if count<3 or not identifiers or identifiers[0] not in {1,2,3,4}:
            raise ValueError(f'invalid coastline polygon at {source}:{index}')
        if index+count>len(lines): raise ValueError(f'truncated coastline polygon at {source}:{index}')
        points: list[tuple[float, float]] = []
        for number,line in enumerate(lines[index:index+count],start=index+1):
            try: point=tuple(float(value) for value in line.replace(',',' ').split()[:2])
            except ValueError as error:
                raise ValueError(f'invalid coastline point at {source}:{number}') from error
            if len(point)!=2: raise ValueError(f'invalid coastline point at {source}:{number}')
            points.append(point)
        index+=count
        coordinates=np.asarray(points,dtype=np.float64)
        polygons.append(CoastPolygon(identifiers[0],coordinates[:,0],coordinates[:,1]))
    if not polygons:
        if allow_empty: return ()
        raise ValueError(f'no coastline polygons found in {source}')
    xmin=min(np.min(item.x) for item in polygons); xmax=max(np.max(item.x) for item in polygons)
    ymin=min(np.min(item.y) for item in polygons); ymax=max(np.max(item.y) for item in polygons)
    grid_xmax=grid.x_origin_km+grid.nx*grid.dx_km
    grid_ymax=grid.y_origin_km+grid.ny*grid.dy_km
    if xmax<grid.x_origin_km or xmin>grid_xmax or ymax<grid.y_origin_km or ymin>grid_ymax:
        raise ValueError('coastline BLN limits are not consistent with the CTGPROC grid')
    return tuple(polygons)


def _polygon_membership(
    px: ArrayLike, py: ArrayLike, polygon: CoastPolygon
) -> Int8Array:
    """Classify points relative to one coastline polygon.

    Args:
        px: Array-like point x coordinates in target-grid kilometres.
        py: Array-like point y coordinates in target-grid kilometres.
        polygon: Coastline polygon in the same target-grid CRS.

    Returns:
        ``int8`` array with ``-1`` outside, ``0`` on the boundary, and ``1``
        inside, with the broadcast point shape.
    """
    px=np.asarray(px,dtype=COORD_DTYPE); py=np.asarray(py,dtype=COORD_DTYPE)
    coordinates=np.column_stack((np.asarray(polygon.x,dtype=COORD_DTYPE),
                                 np.asarray(polygon.y,dtype=COORD_DTYPE)))
    geometry=Polygon(coordinates)
    if not geometry.is_valid:
        geometry=make_valid(geometry)
    inside=contains_xy(geometry,px,py)
    intersects=intersects_xy(geometry,px,py)
    return np.where(inside,1,np.where(intersects,0,-1)).astype(np.int8,copy=False)


def _coastline_types(
    x: ArrayLike, y: ArrayLike, polygons: Sequence[CoastPolygon]
) -> Int8Array:
    """Apply IPTYPE's nested coastline classification to projected points.

    Args:
        x: Array-like point x coordinates in target-grid kilometres.
        y: Array-like point y coordinates in target-grid kilometres.
        polygons: Ordered coastline polygons with nesting types 1 through 4.

    Returns:
        ``int8`` array matching ``x`` with the highest containing polygon
        type; zero denotes marine or otherwise unclassified points.
    """
    result=np.zeros(np.asarray(x).shape,dtype=np.int8)
    flat=result.ravel(); px=np.asarray(x).ravel(); py=np.asarray(y).ravel()
    for polygon in polygons:
        bounds=((px>=np.min(polygon.x))&(px<=np.max(polygon.x))&
                (py>=np.min(polygon.y))&(py<=np.max(polygon.y))&
                (flat<polygon.kind))
        if not bounds.any(): continue
        selected=np.flatnonzero(bounds)
        membership=_polygon_membership(px[selected],py[selected],polygon)
        inside=membership==1; flat[selected[inside]]=np.maximum(flat[selected[inside]],polygon.kind)
        edge=membership==0
        flat[selected[edge]]={1:0,2:2,3:2,4:4}[polygon.kind]
    return result


def _fill_empty_ocean(
    counts: Int64Array, categories: ArrayLike, config: CtgprocConfig
) -> int:
    """Seed unsampled marine target cells with one ocean-category hit.

    Args:
        counts: Mutable ``(category, y, x)`` integer hit array.
        categories: Category identifier corresponding to each first-axis row.
        config: CTGPROC coastline, ocean-category, and target-grid settings.

    Returns:
        Number of previously empty cells classified and filled as ocean.

    Raises:
        ValueError: If no coastline file is configured, the ocean category is
            missing or duplicated, or coastline geometry is invalid.
    """
    if config.coastline_file is None:
        raise ValueError('fill_empty_ocean requires coastline_file/COASTBLN')
    matches=np.flatnonzero(np.asarray(categories)==config.ocean_category)
    if matches.size!=1:
        raise ValueError('ocean_category/IOCEAN must be in the USGS-38 category table')
    polygons=_read_coastline_bln(config.coastline_file,config.grid,
                                  allow_empty=config.gshhs_file is not None)
    x=_grid_centers_1d(config.grid.x_origin_km,config.grid.dx_km,config.grid.nx)
    y=_grid_centers_1d(config.grid.y_origin_km,config.grid.dy_km,config.grid.ny)
    xx,yy=np.meshgrid(x,y,indexing='xy')
    ocean=(counts.sum(axis=0)==0)&(_coastline_types(xx,yy,polygons)==0)
    counts[matches[0],ocean]=1
    return int(ocean.sum())


def _coastline_polygons(config: CtgprocConfig) -> tuple[CoastPolygon, ...]:
    """Load coastline polygons when land/marine category swapping is enabled.

    Args:
        config: CTGPROC coastline-swap and target-grid settings.

    Returns:
        Tuple of coastline polygons, or an empty tuple when swapping is off.

    Raises:
        ValueError: If swapping is enabled without a coastline file or the
            coastline file is invalid.
    """
    if not config.coastline_swap:
        return ()
    if config.coastline_file is None:
        raise ValueError('coastline_swap/LMARSWAP requires coastline_file/COASTBLN')
    return _read_coastline_bln(config.coastline_file,config.grid,
                               allow_empty=config.gshhs_file is not None)


def _swap_coastline_categories(
    categories: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    config: CtgprocConfig,
    polygons: Sequence[CoastPolygon],
) -> tuple[Int16Array, BoolArray]:
    """Apply LUGRID's LMARSWAP rule to projected source samples.

    Args:
        categories: Source land-use categories.
        x: Sample x coordinates in target-grid kilometres.
        y: Sample y coordinates in target-grid kilometres.
        config: Coastline-swap and ocean-category settings.
        polygons: Preloaded coastline polygons in the target-grid CRS.

    Returns:
        Pair ``(categories, keep)`` containing an ``int16`` category array and
        a Boolean inclusion mask. Marine samples are assigned the configured
        ocean category where required.
    """
    values=np.asarray(categories,dtype=np.int16)
    if not polygons and config.coastline_file is None:
        return values,np.ones(values.shape,dtype=bool)
    point_type=_coastline_types(x,y,polygons)
    present=values>0
    marine=point_type==0
    keep=~present|marine|~np.isin(values,(54,55))
    swapped=np.where(present&marine&(values!=54),config.ocean_category,values)
    return swapped,keep

def _remap_categories(
    categories: ArrayLike, counts: Int64Array, mapping: Mapping[int, int]
) -> tuple[Int16Array, Int64Array]:
    """Apply a category map and merge source classes sharing one target.

    Args:
        categories: One-dimensional source category identifiers.
        counts: Integer counts whose first axis corresponds to ``categories``.
        mapping: Source-to-output positive integer category mapping.

    Returns:
        Pair of unique ``int16`` output categories and merged count array with
        the same spatial shape and dtype as ``counts``.

    Raises:
        ValueError: If a mapping key is absent from ``categories`` or a target
            category is not positive.
    """
    source=np.asarray(categories,dtype=np.int16)
    unknown=sorted(set(mapping)-set(int(value) for value in source))
    if unknown: raise ValueError(f'category_map contains absent source categories: {unknown}')
    targets=np.asarray([mapping.get(int(value),int(value)) for value in source],dtype=np.int16)
    if np.any(targets<=0): raise ValueError('category_map target categories must be positive integers')
    output=np.asarray(list(dict.fromkeys(int(value) for value in targets)),dtype=np.int16)
    lookup={int(value):index for index,value in enumerate(output)}
    merged=np.zeros((output.size,*counts.shape[1:]),dtype=counts.dtype)
    for source_index,target in enumerate(targets): merged[lookup[int(target)]]+=counts[source_index]
    return output,merged


def _merge_global_counts(
    accumulated: Int64Array,
    current: Int64Array,
    categories: ArrayLike,
    excluded_water: int = 52,
) -> None:
    """Merge one global-region count grid with GMERGE water precedence.

    Args:
        accumulated: Mutable accumulated ``(category, y, x)`` count array.
        current: Count array for the next global region.
        categories: Category identifiers for the first axis of both arrays.
        excluded_water: Category marking water outside the source continent.

    Returns:
        None. ``accumulated`` is updated in place.

    Raises:
        ValueError: If ``excluded_water`` is absent or duplicated.
    """
    matches=np.flatnonzero(np.asarray(categories)==excluded_water)
    if matches.size!=1: raise ValueError('GMERGE excluded-water category is absent')
    water=int(matches[0])
    accumulated_total=accumulated.sum(axis=0,dtype=np.int64)
    current_total=current.sum(axis=0,dtype=np.int64)
    accumulated_land=accumulated_total>accumulated[water]
    current_land=current_total>current[water]
    replace=~accumulated_land&current_land
    add=(accumulated_land&current_land)|(~accumulated_land&~current_land)
    accumulated[:,replace]=current[:,replace]
    accumulated[:,add]+=current[:,add]


def _select_glaz_markers(
    center_lon: float,
    center_lat: float,
    left_lon: float,
    right_lon: float,
    lower_lat: float,
    upper_lat: float,
) -> set[str]:
    """Select global GLAZ regions using legacy GSELECT overlap tests.

    Args:
        center_lon: Target-grid centre longitude in degrees east.
        center_lat: Target-grid centre latitude in degrees north.
        left_lon: Western corner bound on the normalized longitude branch.
        right_lon: Eastern corner bound on the normalized longitude branch.
        lower_lat: Southern corner bound in degrees north.
        upper_lat: Northern corner bound in degrees north.

    Returns:
        Set of GLAZ input-type identifiers whose regional extents overlap the
        target markers.
    """
    center_lon=float(center_lon); center_lat=float(center_lat)
    left_lon=float(left_lon); right_lon=float(right_lon)
    lower_lat=float(lower_lat); upper_lat=float(upper_lat)
    keep: set[str] = set()
    for kind,limits in GLAZ_LIMITS.items():
        north,south,east,west=map(float,limits)
        center_hits=int(south<=center_lat<=north)+int(west<=center_lon<=east)
        corner_hits=(int(south<=upper_lat<=north)+int(south<=lower_lat<=north)+
                     int(west<left_lon<=east)+int(west<right_lon<=east))
        for shift in (360.,-360.):
            shifted_left=left_lon+shift
            shifted_right=right_lon+shift
            corner_hits+=int(west<shifted_left<=east)
            corner_hits+=int(west<shifted_right<=east)
        if (center_hits==2 and corner_hits==4) or center_hits==2 or corner_hits==3:
            keep.add(kind)
    if {'GLAZEU','GLAZAS'}<=keep: keep.remove('GLAZAS')
    return keep


def _select_glaz_regions(grid: GridConfig) -> set[str]:
    """Transform a target grid and select overlapping GLAZ source regions.

    Args:
        grid: Target grid, interpreted in its configured projection and datum.

    Returns:
        Set of overlapping GLAZ input-type identifiers.

    Raises:
        ValueError: If the grid datum is invalid.
        NotImplementedError: If the grid projection is unsupported.
    """
    x0=float(grid.x_origin_km); y0=float(grid.y_origin_km)
    x1=x0+grid.nx*float(grid.dx_km)
    y1=y0+grid.ny*float(grid.dy_km)
    xc=.5*(x0+x1); yc=.5*(y0+y1)
    x=np.array([xc,x0,x0,x1,x1],dtype=COORD_DTYPE)
    y=np.array([yc,y0,y1,y1,y0],dtype=COORD_DTYPE)
    transform=Transformer.from_crs(_lcc(grid),geographic_crs(grid.datum),always_xy=True)
    longitude,latitude=transform.transform(x,y)
    longitude=np.asarray(longitude,dtype=COORD_DTYPE)
    latitude=np.asarray(latitude,dtype=COORD_DTYPE)
    center=longitude[0]; corners=longitude[1:].copy()
    if center<corners[0]: corners[0]-=360.
    if center<corners[1]: corners[1]-=360.
    if corners[2]<center: corners[2]+=360.
    if corners[3]<center: corners[3]+=360.
    left=min(corners[0],corners[1]); right=max(corners[2],corners[3])
    if left< -180.: left+=360.
    if left>180.: left-=360.
    if right< -180.: right+=360.
    if right>180.: right-=360.
    lower=min(latitude[1],latitude[4]); upper=max(latitude[2],latitude[3])
    if center< -180.: center+=360.
    if center>180.: center-=360.
    return _select_glaz_markers(center,latitude[0],left,right,lower,upper)

def _lcc(g: GridConfig) -> CRS:
    """Construct the kilometre-based target CRS used by raster readers.

    Args:
        g: CTGPROC target grid.

    Returns:
        pyproj CRS with coordinates expressed in kilometres.

    Raises:
        ValueError: If the grid datum is invalid.
        NotImplementedError: If the grid projection is unsupported.
    """
    return grid_crs(g,units='km')

def _glaz_crs(spec: GlazSpec) -> CRS:
    """Construct the spherical Lambert azimuthal CRS of a GLAZ raster.

    Args:
        spec: GLAZ regional raster geometry.

    Returns:
        Kilometre-based pyproj CRS using the GLAZ 6,370,997 m sphere.
    """
    return CRS.from_proj4(f"+proj=laea +lat_0={spec.latitude_origin:g} +lon_0={spec.longitude_origin:g} +R=6370997 +units=km +no_defs")

def _window(
    g: GridConfig,
    reverse: Callable[[Float64Array, Float64Array], tuple[ArrayLike, ArrayLike]],
    spec: GlazSpec,
) -> tuple[range, range]:
    """Estimate a buffered GLAZ source window covering the target grid.

    Args:
        g: Target grid in kilometres.
        reverse: Callable transforming target x/y arrays into GLAZ kilometres.
        spec: GLAZ raster dimensions and pixel-centre origin.

    Returns:
        Pair of clipped Python ranges for source x and y pixel indices.
    """
    x0=float(g.x_origin_km); y0=float(g.y_origin_km)
    x1=x0+g.nx*float(g.dx_km); y1=y0+g.ny*float(g.dy_km)
    x=np.array([x0,x1,x1,x0],dtype=COORD_DTYPE)
    y=np.array([y0,y0,y1,y1],dtype=COORD_DTYPE)
    px,py=reverse(x,y)
    px=np.asarray(px,dtype=COORD_DTYPE); py=np.asarray(py,dtype=COORD_DTYPE)
    x_delta=px-float(spec.x_origin_km); y_delta=py-float(spec.y_origin_km)
    ix0,ix1=int(np.floor(np.min(x_delta))),int(np.ceil(np.max(x_delta)))
    iy0,iy1=int(np.floor(np.min(y_delta))),int(np.ceil(np.max(y_delta)))
    buf=max((ix1-ix0)//10+1,(iy1-iy0)//10+1)
    return range(max(0,ix0-buf),min(spec.nx,ix1+buf+1)),range(max(0,iy0-buf),min(spec.ny,iy1+buf+1))


def _mesh_offsets(mesh: int, cell_size: float = 1.) -> Float64Array:
    """Construct evenly spaced subpixel offsets about a cell centre.

    Args:
        mesh: Positive number of samples along one cell axis.
        cell_size: Signed source-cell size in caller-defined coordinate units.

    Returns:
        One-dimensional ``float64`` array of length ``mesh`` in source units.

    Raises:
        ZeroDivisionError: If ``mesh`` is zero.
    """
    step=float(cell_size)/int(mesh)
    return (np.arange(int(mesh),dtype=COORD_DTYPE)+.5)*step-float(cell_size)/2.


def _mesh_coordinates(
    centers: ArrayLike, mesh: int, cell_size: float = 1.
) -> Float64Array:
    """Expand source-cell centres into regularly spaced mesh coordinates.

    Args:
        centers: Scalar or array-like source-cell centre coordinates.
        mesh: Number of samples along each cell axis.
        cell_size: Signed source-cell size in the coordinate units.

    Returns:
        ``float64`` array with a trailing axis of length ``mesh``.
    """
    center=np.asarray(centers,dtype=COORD_DTYPE)
    return center[...,None]+_mesh_offsets(mesh,cell_size)


def _source_centers(
    origin: float, spacing: float, indices: ArrayLike
) -> Float64Array:
    """Convert zero-based source indices to pixel-centre coordinates.

    Args:
        origin: Coordinate of source index zero in caller-defined units.
        spacing: Signed source-pixel spacing in the same units.
        indices: Scalar or array-like zero-based indices.

    Returns:
        ``float64`` array matching ``indices`` in source-coordinate units.
    """
    return float(origin)+float(spacing)*np.asarray(indices,dtype=COORD_DTYPE)


def _read_categorical_raster(
    path: PathInput, *, fallback_crs: CRS | str | int | None = None
) -> tuple[CRS, float, float, float, float, RasterArray, int, BoolArray]:
    """Read categorical raster pixels and affine geometry with Rasterio.

    Args:
        path: GeoTIFF or other Rasterio-readable categorical raster.
        fallback_crs: CRS used only when the raster carries no CRS metadata.

    Returns:
        Tuple ``(crs, x0, y0, dx, dy, data, photometric, valid)``. ``x0`` and
        ``y0`` are first-pixel centres; ``dx`` and ``dy`` are signed pixel
        spacings in CRS units; ``data`` retains the raster dtype and band
        layout; ``valid`` is a two-dimensional Boolean mask. ``photometric``
        is 1 for one-band values, 2 for RGB(A), and 3 for a palette band.

    Raises:
        ValueError: If the raster cannot be read, has no usable CRS, has a
            rotated transform, or has invalid pixel spacing.
    """
    path=Path(path)
    try:
        with rasterio.open(path) as source:
            source_crs=(CRS.from_user_input(source.crs) if source.crs is not None
                        else (None if fallback_crs is None
                              else CRS.from_user_input(fallback_crs)))
            if source_crs is None:
                raise ValueError(f'categorical GeoTIFF has no CRS: {path}')
            transform=source.transform
            if not np.isclose(transform.b,0.) or not np.isclose(transform.d,0.):
                raise ValueError('rotated categorical GeoTIFF grids are not supported')
            if transform.a<=0. or transform.e==0.:
                raise ValueError('categorical GeoTIFF has invalid pixel spacing')
            x0,y0=transform@(.5,.5)
            dx,dy=float(transform.a),float(transform.e)
            if source.count==1:
                data=source.read(1)
                valid=source.read_masks(1)!=0
                photometric=(3 if source.colorinterp and
                             source.colorinterp[0]==ColorInterp.palette else 1)
            elif source.count in {3,4} and tuple(source.colorinterp[:3])==(
                    ColorInterp.red,ColorInterp.green,ColorInterp.blue):
                data=source.read()
                valid=np.all(source.read_masks()!=0,axis=0)
                photometric=2
            else:
                data=source.read()
                valid=np.all(source.read_masks()!=0,axis=0)
                photometric=0
    except RasterioIOError as error:
        raise ValueError(f'unable to read categorical GeoTIFF: {path}') from error
    return source_crs,float(x0),float(y0),dx,dy,data,photometric,valid


def _accumulate_glazas(
    path: Path,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Window, mesh-sample, and accumulate one raw GLAZ regional raster.

    Args:
        path: Headerless one-byte GLAZ raster path.
        config: Active source type, datum, mesh density, and target-grid settings.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, cells_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If the raster byte size or mesh density is invalid.
        NotImplementedError: If the configured GLAZ region is unsupported.
    """
    kind=config.input_type.upper(); spec=GLAZ_SPECS.get(kind)
    if spec is None: raise NotImplementedError(f"unsupported GLAZ input type: {kind}")
    if path.stat().st_size != spec.nx*spec.ny:
        raise ValueError(f"{kind} raster must contain {spec.nx}x{spec.ny} bytes: {path}")
    mesh=config.mesh_density
    if mesh < 1: raise ValueError("mesh_density must be positive")
    g=config.grid; source=np.memmap(path,dtype="u1",mode="r",shape=(spec.ny,spec.nx))
    source_crs=_glaz_crs(spec)
    source_grid=GridConfig(1,1,1.,1.,projection='LAZA',
                           lat0=spec.latitude_origin,lon0=spec.longitude_origin,
                           datum=config.input_datum)
    def reverse(
        x: ArrayLike, y: ArrayLike
    ) -> tuple[Float64Array, Float64Array]:
        """Transform target-grid coordinates back to the GLAZ source CRS.

        Args:
            x: Array-like target-grid x coordinates in kilometres.
            y: Array-like target-grid y coordinates in kilometres.

        Returns:
            Pair of ``float64`` GLAZ x/y arrays in kilometres.
        """
        return cast(
            tuple[Float64Array, Float64Array],
            unproject_to_crs(x,y,g,source_crs,config.input_datum,units='km'),
        )
    irange,jrange=_window(g,reverse,spec)
    counts=np.zeros((USGS38.size,g.ny*g.nx),dtype=np.int64)
    columns={int(v):i for i,v in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    read=inside=0
    source_index=np.arange(irange.start,irange.stop,dtype=np.float64)
    xs=np.asarray(np.float64(spec.x_origin_km)+source_index,dtype=np.float64)
    for j in jrange:
        raw=np.asarray(source[spec.ny-1-j,irange.start:irange.stop])
        valid=(raw>=1)&(raw<GLAZ_TO_USGS.size)
        if not valid.any(): continue
        base_x=xs[valid]; classes=GLAZ_TO_USGS[raw[valid]]
        source_y=np.asarray(np.float64(spec.y_origin_km)+np.float64(j),dtype=np.float64)
        mesh_x=_mesh_coordinates(base_x,mesh)
        mesh_y=_mesh_coordinates(source_y,mesh)
        sx=np.broadcast_to(mesh_x[:,:,None],(base_x.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(base_x.size,mesh,mesh)).ravel()
        gx,gy=project_glaz_to_lcc(sx,sy,source_grid,g,units='km')
        center_x,center_y=project_glaz_to_lcc(
            base_x,np.full(base_x.size,source_y,dtype=np.float64),
            source_grid,g,units='km')
        _qa_write(config,'qaglaz.dat',center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,g)
        repeated=np.repeat(classes,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=int(valid.sum()); inside+=int(good.sum())
    return counts.reshape(USGS38.size,g.ny,g.nx),read,inside


def _ctg_record(line: str, path: Path, number: int) -> tuple[int, int, int, int]:
    """Parse one free-form or fixed-column compressed CTG record.

    Args:
        line: ASCII source record.
        path: Source path used in diagnostics.
        number: One-based source line number used in diagnostics.

    Returns:
        Four integers ``(run_length, category, x_value, y_value)``.

    Raises:
        ValueError: If the record does not contain four valid integers.
    """
    fields=line.split()
    try:
        if len(fields)==4:
            return cast(tuple[int, int, int, int], tuple(int(value) for value in fields))
        padded=line.ljust(24)
        return (int(padded[0:4]),int(padded[4:7]),
                int(padded[7:15]),int(padded[15:24]))
    except ValueError as error:
        raise ValueError(f'invalid CTG record at {path}:{number}') from error


def _accumulate_ctg(
    path: Path,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Expand USGS 200 m CTG run records and bin their mesh samples.

    Args:
        path: Compressed ASCII USGS CTG source path.
        config: Active source datum, mesh density, coastline, and target grid.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, cells_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If the file header, UTM zone, run record, category,
            coordinate anchors, or mesh density is invalid.
    """
    lines=path.read_text(encoding='ascii',errors='strict').splitlines()
    if len(lines)<6: raise ValueError(f'compressed CTG file is truncated: {path}')
    try: zone=int(lines[0].ljust(55)[50:55])
    except ValueError as error: raise ValueError(f'invalid CTG UTM zone: {path}') from error
    if not 1<=zone<=60: raise ValueError(f'CTG UTM zone must be 1..60: {path}')
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    source_grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=zone,
                           hemisphere='N',datum=config.input_datum)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    x_anchor: int | None
    y_anchor: int | None
    x_anchor=y_anchor=None
    read=inside=0
    for number,line in enumerate(lines[5:],start=6):
        if not line.strip(): continue
        run,category,x_value,y_value=_ctg_record(line,path,number)
        if run<0: raise ValueError(f'negative CTG run length at {path}:{number}')
        if run==0:
            x_anchor=x_value-200; y_anchor=y_value
            continue
        if x_value!=0: continue
        if x_anchor is None: raise ValueError(f'CTG data precede coordinate anchor at {path}:{number}')
        x_m=x_anchor+200*np.arange(1,run+1,dtype=np.int64); x_anchor=int(x_m[-1])
        read+=run
        if not 1<=category<=92: continue
        if category not in columns:
            raise ValueError(f'CTG category {category} is not in the USGS-38 table at {path}:{number}')
        centers_x=np.asarray(np.asarray(x_m,dtype=np.float64)*np.float64(.001),dtype=np.float64)
        center_y=np.asarray(np.float64(y_anchor)*np.float64(.001),dtype=np.float64)
        mesh_x=_mesh_coordinates(centers_x,mesh,np.float64(.2))
        mesh_y=_mesh_coordinates(center_y,mesh,np.float64(.2))
        # IMESH is the outer loop and JMESH the inner loop in LOADOTHER.
        sx=np.broadcast_to(mesh_x[:,:,None],(run,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(run,mesh,mesh)).ravel()
        gx,gy=cast(
            tuple[Float64Array, Float64Array],
            transform_grid(sx,sy,source_grid,config.grid,units='km'),
        )
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        categories=np.full(gx.shape,category,dtype=np.int16)
        categories,coastal=_swap_coastline_categories(categories,gx,gy,config,polygons)
        good=coastal&in_grid
        _qa_write(config,'qamesh.dat',gx[good],gy[good])
        used=good.reshape(run,mesh*mesh).any(axis=1)
        if used.any():
            center_x,center_y=cast(
                tuple[Float64Array, Float64Array],
                transform_grid(centers_x[used],
                    np.full(np.count_nonzero(used),center_y,dtype=np.float64),source_grid,
                    config.grid,units='km'),
            )
            _qa_write(config,'qactg.dat',center_x,center_y,
                      np.full(np.count_nonzero(used),category))
        _bin_samples(counts,columns,gx,gy,categories,good,config,snow)
        inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside

def _read_generic(
    path: Path,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> tuple[Int16Array, Int64Array, int, int]:
    """Read ``GENERIC.LANDUSE`` point records and bin them by target cell.

    Args:
        path: ASCII generic land-use dataset path.
        config: Active target-grid, coastline, and QA settings.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(categories, counts, records_read, samples_inside)``.
        ``categories`` is ``int16``; ``counts`` is ``int64`` with shape
        ``(category, grid.ny, grid.nx)``.

    Raises:
        ValueError: If the header, category table, record, coordinate units,
            latitude, coastline category, or datum is invalid.
    """
    lines=path.read_text(encoding='ascii',errors='strict').splitlines()
    if not lines or lines[0][:16].strip().upper()!='GENERIC.LANDUSE':
        raise ValueError(f'generic land-use dataset header is invalid: {path}')
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error: raise ValueError(f'invalid generic comment count: {path}') from error
    index=2+comments
    try:
        projection=lines[index][:8].strip().upper(); index+=1
        datum=lines[index][:8].strip(); index+=1
        units=lines[index][:4].strip().upper(); index+=1
        count=int(lines[index].split()[0]); index+=1
    except (IndexError,ValueError) as error:
        raise ValueError(f'incomplete generic land-use header: {path}') from error
    if projection!='LL' or units!='DEG':
        raise ValueError('generic land-use coordinates must use LL projection and DEG units')
    categories: list[int] = []
    while len(categories)<count and index<len(lines):
        categories.extend(int(item) for item in lines[index].split()); index+=1
    if len(categories)!=count or len(set(categories))!=count:
        raise ValueError('generic land-use category list is incomplete or contains duplicates')
    records: list[tuple[int, float, float]] = []
    for number,line in enumerate(lines[index:],start=index+1):
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=3: raise ValueError(f'invalid generic record at {path}:{number}')
        category=int(fields[0]); longitude=float(fields[1]); latitude=float(fields[2])
        if category not in categories: raise ValueError(f'undeclared land-use category {category} at {path}:{number}')
        if not -90.<=latitude<=90.: raise ValueError(f'invalid latitude at {path}:{number}')
        records.append((category,longitude,latitude))
    counts=np.zeros((count,config.grid.ny,config.grid.nx),dtype=np.int64)
    if not records: return np.asarray(categories,dtype=np.int16),counts,0,0
    values=np.asarray(records,dtype=float)
    # pyproj currently routes one-element ndarrays through its scalar path;
    # explicit sequences keep the vector API stable across NumPy versions.
    x,y=project_geographic(values[:,1].tolist(),values[:,2].tolist(),
                           config.grid,datum,units='km')
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    _qa_write(config,'qageneric.dat',x,y,values[:,0])
    ix,iy,in_grid=_inside_grid(x,y,config.grid)
    polygons=_coastline_polygons(config)
    sample_categories,coastal=_swap_coastline_categories(
        values[:,0],x,y,config,polygons)
    if config.coastline_swap and config.ocean_category not in categories:
        raise ValueError(
            'ocean_category/IOCEAN must be declared by GENERIC data when coastline_swap is enabled')
    good=coastal&in_grid
    lookup={category:i for i,category in enumerate(categories)}
    _bin_samples(counts.reshape(count,-1),lookup,x,y,sample_categories,good,
                 config,snow)
    return np.asarray(categories,dtype=np.int16),counts,len(records),int(good.sum())


def _read_nzgen(
    path: Path,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Read New Zealand points and apply the legacy NZCAT translation.

    Args:
        path: Free-format ASCII NZGEN point dataset.
        config: Active source datum, target-grid, coastline, and QA settings.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, records_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If a record, NZ category, latitude, or datum is invalid.
    """
    records: list[tuple[int, float, float]] = []
    for number,line in enumerate(path.read_text(encoding='ascii',errors='strict').splitlines(),start=1):
        if not line.strip(): continue
        fields=line.replace(',',' ').split()
        if len(fields)!=4: raise ValueError(f'invalid NZGEN record at {path}:{number}')
        try: _,category,latitude,longitude=(float(value) for value in fields)
        except ValueError as error: raise ValueError(f'invalid NZGEN record at {path}:{number}') from error
        category=int(category)
        if category==0: category=1
        if not 1<=category<NZ_TO_USGS.size or NZ_TO_USGS[category]==0:
            raise ValueError(f'unmapped NZGEN category {category} at {path}:{number}')
        if not -90.<=latitude<=90.: raise ValueError(f'invalid NZGEN latitude at {path}:{number}')
        records.append((int(NZ_TO_USGS[category]),longitude,latitude))
    counts=np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64)
    if not records: return counts,0,0
    values=np.asarray(records,dtype=float)
    x,y=project_geographic(values[:,1].tolist(),values[:,2].tolist(),
                           config.grid,config.input_datum,units='km')
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    _qa_write(config,'qanzgen.dat',x,y)
    ix,iy,in_grid=_inside_grid(x,y,config.grid)
    polygons=_coastline_polygons(config)
    categories,coastal=_swap_coastline_categories(values[:,0],x,y,config,polygons)
    good=coastal&in_grid
    lookup={int(value):index for index,value in enumerate(USGS38)}
    _bin_samples(counts.reshape(USGS38.size,-1),lookup,x,y,categories,good,config,snow)
    return counts,len(records),int(good.sum())


def _nlcd92_state(path: Path) -> str:
    """Infer the legacy NLCD92 state-grid key from a source filename.

    Args:
        path: Source path whose filename precedes ``_nlcd`` or ``-nlcd`` with
            a key in :data:`NLCD92_SPECS`.

    Returns:
        Lowercase state or regional lookup key.

    Raises:
        ValueError: If the filename does not identify a known NLCD92 grid.
    """
    match=re.search(r'^(.+?)[_-]nlcd',path.name,re.IGNORECASE)
    if match is None or match.group(1).lower() not in NLCD92_SPECS:
        raise ValueError(f'NLCD92 filename does not identify a legacy state: {path.name}')
    return match.group(1).lower()


def _accumulate_nlcd92(
    path: Path,
    config: CtgprocConfig,
    spec: Nlcd92Spec | None = None,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Read and accumulate a headerless state NLCD92 byte grid.

    Args:
        path: Headerless one-byte NLCD92 raster path.
        config: Active source datum, target-grid, coastline, and QA settings.
        spec: Optional geometry override; otherwise inferred from ``path``.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, pixels_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If the filename, raster size, source category, or datum is
            invalid.
    """
    spec=NLCD92_SPECS[_nlcd92_state(path)] if spec is None else spec
    if path.stat().st_size!=spec.nx*spec.ny:
        raise ValueError(f'NLCD92 raster must contain {spec.nx}x{spec.ny} bytes: {path}')
    source_grid=GridConfig(1,1,1.,1.,projection='ACEA',lat0=23.,lon0=-96.,
                           lat1=29.5,lat2=45.5,datum=config.input_datum)
    corners_x=np.array([config.grid.x_origin_km,
        config.grid.x_origin_km+config.grid.nx*config.grid.dx_km]*2)
    corners_y=np.repeat([config.grid.y_origin_km,
        config.grid.y_origin_km+config.grid.ny*config.grid.dy_km],2)
    sx,sy=transform_grid(corners_x,corners_y,config.grid,source_grid,units='km')
    ix0=int(np.floor((np.min(sx)-spec.x_origin_km)/.03)); ix1=int(np.ceil((np.max(sx)-spec.x_origin_km)/.03))
    iy0=int(np.floor((spec.y_origin_km-np.max(sy))/.03)); iy1=int(np.ceil((spec.y_origin_km-np.min(sy))/.03))
    buffer=max(abs(ix1-ix0)//10+1,abs(iy1-iy0)//10+1)
    ix0=max(0,ix0-buffer); ix1=min(spec.nx-1,ix1+buffer)
    iy0=max(0,iy0-buffer); iy1=min(spec.ny-1,iy1+buffer)
    if ix0>ix1 or iy0>iy1:
        return np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64),0,0
    source=np.memmap(path,dtype='u1',mode='r',shape=(spec.ny,spec.nx))
    columns={int(value):index for index,value in enumerate(USGS38)}
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    polygons=_coastline_polygons(config)
    x=spec.x_origin_km+.03*np.arange(ix0,ix1+1); read=inside=0
    for j in range(iy0,iy1+1):
        raw=np.asarray(source[j,ix0:ix1+1]); translated=NLCD92_TO_USGS[raw]
        gx,gy=transform_grid(x,np.full(x.size,spec.y_origin_km-.03*j),
                             source_grid,config.grid,units='km')
        _qa_write(config,'qanlcd.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        translated,coastal=_swap_coastline_categories(translated,gx,gy,config,polygons)
        good=(translated>0)&coastal&in_grid
        invalid=(raw>0)&(translated==0)
        if invalid.any(): raise ValueError(f'unmapped NLCD92 category {int(raw[invalid][0])} in {path}')
        _bin_samples(counts,columns,gx,gy,translated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_nlcd_geotiff(
    path: Path,
    config: CtgprocConfig,
    translation: Int16Array,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Read and accumulate an NLCD GeoTIFF using its embedded geometry.

    Args:
        path: NLCD categorical GeoTIFF path.
        config: Active target-grid, coastline, and QA settings.
        translation: Integer lookup from source NLCD values to USGS classes.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, pixels_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If raster geometry, bands, dtype, source classes, mapped
            classes, or target projection is invalid.
    """
    source_crs,x0,y0,dx,dy,data,_,valid_data=_read_categorical_raster(path)
    if data.ndim!=2: raise ValueError('NLCD GeoTIFF must contain one two-dimensional band')
    if data.dtype.kind not in {'u','i'}: raise ValueError('NLCD GeoTIFF categories must be integers')
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    x=_source_centers(x0,dx,np.arange(data.shape[1])); read=inside=0
    for row,raw in enumerate(data):
        valid=valid_data[row]
        if np.any(valid&((raw<0)|(raw>=len(translation)))):
            raise ValueError(f'NLCD GeoTIFF category is outside the byte lookup table: {path}')
        translated=np.zeros(raw.shape,dtype=np.int16)
        translated[valid]=translation[raw[valid].astype(np.int64,copy=False)]
        invalid=valid&(raw!=0)&(translated==0)
        if invalid.any(): raise ValueError(f'unmapped NLCD category {int(raw[invalid][0])} in {path}')
        nonstandard=valid&~np.isin(translated,[0,*columns])
        if nonstandard.any():
            raise ValueError(f'NLCD category maps outside the USGS-38 table: {int(translated[nonstandard][0])}')
        y=_source_centers(y0,dy,row)
        gx,gy=transformer.transform(x,np.full(x.size,y,dtype=COORD_DTYPE))
        _qa_write(config,'qanlcd.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        translated,coastal=_swap_coastline_categories(translated,gx,gy,config,polygons)
        good=valid&(translated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,translated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _corine_indices(
    data: ArrayLike,
    photometric: int,
    path: PathInput,
    valid: ArrayLike | None = None,
) -> Int16Array:
    """Decode index, class-code, or RGB CORINE pixels to class positions.

    Args:
        data: Raster data as one category band or RGB(A) bands.
        photometric: Encoding flag: 2 for RGB, 3 for palette indices, or any
            other value for three-digit CORINE class codes.
        path: Source path used in validation messages.
        valid: Optional Boolean raster validity mask.

    Returns:
        Two-dimensional ``int16`` array of one-based positions in the CORINE
        translation table; invalid raster pixels are zero.

    Raises:
        ValueError: If shape, dtype, RGB range, validity-mask shape, or a
            category value is invalid.
    """
    raw=np.asarray(data)
    if photometric==2:
        if raw.ndim==3 and raw.shape[-1] in {3,4}: rgb=raw[...,:3]
        elif raw.ndim==3 and raw.shape[0] in {3,4}: rgb=np.moveaxis(raw[:3],0,-1)
        else: raise ValueError(f'CORINE RGB GeoTIFF must have three color bands: {path}')
        if rgb.dtype.kind not in {'u','i'} or np.any((rgb<0)|(rgb>255)):
            raise ValueError(f'CORINE RGB values must be bytes: {path}')
        values=(rgb[...,0].astype(np.int32)*1_000_000+
                rgb[...,1].astype(np.int32)*1_000+rgb[...,2].astype(np.int32))
        indices=np.zeros(values.shape,dtype=np.int16)
        for index,value in enumerate(CORINE_RGB,1): indices[values==value]=index
        invalid=indices==0
    else:
        if raw.ndim!=2 or raw.dtype.kind not in {'u','i'}:
            raise ValueError(f'CORINE GeoTIFF must contain one integer category band: {path}')
        values=raw.astype(np.int64,copy=False)
        if photometric==3:
            indices=values.astype(np.int16,copy=False)
            invalid=(values<0)|(values>=len(CORINE_TO_USGS))
        else:
            order=np.argsort(CORINE_CLASSES); sorted_classes=CORINE_CLASSES[order]
            positions=np.searchsorted(sorted_classes,values)
            found=(positions<sorted_classes.size)&(
                sorted_classes[np.minimum(positions,sorted_classes.size-1)]==values)
            indices=np.zeros(values.shape,dtype=np.int16)
            indices[found]=order[positions[found]]+1
            invalid=(values!=0)&~found
    valid_mask=(np.ones(indices.shape,dtype=bool) if valid is None
                else np.asarray(valid,dtype=bool))
    if valid_mask.shape!=indices.shape:
        raise ValueError(f'CORINE validity mask has the wrong shape: {path}')
    invalid &= valid_mask
    if np.any(invalid):
        value=(values[invalid][0] if values.ndim==2 else values.reshape(-1)[0])
        raise ValueError(f'unmapped CORINE category {value!r} in {path}')
    indices=np.asarray(indices,dtype=np.int16).copy()
    indices[~valid_mask]=0
    return indices


def _accumulate_corine(
    path: Path,
    config: CtgprocConfig,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Mesh-sample and accumulate any supported CORINE GeoTIFF encoding.

    Args:
        path: CORINE categorical GeoTIFF path.
        config: Active mesh, target-grid, coastline, and QA settings.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, pixels_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If raster geometry, encoding, categories, mesh density, or
            projection metadata is invalid.
    """
    source_crs,x0,y0,dx,dy,data,photometric,valid_data=_read_categorical_raster(path)
    indices=_corine_indices(data,photometric,path,valid_data)
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config); read=inside=0
    centers=_source_centers(x0,dx,np.arange(indices.shape[1]))
    for row,source_indices in enumerate(indices):
        source_valid=valid_data[row]
        translated=CORINE_TO_USGS[source_indices]
        center_y=_source_centers(y0,dy,row)
        mesh_x=_mesh_coordinates(centers,mesh,dx)
        mesh_y=_mesh_coordinates(center_y,mesh,dy)
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=transformer.transform(sx,sy)
        qa_x,qa_y=transformer.transform(
            centers,np.full(centers.size,center_y,dtype=COORD_DTYPE))
        _qa_write(config,'qacorine.dat',qa_x,qa_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated_valid=np.repeat(source_valid,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=repeated_valid&(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=source_indices.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_global_geotiff(
    path: Path,
    config: CtgprocConfig,
    translation: Int16Array,
    *,
    offset: int = 0,
    missing: Sequence[int] = (),
    fallback_epsg: int | None = None,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Mesh-sample and accumulate a global categorical GeoTIFF.

    Args:
        path: Global categorical GeoTIFF path.
        config: Active source type, mesh, target-grid, coastline, and QA settings.
        translation: Integer lookup from adjusted source values to USGS classes.
        offset: Integer added to each raw source value before lookup.
        missing: Raw source values treated as absent.
        fallback_epsg: EPSG code used only when the raster has no embedded CRS.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, pixels_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If raster geometry, dtype, categories, mesh density, or
            projection metadata is invalid.
    """
    source_crs,x0,y0,dx,dy,data,_,valid_data=_read_categorical_raster(
        path,fallback_crs=(None if fallback_epsg is None else CRS.from_epsg(fallback_epsg)))
    if data.ndim!=2 or data.dtype.kind not in {'u','i'}:
        raise ValueError('global land-use GeoTIFF must be one integer band')
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    read=inside=0
    for row,raw in enumerate(data):
        raster_valid=valid_data[row]
        raw=np.asarray(raw,dtype=np.int64); source_category=raw+offset
        absent=~raster_valid|np.isin(raw,tuple(missing))
        invalid=(source_category<0)|(source_category>=len(translation))
        if np.any(invalid&~absent):
            raise ValueError(f'global land-use category is outside its lookup table: {path}')
        translated=np.zeros(raw.shape,dtype=np.int16)
        valid=~invalid&~absent; translated[valid]=translation[source_category[valid]]
        if np.any((translated==0)&valid):
            raise ValueError(f'unmapped global land-use category in {path}')
        centers=_source_centers(x0,dx,np.arange(data.shape[1]))
        center_y=_source_centers(y0,dy,row)
        mesh_x=_mesh_coordinates(centers,mesh,dx)
        mesh_y=_mesh_coordinates(center_y,mesh,dy)
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=transformer.transform(sx,sy)
        center_x,center_y=transformer.transform(
            centers,np.full(centers.size,center_y,dtype=COORD_DTYPE))
        qa_name='qagcl2k.dat' if config.input_type.upper()=='GLC2K' else 'qaumdglc.dat'
        _qa_write(config,qa_name,center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated_valid=np.repeat(raster_valid,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=repeated_valid&(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_modis(
    path: Path,
    config: CtgprocConfig,
    spec: ModisSpec | None = None,
    snow: SnowGridInfo | None = None,
) -> AccumulationResult:
    """Window, mesh-sample, and accumulate a raw BU MODIS byte grid.

    Args:
        path: Headerless one-byte global MODIS land-cover raster.
        config: Active source datum, mesh, target-grid, coastline, and QA settings.
        spec: Optional raster geometry override; defaults to ``MODIS_SPEC``.
        snow: Optional sparse SNODAS mapping updated for accepted samples.

    Returns:
        Tuple ``(counts, pixels_read, samples_inside)``. ``counts`` is an
        ``int64`` array shaped ``(38, grid.ny, grid.nx)`` in USGS-38 order.

    Raises:
        ValueError: If raster size, mesh density, datum, or grid projection is
            invalid.
    """
    spec=MODIS_SPEC if spec is None else spec
    if path.stat().st_size!=spec.nx*spec.ny:
        raise ValueError(f'MODIS raster must contain {spec.nx}x{spec.ny} bytes: {path}')
    corners_x=np.array([config.grid.x_origin_km,
        config.grid.x_origin_km+config.grid.nx*config.grid.dx_km]*2)
    corners_y=np.repeat([config.grid.y_origin_km,
        config.grid.y_origin_km+config.grid.ny*config.grid.dy_km],2)
    lon,lat=unproject_geographic(corners_x,corners_y,config.grid,
                                 config.input_datum,units='km')
    ix0=int(np.floor((np.min(lon)-spec.x_origin)/spec.spacing))
    ix1=int(np.ceil((np.max(lon)-spec.x_origin)/spec.spacing))
    iy0=int(np.floor((spec.y_origin-np.max(lat))/spec.spacing))
    iy1=int(np.ceil((spec.y_origin-np.min(lat))/spec.spacing))
    buffer=max(abs(ix1-ix0)//10+1,abs(iy1-iy0)//10+1)
    ix0=max(0,ix0-buffer); ix1=min(spec.nx-1,ix1+buffer)
    iy0=max(0,iy0-buffer); iy1=min(spec.ny-1,iy1+buffer)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    if ix0>ix1 or iy0>iy1: return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),0,0
    source=np.memmap(path,dtype='u1',mode='r',shape=(spec.ny,spec.nx))
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    centers=_source_centers(spec.x_origin,spec.spacing,
                            np.arange(ix0,ix1+1)); read=inside=0
    for row in range(iy0,iy1+1):
        raw=np.asarray(source[row,ix0:ix1+1]); valid=(raw>=1)&(raw<=18)
        translated=np.zeros(raw.shape,dtype=np.int16)
        translated[valid]=MODIS_TO_USGS[raw[valid]]
        mesh_x=_mesh_coordinates(centers,mesh,np.float64(spec.spacing))
        center_y=_source_centers(spec.y_origin,-spec.spacing,row)
        mesh_y=_mesh_coordinates(center_y,mesh,np.float64(-spec.spacing))
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=project_geographic(sx,sy,config.grid,config.input_datum,units='km')
        center_x,center_y=cast(
            tuple[Float64Array, Float64Array],
            project_geographic(
                centers,np.full(centers.size,center_y,dtype=np.float64),
                config.grid,config.input_datum,units='km'),
        )
        _qa_write(config,'qamodis.dat',center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside

def _read_previous_landuse(
    path: PathInput, grid: GridConfig
) -> tuple[Int16Array, Int64Array, SnowGridInfo | None]:
    """Read CTGPROC continuation hits and optional sparse snow records.

    Args:
        path: Modern NetCDF or legacy ASCII HITS land-use product.
        grid: Target grid that the continuation product must match.

    Returns:
        Tuple ``(categories, counts, snow)``. Categories are ``int16``;
        counts are non-negative ``int64`` with shape
        ``(category, grid.ny, grid.nx)``; ``snow`` is an optional
        :class:`SnowGridInfo`.

    Raises:
        ValueError: If file type, dimensions, geometry, projection, datum,
            counts, or snow metadata does not match the target grid.
        OSError: If the continuation file cannot be read.
    """
    source=Path(path)
    snow: SnowGridInfo | None
    if source.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(source) as dataset:
            file_type=str(getattr(dataset,'landuse_file_type','')).upper()
            if file_type!='HITS': raise ValueError('previous CTGPROC NetCDF must have HITS file type')
            categories=np.asarray(dataset['landuse_category'][:],dtype=np.int16)
            counts=np.asarray(dataset['sample_count'][:],dtype=np.int64)
            shape=(grid.ny,grid.nx)
            if counts.ndim!=3 or counts.shape!=(categories.size,*shape):
                raise ValueError('previous land-use grid dimensions do not match')
            for name,wanted in {'grid_nx':grid.nx,'grid_ny':grid.ny}.items():
                if name in dataset.ncattrs() and int(getattr(dataset,name))!=wanted:
                    raise ValueError('previous land-use grid dimensions do not match')
            geometry={'grid_dx_km':grid.dx_km,'grid_dy_km':grid.dy_km,
                      'x_origin_km':grid.x_origin_km,'y_origin_km':grid.y_origin_km}
            if any(name in dataset.ncattrs() and
                   not np.isclose(float(getattr(dataset,name)),wanted,rtol=1.e-4,atol=0.)
                   for name,wanted in geometry.items()):
                raise ValueError('previous land-use grid geometry does not match')
            if ('projection' in dataset.ncattrs() and
                    str(dataset.projection).strip().upper()!=grid.projection.upper()):
                raise ValueError('previous land-use projection does not match')
            if ('datum' in dataset.ncattrs() and
                    normalize_datum(str(dataset.datum))!=normalize_datum(grid.datum)):
                raise ValueError('previous land-use datum does not match')
            if np.any(counts<0): raise ValueError('previous land-use counts must be non-negative')
            snow_names={'snow_cell_index','snow_landuse_index','snow_grid_index',
                        'snow_fraction'}
            present=snow_names&set(dataset.variables)
            if present and present!=snow_names:
                raise ValueError('previous land-use snow metadata is incomplete')
            snow=SnowGridInfo.create(grid) if present else None
            if snow is not None:
                cell_values=np.asarray(dataset['snow_cell_index'][:],dtype=np.int64)-1
                land_values=np.asarray(dataset['snow_landuse_index'][:],dtype=np.int64)-1
                snow_grid_values=np.asarray(dataset['snow_grid_index'][:],dtype=np.int64)
                fraction_values=np.asarray(dataset['snow_fraction'][:],dtype=np.float64)
                if not (cell_values.shape==land_values.shape==snow_grid_values.shape==fraction_values.shape):
                    raise ValueError('previous land-use snow arrays do not match')
                if (np.any((cell_values<0)|(cell_values>=grid.nx*grid.ny)) or
                        np.any((land_values<0)|(land_values>=categories.size)) or
                        np.any(~np.isfinite(fraction_values)|(fraction_values<0.))):
                    raise ValueError('previous land-use snow metadata is invalid')
                if 'snow_sample_count' in dataset.variables:
                    snow_count=np.asarray(dataset['snow_sample_count'][:],dtype=np.int64)
                    if snow_count.shape!=cell_values.shape or np.any(snow_count<0):
                        raise ValueError('previous land-use snow counts are invalid')
                else:
                    totals=counts.sum(axis=0,dtype=np.int64).ravel()
                    snow_count=np.rint(fraction_values*totals[cell_values]).astype(np.int64)
                for cell_index,land_index,snow_index,hit_count in zip(
                        cell_values,land_values,snow_grid_values,snow_count,strict=True):
                    if hit_count:
                        snow.add_hits(cell_index,categories[land_index],snow_index,
                                      hit_count)
        return categories,counts,snow
    lines=source.read_text(encoding='ascii',errors='strict').splitlines()
    if len(lines)<6 or not lines[0][:16].strip().upper().startswith('LU.DAT'):
        raise ValueError(f'previous CTGPROC file has an invalid header: {source}')
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error:
        raise ValueError(f'previous CTGPROC file has an invalid comment count: {source}') from error
    index=2+comments
    if index>=len(lines) or lines[index].strip().upper()!='HITS':
        raise ValueError('previous CTGPROC LU.DAT must have HITS file type')
    index+=1
    projection=lines[index][:8].strip().upper(); index+=1
    if projection!=grid.projection.upper(): raise ValueError('previous land-use projection does not match')
    index+=1  # projection parameters
    if projection in {'LCC','LAZA','TTM'}: index+=1
    datum=lines[index][:8].strip().upper(); index+=1
    if normalize_datum(datum)!=normalize_datum(grid.datum):
        raise ValueError('previous land-use datum does not match')
    fields=lines[index].split(); index+=1
    if len(fields)<7: raise ValueError('previous land-use grid header is incomplete')
    nx,ny=int(fields[0]),int(fields[1]); category_count=int(fields[6])
    if (nx,ny)!=(grid.nx,grid.ny): raise ValueError('previous land-use grid dimensions do not match')
    expected=(grid.x_origin_km,grid.y_origin_km,grid.dx_km,grid.dy_km)
    if not np.allclose([float(value) for value in fields[2:6]],expected,rtol=1.e-4,atol=0.):
        raise ValueError('previous land-use grid geometry does not match')
    index+=1  # KM units
    categories=np.fromstring(lines[index],sep=' ',dtype=np.int16); index+=1
    if categories.size!=category_count: raise ValueError('previous land-use category list is incomplete')
    counts=np.zeros((category_count,grid.ny,grid.nx),dtype=np.int64)
    seen=np.zeros((grid.ny,grid.nx),dtype=bool)
    record_count=grid.nx*grid.ny
    records=lines[index:index+record_count]
    if len(records)!=record_count:
        raise ValueError('previous land-use file does not contain every grid cell')
    for number,line in enumerate(records,start=index+1):
        values=np.fromstring(line[:10+5*category_count],sep=' ',dtype=np.int64)
        if values.size<2+category_count:
            raise ValueError(f'invalid previous land-use record at {source}:{number}')
        i,j=int(values[0])-1,int(values[1])-1
        if not (0<=i<grid.nx and 0<=j<grid.ny) or seen[j,i]:
            raise ValueError(f'invalid previous land-use grid index at {source}:{number}')
        counts[:,j,i]=values[2:2+category_count]; seen[j,i]=True
    if not seen.all(): raise ValueError('previous land-use file does not contain every grid cell')
    if np.any(counts<0): raise ValueError('previous land-use counts must be non-negative')
    index+=record_count
    snow = None
    if index<len(lines) and lines[index].strip()=='SNOW Grid Information:':
        snow=SnowGridInfo.create(grid); index+=1
        snow_records=lines[index:index+record_count]
        if len(snow_records)!=record_count:
            raise ValueError('previous land-use snow section is incomplete')
        for number,line in enumerate(snow_records,start=index+1):
            fields=line.split()
            try:
                i=int(fields[0])-1; j=int(fields[1])-1; land_count=int(fields[2])
                if not (0<=i<grid.nx and 0<=j<grid.ny) or land_count<0:
                    raise ValueError
                cursor=3; cell=j*grid.nx+i
                total=int(counts[:,j,i].sum(dtype=np.int64))
                for _ in range(land_count):
                    land=int(fields[cursor])-1; item_count=int(fields[cursor+1]); cursor+=2
                    if not 0<=land<categories.size or item_count<0:
                        raise ValueError
                    for _ in range(item_count):
                        snow_grid=int(fields[cursor]); fraction=float(fields[cursor+1]); cursor+=2
                        hit_count=int(np.rint(fraction*total))
                        if hit_count:
                            snow.add_hits(cell,categories[land],snow_grid,hit_count)
                if cursor!=len(fields):
                    raise ValueError
            except (IndexError,ValueError) as error:
                raise ValueError(
                    f'invalid previous land-use snow record at {source}:{number}') from error
    return categories,counts,snow


def _snow_rows(
    snow: SnowGridInfo | None, categories: Int16Array
) -> tuple[Sequence[SnowRow], Int64Array]:
    """Flatten sparse snow hits into serializable fractional records.

    Args:
        snow: Optional sparse snow mapping.
        categories: Ordered output land-use category identifiers.

    Returns:
        Pair ``(rows, totals)``. Each row contains zero-based target cell,
        one-based category position, SNODAS cell, ``float32`` sample fraction,
        and integer hit count. ``totals`` is an ``int64`` flattened-cell array;
        it is empty when ``snow`` is ``None``.

    Raises:
        ValueError: If snow records contain a category absent from
            ``categories``.
    """
    if snow is None:
        return (),np.empty(0,dtype=np.int64)
    grouped,totals=snow.records(categories)
    rows: list[SnowRow] = []
    for (cell,land),items in sorted(grouped.items()):
        for snow_grid,count in items:
            fraction=COMPUTE_DTYPE(np.float64(count)/np.float64(totals[cell]))
            rows.append((cell,land,snow_grid,fraction,count))
    return rows,totals


def _write_netcdf(
    path: PathInput,
    fractions: Float32Array,
    counts: Int64Array,
    g: GridConfig,
    categories: Int16Array = USGS38,
    *,
    final_output: bool = True,
    snow: SnowGridInfo | None = None,
    qa_statistics: tuple[int, int, int] | None = None,
) -> None:
    """Write a modern NetCDF CTGPROC land-use product.

    Args:
        path: Destination path.
        fractions: Category fractions shaped ``(category, y, x)``.
        counts: Category hit counts shaped ``(category, y, x)``.
        g: Output grid geometry and datum.
        categories: Ordered land-use category identifiers.
        final_output: Mark the product as FRACTION; false marks continuation HITS.
        snow: Optional sparse SNODAS mapping.
        qa_statistics: Optional ``(average hits, threshold, low-cell count)``.

    Returns:
        None. Fractions are stored as ``float32`` and counts as ``int32``.

    Raises:
        ValueError: If array sizes are incompatible with ``g`` or snow
            categories are inconsistent.
        OSError: If the NetCDF file cannot be created or written.
    """
    with Dataset(path,"w",format="NETCDF4") as ds:
        ds.createDimension("category",len(categories)); ds.createDimension("y",g.ny); ds.createDimension("x",g.nx)
        ds.Conventions="CF-1.8"; ds.source="PyCALMET CTGPROC NumPy port"; ds.projection=g.projection; ds.datum=g.datum
        ds.landuse_file_type='FRACTION' if final_output else 'HITS'
        ds.grid_nx=g.nx; ds.grid_ny=g.ny; ds.grid_dx_km=g.dx_km; ds.grid_dy_km=g.dy_km
        ds.x_origin_km=g.x_origin_km; ds.y_origin_km=g.y_origin_km
        ds.false_easting_km=g.false_easting_km; ds.false_northing_km=g.false_northing_km
        ds.ttm_scale=g.ttm_scale
        ds.createVariable("landuse_category","i2",("category",))[:]=categories
        ds.createVariable("landuse_fraction","f4",("category","y","x"),zlib=True)[:]=(
            np.asarray(fractions,dtype=COMPUTE_DTYPE).reshape(len(categories),g.ny,g.nx))
        ds.createVariable("sample_count","i4",("category","y","x"),zlib=True)[:]=(
            np.asarray(counts,dtype=np.int32).reshape(len(categories),g.ny,g.nx))
        if qa_statistics is not None:
            average,threshold,low=qa_statistics
            ds.average_landuse_hits=int(average)
            ds.low_hit_threshold=int(threshold)
            ds.low_hit_cell_count=int(low)
        rows,_=_snow_rows(snow,categories)
        if snow is not None:
            ds.createDimension('snow_record',len(rows))
            ds.snow_grid_information='CTGPROC sparse SNODAS mapping'
            values=np.asarray(rows,dtype=float).reshape((-1,5))
            ds.createVariable('snow_cell_index','i4',('snow_record',))[:]=values[:,0].astype(np.int32)+1
            ds.createVariable('snow_landuse_index','i2',('snow_record',))[:]=values[:,1].astype(np.int16)
            ds.createVariable('snow_grid_index','i4',('snow_record',))[:]=values[:,2].astype(np.int32)
            ds.createVariable('snow_fraction','f4',('snow_record',))[:]=values[:,3]
            ds.createVariable('snow_sample_count','i4',('snow_record',))[:]=values[:,4].astype(np.int32)

def _write_legacy(
    path: Path,
    fractions: Float32Array,
    counts: Int64Array,
    g: GridConfig,
    categories: Int16Array = USGS38,
    *,
    final_output: bool = True,
    snow: SnowGridInfo | None = None,
) -> None:
    """Write a legacy ASCII ``LU.DAT`` CTGPROC product.

    Args:
        path: Destination :class:`Path`.
        fractions: Category fractions shaped ``(category, y, x)``.
        counts: Category hit counts shaped ``(category, y, x)``.
        g: Output grid geometry, projection, and datum.
        categories: Ordered land-use category identifiers.
        final_output: Write FRACTION records when true, otherwise HITS records.
        snow: Optional sparse SNODAS mapping appended to the file.

    Returns:
        None.

    Raises:
        ValueError: If snow categories are inconsistent with ``categories``.
        OSError: If the output file cannot be written.
    """
    with path.open("w",encoding="ascii") as s:
        file_type='FRACTION' if final_output else 'HITS'
        s.write(f"{'LU.DAT':<16}{'2.1':<16}"
                f"{'Coordinate parameters and LandUse categories':<64}\n"
                "   2\nProduced by PyCALMET CTGPROC NumPy implementation\n"
                "Internal Coordinate Transformations  ---  pyproj\n"+file_type+"\n")
        projection=g.projection.upper()
        s.write(f"{projection:<8}\n")
        latitude=lambda value: f'{abs(value):g}{"N" if value>=0 else "S"}'
        longitude=lambda value: f'{abs(value):g}{"E" if value>=0 else "W"}'
        if projection=='UTM':
            s.write(f'{g.utm_zone:4d}{g.hemisphere:<4}\n')
        else:
            parameters=[latitude(g.lat0),longitude(g.lon0)]
            if projection in {'LCC','PS'}: parameters.append(latitude(g.lat1))
            if projection=='LCC': parameters.append(latitude(g.lat2))
            s.write(''.join(f'{value:<16}' for value in parameters)+'\n')
        if projection in {'LCC','LAZA','TTM'}:
            s.write(gfortran_real_pair(g.false_easting_km,g.false_northing_km))
        s.write(f"{g.datum:<8}{NIMA_DATABASE_DATE:<12}\n")
        s.write(f"{g.nx:8d}{g.ny:8d}{g.x_origin_km:12.3f}{g.y_origin_km:12.3f}{g.dx_km:12.3f}{g.dy_km:12.3f}{len(categories):6d}\nKM  \n")
        width=6 if final_output else 5
        s.write("          "+"".join(f"{v:{width}d}" for v in categories)+"\n")
        for j in range(g.ny):
            for i in range(g.nx):
                v=fractions[:,j,i]; flag="   .                " if v.sum() else "   Missing          "
                if final_output:
                    fields="".join(f"{x:6.3f}" for x in v)
                else:
                    fields="".join(f"{int(x):5d}" for x in counts[:,j,i])
                s.write(f"{i+1:5d}{j+1:5d}"+fields+flag+"\n")
        if snow is not None:
            grouped,totals=snow.records(categories)
            s.write('SNOW Grid Information:\n')
            for j in range(g.ny):
                for i in range(g.nx):
                    cell=j*g.nx+i
                    lands=sorted(land for candidate,land in grouped if candidate==cell)
                    fields=f'{i+1:4d}{j+1:4d}{len(lands):3d}'
                    for land_number,land in enumerate(lands):
                        items=grouped[cell,land]
                        separator='' if land_number==0 else ' '
                        fields+=f'{separator}{land:3d}{len(items):4d}'
                        for snow_grid,count in items:
                            fields+=f' {snow_grid:9d}{count/totals[cell]:6.3f}'
                    s.write(fields+'\n')


def _fractions_from_counts(counts: ArrayLike) -> tuple[Int64Array, Float32Array]:
    """Convert integer category hits to model-ready fractions.

    Args:
        counts: Integer hit array shaped ``(category, ...)``.

    Returns:
        Pair ``(totals, fractions)``. ``totals`` is an ``int64`` array over
        the spatial axes; ``fractions`` is ``float32`` with the input shape.
        Cells with no hits contain zero fractions.
    """
    hits=np.asarray(counts,dtype=np.int64)
    totals=hits.sum(axis=0,dtype=np.int64)
    numerator=np.asarray(hits,dtype=np.float64)
    denominator=np.asarray(totals,dtype=np.float64)
    fractions=np.zeros(hits.shape,dtype=np.float64)
    np.divide(numerator,denominator,out=fractions,where=totals>0)
    return totals,fractions.astype(COMPUTE_DTYPE)


def _low_hit_threshold(average_hits: int, threshold_percent: int) -> int:
    """Calculate the integer low-hit threshold used for QA reporting.

    Args:
        average_hits: Integer mean number of source hits per target cell.
        threshold_percent: Integer percentage of the mean used as the cutoff.

    Returns:
        Positive integer threshold, truncated like the legacy calculation.
    """
    product=int(average_hits)*int(threshold_percent)
    return max(1,product//100)


def run(config: CtgprocConfig) -> LanduseResult:
    """Aggregate configured land-use sources onto a CTGPROC target grid.

    Args:
        config: Input datasets and source formats, target grid, category map,
            coastline rules, QA controls, snow mapping, and output settings.

    Returns:
        Summary containing the output path and sampling/coverage statistics.

    Raises:
        FileNotFoundError: If a configured source dataset does not exist.
        ValueError: If inputs, source metadata, target grid, mappings,
            coastline controls, QA threshold, or output settings are invalid.
        NotImplementedError: If a configured source type or projection is not
            supported.
        OSError: If a QA or output file cannot be written.
    """
    if not 0<=config.qa_hit_threshold_percent<=100:
        raise ValueError('qa_hit_threshold_percent/ITHRESH must be between 0 and 100')
    output_format=config.output_format.lower()
    if output_format not in {'auto','legacy','netcdf'}:
        raise ValueError("output_format must be 'auto', 'legacy', or 'netcdf'")
    if config.qa_cell_output:
        directory=_qa_directory(config)
        directory.mkdir(parents=True,exist_ok=True)
        for name in QA_FILENAMES:
            with (directory/name).open('wb'):
                pass
    if config.gshhs_file is not None:
        if config.coastline_file is None:
            raise ValueError('GSHHS extraction requires coastline_file/COASTBLN output')
        extract_gshhs(config.gshhs_file,config.coastline_file,config.grid,
                      source_datum=config.coastline_source_datum,
                      wdbii_datum=config.coastline_wdbii_datum)
    supported={*GLAZ_SPECS,'CTG','NZGEN','NLCD92','NLCD01','NLCDTF',
               'CORINE','GLC2K','UMDGLC','MODIS','GEN'}
    if config.landuse_sources:
        sources=tuple(config.landuse_sources)
    else:
        sources=tuple(CtgprocSource(path,config.input_type,config.input_datum,
                                    config.mesh_density)
                      for path in config.landuse_files)
    if not sources: raise ValueError('CTGPROC requires at least one land-use data file')
    sources=tuple(sources)
    kinds={source.input_type.upper() for source in sources}
    unknown=sorted(kinds-supported)
    if unknown: raise NotImplementedError(f'unsupported CTGPROC input types: {unknown}')
    glaz_count=sum(source.input_type.upper() in GLAZ_SPECS for source in sources)
    if glaz_count>1:
        selected=_select_glaz_regions(config.grid)
        sources=tuple(source for source in sources
                      if source.input_type.upper() not in GLAZ_SPECS or
                      source.input_type.upper() in selected)
    previous_snow: SnowGridInfo | None = None
    categories: Int16Array | None
    counts: Int64Array | None
    if config.previous_landuse is not None:
        categories,counts,previous_snow=_read_previous_landuse(
            config.previous_landuse,config.grid)
    else:
        categories=counts=None
    if not sources and categories is None:
        categories=USGS38
        counts=np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64)
    read=inside=0
    snow=SnowGridInfo.create(config.grid) if config.snow_processing else None
    if snow is not None and previous_snow is not None:
        snow.merge(previous_snow)
    for source in sources:
        kind=source.input_type.upper(); item=Path(source.path)
        active=replace(config,input_type=kind,input_datum=source.input_datum,
                       mesh_density=source.mesh_density,landuse_sources=())
        if kind=='GEN':
            file_categories,partial,file_read,file_inside=_read_with_optional_snow(
                _read_generic,item,active,snow=snow)
            if categories is None: categories=file_categories; counts=np.zeros_like(partial)
            elif not np.array_equal(categories,file_categories): raise ValueError('generic files must declare identical category lists')
        elif kind=='NZGEN':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _read_nzgen,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='NLCD92':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_nlcd92,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind in {'NLCD01','NLCDTF'}:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38
            table=NLCD01_TO_USGS if kind=='NLCD01' else NLCD92_TO_USGS
            partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_nlcd_geotiff,item,active,table,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='CORINE':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_corine,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind in {'GLC2K','UMDGLC'}:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38
            if kind=='GLC2K':
                    partial,file_read,file_inside=_read_with_optional_snow(
                        _accumulate_global_geotiff,item,active,GLC2K_TO_USGS,
                        missing=(24,),fallback_epsg=4326,snow=snow)
            else:
                    partial,file_read,file_inside=_read_with_optional_snow(
                        _accumulate_global_geotiff,item,active,UMDGLC_TO_USGS,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='MODIS':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_modis,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='CTG':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_ctg,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        else:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_glazas,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        counts=cast(Int64Array,counts)
        if kind in GLOBAL_MERGE_TYPES:
            _merge_global_counts(counts,partial,categories)
        else:
            counts+=partial
        read+=file_read; inside+=file_inside
    categories=cast(Int16Array,categories)
    counts=cast(Int64Array,counts)
    raw_totals=counts.sum(axis=0,dtype=np.int64)
    average_hits=int(raw_totals.sum(dtype=np.int64)//(config.grid.nx*config.grid.ny))
    low_hit_threshold=_low_hit_threshold(
        average_hits,config.qa_hit_threshold_percent)
    low_hit_cells=int(np.count_nonzero(raw_totals<low_hit_threshold))
    if config.fill_empty_ocean:
        _fill_empty_ocean(counts,categories,config)
    if config.category_map:
        categories,counts=_remap_categories(categories,counts,config.category_map)
        if snow is not None:
            snow=snow.remap(config.category_map)
    totals,fractions=_fractions_from_counts(counts)
    output=Path(config.output); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+".",delete=False) as tmp: temporary=Path(tmp.name)
    try:
        netcdf=(output_format=='netcdf' or
                (output_format=='auto' and output.suffix.lower() in {'.nc','.nc4','.cdf'}))
        (_write_netcdf(temporary,fractions,counts,config.grid,categories,
                       final_output=config.final_output,snow=snow,
                       qa_statistics=(average_hits,low_hit_threshold,low_hit_cells))
         if netcdf
         else _write_legacy(temporary,fractions,counts,config.grid,categories,
                            final_output=config.final_output,snow=snow))
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    return LanduseResult(output,read,inside,int((totals>0).sum()),average_hits,
                         low_hit_threshold,low_hit_cells)
