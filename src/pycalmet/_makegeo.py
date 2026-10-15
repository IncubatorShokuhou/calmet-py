"""Vectorised MAKEGEO implementation for the standard USGS-38 scheme."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from collections.abc import Iterator,Sequence
from datetime import date,datetime,timedelta
from pathlib import Path
import re
import tempfile
from typing import Final,TextIO,TypeAlias
import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset
from .config import GridConfig,MakegeoConfig
from ._ctgproc import USGS38
from .datums import NIMA_DATABASE_DATE,normalize_datum
from .datetime_utils import parse_date
from .legacy_text import gfortran_real_pair

Float32Array: TypeAlias=npt.NDArray[np.float32]
Float64Array: TypeAlias=npt.NDArray[np.float64]
Int16Array: TypeAlias=npt.NDArray[np.int16]
Int32Array: TypeAlias=npt.NDArray[np.int32]
Int64Array: TypeAlias=npt.NDArray[np.int64]
IndexArray: TypeAlias=npt.NDArray[np.intp]
BoolArray: TypeAlias=npt.NDArray[np.bool_]
NumericArray: TypeAlias=npt.NDArray[np.number]
PropertyRecord: TypeAlias=tuple[float,float,float,float,float,float,int]
GeoFields: TypeAlias=tuple[
    NumericArray,NumericArray,NumericArray,NumericArray,
    NumericArray,NumericArray,NumericArray,
]

OUTCAT: Final[Int16Array]=np.array(
    [10,20,-20,30,40,51,54,55,60,61,62,70,80,90],dtype=np.int16)
MAPCAT: Final[Int16Array]=np.array(
    [10]*7+[20]*4+[30]*3+[40]*3+[51]*3+[54,55,61,62]+[70]*7+[80]*5+[90]*2,
    dtype=np.int16)
MODEL_DTYPE: Final[type[np.float32]]=np.float32
ACCUM_DTYPE: Final[type[np.float64]]=np.float64

Z0: Final[Float32Array]=np.array([.5]+[1.]*6+[.25]*4+[.05]*3+[1.]*3+[.001]*5+[1.,.2]+[.05]*7+[.2]*5+[.05]*2,dtype=MODEL_DTYPE)
ALBEDO: Final[Float32Array]=np.array([.18]*7+[.15]*4+[.25]*3+[.1]*3+[.1]*5+[.1,.1]+[.3]*7+[.3]*5+[.7]*2,dtype=MODEL_DTYPE)
BOWEN: Final[Float32Array]=np.array([1.]+[1.5]*6+[1.]*4+[1.]*3+[1.]*3+[0.]*5+[.5,.1]+[1.]*7+[.5]*5+[.5]*2,dtype=MODEL_DTYPE)
SOIL: Final[Float32Array]=np.array([.2]+[.25]*6+[.15]*4+[.15]*3+[.15]*3+[1.]*5+[.25,.25]+[.15]*7+[.15]*5+[.15]*2,dtype=MODEL_DTYPE)
LAI: Final[Float32Array]=np.array([1.]+[.2]*6+[3.]*4+[.5]*3+[7.]*3+[0.]*5+[2.,1.]+[.05]*7+[0.]*5+[0.]*2,dtype=MODEL_DTYPE)
STANDARD_PROPERTIES: Final[dict[int,PropertyRecord]]={int(category):(float(Z0[i]),float(ALBEDO[i]),float(BOWEN[i]),
    float(SOIL[i]),0.,float(LAI[i]),int(MAPCAT[i])) for i,category in enumerate(USGS38)}


def _category_sum(
    values: npt.ArrayLike,
    mask: npt.ArrayLike | None=None,
) -> Float64Array:
    """Sum category fractions with extra reduction precision.

    Args:
        values: Array shaped ``(category, ...)``; values are read as ``float32``.
        mask: Optional Boolean selector for the category axis.

    Returns:
        ``float64`` array over the remaining axes.

    Raises:
        ValueError: If ``mask`` cannot select the category axis.
    """
    data: Float32Array=np.asarray(values,dtype=MODEL_DTYPE)
    selected: BoolArray=(np.ones(data.shape[0],dtype=bool) if mask is None else
                         np.asarray(mask,dtype=bool))
    return np.sum(data[selected],axis=0,dtype=ACCUM_DTYPE)


def _water_sum(
    fractions: npt.ArrayLike,
    categories: npt.ArrayLike,
    water_categories: Sequence[int],
) -> Float64Array:
    """Sum land-use fractions assigned to configured water categories.

    Args:
        fractions: ``float32``-compatible array shaped ``(category, ...)``.
        categories: Category identifier for each first-axis row.
        water_categories: Category identifiers treated as water.

    Returns:
        ``float64`` water-fraction array over the spatial axes.
    """
    data: Float32Array=np.asarray(fractions,dtype=MODEL_DTYPE)
    category_values: NumericArray=np.asarray(categories)
    return np.sum(data[np.isin(category_values,water_categories)],axis=0,
                  dtype=ACCUM_DTYPE)


def _snow_albedo_values(
    normal: npt.ArrayLike,
    snowy: npt.ArrayLike,
    reduction: npt.ArrayLike,
    maximum_age: npt.ArrayLike,
    age: npt.ArrayLike,
) -> Float32Array:
    """Evaluate WTSWALB's category-specific snow-age albedo decay.

    Args:
        normal: Snow-free albedo values as fractions.
        snowy: Fresh-snow albedo values as fractions.
        reduction: Category snow-albedo reduction coefficients.
        maximum_age: Positive category decay time scales in days.
        age: Snow ages in days.

    Returns:
        ``float32`` albedo array with the broadcast input shape.
    """
    normal_values: Float32Array=np.asarray(normal,dtype=MODEL_DTYPE)
    snowy_values: Float32Array=np.asarray(snowy,dtype=MODEL_DTYPE)
    reduction_values: Float32Array=np.asarray(reduction,dtype=MODEL_DTYPE)
    maximum_age_values: Int32Array=np.asarray(maximum_age,dtype=np.int32)
    age_values: Int64Array=np.asarray(age,dtype=np.int64)
    clipped: Float32Array=np.minimum(
        age_values,maximum_age_values).astype(MODEL_DTYPE)
    ratio: Float32Array=np.asarray(
        clipped/maximum_age_values.astype(MODEL_DTYPE),dtype=MODEL_DTYPE)
    result: Float32Array=snowy_values*(
        1.-(snowy_values-normal_values)*reduction_values*ratio)
    return np.asarray(result,dtype=MODEL_DTYPE)

@dataclass(frozen=True,slots=True)
class GeophysicalResult:
    """Summarize one completed MAKEGEO operation.

    Attributes:
        output: Primary generated geophysical product path.
        populated_cells: Number of grid cells with land-use coverage.
        terrain_minimum_m: Minimum output terrain height in metres.
        terrain_maximum_m: Maximum output terrain height in metres.
        outputs: All generated paths; snow runs contain one path per day.
        qa_cell: Validated one-based QA cell, when configured.
        qa_values: Surface-property values reported for ``qa_cell``.
    """

    output: Path
    populated_cells: int
    terrain_minimum_m: float
    terrain_maximum_m: float
    outputs: tuple[Path,...] = ()
    qa_cell: tuple[int,int] | None = None
    qa_values: tuple[float,...] | None = None


def _qa_cell_values(
    terrain: NumericArray,
    fields: Sequence[NumericArray],
    config: MakegeoConfig,
) -> tuple[float,...] | None:
    """Extract MAKEGEO list-file values at a configured QA cell.

    Args:
        terrain: Terrain-height array shaped ``(y, x)`` in metres.
        fields: Seven output surface fields, each shaped ``(y, x)``.
        config: Target grid and optional one-based ``qa_cell``.

    Returns:
        Tuple of Python floats ordered as dominant land use, terrain, and the
        remaining six surface properties; ``None`` if no valid cell is set.
    """
    if config.qa_cell is None:
        return None
    ix,iy=config.qa_cell
    if not 1<=ix<=config.grid.nx or not 1<=iy<=config.grid.ny:
        return None
    return tuple(float(field[iy-1,ix-1]) for field in (fields[0],terrain,*fields[1:]))


@dataclass(frozen=True,slots=True)
class SnowReference:
    """Store the sparse CTGPROC-to-SNODAS cross-reference.

    Attributes:
        cell_index: Zero-based flattened MAKEGEO target-cell indices.
        category_index: Zero-based input-category indices.
        snow_grid_index: One-based flattened global SNODAS cell indices.
        fraction: ``float32`` sample weights for each reference fragment.
    """

    cell_index: Int64Array
    category_index: Int64Array
    snow_grid_index: Int64Array
    fraction: Float32Array


@dataclass(frozen=True,slots=True)
class SnowDataHeader:
    """Describe the formatted ``SNOW.DAT`` grid and source window.

    Attributes:
        dataset: Dataset declaration string.
        version: File-format version string.
        modification: Producer or modification description.
        projection: Processed-grid projection identifier.
        datum: Processed-grid datum identifier.
        date_system: Datum-table date or date-system label.
        units: Processed-grid coordinate units.
        longitude_origin: Processed-grid longitude origin in degrees east.
        latitude_origin: Processed-grid latitude origin in degrees north.
        nx: Processed-window column count.
        ny: Processed-window row count.
        dx: Processed-grid longitudinal spacing in degrees.
        dy: Processed-grid latitudinal spacing in degrees.
        timezone: Timestamp time-zone label.
        datetime_type: Timestamp representation label.
        raw_longitude_origin: Global SNODAS longitude origin in degrees east.
        raw_latitude_origin: Global SNODAS latitude origin in degrees north.
        raw_nx: Global raw-grid column count.
        raw_ny: Global raw-grid row count.
        raw_dx: Global raw-grid longitudinal spacing in degrees.
        raw_dy: Global raw-grid latitudinal spacing in degrees.
        i_begin: One-based first raw-grid column in the processed window.
        i_end: One-based last raw-grid column in the processed window.
        j_begin: One-based first raw-grid row in the processed window.
        j_end: One-based last raw-grid row in the processed window.
        variables_per_day: Number of packed fields in each daily group.
    """

    dataset: str
    version: str
    modification: str
    projection: str
    datum: str
    date_system: str
    units: str
    longitude_origin: float
    latitude_origin: float
    nx: int
    ny: int
    dx: float
    dy: float
    timezone: str
    datetime_type: str
    raw_longitude_origin: float
    raw_latitude_origin: float
    raw_nx: int
    raw_ny: int
    raw_dx: float
    raw_dy: float
    i_begin: int
    i_end: int
    j_begin: int
    j_end: int
    variables_per_day: int


@dataclass(frozen=True,slots=True)
class SnowVariable:
    """Store one decoded daily ``SNOW.DAT`` variable.

    Attributes:
        timestamp: Time-zone-naive valid time.
        name: Fixed-width variable identifier, such as ``SNOWDEP``.
        values: Two-dimensional ``float32`` field in the variable's physical
            units, with missing packed values represented by NaN.
    """

    timestamp: datetime
    name: str
    values: Float32Array


def _expand_packed_snow(
    packed: npt.ArrayLike,
    minimum: int,
    missing: int,
    scale: float,
    size: int,
) -> Float32Array:
    """Expand RDSN's signed-integer runs and restore physical values.

    Args:
        packed: Encoded value/run-length integer stream.
        minimum: Additive packed-data offset.
        missing: Packed integer used for missing values.
        scale: Nonzero divisor converting packed integers to physical units.
        size: Required number of expanded grid values.

    Returns:
        One-dimensional ``float32`` array of length ``size``; missing values
        are NaN.

    Raises:
        ValueError: If scale is zero or the run stream is truncated, overflows,
            or does not exactly fill the output grid.
    """
    packed_values: Int64Array=np.asarray(packed,dtype=np.int64).ravel()
    if scale==0.:
        raise ValueError('packed snow scale must be non-zero')
    output: Int64Array=np.empty(int(size),dtype=np.int64)
    source=target=0
    while source<packed_values.size:
        value=int(packed_values[source]); source+=1
        if value>=0 or value==missing:
            count=1
        else:
            count=-value
            if source>=packed_values.size:
                raise ValueError('truncated packed snow run')
            value=int(packed_values[source]); source+=1
        if target+count>output.size:
            raise ValueError('packed snow stream expands beyond its grid')
        output[target:target+count]=value
        target+=count
    if target!=output.size:
        raise ValueError('packed snow stream does not fill its grid')
    result: Float32Array=np.asarray(output,dtype=MODEL_DTYPE)
    result=(result+minimum)/scale
    result=np.asarray(result,dtype=MODEL_DTYPE)
    result[output==missing]=np.nan
    return result


def _read_formatted_snow_header(stream: TextIO) -> SnowDataHeader:
    """Parse and validate the header of a formatted ``SNOW.DAT`` stream.

    Args:
        stream: Open text stream positioned at the declaration record.

    Returns:
        Parsed processed-grid, raw-grid, time, and variable-count metadata.

    Raises:
        ValueError: If declaration, comments, geometry, window, or dimensions
            are truncated or invalid.
    """
    first=stream.readline()
    if len(first)<32:
        raise ValueError('SNOW.DAT header record 1 is truncated')
    dataset=first[:16].strip(); version=first[16:32].strip(); modification=first[32:96].strip()
    try:
        comment_count=int(stream.readline().strip())
    except ValueError as error:
        raise ValueError('SNOW.DAT comment count is invalid') from error
    if not 0<=comment_count<=100:
        raise ValueError('SNOW.DAT comment count must be 0..100')
    for _ in range(comment_count):
        if not stream.readline():
            raise ValueError('SNOW.DAT comments are truncated')
    projection=stream.readline()[:8].strip()
    datum_line=stream.readline(); datum=datum_line[:8].strip(); date_system=datum_line[8:18].strip()
    units=stream.readline()[:4].strip()
    try:
        grid=stream.readline().split()
        lon,lat=float(grid[0]),float(grid[1]); nx,ny=int(grid[2]),int(grid[3])
        dx,dy=float(grid[4]),float(grid[5])
        timezone=stream.readline()[:8].strip()
        datetime_type=stream.readline()[:16].strip()
        raw=stream.readline().split()
        raw_lon,raw_lat=float(raw[0]),float(raw[1]); raw_nx,raw_ny=int(raw[2]),int(raw[3])
        raw_dx,raw_dy=float(raw[4]),float(raw[5])
        window=[int(value) for value in stream.readline().split()]
    except (IndexError,ValueError) as error:
        raise ValueError('SNOW.DAT grid header is invalid') from error
    if len(window)!=5 or nx<1 or ny<1 or raw_nx<1 or raw_ny<1:
        raise ValueError('SNOW.DAT grid dimensions/window are invalid')
    return SnowDataHeader(dataset,version,modification,projection,datum,date_system,
        units,lon,lat,nx,ny,dx,dy,timezone,datetime_type,raw_lon,raw_lat,
        raw_nx,raw_ny,raw_dx,raw_dy,*window)


def _formatted_snow_timestamp(
    line: str,
    offset: int,
    label: str,
) -> tuple[datetime,tuple[int,int,int,int,int]]:
    """Parse one fixed-width ``YYYY MM DD HH SSSS`` timestamp.

    Args:
        line: Fixed-width variable-header record.
        offset: Zero-based character offset of the timestamp.
        label: Timestamp label used in error messages.

    Returns:
        Pair of a time-zone-naive :class:`datetime` and its five parsed integer
        fields ``(year, month, day, hour, seconds)``.

    Raises:
        ValueError: If fields or their calendar/time ranges are invalid.
    """
    try:
        fields=(int(line[offset:offset+4]),int(line[offset+4:offset+7]),
                int(line[offset+7:offset+10]),int(line[offset+10:offset+13]),
                int(line[offset+14:offset+18]))
    except ValueError as error:
        raise ValueError(f'SNOW.DAT variable {label} timestamp is invalid') from error
    year,month,day,hour,seconds=fields
    if not 0<=seconds<=3600:
        raise ValueError(f'SNOW.DAT variable {label} timestamp is invalid')
    try:
        timestamp=datetime(year,month,day,hour)+timedelta(seconds=seconds)
    except ValueError as error:
        raise ValueError(f'SNOW.DAT variable {label} timestamp is invalid') from error
    return timestamp,fields


def _read_formatted_snow_variable(
    stream: TextIO,
    header: SnowDataHeader,
) -> SnowVariable | None:
    """Read and expand one packed variable from a formatted snow stream.

    Args:
        stream: Open text stream positioned at a variable header or EOF.
        header: Parsed snow grid dimensions.

    Returns:
        Decoded :class:`SnowVariable` with a ``float32`` ``(ny, nx)`` field,
        or ``None`` at clean end of file.

    Raises:
        ValueError: If timestamps, packed metadata, record counts, or run data
            are inconsistent or truncated.
    """
    line=stream.readline()
    if not line:
        return None
    line=line.rstrip('\n').ljust(86)
    timestamp,first_fields=_formatted_snow_timestamp(line,0,'first')
    _,second_fields=_formatted_snow_timestamp(line,19,'second')
    if second_fields!=first_fields:
        raise ValueError('SNOW.DAT variable header timestamps do not match')
    if first_fields[3]!=0 or first_fields[4]!=0:
        raise ValueError('SNOW.DAT daily variable timestamp must be midnight')
    try:
        name=line[38:46].strip()
        minimum=int(line[47:54]); maximum=int(line[54:61]); missing=int(line[61:68])
        scale=float(line[68:76]); packed_count=int(line[76:86])
    except ValueError as error:
        raise ValueError('SNOW.DAT variable header is invalid') from error
    if maximum-minimum>32767 or packed_count<0:
        raise ValueError('SNOW.DAT packed range/count is invalid')
    packed: list[int]=[]
    while len(packed)<packed_count:
        data=stream.readline()
        if not data:
            raise ValueError('SNOW.DAT packed data are truncated')
        try:
            packed.extend(int(data[index:index+7]) for index in range(0,len(data.rstrip('\n')),7)
                          if data[index:index+7].strip())
        except ValueError as error:
            raise ValueError('SNOW.DAT packed integer is invalid') from error
    if len(packed)!=packed_count:
        raise ValueError('SNOW.DAT packed record contains extra integers')
    values=_expand_packed_snow(packed,minimum,missing,scale,header.nx*header.ny)
    return SnowVariable(timestamp,name,values.reshape(header.ny,header.nx))


def _fill_snow_spatial(values: npt.ArrayLike,radius: int=2) -> Float32Array:
    """Fill missing snow cells from expanding square spatial windows.

    Args:
        values: Two-dimensional snow field; NaN marks missing values.
        radius: Maximum non-negative search radius in grid cells.

    Returns:
        ``float32`` array with the input shape. Each missing cell receives the
        first available square-window mean; unresolved cells remain NaN.

    Raises:
        ValueError: If ``values`` is not two-dimensional or ``radius`` is
            negative.
    """
    source: Float32Array=np.asarray(values,dtype=MODEL_DTYPE)
    if source.ndim!=2 or radius<0:
        raise ValueError('spatial snow fill requires a 2-D field and non-negative radius')
    result: Float32Array=source.copy(); missing: BoolArray=np.isnan(source)
    valid: BoolArray=np.isfinite(source)
    ny,nx=source.shape
    for distance in range(1,radius+1):
        sums: Float64Array=np.zeros(source.shape,dtype=ACCUM_DTYPE)
        counts: Int32Array=np.zeros(source.shape,dtype=np.int32)
        # FILLS scans JJ outside II.  Each offset still updates the whole
        # target grid at once; only the order-dependent 5x5 window is looped.
        for dy in range(-distance,distance+1):
            target_y=slice(max(0,-dy),min(ny,ny-dy))
            source_y=slice(max(0,dy),min(ny,ny+dy))
            for dx in range(-distance,distance+1):
                target_x=slice(max(0,-dx),min(nx,nx-dx))
                source_x=slice(max(0,dx),min(nx,nx+dx))
                good: BoolArray=valid[source_y,source_x]
                region: Float64Array=sums[target_y,target_x]
                region+=np.where(good,source[source_y,source_x],0.)
                counts[target_y,target_x]+=good
        use: BoolArray=missing&np.isnan(result)&(counts>=1)
        averaged: Float64Array=np.divide(
            sums,counts,out=np.zeros_like(sums),where=counts>0)
        result[use]=np.asarray(averaged[use],dtype=MODEL_DTYPE)
    return result


def _fill_snow_temporal(values: npt.ArrayLike,current: int=1) -> Float32Array:
    """Fill missing current-day snow values from a multi-day window.

    Args:
        values: ``(day, y, x)`` snow array; NaN marks missing values.
        current: Zero-based day index retained where it is finite.

    Returns:
        ``float32`` ``(y, x)`` field using the current day where available and
        the mean of all finite window days elsewhere.

    Raises:
        ValueError: If ``values`` is not three-dimensional or ``current`` is
            outside its day axis.
    """
    window: Float32Array=np.asarray(values,dtype=MODEL_DTYPE)
    if window.ndim!=3 or not 0<=current<window.shape[0]:
        raise ValueError('temporal snow fill requires (day,y,x) and a valid current index')
    result: Float32Array=window[current].copy()
    missing: BoolArray=np.isnan(result)
    valid: BoolArray=np.isfinite(window)
    counts: Int32Array=np.zeros(result.shape,dtype=np.int32)
    totals: Float64Array=np.zeros(result.shape,dtype=ACCUM_DTYPE)
    for day in range(window.shape[0]):
        totals+=np.where(valid[day],window[day],0.)
        counts+=valid[day]
    replacement: Float64Array=np.divide(
        totals,counts,out=np.full(result.shape,np.nan,dtype=ACCUM_DTYPE),
        where=counts>0)
    result[missing]=np.asarray(replacement[missing],dtype=MODEL_DTYPE)
    return result


def _snow_control_date(value: date | datetime | str) -> datetime:
    """Normalize a MAKEGEO snow-control value to naive midnight.

    Args:
        value: Supported date, datetime, integer, or string date value.

    Returns:
        Time-zone-naive :class:`datetime` at midnight on the parsed date.

    Raises:
        TypeError: If the value has an unsupported type.
        ValueError: If the value is not a valid calendar date.
    """
    day=parse_date(value,'MAKEGEO snow date')
    return datetime.combine(day,datetime.min.time())


def _iter_formatted_snow_days(
    path: str | Path,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> Iterator[tuple[datetime,Float32Array,Int32Array,SnowDataHeader]]:
    """Stream RDSN's rolling three-day window as daily depth and age fields.

    Args:
        path: Formatted ``SNOW.DAT`` input path.
        start_date: First requested calendar day, inclusive.
        end_date: Last requested calendar day, inclusive.

    Yields:
        Tuples ``(date, depth, age, header)``. ``date`` is naive midnight;
        depth is a temporally/spatially filled ``float32`` ``(ny, nx)`` field
        in the file's physical units; age is ``int32`` days since qualifying
        precipitation; header describes the processed SNODAS window.

    Raises:
        ValueError: If the date interval, SNODAS geometry, daily groups,
            timestamps, variables, continuity, or requested-day coverage is
            invalid.
        OSError: If the snow file cannot be read.
    """
    start=_snow_control_date(start_date); end=_snow_control_date(end_date)
    if start>end:
        raise ValueError('snow_start_date/IDATEBEG must not follow snow_end_date/IDATEEND')
    cutoff=start-timedelta(days=2)
    ages: Int32Array | None=None
    window: deque[tuple[datetime,Float32Array,Int32Array]]=deque(maxlen=3)
    yielded: list[datetime]=[]
    with Path(path).open('r',encoding='ascii',errors='strict') as stream:
        header=_read_formatted_snow_header(stream)
        if header.raw_nx!=6935 or header.raw_ny!=3351:
            raise ValueError('SNOW.DAT raw grid dimensions do not match SNODAS')
        if (abs(header.raw_longitude_origin+124.729584)>1.e-3 or
                abs(header.raw_latitude_origin-52.8704185)>1.e-3 or
                abs(header.raw_dx-8.33333377e-3)>1.e-5 or
                abs(header.raw_dy+8.33333377e-3)>1.e-5):
            raise ValueError('SNOW.DAT raw grid geometry does not match SNODAS')
        ages=np.full((header.ny,header.nx),14,dtype=np.int32)
        while True:
            records: list[SnowVariable]=[]
            for _ in range(header.variables_per_day):
                record=_read_formatted_snow_variable(stream,header)
                if record is None:
                    if records:
                        raise ValueError('SNOW.DAT ends inside a daily variable group')
                    break
                records.append(record)
            if not records:
                break
            for record in records:
                timestamp=record.timestamp
                if timestamp.tzinfo is not None or timestamp.utcoffset() is not None:
                    raise ValueError('SNOW.DAT daily timestamps must be timezone-naive')
                if timestamp.time()!=datetime.min.time():
                    raise ValueError('SNOW.DAT daily timestamps must be midnight')
            timestamps: set[datetime]={record.timestamp for record in records}
            if len(timestamps)!=1:
                raise ValueError('SNOW.DAT variables in one group have different timestamps')
            timestamp=records[0].timestamp
            if timestamp<cutoff:
                continue
            variables: dict[str,Float32Array]={record.name:record.values for record in records}
            if 'SNOWDEP' not in variables or 'PRECIPS' not in variables:
                raise ValueError('SNOW.DAT day lacks SNOWDEP or PRECIPS')
            precipitation: Float32Array=variables['PRECIPS']
            ages=np.where(precipitation>=.1,0,ages+1).astype(np.int32)
            depth: Float32Array=_fill_snow_spatial(
                variables['SNOWDEP'],radius=2)
            if window and timestamp-window[-1][0]!=timedelta(days=1):
                raise ValueError('SNOW.DAT daily timestamps are not consecutive')
            window.append((timestamp,depth,ages.copy()))
            if len(window)<3:
                continue
            current=window[1][0]
            if start<=current<=end:
                depth_current: Float32Array=_fill_snow_temporal(
                    np.stack([item[1] for item in window]),current=1)
                yielded.append(current)
                # Depth and age describe the same (middle) day.  The third
                # record exists only to fill missing depth values temporally.
                yield current,depth_current,window[1][2],header
            if current>=end:
                break
    expected=(end-start).days+1
    if len(yielded)!=expected:
        raise ValueError(f'SNOW.DAT supplies {len(yielded)} of {expected} requested days')


def _read_snow_reference(
    path: str | Path,
    g: GridConfig,
    categories: NumericArray,
) -> SnowReference:
    """Read CTGPROC's sparse SNOWREF cross-reference.

    Args:
        path: NetCDF or legacy ASCII land-use product containing snow metadata.
        g: MAKEGEO target grid.
        categories: Ordered input land-use category identifiers.

    Returns:
        Validated sparse reference with zero-based target/category indices,
        one-based global SNODAS indices, and ``float32`` fractions.

    Raises:
        ValueError: If snow metadata is absent, incomplete, out of bounds,
            duplicated, incorrectly ordered, or contains invalid fractions.
        OSError: If the land-use file cannot be read.
    """
    source=Path(path)
    cell: Int64Array
    land: Int64Array
    snow: Int64Array
    fraction: Float32Array
    if source.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(source) as dataset:
            required: tuple[str,...]=(
                'snow_cell_index','snow_landuse_index','snow_grid_index',
                'snow_fraction')
            missing: list[str]=[
                name for name in required if name not in dataset.variables]
            if missing:
                raise ValueError(f'land-use file has no snow-grid reference: {missing}')
            cell=np.asarray(dataset['snow_cell_index'][:],dtype=np.int64)-1
            land=np.asarray(dataset['snow_landuse_index'][:],dtype=np.int64)-1
            snow=np.asarray(dataset['snow_grid_index'][:],dtype=np.int64)
            fraction=np.asarray(dataset['snow_fraction'][:],dtype=MODEL_DTYPE)
    else:
        lines: list[str]=source.read_text(
            encoding='ascii',errors='strict').splitlines()
        try:
            marker=lines.index('SNOW Grid Information:')
        except ValueError as error:
            raise ValueError('land-use file has no SNOW Grid Information section') from error
        records=lines[marker+1:]
        if len(records)!=g.nx*g.ny:
            raise ValueError('snow-grid reference does not contain every target cell')
        cells: list[int]=[]; lands: list[int]=[]
        snow_grids: list[int]=[]; fractions: list[float]=[]
        seen: set[tuple[int,int]]=set()
        for expected,line in enumerate(records):
            fields=line.split()
            try:
                i,j,type_count=(int(value) for value in fields[:3])
            except (ValueError,TypeError) as error:
                raise ValueError(f'invalid snow-grid reference record {expected+1}') from error
            if (i,j)!=(expected%g.nx+1,expected//g.nx+1):
                raise ValueError('snow-grid reference cell order does not match the target grid')
            if not 0<=type_count<=10:
                raise ValueError('snow-grid reference has more than 10 land-use types in a cell')
            position=3
            for _ in range(type_count):
                if position+2>len(fields):
                    raise ValueError('truncated snow-grid land-use reference')
                land_index=int(fields[position]); count=int(fields[position+1]); position+=2
                if not 1<=land_index<=len(categories):
                    raise ValueError('snow-grid land-use index is outside the category table')
                if not 0<=count<=10:
                    raise ValueError('snow-grid reference has more than 10 snow grids per land-use type')
                if (expected,land_index) in seen:
                    raise ValueError('duplicate land-use type in snow-grid target cell')
                seen.add((expected,land_index))
                for _ in range(count):
                    if position+2>len(fields):
                        raise ValueError('truncated snow-grid index/fraction pair')
                    cells.append(expected); lands.append(land_index-1)
                    snow_grids.append(int(fields[position])); fractions.append(float(fields[position+1]))
                    position+=2
            if position!=len(fields):
                raise ValueError('extra fields in snow-grid reference record')
        cell=np.asarray(cells,dtype=np.int64)
        land=np.asarray(lands,dtype=np.int64)
        snow=np.asarray(snow_grids,dtype=np.int64)
        fraction=np.asarray(fractions,dtype=MODEL_DTYPE)
    size=g.nx*g.ny
    if not (cell.shape==land.shape==snow.shape==fraction.shape):
        raise ValueError('snow-grid reference arrays have inconsistent lengths')
    if np.any((cell<0)|(cell>=size)) or np.any((land<0)|(land>=len(categories))):
        raise ValueError('snow-grid reference index is outside its declared dimensions')
    if np.any(snow<1) or np.any(~np.isfinite(fraction)) or np.any(fraction<0.):
        raise ValueError('snow-grid reference contains invalid index or fraction values')
    return SnowReference(cell,land,snow,fraction)

def _validate_netcdf_grid(
    dataset: Dataset,
    g: GridConfig,
    label: str,
) -> None:
    """Validate optional NetCDF grid metadata against a target grid.

    Args:
        dataset: Open netCDF4 dataset.
        g: Expected grid dimensions, geometry, projection, and datum.
        label: Human-readable input label used in errors.

    Returns:
        None.

    Raises:
        ValueError: If any present grid attribute differs from ``g``.
    """
    expected: dict[str,int | float]={'grid_nx':g.nx,'grid_ny':g.ny,'grid_dx_km':g.dx_km,
              'grid_dy_km':g.dy_km,'x_origin_km':g.x_origin_km,
              'y_origin_km':g.y_origin_km}
    differences: list[str]=[]
    for name,wanted in expected.items():
        if name not in dataset.ncattrs(): continue
        actual=getattr(dataset,name)
        equal=(int(actual)==wanted if isinstance(wanted,int) else
               np.isclose(float(actual),wanted,rtol=1.e-4,atol=0.))
        if not equal: differences.append(name)
    if ('projection' in dataset.ncattrs() and
            str(dataset.projection).strip().upper()!=g.projection.upper()):
        differences.append('projection')
    if ('datum' in dataset.ncattrs() and
            normalize_datum(str(dataset.datum))!=normalize_datum(g.datum)):
        differences.append('datum')
    if differences:
        raise ValueError(f'{label} NetCDF grid mismatch: '+', '.join(differences))


def _legacy_angle(value: str | int | float) -> float:
    """Parse a legacy directional angle with an optional compass suffix.

    Args:
        value: Numeric text optionally suffixed with N, S, E, or W.

    Returns:
        Signed angle as a float in degrees; longitudes are normalized to the
        ``[-180, 180]`` branch when needed.

    Raises:
        ValueError: If the text is empty, nonnumeric, or has multiple suffixes.
    """
    token=re.sub(r'\s+','',str(value))
    if not token:
        raise ValueError('empty directional angle')
    directions=re.findall(r'[NSEW]',token,re.I)
    if len(directions)>1:
        raise ValueError('invalid directional angle')
    direction=directions[0].upper() if directions else ''
    if direction: token=re.sub(r'[NSEW]','',token,flags=re.I)
    number=float(token)
    if direction in {'S','W'}: number=-abs(number)
    elif direction in {'N','E'}: number=abs(number)
    if direction in {'E','W'}:
        if number>180.: number-=360.
        elif number < -180.: number+=360.
    return number


def _legacy_grid_header(
    lines: Sequence[str],
    index: int,
    g: GridConfig,
    label: str,
    *,
    landuse: bool=False,
) -> tuple[int,list[str]]:
    """Read and validate the shared WRTHEAD metadata used by MAKEGEO.

    The legacy reader checks the full projection, datum, and grid geometry for
    each terrain and land-use input.  Return the first line following the grid
    record together with that record's fields.

    Args:
        lines: Complete legacy input file as text records.
        index: Zero-based index of the projection record.
        g: Expected MAKEGEO target grid.
        label: Human-readable input label used in errors.
        landuse: Require the additional land-use category-count field.

    Returns:
        Pair ``(next_index, grid_fields)`` where ``next_index`` points to the
        line after the grid record and ``grid_fields`` contains its tokens.

    Raises:
        ValueError: If projection parameters, datum, dimensions, geometry,
            false offsets, UTM settings, or header structure do not match.
    """
    try:
        projection=lines[index][:8].strip().upper(); index+=1
        parameter_line=lines[index]; index+=1
        false_easting=false_northing=0.
        if projection in {'LCC','LAZA','TTM'}:
            false_fields=lines[index].split(); index+=1
            if len(false_fields)<2:
                raise ValueError('incomplete false easting/northing')
            false_easting,false_northing=map(float,false_fields[:2])
        datum=lines[index][:8].strip().upper(); index+=1
        fields=lines[index].split(); index+=1
        required=7 if landuse else 6
        if len(fields)<required:
            raise ValueError('incomplete grid record')
        nx,ny=int(fields[0]),int(fields[1])
        geometry=tuple(float(value) for value in fields[2:6])
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid {label} grid header') from error

    differences: list[str]=[]
    if projection!=g.projection.upper(): differences.append('projection')
    if normalize_datum(datum)!=normalize_datum(g.datum): differences.append('datum')
    if (nx,ny)!=(g.nx,g.ny): differences.append('dimensions')
    expected=(g.x_origin_km,g.y_origin_km,g.dx_km,g.dy_km)
    if not np.allclose(geometry,expected,rtol=1.e-4,atol=0.):
        differences.append('geometry')

    try:
        if projection=='UTM':
            match=re.fullmatch(r'\s*([+-]?\d+)\s*([NSns])\s*',parameter_line)
            if match is None:
                raise ValueError('invalid UTM parameters')
            if int(match.group(1))!=g.utm_zone or match.group(2).upper()!=g.hemisphere.upper():
                differences.append('UTM zone/hemisphere')
        else:
            count={'LCC':4,'PS':3,'EM':2,'LAZA':2,'TTM':2}.get(projection)
            if count is None:
                raise ValueError('invalid projection parameters')
            fixed=[parameter_line[offset:offset+16].strip()
                   for offset in range(0,16*count,16)]
            parameters=(fixed if len(parameter_line)>=16*count and all(fixed)
                        else parameter_line.split())
            if len(parameters)<count:
                raise ValueError('invalid projection parameters')
            actual=tuple(_legacy_angle(value) for value in parameters[:count])
            wanted=(g.lat0,g.lon0,g.lat1,g.lat2)[:count]
            if not np.allclose(actual,wanted,rtol=1.e-4,atol=0.):
                differences.append('projection parameters')
        if projection in {'LCC','LAZA','TTM'} and not np.allclose(
                (false_easting,false_northing),
                (g.false_easting_km,g.false_northing_km),rtol=1.e-4,atol=0.):
            differences.append('false easting/northing')
    except ValueError as error:
        raise ValueError(f'invalid {label} projection header') from error
    if differences:
        prefix=(f'{label} projection or datum mismatch; '
                if {'projection','datum'}&set(differences) else '')
        raise ValueError(prefix+f'{label} grid mismatch: '+', '.join(differences))
    return index,fields


def _read_terrain(path: str | Path,g: GridConfig) -> Float32Array:
    """Read terrain heights from a NetCDF or legacy TERREL product.

    Args:
        path: Terrain product path.
        g: Expected MAKEGEO target grid.

    Returns:
        ``float32`` terrain-height array shaped ``(g.ny, g.nx)`` in metres,
        ordered south to north along the first axis.

    Raises:
        KeyError: If a NetCDF product lacks ``terrain_height``.
        ValueError: If grid metadata, shape, legacy header, units, ordering, or
            sample count is invalid.
        OSError: If the product cannot be read.
    """
    p=Path(path)
    terrain: Float32Array
    if p.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(p) as ds:
            _validate_netcdf_grid(ds,g,'terrain')
            terrain=np.asarray(ds['terrain_height'][:],dtype=MODEL_DTYPE)
        if terrain.shape!=(g.ny,g.nx):
            raise ValueError(f'terrain NetCDF array shape mismatch in {p}')
        return terrain
    lines=p.read_text(encoding='ascii').splitlines()
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid terrain header in {p}') from error
    index=2+comments
    index,_=_legacy_grid_header(lines,index,g,'terrain')
    try:
        unit_line=lines[index]; index+=1
        xy_unit=unit_line[:4].strip().upper()
        z_unit=unit_line[4:8].strip().upper()
        order=lines[index].split(); index+=1
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid terrain grid header in {p}') from error
    if xy_unit!='KM': raise ValueError(f'terrain horizontal units must be KM in {p}')
    scales={'M':1.,'KM':1000.,'FT':.3048,'YD':.9144}
    if z_unit not in scales: raise ValueError(f'unsupported terrain elevation units in {p}: {z_unit}')
    if len(order)<2 or order[0].upper()!='W_E' or order[1].upper() not in {'N_S','S_N'}:
        raise ValueError(f'unsupported terrain record order in {p}')
    values=np.fromstring(' '.join(lines[index:]),sep=' ')
    if values.size != g.nx*g.ny: raise ValueError(f"terrain size mismatch in {p}")
    terrain=np.asarray(values.reshape(g.ny,g.nx)*scales[z_unit],dtype=MODEL_DTYPE)
    return terrain[::-1] if order[1].upper()=='N_S' else terrain

def _read_landuse(
    path: str | Path,
    g: GridConfig,
    expected_categories: Sequence[int] | None=None,
) -> tuple[Int64Array,Float32Array]:
    """Read category fractions from a NetCDF or legacy CTGPROC product.

    Args:
        path: FRACTION land-use product path.
        g: Expected MAKEGEO target grid.
        expected_categories: Optional required category identifiers and order.

    Returns:
        Pair ``(categories, fractions)``. Categories are ``int64`` and
        fractions are ``float32`` with shape
        ``(category, g.ny, g.nx)``.

    Raises:
        KeyError: If required NetCDF category or fraction variables are absent.
        ValueError: If file type, version, grid metadata, categories, shape,
            cell ordering, units, or trailing records are invalid.
        OSError: If the product cannot be read.
    """
    p=Path(path)
    expected: Int64Array | None=(None if expected_categories is None else
                                 np.asarray(tuple(expected_categories),dtype=np.int64))
    if p.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(p) as ds:
            _validate_netcdf_grid(ds,g,'land-use')
            cats: NumericArray=np.asarray(ds['landuse_category'][:])
            frac: NumericArray=np.asarray(ds['landuse_fraction'][:])
        if frac.ndim!=3 or frac.shape[1:]!=(g.ny,g.nx):
            raise ValueError(f'land-use NetCDF array shape mismatch in {p}')
        if cats.size!=frac.shape[0]:
            raise ValueError(f'land-use NetCDF category dimension mismatch in {p}')
        if expected is not None and not np.array_equal(cats.astype(np.int64),expected):
            raise ValueError(f'land-use categories do not match MAKEGEO input categories in {p}')
        return cats.astype(np.int64),frac.astype(MODEL_DTYPE)
    lines=p.read_text(encoding='ascii').splitlines()
    try:
        version=float(lines[0][16:32].strip())
        comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error: raise ValueError(f'invalid land-use header in {p}') from error
    index=2+comments
    if lines[index].strip().upper()!='FRACTION':
        raise ValueError('MAKEGEO requires a FRACTION land-use file')
    index+=1
    index,fields=_legacy_grid_header(lines,index,g,'land-use',landuse=True)
    try: xy_unit=lines[index][:4].strip().upper(); index+=1
    except IndexError as error: raise ValueError(f'invalid land-use header in {p}') from error
    if xy_unit!='KM': raise ValueError(f'land-use horizontal units must be KM in {p}')
    category_count=int(fields[6])
    if version>2.01:
        categories: Int64Array=np.fromstring(
            lines[index],sep=' ',dtype=np.int64); index+=1
        if categories.size!=category_count:
            raise ValueError(f'land-use category header mismatch in {p}')
    else:
        if expected is None:
            raise ValueError(f'land-use version {version:g} requires MAKEGEO input categories')
        categories=expected.copy()
        if categories.size!=category_count:
            raise ValueError(f'land-use category header mismatch in {p}')
    if expected is not None and not np.array_equal(categories,expected):
        raise ValueError(f'land-use categories do not match MAKEGEO input categories in {p}')
    record_count=g.nx*g.ny
    records=lines[index:index+record_count]
    result: Float32Array=np.empty(
        (categories.size,g.ny,g.nx),dtype=MODEL_DTYPE)
    if len(records)!=record_count: raise ValueError(f"land-use record count mismatch in {p}")
    tail=lines[index+record_count:]
    if tail and tail[0]!='SNOW Grid Information:':
        raise ValueError(f'unrecognized records after land-use grid in {p}')
    for record,line in enumerate(records):
        values: Float64Array=np.fromstring(
            line[:10+6*categories.size],sep=' ')
        if values.size!=categories.size+2:
            raise ValueError(f'invalid land-use fraction record in {p}')
        i,j=int(values[0]),int(values[1])
        expected_index=(record%g.nx+1,record//g.nx+1)
        if (i,j)!=expected_index:
            raise ValueError(f'land-use cell order does not match the target grid in {p}')
        result[:,j-1,i-1]=values[2:]
    return categories,result

def _split_categories(
    categories: NumericArray,
    frac: NumericArray,
    splits: Sequence[tuple[int,int,float]],
) -> tuple[NumericArray,NumericArray]:
    """Transfer configured percentages between existing category fractions.

    Args:
        categories: One-dimensional category identifier array.
        frac: ``float32``-compatible fractions shaped ``(category, y, x)``.
        splits: Sequence of ``(source, target, percent)`` transfers.

    Returns:
        Pair of ``int64`` categories and a ``float32`` fraction copy. With no
        splits, the original objects are returned unchanged.

    Raises:
        ValueError: If a source or target category is absent.
    """
    if not splits: return categories,frac
    category_values: Int64Array=np.asarray(categories,dtype=np.int64)
    result: Float32Array=np.array(frac,dtype=MODEL_DTYPE,copy=True)
    lookup: dict[int,int]={
        int(value):index for index,value in enumerate(category_values)}
    grouped: dict[int,list[tuple[int,float]]]={}
    for source,target,percent in splits: grouped.setdefault(int(source),[]).append((int(target),float(percent)))
    for source,targets in grouped.items():
        if source not in lookup or any(target not in lookup for target,_ in targets):
            raise ValueError('split source and target categories must exist in the input category list')
        source_index=lookup[source]
        percentage_sum=0.
        for target,percent in targets:
            fraction=percent/100.
            contribution=fraction*result[source_index]
            target_index=lookup[target]
            result[target_index]+=contribution
            percentage_sum+=fraction
        result[source_index]-=percentage_sum*result[source_index]
    return category_values,result

def _fields(
    frac: NumericArray,
    config: MakegeoConfig,
    categories: NumericArray=USGS38,
) -> GeoFields:
    """Derive the seven static MAKEGEO surface fields.

    Args:
        frac: Land-use fractions shaped ``(category, y, x)``.
        config: Category properties, mapping, coverage, and water controls.
        categories: Category identifier corresponding to each fraction layer.

    Returns:
        Tuple ``(dominant, roughness, albedo, Bowen ratio, soil heat-flux
        parameter, anthropogenic heat flux, leaf-area index)``. ``dominant``
        is an integer category array; physical fields are ``float32`` and all
        arrays have shape ``(grid.ny, grid.nx)``. Roughness is in metres.

    Raises:
        ValueError: If category properties or output mappings are incomplete,
            coverage is below the configured threshold, a replacement category
            is absent, or weighted values are physically invalid.
    """
    categories,frac=_split_categories(categories,frac,config.category_splits)
    fraction_values: Float32Array=np.asarray(frac,dtype=MODEL_DTYPE)
    properties: dict[int,PropertyRecord]=config.input_properties or STANDARD_PROPERTIES
    missing_properties: list[int]=[
        int(category) for category in categories if int(category) not in properties]
    if missing_properties: raise ValueError(f'missing MAKEGEO properties for categories: {missing_properties}')
    table: Float32Array=np.asarray(
        [properties[int(category)][:6] for category in categories],
        dtype=MODEL_DTYPE)
    mapped: Int16Array=np.asarray(
        [properties[int(category)][6] for category in categories],dtype=np.int16)
    if any(int(value) not in set(config.output_categories) for value in mapped):
        raise ValueError('mapped land-use category is absent from output_categories')
    totals: Float64Array=_category_sum(fraction_values)
    missing: BoolArray=totals < .001
    if np.any((totals>.001)&(totals<config.minimum_coverage)):
        raise ValueError('land-use coverage lies between replacement and FLUMIN thresholds')
    if missing.any():
        found: IndexArray=np.flatnonzero(
            categories==config.missing_category)
        if not found.size: raise ValueError('missing_category is absent from input categories')
        k=int(found[0]); fraction_values=fraction_values.copy()
        fraction_values[k,missing]+=1.-totals[missing]
        totals=_category_sum(fraction_values)
    water_mask: BoolArray=np.isin(categories,config.water_categories)
    water: BoolArray=_water_sum(
        fraction_values,categories,config.water_categories)>(
        config.water_threshold+.0001)
    eligible: BoolArray=np.where(
        water[None,:,:],water_mask[:,None,None],~water_mask[:,None,None])
    selected: IndexArray=np.argmax(
        np.where(eligible,fraction_values,-np.inf),axis=0)
    dominant: Int16Array=mapped[selected]
    def weighted(
        values: npt.ArrayLike,
        *,
        logarithmic: bool=False,
    ) -> Float32Array:
        """Calculate a category-fraction weighted target-grid field.

        Args:
            values: One property value for each input category.
            logarithmic: Use a geometric rather than arithmetic mean.

        Returns:
            ``float32`` array shaped ``(grid.ny, grid.nx)``.
        """
        weights: Float64Array=np.asarray(
            fraction_values,dtype=ACCUM_DTYPE)
        reference: Float64Array=np.asarray(values,dtype=ACCUM_DTYPE)
        if logarithmic:
            reference=np.log(reference)
        numerator: Float64Array=np.sum(
            weights*reference[:,None,None],axis=0,dtype=ACCUM_DTYPE)
        denominator: Float64Array=np.sum(
            weights,axis=0,dtype=ACCUM_DTYPE)
        result: Float64Array=numerator/denominator
        if logarithmic:
            result=np.exp(result)
        return np.asarray(result,dtype=MODEL_DTYPE)
    roughness: Float32Array=weighted(table[:,0],logarithmic=True)
    fields: tuple[Float32Array,...]=tuple(
        weighted(table[:,index]) for index in range(1,6))
    return (dominant,roughness,fields[0],fields[1],fields[2],fields[3],fields[4])


def _snow_weighted_fields(
    frac: NumericArray,
    categories: NumericArray,
    reference: SnowReference,
    snow_depth: npt.ArrayLike,
    snow_age: npt.ArrayLike,
    config: MakegeoConfig,
    *,
    snow_i_offset: int=0,
    snow_j_offset: int=0,
    snow_full_nx: int=6935,
) -> GeoFields:
    """Derive daily snow-adjusted MAKEGEO surface fields.

    Args:
        frac: Static land-use fractions shaped ``(category, y, x)``.
        categories: Category identifier corresponding to each fraction layer.
        reference: Sparse target-cell/category-to-SNODAS sample mapping.
        snow_depth: Two-dimensional snow depth field in metres.
        snow_age: Matching integer snow-age field in days.
        config: Normal/snow property tables and snow algorithm controls.
        snow_i_offset: Zero-based global column of the snow-data window.
        snow_j_offset: Zero-based global row of the snow-data window.
        snow_full_nx: Column count of the flattened global SNODAS grid.

    Returns:
        Tuple ``(dominant, roughness, albedo, Bowen ratio, soil heat-flux
        parameter, anthropogenic heat flux, leaf-area index)`` on the target
        grid. Physical arrays are ``float32``; roughness and snow depth use
        metres.

    Raises:
        ValueError: If property tables, output maps, snow arrays, reference
            indices/weights, roughness controls, or albedo controls are invalid.
    """
    categories,frac=_split_categories(categories,frac,config.category_splits)
    normal: dict[int,PropertyRecord]=config.input_properties or STANDARD_PROPERTIES
    snowy: dict[int,PropertyRecord]=config.snow_properties
    if set(int(value) for value in categories)-set(normal):
        raise ValueError('normal land-use properties do not cover the input categories')
    if set(int(value) for value in categories)-set(snowy):
        raise ValueError('snow land-use properties do not cover the input categories')
    fraction_values: Float32Array=np.asarray(frac,dtype=MODEL_DTYPE)
    normal_table: Float32Array=np.asarray(
        [normal[int(category)][:6] for category in categories],
        dtype=MODEL_DTYPE)
    snow_table: Float32Array=np.asarray(
        [snowy[int(category)][:6] for category in categories],
        dtype=MODEL_DTYPE)
    mapped: Int16Array=np.asarray(
        [normal[int(category)][6] for category in categories],dtype=np.int16)
    if any(snowy[int(category)][6]!=mapped[index]
           for index,category in enumerate(categories)):
        raise ValueError('snow and no-snow output category maps must match')
    depth: Float32Array=np.asarray(snow_depth,dtype=MODEL_DTYPE)
    age: Int64Array=np.asarray(snow_age,dtype=np.int64)
    if depth.shape!=age.shape or depth.ndim!=2:
        raise ValueError('snow depth and age must be matching two-dimensional arrays')
    global_index: Int64Array=reference.snow_grid_index-1
    local_i: Int64Array=global_index%snow_full_nx-int(snow_i_offset)
    local_j: Int64Array=global_index//snow_full_nx-int(snow_j_offset)
    if np.any((local_i<0)|(local_i>=depth.shape[1])|
              (local_j<0)|(local_j>=depth.shape[0])):
        raise ValueError('snow-grid reference lies outside the supplied snow-data window')
    sample_depth: Float32Array=depth[local_j,local_i]
    sample_age: Int64Array=age[local_j,local_i]
    if np.any(~np.isfinite(sample_depth)):
        raise ValueError('snow depth contains missing values at referenced cells')
    land: Int64Array=reference.category_index
    # A target cell can contain many snow-reference fragments.  Accumulate
    # those weights in FP64, then return the physical fields as FP32.
    weights: Float64Array=np.asarray(reference.fraction,dtype=ACCUM_DTYPE)
    cell: Int64Array=reference.cell_index
    cell_count=config.grid.nx*config.grid.ny
    denominator: Float64Array=np.zeros(cell_count,dtype=ACCUM_DTYPE)
    np.add.at(denominator,cell,weights)
    if np.any(denominator<=0.):
        raise ValueError('snow-grid reference has no weight for one or more target cells')

    def aggregate(
        values: npt.ArrayLike,
        *,
        logarithmic: bool=False,
    ) -> Float32Array:
        """Aggregate SNODAS-fragment values to MAKEGEO target cells.

        Args:
            values: One value per sparse reference fragment.
            logarithmic: Use a geometric rather than arithmetic weighted mean.

        Returns:
            ``float32`` array shaped ``(config.grid.ny, config.grid.nx)``.
        """
        value_array: Float64Array=np.asarray(values,dtype=ACCUM_DTYPE)
        if logarithmic:
            value_array=np.log(value_array)
        numerator: Float64Array=np.zeros(cell_count,dtype=ACCUM_DTYPE)
        np.add.at(numerator,cell,weights*value_array)
        result: Float64Array=numerator/denominator
        if logarithmic:
            result=np.exp(result)
        return np.asarray(result.reshape(config.grid.ny,config.grid.nx),
                          dtype=MODEL_DTYPE)

    has_snow: BoolArray=sample_depth>0.
    roughness: Float32Array
    if config.snow_roughness_method==1:
        if config.snow_height_scale_m<=0.:
            raise ValueError('snow_height_scale_m/HSCL must be positive')
        scale=config.snow_height_scale_m
        obstacle: Float32Array=normal_table[:,0]*scale
        minimum: np.float32=obstacle[-1]
        roughness=np.where(
            sample_depth<config.snow_depth_minimum_m,
            normal_table[land,0],
            np.maximum(obstacle[land]-sample_depth,minimum)/scale)
    elif config.snow_roughness_method==2:
        roughness=np.where(has_snow,snow_table[land,0],normal_table[land,0])
    else:
        raise ValueError('snow_roughness_method/MSRL must be 1 or 2')
    roughness=aggregate(roughness,logarithmic=True)

    if config.snow_albedo_method==1:
        parameters: dict[int,tuple[int,float]]=config.snow_age_parameters
        if set(int(value) for value in categories)-set(parameters):
            raise ValueError('snow age parameters do not cover the input categories')
        raw_times: Float64Array=np.asarray(
            [parameters[int(category)][0] for category in categories],
            dtype=ACCUM_DTYPE)
        reductions: Float32Array=np.asarray(
            [parameters[int(category)][1] for category in categories],
            dtype=MODEL_DTYPE)
        times: Int32Array=np.rint(raw_times).astype(np.int32)
        if np.any(times<=0):
            raise ValueError('snow albedo age time scales must be positive')
        snow_albedo: Float32Array=_snow_albedo_values(
            normal_table[land,1],snow_table[land,1],reductions[land],
            times[land],sample_age)
        albedo: Float32Array=np.where(
            sample_depth<config.snow_depth_minimum_m,
            normal_table[land,1],snow_albedo)
    elif config.snow_albedo_method==2:
        albedo=np.where(has_snow,snow_table[land,1],normal_table[land,1])
    else:
        raise ValueError('snow_albedo_method/MSAL must be 1 or 2')
    fields: list[Float32Array]=[aggregate(albedo)]
    for column in range(2,6):
        values: Float32Array=np.where(
            has_snow,snow_table[land,column],normal_table[land,column])
        fields.append(aggregate(values))

    dominant: NumericArray=_fields(fraction_values,config,categories)[0]
    return (dominant,roughness,fields[0],fields[1],fields[2],fields[3],fields[4])

def _merge_landuse(
    primary: NumericArray,
    secondary: NumericArray,
) -> Float32Array:
    """Supplement only cells missing from a primary land-use grid.

    Args:
        primary: Primary ``float32`` fractions shaped ``(category, y, x)``.
        secondary: Secondary fractions with the same category order and shape.

    Returns:
        Array selecting all secondary category layers where the primary total
        is at most ``0.001``; other cells retain the primary values.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    primary_values: Float32Array=np.asarray(primary,dtype=MODEL_DTYPE)
    secondary_values: Float32Array=np.asarray(secondary,dtype=MODEL_DTYPE)
    if primary_values.shape!=secondary_values.shape:
        raise ValueError('primary and secondary land-use grids do not match')
    missing: BoolArray=_category_sum(primary_values)<=.001
    return np.where(missing[None,:,:],secondary_values,primary_values)

def _write_netcdf(
    path: str | Path,
    terrain: NumericArray,
    fields: Sequence[NumericArray],
    g: GridConfig,
    config: MakegeoConfig,
) -> None:
    """Write a modern NetCDF MAKEGEO geophysical product.

    Args:
        path: Destination path.
        terrain: ``(y, x)`` terrain height in metres.
        fields: Seven MAKEGEO output fields shaped ``(y, x)``.
        g: Output grid geometry, projection, and datum.
        config: Output category and water-category metadata.

    Returns:
        None. Continuous variables are stored as ``float32`` and dominant land
        use as ``int16``.

    Raises:
        ValueError: If an input field cannot conform to the grid dimensions.
        OSError: If the NetCDF file cannot be created or written.
    """
    names: tuple[str,...]=('dominant_landuse','roughness_length','albedo','bowen_ratio','soil_heat_flux_parameter','anthropogenic_heat_flux','leaf_area_index')
    terrain_values: Float32Array=np.asarray(terrain,dtype=MODEL_DTYPE)
    field_values: tuple[NumericArray,...]=(
        fields[0],*(np.asarray(value,dtype=MODEL_DTYPE) for value in fields[1:]))
    with Dataset(path,'w',format='NETCDF4') as ds:
        ds.createDimension('y',g.ny); ds.createDimension('x',g.nx); ds.Conventions='CF-1.8'; ds.source='PyCALMET MAKEGEO NumPy port'
        ds.projection=g.projection; ds.datum=g.datum; ds.false_easting_km=g.false_easting_km; ds.false_northing_km=g.false_northing_km
        ds.grid_nx=g.nx; ds.grid_ny=g.ny; ds.grid_dx_km=g.dx_km; ds.grid_dy_km=g.dy_km
        ds.x_origin_km=g.x_origin_km; ds.y_origin_km=g.y_origin_km
        ds.latitude_origin=g.lat0; ds.longitude_origin=g.lon0; ds.standard_parallel_1=g.lat1; ds.standard_parallel_2=g.lat2
        ds.ttm_scale=g.ttm_scale
        ds.utm_zone=g.utm_zone; ds.hemisphere=g.hemisphere
        ds.water_category_min=int(config.output_water_min); ds.water_category_max=int(config.output_water_max)
        ds.landuse_categories=','.join(str(int(value)) for value in config.output_categories)
        ds.createVariable('terrain_height','f4',('y','x'),zlib=True)[:]=terrain_values
        for name,data in zip(names,field_values): ds.createVariable(name,'i2' if name=='dominant_landuse' else 'f4',('y','x'),zlib=True)[:]=data

def _comma_rows(
    stream: TextIO,
    data: NumericArray,
    *,
    width: int,
    decimals: int | None,
    values_per_record: int,
) -> None:
    """Write Fortran-style comma-delimited rows with format reversion.

    Args:
        stream: Open text output stream.
        data: Two-dimensional array written from its last row to first.
        width: Fixed field width in characters.
        decimals: Decimal places for floats; ``None`` writes integers.
        values_per_record: Maximum number of values on one text record.

    Returns:
        None.

    Raises:
        OSError: If the stream cannot be written.
    """
    for row in np.asarray(data)[::-1]:
        fields: list[str]=[
            (f'{int(value):{width}d}' if decimals is None else
             f'{float(value):{width}.{decimals}f}') for value in row]
        for start in range(0,len(fields),values_per_record):
            stop=min(start+values_per_record,len(fields))
            stream.write(''.join(
                value+(',' if index<len(fields)-1 else '')
                for index,value in enumerate(fields[start:stop],start))+'\n')


def _rows(stream: TextIO,data: NumericArray,fmt: str) -> None:
    """Write one legacy MAKEGEO floating grid using its named format.

    Args:
        stream: Open text output stream.
        data: Two-dimensional numeric grid.
        fmt: ``".4f"`` for four decimal places; other supported fields use
            two decimal places.

    Returns:
        None.
    """
    decimals=4 if fmt=='.4f' else 2
    _comma_rows(stream,data,width=7,decimals=decimals,values_per_record=100)

def _write_legacy(
    path: Path,
    terrain: NumericArray,
    fields: Sequence[NumericArray],
    g: GridConfig,
    config: MakegeoConfig,
) -> None:
    """Write a legacy ASCII ``GEO.DAT`` MAKEGEO product.

    Args:
        path: Destination :class:`Path`.
        terrain: ``(y, x)`` terrain height in metres.
        fields: Dominant category and six physical fields shaped ``(y, x)``.
        g: Output grid geometry, projection, and datum.
        config: Title, output categories, and water-category bounds.

    Returns:
        None.

    Raises:
        OSError: If the output file cannot be written.
    """
    dominant,z0,alb,bow,soil,qf,lai=fields
    with path.open('w',encoding='ascii') as s:
        s.write(f"{'GEO.DAT':<16}{'2.1':<16}"
                f"{'Header structure with coordinate parameters':<64}\n"
                "   2\nProduced by PyCALMET MAKEGEO NumPy implementation\n"
                f"{config.title[:80]:<80}\n")
        projection=g.projection.upper()
        latitude=lambda value: f'{abs(value):g}{"N" if value>=0 else "S"}'
        longitude=lambda value: f'{abs(value):g}{"E" if value>=0 else "W"}'
        s.write(f'{projection:<8}\n')
        if projection=='UTM':
            s.write(f'{g.utm_zone:4d}{g.hemisphere:<4}\n')
        else:
            parameters: list[str]=[latitude(g.lat0),longitude(g.lon0)]
            if projection in {'LCC','PS'}: parameters.append(latitude(g.lat1))
            if projection=='LCC': parameters.append(latitude(g.lat2))
            s.write(''.join(f'{value:<16}' for value in parameters)+'\n')
        if projection in {'LCC','LAZA','TTM'}:
            s.write(gfortran_real_pair(g.false_easting_km,g.false_northing_km))
        s.write(f'{g.datum:<8}{NIMA_DATABASE_DATE:<12}\n')
        s.write(f'{g.nx:8d}{g.ny:8d}{g.x_origin_km:12.3f}{g.y_origin_km:12.3f}{g.dx_km:12.3f}{g.dy_km:12.3f}\nKM  M   \n')
        s.write(f' 1          - LAND USE DATA - (1=new categories)\n{len(config.output_categories):5d}{config.output_water_min:4d}{config.output_water_max:4d} - NLU, IWAT1, IWAT2\n '+''.join(f' {v:3d}' for v in config.output_categories)+'\n')
        _comma_rows(s,dominant,width=7,decimals=None,values_per_record=10)
        s.write(f' 1.0000  - {"TERRAIN heights - HTFAC (Conversion to meters)":<70}\n'); _rows(s,terrain,'.2f')
        for label,data,fmt in [('gridded z0 field',z0,'.4f'),('gridded albedo field',alb,'.2f'),('gridded Bowen ratio field',bow,'.2f'),('gridded soil heat flux parameters',soil,'.2f'),('gridded anthropogenic heat flux field',qf,'.2f'),('gridded leaf area index field',lai,'.2f')]:
            s.write(f' 2    - {label:<70}\n'); _rows(s,data,fmt)


def _write_qa_grids(
    terrain: NumericArray,
    dominant: NumericArray,
    g: GridConfig,
    config: MakegeoConfig,
) -> None:
    """Write optional Surfer QA grids and the MAKEGEO color map.

    Args:
        terrain: ``(y, x)`` terrain-height field in metres.
        dominant: ``(y, x)`` dominant output land-use categories.
        g: Grid dimensions, origin, and spacing in kilometres.
        config: QA output paths and IMAGE coordinate-mode setting.

    Returns:
        None. A constant terrain field removes an existing terrain plot rather
        than writing a degenerate Surfer grid.

    Raises:
        ValueError: If ``image_mode`` is not 0 or 1.
        OSError: If a configured QA product cannot be written.
    """
    if config.image_mode not in {0,1}:
        raise ValueError('MAKEGEO IMAGE must be zero or one')
    if config.landuse_plot_output is not None:
        path=Path(config.landuse_plot_output); path.parent.mkdir(parents=True,exist_ok=True)
        if config.image_mode==0:
            xlo=g.x_origin_km+.5*g.dx_km; xhi=g.x_origin_km+(g.nx-.5)*g.dx_km
            ylo=g.y_origin_km+.5*g.dy_km; yhi=g.y_origin_km+(g.ny-.5)*g.dy_km
        else:
            xlo=g.x_origin_km; xhi=g.x_origin_km+g.nx*g.dx_km
            ylo=g.y_origin_km; yhi=g.y_origin_km+g.ny*g.dy_km
        with path.open('w',encoding='ascii') as stream:
            stream.write('DSAA\n')
            stream.write(f'{g.nx:12d}{g.ny:12d}\n{xlo:12.4f}{xhi:12.4f}\n')
            stream.write(f'{ylo:12.4f}{yhi:12.4f}\n{10:12d}{100:12d}\n')
            for row in np.asarray(dominant):
                stream.write(''.join(f'{abs(int(value)):3d}  ' for value in row)+'\n')
    if config.terrain_plot_output is not None:
        path=Path(config.terrain_plot_output); path.parent.mkdir(parents=True,exist_ok=True)
        minimum=float(np.min(terrain)); maximum=float(np.max(terrain))
        if maximum==minimum:
            path.unlink(missing_ok=True)
        else:
            xlo=g.x_origin_km+.5*g.dx_km; xhi=g.x_origin_km+(g.nx-.5)*g.dx_km
            ylo=g.y_origin_km+.5*g.dy_km; yhi=g.y_origin_km+(g.ny-.5)*g.dy_km
            with path.open('w',encoding='ascii') as stream:
                stream.write('DSAA\n')
                stream.write(f'{g.nx:12d}{g.ny:12d}\n{xlo:12.4f}{xhi:12.4f}\n')
                stream.write(f'{ylo:12.4f}{yhi:12.4f}\n{minimum:12.4E}{maximum:12.4E}\n')
                for row in np.asarray(terrain):
                    stream.write(''.join(f'{value:11.4E} ' for value in row)+'\n')
    if config.color_map_output is not None:
        path=Path(config.color_map_output); path.parent.mkdir(parents=True,exist_ok=True)
        colors='''ColorMap 1 1
    0.000000 255 255   0
   11.000000 255 255   0
   11.000000 204 255 102
   22.100000 204 255 102
   22.100000 160 255 160
   33.100000 160 255 160
   33.100000  80 255  80
   44.200000  80 255  80
   44.200000 153 255 255
   50.000000 153 255 255
   50.000000  60 204 255
   55.400000  60 204 255
   55.400000 204 153 204
   66.500000 204 153 204
   66.500000 255 204 153
   77.600000 255 204 153
   77.600000 255 255 204
   88.700000 255 255 204
   88.700000 255 255 255
   99.029557 255 255 255
   99.029557 255   0   0
  100.000000 255   0   0
'''
        path.write_text(colors,encoding='ascii')

def run(config: MakegeoConfig) -> GeophysicalResult:
    """Build static or daily snow-adjusted CALMET geophysical fields.

    Args:
        config: Terrain and land-use inputs, target grid, category properties,
            optional snow data, QA controls, and output settings.

    Returns:
        Summary containing the primary output path, all dated snow outputs,
        coverage count, terrain range in metres, and optional QA-cell values.

    Raises:
        FileNotFoundError: If a configured input file does not exist.
        KeyError: If a required NetCDF variable is absent.
        ValueError: If input grids, categories, properties, coverage, snow
            metadata, dates, or output controls are invalid.
        OSError: If an input cannot be read or an output cannot be written.
    """
    terrain: Float32Array=(
        _read_terrain(config.terrain,config.grid) if config.use_terrain else
        np.zeros((config.grid.ny,config.grid.nx),dtype=MODEL_DTYPE))
    expected_categories=tuple((config.input_properties or STANDARD_PROPERTIES).keys())
    categories: Int64Array
    frac: Float32Array
    categories,frac=_read_landuse(
        config.landuse,config.grid,expected_categories)
    if config.secondary_landuse is not None:
        secondary_categories: Int64Array
        secondary: Float32Array
        secondary_categories,secondary=_read_landuse(
            config.secondary_landuse,config.grid,expected_categories)
        if not np.array_equal(categories,secondary_categories): raise ValueError('primary and secondary land-use categories do not match')
        frac=_merge_landuse(frac,secondary)
    dominant: NumericArray=_fields(frac,config,categories)[0]
    _write_qa_grids(terrain,dominant,config.grid,config)
    output: Path=Path(config.output); output.parent.mkdir(parents=True,exist_ok=True)
    fields: GeoFields
    if config.snow_processing:
        reference: SnowReference=_read_snow_reference(
            config.landuse,config.grid,categories)
        if config.snow_data is None:
            raise ValueError('snow_processing requires snow_data/SNOWDAT')
        if config.snow_start_date is None or config.snow_end_date is None:
            raise ValueError('snow processing requires IDATEBEG and IDATEEND')
        if not output.suffix:
            raise ValueError('dated MAKEGEO snow output requires a filename extension')
        outputs: list[Path]=[]
        for date,depth,age,header in _iter_formatted_snow_days(
                config.snow_data,config.snow_start_date,config.snow_end_date):
            fields=_snow_weighted_fields(
                frac,categories,reference,depth,age,config,
                snow_i_offset=header.i_begin-1,snow_j_offset=header.j_begin-1,
                snow_full_nx=header.raw_nx)
            dated: Path=output.with_name(
                f'{output.stem}{date:%Y%m%d}{output.suffix}')
            with tempfile.NamedTemporaryFile(dir=dated.parent,prefix=dated.name+'.',
                                             delete=False) as tmp:
                temporary=Path(tmp.name)
            try:
                if dated.suffix.lower() in {'.nc','.nc4','.cdf'}:
                    _write_netcdf(temporary,terrain,fields,config.grid,config)
                else:
                    _write_legacy(temporary,terrain,fields,config.grid,config)
                temporary.replace(dated)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(dated)
        first=outputs[0]
        qa_values=_qa_cell_values(terrain,fields,config)
        return GeophysicalResult(first,int(frac.sum(axis=0).astype(bool).sum()),
            float(np.min(terrain)),float(np.max(terrain)),tuple(outputs),
            config.qa_cell if qa_values is not None else None,qa_values)
    fields=_fields(frac,config,categories)
    qa_values=_qa_cell_values(terrain,fields,config)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',delete=False) as tmp: temporary=Path(tmp.name)
    try:
        if output.suffix.lower() in {'.nc','.nc4','.cdf'}: _write_netcdf(temporary,terrain,fields,config.grid,config)
        else: _write_legacy(temporary,terrain,fields,config.grid,config)
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    return GeophysicalResult(output,int(frac.sum(axis=0).astype(bool).sum()),
        float(np.min(terrain)),float(np.max(terrain)),(output,),
        config.qa_cell if qa_values is not None else None,qa_values)
