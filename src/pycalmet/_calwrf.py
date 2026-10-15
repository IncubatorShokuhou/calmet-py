"""Vectorized WRF extraction corresponding to CALWRF 2.0.2."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime,timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import netCDF4 as nc
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import fortran_inclusive_slice
from ._calwrf_listing import write_calwrf_listing
from ._path_safety import validate_calwrf_paths

from .config import CalwrfConfig
from .legacy_calwrf import write_m3d
from .datetime_utils import parse_timestamp

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
Shape2D: TypeAlias = tuple[int, int]
Shape3D: TypeAlias = tuple[int, int, int]
IndexRange: TypeAlias = tuple[int, int] | None
WrfRecord: TypeAlias = tuple[Path, int, str]
RainState: TypeAlias = tuple[Float32Array, Float32Array]
ValueLimits: TypeAlias = dict[str, tuple[float, float]]
NetCDFDType: TypeAlias = Literal["f4", "f8", "i2"]
NetCDFScalar: TypeAlias = np.float32 | np.float64 | np.int16
CompressionLevel: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if TYPE_CHECKING:
    NetCDFBytesVariable: TypeAlias = nc.Variable[np.bytes_]
    NetCDFNumericVariable: TypeAlias = nc.Variable[NetCDFScalar]
else:
    # netCDF4's compiled runtime class is not subscriptable, while its stubs
    # model Variable as generic.  These aliases keep runtime reflection valid.
    NetCDFBytesVariable = nc.Variable
    NetCDFNumericVariable = nc.Variable

_LIMITS_3D: ValueLimits = {
    "pressure": (10.0, 1100.0), "temperature": (150.0, 330.0),
    "wind_direction": (0.0, 360.0), "wind_speed": (0.0, 150.0),
    "vertical_velocity": (-50.0, 50.0), "qvapor": (0.0, 0.05),
    "qcloud": (0.0, 0.05), "qrain": (0.0, 0.05),
    "qice": (0.0, 0.05), "qsnow": (0.0, 0.05), "qgraupel": (0.0, 0.05),
}
_LIMITS_2D: ValueLimits = {
    "surface_pressure": (50000.0, 110000.0), "precipitation": (0.0, 500.0),
    "snow_cover": (0.0, 1.1), "shortwave_down": (0.0, 1500.0),
    "longwave_down": (0.0, 1000.0), "temperature_2m": (210.0, 330.0),
    "qvapor_2m": (0.0, 0.05), "wind_direction_10m": (0.0, 360.0),
    "wind_speed_10m": (0.0, 50.0), "sst": (0.0, 350.0),
}


def _horizontal_range_check(
    values: ArrayLike,
    low: float,
    high: float,
    ys: slice,
    xs: slice,
) -> Float32Array:
    """Replace out-of-range values from valid horizontal neighbours.

    Args:
        values: Array whose last two axes are the full WRF ``(y, x)`` grid.
            Leading axes, such as the vertical level, are preserved.
        low: Inclusive lower validity limit in the units of ``values``.
        high: Inclusive upper validity limit in the units of ``values``.
        ys: Zero-based slice selecting rows from the full source grid.
        xs: Zero-based slice selecting columns from the full source grid.

    Returns:
        A float32 array over the selected horizontal domain. Its shape is
        ``values.shape[:-2] + (len(ys), len(xs))``. Invalid cells are replaced
        by the mean of valid north, south, east, and west neighbours, or by
        the nearest range bound when no valid neighbour exists.
    """
    original: Float32Array = np.asarray(values, dtype="f4")
    result: Float32Array = original[..., ys, xs].copy()
    ny, nx = original.shape[-2:]
    jb, je, ib, ie = ys.start + 1, ys.stop, xs.start + 1, xs.stop
    jb_edge, je_edge = min(jb, 2), max(je, ny - 1)
    ib_edge, ie_edge = min(ib, 2), max(ie, nx - 1)
    jj, ii = np.arange(jb, je + 1), np.arange(ib, ie + 1)
    j1 = np.maximum(jj - 1, jb_edge) - 1
    j2 = np.minimum(jj + 1, je_edge) - 1
    i1 = np.maximum(ii - 1, ib_edge) - 1
    i2 = np.minimum(ii + 1, ie_edge) - 1
    neighbours: Float32Array = np.stack((
        original[..., jj[:, None] - 1, i1[None, :]],
        original[..., jj[:, None] - 1, i2[None, :]],
        original[..., j1[:, None], ii[None, :] - 1],
        original[..., j2[:, None], ii[None, :] - 1],
    ), axis=0)
    good: BoolArray = (neighbours >= low) & (neighbours <= high)
    count = good.sum(axis=0)
    total: Float32Array = np.where(good, neighbours, 0.0).sum(axis=0, dtype="f4")
    fallback: Float32Array = np.where(result < low, low, high).astype("f4")
    replacement = np.divide(total, count, out=fallback, where=count > 0)
    return np.where((result < low) | (result > high), replacement, result).astype("f4")


def _vertical_range_check(
    values: ArrayLike,
    low: float,
    high: float,
    zs: slice,
    ys: slice,
    xs: slice,
) -> Float32Array:
    """Replace invalid hydrometeors from adjacent vertical mass levels.

    Args:
        values: Three-dimensional WRF field ordered as ``(level, y, x)``.
        low: Inclusive lower validity limit in the units of ``values``.
        high: Inclusive upper validity limit in the units of ``values``.
        zs: Zero-based slice selecting mass levels.
        ys: Zero-based slice selecting source-grid rows.
        xs: Zero-based slice selecting source-grid columns.

    Returns:
        A float32 ``(level, y, x)`` array over the requested subdomain.
        Invalid values are replaced by the mean of valid levels immediately
        above and below, or by the nearest bound when neither is valid.
    """
    original: Float32Array = np.asarray(values, dtype="f4")
    result: Float32Array = original[zs, ys, xs].copy()
    kk = np.arange(zs.start, zs.stop)
    neighbours: Float32Array = np.stack((
        original[np.maximum(kk - 1, 0), ys, xs],
        original[np.minimum(kk + 1, original.shape[0] - 1), ys, xs],
    ), axis=0)
    good: BoolArray = (neighbours >= low) & (neighbours <= high)
    count = good.sum(axis=0)
    total: Float32Array = np.where(good, neighbours, 0.0).sum(axis=0, dtype="f4")
    fallback: Float32Array = np.where(result < low, low, high).astype("f4")
    replacement = np.divide(total, count, out=fallback, where=count > 0)
    return np.where((result < low) | (result > high), replacement, result).astype("f4")


def _times(ds: nc.Dataset) -> list[str]:
    """Decode all WRF ``Times`` records.

    Args:
        ds: Open WRF NetCDF dataset containing the character variable
            ``Times(time, DateStrLen)``.

    Returns:
        Timestamp strings in source-record order, normally formatted as
        ``YYYY-MM-DD_HH:MM:SS``.

    Raises:
        KeyError: If the dataset has no ``Times`` variable.
    """
    values: NDArray[np.str_] = cast(
        NDArray[np.str_], nc.chartostring(ds.variables["Times"][:])
    )
    return [str(x) for x in np.asarray(values).tolist()]


def _wrf_timestamp(value: str) -> datetime:
    """Parse one WRF timestamp into a naive calendar datetime.

    Args:
        value: WRF timestamp text accepted by
            :func:`pycalmet.datetime_utils.parse_timestamp`.

    Returns:
        A naive :class:`datetime.datetime` containing the calendar fields in
        ``value``.

    Raises:
        ValueError: If ``value`` is not a supported timestamp.
    """
    try: return parse_timestamp(value)
    except ValueError as error: raise ValueError(f"invalid CALWRF timestamp: {value!r}") from error


def _write_time_text(
    variable: NetCDFBytesVariable, index: int, text: str
) -> None:
    """Write one WRF-style timestamp without scalar-string coercion.

    Args:
        variable: NetCDF character variable with dimensions
            ``(time, date_strlen)`` and a date length of 19 bytes.
        index: Zero-based time-record index to update.
        text: Timestamp convertible to exactly 19 ASCII bytes, normally
            ``YYYY-MM-DD_HH:MM:SS``.

    Returns:
        None.

    Raises:
        UnicodeEncodeError: If ``text`` contains non-ASCII characters.
        ValueError: If the encoded timestamp is not exactly 19 bytes.
        IndexError: If ``index`` lies outside ``variable``.
    """
    encoded=str(text).encode('ascii')
    if len(encoded)!=19:
        raise ValueError('CALWRF time_text values must contain 19 ASCII bytes')
    variable[index]=np.frombuffer(encoded,dtype='S1')

def _selection(value: IndexRange, size: int, label: str) -> slice:
    """Convert an inclusive one-based CALWRF range to a Python slice.

    Args:
        value: Inclusive ``(start, end)`` indices, or ``None`` to select the
            complete axis.
        size: Number of elements available on the source axis.
        label: Axis label included in validation errors, such as ``I``.

    Returns:
        A zero-based, stop-exclusive :class:`slice` bounded by ``size``.

    Raises:
        ValueError: If the requested indices are invalid or outside the axis.
    """
    return fortran_inclusive_slice(value,size,label=f'CALWRF {label}')


def destagger_x(a: ArrayLike) -> Float32Array:
    """Average adjacent points along the WRF west-east staggered axis.

    Args:
        a: Numeric array whose last axis has length ``nx + 1``. Other axes are
            preserved.

    Returns:
        A float32 array with the same leading dimensions and last-axis length
        ``nx``. Values are arithmetic means of adjacent staggered points.
    """
    values: Float32Array = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[..., :-1] + values[..., 1:])).astype(
        np.float32, copy=False,
    )


def destagger_y(a: ArrayLike) -> Float32Array:
    """Average adjacent points along the WRF south-north staggered axis.

    Args:
        a: Numeric array whose penultimate axis has length ``ny + 1``. Other
            axes are preserved.

    Returns:
        A float32 array with penultimate-axis length ``ny`` and all other
        dimensions unchanged.
    """
    values: Float32Array = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[..., :-1, :] + values[..., 1:, :])).astype(
        np.float32, copy=False,
    )


def destagger_z(a: ArrayLike) -> Float32Array:
    """Average adjacent points along the leading WRF vertical-stagger axis.

    Args:
        a: Numeric array ordered with a leading staggered vertical axis of
            length ``nz + 1``.

    Returns:
        A float32 mass-level array with leading-axis length ``nz`` and all
        remaining dimensions unchanged.
    """
    values: Float32Array = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[:-1, ...] + values[1:, ...])).astype(
        np.float32, copy=False,
    )


def _geopotential_height(ph: ArrayLike, phb: ArrayLike) -> Float32Array:
    """Convert WRF staggered geopotential to mass-level geometric height.

    Args:
        ph: Perturbation geopotential in m2 s-2, ordered as
            ``(staggered_level, y, x)``.
        phb: Base-state geopotential in m2 s-2 with the same shape as ``ph``.

    Returns:
        Float32 geometric height in metres on mass levels, with one fewer
        element along the leading vertical axis.
    """
    geopotential: Float32Array = np.asarray(ph, dtype=np.float32) + np.asarray(
        phb, dtype=np.float32,
    )
    return destagger_z(geopotential / np.float32(9.81))


def _cone(truelat1: float, truelat2: float) -> float:
    """Calculate the Lambert conformal cone parameter used by WRF.

    Args:
        truelat1: First true latitude in decimal degrees.
        truelat2: Second true latitude in decimal degrees.

    Returns:
        Dimensionless Lambert cone parameter. Nearly equal standard
        parallels use the tangent-cone limit ``sin(abs(truelat1))``.
    """
    first=float(truelat1); second=float(truelat2)
    if abs(first-second)<=0.1:
        return float(np.sin(np.deg2rad(abs(first))))
    numerator=np.log(np.cos(np.deg2rad(first))/np.cos(np.deg2rad(second)))
    denominator=np.log(
        np.tan(np.deg2rad(45.-abs(first)/2.)) /
        np.tan(np.deg2rad(45.-abs(second)/2.)))
    return float(numerator/denominator)


def rotation_angle(
    ds: nc.Dataset,
    lat: ArrayLike,
    lon: ArrayLike,
) -> Float32Array:
    """Calculate the CALWRF grid-to-earth wind rotation angle.

    Args:
        ds: Open WRF NetCDF dataset supplying ``MAP_PROJ``, ``STAND_LON``,
            and, for Lambert grids, ``TRUELAT1`` and ``TRUELAT2`` attributes.
        lat: Mass-point latitude in decimal degrees.
        lon: Mass-point longitude in decimal degrees, broadcast-compatible
            with ``lat``.

    Returns:
        A float32 array in radians with the broadcast shape of ``lat`` and
        ``lon``. WRF projection identifiers greater than or equal to three
        follow CALWRF and return zero rotation.

    Raises:
        ValueError: If ``MAP_PROJ`` is neither 1, 2, nor a value at least 3.
        AttributeError: If an attribute required by the projection is absent.
    """
    map_projection=int(getattr(ds,"MAP_PROJ",0))
    # CALWRF sets OP_ROT=.FALSE. for every WRF projection code >= 3, not
    # only Mercator (3).  This also covers global/rotated lat-lon products
    # using later WRF projection identifiers.
    if map_projection>=3:
        return np.zeros_like(lon, dtype="f4")
    if map_projection not in {1,2}:
        raise ValueError(f"CALWRF does not support WRF MAP_PROJ={map_projection}")
    # Keep coordinate arithmetic in double precision, then cross into the
    # model's FP32 boundary once the angle has been calculated.
    latitude: Float64Array = np.asarray(lat,dtype=np.float64)
    longitude: Float64Array = np.asarray(lon,dtype=np.float64)
    stand_lon=float(getattr(ds,"STAND_LON"))
    cone=1.0 if map_projection==2 else _cone(
        getattr(ds,"TRUELAT1"),getattr(ds,"TRUELAT2"),
    )
    delta=longitude-stand_lon
    delta=np.where(delta>180.0,delta-360.0,delta)
    delta=np.where(delta<-180.0,delta+360.0,delta)
    alpha=np.deg2rad(delta*cone)
    return np.asarray(np.where(latitude<0.0,-alpha,alpha),dtype=np.float32)


def earth_wind(
    u: ArrayLike,
    v: ArrayLike,
    alpha: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Rotate WRF grid-relative wind to earth-relative components.

    Args:
        u: Grid-x wind component in m s-1.
        v: Grid-y wind component in m s-1, broadcast-compatible with ``u``.
        alpha: Grid-to-earth rotation angle in radians, broadcast-compatible
            with the wind arrays.

    Returns:
        A pair ``(eastward, northward)`` of float32 arrays in m s-1 with the
        broadcast input shape.
    """
    u_values: Float32Array = np.asarray(u,dtype=np.float32)
    v_values: Float32Array = np.asarray(v,dtype=np.float32)
    angle: Float32Array = np.asarray(alpha,dtype=np.float32)
    cosine=np.cos(angle)
    sine=np.sin(angle)
    east=v_values*sine+u_values*cosine
    north=v_values*cosine-u_values*sine
    return (np.asarray(east,dtype=np.float32),
            np.asarray(north,dtype=np.float32))


def speed_direction(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Convert earth-relative vector wind to meteorological polar form.

    Args:
        u: Eastward wind component in m s-1.
        v: Northward wind component in m s-1, broadcast-compatible with ``u``.

    Returns:
        A pair ``(speed, direction)`` of float32 arrays. Speed is in m s-1;
        direction is the clockwise direction from which the wind blows in
        degrees in ``(0, 360]``. The historical CALWRF wrapping rule reports
        calm cells as 360 degrees.
    """
    u_values: Float32Array = np.asarray(u,dtype=np.float32)
    v_values: Float32Array = np.asarray(v,dtype=np.float32)
    speed=np.hypot(u_values,v_values).astype(np.float32,copy=False)
    calculated=np.float32(270.0)-np.rad2deg(np.arctan2(v_values,u_values))
    direction=np.where(speed>=np.float32(1.e-9),calculated,np.float32(0.0))
    direction=np.mod(direction,np.float32(360.0))
    direction=np.where(direction<=np.float32(0.0),
                       direction+np.float32(360.0),direction)
    return speed,np.asarray(direction,dtype=np.float32)


def _var(
    dst: nc.Dataset, name: str, dims: tuple[str, ...], units: str,
    compression: int = 1, dtype: NetCDFDType = "f4",
) -> NetCDFNumericVariable:
    """Create a compressed NetCDF variable and assign its physical units.

    Args:
        dst: Open writable NetCDF dataset.
        name: Variable name to create.
        dims: Ordered dimension names already defined in ``dst``.
        units: CF-style unit text stored on the variable.
        compression: NetCDF4 deflate level from 0 through 9. Zero disables
            deflation and shuffling.
        dtype: NetCDF4 or NumPy data-type specification.

    Returns:
        The newly created writable :class:`netCDF4.Variable`.

    Raises:
        RuntimeError: If the variable cannot be created, for example because
            its name already exists or a dimension is missing.
    """
    level=cast(CompressionLevel,compression)
    var=cast("NetCDFNumericVariable",dst.createVariable(
        name,dtype,dims,zlib=level>0,complevel=level,shuffle=level>0))
    var.units = units
    return var


def _mass_coordinate(
    source: nc.Dataset,
    name: str,
    time_index: int = 0,
) -> Float64Array:
    """Read a WRF mass-point coordinate using CALWRF's alias fallback.

    Args:
        source: Open WRF NetCDF dataset.
        name: Primary coordinate variable name, normally ``XLAT`` or
            ``XLONG``. The ``_M`` suffix is tried second.
        time_index: Zero-based source time index.

    Returns:
        A float64 ``(y, x)`` coordinate array in the units declared by WRF,
        normally decimal degrees.

    Raises:
        ValueError: If neither ``name`` nor ``name + '_M'`` is present.
        IndexError: If ``time_index`` is outside the coordinate variable.
    """
    for candidate in (name,f'{name}_M'):
        if candidate in source.variables:
            return np.asarray(source[candidate][time_index],dtype='f8')
    raise ValueError(f'WRF file has neither {name} nor {name}_M')


def _rain_increment(
    current: ArrayLike,
    previous: ArrayLike | None,
) -> Float32Array:
    """Calculate a precipitation accumulation increment across one record.

    Args:
        current: Current cumulative precipitation in millimetres.
        previous: Previous cumulative field in millimetres, or ``None`` for
            the first available record.

    Returns:
        A float32 array in millimetres with the shape of ``current``. Negative
        differences are treated as accumulation-counter resets, in which case
        the current cumulative value is returned.
    """
    values: Float32Array = np.asarray(current,dtype='f4')
    if previous is None:
        return values.copy()
    delta: Float32Array = values-np.asarray(previous,dtype='f4')
    return np.where(delta>=0.,delta,values).astype('f4',copy=False)


def _rain_increments(
    rainnc: ArrayLike,
    rainc: ArrayLike,
    previous: tuple[ArrayLike, ArrayLike] | None,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate reset-aware total and component precipitation increments.

    Args:
        rainnc: Current WRF ``RAINNC`` grid-scale accumulation in millimetres.
        rainc: Current WRF ``RAINC`` convective accumulation in millimetres.
        previous: Pair ``(previous_rainnc, previous_rainc)`` in millimetres,
            or ``None`` when no earlier record is available.

    Returns:
        A tuple ``(total, grid, convective)`` of float32 arrays in millimetres
        with the broadcast input shape.
    """
    nonconvective: Float32Array = np.asarray(rainnc,dtype='f4')
    convective: Float32Array = np.asarray(rainc,dtype='f4')
    previous_nonconvective: ArrayLike | None = (
        None if previous is None else previous[0]
    )
    previous_convective: ArrayLike | None = (
        None if previous is None else previous[1]
    )
    grid=_rain_increment(nonconvective,previous_nonconvective)
    convective_increment=_rain_increment(convective,previous_convective)
    return np.asarray(grid+convective_increment,dtype='f4'),grid,convective_increment


def _surface_variable(
    source: nc.Dataset,
    index: int,
    name: str,
    shape: Shape2D,
    *,
    required: bool = False,
) -> Float32Array:
    """Read one WRF surface field using CALWRF's missing-field policy.

    Args:
        source: Open WRF NetCDF dataset.
        index: Zero-based source time index.
        name: WRF variable name.
        shape: Expected ``(y, x)`` shape used for an absent optional field.
        required: Whether absence of ``name`` is an error.

    Returns:
        A float32 ``(y, x)`` array in the source variable's physical units.
        An absent optional field is represented by zeros.

    Raises:
        ValueError: If a required variable is absent.
        IndexError: If ``index`` is outside the variable's time dimension.
    """
    if name in source.variables:
        return np.asarray(source[name][index],dtype='f4')
    if required:
        raise ValueError(f'WRF file is missing required 2-D variable {name}')
    # GET2D clears X2D before each time record and skips absent optional
    # variables, so their stored field remains exactly zero before QA.
    return np.zeros(shape,dtype='f4')


def _volume_variable(
    source: nc.Dataset,
    index: int,
    name: str,
    shape: Shape3D,
    *,
    required: bool = False,
) -> Float32Array:
    """Read one WRF volume field using CALWRF's missing-field policy.

    Args:
        source: Open WRF NetCDF dataset.
        index: Zero-based source time index.
        name: WRF variable name.
        shape: Expected ``(level, y, x)`` shape for an absent optional field.
        required: Whether absence of ``name`` is an error.

    Returns:
        A float32 ``(level, y, x)`` array in the source variable's physical
        units. An absent optional field is represented by zeros.

    Raises:
        ValueError: If a required variable is absent.
        IndexError: If ``index`` is outside the variable's time dimension.
    """
    if name in source.variables:
        return np.asarray(source[name][index],dtype='f4')
    if required:
        raise ValueError(f'WRF file is missing required 3-D variable {name}')
    return np.zeros(shape,dtype='f4')


def _surface_wind(
    source: nc.Dataset,
    index: int,
    shape: Shape2D,
    alpha: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Read and rotate WRF 10-m wind into CALWRF speed and direction.

    Args:
        source: Open WRF NetCDF dataset.
        index: Zero-based source time index.
        shape: Expected horizontal ``(y, x)`` shape.
        alpha: Grid-to-earth rotation angles in radians with ``shape``.

    Returns:
        A pair ``(speed, direction)`` of float32 ``(y, x)`` arrays. Speed is
        in m s-1 and meteorological direction is in degrees. If ``U10`` is
        absent, both arrays are zero without requiring ``V10``.
    """
    if 'U10' not in source.variables:
        zeros: Float32Array = np.zeros(shape,dtype='f4')
        return zeros.copy(),zeros
    u10=_surface_variable(source,index,'U10',shape)
    v10=_surface_variable(source,index,'V10',shape)
    east,north=earth_wind(u10,v10,alpha)
    speed,direction=speed_direction(east,north)
    return speed,direction


def _pressure_temperature(
    perturbation_pressure: ArrayLike,
    base_pressure: ArrayLike,
    perturbation_theta: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Convert WRF pressure and potential temperature to CALWRF fields.

    Args:
        perturbation_pressure: WRF ``P`` perturbation pressure in pascals.
        base_pressure: WRF ``PB`` base-state pressure in pascals.
        perturbation_theta: WRF ``T`` perturbation potential temperature in
            kelvins relative to 300 K.

    Returns:
        A pair ``(pressure, temperature)`` of float32 arrays with the broadcast
        input shape. Pressure is in hectopascals and temperature is in kelvins.
    """
    pressure: Float32Array = (
        np.asarray(perturbation_pressure,dtype=np.float32)+
        np.asarray(base_pressure,dtype=np.float32)
    )*np.float32(0.01)
    theta: Float32Array = (
        np.asarray(perturbation_theta,dtype=np.float32)+np.float32(300.0)
    )
    exponent=np.float32(287.04/1004.0)
    temperature=theta*np.power(pressure/np.float32(1000.0),exponent)
    return (np.asarray(pressure,dtype=np.float32),
            np.asarray(temperature,dtype=np.float32))


def _select_wrf_times(
    paths: Sequence[Path],
    begin: datetime,
    end: datetime,
) -> tuple[list[WrfRecord], WrfRecord | None]:
    """Select unique hourly WRF records in an inclusive time interval.

    Args:
        paths: WRF NetCDF paths searched in order.
        begin: Inclusive naive start datetime.
        end: Inclusive naive end datetime.

    Returns:
        A pair ``(selected, baseline)``. ``selected`` is chronologically
        ordered and contains ``(path, record_index, timestamp_text)`` tuples,
        keeping the first file for duplicate times. ``baseline`` has the same
        structure for ``begin - 1 hour`` or is ``None``.

    Raises:
        OSError: If a WRF file cannot be opened.
        KeyError: If a file has no ``Times`` variable.
        ValueError: If a source timestamp cannot be parsed.
    """
    selected_by_time: dict[datetime, WrfRecord] = {}
    baseline: WrfRecord | None = None
    baseline_time=begin-timedelta(hours=1)
    dataset: nc.Dataset
    for path in paths:
        with nc.Dataset(path) as dataset:
            for index,text in enumerate(_times(dataset)):
                when=_wrf_timestamp(text)
                canonical=when.strftime('%Y-%m-%d_%H:%M:%S')
                if when==baseline_time and baseline is None:
                    baseline=(path,index,canonical)
                if begin<=when<=end:
                    selected_by_time.setdefault(when,(path,index,canonical))
    selected: list[WrfRecord] = [
        selected_by_time[when] for when in sorted(selected_by_time)
    ]
    return selected,baseline


def _copy_static(
    src: nc.Dataset,
    dst: nc.Dataset,
    ys: slice,
    xs: slice,
    zs: slice,
    compression: int = 1,
) -> None:
    """Copy time-invariant WRF grid fields into a CALWRF NetCDF dataset.

    Args:
        src: Open source WRF NetCDF dataset.
        dst: Open writable CALWRF NetCDF dataset with ``level``, ``y``, and
            ``x`` dimensions already defined.
        ys: Zero-based slice selecting source rows.
        xs: Zero-based slice selecting source columns.
        zs: Zero-based slice selecting source mass levels.
        compression: NetCDF4 deflate level used for created variables.

    Returns:
        None. ``dst`` is modified in place with latitude and longitude in
        degrees, terrain in metres, land-use category, and dimensionless sigma.

    Raises:
        KeyError: If a required WRF grid variable is absent.
        RuntimeError: If an output variable cannot be created or written.
    """
    for name, units in (("terrain", "m"),):
        _var(dst,name,("y","x"),units,compression)
    for name,units in (("latitude","degrees_north"),("longitude","degrees_east")):
        _var(dst,name,("y","x"),units,compression,dtype="f8")
    _var(dst,"landuse",("y","x"),"category",compression,dtype="i2")
    dst["latitude"][:] = _mass_coordinate(src,"XLAT")[ys,xs]
    dst["longitude"][:] = _mass_coordinate(src,"XLONG")[ys,xs]
    dst["terrain"][:] = np.asarray(src["HGT"][0,ys,xs], dtype="f4")
    dst["landuse"][:] = np.asarray(src["LU_INDEX"][0,ys,xs], dtype="i2")
    _var(dst,"sigma",("level",),"1",compression)[:]=np.asarray(src["ZNU"][0,zs],dtype="f4")


def _create_output(
    path: Path,
    src: nc.Dataset,
    nt: int,
    ny: int,
    nx: int,
    nz: int,
    ys: slice,
    xs: slice,
    zs: slice,
    *,
    compression: int = 1,
) -> nc.Dataset:
    """Create and initialize the canonical CALWRF NetCDF product.

    Args:
        path: Destination NetCDF path. Missing parent directories are created.
        src: Open WRF dataset supplying static grid data and metadata.
        nt: Number of hourly output records.
        ny: Number of selected south-north cells.
        nx: Number of selected west-east cells.
        nz: Number of selected mass levels.
        ys: Zero-based slice selecting source rows.
        xs: Zero-based slice selecting source columns.
        zs: Zero-based slice selecting source mass levels.
        compression: NetCDF4 deflate level from 0 through 9.

    Returns:
        An open writable :class:`netCDF4.Dataset`. The caller owns it and must
        close it. Variables use float32 model fields except float64 geographic
        coordinates and int16 land-use categories.

    Raises:
        OSError: If ``path`` cannot be created.
        KeyError: If required WRF dimensions or variables are absent.
        AttributeError: If required WRF grid-spacing metadata is absent.
        RuntimeError: If NetCDF dimensions or variables cannot be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dst: nc.Dataset = nc.Dataset(path, "w", format="NETCDF4")
    dst.createDimension("time", nt)
    dst.createDimension("level", nz)
    dst.createDimension("y", ny)
    dst.createDimension("x", nx)
    dst.createDimension("date_strlen", 19)
    dst.createVariable("time_text", "S1", ("time", "date_strlen"))
    _copy_static(src,dst,ys,xs,zs,compression)
    fields3: dict[str, str] = {
        "pressure": "hPa", "height": "m", "temperature": "K",
        "wind_speed": "m s-1", "wind_direction": "degree",
        "vertical_velocity": "m s-1", "qvapor": "kg kg-1",
        "qcloud": "kg kg-1", "qrain": "kg kg-1", "qice": "kg kg-1",
        "qsnow": "kg kg-1", "qgraupel": "kg kg-1",
    }
    fields2: dict[str, str] = {
        "surface_pressure": "Pa", "precipitation": "mm", "snow_cover": "1",
        "shortwave_down": "W m-2", "longwave_down": "W m-2",
        "temperature_2m": "K", "qvapor_2m": "kg kg-1",
        "wind_speed_10m": "m s-1", "wind_direction_10m": "degree",
        "sst": "K", "skin_temperature": "K", "pbl_height": "m",
        "sensible_heat_flux": "W m-2", "latent_heat_flux": "W m-2",
        "friction_velocity": "m s-1", "convective_precipitation": "mm",
        "grid_precipitation": "mm",
    }
    for name, units in fields3.items():
        _var(dst,name,("time","level","y","x"),units,compression)
    # IOSRF only controls the additional 2D.DAT file.  GET2D still runs and
    # WRTOUT embeds the surface record in 3D.DAT for every grid point.
    for name,units in fields2.items():
        _var(dst,name,("time","y","x"),units,compression)
    dst.source_format = "WRF NetCDF"
    dst.pycalmet_stage = "calwrf"
    dst.wrf_dx_m = np.float64(getattr(src, "DX"))
    dst.wrf_dy_m = np.float64(getattr(src, "DY"))
    dst.source_i_start=xs.start+1; dst.source_i_end=xs.stop
    dst.source_j_start=ys.start+1; dst.source_j_end=ys.stop
    dst.source_k_start=zs.start+1; dst.source_k_end=zs.stop
    dst.source_grid_nx=len(src.dimensions["west_east"])
    dst.source_grid_ny=len(src.dimensions["south_north"])
    dst.source_grid_nz=len(src.dimensions["bottom_top"])
    source_latitude: Float64Array = _mass_coordinate(src,"XLAT")
    source_longitude: Float64Array = _mass_coordinate(src,"XLONG")
    dst.source_latitude_1_1=np.float64(source_latitude[0,0])
    dst.source_longitude_1_1=np.float64(source_longitude[0,0])
    for attr in ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
                 "MOAD_CEN_LAT"):
        if hasattr(src, attr):
            dst.setncattr(attr.lower(), getattr(src, attr))
    for attr in ('MP_PHYSICS','CU_PHYSICS','BL_PBL_PHYSICS','RA_SW_PHYSICS',
                 'SF_SURFACE_PHYSICS','GRID_FDDA','OBS_NUDGE_OPT'):
        if hasattr(src,attr): dst.setncattr(attr.lower(),getattr(src,attr))
    present: list[bool] = [
        name in src.variables for name in ('QCLOUD','QRAIN','QICE','QSNOW','QGRAUP')
    ]
    dst.legacy_cloud_component_count=(5 if present[4] else 4 if present[2] and present[3]
                                      else 2 if present[1] else 1 if present[0] else 1)
    return dst


def run(config: CalwrfConfig) -> Path:
    """Extract an hourly WRF interval into CALWRF meteorological products.

    Args:
        config: Validated CALWRF configuration containing WRF input paths, an
            inclusive hourly interval, optional one-based ``i``, ``j``, and
            ``k`` subsets, compression, and output paths.

    Returns:
        Path to the canonical NetCDF output specified by ``config.output``.

    Raises:
        FileNotFoundError: If any configured WRF input does not exist.
        ValueError: If paths overlap unsafely, the interval or compression is
            invalid, no complete hourly coverage is available, a configured
            subdomain is invalid, or required WRF content is missing.
        OSError: If an input cannot be opened or an output cannot be written.

    Notes:
        The canonical NetCDF output is always written. Depending on ``config``,
        this function also writes legacy M3D/M2D files and a text listing.
        Existing destination files may be replaced.
    """
    active_surface_output: str | Path | None = (
        config.surface_output
        if config.create_surface and config.legacy_output is not None
        else None
    )
    validate_calwrf_paths(
        config.wrf_files, config.output, config.legacy_output,
        active_surface_output, config.listing_output,
    )
    paths: list[Path] = [Path(p) for p in config.wrf_files]
    if not paths:
        raise ValueError("wrf_files is empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    begin,end=config.start,config.end
    if end < begin:
        raise ValueError("end precedes start")
    if not 0<=config.compression<=9:
        raise ValueError('compression must be between zero and nine')

    selected,rain_baseline=_select_wrf_times(paths,begin,end)
    if not selected:
        raise ValueError("no WRF times fall inside requested interval")
    selected_times: list[datetime] = [_wrf_timestamp(item[2]) for item in selected]
    if selected_times[0]!=begin or selected_times[-1]!=end or any(
            right-left!=timedelta(hours=1) for left,right in zip(selected_times[:-1],selected_times[1:])):
        raise ValueError('selected CALWRF output times must cover the requested interval hourly')

    first: nc.Dataset
    with nc.Dataset(selected[0][0]) as first:
        source_ny=len(first.dimensions["south_north"]); source_nx=len(first.dimensions["west_east"])
        source_nz=len(first.dimensions["bottom_top"])
        xs=_selection(config.i_range,source_nx,'I'); ys=_selection(config.j_range,source_ny,'J')
        zs=_selection(config.k_range,source_nz,'K')
        nx=xs.stop-xs.start; ny=ys.stop-ys.start; nz=zs.stop-zs.start
        dst: nc.Dataset = _create_output(
            Path(config.output),first,len(selected),ny,nx,nz,ys,xs,zs,
            compression=config.compression,
        )
        dst.title=config.title
    previous_rain: RainState | None = None
    if rain_baseline is not None:
        baseline_path,baseline_index,_=rain_baseline
        baseline_source: nc.Dataset
        with nc.Dataset(baseline_path) as baseline_source:
            baseline_shape: Shape2D = (
                len(baseline_source.dimensions['south_north']),
                len(baseline_source.dimensions['west_east']),
            )
            rainnc=_surface_variable(baseline_source,baseline_index,'RAINNC',baseline_shape,
                                     required=True)
            rainc=_surface_variable(baseline_source,baseline_index,'RAINC',baseline_shape)
            previous_rain=(rainnc,rainc)
    try:
        for out_index, (path, time_index, text) in enumerate(selected):
            src: nc.Dataset
            with nc.Dataset(path) as src:
                take: Callable[[str], Float32Array] = lambda name: np.asarray(
                    src[name][time_index], dtype="f4",
                )
                crop2: Callable[[Float32Array], Float32Array] = (
                    lambda values: np.asarray(values)[ys,xs]
                )
                crop3: Callable[[Float32Array], Float32Array] = (
                    lambda values: np.asarray(values)[zs,ys,xs]
                )
                lat: Float64Array
                lon: Float64Array
                lat,lon=_mass_coordinate(src,"XLAT",time_index),_mass_coordinate(
                    src,"XLONG",time_index)
                alpha: Float32Array = rotation_angle(src, lat, lon)
                pressure_hpa: Float32Array
                temperature: Float32Array
                pressure_hpa,temperature=_pressure_temperature(take("P"),take("PB"),take("T"))
                height: Float32Array = _geopotential_height(take("PH"), take("PHB"))
                vertical: Float32Array = destagger_z(take("W"))
                u: Float32Array = destagger_x(take("U"))
                v: Float32Array = destagger_y(take("V"))
                ue: Float32Array
                vn: Float32Array
                ue, vn = earth_wind(u, v, alpha[None, ...])
                speed: Float32Array
                direction: Float32Array
                speed, direction = speed_direction(ue, vn)
                surface_shape: Shape2D = alpha.shape
                speed10: Float32Array
                direction10: Float32Array
                speed10,direction10=_surface_wind(
                    src,time_index,surface_shape,alpha)

                _write_time_text(dst["time_text"],out_index,text)
                fields3: dict[str, Float32Array] = {
                    "pressure":pressure_hpa, "temperature":temperature,
                    "wind_speed":speed, "wind_direction":direction,
                    "vertical_velocity":vertical,
                }
                fields3["qvapor"]=_volume_variable(
                    src,time_index,"QVAPOR",pressure_hpa.shape,required=True)
                for source, target in (("QCLOUD", "qcloud"), ("QRAIN", "qrain"),
                                       ("QICE", "qice"),
                                       ("QSNOW", "qsnow"), ("QGRAUP", "qgraupel")):
                    fields3[target]=_volume_variable(
                        src,time_index,source,pressure_hpa.shape)
                dst["pressure"][out_index] = _horizontal_range_check(fields3["pressure"],*_LIMITS_3D["pressure"],ys,xs)[zs]
                dst["height"][out_index] = crop3(height)
                for target in ("temperature","wind_speed","wind_direction","vertical_velocity","qvapor"):
                    dst[target][out_index]=_horizontal_range_check(fields3[target],*_LIMITS_3D[target],ys,xs)[zs]
                for target in ("qcloud","qrain","qice","qsnow","qgraupel"):
                    dst[target][out_index]=_vertical_range_check(fields3[target],*_LIMITS_3D[target],zs,ys,xs)

                rainnc=_surface_variable(src,time_index,'RAINNC',surface_shape,required=True)
                rainc=_surface_variable(src,time_index,'RAINC',surface_shape)
                increment,grid_increment,convective_increment=_rain_increments(
                    rainnc,rainc,previous_rain)
                previous_rain=(rainnc.copy(),rainc.copy())
                surface: dict[str, Float32Array] = {
                    "surface_pressure":_surface_variable(
                        src,time_index,'PSFC',surface_shape,required=True),
                    "precipitation":increment,
                    "grid_precipitation":grid_increment,
                    "convective_precipitation":convective_increment,
                    "snow_cover":_surface_variable(
                        src,time_index,'SNOWC',surface_shape,required=True),
                    "shortwave_down":_surface_variable(src,time_index,'SWDOWN',surface_shape),
                    "longwave_down":_surface_variable(src,time_index,'GLW',surface_shape),
                    "temperature_2m":_surface_variable(src,time_index,'T2',surface_shape),
                    "qvapor_2m":_surface_variable(src,time_index,'Q2',surface_shape),
                    "wind_speed_10m":speed10,"wind_direction_10m":direction10,
                    "sst":_surface_variable(src,time_index,'SST',surface_shape),
                    "skin_temperature":_surface_variable(src,time_index,'TSK',surface_shape),
                    "pbl_height":_surface_variable(src,time_index,'PBLH',surface_shape),
                    "sensible_heat_flux":_surface_variable(src,time_index,'HFX',surface_shape),
                    "latent_heat_flux":_surface_variable(src,time_index,'LH',surface_shape),
                    "friction_velocity":_surface_variable(src,time_index,'UST',surface_shape),
                }
                for target, values in surface.items():
                    if target in _LIMITS_2D:
                        values=_horizontal_range_check(values,*_LIMITS_2D[target],ys,xs)
                    else:
                        values=crop2(values)
                    dst[target][out_index] = values
    finally:
        dst.close()
    if config.legacy_output is not None:
        write_m3d(config.output,config.legacy_output,
                  config.surface_output if config.create_surface else None)
    if config.listing_output is not None:
        write_calwrf_listing(
            config.listing_output,title=config.title,wrf_files=paths,
            requested_start=begin,requested_end=end,output=config.output,
            legacy_output=config.legacy_output,
            surface_output=active_surface_output,written_times=selected_times,
            nx=nx,ny=ny,nz=nz,
        )
    return Path(config.output)
