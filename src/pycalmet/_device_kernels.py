"""JAX implementations of the vectorized stage kernels.

Each function follows the active :mod:`pycalmet.backend` and returns NumPy
arrays. Callers invoke these only after selecting JAX. The formulas mirror
the NumPy kernels; transcendental results can differ in the last bits.
In-place Gauss-Seidel and divergence-minimization sweeps are intentionally
absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backend import asarray, get_namespace, to_numpy


Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]
WaterRange: TypeAlias = tuple[int, int]


def _xp() -> Any:
    """Return the active array namespace.

    Returns:
        ``jax.numpy`` when JAX is selected, otherwise NumPy.
    """
    return get_namespace()


def _f32(value: ArrayLike) -> Float32Array:
    """Copy ``value`` to a host ``float32`` array.

    Args:
        value: Scalar or array from the active namespace.

    Returns:
        A NumPy ``float32`` array.
    """
    return cast(Float32Array, to_numpy(value, dtype=np.float32))


def _f64(value: ArrayLike) -> Float64Array:
    """Copy ``value`` to a host ``float64`` array.

    Args:
        value: Scalar or array from the active namespace.

    Returns:
        A NumPy ``float64`` array.
    """
    return cast(Float64Array, to_numpy(value, dtype=np.float64))


def _i64(value: ArrayLike) -> Int64Array:
    """Copy ``value`` to a host ``int64`` array.

    Args:
        value: Scalar or array from the active namespace.

    Returns:
        A NumPy ``int64`` array.
    """
    return cast(Int64Array, to_numpy(value, dtype=np.int64))


def _promoted(value: ArrayLike) -> Any:
    """Round to model ``float32`` and then widen to ``float64``.

    Args:
        value: Scalar or array-like physical field.

    Returns:
        An active-backend ``float64`` array. This matches NumPy's promotion
        when a ``float32`` array is combined with a Python float and JAX
        64-bit mode is on.
    """
    return asarray(value, dtype=np.float32).astype(_xp().float64)


def destagger_x(values: ArrayLike) -> Float32Array:
    """Average adjacent points on a staggered west-east axis.

    Args:
        values: Array whose last axis is staggered.

    Returns:
        Host ``float32`` array with one fewer element on the last axis.
    """
    data = asarray(values, dtype=np.float32)
    half = _xp().asarray(np.float32(0.5))
    return _f32(half * (data[..., :-1] + data[..., 1:]))


def destagger_y(values: ArrayLike) -> Float32Array:
    """Average adjacent points on a staggered south-north axis.

    Args:
        values: Array whose penultimate axis is staggered.

    Returns:
        Host ``float32`` array with one fewer element on the penultimate axis.
    """
    data = asarray(values, dtype=np.float32)
    half = _xp().asarray(np.float32(0.5))
    return _f32(half * (data[..., :-1, :] + data[..., 1:, :]))


def destagger_z(values: ArrayLike) -> Float32Array:
    """Average adjacent points on a leading staggered vertical axis.

    Args:
        values: Array whose leading axis is staggered.

    Returns:
        Host ``float32`` array with one fewer element on the leading axis.
    """
    data = asarray(values, dtype=np.float32)
    half = _xp().asarray(np.float32(0.5))
    return _f32(half * (data[:-1, ...] + data[1:, ...]))


def geopotential_height(ph: ArrayLike, phb: ArrayLike) -> Float32Array:
    """Convert staggered WRF geopotential to mass-level height.

    Args:
        ph: Perturbation geopotential in m2 s-2.
        phb: Base-state geopotential in m2 s-2.

    Returns:
        Host ``float32`` geometric height in metres on mass levels.
    """
    geopotential = asarray(ph, dtype=np.float32) + asarray(phb, dtype=np.float32)
    gravity = _xp().asarray(np.float32(9.81))
    return destagger_z(geopotential / gravity)


def rotation_angle_field(
    latitude: ArrayLike,
    longitude: ArrayLike,
    stand_lon: float,
    cone: float,
) -> Float32Array:
    """Calculate the CALWRF grid-to-earth rotation angle.

    Args:
        latitude: Mass-point latitude in degrees.
        longitude: Mass-point longitude in degrees.
        stand_lon: WRF standard longitude in degrees.
        cone: Lambert cone parameter. Polar stereographic uses ``1``.

    Returns:
        Host ``float32`` rotation angle in radians, negative in the southern
        hemisphere.
    """
    xp = _xp()
    lat = asarray(latitude, dtype=np.float64)
    lon = asarray(longitude, dtype=np.float64)
    delta = lon - xp.asarray(stand_lon, dtype=xp.float64)
    delta = xp.where(delta > 180.0, delta - 360.0, delta)
    delta = xp.where(delta < -180.0, delta + 360.0, delta)
    alpha = xp.deg2rad(delta * xp.asarray(cone, dtype=xp.float64))
    return _f32(xp.where(lat < 0.0, -alpha, alpha))


def earth_wind(
    u: ArrayLike,
    v: ArrayLike,
    alpha: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Rotate grid-relative wind to earth-relative components.

    Args:
        u: Grid-x wind component in m s-1.
        v: Grid-y wind component in m s-1.
        alpha: Rotation angle in radians.

    Returns:
        Host ``float32`` pair ``(eastward, northward)`` in m s-1.
    """
    xp = _xp()
    u_values = asarray(u, dtype=np.float32)
    v_values = asarray(v, dtype=np.float32)
    angle = asarray(alpha, dtype=np.float32)
    cosine = xp.cos(angle)
    sine = xp.sin(angle)
    east = v_values * sine + u_values * cosine
    north = v_values * cosine - u_values * sine
    return _f32(east), _f32(north)


def speed_direction(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Convert earth-relative wind to CALWRF speed and direction.

    Args:
        u: Eastward wind component in m s-1.
        v: Northward wind component in m s-1.

    Returns:
        Host ``float32`` pair ``(speed, direction)``. Speed is m s-1 and
        direction is degrees in ``(0, 360]``, with calm reported as 360.
    """
    xp = _xp()
    u_values = asarray(u, dtype=np.float32)
    v_values = asarray(v, dtype=np.float32)
    speed = xp.hypot(u_values, v_values)
    zero = xp.asarray(np.float32(0.0))
    full_circle = xp.asarray(np.float32(360.0))
    calculated = xp.asarray(np.float32(270.0)) - xp.rad2deg(
        xp.arctan2(v_values, u_values)
    )
    direction = xp.where(
        speed >= xp.asarray(np.float32(1.0e-9)),
        calculated,
        zero,
    )
    direction = xp.mod(direction, full_circle)
    direction = xp.where(direction <= zero, direction + full_circle, direction)
    return _f32(speed), _f32(direction)


def rain_increment(
    current: ArrayLike,
    previous: ArrayLike | None,
) -> Float32Array:
    """Calculate one reset-aware precipitation accumulation increment.

    Args:
        current: Current cumulative precipitation in millimetres.
        previous: Previous cumulative field, or ``None`` for the first record.

    Returns:
        Host ``float32`` increment in millimetres. A negative difference is
        treated as a counter reset and replaced by ``current``.
    """
    values = asarray(current, dtype=np.float32)
    if previous is None:
        return _f32(values).copy()
    delta = values - asarray(previous, dtype=np.float32)
    return _f32(_xp().where(delta >= 0.0, delta, values))


def pressure_temperature(
    perturbation_pressure: ArrayLike,
    base_pressure: ArrayLike,
    perturbation_theta: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Convert WRF pressure and potential temperature to CALWRF fields.

    Args:
        perturbation_pressure: WRF ``P`` in pascals.
        base_pressure: WRF ``PB`` in pascals.
        perturbation_theta: WRF ``T`` perturbation potential temperature in
            kelvins relative to 300 K.

    Returns:
        Host ``float32`` pair ``(pressure_hpa, temperature_k)``.
    """
    xp = _xp()
    pressure = (
        asarray(perturbation_pressure, dtype=np.float32)
        + asarray(base_pressure, dtype=np.float32)
    ) * xp.asarray(np.float32(0.01))
    theta = asarray(perturbation_theta, dtype=np.float32) + xp.asarray(
        np.float32(300.0)
    )
    exponent = xp.asarray(np.float32(287.04 / 1004.0))
    temperature = theta * xp.power(
        pressure / xp.asarray(np.float32(1000.0)),
        exponent,
    )
    return _f32(pressure), _f32(temperature)


def cell_averages(sums: ArrayLike, counts: ArrayLike) -> Float32Array:
    """Reduce terrain elevation sums to a float32 cell mean.

    Args:
        sums: Elevation sums in metres.
        counts: Sample counts with the same shape as ``sums``.

    Returns:
        Host ``float32`` means in metres. Cells with no samples are ``-999``.
    """
    xp = _xp()
    totals = asarray(sums, dtype=np.float64)
    hits = asarray(counts)
    denominator = xp.asarray(hits, dtype=xp.float64)
    missing = xp.full(tuple(totals.shape), -999.0, dtype=xp.float64)
    return _f32(xp.where(hits > 0, totals / denominator, missing))


def grid_indices(
    x: ArrayLike,
    y: ArrayLike,
    x_origin_km: float,
    y_origin_km: float,
    dx_km: float,
    dy_km: float,
) -> tuple[Int64Array, Int64Array]:
    """Convert projected kilometres to zero-based cell indices.

    Args:
        x: X coordinate in grid kilometres.
        y: Y coordinate in grid kilometres.
        x_origin_km: Easting of the south-west corner of cell ``(0, 0)``.
        y_origin_km: Northing of the south-west corner of cell ``(0, 0)``.
        dx_km: East-west cell size in kilometres.
        dy_km: South-north cell size in kilometres.

    Returns:
        Host ``int64`` pair ``(ix, iy)``. Indices are not clipped.
    """
    xp = _xp()
    x_values = asarray(x, dtype=np.float64)
    y_values = asarray(y, dtype=np.float64)
    ix = xp.floor(
        (x_values - xp.asarray(x_origin_km, dtype=xp.float64))
        / xp.asarray(dx_km, dtype=xp.float64)
    )
    iy = xp.floor(
        (y_values - xp.asarray(y_origin_km, dtype=xp.float64))
        / xp.asarray(dy_km, dtype=xp.float64)
    )
    return _i64(ix), _i64(iy)


def bincount_hits(cell_index: ArrayLike, minlength: int) -> Int64Array:
    """Count samples in flattened target cells.

    Args:
        cell_index: Nonnegative flattened cell indices.
        minlength: Output length. Shorter histograms are padded with zeros.

    Returns:
        Host ``int64`` counts of shape ``(minlength,)``.
    """
    counts = _xp().bincount(asarray(cell_index, dtype=np.int64), minlength=minlength)
    return _i64(counts)


def category_sum(values: ArrayLike, mask: ArrayLike | None = None) -> Float64Array:
    """Sum land-use fractions along the category axis.

    Args:
        values: Array shaped ``(category, ...)``.
        mask: Optional Boolean selector for the category axis. ``None`` sums
            every category.

    Returns:
        Host ``float64`` array over the remaining axes.
    """
    data = asarray(values, dtype=np.float32)
    if mask is None:
        selected = data
    else:
        selected = data[asarray(mask, dtype=bool)]
    return _f64(_xp().sum(selected, axis=0, dtype=_xp().float64))


def water_sum(
    fractions: ArrayLike,
    categories: ArrayLike,
    water_categories: Sequence[int],
) -> Float64Array:
    """Sum fractions whose category is in ``water_categories``.

    Args:
        fractions: Array shaped ``(category, ...)``.
        categories: Category identifier for each first-axis row.
        water_categories: Category identifiers treated as water.

    Returns:
        Host ``float64`` water-fraction array over the spatial axes.
    """
    xp = _xp()
    data = asarray(fractions, dtype=np.float32)
    category_values = asarray(categories)
    water = asarray(np.asarray(tuple(water_categories)))
    return _f64(xp.sum(data[xp.isin(category_values, water)], axis=0, dtype=xp.float64))


def weighted_surface_field(
    fractions: ArrayLike,
    reference: ArrayLike,
    *,
    logarithmic: bool = False,
) -> Float32Array:
    """Calculate a category-fraction weighted surface field.

    Args:
        fractions: Land-use fractions shaped ``(category, y, x)``.
        reference: One property value for each category.
        logarithmic: Use a geometric rather than arithmetic mean.

    Returns:
        Host ``float32`` array shaped ``(y, x)``.
    """
    xp = _xp()
    weights = asarray(fractions, dtype=np.float64)
    values = asarray(reference, dtype=np.float64)
    if logarithmic:
        values = xp.log(values)
    numerator = xp.sum(weights * values[:, None, None], axis=0, dtype=xp.float64)
    denominator = xp.sum(weights, axis=0, dtype=xp.float64)
    result = numerator / denominator
    if logarithmic:
        result = xp.exp(result)
    return _f32(result)


def density_from_pressure_temperature(
    pressure_mb: ArrayLike,
    temperature_k: ArrayLike,
) -> Float32Array:
    """Calculate dry-air density with CALMET's ideal-gas coefficient.

    Args:
        pressure_mb: Pressure in millibars.
        temperature_k: Temperature in kelvins.

    Returns:
        Host ``float32`` density in kg m-3.
    """
    pressure = _promoted(pressure_mb)
    temperature = _promoted(temperature_k)
    return _f32(0.3484321 * pressure / temperature)


def coriolis_parameter(
    latitude_degrees: ArrayLike,
    configured_value: float = 999.0,
) -> Float32Array:
    """Calculate CALMET's absolute Coriolis field with an equatorial floor.

    Args:
        latitude_degrees: Latitude in degrees.
        configured_value: Coriolis parameter in s-1. Values above 998 request
            calculation from latitude.

    Returns:
        Host ``float32`` nonnegative Coriolis parameter in s-1, floored at
        ``2.5e-5``.
    """
    xp = _xp()
    latitude = asarray(latitude_degrees, dtype=np.float64)
    if configured_value > 998.0:
        omega = 2.0 * xp.pi / 86400.0
        value = 2.0 * omega * xp.sin(xp.deg2rad(latitude))
    else:
        value = xp.full_like(latitude, xp.asarray(configured_value, dtype=xp.float64))
    floor = xp.asarray(0.25e-4, dtype=xp.float64)
    return _f32(xp.maximum(xp.abs(value), floor))


def inverse_distance_squared(
    values: ArrayLike,
    point_x: ArrayLike,
    point_y: ArrayLike,
    sample_x: ArrayLike,
    sample_y: ArrayLike,
    *,
    ignore_zero: bool = False,
) -> Float32Array:
    """Interpolate samples with inverse-square weighting.

    Args:
        values: Sample values whose last axis indexes samples.
        point_x: Target x coordinates.
        point_y: Target y coordinates.
        sample_x: One-dimensional sample x coordinates.
        sample_y: One-dimensional sample y coordinates.
        ignore_zero: Exclude sample values below ``1e-6``.

    Returns:
        Host ``float32`` interpolated values. Targets with no retained weight
        are zero.
    """
    xp = _xp()
    samples = asarray(values, dtype=np.float32)
    dx = asarray(sample_x, dtype=np.float64) - asarray(point_x, dtype=np.float64)[
        ..., None
    ]
    dy = asarray(sample_y, dtype=np.float64) - asarray(point_y, dtype=np.float64)[
        ..., None
    ]
    weights = 1.0 / xp.maximum(dx * dx + dy * dy, 0.001)
    if ignore_zero:
        threshold = xp.asarray(np.float32(1.0e-6))
        weights = xp.where(samples >= threshold, weights, 0.0)
    denominator = xp.sum(weights, axis=-1, dtype=xp.float64)
    numerator = xp.sum(samples * weights, axis=-1, dtype=xp.float64)
    result = xp.where(denominator > 1.0e-9, numerator / denominator, 0.0)
    return _f32(result)


def wind_speed_direction(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[Float32Array, Float32Array]:
    """Convert vector wind to CALMET speed and meteorological direction.

    Args:
        u: Eastward component in m s-1.
        v: Northward component in m s-1.

    Returns:
        Host ``float32`` pair ``(speed, direction)``. Calm direction is zero.
    """
    xp = _xp()
    u_values = asarray(u, dtype=np.float32)
    v_values = asarray(v, dtype=np.float32)
    squared = u_values * u_values + v_values * v_values
    speed = xp.sqrt(squared)
    direction = xp.fmod(270.0 - xp.rad2deg(xp.arctan2(v_values, u_values)), 360.0)
    direction = xp.where(
        squared < 1.0e-10,
        0.0,
        xp.where(direction == 0.0, 360.0, direction),
    )
    return _f32(speed), _f32(direction)


def cloud_fraction(relative_humidity_percent: ArrayLike) -> Float32Array:
    """Diagnose cloud fraction with the Teixeira square-root relation.

    Args:
        relative_humidity_percent: Relative humidity in percent.

    Returns:
        Host ``float32`` cloud fraction. Values at or above 99 percent map
        to one.
    """
    xp = _xp()
    rh = _promoted(relative_humidity_percent)
    gap = 1.0 - rh / 100.0
    updated = 0.02 * (-1.0 + xp.sqrt(1.0 + 100.0 * gap)) / gap
    return _f32(xp.where(rh < 99.0, updated, xp.ones_like(rh)))


def solar_sine(
    julian_day: int,
    base_timezone: float,
    latitude: ArrayLike,
    longitude_east: ArrayLike,
) -> Float32Array:
    """Calculate CALMET half-hour solar-elevation sine fields.

    Args:
        julian_day: One-based day of year.
        base_timezone: Hours west of UTC.
        latitude: Latitude in degrees north.
        longitude_east: Longitude in degrees east.

    Returns:
        Host ``float32`` array shaped ``(26, ...)`` with the sine of solar
        elevation from 00:30 before through 00:30 after the modeled day.
    """
    xp = _xp()
    lat = asarray(latitude, dtype=np.float64)
    lon = asarray(longitude_east, dtype=np.float64)
    day = xp.asarray(julian_day, dtype=xp.float64)
    timezone = xp.asarray(base_timezone, dtype=xp.float64)
    d = (day - 1.0) * 0.9856479
    radd = xp.deg2rad(d)
    sd = xp.sin(radd)
    cd = xp.cos(radd)
    rad2d = 2.0 * radd
    s2 = xp.sin(rad2d)
    c2 = xp.cos(rad2d)
    em = 12.0 + 0.12357 * sd - 0.004289 * cd + 0.153809 * s2 + 0.060783 * c2
    sigma = (
        279.9348
        + d
        + 1.914827 * sd
        - 0.079525 * cd
        + 0.019938 * s2
        - 0.00162 * c2
    )
    sincd = 0.39784989 * xp.sin(xp.deg2rad(sigma))
    capd = xp.arcsin(sincd)
    coscd = xp.cos(capd)
    radlat = xp.deg2rad(lat)
    sinlat = xp.sin(radlat)
    coslat = xp.cos(radlat)
    hours = xp.arange(26, dtype=xp.float64) - 0.5 + timezone
    solha = 15.0 * (hours[:, None, None] - em) + lon[None, :, :]
    result = sinlat[None, :, :] * sincd + coslat[None, :, :] * coscd * xp.cos(
        xp.deg2rad(solha)
    )
    return _f32(result)


def heat_flux(
    solar_sine_field: ArrayLike,
    cloud_fraction_field: ArrayLike,
    temperature_k: ArrayLike,
    landuse: ArrayLike,
    albedo: ArrayLike,
    bowen: ArrayLike,
    soil_heat_flux: ArrayLike,
    anthropogenic_heat: ArrayLike,
    water_range: WaterRange = (50, 55),
    *,
    turbidity_coefficient_1: float = 990.0,
    turbidity_coefficient_2: float = -30.0,
    cloudiness_coefficient_1: float = -0.75,
    cloudiness_coefficient_2: float = 3.4,
    net_radiation_coefficient_1: float = 5.31e-13,
    net_radiation_coefficient_2: float = 60.0,
    net_radiation_coefficient_3: float = 0.12,
) -> tuple[Float32Array, Float32Array]:
    """Calculate short-wave radiation and land sensible heat flux.

    Args:
        solar_sine_field: Sine of solar elevation.
        cloud_fraction_field: Cloud fraction from 0 to 1.
        temperature_k: Surface air temperature in kelvins.
        landuse: Integer land-use categories.
        albedo: Surface albedo fraction.
        bowen: Bowen ratio.
        soil_heat_flux: Fraction of net radiation stored as soil heat.
        anthropogenic_heat: Anthropogenic heat flux in W m-2.
        water_range: Inclusive land-use categories treated as water.
        turbidity_coefficient_1: Clear-sky solar coefficient in W m-2.
        turbidity_coefficient_2: Clear-sky solar offset in W m-2.
        cloudiness_coefficient_1: Dimensionless cloud attenuation coefficient.
        cloudiness_coefficient_2: Dimensionless cloud-fraction exponent.
        net_radiation_coefficient_1: Temperature-sixth-power coefficient in
            W m-2 K-6.
        net_radiation_coefficient_2: Cloud radiation coefficient in W m-2.
        net_radiation_coefficient_3: Dimensionless net-radiation denominator
            coefficient.

    Returns:
        Host ``float32`` pair ``(sensible_heat, shortwave)`` in W m-2.
    """
    xp = _xp()
    sun = _promoted(solar_sine_field)
    cloud = _promoted(cloud_fraction_field)
    temp = _promoted(temperature_k)
    albedo_values = _promoted(albedo)
    bowen_values = _promoted(bowen)
    soil = _promoted(soil_heat_flux)
    anthropogenic = _promoted(anthropogenic_heat)
    cloud_power = xp.power(cloud, cloudiness_coefficient_2)
    shortwave = xp.maximum(
        (turbidity_coefficient_1 * sun + turbidity_coefficient_2)
        * (1.0 + cloudiness_coefficient_1 * cloud_power),
        0.0,
    )
    temp2 = temp * temp
    temp3 = temp2 * temp
    temp6 = temp3 * temp3
    net = (
        (1.0 - albedo_values) * shortwave
        + net_radiation_coefficient_1 * temp6
        - 5.67e-8 * (temp2 * temp2)
        + net_radiation_coefficient_2 * cloud
    ) / (net_radiation_coefficient_3 + 1.0)
    sensible = bowen_values * (net * (1.0 - soil) + anthropogenic) / (
        1.0 + bowen_values
    )
    sensible = xp.where(sun <= 0.0, -0.1, sensible)
    categories = asarray(landuse)
    water = (categories >= water_range[0]) & (categories <= water_range[1])
    return _f32(xp.where(water, 0.0, sensible)), _f32(shortwave)


def divergence_and_vertical_velocity(
    u: ArrayLike,
    v: ArrayLike,
    z_faces_m: ArrayLike,
    dx_m: float,
    dy_m: float,
    *,
    obrien: bool = False,
    boundaries: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike] | None = None,
) -> tuple[Float32Array, Float32Array]:
    """Calculate horizontal divergence and integrate vertical velocity.

    Args:
        u: Eastward wind shaped ``(level, y, x)`` in m s-1.
        v: Northward wind shaped ``(level, y, x)`` in m s-1.
        z_faces_m: Layer-face heights shaped ``(level + 1,)`` in metres.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.
        obrien: Apply the O'Brien correction that forces zero vertical
            velocity at the domain top.
        boundaries: Optional ``(u_west, u_east, v_south, v_north)``. Omitted
            boundaries repeat the edge winds.

    Returns:
        Host ``float32`` pair ``(divergence, w)``. Divergence has the wind
        shape in s-1. ``w`` has one extra vertical face and is in m s-1.
    """
    xp = _xp()
    u_values = asarray(u, dtype=np.float64)
    v_values = asarray(v, dtype=np.float64)
    faces = asarray(z_faces_m, dtype=np.float64)
    if boundaries is None:
        u_west, u_east = u_values[:, :, 0], u_values[:, :, -1]
        v_south, v_north = v_values[:, 0, :], v_values[:, -1, :]
    else:
        u_west, u_east, v_south, v_north = (
            asarray(value, dtype=np.float64) for value in boundaries
        )
    padded_u = xp.concatenate(
        (u_west[:, :, None], u_values, u_east[:, :, None]),
        axis=2,
    )
    padded_v = xp.concatenate(
        (v_south[:, None, :], v_values, v_north[:, None, :]),
        axis=1,
    )
    dxi = xp.asarray(1.0, dtype=xp.float64) / (
        xp.asarray(2.0, dtype=xp.float64) * xp.asarray(dx_m, dtype=xp.float64)
    )
    dyi = xp.asarray(1.0, dtype=xp.float64) / (
        xp.asarray(2.0, dtype=xp.float64) * xp.asarray(dy_m, dtype=xp.float64)
    )
    divergence = (padded_u[:, :, 2:] - padded_u[:, :, :-2]) * dxi + (
        padded_v[:, 2:, :] - padded_v[:, :-2, :]
    ) * dyi
    levels: list[Any] = [xp.zeros(tuple(u_values.shape[1:]), dtype=xp.float64)]
    level_count = int(u_values.shape[0])
    for level in range(level_count):
        levels.append(
            -((faces[level + 1] - faces[level]) * divergence[level]) + levels[-1]
        )
    vertical = xp.stack(levels, axis=0)
    if obrien:
        top = vertical[-1]
        corrected: list[Any] = [vertical[0]]
        for level in range(level_count):
            ratio = faces[level + 1] / faces[-1]
            corrected.append(vertical[level + 1] - ratio * top)
        vertical = xp.stack(corrected, axis=0)
    return _f32(divergence), _f32(vertical)
