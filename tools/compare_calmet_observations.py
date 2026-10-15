#!/usr/bin/env python3
"""Compare a PyCALMET NetCDF result with a Fortran CALMET.DAT result.

The comparison aligns each NetCDF timestamp with the corresponding explicit
CALMET.DAT interval and reports ``Python - Fortran`` statistics.  It uses the
runtime CALMET.DAT reader so record layout, integer fields, and interval times
are validated in the same way as the library's initial-guess input path.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import netCDF4
import numpy as np
import numpy.typing as npt

from pycalmet.calmet_solver import target_latitude_longitude
from pycalmet.config import GridConfig
from pycalmet.datums import normalize_datum
from pycalmet.igf_data import FieldArray, InitialGuessCalmetData


FloatArray: TypeAlias = npt.NDArray[np.float64]
TimeMode: TypeAlias = Literal["auto", "begin", "end"]
LegacyExtractor: TypeAlias = Callable[[Mapping[str, FieldArray], int], npt.ArrayLike]
NetCDFExtractor: TypeAlias = Callable[[Any, int, int], npt.ArrayLike]


@dataclass(frozen=True)
class FieldSpec:
    """Describe one comparable time-dependent output field.

    Attributes:
        name: Human-readable row name used in the report.
        netcdf_name: Variable name in the PyCALMET NetCDF output.
        legacy: Extractor for records in one Fortran output period.
        netcdf: Extractor for one NetCDF time record.
        required: Whether absence in either output is an error.
        integer: Whether to report the exact categorical mismatch fraction.
        missing_absolute_at_least: Optional absolute sentinel threshold whose
            values are treated as missing in both outputs.
    """

    name: str
    netcdf_name: str
    legacy: LegacyExtractor
    netcdf: NetCDFExtractor
    required: bool = True
    integer: bool = False
    missing_absolute_at_least: float | None = None


@dataclass(frozen=True)
class StaticFieldSpec:
    """Describe one static field embedded in both CALMET outputs.

    Attributes:
        name: Human-readable row name used in the report.
        netcdf_name: Variable name in the PyCALMET NetCDF output.
        legacy_attribute: Attribute name on the decoded CALMET.DAT object.
        integer: Whether to report the exact categorical mismatch fraction.
    """

    name: str
    netcdf_name: str
    legacy_attribute: str
    integer: bool = False


@dataclass(frozen=True)
class Statistics:
    """Aggregate paired-grid statistics for one output field.

    Attributes:
        count: Number of cells with finite values in both outputs.
        total_count: Number of compared cells, including missing values.
        netcdf_missing_count: Missing-cell count in the Python output.
        fortran_missing_count: Missing-cell count in the Fortran output.
        missing_mask_disagreement_count: Cells missing in exactly one output.
        missing_mask_disagreement_fraction: Mask disagreement divided by all
            compared cells.
        bias: Mean paired difference, defined as Python minus Fortran.
        rmse: Root-mean-square paired difference.
        p95_absolute: 95th percentile of paired absolute differences.
        maximum_absolute: Maximum paired absolute difference.
        mismatch_fraction: Exact mismatch fraction for categorical fields, or
            ``None`` for continuous fields.
    """

    count: int
    total_count: int
    netcdf_missing_count: int
    fortran_missing_count: int
    missing_mask_disagreement_count: int
    missing_mask_disagreement_fraction: float
    bias: float
    rmse: float
    p95_absolute: float
    maximum_absolute: float
    mismatch_fraction: float | None


@dataclass(frozen=True)
class ComparisonSample:
    """Hold differences and missing-mask counts for one comparison sample.

    Attributes:
        differences: Finite paired differences, defined as Python minus
            Fortran.
        total_count: Number of source cells before finite pairing.
        netcdf_missing_count: Missing-cell count in the Python sample.
        fortran_missing_count: Missing-cell count in the Fortran sample.
        missing_mask_disagreement_count: Cells missing in exactly one sample.
    """

    differences: FloatArray
    total_count: int
    netcdf_missing_count: int
    fortran_missing_count: int
    missing_mask_disagreement_count: int


def _legacy_levels(
    fields: Mapping[str, FieldArray],
    levels: int,
    prefix: str,
) -> npt.NDArray[np.generic]:
    """Stack one numbered CALMET.DAT field over model levels.

    Args:
        fields: Legacy records for one output period.
        levels: Number of CALMET cell-centre levels expected.
        prefix: Fixed record-label prefix, such as ``U-LEV`` or ``T-LEV``.

    Returns:
        Array ordered as ``(level, y, x)``.

    Raises:
        KeyError: If any numbered level record is absent.
    """
    labels: tuple[str, ...] = tuple(f"{prefix}{level:3d}" for level in range(1, levels + 1))
    missing: tuple[str, ...] = tuple(label for label in labels if label not in fields)
    if missing:
        raise KeyError(", ".join(missing))
    return np.stack([np.asarray(fields[label]) for label in labels])


def _legacy_level_field(prefix: str) -> LegacyExtractor:
    """Create an extractor for a numbered three-dimensional legacy field.

    Args:
        prefix: Fixed CALMET.DAT record-label prefix.

    Returns:
        Callable that stacks all numbered records for a period.
    """
    return lambda fields, levels: _legacy_levels(fields, levels, prefix)


def _legacy_field(label: str) -> LegacyExtractor:
    """Create an extractor for one two-dimensional legacy field.

    Args:
        label: Exact CALMET.DAT record label.

    Returns:
        Callable that retrieves ``label`` from a period's record mapping.
    """
    def extract(fields: Mapping[str, FieldArray], levels: int) -> npt.ArrayLike:
        """Extract the configured two-dimensional legacy field.

        Args:
            fields: Legacy records for one output period.
            levels: Unused model-level count accepted by the common extractor
                interface.

        Returns:
            Array-like field stored under the configured record label.

        Raises:
            KeyError: If the configured record label is absent.
        """
        del levels
        return fields[label]
    return extract


def _netcdf_record(
    variable: netCDF4.Variable[Any], index: int, levels: int
) -> npt.ArrayLike:
    """Read one full time record from a NetCDF variable.

    Args:
        variable: NetCDF output variable whose leading dimension is time.
        index: Zero-based time-record index.
        levels: Unused level count accepted by the common extractor interface.

    Returns:
        Array-like view of the selected record.
    """
    del levels
    return cast(npt.ArrayLike, variable[index])


def _netcdf_top_faces(
    variable: netCDF4.Variable[Any],
    index: int,
    levels: int,
) -> npt.ArrayLike:
    """Read vertical velocity at the top face of every CALMET layer.

    CALMET.DAT writes ``WFACE 1..NZ`` for layer-top faces. NetCDF also stores the
    lower domain boundary at face index zero, which this extractor omits.

    Args:
        variable: NetCDF vertical-velocity variable ordered by time and face.
        index: Zero-based time-record index.
        levels: Number of CALMET layers and therefore top faces to return.

    Returns:
        Array-like record ordered as ``(level_top_face, y, x)``.
    """
    return cast(npt.ArrayLike, variable[index, 1 : levels + 1])


FIELD_SPECS: Final[tuple[FieldSpec, ...]] = (
    FieldSpec("wind_u", "wind_u", _legacy_level_field("U-LEV"), _netcdf_record),
    FieldSpec("wind_v", "wind_v", _legacy_level_field("V-LEV"), _netcdf_record),
    FieldSpec(
        "vertical_velocity_top_faces",
        "vertical_velocity",
        _legacy_level_field("WFACE"),
        _netcdf_top_faces,
        required=False,
    ),
    FieldSpec(
        "temperature", "temperature", _legacy_level_field("T-LEV"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "pgt_stability_class", "pgt_stability_class", _legacy_field("IPGT"),
        _netcdf_record, required=False, integer=True,
    ),
    FieldSpec(
        "friction_velocity", "friction_velocity", _legacy_field("USTAR"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "mixing_height", "mixing_height", _legacy_field("ZI"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "monin_obukhov_length", "monin_obukhov_length", _legacy_field("EL"),
        _netcdf_record, required=False, missing_absolute_at_least=9.0e9,
    ),
    FieldSpec(
        "convective_velocity_scale", "convective_velocity_scale",
        _legacy_field("WSTAR"), _netcdf_record, required=False,
    ),
    FieldSpec(
        "precipitation_rate", "precipitation_rate", _legacy_field("RMM"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "surface_temperature", "surface_temperature", _legacy_field("TEMPK"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "surface_density", "surface_density", _legacy_field("RHO"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "shortwave_flux", "shortwave_flux", _legacy_field("QSW"),
        _netcdf_record, required=False,
    ),
    FieldSpec(
        "surface_relative_humidity", "surface_relative_humidity",
        _legacy_field("IRH"), _netcdf_record, required=False,
    ),
    FieldSpec(
        "precipitation_code", "precipitation_code", _legacy_field("IPCODE"),
        _netcdf_record, required=False, integer=True,
    ),
)

STATIC_FIELD_SPECS: Final[tuple[StaticFieldSpec, ...]] = (
    StaticFieldSpec("terrain_height", "terrain_height", "terrain_m"),
    StaticFieldSpec("landuse", "landuse", "landuse", integer=True),
)


def _as_datetime(value: object) -> datetime:
    """Convert a decoded NetCDF timestamp to ``datetime``.

    Args:
        value: Python ``datetime`` or cftime-like object exposing year through
            second and, optionally, microsecond fields.

    Returns:
        Timezone-naive datetime with the same calendar fields.

    Raises:
        TypeError: If ``value`` lacks the required calendar fields.
        ValueError: If its fields cannot form a valid Python datetime.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    required: tuple[str, ...] = ("year", "month", "day", "hour", "minute", "second")
    if not all(hasattr(value, name) for name in required):
        raise TypeError(f"unsupported NetCDF timestamp {value!r}")
    microsecond: int = int(getattr(value, "microsecond", 0))
    parts: tuple[int, ...] = tuple(int(getattr(value, name)) for name in required)
    return datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], microsecond)


def _netcdf_times(dataset: netCDF4.Dataset) -> tuple[datetime, ...]:
    """Decode and validate the NetCDF time coordinate.

    Args:
        dataset: Open PyCALMET NetCDF output dataset.

    Returns:
        Unique timezone-naive datetimes in file order.

    Raises:
        TypeError: If a decoded timestamp is not datetime-compatible.
        ValueError: If the coordinate or its units are absent, decoding fails to
            produce valid calendar values, or timestamps are duplicated.
    """
    if "time" not in dataset.variables:
        raise ValueError("Python NetCDF output has no time coordinate")
    variable: netCDF4.Variable[Any] = dataset.variables["time"]
    units: str | None = getattr(variable, "units", None)
    if not units:
        raise ValueError("Python NetCDF time coordinate has no units")
    calendar: str = str(getattr(variable, "calendar", "standard"))
    decoded: object = netCDF4.num2date(
        variable[:], units=units, calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    flattened: npt.NDArray[np.object_] = np.asarray(decoded, dtype=object).reshape(-1)
    times: tuple[datetime, ...] = tuple(_as_datetime(value) for value in flattened)
    if len(times) != len(set(times)):
        raise ValueError("Python NetCDF output contains duplicate timestamps")
    return times


def _choose_time_mode(
    requested: TimeMode,
    netcdf_times: Sequence[datetime],
    beginnings: Sequence[datetime],
    endings: Sequence[datetime],
) -> Literal["begin", "end"]:
    """Select CALMET interval-beginning or interval-ending alignment.

    Args:
        requested: Explicit alignment mode, or ``auto`` for exact set detection.
        netcdf_times: Decoded timestamps from the PyCALMET output.
        beginnings: Interval beginning times from CALMET.DAT.
        endings: Interval ending times from CALMET.DAT.

    Returns:
        Resolved ``begin`` or ``end`` alignment mode.

    Raises:
        ValueError: If automatic detection finds no exact match or an explicit
            mode leaves missing or extra timestamps.
    """
    netcdf_set: set[datetime] = set(netcdf_times)
    beginning_set: set[datetime] = set(beginnings)
    ending_set: set[datetime] = set(endings)
    if requested == "auto":
        if netcdf_set == beginning_set:
            return "begin"
        if netcdf_set == ending_set:
            return "end"
        raise ValueError(
            "NetCDF timestamps match neither CALMET.DAT interval beginnings "
            "nor interval endings"
        )
    expected: set[datetime] = beginning_set if requested == "begin" else ending_set
    missing: list[datetime] = sorted(expected - netcdf_set)
    extra: list[datetime] = sorted(netcdf_set - expected)
    if missing or extra:
        missing_text: str = ", ".join(value.isoformat(sep=" ") for value in missing[:4])
        extra_text: str = ", ".join(value.isoformat(sep=" ") for value in extra[:4])
        raise ValueError(
            f"NetCDF/CALMET.DAT timestamp mismatch for {requested} alignment; "
            f"missing=[{missing_text}], extra=[{extra_text}]"
        )
    return requested


def _dimension_size(dataset: netCDF4.Dataset, name: str) -> int:
    """Return one required NetCDF dimension size.

    Args:
        dataset: Open PyCALMET output dataset.
        name: Required dimension name.

    Returns:
        Current dimension size.

    Raises:
        ValueError: If the named dimension is absent.
    """
    if name not in dataset.dimensions:
        raise ValueError(f"Python NetCDF output has no {name} dimension")
    return int(dataset.dimensions[name].size)


def _linear_values_km(
    variable: netCDF4.Variable[Any],
    *,
    default_units: Literal["m", "km"],
) -> FloatArray:
    """Read a finite linear coordinate and normalize it to kilometres.

    Args:
        variable: NetCDF coordinate variable.
        default_units: Units assumed when the variable has no ``units``
            attribute.

    Returns:
        Float64 coordinate values in kilometres.

    Raises:
        ValueError: If units are unsupported or values are missing/nonfinite.
    """
    raw_units: str = str(getattr(variable, "units", default_units))
    units: str = raw_units.strip().lower().replace(" ", "_")
    metres: frozenset[str] = frozenset(
        {"m", "meter", "meters", "metre", "metres"}
    )
    kilometres: frozenset[str] = frozenset(
        {"km", "kilometer", "kilometers", "kilometre", "kilometres"}
    )
    if units not in metres | kilometres:
        raise ValueError(
            f"Python NetCDF {variable.name} coordinate has unsupported units "
            f"{raw_units!r}"
        )
    values: FloatArray = _float_values(variable[:])
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Python NetCDF {variable.name} coordinate contains missing or "
            "nonfinite values"
        )
    if units in metres:
        values = values / 1000.0
    return np.asarray(values, dtype=np.float64)


def _require_close(
    name: str,
    actual: npt.ArrayLike,
    expected: npt.ArrayLike,
    *,
    rtol: float,
    atol: float,
    longitude: bool = False,
) -> None:
    """Require equally shaped, finite coordinate arrays within tolerance.

    Args:
        name: Coordinate label used in validation errors.
        actual: Values read from the NetCDF output.
        expected: Values derived from CALMET.DAT metadata.
        rtol: Relative comparison tolerance.
        atol: Absolute comparison tolerance.
        longitude: Compare angular differences modulo 360 degrees.

    Raises:
        ValueError: If shape, finiteness, or numeric values differ.
    """
    actual_values: FloatArray = _float_values(actual)
    expected_values: FloatArray = _float_values(expected)
    if actual_values.shape != expected_values.shape:
        raise ValueError(
            f"Python NetCDF {name} coordinate shape {actual_values.shape} differs "
            f"from CALMET.DAT {expected_values.shape}"
        )
    if not np.all(np.isfinite(actual_values)):
        raise ValueError(
            f"Python NetCDF {name} coordinate contains missing or nonfinite values"
        )
    if longitude:
        difference: FloatArray = np.asarray(
            (actual_values - expected_values + 180.0) % 360.0 - 180.0,
            dtype=np.float64,
        )
        close: npt.NDArray[np.bool_] = np.isclose(
            difference, 0.0, rtol=0.0, atol=atol
        )
        maximum: float = float(np.max(np.abs(difference), initial=0.0))
    else:
        close = np.isclose(actual_values, expected_values, rtol=rtol, atol=atol)
        maximum = float(
            np.max(np.abs(actual_values - expected_values), initial=0.0)
        )
    if not np.all(close):
        raise ValueError(
            f"Python NetCDF {name} coordinate differs from CALMET.DAT "
            f"(maximum absolute difference {maximum:.7g})"
        )


def _validate_projection_metadata(
    dataset: netCDF4.Dataset,
    grid: GridConfig,
) -> None:
    """Validate every recognized projection attribute present in NetCDF.

    Args:
        dataset: Open PyCALMET output dataset.
        grid: Grid metadata decoded from CALMET.DAT.

    Raises:
        ValueError: If a present projection attribute is incompatible.
    """
    attributes: set[str] = set(dataset.ncattrs())
    text_expected: dict[str, str] = {"projection": grid.projection}
    if grid.projection.upper() == "UTM":
        text_expected["hemisphere"] = grid.hemisphere
    for name, expected in text_expected.items():
        if name not in attributes:
            continue
        actual: str = str(getattr(dataset, name)).strip().upper()
        if actual != expected.strip().upper():
            raise ValueError(
                f"Python NetCDF projection metadata {name}={actual!r} differs "
                f"from CALMET.DAT {expected!r}"
            )
    if "datum" in attributes:
        actual_datum: str = str(getattr(dataset, "datum"))
        if normalize_datum(actual_datum) != normalize_datum(grid.datum):
            raise ValueError(
                f"Python NetCDF projection metadata datum={actual_datum!r} "
                f"differs from CALMET.DAT {grid.datum!r}"
            )

    numeric_expected: dict[str, float | int] = {
        "grid_nx": grid.nx,
        "grid_ny": grid.ny,
        "grid_dx_km": grid.dx_km,
        "grid_dy_km": grid.dy_km,
        "nx": grid.nx,
        "ny": grid.ny,
        "dx_km": grid.dx_km,
        "dy_km": grid.dy_km,
        "x_origin_km": grid.x_origin_km,
        "y_origin_km": grid.y_origin_km,
    }
    projection: str = grid.projection.upper()
    if projection == "UTM":
        numeric_expected["utm_zone"] = grid.utm_zone
    if projection in {"TTM", "LCC", "LAZA"}:
        numeric_expected.update(
            false_easting_km=grid.false_easting_km,
            false_northing_km=grid.false_northing_km,
        )
    if projection == "TTM":
        numeric_expected["ttm_scale"] = grid.ttm_scale
    if projection in {"TTM", "LCC", "PS", "EM", "LAZA"}:
        numeric_expected.update(
            latitude_origin=grid.lat0,
            longitude_origin=grid.lon0,
            lat0=grid.lat0,
            lon0=grid.lon0,
        )
    if projection in {"LCC", "PS"}:
        numeric_expected.update(standard_parallel_1=grid.lat1, lat1=grid.lat1)
    if projection == "LCC":
        numeric_expected.update(standard_parallel_2=grid.lat2, lat2=grid.lat2)
    integer_names: frozenset[str] = frozenset(
        {"grid_nx", "grid_ny", "nx", "ny", "utm_zone"}
    )
    for name, expected_value in numeric_expected.items():
        if name not in attributes:
            continue
        raw_actual: Any = getattr(dataset, name)
        try:
            if name in integer_names:
                equal: bool = int(raw_actual) == int(expected_value)
            elif name in {"longitude_origin", "lon0"}:
                delta: float = (
                    (float(raw_actual) - float(expected_value) + 180.0) % 360.0
                    - 180.0
                )
                equal = bool(np.isclose(delta, 0.0, rtol=0.0, atol=1.0e-5))
            else:
                equal = bool(
                    np.isclose(
                        float(raw_actual), float(expected_value),
                        rtol=1.0e-5, atol=1.0e-6,
                    )
                )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Python NetCDF projection metadata {name}={raw_actual!r} is invalid"
            ) from error
        if not equal:
            raise ValueError(
                f"Python NetCDF projection metadata {name}={raw_actual!r} "
                f"differs from CALMET.DAT {expected_value!r}"
            )


def _validate_horizontal_coordinates(
    dataset: netCDF4.Dataset,
    grid: GridConfig,
) -> None:
    """Validate projected or geographic NetCDF cell-centre coordinates.

    Args:
        dataset: Open PyCALMET output dataset.
        grid: Grid decoded from CALMET.DAT.

    Raises:
        ValueError: If no complete coordinate pair exists or values differ.
    """
    variables: Mapping[str, netCDF4.Variable[Any]] = dataset.variables
    has_x: bool = "x" in variables
    has_y: bool = "y" in variables
    has_longitude: bool = "longitude" in variables
    has_latitude: bool = "latitude" in variables
    if has_x != has_y:
        missing: str = "y" if has_x else "x"
        raise ValueError(f"Python NetCDF output has no {missing} coordinate")
    if has_longitude != has_latitude:
        missing = "latitude" if has_longitude else "longitude"
        raise ValueError(f"Python NetCDF output has no {missing} coordinate")
    if not ((has_x and has_y) or (has_longitude and has_latitude)):
        raise ValueError(
            "Python NetCDF output has neither x/y nor longitude/latitude coordinates"
        )

    expected_x: FloatArray = np.asarray(
        grid.x_origin_km + (np.arange(grid.nx, dtype=np.float64) + 0.5) * grid.dx_km
    )
    expected_y: FloatArray = np.asarray(
        grid.y_origin_km + (np.arange(grid.ny, dtype=np.float64) + 0.5) * grid.dy_km
    )
    if has_x and has_y:
        x_values: FloatArray = _linear_values_km(variables["x"], default_units="km")
        y_values: FloatArray = _linear_values_km(variables["y"], default_units="km")
        if x_values.ndim == y_values.ndim == 1:
            _require_close("x", x_values, expected_x, rtol=1.0e-6, atol=1.0e-3)
            _require_close("y", y_values, expected_y, rtol=1.0e-6, atol=1.0e-3)
        elif x_values.shape == y_values.shape == (grid.ny, grid.nx):
            expected_x_2d, expected_y_2d = np.meshgrid(expected_x, expected_y)
            _require_close(
                "x", x_values, expected_x_2d, rtol=1.0e-6, atol=1.0e-3
            )
            _require_close(
                "y", y_values, expected_y_2d, rtol=1.0e-6, atol=1.0e-3
            )
        else:
            raise ValueError(
                "Python NetCDF x/y coordinates must be matching (x)/(y) axes "
                f"or (y, x) grids; found {x_values.shape} and {y_values.shape}"
            )

    if has_longitude and has_latitude:
        expected_latitude, expected_longitude = target_latitude_longitude(grid)
        latitude: FloatArray = _float_values(variables["latitude"][:])
        longitude: FloatArray = _float_values(variables["longitude"][:])
        _require_close(
            "latitude", latitude, expected_latitude, rtol=0.0, atol=2.0e-5
        )
        _require_close(
            "longitude", longitude, expected_longitude,
            rtol=0.0, atol=2.0e-5, longitude=True,
        )


def _validate_grid_and_vertical_coordinates(
    dataset: netCDF4.Dataset,
    legacy: InitialGuessCalmetData,
) -> None:
    """Validate horizontal grid, projection, and vertical coordinates.

    Args:
        dataset: Open PyCALMET output dataset.
        legacy: Decoded Fortran CALMET.DAT output.

    Raises:
        ValueError: If either output describes a different model grid.
    """
    grid: GridConfig = legacy.grid
    actual_horizontal: tuple[int, int] = (
        _dimension_size(dataset, "y"),
        _dimension_size(dataset, "x"),
    )
    expected_horizontal: tuple[int, int] = (grid.ny, grid.nx)
    if actual_horizontal != expected_horizontal:
        raise ValueError(
            f"horizontal grid shape differs: NetCDF {actual_horizontal}, "
            f"CALMET.DAT {expected_horizontal}"
        )
    faces: FloatArray = np.asarray(legacy.z_faces_m, dtype=np.float64)
    expected_sizes: dict[str, int] = {
        "level": len(faces) - 1,
        "face": len(faces),
    }
    for name, expected_size in expected_sizes.items():
        actual_size: int = _dimension_size(dataset, name)
        if actual_size != expected_size:
            raise ValueError(
                f"vertical {name} count differs: NetCDF {actual_size}, "
                f"CALMET.DAT {expected_size}"
            )
    _validate_projection_metadata(dataset, grid)
    _validate_horizontal_coordinates(dataset, grid)

    for name, expected in (
        ("face_height", faces),
        ("level_height", 0.5 * (faces[:-1] + faces[1:])),
    ):
        if name not in dataset.variables:
            raise ValueError(f"Python NetCDF output has no {name} coordinate")
        variable: netCDF4.Variable[Any] = dataset.variables[name]
        actual: FloatArray = _linear_values_km(variable, default_units="m") * 1000.0
        _require_close(name, actual, expected, rtol=1.0e-6, atol=1.0e-3)


def _float_values(value: npt.ArrayLike) -> FloatArray:
    """Convert numeric input to float64 and preserve missingness as NaN.

    Args:
        value: NumPy-compatible numeric or masked array.

    Returns:
        Float64 array with masked elements filled by ``NaN``.
    """
    array: npt.NDArray[np.generic] = np.asanyarray(value)
    if np.ma.isMaskedArray(array):
        masked: np.ma.MaskedArray[Any, Any] = np.ma.asarray(array)
        values: FloatArray = np.asarray(np.ma.getdata(masked), dtype=np.float64)
        mask: npt.NDArray[np.bool_] = np.ma.getmaskarray(masked)
        if np.any(mask):
            values = values.copy()
            values[mask] = np.nan
        return values
    return np.asarray(array, dtype=np.float64)


def _field_values(value: npt.ArrayLike, spec: FieldSpec) -> FloatArray:
    """Normalize one field and apply its documented sentinel convention.

    Args:
        value: Numeric NetCDF or CALMET.DAT field values.
        spec: Field description, including any field-specific sentinel limit.

    Returns:
        Float64 values with masked elements and model sentinel states replaced
        by ``NaN``. CALMET uses approximately ``9.9e9 m`` for an effectively
        infinite/undefined Monin-Obukhov length; treating it as a physical
        number would dominate the comparison statistics.
    """
    values: FloatArray=_float_values(value)
    threshold: float | None=spec.missing_absolute_at_least
    if threshold is not None:
        sentinel: npt.NDArray[np.bool_]=np.abs(values)>=threshold
        if np.any(sentinel):
            values=values.copy()
            values[sentinel]=np.nan
    return values


def _comparison_sample(
    python_values: FloatArray,
    legacy_values: FloatArray,
) -> ComparisonSample:
    """Pair finite Python and Fortran values for one record.

    Args:
        python_values: PyCALMET values already converted to float64.
        legacy_values: Shape-compatible CALMET.DAT values in the same units.

    Returns:
        Paired ``Python - Fortran`` differences and directional missing-mask
        counts for the whole record.
    """
    python_missing: npt.NDArray[np.bool_] = ~np.isfinite(python_values)
    legacy_missing: npt.NDArray[np.bool_] = ~np.isfinite(legacy_values)
    paired: npt.NDArray[np.bool_] = ~(python_missing | legacy_missing)
    return ComparisonSample(
        differences=np.asarray(
            python_values[paired] - legacy_values[paired], dtype=np.float64
        ),
        total_count=int(python_values.size),
        netcdf_missing_count=int(np.count_nonzero(python_missing)),
        fortran_missing_count=int(np.count_nonzero(legacy_missing)),
        missing_mask_disagreement_count=int(
            np.count_nonzero(python_missing != legacy_missing)
        ),
    )


def _statistics(
    samples: Sequence[ComparisonSample],
    *,
    integer: bool,
) -> Statistics:
    """Summarize paired differences across all time records of a field.

    Args:
        samples: Per-record differences and missing-mask counts.
        integer: Also calculate the fraction of paired values that differ exactly.

    Returns:
        Counts, missing-mask agreement, bias, RMSE, 95th-percentile absolute
        difference, maximum absolute difference, and optional integer mismatch.

    Raises:
        ValueError: If no record samples are supplied.
    """
    if not samples:
        raise ValueError("cannot summarize an empty field")
    values: FloatArray = np.concatenate(
        [sample.differences.reshape(-1) for sample in samples]
    )
    total_count: int = sum(sample.total_count for sample in samples)
    netcdf_missing_count: int = sum(
        sample.netcdf_missing_count for sample in samples
    )
    fortran_missing_count: int = sum(
        sample.fortran_missing_count for sample in samples
    )
    disagreement_count: int = sum(
        sample.missing_mask_disagreement_count for sample in samples
    )
    if values.size:
        absolute: FloatArray = np.abs(values)
        bias: float = float(np.mean(values))
        rmse: float = float(np.sqrt(np.mean(np.square(values))))
        p95_absolute: float = float(np.percentile(absolute, 95.0))
        maximum_absolute: float = float(np.max(absolute))
        mismatch: float | None = (
            float(np.count_nonzero(values) / values.size) if integer else None
        )
    else:
        bias = rmse = p95_absolute = maximum_absolute = float("nan")
        mismatch = float("nan") if integer else None
    return Statistics(
        count=int(values.size),
        total_count=total_count,
        netcdf_missing_count=netcdf_missing_count,
        fortran_missing_count=fortran_missing_count,
        missing_mask_disagreement_count=disagreement_count,
        missing_mask_disagreement_fraction=(
            float(disagreement_count / total_count) if total_count else float("nan")
        ),
        bias=bias,
        rmse=rmse,
        p95_absolute=p95_absolute,
        maximum_absolute=maximum_absolute,
        mismatch_fraction=mismatch,
    )


def compare(
    netcdf_path: str | Path,
    legacy_path: str | Path,
    *,
    time_mode: TimeMode = "auto",
) -> tuple[Literal["begin", "end"], dict[str, Statistics], tuple[str, ...]]:
    """Compare all available mapped fields in two CALMET outputs.

    Args:
        netcdf_path: PyCALMET NetCDF output path.
        legacy_path: Fortran CALMET.DAT v2.1 path.
        time_mode: Whether NetCDF timestamps denote interval beginnings,
            endings, or should be detected from complete timestamp sets.

    Returns:
        Selected time alignment, field statistics, and skipped optional field
        descriptions.

    Raises:
        OSError: If either input cannot be opened.
        ValueError: If metadata, times, shapes, or required fields differ.
    """
    legacy: InitialGuessCalmetData = InitialGuessCalmetData(
        legacy_path, retain_fields=True
    )
    periods: list[tuple[datetime, datetime, Mapping[str, FieldArray]]] = sorted(
        (begin, end, fields)
        for end, (begin, fields) in legacy.fields.items()
    )
    if not periods:
        raise ValueError("Fortran CALMET.DAT contains no retained field periods")
    beginnings: tuple[datetime, ...] = tuple(item[0] for item in periods)
    endings: tuple[datetime, ...] = tuple(item[1] for item in periods)
    levels: int = len(legacy.z_faces_m) - 1
    results: dict[str, Statistics] = {}
    skipped: list[str] = []

    with netCDF4.Dataset(netcdf_path) as dataset:
        _validate_grid_and_vertical_coordinates(dataset, legacy)
        netcdf_times: tuple[datetime, ...] = _netcdf_times(dataset)
        selected_mode: Literal["begin", "end"] = _choose_time_mode(
            time_mode, netcdf_times, beginnings, endings
        )
        time_to_index: dict[datetime, int] = {
            value: index for index, value in enumerate(netcdf_times)
        }
        for static_spec in STATIC_FIELD_SPECS:
            if static_spec.netcdf_name not in dataset.variables:
                skipped.append(f"{static_spec.name}: NetCDF variable is absent")
                continue
            static_python_values: FloatArray = _float_values(
                dataset.variables[static_spec.netcdf_name][:]
            )
            static_legacy_values: FloatArray = _float_values(
                cast(npt.ArrayLike, getattr(legacy, static_spec.legacy_attribute))
            )
            if static_python_values.shape != static_legacy_values.shape:
                raise ValueError(
                    f"{static_spec.name} shape differs: NetCDF "
                    f"{static_python_values.shape}, CALMET.DAT "
                    f"{static_legacy_values.shape}"
                )
            results[static_spec.name] = _statistics(
                [_comparison_sample(static_python_values, static_legacy_values)],
                integer=static_spec.integer,
            )
        first_fields: Mapping[str, FieldArray] = periods[0][2]
        for spec in FIELD_SPECS:
            if spec.netcdf_name not in dataset.variables:
                if spec.required:
                    raise ValueError(
                        f"Python NetCDF output has no required {spec.netcdf_name} variable"
                    )
                skipped.append(f"{spec.name}: NetCDF variable is absent")
                continue
            try:
                spec.legacy(first_fields, levels)
            except KeyError:
                if spec.required:
                    raise ValueError(
                        f"Fortran CALMET.DAT has no required records for {spec.name}"
                    ) from None
                skipped.append(f"{spec.name}: CALMET.DAT record is absent")
                continue
            variable: netCDF4.Variable[Any] = dataset.variables[spec.netcdf_name]
            samples: list[ComparisonSample] = []
            for begin, end, fields in periods:
                index: int = time_to_index[begin if selected_mode == "begin" else end]
                try:
                    legacy_values: FloatArray = _field_values(
                        spec.legacy(fields, levels),spec,
                    )
                except KeyError as error:
                    raise ValueError(
                        f"Fortran CALMET.DAT is missing {error.args[0]} at {end}"
                    ) from None
                python_values: FloatArray = _field_values(
                    spec.netcdf(variable, index, levels),spec,
                )
                if python_values.shape != legacy_values.shape:
                    raise ValueError(
                        f"{spec.name} shape differs at {begin}: "
                        f"NetCDF {python_values.shape}, CALMET.DAT {legacy_values.shape}"
                    )
                samples.append(_comparison_sample(python_values, legacy_values))
            results[spec.name] = _statistics(samples, integer=spec.integer)

        if "wind_u" in results and "wind_v" in results:
            speed_samples: list[ComparisonSample] = []
            u_variable: netCDF4.Variable[Any] = dataset.variables["wind_u"]
            v_variable: netCDF4.Variable[Any] = dataset.variables["wind_v"]
            for begin, end, fields in periods:
                index = time_to_index[begin if selected_mode == "begin" else end]
                python_u: FloatArray = _float_values(u_variable[index])
                python_v: FloatArray = _float_values(v_variable[index])
                legacy_u: FloatArray = _float_values(_legacy_levels(fields, levels, "U-LEV"))
                legacy_v: FloatArray = _float_values(_legacy_levels(fields, levels, "V-LEV"))
                python_speed: FloatArray = np.asarray(
                    np.hypot(python_u, python_v), dtype=np.float64
                )
                legacy_speed: FloatArray = np.asarray(
                    np.hypot(legacy_u, legacy_v), dtype=np.float64
                )
                speed_samples.append(_comparison_sample(python_speed, legacy_speed))
            results["wind_speed"] = _statistics(speed_samples, integer=False)

    return selected_mode, results, tuple(skipped)


def _parser() -> argparse.ArgumentParser:
    """Create the CALMET-output comparison command-line parser.

    Returns:
        Configured parser for NetCDF path, CALMET.DAT path, and time alignment.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyCALMET NetCDF fields with an official Fortran "
            "CALMET.DAT output. Statistics are Python minus Fortran."
        )
    )
    parser.add_argument("python_netcdf", type=Path, help="PyCALMET NetCDF output")
    parser.add_argument("fortran_calmet", type=Path, help="Fortran CALMET.DAT output")
    parser.add_argument(
        "--netcdf-time",
        choices=("auto", "begin", "end"),
        default="auto",
        help=(
            "meaning of NetCDF timestamps; auto requires an exact match to "
            "all Fortran interval beginnings or endings (default: auto)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison CLI and print an aligned statistics table.

    Args:
        argv: Command-line arguments without the executable name, or ``None`` to
            let :mod:`argparse` read :data:`sys.argv`.

    Returns:
        Process exit status, with zero indicating a completed comparison.

    Raises:
        OSError: If an input file cannot be opened.
        ValueError: If output metadata, timestamps, shapes, or required fields are
            incompatible.
    """
    arguments = _parser().parse_args(argv)
    mode, results, skipped = compare(
        arguments.python_netcdf,
        arguments.fortran_calmet,
        time_mode=cast(TimeMode, arguments.netcdf_time),
    )
    print(f"time_alignment={mode}; differences=Python-Fortran")
    print(
        f"{'field':34s} {'paired':>11s} {'total':>11s} {'bias':>14s} "
        f"{'rmse':>14s} {'p95_abs':>14s} {'max_abs':>14s} "
        f"{'mismatch':>11s} {'nc_missing':>11s} {'ft_missing':>11s} "
        f"{'mask_disagree':>21s}"
    )
    for name, item in results.items():
        mismatch: str = (
            "-" if item.mismatch_fraction is None
            else f"{100.0 * item.mismatch_fraction:.6g}%"
        )
        mask_disagreement: str = (
            f"{item.missing_mask_disagreement_count}/{item.total_count} "
            f"({100.0 * item.missing_mask_disagreement_fraction:.6g}%)"
        )
        print(
            f"{name:34s} {item.count:11d} {item.total_count:11d} "
            f"{item.bias:14.7g} "
            f"{item.rmse:14.7g} {item.p95_absolute:14.7g} "
            f"{item.maximum_absolute:14.7g} {mismatch:>11s} "
            f"{item.netcdf_missing_count:11d} {item.fortran_missing_count:11d} "
            f"{mask_disagreement:>21s}"
        )
    for message in skipped:
        print(f"skipped: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
