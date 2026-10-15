"""Compatibility between legacy control names and Python configuration fields."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
import re
import shlex
from typing import Any, Callable, Final, TypeAlias

from .config import (
    CalmetConfig, CalwrfConfig, CtgprocConfig, CtgprocSource, GridConfig,
    MakegeoConfig, TerrelConfig, TerrelSource,
)
from .datetime_utils import exact_integer,parse_calendar,parse_timezone_label


@dataclass(frozen=True, slots=True)
class NormalizedParameters:
    """Canonical parameters separated by their configuration destination.

    Attributes:
        values: Top-level configuration fields keyed by their Python names.
            Required.
        grid: Flattened :class:`GridConfig` fields, without the ``grid.``
            prefix used internally by the alias table. Required.
        special: Intermediate values needed to assemble structured fields or
            implement legacy control-file switches. Keys omit the internal
            ``$`` prefix. Required.
    """

    values: dict[str, Any]
    grid: dict[str, Any]
    special: dict[str, Any]


def _single(value: Any) -> Any:
    """Unwrap a non-text sequence that contains exactly one item.

    Args:
        value: Scalar or sequence supplied through the public or legacy API.

    Returns:
        The sequence's only item when ``value`` is a one-item, non-text
        sequence; otherwise ``value`` unchanged.
    """
    if (isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, Path))
            and len(value) == 1):
        return value[0]
    return value


def _identity(value: Any) -> Any:
    """Apply only the common one-item sequence normalization.

    Args:
        value: Parameter value to normalize.

    Returns:
        ``value`` after unwrapping a one-item non-text sequence.
    """
    return _single(value)


def _number(value: Any) -> int | float:
    """Parse a Python or Fortran numeric scalar without losing integers.

    Fortran double-precision exponent markers (``D`` or ``d``) are accepted.

    Args:
        value: Numeric value or text, optionally wrapped in a one-item
            sequence.

    Returns:
        An ``int`` for integral text and a ``float`` for decimal or
        exponential text.

    Raises:
        ValueError: If the normalized text is not a valid number.
    """
    value = _single(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    token = str(value).strip().replace("D", "E").replace("d", "e")
    return float(token) if any(char in token for char in ".Ee") else int(token)


def _integer(value: Any) -> int:
    """Convert a scalar to an exact integer.

    Args:
        value: Integral number or numeric text, optionally in a one-item
            sequence.

    Returns:
        The exactly represented integer.

    Raises:
        ValueError: If ``value`` is nonnumeric, non-finite, or not integral.
    """
    return exact_integer(_single(value))


def _named_integer(label: str) -> Callable[[Any], int]:
    """Build an exact-integer converter that names its control variable.

    Args:
        label: Legacy control-variable name used in conversion errors.

    Returns:
        A converter accepting one parameter value and returning an integer.
    """
    return lambda value: exact_integer(_single(value),label)


def _real(value: Any) -> float:
    """Convert a scalar, including Fortran numeric text, to ``float``.

    Args:
        value: Numeric value or text, optionally in a one-item sequence.

    Returns:
        The floating-point value.

    Raises:
        ValueError: If ``value`` cannot be parsed as a number.
    """
    return float(_number(value))


def _logical(value: Any) -> bool:
    """Parse a Python or Fortran logical scalar.

    Accepted true tokens are ``T``, ``.TRUE.``, ``TRUE``, and ``1``;
    accepted false tokens are ``F``, ``.FALSE.``, ``FALSE``, and ``0``.

    Args:
        value: Boolean, zero/one number, or logical text, optionally wrapped
            in a one-item sequence.

    Returns:
        The parsed Boolean value.

    Raises:
        ValueError: If ``value`` is not a recognized logical representation.
    """
    value = _single(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value).strip().upper()
    if token in {"T", ".TRUE.", "TRUE", "1"}:
        return True
    if token in {"F", ".FALSE.", "FALSE", "0"}:
        return False
    raise ValueError(f"invalid Fortran logical value: {value!r}")


def _upper(value: Any) -> str:
    """Normalize a scalar as stripped uppercase text.

    Args:
        value: Value to normalize, optionally in a one-item sequence.

    Returns:
        The stripped, uppercase string representation.
    """
    return str(_single(value)).strip().upper()


def _text(value: Any) -> str:
    """Normalize a scalar as text without changing whitespace or case.

    Args:
        value: Value to normalize, optionally in a one-item sequence.

    Returns:
        The string representation of the normalized value.
    """
    return str(_single(value))


def _angle(value: Any) -> float:
    """Parse a signed latitude or longitude with an optional compass suffix.

    ``S`` and ``W`` force a negative sign, while ``N`` and ``E`` force a
    positive sign. East/west values outside [-180, 180] are wrapped once to
    match the traditional control-file convention.

    Args:
        value: Numeric angle or text containing at most one of ``NSEW``.

    Returns:
        The signed angle in decimal degrees.

    Raises:
        ValueError: If the value is not numeric or contains multiple compass
            directions.
    """
    token = re.sub(r"\s+", "", str(_single(value)))
    directions = re.findall(r"[NSEW]", token, re.I)
    if len(directions) > 1:
        raise ValueError(f"invalid directional angle: {value!r}")
    direction = directions[0].upper() if directions else ""
    if direction:
        token = re.sub(r"[NSEW]", "", token, flags=re.I)
    try:
        result = float(token)
    except ValueError as error:
        raise ValueError(f"invalid directional angle: {value!r}") from error
    if direction in {"S", "W"}:
        result = -abs(result)
    elif direction in {"N", "E"}:
        result = abs(result)
    if direction in {"E", "W"}:
        if result > 180.0:
            result -= 360.0
        elif result < -180.0:
            result += 360.0
    return result


def _tokens(value: Any) -> list[Any]:
    """Split a scalar or sequence into whitespace/comma-delimited tokens.

    Args:
        value: Text, a sequence of values, or a scalar. String elements of a
            sequence are split further when they contain delimiters.

    Returns:
        Tokens in their original order. Non-string tokens retain their types.
    """
    if isinstance(value, str):
        return [item for item in re.split(r"[\s,]+", value.strip()) if item]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, Path)):
        result: list[Any] = []
        for item in value:
            if isinstance(item, str) and ("," in item or len(item.split()) > 1):
                result.extend(token for token in re.split(r"[\s,]+", item.strip()) if token)
            else:
                result.append(item)
        return result
    return [value]


def _integers(value: Any) -> tuple[int, ...]:
    """Parse every token in a value as an exact integer.

    Args:
        value: Scalar, delimited text, or sequence of integer values.

    Returns:
        Parsed integers in input order.

    Raises:
        ValueError: If a token is nonnumeric, non-finite, or not integral.
    """
    return tuple(_integer(item) for item in _tokens(value))


def _nonzero_integers(value: Any) -> tuple[int, ...]:
    """Parse integer tokens and discard zero padding values.

    Args:
        value: Scalar, delimited text, or sequence of integer values.

    Returns:
        Parsed nonzero integers in input order.

    Raises:
        ValueError: If a token is nonnumeric, non-finite, or not integral.
    """
    return tuple(item for item in _integers(value) if item != 0)


def _reals(value: Any) -> tuple[float, ...]:
    """Parse every token in a value as a floating-point number.

    Args:
        value: Scalar, delimited text, or sequence of numeric values.

    Returns:
        Parsed floating-point values in input order.

    Raises:
        ValueError: If a token is not numeric.
    """
    return tuple(_real(item) for item in _tokens(value))


def _files(value: Any) -> tuple[str | Path, ...]:
    """Normalize one path or a path sequence to a tuple.

    Args:
        value: A string/path scalar or iterable of strings and paths.

    Returns:
        A tuple containing the supplied path values.
    """
    if isinstance(value, (str, Path)):
        return (value,)
    return tuple(value)


def _file_or_files(value: Any) -> str | Path | tuple[str | Path, ...]:
    """Normalize path input while preserving a scalar for one file.

    Args:
        value: A string/path scalar or iterable of strings and paths.

    Returns:
        The sole path when one was supplied, otherwise a tuple of paths.
    """
    result = _files(value)
    return result[0] if len(result) == 1 else result


def _sampling(value: Any) -> str:
    """Translate TERREL's ``MSHEET`` switch to a sampling method name.

    Args:
        value: ``0`` for legacy sheet-corner interpolation or ``1`` for
            direct transformation of every terrain sample.

    Returns:
        ``"legacy"`` for zero or ``"direct"`` for one.

    Raises:
        ValueError: If the value is not exactly 0 or 1.
    """
    method = _integer(value)
    if method not in {0, 1}:
        raise ValueError("MSHEET must be 0 or 1")
    return "direct" if method else "legacy"


Alias: TypeAlias = tuple[tuple[str, ...], Callable[[Any], Any]]


def _alias(target: str | tuple[str, ...], converter: Callable[[Any], Any] = _identity) -> Alias:
    """Create an alias-table entry for one or more canonical destinations.

    Args:
        target: Canonical field name or names populated by the alias.
        converter: Function converting the supplied legacy value.

    Returns:
        The normalized target tuple paired with ``converter``.
    """
    return ((target,) if isinstance(target, str) else target, converter)


def _add(mapping: dict[str, Alias], converter: Callable[[Any], Any], **items: str) -> None:
    """Add multiple legacy-to-canonical aliases using one converter.

    Args:
        mapping: Alias table modified in place.
        converter: Value converter shared by all new entries.
        **items: Mapping from legacy control names to canonical field names.
    """
    for legacy, target in items.items():
        mapping[legacy] = _alias(target, converter)


GRID: dict[str, Alias] = {
    "DGRIDKM": _alias(("grid.dx_km", "grid.dy_km"), _real),
}
_add(GRID, _integer, NX="grid.nx", NY="grid.ny", IUTMZN="grid.utm_zone",
     IGRID="grid.grid_type", IPROC="grid.polar_processing")
_add(GRID, _real, DXKM="grid.dx_km", DYKM="grid.dy_km",
     XREFKM="grid.x_origin_km", XORIGKM="grid.x_origin_km",
     YREFKM="grid.y_origin_km", YORIGKM="grid.y_origin_km",
     FEAST="grid.false_easting_km", FNORTH="grid.false_northing_km",
     TMSCALE="grid.ttm_scale")
_add(GRID, _angle, RLAT0="grid.lat0", XLAT0="grid.lat0", RLON0="grid.lon0",
     XLON0="grid.lon0", RLAT1="grid.lat1", XLAT1="grid.lat1",
     RLAT2="grid.lat2", XLAT2="grid.lat2")
_add(GRID, _upper, PMAP="grid.projection", UTMHEM="grid.hemisphere")
_add(GRID, _reals, DISKM="grid.ring_distances_km", ANGDEG="grid.ray_angles_deg",
     DISK="grid.ring_distances_km", ANG="grid.ray_angles_deg")


CALWRF: dict[str, Alias] = {}
_add(CALWRF, _files, FLNAMES="wrf_files")
_add(CALWRF, _identity, IDATEB="start", IDATEE="end", FIO="$fio",
     FLG="listing_output", FSRF="surface_output")
_add(CALWRF, _text, TITLE="title")
_add(CALWRF, _logical, IOSRF="create_surface")
_add(CALWRF, _integer, NFILES="$nfiles", IB="$ib", IE="$ie", JB="$jb",
     JE="$je", KB="$kb", KE="$ke")


TERREL_KINDS: Final[tuple[str, ...]] = ("USGS90", "USGS30", "ARM3", "3CD", "DMDF", "CDED", "SRTM1",
                 "SRTM3", "GTOPO30", "USGSLA", "NZGEN", "GEN", "GEOTIFF")
TERREL_DATUM_KEYS: Final[dict[str, str]] = {"USGS90":"DUSGS90", "USGS30":"DUSGS30", "ARM3":"DARM3",
    "3CD":"D3CD", "DMDF":"DDMDF", "CDED":"DCDED", "SRTM1":"DSRTM1",
    "SRTM3":"DSRTM3", "GTOPO30":"DGTOPO30", "USGSLA":"DUSGSLA",
    "NZGEN":"DNZGEN", "GEN":"DGEN", "GEOTIFF":"DGEOTIFF"}
TERREL_DATUM_DEFAULTS: Final[dict[str, str]] = {"USGS90":"WGS-72", "USGS30":"NAS-C", "ARM3":"NAS-C",
    "3CD":"WGS-72", "DMDF":"NAS-C", "CDED":"NAR-C", "SRTM1":"WGS-96",
    "SRTM3":"WGS-96", "GTOPO30":"WGS-84", "USGSLA":"ESR-S",
    "NZGEN":"WGS-84", "GEN":"WGS-84", "GEOTIFF":"WGS-84"}
TERREL: dict[str, Alias] = dict(GRID)
_add(TERREL, _identity, OUTFIL="output", XYINP="receptor_input", XYOUT="receptor_output",
     PREVFIL="previous_save", SAVEFIL="save_output", COASTBLN="coastline_file",
     GSHHSIN="gshhs_file", PLTFIL="plot_output", RAWECHO="raw_echo_output",
     SRTMDIR="srtm_dir")
_add(TERREL, _text, DATUM="datum", DWVS="coastline_source_datum",
     DWDBII="coastline_wdbii_datum")
TERREL["DATUM"] = _alias(("datum", "grid.datum"), _text)
_add(TERREL, _sampling, MSHEET="sampling")
_add(TERREL, _logical, LVOIDFIL="void_fill", LINTXY="interpolate_receptors",
     LCFILES="$lcfiles", LXY="$lxy", LPREV="$lprev", LCOAST="$lcoast",
     LBLNREAD="$lblnread", LRAWECHO="$lrawecho")
_add(TERREL, _integer, IMODEL="model", NXYCOL="receptor_columns",
     ITHRES="qa_hit_threshold_percent", NTDF="$source_count", NRING="$nring",
     NRAYS="$nrays")
TERREL["LSTFIL"] = _alias("$listing")
_add(TERREL, _real, CELLRADKM="cell_radius_km", XYRADKM="receptor_radius_km")
_add(TERREL, _integers, INOISEREP="noise_replacement", ITERREP="missing_replacement")
_add(TERREL, _reals, ZNOISE="noise_minimum_m", TERDEF="default_elevation_m")
for kind in TERREL_KINDS:
    TERREL[kind] = _alias(f"$source.{kind}", _files)
    TERREL[TERREL_DATUM_KEYS[kind]] = _alias(f"$source_datum.{kind}", _text)


CTG_KINDS: Final[tuple[str, ...]] = ("GLC2K", "UMDGLC", "CTG", "NZGEN", "GLAZNA", "GLAZSA", "GLAZEU",
             "GLAZAS", "GLAZAF", "GLAZAP", "NLCD92", "NLCDTF", "NLCD01",
             "CORINE", "MODIS", "GEN")
CTG: dict[str, Alias] = dict(GRID)
_add(CTG, _identity, LUDAT="output", LUINDAT="$luindat", COASTBLN="coastline_file",
     GSHHSIN="gshhs_file", PREVDAT="previous_landuse")
_add(CTG, _text, DATUM="grid.datum", DWVS="coastline_source_datum",
     DWDBII="coastline_wdbii_datum",
     DCTG="$source_datum.CTG", DNZGEN="$source_datum.NZGEN",
     DNLCD="$source_datum.NLCD", DCORINE="$source_datum.CORINE",
     DGLC2K="$source_datum.GLC2K", DUMDGLC="$source_datum.UMDGLC",
     DMODIS="$source_datum.MODIS", DUSGSLA="$source_datum.GLAZ")
_add(CTG, _integer, NDBF="$source_count", LULC="$lulc", IGLAZR="$iglazr",
     MESHCTG="$mesh.CTG", MESHCORN="$mesh.CORINE", MESHGLAZ="$mesh.default",
     IOCEAN="ocean_category", ITHRESH="qa_hit_threshold_percent")
_add(CTG, _logical, LMARSWAP="coastline_swap", LMARFILL="fill_empty_ocean",
     LFINAL="final_output", LSNOW="snow_processing", LQACELL="qa_cell_output",
     LCFILES="$lcfiles", LCOAST="$lcoast", LBLNREAD="$lblnread", LPREV="$lprev")
CTG["RUNLST"] = _alias("$listing")
for kind in CTG_KINDS:
    CTG[kind] = _alias(f"$source.{kind}", _files)


def _records(value: Any, count: int) -> list[list[Any]]:
    """Normalize legacy fixed-width records to lists of fields.

    Mapping input is converted from ``key -> remaining fields`` to records
    beginning with the key. Text records are split on whitespace and commas.

    Args:
        value: One record, a sequence of records, or a mapping of record keys
            to their remaining fields.
        count: Required number of fields in every normalized record.

    Returns:
        A list of mutable field lists, one for each record.

    Raises:
        ValueError: If any record does not contain exactly ``count`` fields.
    """
    if isinstance(value, Mapping):
        value = [[key, *record] for key, record in value.items()]
    elif isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        value = [value]
    result: list[list[Any]] = []
    for record in value:
        parts = ([part for part in re.split(r"[\s,]+", record.strip()) if part]
                 if isinstance(record, str) else list(record))
        if len(parts) != count:
            raise ValueError(f"legacy record must contain {count} values: {record!r}")
        result.append(parts)
    return result


def _properties(value: Any) -> dict[int, tuple[float, float, float, float, float, float, int]]:
    """Parse MAKEGEO surface-property records keyed by input category.

    Args:
        value: Eight-field records containing an input category, six real
            properties, and the mapped output category.

    Returns:
        Input categories mapped to six floating-point properties followed by
        the integer output category.

    Raises:
        ValueError: If a record has the wrong width or a field is not numeric.
    """
    return {
        _integer(row[0]): (
            _real(row[1]), _real(row[2]), _real(row[3]),
            _real(row[4]), _real(row[5]), _real(row[6]),
            _integer(row[7]),
        )
        for row in _records(value, 8)
    }


def _snow_age(value: Any) -> dict[int, tuple[int, float]]:
    """Parse MAKEGEO snow-age parameter records.

    Args:
        value: Three-field records containing category, integer age time
            scale in days, and albedo-reduction factor.

    Returns:
        Categories mapped to ``(age_time_scale_days, reduction_factor)``.

    Raises:
        ValueError: If a record has the wrong width or a field is not numeric.
    """
    return {_integer(row[0]): (_integer(row[1]), _real(row[2])) for row in _records(value, 3)}


def _splits(value: Any) -> tuple[tuple[int, int, float], ...]:
    """Parse MAKEGEO land-use category split records.

    Args:
        value: Three-field records containing source category, target
            category, and percentage.

    Returns:
        Parsed ``(source_category, target_category, percent)`` records.

    Raises:
        ValueError: If a record has the wrong width or a field is not numeric.
    """
    return tuple((_integer(row[0]), _integer(row[1]), _real(row[2]))
                 for row in _records(value, 3))


MAKEGEO: dict[str, Alias] = dict(GRID)
_add(MAKEGEO, _identity, TERRDAT="terrain", LUDAT="landuse", GEODAT="output",
     LU2DAT="secondary_landuse", SNOWDAT="snow_data", LUGRD="landuse_plot_output",
     TEGRD="terrain_plot_output")
_add(MAKEGEO, _text, DATUM="grid.datum")
_add(MAKEGEO, _real, CFRACT="water_threshold", FLUMIN="minimum_coverage",
     HSCL="snow_height_scale_m", SDPMIN="snow_depth_minimum_m")
_add(MAKEGEO, _integer, IMISS="missing_category", IWAT1="output_water_min",
     IWAT2="output_water_max",
     MSRL="snow_roughness_method", MSAL="snow_albedo_method", IMAGE="image_mode",
     IXQA="$ixqa", IYQA="$iyqa", NINCAT="$nincat", NOUTCAT="$noutcat",
     NUMWAT="$numwat", NSPLIT="$nsplit")
MAKEGEO["IFMTGEO"] = _alias("$ifmtgeo", _integer)
_add(MAKEGEO, _identity, IDATEBEG="snow_start_date", IDATEEND="snow_end_date")
_add(MAKEGEO, _logical, LTERR="use_terrain", LSNOW="snow_processing",
     LCFILES="$lcfiles", LLU2="$llu2")
_add(MAKEGEO, _integers, IWAT="water_categories")
MAKEGEO["OUTCAT"] = _alias("output_categories", _nonzero_integers)
MAKEGEO.update(X=_alias("input_properties", _properties),
               XS=_alias("snow_properties", _properties),
               XF=_alias("snow_age_parameters", _snow_age),
               XSPLIT=_alias("category_splits", _splits))
MAKEGEO["RUNLST"] = _alias("$listing")


CALMET: dict[str, Alias] = dict(GRID)
_add(CALMET, _identity, PRGDAT="meteorology", GEODAT="geophysical", METDAT="output",
     PACDAT="mesopac_output", DCSTGD="distance_to_coast_output", METLST="listing_output",
     CLDDAT="cloud_data", SRFDAT="surface_data", DIADAT="diagnostic_data",
     WTDAT="weight_data", PRCDAT="precipitation_data", TSTOUT="final_wind_debug_output",
     TSTKIN="kinematic_wind_debug_output", TSTFRD="froude_wind_debug_output",
     TSTSLP="slope_wind_debug_output", TSTPRT="wind_print_output")
_add(CALMET, _file_or_files, M3DDAT="meteorology")
CALMET["MM4DAT"] = _alias("$mm4dat", _identity)
_add(CALMET, _files, UPDAT="upper_air_files", SEADAT="sea_files",
     IGFDAT="initial_guess_calmet_files")
_add(CALMET, _text, DATUM="grid.datum", ABTZ="$abtz")
_add(CALMET, _logical, LSAVE="save_output", LPRINT="print_fields",
     LDB="micrometeorology_debug", LCALGRD="include_w_temperature",
     LVARY="varying_radius", LCFILES="$lcfiles", LDBCST="$ldbcst")
_add(CALMET, _reals, ZFACE="z_faces_m", ZUPWND="initial_guess_height_range_m",
     BIAS="wind_bias", FEXTR2="wind_extrapolation_factors")
_add(CALMET, _integers, IUVOUT="print_uv_levels", IWOUT="print_w_levels",
     ITOUT="print_temperature_levels", NINTR2="stations_per_level", NSMTH="smoothing_passes")
for index, name in enumerate(("STABILITY", "USTAR", "MONIN", "MIXHT", "WSTAR",
                              "PRECIP", "SENSHEAT", "CONVZI")):
    CALMET[name] = _alias(f"$surface_field.{index}", _integer)
for index in range(9):
    CALMET[f"IPR{index}"] = _alias(f"$wind_debug_field.{index}", _integer)
_add(CALMET, _logical, LLBREZE="$lake.enabled")
_add(CALMET, _integer, NBOX="$lake.count")
_add(CALMET, _integers, NLB="$lake.station_counts", METBXID="$lake.station_ids")
_add(CALMET, _reals, XBBAR="$barrier.xb", YBBAR="$barrier.yb",
     XEBAR="$barrier.xe", YEBAR="$barrier.ye", XG1="$lake.xg1",
     XG2="$lake.xg2", YG1="$lake.yg1", YG2="$lake.yg2",
     XBCST="$lake.xbcst", YBCST="$lake.ybcst", XECST="$lake.xecst",
     YECST="$lake.yecst")

CALMET_INT: Final[dict[str, str]] = {
 "NM3D":"$nm3d", "NZ":"$nz", "MREG":"regulatory_mode", "IRTYPE":"run_type",
 "IFORMO":"legacy_output_format", "IPRINF":"print_interval_hours", "NN1":"debug_first_step",
 "NN2":"debug_last_step", "IOUTD":"wind_debug_output", "NZPRN2":"wind_debug_levels",
 "ITEST":"test_mode", "NSECDT":"timestep_seconds", "ISTEPPG":"$isteppg",
 "ISTEPPGS":"$isteppgs",
 "MCLOUD":"cloud_method", "IFORMC":"cloud_data_format", "ICLDOUT":"cloud_output",
 "IFORMS":"surface_data_format", "IDIOPT1":"diagnostic_surface_temperature",
 "IDIOPT2":"diagnostic_lapse_rate", "IDIOPT3":"diagnostic_initial_wind",
 "IDIOPT4":"diagnostic_surface_wind", "IDIOPT5":"diagnostic_upper_wind",
 "IUPWND":"initial_guess_upper_station", "IGFMET":"use_initial_guess_calmet",
 "NPSTA":"precipitation_station_count", "IFORMP":"precipitation_data_format", "NOOBS":"noobs",
 "IPROG":"iprog", "IRHPROG":"irhprog", "ITPROG":"itprog", "IMIXH":"imixh",
 "IAVET":"iavet", "IAVEZI":"iavezi", "ILEVZI":"mixing_wind_level", "MNMDAV":"mnmdav",
 "ITWPROG":"water_temperature_mode", "ILUOC3D":"prognostic_ocean_category",
 "ICOARE":"overwater_flux_method", "IWARM":"warm_layer", "ICOOL":"cool_skin",
 "IZICRLX":"convective_relaxation", "NFLAGP":"precipitation_interpolation",
 "ISURFT":"surface_temperature_option", "IUPT":"upper_temperature_option",
 "IRAD":"temperature_interpolation_power", "NUMTS":"temperature_station_limit",
 "JWAT1":"water_temperature_category_min", "JWAT2":"water_temperature_category_max",
 "IWFCOD":"iwfcod", "IEXTRP":"wind_extrapolation", "ICALM":"extrapolate_calm",
 "NBAR":"barrier_count", "KBAR":"barrier_top_level", "IFRADJ":"ifradj", "IKINE":"ikine",
 "IOBR":"iobr", "ISLOPE":"islope", "NITER":"divergence_iterations", "IBTZ":"base_timezone",
 "IRLG":"$irlg", "ICLOUD":"$icloud", "NSSTA":"$nssta", "NUSTA":"$nusta",
 "NOWSTA":"$nowsta", "NIGF":"$nigf",
}
_add(CALMET, _integer, **CALMET_INT)
for name in ("NSECDT","ISTEPPG","ISTEPPGS"):
    CALMET[name]=_alias(CALMET_INT[name],_named_integer(name))
CALMET_REAL: Final[dict[str, str]] = {
 "HAFANG":"half_average_angle", "CONSTB":"constb", "CONSTE":"conste", "CONSTN":"constn",
 "CONSTW":"constw", "FCORIOL":"coriolis", "DPTMIN":"minimum_lapse_rate",
 "DZZI":"mixing_lapse_depth_m", "ZIMIN":"minimum_mixing_height_m",
 "ZIMAX":"maximum_mixing_height_m", "ZIMINW":"minimum_water_mixing_height_m",
 "ZIMAXW":"maximum_water_mixing_height_m", "THRESHL":"land_flux_threshold",
 "THRESHW":"water_flux_threshold", "DSHELF":"coastal_shelf_length_km",
 "TZICRLX":"convective_relaxation_time_s", "SIGMAP":"precipitation_radius_km",
 "CUTP":"precipitation_cutoff", "ZUPT":"lapse_height_m", "TRADKM":"temperature_radius_km",
 "TGDEFB":"water_temperature_gradient_below", "TGDEFA":"water_temperature_gradient_above",
 "HA1":"turbidity_coefficient_1", "HA2":"turbidity_coefficient_2",
 "HB1":"cloudiness_coefficient_1", "HB2":"cloudiness_coefficient_2",
 "HC1":"net_radiation_coefficient_1", "HC2":"net_radiation_coefficient_2",
 "HC3":"net_radiation_coefficient_3",
 "RMIN2":"upper_air_separation_cells", "RMIN":"minimum_station_radius_km",
 "RMAX1":"maximum_surface_radius_km", "RMAX2":"maximum_upper_radius_km",
 "RMAX3":"maximum_water_radius_km", "R1":"surface_background_radius_km",
 "R2":"upper_background_radius_km", "RPROG":"prognostic_radius_km", "CRITFN":"critical_froude",
 "DIVLIM":"divergence_limit", "ALPHA":"kinematic_alpha", "TERRAD":"terrain_radius_km",
}
_add(CALMET, _real, **CALMET_REAL)
for prefix in ("IB", "IE"):
    for suffix in ("YR", "MO", "DY", "HR", "SEC"):
        name=prefix+suffix
        CALMET[name]=_alias("$date."+name,_named_integer(name))


ALIASES: dict[type[Any], dict[str, Alias]] = {
    CalwrfConfig: CALWRF, TerrelConfig: TERREL, CtgprocConfig: CTG,
    MakegeoConfig: MAKEGEO, CalmetConfig: CALMET,
}


def _duplicate(cls: type[Any], target: str, old: str, new: str) -> TypeError:
    """Create the error used when two spellings claim one parameter.

    Args:
        cls: Configuration class whose parameter is duplicated.
        target: Canonical field name, possibly with ``grid.`` or ``$``.
        old: First spelling that supplied the field.
        new: Later spelling that supplied the same field.

    Returns:
        A descriptive ``TypeError`` ready to be raised by the caller.
    """
    label = target.removeprefix("grid.").removeprefix("$")
    return TypeError(f"duplicate {cls.__name__} parameter {label!r}: {old!r} and {new!r}")


def _timezone(value: str) -> tuple[int, str]:
    """Parse a CALMET ``ABTZ`` timezone label for a whole-hour offset.

    Args:
        value: CALMET timezone label, with insignificant whitespace and case.

    Returns:
        A pair containing the UTC offset in hours and canonical label.

    Raises:
        ValueError: If the label is unsupported or encodes a fractional-hour
            offset that CALMET cannot represent.
    """
    try:
        return parse_timezone_label("".join(str(value).split()).upper())
    except (TypeError,ValueError) as error:
        if "whole-hour" in str(error):
            raise ValueError(
                "CALMET base_timezone only supports whole-hour ABTZ values") from error
        raise ValueError(f"unsupported CALMET timezone: {value!r}") from error


def _date(parts: Mapping[str, Any], prefix: str) -> datetime:
    """Assemble a CALMET begin or end timestamp from legacy date fields.

    CALMET accepts hour 24 and second 3600 boundary notation; the shared
    calendar parser normalizes those values into the following day or hour.

    Args:
        parts: Date components keyed by names such as ``IBYR`` and ``IESEC``.
        prefix: ``"IB"`` for the begin time or ``"IE"`` for the end time.

    Returns:
        A naive, second-precision ``datetime``.

    Raises:
        TypeError: If any required year, month, day, or hour field is absent.
        ValueError: If a component is invalid or outside CALMET's accepted
            hour and second boundary ranges.
    """
    required = [prefix + suffix for suffix in ("YR", "MO", "DY", "HR")]
    missing = [name for name in required if name not in parts]
    if missing:
        raise TypeError("incomplete CALMET date parameters: " + ", ".join(missing))
    try:
        return parse_calendar(
            parts[prefix + "YR"],parts[prefix + "MO"],
            parts[prefix + "DY"],parts[prefix + "HR"],
            parts.get(prefix + "SEC",0),
        )
    except ValueError as error:
        if "invalid hour" in str(error):
            raise ValueError(f"{prefix}HR must be between 0 and 24") from error
        if "invalid seconds" in str(error):
            raise ValueError(f"{prefix}SEC must be between 0 and 3600") from error
        raise


def _station(value: Any, count: int) -> tuple[Any, ...]:
    """Parse one legacy CALMET station-control row.

    Args:
        value: Shell-like text or an iterable containing station fields.
        count: Required row width for the selected station type.

    Returns:
        A tuple containing the station name, integer identifier, and remaining
        coordinates or metadata converted to floats.

    Raises:
        ValueError: If the row width is wrong or a numeric field is invalid.
    """
    if isinstance(value, str):
        parts = shlex.split(value)
    else:
        parts = list(value)
    if len(parts) != count:
        raise ValueError(f"station row must contain {count} fields: {value!r}")
    return (str(parts[0]), _integer(parts[1]), *(_real(item) for item in parts[2:]))


def _ctg_datum(kind: str, special: Mapping[str, Any]) -> str:
    """Select the source datum for one CTGPROC land-use input type.

    Args:
        kind: Uppercase CTGPROC input type, such as ``CTG`` or ``NLCD01``.
        special: Parsed legacy values containing optional ``source_datum.*``
            overrides.

    Returns:
        The explicit datum override or the format-specific traditional
        default.
    """
    if kind == "CTG": return str(special.get("source_datum.CTG", "NAS-C"))
    if kind == "NZGEN": return str(special.get("source_datum.NZGEN", "WGS-84"))
    if kind.startswith("NLCD"): return str(special.get("source_datum.NLCD", "NAR-C"))
    if kind == "CORINE": return str(special.get("source_datum.CORINE", "WGS-84"))
    if kind.startswith("GLAZ"): return str(special.get("source_datum.GLAZ", "ESR-S"))
    default = "ESR-S" if kind == "MODIS" else "WGS-84"
    return str(special.get(f"source_datum.{kind}", default))


def normalize_public_parameters(cls: type[Any], parameters: Mapping[str, Any]) -> NormalizedParameters:
    """Normalize descriptive and legacy keywords for a configuration class.

    Names are matched case-insensitively. Descriptive Python fields are kept
    as top-level values, flattened grid fields are separated into ``grid``,
    and legacy switches are resolved into structured sources, station rows,
    time ranges, barriers, and lake-breeze regions where applicable.

    Args:
        cls: One of the supported stage configuration dataclasses.
        parameters: User-supplied keyword names and values. Names may be
            descriptive Python fields, flattened grid fields, or traditional
            control-file variables.

    Returns:
        Canonical top-level, grid, and intermediate special parameters.

    Raises:
        TypeError: If a name is unknown, two spellings set the same field, or
            a required group of companion fields is incomplete.
        ValueError: If a value, declared count, date, source selection, or
            structured legacy record is invalid or inconsistent.
    """
    aliases = ALIASES.get(cls, {})
    canonical = {field.name.casefold(): field.name for field in fields(cls)}
    grid_canonical = {field.name.casefold(): field.name for field in fields(GridConfig)}
    values: dict[str, Any] = {}
    grid: dict[str, Any] = {}
    special: dict[str, Any] = {}
    claimed: dict[str, str] = {}
    unknown: list[object] = []
    explicit_grid = any(isinstance(name, str) and name.casefold() == "grid"
                        for name in parameters)

    def assign(target: str, value: Any, spelling: str) -> None:
        """Store one canonical field and reject duplicate spellings.

        Args:
            target: Canonical top-level, ``grid.*``, or ``$`` special key.
            value: Converted value to store.
            spelling: User-facing name that supplied the value.

        Raises:
            TypeError: If another spelling already claimed ``target``.
        """
        if target in claimed:
            raise _duplicate(cls, target, claimed[target], spelling)
        claimed[target] = spelling
        if target.startswith("grid."): grid[target[5:]] = value
        elif target.startswith("$"): special[target[1:]] = value
        else: values[target] = value

    for name, value in parameters.items():
        if not isinstance(name, str):
            unknown.append(name); continue
        folded = name.casefold()
        alias = aliases.get(name.upper())
        # TERREL has a top-level descriptive datum field while its DATUM
        # control also defines the nested output-grid datum.  An explicitly
        # supplied GridConfig already carries the latter.
        if cls is TerrelConfig and name.upper() == "DATUM" and explicit_grid:
            alias = _alias("datum", _text)
        if alias is None and cls is CalmetConfig:
            station = re.fullmatch(r"(SS|US|PS)(\d+)", name.upper())
            if station:
                alias = _alias(
                    f"$station.{station.group(1)}.{int(station.group(2))}",
                    _identity,
                )
        if alias is not None:
            targets, converter = alias
            converted = converter(value)
            for target in targets: assign(target, converted, name)
        elif folded in canonical:
            target = canonical[folded]
            assign(target, value, name)
        elif folded in grid_canonical:
            assign("grid." + grid_canonical[folded], value, name)
        else:
            unknown.append(name)
    if unknown:
        names = ", ".join(sorted((str(name) for name in unknown), key=str.casefold))
        raise TypeError(f"unknown {cls.__name__} parameters: {names}")
    if cls is CalwrfConfig:
        if "fio" in special:
            output = Path(special["fio"])
            if output.suffix.casefold() == ".nc":
                assign("output", output, claimed["$fio"])
            else:
                assign("legacy_output", output, claimed["$fio"])
                assign("output", output.with_suffix(".nc"), claimed["$fio"])
            if values.get("create_surface", True) and "surface_output" not in values:
                assign("surface_output", Path(output).with_suffix(".m2d"),
                       claimed["$fio"])
        for axis in "ijk":
            first, last = special.get(axis + "b"), special.get(axis + "e")
            if first is None and last is None: continue
            if first is None or last is None:
                raise TypeError(f"CALWRF {axis.upper()}B and {axis.upper()}E must be supplied together")
            assign(axis + "_range", None if first < 1 or last < 1 else (first, last),
                   claimed["$" + axis + "b"])
        if "nfiles" in special and "wrf_files" in values and special["nfiles"] != len(values["wrf_files"]):
            raise ValueError("CALWRF NFILES does not match FLNAMES")

    elif cls is TerrelConfig:
        terrain_sources: list[TerrelSource] = [TerrelSource(path, kind, str(special.get("source_datum." + kind,
                            TERREL_DATUM_DEFAULTS[kind]))) for kind in TERREL_KINDS
                           for path in special.get("source." + kind, ())]
        if terrain_sources:
            spelling = claimed["$source." + terrain_sources[0].input_type]
            assign("terrain_sources", tuple(terrain_sources), spelling)
            assign("terrain_files", tuple(source.path for source in terrain_sources), spelling)
            values.setdefault("srtm_dir", Path("."))
        if "source_count" in special and special["source_count"] != len(terrain_sources):
            raise ValueError(
                f"TERREL NTDF declares {special['source_count']} files but "
                f"{len(terrain_sources)} are active"
            )
        if "polar_processing" in grid and "processing_method" not in values:
            values["processing_method"] = grid["polar_processing"]
        for target, defaults in (
            ("noise_replacement", (0, 0, 0, 0, 0)),
            ("noise_minimum_m", (0., 1., 1., 1., 1.)),
            ("missing_replacement", (3, 0, 0, 0, 0)),
            ("default_elevation_m", (0., 0., 0., 0., 0.)),
        ):
            if target in values:
                if len(values[target]) == 2:
                    values[target] = tuple(values[target]) + defaults[2:]
                elif len(values[target]) != 5:
                    raise ValueError(f"TERREL {claimed[target]} must contain two values")
        if special.get("lxy") is False:
            values["receptor_input"] = None
            values["receptor_output"] = None
        elif special.get("lxy") is True:
            values.setdefault("receptor_input", "xyinp.dat")
            values.setdefault("receptor_output", "xyout.dat")
        if special.get("lprev") is False:
            values["previous_save"] = None
        elif special.get("lprev") is True:
            values.setdefault("previous_save", "prev.sav")
        if special.get("lcoast") is False:
            values["coastline_file"] = None
            values["gshhs_file"] = None
        elif special.get("lcoast") is True:
            values.setdefault("coastline_file", "coast.bln")
            if special.get("lblnread") is True:
                values["gshhs_file"] = None
            else:
                values.setdefault("gshhs_file", "gshhs_f.b")
        if special.get("lrawecho") is False:
            values["raw_echo_output"] = None
        elif special.get("lrawecho") is True:
            values.setdefault("raw_echo_output", "rawecho.dat")
        if grid.get("grid_type") == 2:
            if "x_origin_km" in grid and "dx_km" in grid: grid["x_origin_km"] -= .5 * grid["dx_km"]
            if "y_origin_km" in grid and "dy_km" in grid: grid["y_origin_km"] -= .5 * grid["dy_km"]
        if values.get("model") == 3: grid["grid_type"] = 3
        if special.get("lxy") is True and "receptor_columns" not in values:
            values["receptor_columns"] = 4
        if grid.get("grid_type") == 3:
            if "nring" in special: grid["nx"] = special["nring"]
            if "nrays" in special: grid["ny"] = special["nrays"]

    elif cls is CtgprocConfig:
        items = [(kind, path) for kind in CTG_KINDS for path in special.get("source." + kind, ())]
        if "luindat" in special:
            types = {1:"CTG", 3:"NZGEN", 4:"NLCD92", 5:"NLCD01", 6:"NLCDTF",
                     7:"CORINE", 8:"GLC2K", 9:"UMDGLC", 10:"MODIS", 11:"GEN"}
            lulc = special.get("lulc", 1)
            if lulc == 2:
                region = special.get("iglazr", 0)
                if region not in range(1, 7):
                    raise ValueError("CTGPROC legacy LULC=2 requires IGLAZR from 1 through 6")
                kind = ("GLAZNA", "GLAZSA", "GLAZEU", "GLAZAS", "GLAZAF", "GLAZAP")[region - 1]
            elif lulc in types: kind = types[lulc]
            else: raise ValueError(f"CTGPROC legacy LULC must be from 1 through 11, got {lulc}")
            items.append((kind, special["luindat"]))
        if items:
            landuse_sources: tuple[CtgprocSource, ...] = tuple(CtgprocSource(path, kind, _ctg_datum(kind, special),
                special.get("mesh." + kind, special.get("mesh.default", 1))) for kind, path in items)
            spelling = claimed.get("$source." + items[0][0], claimed.get("$luindat", "LUINDAT"))
            for target, value in (("landuse_sources", landuse_sources),
                ("landuse_files", tuple(item.path for item in landuse_sources)), ("input_type", landuse_sources[0].input_type),
                ("mesh_density", landuse_sources[0].mesh_density), ("input_datum", landuse_sources[0].input_datum)):
                assign(target, value, spelling)
        if ("source_count" in special and
                not (special["source_count"] == 0 and "luindat" in special) and
                special["source_count"] != len(items)):
            raise ValueError(f"CTGPROC NDBF declares {special['source_count']} files but {len(items)} are active")
        if special.get("lcoast") is False:
            values["coastline_file"] = None
            values["gshhs_file"] = None
            values["coastline_swap"] = False
            values["fill_empty_ocean"] = False
        elif special.get("lcoast") is True:
            values.setdefault("coastline_file", "coast.bln")
            if special.get("lblnread") is True:
                values["gshhs_file"] = None
            elif not values.get("gshhs_file"):
                values["gshhs_file"] = "GSHHS_F.B"
        if special.get("lprev") is False:
            values["previous_landuse"] = None
        elif special.get("lprev") is True:
            values.setdefault("previous_landuse", "prev.dat")

    elif cls is MakegeoConfig:
        if special.get("ifmtgeo", 1) != 1:
            raise ValueError("MAKEGEO only supports IFMTGEO=1")
        if "ixqa" in special or "iyqa" in special:
            if "ixqa" not in special or "iyqa" not in special:
                raise TypeError("MAKEGEO IXQA and IYQA must be supplied together")
            qa = ((special["ixqa"], special["iyqa"])
                  if special["ixqa"] > 0 and special["iyqa"] > 0 else None)
            assign("qa_cell", qa, claimed["$ixqa"])
        for count, target, label in (("nincat", "input_properties", "NINCAT"),
             ("noutcat", "output_categories", "NOUTCAT"), ("numwat", "water_categories", "NUMWAT")):
            if count in special and target in values and special[count] != len(values[target]):
                raise ValueError(f"MAKEGEO {label} does not match {target}")
        if "nsplit" in special and "category_splits" in values:
            if special["nsplit"] != len({item[0] for item in values["category_splits"]}):
                raise ValueError("MAKEGEO NSPLIT does not match category_splits")
        if special.get("llu2") is False:
            values["secondary_landuse"] = None
        elif special.get("llu2") is True:
            values.setdefault("secondary_landuse", "lu2.dat")
        if values.get("snow_processing"):
            values.setdefault("snow_data", "snow.dat")
            values.setdefault("snow_height_scale_m", 10.0)
            values.setdefault("snow_depth_minimum_m", 0.01)

    elif cls is CalmetConfig:
        parts = {key.removeprefix("date."): value for key, value in special.items()
                 if key.startswith("date.")}
        if parts:
            begin = _date(parts, "IB")
            if any(key.startswith("IE") for key in parts): end = _date(parts, "IE")
            elif "irlg" in special:
                begin -= timedelta(hours=1); end = begin + timedelta(hours=special["irlg"])
            else: raise TypeError("CALMET ending date fields or IRLG are required")
            assign("start", begin, "IBYR")
            assign("end", end, "IEYR")
        if "abtz" in special:
            zone, label = _timezone(special["abtz"])
            assign("base_timezone", zone, claimed["$abtz"])
            assign("base_timezone_label", label, claimed["$abtz"])
        if "icloud" in special:
            cloud = special["icloud"]
            assign("cloud_method", 1 if cloud in {0, 1} else cloud, claimed["$icloud"])
            assign("cloud_output", int(cloud == 1), claimed["$icloud"])
        if "isteppg" in special or "isteppgs" in special:
            seconds = special.get("isteppgs", 0)
            source = "$isteppgs" if "isteppgs" in special else "$isteppg"
            if seconds == 0:
                seconds = special.get("isteppg", 1) * 3600
                if "isteppg" in special:
                    source = "$isteppg"
            assign("prognostic_timestep_seconds", seconds, claimed[source])
        if ("mm4dat" in special and special.get("nm3d", 0) == 0 and
                "meteorology" not in values):
            assign("meteorology", special["mm4dat"], claimed["$mm4dat"])
            special["nm3d"] = 1
        if "nm3d" in special and "meteorology" in values:
            if special["nm3d"] != len(_files(values["meteorology"])):
                raise ValueError("CALMET NM3D does not match M3DDAT")
        if "nz" in special and "z_faces_m" in values:
            if special["nz"] != len(values["z_faces_m"]) - 1:
                raise ValueError("CALMET NZ does not match ZFACE")
        surface_fields = {int(key.split(".")[-1]): value for key, value in special.items()
                          if key.startswith("surface_field.")}
        if surface_fields:
            assign("print_surface_fields", tuple(surface_fields.get(index, 0)
                   for index in range(8)), claimed["$surface_field." + str(min(surface_fields))])
        debug_fields = {int(key.split(".")[-1]): value for key, value in special.items()
                        if key.startswith("wind_debug_field.")}
        if debug_fields:
            assign("wind_debug_fields", tuple(debug_fields.get(index, 0)
                   for index in range(9)), claimed["$wind_debug_field." + str(min(debug_fields))])

        station_specs = (("SS", "nssta", "surface_control_stations", 6),
                         ("US", "nusta", "upper_control_stations", 5),
                         ("PS", None, "precipitation_control_stations", 4))
        for prefix, count_key, target, width in station_specs:
            rows = {int(key.rsplit(".", 1)[1]): value for key, value in special.items()
                    if key.startswith("station." + prefix + ".")}
            count = (special.get(count_key, 0) if count_key else
                     values.get("precipitation_station_count", -1))
            if rows and count > 0:
                expected = list(range(1, count + 1))
                if sorted(rows) != expected:
                    raise ValueError(f"{prefix} station rows must cover 1..{count}")
                spelling = claimed[f"$station.{prefix}.{min(rows)}"]
                assign(target, tuple(_station(rows[index], width) for index in expected), spelling)

        barrier_names = ("xb", "yb", "xe", "ye")
        barrier_columns = [special.get("barrier." + name, ()) for name in barrier_names]
        count = values.get("barrier_count", 0)
        if any(barrier_columns) and count > 0:
            if any(len(column) != count for column in barrier_columns):
                raise ValueError("NBAR does not match XBBAR/YBBAR/XEBAR/YEBAR lengths")
            assign("barriers", tuple(zip(*barrier_columns)), claimed["$barrier.xb"])

        if special.get("lake.enabled", False):
            count = special.get("lake.count", 0)
            coordinate_names = ("xg1", "xg2", "yg1", "yg2", "xbcst", "ybcst",
                                "xecst", "yecst")
            columns = [special.get("lake." + name, ()) for name in coordinate_names]
            station_counts = special.get("lake.station_counts", ())
            station_ids = special.get("lake.station_ids", ())
            if count < 1 or any(len(column) != count for column in columns):
                raise ValueError("LLBREZE requires NBOX matching all lake-breeze coordinate arrays")
            if len(station_counts) != count or sum(station_counts) != len(station_ids):
                raise ValueError("lake-breeze NLB counts do not match NBOX/METBXID")
            supplied_grid = values.get("grid")
            x_origin = grid.get("x_origin_km", getattr(supplied_grid, "x_origin_km", 0.))
            y_origin = grid.get("y_origin_km", getattr(supplied_grid, "y_origin_km", 0.))
            regions: list[
                tuple[
                    float, float, float, float,
                    float, float, float, float,
                    tuple[int, ...],
                ]
            ] = []
            offset = 0
            for index, station_count in enumerate(station_counts):
                coordinates = [column[index] for column in columns]
                for position, origin in ((4, x_origin), (5, y_origin),
                                         (6, x_origin), (7, y_origin)):
                    coordinates[position] = (coordinates[position] - origin) * 1000.
                selected = tuple(station_ids[offset:offset + station_count])
                offset += station_count
                regions.append((*coordinates, selected))
            assign("lake_breeze_regions", tuple(regions), claimed["$lake.enabled"])

    return NormalizedParameters(values, grid, special)


__all__: Final[list[str]] = ["NormalizedParameters", "normalize_public_parameters"]
