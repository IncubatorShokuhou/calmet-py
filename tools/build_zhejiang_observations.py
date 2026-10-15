#!/usr/bin/env python3
"""Convert archived NOAA ISD/Global Hourly and IGRA v2 data to CALMET 2.1.

The converter is intentionally download-free: every raw file and its source
URL are explicit command-line inputs.  This keeps the generated CALMET files
reproducible and lets the accompanying manifest identify the exact archived
bytes used for a case.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import tempfile
from typing import Final, Mapping, Protocol, Sequence
from urllib.parse import urlparse
import zipfile


SURFACE_MISSING: Final[int] = 9999
UPPER_MISSING: Final[float] = 999.9
CALMET_LEGACY_MISSING_HEIGHT_M: Final[float] = 9998.0
READ_CHUNK_BYTES: Final[int] = 1024 * 1024
MAX_ARCHIVED_SOURCE_BYTES: Final[int] = 256 * 1024 * 1024
MAX_TEXT_SOURCE_BYTES: Final[int] = 256 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO: Final[float] = 64.0
REJECTED_ISD_QUALITY: Final[frozenset[str]] = frozenset({"3", "7"})
KNOWN_ISD_QUALITY: Final[frozenset[str]] = frozenset(
    {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
     "A", "C", "I", "M", "P", "R", "U"}
)
KNOWN_IGRA_QUALITY: Final[frozenset[str]] = frozenset({"", "A", "B"})
ISD_SKY_TO_TENTHS: Final[dict[int, int]] = {
    0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6,
    6: 8, 7: 9, 8: 10, 9: 10, 10: 10,
}
ISD_QUALITATIVE_SKY_TO_TENTHS: Final[dict[int, int]] = {
    0: 0, 1: 2, 2: 4, 3: 8, 4: 10, 5: 10, 6: 10,
}
ISD_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME"}
)
REJECTED_ISD_REPORT_TYPES: Final[frozenset[str]] = frozenset(
    {"BOGUS", "COOPD", "PCP15", "SOD", "SOM"}
)
ISD_MANUAL_LIQUID_WEATHER: Final[frozenset[int]] = frozenset(
    {20, 21, 23, 24, 25, *range(50, 70), *range(80, 85), 91, 92}
)
ISD_MANUAL_FROZEN_WEATHER: Final[frozenset[int]] = frozenset(
    {22, *range(36, 40), *range(70, 80), 85, 86, 96, 99}
)
ISD_MANUAL_INDETERMINATE_WEATHER: Final[frozenset[int]] = frozenset(
    {*range(26, 30), *range(87, 91), *range(93, 96), 97, 98}
)
ISD_AUTOMATED_LIQUID_WEATHER: Final[frozenset[int]] = frozenset(
    {
        21, 22, 23, 25,
        *range(40, 45), 47, 48,
        *range(50, 59), *range(60, 69),
        *range(80, 85),
    }
)
ISD_AUTOMATED_FROZEN_WEATHER: Final[frozenset[int]] = frozenset(
    {
        11, 24, 27, 28, 29, 45, 46,
        *range(70, 79), *range(85, 90), 93, 96,
    }
)
ISD_AUTOMATED_INDETERMINATE_WEATHER: Final[frozenset[int]] = frozenset(
    {
        *range(6, 10), *range(13, 18), 19, 20, 26,
        *range(36, 40), 49, 59, 69, 79, 90, 92, 95, 97, 98,
    }
)


class _BinaryReader(Protocol):
    """Structural type for binary streams used by bounded readers."""

    def read(self, size: int = -1, /) -> bytes:
        """Read at most ``size`` bytes from the stream."""
        ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Describe one archived source file and its public provenance URL.

    Attributes:
        path: Local archived file. Plain text, gzip, and single-member zip
            files are accepted.
        source_url: URL from which the archived bytes were obtained.
    """

    path: Path
    source_url: str


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """Hold decoded source text and integrity metadata."""

    spec: SourceSpec
    text: str
    sha256: str
    size_bytes: int
    archive_member: str | None


@dataclass(frozen=True, slots=True)
class SurfaceValues:
    """Decoded values from one NOAA Global Hourly report."""

    wind_speed_m_s: float | None
    wind_direction_deg: float | None
    ceiling_hundreds_ft: int | None
    cloud_tenths: int | None
    temperature_k: float | None
    relative_humidity_percent: int | None
    station_pressure_mb: float | None
    precipitation_code: int | None

    def completeness(self) -> int:
        """Return the number of decoded fields available in this report.

        Returns:
            Count of nonmissing wind, cloud, temperature, humidity, pressure, and
            precipitation fields.
        """
        fields: tuple[object | None, ...] = (
            self.wind_speed_m_s,
            self.wind_direction_deg,
            self.ceiling_hundreds_ft,
            self.cloud_tenths,
            self.temperature_k,
            self.relative_humidity_percent,
            self.station_pressure_mb,
            self.precipitation_code,
        )
        return sum(value is not None for value in fields)


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    """Associate a decoded ISD report with its UTC timestamp."""

    timestamp: datetime
    values: SurfaceValues
    report_type: str
    input_order: int
    latitude: float
    longitude: float
    elevation_m: float
    full_name: str


@dataclass(frozen=True, slots=True)
class SurfaceStationData:
    """Store one ISD station and all reports available for selection."""

    source: LoadedSource
    input_format: str
    source_station_id: str
    identifier: int
    full_name: str
    latitude: float
    longitude: float
    elevation_m: float
    reports: tuple[SurfaceCandidate, ...]


@dataclass(frozen=True, slots=True)
class UpperLevel:
    """One pressure-height level decoded from an IGRA v2 sounding."""

    level_type: str
    is_surface: bool
    pressure_mb: float
    height_msl: float
    temperature_k: float | None
    wind_direction_deg: float | None
    wind_speed_m_s: float | None

    @property
    def has_wind(self) -> bool:
        """Report whether this level contains a complete wind observation.

        Returns:
            ``True`` when both wind direction and wind speed are available.
        """
        return self.wind_direction_deg is not None and self.wind_speed_m_s is not None


@dataclass(frozen=True, slots=True)
class UpperSounding:
    """One selected IGRA sounding, including valid levels CALMET cannot emit."""

    timestamp: datetime
    latitude: float
    longitude: float
    levels: tuple[UpperLevel, ...]


@dataclass(frozen=True, slots=True)
class UpperStationData:
    """Store selected soundings and CALMET metadata for one IGRA station."""

    source: LoadedSource
    igra_id: str
    identifier: int
    latitude: float
    longitude: float
    elevation_m: float
    elevation_source: str
    soundings: tuple[UpperSounding, ...]


@dataclass(frozen=True, slots=True)
class RawUpperSounding:
    """Hold one IGRA sounding before CALMET vertical validation."""

    igra_id: str
    timestamp: datetime
    latitude: float | None
    longitude: float | None
    levels: tuple[UpperLevel, ...]
    missing_surface_height: bool


def _sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file.

    Args:
        path: File whose archived bytes are hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_spec(spec: SourceSpec) -> None:
    """Validate a raw-data path and its provenance URL.

    Args:
        spec: Local archive and the public URL from which it was obtained.

    Raises:
        FileNotFoundError: If ``spec.path`` is not a regular file.
        ValueError: If ``source_url`` is not an absolute HTTP or HTTPS URL.
    """
    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    parsed = urlparse(spec.source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"source URL must be absolute HTTP(S): {spec.source_url!r}")


def _read_limited(
    stream: _BinaryReader,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read a binary stream without allowing unbounded memory growth.

    Args:
        stream: Binary input stream positioned at its payload.
        max_bytes: Maximum payload size accepted in bytes.
        label: Human-readable source description used in an error message.

    Returns:
        The complete payload, provided it does not exceed ``max_bytes``.

    Raises:
        ValueError: If ``max_bytes`` is not positive or the payload exceeds it.
        OSError: If the stream cannot be read.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    chunks: list[bytes] = []
    total = 0
    while True:
        request_size = min(READ_CHUNK_BYTES, max_bytes - total + 1)
        chunk = stream.read(request_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"{label} expands beyond the {max_bytes}-byte safety limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_zip_member(info: zipfile.ZipInfo, path: Path) -> None:
    """Reject unsafe or excessively compressed single-member ZIP payloads.

    Args:
        info: Metadata for the sole ZIP member.
        path: Archive path used in diagnostics.

    Raises:
        ValueError: If the member is a symbolic link, is too large, has an
            invalid compressed size, or exceeds the compression-ratio limit.
    """
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise ValueError(f"archive member must not be a symbolic link: {path}")
    if info.file_size > MAX_TEXT_SOURCE_BYTES:
        raise ValueError(
            f"archive member in {path} is {info.file_size} bytes; maximum is "
            f"{MAX_TEXT_SOURCE_BYTES}"
        )
    if info.file_size > 0 and info.compress_size <= 0:
        raise ValueError(f"archive member in {path} has an invalid compressed size")
    ratio = info.file_size / max(1, info.compress_size)
    if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
        raise ValueError(
            f"archive member in {path} has compression ratio {ratio:.1f}; "
            f"maximum is {MAX_ARCHIVE_COMPRESSION_RATIO:g}"
        )


def _load_text(spec: SourceSpec, *, encoding: str) -> LoadedSource:
    """Read plain, gzip, or single-member zip text without changing bytes.

    Args:
        spec: Local file and public provenance URL.
        encoding: Character encoding of the uncompressed payload.

    Returns:
        Decoded text plus integrity metadata for the original archived bytes.

    Raises:
        OSError: If the source path cannot be opened or read.
        ValueError: If the URL, archive contents, or text encoding is invalid.
    """
    _validate_source_spec(spec)
    archived_size = spec.path.stat().st_size
    if archived_size > MAX_ARCHIVED_SOURCE_BYTES:
        raise ValueError(
            f"source {spec.path} is {archived_size} bytes; maximum archived "
            f"size is {MAX_ARCHIVED_SOURCE_BYTES}"
        )
    member: str | None = None
    if zipfile.is_zipfile(spec.path):
        with zipfile.ZipFile(spec.path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise ValueError(f"archive must contain exactly one file: {spec.path}")
            info = members[0]
            _validate_zip_member(info, spec.path)
            member = info.filename
            with archive.open(info) as stream:
                raw = _read_limited(
                    stream,
                    max_bytes=MAX_TEXT_SOURCE_BYTES,
                    label=f"archive member in {spec.path}",
                )
    elif spec.path.suffix.lower() == ".gz":
        with gzip.open(spec.path, "rb") as stream:
            raw = _read_limited(
                stream,
                max_bytes=MAX_TEXT_SOURCE_BYTES,
                label=f"gzip payload in {spec.path}",
            )
    else:
        with spec.path.open("rb") as stream:
            raw = _read_limited(
                stream,
                max_bytes=MAX_TEXT_SOURCE_BYTES,
                label=f"text payload in {spec.path}",
            )
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot decode {spec.path.name} as {encoding}") from error
    return LoadedSource(
        spec=spec,
        text=text,
        sha256=_sha256(spec.path),
        size_bytes=archived_size,
        archive_member=member,
    )


def parse_utc(text: str) -> datetime:
    """Parse an explicitly UTC ISO-8601 instant into a naive UTC datetime.

    Args:
        text: ISO-8601 text ending in ``Z`` or carrying a zero UTC offset.

    Returns:
        A timezone-naive datetime whose fields are UTC.

    Raises:
        ValueError: If the value lacks an offset or is not UTC.
    """
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        value = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid ISO-8601 time {text!r}") from error
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"time must carry an explicit UTC offset: {text!r}")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_isd_datetime(text: str) -> datetime:
    """Parse a NOAA ISD ``DATE`` value as UTC.

    Args:
        text: ISO-8601 date and time from Global Hourly or a converted annual
            fixed-width record. A missing offset is interpreted as UTC because
            the ISD field is defined in UTC.

    Returns:
        Timezone-naive UTC datetime.

    Raises:
        ValueError: If the text is malformed or carries a non-UTC offset.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("empty ISD DATE")
    if normalized.endswith(("Z", "z")):
        return parse_utc(normalized)
    try:
        value = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid ISD DATE {text!r}") from error
    if value.tzinfo is not None:
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"ISD DATE is not UTC: {text!r}")
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _quality_valid(code: str) -> bool:
    """Check whether an ISD quality flag permits use of a value.

    Args:
        code: One-character ISD quality flag.

    Returns:
        ``True`` for a recognized flag that is not classified as erroneous.
    """
    flag = code.strip().upper()
    return flag in KNOWN_ISD_QUALITY and flag not in REJECTED_ISD_QUALITY


def _group(row: dict[str, str], name: str) -> list[str]:
    """Split one optional ISD additional-data group.

    Args:
        row: Decoded ISD row keyed by group name.
        name: Additional-data group name, such as ``WND`` or ``CIG``.

    Returns:
        Whitespace-stripped comma-delimited fields, or an empty list when the
        group is absent.
    """
    value = row.get(name, "").strip()
    return [] if not value else [item.strip() for item in value.split(",")]


def _scaled_isd(
    encoded: str,
    quality: str,
    *,
    missing: frozenset[str],
    scale: float,
) -> float | None:
    """Decode one quality-controlled fixed-scale ISD number.

    Args:
        encoded: Signed or unsigned integer text from ISD.
        quality: ISD quality flag associated with ``encoded``.
        missing: Encoded sentinel strings that represent missing data.
        scale: Multiplier that converts the encoded integer to physical units.

    Returns:
        Scaled value, or ``None`` for a sentinel, invalid quality flag, or
        malformed integer.
    """
    text = encoded.strip()
    if text in missing or not _quality_valid(quality):
        return None
    if re.fullmatch(r"[+-]?\d+", text) is None:
        return None
    return int(text) * scale


def _temperature_c(row: dict[str, str], field: str) -> float | None:
    """Decode one ISD temperature group.

    Args:
        row: Decoded ISD row.
        field: Temperature group name, normally ``TMP`` or ``DEW``.

    Returns:
        Quality-controlled temperature in degrees Celsius, or ``None`` when the
        group is absent, invalid, or outside the supported physical range.
    """
    parts = _group(row, field)
    if len(parts) < 2:
        return None
    value = _scaled_isd(
        parts[0], parts[1], missing=frozenset({"+9999", "-9999", "9999"}),
        scale=0.1,
    )
    if value is None or not -120.0 <= value <= 60.0:
        return None
    return value


def _wind(row: dict[str, str]) -> tuple[float | None, float | None]:
    """Decode the ISD wind group for CALMET surface input.

    Args:
        row: Decoded ISD row containing the optional ``WND`` group.

    Returns:
        Pair ``(speed_m_s, direction_degrees)``. Both values are ``None`` for
        variable, missing, invalid, or implausible wind; calm is ``(0, 0)``.
    """
    parts = _group(row, "WND")
    if len(parts) < 5:
        return None, None
    wind_type = parts[2].upper()
    if wind_type == "C":
        return 0.0, 0.0
    if wind_type in {"V", "9"}:
        return None, None
    speed = _scaled_isd(
        parts[3], parts[4], missing=frozenset({"9999"}), scale=0.1,
    )
    if speed is None or speed < 0.0 or speed > 150.0:
        return None, None
    if speed < 0.1:
        return 0.0, 0.0
    direction = _scaled_isd(
        parts[0], parts[1], missing=frozenset({"999"}), scale=1.0,
    )
    if direction is None or not 0.0 <= direction <= 360.0:
        return None, None
    return speed, direction


def _ceiling(row: dict[str, str]) -> int | None:
    """Decode the ISD ceiling group.

    Args:
        row: Decoded ISD row containing the optional ``CIG`` group.

    Returns:
        Rounded cloud ceiling in hundreds of feet, ``999`` for NOAA's unlimited
        ceiling, or ``None`` when no valid ceiling is available.
    """
    parts = _group(row, "CIG")
    if len(parts) < 2:
        return None
    metres = _scaled_isd(
        parts[0], parts[1], missing=frozenset({"99999"}), scale=1.0,
    )
    if metres is None or metres < 0.0:
        return None
    if metres == 22000.0:
        # NOAA's unlimited-ceiling code. SURFGEN writes 999 hundred feet,
        # which remains nonmissing to CALMET's MISSFC validation.
        return 999
    return int(math.floor(metres / 30.48 + 0.5))


def _sky_cover(
    parts: Sequence[str],
    value_index: int,
    quality_index: int,
    mapping: dict[int, int] = ISD_SKY_TO_TENTHS,
) -> int | None:
    """Decode one quality-controlled ISD sky-cover code.

    Args:
        parts: Fields from an ISD cloud group.
        value_index: Index of the encoded sky-cover field.
        quality_index: Index of the quality flag for that field.
        mapping: Mapping from the ISD code to CALMET tenths.

    Returns:
        Cloud cover in tenths, or ``None`` when the group or flag is unusable.
    """
    if len(parts) <= max(value_index, quality_index):
        return None
    encoded = parts[value_index].strip()
    if not _quality_valid(parts[quality_index]) or re.fullmatch(r"\d{1,2}", encoded) is None:
        return None
    return mapping.get(int(encoded))


def _summation_cloud_tenths(row: dict[str, str]) -> int | None:
    """Decode ISD sky-cover summation groups.

    Args:
        row: Decoded ISD row containing zero or more ``GD`` groups.

    Returns:
        The last usable summation state in cloud-cover tenths, or ``None`` when
        no valid state is present.
    """
    selected: int | None = None
    for index in range(1, 7):
        parts = _group(row, f"GD{index}")
        secondary = _sky_cover(parts, 1, 2)
        primary = _sky_cover(
            parts, 0, 2, mapping=ISD_QUALITATIVE_SKY_TO_TENTHS,
        )
        value = secondary if secondary is not None else primary
        if value is not None:
            selected = value
    return selected


def _cloud_tenths(row: dict[str, str]) -> int | None:
    """Select the best available ISD cloud-cover observation.

    Args:
        row: Decoded ISD row containing optional ``GF``, ``GD``, and ``GA``
            groups.

    Returns:
        Opaque or total cloud cover in tenths, or ``None`` when all supported
        groups are missing or invalid.
    """
    gf1 = _group(row, "GF1")
    # GF uses one shared quality flag after total and opaque cover.
    opaque = _sky_cover(gf1, 1, 2)
    total = _sky_cover(gf1, 0, 2)
    selected = opaque if opaque is not None else total
    if selected is None:
        selected = _summation_cloud_tenths(row)
    if selected is None:
        layers = [
            value
            for index in range(1, 7)
            if (value := _sky_cover(_group(row, f"GA{index}"), 0, 1)) is not None
        ]
        selected = max(layers) if layers else None
    return selected


def _station_pressure(
    row: dict[str, str], elevation_m: float, temperature_k: float | None
) -> float | None:
    """Obtain pressure at station elevation from an ISD report.

    Args:
        row: Decoded ISD row containing optional ``MA1`` and ``SLP`` groups.
        elevation_m: Station elevation in metres above mean sea level.
        temperature_k: Observed air temperature in kelvin, used when reducing
            sea-level pressure; 288.15 K is used when it is unavailable.

    Returns:
        Direct or reduced station pressure in millibars, or ``None`` when no
        supported pressure value can be validated.
    """
    ma1 = _group(row, "MA1")
    if len(ma1) >= 4:
        station = _scaled_isd(
            ma1[2], ma1[3], missing=frozenset({"99999"}), scale=0.1,
        )
        if station is not None and 100.0 <= station <= 1100.0:
            return station
        altimeter = _scaled_isd(
            ma1[0], ma1[1], missing=frozenset({"99999"}), scale=0.1,
        )
        if altimeter is not None and 800.0 <= altimeter <= 1100.0:
            base = altimeter ** 0.19026 - 8.41717e-5 * elevation_m
            if base > 0.0:
                reduced = float(base ** 5.25593)
                if 100.0 <= reduced <= 1100.0:
                    return reduced
    slp = _group(row, "SLP")
    if len(slp) >= 2:
        sea_level = _scaled_isd(
            slp[0], slp[1], missing=frozenset({"99999"}), scale=0.1,
        )
        if sea_level is not None and 800.0 <= sea_level <= 1100.0:
            reference_temperature = 288.15 if temperature_k is None else temperature_k
            reduced = sea_level * math.exp(
                -elevation_m / (29.3 * reference_temperature)
            )
            if 100.0 <= reduced <= 1100.0:
                return reduced
    return None


def _relative_humidity(temperature_c: float | None, dewpoint_c: float | None) -> int | None:
    """Calculate relative humidity from temperature and dew point.

    Args:
        temperature_c: Air temperature in degrees Celsius.
        dewpoint_c: Dew-point temperature in degrees Celsius.

    Returns:
        Relative humidity rounded to an integer percent and bounded to 0--100,
        or ``None`` when either temperature is unavailable.
    """
    if temperature_c is None or dewpoint_c is None:
        return None
    dewpoint = min(dewpoint_c, temperature_c)
    vapor = 6.112 * math.exp(17.67 * dewpoint / (dewpoint + 243.5))
    saturation = 6.112 * math.exp(17.67 * temperature_c / (temperature_c + 243.5))
    value = min(100.0, max(0.0, 100.0 * vapor / saturation))
    return int(math.floor(value + 0.5))


def _weather_codes(row: dict[str, str]) -> tuple[tuple[str, int], ...]:
    """Collect usable ISD present-weather codes.

    Args:
        row: Decoded ISD row containing automated ``AW`` and manual ``MW``
            groups.

    Returns:
        Ordered ``(group_prefix, code)`` pairs whose quality flags are valid.
    """
    codes: list[tuple[str, int]] = []
    for prefix in ("AW", "MW"):
        for index in range(1, 8):
            parts = _group(row, f"{prefix}{index}")
            if len(parts) < 2 or not _quality_valid(parts[1]):
                continue
            if re.fullmatch(r"\d{2}", parts[0]) is not None:
                codes.append((prefix, int(parts[0])))
    return tuple(codes)


def _precipitation_code(row: dict[str, str], temperature_k: float | None) -> int:
    """Diagnose CALMET precipitation phase from an ISD report.

    Args:
        row: Decoded ISD row containing accumulation and present-weather groups.
        temperature_k: Air temperature in kelvin for unresolved precipitation
            phase.

    Returns:
        CALMET code 0 for none, 10 for liquid, or 20 for frozen precipitation.
    """
    active = False
    for index in range(1, 5):
        parts = _group(row, f"AA{index}")
        if len(parts) < 4 or parts[0] != "01" or not _quality_valid(parts[3]):
            continue
        if re.fullmatch(r"\d{4}", parts[1]) is None or parts[1] == "9999":
            continue
        active = int(parts[1]) > 0 or parts[2] == "1"
        if active:
            break
    weather = _weather_codes(row)
    liquid = any(
        code in (
            ISD_AUTOMATED_LIQUID_WEATHER
            if prefix == "AW" else ISD_MANUAL_LIQUID_WEATHER
        )
        for prefix, code in weather
    )
    frozen = any(
        code in (
            ISD_AUTOMATED_FROZEN_WEATHER
            if prefix == "AW" else ISD_MANUAL_FROZEN_WEATHER
        )
        for prefix, code in weather
    )
    indeterminate = any(
        code in (
            ISD_AUTOMATED_INDETERMINATE_WEATHER
            if prefix == "AW" else ISD_MANUAL_INDETERMINATE_WEATHER
        )
        for prefix, code in weather
    )
    if liquid:
        return 10
    if frozen:
        return 20
    if not active and not indeterminate:
        return 0
    below_freezing = temperature_k is not None and temperature_k < 273.15
    return 20 if below_freezing else 10


def decode_isd_row(row: dict[str, str], elevation_m: float) -> SurfaceValues:
    """Decode one NOAA Global Hourly row into CALMET surface units.

    Args:
        row: Mapping produced by :class:`csv.DictReader`.
        elevation_m: Station elevation used to reduce altimeter or sea-level
            pressure when a direct station pressure is unavailable.

    Returns:
        Quality-controlled values; unavailable fields are represented by
        ``None`` until file serialization converts them to 9999.
    """
    speed, direction = _wind(row)
    temperature_c = _temperature_c(row, "TMP")
    dewpoint_c = _temperature_c(row, "DEW")
    temperature_k = None if temperature_c is None else temperature_c + 273.15
    cloud_tenths = _cloud_tenths(row)
    ceiling_hundreds_ft = _ceiling(row)
    cig = _group(row, "CIG")
    cavok = len(cig) >= 4 and cig[3].upper() == "Y"
    if cavok and cloud_tenths is None:
        cloud_tenths = 0
    if cavok and ceiling_hundreds_ft is None:
        ceiling_hundreds_ft = 999
    if cloud_tenths == 0 and ceiling_hundreds_ft is None:
        # A clear report has no physical cloud base. Use SURFGEN's finite
        # unlimited-ceiling code so CALMET does not reject the observation as
        # missing before its cloud-cover logic can ignore the ceiling.
        ceiling_hundreds_ft = 999
    return SurfaceValues(
        wind_speed_m_s=speed,
        wind_direction_deg=direction,
        ceiling_hundreds_ft=ceiling_hundreds_ft,
        cloud_tenths=cloud_tenths,
        temperature_k=temperature_k,
        relative_humidity_percent=_relative_humidity(temperature_c, dewpoint_c),
        station_pressure_mb=_station_pressure(row, elevation_m, temperature_k),
        precipitation_code=_precipitation_code(row, temperature_k),
    )


def _surface_identifier(source_station_id: str) -> int:
    """Derive a numeric CALMET station identifier from NOAA text.

    Args:
        source_station_id: NOAA ``USAF+WBAN`` station identifier.

    Returns:
        USAF identifier, or the WBAN identifier when USAF is the missing sentinel.

    Raises:
        ValueError: If neither a usable six-digit USAF nor five-digit WBAN
            identifier is present.
    """
    digits = re.sub(r"\D", "", source_station_id)
    if len(digits) < 6:
        raise ValueError(f"invalid ISD station identifier {source_station_id!r}")
    usaf = digits[:6]
    if usaf != "999999":
        return int(usaf)
    if len(digits) >= 11 and digits[-5:] != "99999":
        return int(digits[-5:])
    raise ValueError(f"ISD station has neither a USAF nor WBAN identifier: {source_station_id!r}")


def _fixed_isd_group(
    additional: str,
    tag: str,
    widths: Sequence[int],
) -> str:
    """Extract an ISD additional-data group from a fixed-width record.

    Args:
        additional: Additional-data portion following the ``ADD`` marker.
        tag: Group tag to locate, such as ``GA1`` or ``MA1``.
        widths: Fixed field widths following the tag.

    Returns:
        Comma-delimited group fields, or an empty string when the group is absent
        or truncated.
    """
    index = additional.find(tag)
    if index < 0:
        return ""
    payload_start = index + len(tag)
    payload_end = payload_start + sum(widths)
    if payload_end > len(additional):
        return ""
    payload = additional[payload_start:payload_end]
    values: list[str] = []
    cursor = 0
    for width in widths:
        values.append(payload[cursor:cursor + width])
        cursor += width
    return ",".join(values)


def _fixed_isd_row(line: str, line_number: int) -> dict[str, str]:
    """Convert one official annual ISD record to decoder fields.

    Args:
        line: One fixed-width record, including or excluding its line ending.
        line_number: One-based source line number used in validation errors.

    Returns:
        Mapping compatible with the Global Hourly CSV decoder.

    Raises:
        ValueError: If the record header, timestamp, or ``ADD`` marker is invalid.
    """
    record = line.rstrip("\r\n")
    if len(record) < 108 or not record[:4].isdigit():
        raise ValueError(f"invalid fixed-width ISD record at line {line_number}")
    if record[105:108] != "ADD":
        raise ValueError(f"fixed-width ISD record lacks ADD at line {line_number}")
    date = record[15:23]
    time = record[23:27]
    if re.fullmatch(r"\d{8}", date) is None or re.fullmatch(r"\d{4}", time) is None:
        raise ValueError(f"invalid fixed-width ISD timestamp at line {line_number}")
    timestamp = (
        f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{time[0:2]}:{time[2:4]}:00"
    )
    _parse_isd_datetime(timestamp)
    source_id = record[4:15]
    call_sign = record[51:56].strip()
    name = source_id if call_sign in {"", "99999"} else call_sign
    marker_positions = [
        position for marker in ("REM", "EQD")
        if (position := record.find(marker, 108)) >= 0
    ]
    end = min(marker_positions, default=len(record))
    additional = record[108:end]
    row: dict[str, str] = {
        "STATION": source_id,
        "DATE": timestamp,
        "LATITUDE": str(int(record[28:34]) / 1000.0),
        "LONGITUDE": str(int(record[34:41]) / 1000.0),
        "ELEVATION": str(int(record[46:51])),
        "NAME": name,
        "REPORT_TYPE": record[41:46].strip(),
        "WND": ",".join((
            record[60:63], record[63:64], record[64:65],
            record[65:69], record[69:70],
        )),
        "CIG": ",".join((
            record[70:75], record[75:76], record[76:77], record[77:78],
        )),
        "TMP": ",".join((record[87:92], record[92:93])),
        "DEW": ",".join((record[93:98], record[98:99])),
        "SLP": ",".join((record[99:104], record[104:105])),
        "GF1": _fixed_isd_group(
            additional, "GF1", (2, 2, 1, 2, 1, 2, 1, 5, 1, 2, 1, 2, 1)
        ),
        "MA1": _fixed_isd_group(additional, "MA1", (5, 1, 5, 1)),
    }
    for prefix, stop, widths in (
        ("GA", 6, (2, 1, 6, 1, 2, 1)),
        ("GD", 6, (1, 2, 1, 6, 1, 1)),
        ("AA", 4, (2, 4, 1, 1)),
        ("AW", 7, (2, 1)),
        ("MW", 7, (2, 1)),
    ):
        for index in range(1, stop + 1):
            tag = f"{prefix}{index}"
            row[tag] = _fixed_isd_group(additional, tag, widths)
    return row


def _isd_rows(source: LoadedSource) -> tuple[list[dict[str, str]], str]:
    """Decode either Global Hourly CSV or annual fixed-width ISD text.

    Args:
        source: Loaded and decoded NOAA source plus provenance metadata.

    Returns:
        Pair containing decoded row mappings and the detected input-format label.

    Raises:
        ValueError: If a CSV lacks required columns or a fixed-width record is
            malformed.
    """
    first = next((line for line in source.text.splitlines() if line.strip()), "")
    normalized = first.lstrip()
    if normalized.startswith("STATION,") or normalized.startswith('"STATION"'):
        reader = csv.DictReader(io.StringIO(source.text))
        fields = frozenset(reader.fieldnames or ())
        missing_columns = sorted(ISD_REQUIRED_COLUMNS - fields)
        if missing_columns:
            raise ValueError(
                f"{source.spec.path.name} lacks required ISD columns: "
                f"{', '.join(missing_columns)}"
            )
        return list(reader), "global_hourly_csv"
    rows = [
        _fixed_isd_row(line, number)
        for number, line in enumerate(source.text.splitlines(), 1)
        if line.strip()
    ]
    return rows, "isd_fixed_width"


def _surface_station_from_rows(
    source: LoadedSource,
    input_format: str,
    source_id: str,
    rows: Sequence[dict[str, str]],
) -> SurfaceStationData:
    """Build one surface station from its ISD report rows.

    Args:
        source: Loaded source shared by the report rows.
        input_format: Detected input-format label for provenance.
        source_id: NOAA station identifier common to ``rows``.
        rows: Report mappings belonging to the station.

    Returns:
        Valid station metadata and all usable candidate reports.

    Raises:
        ValueError: If station location, timestamps, identifier, or report data
            do not yield a usable station.
    """
    metadata_row: dict[str, str] | None = None
    for row in rows:
        try:
            latitude = float(row["LATITUDE"])
            longitude = float(row["LONGITUDE"])
            elevation = float(row["ELEVATION"])
        except (TypeError, ValueError):
            continue
        if (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0
                and -500.0 <= elevation < 9000.0):
            metadata_row = row
            break
    if metadata_row is None:
        raise ValueError(f"ISD station {source_id} has no valid location")
    latitude = float(metadata_row["LATITUDE"])
    longitude = float(metadata_row["LONGITUDE"])
    elevation = float(metadata_row["ELEVATION"])
    reports: list[SurfaceCandidate] = []
    for order, row in enumerate(rows):
        report_type = row.get("REPORT_TYPE", "").strip()
        if report_type.upper() in REJECTED_ISD_REPORT_TYPES:
            continue
        try:
            report_latitude = float(row["LATITUDE"])
            report_longitude = float(row["LONGITUDE"])
            report_elevation = float(row["ELEVATION"])
        except (TypeError, ValueError):
            continue
        if not (-90.0 <= report_latitude <= 90.0
                and -180.0 <= report_longitude <= 180.0
                and -500.0 <= report_elevation < 9000.0):
            continue
        timestamp = _parse_isd_datetime(row["DATE"])
        reports.append(SurfaceCandidate(
            timestamp=timestamp,
            values=decode_isd_row(row, report_elevation),
            report_type=report_type,
            input_order=order,
            latitude=report_latitude,
            longitude=report_longitude,
            elevation_m=report_elevation,
            full_name=row["NAME"].strip() or source_id,
        ))
    if not reports:
        raise ValueError(f"ISD station {source_id} contains no usable reports")
    return SurfaceStationData(
        source=source,
        input_format=input_format,
        source_station_id=source_id,
        identifier=_surface_identifier(source_id),
        full_name=metadata_row["NAME"].strip() or source_id,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation,
        reports=tuple(reports),
    )


def read_isd_stations(spec: SourceSpec) -> tuple[SurfaceStationData, ...]:
    """Read all stations from Global Hourly CSV or fixed-width ISD data.

    Args:
        spec: CSV/fixed-width file or archive and its public provenance URL.

    Returns:
        One decoded station per distinct NOAA station identifier. A Global
        Hourly CSV may contain several stations; an annual fixed-width file
        normally contains one.

    Raises:
        OSError: If the source file cannot be opened or read.
        ValueError: If the schema, station identity, metadata, timestamps, or
            meteorological fields are invalid.
    """
    source = _load_text(spec, encoding="utf-8-sig")
    rows, input_format = _isd_rows(source)
    if not rows:
        raise ValueError(f"empty ISD source: {spec.path}")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source_id = row.get("STATION", "").strip()
        if not source_id:
            raise ValueError(f"ISD row in {spec.path} has no station identifier")
        grouped.setdefault(source_id, []).append(row)
    return tuple(
        _surface_station_from_rows(source, input_format, source_id, station_rows)
        for source_id, station_rows in grouped.items()
    )


def read_isd_station(spec: SourceSpec) -> SurfaceStationData:
    """Read an ISD source that must contain exactly one station.

    Args:
        spec: CSV/fixed-width file or archive and its public provenance URL.

    Returns:
        The sole decoded surface station.

    Raises:
        OSError: If the source file cannot be opened or read.
        ValueError: If parsing fails or the source contains zero or multiple
            station identifiers.
    """
    stations = read_isd_stations(spec)
    if len(stations) != 1:
        raise ValueError(f"ISD source contains {len(stations)} stations: {spec.path}")
    return stations[0]


def _report_priority(report_type: str) -> int:
    """Rank an ISD report type for deterministic tie-breaking.

    Args:
        report_type: ISD report-type code, such as ``FM-12`` or ``FM-15``.

    Returns:
        Lower priority number for preferred synoptic/METAR types; 99 for an
        unrecognized type.
    """
    order: dict[str, int] = {
        "FM-12": 0,
        "FM-15": 1,
        "FM-16": 2,
        "SAO": 3,
    }
    return order.get(report_type.upper(), 99)


def select_surface_report(
    station: SurfaceStationData,
    target: datetime,
    tolerance: timedelta,
) -> SurfaceCandidate | None:
    """Select the nearest, most complete ISD report for one target hour.

    Args:
        station: Source station containing candidate reports.
        target: Exact naive UTC target hour.
        tolerance: Maximum absolute offset accepted around ``target``.

    Returns:
        The deterministic best candidate, or ``None`` when none is close
        enough.
    """
    candidates = [
        report for report in station.reports
        if abs(report.timestamp - target) <= tolerance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda report: (
            abs((report.timestamp - target).total_seconds()),
            -report.values.completeness(),
            _report_priority(report.report_type),
            report.input_order,
        ),
    )


def _fixed_integer(line: str, begin: int, end: int, label: str) -> int:
    """Decode one integer field from an IGRA record.

    Args:
        line: Fixed-width IGRA record.
        begin: Zero-based inclusive field offset.
        end: Zero-based exclusive field offset.
        label: Field name included in an error message.

    Returns:
        Decoded integer.

    Raises:
        ValueError: If the selected field is not valid integer text.
    """
    field = line[begin:end].strip()
    try:
        return int(field)
    except ValueError as error:
        raise ValueError(f"invalid IGRA {label} field {field!r}") from error


def _optional_igra_coordinate(raw: int, *, latitude: bool) -> float | None:
    """Decode an optional IGRA coordinate.

    Args:
        raw: Coordinate in ten-thousandths of a degree or an IGRA sentinel.
        latitude: Apply latitude bounds when true, otherwise longitude bounds.

    Returns:
        Decimal degrees, or ``None`` for a missing or out-of-range coordinate.
    """
    if raw in {-9999, -8888}:
        return None
    value = raw / 10000.0
    limit = 90.0 if latitude else 180.0
    return value if -limit <= value <= limit else None


def _igra_quality_valid(flag: str) -> bool:
    """Check whether an IGRA processing flag permits use of a value.

    Args:
        flag: IGRA pressure, height, or temperature processing flag.

    Returns:
        ``True`` for blank, ``A``, or ``B``; ``False`` otherwise.
    """
    return flag.strip().upper() in KNOWN_IGRA_QUALITY


def parse_igra_level(
    line: str,
    station_elevation_m: float | None = None,
) -> UpperLevel | None:
    """Decode one IGRA v2 fixed-width level record.

    Non-pressure levels and levels without pressure or geopotential height are
    discarded. A missing height on an explicit surface level can be filled by
    a documented station-elevation override. IGRA values removed by quality
    assurance (``-8888``), missing values (``-9999``), and unknown processing
    flags are not used.

    Args:
        line: One official IGRA v2 fixed-width level record.
        station_elevation_m: Optional documented MSL elevation used only for
            an explicit ``LVLTYP2=1`` surface record with missing GPH.

    Returns:
        Converted level in CALMET units, or ``None`` for an unusable level.

    Raises:
        ValueError: If a required fixed-width integer field is malformed.
    """
    padded = line.ljust(52)
    level_type = padded[0:1]
    minor_level_type = padded[1:2]
    if level_type not in {"1", "2"} or minor_level_type not in {"0", "1", "2"}:
        return None
    pressure_pa = _fixed_integer(padded, 9, 15, "pressure")
    height_raw = _fixed_integer(padded, 16, 21, "height")
    if (pressure_pa in {-9999, -8888} or pressure_pa <= 0
            or not _igra_quality_valid(padded[15:16])):
        return None
    is_surface = minor_level_type == "1"
    height: float
    if height_raw in {-9999, -8888}:
        if not is_surface or station_elevation_m is None:
            return None
        height = station_elevation_m
    elif not _igra_quality_valid(padded[21:22]):
        return None
    else:
        height = float(height_raw)
    temperature_raw = _fixed_integer(padded, 22, 27, "temperature")
    direction_raw = _fixed_integer(padded, 40, 45, "wind direction")
    speed_raw = _fixed_integer(padded, 46, 51, "wind speed")
    temperature: float | None
    if (temperature_raw in {-9999, -8888}
            or not _igra_quality_valid(padded[27:28])):
        temperature = None
    else:
        temperature = temperature_raw / 10.0 + 273.15
        if not 100.0 <= temperature <= 350.0:
            temperature = None
    direction: float | None
    speed: float | None
    if direction_raw in {-9999, -8888} or speed_raw in {-9999, -8888}:
        direction = speed = None
    elif not 0 <= direction_raw <= 360 or not 0 <= speed_raw <= 2000:
        direction = speed = None
    else:
        direction = float(direction_raw)
        speed = speed_raw / 10.0
        if speed < 0.1:
            direction = 0.0
            speed = 0.0
    return UpperLevel(
        level_type=level_type,
        is_surface=is_surface,
        pressure_mb=pressure_pa / 100.0,
        height_msl=height,
        temperature_k=temperature,
        wind_direction_deg=direction,
        wind_speed_m_s=speed,
    )


def _raw_igra_soundings(
    source: LoadedSource,
    station_elevations_m: Mapping[str, float],
) -> tuple[RawUpperSounding, ...]:
    """Parse complete sounding blocks from one IGRA v2 source.

    Args:
        source: Loaded ASCII IGRA fixed-width source.
        station_elevations_m: Documented station elevations keyed by IGRA ID,
            used only when an explicit surface record has missing height.

    Returns:
        Soundings in source order before vertical coverage validation.

    Raises:
        ValueError: If headers, fixed-width fields, station identity, sounding
            length, or timestamps are malformed, or no soundings are present.
    """
    lines = source.text.splitlines()
    soundings: list[RawUpperSounding] = []
    index = 0
    expected_id: str | None = None
    while index < len(lines):
        header = lines[index]
        index += 1
        if not header.strip():
            continue
        if not header.startswith("#"):
            raise ValueError(f"IGRA record does not begin with '#': line {index}")
        padded = header.ljust(71)
        igra_id = padded[1:12].strip()
        if not igra_id:
            raise ValueError(f"missing IGRA station ID at line {index}")
        if expected_id is None:
            expected_id = igra_id
        elif igra_id != expected_id:
            raise ValueError(f"IGRA file contains multiple stations: {source.spec.path}")
        year = _fixed_integer(padded, 13, 17, "year")
        month = _fixed_integer(padded, 18, 20, "month")
        day = _fixed_integer(padded, 21, 23, "day")
        hour = _fixed_integer(padded, 24, 26, "hour")
        count = _fixed_integer(padded, 32, 36, "level count")
        if count < 0 or index + count > len(lines):
            raise ValueError(f"truncated IGRA sounding at line {index}")
        level_lines = lines[index:index + count]
        index += count
        if not 0 <= hour <= 23:
            continue
        try:
            timestamp = datetime(year, month, day, hour)
        except ValueError as error:
            raise ValueError(f"invalid IGRA sounding timestamp at line {index - count}") from error
        latitude = _optional_igra_coordinate(
            _fixed_integer(padded, 55, 62, "latitude"), latitude=True
        )
        longitude = _optional_igra_coordinate(
            _fixed_integer(padded, 63, 71, "longitude"), latitude=False
        )
        elevation = station_elevations_m.get(igra_id)
        missing_surface_height = any(
            item.ljust(52)[1:2] == "1"
            and _fixed_integer(item.ljust(52), 16, 21, "height") in {-9999, -8888}
            for item in level_lines
        )
        levels = tuple(
            level for item in level_lines
            if (level := parse_igra_level(item, elevation)) is not None
        )
        soundings.append(RawUpperSounding(
            igra_id=igra_id,
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            levels=levels,
            missing_surface_height=missing_surface_height and elevation is None,
        ))
    if not soundings:
        raise ValueError(f"IGRA file contains no dated soundings: {source.spec.path}")
    return tuple(soundings)


def _level_score(level: UpperLevel) -> tuple[int, int, int]:
    """Score duplicate-height IGRA levels for retention.

    Args:
        level: Candidate decoded upper-air level.

    Returns:
        Tuple ordered by data completeness, explicit-surface status, and standard
        pressure-level status; larger tuples are preferred.
    """
    completeness = int(level.temperature_k is not None) + 2 * int(level.has_wind)
    return completeness, int(level.is_surface), int(level.level_type == "1")


def _prepare_sounding(
    raw: RawUpperSounding,
    minimum_top_height_m: float,
) -> UpperSounding:
    """Prepare one decoded IGRA sounding for CALMET interpolation.

    The returned sounding retains valid source levels above CALMET's 9998 m
    legacy output limit for provenance. Coverage checks use only levels that can
    be serialized safely below that limit.

    Args:
        raw: Sounding before height sorting, deduplication, and coverage checks.
        minimum_top_height_m: Required valid wind and temperature depth above the
            retained surface level, in metres AGL.

    Returns:
        Prepared sounding with levels sorted and deduplicated by height.

    Raises:
        ValueError: If the surface height is unresolved or the profile lacks a
            surface origin, temperature, wind, or required vertical depth.
    """
    if raw.missing_surface_height:
        raise ValueError(
            "surface GPH is missing; provide an explicit station-elevation override"
        )
    by_height: dict[float, UpperLevel] = {}
    for level in raw.levels:
        current = by_height.get(level.height_msl)
        if current is None or _level_score(level) > _level_score(current):
            by_height[level.height_msl] = level
    levels = sorted(by_height.values(), key=lambda item: item.height_msl)
    wind_indices = [index for index, level in enumerate(levels) if level.has_wind]
    if len(wind_indices) < 2:
        raise ValueError("fewer than two valid wind levels")
    if not levels[wind_indices[0]].is_surface:
        raise ValueError("lowest valid wind level is not an explicit surface level")
    levels = levels[wind_indices[0]:]
    if levels[0].temperature_k is None:
        raise ValueError("lowest valid-wind level has no temperature")
    emitted_levels = [
        level
        for level in levels
        if level.height_msl < CALMET_LEGACY_MISSING_HEIGHT_M
    ]
    valid_temperatures = [
        level for level in emitted_levels if level.temperature_k is not None
    ]
    if len(valid_temperatures) < 2:
        raise ValueError(
            "fewer than two valid temperature levels below CALMET's "
            "9998 m missing-height threshold"
        )
    valid_winds = [level for level in emitted_levels if level.has_wind]
    if len(valid_winds) < 2:
        raise ValueError(
            "fewer than two valid wind levels below CALMET's "
            "9998 m missing-height threshold"
        )
    depth = valid_winds[-1].height_msl - levels[0].height_msl
    if depth < minimum_top_height_m:
        raise ValueError(
            f"valid wind profile reaches only {depth:.0f} m AGL; "
            f"{minimum_top_height_m:.0f} m required"
        )
    temperature_depth = valid_temperatures[-1].height_msl - levels[0].height_msl
    if temperature_depth < minimum_top_height_m:
        raise ValueError(
            f"valid temperature profile reaches only {temperature_depth:.0f} m AGL; "
            f"{minimum_top_height_m:.0f} m required"
        )
    return UpperSounding(
        timestamp=raw.timestamp,
        latitude=(raw.latitude if raw.latitude is not None else math.nan),
        longitude=(raw.longitude if raw.longitude is not None else math.nan),
        levels=tuple(levels),
    )


def _upper_identifier(igra_id: str) -> int:
    """Convert an IGRA identifier to CALMET's numeric I8 identifier.

    Args:
        igra_id: Eleven-character IGRA station ID (three letters and eight digits).

    Returns:
        Numeric eight-character station portion.

    Raises:
        ValueError: If the ID does not follow the supported IGRA form.
    """
    match = re.fullmatch(r"[A-Z]{3}(\d{8})", igra_id.upper())
    if match is None:
        raise ValueError(f"unsupported IGRA station identifier {igra_id!r}")
    return int(match.group(1))


def read_igra_station(
    spec: SourceSpec,
    start: datetime,
    end: datetime,
    minimum_top_height_m: float,
    station_elevations_m: Mapping[str, float],
) -> UpperStationData:
    """Read one IGRA archive and retain valid soundings bracketing a run.

    Args:
        spec: Fixed-width IGRA file or archive and its provenance URL.
        start: Inclusive naive UTC beginning of the CALMET run.
        end: Exclusive naive UTC ending of the CALMET run. A valid sounding
            strictly later than this time is required so runtime endpoint
            selection retains a right-hand snapshot.
        minimum_top_height_m: Minimum valid-wind and valid-temperature depth
            above the first retained level, using only heights that can be
            emitted below CALMET's 9998 m legacy missing-value threshold.
        station_elevations_m: Documented MSL elevations keyed by IGRA ID.
            An override is required when IGRA omits GPH on surface records.

    Returns:
        Station metadata and the smallest chronological set of valid
        soundings from the latest snapshot at or before ``start`` through the
        first snapshot strictly after ``end``.

    Raises:
        OSError: If the source file cannot be opened or read.
        ValueError: If records are malformed or no valid profiles bracket the
            requested run.
    """
    source = _load_text(spec, encoding="ascii")
    raw_soundings = _raw_igra_soundings(source, station_elevations_m)
    prepared: dict[datetime, UpperSounding] = {}
    rejected: list[str] = []
    for raw in raw_soundings:
        try:
            sounding = _prepare_sounding(raw, minimum_top_height_m)
        except ValueError as error:
            rejected.append(f"{raw.timestamp.isoformat()}: {error}")
            continue
        if not math.isfinite(sounding.latitude) or not math.isfinite(sounding.longitude):
            rejected.append(f"{raw.timestamp.isoformat()}: missing station coordinates")
            continue
        current = prepared.get(sounding.timestamp)
        if current is None or len(sounding.levels) > len(current.levels):
            prepared[sounding.timestamp] = sounding
    ordered = tuple(prepared[key] for key in sorted(prepared))
    left = [item for item in ordered if item.timestamp <= start]
    right = [item for item in ordered if item.timestamp > end]
    if not left or not right:
        detail = "; ".join(rejected[-3:])
        raise ValueError(
            f"IGRA station {raw_soundings[0].igra_id} has no valid soundings "
            f"bracketing {start.isoformat()}..{end.isoformat()} with a "
            "snapshot strictly after the run end"
            + (f" ({detail})" if detail else "")
        )
    first = left[-1].timestamp
    last = right[0].timestamp
    selected = tuple(item for item in ordered if first <= item.timestamp <= last)
    if len(selected) < 2 or selected[0].timestamp >= selected[-1].timestamp:
        raise ValueError("IGRA bracketing requires two distinct sounding times")
    latitude = selected[0].latitude
    longitude = selected[0].longitude
    if any(abs(item.latitude - latitude) > 0.02 or
           abs(item.longitude - longitude) > 0.02 for item in selected[1:]):
        raise ValueError(f"IGRA station location changes within {spec.path}")
    igra_id = raw_soundings[0].igra_id
    if igra_id in station_elevations_m:
        elevation = station_elevations_m[igra_id]
        elevation_source = "explicit override"
    else:
        surface_heights = [
            level.height_msl
            for item in selected
            for level in item.levels
            if level.is_surface
        ]
        if not surface_heights:
            raise ValueError(
                f"IGRA station {igra_id} has no usable surface-level elevation"
            )
        elevation = float(statistics.median(surface_heights))
        elevation_source = "IGRA surface-level GPH"
    return UpperStationData(
        source=source,
        igra_id=igra_id,
        identifier=_upper_identifier(igra_id),
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation,
        elevation_source=elevation_source,
        soundings=selected,
    )


def _ordinal(value: datetime) -> tuple[int, int, int, int]:
    """Convert a datetime to CALMET ordinal time fields.

    Args:
        value: Date and time to encode.

    Returns:
        Tuple ``(year, Julian day, hour, seconds within hour)``.
    """
    return value.year, value.timetuple().tm_yday, value.hour, value.minute * 60 + value.second


def _coordinate(value: float, *, latitude: bool) -> str:
    """Format a geographic coordinate for a CALMET station header.

    Args:
        value: Signed decimal coordinate in degrees.
        latitude: Use north/south suffixes when true, otherwise east/west.

    Returns:
        Absolute coordinate with six decimal places and a hemisphere suffix.
    """
    positive = "N" if latitude else "E"
    negative = "S" if latitude else "W"
    suffix = positive if value >= 0.0 else negative
    return f"{abs(value):.6f}{suffix}"


def _surface_value(value: float | int | None, decimals: int = 0) -> str:
    """Format one field for a SURF.DAT hourly record.

    Args:
        value: Numeric value, or ``None`` for CALMET's missing sentinel.
        decimals: Number of fractional decimal places; zero emits an integer.

    Returns:
        Formatted value or the integer ``9999`` missing sentinel.
    """
    if value is None:
        return str(SURFACE_MISSING)
    if decimals == 0:
        return str(int(value))
    return f"{float(value):.{decimals}f}"


def _surface_site(
    station: SurfaceStationData,
    chosen: Sequence[SurfaceCandidate | None],
    station_names: Mapping[str, str],
) -> tuple[float, float, float, str]:
    """Resolve and validate the site represented by selected reports.

    Args:
        station: Station-level metadata and candidate reports.
        chosen: Selected candidate, or ``None``, for each requested hour.
        station_names: Optional descriptive-name overrides keyed by NOAA station
            identifier.

    Returns:
        Tuple ``(latitude, longitude, elevation_m, descriptive_name)``.

    Raises:
        ValueError: If selected reports mix distinct physical sites or the
            resulting station name is empty.
    """
    available = [item for item in chosen if item is not None]
    if available:
        reference = available[0]
        if any(
            abs(item.latitude - reference.latitude) > 0.01
            or abs(item.longitude - reference.longitude) > 0.01
            or abs(item.elevation_m - reference.elevation_m) > 10.0
            for item in available[1:]
        ):
            raise ValueError(
                f"selected ISD reports for {station.source_station_id} mix "
                "different station sites"
            )
        latitude = reference.latitude
        longitude = reference.longitude
        elevation = reference.elevation_m
        source_name = reference.full_name
    else:
        latitude = station.latitude
        longitude = station.longitude
        elevation = station.elevation_m
        source_name = station.full_name
    name = station_names.get(station.source_station_id, source_name).strip()
    if not name:
        raise ValueError(f"empty name for ISD station {station.source_station_id}")
    return latitude, longitude, elevation, name


def _validate_surface_coverage(
    hours: Sequence[datetime],
    selected_by_station: Sequence[Sequence[SurfaceCandidate | None]],
    require_cloud_ceiling: bool,
) -> None:
    """Validate cross-station coverage for every surface hour.

    Args:
        hours: Requested UTC hours in output order.
        selected_by_station: Per-station selections aligned with ``hours``.
        require_cloud_ceiling: Also require at least one cloud-cover and ceiling
            observation per hour for an ``MCLOUD=1`` run.

    Raises:
        ValueError: If an hour lacks wind, temperature, humidity, pressure, or a
            requested cloud field across all stations.
    """
    for hour_index, hour in enumerate(hours):
        values = [
            candidate.values
            for station in selected_by_station
            if (candidate := station[hour_index]) is not None
        ]
        available: dict[str, bool] = {
            "wind": any(
                item.wind_speed_m_s is not None
                and item.wind_direction_deg is not None
                for item in values
            ),
            "temperature": any(item.temperature_k is not None for item in values),
            "relative humidity": any(
                item.relative_humidity_percent is not None for item in values
            ),
            "station pressure": any(
                item.station_pressure_mb is not None for item in values
            ),
        }
        if require_cloud_ceiling:
            available["cloud cover"] = any(
                item.cloud_tenths is not None for item in values
            )
            available["ceiling"] = any(
                item.ceiling_hundreds_ft is not None for item in values
            )
        missing = [field for field, present in available.items() if not present]
        if missing:
            raise ValueError(
                f"surface observations at {hour.isoformat()}Z lack cross-station "
                f"coverage for: {', '.join(missing)}"
            )


def write_surface_dat(
    path: Path,
    stations: Sequence[SurfaceStationData],
    start: datetime,
    end: datetime,
    tolerance: timedelta,
    anemometer_height_m: float,
    datum_date: str,
    station_names: Mapping[str, str],
    require_cloud_ceiling: bool,
) -> list[dict[str, object]]:
    """Write one unpacked, formatted SURF.DAT 2.1 file.

    Args:
        path: Output path.
        stations: Parsed NOAA surface stations.
        start: Inclusive naive UTC beginning of the file period.
        end: Exclusive naive UTC ending of the file period.
        tolerance: Maximum offset allowed when choosing an hourly report.
        anemometer_height_m: Assumed sensor height above local ground.
        datum_date: Optional ten-character datum database date.
        station_names: Optional source-station names keyed by NOAA station ID.
        require_cloud_ceiling: Whether every hour must have cross-station
            cloud-cover and ceiling observations for ``MCLOUD=1``.

    Returns:
        Per-station selection metadata suitable for the provenance manifest.

    Raises:
        OSError: If the output file cannot be written.
        ValueError: If stations are absent or collide, selected reports mix
            sites, or any hour lacks the required cross-station fields.
    """
    if not stations:
        raise ValueError("at least one ISD station is required")
    ordered = sorted(stations, key=lambda item: (item.identifier, item.source_station_id))
    if len({item.identifier for item in ordered}) != len(ordered):
        raise ValueError("derived SURF.DAT station identifiers are not unique")
    hours: list[datetime] = []
    cursor = start
    while cursor < end:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    begin = _ordinal(start)
    finish = _ordinal(end)
    lines = [
        f"{'SURF.DAT':<16}{'2.1':<16}{'NOAA ISD observations':<64}",
        "1",
        "Generated from archived NOAA ISD data; see manifest.json",
        "LL",
        f"{'WGS-84':<8}{datum_date:<10}",
        "DEG",
        "UTC+0000",
        " ".join(str(value) for value in (*begin, *finish, len(ordered))),
    ]
    selections: list[dict[str, object]] = []
    selected_by_station: list[list[SurfaceCandidate | None]] = []
    for station in ordered:
        chosen = [select_surface_report(station, hour, tolerance) for hour in hours]
        selected_by_station.append(chosen)
    _validate_surface_coverage(hours, selected_by_station, require_cloud_ceiling)
    for number, station in enumerate(ordered, 1):
        short_name = f"S{number:03d}"
        chosen = selected_by_station[number - 1]
        latitude, longitude, elevation, source_name = _surface_site(
            station, chosen, station_names
        )
        lines.append(
            f"{station.identifier} {short_name} "
            f"{_coordinate(latitude, latitude=True)} "
            f"{_coordinate(longitude, latitude=False)} "
            f"{anemometer_height_m:.1f}"
        )
        selections.append({
            "calmet_name": short_name,
            "calmet_station_id": station.identifier,
            "source_station_id": station.source_station_id,
            "source_name": source_name,
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
            "anemometer_height_m": anemometer_height_m,
            "selected_reports_utc": [
                None if item is None else item.timestamp.isoformat() + "Z"
                for item in chosen
            ],
            "selected_report_types": [
                None if item is None else item.report_type for item in chosen
            ],
        })
    missing = SurfaceValues(None, None, None, None, None, None, None, None)
    for hour_index, hour in enumerate(hours):
        record_end = hour + timedelta(hours=1)
        tokens = [str(value) for value in (*_ordinal(hour), *_ordinal(record_end))]
        for station_index in range(len(ordered)):
            candidate = selected_by_station[station_index][hour_index]
            values = missing if candidate is None else candidate.values
            tokens.extend((
                _surface_value(values.wind_speed_m_s, 2),
                _surface_value(values.wind_direction_deg, 1),
                _surface_value(values.ceiling_hundreds_ft),
                _surface_value(values.cloud_tenths),
                _surface_value(values.temperature_k, 2),
                _surface_value(values.relative_humidity_percent),
                _surface_value(values.station_pressure_mb, 1),
                _surface_value(values.precipitation_code),
            ))
        lines.append(" ".join(tokens))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return selections


def _up_header_time(value: datetime) -> tuple[int, int, int, int]:
    """Convert a datetime to fields used by the UP.DAT file header.

    Args:
        value: Date and time to encode.

    Returns:
        Tuple ``(year, Julian day, hour, seconds within hour)``.
    """
    return _ordinal(value)


def _sounding_header(station_id: int, timestamp: datetime, levels: int) -> str:
    """Format one fixed-column READ62/RDUPN sounding header.

    Args:
        station_id: Numeric CALMET upper-air station identifier.
        timestamp: UTC sounding time written as both beginning and ending time.
        levels: Number of following profile-level records.

    Returns:
        Exact 76-column header accepted by CALMET 6.5.
    """
    fields = (timestamp.year, timestamp.month, timestamp.day,
              timestamp.hour, timestamp.minute * 60 + timestamp.second)
    return (
        " " * 9 + f"{station_id:8d}" +
        " " * 4 + f"{fields[0]:4d}{fields[1]:4d}{fields[2]:3d}"
        f"{fields[3]:3d}{fields[4]:5d}" +
        " " * 4 + f"{fields[0]:4d}{fields[1]:4d}{fields[2]:3d}"
        f"{fields[3]:3d}{fields[4]:5d}" +
        " " * 8 + f"{levels:5d}"
    )


def write_upper_dat(
    path: Path,
    station: UpperStationData,
    station_number: int,
    datum_date: str,
) -> dict[str, object]:
    """Write one formatted UP.DAT 2.1 file for one IGRA station.

    CALMET 6.5 ``RDUP`` treats every height at or above 9998 m as missing.
    Such otherwise valid IGRA levels remain in ``station`` for provenance but
    are intentionally omitted from the emitted profile.

    Args:
        path: Output path.
        station: Validated upper-air station and selected soundings.
        station_number: One-based station number used for the four-character
            CALMET station name.
        datum_date: Optional ten-character datum database date.

    Returns:
        Station and sounding metadata suitable for the provenance manifest.

    Raises:
        OSError: If the output file cannot be written.
        ValueError: If any sounding has no level below CALMET's 9998 m legacy
            missing-height threshold.
    """
    beginning = station.soundings[0].timestamp
    ending = station.soundings[-1].timestamp
    emitted_soundings = tuple(
        tuple(
            level
            for level in sounding.levels
            if level.height_msl < CALMET_LEGACY_MISSING_HEIGHT_M
        )
        for sounding in station.soundings
    )
    if any(not levels for levels in emitted_soundings):
        raise ValueError(
            f"IGRA station {station.igra_id} has a sounding with no levels "
            "below CALMET's 9998 m missing-height threshold"
        )
    top_pressure = min(
        level.pressure_mb
        for levels in emitted_soundings
        for level in levels
    )
    header_values = (*_up_header_time(beginning), *_up_header_time(ending))
    lines = [
        f"{'UP.DAT':<16}{'2.1':<16}{'IGRA v2 upper-air observations':<64}",
        "1",
        "Generated from archived IGRA v2 fixed-width data; see manifest.json",
        "LL",
        f"{'WGS-84':<8}{datum_date:<10}",
        "DEG",
        "UTC+0000",
        " " + "".join(f"{value:5d}" for value in header_values)
        + f"{top_pressure:5.0f}{3:5d}{2:5d}",
        "     T    F    F    F",
        f"{station.identifier} U{station_number:03d} "
        f"{_coordinate(station.latitude, latitude=True)} "
        f"{_coordinate(station.longitude, latitude=False)} "
        f"{int(math.floor(station.elevation_m + 0.5))}",
    ]
    profile_metadata: list[dict[str, object]] = []
    for sounding, emitted_levels in zip(
            station.soundings, emitted_soundings, strict=True):
        lines.append(_sounding_header(
            station.identifier, sounding.timestamp, len(emitted_levels)
        ))
        for level in emitted_levels:
            temperature = (UPPER_MISSING if level.temperature_k is None
                           else level.temperature_k)
            direction = (UPPER_MISSING if level.wind_direction_deg is None
                         else level.wind_direction_deg)
            speed = UPPER_MISSING if level.wind_speed_m_s is None else level.wind_speed_m_s
            lines.append(
                f"{level.pressure_mb:.2f} {level.height_msl:.1f} "
                f"{temperature:.2f} {direction:.1f} {speed:.1f}"
            )
        valid_winds = [level for level in emitted_levels if level.has_wind]
        valid_temperatures = [
            level for level in emitted_levels if level.temperature_k is not None
        ]
        profile_metadata.append({
            "timestamp_utc": sounding.timestamp.isoformat() + "Z",
            "source_valid_level_count": len(sounding.levels),
            "removed_height_ge_9998_m_count": (
                len(sounding.levels) - len(emitted_levels)
            ),
            "emitted_level_count": len(emitted_levels),
            "emitted_top_height_msl_m": emitted_levels[-1].height_msl,
            "emitted_top_height_agl_m": (
                emitted_levels[-1].height_msl - emitted_levels[0].height_msl
            ),
            "emitted_valid_wind_top_height_msl_m": (
                valid_winds[-1].height_msl
            ),
            "emitted_valid_temperature_top_height_msl_m": (
                valid_temperatures[-1].height_msl
            ),
        })
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "calmet_name": f"U{station_number:03d}",
        "calmet_station_id": station.identifier,
        "igra_id": station.igra_id,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation_m": station.elevation_m,
        "elevation_source": station.elevation_source,
        "soundings_utc": [
            item.timestamp.isoformat() + "Z" for item in station.soundings
        ],
        "level_counts": [len(item) for item in emitted_soundings],
        "source_valid_level_counts": [
            len(item.levels) for item in station.soundings
        ],
        "profiles": profile_metadata,
    }


def _input_manifest(kind: str, source: LoadedSource) -> dict[str, object]:
    """Build a provenance entry for one raw input archive.

    Args:
        kind: Stable source-format label used by the manifest.
        source: Loaded source and its URL, digest, size, and optional member name.

    Returns:
        JSON-serializable input metadata.
    """
    result: dict[str, object] = {
        "kind": kind,
        "file_name": source.spec.path.name,
        "source_url": source.spec.source_url,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }
    if source.archive_member is not None:
        result["archive_member"] = source.archive_member
    return result


def _output_manifest(kind: str, path: Path) -> dict[str, object]:
    """Build an integrity entry for one generated output.

    Args:
        kind: Stable CALMET output-format label used by the manifest.
        path: Generated file to hash and measure.

    Returns:
        JSON-serializable output name, SHA-256 digest, and byte count.

    Raises:
        OSError: If the generated file cannot be read or inspected.
    """
    return {
        "kind": kind,
        "file_name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _build_observations_into(
    isd_sources: Sequence[SourceSpec],
    igra_sources: Sequence[SourceSpec],
    *,
    start: datetime,
    end: datetime,
    output_dir: Path,
    surface_tolerance_minutes: int = 0,
    minimum_upper_wind_height_m: float = 3000.0,
    anemometer_height_m: float = 10.0,
    require_surface_cloud_ceiling: bool = True,
    surface_station_names: Mapping[str, str] | None = None,
    igra_station_elevations_m: Mapping[str, float] | None = None,
    igra_elevation_source: SourceSpec | None = None,
    datum_date: str = "",
    manifest_name: str = "manifest.json",
) -> tuple[Path, tuple[Path, ...], Path]:
    """Convert raw observation archives into a new CALMET output directory.

    Args:
        isd_sources: NOAA Global Hourly CSV or official fixed-width ISD
            sources. A CSV source may contain several stations.
        igra_sources: One IGRA v2 fixed-width source per upper-air station.
        start: Inclusive naive UTC beginning, aligned to an hour.
        end: Exclusive naive UTC ending, aligned to an hour.
        output_dir: New staging directory for generated files and manifest.
        surface_tolerance_minutes: Maximum distance from an exact target hour
            when selecting an ISD report; 0 requires an exact report.
        minimum_upper_wind_height_m: Required valid wind and temperature
            depth above the lowest retained level. Coverage is checked using
            only levels below CALMET's 9998 m legacy height threshold.
        anemometer_height_m: Documented AGL assumption written for every ISD
            station because Global Hourly does not provide sensor height.
        require_surface_cloud_ceiling: Require cloud and ceiling coverage for
            a pure-observation ``MCLOUD=1`` run. Set false when prognostic data
            supplies cloud fields.
        surface_station_names: Optional descriptive names keyed by NOAA's
            eleven-digit ``USAF+WBAN`` station identifier.
        igra_station_elevations_m: Documented MSL elevation overrides keyed by
            eleven-character IGRA ID. These retain surface records whose GPH
            is missing.
        igra_elevation_source: Optional archived catalog that documents the
            explicit IGRA elevation overrides and is hashed in the manifest.
        datum_date: Optional datum database date written after ``WGS-84``.
        manifest_name: File name for the JSON provenance manifest.

    Returns:
        Tuple ``(surface_path, upper_paths, manifest_path)``.

    Raises:
        OSError: If a source cannot be read or an output cannot be written.
        ValueError: If time, source, station, or profile data are invalid.
    """
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("build_observations expects naive UTC datetimes")
    if (start.minute or start.second or start.microsecond
            or end.minute or end.second or end.microsecond):
        raise ValueError("start and end must align to whole UTC hours")
    if end <= start:
        raise ValueError("end must be later than start")
    if not isd_sources or not igra_sources:
        raise ValueError("at least one ISD and one IGRA source are required")
    if not 0 <= surface_tolerance_minutes <= 30:
        raise ValueError("surface_tolerance_minutes must be between 0 and 30")
    if minimum_upper_wind_height_m <= 0.0:
        raise ValueError("minimum_upper_wind_height_m must be positive")
    if anemometer_height_m <= 0.0:
        raise ValueError("anemometer_height_m must be positive")
    if len(datum_date) > 10 or not datum_date.isascii():
        raise ValueError("datum_date must be at most 10 ASCII characters")
    if Path(manifest_name).name != manifest_name:
        raise ValueError("manifest_name must be a file name without directories")
    names = {} if surface_station_names is None else dict(surface_station_names)
    elevations = {
        key.upper(): float(value)
        for key, value in (
            {} if igra_station_elevations_m is None
            else igra_station_elevations_m
        ).items()
    }
    for igra_id, elevation in elevations.items():
        if re.fullmatch(r"[A-Z]{3}\d{8}", igra_id) is None:
            raise ValueError(f"invalid IGRA elevation-override ID {igra_id!r}")
        if not math.isfinite(elevation) or not -500.0 <= elevation < 9000.0:
            raise ValueError(f"invalid elevation override for {igra_id}: {elevation}")
    elevation_provenance = (
        None if igra_elevation_source is None
        else _load_text(igra_elevation_source, encoding="utf-8")
    )
    surface_stations = tuple(
        station
        for spec in isd_sources
        for station in read_isd_stations(spec)
    )
    upper_stations = tuple(read_igra_station(
        spec, start, end, minimum_upper_wind_height_m, elevations
    ) for spec in igra_sources)
    unused_names = set(names) - {
        item.source_station_id for item in surface_stations
    }
    if unused_names:
        raise ValueError(
            "station-name overrides do not match an ISD source: "
            + ", ".join(sorted(unused_names))
        )
    unused_elevations = set(elevations) - {
        item.igra_id for item in upper_stations
    }
    if unused_elevations:
        raise ValueError(
            "elevation overrides do not match an IGRA source: "
            + ", ".join(sorted(unused_elevations))
        )
    if len({item.identifier for item in upper_stations}) != len(upper_stations):
        raise ValueError("derived UP.DAT station identifiers are not unique")
    output_names = {
        "SURF.DAT",
        *(f"UP_{item.igra_id.upper()}.DAT" for item in upper_stations),
    }
    if manifest_name.casefold() in {name.casefold() for name in output_names}:
        raise ValueError(
            f"manifest_name {manifest_name!r} conflicts with a generated CALMET file"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "SURF.DAT"
    tolerance = timedelta(minutes=surface_tolerance_minutes)
    surface_metadata = write_surface_dat(
        surface_path, surface_stations, start, end, tolerance,
        anemometer_height_m, datum_date, names,
        require_surface_cloud_ceiling,
    )
    upper_paths: list[Path] = []
    upper_metadata: list[dict[str, object]] = []
    for number, station in enumerate(sorted(
            upper_stations, key=lambda item: (item.identifier, item.igra_id)), 1):
        output = output_dir / f"UP_{station.igra_id.upper()}.DAT"
        upper_paths.append(output)
        upper_metadata.append(write_upper_dat(output, station, number, datum_date))
    manifest_path = output_dir / manifest_name
    inputs: list[dict[str, object]] = []
    seen_inputs: set[tuple[Path, str]] = set()
    for surface_station in surface_stations:
        key = (
            surface_station.source.spec.path,
            surface_station.source.spec.source_url,
        )
        if key not in seen_inputs:
            inputs.append(_input_manifest(
                f"noaa_{surface_station.input_format}", surface_station.source
            ))
            seen_inputs.add(key)
    for upper_station in upper_stations:
        key = (upper_station.source.spec.path, upper_station.source.spec.source_url)
        if key not in seen_inputs:
            inputs.append(_input_manifest(
                "igra_v2_fixed_width", upper_station.source
            ))
            seen_inputs.add(key)
    if elevation_provenance is not None:
        inputs.append(_input_manifest(
            "igra_station_elevation_provenance", elevation_provenance
        ))
    outputs = [_output_manifest("calmet_surface_2.1", surface_path)]
    outputs.extend(_output_manifest("calmet_upper_air_2.1", path) for path in upper_paths)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": "tools/build_zhejiang_observations.py",
        "period_utc": {
            "start_inclusive": start.isoformat() + "Z",
            "end_exclusive": end.isoformat() + "Z",
        },
        "inputs": inputs,
        "outputs": outputs,
        "surface": {
            "temperature_unit": "K",
            "pressure_unit": "mb",
            "wind_speed_unit": "m s-1",
            "selection_tolerance_minutes": surface_tolerance_minutes,
            "requires_cloud_and_ceiling": require_surface_cloud_ceiling,
            "missing_value": SURFACE_MISSING,
            "isd_quality_flags": {
                "known": "".join(sorted(KNOWN_ISD_QUALITY)),
                "rejected_as_erroneous": "".join(sorted(REJECTED_ISD_QUALITY)),
                "unknown_flags": "rejected",
            },
            "cloud_cover_preference": [
                "GF opaque", "GF total", "last valid GD summation",
                "maximum valid GA layer",
            ],
            "clear_sky_ceiling_hundreds_ft": 999,
            "cavok_fallback": "cloud cover 0 tenths; ceiling 999 hundred ft",
            "selected_site_tolerance": {
                "latitude_degrees": 0.01,
                "longitude_degrees": 0.01,
                "elevation_m": 10.0,
            },
            "relative_humidity_formula": (
                "100 * es(dewpoint) / es(temperature), "
                "es=6.112*exp(17.67*T/(T+243.5))"
            ),
            "station_pressure_preference": [
                "MA1 station pressure", "MA1 altimeter reduced to station",
                "SLP reduced to station",
            ],
            "precipitation_weather_mapping": (
                "SURFGEN 7 ISD AW/MW mapping, expressed as CALMET 0/10/20"
            ),
            "stations": surface_metadata,
        },
        "upper_air": {
            "temperature_unit": "K",
            "pressure_unit": "mb",
            "height_unit": "m MSL",
            "wind_speed_unit": "m s-1",
            "missing_temperature_and_wind": UPPER_MISSING,
            "missing_height": SURFACE_MISSING,
            "emitted_height_policy": (
                "CALMET legacy missing-height sentinel limits emitted heights "
                "to <9998m"
            ),
            "minimum_valid_wind_depth_m": minimum_upper_wind_height_m,
            "minimum_valid_temperature_depth_m": minimum_upper_wind_height_m,
            "igra_processing_flags_accepted": ["blank", "A", "B"],
            "surface_level_policy": (
                "LVLTYP2=1 required at profile origin; missing surface GPH "
                "requires an explicit elevation override"
            ),
            "station_elevation_overrides_m": elevations,
            "stations": upper_metadata,
        },
        "assumptions": [
            "All NOAA Global Hourly DATE values are UTC.",
            f"ISD anemometer height is assumed to be {anemometer_height_m:g} m AGL.",
            "Clear reports without a finite CIG use the SURFGEN unlimited-ceiling code 999.",
            "Missing valid ISD precipitation groups are interpreted as no precipitation.",
            "IGRA profile height is relative to its explicit surface level.",
            "CALMET legacy missing-height sentinel limits emitted heights to <9998m.",
            "UP.DAT 2.1 soundings are instantaneous; begin and end are identical.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return surface_path, tuple(upper_paths), manifest_path


def _install_output_directory(staging: Path, output: Path) -> None:
    """Install a complete generated directory and preserve recoverability.

    Args:
        staging: Complete newly generated directory on the destination filesystem.
        output: Dedicated observation-output directory to replace.

    Raises:
        OSError: If the existing directory cannot be preserved, the new directory
            cannot be installed, or a failed installation cannot be restored.
        ValueError: If ``staging`` is not a real directory.
    """
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError(f"staged observation directory is invalid: {staging}")
    backup = output.parent / (
        f".zhejiang-build-previous-{output.name}-{staging.parent.name}"
    )
    had_output = os.path.lexists(output)
    if had_output:
        try:
            os.replace(output, backup)
        except OSError as error:
            raise OSError(
                f"cannot preserve existing observation output at {output}: {error}"
            ) from error
    try:
        os.replace(staging, output)
    except OSError as install_error:
        if had_output:
            try:
                os.replace(backup, output)
            except OSError as restore_error:
                raise OSError(
                    f"cannot install observation output at {output}: "
                    f"{install_error}; restoring the previous directory also "
                    f"failed: {restore_error}; recovery copy remains at {backup}"
                ) from restore_error
        raise OSError(
            f"cannot install observation output at {output}: {install_error}"
        ) from install_error
    if had_output:
        retired = staging.parent / "previous-output"
        try:
            os.replace(backup, retired)
        except OSError as error:
            raise OSError(
                f"installed observation output at {output}, but could not retire "
                f"the previous directory at {backup}: {error}"
            ) from error


def build_observations(
    isd_sources: Sequence[SourceSpec],
    igra_sources: Sequence[SourceSpec],
    *,
    start: datetime,
    end: datetime,
    output_dir: Path,
    surface_tolerance_minutes: int = 0,
    minimum_upper_wind_height_m: float = 3000.0,
    anemometer_height_m: float = 10.0,
    require_surface_cloud_ceiling: bool = True,
    surface_station_names: Mapping[str, str] | None = None,
    igra_station_elevations_m: Mapping[str, float] | None = None,
    igra_elevation_source: SourceSpec | None = None,
    datum_date: str = "",
    manifest_name: str = "manifest.json",
) -> tuple[Path, tuple[Path, ...], Path]:
    """Build and atomically install one complete CALMET observation dataset.

    The destination is replaced as a directory only after every source has been
    parsed and every output and manifest has been written successfully. If
    installation fails, an existing destination is restored.

    Args:
        isd_sources: NOAA Global Hourly CSV or fixed-width ISD sources.
        igra_sources: One IGRA v2 source per upper-air station.
        start: Inclusive naive UTC beginning aligned to an hour.
        end: Exclusive naive UTC ending aligned to an hour.
        output_dir: Dedicated directory to replace with the complete result.
        surface_tolerance_minutes: Maximum report offset from each target hour.
        minimum_upper_wind_height_m: Required valid wind and temperature depth.
        anemometer_height_m: Assumed ISD sensor height above local ground.
        require_surface_cloud_ceiling: Require surface cloud and ceiling coverage.
        surface_station_names: Optional names keyed by NOAA station identifier.
        igra_station_elevations_m: Optional elevations keyed by IGRA identifier.
        igra_elevation_source: Optional catalog documenting elevation overrides.
        datum_date: Optional datum database date written after ``WGS-84``.
        manifest_name: Non-conflicting file name for the provenance manifest.

    Returns:
        Final ``(surface_path, upper_paths, manifest_path)`` under ``output_dir``.

    Raises:
        OSError: If source, staging, output, or directory installation I/O fails.
        ValueError: If configuration or observation data are invalid.
    """
    output_dir = Path(output_dir)
    if output_dir.name in {"", ".", ".."}:
        raise ValueError("output_dir must name a dedicated child directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".zhejiang-build-observations-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "calmet"
        surface_path, upper_paths, manifest_path = _build_observations_into(
            isd_sources,
            igra_sources,
            start=start,
            end=end,
            output_dir=staging,
            surface_tolerance_minutes=surface_tolerance_minutes,
            minimum_upper_wind_height_m=minimum_upper_wind_height_m,
            anemometer_height_m=anemometer_height_m,
            require_surface_cloud_ceiling=require_surface_cloud_ceiling,
            surface_station_names=surface_station_names,
            igra_station_elevations_m=igra_station_elevations_m,
            igra_elevation_source=igra_elevation_source,
            datum_date=datum_date,
            manifest_name=manifest_name,
        )
        _install_output_directory(staging, output_dir)
    return (
        output_dir / surface_path.name,
        tuple(output_dir / path.name for path in upper_paths),
        output_dir / manifest_path.name,
    )


def _source_specs(values: Sequence[Sequence[str]]) -> tuple[SourceSpec, ...]:
    """Convert repeated command-line path/URL pairs to source specs.

    Args:
        values: Two-item sequences containing local path then provenance URL.

    Returns:
        Immutable source specifications in command-line order.
    """
    return tuple(SourceSpec(Path(value[0]), value[1]) for value in values)


def _string_overrides(values: Sequence[Sequence[str]] | None) -> dict[str, str]:
    """Convert repeated command-line key/value overrides.

    Args:
        values: Optional two-item key/value sequences.

    Returns:
        Mapping of each unique key to its supplied string value.

    Raises:
        ValueError: If a key is supplied more than once.
    """
    result: dict[str, str] = {}
    for key, value in values or ():
        if key in result:
            raise ValueError(f"duplicate override for {key}")
        result[key] = value
    return result


def _float_overrides(values: Sequence[Sequence[str]] | None) -> dict[str, float]:
    """Convert repeated command-line key/value pairs to floats.

    Args:
        values: Optional two-item key/numeric-text sequences.

    Returns:
        Mapping of each unique key to its floating-point value.

    Raises:
        ValueError: If a key is duplicated or a value is not valid float text.
    """
    return {
        key: float(value)
        for key, value in _string_overrides(values).items()
    }


def _parser() -> argparse.ArgumentParser:
    """Create the observation-conversion command-line parser.

    Returns:
        Configured argument parser for archived NOAA ISD and IGRA inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isd-source", nargs=2, action="append", required=True,
        metavar=("ISD_FILE_OR_ARCHIVE", "SOURCE_URL"),
        help=("archived NOAA Global Hourly CSV or annual fixed-width ISD file "
              "and its source URL; repeat per source"),
    )
    parser.add_argument(
        "--igra-source", nargs=2, action="append", required=True,
        metavar=("TEXT_OR_ARCHIVE", "SOURCE_URL"),
        help="archived IGRA v2 fixed-width data and its source URL; repeat per station",
    )
    parser.add_argument("--start", required=True, help="inclusive UTC ISO-8601 hour")
    parser.add_argument("--end", required=True, help="exclusive UTC ISO-8601 hour")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-tolerance-minutes", type=int, default=0)
    parser.add_argument("--minimum-upper-wind-height-m", type=float, default=3000.0)
    parser.add_argument("--anemometer-height-m", type=float, default=10.0)
    parser.add_argument(
        "--surface-cloud-source", choices=("surface", "prognostic"),
        default="surface",
        help="require hourly cloud/ceiling only when surface observations supply them",
    )
    parser.add_argument(
        "--isd-station-name", nargs=2, action="append", default=[],
        metavar=("USAF_WBAN", "NAME"),
        help="descriptive station-name override; repeat per station",
    )
    parser.add_argument(
        "--igra-station-elevation", nargs=2, action="append", default=[],
        metavar=("IGRA_ID", "METRES_MSL"),
        help="documented elevation used for missing surface GPH; repeat per station",
    )
    parser.add_argument(
        "--igra-elevation-source", nargs=2,
        metavar=("CATALOG_FILE", "SOURCE_URL"),
        help="archived catalog documenting elevation overrides for the manifest",
    )
    parser.add_argument("--datum-date", default="")
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archived-observation conversion CLI.

    Args:
        argv: Command-line arguments without the executable name, or ``None``
            to use :data:`sys.argv` through :mod:`argparse`.

    Returns:
        Process exit status, with zero indicating successful conversion.
    """
    arguments = _parser().parse_args(argv)
    elevation_source = (
        None if arguments.igra_elevation_source is None
        else SourceSpec(
            Path(arguments.igra_elevation_source[0]),
            arguments.igra_elevation_source[1],
        )
    )
    surface_path, upper_paths, manifest_path = build_observations(
        _source_specs(arguments.isd_source),
        _source_specs(arguments.igra_source),
        start=parse_utc(arguments.start),
        end=parse_utc(arguments.end),
        output_dir=arguments.output_dir,
        surface_tolerance_minutes=arguments.surface_tolerance_minutes,
        minimum_upper_wind_height_m=arguments.minimum_upper_wind_height_m,
        anemometer_height_m=arguments.anemometer_height_m,
        require_surface_cloud_ceiling=arguments.surface_cloud_source == "surface",
        surface_station_names=_string_overrides(arguments.isd_station_name),
        igra_station_elevations_m=_float_overrides(
            arguments.igra_station_elevation
        ),
        igra_elevation_source=elevation_source,
        datum_date=arguments.datum_date,
        manifest_name=arguments.manifest_name,
    )
    print(surface_path)
    for path in upper_paths:
        print(path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
