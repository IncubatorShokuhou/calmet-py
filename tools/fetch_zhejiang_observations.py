#!/usr/bin/env python3
"""Fetch the official observations used by the Zhejiang examples.

The downloaded observations are local, restricted inputs and are deliberately
ignored by Git.  Source locations and reviewable checksum expectations live in
``data/observations/zhejiang-20180318/acquisition.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_PROVENANCE: Final[Path] = (
    ROOT / "data" / "observations" / "zhejiang-20180318" /
    "acquisition.json"
)
DEFAULT_OUTPUT: Final[Path] = DEFAULT_PROVENANCE.parent / "raw"
DEFAULT_PREPARED: Final[Path] = DEFAULT_PROVENANCE.parent / "calmet"
USER_AGENT: Final[str] = "py-calmet-observation-fetch/1.0"
TARGET_DATE: Final[str] = "20180318"
TARGET_SOUNDING_HOURS: Final[frozenset[int]] = frozenset((0, 12))
EXPECTED_CASE_PERIOD: Final[dict[str, str]] = {
    "start_inclusive": "2018-03-18T00:00:00Z",
    "end_exclusive": "2018-03-18T01:00:00Z",
}
EXPECTED_SURFACE_DATE: Final[str] = "2018-03-18"
EXPECTED_UPPER_TIMES: Final[tuple[str, str]] = (
    "2018-03-18T00:00:00Z",
    "2018-03-18T12:00:00Z",
)
ELEVATION_TOLERANCE_M: Final[float] = 0.05
COORDINATE_TOLERANCE_DEG: Final[float] = 0.00005
DOWNLOAD_CHUNK_BYTES: Final[int] = 1024 * 1024
REFERENCE_DOWNLOAD_HEADROOM_BYTES: Final[int] = 8 * 1024 * 1024
REFERENCE_DOWNLOAD_GROWTH_FACTOR: Final[int] = 2
ISD_EXPANSION_FACTOR: Final[int] = 32
IGRA_MEMBER_EXPANSION_FACTOR: Final[int] = 8
MAX_ZIP_COMPRESSION_RATIO: Final[float] = 64.0
MAX_OBSERVATION_LINE_BYTES: Final[int] = 64 * 1024
DEFAULT_ISD_UNCOMPRESSED_LIMIT: Final[int] = 64 * 1024 * 1024
DEFAULT_IGRA_MEMBER_LIMIT: Final[int] = 256 * 1024 * 1024
DEFAULT_IGRA_SLICE_LIMIT: Final[int] = 1024 * 1024


class _BinaryLineReader(Protocol):
    """Structural type for binary streams that support bounded line reads."""

    def readline(self, size: int = -1, /) -> bytes:
        """Read one line containing at most ``size`` bytes."""
        ...


class AcquisitionError(RuntimeError):
    """Report an unsafe upstream change or an invalid local acquisition."""


@dataclass(frozen=True)
class DownloadResult:
    """Metadata captured while downloading one official source."""

    path: Path
    requested_url: str
    resolved_url: str
    sha256: str
    size_bytes: int
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class ProvenanceNames:
    """Validated file names declared by the acquisition provenance."""

    surface_archives: tuple[str, ...]
    upper_archives: tuple[str, ...]
    upper_slices: tuple[str, ...]
    station_list: str
    elevation_catalog: str
    prepared_outputs: frozenset[str]


@dataclass(frozen=True)
class ProvenanceSizes:
    """Validated byte counts declared by the acquisition provenance."""

    surface_archives: tuple[int, ...]
    upper_reference_archives: tuple[int, ...]
    upper_slices: tuple[int, ...]
    station_list_reference: int
    elevation_catalog: int


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file.

    Args:
        path: File whose bytes are hashed.

    Returns:
        The 64-character lowercase hexadecimal digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or raise a concise acquisition error.

    Args:
        path: UTF-8 JSON document to read.

    Returns:
        The decoded top-level JSON object.

    Raises:
        AcquisitionError: If the file cannot be read, is invalid JSON, or
            does not contain an object at its top level.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read provenance file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcquisitionError(f"provenance file {path} must contain a JSON object")
    return value


def _safe_basename(value: object, label: str) -> str:
    """Validate a provenance value as one unqualified file name.

    Args:
        value: Candidate value read from the provenance document.
        label: Human-readable field label used in an error message.

    Returns:
        The validated file name.

    Raises:
        AcquisitionError: If ``value`` is empty, absolute, contains a path
            separator, or otherwise names a path rather than one file.
    """
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionError(f"{label} must be a nonempty file name")
    candidate = Path(value)
    if (
        value in {".", ".."}
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise AcquisitionError(
            f"{label} must be a safe base name without directories: {value!r}"
        )
    return value


def _provenance_names(provenance: Mapping[str, Any]) -> ProvenanceNames:
    """Validate every provenance-controlled file name before local I/O.

    Args:
        provenance: Parsed acquisition metadata.

    Returns:
        Validated raw and prepared file names in source-list order.

    Raises:
        AcquisitionError: If source metadata is missing or any name could
            address a directory or escape its intended output directory.
    """
    surface = provenance.get("surface")
    upper = provenance.get("upper_air")
    if not isinstance(surface, dict) or not isinstance(upper, dict):
        raise AcquisitionError("acquisition.json has invalid observation metadata")
    isd_sources = surface.get("archives")
    igra_sources = upper.get("archives")
    station_list = upper.get("station_list")
    if not isinstance(isd_sources, list) or not isinstance(igra_sources, list):
        raise AcquisitionError("acquisition.json has invalid source lists")
    if not isinstance(station_list, dict):
        raise AcquisitionError("acquisition.json has an invalid IGRA station list")
    catalog = station_list.get("elevation_catalog")
    if not isinstance(catalog, dict):
        raise AcquisitionError(
            "acquisition.json has no pinned IGRA elevation catalog"
        )

    surface_names: list[str] = []
    for index, item in enumerate(isd_sources):
        if not isinstance(item, dict):
            raise AcquisitionError(
                f"surface.archives[{index}] must be a JSON object"
            )
        surface_names.append(_safe_basename(
            item.get("filename"), f"surface.archives[{index}].filename"
        ))

    archive_names: list[str] = []
    slice_names: list[str] = []
    prepared_names = {_safe_basename("SURF.DAT", "prepared surface file")}
    for index, item in enumerate(igra_sources):
        if not isinstance(item, dict):
            raise AcquisitionError(
                f"upper_air.archives[{index}] must be a JSON object"
            )
        archive_names.append(_safe_basename(
            item.get("filename"), f"upper_air.archives[{index}].filename"
        ))
        slice_names.append(_safe_basename(
            item.get("slice_filename"),
            f"upper_air.archives[{index}].slice_filename",
        ))
        station = item.get("station")
        if not isinstance(station, str) or not station:
            raise AcquisitionError(
                f"upper_air.archives[{index}].station must be a nonempty string"
            )
        prepared_names.add(_safe_basename(
            f"UP_{station}.DAT",
            f"upper_air.archives[{index}] prepared output",
        ))

    return ProvenanceNames(
        surface_archives=tuple(surface_names),
        upper_archives=tuple(archive_names),
        upper_slices=tuple(slice_names),
        station_list=_safe_basename(
            station_list.get("filename"), "upper_air.station_list.filename"
        ),
        elevation_catalog=_safe_basename(
            catalog.get("filename"),
            "upper_air.station_list.elevation_catalog.filename",
        ),
        prepared_outputs=frozenset(prepared_names),
    )


def _positive_size(value: object, label: str) -> int:
    """Validate one provenance byte count.

    Args:
        value: Candidate JSON value.
        label: Human-readable field name used in an error message.

    Returns:
        The positive integer byte count.

    Raises:
        AcquisitionError: If ``value`` is not a positive JSON integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AcquisitionError(f"{label} must be a positive integer byte count")
    return value


def _provenance_sizes(provenance: Mapping[str, Any]) -> ProvenanceSizes:
    """Validate source and selected-slice sizes before network or output I/O.

    Args:
        provenance: Parsed acquisition metadata.

    Returns:
        Validated byte counts in the same order as their source lists.

    Raises:
        AcquisitionError: If source metadata is incomplete or a size is not a
            positive integer.
    """
    surface = provenance.get("surface")
    upper = provenance.get("upper_air")
    if not isinstance(surface, dict) or not isinstance(upper, dict):
        raise AcquisitionError("acquisition.json has invalid observation metadata")
    isd_sources = surface.get("archives")
    igra_sources = upper.get("archives")
    station_list = upper.get("station_list")
    if not isinstance(isd_sources, list) or not isinstance(igra_sources, list):
        raise AcquisitionError("acquisition.json has invalid source lists")
    if not isinstance(station_list, dict):
        raise AcquisitionError("acquisition.json has an invalid IGRA station list")
    catalog = station_list.get("elevation_catalog")
    if not isinstance(catalog, dict):
        raise AcquisitionError(
            "acquisition.json has no pinned IGRA elevation catalog"
        )

    surface_sizes: list[int] = []
    for index, item in enumerate(isd_sources):
        if not isinstance(item, dict):
            raise AcquisitionError(
                f"surface.archives[{index}] must be a JSON object"
            )
        surface_sizes.append(_positive_size(
            item.get("size_bytes"), f"surface.archives[{index}].size_bytes"
        ))

    archive_sizes: list[int] = []
    slice_sizes: list[int] = []
    for index, item in enumerate(igra_sources):
        if not isinstance(item, dict):
            raise AcquisitionError(
                f"upper_air.archives[{index}] must be a JSON object"
            )
        archive_sizes.append(_positive_size(
            item.get("reference_archive_size_bytes"),
            f"upper_air.archives[{index}].reference_archive_size_bytes",
        ))
        slice_sizes.append(_positive_size(
            item.get("slice_size_bytes"),
            f"upper_air.archives[{index}].slice_size_bytes",
        ))

    return ProvenanceSizes(
        surface_archives=tuple(surface_sizes),
        upper_reference_archives=tuple(archive_sizes),
        upper_slices=tuple(slice_sizes),
        station_list_reference=_positive_size(
            station_list.get("reference_size_bytes"),
            "upper_air.station_list.reference_size_bytes",
        ),
        elevation_catalog=_positive_size(
            catalog.get("size_bytes"),
            "upper_air.station_list.elevation_catalog.size_bytes",
        ),
    )


def _reference_download_limit(reference_size: int) -> int:
    """Return a bounded allowance for an append-only live source.

    Args:
        reference_size: Reviewed source size in bytes.

    Returns:
        Twice the reference size or eight MiB of headroom, whichever is larger.
    """
    return max(
        reference_size * REFERENCE_DOWNLOAD_GROWTH_FACTOR,
        reference_size + REFERENCE_DOWNLOAD_HEADROOM_BYTES,
    )


def _validate_case(provenance: Mapping[str, Any]) -> None:
    """Keep case metadata consistent with the pinned record selectors.

    Args:
        provenance: Parsed acquisition metadata containing the case section.

    Raises:
        AcquisitionError: If the configured period, surface date, or
            upper-air profile times differ from the Zhejiang case selectors.
    """
    case = provenance.get("case")
    if not isinstance(case, dict):
        raise AcquisitionError("acquisition.json has no case metadata")
    period = case.get("calmet_period_utc")
    upper = case.get("upper_air_selection_utc")
    if period != EXPECTED_CASE_PERIOD:
        raise AcquisitionError(
            "acquisition.json CALMET period does not match the pinned Zhejiang "
            f"case: expected {EXPECTED_CASE_PERIOD}, got {period!r}"
        )
    if case.get("surface_selection_date_utc") != EXPECTED_SURFACE_DATE:
        raise AcquisitionError(
            "acquisition.json surface date does not match the pinned Zhejiang case"
        )
    if upper != list(EXPECTED_UPPER_TIMES):
        raise AcquisitionError(
            "acquisition.json upper-air times must be exactly "
            + ", ".join(EXPECTED_UPPER_TIMES)
        )


def _download(
    url: str,
    destination: Path,
    timeout: float,
    *,
    max_bytes: int,
) -> DownloadResult:
    """Download one URL while hashing its exact response bytes.

    Args:
        url: Official upstream URL to request.
        destination: Staging file that receives the response bytes.
        timeout: Network timeout in seconds.
        max_bytes: Maximum response body size accepted in bytes.

    Returns:
        Response metadata and the digest of the downloaded bytes.

    Raises:
        AcquisitionError: If the request or destination write fails, the
            declared content length is invalid, or the body exceeds
            ``max_bytes``.
    """
    if max_bytes <= 0:
        raise AcquisitionError("download max_bytes must be positive")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            resolved_url = response.geturl()
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise AcquisitionError(
                        f"download {url} returned invalid Content-Length "
                        f"{content_length!r}"
                    ) from exc
                if declared_size < 0 or declared_size > max_bytes:
                    raise AcquisitionError(
                        f"download {url} declares {declared_size} bytes; safety "
                        f"limit is {max_bytes}"
                    )
            with destination.open("wb") as output:
                while True:
                    request_size = min(
                        DOWNLOAD_CHUNK_BYTES, max_bytes - size + 1
                    )
                    chunk = response.read(request_size)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise AcquisitionError(
                            f"download {url} exceeds the {max_bytes}-byte safety limit"
                        )
                    output.write(chunk)
                    digest.update(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise AcquisitionError(f"failed to download {url}: {exc}") from exc
    return DownloadResult(
        path=destination,
        requested_url=url,
        resolved_url=resolved_url,
        sha256=digest.hexdigest(),
        size_bytes=size,
        etag=etag,
        last_modified=last_modified,
    )


def _require_size(result: DownloadResult, expected: int, *, label: str) -> None:
    """Reject a pinned source whose byte count differs from provenance.

    Args:
        result: Metadata for the downloaded source.
        expected: Pinned source size in bytes.
        label: Source description included in an error message.

    Raises:
        AcquisitionError: If the received byte count differs from ``expected``.
    """
    if result.size_bytes != expected:
        raise AcquisitionError(
            f"{label} changed size: expected {expected} bytes, got "
            f"{result.size_bytes} from {result.resolved_url}"
        )


def _require_digest(
    result: DownloadResult,
    expected: str,
    *,
    label: str,
) -> None:
    """Reject a changed immutable input with review instructions.

    Args:
        result: Metadata for the downloaded source.
        expected: Pinned lowercase SHA-256 digest.
        label: Source description included in an error message.

    Raises:
        AcquisitionError: If the downloaded digest differs from ``expected``.
    """
    if result.sha256 == expected:
        return
    raise AcquisitionError(
        f"{label} changed upstream: expected SHA-256 {expected}, got "
        f"{result.sha256} ({result.size_bytes} bytes) from "
        f"{result.resolved_url}. Do not silently accept this file. Inspect "
        "the NCEI revision and update acquisition.json only after validating "
        "the target observations."
    )


def _readline_limited(stream: _BinaryLineReader, *, label: str) -> bytes:
    """Read one bounded binary observation record.

    Args:
        stream: Binary input stream positioned at a record boundary.
        label: Source description used in an error message.

    Returns:
        One record including its line ending, or ``b""`` at end of input.

    Raises:
        AcquisitionError: If a record exceeds
            :data:`MAX_OBSERVATION_LINE_BYTES`.
        OSError: If the stream cannot be read.
    """
    line = stream.readline(MAX_OBSERVATION_LINE_BYTES + 1)
    if len(line) > MAX_OBSERVATION_LINE_BYTES:
        raise AcquisitionError(
            f"{label} contains a record longer than "
            f"{MAX_OBSERVATION_LINE_BYTES} bytes"
        )
    return line


def _inspect_isd(
    path: Path,
    station: str,
    *,
    max_uncompressed_bytes: int = DEFAULT_ISD_UNCOMPRESSED_LIMIT,
) -> dict[str, Any]:
    """Validate an annual fixed-width ISD archive and summarize target rows.

    Args:
        path: Gzip-compressed annual ISD file.
        station: Expected six-digit USAF station identifier.
        max_uncompressed_bytes: Maximum expanded payload size accepted.

    Returns:
        Target-date row count, report types, and raw site metadata.

    Raises:
        AcquisitionError: If decompression fails, a record is malformed, the
            station identity changes, or the target date has no observations.
    """
    target_rows = 0
    report_types: set[str] = set()
    sites: set[tuple[str, str, str]] = set()
    expanded_size = 0
    try:
        with gzip.open(path, "rb") as stream:
            number = 0
            while raw_line := _readline_limited(stream, label=path.name):
                number += 1
                expanded_size += len(raw_line)
                if expanded_size > max_uncompressed_bytes:
                    raise AcquisitionError(
                        f"{path.name} expands beyond the "
                        f"{max_uncompressed_bytes}-byte safety limit"
                    )
                line = raw_line.rstrip(b"\r\n")
                if len(line) < 105:
                    raise AcquisitionError(
                        f"{path.name}:{number} is shorter than the ISD "
                        "mandatory section"
                    )
                try:
                    line_station = line[4:10].decode("ascii")
                    wban = line[10:15].decode("ascii")
                    timestamp = line[15:27].decode("ascii")
                    report_type = line[41:46].decode("ascii").strip()
                    latitude = line[28:34].decode("ascii")
                    longitude = line[34:41].decode("ascii")
                    elevation = line[46:51].decode("ascii")
                except UnicodeDecodeError as exc:
                    raise AcquisitionError(
                        f"{path.name}:{number} is not an ASCII ISD record"
                    ) from exc
                if line_station != station or wban != "99999":
                    raise AcquisitionError(
                        f"{path.name}:{number} identifies {line_station}-{wban}, "
                        f"expected {station}-99999"
                    )
                if timestamp.startswith(TARGET_DATE):
                    target_rows += 1
                    report_types.add(report_type)
                    sites.add((latitude, longitude, elevation))
    except (OSError, EOFError) as exc:
        raise AcquisitionError(f"cannot decompress {path}: {exc}") from exc
    if target_rows == 0:
        raise AcquisitionError(
            f"{path.name} contains no observations for 2018-03-18 UTC"
        )
    return {
        "target_date": "2018-03-18",
        "target_rows": target_rows,
        "report_types": sorted(report_types),
        "reported_sites": [
            {"latitude_raw": lat, "longitude_raw": lon, "elevation_raw": elev}
            for lat, lon, elev in sorted(sites)
        ],
    }


def _igra_header(line: bytes, expected_station: str) -> tuple[int, int, int, int, int]:
    """Parse fields needed to select one IGRA fixed-width profile header.

    Args:
        line: Raw IGRA header record.
        expected_station: Eleven-character station identifier required in the
            header.

    Returns:
        Year, month, day, nominal UTC hour, and following level count.

    Raises:
        AcquisitionError: If the header is malformed, belongs to another
            station, or declares no levels.
    """
    if len(line) < 36 or not line.startswith(b"#"):
        raise AcquisitionError("encountered an invalid IGRA profile header")
    try:
        station = line[1:12].decode("ascii")
        year = int(line[13:17])
        month = int(line[18:20])
        day = int(line[21:23])
        hour = int(line[24:26])
        level_count = int(line[32:36])
    except (UnicodeDecodeError, ValueError) as exc:
        raise AcquisitionError("cannot parse an IGRA profile header") from exc
    if station != expected_station:
        raise AcquisitionError(
            f"IGRA archive contains station {station}, expected {expected_station}"
        )
    if level_count <= 0:
        raise AcquisitionError(f"IGRA profile has invalid level count {level_count}")
    return year, month, day, hour, level_count


def _validate_igra_member(
    info: zipfile.ZipInfo,
    archive: Path,
    *,
    max_member_bytes: int,
) -> None:
    """Reject an unsafe or implausibly compressed IGRA ZIP member.

    Args:
        info: Metadata for the sole expected archive member.
        archive: ZIP path used in diagnostics.
        max_member_bytes: Maximum uncompressed member size accepted.

    Raises:
        AcquisitionError: If the member is encrypted, is a symbolic link,
            exceeds ``max_member_bytes``, or has an excessive compression ratio.
    """
    if max_member_bytes <= 0:
        raise AcquisitionError("IGRA max_member_bytes must be positive")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise AcquisitionError(f"{archive.name} contains a symbolic-link member")
    if info.flag_bits & 0x1:
        raise AcquisitionError(f"{archive.name} contains an encrypted member")
    if info.file_size > max_member_bytes:
        raise AcquisitionError(
            f"{archive.name} member is {info.file_size} bytes; safety limit is "
            f"{max_member_bytes}"
        )
    if info.file_size > 0 and info.compress_size <= 0:
        raise AcquisitionError(f"{archive.name} member has invalid compressed size")
    ratio = info.file_size / max(1, info.compress_size)
    if ratio > MAX_ZIP_COMPRESSION_RATIO:
        raise AcquisitionError(
            f"{archive.name} member compression ratio is {ratio:.1f}; maximum is "
            f"{MAX_ZIP_COMPRESSION_RATIO:g}"
        )


def _extract_igra_slice(
    archive: Path,
    station: str,
    destination: Path,
    *,
    max_member_bytes: int = DEFAULT_IGRA_MEMBER_LIMIT,
    max_slice_bytes: int = DEFAULT_IGRA_SLICE_LIMIT,
) -> tuple[int, int]:
    """Copy exact 00/12 UTC profile blocks from an IGRA period archive.

    Args:
        archive: Official single-member period-of-record ZIP file.
        station: Eleven-character IGRA station identifier.
        destination: File that receives the selected unmodified profile bytes.
        max_member_bytes: Maximum uncompressed period-of-record member size.
        max_slice_bytes: Maximum combined size of the two selected profiles.

    Returns:
        Number of selected profiles and their total reported level count.

    Raises:
        AcquisitionError: If the archive layout or profile records are
            malformed, or exactly one 00 UTC and one 12 UTC profile are not
            available for the target date.
    """
    if max_slice_bytes <= 0:
        raise AcquisitionError("IGRA max_slice_bytes must be positive")
    member_name = f"{station}-data.txt"
    selected: list[bytes] = []
    selected_size = 0
    profiles = 0
    levels = 0
    selected_hour_counts = {hour: 0 for hour in TARGET_SOUNDING_HOURS}
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if len(members) != 1 or members[0].filename != member_name:
                raise AcquisitionError(
                    f"{archive.name} must contain only {member_name}; found "
                    f"{[item.filename for item in members]!r}"
                )
            info = members[0]
            if info.is_dir():
                raise AcquisitionError(
                    f"{archive.name} member {member_name} must be a regular file"
                )
            _validate_igra_member(
                info, archive, max_member_bytes=max_member_bytes
            )
            expanded_size = 0
            line_number = 0
            with source.open(info) as stream:
                while True:
                    header = _readline_limited(stream, label=archive.name)
                    if not header:
                        break
                    line_number += 1
                    expanded_size += len(header)
                    if expanded_size > max_member_bytes:
                        raise AcquisitionError(
                            f"{archive.name} expands beyond the "
                            f"{max_member_bytes}-byte safety limit"
                        )
                    if not header.startswith(b"#"):
                        raise AcquisitionError(
                            f"{archive.name} has a non-header line at profile "
                            f"boundary {line_number}"
                        )
                    year, month, day, hour, level_count = _igra_header(
                        header, station
                    )
                    is_target = (
                        (year, month, day) == (2018, 3, 18)
                        and hour in TARGET_SOUNDING_HOURS
                    )
                    block = [header] if is_target else []
                    for _ in range(level_count):
                        line = _readline_limited(stream, label=archive.name)
                        if not line:
                            raise AcquisitionError(
                                f"{archive.name} ends inside the "
                                f"{year:04d}-{month:02d}-{day:02d} "
                                f"{hour:02d} UTC profile"
                            )
                        line_number += 1
                        expanded_size += len(line)
                        if expanded_size > max_member_bytes:
                            raise AcquisitionError(
                                f"{archive.name} expands beyond the "
                                f"{max_member_bytes}-byte safety limit"
                            )
                        if is_target:
                            block.append(line)
                    if is_target:
                        block_size = sum(len(line) for line in block)
                        selected_size += block_size
                        if selected_size > max_slice_bytes:
                            raise AcquisitionError(
                                f"{archive.name} selected profiles exceed the "
                                f"{max_slice_bytes}-byte safety limit"
                            )
                        selected.extend(block)
                        profiles += 1
                        levels += level_count
                        selected_hour_counts[hour] += 1
            if expanded_size != info.file_size:
                raise AcquisitionError(
                    f"{archive.name} expanded to {expanded_size} bytes; ZIP metadata "
                    f"declares {info.file_size}"
                )
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise AcquisitionError(f"cannot read {archive}: {exc}") from exc
    if selected_hour_counts != {0: 1, 12: 1}:
        raise AcquisitionError(
            f"{archive.name} yielded target profile counts {selected_hour_counts}; "
            "expected exactly one profile at 2018-03-18 00 UTC and one at 12 UTC"
        )
    destination.write_bytes(b"".join(selected))
    return profiles, levels


def _validate_station_list(path: Path, stations: Sequence[Mapping[str, Any]]) -> None:
    """Ensure an IGRA station list contains every selected station.

    Args:
        path: Fixed-width IGRA station-list file.
        stations: Selected station metadata from the acquisition document.

    Raises:
        AcquisitionError: If the list cannot be read, station metadata lacks
            an identifier, or a selected station is absent.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AcquisitionError(f"cannot read IGRA station list {path}: {exc}") from exc
    by_id = {line[:11]: line for line in lines if len(line) >= 11}
    identifiers = [str(item.get("station", item.get("id", ""))) for item in stations]
    if any(not identifier for identifier in identifiers):
        raise AcquisitionError("selected IGRA station metadata lacks an identifier")
    missing = [identifier for identifier in identifiers if identifier not in by_id]
    if missing:
        raise AcquisitionError(
            "current IGRA station list is missing selected station(s): "
            + ", ".join(missing)
        )


def _station_catalog(
    path: Path,
    identifiers: set[str],
) -> dict[str, tuple[float, float, float]]:
    """Read selected IGRA station metadata from documented fixed columns.

    Args:
        path: Fixed-width IGRA station-list file.
        identifiers: Station identifiers to retain.

    Returns:
        Mapping from station identifier to latitude, longitude, and elevation
        in metres.

    Raises:
        AcquisitionError: If the file cannot be read or selected coordinates
            cannot be parsed.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AcquisitionError(f"cannot read IGRA station list {path}: {exc}") from exc
    catalog: dict[str, tuple[float, float, float]] = {}
    for number, line in enumerate(lines, 1):
        if len(line) < 37:
            continue
        identifier = line[0:11]
        if identifier not in identifiers:
            continue
        try:
            catalog[identifier] = (
                float(line[12:20]),
                float(line[21:30]),
                float(line[31:37]),
            )
        except ValueError as exc:
            raise AcquisitionError(
                f"invalid fixed-column IGRA station metadata at {path}:{number}"
            ) from exc
    return catalog


def _validate_elevation_catalog(
    path: Path,
    stations: Sequence[Mapping[str, Any]],
) -> None:
    """Verify configured station locations and elevations against a catalog.

    Args:
        path: Pinned fixed-width IGRA station catalog.
        stations: Acquisition entries containing coordinates and elevation
            overrides.

    Raises:
        AcquisitionError: If a station is missing or its configured location
            or elevation differs from the pinned catalog beyond tolerance.
        KeyError: If a station entry lacks required coordinate or elevation
            fields.
        ValueError: If a required numeric field cannot be converted.
    """
    identifiers = {str(item.get("station", "")) for item in stations}
    catalog = _station_catalog(path, identifiers)
    for item in stations:
        identifier = str(item.get("station", ""))
        if identifier not in catalog:
            raise AcquisitionError(
                f"pinned elevation catalog lacks IGRA station {identifier}"
            )
        latitude, longitude, elevation = catalog[identifier]
        expected_latitude = float(item["latitude"])
        expected_longitude = float(item["longitude"])
        expected_elevation = float(item["elevation_override_m"])
        if (abs(latitude - expected_latitude) > COORDINATE_TOLERANCE_DEG
                or abs(longitude - expected_longitude) > COORDINATE_TOLERANCE_DEG):
            raise AcquisitionError(
                f"pinned catalog coordinates for {identifier} are "
                f"({latitude}, {longitude}); acquisition.json expects "
                f"({expected_latitude}, {expected_longitude}) within "
                f"{COORDINATE_TOLERANCE_DEG:g} degrees"
            )
        if abs(elevation - expected_elevation) > ELEVATION_TOLERANCE_M:
            raise AcquisitionError(
                f"pinned catalog elevation for {identifier} is {elevation} m; "
                f"acquisition.json requests {expected_elevation} m, exceeding "
                f"the {ELEVATION_TOLERANCE_M:g} m tolerance"
            )


def _file_record(
    result: DownloadResult,
    relative_path: Path,
    *,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one deterministic runtime-manifest file record.

    Args:
        result: Download metadata for the source file.
        relative_path: Installed path relative to the raw-data directory.
        kind: Stable source-kind identifier.
        extra: Optional source-specific manifest fields.

    Returns:
        JSON-serializable manifest record.
    """
    record: dict[str, Any] = {
        "kind": kind,
        "path": relative_path.as_posix(),
        "source_url": result.requested_url,
        "resolved_url": result.resolved_url,
        "sha256": result.sha256,
        "size_bytes": result.size_bytes,
        "http_etag": result.etag,
        "http_last_modified": result.last_modified,
    }
    if extra:
        record.update(extra)
    return record


def _install_directory(staging: Path, output: Path) -> None:
    """Atomically replace a raw-data directory and restore it on failure.

    Args:
        staging: Complete temporary raw-data directory containing validated files.
        output: Local raw-data destination directory.

    Raises:
        AcquisitionError: If the staged directory cannot be installed. When an
            earlier output existed, it is restored before the error is raised.
    """
    if not staging.is_dir() or staging.is_symlink():
        raise AcquisitionError(f"staged raw-data directory is invalid: {staging}")
    backup = output.parent / f".{output.name}.previous-{staging.parent.name}"
    had_output = os.path.lexists(output)
    if had_output:
        try:
            os.replace(output, backup)
        except OSError as exc:
            raise AcquisitionError(
                f"cannot preserve existing raw observations at {output}: {exc}"
            ) from exc
    try:
        os.replace(staging, output)
    except OSError as install_error:
        if had_output:
            try:
                os.replace(backup, output)
            except OSError as restore_error:
                raise AcquisitionError(
                    f"cannot install raw observations at {output}: {install_error}; "
                    f"restoring the previous directory also failed: {restore_error}; "
                    f"recovery copy remains at {backup}"
                ) from restore_error
        raise AcquisitionError(
            f"cannot install raw observations at {output}: {install_error}"
        ) from install_error
    if had_output:
        retired = staging.parent / "previous-raw"
        try:
            os.replace(backup, retired)
        except OSError as exc:
            raise AcquisitionError(
                f"installed raw observations at {output}, but could not retire "
                f"the previous directory at {backup}: {exc}"
            ) from exc


def fetch(provenance_path: Path, output: Path, timeout: float) -> None:
    """Download, validate, and atomically install all observation sources.

    Args:
        provenance_path: Tracked acquisition metadata with URLs and checksums.
        output: Local-only directory for downloaded raw data.
        timeout: Per-request network timeout in seconds.

    Raises:
        AcquisitionError: If metadata is unsafe, a download fails, an input
            changes, selected records do not match the pinned case, or the
            complete staged directory cannot be installed.
    """
    provenance = _load_json(provenance_path)
    _validate_case(provenance)
    names = _provenance_names(provenance)
    sizes = _provenance_sizes(provenance)
    isd_sources = provenance.get("surface", {}).get("archives", [])
    igra_sources = provenance.get("upper_air", {}).get("archives", [])
    station_list = provenance.get("upper_air", {}).get("station_list", {})

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".zhejiang-observations-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "raw"
        staging.mkdir()
        records: list[dict[str, Any]] = []

        for item, filename, expected_size in zip(
            isd_sources,
            names.surface_archives,
            sizes.surface_archives,
            strict=True,
        ):
            station = str(item["station"])
            relative = Path("surface") / filename
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = _download(
                str(item["url"]), destination, timeout, max_bytes=expected_size
            )
            _require_size(result, expected_size, label=f"ISD archive {station}")
            _require_digest(
                result, str(item["sha256"]), label=f"ISD archive {station}"
            )
            inspection = _inspect_isd(
                destination,
                station,
                max_uncompressed_bytes=expected_size * ISD_EXPANSION_FACTOR,
            )
            records.append(_file_record(
                result, relative, kind="noaa_isd_annual_fixed_width_gzip",
                extra={"station": station, "inspection": inspection},
            ))
            print(f"verified ISD {station}: {result.sha256}")

        for item, archive_name, slice_name, reference_size, slice_size in zip(
            igra_sources,
            names.upper_archives,
            names.upper_slices,
            sizes.upper_reference_archives,
            sizes.upper_slices,
            strict=True,
        ):
            station = str(item["station"])
            archive_relative = Path("upper_air") / "archives" / archive_name
            archive_path = staging / archive_relative
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            result = _download(
                str(item["url"]),
                archive_path,
                timeout,
                max_bytes=_reference_download_limit(reference_size),
            )
            reference_digest = str(item["reference_archive_sha256"])
            archive_matches_reference = result.sha256 == reference_digest

            slice_relative = Path("upper_air") / slice_name
            slice_path = staging / slice_relative
            slice_path.parent.mkdir(parents=True, exist_ok=True)
            profiles, levels = _extract_igra_slice(
                archive_path,
                station,
                slice_path,
                max_member_bytes=reference_size * IGRA_MEMBER_EXPANSION_FACTOR,
                max_slice_bytes=slice_size,
            )
            actual_slice_size = slice_path.stat().st_size
            if actual_slice_size != slice_size:
                raise AcquisitionError(
                    f"IGRA historical slice for {station} changed size: expected "
                    f"{slice_size} bytes, got {actual_slice_size}"
                )
            slice_digest = _sha256(slice_path)
            expected_slice = str(item["slice_sha256"])
            if slice_digest != expected_slice:
                raise AcquisitionError(
                    f"IGRA historical records for {station} changed upstream: "
                    f"expected slice SHA-256 {expected_slice}, got {slice_digest}. "
                    "Do not silently accept the revision; inspect the two target "
                    "profiles and update acquisition.json only after review."
                )
            records.append(_file_record(
                result,
                archive_relative,
                kind="noaa_igra_period_of_record_zip",
                extra={
                    "station": station,
                    "reference_archive_sha256": reference_digest,
                    "archive_matches_reference": archive_matches_reference,
                },
            ))
            records.append({
                "kind": "noaa_igra_exact_profile_slice",
                "path": slice_relative.as_posix(),
                "source_path": archive_relative.as_posix(),
                "source_member": f"{station}-data.txt",
                "sha256": slice_digest,
                "size_bytes": actual_slice_size,
                "profiles": profiles,
                "levels": levels,
                "selection": "2018-03-18 00 and 12 UTC",
            })
            state = (
                "reference archive"
                if archive_matches_reference
                else "refreshed archive"
            )
            print(f"verified IGRA {station} slice from {state}: {slice_digest}")

        station_relative = Path("upper_air") / names.station_list
        station_path = staging / station_relative
        station_path.parent.mkdir(parents=True, exist_ok=True)
        station_result = _download(
            str(station_list["url"]),
            station_path,
            timeout,
            max_bytes=_reference_download_limit(sizes.station_list_reference),
        )
        _validate_station_list(station_path, igra_sources)
        station_reference = str(station_list.get("reference_sha256", ""))
        records.append(_file_record(
            station_result,
            station_relative,
            kind="noaa_igra_station_list",
            extra={
                "reference_sha256": station_reference or None,
                "matches_reference": station_result.sha256 == station_reference,
            },
        ))
        catalog = station_list["elevation_catalog"]
        catalog_relative = Path("upper_air") / names.elevation_catalog
        catalog_path = staging / catalog_relative
        catalog_result = _download(
            str(catalog["url"]),
            catalog_path,
            timeout,
            max_bytes=sizes.elevation_catalog,
        )
        _require_size(
            catalog_result,
            sizes.elevation_catalog,
            label="frozen IGRA elevation catalog",
        )
        _require_digest(
            catalog_result,
            str(catalog["sha256"]),
            label="frozen IGRA elevation catalog",
        )
        _validate_station_list(catalog_path, igra_sources)
        _validate_elevation_catalog(catalog_path, igra_sources)
        records.append(_file_record(
            catalog_result,
            catalog_relative,
            kind="pinned_igra_station_elevation_catalog",
            extra={"source_commit": catalog.get("commit")},
        ))
        manifest_relative = Path("manifest.json")
        manifest_path = staging / manifest_relative
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "tools/fetch_zhejiang_observations.py",
            "provenance_template": (
                provenance_path.resolve().relative_to(ROOT).as_posix()
                if provenance_path.resolve().is_relative_to(ROOT)
                else provenance_path.name
            ),
            "provenance_sha256": _sha256(provenance_path),
            "redistribution": provenance.get("redistribution"),
            "files": records,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _install_directory(staging, output)
    print(f"installed local observation inputs under {output}")


def _case_time(value: str) -> datetime:
    """Convert a tracked ISO timestamp to a naive UTC datetime.

    Args:
        value: ISO 8601 timestamp containing a UTC offset or ``Z`` suffix.

    Returns:
        Equivalent timezone-naive UTC datetime for the preprocessing API.

    Raises:
        AcquisitionError: If the timestamp is invalid or lacks a UTC offset.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcquisitionError(f"invalid case timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AcquisitionError(f"case timestamp must include UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def prepare(
    provenance_path: Path,
    raw: Path,
    output: Path,
    *,
    announce: bool = True,
) -> None:
    """Build all CALMET SURF/UP inputs from an acquired raw dataset.

    Args:
        provenance_path: Tracked acquisition metadata used for the raw files.
        raw: Verified local raw-data directory.
        output: Local-only destination for prepared CALMET observation files.
        announce: Whether to print the paths of installed outputs.

    Raises:
        AcquisitionError: If provenance is unsafe or incomplete, required raw
            files are absent, the frozen elevation catalog disagrees, the
            converter cannot be imported, or preprocessing fails.
    """
    provenance = _load_json(provenance_path)
    _validate_case(provenance)
    names = _provenance_names(provenance)

    tools_directory = str(Path(__file__).resolve().parent)
    if tools_directory not in sys.path:
        sys.path.insert(0, tools_directory)
    try:
        from build_zhejiang_observations import (
            SourceSpec,
            _build_observations_into,
        )
    except ImportError as exc:
        raise AcquisitionError(
            "cannot import tools/build_zhejiang_observations.py"
        ) from exc

    surface = provenance.get("surface", {})
    upper = provenance.get("upper_air", {})
    isd_items = surface.get("archives", [])
    igra_items = upper.get("archives", [])
    station_list = upper.get("station_list", {})
    catalog = station_list.get("elevation_catalog", {})
    period = provenance.get("case", {}).get("calmet_period_utc", {})
    try:
        start = _case_time(str(period["start_inclusive"]))
        end = _case_time(str(period["end_exclusive"]))
        catalog_path = raw / "upper_air" / names.elevation_catalog
        catalog_url = str(catalog["url"])
    except (KeyError, TypeError) as exc:
        raise AcquisitionError("acquisition.json lacks preparation metadata") from exc

    isd_sources = [
        SourceSpec(
            raw / "surface" / filename,
            str(item["url"]),
        )
        for item, filename in zip(isd_items, names.surface_archives, strict=True)
    ]
    igra_sources = [
        SourceSpec(
            raw / "upper_air" / filename,
            str(item["url"]),
        )
        for item, filename in zip(igra_items, names.upper_slices, strict=True)
    ]
    station_names = {
        f"{item['station']}99999": str(
            item.get("selected_name", item["name"])
        )
        for item in isd_items
    }
    elevations = {
        str(item["station"]): float(item["elevation_override_m"])
        for item in igra_items
    }
    missing = [
        source.path for source in (*isd_sources, *igra_sources)
        if not source.path.is_file()
    ]
    if not catalog_path.is_file():
        missing.append(catalog_path)
    if missing:
        raise AcquisitionError(
            "cannot prepare observations; run the fetch step first. Missing: "
            + ", ".join(str(path) for path in missing)
        )
    _validate_elevation_catalog(catalog_path, igra_items)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".zhejiang-calmet-observations-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "calmet"
        try:
            surface_path, upper_paths, manifest_path = _build_observations_into(
                isd_sources,
                igra_sources,
                start=start,
                end=end,
                output_dir=staging,
                require_surface_cloud_ceiling=True,
                surface_station_names=station_names,
                igra_station_elevations_m=elevations,
                igra_elevation_source=SourceSpec(catalog_path, catalog_url),
            )
        except (OSError, ValueError) as exc:
            raise AcquisitionError(
                f"CALMET observation preparation failed: {exc}"
            ) from exc
        _install_directory(staging, output)
    if announce:
        print(f"prepared {output / surface_path.name}")
        for path in upper_paths:
            print(f"prepared {output / path.name}")
        print(f"wrote preprocessing provenance {output / manifest_path.name}")


def _raw_file_set(output: Path) -> set[str]:
    """Enumerate regular files in a raw-data directory without following links.

    Args:
        output: Raw observation directory to inspect.

    Returns:
        POSIX-style file paths relative to ``output``.

    Raises:
        AcquisitionError: If ``output`` is missing, is a symbolic link, or
            contains a symbolic link or non-regular filesystem entry.
    """
    if output.is_symlink() or not output.is_dir():
        raise AcquisitionError(f"raw observation path is not a directory: {output}")
    files: set[str] = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            raise AcquisitionError(
                f"raw observation directory contains symbolic link: {relative}"
            )
        if path.is_file():
            files.add(relative)
        elif not path.is_dir():
            raise AcquisitionError(
                f"raw observation directory contains non-regular entry: {relative}"
            )
    return files


def check_raw(provenance_path: Path, output: Path) -> dict[str, str]:
    """Validate an existing local acquisition without network access.

    Args:
        provenance_path: Tracked acquisition metadata with checksum pins.
        output: Local raw-data directory to verify.

    Returns:
        Mapping from each installed base name to its verified SHA-256 digest.

    Raises:
        AcquisitionError: If provenance or the runtime manifest is unsafe,
            incomplete, stale, or inconsistent with local file bytes.
    """
    provenance = _load_json(provenance_path)
    _validate_case(provenance)
    names = _provenance_names(provenance)
    surface_items = provenance.get("surface", {}).get("archives", [])
    upper_items = provenance.get("upper_air", {}).get("archives", [])
    station_list = provenance.get("upper_air", {}).get("station_list", {})
    expected_surface = {
        filename: str(item["sha256"])
        for item, filename in zip(
            surface_items, names.surface_archives, strict=True
        )
    }
    expected_slices = {
        filename: str(item["slice_sha256"])
        for item, filename in zip(upper_items, names.upper_slices, strict=True)
    }
    expected_paths = {
        f"surface/{filename}" for filename in names.surface_archives
    } | {
        f"upper_air/archives/{filename}" for filename in names.upper_archives
    } | {
        f"upper_air/{filename}" for filename in names.upper_slices
    } | {
        f"upper_air/{names.station_list}",
        f"upper_air/{names.elevation_catalog}",
    }
    actual_paths = _raw_file_set(output)
    expected_disk_paths = expected_paths | {"manifest.json"}
    if actual_paths != expected_disk_paths:
        raise AcquisitionError(
            "raw directory file set changed: expected "
            f"{sorted(expected_disk_paths)}, got {sorted(actual_paths)}"
        )
    manifest_path = output / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise AcquisitionError(f"runtime manifest {manifest_path} has no file records")
    if (manifest.get("schema_version") != 1
            or manifest.get("generator") != "tools/fetch_zhejiang_observations.py"):
        raise AcquisitionError(
            f"runtime manifest {manifest_path} has an unsupported schema or generator"
        )
    provenance_digest = _sha256(provenance_path)
    if manifest.get("provenance_sha256") != provenance_digest:
        raise AcquisitionError(
            f"runtime manifest {manifest_path} was not generated from the current "
            f"acquisition.json ({provenance_digest})"
        )
    manifest_paths = [str(record.get("path", "")) for record in files]
    if (
        len(manifest_paths) != len(set(manifest_paths))
        or set(manifest_paths) != expected_paths
    ):
        raise AcquisitionError(
            f"runtime manifest file set changed: expected {sorted(expected_paths)}, "
            f"got {sorted(manifest_paths)}"
        )
    acquired: dict[str, str] = {}
    catalog = provenance.get("upper_air", {}).get("station_list", {}).get(
        "elevation_catalog", {}
    )
    expected_catalog = str(catalog.get("sha256", ""))
    for record in files:
        relative_text = str(record.get("path", ""))
        relative = Path(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise AcquisitionError(
                f"runtime manifest contains unsafe path {relative_text!r}"
            )
        path = output / relative
        if not path.is_file():
            raise AcquisitionError(f"missing acquired observation file: {path}")
        actual = _sha256(path)
        manifest_digest = str(record.get("sha256", ""))
        if actual != manifest_digest:
            raise AcquisitionError(
                f"local file changed: {path}; manifest has {manifest_digest}, "
                f"got {actual}"
            )
        acquired[relative.name] = actual
        if relative.parent == Path("surface"):
            pinned = expected_surface.get(relative.name)
            if pinned is None or actual != pinned:
                raise AcquisitionError(
                    f"local ISD checksum is not pinned by acquisition.json: {path}"
                )
        if relative.parent == Path("upper_air") and relative.name.startswith("CHM"):
            pinned = expected_slices.get(relative.name)
            if pinned is None or actual != pinned:
                raise AcquisitionError(
                    "local IGRA slice checksum is not pinned by "
                    f"acquisition.json: {path}"
                )
        if record.get("kind") == "pinned_igra_station_elevation_catalog":
            if not expected_catalog or actual != expected_catalog:
                raise AcquisitionError(
                    f"local elevation catalog is not pinned by acquisition.json: {path}"
                )
        print(f"ok {relative}: {actual}")
    print(f"verified {len(files)} local observation files under {output}")
    return acquired


def check_prepared(
    provenance_path: Path,
    raw: Path,
    output: Path,
    acquired: Mapping[str, str],
) -> None:
    """Verify derived CALMET files and their linkage to acquired inputs.

    Args:
        provenance_path: Tracked acquisition metadata used for preprocessing.
        raw: Verified local raw-data directory.
        output: Prepared CALMET observation directory to verify.
        acquired: Raw base-name to verified SHA-256 mapping from
            :func:`check_raw`.

    Raises:
        AcquisitionError: If output names, checksums, input links, or a fresh
            deterministic rebuild differ from the installed files.
    """
    provenance = _load_json(provenance_path)
    names = _provenance_names(provenance)
    manifest_path = output / "manifest.json"
    manifest = _load_json(manifest_path)
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    if not isinstance(outputs, list) or not isinstance(inputs, list):
        raise AcquisitionError(
            f"preprocessing manifest {manifest_path} lacks inputs or outputs"
        )
    expected_names = set(names.prepared_outputs)
    actual_names = {
        path.name for path in output.glob("*.DAT") if path.is_file()
    }
    if actual_names != expected_names:
        raise AcquisitionError(
            f"prepared directory file set changed: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )
    output_names = {str(item.get("file_name", "")) for item in outputs}
    if output_names != expected_names:
        raise AcquisitionError(
            f"prepared output set changed: expected {sorted(expected_names)}, "
            f"got {sorted(output_names)}"
        )
    for record in outputs:
        path = output / str(record["file_name"])
        if not path.is_file():
            raise AcquisitionError(f"missing prepared CALMET file: {path}")
        actual = _sha256(path)
        expected = str(record.get("sha256", ""))
        if actual != expected:
            raise AcquisitionError(
                f"prepared CALMET file changed: {path}; manifest has "
                f"{expected}, got {actual}"
            )
        print(f"ok {path.relative_to(output)}: {actual}")
    for record in inputs:
        name = str(record.get("file_name", ""))
        digest = str(record.get("sha256", ""))
        if acquired.get(name) != digest:
            raise AcquisitionError(
                f"prepared manifest input {name} is not the verified raw file "
                f"(manifest SHA-256 {digest})"
            )
    with tempfile.TemporaryDirectory(
        prefix=".zhejiang-observation-check-", dir=output.parent
    ) as temporary:
        regenerated = Path(temporary) / "calmet"
        prepare(
            provenance_path,
            raw,
            regenerated,
            announce=False,
        )
        for name in sorted(expected_names | {"manifest.json"}):
            actual_path = output / name
            reference_path = regenerated / name
            if (
                not actual_path.is_file()
                or actual_path.read_bytes() != reference_path.read_bytes()
            ):
                raise AcquisitionError(
                    f"prepared file is not the deterministic result of the verified "
                    f"inputs and acquisition.json: {actual_path}"
                )
    print(f"verified {len(outputs)} prepared CALMET files under {output}")


def check(provenance_path: Path, raw: Path, prepared: Path) -> None:
    """Verify acquired and prepared observations without network access.

    Args:
        provenance_path: Tracked acquisition metadata with checksum pins.
        raw: Local raw-data directory to verify.
        prepared: Prepared CALMET observation directory to verify.

    Raises:
        AcquisitionError: If either raw or prepared verification fails.
    """
    acquired = check_raw(provenance_path, raw)
    check_prepared(provenance_path, raw, prepared, acquired)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Returns:
        Configured argument parser for fetch, prepare, and offline check modes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance", type=Path, default=DEFAULT_PROVENANCE,
        help="tracked acquisition.json containing official URLs and checksum pins",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="local-only raw observation directory",
    )
    parser.add_argument(
        "--prepared-dir", type=Path, default=DEFAULT_PREPARED,
        help="local-only directory for generated CALMET SURF/UP files",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="per-request network timeout in seconds",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true",
        help="verify existing raw and prepared files without downloading",
    )
    modes.add_argument(
        "--fetch-only", action="store_true",
        help="download and verify raw sources without preparing CALMET files",
    )
    modes.add_argument(
        "--prepare-only", action="store_true",
        help="prepare CALMET files from an existing verified raw acquisition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fetch, prepare, or offline-check workflow.

    Args:
        argv: Command-line arguments excluding the executable name. ``None``
            reads arguments from :data:`sys.argv`.

    Returns:
        Process status: zero on success, one for an acquisition failure, or
        two for an invalid timeout.
    """
    arguments = _parser().parse_args(argv)
    if arguments.timeout <= 0.0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    try:
        if arguments.check:
            check(
                arguments.provenance,
                arguments.output_dir,
                arguments.prepared_dir,
            )
        elif arguments.prepare_only:
            check_raw(arguments.provenance, arguments.output_dir)
            prepare(
                arguments.provenance,
                arguments.output_dir,
                arguments.prepared_dir,
            )
        else:
            fetch(arguments.provenance, arguments.output_dir, arguments.timeout)
            if not arguments.fetch_only:
                prepare(
                    arguments.provenance,
                    arguments.output_dir,
                    arguments.prepared_dir,
                )
    except AcquisitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
