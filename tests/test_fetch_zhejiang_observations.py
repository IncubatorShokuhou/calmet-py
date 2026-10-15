"""Tests for the one-command Zhejiang observation acquisition workflow."""

from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import pytest

from tools import fetch_zhejiang_observations as fetcher


def _minimal_prepared_provenance() -> dict[str, Any]:
    """Return safe file metadata sufficient for prepared-output checks."""
    return {
        "surface": {"archives": []},
        "upper_air": {
            "archives": [{
                "station": "CHM00058457",
                "filename": "CHM00058457-data.txt.zip",
                "slice_filename": "CHM00058457-20180318-00-12Z.txt",
            }],
            "station_list": {
                "filename": "igra2-station-list.txt",
                "elevation_catalog": {
                    "filename": "igra2-station-list-frozen.txt",
                },
            },
        },
    }


def _write_minimal_raw_acquisition(tmp_path: Path) -> tuple[Path, Path]:
    """Write a complete compact raw tree and matching acquisition metadata."""
    raw = tmp_path / "raw"
    surface = raw / "surface" / "surface.gz"
    station_list = raw / "upper_air" / "station-list.txt"
    catalog = raw / "upper_air" / "frozen-station-list.txt"
    surface.parent.mkdir(parents=True)
    station_list.parent.mkdir(parents=True)
    surface.write_bytes(b"surface archive")
    station_list.write_bytes(b"live station list")
    catalog.write_bytes(b"frozen station list")

    def digest(path: Path) -> str:
        """Return a compact fixture file's SHA-256 digest."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance_data = {
        "case": {
            "calmet_period_utc": fetcher.EXPECTED_CASE_PERIOD,
            "surface_selection_date_utc": fetcher.EXPECTED_SURFACE_DATE,
            "upper_air_selection_utc": list(fetcher.EXPECTED_UPPER_TIMES),
        },
        "surface": {"archives": [{
            "station": "000000",
            "filename": surface.name,
            "size_bytes": surface.stat().st_size,
            "sha256": digest(surface),
        }]},
        "upper_air": {
            "archives": [],
            "station_list": {
                "filename": station_list.name,
                "reference_size_bytes": station_list.stat().st_size,
                "elevation_catalog": {
                    "filename": catalog.name,
                    "size_bytes": catalog.stat().st_size,
                    "sha256": digest(catalog),
                },
            },
        },
    }
    provenance = tmp_path / "acquisition.json"
    provenance.write_text(json.dumps(provenance_data), encoding="utf-8")
    records = [
        {"kind": "surface", "path": "surface/surface.gz", "sha256": digest(surface)},
        {
            "kind": "station_list",
            "path": "upper_air/station-list.txt",
            "sha256": digest(station_list),
        },
        {
            "kind": "pinned_igra_station_elevation_catalog",
            "path": "upper_air/frozen-station-list.txt",
            "sha256": digest(catalog),
        },
    ]
    (raw / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "generator": "tools/fetch_zhejiang_observations.py",
            "provenance_sha256": digest(provenance),
            "files": records,
        }),
        encoding="utf-8",
    )
    return provenance, raw


def _write_malicious_provenance(
    tmp_path: Path,
    field: str,
) -> tuple[Path, Path]:
    """Write a tracked provenance copy with one path-escaping value."""
    data = json.loads(fetcher.DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
    escaped = tmp_path / "escaped-observation.dat"
    if field == "surface archive":
        data["surface"]["archives"][0]["filename"] = str(escaped)
    elif field == "upper archive":
        data["upper_air"]["archives"][0]["filename"] = str(escaped)
    elif field == "upper slice":
        data["upper_air"]["archives"][0]["slice_filename"] = str(escaped)
    elif field == "station list":
        data["upper_air"]["station_list"]["filename"] = str(escaped)
    elif field == "elevation catalog":
        data["upper_air"]["station_list"]["elevation_catalog"][
            "filename"
        ] = str(escaped)
    elif field == "prepared output":
        data["upper_air"]["archives"][0]["station"] = "../../../escaped"
    else:  # pragma: no cover - protects the parametrized test itself
        raise AssertionError(f"unknown malicious field {field}")
    provenance = tmp_path / "malicious-acquisition.json"
    provenance.write_text(json.dumps(data), encoding="utf-8")
    return provenance, escaped


def _igra_header(station: str, hour: int, levels: int = 1) -> bytes:
    """Build the fixed fields used by the fetcher's profile selector."""
    line = [" "] * 80
    line[0] = "#"
    line[1:12] = f"{station:<11}"
    line[13:17] = "2018"
    line[18:20] = "03"
    line[21:23] = "18"
    line[24:26] = f"{hour:02d}"
    line[27:31] = "9999"
    line[32:36] = f"{levels:4d}"
    return ("".join(line).rstrip() + "\n").encode("ascii")


def test_extract_igra_slice_preserves_complete_target_blocks(tmp_path: Path) -> None:
    station = "CHM00058457"
    archive = tmp_path / f"{station}-data.txt.zip"
    ignored = _igra_header(station, 6) + b"ignored-level\n"
    at_zero = _igra_header(station, 0) + b"zero-level\n"
    at_twelve = _igra_header(station, 12) + b"twelve-level\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{station}-data.txt", ignored + at_zero + at_twelve)
    selected = tmp_path / "selected.txt"

    profiles, levels = fetcher._extract_igra_slice(archive, station, selected)

    assert (profiles, levels) == (2, 2)
    assert selected.read_bytes() == at_zero + at_twelve


def test_extract_igra_slice_requires_both_bracketing_profiles(tmp_path: Path) -> None:
    station = "CHM00058457"
    archive = tmp_path / f"{station}-data.txt.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            f"{station}-data.txt", _igra_header(station, 0) + b"level\n"
        )

    with pytest.raises(fetcher.AcquisitionError, match="target profile counts"):
        fetcher._extract_igra_slice(archive, station, tmp_path / "selected.txt")


def test_extract_igra_slice_rejects_duplicate_nominal_hour(tmp_path: Path) -> None:
    station = "CHM00058457"
    archive = tmp_path / f"{station}-data.txt.zip"
    duplicate = _igra_header(station, 0) + b"level\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{station}-data.txt", duplicate + duplicate)

    with pytest.raises(fetcher.AcquisitionError, match=r"\{0: 2, 12: 0\}"):
        fetcher._extract_igra_slice(archive, station, tmp_path / "selected.txt")


def test_extract_igra_slice_rejects_oversize_member(tmp_path: Path) -> None:
    station = "CHM00058457"
    archive = tmp_path / f"{station}-data.txt.zip"
    payload = (
        _igra_header(station, 0) + b"zero-level\n"
        + _igra_header(station, 12) + b"twelve-level\n"
    )
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{station}-data.txt", payload)

    with pytest.raises(fetcher.AcquisitionError, match="member is .* safety limit"):
        fetcher._extract_igra_slice(
            archive,
            station,
            tmp_path / "selected.txt",
            max_member_bytes=len(payload) - 1,
        )


def test_extract_igra_slice_rejects_zip_bomb_ratio(tmp_path: Path) -> None:
    station = "CHM00058457"
    archive = tmp_path / f"{station}-data.txt.zip"
    payload = b"A" * (1024 * 1024)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(f"{station}-data.txt", payload)

    with pytest.raises(fetcher.AcquisitionError, match="compression ratio"):
        fetcher._extract_igra_slice(
            archive,
            station,
            tmp_path / "selected.txt",
            max_member_bytes=2 * 1024 * 1024,
        )


def test_download_rejects_response_beyond_size_limit(tmp_path: Path) -> None:
    destination = tmp_path / "download.bin"

    with pytest.raises(fetcher.AcquisitionError, match="safety limit"):
        fetcher._download(
            "data:application/octet-stream,abcdef",
            destination,
            1.0,
            max_bytes=5,
        )


def test_inspect_isd_retains_per_report_site_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "584570-99999-2018.gz"

    def record(report: str, latitude: str, longitude: str, elevation: str) -> bytes:
        body = "".join((
            "0000", "584570", "99999", "20180318", "0000", "4",
            latitude, longitude, report, elevation, "99999", "V020",
            "999", "1", "N", "0000", "1", "99999", "9", "9", "N",
            "999999", "9", "N", "9", "+9999", "9", "+9999", "9",
            "99999", "9",
        ))
        return (body + "\n").encode("ascii")

    with gzip.open(archive, "wb") as output:
        output.write(record("FM-12", "+30233", "+120167", "+0043"))
        output.write(record("FM-15", "+30228", "+120432", "+0007"))

    result = fetcher._inspect_isd(archive, "584570")

    assert result["target_rows"] == 2
    assert result["report_types"] == ["FM-12", "FM-15"]
    assert result["reported_sites"] == [
        {
            "latitude_raw": "+30228",
            "longitude_raw": "+120432",
            "elevation_raw": "+0007",
        },
        {
            "latitude_raw": "+30233",
            "longitude_raw": "+120167",
            "elevation_raw": "+0043",
        },
    ]


def test_case_time_returns_naive_utc() -> None:
    assert fetcher._case_time("2018-03-18T08:00:00+08:00") == datetime(
        2018, 3, 18
    )
    with pytest.raises(fetcher.AcquisitionError, match="must include UTC offset"):
        fetcher._case_time("2018-03-18T00:00:00")


def test_changed_pinned_download_requires_explicit_review(tmp_path: Path) -> None:
    result = fetcher.DownloadResult(
        path=tmp_path / "changed.gz",
        requested_url="https://example.test/changed.gz",
        resolved_url="https://example.test/changed.gz",
        sha256="b" * 64,
        size_bytes=42,
        etag=None,
        last_modified=None,
    )

    with pytest.raises(fetcher.AcquisitionError, match="Do not silently accept"):
        fetcher._require_digest(result, "a" * 64, label="fixture")


def test_elevation_catalog_must_match_configured_override(tmp_path: Path) -> None:
    catalog = tmp_path / "igra2-station-list.txt"
    catalog.write_text(
        f"{'CHM00058457':<11} {30.2333:8.4f} {120.1667:9.4f} {43.0:6.1f}\n",
        encoding="ascii",
    )
    station = {
        "station": "CHM00058457",
        "latitude": 30.2333,
        "longitude": 120.1667,
        "elevation_override_m": 44.0,
    }

    with pytest.raises(fetcher.AcquisitionError, match="catalog elevation"):
        fetcher._validate_elevation_catalog(catalog, [station])


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        ".",
        "..",
        "/tmp/observation.dat",
        "../observation.dat",
        "directory/observation.dat",
        r"directory\observation.dat",
        r"C:\observation.dat",
    ],
)
def test_safe_basename_rejects_empty_absolute_and_qualified_names(
    value: str,
) -> None:
    with pytest.raises(fetcher.AcquisitionError, match="file name|base name"):
        fetcher._safe_basename(value, "fixture filename")


def test_safe_basename_returns_an_unqualified_name() -> None:
    assert fetcher._safe_basename("observation.dat", "fixture") == "observation.dat"


@pytest.mark.parametrize(
    "field",
    [
        "surface archive",
        "upper archive",
        "upper slice",
        "station list",
        "elevation catalog",
        "prepared output",
    ],
)
@pytest.mark.parametrize(
    "operation",
    ["fetch", "prepare", "check", "check_raw", "check_prepared"],
)
def test_malicious_provenance_is_rejected_before_any_output_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    operation: str,
) -> None:
    provenance, escaped = _write_malicious_provenance(tmp_path, field)
    work = tmp_path / "operation-output"

    def unexpected_download(
        url: str,
        destination: Path,
        timeout: float,
    ) -> fetcher.DownloadResult:
        del url, destination, timeout
        raise AssertionError("download started before provenance validation")

    monkeypatch.setattr(fetcher, "_download", unexpected_download)
    with pytest.raises(fetcher.AcquisitionError, match="safe base name"):
        if operation == "fetch":
            fetcher.fetch(provenance, work / "raw", 1.0)
        elif operation == "prepare":
            fetcher.prepare(provenance, work / "raw", work / "calmet")
        elif operation == "check":
            fetcher.check(provenance, work / "raw", work / "calmet")
        elif operation == "check_raw":
            fetcher.check_raw(provenance, work / "raw")
        else:
            fetcher.check_prepared(
                provenance,
                work / "raw",
                work / "calmet",
                {},
            )

    assert not work.exists()
    assert not escaped.exists()


def test_check_prepared_rejects_stale_dat_file(tmp_path: Path) -> None:
    provenance = tmp_path / "acquisition.json"
    provenance.write_text(json.dumps(_minimal_prepared_provenance()), encoding="utf-8")
    output = tmp_path / "calmet"
    output.mkdir()
    (output / "SURF.DAT").write_bytes(b"surface")
    (output / "UP_CHM00058457.DAT").write_bytes(b"upper")
    (output / "UP_STALE.DAT").write_bytes(b"stale")
    (output / "manifest.json").write_text(
        '{"outputs":[],"inputs":[]}', encoding="utf-8"
    )

    with pytest.raises(fetcher.AcquisitionError, match="directory file set"):
        fetcher.check_prepared(provenance, tmp_path / "raw", output, {})


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_check_raw_rejects_actual_file_set_changes(
    tmp_path: Path,
    change: str,
) -> None:
    provenance, raw = _write_minimal_raw_acquisition(tmp_path)
    if change == "extra":
        (raw / "unexpected.bin").write_bytes(b"stale")
    else:
        (raw / "surface" / "surface.gz").unlink()

    with pytest.raises(fetcher.AcquisitionError, match="raw directory file set"):
        fetcher.check_raw(provenance, raw)


def test_check_raw_accepts_the_exact_file_set(tmp_path: Path) -> None:
    provenance, raw = _write_minimal_raw_acquisition(tmp_path)

    acquired = fetcher.check_raw(provenance, raw)

    assert set(acquired) == {
        "surface.gz", "station-list.txt", "frozen-station-list.txt",
    }


def test_install_directory_replaces_the_complete_tree(tmp_path: Path) -> None:
    transaction = tmp_path / "transaction"
    staging = transaction / "raw"
    output = tmp_path / "raw"
    staging.mkdir(parents=True)
    output.mkdir()
    (staging / "new.bin").write_bytes(b"new")
    (output / "stale.bin").write_bytes(b"stale")

    fetcher._install_directory(staging, output)

    assert (output / "new.bin").read_bytes() == b"new"
    assert not (output / "stale.bin").exists()


def test_install_directory_restores_previous_tree_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = tmp_path / "transaction"
    staging = transaction / "raw"
    output = tmp_path / "raw"
    staging.mkdir(parents=True)
    output.mkdir()
    (staging / "new.bin").write_bytes(b"new")
    (output / "old.bin").write_bytes(b"old")
    real_replace = fetcher.os.replace

    def fail_staged_install(source: Path, destination: Path) -> None:
        """Inject a failure only after the previous output is preserved."""
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected install failure")
        real_replace(source, destination)

    monkeypatch.setattr(fetcher.os, "replace", fail_staged_install)
    with pytest.raises(fetcher.AcquisitionError, match="cannot install"):
        fetcher._install_directory(staging, output)

    assert (output / "old.bin").read_bytes() == b"old"
    assert not (output / "new.bin").exists()
    assert not list(tmp_path.glob(".raw.previous-*"))


def test_check_prepared_regenerates_instead_of_trusting_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tmp_path / "acquisition.json"
    provenance.write_text(json.dumps(_minimal_prepared_provenance()), encoding="utf-8")
    output = tmp_path / "calmet"
    output.mkdir()
    surface = output / "SURF.DAT"
    upper = output / "UP_CHM00058457.DAT"
    surface.write_bytes(b"surface")
    upper.write_bytes(b"upper")
    (output / "manifest.json").write_text(
        """{
          "period_utc": {"start_inclusive": "2099-01-01T00:00:00Z"},
          "inputs": [],
          "outputs": [
            {"file_name": "SURF.DAT", "sha256": "%s"},
            {"file_name": "UP_CHM00058457.DAT", "sha256": "%s"}
          ]
        }""" % (fetcher._sha256(surface), fetcher._sha256(upper)),
        encoding="utf-8",
    )

    def regenerate(
        provenance_path: Path,
        raw: Path,
        destination: Path,
        *,
        announce: bool = True,
    ) -> None:
        del provenance_path, raw, announce
        destination.mkdir(parents=True)
        (destination / "SURF.DAT").write_bytes(b"surface")
        (destination / "UP_CHM00058457.DAT").write_bytes(b"upper")
        (destination / "manifest.json").write_text(
            '{"period_utc":{"start_inclusive":"2018-03-18T00:00:00Z"}}',
            encoding="utf-8",
        )

    monkeypatch.setattr(fetcher, "prepare", regenerate)
    with pytest.raises(fetcher.AcquisitionError, match="deterministic result"):
        fetcher.check_prepared(provenance, tmp_path / "raw", output, {})


def test_case_metadata_must_match_pinned_selectors() -> None:
    case = {
        "case": {
            "calmet_period_utc": fetcher.EXPECTED_CASE_PERIOD,
            "surface_selection_date_utc": fetcher.EXPECTED_SURFACE_DATE,
            "upper_air_selection_utc": [
                "2018-03-18T00:00:00Z",
                "2018-03-18T06:00:00Z",
            ],
        }
    }

    with pytest.raises(fetcher.AcquisitionError, match="upper-air times"):
        fetcher._validate_case(case)


def test_default_cli_fetches_then_prepares(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        fetcher, "fetch", lambda *args: calls.append(("fetch", args))
    )
    monkeypatch.setattr(
        fetcher, "prepare", lambda *args: calls.append(("prepare", args))
    )

    assert fetcher.main([]) == 0
    assert [name for name, _ in calls] == ["fetch", "prepare"]


def test_fetch_only_does_not_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(fetcher, "fetch", lambda *args: calls.append("fetch"))
    monkeypatch.setattr(fetcher, "prepare", lambda *args: calls.append("prepare"))

    assert fetcher.main(["--fetch-only"]) == 0
    assert calls == ["fetch"]
