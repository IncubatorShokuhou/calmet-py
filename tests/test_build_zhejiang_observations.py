from __future__ import annotations

import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from pycalmet.surface_data import SurfaceData
from pycalmet.upper_air import UpperAirData
from tools import build_zhejiang_observations as builder
from tools.build_zhejiang_observations import (
    SourceSpec,
    build_observations,
    decode_isd_row,
    parse_igra_level,
    parse_utc,
    read_igra_station,
    read_isd_station,
    read_isd_stations,
)


def _write_isd(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a compact Global Hourly CSV fixture."""
    fields = [
        "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
        "REPORT_TYPE", "WND", "CIG", "TMP", "DEW", "SLP", "MA1",
        "GF1", "GA1", "GD1", "GD2", "AA1", "AW1", "MW1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _igra_header(
    station: str,
    timestamp: datetime,
    levels: int,
    latitude: float = 30.2333,
    longitude: float = 120.1667,
) -> str:
    """Build one fixed-column IGRA v2 sounding header."""
    line = [" "] * 80
    line[0] = "#"
    line[1:12] = f"{station:<11}"
    line[13:17] = f"{timestamp.year:4d}"
    line[18:20] = f"{timestamp.month:02d}"
    line[21:23] = f"{timestamp.day:02d}"
    line[24:26] = f"{timestamp.hour:02d}"
    line[27:31] = "9999"
    line[32:36] = f"{levels:4d}"
    line[55:62] = f"{round(latitude * 10000):7d}"
    line[63:71] = f"{round(longitude * 10000):8d}"
    return "".join(line).rstrip()


def _igra_level(
    pressure_pa: int,
    height_m: int,
    temperature_tenths_c: int,
    direction_deg: int,
    speed_tenths_m_s: int,
    level_type: str = "1",
    minor_level_type: str = "0",
    pressure_flag: str = " ",
    height_flag: str = " ",
    temperature_flag: str = " ",
) -> str:
    """Build one fixed-column IGRA v2 level record."""
    line = [" "] * 52
    line[0] = level_type
    line[1] = minor_level_type
    line[3:8] = f"{0:5d}"
    line[9:15] = f"{pressure_pa:6d}"
    line[15] = pressure_flag
    line[16:21] = f"{height_m:5d}"
    line[21] = height_flag
    line[22:27] = f"{temperature_tenths_c:5d}"
    line[27] = temperature_flag
    line[28:33] = f"{-9999:5d}"
    line[34:39] = f"{-9999:5d}"
    line[40:45] = f"{direction_deg:5d}"
    line[46:51] = f"{speed_tenths_m_s:5d}"
    return "".join(line).rstrip()


def _igra_text(
    top_height: int = 4000,
    top_temperature: int | None = None,
    surface_height: int = 43,
    include_legacy_missing_height_layers: bool = False,
    station: str = "CHM00058457",
    latitude: float = 30.2333,
    longitude: float = 120.1667,
    timestamps: tuple[datetime, ...] | None = None,
) -> str:
    """Return Zhejiang soundings at the requested timestamps."""
    if timestamps is None:
        timestamps = (
            datetime(2018, 3, 18),
            datetime(2018, 3, 18, 12),
        )
    chunks: list[str] = []
    for index, timestamp in enumerate(timestamps):
        offset = index * 20
        levels = [
            _igra_level(
                100000, surface_height, 100 + offset, 270, 50,
                minor_level_type="1",
            ),
            _igra_level(90000, 1000, 40 + offset, 260, 70),
            _igra_level(
                70000, top_height,
                (-80 + offset if top_temperature is None else top_temperature),
                250, 100,
            ),
        ]
        if include_legacy_missing_height_layers:
            levels.extend((
                _igra_level(30000, 9998, -400 + offset, 240, 120),
                _igra_level(25000, 10660, -450 + offset, 230, 140),
            ))
        chunks.append(_igra_header(
            station,
            timestamp,
            len(levels),
            latitude,
            longitude,
        ))
        chunks.extend(levels)
    return "\n".join(chunks) + "\n"


def _observation_sources(
    tmp_path: Path,
    upper_stations: tuple[str, ...] = ("CHM00058457",),
) -> tuple[list[SourceSpec], list[SourceSpec]]:
    """Write compact source fixtures for transactional builder tests."""
    isd = tmp_path / "58457099999.csv"
    _write_isd(isd, [_base_isd_row()])
    surface_sources = [SourceSpec(isd, "https://example.test/isd.csv")]
    upper_sources: list[SourceSpec] = []
    for index, station in enumerate(upper_stations):
        source = tmp_path / f"{station}.txt"
        source.write_text(
            _igra_text(
                station=station,
                latitude=30.2333 - 0.5 * index,
                longitude=120.1667 - 0.5 * index,
            ),
            encoding="ascii",
        )
        upper_sources.append(SourceSpec(
            source, f"https://example.test/{station}.txt"
        ))
    return surface_sources, upper_sources


def _base_isd_row() -> dict[str, str]:
    """Return one complete Hangzhou-like Global Hourly report."""
    return {
        "STATION": "58457099999",
        "DATE": "2018-03-18T00:00:00",
        "LATITUDE": "30.2333",
        "LONGITUDE": "120.1667",
        "ELEVATION": "43.0",
        "NAME": "HANGZHOU, CH",
        "REPORT_TYPE": "FM-12",
        "WND": "270,1,N,0050,1",
        "CIG": "00305,1,N,N",
        "TMP": "+0100,1",
        "DEW": "+0050,1",
        "SLP": "10100,1",
        "MA1": "10132,1,10000,1",
        "GF1": "08,04,1,08,1,99,9,00900,1,99,9,99,9",
        "GA1": "06,1,00900,1,06,1",
        "AA1": "01,0005,0,1",
        "AW1": "",
        "MW1": "",
    }


def _fixed_isd_line(*, cavok: str = "N") -> str:
    """Build one official annual fixed-width ISD record."""
    mandatory = "".join((
        "270", "1", "N", "0050", "1",
        "22000", "1", "9", cavok,
        "016000", "1", "N", "9",
        "+0100", "1", "+0050", "1", "10100", "1",
    ))
    additional = "".join((
        "ADD",
        "GF1", "08", "04", "1", "08", "1", "99", "9", "00900", "1",
        "99", "9", "99", "9",
        "MA1", "10132", "1", "10000", "1",
        "AA1", "01", "0005", "0", "1",
        "REM",
    ))
    body = "".join((
        "584570", "99999", "20180318", "0000", "4",
        "+30233", "+120167", "FM-12", "+0043", "99999", "V020",
        mandatory, additional,
    ))
    return f"{len(additional):04d}{body}"


def test_build_observations_writes_reader_compatible_files_and_manifest(tmp_path: Path) -> None:
    isd = tmp_path / "58457099999.csv"
    _write_isd(isd, [_base_isd_row()])
    igra_zip = tmp_path / "CHM00058457-data.txt.zip"
    with zipfile.ZipFile(igra_zip, "w") as archive:
        archive.writestr("CHM00058457-data.txt", _igra_text())
    output = tmp_path / "calmet"

    surface_path, upper_paths, manifest_path = build_observations(
        [SourceSpec(isd, "https://example.test/global-hourly/58457099999.csv")],
        [SourceSpec(igra_zip, "https://example.test/igra/CHM00058457-data.txt.zip")],
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        output_dir=output,
    )

    surface = SurfaceData(surface_path)
    assert surface.start == datetime(2018, 3, 18)
    assert surface.end == datetime(2018, 3, 18, 1)
    assert surface.timezone == "UTC+0000"
    assert surface.stations[0].identifier == 584570
    assert surface.stations[0].name == "S001"
    record = surface.records[0]
    assert record.wind_speed[0] == pytest.approx(5.0)
    assert record.wind_direction[0] == pytest.approx(270.0)
    assert record.ceiling_hundreds_ft[0] == 10
    assert record.cloud_tenths[0] == 5
    assert record.temperature_k[0] == pytest.approx(283.15, abs=1e-4)
    assert record.relative_humidity[0] == 71
    assert record.pressure_mb[0] == pytest.approx(1000.0)
    assert record.precipitation_code[0] == 10

    assert [path.name for path in upper_paths] == ["UP_CHM00058457.DAT"]
    upper = UpperAirData(upper_paths[0])
    assert upper.station.identifier == 58457
    assert [item.timestamp for item in upper.soundings] == [
        datetime(2018, 3, 18), datetime(2018, 3, 18, 12)
    ]
    assert upper.soundings[0].temperature_k[0] == pytest.approx(283.15, abs=1e-4)
    assert upper.soundings[0].pressure_mb[-1] == pytest.approx(700.0)
    assert upper.soundings[0].wind_speed[-1] == pytest.approx(10.0)
    sounding_headers = [line for line in upper_paths[0].read_text(encoding="ascii").splitlines()
                        if line.startswith("         ")]
    assert sounding_headers and all(len(line) == 76 for line in sounding_headers)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["period_utc"] == {
        "start_inclusive": "2018-03-18T00:00:00Z",
        "end_exclusive": "2018-03-18T01:00:00Z",
    }
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(isd.read_bytes()).hexdigest()
    assert manifest["inputs"][1]["archive_member"] == "CHM00058457-data.txt"
    assert manifest["surface"]["stations"][0]["selected_reports_utc"] == [
        "2018-03-18T00:00:00Z"
    ]


def test_build_requires_upper_snapshot_strictly_after_run_end(tmp_path: Path) -> None:
    isd = tmp_path / "58457099999.csv"
    rows: list[dict[str, str]] = []
    for hour in range(12):
        row = _base_isd_row()
        row["DATE"] = f"2018-03-18T{hour:02d}:00:00"
        rows.append(row)
    _write_isd(isd, rows)
    igra = tmp_path / "CHM00058457.txt"
    igra.write_text(_igra_text(), encoding="ascii")

    with pytest.raises(ValueError, match="snapshot strictly after the run end"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")],
            [SourceSpec(igra, "https://example.test/igra.txt")],
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 12),
            output_dir=tmp_path / "calmet",
        )


def test_built_upper_air_supports_exact_run_end_selection(tmp_path: Path) -> None:
    isd = tmp_path / "58457099999.csv"
    rows: list[dict[str, str]] = []
    for hour in range(12):
        row = _base_isd_row()
        row["DATE"] = f"2018-03-18T{hour:02d}:00:00"
        rows.append(row)
    _write_isd(isd, rows)
    igra = tmp_path / "CHM00058457.txt"
    timestamps = (
        datetime(2018, 3, 18),
        datetime(2018, 3, 18, 12),
        datetime(2018, 3, 19),
    )
    igra.write_text(_igra_text(timestamps=timestamps), encoding="ascii")

    _, upper_paths, _ = build_observations(
        [SourceSpec(isd, "https://example.test/isd.csv")],
        [SourceSpec(igra, "https://example.test/igra.txt")],
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 12),
        output_dir=tmp_path / "calmet",
    )

    upper = UpperAirData(upper_paths[0])
    assert [sounding.timestamp for sounding in upper.soundings] == list(timestamps)
    assert upper.mixing_sounding_at(datetime(2018, 3, 18, 12)).timestamp == timestamps[1]


@pytest.mark.parametrize(
    "manifest_name",
    ["SURF.DAT", "surf.dat", "UP_CHM00058457.DAT"],
)
def test_build_rejects_manifest_name_that_conflicts_with_generated_data(
    tmp_path: Path,
    manifest_name: str,
) -> None:
    surface_sources, upper_sources = _observation_sources(tmp_path)
    output = tmp_path / "calmet"

    with pytest.raises(ValueError, match="conflicts with a generated CALMET file"):
        build_observations(
            surface_sources,
            upper_sources,
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 1),
            output_dir=output,
            manifest_name=manifest_name,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".zhejiang-build-observations-*"))


def test_build_atomically_replaces_complete_directory_without_stale_upper_files(
    tmp_path: Path,
) -> None:
    surface_sources, upper_sources = _observation_sources(
        tmp_path, ("CHM00058457", "CHM00058633")
    )
    output = tmp_path / "calmet"
    build_observations(
        surface_sources,
        upper_sources,
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        output_dir=output,
    )
    assert (output / "UP_CHM00058633.DAT").is_file()

    build_observations(
        surface_sources,
        upper_sources[:1],
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        output_dir=output,
    )

    assert {path.name for path in output.iterdir()} == {
        "SURF.DAT", "UP_CHM00058457.DAT", "manifest.json",
    }
    assert not list(tmp_path.glob(".zhejiang-build-observations-*"))
    assert not list(tmp_path.glob(".zhejiang-build-previous-*"))


def test_build_restores_existing_directory_when_atomic_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface_sources, upper_sources = _observation_sources(tmp_path)
    output = tmp_path / "calmet"
    build_observations(
        surface_sources,
        upper_sources,
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        output_dir=output,
    )
    previous = {
        path.name: path.read_bytes()
        for path in output.iterdir()
    }
    real_replace = builder.os.replace

    def fail_staged_install(source: Path, destination: Path) -> None:
        """Fail only the staged-directory publication, then allow restore."""
        source_path = Path(source)
        if (
            Path(destination) == output
            and source_path.name == "calmet"
            and source_path.parent.name.startswith(".zhejiang-build-observations-")
        ):
            raise OSError("injected observation install failure")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_staged_install)
    with pytest.raises(OSError, match="cannot install observation output"):
        build_observations(
            surface_sources,
            upper_sources,
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 1),
            output_dir=output,
        )

    assert {
        path.name: path.read_bytes()
        for path in output.iterdir()
    } == previous
    assert not list(tmp_path.glob(".zhejiang-build-observations-*"))
    assert not list(tmp_path.glob(".zhejiang-build-previous-*"))


def test_read_isd_station_autodetects_official_fixed_width_gzip(tmp_path: Path) -> None:
    source = tmp_path / "584570-99999-2018.gz"
    with gzip.open(source, "wt", encoding="ascii", newline="\n") as stream:
        stream.write(_fixed_isd_line() + "\n")

    station = read_isd_station(SourceSpec(
        source,
        "https://www.ncei.noaa.gov/pub/data/noaa/2018/584570-99999-2018.gz",
    ))

    assert station.input_format == "isd_fixed_width"
    assert station.source_station_id == "58457099999"
    assert station.latitude == pytest.approx(30.233)
    assert station.longitude == pytest.approx(120.167)
    report = station.reports[0]
    assert report.timestamp == datetime(2018, 3, 18)
    assert report.values.wind_speed_m_s == pytest.approx(5.0)
    assert report.values.cloud_tenths == 5
    assert report.values.station_pressure_mb == pytest.approx(1000.0)


@pytest.mark.parametrize("compression", ["plain", "gzip"])
def test_load_text_rejects_payload_beyond_expansion_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
) -> None:
    source = tmp_path / ("source.gz" if compression == "gzip" else "source.txt")
    if compression == "gzip":
        with gzip.open(source, "wb") as stream:
            stream.write(b"A" * 65)
    else:
        source.write_bytes(b"A" * 65)
    monkeypatch.setattr(builder, "MAX_TEXT_SOURCE_BYTES", 64)

    with pytest.raises(ValueError, match="safety limit"):
        builder._load_text(
            builder.SourceSpec(source, "https://example.test/source"),
            encoding="ascii",
        )


def test_read_isd_stations_flattens_multistation_global_hourly_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stations.csv"
    second = _base_isd_row() | {
        "STATION": "58367099999",
        "LATITUDE": "31.198",
        "LONGITUDE": "121.336",
        "ELEVATION": "3.0",
        "NAME": "SHANGHAI HONGQIAO, CH",
    }
    _write_isd(source, [_base_isd_row(), second])

    stations = read_isd_stations(SourceSpec(source, "https://example.test/stations.csv"))

    assert [item.source_station_id for item in stations] == [
        "58457099999", "58367099999",
    ]


def test_decode_isd_row_applies_qc_fallbacks_and_frozen_weather() -> None:
    row = _base_isd_row()
    row.update({
        "WND": "270,3,N,0050,1",
        "CIG": "00305,7,N,N",
        "TMP": "-0050,1",
        "DEW": "-0100,1",
        "MA1": "10132,1,10000,3",
        "GF1": "06,04,3,08,1,99,9,00900,1,99,9,99,9",
        "AA1": "",
        "MW1": "75,1",
    })

    values = decode_isd_row(row, elevation_m=100.0)

    assert values.wind_speed_m_s is None
    assert values.wind_direction_deg is None
    assert values.ceiling_hundreds_ft is None
    assert values.temperature_k == pytest.approx(268.15)
    assert values.relative_humidity_percent == 68
    assert values.station_pressure_mb == pytest.approx(1001.3, abs=0.2)
    assert values.cloud_tenths == 8
    assert values.precipitation_code == 20


def test_decode_isd_wind_type_codes_handle_calm_and_variable() -> None:
    calm = _base_isd_row()
    calm["WND"] = "999,7,C,9999,7"
    calm_values = decode_isd_row(calm, elevation_m=43.0)
    assert calm_values.wind_speed_m_s == 0.0
    assert calm_values.wind_direction_deg == 0.0

    variable = _base_isd_row()
    variable["WND"] = "999,1,V,0040,1"
    variable_values = decode_isd_row(variable, elevation_m=43.0)
    assert variable_values.wind_speed_m_s is None
    assert variable_values.wind_direction_deg is None


def test_decode_isd_uses_surfgen_present_weather_mapping() -> None:
    automated_snow = _base_isd_row() | {
        "AA1": "", "AW1": "24,1", "MW1": "",
    }
    manual_drizzle = _base_isd_row() | {
        "TMP": "-0050,1", "DEW": "-0060,1",
        "AA1": "", "AW1": "", "MW1": "20,1",
    }

    assert decode_isd_row(automated_snow, 43.0).precipitation_code == 20
    assert decode_isd_row(manual_drizzle, 43.0).precipitation_code == 10


def test_decode_isd_cloud_uses_gf_layout_then_gd_and_ga_fallbacks() -> None:
    row = _base_isd_row()
    row["GF1"] = "08,09,1,09,1,99,9,99999,9,99,9,99,9"
    assert decode_isd_row(row, elevation_m=43.0).cloud_tenths == 10

    row["GF1"] = "99,99,9,99,9,99,9,99999,9,99,9,99,9"
    row["GD1"] = "2,04,1,+99999,9"
    row["GD2"] = "2,07,1,+99999,9"
    row["GA1"] = "08,1,+00900,1,06,1"
    assert decode_isd_row(row, elevation_m=43.0).cloud_tenths == 9

    row["GD2"] = "2,07,3,+99999,9"
    assert decode_isd_row(row, elevation_m=43.0).cloud_tenths == 5


@pytest.mark.parametrize("encoded_ceiling", ["22000,1,9,N", "99999,9,9,N"])
def test_clear_sky_uses_nonmissing_unlimited_ceiling(encoded_ceiling: str) -> None:
    row = _base_isd_row()
    row["CIG"] = encoded_ceiling
    row["GF1"] = "00,00,1,99,9,99,9,99999,9,99,9,99,9"

    values = decode_isd_row(row, elevation_m=43.0)

    assert values.cloud_tenths == 0
    assert values.ceiling_hundreds_ft == 999


def test_cavok_supplies_clear_cloud_and_unlimited_ceiling() -> None:
    row = _base_isd_row()
    row["CIG"] = "99999,9,9,Y"
    row["GF1"] = ""
    row["GA1"] = ""

    values = decode_isd_row(row, elevation_m=43.0)

    assert values.cloud_tenths == 0
    assert values.ceiling_hundreds_ft == 999


def test_cloudy_report_does_not_invent_a_missing_ceiling() -> None:
    row = _base_isd_row()
    row["CIG"] = "99999,9,9,N"

    values = decode_isd_row(row, elevation_m=43.0)

    assert values.cloud_tenths == 5
    assert values.ceiling_hundreds_ft is None


def test_parse_igra_level_converts_units_and_missing_values() -> None:
    level = parse_igra_level(_igra_level(92500, 800, -55, 315, 123))
    assert level is not None
    assert level.pressure_mb == pytest.approx(925.0)
    assert level.height_msl == pytest.approx(800.0)
    assert level.temperature_k == pytest.approx(267.65)
    assert level.wind_direction_deg == pytest.approx(315.0)
    assert level.wind_speed_m_s == pytest.approx(12.3)
    missing = parse_igra_level(_igra_level(92500, 800, -9999, -9999, -9999))
    assert missing is not None
    assert missing.temperature_k is None and missing.wind_speed_m_s is None
    assert parse_igra_level(
        _igra_level(92500, 800, -55, 315, 123, level_type="3")
    ) is None


def test_parse_igra_level_applies_processing_flags_and_surface_override() -> None:
    assert parse_igra_level(
        _igra_level(92500, 800, -55, 315, 123, pressure_flag="X")
    ) is None
    assert parse_igra_level(
        _igra_level(92500, 800, -55, 315, 123, height_flag="X")
    ) is None
    bad_temperature = parse_igra_level(
        _igra_level(92500, 800, -55, 315, 123, temperature_flag="X")
    )
    assert bad_temperature is not None
    assert bad_temperature.temperature_k is None
    surface = parse_igra_level(
        _igra_level(
            100000, -9999, 100, 270, 50,
            minor_level_type="1", pressure_flag="B", temperature_flag="B",
        ),
        station_elevation_m=43.0,
    )
    assert surface is not None
    assert surface.is_surface
    assert surface.height_msl == pytest.approx(43.0)


def test_upper_writer_retains_raw_high_levels_but_omits_calmet_sentinel_range(
    tmp_path: Path,
) -> None:
    isd = tmp_path / "station.csv"
    _write_isd(isd, [_base_isd_row()])
    igra = tmp_path / "station.txt"
    igra.write_text(
        _igra_text(include_legacy_missing_height_layers=True),
        encoding="ascii",
    )
    igra_source = SourceSpec(igra, "https://example.test/igra.txt")

    retained = read_igra_station(
        igra_source,
        datetime(2018, 3, 18),
        datetime(2018, 3, 18, 1),
        3000.0,
        {},
    )
    assert [level.height_msl for level in retained.soundings[0].levels][-2:] == [
        9998.0,
        10660.0,
    ]

    _, upper_paths, manifest_path = build_observations(
        [SourceSpec(isd, "https://example.test/isd.csv")],
        [igra_source],
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        output_dir=tmp_path / "output",
    )
    upper = UpperAirData(upper_paths[0])
    assert all(
        height < 9998.0
        for sounding in upper.soundings
        for height in sounding.height_msl
    )
    assert [max(sounding.height_msl) for sounding in upper.soundings] == [
        4000.0,
        4000.0,
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upper_air"]["emitted_height_policy"] == (
        "CALMET legacy missing-height sentinel limits emitted heights to <9998m"
    )
    profiles = manifest["upper_air"]["stations"][0]["profiles"]
    assert [item["source_valid_level_count"] for item in profiles] == [5, 5]
    assert [item["removed_height_ge_9998_m_count"] for item in profiles] == [2, 2]
    assert [item["emitted_level_count"] for item in profiles] == [3, 3]
    assert [item["emitted_top_height_msl_m"] for item in profiles] == [
        4000.0,
        4000.0,
    ]


def test_build_rejects_upper_profile_without_requested_depth(tmp_path: Path) -> None:
    isd = tmp_path / "station.csv"
    _write_isd(isd, [_base_isd_row()])
    igra = tmp_path / "station.txt"
    igra.write_text(_igra_text(top_height=1500), encoding="ascii")

    with pytest.raises(ValueError, match="no valid soundings bracketing"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")],
            [SourceSpec(igra, "https://example.test/igra.txt")],
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 1),
            output_dir=tmp_path / "output",
            minimum_upper_wind_height_m=3000.0,
        )


def test_build_rejects_temperature_profile_below_requested_depth(
    tmp_path: Path,
) -> None:
    isd = tmp_path / "station.csv"
    _write_isd(isd, [_base_isd_row()])
    igra = tmp_path / "station.txt"
    igra.write_text(_igra_text(top_temperature=-9999), encoding="ascii")

    with pytest.raises(ValueError, match="no valid soundings bracketing"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")],
            [SourceSpec(igra, "https://example.test/igra.txt")],
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 1),
            output_dir=tmp_path / "output",
            minimum_upper_wind_height_m=3000.0,
        )


def test_build_requires_elevation_override_for_missing_igra_surface_gph(
    tmp_path: Path,
) -> None:
    isd = tmp_path / "station.csv"
    _write_isd(isd, [_base_isd_row()])
    igra = tmp_path / "station.txt"
    igra.write_text(_igra_text(surface_height=-9999), encoding="ascii")
    sources = [SourceSpec(igra, "https://example.test/igra.txt")]

    with pytest.raises(ValueError, match="no valid soundings bracketing"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")], sources,
            start=datetime(2018, 3, 18), end=datetime(2018, 3, 18, 1),
            output_dir=tmp_path / "without-override",
        )

    _, upper_paths, manifest_path = build_observations(
        [SourceSpec(isd, "https://example.test/isd.csv")], sources,
        start=datetime(2018, 3, 18), end=datetime(2018, 3, 18, 1),
        output_dir=tmp_path / "with-override",
        igra_station_elevations_m={"CHM00058457": 43.0},
    )
    upper = UpperAirData(upper_paths[0])
    assert upper.soundings[0].height_msl[0] == pytest.approx(43.0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upper_air"]["station_elevation_overrides_m"] == {
        "CHM00058457": 43.0,
    }


def test_build_rejects_selected_reports_that_mix_station_sites(tmp_path: Path) -> None:
    isd = tmp_path / "station.csv"
    first = _base_isd_row()
    second = _base_isd_row() | {
        "DATE": "2018-03-18T01:00:00",
        "LATITUDE": "30.228",
        "LONGITUDE": "120.432",
        "ELEVATION": "7.0",
        "REPORT_TYPE": "FM-15",
    }
    _write_isd(isd, [first, second])
    igra = tmp_path / "station.txt"
    igra.write_text(_igra_text(), encoding="ascii")

    with pytest.raises(ValueError, match="mix different station sites"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")],
            [SourceSpec(igra, "https://example.test/igra.txt")],
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 2),
            output_dir=tmp_path / "output",
        )


def test_build_rejects_hour_without_required_cross_station_surface_data(
    tmp_path: Path,
) -> None:
    isd = tmp_path / "station.csv"
    row = _base_isd_row()
    row["DATE"] = "2018-03-17T23:00:00"
    _write_isd(isd, [row])
    igra = tmp_path / "station.txt"
    igra.write_text(_igra_text(), encoding="ascii")

    with pytest.raises(ValueError, match="lack cross-station coverage"):
        build_observations(
            [SourceSpec(isd, "https://example.test/isd.csv")],
            [SourceSpec(igra, "https://example.test/igra.txt")],
            start=datetime(2018, 3, 18),
            end=datetime(2018, 3, 18, 1),
            output_dir=tmp_path / "output",
        )


def test_parse_utc_requires_explicit_zero_offset() -> None:
    assert parse_utc("2018-03-18T00:00:00Z") == datetime(2018, 3, 18)
    with pytest.raises(ValueError, match="explicit UTC"):
        parse_utc("2018-03-18T00:00:00")
    with pytest.raises(ValueError, match="explicit UTC"):
        parse_utc("2018-03-18T08:00:00+08:00")
