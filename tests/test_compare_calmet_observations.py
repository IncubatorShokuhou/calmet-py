"""Focused tests for the Python/Fortran CALMET output comparator."""

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from pycalmet.calmet_output import LegacyCalmetWriter
from pycalmet.calmet_solver import target_latitude_longitude
from pycalmet.config import GridConfig
from tools.compare_calmet_observations import compare, main


GRID = GridConfig(
    2,
    2,
    2.0,
    2.0,
    projection="LCC",
    datum="WGS-84",
    lat0=30.0,
    lon0=120.0,
    lat1=30.0,
    lat2=60.0,
    x_origin_km=10.0,
    y_origin_km=20.0,
)
FACES = np.array([0.0, 20.0, 60.0], dtype=np.float32)
START = datetime(2018, 3, 18)
END = START + timedelta(hours=1)


def _write_legacy(path: Path) -> None:
    """Write one complete synthetic CALMET.DAT period."""
    shape = (GRID.ny, GRID.nx)
    geophysical = {
        "roughness_length": np.full(shape, 0.1, dtype=np.float32),
        "dominant_landuse": np.full(shape, 20, dtype=np.int32),
        "terrain_height": np.zeros(shape, dtype=np.float32),
        "leaf_area_index": np.ones(shape, dtype=np.float32),
    }
    writer = LegacyCalmetWriter(
        path,
        GRID,
        FACES,
        geophysical,
        START,
        END,
        run_type=1,
        precipitation_station_count=0,
    )
    for level in range(1, len(FACES)):
        writer._field_record(
            f"U-LEV{level:3d}", np.full(shape, level, dtype=np.float32), START, END
        )
        writer._field_record(
            f"V-LEV{level:3d}", np.full(shape, level + 2, dtype=np.float32), START, END
        )
    writer._field_record("IPGT", np.ones(shape, dtype=np.int32), START, END, integer=True)
    writer._field_record(
        "USTAR",
        np.array([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32),
        START,
        END,
    )
    writer._field_record("ZI", np.ones(shape, dtype=np.float32), START, END)
    writer._field_record(
        "EL",
        np.array([[9.9e9, 1.0], [1.0, 1.0]], dtype=np.float32),
        START,
        END,
    )
    for label in ("WSTAR", "TEMPK", "RHO", "QSW"):
        writer._field_record(label, np.ones(shape, dtype=np.float32), START, END)
    writer._field_record("IRH", np.ones(shape, dtype=np.int32), START, END, integer=True)
    writer.close()


def _write_netcdf(
    path: Path,
    *,
    grid: GridConfig = GRID,
    coordinates: str = "geographic",
) -> None:
    """Write the corresponding compact PyCALMET-style NetCDF output."""
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("level", len(FACES) - 1)
        dataset.createDimension("face", len(FACES))
        dataset.createDimension("y", grid.ny)
        dataset.createDimension("x", grid.nx)
        dataset.projection = grid.projection
        dataset.datum = grid.datum
        dataset.x_origin_km = grid.x_origin_km
        dataset.y_origin_km = grid.y_origin_km

        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 1970-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = netCDF4.date2num([START], time.units, time.calendar)

        level_height = dataset.createVariable("level_height", "f4", ("level",))
        level_height.units = "m"
        level_height[:] = 0.5 * (FACES[:-1] + FACES[1:])
        face_height = dataset.createVariable("face_height", "f4", ("face",))
        face_height.units = "m"
        face_height[:] = FACES

        terrain = dataset.createVariable("terrain_height", "f4", ("y", "x"))
        terrain.units = "m"
        terrain[:, :] = np.zeros((grid.ny, grid.nx), dtype=np.float32)
        landuse = dataset.createVariable("landuse", "i4", ("y", "x"))
        landuse[:, :] = np.full((grid.ny, grid.nx), 20, dtype=np.int32)

        if coordinates == "geographic":
            latitude_values, longitude_values = target_latitude_longitude(grid)
            latitude = dataset.createVariable("latitude", "f8", ("y", "x"))
            longitude = dataset.createVariable("longitude", "f8", ("y", "x"))
            latitude.units = "degrees_north"
            longitude.units = "degrees_east"
            latitude[:, :] = np.asarray(latitude_values).copy()
            longitude[:, :] = np.asarray(longitude_values).copy()
        elif coordinates == "projected":
            x = dataset.createVariable("x", "f8", ("x",))
            y = dataset.createVariable("y", "f8", ("y",))
            x.units = "m"
            y.units = "m"
            x[:] = 1000.0 * (
                grid.x_origin_km
                + (np.arange(grid.nx, dtype=np.float64) + 0.5) * grid.dx_km
            )
            y[:] = 1000.0 * (
                grid.y_origin_km
                + (np.arange(grid.ny, dtype=np.float64) + 0.5) * grid.dy_km
            )
        else:
            raise AssertionError(f"unsupported test coordinate mode {coordinates}")

        wind_shape = (1, len(FACES) - 1, grid.ny, grid.nx)
        wind_u = dataset.createVariable("wind_u", "f4", ("time", "level", "y", "x"))
        wind_v = dataset.createVariable("wind_v", "f4", ("time", "level", "y", "x"))
        wind_u[:, :, :, :] = np.broadcast_to(
            np.arange(1, len(FACES), dtype=np.float32)[None, :, None, None],
            wind_shape,
        ).copy()
        wind_v[:, :, :, :] = np.broadcast_to(
            np.arange(3, len(FACES) + 2, dtype=np.float32)[None, :, None, None],
            wind_shape,
        ).copy()

        friction = dataset.createVariable(
            "friction_velocity",
            "f4",
            ("time", "y", "x"),
            fill_value=np.float32(-9999.0),
        )
        friction_values = np.ones((grid.ny, grid.nx), dtype=np.float32)
        if (grid.ny, grid.nx) == (2, 2):
            friction_values[:, :] = np.array(
                [[1.0, -9999.0], [3.0, -9999.0]], dtype=np.float32
            )
        friction[0, :, :] = friction_values
        monin = dataset.createVariable(
            "monin_obukhov_length", "f4", ("time", "y", "x")
        )
        monin_values = np.ones((grid.ny, grid.nx), dtype=np.float32)
        monin_values[0, 0] = np.float32(9.9e9)
        monin[0, :, :] = monin_values


def _outputs(tmp_path: Path, *, coordinates: str = "geographic") -> tuple[Path, Path]:
    legacy = tmp_path / "calmet.dat"
    netcdf = tmp_path / "calmet.nc"
    _write_legacy(legacy)
    _write_netcdf(netcdf, coordinates=coordinates)
    return netcdf, legacy


def test_compare_reports_directional_and_disagreeing_missing_masks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    netcdf, legacy = _outputs(tmp_path)

    mode, results, _ = compare(netcdf, legacy)

    assert mode == "begin"
    friction = results["friction_velocity"]
    assert friction.count == 1
    assert friction.total_count == 4
    assert friction.netcdf_missing_count == 2
    assert friction.fortran_missing_count == 2
    assert friction.missing_mask_disagreement_count == 2
    assert friction.missing_mask_disagreement_fraction == pytest.approx(0.5)
    assert friction.bias == pytest.approx(0.0)
    monin = results["monin_obukhov_length"]
    assert monin.count == 3
    assert monin.netcdf_missing_count == 1
    assert monin.fortran_missing_count == 1
    assert monin.missing_mask_disagreement_count == 0
    assert monin.rmse == pytest.approx(0.0)
    assert results["terrain_height"].rmse == pytest.approx(0.0)
    assert results["landuse"].mismatch_fraction == pytest.approx(0.0)

    assert main([str(netcdf), str(legacy)]) == 0
    output = capsys.readouterr().out
    assert "nc_missing" in output
    assert "ft_missing" in output
    assert "mask_disagree" in output
    assert "50%" in next(
        line for line in output.splitlines() if line.startswith("friction_velocity")
    )


def test_compare_reports_static_geophysical_differences(tmp_path: Path) -> None:
    netcdf, legacy = _outputs(tmp_path)
    with netCDF4.Dataset(netcdf, "r+") as dataset:
        dataset.variables["terrain_height"][0, 0] = np.float32(3.0)
        dataset.variables["landuse"][0, 1] = np.int32(40)

    _, results, _ = compare(netcdf, legacy)

    assert results["terrain_height"].rmse == pytest.approx(1.5)
    assert results["landuse"].mismatch_fraction == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("projection", "projection metadata projection"),
        ("datum", "projection metadata datum"),
        ("x_origin", "projection metadata x_origin_km"),
        ("latitude", "latitude coordinate differs"),
        ("level_height", "level_height coordinate differs"),
        ("face_height", "face_height coordinate differs"),
    ],
)
def test_compare_rejects_incompatible_metadata_and_coordinates(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    netcdf, legacy = _outputs(tmp_path)
    with netCDF4.Dataset(netcdf, "r+") as dataset:
        if change == "projection":
            dataset.projection = "UTM"
        elif change == "datum":
            dataset.datum = "NWS-84"
        elif change == "x_origin":
            dataset.x_origin_km = float(dataset.x_origin_km) + 1.0
        else:
            variable = dataset.variables[change]
            values = np.asarray(variable[:])
            values.flat[0] += 1.0
            variable[...] = values

    with pytest.raises(ValueError, match=message):
        compare(netcdf, legacy)


def test_compare_rejects_horizontal_grid_shape_before_field_statistics(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "calmet.dat"
    netcdf = tmp_path / "wrong-grid.nc"
    _write_legacy(legacy)
    wrong_grid = GridConfig(
        3,
        GRID.ny,
        GRID.dx_km,
        GRID.dy_km,
        projection=GRID.projection,
        datum=GRID.datum,
        lat0=GRID.lat0,
        lon0=GRID.lon0,
        lat1=GRID.lat1,
        lat2=GRID.lat2,
        x_origin_km=GRID.x_origin_km,
        y_origin_km=GRID.y_origin_km,
    )
    _write_netcdf(netcdf, grid=wrong_grid)

    with pytest.raises(ValueError, match="horizontal grid shape differs"):
        compare(netcdf, legacy)


def test_compare_accepts_projected_axes_and_rejects_shifted_axis(
    tmp_path: Path,
) -> None:
    netcdf, legacy = _outputs(tmp_path, coordinates="projected")

    mode, results, _ = compare(netcdf, legacy)

    assert mode == "begin"
    assert results["wind_u"].count == 8

    with netCDF4.Dataset(netcdf, "r+") as dataset:
        dataset.variables["x"][0] += 100.0
    with pytest.raises(ValueError, match="x coordinate differs"):
        compare(netcdf, legacy)
