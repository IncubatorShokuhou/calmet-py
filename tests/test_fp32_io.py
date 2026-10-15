from types import SimpleNamespace

import netCDF4 as nc
import numpy as np

from pycalmet._calwrf import (
    _create_output,
    _geopotential_height,
    _pressure_temperature,
    destagger_x,
    earth_wind,
    rotation_angle,
    speed_direction,
)
from pycalmet.calmet_output import CalmetNetCDFWriter
from pycalmet.config import GridConfig


def test_calwrf_physical_math_crosses_to_fp32_at_input_boundary():
    values = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    pressure, temperature = _pressure_temperature(values, values + 90_000.0, values)
    height = _geopotential_height(
        np.zeros((2, 1, 1), dtype=np.float64),
        np.array([0.0, 196.2], dtype=np.float64).reshape(2, 1, 1),
    )
    angle = rotation_angle(
        SimpleNamespace(MAP_PROJ=2, STAND_LON=120.0),
        np.array([[30.0]], dtype=np.float64),
        np.array([[121.0]], dtype=np.float64),
    )
    east, north = earth_wind(values, values + 1.0, np.zeros_like(values))
    speed, direction = speed_direction(east, north)

    for result in (
        destagger_x(values), pressure, temperature, height, angle,
        east, north, speed, direction,
    ):
        assert result.dtype == np.float32


def test_calmet_netcdf_schema_uses_fp32_for_fields_and_fp64_for_time(tmp_path):
    path = tmp_path / "calmet.nc"
    geo_path = tmp_path / "geo.nc"
    grid = GridConfig(1, 1, 1.0, 1.0)
    with nc.Dataset(geo_path, "w") as geophysical:
        geophysical.createDimension("y", 1)
        geophysical.createDimension("x", 1)
        geophysical.createVariable("terrain_height", "f8", ("y", "x"))[:] = np.ones(
            (1, 1), dtype=np.float64,
        )
        geophysical.createVariable("dominant_landuse", "f8", ("y", "x"))[:] = np.full(
            (1, 1), 20.0, dtype=np.float64,
        )
        with CalmetNetCDFWriter(path, grid, [0.0, 20.0], geophysical):
            pass

    with nc.Dataset(path) as dataset:
        assert dataset["time"].dtype == np.dtype("float64")
        assert dataset["latitude"].dtype == np.dtype("float64")
        assert dataset["longitude"].dtype == np.dtype("float64")
        assert dataset["level_height"].dtype == np.dtype("float32")
        assert dataset["terrain_height"].dtype == np.dtype("float32")
        assert dataset["wind_u"].dtype == np.dtype("float32")
        assert dataset["surface_relative_humidity"].dtype == np.dtype("float32")
        assert dataset["landuse"].dtype == np.dtype("int16")
        assert dataset["precipitation_code"].dtype == np.dtype("int8")


def test_calwrf_netcdf_schema_uses_fp32_fields_and_integer_landuse(tmp_path):
    source_path = tmp_path / "wrf.nc"
    output_path = tmp_path / "calwrf.nc"
    with nc.Dataset(source_path, "w") as source:
        for name, size in (
            ("Time", 1), ("south_north", 1), ("west_east", 1),
            ("bottom_top", 1),
        ):
            source.createDimension(name, size)
        surface_dims = ("Time", "south_north", "west_east")
        for name, value in (
            ("XLAT", 30.0), ("XLONG", 120.0), ("HGT", 10.0),
            ("LU_INDEX", 20.0),
        ):
            source.createVariable(name, "f8", surface_dims)[:] = np.full(
                (1, 1, 1), value, dtype=np.float64,
            )
        source.createVariable("ZNU", "f8", ("Time", "bottom_top"))[:] = np.ones(
            (1, 1), dtype=np.float64,
        )
        source.DX = 1_000.0
        source.DY = 1_000.0
        output = _create_output(
            output_path, source, 1, 1, 1, 1,
            slice(0, 1), slice(0, 1), slice(0, 1),
        )
        output.close()

    with nc.Dataset(output_path) as dataset:
        assert dataset["pressure"].dtype == np.dtype("float32")
        assert dataset["latitude"].dtype == np.dtype("float64")
        assert dataset["longitude"].dtype == np.dtype("float64")
        assert dataset["terrain"].dtype == np.dtype("float32")
        assert dataset["sigma"].dtype == np.dtype("float32")
        assert dataset["landuse"].dtype == np.dtype("int16")
