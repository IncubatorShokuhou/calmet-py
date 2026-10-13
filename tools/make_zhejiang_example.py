"""Create a losslessly compressed, CALWRF-complete one-record WRF example."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc


CALWRF_VARIABLES = (
    "Times", "XLAT", "XLONG", "HGT", "LU_INDEX", "ZNU",
    "U", "V", "W", "PH", "PHB", "T", "P", "PB", "QVAPOR",
    "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    "U10", "V10", "PSFC", "SNOWC", "T2", "Q2", "RAINC", "RAINNC",
    "SWDOWN", "GLW", "SST", "TSK", "PBLH", "HFX", "LH", "UST",
)


def create_example(source: Path, output: Path, time_index: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(source) as src, nc.Dataset(output, "w", format="NETCDF4") as dst:
        if not 0 <= time_index < len(src.dimensions["Time"]):
            raise IndexError(f"time index {time_index} is outside {source}")
        missing = [name for name in CALWRF_VARIABLES if name not in src.variables]
        if missing:
            raise ValueError("source is missing CALWRF variables: " + ", ".join(missing))

        for name, dimension in src.dimensions.items():
            dst.createDimension(name, 1 if name == "Time" else len(dimension))
        dst.setncatts({name: src.getncattr(name) for name in src.ncattrs()})

        for name in CALWRF_VARIABLES:
            source_variable = src.variables[name]
            fill = (source_variable.getncattr("_FillValue")
                    if "_FillValue" in source_variable.ncattrs() else None)
            options = {}
            if source_variable.dtype.kind not in "SU":
                options.update(zlib=True, complevel=9, shuffle=True)
            if fill is not None:
                options["fill_value"] = fill
            target = dst.createVariable(
                name, source_variable.dtype, source_variable.dimensions, **options
            )
            target.setncatts({
                attr: source_variable.getncattr(attr)
                for attr in source_variable.ncattrs() if attr != "_FillValue"
            })
            selection = [slice(None)] * source_variable.ndim
            if "Time" in source_variable.dimensions:
                axis = source_variable.dimensions.index("Time")
                selection[axis] = slice(time_index, time_index + 1)
            target[...] = source_variable[tuple(selection)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="full WRF output")
    parser.add_argument("output", type=Path, help="example NetCDF path")
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()
    create_example(args.source, args.output, args.time_index)


if __name__ == "__main__":
    main()
