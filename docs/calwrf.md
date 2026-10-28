# CALWRF

CALWRF reads WRF-ARW NetCDF and writes the three-dimensional meteorology
CALMET expects: pressure, temperature, water vapor, hydrometeors, and
earth-relative wind on mass levels, plus surface wind and precipitation
increments.

```python
from pathlib import Path
import pycalmet

output = pycalmet.calwrf(
    FLNAMES=[Path("wrfout_d01_2018-03-18_00_00_00.nc")],
    NFILES=1,
    IDATEB="2018031800",
    IDATEE="2018031800",
    FIO=Path("output/calwrf.nc"),
)
```

`calwrf` accepts either that legacy control vocabulary or a `CalwrfConfig`.
`output` is the NetCDF path. A traditional M3D file is written only when
`legacy_output` is set. Horizontal and vertical subsets use one-based
inclusive index ranges (`i_range`, `j_range`, `k_range`); the default keeps
the full WRF grid.

## What the kernel does

For each selected hour the reader pulls WRF fields with `netCDF4`, then:

1. Destaggers U, V, W, and geopotential onto mass points by averaging
   neighbors.
2. Converts perturbation pressure and potential temperature to hectopascals
   and kelvins.
3. Rotates grid-relative wind into eastward and northward components using
   the WRF map projection, then reports meteorological speed and direction.
4. Turns cumulative `RAINC` and `RAINNC` into hourly increments, treating a
   drop in the accumulator as a reset.

Those four steps are the CALWRF kernels on the [array backend](backends.md).
Range checks that replace a few invalid cells from their neighbors stay on
NumPy; they are small compared with the volume fields and they use masked
index updates.

WRF times are naive timestamps in the file. CALWRF's window is inclusive, so
a one-hour CALMET period needs the record at the start and the record at the
end. The Zhejiang configuration documents that bracket in
`examples/zhejiang/config/calwrf.inp`.
