# Zhejiang one-hour example

The three WRF files are the 2018-03-18 00:00, 01:00, and 02:00 UTC records
from the original Zhejiang 264 × 264 × 35 WRF output. They retain every
variable read by CALWRF and use lossless NetCDF4 compression; unrelated WRF
variables and later records were removed. Running either the GNU Fortran or
Python CALWRF implementation on any file produces the same CALWRF fields as
the corresponding record in the original 3.0 GiB file. The third record is
the following prognostic boundary required for a 00:00–01:00 CALMET run.

`geo.nc` is the canonical NetCDF conversion of the bundled Zhejiang
`MAKEGEO/geo.dat`. The two files support a one-hour validation without the
external land-use source used to regenerate the static field.

From the repository root:

```bash
sha256sum -c example/zhejiang/SHA256SUMS

./run_zhejiang_python.sh \
  --wrf example/zhejiang \
  --geo example/zhejiang/geo.nc \
  --start 2018031800 --end 2018031801 \
  --skip-static \
  --output output/zhejiang_example_python.nc
```

For the GNU Fortran path, first run `./build_gnu.sh`, then:

```bash
./run_zhejiang.sh \
  --wrf example/zhejiang \
  --start 2018031800 --end 2018031801 \
  --skip-static \
  --output output/zhejiang_example_fortran.dat
```

The raw Zhejiang land-use raster is not bundled. Regenerating `geo.dat` from
scratch therefore still requires a CTGPROC-compatible land-use input.

The Python runner keeps a canonical `calwrf.nc` intermediate and also writes a
traditional `calwrf.m3d`. CALMET reads the M3D so its fixed-width input
quantization matches the GNU Fortran workflow.

Release validation compared all 108 retained variable/time pairs
(107,317,746 values) with the 3.0 GiB source without a mismatch. Across the
three hours, Python and GNU CALWRF agreed bit-for-bit for 2,090,880 surface,
58,544,640 level, and 36,590,400 water-species values. The resulting one-hour
Python and GNU CALMET runs also agreed bit-for-bit for all 31 output fields
(9,436,090 values).
