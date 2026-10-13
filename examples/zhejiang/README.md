# Zhejiang one-hour example

The three WRF files are the 2018-03-18 00:00, 01:00, and 02:00 UTC records
from the original Zhejiang 264 × 264 × 35 WRF output. They retain every
variable read by CALWRF and use lossless NetCDF4 compression; unrelated WRF
variables and later records were removed. Running the Python CALWRF
implementation on any file produces the same CALWRF fields as
the corresponding record in the original 3.0 GiB file. The third record is
the following prognostic boundary required for a 00:00–01:00 CALMET run.

`geo.nc` is the prepared canonical geophysical dataset. It supports a
one-hour validation without the external land-use source used to regenerate
the static field.

From the repository root:

```bash
sha256sum -c examples/zhejiang/SHA256SUMS

./examples/zhejiang/run.sh \
  --wrf examples/zhejiang \
  --geo examples/zhejiang/geo.nc \
  --start 2018031800 --end 2018031801 \
  --skip-static \
  --output output/zhejiang_example_python.nc
```

The raw Zhejiang land-use raster is not bundled. Regenerating `geo.dat` from
scratch therefore requires a CTGPROC-compatible land-use input. The 42 SRTM3
tiles used by this domain are stored separately in `terrain/srtm3/`.

The Python runner keeps a canonical `calwrf.nc` intermediate and also writes a
traditional `calwrf.m3d`. CALMET reads the M3D so its fixed-width input
quantization matches the GNU Fortran workflow.

Release validation compared all 108 retained variable/time pairs
(107,317,746 values) with the 3.0 GiB source without a mismatch. Across the
three hours, Python and the retained Fortran reference agreed bit-for-bit for
2,090,880 surface, 58,544,640 level, and 36,590,400 water-species values. The
resulting one-hour CALMET runs also agreed bit-for-bit for all 31 output fields
(9,436,090 values).
