# Raw geospatial inputs

This directory keeps the two raw geospatial inputs used by the Zhejiang
example at the same level:

- `landuse/usgs-glcc/` contains the original USGS GLCC categorical raster read
  by CTGPROC.
- `terrain/srtm3/` contains the original SRTM3 elevation tiles read by TERREL.

Derived products such as `LULC.DAT`, `TERR.DAT`, and `geo.dat` do not belong
here. The staged example may write their NetCDF equivalents under `output/`,
while `examples/zhejiang/run.py` passes them in memory and retains only the
final CALMET file.

Each source directory includes its own format notes and SHA-256 manifest.
