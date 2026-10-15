# Raw geospatial data

[中文文档](README.zh-CN.md)

This directory keeps the versioned raw geospatial sources separate from the
Python package and runnable examples:

- [`landuse/usgs-glcc/`](landuse/usgs-glcc/) contains the Asia and Europe
  regional USGS GLCC rasters archived for CTGPROC. The Zhejiang example reads
  only the Asia raster, `EAUSGS1.IMG`.
- [`terrain/srtm3/`](terrain/srtm3/) contains the original SRTM3 elevation
  tiles read by TERREL for the Zhejiang domain.

Derived products such as `LULC.DAT`, `TERR.DAT`, and `geo.dat` do not belong
here. The staged example may write their NetCDF equivalents under `output/`,
while `examples/zhejiang/run.py` passes them in memory and retains only the
final CALMET file.

Each source directory includes format, provenance, and SHA-256 information.
