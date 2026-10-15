# Raw geospatial data

[中文文档](README.zh-CN.md)

This directory keeps scientific inputs separate from the Python package and
runnable examples. Redistributable geospatial sources are versioned here;
restricted meteorological observations are reproducibly acquired for local use:

- [`landuse/usgs-glcc/`](landuse/usgs-glcc/) contains the Asia and Europe
  regional USGS GLCC rasters archived for CTGPROC. The Zhejiang example reads
  only the Asia raster, `EAUSGS1.IMG`.
- [`terrain/srtm3/`](terrain/srtm3/) contains the original SRTM3 elevation
  tiles read by TERREL for the Zhejiang domain.
- [`observations/zhejiang-20180318/`](observations/zhejiang-20180318/) defines
  the official NCEI surface and upper-air observations for the Zhejiang cases,
  including acquisition URLs and checksum pins. Observation bytes remain local
  because non-U.S. ISD data cannot be redistributed.

Derived products such as `LULC.DAT`, `TERR.DAT`, and `geo.dat` do not belong
here. The staged example may write their NetCDF equivalents under `output/`.
The three typed Zhejiang runners pass them through NetCDF files in private
temporary directories and retain only their final CALMET files.

Each source directory includes format, provenance, and SHA-256 information.
