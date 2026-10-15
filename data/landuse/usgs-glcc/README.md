# USGS global land-use raster for Asia

`EAUSGS1.IMG` is the original categorical raster consumed by the Zhejiang
CTGPROC stage. It is a headerless, row-major `uint8` image with 13,000 columns
and 12,000 rows (156,000,000 bytes). CTGPROC reads it as `GLAZAS`: the
Asia-optimized Eurasia grid in a 1 km Lambert azimuthal equal-area projection,
using the `ESR-S` spherical datum, latitude of origin 45° N, and central
longitude 100° E.

The legacy Zhejiang workflow called this file `EAUSGS1.IMG`. The byte content
here is `eausgs2_0la.img`, extracted from the fixed
[`eausgs2_0la.7z`](https://raw.githubusercontent.com/ydopku/parallel-CALPUFF/aa04af6399cc5080a7776d92dec7316acb3488c6/CalpuffWorkerG/eausgs2_0la.7z)
revision in the public third-party `ydopku/parallel-CALPUFF` mirror. The
archive SHA-256, calculated by this project, is
`2ae72d2824bb0c5ba9ad9690563fc7d556d8b29446713861a5988e1dda6e2e16`.
No upstream USGS checksum or retained checksum for the formerly ignored local
file was available. As a cross-check, the historical CTGPROC implementation
produces 304,390 valid Zhejiang cells and 43,548,891 samples from this raster,
exactly matching the two statistics recorded for the old local
`EAUSGS1.IMG`; this strongly validates the data but cannot prove byte identity.

`EAUSGS2.IMG` is not included because its historical content could not be
established and the Zhejiang control uses `NDBF=1` to read only this Asia
raster. The mirror's separate Europe raster is `GLAZEU`, not an additional
input for this `GLAZAS` configuration. Adding it as a second source would
change the aggregation and can double-count overlapping coverage.

The `.IMG` file is stored with Git LFS. After cloning, run `git lfs pull` if
the checkout contains an LFS pointer instead of the 156 MB raster. Verify the
materialized file from this directory with:

```bash
sha256sum -c SHA256SUMS
```
