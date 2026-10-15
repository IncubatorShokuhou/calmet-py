# USGS GLCC land-use rasters

[中文文档](README.zh-CN.md) · [Data overview](../../README.md)

This directory archives two headerless, one-byte-per-cell USGS Global Land
Cover Characterization 2.0 rasters. CTGPROC assigns their fixed 1 km Lambert
azimuthal equal-area geometries from the reader name rather than a file header.

| Local file | Public source image | CTGPROC reader and fixed geometry | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| `EAUSGS1.IMG` | `eausgs2_0la.img` | `GLAZAS`; 13,000 × 12,000; 45° N, 100° E | 156,000,000 bytes | `4f2a1977d141e2040da4d36ec6a4826c5c960c84686eaf10f226f388cb3ab90a` |
| `EAUSGS2.IMG` | `eausgs2_0le.img` | `GLAZEU`; 13,000 × 13,000; 55° N, 20° E | 169,000,000 bytes | `0fafa581ece7b23b595aa895832e94034c0e93e4efaf6aef60fbcc6be57860da` |

Both use the `ESR-S` spherical datum. The images were extracted from fixed
revisions of [`eausgs2_0la.7z`](https://raw.githubusercontent.com/ydopku/parallel-CALPUFF/aa04af6399cc5080a7776d92dec7316acb3488c6/CalpuffWorkerG/eausgs2_0la.7z)
and [`eausgs2_0le.7z`](https://raw.githubusercontent.com/ydopku/parallel-CALPUFF/aa04af6399cc5080a7776d92dec7316acb3488c6/CalpuffWorkerG/eausgs2_0le.7z)
in the public third-party `ydopku/parallel-CALPUFF` mirror. Their archive
SHA-256 values, calculated by this project, are respectively
`2ae72d2824bb0c5ba9ad9690563fc7d556d8b29446713861a5988e1dda6e2e16` and
`6a56fb920f3c77ca8a1054f7e8fa287859a9b47e76b5a7caefc831e1dc57333a`.

`EAUSGS1.IMG` and `EAUSGS2.IMG` are historical local names, not standard
CTGPROC identifiers. The old files were ignored by Git and no checksums were
retained. The Asia mapping is strongly validated because the historical
CTGPROC implementation produces 304,390 valid Zhejiang cells and 43,548,891
samples from this image, exactly matching the two recorded legacy statistics.
The Europe image is the corresponding file distributed beside the Asia image
and is the best available reconstruction of the requested `EAUSGS2.IMG`.
Without retained historical hashes, byte identity with either old local file
cannot be proved.

The Zhejiang control remains `NDBF=1` and reads only `EAUSGS1.IMG` as
`GLAZAS`. `EAUSGS2.IMG` is retained as the `GLAZEU` companion and does not
participate in the Zhejiang calculation.

Both `.IMG` files use Git LFS. After cloning, run `git lfs pull` if either
checkout contains an LFS pointer. Verify both materialized files from this
directory with:

```bash
sha256sum -c SHA256SUMS
```
