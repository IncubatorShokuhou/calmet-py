# Zhejiang SRTM3 subset

This directory contains only the 42 SRTM3 tiles referenced by
`examples/zhejiang/config/terrel.inp`: latitudes N26 through N31 and longitudes E117 through
E123. Each file is a 1201 × 1201 signed 16-bit elevation grid in the standard
SRTM `.hgt` layout.

Run `cd data/terrain/srtm3 && sha256sum -c SHA256SUMS` to verify the files. The
full local SRTM archive is intentionally excluded from Git.
