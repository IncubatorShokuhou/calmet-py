# Zhejiang observations for 18 March 2018

[中文文档](README.zh-CN.md) · [Data overview](../../README.md)

This directory defines the real surface and upper-air observations used by the
Zhejiang observation-only and WRF-plus-observation examples. Observation bytes
are acquired directly from NOAA NCEI and remain local; the repository contains
only this documentation and the machine-readable
[`acquisition.json`](acquisition.json) provenance and checksum pins. No value in
this workflow is synthesized from WRF output.

## Redistribution and local acquisition

NCEI's [ISD access-point readme](https://www.ncei.noaa.gov/pub/data/noaa/readme.txt)
states: “The non-U.S. data in ISD are subject to WMO Resolution 40 restrictions,
and cannot be redistributed to other users or customers.” For that reason,
neither the Chinese ISD records nor any CALMET files derived from them are
committed. IGRA describes its `access/` files as intended for public access but
does not state a redistribution grant in its readme, so the radiosonde records
follow the same direct-acquisition workflow.

Fetch the sources from NCEI and verify the pinned historical content:

```bash
python tools/fetch_zhejiang_observations.py
python tools/fetch_zhejiang_observations.py --check
```

The first command is the complete preparation workflow: it downloads into the
Git-ignored `raw/` directory, verifies every source, and generates `SURF.DAT`,
four `UP_*.DAT` files, and their manifest under the Git-ignored `calmet/`
directory. The second command performs an offline SHA-256 check of both layers.
A checksum failure is intentional: review an upstream NCEI revision before
updating [`acquisition.json`](acquisition.json).

For troubleshooting, `--fetch-only` stops after acquiring `raw/`, and
`--prepare-only` rebuilds `calmet/` from an already verified local acquisition.
The fetcher passes all ten ISD sources, four IGRA slices, station names,
documented elevation overrides, and the pinned elevation catalog to
`tools/build_zhejiang_observations.py`; no manual parameter list is required.

## Period and layout

The CALMET example runs from `2018-03-18 00:00 UTC` inclusive to `01:00 UTC`
exclusive. Surface preprocessing selects observations for that hourly period
from the 2018 annual ISD archives. Upper-air interpolation is bracketed by the
00 and 12 UTC profiles on 18 March.

```text
zhejiang-20180318/
├── acquisition.json             # tracked official URLs and checksum pins
├── README.md
├── README.zh-CN.md
├── raw/                         # local only; created by the fetcher
│   ├── manifest.json            # actual URL, HTTP metadata, size and SHA-256
│   ├── surface/                 # ten complete 2018 ISD .gz archives
│   └── upper_air/
│       ├── archives/            # four complete IGRA period-of-record ZIPs
│       ├── *-20180318-00-12Z.txt# exact byte slices used by preprocessing
│       ├── igra2-station-list.txt
│       └── igra2-station-list-frozen-f9b7fd27.txt # elevation provenance
└── calmet/                      # local only; created by preprocessing
    ├── SURF.DAT
    ├── UP_CHM00058362.DAT
    ├── UP_CHM00058457.DAT
    ├── UP_CHM00058633.DAT
    ├── UP_CHM00058665.DAT
    └── manifest.json
```

`raw/` contains source bytes. `calmet/` contains deterministic CALMET 2.1 text
files derived by `tools/build_zhejiang_observations.py`; they are model inputs,
not additional observations. Both directories are explicitly ignored by Git.

## Surface observations

The canonical source is the annual gzip-compressed, fixed-width
[Integrated Surface Database](https://www.ncei.noaa.gov/pub/data/noaa/2018/).
The fixed-width files retain the per-report site metadata. This matters for
station 584570: the archive contains Hangzhou `FM-12` reports at
30.233° N, 120.167° E, 43 m and Xiaoshan `FM-15` reports at
30.228° N, 120.432° E, 7 m. The preprocessing selection gives the synoptic
`FM-12` report priority when time distance and data completeness are equal; an
NCEI date-service CSV would normalize both rows to Xiaoshan metadata and lose
that distinction.

| USAF-WBAN | Station label | Archive bytes | SHA-256 |
| --- | --- | ---: | --- |
| `582390-99999` | Lishe | 213,626 | `780dd554b1204df3b0dfebd59c6c16bc0406c2c0d6c73906ebca5aa9c12784fd` |
| `583670-99999` | Hongqiao International | 363,795 | `721df51fe64c12bdaede856fab342c003a8912c0e3dc8ea0f17a4b8254f974e8` |
| `584570-99999` | Hangzhou / Xiaoshan | 544,427 | `5973f915b6644bca1f1e0ac67288d826913a33a23dece51096c9243544cbdeea` |
| `584770-99999` | Dinghai | 119,031 | `f1735adb425034f82156f24940b971eadb0da7b20514c2a09335165bce6e7fe3` |
| `585560-99999` | Shengxian | 103,612 | `f818b19dcbfdb972f859d2078b7e6ce2e981452aa63d4f9aa0a44a7fb1a48195` |
| `585690-99999` | Shipu | 106,193 | `e027723db4dc0431ce21b1929c1019c4eae96ab5f2cc61d2f55084c154ba3fc2` |
| `586330-99999` | Qu Xian | 74,493 | `57048dac01aaea971e4c7b96ea2aca3f01ad1fdc6243d2cf542530c9f7debf74` |
| `586460-99999` | Lishui | 99,265 | `4d58c6cbf803841f6f686644bf8ce142d2d44cdd1daafa2513dbd275e57866bd` |
| `586660-99999` | Dachen Dao | 120,802 | `749d200ac7658cd8cc5fdd84dbf9b8c92c020415dc30af09c500d995611383b7` |
| `587520-99999` | Ruian | 117,648 | `5ac9db168e547fa98dd149eb41441c354b904aca79cd4910c56f4b60a0958248` |

Each URL is
`https://www.ncei.noaa.gov/pub/data/noaa/2018/{USAF}-99999-2018.gz`.
The fetcher verifies the complete compressed archive before inspecting its ISD
records, including station identity and the presence of 18 March data.

## Upper-air observations

The source is NCEI's
[Integrated Global Radiosonde Archive v2.2](https://www.ncei.noaa.gov/pub/data/igra/).
The fetcher downloads each period-of-record ZIP, then copies the complete fixed
width record blocks for `2018-03-18 00 UTC` and `12 UTC` byte for byte. Full
archives can change as NCEI appends recent profiles, so their digests are
reference identities. The extracted historical slice digest is the enforced
reproducibility check.

The raw slices retain every valid IGRA level. When writing CALMET `UP.DAT`,
the converter emits only levels below 9,998 m MSL because the official CALMET
6.5 `RDUP` reader reserves every height at or above 9,998 m as missing. The
resulting profiles still reach 9.45–9.53 km MSL, well above this example's
3 km model top. The conversion manifest records the number of omitted levels
and the emitted wind and temperature top for every sounding.

| IGRA ID | Station | Role | Reference ZIP SHA-256 | Selected-slice SHA-256 |
| --- | --- | --- | --- | --- |
| `CHM00058362` | Shanghai (Baoshan) | About 1.6 km north of the grid; used to constrain the north edge | `33a650675ce719d0a8b4d6b6ebf68fe1d266ed3c4a0d2bc64fee432cd7bffdef` | `554fff21adabac78cdf380d89b7e9c84ea0d8bbb5d18e88e651fc5476447eb12` |
| `CHM00058457` | Hangzhou | In domain | `2c719551602eb0dcb0cf291bdf220e3682199c03133f51fab39f71d10c9c11b2` | `06bfe5a02a691370c8f8a5f28efd0b2d36cdf7dc20762410ae2babce5d35e5ad` |
| `CHM00058633` | Qu Xian | In domain | `a0e4abf998cac2a48ad75c9958b940891068c0800be344185a37e8e13550d955` | `f77a15a5b96fb73984c442f60c558135813471dae49e681acff846c70042b761` |
| `CHM00058665` | Hongjia | In domain | `5a571097b9e6e3384d8754479a2fd559ae23b9dbc10d0f4f1368cf3e6ddcc83b` | `6400722e91d6b074f0c745214ee6ef97bb30d021438174711d7f15e185c9255b` |

Each period archive URL is
`https://www.ncei.noaa.gov/pub/data/igra/data/data-por/{IGRA_ID}-data.txt.zip`.
The live IGRA station list is acquired with the observations. The preprocessing
provenance records explicit station elevations of 9.0, 43.0, 82.9 and 4.6 m
MSL for Baoshan, Hangzhou, Qu Xian and Hongjia respectively; the first three
are currently missing in the live NCEI station list. These are metadata
overrides only and do not alter measured profile values.

The elevations are audited against an older NCEI station catalog preserved by
the `Matthieu-Lacroix/igra2_interactivemap` mirror at commit
[`f9b7fd27adc6c9cee0ba1aa653cf4c99caf91ce6`](https://github.com/Matthieu-Lacroix/igra2_interactivemap/blob/f9b7fd27adc6c9cee0ba1aa653cf4c99caf91ce6/igra2-station-list.txt).
That catalog is 260,147 bytes with SHA-256
`d1deb3396f6b8b5d8d49c8338d2ac62a842fed1c7d06fd7723890a6f5ce669c5`.
The fetcher downloads and verifies this exact catalog locally, and the
preprocessing manifest cites it as the elevation source. The wrapper also
checks each configured coordinate within `0.00005°` and each elevation within
`0.05 m` of the fixed-column catalog value before it can build an `UP_*.DAT`.

## Provenance boundary

The tracked checksums identify the source revisions reviewed for this case.
`raw/manifest.json` records the bytes actually retrieved, including redirect
targets, ETags and `Last-Modified` values when NCEI supplies them.
`calmet/manifest.json` records all preprocessing choices and output checksums.
Keep both manifests with local scientific results, but do not redistribute the
observation-bearing directories.
