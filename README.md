# PyCALMET

[中文文档](README.zh-CN.md)

PyCALMET is a NumPy implementation of the CALMET meteorological preprocessing
and diagnostic wind-field workflow. It provides Python APIs for CALWRF,
TERREL, CTGPROC, MAKEGEO, and CALMET, with NetCDF as the preferred interchange
format and compatibility readers and writers for legacy CALPUFF files.

The package is intended for reproducible research, automated data processing,
and modernization of existing CALMET workflows. A complete Zhejiang example,
the required SRTM3 terrain tiles, and the original Fortran sources used as
numerical references are kept in separate, clearly scoped directories.

## Highlights

- Pure Python and NumPy/SciPy execution; no Fortran compiler is required.
- Five public processing stages exposed through one package.
- WRF NetCDF input and CF-compatible NetCDF output.
- Compatibility with legacy CALWRF, TERREL, CTGPROC, MAKEGEO, and CALMET files.
- Streaming WRF processing for datasets larger than memory.
- A real-data Zhejiang example with losslessly compressed WRF inputs.
- Numerical regression tests derived from the retained Fortran reference code.

## Installation

PyCALMET requires Python 3.10 or newer.

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Quick start

Each stage accepts either explicit keyword arguments or its legacy
CALMET-family control file. These `.inp` files are program-specific control
files rather than Fortran `NAMELIST` files. The Zhejiang example defines all
five:

| Stage | Example control file |
| --- | --- |
| CALWRF | `examples/zhejiang/config/calwrf.inp` |
| TERREL | `examples/zhejiang/config/terrel.inp` |
| CTGPROC | `examples/zhejiang/config/ctgproc.inp` |
| MAKEGEO | `examples/zhejiang/config/makegeo.inp` |
| CALMET | `examples/zhejiang/config/calmet.inp` |

Paths inside a control file are resolved relative to that file. Keyword
arguments supplied with a control-file path override its values, which is
useful for selecting user inputs and output locations. Stage functions return
the path of their primary output.

```python
from pathlib import Path

import pycalmet

meteorology = pycalmet.calwrf(
    "examples/zhejiang/config/calwrf.inp",
    wrf_files=sorted(Path("examples/zhejiang").glob("wrfout*.nc")),
    start="2018031800",
    end="2018031802",  # following-hour boundary for CALMET averaging
    output="output/calwrf.nc",
    legacy_output="output/calwrf.m3d",
    create_surface=False,
)

result = pycalmet.calmet(
    "examples/zhejiang/config/calmet.inp",
    meteorology="output/calwrf.m3d",
    geophysical="examples/zhejiang/geo.nc",
    output="output/calmet.nc",
    start="2018031800",
    end="2018031801",
)
```

`TERREL`, `CTGPROC`, and `MAKEGEO` controls are used when rebuilding the
static geophysical grid. That route also requires an external land-use raster;
the runnable command below reuses the bundled `geo.nc` instead.

The public processing API is:

```python
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc is a compatibility alias
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

## Zhejiang example

The bundled example contains three consecutive WRF records and a prepared
geophysical grid. Run the full Python workflow from the repository root:

```bash
sha256sum -c examples/zhejiang/SHA256SUMS

./examples/zhejiang/run.sh \
  --wrf examples/zhejiang \
  --geo examples/zhejiang/geo.nc \
  --start 2018031800 \
  --end 2018031801 \
  --skip-static \
  --output output/zhejiang.nc
```

See [examples/zhejiang/README.md](examples/zhejiang/README.md) for provenance,
input boundaries, and instructions for rebuilding the static fields.

## Repository layout

```text
src/pycalmet/          Python package
tests/                 Unit, integration, and numerical regression tests
examples/zhejiang/     Runnable real-data example and control files
terrain/srtm3/         SRTM3 tiles used by the Zhejiang domain
fortran/               Reference Fortran sources; not used at runtime
tools/                 Python diagnostics and regression utilities
```

The `fortran/` tree contains source material only. Legacy NCL programs,
regional C-shell runners, build wrappers, generated executables, and runtime
products are intentionally excluded from the maintained library.

## Data and output conventions

Canonical outputs are NetCDF files readable by `xarray.open_dataset`. Legacy
formatted and unformatted CALMET-family files remain available when an output
path or compatibility option explicitly requests them. Large inputs should be
kept outside the source tree; generated files belong under `output/`, which is
ignored by Git.

The Zhejiang example uses an LCC grid with a 1 km spacing and a `610 × 499`
domain. Its WRF records cover 2018-03-18 00:00–02:00 UTC. The final record is
the following-hour boundary required to produce the 00:00–01:00 CALMET period.

## Validation

Run the complete Python suite with:

```bash
python -m pytest -q
```

Most regression expectations are platform-independent. A small group of tests
checks exact GNU `REAL*4` transcendental-function bit patterns; different glibc
versions can differ by one or two ULP while producing equivalent physical
results.

The scripts in `tools/` compare canonical or legacy outputs field by field.
Fortran sources under `fortran/` are retained for audit and numerical reference
and are not an installation dependency.

## Contributing

Keep runtime code in `src/pycalmet`, add focused tests for behavioral changes,
and avoid committing generated model products. Preserve fixed-form layout when
editing reference Fortran. Pull requests should identify the affected stage,
the input configuration, and the exact validation command used.

## Acknowledgements

PyCALMET follows the processing model and file conventions of the CALPUFF
system. The included WRF subset and terrain data are provided to make numerical
validation reproducible; their provenance and checksums are documented beside
the data.
