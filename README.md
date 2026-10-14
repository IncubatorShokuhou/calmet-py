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
- A separate opt-in legacy API for reproducing historical Fortran-oriented runs.

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

Paths inside a control file are resolved relative to that file. The bundled
example is fully defined by these files, so no duplicate Python arguments are
needed. Keyword overrides remain available for applications that deliberately
reuse a control file with different data. Stage functions return the path of
their primary output.

```python
import pycalmet

base = "examples/zhejiang/config"
pycalmet.calwrf(f"{base}/calwrf.inp")
pycalmet.terrel(f"{base}/terrel.inp")
pycalmet.ctrproc(f"{base}/ctgproc.inp")
pycalmet.makegeo(f"{base}/makegeo.inp")
result = pycalmet.calmet(f"{base}/calmet.inp")
```

Every stage also has a complete typed keyword API. Leave `inp` at its default
`None` and pass the control values directly; grid controls such as `nx`,
`dx_km`, and `projection` are ordinary parameters. Dates accept `datetime`,
paths accept `str` or `Path`, and repeated controls accept Python sequences.
Parameters omitted by the caller use the defaults defined by the original
program. For example:

```python
from datetime import datetime
from pathlib import Path

result = pycalmet.calmet(
    inp=None,
    meteorology=Path("output/calwrf.nc"),
    geophysical=Path("output/geo.nc"),
    output=Path("output/calmet.nc"),
    start=datetime(2018, 3, 18, 0),
    end=datetime(2018, 3, 18, 1),
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    lat0=27.0,
    lon0=117.5,
)
```

The signatures list every supported control, including controls with Fortran
defaults that may be absent from a particular `.inp` file. They are visible to
IDE completion, type checkers, `help()`, and `inspect.signature()`.

The public processing API is:

```python
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc is a compatibility alias
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

The default API uses regular NumPy calculations, `pyproj`, `datetime`, and a
direct NetCDF flow from CALWRF to CALMET. Workflows that specifically require
the former single-precision operation order or M3D staging can opt in to the
frozen compatibility implementation:

```python
pycalmet.legacy.calwrf(...)
pycalmet.legacy.calmet(...)
```

## Zhejiang example

The bundled example contains three consecutive WRF records, 42 original SRTM3
`.hgt` tiles, and a categorical land-use point dataset. It rebuilds terrain,
land use, and the geophysical grid instead of using a prepared `geo.nc`.
Run all five stages from the repository root:

```bash
sha256sum -c examples/zhejiang/SHA256SUMS
cd terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../..

./examples/zhejiang/run.sh
```

See [examples/zhejiang/README.md](examples/zhejiang/README.md) for provenance,
input boundaries, source data, and generated products.

## Repository layout

```text
src/pycalmet/          Python package and opt-in legacy namespace
tests/                 Unit and integration tests
examples/zhejiang/     Runnable real-data example and control files
terrain/srtm3/         SRTM3 tiles used by the Zhejiang domain
fortran/               Reference Fortran sources; not used at runtime
tools/                 Maintainer utilities
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

The default suite checks numerical behavior and complete data flow without
requiring a particular compiler or C math library. A focused legacy test keeps
the compatibility namespace separate and verifies its historical
single-precision path. Fortran sources under `fortran/` remain available for
audit and are not an installation dependency.

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
