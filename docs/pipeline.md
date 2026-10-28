# End-to-end pipeline

[中文](zh/pipeline.md)

`pycalmet.run_pipeline` runs CALWRF, TERREL, CTGPROC, MAKEGEO, and CALMET as
one call. Intermediate NetCDF files live in a temporary directory that the
pipeline deletes. The path you pass as `output` is the CALMET dataset that
remains.

The stages share one horizontal grid. Build it once and pass it to every
stage that asks for geometry:

```python
from datetime import datetime
from pathlib import Path

import pycalmet

grid = pycalmet.GridConfig(
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
)

example = Path("examples/zhejiang")
result = pycalmet.run_pipeline(
    inp=None,
    calwrf_wrf_files=[
        example / "wrfout_d01_2018-03-18_00_00_00.nc",
        example / "wrfout_d01_2018-03-18_01_00_00.nc",
    ],
    calwrf_start=datetime(2018, 3, 18, 0),
    calwrf_end=datetime(2018, 3, 18, 1),
    terrel_srtm_dir=Path("data/terrain/srtm3"),
    terrel_grid=grid,
    terrel_sampling="bilinear",
    ctgproc_landuse_files=[Path("data/landuse/usgs-glcc/EAUSGS1.IMG")],
    ctgproc_grid=grid,
    ctgproc_input_type="GLAZAS",
    ctgproc_input_datum="ESR-S",
    ctgproc_mesh_density=12,
    makegeo_grid=grid,
    calmet_start=datetime(2018, 3, 18, 0),
    calmet_end=datetime(2018, 3, 18, 1),
    calmet_grid=grid,
    calmet_noobs=2,
    calmet_iprog=14,
    output=Path("output/zhejiang/calmet.nc"),
)
```

Keyword names are the stage field with a prefix (`calwrf_`, `terrel_`,
`ctgproc_`, `makegeo_`, `calmet_`). A grid may instead be split into prefixed
scalars such as `terrel_nx` and `terrel_dx_km`. The full table is the
parameter reference in the README.

Three pipeline-only constraints are worth knowing before the first run:

- `makegeo_snow_processing` stays false. Snow processing writes one GEO file
  per day, which breaks the single-output contract.
- `calmet_save_output` stays true and `calmet_test_mode` stays `2`, so the
  final hour is both computed and written.
- CALWRF's `start` and `end` are inclusive WRF records. They bracket the
  CALMET interval, which is half-open on the end. The example above feeds
  00:00 and 01:00 so CALMET can fill `[00:00, 01:00)`.

An existing directory of the five control files is the other entry point.
Paths inside each file are resolved next to that file:

```python
from pathlib import Path
import pycalmet

result = pycalmet.run_pipeline(
    Path("examples/zhejiang/config"),
    output=Path("output/zhejiang/calmet.nc"),
)
```

`PipelineConfig` groups five typed stage objects when a caller already built
them. `run_pipeline` accepts that object as its first argument.

## Where the backend applies

Call :func:`pycalmet.set_backend` or enter :func:`pycalmet.use_backend` before
`run_pipeline`. The pipeline does not pick a backend of its own. With the
default NumPy backend the Zhejiang runners behave as they did before JAX
existed. With JAX selected, the kernels listed in
[Array backends](backends.md) run on the device; file I/O and the iterative
wind solver stay on the host.

The runnable example is `examples/zhejiang/run.py`, started with
`./examples/zhejiang/run.sh`. `examples/zhejiang/run_stages.py` runs the five
control files one at a time when you want to inspect an intermediate file.
