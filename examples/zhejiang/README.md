# Zhejiang end-to-end pipeline example

This example runs all five PyCALMET stages from source datasets and traditional
control files through one `run_pipeline` call. It does not use a prepared
geophysical grid.

## Bundled inputs

- Three WRF records for 2018-03-18 00:00, 01:00, and 02:00 UTC. The configured
  00:00–01:00 CALMET period uses the first two; the final record supports
  extending the example through the next interval.
- 42 original SRTM3 tiles under `terrain/srtm3/`, covering N26–N31 and
  E117–E123. Each `.hgt` file is a 1201 × 1201 signed 16-bit elevation grid.
- `landuse/zhejiang_generic.landuse`, a WGS-84 categorical point dataset with
  one source record at each target-grid cell center. Its categories preserve
  the validated Zhejiang example's dominant land use using the corresponding
  USGS classes.
- One control file per stage under `config/`. Their relative input and output
  paths form a complete processing chain.

Verify the source files from the repository root:

```bash
sha256sum -c examples/zhejiang/SHA256SUMS
cd terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../..
```

## Run

Install the package and execute the example from the repository root:

```bash
python -m pip install -e .
./examples/zhejiang/run.sh
```

The default runner passes the directory containing all five control files to
the complete pipeline API:

```python
from pathlib import Path

import pycalmet

result = pycalmet.run_pipeline(
    Path("examples/zhejiang/config"),
    output=Path("output/zhejiang/calmet.nc"),
)
```

`run_pipeline` creates the CALWRF, terrain, land-use, and geophysical NetCDF
datasets in a private temporary directory and removes them after CALMET
finishes. The only persistent product is:

```text
calmet.nc       final one-hour CALMET result
```

The pipeline does not create traditional M3D or M2D files. CALWRF passes its
canonical NetCDF dataset directly to CALMET. The WRF subsets retain every
variable read by CALWRF and use lossless NetCDF4 compression. PyCALMET reads
and writes these files through `netCDF4`; it does not require xarray.

## Step-by-step debugging

To inspect the output from every stage, run the separate debugging example:

```bash
PYTHONPATH=src python examples/zhejiang/run_stages.py
```

This script does not read any `.inp` file. It supplies typed Python keyword
parameters to the five public stage APIs and lets omitted controls use their
documented Python defaults. The parameters that affect the Zhejiang run are
written explicitly, including the source datum for every SRTM tile and the
prognostic meteorology, humidity, temperature, precipitation, and mixing-height
settings. It retains `calwrf.nc`, `terrain.nc`, `landuse.nc`, `geo.nc`, and
`calmet.nc` under `output/zhejiang/`.
