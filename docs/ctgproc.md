# CTGPROC

[中文](zh/ctgproc.md)

CTGPROC resamples a categorical land-use raster onto the CALMET grid and
stores hit counts per category. MAKEGEO turns those counts into fractions.
The Zhejiang land-use source is `EAUSGS1.IMG`, a headerless 8-bit raster on
the Asia Lambert azimuthal grid. `input_type="GLAZAS"` selects that geometry
and `input_datum="ESR-S"` names its datum. Neither is implied by the target
grid's WGS-84 datum.

```python
from pathlib import Path
import pycalmet

grid = pycalmet.GridConfig(nx=80, ny=60, dx_km=1.0, dy_km=1.0, projection="LCC")
landuse = pycalmet.ctgproc(
    inp=None,
    grid=grid,
    landuse_files=[Path("data/landuse/usgs-glcc/EAUSGS1.IMG")],
    input_type="GLAZAS",
    input_datum="ESR-S",
    mesh_density=12,
    output=Path("output/landuse.nc"),
)
```

`mesh_density=12` is the Python spelling of `MESHGLAZ=12`: each source cell
is split into a 12 by 12 mesh before the samples are binned. `ctgproc` and
`ctrproc` are the same stage; `ctrproc` is the historical entry-point name.

## Binning and the backend

For every accepted sample the kernel:

1. Projects the sample into the target grid's kilometres.
2. Converts kilometres to a zero-based cell index with `floor((x - origin) / dx)`.
3. Histograms the samples by category.

Steps 2 and 3 use the active backend. The histogram is added into a host
`int64` count array because later passes, coastline edits, and the NetCDF
writer all mutate that buffer in place. Coastline polygons, GeoTIFF windows,
and datum shifts stay on Shapely, Rasterio, and PROJ.

Indices are not clipped. Samples outside the grid are the caller's
responsibility to drop before binning; the Zhejiang reader does that with the
mesh window. A category that is not in the output map raises, on either
backend, when the histogram tries to address it.

Snow bookkeeping (`snow` grids for a later MAKEGEO pass) is a sparse host
dictionary. It is unchanged by the JAX backend. The pipeline turns snow
processing off so it can promise one output file.
