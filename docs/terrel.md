# TERREL

[中文](zh/terrel.md)

TERREL samples a digital elevation model onto the CALMET grid and writes the
terrain height MAKEGEO will read. Rasterio decodes SRTM HGT and GeoTIFF.
Older CALMET formats (USGS DEM, 3CD, ARM3, and the rest) have readers in the
package because GDAL does not identify them reliably.

```python
from pathlib import Path
import pycalmet

grid = pycalmet.GridConfig(
    nx=80,
    ny=60,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
    x_origin_km=0.0,
    y_origin_km=0.0,
)

terrain = pycalmet.terrel(
    inp=None,
    grid=grid,
    srtm_dir=Path("data/terrain/srtm3"),
    sampling="bilinear",
    output=Path("output/terrain.nc"),
)
```

The control-file form is `pycalmet.terrel("terrel.inp")`. Source datum and
target datum are separate: the grid's `datum` is where the output lives, and
each terrain file names the datum of its own coordinates.

## Sampling and the backend

Terrain samples are accumulated in float64 sums and integer hit counts so a
1 km cell that covers many SRTM posts does not lose the mean to float32
rounding. The finish step divides those sums on the active backend and writes
float32 heights. Empty cells stay at the TERREL missing value, `-999`.

Nearest-neighbour searches and coastline clipping stay on SciPy and Shapely.
Polar gap filling uses a `cKDTree` on the host. Selecting JAX therefore speeds
the cell-mean reduction, which is the wide array step, and leaves the spatial
index on CPU.

`sampling` chooses how posts land in a cell. The Zhejiang case uses
`bilinear` on SRTM3. Peak-preserving output is a separate switch on the
TERREL configuration; it replaces the mean with the maximum sample and does
not change the backend contract.

Output defaults follow the filename: `.nc` is NetCDF. Legacy SURFER and
TERREL save files are written only when their paths are set.
