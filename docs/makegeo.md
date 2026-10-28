# MAKEGEO

[中文](zh/makegeo.md)

MAKEGEO joins the TERREL terrain grid and the CTGPROC land-use counts into
the geophysical fields CALMET reads: dominant land use, roughness length,
albedo, Bowen ratio, soil heat-flux parameter, anthropogenic heat, and
leaf-area index.

```python
from pathlib import Path
import pycalmet

geo = pycalmet.makegeo(
    inp=None,
    grid=pycalmet.GridConfig(nx=80, ny=60, dx_km=1.0, dy_km=1.0),
    terrain=Path("output/terrain.nc"),
    landuse=Path("output/landuse.nc"),
    output=Path("output/geo.nc"),
)
```

With no property table, MAKEGEO uses the standard USGS-38 records. Roughness
is a geometric mean of the category values, weighted by land-use fraction.
The other physical fields are arithmetic means. Dominant land use is the
category with the largest fraction, chosen among water categories only when
the water fraction exceeds `water_threshold` (default 0.5).

## Weighted fields and the backend

Fraction sums, water-fraction sums, and both the arithmetic and logarithmic
weighted means run on the active backend in float64 and come back as float32
fields. The dominant-category index uses a host `argmax` because it feeds an
integer map, not a floating-point reduction.

Coverage below `0.001` is filled with `missing_category` (default 55, water).
Coverage between that floor and `minimum_coverage` (default 0.96) is an
error, on either backend. The test that guards this is numerical, not a
device check: a JAX run must raise for the same grid the NumPy run rejects.

Snow-adjusted daily GEO files are a separate code path. The pipeline refuses
them because they write more than one output. Call `makegeo` directly when a
study needs `snow_processing=True`; the snow-age albedo decay itself stays on
NumPy.

The control-file equivalent is `pycalmet.makegeo("makegeo.inp")`. Grid datum,
land-use datum, and terrain datum are not required to match each other here:
CTGPROC and TERREL have already resampled both datasets onto `grid`.
