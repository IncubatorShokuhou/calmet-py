# CALMET

CALMET diagnoses a mass-consistent wind field and the mixing heights,
temperatures, and turbulence scales that a dispersion model reads. The public
entry point is `pycalmet.calmet`. It accepts a control file or keywords.

A prognostic-only hour, which is the shape of the Zhejiang WRF run, looks
like this:

```python
from datetime import datetime
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
)

result = pycalmet.calmet(
    inp=None,
    start=datetime(2018, 3, 18, 0),
    end=datetime(2018, 3, 18, 1),
    grid=grid,
    geophysical=Path("output/geo.nc"),
    meteorology=[Path("output/calwrf.nc")],
    noobs=2,
    iprog=14,
    irhprog=1,
    itprog=2,
    output=Path("output/calmet.nc"),
)
```

`noobs=2` with `iprog=14` tells CALMET to take its initial wind and scalar
fields from the CALWRF dataset and not from surface or upper-air stations.
`end` is the exclusive end of the modeled interval when you use the keyword
API in this form; the README parameter table is the authority for each
control's inclusive or exclusive convention. Observation-based runs add
surface, precipitation, and sounding paths instead of `noobs=2`. The Zhejiang
guide (`examples/zhejiang/README.md`) shows both.

## Which numerics use JAX

The diagnostic building blocks in `pycalmet.calmet_numerics` are plain
functions. The ones that are array-wide and free of in-place sweeps follow
the [backend](backends.md):

- dry-air density and the Coriolis parameter
- inverse-distance interpolation of station values
- vector wind to speed and meteorological direction
- Teixeira cloud fraction from relative humidity
- the half-hour solar-elevation sine used by the heat-flux scheme
- sensible heat and short-wave radiation
- horizontal divergence and the vertical velocity integrated upward from it,
  including the optional O'Brien correction

`minimize_divergence` and `smooth_wind` walk each cell in a fixed order and
update the wind they just wrote. They stay on NumPy under every backend.
Objective analysis and the barrier routine do the same. A JAX selection still
runs those sweeps; it only moves the surrounding array kernels.

Model state inside the solver is float32. The divergence integration
accumulates in float64, then the kernel casts the stored field back to
float32. That split is the same one the NumPy path uses, and it is why the
JAX backend turns 64-bit mode on. Leaving it off would change which cells the
convergence test accepts.

The legacy API (`pycalmet.legacy.calmet`) reproduces the historical
float32-at-every-statement path. It does not consult the backend. Use it
when the comparison target is the Fortran binary, and use `pycalmet.calmet`
when the comparison target is this package's NumPy solver.

## Reading the result

The NetCDF written by `calmet` is the product `run_pipeline` keeps. Variable
names follow the CALMET output convention already documented in the README
(wind components, mixing heights, temperature, and the two-dimensional
surface fields). Legacy `CALMET.DAT` plus its `.aux` companion are written
only when the output path or the compatibility switch asks for them.

A one-hour WRF window is a regression case, not a boundary-layer spin-up.
Production runs need the previous night of meteorology so the mixing height
has somewhere to start. The Zhejiang README states that limit next to the
reproduction command.
