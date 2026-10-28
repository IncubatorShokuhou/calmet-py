# Array backends

[中文](zh/backends.md)

PyCALMET keeps one dispatcher, :mod:`pycalmet.backend`, and calls it from the
vectorized kernels in each stage. There is no second copy of the file readers,
NetCDF writers, or the in-place wind solver.

```python
import pycalmet

pycalmet.set_backend("numpy")          # default
pycalmet.set_backend("jax")             # requires the gpu extra
pycalmet.set_backend("jax", device="cpu")
pycalmet.set_backend("auto")            # JAX if it imports, otherwise NumPy

with pycalmet.use_backend("jax", device="gpu"):
    path = pycalmet.run_pipeline(...)
```

The environment is read the first time the backend is needed, and again after
:func:`pycalmet.reset_backend`:

| Variable | Values | Effect |
| --- | --- | --- |
| `PYCALMET_BACKEND` | `numpy` (default), `jax`, `auto` | Selects the backend when nothing has called `set_backend` yet. |
| `PYCALMET_DEVICE` | `auto` (default), `cpu`, `gpu`, `cuda`, `cuda:0`, `gpu:1` | Used only by the JAX backend. |

`auto` for the device prefers the first GPU JAX can see, and otherwise CPU.
`device="gpu"` or `PYCALMET_DEVICE=gpu` raises `RuntimeError` when JAX has no
GPU, so a batch script does not silently spend the run on the host.

Selection is process-global. :func:`pycalmet.use_backend` restores the previous
backend, and it restores the JAX x64 flag and default device if this package
changed them. Do not switch backends from two threads at once.

## What runs on the selected device

These kernels honor the backend. They run eagerly on the JAX device (they are
not one fused `jax.jit` of a whole stage) and copy the result back to a NumPy
array before the caller writes NetCDF or mutates a host buffer:

| Stage | Kernels |
| --- | --- |
| CALWRF | Stagger removal, geopotential height, grid-to-earth rotation, earth-relative wind, speed and direction, precipitation increments, pressure and temperature |
| TERREL | Cell-mean reduction of elevation sums |
| CTGPROC | Projected coordinate to cell index, and the land-use histogram |
| MAKEGEO | Category sums, water-fraction sums, and the weighted surface fields |
| CALMET | Density, Coriolis, inverse-distance interpolation, wind speed and direction, Teixeira cloud fraction, solar elevation, sensible-heat flux, horizontal divergence and integrated vertical velocity |

Geometry that depends on GDAL, PROJ, Shapely, or SciPy `cKDTree` stays on the
host. So do the sequential NumPy sweeps: divergence minimization, Gauss-Seidel
smoothing, and objective analysis. Those updates are ordered in place; JAX
arrays cannot host that pattern without a separate solver. The legacy
namespace under `pycalmet.legacy` is unchanged and always uses NumPy.

A CALMET hour therefore moves the large diagnostic arrays onto the device for
the kernels above, then returns to NumPy for the iterative wind adjustment.
That is the boundary that keeps the existing Zhejiang NumPy results intact
when JAX is not selected.

## Dtypes and device placement

Physical model fields are `float32`. Time, map coordinates, and the documented
reductions stay `float64`. JAX's own default is to disable 64-bit mode, which
would silently narrow those reductions. Selecting the JAX backend sets
`jax_enable_x64` to `True` for the process and sets `jax_default_device` to
the resolved device. Leaving the JAX backend through `set_backend("numpy")`,
`reset_backend()`, or the end of `use_backend` restores the previous JAX
settings.

Consequences:

- Host NumPy code outside PyCALMET is unaffected, because it never reads the
  JAX flags.
- Other JAX work in the same process sees x64 enabled and the default device
  this package selected, until the backend is restored.
- Transcendental functions (`sin`, `exp`, `power`) can differ from NumPy in
  the last bits. Treat JAX output as numerically equivalent, not bitwise
  identical to the NumPy or Fortran reference.
- Every accelerated kernel returns a NumPy `ndarray`. Downstream code that
  indexes NetCDF variables does not need to know which backend produced the
  array.
- The first JAX call can spend time initializing the GPU runtime. Later calls
  reuse that process.

```python
import numpy as np
import pycalmet
from pycalmet.calmet_numerics import density_from_pressure_temperature

with pycalmet.use_backend("jax", device="cpu"):
    density = density_from_pressure_temperature(
        np.array([1000.0], dtype=np.float32),
        np.array([300.0], dtype=np.float32),
    )

assert isinstance(density, np.ndarray)
assert density.dtype == np.float32
print(pycalmet.backend_info())
```

`backend_info()` while JAX is active reports `model_dtype="float32"`,
`x64_enabled=True`, and the JAX device label. That label contains `cpu` on
the host and `cuda` or `gpu` on an NVIDIA device; the exact spelling follows
the installed JAX release.
