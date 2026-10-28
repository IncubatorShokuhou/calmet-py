# Installation

[中文](zh/install.md)

PyCALMET requires Python 3.10 or newer. The published package name is
`pycalmet`.

## Default install

The default install is NumPy, SciPy, and the geospatial libraries the readers
already use. It does not install JAX:

```bash
python -m pip install pycalmet
```

From a source checkout the same choice is:

```bash
python -m pip install .
```

Importing the package and running the Zhejiang example both stay on NumPy
until a caller selects another backend. See [Array backends](backends.md).

## Optional JAX / GPU extra

```bash
python -m pip install "pycalmet[gpu]"
```

From a source checkout:

```bash
python -m pip install ".[gpu]"
```

What that extra resolves depends on the platform:

| Platform | Wheels pulled in by `[gpu]` |
| --- | --- |
| Linux x86_64 and aarch64 | `jax[cuda12]`. This is JAX plus the CUDA 12 plugin and the NVIDIA pip wheels. The driver must be 525 or newer. |
| macOS, Windows, and other Linux architectures | The portable `jax` wheel. JAX then uses its CPU backend unless you add a platform plugin yourself. |

`[gpu]` does not switch the runtime. After installation, `pycalmet.get_backend()`
is still `"numpy"` until you select JAX.

CUDA 13 wheels (`jax[cuda13]`) need a newer driver. Install those yourself
when the machine provides CUDA 13 and you do not want the CUDA 12 plugin.
A machine that already has a CUDA toolkit and cuDNN can use `jax[cuda12-local]`
instead of the bundled NVIDIA wheels. Both of those are upstream JAX choices;
the extra itself stays on the CUDA 12 pip stack because that matches the
drivers commonly installed on clusters.

Pip will not install a JAX release that the running interpreter cannot use.
Python 3.10 resolves an older JAX; newer Python versions resolve a current
JAX. The import surface used by PyCALMET (`jax`, `jax.numpy`, and the x64
and default-device configuration) is stable across that range.

## Check the selection

```python
import pycalmet

print(pycalmet.available_backends())
print(pycalmet.backend_info())
```

`available_backends()` is `("numpy",)` until JAX imports, and
`("numpy", "jax")` after the extra is installed. `backend_info()` reports
the active backend, whether JAX imported, and, once JAX is selected, the
device label.

## Development install

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The default test run does not install or import JAX. Tests that compare JAX
with NumPy skip when `jax` is absent, so a NumPy-only environment stays green.

To exercise the JAX tests on a machine without a GPU, install the portable
CPU build rather than the CUDA extra:

```bash
python -m pip install "jax>=0.4.31"
python -m pytest -q tests/test_backend.py
```

CPU JAX still runs the same kernels. It does not download the NVIDIA wheels.
