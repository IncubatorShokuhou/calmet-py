"""Array-backend selection and JAX parity for vectorized stage kernels."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import numpy as np
import pytest

import pycalmet
from pycalmet import _device_kernels
from pycalmet._calwrf import (
    _pressure_temperature,
    _rain_increment,
    destagger_x,
    destagger_y,
    destagger_z,
    earth_wind,
    speed_direction,
)
from pycalmet._ctgproc import _bin_samples, _grid_indices
from pycalmet._makegeo import _category_sum, _fields, _water_sum
from pycalmet._terrel import _cell_averages
from pycalmet.calmet_numerics import (
    cloud_fraction,
    coriolis_parameter,
    density_from_pressure_temperature,
    divergence_and_vertical_velocity,
    heat_flux,
    inverse_distance_squared,
    solar_sine,
    wind_speed_direction,
)
from pycalmet.config import GridConfig, MakegeoConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_array_backend():
    """Keep backend selection from leaking between tests."""
    pycalmet.reset_backend()
    yield
    pycalmet.reset_backend()


def test_default_backend_is_numpy_without_importing_jax():
    """The import and default selection stay on NumPy even if JAX is installed."""
    env = os.environ.copy()
    env.pop("PYCALMET_BACKEND", None)
    env.pop("PYCALMET_DEVICE", None)
    script = (
        "import sys; import pycalmet; "
        "assert pycalmet.get_backend() == 'numpy'; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_default_dependencies_do_not_require_jax():
    """The base install stays free of JAX; the gpu extra carries it."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    required = [item.lower() for item in project["dependencies"]]
    assert all(not item.startswith("jax") for item in required)
    extra = project["optional-dependencies"]["gpu"]
    assert any(item.startswith("jax[cuda12]") for item in extra)
    assert any(item.startswith("jax>=") for item in extra)
    assert any("platform_system == 'Linux'" in item for item in extra)


def test_public_backend_api_is_exported():
    """Selection helpers are part of the package surface."""
    for name in (
        "get_backend",
        "set_backend",
        "reset_backend",
        "use_backend",
        "backend_info",
        "available_backends",
        "jax_available",
    ):
        assert name in pycalmet.__all__
        assert callable(getattr(pycalmet, name))


def test_numpy_backend_reports_host_status():
    """NumPy selection does not advertise a device or x64 mode."""
    assert pycalmet.set_backend("numpy") == "numpy"
    info = pycalmet.backend_info()
    assert info["backend"] == "numpy"
    assert info["model_dtype"] == "float32"
    assert info["device"] is None
    assert info["x64_enabled"] is None
    assert info["jax_installed"] is pycalmet.jax_available()
    assert "numpy" in pycalmet.available_backends()


def test_unknown_backend_is_rejected():
    """Backend names outside the dispatcher are an error."""
    with pytest.raises(ValueError, match="Unknown array backend"):
        pycalmet.set_backend("cupy")  # type: ignore[arg-type]


def test_auto_without_jax_stays_on_numpy(monkeypatch):
    """``auto`` is a portable default when the optional extra is absent."""
    monkeypatch.setattr("pycalmet.backend.jax_available", lambda: False)
    assert pycalmet.set_backend("auto") == "numpy"


def test_missing_jax_install_names_the_extra(monkeypatch):
    """A JAX request explains the optional extra instead of failing raw."""
    def fail_jax(name: str, package: str | None = None):
        """Raise ImportError for JAX and import everything else normally."""
        if name == "jax.numpy" or name.startswith("jax"):
            raise ImportError("mocked missing jax")
        return original(name, package)

    original = pycalmet.backend.import_module
    monkeypatch.setattr("pycalmet.backend.import_module", fail_jax)
    monkeypatch.setattr("pycalmet.backend.jax_available", lambda: False)
    with pytest.raises(ImportError, match=r"pycalmet\[gpu\]"):
        pycalmet.set_backend("jax")


def test_environment_selects_numpy_explicitly(monkeypatch):
    """``PYCALMET_BACKEND`` is read when no explicit selection is stored."""
    monkeypatch.setenv("PYCALMET_BACKEND", "numpy")
    pycalmet.reset_backend()
    assert pycalmet.get_backend() == "numpy"


def test_use_backend_restores_the_previous_choice():
    """The context manager does not leak its selection."""
    pycalmet.set_backend("numpy")
    with pycalmet.use_backend("numpy") as selected:
        assert selected == "numpy"
        assert pycalmet.get_backend() == "numpy"
    assert pycalmet.get_backend() == "numpy"


def test_destagger_numpy_matches_the_arithmetic_mean():
    """The NumPy stagger removal path keeps its original definition."""
    values = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    expected = np.float32(0.5) * (values[:-1] + values[1:])
    np.testing.assert_array_equal(destagger_x(values), expected)


@pytest.fixture
def jax_cpu():
    """Select the JAX backend on CPU for one test."""
    pytest.importorskip("jax.numpy")
    with pycalmet.use_backend("jax", device="cpu") as selected:
        assert selected == "jax"
        info = pycalmet.backend_info()
        assert info["backend"] == "jax"
        assert info["x64_enabled"] is True
        assert info["device"] is not None
        assert "cpu" in info["device"].lower()
        yield


def _compare(function, *args, rtol=1e-5, atol=1e-5, **kwargs):
    """Return JAX output and assert it matches a NumPy call of ``function``."""
    with pycalmet.use_backend("numpy"):
        expected = function(*args, **kwargs)
    actual = function(*args, **kwargs)
    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        pairs = zip(actual, expected, strict=True)
    else:
        pairs = ((actual, expected),)
    for left, right in pairs:
        assert left.shape == right.shape
        if np.issubdtype(left.dtype, np.integer) or np.issubdtype(right.dtype, np.integer):
            assert left.dtype == right.dtype
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)
    return actual


def test_calwrf_vector_kernels_match_numpy(jax_cpu):
    """CALWRF stagger, wind, rain, and thermodynamics agree on CPU JAX."""
    staggered = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    _compare(destagger_x, staggered, rtol=0, atol=0)
    _compare(destagger_y, staggered, rtol=0, atol=0)
    _compare(destagger_z, staggered, rtol=0, atol=0)
    angle = np.linspace(-0.4, 0.4, 6, dtype=np.float32).reshape(2, 3)
    wind = staggered[:, :2, :3]
    _compare(earth_wind, wind, wind * np.float32(0.5), angle)
    _compare(speed_direction, wind[0], wind[0] * np.float32(-0.25))
    current = np.array([[1.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    previous = np.array([[0.5, 5.0], [0.0, 2.0]], dtype=np.float32)
    _compare(_rain_increment, current, previous, rtol=0, atol=0)
    _compare(_rain_increment, current, None, rtol=0, atol=0)
    _compare(
        _pressure_temperature,
        np.full((2, 2), 100.0, dtype=np.float32),
        np.full((2, 2), 100000.0, dtype=np.float32),
        np.full((2, 2), -10.0, dtype=np.float32),
        rtol=1e-6,
        atol=1e-5,
    )


def test_terrain_landuse_and_makegeo_match_numpy(jax_cpu):
    """TERREL, CTGPROC, and MAKEGEO reductions agree on CPU JAX."""
    _compare(
        _cell_averages,
        np.array([[10.0, 0.0], [3.0, 5.0]]),
        np.array([[2, 0], [1, 4]]),
        rtol=0,
        atol=0,
    )
    grid = GridConfig(nx=2, ny=2, dx_km=1.0, dy_km=1.0, x_origin_km=0.0, y_origin_km=0.0)
    _compare(
        _grid_indices,
        np.array([0.2, 1.2, -0.1]),
        np.array([0.2, 1.8, 0.2]),
        grid,
        rtol=0,
        atol=0,
    )
    samples_x = np.array([0.2, 1.2, 0.2])
    samples_y = np.array([0.2, 0.2, 1.2])
    categories = np.array([10, 10, 20], dtype=np.int16)
    good = np.ones(3, dtype=bool)
    config = SimpleNamespace(grid=grid)
    columns = {10: 0, 20: 1}

    def accumulate():
        """Bin one fixed sample set into a fresh count array."""
        counts = np.zeros((2, 4), dtype=np.int64)
        _bin_samples(counts, columns, samples_x, samples_y, categories, good, config)
        return counts

    with pycalmet.use_backend("numpy"):
        expected = accumulate()
    actual = accumulate()
    np.testing.assert_array_equal(actual, expected)
    fractions = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    fractions /= fractions.sum(axis=0, keepdims=True)
    _compare(_category_sum, fractions, np.array([True, False, True]), rtol=1e-6, atol=1e-6)
    _compare(_water_sum, fractions, np.array([10, 51, 20]), (51, 55), rtol=1e-6, atol=1e-6)
    field_fractions = np.zeros((38, 1, 1), dtype=np.float32)
    field_fractions[0, 0, 0] = 1.0
    config_geo = MakegeoConfig("t", "l", "o", GridConfig(1, 1, 1, 1))
    _compare(_fields, field_fractions, config_geo, rtol=1e-6, atol=1e-6)


def test_calmet_vector_kernels_match_numpy(jax_cpu):
    """CALMET diagnostic kernels agree; the in-place sweeps are not invoked."""
    pressure = np.array([1000.0, 850.0], dtype=np.float32)
    temperature = np.array([300.0, 280.0], dtype=np.float32)
    _compare(density_from_pressure_temperature, pressure, temperature, rtol=1e-6, atol=1e-6)
    _compare(coriolis_parameter, np.array([0.0, 30.0, -45.0]), rtol=1e-6, atol=1e-7)
    _compare(coriolis_parameter, np.array([10.0, 20.0]), 1.0e-4, rtol=1e-6, atol=1e-7)
    _compare(
        inverse_distance_squared,
        np.array([1.0, 3.0, 0.0], dtype=np.float32),
        np.array([0.0, 2.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.0, 2.0]),
        ignore_zero=True,
        rtol=1e-6,
        atol=1e-6,
    )
    wind = np.array([[1.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    _compare(wind_speed_direction, wind, wind * 0.5, rtol=1e-6, atol=1e-5)
    _compare(cloud_fraction, np.array([50.0, 98.0, 99.0, 100.0], dtype=np.float32))
    latitude = np.array([[30.0, 31.0], [32.0, 33.0]])
    longitude = np.array([[120.0, 121.0], [120.5, 121.5]])
    _compare(solar_sine, 77, 8.0, latitude, longitude, rtol=1e-6, atol=1e-6)
    levels = 2
    shape = (levels, 3, 4)
    rng = np.random.default_rng(4)
    u = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)
    faces = np.array([0.0, 20.0, 80.0], dtype=np.float64)
    _compare(divergence_and_vertical_velocity, u, v, faces, 1000.0, 1000.0, rtol=1e-6, atol=1e-6)
    _compare(
        divergence_and_vertical_velocity,
        u,
        v,
        faces,
        1000.0,
        1000.0,
        obrien=True,
        rtol=1e-6,
        atol=1e-6,
    )
    landuse = np.array([[10, 55], [20, 52]])
    ones = np.ones((2, 2), dtype=np.float32)
    _compare(
        heat_flux,
        np.array([[0.4, 0.0], [0.2, 0.8]], dtype=np.float32),
        np.array([[0.1, 0.9], [0.0, 0.3]], dtype=np.float32),
        np.full((2, 2), 290.0, dtype=np.float32),
        landuse,
        np.full((2, 2), 0.2, dtype=np.float32),
        ones,
        np.full((2, 2), 0.15, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )


def test_jax_namespace_is_used_and_results_return_to_numpy(jax_cpu):
    """Device kernels place inputs on JAX and hand NumPy arrays back."""
    jax_numpy = pytest.importorskip("jax.numpy")
    assert pycalmet.backend.get_namespace() is jax_numpy
    values = _device_kernels.destagger_x(np.array([0.0, 2.0, 4.0], dtype=np.float32))
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32


def test_gpu_request_without_a_gpu_is_explicit(jax_cpu):
    """Asking for a GPU when JAX has none fails before any kernel runs."""
    jax = pytest.importorskip("jax")
    if any(str(device.platform).lower() in {"gpu", "cuda", "rocm"} for device in jax.devices()):
        pycalmet.set_backend("jax", device="gpu")
        assert "cpu" not in str(pycalmet.backend_info()["device"]).lower()
        return
    with pytest.raises(RuntimeError, match="did not find a GPU"):
        pycalmet.set_backend("jax", device="gpu")
