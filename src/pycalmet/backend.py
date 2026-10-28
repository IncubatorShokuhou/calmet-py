"""Optional NumPy and JAX array backends.

The default install and the default runtime are NumPy. Installing the
``gpu`` extra makes JAX importable; it does not change results until a caller
selects JAX with :func:`set_backend` or ``PYCALMET_BACKEND``.

Selection is process-global. :func:`use_backend` restores the previous choice
when its block exits, including the JAX x64 flag and default device this
module changed. Vectorized stage kernels place work on that device and return
NumPy arrays so NetCDF writers stay on the host. In-place diagnostic sweeps
stay on NumPy. See the user guide for the kernel list and dtype caveats.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from threading import RLock
from typing import Any, Final, Literal, TypeAlias, TypedDict
import os

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


BackendName: TypeAlias = Literal["numpy", "jax"]
BackendRequest: TypeAlias = Literal["numpy", "jax", "auto"]

_ENV_BACKEND: Final[str] = "PYCALMET_BACKEND"
_ENV_DEVICE: Final[str] = "PYCALMET_DEVICE"
_GPU_PLATFORMS: Final[frozenset[str]] = frozenset({"gpu", "cuda", "rocm"})


class BackendStatus(TypedDict):
    """Snapshot of the active array backend.

    Attributes:
        backend: Resolved backend name, ``numpy`` or ``jax``.
        jax_installed: Whether ``jax`` and ``jax.numpy`` import successfully.
        device: JAX device label when the JAX backend is active, otherwise
            ``None``.
        x64_enabled: Whether this module enabled JAX 64-bit mode. ``None``
            when the NumPy backend is active.
        model_dtype: Dtype used for physical model fields. Always ``float32``.
    """

    backend: str
    jax_installed: bool
    device: str | None
    x64_enabled: bool | None
    model_dtype: str


_lock: Final[Any] = RLock()
_selected: BackendName | None = None
_device_override: str | None = None
_jax_snapshot: tuple[bool, Any] | None = None


def jax_available() -> bool:
    """Report whether the optional JAX dependency imports.

    Returns:
        ``True`` when ``jax.numpy`` imports. A missing install returns
        ``False``. Other import failures propagate so a broken CUDA plugin
        is visible.
    """
    try:
        import_module("jax.numpy")
    except ImportError:
        return False
    return True


def available_backends() -> tuple[BackendName, ...]:
    """List backends that can be selected in this process.

    Returns:
        ``("numpy",)`` or ``("numpy", "jax")`` when JAX imports.
    """
    if jax_available():
        return ("numpy", "jax")
    return ("numpy",)


def get_backend() -> BackendName:
    """Return the active backend, resolving the environment on first use.

    Returns:
        ``numpy`` or ``jax``.

    Raises:
        ValueError: If ``PYCALMET_BACKEND`` names an unknown backend.
        ImportError: If the environment requests JAX and it is not installed.
        RuntimeError: If the environment requests a missing JAX device.
    """
    with _lock:
        return _ensure_selected()


def set_backend(
    name: BackendRequest,
    *,
    device: str | None = None,
) -> BackendName:
    """Select the array backend used by vectorized stage kernels.

    Args:
        name: ``numpy``, ``jax``, or ``auto``. ``auto`` uses JAX when it
            imports and otherwise stays on NumPy.
        device: JAX device request such as ``auto``, ``cpu``, ``gpu``,
            ``cuda:0``, or ``gpu:1``. ``None`` reads ``PYCALMET_DEVICE`` and
            otherwise prefers the first GPU, then CPU. Ignored by NumPy.

    Returns:
        The resolved backend name.

    Raises:
        ValueError: If ``name`` is not a known backend request.
        ImportError: If ``name`` resolves to JAX and JAX is not installed.
        RuntimeError: If ``device`` does not match an available JAX device.
    """
    with _lock:
        resolved = _normalize(name)
        _activate(resolved, device)
        return resolved


def reset_backend() -> None:
    """Forget the selected backend so the next use reads the environment.

    Returns:
        None. A later :func:`get_backend` call reads ``PYCALMET_BACKEND``
        again, or NumPy when that variable is unset. JAX configuration this
        module changed is restored.
    """
    global _selected, _device_override
    with _lock:
        _restore_jax_config()
        cleared_backend: BackendName | None = None
        cleared_device: str | None = None
        _selected = cleared_backend
        _device_override = cleared_device


@contextmanager
def use_backend(
    name: BackendRequest,
    *,
    device: str | None = None,
) -> Iterator[BackendName]:
    """Select a backend for the duration of a ``with`` block.

    Args:
        name: ``numpy``, ``jax``, or ``auto``.
        device: Optional JAX device request. See :func:`set_backend`.

    Yields:
        The resolved backend name active inside the block.

    Raises:
        ValueError: If ``name`` is not a known backend request.
        ImportError: If the block resolves to JAX and JAX is not installed.
        RuntimeError: If ``device`` does not match an available JAX device.
    """
    with _lock:
        previous = _ensure_selected()
        previous_device = _device_override
    try:
        yield set_backend(name, device=device)
    finally:
        set_backend(previous, device=previous_device)


def using_jax() -> bool:
    """Report whether vectorized kernels should run on JAX.

    Returns:
        ``True`` when :func:`get_backend` resolves to ``jax``.
    """
    return get_backend() == "jax"


def get_namespace() -> Any:
    """Return the active NumPy-like module.

    Returns:
        The ``numpy`` module, or ``jax.numpy`` when the JAX backend is active.
        Callers that write NetCDF or other host formats should convert with
        :func:`to_numpy`.
    """
    if get_backend() == "jax":
        return import_module("jax.numpy")
    return np


def asarray(values: ArrayLike, dtype: DTypeLike | None = None) -> Any:
    """Convert values with the active backend and place them on its device.

    Args:
        values: Scalar or array-like numeric data. JAX arrays are accepted
            when that backend is active.
        dtype: Optional NumPy dtype. ``None`` preserves the backend's default
            inference.

    Returns:
        A NumPy array on the NumPy backend, or a JAX array on the selected
        JAX device.
    """
    if get_backend() != "jax":
        if dtype is None:
            return np.asarray(values)
        return np.asarray(values, dtype=dtype)
    namespace = import_module("jax.numpy")
    if dtype is None:
        array = namespace.asarray(values)
    else:
        array = namespace.asarray(values, dtype=dtype)
    jax = import_module("jax")
    device = _jax_config_value("jax_default_device")
    if device is None:
        return array
    return jax.device_put(array, device)


def to_numpy(
    values: ArrayLike,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    """Copy an active-backend array onto the host as a NumPy array.

    Args:
        values: Scalar or array produced by :func:`get_namespace`.
        dtype: Optional NumPy dtype applied on the host.

    Returns:
        A NumPy array. JAX results are fully copied off the device first.
    """
    if get_backend() == "jax":
        jax = import_module("jax")
        values = jax.device_get(values)
    if dtype is None:
        return np.asarray(values)
    return np.asarray(values, dtype=dtype)


def backend_info() -> BackendStatus:
    """Describe the active backend, JAX install, and device.

    Returns:
        A :class:`BackendStatus` mapping. Probing ``jax_installed`` imports
        JAX when it is installed. ``device`` and ``x64_enabled`` are populated
        only while the JAX backend is selected.
    """
    backend = get_backend()
    device: str | None = None
    x64_enabled: bool | None = None
    if backend == "jax":
        current = _jax_config_value("jax_default_device")
        device = None if current is None else str(current)
        x64_enabled = bool(_jax_config_value("jax_enable_x64"))
    return {
        "backend": backend,
        "jax_installed": jax_available(),
        "device": device,
        "x64_enabled": x64_enabled,
        "model_dtype": "float32",
    }


def _ensure_selected() -> BackendName:
    """Resolve and activate the backend if selection has not happened yet.

    Returns:
        The active backend name.

    Raises:
        ValueError: If ``PYCALMET_BACKEND`` names an unknown backend.
        ImportError: If the environment requests JAX and it is not installed.
        RuntimeError: If the environment requests a missing JAX device.
    """
    global _selected
    if _selected is None:
        _activate(_normalize(os.environ.get(_ENV_BACKEND, "numpy")), None)
    if _selected is None:
        return "numpy"
    return _selected


def _activate(name: BackendName, device: str | None) -> None:
    """Store ``name`` and apply or restore JAX process settings.

    Args:
        name: Resolved backend.
        device: Explicit device request. ``None`` reads ``PYCALMET_DEVICE``.

    Returns:
        None.

    Raises:
        RuntimeError: If a JAX device request cannot be satisfied.
    """
    global _selected, _device_override
    _device_override = device
    _selected = name
    if name == "jax":
        _configure_jax(device)
        return
    _restore_jax_config()


def _normalize(name: str) -> BackendName:
    """Map a user or environment token onto ``numpy`` or ``jax``.

    Args:
        name: Raw backend request. Blank text selects NumPy.

    Returns:
        The resolved backend name.

    Raises:
        ValueError: If ``name`` is not ``numpy``, ``jax``, or ``auto``.
        ImportError: If ``name`` requests JAX and it is not installed.
    """
    token = name.strip().lower()
    if token in {"", "numpy"}:
        return "numpy"
    if token == "auto":
        if jax_available():
            return "jax"
        return "numpy"
    if token == "jax":
        _require_jax()
        return "jax"
    raise ValueError(
        f"Unknown array backend {name!r}. Choose 'numpy', 'jax', or 'auto'."
    )


def _require_jax() -> None:
    """Import JAX or raise an install hint.

    Returns:
        None.

    Raises:
        ImportError: If ``jax.numpy`` cannot be imported.
    """
    try:
        import_module("jax.numpy")
    except ImportError as error:
        raise ImportError(
            "The JAX backend requires the optional GPU extra. Install it "
            "with `python -m pip install pycalmet[gpu]` or, from a source "
            "checkout, `python -m pip install \".[gpu]\"`. "
            f"Import failed with: {error}"
        ) from error


def _configure_jax(device: str | None) -> None:
    """Enable float64 and place new JAX work on the requested device.

    Args:
        device: Device request forwarded to :func:`_choose_device`.

    Returns:
        None. The previous JAX x64 flag and default device are saved once
        so :func:`_restore_jax_config` can put them back.

    Raises:
        RuntimeError: If ``device`` does not match an available JAX device.
    """
    global _jax_snapshot
    jax = import_module("jax")
    if _jax_snapshot is None:
        _jax_snapshot = (
            bool(_jax_config_value("jax_enable_x64")),
            _jax_config_value("jax_default_device"),
        )
    selected = _choose_device(device)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_device", selected)


def _restore_jax_config() -> None:
    """Restore JAX flags saved by :func:`_configure_jax`.

    Returns:
        None. Nothing is changed when this module has not touched JAX.
    """
    global _jax_snapshot
    if _jax_snapshot is None:
        return
    jax = import_module("jax")
    enabled, default_device = _jax_snapshot
    jax.config.update("jax_enable_x64", enabled)
    jax.config.update("jax_default_device", default_device)
    cleared: tuple[bool, Any] | None = None
    _jax_snapshot = cleared


def _jax_config_value(name: str) -> Any:
    """Read one JAX configuration value across recent JAX releases.

    Args:
        name: JAX configuration key, such as ``jax_enable_x64``.

    Returns:
        The current value stored for ``name``.
    """
    jax = import_module("jax")
    # ``jax.config.read`` rejects device flags that also have a context
    # manager. Attribute access works for both those flags and ``jax_enable_x64``.
    if hasattr(jax.config, name):
        return getattr(jax.config, name)
    return jax.config.read(name)


def _choose_device(device: str | None) -> Any:
    """Resolve a JAX device from an explicit request or the environment.

    Args:
        device: Request such as ``auto``, ``cpu``, ``gpu``, or ``cuda:0``.
            ``None`` uses ``PYCALMET_DEVICE`` and then ``auto``.

    Returns:
        The matching JAX device object.

    Raises:
        RuntimeError: If the request is unknown or no matching device exists.
    """
    jax = import_module("jax")
    raw = device if device is not None else os.environ.get(_ENV_DEVICE, "auto")
    text = raw.strip().lower()
    if text in {"", "auto"}:
        gpus = _gpu_devices()
        if gpus:
            return gpus[0]
        return jax.devices("cpu")[0]
    if text == "cpu":
        return jax.devices("cpu")[0]
    if text in {"gpu", "cuda"}:
        gpus = _gpu_devices()
        if not gpus:
            raise RuntimeError(
                "A GPU was requested, but JAX did not find a GPU device."
            )
        return gpus[0]
    if ":" in text:
        kind, _, index_text = text.partition(":")
        try:
            index = int(index_text)
        except ValueError as error:
            raise RuntimeError(
                f"JAX device request {raw!r} must use an integer index."
            ) from error
        if kind in {"gpu", "cuda"}:
            pool = _gpu_devices()
        elif kind == "cpu":
            pool = list(jax.devices("cpu"))
        else:
            pool = [
                item
                for item in jax.devices()
                if str(item.platform).lower() == kind
            ]
        if index < 0 or index >= len(pool):
            raise RuntimeError(
                f"JAX device request {raw!r} is outside the "
                f"{len(pool)} available device(s)."
            )
        return pool[index]
    for item in jax.devices():
        if text == str(item).lower():
            return item
    raise RuntimeError(f"JAX has no device matching {raw!r}.")


def _gpu_devices() -> list[Any]:
    """Return JAX devices whose platform is a GPU.

    Returns:
        GPU devices in JAX's order. The list is empty when JAX only sees CPU.
    """
    jax = import_module("jax")
    return [
        device
        for device in jax.devices()
        if str(device.platform).lower() in _GPU_PLATFORMS
    ]
