"""Guard the project's netCDF4-only I/O policy."""

import ast
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PACKAGES = frozenset({"xarray", "rioxarray", "xwrf", "metpy"})
FORBIDDEN_NETCDF_PACKAGES = frozenset({"h5netcdf", "h5py", "zarr"})
FORBIDDEN_IMPORTS = FORBIDDEN_PACKAGES | FORBIDDEN_NETCDF_PACKAGES
FORBIDDEN_IMPORT_PREFIXES = ("scipy.io.netcdf",)


def _dependency_name(requirement: str) -> str:
    """Return the normalized leading package name from a simple requirement."""
    name = requirement.strip()
    for marker in ("[", "<", ">", "=", "!", "~", ";", " "):
        name = name.split(marker, 1)[0]
    return name.lower()


def test_runtime_does_not_import_xarray_ecosystem():
    violations = []
    for path in (ROOT / "src" / "pycalmet").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                continue
            if any(
                name.split(".", 1)[0].lower() in FORBIDDEN_IMPORTS
                or name.lower().startswith(FORBIDDEN_IMPORT_PREFIXES)
                for name in imported
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "xarray ecosystem imports found: " + ", ".join(violations)


def test_project_does_not_depend_on_xarray_ecosystem():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    forbidden = [
        dependency
        for dependency in dependencies
        if _dependency_name(dependency) in FORBIDDEN_IMPORTS
    ]
    assert not forbidden


def test_normal_import_does_not_load_legacy_compatibility_graph():
    script = (
        "import sys; import pycalmet; "
        "assert not any(name.startswith('pycalmet.legacy._compat') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
