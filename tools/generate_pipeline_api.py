#!/usr/bin/env python3
"""Generate the statically typed, flattened ``run_pipeline`` wrapper."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import inspect
import sys
import types
from dataclasses import MISSING, fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
OUTPUT = SOURCE / "pycalmet" / "_pipeline_api.py"
STAGES = ("calwrf", "terrel", "ctgproc", "makegeo", "calmet")
_SOURCE_PACKAGE = "_pycalmet_pipeline_api_codegen"


def _annotation(parameter: inspect.Parameter) -> str:
    value = parameter.annotation
    if value is inspect.Parameter.empty:
        return "Any"
    return value if isinstance(value, str) else inspect.formatannotation(value)


def _default(parameter: inspect.Parameter) -> str:
    if parameter.default is inspect.Parameter.empty:
        raise ValueError(f"pipeline parameter {parameter.name!r} has no default")
    return repr(parameter.default)


def _omitted_parameters(config_type: type, grid_type: type) -> tuple[str, ...]:
    """Return the ``None`` sentinels omitted by the individual stage API."""
    omitted: set[str] = set()
    for field in (*fields(config_type), *fields(grid_type)):
        if field.default is MISSING or field.default_factory is not MISSING:
            omitted.add(field.name)
    return tuple(sorted(omitted))


def _source_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Load API metadata without executing ``pycalmet.__init__``.

    The real package imports the generated module through ``pipeline``.  A
    private, lightweight package namespace breaks that bootstrap cycle while
    preserving the relative imports used by ``api.py`` and ``config.py``.
    """
    for name in tuple(sys.modules):
        if name == _SOURCE_PACKAGE or name.startswith(f"{_SOURCE_PACKAGE}."):
            del sys.modules[name]

    package_directory = SOURCE / "pycalmet"
    specification = importlib.machinery.ModuleSpec(
        _SOURCE_PACKAGE, loader=None, is_package=True
    )
    specification.submodule_search_locations = [str(package_directory)]
    package = types.ModuleType(_SOURCE_PACKAGE)
    package.__package__ = _SOURCE_PACKAGE
    package.__path__ = [str(package_directory)]
    package.__spec__ = specification
    sys.modules[_SOURCE_PACKAGE] = package

    api = importlib.import_module(f"{_SOURCE_PACKAGE}.api")
    config = importlib.import_module(f"{_SOURCE_PACKAGE}.config")
    return api, config


def render() -> str:
    api, config = _source_modules()
    CalmetConfig = config.CalmetConfig
    CalwrfConfig = config.CalwrfConfig
    CtgprocConfig = config.CtgprocConfig
    GridConfig = config.GridConfig
    MakegeoConfig = config.MakegeoConfig
    TerrelConfig = config.TerrelConfig

    functions = {name: getattr(api, name) for name in STAGES}
    config_types = {
        "calwrf": CalwrfConfig,
        "terrel": TerrelConfig,
        "ctgproc": CtgprocConfig,
        "makegeo": MakegeoConfig,
        "calmet": CalmetConfig,
    }
    parameters: dict[str, list[inspect.Parameter]] = {}
    for name, function in functions.items():
        parameters[name] = [
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.name not in {"inp", "legacy_parameters"}
        ]

    lines = [
        '"""Generated static public wrapper for the complete CALMET pipeline.',
        "",
        "Regenerate this file with ``python tools/generate_pipeline_api.py``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Mapping, Sequence",
        "from datetime import datetime",
        "from pathlib import Path",
        "from typing import Any, TypeAlias",
        "",
        "from .api import _EXPLICIT_ARGUMENTS, _capture_explicit_arguments",
        "from .config import (",
        "    CalmetConfig, CalwrfConfig, CtgprocConfig, CtgprocSource,",
        "    GridConfig, MakegeoConfig, PipelineConfig, TerrelConfig,",
        "    TerrelSource,",
        ")",
        "",
        "StageSpec: TypeAlias = Mapping[str, Any] | str | Path",
        "CalwrfSpec: TypeAlias = CalwrfConfig | StageSpec",
        "TerrelSpec: TypeAlias = TerrelConfig | StageSpec",
        "CtgprocSpec: TypeAlias = CtgprocConfig | StageSpec",
        "MakegeoSpec: TypeAlias = MakegeoConfig | StageSpec",
        "CalmetSpec: TypeAlias = CalmetConfig | StageSpec",
        "",
        "_STAGE_PARAMETERS = {",
    ]
    for stage in STAGES:
        names = tuple(parameter.name for parameter in parameters[stage])
        lines.append(f"    {stage!r}: {names!r},")
    lines.extend(["}", "_OMITTED_PARAMETERS = {"])
    for stage in STAGES:
        omitted = _omitted_parameters(config_types[stage], GridConfig)
        # Only public parameters can be selected by the generated wrapper.
        names = {parameter.name for parameter in parameters[stage]}
        omitted = tuple(name for name in omitted if name in names)
        lines.append(f"    {stage!r}: {omitted!r},")
    lines.extend([
        "}",
        "",
        "",
        "@_capture_explicit_arguments",
        "def run_pipeline(",
        "    inp: PipelineConfig | Mapping[str, Any] | str | Path | None = None,",
        "    *,",
        "    calwrf: CalwrfSpec | None = None,",
        "    terrel: TerrelSpec | None = None,",
        "    ctgproc: CtgprocSpec | None = None,",
        "    makegeo: MakegeoSpec | None = None,",
        "    calmet: CalmetSpec | None = None,",
        "    output: str | Path | None = None,",
    ])
    for stage in STAGES:
        for parameter in parameters[stage]:
            lines.append(
                f"    {stage}_{parameter.name}: {_annotation(parameter)} = "
                f"{_default(parameter)},"
            )
    lines.extend([
        ") -> Path:",
        '    """Run all five stages from controls or fully typed parameters.',
        "",
        "    Every stage parameter uses its stage name as a prefix.  ``output``",
        "    overrides ``calmet_output`` as the final product path.",
        '    """',
        "    values = locals()",
        "    supplied = _EXPLICIT_ARGUMENTS.get() or frozenset()",
        "    parameter_only = (",
        "        inp is None",
        "        and calwrf is None",
        "        and terrel is None",
        "        and ctgproc is None",
        "        and makegeo is None",
        "        and calmet is None",
        "    )",
        "    from .pipeline import _run_pipeline, _select_stage_parameters",
        "",
        "    stage_parameters = {",
        "        stage: _select_stage_parameters(",
        "            stage, values, supplied, _STAGE_PARAMETERS[stage],",
        "            _OMITTED_PARAMETERS[stage], parameter_only=parameter_only,",
        "        )",
        "        for stage in _STAGE_PARAMETERS",
        "    }",
        "    return _run_pipeline(",
        "        inp,",
        "        calwrf=calwrf,",
        "        terrel=terrel,",
        "        ctgproc=ctgproc,",
        "        makegeo=makegeo,",
        "        calmet=calmet,",
        "        output=output,",
        "        stage_parameters=stage_parameters,",
        "    )",
        "",
        "",
        '__all__ = ["run_pipeline"]',
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if the generated file is stale"
    )
    options = parser.parse_args()
    rendered = render()
    if options.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(f"{OUTPUT.relative_to(ROOT)} is out of date", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
