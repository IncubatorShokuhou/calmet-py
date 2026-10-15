#!/usr/bin/env python3
"""Generate the statically typed, flattened ``run_pipeline`` wrapper."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import inspect
import sys
import textwrap
import types
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SOURCE: Final[Path] = ROOT / "src"
OUTPUT: Final[Path] = SOURCE / "pycalmet" / "_pipeline_api.py"
STAGES: Final[tuple[str, ...]] = (
    "calwrf", "terrel", "ctgproc", "makegeo", "calmet"
)
_SOURCE_PACKAGE: Final[str] = "_pycalmet_pipeline_api_codegen"


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


def _source_modules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
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
    parameter_docs = importlib.import_module(
        f"{_SOURCE_PACKAGE}._parameter_docs"
    )
    return api, config, parameter_docs


def _doc_entry(name: str, description: str) -> list[str]:
    """Render one indented Google ``Args`` entry for generated source.

    Args:
        name: Parameter name shown in the generated docstring.
        description: Complete human-readable parameter description.

    Returns:
        Source lines wrapped to the project's normal line width.
    """
    prefix = f"        {name}: "
    return textwrap.wrap(
        description,
        width=88,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def render() -> str:
    api, config, parameter_docs = _source_modules()
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
        "from datetime import date, datetime",
        "from pathlib import Path",
        "from typing import Any, Final, TypeAlias",
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
        "_STAGE_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {",
    ]
    for stage in STAGES:
        names = tuple(parameter.name for parameter in parameters[stage])
        lines.append(f"    {stage!r}: {names!r},")
    lines.extend([
        "}",
        "_OMITTED_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {",
    ])
    for stage in STAGES:
        omitted = _omitted_parameters(config_types[stage], GridConfig)
        # Only public parameters can be selected by the generated wrapper.
        public_names = {parameter.name for parameter in parameters[stage]}
        omitted = tuple(name for name in omitted if name in public_names)
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
        '    """Run all five CALMET workflow stages without persistent intermediates.',
        "",
        "    Args:",
    ])
    pipeline_arguments = {
        "inp": (
            "Pipeline configuration input. Pass a PipelineConfig or complete five-stage "
            "mapping by itself, a directory containing calwrf.inp, terrel.inp, "
            "ctgproc.inp, makegeo.inp, and calmet.inp with optional prefixed overrides, "
            "or None to use explicit stage specifications or prefixed typed parameters."
        ),
        "calwrf": (
            "CALWRF configuration object, mapping, or control-file path. Supply this "
            "together with terrel, ctgproc, makegeo, and calmet when inp is None, or "
            "omit all five stage specifications to use prefixed parameters."
        ),
        "terrel": "TERREL configuration object, mapping, or control-file path.",
        "ctgproc": "CTGPROC configuration object, mapping, or control-file path.",
        "makegeo": "MAKEGEO configuration object, mapping, or control-file path.",
        "calmet": "CALMET configuration object, mapping, or control-file path.",
        "output": (
            "Final CALMET product path; when supplied, this overrides calmet_output."
        ),
    }
    for name, description in pipeline_arguments.items():
        lines.extend(_doc_entry(name, description))
    for stage in STAGES:
        for parameter in parameters[stage]:
            description = parameter_docs.parameter_meaning(
                stage, parameter.name, "en"
            )
            lines.extend(_doc_entry(
                f"{stage}_{parameter.name}",
                f"{stage.upper()} stage. {description}",
            ))
    lines.extend([
        "",
        "    Returns:",
        "        Path to the final CALMET meteorological output.",
        "",
        "    Raises:",
        "        TypeError: If pipeline input modes are combined, a stage specification",
        "            is missing, or a parameter is unknown, duplicated, or invalid.",
        "        ValueError: If a stage configuration fails validation.",
        "        OSError: If a required input cannot be read or the final output cannot",
        "            be written.",
        '    """',
        "    values: dict[str, Any] = locals()",
        "    supplied: frozenset[str] = _EXPLICIT_ARGUMENTS.get() or frozenset()",
        "    parameter_only: bool = (",
        "        inp is None",
        "        and calwrf is None",
        "        and terrel is None",
        "        and ctgproc is None",
        "        and makegeo is None",
        "        and calmet is None",
        "    )",
        "    from .pipeline import _run_pipeline, _select_stage_parameters",
        "",
        "    stage_parameters: dict[str, dict[str, Any]] = {",
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
        '__all__: Final[list[str]] = ["run_pipeline"]',
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
