"""Run the complete CALWRF-to-CALMET workflow as one operation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeAlias

from .api import _config
from .calmet import run as _run_calmet
from .calwrf import run as _run_calwrf
from .config import (CalmetConfig, CalwrfConfig, CtgprocConfig, MakegeoConfig,
                     PipelineConfig, TerrelConfig)
from .control import (calmet_config, calwrf_config, ctgproc_config,
                      makegeo_config, terrel_config)
from .ctrproc import run as _run_ctgproc
from .makegeo import run as _run_makegeo
from .parameters import normalize_public_parameters
from .terrel import run as _run_terrel

StageSpec: TypeAlias = Mapping[str, Any] | str | Path
CalwrfSpec: TypeAlias = CalwrfConfig | StageSpec
TerrelSpec: TypeAlias = TerrelConfig | StageSpec
CtgprocSpec: TypeAlias = CtgprocConfig | StageSpec
MakegeoSpec: TypeAlias = MakegeoConfig | StageSpec
CalmetSpec: TypeAlias = CalmetConfig | StageSpec

_CONTROL_FILES = {
    "calwrf": "calwrf.inp",
    "terrel": "terrel.inp",
    "ctgproc": "ctgproc.inp",
    "makegeo": "makegeo.inp",
    "calmet": "calmet.inp",
}


def _directory_specs(path: str | Path) -> dict[str, Path]:
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"pipeline input must be a directory containing the five control "
            f"files: {directory}"
        )
    specs = {name: directory / filename
             for name, filename in _CONTROL_FILES.items()}
    missing = [path.name for path in specs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"pipeline control directory {directory} is missing: "
            + ", ".join(missing)
        )
    return specs


def _mapping_specs(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - set(_CONTROL_FILES)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise TypeError(f"unknown pipeline configuration field(s): {names}")
    missing = set(_CONTROL_FILES) - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"missing pipeline configuration field(s): {names}")
    return {name: value[name] for name in _CONTROL_FILES}


def _mapping_defaults(cls: Any, value: Any, **defaults: Any) -> Any:
    """Supply fields whose paths are managed by the pipeline."""
    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    # Normalize first so legacy spellings such as OUTFIL, LUDAT, GEODAT,
    # METDAT, M3DDAT, and CALWRF's FIO count as their canonical fields.
    # Adding a canonical default beside one of those aliases would otherwise
    # make the stage parser correctly report a duplicate parameter.
    present = normalize_public_parameters(cls, value).values
    for name, default in defaults.items():
        if name not in present:
            result[name] = default
    return result


def _annotate_error(error: Exception, stage: str) -> None:
    message = f"pycalmet.run_pipeline stage: {stage}"
    if hasattr(error, "add_note"):
        error.add_note(message)
    elif error.args:
        error.args = (f"{error.args[0]} ({message})", *error.args[1:])
    else:
        error.args = (message,)


def _resolve_stage(name: str, cls: Any, value: Any, loader: Any) -> Any:
    try:
        return _config(cls, value, {}, loader)
    except Exception as error:
        _annotate_error(error, name)
        raise


def _resolve_config(
    inp: PipelineConfig | Mapping[str, Any] | str | Path | None,
    *,
    calwrf: CalwrfSpec | None,
    terrel: TerrelSpec | None,
    ctgproc: CtgprocSpec | None,
    makegeo: MakegeoSpec | None,
    calmet: CalmetSpec | None,
    output: str | Path | None,
) -> PipelineConfig:
    explicit = {
        "calwrf": calwrf,
        "terrel": terrel,
        "ctgproc": ctgproc,
        "makegeo": makegeo,
        "calmet": calmet,
    }
    supplied = {name: value for name, value in explicit.items()
                if value is not None}
    if inp is not None and supplied:
        raise TypeError(
            "pass either a pipeline config/control directory or the five stage "
            "configurations, not both"
        )
    if isinstance(inp, PipelineConfig):
        return inp
    if isinstance(inp, (str, Path)):
        specs: Mapping[str, Any] = _directory_specs(inp)
    elif isinstance(inp, Mapping):
        specs = _mapping_specs(inp)
    elif inp is None:
        if len(supplied) != len(_CONTROL_FILES):
            missing = [name for name in _CONTROL_FILES if name not in supplied]
            raise TypeError(
                "run_pipeline requires all five stage configurations; missing: "
                + ", ".join(missing)
            )
        specs = explicit
    else:
        raise TypeError(
            "inp must be a PipelineConfig, mapping, or control-file directory"
        )

    specs = dict(specs)
    specs["calwrf"] = _mapping_defaults(
        CalwrfConfig, specs["calwrf"],
        output="__pycalmet_pipeline_calwrf__.nc")
    specs["terrel"] = _mapping_defaults(
        TerrelConfig, specs["terrel"],
        output="__pycalmet_pipeline_terrain__.nc")
    specs["ctgproc"] = _mapping_defaults(
        CtgprocConfig, specs["ctgproc"],
        output="__pycalmet_pipeline_landuse__.nc")
    specs["makegeo"] = _mapping_defaults(
        MakegeoConfig, specs["makegeo"],
        terrain="__pycalmet_pipeline_terrain__.nc",
        landuse="__pycalmet_pipeline_landuse__.nc",
        output="__pycalmet_pipeline_geo__.nc",
    )
    calmet_defaults: dict[str, Any] = {
        "meteorology": "__pycalmet_pipeline_calwrf__.nc",
        "geophysical": "__pycalmet_pipeline_geo__.nc",
    }
    if output is not None:
        calmet_defaults["output"] = output
    specs["calmet"] = _mapping_defaults(
        CalmetConfig, specs["calmet"], **calmet_defaults)

    # Resolve every control before starting expensive processing.  A malformed
    # late-stage control therefore cannot leave work products behind.
    return PipelineConfig(
        calwrf=_resolve_stage(
            "CALWRF", CalwrfConfig, specs["calwrf"], calwrf_config),
        terrel=_resolve_stage(
            "TERREL", TerrelConfig, specs["terrel"], terrel_config),
        ctgproc=_resolve_stage(
            "CTGPROC", CtgprocConfig, specs["ctgproc"], ctgproc_config),
        makegeo=_resolve_stage(
            "MAKEGEO", MakegeoConfig, specs["makegeo"], makegeo_config),
        calmet=_resolve_stage(
            "CALMET", CalmetConfig, specs["calmet"], calmet_config),
    )


def _run_stage(name: str, runner: Any, config: Any) -> Path:
    try:
        return Path(runner(config))
    except Exception as error:
        _annotate_error(error, name)
        raise


def _effective_output(config: CalmetConfig, output: str | Path | None) -> Path:
    if output is not None:
        return Path(output)
    if config.legacy_output_format == 2 and config.mesopac_output is not None:
        return Path(config.mesopac_output)
    return Path(config.output)


def run_pipeline(
    inp: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    calwrf: CalwrfSpec | None = None,
    terrel: TerrelSpec | None = None,
    ctgproc: CtgprocSpec | None = None,
    makegeo: MakegeoSpec | None = None,
    calmet: CalmetSpec | None = None,
    output: str | Path | None = None,
) -> Path:
    """Run CALWRF, TERREL, CTGPROC, MAKEGEO, and CALMET in sequence.

    ``inp`` may be a directory containing ``calwrf.inp``, ``terrel.inp``,
    ``ctgproc.inp``, ``makegeo.inp``, and ``calmet.inp``.  Alternatively,
    supply all five named stage configurations as dataclass instances,
    mappings, or control-file paths.

    The four preprocessing products are written to a private temporary
    directory and removed when the run finishes.  Stage listing, plot, QA,
    legacy M3D/M2D, and debug outputs are disabled.  CALMET writes its normal
    configured final format (a legacy CALMET file can also require its ``.aux``
    companion).
    """
    config = _resolve_config(
        inp,
        calwrf=calwrf,
        terrel=terrel,
        ctgproc=ctgproc,
        makegeo=makegeo,
        calmet=calmet,
        output=output,
    )
    if config.makegeo.snow_processing:
        raise ValueError(
            "run_pipeline does not support MAKEGEO snow_processing because it "
            "produces multiple dated geophysical files"
        )
    if not config.calmet.save_output:
        raise ValueError("run_pipeline requires CALMET save_output=True")
    if config.calmet.test_mode != 2:
        raise ValueError("run_pipeline requires CALMET test_mode=2")

    final_output = _effective_output(config.calmet, output)
    with TemporaryDirectory(prefix="pycalmet-pipeline-") as temporary:
        workspace = Path(temporary)
        meteorology = workspace / "calwrf.nc"
        terrain = workspace / "terrain.nc"
        landuse = workspace / "landuse.nc"
        geophysical = workspace / "geo.nc"

        calwrf_configured = replace(
            config.calwrf,
            output=meteorology,
            create_surface=False,
            legacy_output=None,
            surface_output=None,
            listing_output=None,
        )
        terrel_configured = replace(
            config.terrel,
            output=terrain,
            receptor_output=(workspace / "terrel-receptors.dat"
                             if config.terrel.receptor_input is not None else None),
            save_output=None,
            plot_output=None,
            raw_echo_output=None,
            coastline_file=(workspace / "terrel-coastline.bln"
                            if config.terrel.gshhs_file is not None
                            else config.terrel.coastline_file),
        )
        ctgproc_configured = replace(
            config.ctgproc,
            output=landuse,
            output_format="netcdf",
            final_output=True,
            qa_cell_output=False,
            qa_output_directory=None,
            coastline_file=(workspace / "ctgproc-coastline.bln"
                            if config.ctgproc.gshhs_file is not None
                            else config.ctgproc.coastline_file),
        )
        makegeo_configured = replace(
            config.makegeo,
            terrain=terrain,
            landuse=landuse,
            output=geophysical,
            landuse_plot_output=None,
            terrain_plot_output=None,
            color_map_output=None,
        )
        calmet_changes: dict[str, Any] = {
            "meteorology": meteorology,
            "geophysical": geophysical,
            "output": final_output,
            "distance_to_coast_output": None,
            "listing_output": None,
            "print_fields": False,
            "micrometeorology_debug": False,
            "wind_debug_output": 0,
            "wind_debug_fields": (0,) * 9,
            "final_wind_debug_output": None,
            "kinematic_wind_debug_output": None,
            "froude_wind_debug_output": None,
            "slope_wind_debug_output": None,
            "wind_print_output": None,
            "cloud_output": 0,
        }
        if config.calmet.legacy_output_format == 2:
            calmet_changes["mesopac_output"] = final_output
        else:
            calmet_changes["mesopac_output"] = None
        calmet_configured = replace(config.calmet, **calmet_changes)

        _run_stage("CALWRF", _run_calwrf, calwrf_configured)
        _run_stage("TERREL", _run_terrel, terrel_configured)
        _run_stage("CTGPROC", _run_ctgproc, ctgproc_configured)
        _run_stage("MAKEGEO", _run_makegeo, makegeo_configured)
        return _run_stage("CALMET", _run_calmet, calmet_configured)


__all__ = ["PipelineConfig", "run_pipeline"]
