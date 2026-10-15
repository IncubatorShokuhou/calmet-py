"""Typed public API for every workflow stage."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import fields, replace
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Final, ParamSpec, TypeVar, cast

from ._parameter_docs import document_stage_api
from .config import (CalmetConfig, CalwrfConfig, CtgprocConfig, GridConfig,
                     MakegeoConfig, TerrelConfig, CtgprocSource, TerrelSource)
from .parameters import normalize_public_parameters

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
_GRID_PARAMETER_NAMES: Final[set[str]] = {
    field.name for field in fields(GridConfig)
}
_EXPLICIT_ARGUMENTS: ContextVar[frozenset[str] | None] = ContextVar(
    "pycalmet_explicit_arguments", default=None)


def _capture_explicit_arguments(function: Callable[P, R]) -> Callable[P, R]:
    """Preserve which keyword arguments a caller supplied explicitly.

    Args:
        function: Public API callable whose keyword use must be captured.

    Returns:
        A signature-preserving wrapper that exposes the supplied keyword names
        through the call-local context variable.
    """
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call the API function with explicit-keyword tracking enabled.

        Args:
            *args: Positional arguments forwarded unchanged.
            **kwargs: Keyword arguments forwarded unchanged and recorded for
                configuration-source selection.

        Returns:
            The wrapped API result.
        """
        token=_EXPLICIT_ARGUMENTS.set(frozenset(kwargs))
        try:
            return function(*args,**kwargs)
        finally:
            _EXPLICIT_ARGUMENTS.reset(token)
    return wrapped


def _source_control_overrides(cls: type[T], result: T,
                              special: Mapping[str, Any]) -> T:
    """Apply source-specific datum and mesh controls parsed from legacy input.

    Args:
        cls: Configuration class identifying the workflow stage.
        result: Parsed configuration instance to update.
        special: Normalized source-specific values that do not map directly to
            configuration fields.

    Returns:
        The updated TERREL or CTGPROC configuration, or ``result`` unchanged
        for other stages.
    """
    if isinstance(result, TerrelConfig) and result.terrain_sources:
        terrain_sources=tuple(replace(source,input_datum=str(special.get(
            "source_datum."+source.input_type.upper(),source.input_datum)))
            for source in result.terrain_sources)
        return cast(T, replace(result,terrain_sources=terrain_sources))
    if isinstance(result, CtgprocConfig) and result.landuse_sources:
        def datum_key(kind: str) -> str:
            """Return the normalized legacy datum key for a source type.

            Args:
                kind: Uppercase CTGPROC land-use input type.

            Returns:
                Source-specific datum special-parameter key.
            """
            if kind.startswith("NLCD"): return "source_datum.NLCD"
            if kind.startswith("GLAZ"): return "source_datum.GLAZ"
            return "source_datum."+kind
        def mesh_key(kind: str) -> str:
            """Return the normalized legacy mesh key for a source type.

            Args:
                kind: Uppercase CTGPROC land-use input type.

            Returns:
                Source-specific or default mesh special-parameter key.
            """
            if kind=="CTG": return "mesh.CTG"
            if kind=="CORINE": return "mesh.CORINE"
            return "mesh.default"
        landuse_sources=tuple(replace(
            source,
            input_datum=str(special.get(datum_key(source.input_type.upper()),
                                        source.input_datum)),
            mesh_density=int(special.get(mesh_key(source.input_type.upper()),
                                         source.mesh_density)),
        ) for source in result.landuse_sources)
        return cast(T, replace(result,landuse_sources=landuse_sources,
                              input_type=landuse_sources[0].input_type,
                              input_datum=landuse_sources[0].input_datum,
                              mesh_density=landuse_sources[0].mesh_density))
    return result


def _merge_grid(supplied: GridConfig | Mapping[str, Any] | None,
                updates: Mapping[str, Any], base: GridConfig | None = None) -> GridConfig:
    """Combine a complete grid value with flattened grid keyword updates.

    Args:
        supplied: Explicit grid object or mapping, if one was provided.
        updates: Flattened grid fields supplied by the caller.
        base: Grid loaded from a control file and used when ``supplied`` is
            absent.

    Returns:
        A complete grid configuration containing the requested updates.

    Raises:
        TypeError: If ``supplied`` is neither a grid, mapping, nor ``None``.
    """
    if supplied is None:
        grid=base
    elif isinstance(supplied,GridConfig):
        grid=supplied
    elif isinstance(supplied,Mapping):
        grid=(GridConfig(**supplied) if base is None
              else replace(base,**supplied))
    else:
        raise TypeError("grid must be a GridConfig or mapping")
    if grid is None:
        return GridConfig(**updates)
    return replace(grid,**updates) if updates else grid


def _config(cls: type[T], value: T | Mapping[str, Any] | str | Path | None,
            kwargs: dict[str, Any], loader: Callable[...,T] | None=None) -> T:
    """Resolve one public stage input into its typed configuration object.

    Args:
        cls: Expected configuration class.
        value: Configuration object, mapping, control-file path, or ``None``.
        kwargs: Canonical and legacy keyword overrides.
        loader: Optional control-file parser for the stage.

    Returns:
        The fully normalized typed configuration.

    Raises:
        TypeError: If input modes are combined or ``value`` has an unsupported
            type.
    """
    if isinstance(value,(str,Path)):
        if loader is None: raise TypeError(f"{cls.__name__} does not support control files")
        parameters=normalize_public_parameters(cls,kwargs)
        supplied_grid=parameters.values.pop("grid",None)
        result=loader(value,**parameters.values)
        if supplied_grid is not None or parameters.grid:
            result=cast(T, replace(cast(Any, result),grid=_merge_grid(
                supplied_grid, parameters.grid,
                cast(GridConfig, getattr(result, "grid")))))
        return _source_control_overrides(cls,result,parameters.special)
    if value is not None and kwargs:
        raise TypeError("pass either a config/mapping or keyword parameters, not both")
    if isinstance(value, cls):
        return value
    source = kwargs if value is None else cast(Mapping[str, Any], value)
    parameters=normalize_public_parameters(cls,dict(source))
    supplied_grid=parameters.values.get("grid")
    if supplied_grid is not None or parameters.grid:
        parameters.values["grid"]=_merge_grid(supplied_grid,parameters.grid)
    return cls(**parameters.values)


def _changed(value: Any, default: Any) -> bool:
    """Determine whether a supplied value differs from its public default.

    Args:
        value: Candidate value supplied to the API.
        default: Declared signature default.

    Returns:
        ``True`` when the values differ or cannot be compared as scalars.
    """
    try:
        return bool(value != default)
    except (TypeError, ValueError):
        return True


def _arguments(inp: Any, values: dict[str, Any], defaults: Mapping[str, Any],
               omitted: set[str], legacy: dict[str, Any]) -> tuple[Any,dict[str,Any]]:
    """Select parameters for control-file or explicit-parameter mode.

    Args:
        inp: Configuration source passed to the stage function.
        values: Local stage arguments keyed by canonical parameter name.
        defaults: Public signature defaults for canonical parameters.
        omitted: ``None`` sentinel parameters excluded in keyword-only mode.
        legacy: Additional legacy-named keyword arguments.

    Returns:
        The input source and the parameter mapping to pass to ``_config``.
    """
    values={name:values[name] for name in defaults}
    result=dict(legacy)
    if inp is None and not legacy:
        supplied_grid = values.get("grid") is not None
        supplied=_EXPLICIT_ARGUMENTS.get()
        result.update({name:value for name,value in values.items()
                       if not (name in omitted and value is None)
                       and not (name == "grid" and value is None)
                       and not (supplied_grid and name in _GRID_PARAMETER_NAMES
                                and (name not in supplied if supplied is not None
                                     else not _changed(value,defaults[name])))})
    else:
        supplied=_EXPLICIT_ARGUMENTS.get()
        result.update({name:value for name,value in values.items()
                       if (name in supplied if supplied is not None
                           else _changed(value,defaults[name]))})
    return inp,result


@_capture_explicit_arguments
@document_stage_api("calwrf")
def calwrf(
    inp: CalwrfConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    wrf_files: Sequence[str | Path] | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    output: str | Path | None = None,
    create_surface: bool = True,
    compression: int = 1,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
    k_range: tuple[int, int] | None = None,
    legacy_output: str | Path | None = None,
    surface_output: str | Path | None = None,
    title: str = 'Create 3D.DAT file for WRF output',
    listing_output: str | Path | None = None,
    **legacy_parameters: Any,
) -> Path:
    """Run CALWRF from ``inp`` or fully typed keyword parameters.

    Returns:
        Path to the canonical CALWRF NetCDF meteorological file.
    """
    from .calwrf import run
    from .control import calwrf_config
    source,parameters=_arguments(inp,locals(),{'wrf_files': None, 'start': None, 'end': None, 'output': None, 'create_surface': True, 'compression': 1, 'i_range': None, 'j_range': None, 'k_range': None, 'legacy_output': None, 'surface_output': None, 'title': 'Create 3D.DAT file for WRF output', 'listing_output': None},{'wrf_files', 'start', 'end', 'output'},legacy_parameters)
    return run(_config(CalwrfConfig,source,parameters,calwrf_config))


@_capture_explicit_arguments
@document_stage_api("terrel")
def terrel(
    inp: TerrelConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    srtm_dir: str | Path | None = None,
    output: str | Path | None = None,
    datum: str = 'WGS-84',
    sampling: str = 'bilinear',
    terrain_files: Sequence[str | Path] = (),
    terrain_sources: Sequence[TerrelSource] = (),
    void_fill: bool = False,
    cell_radius_km: float = 0.0,
    model: int = 1,
    processing_method: int | None = None,
    receptor_input: str | Path | None = None,
    receptor_output: str | Path | None = None,
    receptor_columns: int = 4,
    interpolate_receptors: bool = False,
    receptor_radius_km: float = 0.0,
    previous_save: str | Path | None = None,
    save_output: str | Path | None = None,
    coastline_file: str | Path | None = None,
    gshhs_file: str | Path | None = None,
    coastline_source_datum: str = 'WGS-84',
    coastline_wdbii_datum: str = 'WGS-72',
    noise_replacement: Sequence[int] = (0, 0, 0, 0, 0),
    noise_minimum_m: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 1.0),
    missing_replacement: Sequence[int] = (3, 0, 0, 0, 0),
    default_elevation_m: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    plot_output: str | Path | None = None,
    raw_echo_output: str | Path | None = None,
    qa_hit_threshold_percent: int = 75,
    grid: GridConfig | Mapping[str, Any] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    dx_km: float | None = None,
    dy_km: float | None = None,
    projection: str = 'UTM',
    lat0: float = 27.0,
    lon0: float = 117.5,
    lat1: float = 30.0,
    lat2: float = 60.0,
    x_origin_km: float = 0.0,
    y_origin_km: float = 0.0,
    false_easting_km: float = 0.0,
    false_northing_km: float = 0.0,
    utm_zone: int = 50,
    hemisphere: str = 'N',
    grid_type: int = 1,
    ring_distances_km: Sequence[float] = (),
    ray_angles_deg: Sequence[float] = (),
    polar_processing: int = 2,
    ttm_scale: float = 1.0,
    **legacy_parameters: Any,
) -> Path:
    """Run TERREL from ``inp`` or fully typed keyword parameters.

    Returns:
        Path to the generated terrain file.
    """
    from .terrel import run
    from .control import terrel_config
    source,parameters=_arguments(inp,locals(),{'srtm_dir': None, 'output': None, 'datum': 'WGS-84', 'sampling': 'bilinear', 'terrain_files': (), 'terrain_sources': (), 'void_fill': False, 'cell_radius_km': 0.0, 'model': 1, 'processing_method': None, 'receptor_input': None, 'receptor_output': None, 'receptor_columns': 4, 'interpolate_receptors': False, 'receptor_radius_km': 0.0, 'previous_save': None, 'save_output': None, 'coastline_file': None, 'gshhs_file': None, 'coastline_source_datum': 'WGS-84', 'coastline_wdbii_datum': 'WGS-72', 'noise_replacement': (0, 0, 0, 0, 0), 'noise_minimum_m': (0.0, 1.0, 1.0, 1.0, 1.0), 'missing_replacement': (3, 0, 0, 0, 0), 'default_elevation_m': (0.0, 0.0, 0.0, 0.0, 0.0), 'plot_output': None, 'raw_echo_output': None, 'qa_hit_threshold_percent': 75, 'grid': None, 'nx': None, 'ny': None, 'dx_km': None, 'dy_km': None, 'projection': 'UTM', 'lat0': 27.0, 'lon0': 117.5, 'lat1': 30.0, 'lat2': 60.0, 'x_origin_km': 0.0, 'y_origin_km': 0.0, 'false_easting_km': 0.0, 'false_northing_km': 0.0, 'utm_zone': 50, 'hemisphere': 'N', 'grid_type': 1, 'ring_distances_km': (), 'ray_angles_deg': (), 'polar_processing': 2, 'ttm_scale': 1.0},{'srtm_dir', 'output', 'nx', 'ny', 'dx_km', 'dy_km'},legacy_parameters)
    return run(_config(TerrelConfig,source,parameters,terrel_config))


@_capture_explicit_arguments
@document_stage_api("ctrproc")
def ctrproc(
    inp: CtgprocConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    landuse_files: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
    category_map: dict[int, int] | None = None,
    input_type: str = 'CTG',
    mesh_density: int = 1,
    input_datum: str = 'NAS-C',
    output_format: str = 'auto',
    coastline_file: str | Path | None = None,
    gshhs_file: str | Path | None = None,
    coastline_source_datum: str = 'WGS-84',
    coastline_wdbii_datum: str = 'WGS-72',
    coastline_swap: bool = False,
    fill_empty_ocean: bool | None = None,
    ocean_category: int = 55,
    previous_landuse: str | Path | None = None,
    final_output: bool = True,
    landuse_sources: Sequence[CtgprocSource] = (),
    snow_processing: bool = False,
    qa_cell_output: bool = False,
    qa_hit_threshold_percent: int = 75,
    qa_output_directory: str | Path | None = None,
    grid: GridConfig | Mapping[str, Any] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    dx_km: float | None = None,
    dy_km: float | None = None,
    projection: str = 'UTM',
    lat0: float = 27.0,
    lon0: float = 117.5,
    lat1: float = 30.0,
    lat2: float = 60.0,
    x_origin_km: float = 0.0,
    y_origin_km: float = 0.0,
    false_easting_km: float = 0.0,
    false_northing_km: float = 0.0,
    utm_zone: int = 50,
    hemisphere: str = 'N',
    datum: str = 'WGS-84',
    grid_type: int = 1,
    ring_distances_km: Sequence[float] = (),
    ray_angles_deg: Sequence[float] = (),
    polar_processing: int = 2,
    ttm_scale: float = 1.0,
    **legacy_parameters: Any,
) -> Path:
    """Run land-use preprocessing using the public CTRPROC spelling.

    Returns:
        Path to the generated land-use file.
    """
    from .ctrproc import run
    from .control import ctgproc_config
    source,parameters=_arguments(inp,locals(),{'landuse_files': None, 'output': None, 'category_map': None, 'input_type': 'CTG', 'mesh_density': 1, 'input_datum': 'NAS-C', 'output_format': 'auto', 'coastline_file': None, 'gshhs_file': None, 'coastline_source_datum': 'WGS-84', 'coastline_wdbii_datum': 'WGS-72', 'coastline_swap': False, 'fill_empty_ocean': None, 'ocean_category': 55, 'previous_landuse': None, 'final_output': True, 'landuse_sources': (), 'snow_processing': False, 'qa_cell_output': False, 'qa_hit_threshold_percent': 75, 'qa_output_directory': None, 'grid': None, 'nx': None, 'ny': None, 'dx_km': None, 'dy_km': None, 'projection': 'UTM', 'lat0': 27.0, 'lon0': 117.5, 'lat1': 30.0, 'lat2': 60.0, 'x_origin_km': 0.0, 'y_origin_km': 0.0, 'false_easting_km': 0.0, 'false_northing_km': 0.0, 'utm_zone': 50, 'hemisphere': 'N', 'datum': 'WGS-84', 'grid_type': 1, 'ring_distances_km': (), 'ray_angles_deg': (), 'polar_processing': 2, 'ttm_scale': 1.0},{'landuse_files', 'output', 'category_map', 'nx', 'ny', 'dx_km', 'dy_km'},legacy_parameters)
    return run(_config(CtgprocConfig,source,parameters,ctgproc_config))


@_capture_explicit_arguments
@document_stage_api("ctgproc")
def ctgproc(
    inp: CtgprocConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    landuse_files: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
    category_map: dict[int, int] | None = None,
    input_type: str = 'CTG',
    mesh_density: int = 1,
    input_datum: str = 'NAS-C',
    output_format: str = 'auto',
    coastline_file: str | Path | None = None,
    gshhs_file: str | Path | None = None,
    coastline_source_datum: str = 'WGS-84',
    coastline_wdbii_datum: str = 'WGS-72',
    coastline_swap: bool = False,
    fill_empty_ocean: bool | None = None,
    ocean_category: int = 55,
    previous_landuse: str | Path | None = None,
    final_output: bool = True,
    landuse_sources: Sequence[CtgprocSource] = (),
    snow_processing: bool = False,
    qa_cell_output: bool = False,
    qa_hit_threshold_percent: int = 75,
    qa_output_directory: str | Path | None = None,
    grid: GridConfig | Mapping[str, Any] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    dx_km: float | None = None,
    dy_km: float | None = None,
    projection: str = 'UTM',
    lat0: float = 27.0,
    lon0: float = 117.5,
    lat1: float = 30.0,
    lat2: float = 60.0,
    x_origin_km: float = 0.0,
    y_origin_km: float = 0.0,
    false_easting_km: float = 0.0,
    false_northing_km: float = 0.0,
    utm_zone: int = 50,
    hemisphere: str = 'N',
    datum: str = 'WGS-84',
    grid_type: int = 1,
    ring_distances_km: Sequence[float] = (),
    ray_angles_deg: Sequence[float] = (),
    polar_processing: int = 2,
    ttm_scale: float = 1.0,
    **legacy_parameters: Any,
) -> Path:
    """Run land-use preprocessing using the CTGPROC program spelling.

    Returns:
        Path to the generated land-use file.
    """
    from .ctgproc import run
    from .control import ctgproc_config
    source,parameters=_arguments(inp,locals(),{'landuse_files': None, 'output': None, 'category_map': None, 'input_type': 'CTG', 'mesh_density': 1, 'input_datum': 'NAS-C', 'output_format': 'auto', 'coastline_file': None, 'gshhs_file': None, 'coastline_source_datum': 'WGS-84', 'coastline_wdbii_datum': 'WGS-72', 'coastline_swap': False, 'fill_empty_ocean': None, 'ocean_category': 55, 'previous_landuse': None, 'final_output': True, 'landuse_sources': (), 'snow_processing': False, 'qa_cell_output': False, 'qa_hit_threshold_percent': 75, 'qa_output_directory': None, 'grid': None, 'nx': None, 'ny': None, 'dx_km': None, 'dy_km': None, 'projection': 'UTM', 'lat0': 27.0, 'lon0': 117.5, 'lat1': 30.0, 'lat2': 60.0, 'x_origin_km': 0.0, 'y_origin_km': 0.0, 'false_easting_km': 0.0, 'false_northing_km': 0.0, 'utm_zone': 50, 'hemisphere': 'N', 'datum': 'WGS-84', 'grid_type': 1, 'ring_distances_km': (), 'ray_angles_deg': (), 'polar_processing': 2, 'ttm_scale': 1.0},{'landuse_files', 'output', 'category_map', 'nx', 'ny', 'dx_km', 'dy_km'},legacy_parameters)
    return run(_config(CtgprocConfig,source,parameters,ctgproc_config))


@_capture_explicit_arguments
@document_stage_api("makegeo")
def makegeo(
    inp: MakegeoConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    terrain: str | Path | None = None,
    landuse: str | Path | None = None,
    output: str | Path | None = None,
    water_threshold: float = 0.5,
    missing_category: int = 55,
    minimum_coverage: float = 0.96,
    secondary_landuse: str | Path | None = None,
    use_terrain: bool = True,
    input_properties: dict[int, tuple[float, float, float, float, float, float, int]] | None = None,
    water_categories: Sequence[int] = (51, 52, 53, 54, 55),
    output_categories: Sequence[int] = (10, 20, -20, 30, 40, 51, 54, 55, 60, 61, 62, 70, 80, 90),
    output_water_min: int = 50,
    output_water_max: int = 55,
    category_splits: Sequence[tuple[int, int, float]] = (),
    qa_cell: tuple[int, int] | None = None,
    snow_processing: bool = False,
    snow_data: str | Path | None = None,
    snow_start_date: date | datetime | str | None = None,
    snow_end_date: date | datetime | str | None = None,
    snow_properties: dict[int, tuple[float, float, float, float, float, float, int]] | None = None,
    snow_age_parameters: dict[int, tuple[int, float]] | None = None,
    snow_roughness_method: int = 1,
    snow_height_scale_m: float = 10.0,
    snow_albedo_method: int = 1,
    snow_depth_minimum_m: float = 0.01,
    landuse_plot_output: str | Path | None = None,
    terrain_plot_output: str | Path | None = None,
    color_map_output: str | Path | None = None,
    image_mode: int = 0,
    title: str = 'PyCALMET Application',
    grid: GridConfig | Mapping[str, Any] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    dx_km: float | None = None,
    dy_km: float | None = None,
    projection: str = 'UTM',
    lat0: float = 27.0,
    lon0: float = 117.5,
    lat1: float = 30.0,
    lat2: float = 60.0,
    x_origin_km: float = 0.0,
    y_origin_km: float = 0.0,
    false_easting_km: float = 0.0,
    false_northing_km: float = 0.0,
    utm_zone: int = 50,
    hemisphere: str = 'N',
    datum: str = 'WGS-84',
    grid_type: int = 1,
    ring_distances_km: Sequence[float] = (),
    ray_angles_deg: Sequence[float] = (),
    polar_processing: int = 2,
    ttm_scale: float = 1.0,
    **legacy_parameters: Any,
) -> Path:
    """Run MAKEGEO from ``inp`` or fully typed keyword parameters.

    Returns:
        Path to the generated geophysical file.
    """
    from .makegeo import run
    from .control import makegeo_config
    source,parameters=_arguments(inp,locals(),{'terrain': None, 'landuse': None, 'output': None, 'water_threshold': 0.5, 'missing_category': 55, 'minimum_coverage': 0.96, 'secondary_landuse': None, 'use_terrain': True, 'input_properties': None, 'water_categories': (51, 52, 53, 54, 55), 'output_categories': (10, 20, -20, 30, 40, 51, 54, 55, 60, 61, 62, 70, 80, 90), 'output_water_min': 50, 'output_water_max': 55, 'category_splits': (), 'qa_cell': None, 'snow_processing': False, 'snow_data': None, 'snow_start_date': None, 'snow_end_date': None, 'snow_properties': None, 'snow_age_parameters': None, 'snow_roughness_method': 1, 'snow_height_scale_m': 10.0, 'snow_albedo_method': 1, 'snow_depth_minimum_m': 0.01, 'landuse_plot_output': None, 'terrain_plot_output': None, 'color_map_output': None, 'image_mode': 0, 'title': 'PyCALMET Application', 'grid': None, 'nx': None, 'ny': None, 'dx_km': None, 'dy_km': None, 'projection': 'UTM', 'lat0': 27.0, 'lon0': 117.5, 'lat1': 30.0, 'lat2': 60.0, 'x_origin_km': 0.0, 'y_origin_km': 0.0, 'false_easting_km': 0.0, 'false_northing_km': 0.0, 'utm_zone': 50, 'hemisphere': 'N', 'datum': 'WGS-84', 'grid_type': 1, 'ring_distances_km': (), 'ray_angles_deg': (), 'polar_processing': 2, 'ttm_scale': 1.0},{'terrain', 'landuse', 'output', 'input_properties', 'snow_properties', 'snow_age_parameters', 'nx', 'ny', 'dx_km', 'dy_km'},legacy_parameters)
    return run(_config(MakegeoConfig,source,parameters,makegeo_config))


@_capture_explicit_arguments
@document_stage_api("calmet")
def calmet(
    inp: CalmetConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    meteorology: str | Path | Sequence[str | Path] | None = None,
    geophysical: str | Path | None = None,
    output: str | Path | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    noobs: int = 0,
    run_type: int = 1,
    save_output: bool = True,
    legacy_output_format: int = 1,
    test_mode: int = 2,
    regulatory_mode: int = 0,
    timestep_seconds: int = 3600,
    prognostic_timestep_seconds: int = 3600,
    cloud_method: int = 4,
    cloud_condensate_threshold_g_kg: float = 0.005,
    cloud_data: str | Path | None = None,
    cloud_data_format: int = 2,
    cloud_output: int = 0,
    surface_data: str | Path | None = None,
    surface_data_format: int = 2,
    surface_control_stations: Sequence[tuple[str, int, float, float, float, float]] = (),
    upper_air_files: Sequence[str | Path] = (),
    upper_control_stations: Sequence[tuple[str, int, float, float, float]] = (),
    sea_files: Sequence[str | Path] = (),
    diagnostic_data: str | Path | None = None,
    diagnostic_surface_temperature: int = 0,
    diagnostic_lapse_rate: int = 0,
    diagnostic_initial_wind: int = 0,
    diagnostic_surface_wind: int = 0,
    diagnostic_upper_wind: int = 0,
    initial_guess_upper_station: int = -1,
    initial_guess_height_range_m: Sequence[float] = (1.0, 1000.0),
    use_initial_guess_calmet: int = 0,
    initial_guess_calmet_files: Sequence[str | Path] = (),
    weight_data: str | Path | None = None,
    precipitation_station_count: int = 0,
    precipitation_data: str | Path | None = None,
    precipitation_data_format: int = 2,
    precipitation_control_stations: Sequence[tuple[str, int, float, float]] = (),
    iprog: int = 0,
    irhprog: int = 0,
    itprog: int = 0,
    imixh: int = 1,
    constb: float = 1.41,
    conste: float = 0.15,
    constn: float = 2400.0,
    constw: float = 0.16,
    coriolis: float = 999.0,
    minimum_lapse_rate: float = 0.001,
    mixing_lapse_depth_m: float = 200.0,
    minimum_mixing_height_m: float = 50.0,
    maximum_mixing_height_m: float = 3000.0,
    minimum_water_mixing_height_m: float = 50.0,
    maximum_water_mixing_height_m: float = 3000.0,
    land_flux_threshold: float = 0.0,
    water_flux_threshold: float = 0.05,
    water_temperature_mode: int = 0,
    prognostic_ocean_category: int = 16,
    overwater_flux_method: int = 10,
    coastal_shelf_length_km: float = 0.0,
    warm_layer: int = 0,
    cool_skin: int = 0,
    convective_relaxation: int = 1,
    convective_relaxation_time_s: float = 800.0,
    precipitation_interpolation: int = 2,
    precipitation_radius_km: float = 100.0,
    precipitation_cutoff: float = 0.01,
    turbidity_coefficient_1: float = 990.0,
    turbidity_coefficient_2: float = -30.0,
    cloudiness_coefficient_1: float = -0.75,
    cloudiness_coefficient_2: float = 3.4,
    net_radiation_coefficient_1: float = 5.31e-13,
    net_radiation_coefficient_2: float = 60.0,
    net_radiation_coefficient_3: float = 0.12,
    surface_temperature_option: int = -1,
    upper_temperature_option: int = -1,
    lapse_height_m: float = 200.0,
    temperature_interpolation_power: int = 1,
    temperature_radius_km: float = 500.0,
    temperature_station_limit: int = 5,
    water_temperature_gradient_below: float = -0.0098,
    water_temperature_gradient_above: float = -0.0045,
    water_temperature_category_min: int = 999,
    water_temperature_category_max: int = 999,
    iavet: int = 1,
    iavezi: int = 1,
    mixing_wind_level: int = 1,
    mnmdav: int = 1,
    iwfcod: int = 1,
    wind_extrapolation: int = -4,
    upper_air_separation_cells: float = 4.0,
    extrapolate_calm: int = 0,
    wind_bias: Sequence[float] | None = None,
    varying_radius: bool = False,
    minimum_station_radius_km: float = 0.1,
    maximum_surface_radius_km: float = 30.0,
    maximum_upper_radius_km: float = 30.0,
    maximum_water_radius_km: float = 50.0,
    surface_background_radius_km: float = 1.0,
    upper_background_radius_km: float = 1.0,
    prognostic_radius_km: float = 0.0,
    stations_per_level: Sequence[int] | None = None,
    wind_extrapolation_factors: Sequence[float] | None = None,
    barrier_count: int = 0,
    barrier_top_level: int | None = None,
    barriers: Sequence[tuple[float, float, float, float]] = (),
    lake_breeze_regions: Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]] = (),
    ifradj: int = 1,
    ikine: int = 0,
    iobr: int = 0,
    islope: int = 1,
    critical_froude: float = 1.0,
    smoothing_passes: Sequence[int] | None = None,
    divergence_limit: float = 5e-06,
    divergence_iterations: int = 50,
    kinematic_alpha: float = 0.1,
    z_faces_m: Sequence[float] = (0.0, 20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0),
    base_timezone: int = 0,
    base_timezone_label: str | None = None,
    terrain_radius_km: float = 15.0,
    half_average_angle: float = 30.0,
    include_w_temperature: bool = True,
    compression: int = 1,
    control_lines: Sequence[str] = (),
    mesopac_output: str | Path | None = None,
    distance_to_coast_output: str | Path | None = None,
    listing_output: str | Path | None = None,
    print_fields: bool = False,
    print_interval_hours: int = 1,
    print_uv_levels: Sequence[int] = (),
    print_w_levels: Sequence[int] = (),
    print_temperature_levels: Sequence[int] = (),
    print_surface_fields: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0),
    micrometeorology_debug: bool = False,
    debug_first_step: int = 1,
    debug_last_step: int = 1,
    wind_debug_output: int = 0,
    wind_debug_levels: int = 1,
    wind_debug_fields: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0, 0),
    final_wind_debug_output: str | Path | None = None,
    kinematic_wind_debug_output: str | Path | None = None,
    froude_wind_debug_output: str | Path | None = None,
    slope_wind_debug_output: str | Path | None = None,
    wind_print_output: str | Path | None = None,
    grid: GridConfig | Mapping[str, Any] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    dx_km: float | None = None,
    dy_km: float | None = None,
    projection: str = 'UTM',
    lat0: float = 27.0,
    lon0: float = 117.5,
    lat1: float = 30.0,
    lat2: float = 60.0,
    x_origin_km: float = 0.0,
    y_origin_km: float = 0.0,
    false_easting_km: float = 0.0,
    false_northing_km: float = 0.0,
    utm_zone: int = 50,
    hemisphere: str = 'N',
    datum: str = 'WGS-84',
    grid_type: int = 1,
    ring_distances_km: Sequence[float] = (),
    ray_angles_deg: Sequence[float] = (),
    polar_processing: int = 2,
    ttm_scale: float = 1.0,
    **legacy_parameters: Any,
) -> Path:
    """Run CALMET from ``inp`` or fully typed keyword parameters.

    Returns:
        Path to the final CALMET meteorological output.
    """
    from .calmet import run
    from .control import calmet_config
    source,parameters=_arguments(inp,locals(),{'meteorology': None, 'geophysical': None, 'output': None, 'start': None, 'end': None, 'noobs': 0, 'run_type': 1, 'save_output': True, 'legacy_output_format': 1, 'test_mode': 2, 'regulatory_mode': 0, 'timestep_seconds': 3600, 'prognostic_timestep_seconds': 3600, 'cloud_method': 4, 'cloud_condensate_threshold_g_kg': 0.005, 'cloud_data': None, 'cloud_data_format': 2, 'cloud_output': 0, 'surface_data': None, 'surface_data_format': 2, 'surface_control_stations': (), 'upper_air_files': (), 'upper_control_stations': (), 'sea_files': (), 'diagnostic_data': None, 'diagnostic_surface_temperature': 0, 'diagnostic_lapse_rate': 0, 'diagnostic_initial_wind': 0, 'diagnostic_surface_wind': 0, 'diagnostic_upper_wind': 0, 'initial_guess_upper_station': -1, 'initial_guess_height_range_m': (1.0, 1000.0), 'use_initial_guess_calmet': 0, 'initial_guess_calmet_files': (), 'weight_data': None, 'precipitation_station_count': 0, 'precipitation_data': None, 'precipitation_data_format': 2, 'precipitation_control_stations': (), 'iprog': 0, 'irhprog': 0, 'itprog': 0, 'imixh': 1, 'constb': 1.41, 'conste': 0.15, 'constn': 2400.0, 'constw': 0.16, 'coriolis': 999.0, 'minimum_lapse_rate': 0.001, 'mixing_lapse_depth_m': 200.0, 'minimum_mixing_height_m': 50.0, 'maximum_mixing_height_m': 3000.0, 'minimum_water_mixing_height_m': 50.0, 'maximum_water_mixing_height_m': 3000.0, 'land_flux_threshold': 0.0, 'water_flux_threshold': 0.05, 'water_temperature_mode': 0, 'prognostic_ocean_category': 16, 'overwater_flux_method': 10, 'coastal_shelf_length_km': 0.0, 'warm_layer': 0, 'cool_skin': 0, 'convective_relaxation': 1, 'convective_relaxation_time_s': 800.0, 'precipitation_interpolation': 2, 'precipitation_radius_km': 100.0, 'precipitation_cutoff': 0.01, 'turbidity_coefficient_1': 990.0, 'turbidity_coefficient_2': -30.0, 'cloudiness_coefficient_1': -0.75, 'cloudiness_coefficient_2': 3.4, 'net_radiation_coefficient_1': 5.31e-13, 'net_radiation_coefficient_2': 60.0, 'net_radiation_coefficient_3': 0.12, 'surface_temperature_option': -1, 'upper_temperature_option': -1, 'lapse_height_m': 200.0, 'temperature_interpolation_power': 1, 'temperature_radius_km': 500.0, 'temperature_station_limit': 5, 'water_temperature_gradient_below': -0.0098, 'water_temperature_gradient_above': -0.0045, 'water_temperature_category_min': 999, 'water_temperature_category_max': 999, 'iavet': 1, 'iavezi': 1, 'mixing_wind_level': 1, 'mnmdav': 1, 'iwfcod': 1, 'wind_extrapolation': -4, 'upper_air_separation_cells': 4.0, 'extrapolate_calm': 0, 'wind_bias': None, 'varying_radius': False, 'minimum_station_radius_km': 0.1, 'maximum_surface_radius_km': 30.0, 'maximum_upper_radius_km': 30.0, 'maximum_water_radius_km': 50.0, 'surface_background_radius_km': 1.0, 'upper_background_radius_km': 1.0, 'prognostic_radius_km': 0.0, 'stations_per_level': None, 'wind_extrapolation_factors': None, 'barrier_count': 0, 'barrier_top_level': None, 'barriers': (), 'lake_breeze_regions': (), 'ifradj': 1, 'ikine': 0, 'iobr': 0, 'islope': 1, 'critical_froude': 1.0, 'smoothing_passes': None, 'divergence_limit': 5e-06, 'divergence_iterations': 50, 'kinematic_alpha': 0.1, 'z_faces_m': (0.0, 20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0), 'base_timezone': 0, 'base_timezone_label': None, 'terrain_radius_km': 15.0, 'half_average_angle': 30.0, 'include_w_temperature': True, 'compression': 1, 'control_lines': (), 'mesopac_output': None, 'distance_to_coast_output': None, 'listing_output': None, 'print_fields': False, 'print_interval_hours': 1, 'print_uv_levels': (), 'print_w_levels': (), 'print_temperature_levels': (), 'print_surface_fields': (0, 0, 0, 0, 0, 0, 0, 0), 'micrometeorology_debug': False, 'debug_first_step': 1, 'debug_last_step': 1, 'wind_debug_output': 0, 'wind_debug_levels': 1, 'wind_debug_fields': (0, 0, 0, 0, 0, 0, 0, 0, 0), 'final_wind_debug_output': None, 'kinematic_wind_debug_output': None, 'froude_wind_debug_output': None, 'slope_wind_debug_output': None, 'wind_print_output': None, 'grid': None, 'nx': None, 'ny': None, 'dx_km': None, 'dy_km': None, 'projection': 'UTM', 'lat0': 27.0, 'lon0': 117.5, 'lat1': 30.0, 'lat2': 60.0, 'x_origin_km': 0.0, 'y_origin_km': 0.0, 'false_easting_km': 0.0, 'false_northing_km': 0.0, 'utm_zone': 50, 'hemisphere': 'N', 'datum': 'WGS-84', 'grid_type': 1, 'ring_distances_km': (), 'ray_angles_deg': (), 'polar_processing': 2, 'ttm_scale': 1.0},{'geophysical', 'output', 'start', 'end', 'nx', 'ny', 'dx_km', 'dy_km'},legacy_parameters)
    return run(_config(CalmetConfig,source,parameters,calmet_config))
