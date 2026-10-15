"""Stable public API for all workflow stages."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import (CalmetConfig, CalwrfConfig, CtgprocConfig, GridConfig,
                     MakegeoConfig, TerrelConfig)
from .parameters import normalize_public_parameters

T = TypeVar("T")


def _source_control_overrides(cls: type[T], result: T,
                              special: Mapping[str, Any]) -> T:
    """Apply source-specific legacy kwargs to sources loaded from a control.

    The normalizer can build source metadata when the file keyword is supplied
    in the same call.  With a control-file positional argument, however, the
    source paths come from the control and only become available afterwards.
    Preserve the same DSRTM3/DUSGSLA/MESHGLAZ semantics in that form too.
    """
    if cls is TerrelConfig and result.terrain_sources:
        sources=tuple(replace(source,input_datum=str(special.get(
            "source_datum."+source.input_type.upper(),source.input_datum)))
            for source in result.terrain_sources)
        return replace(result,terrain_sources=sources)
    if cls is CtgprocConfig and result.landuse_sources:
        def datum_key(kind: str) -> str:
            if kind.startswith("NLCD"): return "source_datum.NLCD"
            if kind.startswith("GLAZ"): return "source_datum.GLAZ"
            return "source_datum."+kind
        def mesh_key(kind: str) -> str:
            if kind=="CTG": return "mesh.CTG"
            if kind=="CORINE": return "mesh.CORINE"
            return "mesh.default"
        sources=tuple(replace(
            source,
            input_datum=str(special.get(datum_key(source.input_type.upper()),
                                        source.input_datum)),
            mesh_density=int(special.get(mesh_key(source.input_type.upper()),
                                         source.mesh_density)),
        ) for source in result.landuse_sources)
        return replace(result,landuse_sources=sources,
                       input_type=sources[0].input_type,
                       input_datum=sources[0].input_datum,
                       mesh_density=sources[0].mesh_density)
    return result


def _merge_grid(supplied: GridConfig | Mapping[str, Any] | None,
                updates: Mapping[str, Any], base: GridConfig | None = None) -> GridConfig:
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
    if isinstance(value,(str,Path)):
        if loader is None: raise TypeError(f"{cls.__name__} does not support control files")
        parameters=normalize_public_parameters(cls,kwargs)
        supplied_grid=parameters.values.pop("grid",None)
        result=loader(value,**parameters.values)
        if supplied_grid is not None or parameters.grid:
            result=replace(result,grid=_merge_grid(
                supplied_grid,parameters.grid,result.grid))
        return _source_control_overrides(cls,result,parameters.special)
    if value is not None and kwargs:
        raise TypeError("pass either a config/mapping or keyword parameters, not both")
    if isinstance(value, cls):
        return value
    parameters=normalize_public_parameters(cls,dict(value or kwargs))
    supplied_grid=parameters.values.get("grid")
    if supplied_grid is not None or parameters.grid:
        parameters.values["grid"]=_merge_grid(supplied_grid,parameters.grid)
    return cls(**parameters.values)


def calwrf(config: CalwrfConfig | Mapping[str, Any] | str | Path | None = None,
           **kwargs: Any) -> Path:
    from .calwrf import run
    from .control import calwrf_config
    return run(_config(CalwrfConfig, config, kwargs, calwrf_config))


def terrel(config: TerrelConfig | Mapping[str, Any] | str | Path | None = None,
           **kwargs: Any) -> Path:
    from .terrel import run
    from .control import terrel_config
    return run(_config(TerrelConfig, config, kwargs, terrel_config))


def ctrproc(config: CtgprocConfig | Mapping[str, Any] | str | Path | None = None,
            **kwargs: Any) -> Path:
    """Run land-use preprocessing using the workflow's public spelling."""
    from .ctrproc import run
    from .control import ctgproc_config
    return run(_config(CtgprocConfig, config, kwargs, ctgproc_config))


def ctgproc(config: CtgprocConfig | Mapping[str, Any] | str | Path | None = None,
            **kwargs: Any) -> Path:
    """Compatibility alias for the legacy CTGPROC program name."""
    return ctrproc(config, **kwargs)


def makegeo(config: MakegeoConfig | Mapping[str, Any] | str | Path | None = None,
            **kwargs: Any) -> Path:
    from .makegeo import run
    from .control import makegeo_config
    return run(_config(MakegeoConfig, config, kwargs, makegeo_config))


def calmet(config: CalmetConfig | Mapping[str, Any] | str | Path | None = None,
           **kwargs: Any) -> Path:
    from .calmet import run
    from .control import calmet_config
    return run(_config(CalmetConfig, config, kwargs, calmet_config))
