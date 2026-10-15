from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import Any

import pycalmet

from pycalmet.control import (
    calmet_config,
    calwrf_config,
    ctgproc_config,
    makegeo_config,
    terrel_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "zhejiang"
CONFIG = EXAMPLE / "config"
SCRIPT = EXAMPLE / "run.sh"
PIPELINE_SCRIPT = EXAMPLE / "run.py"
STAGES_SCRIPT = EXAMPLE / "run_stages.py"


def test_example_controls_form_a_complete_source_to_output_chain():
    calwrf = calwrf_config(CONFIG / "calwrf.inp")
    terrain = terrel_config(CONFIG / "terrel.inp")
    landuse = ctgproc_config(CONFIG / "ctgproc.inp")
    geo = makegeo_config(CONFIG / "makegeo.inp")
    calmet = calmet_config(CONFIG / "calmet.inp")

    assert len(calwrf.wrf_files) == 2
    assert all(Path(path).is_file() for path in calwrf.wrf_files)
    assert calwrf.start == calmet.start
    assert calwrf.end == calmet.end
    assert len(terrain.terrain_files) == 42
    assert all(Path(path).is_file() for path in terrain.terrain_files)
    assert landuse.input_type == "GLAZAS"
    assert landuse.input_datum == "ESR-S"
    assert landuse.mesh_density == 12
    active_landuse = ROOT / "data" / "landuse" / "usgs-glcc" / "EAUSGS1.IMG"
    archived_europe_landuse = (
        ROOT / "data" / "landuse" / "usgs-glcc" / "EAUSGS2.IMG"
    )
    assert tuple(Path(path).resolve() for path in landuse.landuse_files) == (
        active_landuse,
    )
    assert [Path(path).stat().st_size for path in landuse.landuse_files] == [
        156_000_000
    ]
    assert archived_europe_landuse.stat().st_size == 169_000_000
    assert geo.terrain == terrain.output
    assert geo.landuse == landuse.output
    assert calmet.geophysical == geo.output
    assert calmet.meteorology == calwrf.output


def test_runner_calls_parameter_only_pipeline_once(tmp_path):
    capture = tmp_path / "calls.jsonl"
    patch_dir = tmp_path / "python-patch"
    patch_dir.mkdir()
    (patch_dir / "sitecustomize.py").write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path
            import pycalmet

            def run_pipeline(*args, **kwargs):
                grid = kwargs["calmet_grid"]
                with Path(os.environ["PYCALMET_TEST_CAPTURE"]).open("a") as stream:
                    stream.write(json.dumps({
                        "args": [str(item) for item in args],
                        "inp": kwargs["inp"],
                        "keys": sorted(kwargs),
                        "wrf_files": [
                            str(path) for path in kwargs["calwrf_wrf_files"]
                        ],
                        "wrf_start": kwargs["calwrf_start"].isoformat(),
                        "wrf_end": kwargs["calwrf_end"].isoformat(),
                        "terrain_sources": [{
                            "path": str(source.path),
                            "input_type": source.input_type,
                            "input_datum": source.input_datum,
                        } for source in kwargs["terrel_terrain_sources"]],
                        "landuse_files": [
                            str(path) for path in kwargs["ctgproc_landuse_files"]
                        ],
                        "grid": {
                            "nx": grid.nx,
                            "ny": grid.ny,
                            "dx_km": grid.dx_km,
                            "dy_km": grid.dy_km,
                            "projection": grid.projection,
                            "datum": grid.datum,
                        },
                        "shared_grid": all(
                            kwargs[name] is grid for name in (
                                "terrel_grid",
                                "ctgproc_grid",
                                "makegeo_grid",
                                "calmet_grid",
                            )
                        ),
                        "stage_controls": {
                            "calwrf_create_surface": kwargs[
                                "calwrf_create_surface"
                            ],
                            "calwrf_title": kwargs["calwrf_title"],
                            "terrel_srtm_dir": str(kwargs["terrel_srtm_dir"]),
                            "terrel_sampling": kwargs["terrel_sampling"],
                            "ctgproc_input_type": kwargs[
                                "ctgproc_input_type"
                            ],
                            "ctgproc_input_datum": kwargs[
                                "ctgproc_input_datum"
                            ],
                            "ctgproc_mesh_density": kwargs[
                                "ctgproc_mesh_density"
                            ],
                            "makegeo_title": kwargs["makegeo_title"],
                        },
                        "calmet_end": kwargs["calmet_end"].isoformat(),
                        "calmet_controls": {
                            name: kwargs[name] for name in (
                                "calmet_noobs",
                                "calmet_iprog",
                                "calmet_irhprog",
                                "calmet_itprog",
                                "calmet_coriolis",
                                "calmet_maximum_mixing_height_m",
                                "calmet_maximum_water_mixing_height_m",
                                "calmet_precipitation_radius_km",
                                "calmet_mnmdav",
                                "calmet_wind_extrapolation",
                                "calmet_precipitation_station_count",
                                "calmet_include_w_temperature",
                            )
                        },
                        "output": str(kwargs["output"]),
                    }) + "\\n")
                return Path(kwargs["output"])

            pycalmet.run_pipeline = run_pipeline
            """
        ),
        encoding="ascii",
    )
    environment = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "PYTHONPATH": str(patch_dir),
        "PYCALMET_TEST_CAPTURE": str(capture),
    }
    result = subprocess.run(
        (SCRIPT,),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == []
    assert call["inp"] is None
    assert ".inp" not in PIPELINE_SCRIPT.read_text(encoding="utf-8")
    assert set(call["keys"]) == {
        "inp",
        "calwrf_wrf_files",
        "calwrf_start",
        "calwrf_end",
        "calwrf_create_surface",
        "calwrf_title",
        "terrel_srtm_dir",
        "terrel_sampling",
        "terrel_terrain_sources",
        "terrel_grid",
        "ctgproc_landuse_files",
        "ctgproc_input_type",
        "ctgproc_input_datum",
        "ctgproc_mesh_density",
        "ctgproc_grid",
        "makegeo_title",
        "makegeo_grid",
        "calmet_start",
        "calmet_end",
        "calmet_grid",
        "calmet_noobs",
        "calmet_iprog",
        "calmet_irhprog",
        "calmet_itprog",
        "calmet_coriolis",
        "calmet_maximum_mixing_height_m",
        "calmet_maximum_water_mixing_height_m",
        "calmet_precipitation_radius_km",
        "calmet_mnmdav",
        "calmet_wind_extrapolation",
        "calmet_precipitation_station_count",
        "calmet_include_w_temperature",
        "output",
    }
    assert call["wrf_files"] == [
        str(EXAMPLE / f"wrfout_d01_2018-03-18_{hour:02d}_00_00.nc")
        for hour in range(2)
    ]
    assert call["wrf_start"] == "2018-03-18T00:00:00"
    assert call["wrf_end"] == "2018-03-18T01:00:00"
    assert len(call["terrain_sources"]) == 42
    assert all(
        source["input_type"] == "SRTM3"
        and source["input_datum"] == "WGS-96"
        for source in call["terrain_sources"]
    )
    assert call["landuse_files"] == [
        str(ROOT / "data" / "landuse" / "usgs-glcc" / "EAUSGS1.IMG")
    ]
    assert call["grid"] == {
        "nx": 610,
        "ny": 499,
        "dx_km": 1.0,
        "dy_km": 1.0,
        "projection": "LCC",
        "datum": "WGS-84",
    }
    assert call["shared_grid"] is True
    assert call["stage_controls"] == {
        "calwrf_create_surface": False,
        "calwrf_title": "Create CALWRF data for the Zhejiang example",
        "terrel_srtm_dir": str(ROOT / "data" / "terrain" / "srtm3"),
        "terrel_sampling": "bilinear",
        "ctgproc_input_type": "GLAZAS",
        "ctgproc_input_datum": "ESR-S",
        "ctgproc_mesh_density": 12,
        "makegeo_title": "Demo Application",
    }
    assert call["calmet_end"] == "2018-03-18T01:00:00"
    assert call["calmet_controls"] == {
        "calmet_noobs": 2,
        "calmet_iprog": 14,
        "calmet_irhprog": 1,
        "calmet_itprog": 2,
        "calmet_coriolis": 1.0e-4,
        "calmet_maximum_mixing_height_m": 2500.0,
        "calmet_maximum_water_mixing_height_m": 2500.0,
        "calmet_precipitation_radius_km": 50.0,
        "calmet_mnmdav": 10,
        "calmet_wind_extrapolation": -1,
        "calmet_precipitation_station_count": -1,
        "calmet_include_w_temperature": False,
    }
    final_output = ROOT / "output" / "zhejiang" / "calmet.nc"
    assert call["output"] == str(final_output)
    assert f"CALMET -> {final_output}" in result.stdout


def test_stage_runner_calls_each_stage_with_its_control_file(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def replacement(name: str) -> Callable[..., Path]:
        def run(*args: Any, **kwargs: Any) -> Path:
            calls.append((name, args, kwargs))
            return tmp_path / f"{name}.nc"

        return run

    for name in ("calwrf", "terrel", "ctrproc", "makegeo", "calmet"):
        monkeypatch.setattr(pycalmet, name, replacement(name))

    namespace = runpy.run_path(str(STAGES_SCRIPT))
    main = namespace["main"]
    main.__globals__["OUTPUT"] = tmp_path / "output"
    main()
    capsys.readouterr()

    assert [(name, args, kwargs) for name, args, kwargs in calls] == [
        ("calwrf", (CONFIG / "calwrf.inp",), {}),
        ("terrel", (CONFIG / "terrel.inp",), {}),
        ("ctrproc", (CONFIG / "ctgproc.inp",), {}),
        ("makegeo", (CONFIG / "makegeo.inp",), {}),
        ("calmet", (CONFIG / "calmet.inp",), {}),
    ]
