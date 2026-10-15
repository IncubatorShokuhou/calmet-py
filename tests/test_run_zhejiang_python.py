from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
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
STAGES_SCRIPT = EXAMPLE / "run_stages.py"


def test_example_controls_form_a_complete_source_to_output_chain():
    calwrf = calwrf_config(CONFIG / "calwrf.inp")
    terrain = terrel_config(CONFIG / "terrel.inp")
    landuse = ctgproc_config(CONFIG / "ctgproc.inp")
    geo = makegeo_config(CONFIG / "makegeo.inp")
    calmet = calmet_config(CONFIG / "calmet.inp")

    assert len(calwrf.wrf_files) == 3
    assert all(Path(path).is_file() for path in calwrf.wrf_files)
    assert len(terrain.terrain_files) == 42
    assert all(Path(path).is_file() for path in terrain.terrain_files)
    assert landuse.input_type == "GEN"
    assert all(Path(path).is_file() for path in landuse.landuse_files)
    assert geo.terrain == terrain.output
    assert geo.landuse == landuse.output
    assert calmet.geophysical == geo.output
    assert calmet.meteorology == calwrf.output


def test_runner_calls_the_complete_pipeline_once(tmp_path):
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
                with Path(os.environ["PYCALMET_TEST_CAPTURE"]).open("a") as stream:
                    stream.write(json.dumps({
                        "args": [str(item) for item in args],
                        "kwargs": {
                            name: str(value) for name, value in kwargs.items()
                        },
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
    assert calls == [
        {
            "args": [str(CONFIG)],
            "kwargs": {
                "output": str(ROOT / "output" / "zhejiang" / "calmet.nc")
            },
        }
    ]
    final_output = ROOT / "output" / "zhejiang" / "calmet.nc"
    assert f"CALMET -> {final_output}" in result.stdout


def test_stage_runner_uses_only_typed_keyword_parameters(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def replacement(name: str) -> Callable[..., Path]:
        def run(*args: Any, **kwargs: Any) -> Path:
            calls.append((name, args, kwargs))
            return Path(kwargs["output"])

        return run

    for name in ("calwrf", "terrel", "ctrproc", "makegeo", "calmet"):
        monkeypatch.setattr(pycalmet, name, replacement(name))

    source = STAGES_SCRIPT.read_text(encoding="utf-8")
    assert ".inp" not in source
    namespace = runpy.run_path(str(STAGES_SCRIPT))
    main = namespace["main"]
    main.__globals__["OUTPUT"] = tmp_path / "output"
    main()
    capsys.readouterr()

    assert [name for name, _, _ in calls] == [
        "calwrf",
        "terrel",
        "ctrproc",
        "makegeo",
        "calmet",
    ]
    assert all(not args for _, args, _ in calls)
    assert all(parameters["inp"] is None for _, _, parameters in calls)

    parameters = {name: kwargs for name, _, kwargs in calls}
    assert parameters["calwrf"]["wrf_files"] == tuple(
        EXAMPLE / f"wrfout_d01_2018-03-18_{hour:02d}_00_00.nc"
        for hour in range(3)
    )
    assert parameters["calwrf"]["start"] == datetime(2018, 3, 18, 0)
    assert parameters["calwrf"]["end"] == datetime(2018, 3, 18, 2)
    assert parameters["terrel"]["srtm_dir"] == ROOT / "terrain" / "srtm3"
    assert len(parameters["terrel"]["terrain_sources"]) == 42
    assert all(
        source.input_datum == "WGS-96"
        for source in parameters["terrel"]["terrain_sources"]
    )
    assert parameters["ctrproc"]["landuse_files"] == (
        EXAMPLE / "landuse" / "zhejiang_generic.landuse",
    )
    assert parameters["ctrproc"]["input_type"] == "GEN"
    assert parameters["ctrproc"]["mesh_density"] == 12
    assert "input_datum" not in parameters["ctrproc"]
    assert parameters["makegeo"]["terrain"] == parameters["terrel"]["output"]
    assert parameters["makegeo"]["landuse"] == parameters["ctrproc"]["output"]
    assert parameters["calmet"]["meteorology"] == parameters["calwrf"]["output"]
    assert parameters["calmet"]["geophysical"] == parameters["makegeo"]["output"]
    grid = parameters["terrel"]["grid"]
    assert (grid.nx, grid.ny, grid.dx_km, grid.dy_km) == (610, 499, 1.0, 1.0)
    assert (grid.projection, grid.datum) == ("LCC", "WGS-84")
    assert all(
        parameters[stage]["grid"] is grid
        for stage in ("terrel", "ctrproc", "makegeo", "calmet")
    )
    assert parameters["calmet"]["end"] == datetime(2018, 3, 18, 1)
    assert parameters["calmet"]["noobs"] == 2
    assert parameters["calmet"]["iprog"] == 14
    assert parameters["calmet"]["irhprog"] == 1
    assert parameters["calmet"]["itprog"] == 2
    assert parameters["calmet"]["precipitation_station_count"] == -1
    assert parameters["calmet"]["coriolis"] == 1.0e-4
    assert parameters["calmet"]["maximum_mixing_height_m"] == 2500.0
    assert parameters["calmet"]["maximum_water_mixing_height_m"] == 2500.0
    assert parameters["calmet"]["precipitation_radius_km"] == 50.0
    assert parameters["calmet"]["mnmdav"] == 10
    assert parameters["calmet"]["wind_extrapolation"] == -1
    assert parameters["calmet"]["include_w_temperature"] is False
