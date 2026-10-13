from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

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


def test_runner_calls_each_stage_with_only_its_control_file(tmp_path):
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

            def replacement(name):
                def run(*args, **kwargs):
                    with Path(os.environ["PYCALMET_TEST_CAPTURE"]).open("a") as stream:
                        stream.write(json.dumps({
                            "name": name,
                            "args": [str(item) for item in args],
                            "kwargs": kwargs,
                        }) + "\\n")
                    return Path("output") / (name + ".nc")
                return run

            for name in ("calwrf", "terrel", "ctrproc", "makegeo", "calmet"):
                setattr(pycalmet, name, replacement(name))
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
    assert [call["name"] for call in calls] == [
        "calwrf",
        "terrel",
        "ctrproc",
        "makegeo",
        "calmet",
    ]
    assert all(len(call["args"]) == 1 and not call["kwargs"] for call in calls)
