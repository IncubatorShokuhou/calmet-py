from dataclasses import replace
from pathlib import Path

import pytest

import pycalmet
import pycalmet.pipeline as pipeline
from pycalmet.config import (
    CalmetConfig,
    CalwrfConfig,
    CtgprocConfig,
    MakegeoConfig,
    PipelineConfig,
    TerrelConfig,
)


STAGES = ("CALWRF", "TERREL", "CTGPROC", "MAKEGEO", "CALMET")
RUNNERS = {
    "CALWRF": "_run_calwrf",
    "TERREL": "_run_terrel",
    "CTGPROC": "_run_ctgproc",
    "MAKEGEO": "_run_makegeo",
    "CALMET": "_run_calmet",
}


def _pipeline_config(tmp_path: Path) -> PipelineConfig:
    grid = pycalmet.GridConfig(2, 2, 1.0, 1.0)
    return PipelineConfig(
        calwrf=CalwrfConfig(
            wrf_files=(tmp_path / "wrf.nc",),
            start="2018010100",
            end="2018010101",
            output=tmp_path / "configured-calwrf.nc",
            legacy_output=tmp_path / "configured.m3d",
            surface_output=tmp_path / "configured.m2d",
            listing_output=tmp_path / "calwrf.lst",
        ),
        terrel=TerrelConfig(
            srtm_dir=tmp_path / "srtm",
            output=tmp_path / "configured-terrain.nc",
            grid=grid,
            receptor_input=tmp_path / "receptors.inp",
            receptor_output=tmp_path / "receptors.out",
            save_output=tmp_path / "terrain.sav",
            plot_output=tmp_path / "terrain.grd",
            raw_echo_output=tmp_path / "raw-terrain.dat",
            gshhs_file=tmp_path / "gshhs.b",
            coastline_file=tmp_path / "terrain-coast.bln",
        ),
        ctgproc=CtgprocConfig(
            landuse_files=(tmp_path / "landuse.dat",),
            output=tmp_path / "configured-landuse.nc",
            grid=grid,
            output_format="legacy",
            final_output=False,
            qa_cell_output=True,
            qa_output_directory=tmp_path / "ctg-qa",
            gshhs_file=tmp_path / "gshhs.b",
            coastline_file=tmp_path / "landuse-coast.bln",
        ),
        makegeo=MakegeoConfig(
            terrain=tmp_path / "configured-terrain.nc",
            landuse=tmp_path / "configured-landuse.nc",
            output=tmp_path / "configured-geo.nc",
            grid=grid,
            landuse_plot_output=tmp_path / "landuse.grd",
            terrain_plot_output=tmp_path / "terrain.grd",
            color_map_output=tmp_path / "colors.txt",
        ),
        calmet=CalmetConfig(
            meteorology=tmp_path / "configured-calwrf.nc",
            geophysical=tmp_path / "configured-geo.nc",
            output=tmp_path / "configured-calmet.nc",
            start="2018010100",
            end="2018010101",
            grid=grid,
            cloud_output=1,
            distance_to_coast_output=tmp_path / "coast-distance.dat",
            listing_output=tmp_path / "calmet.lst",
            print_fields=True,
            micrometeorology_debug=True,
            wind_debug_output=1,
            wind_debug_fields=(1,) * 9,
            final_wind_debug_output=tmp_path / "final-wind.dat",
            kinematic_wind_debug_output=tmp_path / "kinematic-wind.dat",
            froude_wind_debug_output=tmp_path / "froude-wind.dat",
            slope_wind_debug_output=tmp_path / "slope-wind.dat",
            wind_print_output=tmp_path / "wind-print.dat",
        ),
    )


def _install_recording_runners(monkeypatch, *, write_outputs=False):
    calls = []

    def runner(stage):
        def run(config):
            calls.append((stage, config))
            output = (
                config.mesopac_output
                if stage == "CALMET"
                and config.legacy_output_format == 2
                and config.mesopac_output is not None
                else config.output
            )
            path = Path(output)
            if write_outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(stage, encoding="utf-8")
            return path

        return run

    for stage, name in RUNNERS.items():
        monkeypatch.setattr(pipeline, name, runner(stage))
    return calls


def test_run_pipeline_is_public():
    from pycalmet.pipeline import run_pipeline

    assert pycalmet.run_pipeline is run_pipeline
    assert pycalmet.PipelineConfig is PipelineConfig
    assert "run_pipeline" in pycalmet.__all__
    assert "PipelineConfig" in pycalmet.__all__


def test_run_pipeline_discovers_the_five_control_files(monkeypatch, tmp_path):
    calls = _install_recording_runners(monkeypatch)
    config_directory = Path(__file__).parents[1] / "examples" / "zhejiang" / "config"
    final_output = tmp_path / "calmet.nc"

    result = pycalmet.run_pipeline(config_directory, output=final_output)

    assert result == final_output
    assert [stage for stage, _ in calls] == list(STAGES)
    assert [type(config) for _, config in calls] == [
        CalwrfConfig,
        TerrelConfig,
        CtgprocConfig,
        MakegeoConfig,
        CalmetConfig,
    ]
    assert Path(calls[0][1].wrf_files[0]).name == (
        "wrfout_d01_2018-03-18_00_00_00.nc"
    )
    temporary_directory = Path(calls[0][1].output).parent
    assert [Path(config.output).name for _, config in calls[:-1]] == [
        "calwrf.nc",
        "terrain.nc",
        "landuse.nc",
        "geo.nc",
    ]
    assert all(
        Path(config.output).parent == temporary_directory
        for _, config in calls[:-1]
    )
    assert not temporary_directory.exists()


def test_run_pipeline_rewires_inputs_and_suppresses_intermediate_outputs(
    monkeypatch, tmp_path
):
    calls = _install_recording_runners(monkeypatch, write_outputs=True)
    final_output = tmp_path / "final.nc"

    result = pycalmet.run_pipeline(
        _pipeline_config(tmp_path), output=final_output
    )

    assert result == final_output
    configured = {stage: config for stage, config in calls}
    calwrf = configured["CALWRF"]
    terrel = configured["TERREL"]
    ctgproc = configured["CTGPROC"]
    makegeo = configured["MAKEGEO"]
    calmet = configured["CALMET"]

    assert calwrf.create_surface is False
    assert calwrf.legacy_output is None
    assert calwrf.surface_output is None
    assert calwrf.listing_output is None

    assert terrel.receptor_output.parent == Path(terrel.output).parent
    assert terrel.save_output is None
    assert terrel.plot_output is None
    assert terrel.raw_echo_output is None
    assert terrel.coastline_file.parent == Path(terrel.output).parent

    assert ctgproc.output_format == "netcdf"
    assert ctgproc.final_output is True
    assert ctgproc.qa_cell_output is False
    assert ctgproc.qa_output_directory is None
    assert ctgproc.coastline_file.parent == Path(ctgproc.output).parent

    assert makegeo.terrain == terrel.output
    assert makegeo.landuse == ctgproc.output
    assert makegeo.landuse_plot_output is None
    assert makegeo.terrain_plot_output is None
    assert makegeo.color_map_output is None

    assert calmet.meteorology == calwrf.output
    assert calmet.geophysical == makegeo.output
    assert calmet.output == final_output
    assert calmet.cloud_output == 0
    assert calmet.distance_to_coast_output is None
    assert calmet.listing_output is None
    assert calmet.print_fields is False
    assert calmet.micrometeorology_debug is False
    assert calmet.wind_debug_output == 0
    assert tuple(calmet.wind_debug_fields) == (0,) * 9
    assert calmet.final_wind_debug_output is None
    assert calmet.kinematic_wind_debug_output is None
    assert calmet.froude_wind_debug_output is None
    assert calmet.slope_wind_debug_output is None
    assert calmet.wind_print_output is None

    temporary_directory = Path(calwrf.output).parent
    assert not temporary_directory.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["final.nc"]


def test_run_pipeline_accepts_grouped_configs_mappings_and_paths(
    monkeypatch, tmp_path
):
    base = _pipeline_config(tmp_path)
    calls = _install_recording_runners(monkeypatch)
    control_directory = Path(__file__).parents[1] / "examples" / "zhejiang" / "config"

    result = pycalmet.run_pipeline(
        calwrf=base.calwrf,
        terrel={
            "srtm_dir": tmp_path / "srtm",
            "output": tmp_path / "terrain.nc",
            "grid": base.terrel.grid,
        },
        ctgproc=control_directory / "ctgproc.inp",
        makegeo=base.makegeo,
        calmet={
            "meteorology": tmp_path / "met.nc",
            "geophysical": tmp_path / "geo.nc",
            "output": tmp_path / "calmet.nc",
            "start": "2018010100",
            "end": "2018010101",
            "grid": base.calmet.grid,
        },
    )

    assert result == tmp_path / "calmet.nc"
    assert [stage for stage, _ in calls] == list(STAGES)
    assert all(
        isinstance(config, expected)
        for (_, config), expected in zip(
            calls,
            (CalwrfConfig, TerrelConfig, CtgprocConfig, MakegeoConfig, CalmetConfig),
        )
    )


def test_run_pipeline_accepts_top_level_mapping(monkeypatch, tmp_path):
    config = _pipeline_config(tmp_path)
    calls = _install_recording_runners(monkeypatch)

    result = pycalmet.run_pipeline(
        {
            "calwrf": config.calwrf,
            "terrel": config.terrel,
            "ctgproc": config.ctgproc,
            "makegeo": config.makegeo,
            "calmet": config.calmet,
        }
    )

    assert result == config.calmet.output
    assert [stage for stage, _ in calls] == list(STAGES)


def test_run_pipeline_routes_mesopac_to_the_selected_final_output(
    monkeypatch, tmp_path
):
    config = _pipeline_config(tmp_path)
    config = replace(
        config,
        calmet=replace(
            config.calmet,
            legacy_output_format=2,
            mesopac_output=tmp_path / "configured.pac",
        ),
    )
    calls = _install_recording_runners(monkeypatch)
    final_output = tmp_path / "selected.pac"

    result = pycalmet.run_pipeline(config, output=final_output)

    calmet = calls[-1][1]
    assert result == final_output
    assert calmet.output == final_output
    assert calmet.mesopac_output == final_output


def test_run_pipeline_managed_defaults_recognize_legacy_aliases(
    monkeypatch, tmp_path
):
    grid = pycalmet.GridConfig(2, 2, 1.0, 1.0)
    calls = _install_recording_runners(monkeypatch)

    result = pycalmet.run_pipeline(
        calwrf={
            "FLNAMES": (tmp_path / "wrf.nc",),
            "IDATEB": "2018010100",
            "IDATEE": "2018010101",
            "FIO": tmp_path / "configured-calwrf.nc",
        },
        terrel={
            "SRTMDIR": tmp_path / "srtm",
            "OUTFIL": tmp_path / "configured-terrain.nc",
            "GRID": grid,
        },
        ctgproc={
            "landuse_files": (tmp_path / "landuse.dat",),
            "LUDAT": tmp_path / "configured-landuse.nc",
            "GRID": grid,
        },
        makegeo={
            "TERRDAT": tmp_path / "configured-terrain.nc",
            "LUDAT": tmp_path / "configured-landuse.nc",
            "GEODAT": tmp_path / "configured-geo.nc",
            "GRID": grid,
        },
        calmet={
            "M3DDAT": tmp_path / "configured-calwrf.nc",
            "GEODAT": tmp_path / "configured-geo.nc",
            "METDAT": tmp_path / "configured-calmet.nc",
            "start": "2018010100",
            "end": "2018010101",
            "GRID": grid,
        },
    )

    assert result == tmp_path / "configured-calmet.nc"
    assert [stage for stage, _ in calls] == list(STAGES)


def test_run_pipeline_cleans_temporary_files_after_stage_failure(
    monkeypatch, tmp_path
):
    calls = []
    temporary_directories = []

    def calwrf(config):
        calls.append("CALWRF")
        temporary_directories.append(Path(config.output).parent)
        Path(config.output).write_text("temporary", encoding="utf-8")
        return config.output

    def terrel(config):
        calls.append("TERREL")
        raise RuntimeError("terrain failed")

    monkeypatch.setattr(pipeline, "_run_calwrf", calwrf)
    monkeypatch.setattr(pipeline, "_run_terrel", terrel)
    for stage in ("CTGPROC", "MAKEGEO", "CALMET"):
        monkeypatch.setattr(
            pipeline,
            RUNNERS[stage],
            lambda config, stage=stage: calls.append(stage),
        )

    with pytest.raises(RuntimeError, match="terrain failed") as caught:
        pycalmet.run_pipeline(_pipeline_config(tmp_path))

    assert calls == ["CALWRF", "TERREL"]
    assert not temporary_directories[0].exists()
    assert any("stage: TERREL" in note for note in caught.value.__notes__)


def test_run_pipeline_reports_missing_control_file_before_running(
    monkeypatch, tmp_path
):
    for filename in ("calwrf.inp", "terrel.inp", "ctgproc.inp", "makegeo.inp"):
        (tmp_path / filename).touch()
    calls = _install_recording_runners(monkeypatch)

    with pytest.raises(FileNotFoundError, match="calmet.inp"):
        pycalmet.run_pipeline(tmp_path)

    assert calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("snow", "snow_processing"),
        ("save", "save_output=True"),
        ("test", "test_mode=2"),
    ),
)
def test_run_pipeline_rejects_modes_without_one_final_calmet_product(
    monkeypatch, tmp_path, change, message
):
    config = _pipeline_config(tmp_path)
    if change == "snow":
        config = replace(
            config, makegeo=replace(config.makegeo, snow_processing=True)
        )
    elif change == "save":
        config = replace(
            config, calmet=replace(config.calmet, save_output=False)
        )
    else:
        config = replace(config, calmet=replace(config.calmet, test_mode=1))
    calls = _install_recording_runners(monkeypatch)

    with pytest.raises(ValueError, match=message):
        pycalmet.run_pipeline(config)

    assert calls == []


def test_run_pipeline_requires_every_explicit_stage(monkeypatch, tmp_path):
    config = _pipeline_config(tmp_path)
    calls = _install_recording_runners(monkeypatch)

    with pytest.raises(TypeError, match="missing: calmet"):
        pycalmet.run_pipeline(
            calwrf=config.calwrf,
            terrel=config.terrel,
            ctgproc=config.ctgproc,
            makegeo=config.makegeo,
        )

    assert calls == []
