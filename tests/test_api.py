from dataclasses import dataclass, fields, MISSING
from copy import copy
import inspect
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

import pycalmet
from pycalmet._calmet import (_noobs_step_transform,_prognostic_precipitation_radius,_validate,
                              _validate_geophysical_grid,_validate_input_span,
                              _validate_input_timezone,_output_path,
                              _write_distance_to_coast_grid,
                              _validate_sea_metadata,_validate_upper_span)
from pycalmet.config import CalmetConfig
from pycalmet.legacy._compat._calmet import (
    _prognostic_precipitation_radius as _legacy_prognostic_precipitation_radius,
    _validate as _legacy_validate,
    _validate_upper_span as _legacy_validate_upper_span,
)
from pycalmet.sea_data import GriddedSeaData

CALMET_CONTROL = Path(__file__).with_name("fixtures") / "calmet.inp"


def _sea_metadata(*,version='2.11',timezone='UTC-0000',**changes):
    values=dict(version=version,timezone=timezone,
        source_grid=pycalmet.GridConfig(1,1,1.,1.,projection='LCC',
            lat0=30.,lon0=-120.,lat1=30.,lat2=60.),
        projection='LCC',datum='WGS-84',datum_date='20200101')
    values.update(changes)
    return SimpleNamespace(**values)


def _prognostic_calmet_config(*args, **kwargs):
    """Build the prognostic baseline used by the validation tests."""
    defaults = {
        "noobs": 2,
        "iprog": 14,
        "irhprog": 1,
        "itprog": 2,
        "water_temperature_category_min": 55,
        "water_temperature_category_max": 55,
        "wind_extrapolation": -1,
        "varying_radius": True,
        "stations_per_level": (5,) * 10,
        "barrier_top_level": 0,
    }
    defaults.update(kwargs)
    return CalmetConfig(*args, **defaults)


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError, match="unknown CalwrfConfig parameters"):
        pycalmet.calwrf({"not_a_parameter": 1})


def test_modern_configs_store_public_times_as_datetime():
    calwrf=pycalmet.CalwrfConfig(
        ['wrf.nc'],'2018031800','2018-03-18_01:00:00','calwrf.nc')
    calmet=pycalmet.CalmetConfig(
        'met.nc','geo.nc','calmet.nc','2018-03-18 00:00:00',
        datetime(2018,3,18,1),pycalmet.GridConfig(1,1,1.,1.))
    assert (calwrf.start,calwrf.end)==(
        datetime(2018,3,18),datetime(2018,3,18,1))
    assert (calmet.start,calmet.end)==(
        datetime(2018,3,18),datetime(2018,3,18,1))


@pytest.mark.parametrize('value',[
    datetime(2018,3,18,tzinfo=timezone.utc),
    datetime(2018,3,18,microsecond=1),
])
def test_modern_configs_reject_datetime_information_the_model_cannot_store(value):
    with pytest.raises(ValueError):
        pycalmet.CalwrfConfig(['wrf.nc'],value,datetime(2018,3,18,1),'out.nc')
    with pytest.raises(ValueError):
        pycalmet.CalmetConfig(
            'met.nc','geo.nc','out.nc',value,datetime(2018,3,18,1),
            pycalmet.GridConfig(1,1,1.,1.))


@pytest.mark.parametrize(('field','value'),(
    ('base_timezone',8.5),('timestep_seconds',900.5),
    ('prognostic_timestep_seconds',3600.5),('print_interval_hours',1.5),
    ('debug_first_step',1.5),('debug_last_step',2.5),
))
def test_explicit_calmet_config_rejects_fractional_time_integer_fields(
        field,value):
    options={field:value}
    with pytest.raises(ValueError,match=field):
        pycalmet.CalmetConfig(
            'met.nc','geo.nc','out.nc',datetime(2018,3,18),
            datetime(2018,3,18,1),pycalmet.GridConfig(1,1,1.,1.),
            **options)


def test_explicit_calmet_config_normalizes_exact_time_integer_values():
    config=pycalmet.CalmetConfig(
        'met.nc','geo.nc','out.nc',datetime(2018,3,18),
        datetime(2018,3,18,1),pycalmet.GridConfig(1,1,1.,1.),
        base_timezone=8.,timestep_seconds=900.,
        prognostic_timestep_seconds=3600.)
    assert (config.base_timezone,config.timestep_seconds,
            config.prognostic_timestep_seconds)==(8,900,3600)
    assert all(isinstance(value,int) for value in (
        config.base_timezone,config.timestep_seconds,
        config.prognostic_timestep_seconds))


@pytest.mark.parametrize('sentinel',[-1,'-2018031800'])
def test_modern_calwrf_rejects_legacy_negative_date_sentinel(sentinel):
    with pytest.raises((TypeError,ValueError),match='CALWRF start'):
        pycalmet.CalwrfConfig(
            ['wrf.nc'],sentinel,'2018031801','calwrf.nc')


def test_makegeo_public_snow_dates_are_calendar_types_or_explicit_strings():
    grid=pycalmet.GridConfig(1,1,1.,1.)
    config=pycalmet.MakegeoConfig(
        'terrain','landuse','geo',grid,
        snow_start_date='2020-01-02',snow_end_date=datetime(2020,1,3))
    assert (config.snow_start_date,config.snow_end_date)==(
        date(2020,1,2),date(2020,1,3))
    with pytest.raises(TypeError,match='date'):
        pycalmet.MakegeoConfig(
            'terrain','landuse','geo',grid,
            snow_start_date=20200102,snow_end_date=date(2020,1,3))
    with pytest.raises(ValueError,match='midnight'):
        pycalmet.MakegeoConfig(
            'terrain','landuse','geo',grid,
            snow_start_date=datetime(2020,1,2,1),
            snow_end_date=date(2020,1,3))


@pytest.mark.parametrize(("stage", "config_type", "has_grid"), (
    ("calwrf", pycalmet.CalwrfConfig, False),
    ("terrel", pycalmet.TerrelConfig, True),
    ("ctrproc", pycalmet.CtgprocConfig, True),
    ("ctgproc", pycalmet.CtgprocConfig, True),
    ("makegeo", pycalmet.MakegeoConfig, True),
    ("calmet", pycalmet.CalmetConfig, True),
))
def test_stage_signatures_expose_every_typed_configuration_parameter(
        stage, config_type, has_grid):
    signature = inspect.signature(getattr(pycalmet, stage))
    expected = {field.name for field in fields(config_type)}
    if has_grid:
        expected.update(field.name for field in fields(pycalmet.GridConfig))
    assert expected <= signature.parameters.keys()
    assert signature.parameters["inp"].default is None
    for field in fields(config_type):
        if field.name == "grid" or field.default is MISSING:
            continue
        if field.default_factory is not MISSING:
            assert signature.parameters[field.name].default is None
        else:
            assert signature.parameters[field.name].default == field.default


def test_preprocessor_defaults_match_fortran_control_defaults():
    grid = pycalmet.GridConfig(nx=1, ny=1, dx_km=1.0, dy_km=1.0)
    makegeo = pycalmet.MakegeoConfig("terrain", "landuse", "geo", grid)

    assert grid.polar_processing == 2
    assert grid.projection == "UTM"
    assert pycalmet.TerrelConfig("terrain", "out", grid).receptor_columns == 4
    ctgproc = pycalmet.CtgprocConfig(("landuse",), "out", grid)
    assert (ctgproc.input_type, ctgproc.mesh_density,
            ctgproc.input_datum) == ("CTG", 1, "NAS-C")
    assert ctgproc.fill_empty_ocean is False
    assert pycalmet.CtgprocConfig(
        ("landuse",), "out", grid, coastline_file="coast.bln",
    ).fill_empty_ocean is True
    assert makegeo.snow_height_scale_m == 10.0
    assert makegeo.snow_depth_minimum_m == 0.01

    for stage in ("terrel", "ctrproc", "ctgproc", "makegeo", "calmet"):
        assert inspect.signature(getattr(pycalmet, stage)).parameters[
            "polar_processing"
        ].default == 2
    makegeo_signature = inspect.signature(pycalmet.makegeo)
    assert makegeo_signature.parameters["snow_height_scale_m"].default == 10.0
    assert makegeo_signature.parameters["snow_depth_minimum_m"].default == 0.01


def test_calmet_defaults_match_fortran_control_defaults():
    config = CalmetConfig(
        "met.nc", "geo.nc", "out.nc", "2018010100", "2018010101",
        pycalmet.GridConfig(1, 1, 1.0, 1.0),
    )
    expected = {
        "noobs": 0,
        "iprog": 0,
        "irhprog": 0,
        "itprog": 0,
        "precipitation_station_count": 0,
        "water_temperature_category_min": 999,
        "water_temperature_category_max": 999,
        "coriolis": 999.0,
        "wind_extrapolation": -4,
        "varying_radius": False,
        "include_w_temperature": True,
    }
    signature = inspect.signature(pycalmet.calmet)
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert signature.parameters[name].default == value
    dynamic = {
        "wind_bias": (0.0,) * 10,
        "stations_per_level": (99,) * 10,
        "wind_extrapolation_factors": (0.0,) * 10,
        "smoothing_passes": (2,) + (4,) * 9,
    }
    for name, value in dynamic.items():
        assert getattr(config, name) == value
        assert signature.parameters[name].default is None
    assert config.barrier_top_level == len(config.z_faces_m) - 1
    assert signature.parameters["barrier_top_level"].default is None


def test_calmet_default_barrier_top_level_tracks_vertical_grid():
    config = _prognostic_calmet_config(
        "met.nc", "geo.nc", "out.nc", "2018010100", "2018010101",
        pycalmet.GridConfig(1, 1, 1.0, 1.0),
        barrier_top_level=None,
        z_faces_m=(0.0, 20.0, 60.0),
        smoothing_passes=(0, 0),
    )
    _validate(config)
    assert config.barrier_top_level == 2


def test_parameter_only_calmet_accepts_datetime_and_flat_grid(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._calmet.run",
                        lambda config: captured.append(config) or Path(config.output))
    result = pycalmet.calmet(
        inp=None,
        meteorology=tmp_path / "calwrf.nc",
        geophysical=tmp_path / "geo.nc",
        output=tmp_path / "calmet.nc",
        start=datetime(2018, 3, 18),
        end=datetime(2018, 3, 18, 1),
        nx=4,
        ny=3,
        dx_km=1.0,
        dy_km=1.0,
        lat0=30.0,
        lon0=120.0,
        z_faces_m=(0.0, 20.0, 60.0),
    )
    config = captured[0]
    assert result == tmp_path / "calmet.nc"
    assert config.start == datetime(2018, 3, 18)
    assert config.end == datetime(2018, 3, 18, 1)
    assert (config.grid.nx, config.grid.ny) == (4, 3)
    assert config.noobs == 0
    assert config.precipitation_station_count == 0
    assert config.coriolis == 999.0
    assert config.include_w_temperature is True
    assert config.wind_bias == (0.0, 0.0)
    assert config.stations_per_level == (99, 99)
    assert config.wind_extrapolation_factors == (0.0, 0.0)
    assert config.smoothing_passes == (2, 4)
    assert config.timestep_seconds == 3600


def test_control_file_honours_explicit_overrides_equal_to_python_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr("pycalmet._calmet.run",
                        lambda config: captured.append(config) or Path(config.output))
    pycalmet.calmet(
        CALMET_CONTROL,
        noobs=0,
        iprog=0,
        coriolis=999.0,
        include_w_temperature=True,
    )
    config = captured[0]
    assert config.noobs == 0
    assert config.iprog == 0
    assert config.coriolis == 999.0
    assert config.include_w_temperature is True


def test_grid_object_is_preserved_unless_flattened_grid_value_is_explicit(monkeypatch):
    captured = []
    monkeypatch.setattr("pycalmet._terrel.run",
                        lambda config: captured.append(config) or
                        SimpleNamespace(output=config.output))
    grid = pycalmet.GridConfig(
        2, 3, 1.0, 1.0, projection="LCC", lat0=27.0, lon0=117.5,
        lat1=30.0, lat2=60.0,
    )
    pycalmet.terrel(srtm_dir="terrain", output="terrain.nc", grid=grid)
    assert captured[-1].grid.projection == "LCC"
    pycalmet.terrel(
        srtm_dir="terrain", output="terrain.nc", grid=grid,
        projection="UTM",
    )
    assert captured[-1].grid.projection == "UTM"
    pycalmet.terrel(
        srtm_dir="terrain", output="terrain.nc",
        grid={"nx": 2, "ny": 3, "dx_km": 1.0, "dy_km": 1.0,
              "projection": "LCC"},
    )
    assert isinstance(captured[-1].grid, pycalmet.GridConfig)
    assert captured[-1].grid.projection == "LCC"


def test_modern_calmet_accepts_supported_wind_pair_rejected_by_legacy():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=13,iwfcod=1,
        prognostic_radius_km=10.,z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='does not allow'):
        _legacy_validate(config)


def test_calmet_rejects_unknown_wind_codes_and_accepts_rectangular_cells():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.,projection='LCC'),
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.iprog=6
    with pytest.raises(ValueError,match='IPROG must be'): _validate(config)
    config.iprog=14; config.iwfcod=2
    with pytest.raises(ValueError,match='IWFCOD must be'): _validate(config)
    config.iwfcod=1; config.grid.dy_km=2.
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='DGRID'): _legacy_validate(config)


@pytest.mark.parametrize(('dx_km','dy_km'),[
    (0.,1.),(1.,0.),(-1.,1.),(1.,-1.),(np.inf,1.),(1.,np.nan),
])
def test_calmet_rejects_nonpositive_or_nonfinite_grid_spacing(dx_km,dy_km):
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,dx_km,dy_km),z_faces_m=(0.,20.),
        smoothing_passes=(0,))
    with pytest.raises(ValueError,match='finite and positive'):
        _validate(config)


def test_calmet_first_layer_midpoint_check_exempts_wind_only_runs():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,30.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='first layer midpoint'): _validate(config)
    config.run_type=0
    assert _validate(config).tolist()==[0.,30.]


def test_calmet_first_layer_midpoint_uses_ordinary_model_precision():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),
        smoothing_passes=(0,))
    assert _validate(config).dtype==np.float32
    legacy_config=copy(config)
    legacy_config.start=config.start.strftime('%Y%m%d%H')
    legacy_config.end=config.end.strftime('%Y%m%d%H')
    assert _legacy_validate(legacy_config).dtype==np.float64


def test_calmet_projection_qa_matches_fortran_set():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.,projection='EM'),z_faces_m=(0.,20.),
        smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.grid.projection='LAZA'; config.grid.lat0=30.; config.grid.lon0=-120.
    assert _validate(config).tolist()==[0.,20.]
    config.grid.projection='ACEA'
    with pytest.raises(ValueError,match='PMAP'): _validate(config)
    config.grid.projection='UTM'; config.grid.utm_zone=0
    with pytest.raises(ValueError,match='zone'): _validate(config)
    config.grid.utm_zone=50; config.grid.hemisphere='X'
    with pytest.raises(ValueError,match='hemisphere'): _validate(config)
    config.grid.hemisphere='N'
    assert _validate(config).tolist()==[0.,20.]

def test_calmet_readcf_validates_grid_timezone_and_start_time():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.,projection='LCC'),
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.base_timezone=13
    with pytest.raises(ValueError,match='timezone'): _validate(config)
    config.base_timezone=8; config.base_timezone_label='UTC+0800'
    with pytest.raises(ValueError,match='timezone label'): _validate(config)
    config.base_timezone=0; config.base_timezone_label=None; config.grid.nx=0
    with pytest.raises(ValueError,match='NX'): _validate(config)
    config.grid.nx=1; config.grid.datum='UNKNOWN'
    with pytest.raises(ValueError,match='DATUM'): _validate(config)
    config.grid.datum='WGS-84'; config.grid.lat0=91.
    with pytest.raises(ValueError,match='origin'): _validate(config)
    config.grid.lat0=27.; config.start=datetime(2018,1,1,5)
    with pytest.raises(ValueError,match='before 05 LST'): _validate(config)
    config.start=datetime(2018,1,1,4)
    assert _validate(config).tolist()==[0.,20.]
    config.start=datetime(2018,1,1,5)
    config.run_type=0
    assert _validate(config).tolist()==[0.,20.]
    config.start=datetime(2018,1,1,0,0,1)
    with pytest.raises(ValueError,match='NSECDT'): _validate(config)

def test_calmet_ignores_mixing_level_when_upwind_average_is_disabled():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iavezi=0,mixing_wind_level=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.iavezi=1
    with pytest.raises(ValueError,match='mixing_wind_level'): _validate(config)


def test_calmet_accepts_legacy_output_and_setup_controls():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.dat','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.save_output=False
    assert _validate(config).tolist()==[0.,20.]
    config.save_output=True; config.legacy_output_format=2
    assert _validate(config).tolist()==[0.,20.]
    config.timestep_seconds=900
    with pytest.raises(ValueError,match='MESOPAC'): _validate(config)
    config.timestep_seconds=3600
    config.legacy_output_format=1; config.test_mode=1
    assert _validate(config).tolist()==[0.,20.]
    config.test_mode=3
    with pytest.raises(ValueError,match='ITEST'):
        _validate(config)


def test_calmet_accepts_print_and_debug_outputs_fixed_in_modern_runtime():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.dat','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,),
        print_fields=True,print_uv_levels=(1,))
    assert _validate(config).tolist()==[0.,20.]
    config.micrometeorology_debug=True
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=0
    config.surface_temperature_option=-2
    config.surface_data='surf.dat'; config.upper_air_files=('up.dat',)
    config.stations_per_level=(1,); config.wind_extrapolation_factors=(1.,)
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1
    assert _validate(config).tolist()==[0.,20.]
    config.surface_temperature_option=-1
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='undefined TINF'):
        _legacy_validate(config)
    config.surface_temperature_option=-2
    config.noobs=2
    config.micrometeorology_debug=False
    config.print_fields=False; config.wind_debug_fields=(0,0,0,0,0,0,0,0,1)
    assert _validate(config).tolist()==[0.,20.]
    config.wind_debug_output=1
    assert _validate(config).tolist()==[0.,20.]
    config.wind_debug_fields=(0,0,0,0,0,1,1,1,1)
    assert _validate(config).tolist()==[0.,20.]
    config.wind_debug_fields=(0,0,0,0,1,0,0,0,0)
    assert _validate(config).tolist()==[0.,20.]
    config.wind_debug_fields=(0,0,0,1,0,0,0,0,0)
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='FORMAT 2820'):
        _legacy_validate(config)
    config.wind_debug_fields=(0,0,1,0,0,0,0,0,0)
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_selects_pacdat_only_for_mesopac_output():
    config=_prognostic_calmet_config('met.nc','geo.nc','calmet.dat','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),mesopac_output='pacout.dat')
    assert _output_path(config)==Path('calmet.dat')
    config.legacy_output_format=2
    assert _output_path(config)==Path('pacout.dat')


def test_calmet_writes_setcoast_surfer_grid(tmp_path):
    grid=pycalmet.GridConfig(3,3,1.,1.,x_origin_km=10.,y_origin_km=20.)
    landuse=np.full((3,3),55,dtype=np.int16); landuse[0,0]=10
    path=tmp_path/'dcst.grd'
    distance=_write_distance_to_coast_grid(path,landuse,grid,(50,55))
    np.testing.assert_allclose(distance,
        [[0.,1.,2.],[1.,np.sqrt(2.),np.sqrt(5.)],[2.,np.sqrt(5.),np.sqrt(8.)]])
    lines=path.read_text(encoding='ascii').splitlines()
    assert lines[:4]==['DSAA','           3           3',
                       '     11.0000     13.0000','     21.0000     23.0000']
    assert [float(value) for value in ' '.join(lines[5:]).split()]==pytest.approx(
        distance.ravel().tolist(),rel=5.e-5)


def test_calmet_iprog13_objective_noobs_mode_is_supported():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=13,iwfcod=0,prognostic_radius_km=10.,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    with pytest.raises(ValueError,match='WTDAT'): _validate(config)
    config.weight_data='wt.dat'; config.prognostic_radius_km=10.
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_iprog15_noobs_prognostic_observations_are_supported():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=15,iwfcod=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='requires IDIOPT3=1'):
        _legacy_validate(config)
    config.diagnostic_initial_wind=1; config.diagnostic_data='diag.dat'
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    with pytest.raises(ValueError,match='WTDAT'): _validate(config)
    config.weight_data='wt.dat'
    assert _validate(config).tolist()==[0.,20.]
    config.use_initial_guess_calmet=1
    config.initial_guess_calmet_files=('coarse-calmet.dat',)
    assert _validate(config).tolist()==[0.,20.]
    config.iprog=13; config.iwfcod=0; config.prognostic_radius_km=10.
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match='does not allow IGFMET'):
        _legacy_validate(config)


def test_calmet_iprog15_objective_analysis_mode_is_supported():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=15,iwfcod=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_mm4_wind_modes_map_to_their_mm5_counterparts():
    grid=pycalmet.GridConfig(1,1,1.,1.)
    for iprog,iwfcod in ((3,0),(4,1),(5,0)):
        config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
            grid,iprog=iprog,iwfcod=iwfcod,
            prognostic_radius_km=(10. if iprog==3 else 0.),
            z_faces_m=(0.,20.),smoothing_passes=(0,))
        assert _validate(config).tolist()==[0.,20.]
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        grid,iprog=5,iwfcod=1,diagnostic_initial_wind=1,diagnostic_data='diag.dat',
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    with pytest.raises(ValueError,match='WTDAT'):
        _validate(config)


def test_calmet_prog_wind_modes_are_supported_with_observations():
    common=dict(meteorology='PROG.DAT',geophysical='geo.nc',output='out.nc',
        start='2018010100',end='2018010101',grid=pycalmet.GridConfig(1,1,1.,1.),
        noobs=0,surface_data='surf.dat',upper_air_files=('up.dat',),itprog=0,irhprog=0,
        precipitation_station_count=0,z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,))
    first=_prognostic_calmet_config(**common,iprog=1,iwfcod=0,prognostic_radius_km=10.)
    second=_prognostic_calmet_config(**common,iprog=2,iwfcod=1)
    assert _validate(first).tolist()==[0.,20.]
    assert _validate(second).tolist()==[0.,20.]
    first.prognostic_radius_km=0.
    with pytest.raises(ValueError,match='RPROG'):
        _validate(first)
    second.timestep_seconds=900
    assert _validate(second).tolist()==[0.,20.]


def test_calmet_iprog0_wind1_observation_mode_is_supported():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=0,iwfcod=1,noobs=0,
        surface_data='surf.dat',upper_air_files=('up.dat',),itprog=0,irhprog=0,
        precipitation_station_count=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,))
    assert _validate(config).tolist()==[0.,20.]
    config.iwfcod=0
    assert _validate(config).tolist()==[0.,20.]
    config.iwfcod=1
    config.noobs=1
    with pytest.raises(ValueError,match='NOOBS=0'): _validate(config)
    config.noobs=0; config.itprog=2
    with pytest.raises(ValueError,match='ITPROG=0'): _validate(config)
    config.itprog=0; config.irhprog=1
    with pytest.raises(ValueError,match='IRHPROG=0'): _validate(config)


def test_observed_similarity_wind_passes_setup_for_runtime_stheor_dispatch():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=0,iwfcod=1,noobs=0,
        surface_data='surf.dat',upper_air_files=('up.dat',),itprog=0,irhprog=0,
        precipitation_station_count=0,wind_extrapolation=4,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,))
    np.testing.assert_array_equal(_validate(config),[0.,20.])


def test_calmet_selected_upper_initial_wind_controls_are_validated():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=0,iwfcod=1,noobs=0,
        surface_data='surf.dat',upper_air_files=('up.dat',),itprog=0,irhprog=0,
        precipitation_station_count=0,z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,),
        initial_guess_upper_station=2)
    with pytest.raises(ValueError,match='IUPWND'): _validate(config)
    config.initial_guess_upper_station=1
    assert _validate(config).tolist()==[0.,20.]
    config.initial_guess_height_range_m=(1000.,1.)
    with pytest.raises(ValueError,match='ZUPWND'): _validate(config)


def test_calmet_diagnostic_data_controls_are_validated():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,surface_data='surf.dat',
        upper_air_files=('up.dat',),z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,),
        diagnostic_initial_wind=1)
    with pytest.raises(ValueError,match='DIADAT'): _validate(config)
    config.diagnostic_data='diag.dat'; config.diagnostic_initial_wind=0
    config.diagnostic_surface_wind=1
    assert _validate(config).tolist()==[0.,20.]
    config.run_type=0
    config.surface_data=None
    with pytest.raises(ValueError,match='IDIOPT4.*SS records'): _validate(config)
    config.surface_control_stations=(('SFC',1,.5,.5,0.,10.),)
    assert _validate(config).tolist()==[0.,20.]
    config.diagnostic_surface_wind=0; config.diagnostic_surface_temperature=1
    config.diagnostic_data=None; config.surface_data='surf.dat'
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_noobs2_requires_only_consumed_diagnostic_data():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),diagnostic_lapse_rate=1,
        diagnostic_initial_wind=1,diagnostic_surface_wind=1,
        diagnostic_upper_wind=1,z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.diagnostic_surface_temperature=1
    assert _validate(config).tolist()==[0.,20.]
    config.surface_temperature_option=-2
    with pytest.raises(ValueError,match='DIADAT'): _validate(config)


def test_calmet_initial_guess_file_controls_follow_fortran_compatibility():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),use_initial_guess_calmet=1,
        iprog=0,iwfcod=1,noobs=0,surface_data='surf.dat',upper_air_files=('up.dat',),
        itprog=0,irhprog=0,precipitation_station_count=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,))
    with pytest.raises(ValueError,match='IGFDAT'): _validate(config)
    config.initial_guess_calmet_files=('coarse.dat',)
    assert _validate(config).tolist()==[0.,20.]
    config.iprog=14; config.itprog=2; config.irhprog=1
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_spatial_averaging_switches_are_supported():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iavet=0,iavezi=0,z_faces_m=(0.,20.),
        smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.iobr=1
    assert _validate(config).tolist()==[0.,20.]
    config.run_type=0
    assert _validate(config).tolist()==[0.,20.]
    config.timestep_seconds=900
    assert _validate(config).tolist()==[0.,20.]
    config.timestep_seconds=1000
    with pytest.raises(ValueError,match='NSECDT'):
        _validate(config)
    config.timestep_seconds=3600; config.cloud_method=3
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_method=2; config.cloud_data='cloud.dat'
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(1.,)
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_data=None
    with pytest.raises(ValueError,match='CLDDAT'):
        _validate(config)
    config.cloud_method=4; config.timestep_seconds=900; config.prognostic_timestep_seconds=1000
    with pytest.raises(ValueError,match='ISTEPPGS'):
        _validate(config)


def test_calmet_overwater_and_regulatory_controls_are_validated():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.overwater_flux_method=9
    with pytest.raises(ValueError,match='ICOARE'): _validate(config)
    config.overwater_flux_method=10; config.water_temperature_mode=2; config.iprog=0
    config.noobs=0; config.surface_data='surf.dat'; config.upper_air_files=('up.dat',)
    config.itprog=0; config.irhprog=0; config.precipitation_station_count=0
    config.stations_per_level=(1,); config.wind_extrapolation_factors=(0.,)
    with pytest.raises(ValueError,match='ITWPROG=2'): _validate(config)
    config.iprog=14; config.water_temperature_mode=0; config.regulatory_mode=1
    with pytest.raises(ValueError,match='MREG=1'): _validate(config)
    config.overwater_flux_method=0; config.imixh=-1; config.convective_relaxation=0
    config.itprog=2
    config.surface_temperature_option=-2; config.upper_temperature_option=-2
    assert _validate(config).tolist()==[0.,20.]


def test_negative_icoare_requires_wave_capable_sea_data():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),overwater_flux_method=-11,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    old=_sea_metadata(version='2.00',timezone=None)
    with pytest.raises(ValueError,match='negative ICOARE.*2.1'):
        _validate_sea_metadata(config,(old,))
    modern=_sea_metadata(version='2.10',timezone=None)
    _validate_sea_metadata(config,(modern,))


def test_sea_metadata_timezone_uses_calmet_positive_west_convention():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),base_timezone=8,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    _validate_sea_metadata(config,(_sea_metadata(timezone='UTC-0800'),))
    with pytest.raises(ValueError,match='whole-hour'):
        _validate_sea_metadata(config,(_sea_metadata(timezone='UTC-0830'),))
    with pytest.raises(ValueError,match='timezone'):
        _validate_sea_metadata(config,(_sea_metadata(timezone='UTC+0800'),))
    with pytest.raises(ValueError,match='invalid SEA.DAT timezone'):
        _validate_sea_metadata(config,(_sea_metadata(timezone='PST'),))


def test_surface_and_precip_timezones_accept_modern_whole_hour_and_old_integer_forms():
    _validate_input_timezone('UTC-0800',8,'SURF.DAT')
    _validate_input_timezone(8,8,'PRECIP.DAT')
    with pytest.raises(ValueError,match='whole-hour'):
        _validate_input_timezone('UTC-0830',8,'SURF.DAT')
    with pytest.raises(ValueError,match='PRECIP.DAT timezone'):
        _validate_input_timezone(-8,8,'PRECIP.DAT')
    with pytest.raises(ValueError,match='must be an integer'):
        _validate_input_timezone(8.5,8,'PRECIP.DAT')
    with pytest.raises(ValueError,match='invalid SURF.DAT timezone'):
        _validate_input_timezone('UTC-ABCD',0,'SURF.DAT')


def test_calmet_timestep_alignment_includes_subsecond_datetime_remainder():
    config=_prognostic_calmet_config(
        'met.nc','geo.nc','out.nc',datetime(2018,1,1,0,15),
        datetime(2018,1,1,1),pycalmet.GridConfig(1,1,1.,1.),
        run_type=0,timestep_seconds=900,z_faces_m=(0.,20.),
        smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.start=datetime(2018,1,1,0,15,microsecond=1)
    with pytest.raises(ValueError,match='align with NSECDT'):
        _validate(config)


def test_surface_and_precip_declared_spans_must_cover_the_run():
    run_start=datetime(2018,1,1); run_end=datetime(2018,1,1,2)
    covering=SimpleNamespace(start=run_start,end=run_end)
    _validate_input_span(covering,run_start,run_end,'SURF.DAT')
    late=SimpleNamespace(start=datetime(2018,1,1,1),end=run_end)
    with pytest.raises(ValueError,match='SURF.DAT.*does not cover'):
        _validate_input_span(late,run_start,run_end,'SURF.DAT')
    early=SimpleNamespace(start=run_start,end=datetime(2018,1,1,1))
    with pytest.raises(ValueError,match='PRECIP.DAT.*does not cover'):
        _validate_input_span(early,run_start,run_end,'PRECIP.DAT')


def test_upper_air_declared_span_is_compared_in_utc():
    run_start=datetime(2018,1,1); run_end=datetime(2018,1,1,2)
    covering=SimpleNamespace(start=datetime(2018,1,1,8),
                             end=datetime(2018,1,1,10))
    _validate_upper_span(covering,run_start,run_end,8)
    late=SimpleNamespace(start=datetime(2018,1,1,9),
                         end=datetime(2018,1,1,10))
    with pytest.raises(ValueError,match='run beginning'):
        _validate_upper_span(late,run_start,run_end,8)
    early=SimpleNamespace(start=datetime(2018,1,1,8),
                          end=datetime(2018,1,1,9))
    with pytest.raises(ValueError,match='run ending'):
        _validate_upper_span(early,run_start,run_end,8)


def test_upper_air_equal_end_hour_uses_actual_end_second_while_legacy_keeps_quirk():
    run_start=datetime(2018,1,1,0,30); run_end=datetime(2018,1,1,2)
    data=SimpleNamespace(start=datetime(2018,1,1,0),
                         end=datetime(2018,1,1,2,59))
    _validate_upper_span(data,run_start,run_end,0)
    with pytest.raises(ValueError,match='run ending'):
        _legacy_validate_upper_span(data,run_start,run_end,0)

    early=SimpleNamespace(start=datetime(2018,1,1,0),
                          end=datetime(2018,1,1,1,59,59))
    with pytest.raises(ValueError,match='run ending'):
        _validate_upper_span(early,run_start,run_end,0)


def test_sea_files_require_shared_projection_and_datum_metadata():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    first=_sea_metadata()
    matching=SimpleNamespace(**vars(first))
    _validate_sea_metadata(config,(first,matching))
    changed=SimpleNamespace(**{**vars(first),'datum_date':'20210101'})
    with pytest.raises(ValueError,match='projection and datum'):
        _validate_sea_metadata(config,(first,changed))
    shifted=SimpleNamespace(**{**vars(first),'source_grid':pycalmet.GridConfig(
        1,1,1.,1.,projection='LCC',lat0=30.,lon0=-120.,lat1=30.,lat2=60.,
        false_easting_km=1.)})
    with pytest.raises(ValueError,match='projection and datum'):
        _validate_sea_metadata(config,(first,shifted))

def test_calmet_noobs1_requires_surface_data_and_rejects_unported_barriers():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=1,z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,))
    with pytest.raises(ValueError,match='SRFDAT'):
        _validate(config)
    config.surface_data='surf.dat'; assert _validate(config).tolist()==[0.,20.]
    config.barrier_count=1
    with pytest.raises(ValueError,match='barrier_count'):
        _validate(config)
    config.barriers=((0.,0.,1.,1.),); config.barrier_top_level=1
    assert _validate(config).tolist()==[0.,20.]

def test_calmet_noobs0_requires_surface_and_upper_air_files():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,))
    with pytest.raises(ValueError,match='SRFDAT'): _validate(config)
    config.surface_data='surf.dat'
    with pytest.raises(ValueError,match='UPDAT'): _validate(config)
    config.upper_air_files=('up.dat',)
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_idiopt4_5_accept_control_metadata_without_observation_files():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,run_type=0,
        diagnostic_data='diag.dat',diagnostic_surface_wind=1,
        diagnostic_upper_wind=1,surface_control_stations=(('SFC',1,.5,.5,0.,10.),),
        upper_control_stations=(('UPR',2,.5,.5,0.),),
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,))
    assert _validate(config).tolist()==[0.,20.]
    config.surface_control_stations=()
    with pytest.raises(ValueError,match='IDIOPT4.*SS records'): _validate(config)
    config.surface_control_stations=(('SFC',1,.5,.5,0.,10.),)
    config.upper_control_stations=()
    with pytest.raises(ValueError,match='IDIOPT5.*US records'): _validate(config)

def test_calmet_noobs2_accepts_external_gridded_cloud_data():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=2,cloud_method=2,
        cloud_data='cloud.dat',z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_method=1
    with pytest.raises(ValueError,match='MCLOUD=1'):
        _validate(config)

def test_calmet_cloud_output_controls_match_legacy_rules():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),cloud_method=4,cloud_output=True,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='CLDDAT'): _validate(config)
    config.cloud_data='cloud.dat'
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_method=2
    with pytest.raises(ValueError,match='MCLOUD=2'): _validate(config)


@pytest.mark.parametrize('threshold',[0.,-0.001,np.inf,np.nan])
def test_calmet_cloud_condensate_threshold_must_be_finite_and_positive(threshold):
    config=_prognostic_calmet_config(
        'met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),
        cloud_condensate_threshold_g_kg=threshold,
        z_faces_m=(0.,20.),smoothing_passes=(0,))

    with pytest.raises(ValueError,match='finite and positive'):
        _validate(config)


def test_calmet_itprog1_accepts_observed_surface_temperature_controls():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=1,surface_data='surf.dat',itprog=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,),temperature_interpolation_power=2,
        temperature_radius_km=20.,temperature_station_limit=3)
    assert _validate(config).tolist()==[0.,20.]
    config.temperature_interpolation_power=3
    with pytest.raises(ValueError,match='IRAD'): _validate(config)
    config.temperature_interpolation_power=1; config.temperature_station_limit=0
    with pytest.raises(ValueError,match='NUMTS'): _validate(config)
    config.temperature_station_limit=3; config.itprog=0
    with pytest.raises(ValueError,match='NOOBS=0'): _validate(config)


def test_calmet_noobs2_uses_prognostic_humidity_without_mutating_config():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),irhprog=0,wind_extrapolation=4,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    assert config.irhprog==0
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    config.irhprog=0
    assert _validate(config).tolist()==[0.,20.]

def test_calmet_prognostic_and_relaxation_timesteps_validate_data_requirements():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.prognostic_timestep_seconds=43200
    assert _validate(config).tolist()==[0.,20.]
    config.prognostic_timestep_seconds=46800
    with pytest.raises(ValueError,match='12 hours'): _validate(config)
    config.prognostic_timestep_seconds=7200; config.iprog=5; config.iwfcod=0
    assert _validate(config).tolist()==[0.,20.]
    config.prognostic_timestep_seconds=3600; config.convective_relaxation_time_s=.5
    with pytest.raises(ValueError,match='TZICRLX'): _validate(config)
    config.convective_relaxation_time_s=1.
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_selected_surface_temperature_is_observation_only():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),surface_temperature_option=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='ITPROG>0'): _validate(config)
    config.itprog=0; config.noobs=0; config.surface_data='surf.dat'; config.upper_air_files=('up.dat',)
    config.stations_per_level=(1,); config.wind_extrapolation_factors=(0.,)
    assert _validate(config).tolist()==[0.,20.]
    config.surface_temperature_option=-2
    assert _validate(config).tolist()==[0.,20.]
    with pytest.raises(ValueError,match=r'TEMPK\(-2\)'):
        _legacy_validate(config)
    config.surface_temperature_option=1
    config.upper_temperature_option=1
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_domain_prognostic_surface_temperature_is_valid_with_itprog1():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=1,surface_data='surf.dat',
        iprog=14,iwfcod=1,itprog=1,surface_temperature_option=-2,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(0.,))
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_itprog0_iupt_minus2_requires_diagnostic_lapse_rate():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,surface_data='surf.dat',
        upper_air_files=('up.dat',),itprog=0,irhprog=0,
        upper_temperature_option=-2,precipitation_station_count=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(0.,))
    with pytest.raises(ValueError,match='IDIOPT2'): _validate(config)
    config.diagnostic_lapse_rate=1
    config.diagnostic_data='diag.dat'
    assert _validate(config).tolist()==[0.,20.]

def test_calmet_positive_npsta_requires_file_but_is_valid_in_noobs_mode():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),precipitation_station_count=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='PRCDAT'): _validate(config)
    config.precipitation_data='precip.dat'
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_rprog_matches_readcf_nonzero_rule():
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=13,iwfcod=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='non-zero RPROG'): _validate(config)
    config.prognostic_radius_km=-10.
    assert _validate(config).tolist()==[0.,20.]


def test_noobs_mode_grids_positive_npsta_precipitation(monkeypatch):
    @dataclass(frozen=True)
    class Step:
        surface_temperature: np.ndarray
        precipitation_rate: np.ndarray

    record=SimpleNamespace(rate_mm_h=np.array([2.5],dtype=np.float32))
    precipitation=SimpleNamespace(at_begin=lambda timestamp: record)
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),precipitation_station_count=1,
        precipitation_data='precip.dat',z_faces_m=(0.,20.),smoothing_passes=(0,))
    transform=_noobs_step_transform(
        config,None,None,precipitation,np.array([[.5,.5]],dtype=np.float32))
    step=Step(np.array([[280.]],dtype=np.float32),np.array([[0.]],dtype=np.float32))
    result=transform(step,datetime(2018,1,1))
    np.testing.assert_array_equal(result.precipitation_rate,
                                  np.array([[2.5]],dtype=np.float32))
    config.run_type=0
    config.diagnostic_lapse_rate=1
    monkeypatch.setattr('pycalmet._calmet.grid_precipitation',
                        lambda *args,**kwargs: pytest.fail('wind-only run gridded rain'))
    wind_only=_noobs_step_transform(
        config,None,None,precipitation,np.array([[.5,.5]],dtype=np.float32))
    np.testing.assert_array_equal(
        wind_only(step,datetime(2018,1,1)).precipitation_rate,
        step.precipitation_rate)


def test_noobs_mode_preserves_sea_data_for_overwater_physics():
    @dataclass(frozen=True)
    class Step:
        surface_temperature: np.ndarray
        precipitation_rate: np.ndarray
        observed_water_mixing_height: np.ndarray | None = None
        overwater_data: object | None = None
        water_gradient_below: np.ndarray | None = None
        water_gradient_above: np.ndarray | None = None
        water_mixing_gradient_above: np.ndarray | None = None

    record=SimpleNamespace(mixing_height_m=125.,air_water_delta_k=2.,
        air_temperature_k=290.,relative_humidity=80.,anemometer_height_m=10.,
        air_sensor_height_m=4.,water_sensor_depth_m=.6,wave_period_s=8.,
        wave_height_m=1.5,gradient_below_k_m=-.008,
        gradient_above_k_m=-.004)
    sea=SimpleNamespace(at_begin=lambda timestamp: record,
        point=lambda grid: np.array([.5,.5]),
        longitude_latitude=lambda: (120.,30.))
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=2,itprog=2,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    step=Step(np.array([[290.]],dtype=np.float32),
              np.array([[0.]],dtype=np.float32))
    result=_noobs_step_transform(config,None,None,sea=(sea,))(
        step,datetime(2018,1,1))
    np.testing.assert_array_equal(result.observed_water_mixing_height,[[125.]])
    assert result.overwater_data is not None
    np.testing.assert_allclose(result.overwater_data.sea_temperature_k,[[288.]])
    np.testing.assert_allclose(result.water_gradient_below,[[-.008]])
    np.testing.assert_allclose(result.water_gradient_above,[[-.004]])
    np.testing.assert_allclose(result.water_mixing_gradient_above,[[-.004]])


def test_noobs_waterp_keeps_prognostic_thermodynamics_and_merges_only_waves():
    @dataclass(frozen=True)
    class Step:
        surface_temperature: np.ndarray
        precipitation_rate: np.ndarray
        observed_water_mixing_height: np.ndarray | None = None
        overwater_data: object | None = None

    record=SimpleNamespace(mixing_height_m=125.,air_water_delta_k=9999.,
        air_temperature_k=290.,relative_humidity=80.,anemometer_height_m=10.,
        air_sensor_height_m=4.,water_sensor_depth_m=.6,wave_period_s=9.,
        wave_height_m=2.,gradient_below_k_m=-.008,gradient_above_k_m=-.004)
    sea=SimpleNamespace(at_begin=lambda timestamp: record,
        point=lambda grid: np.array([.5,.5]),longitude_latitude=lambda: (120.,30.))
    config=_prognostic_calmet_config('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=2,itprog=2,water_temperature_mode=2,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    shape=(1,1)
    prognostic=GriddedSeaData(*[np.full(shape,value,dtype=np.float32) for value in
        (281.,285.,77.,12.,6.,.05,-999.,-999.,118.)])
    step=Step(np.array([[290.]],dtype=np.float32),np.array([[0.]],dtype=np.float32),
              overwater_data=prognostic)
    result=_noobs_step_transform(config,None,None,sea=(sea,))(
        step,datetime(2018,1,1))
    np.testing.assert_array_equal(result.overwater_data.air_temperature_k,[[281.]])
    np.testing.assert_array_equal(result.overwater_data.sea_temperature_k,[[285.]])
    np.testing.assert_array_equal(result.overwater_data.water_sensor_depth_m,
                                  np.array([[.05]],dtype=np.float32))
    np.testing.assert_array_equal(result.overwater_data.wave_period_s,[[9.]])
    np.testing.assert_array_equal(result.overwater_data.wave_height_m,[[2.]])

def test_calmet_rejects_mismatched_geophysical_grid_metadata(tmp_path):
    path=tmp_path/'geo.nc'; grid=pycalmet.GridConfig(2,3,1.,1.,x_origin_km=4.,datum='WGS-84')
    with Dataset(path,'w') as dataset:
        dataset.grid_nx=2; dataset.grid_ny=3; dataset.grid_dx_km=1.; dataset.x_origin_km=5.
        dataset.projection='LCC'; dataset.datum='WGS-84'; dataset.hemisphere='N'
    with Dataset(path) as dataset:
        with pytest.raises(ValueError,match='x_origin_km'):
            _validate_geophysical_grid(dataset,grid)

def test_calmet_accepts_legacy_geo_without_optional_grid_attributes(tmp_path):
    path=tmp_path/'geo.nc'
    with Dataset(path,'w'): pass
    with Dataset(path) as dataset:
        _validate_geophysical_grid(dataset,pycalmet.GridConfig(1,1,1.,1.))


def test_calmet_accepts_equivalent_geophysical_datum_spelling(tmp_path):
    path=tmp_path/'geo.nc'
    with Dataset(path,'w') as dataset: dataset.datum='WGS84'
    with Dataset(path) as dataset:
        _validate_geophysical_grid(
            dataset,pycalmet.GridConfig(1,1,1.,1.,datum='WGS-84'))


def test_calmet_ignores_utm_metadata_for_lcc_geophysical_grid(tmp_path):
    path=tmp_path/'geo.nc'
    grid=pycalmet.GridConfig(1,1,1.,1.,projection='LCC',utm_zone=50,hemisphere='N')
    with Dataset(path,'w') as dataset:
        dataset.projection='LCC'; dataset.datum='WGS-84'
        dataset.utm_zone=0; dataset.hemisphere='S'
    with Dataset(path) as dataset:
        _validate_geophysical_grid(dataset,grid)


def test_calmet_checks_utm_metadata_for_utm_geophysical_grid(tmp_path):
    path=tmp_path/'geo.nc'
    grid=pycalmet.GridConfig(1,1,1.,1.,projection='UTM',utm_zone=50,hemisphere='N')
    with Dataset(path,'w') as dataset:
        dataset.projection='UTM'; dataset.datum='WGS-84'
        dataset.utm_zone=49; dataset.hemisphere='S'
    with Dataset(path) as dataset:
        with pytest.raises(ValueError,match='utm_zone'):
            _validate_geophysical_grid(dataset,grid)

def test_prognostic_precipitation_radius_uses_projected_fp64_spacing():
    mapper=SimpleNamespace(source_shape=(2,2),source_coordinates=np.asarray(
        [[0.,0.],[0.,9.],[9.0125,.06],[9.0125,9.06]],dtype=np.float64))
    dx=np.linalg.norm(mapper.source_coordinates[0]-mapper.source_coordinates[2])
    dy=np.linalg.norm(mapper.source_coordinates[0]-mapper.source_coordinates[1])
    expected=float(np.hypot(dx,dy)*1.05)
    assert _prognostic_precipitation_radius(mapper,50.,-1)==expected
    assert _prognostic_precipitation_radius(mapper,10.,-1)==10.
    assert _prognostic_precipitation_radius(mapper,50.,0)==50.


def test_prognostic_precipitation_radius_no_longer_uses_legacy_fp32_boundary():
    spacing=10.0000004
    requested=10.0000002
    mapper=SimpleNamespace(source_shape=(2,2),source_coordinates=np.asarray(
        [[0.,0.],[0.,1.],[spacing,0.],[spacing,1.]],dtype=np.float64))

    expected=float(np.hypot(spacing,1.)*1.05)
    assert _prognostic_precipitation_radius(mapper,requested,-1)==expected
    assert _legacy_prognostic_precipitation_radius(mapper,requested,-1)==10.


def test_prognostic_precipitation_radius_uses_both_rectangular_grid_spacings():
    mapper=SimpleNamespace(source_shape=(2,2),source_coordinates=np.asarray(
        [[0.,0.],[0.,2.],[5.,0.],[5.,2.]],dtype=np.float64))
    expected=np.hypot(5.,2.)*1.05
    # Three kilometres exceeds dy but cannot span the five-kilometre x edge,
    # so the automatic radius covers the complete source-cell diagonal.
    assert _prognostic_precipitation_radius(mapper,3.,-1)==expected
    assert _prognostic_precipitation_radius(mapper,5.2,-1)==5.2
