from dataclasses import dataclass
from datetime import datetime
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
from pycalmet.sea_data import GriddedSeaData


def _sea_metadata(*,version='2.11',timezone='UTC-0000',**changes):
    values=dict(version=version,timezone=timezone,
        source_grid=pycalmet.GridConfig(1,1,1.,1.,projection='LCC',
            lat0=30.,lon0=-120.,lat1=30.,lat2=60.),
        projection='LCC',datum='WGS-84',datum_date='20200101')
    values.update(changes)
    return SimpleNamespace(**values)


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError, match="unknown CalwrfConfig parameters"):
        pycalmet.calwrf({"not_a_parameter": 1})


def test_fortran_invalid_calmet_wind_pair_fails_explicitly():
    with pytest.raises(ValueError, match="does not allow"):
        pycalmet.calmet({
            "meteorology": "calwrf.nc", "geophysical": "geo.nc", "output": "calmet.nc",
            "start": "2018-03-18_00:00:00", "end": "2018-03-18_01:00:00",
            "grid": pycalmet.GridConfig(nx=1, ny=1, dx_km=1, dy_km=1),
            "iprog": 13,
        })

def test_calmet_rejects_nonlegacy_wind_codes_and_rectangular_cells():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.iprog=6
    with pytest.raises(ValueError,match='IPROG must be'): _validate(config)
    config.iprog=14; config.iwfcod=2
    with pytest.raises(ValueError,match='IWFCOD must be'): _validate(config)
    config.iwfcod=1; config.grid.dy_km=2.
    with pytest.raises(ValueError,match='DGRID'): _validate(config)


def test_calmet_first_layer_midpoint_check_exempts_wind_only_runs():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,30.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='first layer midpoint'): _validate(config)
    config.run_type=0
    assert _validate(config).tolist()==[0.,30.]


def test_calmet_projection_qa_matches_fortran_set():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
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
    config.grid.lat0=27.; config.start='2018010105'
    with pytest.raises(ValueError,match='before 05 LST'): _validate(config)
    config.start='2018010104'
    assert _validate(config).tolist()==[0.,20.]
    config.start='2018010105'
    config.run_type=0
    assert _validate(config).tolist()==[0.,20.]
    config.start='2018-01-01_00:00:01'
    with pytest.raises(ValueError,match='NSECDT'): _validate(config)

def test_calmet_ignores_mixing_level_when_upwind_average_is_disabled():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iavezi=0,mixing_wind_level=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.iavezi=1
    with pytest.raises(ValueError,match='mixing_wind_level'): _validate(config)


def test_calmet_accepts_legacy_output_and_setup_controls():
    config=CalmetConfig('met.nc','geo.nc','out.dat','2018010100','2018010101',
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


def test_calmet_rejects_unported_print_and_debug_outputs():
    config=CalmetConfig('met.nc','geo.nc','out.dat','2018010100','2018010101',
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
    with pytest.raises(ValueError,match='undefined TINF'):
        _validate(config)
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
    with pytest.raises(ValueError,match='FORMAT 2820'):
        _validate(config)
    config.wind_debug_fields=(0,0,1,0,0,0,0,0,0)
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_selects_pacdat_only_for_mesopac_output():
    config=CalmetConfig('met.nc','geo.nc','calmet.dat','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=13,iwfcod=0,prognostic_radius_km=10.,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    with pytest.raises(ValueError,match='WTDAT'): _validate(config)
    config.weight_data='wt.dat'; config.prognostic_radius_km=10.
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_iprog15_noobs_prognostic_observations_are_supported():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=15,iwfcod=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='requires IDIOPT3=1'):
        _validate(config)
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
    config.iprog=13; config.iwfcod=0
    with pytest.raises(ValueError,match='does not allow IGFMET'):
        _validate(config)


def test_calmet_iprog15_objective_analysis_mode_is_supported():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=15,iwfcod=0,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_mm4_wind_modes_map_to_their_mm5_counterparts():
    grid=pycalmet.GridConfig(1,1,1.,1.)
    for iprog,iwfcod in ((3,0),(4,1),(5,0)):
        config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
            grid,iprog=iprog,iwfcod=iwfcod,
            prognostic_radius_km=(10. if iprog==3 else 0.),
            z_faces_m=(0.,20.),smoothing_passes=(0,))
        assert _validate(config).tolist()==[0.,20.]
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    first=CalmetConfig(**common,iprog=1,iwfcod=0,prognostic_radius_km=10.)
    second=CalmetConfig(**common,iprog=2,iwfcod=1)
    assert _validate(first).tolist()==[0.,20.]
    assert _validate(second).tolist()==[0.,20.]
    first.prognostic_radius_km=0.
    with pytest.raises(ValueError,match='RPROG'):
        _validate(first)
    second.timestep_seconds=900
    assert _validate(second).tolist()==[0.,20.]


def test_calmet_iprog0_wind1_observation_mode_is_supported():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),iprog=0,iwfcod=1,noobs=0,
        surface_data='surf.dat',upper_air_files=('up.dat',),itprog=0,irhprog=0,
        precipitation_station_count=0,wind_extrapolation=4,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,))
    np.testing.assert_array_equal(_validate(config),[0.,20.])


def test_calmet_selected_upper_initial_wind_controls_are_validated():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,surface_data='surf.dat',
        upper_air_files=('up.dat',),z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,),
        diagnostic_initial_wind=1)
    with pytest.raises(ValueError,match='DIADAT'): _validate(config)
    config.diagnostic_data='diag.dat'; config.diagnostic_initial_wind=0
    config.diagnostic_surface_wind=1
    with pytest.raises(ValueError,match='IRTYPE=0'): _validate(config)
    config.run_type=0
    config.surface_control_stations=(('SFC',1,.5,.5,0.,10.),)
    assert _validate(config).tolist()==[0.,20.]
    config.diagnostic_surface_wind=0; config.diagnostic_surface_temperature=1
    with pytest.raises(ValueError,match='ISURFT=-1'): _validate(config)


def test_calmet_initial_guess_file_controls_follow_fortran_compatibility():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),overwater_flux_method=-11,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    old=_sea_metadata(version='2.00',timezone=None)
    with pytest.raises(ValueError,match='negative ICOARE.*2.1'):
        _validate_sea_metadata(config,(old,))
    modern=_sea_metadata(version='2.10',timezone=None)
    _validate_sea_metadata(config,(modern,))


def test_sea_metadata_timezone_uses_calmet_positive_west_convention():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),base_timezone=8,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    _validate_sea_metadata(config,(_sea_metadata(timezone='UTC-0800'),))
    _validate_sea_metadata(config,(_sea_metadata(timezone='UTC-0830'),))
    with pytest.raises(ValueError,match='timezone'):
        _validate_sea_metadata(config,(_sea_metadata(timezone='UTC+0800'),))
    with pytest.raises(ValueError,match='invalid SEA.DAT timezone'):
        _validate_sea_metadata(config,(_sea_metadata(timezone='PST'),))


def test_surface_and_precip_timezones_accept_modern_fractional_and_old_integer_forms():
    _validate_input_timezone('UTC-0830',8,'SURF.DAT')
    _validate_input_timezone(8,8,'PRECIP.DAT')
    with pytest.raises(ValueError,match='PRECIP.DAT timezone'):
        _validate_input_timezone(-8,8,'PRECIP.DAT')
    with pytest.raises(ValueError,match='invalid SURF.DAT timezone'):
        _validate_input_timezone('UTC-ABCD',0,'SURF.DAT')


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


def test_upper_air_equal_end_hour_preserves_readhd_begin_second_quirk():
    run_start=datetime(2018,1,1,0,30); run_end=datetime(2018,1,1,2)
    data=SimpleNamespace(start=datetime(2018,1,1,0),
                         end=datetime(2018,1,1,2,59))
    with pytest.raises(ValueError,match='run ending'):
        _validate_upper_span(data,run_start,run_end,0)


def test_sea_files_require_shared_projection_and_datum_metadata():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=0,z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(1.,))
    with pytest.raises(ValueError,match='SRFDAT'): _validate(config)
    config.surface_data='surf.dat'
    with pytest.raises(ValueError,match='UPDAT'): _validate(config)
    config.upper_air_files=('up.dat',)
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_idiopt4_5_accept_control_metadata_without_observation_files():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=2,cloud_method=2,
        cloud_data='cloud.dat',z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_method=1
    with pytest.raises(ValueError,match='MCLOUD=1'):
        _validate(config)

def test_calmet_cloud_output_controls_match_legacy_rules():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),cloud_method=4,cloud_output=True,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='CLDDAT'): _validate(config)
    config.cloud_data='cloud.dat'
    assert _validate(config).tolist()==[0.,20.]
    config.cloud_method=2
    with pytest.raises(ValueError,match='MCLOUD=2'): _validate(config)


def test_calmet_itprog1_accepts_observed_surface_temperature_controls():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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


def test_calmet_irhprog0_is_reset_in_full_noobs_mode():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),irhprog=0,z_faces_m=(0.,20.),smoothing_passes=(0,))
    assert _validate(config).tolist()==[0.,20.]
    assert config.irhprog==1
    config.noobs=1; config.surface_data='surf.dat'; config.stations_per_level=(1,)
    config.wind_extrapolation_factors=(0.,)
    config.irhprog=0
    assert _validate(config).tolist()==[0.,20.]

def test_calmet_prognostic_and_relaxation_timesteps_match_readcf_limits():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),smoothing_passes=(0,))
    config.prognostic_timestep_seconds=43200
    assert _validate(config).tolist()==[0.,20.]
    config.prognostic_timestep_seconds=46800
    with pytest.raises(ValueError,match='12 hours'): _validate(config)
    config.prognostic_timestep_seconds=7200; config.iprog=5; config.iwfcod=0
    with pytest.raises(ValueError,match='ISTEPPGS=3600'): _validate(config)
    config.prognostic_timestep_seconds=3600; config.convective_relaxation_time_s=.5
    with pytest.raises(ValueError,match='TZICRLX'): _validate(config)
    config.convective_relaxation_time_s=1.
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_selected_surface_temperature_is_observation_only():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),surface_temperature_option=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='ITPROG>0'): _validate(config)
    config.itprog=0; config.noobs=0; config.surface_data='surf.dat'; config.upper_air_files=('up.dat',)
    config.stations_per_level=(1,); config.wind_extrapolation_factors=(0.,)
    assert _validate(config).tolist()==[0.,20.]
    config.surface_temperature_option=-2
    with pytest.raises(ValueError,match=r'TEMPK\(-2\)'): _validate(config)
    config.surface_temperature_option=1
    config.upper_temperature_option=1
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_domain_prognostic_surface_temperature_is_valid_with_itprog1():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),noobs=1,surface_data='surf.dat',
        iprog=14,iwfcod=1,itprog=1,surface_temperature_option=-2,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(0.,))
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_itprog0_iupt_minus2_requires_diagnostic_lapse_rate():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
        pycalmet.GridConfig(1,1,1.,1.),precipitation_station_count=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,))
    with pytest.raises(ValueError,match='PRCDAT'): _validate(config)
    config.precipitation_data='precip.dat'
    assert _validate(config).tolist()==[0.,20.]


def test_calmet_rprog_matches_readcf_nonzero_rule():
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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
    config=CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
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

def test_prognostic_precipitation_radius_matches_readhd_guard():
    mapper=SimpleNamespace(source_shape=(2,2),source_coordinates=np.asarray(
        [[0.,0.],[0.,9.],[9.0125,.06],[9.0125,9.06]],dtype=np.float32))
    dx=np.float32(9.0125); dy=np.float32(.06)
    spacing=np.sqrt(np.float32(dx*dx+dy*dy))
    expected=float(np.float32(np.sqrt(np.float32(2.))*spacing*np.float32(1.05)))
    assert _prognostic_precipitation_radius(mapper,50.,-1)==expected
    assert _prognostic_precipitation_radius(mapper,10.,-1)==10.
    assert _prognostic_precipitation_radius(mapper,50.,0)==50.
