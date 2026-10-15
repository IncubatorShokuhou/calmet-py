from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import pycalmet
from pycalmet.config import CalwrfConfig
from pycalmet.legacy import CalwrfConfig as LegacyCalwrfConfig
from pycalmet._calwrf import run as run_calwrf
from pycalmet.legacy._compat._calwrf import run as run_legacy_calwrf
from pycalmet.legacy._compat._calwrf import (
    _horizontal_range_check as _legacy_horizontal_range_check,
    _rain_increments as _legacy_rain_increments,
    _reset_total_rain_at_file_boundary as _legacy_reset_total_rain_at_file_boundary,
)
from pycalmet._calwrf import (_cone,_create_output,_geopotential_height,_horizontal_range_check,_mass_coordinate,_pressure_temperature,_rain_increment,_rain_increments,_select_wrf_times,
                              _selection,_wrf_timestamp,_surface_variable,_surface_wind,
                              _var,_vertical_range_check,_volume_variable,
                              _write_time_text,destagger_x,
                              destagger_y,destagger_z,earth_wind,rotation_angle,
                              speed_direction)

ROOT=Path(__file__).resolve().parents[1]
CALWRF_CONTROL=Path(__file__).with_name('fixtures')/'calwrf.inp'


def _write_minimal_calwrf_wrf(path, timestamp="2018-03-18_00:00:00"):
    """Create the smallest WRF file that exercises a complete CALWRF run."""
    with nc.Dataset(path, "w") as dataset:
        for name, size in (
            ("Time", 1), ("DateStrLen", 19), ("bottom_top", 1),
            ("bottom_top_stag", 2), ("south_north", 1),
            ("south_north_stag", 2), ("west_east", 1),
            ("west_east_stag", 2),
        ):
            dataset.createDimension(name, size)
        dataset.DX=1000.; dataset.DY=1000.; dataset.MAP_PROJ=6
        dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))[:]=(
            np.asarray([list(timestamp)], dtype="S1")
        )
        surface_dims=("Time", "south_north", "west_east")
        for name, value in (
            ("XLAT", 30.), ("XLONG", 120.), ("HGT", 10.),
            ("LU_INDEX", 1.), ("RAINNC", 1.), ("PSFC", 100000.),
            ("SNOWC", 0.),
        ):
            dataset.createVariable(name, "f4", surface_dims)[:]=value
        mass_dims=("Time", "bottom_top", "south_north", "west_east")
        for name, value in (("P", 0.), ("PB", 100000.), ("T", 0.),
                            ("QVAPOR", .01)):
            dataset.createVariable(name, "f4", mass_dims)[:]=value
        staggered_z=("Time", "bottom_top_stag", "south_north", "west_east")
        dataset.createVariable("PH", "f4", staggered_z)[:]=0.
        dataset.createVariable("PHB", "f4", staggered_z)[:]=(
            np.asarray([0., 196.2], dtype="f4").reshape(1, 2, 1, 1)
        )
        dataset.createVariable("W", "f4", staggered_z)[:]=0.
        dataset.createVariable(
            "U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"),
        )[:]=1.
        dataset.createVariable(
            "V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"),
        )[:]=0.
        dataset.createVariable("ZNU", "f4", ("Time", "bottom_top"))[:]=.5


@pytest.mark.parametrize(
    ("entrypoint", "run_target"),
    (
        (pycalmet.calwrf, "pycalmet._calwrf.run"),
        (pycalmet.legacy.calwrf, "pycalmet.legacy._compat._calwrf.run"),
    ),
    ids=("public", "legacy"),
)
def test_calwrf_fio_netcdf_is_only_the_canonical_output(
        monkeypatch, tmp_path, entrypoint, run_target):
    monkeypatch.setattr(run_target, lambda config: config)
    target = tmp_path / "calwrf.nc"
    config = entrypoint(
        FLNAMES=[tmp_path / "wrf.nc"], NFILES=1,
        IDATEB="2018010100", IDATEE="2018010101", FIO=target,
    )
    assert config.output == target
    assert config.legacy_output is None
    assert config.surface_output == target.with_suffix(".m2d")


@pytest.mark.parametrize(
    "config_type", (CalwrfConfig, LegacyCalwrfConfig), ids=("public", "legacy"),
)
@pytest.mark.parametrize(
    ("first", "second"),
    (
        ("output", "legacy_output"),
        ("output", "surface_output"),
        ("legacy_output", "surface_output"),
        ("output", "listing_output"),
        ("legacy_output", "listing_output"),
        ("surface_output", "listing_output"),
    ),
)
def test_calwrf_config_rejects_colliding_outputs(
        tmp_path, config_type, first, second):
    paths = {
        "output": tmp_path / "calwrf.nc",
        "legacy_output": tmp_path / "calwrf.m3d",
        "surface_output": tmp_path / "calwrf.m2d",
        "listing_output": tmp_path / "calwrf.lst",
    }
    paths[second] = paths[first].parent / "unused" / ".." / paths[first].name
    with pytest.raises(ValueError, match="CALWRF output paths collide"):
        config_type(
            [tmp_path / "wrf.nc"], "2018010100", "2018010101",
            **paths,
        )


@pytest.mark.parametrize(
    "config_type", (CalwrfConfig, LegacyCalwrfConfig), ids=("public", "legacy"),
)
@pytest.mark.parametrize(
    "output_name", ("output", "legacy_output", "surface_output", "listing_output"),
)
def test_calwrf_config_rejects_every_output_that_aliases_a_wrf_input(
        tmp_path, config_type, output_name):
    inputs = [tmp_path / "first-wrf.nc", tmp_path / "second-wrf.nc"]
    paths = {
        "output": tmp_path / "calwrf.nc",
        "legacy_output": tmp_path / "calwrf.m3d",
        "surface_output": tmp_path / "calwrf.m2d",
        "listing_output": tmp_path / "calwrf.lst",
    }
    paths[output_name] = inputs[1]
    with pytest.raises(ValueError, match=rf"{output_name} conflicts with wrf_files\[1\]"):
        config_type(inputs, "2018010100", "2018010101", **paths)


@pytest.mark.parametrize(
    ("config_type", "runner"),
    (
        (CalwrfConfig, run_calwrf),
        (LegacyCalwrfConfig, run_legacy_calwrf),
    ),
    ids=("public", "legacy"),
)
def test_calwrf_run_rechecks_paths_after_config_mutation(
        tmp_path, config_type, runner):
    source = tmp_path / "wrf.nc"
    config = config_type(
        [source], "2018010100", "2018010101", tmp_path / "calwrf.nc",
    )
    config.output = source
    with pytest.raises(ValueError, match=r"output conflicts with wrf_files\[0\]"):
        runner(config)


@pytest.mark.parametrize(
    ("config_type", "runner"),
    (
        (CalwrfConfig, run_calwrf),
        (LegacyCalwrfConfig, run_legacy_calwrf),
    ),
    ids=("public", "legacy"),
)
def test_calwrf_listing_is_written_after_a_successful_run(
        tmp_path, config_type, runner):
    source=tmp_path/"wrf.nc"
    output=tmp_path/"calwrf.nc"
    listing=tmp_path/"reports"/"calwrf.lst"
    _write_minimal_calwrf_wrf(source)
    result=runner(config_type(
        [source], "2018031800", "2018031800", output,
        title="测试 CALWRF", listing_output=listing,
    ))
    assert result==output
    assert output.is_file()
    text=listing.read_text(encoding="utf-8")
    assert "PYCALMET CALWRF RUN LISTING" in text
    assert "Title: 测试 CALWRF" in text
    assert f"  - {source}" in text
    assert "Requested start: 2018-03-18 00:00:00" in text
    assert "Requested end:   2018-03-18 00:00:00" in text
    assert "Time records written: 1" in text
    assert "Output grid (nx, ny, nz): 1 x 1 x 1" in text
    assert f"Main NetCDF output: {output}" in text
    assert "Legacy M3D output:" not in text
    assert "Surface M2D output:" not in text


@pytest.mark.parametrize(
    ("config_type", "runner", "write_target"),
    (
        (CalwrfConfig, run_calwrf, "pycalmet._calwrf.write_m3d"),
        (LegacyCalwrfConfig, run_legacy_calwrf,
         "pycalmet.legacy._compat._calwrf.write_m3d"),
    ),
    ids=("public", "legacy"),
)
def test_calwrf_listing_records_only_successfully_created_legacy_outputs(
        monkeypatch, tmp_path, config_type, runner, write_target):
    source=tmp_path/"wrf.nc"
    output=tmp_path/"calwrf.nc"
    legacy_output=tmp_path/"calwrf.m3d"
    surface_output=tmp_path/"calwrf.m2d"
    listing=tmp_path/"calwrf.lst"
    _write_minimal_calwrf_wrf(source)

    def write_legacy(canonical, legacy, surface):
        assert Path(canonical)==output
        assert not listing.exists()
        Path(legacy).write_bytes(b"M3D")
        Path(surface).write_bytes(b"M2D")

    monkeypatch.setattr(write_target, write_legacy)
    runner(config_type(
        [source], "2018031800", "2018031800", output,
        legacy_output=legacy_output, surface_output=surface_output,
        listing_output=listing,
    ))
    text=listing.read_text(encoding="utf-8")
    assert f"Legacy M3D output: {legacy_output}" in text
    assert f"Surface M2D output: {surface_output}" in text


@pytest.mark.parametrize(
    ("config_type", "runner"),
    (
        (CalwrfConfig, run_calwrf),
        (LegacyCalwrfConfig, run_legacy_calwrf),
    ),
    ids=("public", "legacy"),
)
def test_calwrf_listing_collision_fails_before_touching_wrf_input(
        tmp_path, config_type, runner):
    source=tmp_path/"wrf.nc"
    original=b"WRF input must remain unchanged"
    source.write_bytes(original)
    output=tmp_path/"calwrf.nc"
    config=config_type(
        [source], "2018031800", "2018031800", output,
        listing_output=tmp_path/"calwrf.lst",
    )
    config.listing_output=source
    with pytest.raises(ValueError, match=r"listing_output conflicts with wrf_files\[0\]"):
        runner(config)
    assert source.read_bytes()==original
    assert not output.exists()


@pytest.mark.parametrize(
    ("config_type", "runner"),
    (
        (CalwrfConfig, run_calwrf),
        (LegacyCalwrfConfig, run_legacy_calwrf),
    ),
    ids=("public", "legacy"),
)
@pytest.mark.parametrize("surface_alias", ("input", "output"))
def test_calwrf_ignores_inactive_surface_output_for_path_validation(
        tmp_path, config_type, runner, surface_alias):
    source=tmp_path/"wrf.nc"
    output=tmp_path/"calwrf.nc"
    surface_output=source if surface_alias=="input" else output
    config=config_type(
        [source], "2018031800", "2018031800", output,
        legacy_output=tmp_path/"calwrf.m3d", surface_output=surface_output,
        create_surface=False,
    )
    with pytest.raises(FileNotFoundError, match="wrf.nc"):
        runner(config)


def test_public_calwrf_component_module_delegates_to_verified_engine(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr('pycalmet._calwrf.run',
                        lambda config: captured.append(config) or Path(config.output))
    output=tmp_path/'calwrf.nc'
    result=pycalmet.calwrf(
        WRF_FILES=[tmp_path/'wrf.nc'], START='2018010100',
        END='2018010101', OUTPUT=output,
    )
    assert result==output
    assert isinstance(captured[0],CalwrfConfig)
    assert captured[0].wrf_files==[tmp_path/'wrf.nc']
    # Importing and using the component must not replace the package function.
    from pycalmet.calwrf import run
    assert run(captured[0])==output
    assert callable(pycalmet.calwrf)


def test_public_calwrf_mapping_keys_are_case_insensitive(monkeypatch,tmp_path):
    monkeypatch.setattr('pycalmet._calwrf.run',lambda config: config)
    config=pycalmet.calwrf({
        'WrF_FiLeS':[tmp_path/'wrf.nc'], 'sTaRt':'2018010100',
        'EnD':'2018010101', 'OuTpUt':tmp_path/'calwrf.nc',
    })
    assert isinstance(config,CalwrfConfig)
    assert config.output==tmp_path/'calwrf.nc'


def test_public_calwrf_control_overrides_are_case_insensitive(monkeypatch,tmp_path):
    monkeypatch.setattr('pycalmet._calwrf.run',lambda config: config)
    output=tmp_path/'redirected.nc'
    config=pycalmet.calwrf(CALWRF_CONTROL,OUTPUT=output)
    assert config.output==output
    assert config.legacy_output is None
    assert config.surface_output is None


def test_public_calwrf_rejects_case_duplicate_parameters(monkeypatch,tmp_path):
    monkeypatch.setattr('pycalmet._calwrf.run',lambda config: config)
    with pytest.raises(TypeError,match="duplicate CalwrfConfig parameter 'output'"):
        pycalmet.calwrf({
            'wrf_files':[tmp_path/'wrf.nc'], 'start':'2018010100',
            'end':'2018010101', 'output':tmp_path/'first.nc',
            'OUTPUT':tmp_path/'second.nc',
        })


def test_calwrf_accepts_legacy_and_iso_timestamps():
    assert _wrf_timestamp('2018031800')==_wrf_timestamp('2018-03-18_00:00:00')


def test_calwrf_inclusive_one_based_subdomain_selection():
    assert _selection(None,5,'I')==slice(0,5)
    assert _selection((2,4),5,'I')==slice(1,4)
    with pytest.raises(ValueError,match='outside'):
        _selection((0,4),5,'I')


def test_calwrf_netcdf_compression_switch(tmp_path):
    path=tmp_path/'compression.nc'
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('x',1)
        plain=_var(dataset,'plain',('x',),'1',0)
        compressed=_var(dataset,'compressed',('x',),'1',4)
        assert not plain.filters()['zlib']
        assert compressed.filters()['zlib'] and compressed.filters()['complevel']==4


def test_canonical_output_always_contains_embedded_surface_fields(tmp_path):
    source_path=tmp_path/'wrf.nc'; output_path=tmp_path/'calwrf.nc'
    with nc.Dataset(source_path,'w') as source:
        source.createDimension('Time',1); source.createDimension('bottom_top',1)
        source.createDimension('south_north',1); source.createDimension('west_east',1)
        source.DX=1000.; source.DY=1000.
        for name in ('XLAT','XLONG','HGT','LU_INDEX'):
            source.createVariable(name,'f4',('Time','south_north','west_east'))[:]=1.
        source.createVariable('ZNU','f4',('Time','bottom_top'))[:]=1.
    with nc.Dataset(source_path) as source:
        output=_create_output(output_path,source,1,1,1,1,
                              slice(0,1),slice(0,1),slice(0,1))
        output.close()
    with nc.Dataset(output_path) as output:
        for name in ('surface_pressure','precipitation','snow_cover','temperature_2m',
                     'wind_speed_10m','sst','pbl_height'):
            assert name in output.variables
        assert {name:len(output.dimensions[name]) for name in ('time','level','y','x')} == {
            'time':1, 'level':1, 'y':1, 'x':1,
        }
        assert output['surface_pressure'].dimensions == ('time','y','x')


def test_time_text_writes_real_netcdf_character_variable(tmp_path):
    path=tmp_path/'time.nc'
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('time',1); dataset.createDimension('width',19)
        variable=dataset.createVariable('time_text','S1',('time','width'))
        _write_time_text(variable,0,'2018-03-18_00:00:00')
        with pytest.raises(ValueError,match='19 ASCII bytes'):
            _write_time_text(variable,0,'short')
    with nc.Dataset(path) as dataset:
        assert nc.chartostring(dataset['time_text'][:]).item()=='2018-03-18_00:00:00'


def test_mass_coordinate_uses_wrf_m_alias_and_reports_missing(tmp_path):
    path=tmp_path/'coordinates.nc'
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('time',1); dataset.createDimension('y',1); dataset.createDimension('x',2)
        dataset.createVariable('XLAT_M','f4',('time','y','x'))[:]=[[[30.,31.]]]
    with nc.Dataset(path) as dataset:
        np.testing.assert_array_equal(_mass_coordinate(dataset,'XLAT'),[[30.,31.]])
        with pytest.raises(ValueError,match='neither XLONG nor XLONG_M'):
            _mass_coordinate(dataset,'XLONG')


def test_rain_increment_uses_full_first_accumulation_without_prior_time():
    current=np.array([[12.,20.]],dtype='f4')
    np.testing.assert_array_equal(_rain_increment(current,None),current)
    np.testing.assert_array_equal(_rain_increment(current,np.array([[5.,7.]],dtype='f4')),
                                  [[7.,13.]])
    np.testing.assert_array_equal(_rain_increment([[1.]],[[5.]]),[[1.]])


def test_rain_increments_continue_across_wrf_file_boundaries_by_default():
    previous=(np.array([[12.,20.]],dtype='f4'),
              np.array([[1.,2.]],dtype='f4'))
    total,grid,convective=_rain_increments(
        np.array([[14.5,20.25]],dtype='f4'),
        np.array([[1.25,2.5]],dtype='f4'),previous)
    np.testing.assert_array_equal(total,[[2.75,.75]])
    np.testing.assert_array_equal(grid,[[2.5,.25]])
    np.testing.assert_array_equal(convective,[[.25,.5]])


def test_rain_increments_detect_component_counter_resets():
    previous=(np.array([[12.,20.]],dtype='f4'),
              np.array([[1.,2.]],dtype='f4'))
    rainnc=np.array([[2.5,20.25]],dtype='f4')
    rainc=np.array([[1.25,.5]],dtype='f4')
    total,grid,convective=_rain_increments(rainnc,rainc,previous)
    np.testing.assert_array_equal(total,[[2.75,.75]])
    np.testing.assert_array_equal(grid,[[2.5,.25]])
    np.testing.assert_array_equal(convective,[[.25,.5]])


def test_legacy_total_rain_reset_remains_available_in_legacy_namespace():
    previous=(np.array([[1.]],dtype='f4'),np.array([[0.]],dtype='f4'))
    first=Path('first.nc'); second=Path('second.nc')
    assert not _legacy_reset_total_rain_at_file_boundary(False,previous,first,second)
    assert not _legacy_reset_total_rain_at_file_boundary(True,None,first,second)
    assert not _legacy_reset_total_rain_at_file_boundary(True,previous,first,first)
    assert _legacy_reset_total_rain_at_file_boundary(True,previous,first,second)
    total,grid,convective=_legacy_rain_increments(
        np.array([[2.]],dtype='f4'),np.array([[.5]],dtype='f4'),previous,
        reset_total=True,
    )
    np.testing.assert_array_equal(total,[[2.5]])
    np.testing.assert_array_equal(grid,[[1.]])
    np.testing.assert_array_equal(convective,[[.5]])


def test_optional_calwrf_surface_variable_is_zero_but_required_one_fails(tmp_path):
    path=tmp_path/'surface.nc'
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('time',1); dataset.createDimension('y',1); dataset.createDimension('x',2)
        dataset.createVariable('T2','f4',('time','y','x'))[:]=[[[280.,281.]]]
    with nc.Dataset(path) as dataset:
        np.testing.assert_array_equal(_surface_variable(dataset,0,'T2',(1,2)),[[280.,281.]])
        np.testing.assert_array_equal(_surface_variable(dataset,0,'SWDOWN',(1,2)),
                                      np.zeros((1,2),dtype='f4'))
        with pytest.raises(ValueError,match='required.*SNOWC'):
            _surface_variable(dataset,0,'SNOWC',(1,2),required=True)
        speed,direction=_surface_wind(dataset,0,(1,2),np.zeros((1,2),dtype='f4'))
        np.testing.assert_array_equal(speed,0.)
        # Missing U10 skips CALWRF's wind calculation, whereas a present
        # zero-valued U10/V10 pair is represented as direction 360 degrees.
        np.testing.assert_array_equal(direction,0.)


def test_qvapor_is_required_but_hydrometeors_are_optional(tmp_path):
    path=tmp_path/'moisture.nc'
    with nc.Dataset(path,'w') as dataset:
        dataset.createDimension('time',1); dataset.createDimension('z',1)
        dataset.createDimension('y',1); dataset.createDimension('x',2)
        dataset.createVariable('QVAPOR','f4',('time','z','y','x'))[:]=.01
    with nc.Dataset(path) as dataset:
        shape=(1,1,2)
        np.testing.assert_array_equal(
            _volume_variable(dataset,0,'QVAPOR',shape,required=True),
            np.full(shape,.01,dtype='f4'))
        np.testing.assert_array_equal(
            _volume_variable(dataset,0,'QCLOUD',shape),np.zeros(shape,dtype='f4'))
        with pytest.raises(ValueError,match='required 3-D variable QRAIN'):
            _volume_variable(dataset,0,'QRAIN',shape,required=True)


def test_calwrf_selects_preceding_hour_as_rain_baseline(tmp_path):
    paths=[]
    for number,times in enumerate((('2018-03-17_23:00:00','2018-03-18_00:00:00'),
                                   ('2018-03-18_01:00:00',))):
        path=tmp_path/f'wrf-{number}.nc'; paths.append(path)
        with nc.Dataset(path,'w') as dataset:
            dataset.createDimension('Time',len(times)); dataset.createDimension('DateStrLen',19)
            variable=dataset.createVariable('Times','S1',('Time','DateStrLen'))
            variable[:]=np.asarray([list(value) for value in times],dtype='S1')
    begin=_wrf_timestamp('2018-03-18_00:00:00')
    end=_wrf_timestamp('2018-03-18_01:00:00')
    selected,baseline=_select_wrf_times(paths,begin,end)
    assert baseline==(paths[0],0,'2018-03-17_23:00:00')
    assert [(path,index) for path,index,_ in selected]==[(paths[0],1),(paths[1],0)]


def test_calwrf_times_are_deduplicated_and_sorted_as_datetimes(tmp_path):
    paths=[]
    for number,times in enumerate((
            ('2018-01-01 01:00:00','2018-01-01_02:00:00','2018010100'),
            ('2018-01-01_01:00:00',))):
        path=tmp_path/f'wrf-{number}.nc'; paths.append(path)
        with nc.Dataset(path,'w') as dataset:
            dataset.createDimension('Time',len(times))
            dataset.createDimension('DateStrLen',19)
            variable=dataset.createVariable('Times','S1',('Time','DateStrLen'))
            variable[:]=np.asarray([list(value.ljust(19)) for value in times],dtype='S1')

    begin=datetime(2018,1,1); end=datetime(2018,1,1,2)
    selected,baseline=_select_wrf_times(paths,begin,end)

    assert baseline is None
    assert selected==[
        (paths[0],2,'2018-01-01_00:00:00'),
        (paths[0],0,'2018-01-01_01:00:00'),
        (paths[0],1,'2018-01-01_02:00:00'),
    ]


def test_destagger_is_vectorized_midpoint():
    a = np.arange(24, dtype="f4").reshape(2, 3, 4)
    np.testing.assert_array_equal(destagger_x(a), (a[..., :-1] + a[..., 1:]) / 2)
    np.testing.assert_array_equal(destagger_y(a), (a[..., :-1, :] + a[..., 1:, :]) / 2)
    np.testing.assert_array_equal(destagger_z(a), (a[:-1] + a[1:]) / 2)


def test_wind_rotation_and_meteorological_direction():
    u = np.array([[1.0]], dtype="f4")
    v = np.array([[0.0]], dtype="f4")
    east, north = earth_wind(u, v, np.array([[0.0]], dtype="f4"))
    speed, direction = speed_direction(east, north)
    np.testing.assert_allclose(speed, 1.0)
    np.testing.assert_allclose(direction, 270.0)


def test_polar_stereographic_rotation_uses_unit_cone_and_hemisphere_sign():
    dataset=SimpleNamespace(MAP_PROJ=2,STAND_LON=170.)
    latitude=np.array([[60.,-60.,60.]],dtype='f4')
    longitude=np.array([[-170.,-170.,160.]],dtype='f4')
    actual=rotation_angle(dataset,latitude,longitude)
    np.testing.assert_allclose(actual,np.deg2rad([[20.,-20.,-10.]]),rtol=0.,atol=2.e-8)


def test_calwrf_disables_rotation_for_every_wrf_projection_at_least_three():
    coordinates=np.ones((1,1),dtype='f4')
    for projection in (3,4,6):
        np.testing.assert_array_equal(rotation_angle(SimpleNamespace(MAP_PROJ=projection),
                                                      coordinates,coordinates),0.)
    with pytest.raises(ValueError,match='MAP_PROJ=0'):
        rotation_angle(SimpleNamespace(MAP_PROJ=0),coordinates,coordinates)


def test_calm_wind_direction_matches_calwrf_wrap_to_360():
    speed, direction = speed_direction(np.zeros((2, 2)), np.zeros((2, 2)))
    np.testing.assert_array_equal(speed, 0.0)
    np.testing.assert_array_equal(direction, 360.0)


def test_calwrf_horizontal_range_check_uses_original_four_neighbours():
    values=np.full((5,5),10.,dtype='f4')
    values[2,2]=99.; values[2,3]=99.
    values[1,2]=2.; values[3,2]=6.; values[2,1]=4.
    checked=_horizontal_range_check(values,0.,20.,slice(0,5),slice(0,5))
    # The adjacent invalid value is excluded and cannot be replaced first.
    assert checked[2,2]==pytest.approx(4.)


def test_calwrf_horizontal_range_check_falls_back_to_violated_limit():
    values=np.full((5,5),99.,dtype='f4')
    values[2,2]=-3.
    checked=_horizontal_range_check(values,0.,20.,slice(0,5),slice(0,5))
    assert checked[2,2]==0.


def test_calwrf_horizontal_range_check_uses_each_axis_bounds():
    values=np.zeros((3,5),dtype=np.float32)
    values[1,3]=100.
    values[1,2]=2.
    values[1,4]=10.
    values[0,3]=4.
    values[2,3]=6.

    checked=_horizontal_range_check(values,0.,20.,slice(0,3),slice(0,5))
    legacy=_legacy_horizontal_range_check(values,0.,20.,slice(0,3),slice(0,5))

    assert checked[1,3] == np.float32(5.5)
    assert legacy[1,3] == np.float32(3.5)


def test_calwrf_hydrometeor_range_check_uses_vertical_neighbours():
    values=np.zeros((3,3,3),dtype='f4')
    values[0,1,1]=.01; values[1,1,1]=.2; values[2,1,1]=.03
    checked=_vertical_range_check(values,0.,.05,slice(0,3),slice(0,3),slice(0,3))
    assert checked[1,1,1]==pytest.approx(.02)
