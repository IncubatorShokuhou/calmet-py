from types import SimpleNamespace
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import pycalmet
import xarray as xr
from pycalmet.config import CalwrfConfig
from pycalmet._calwrf import (_cone,_create_output,_geopotential_height,_horizontal_range_check,_mass_coordinate,_pressure_temperature,_rain_increment,_rain_increments,_requested_interval,_reset_total_rain_at_file_boundary,_select_wrf_times,
                              _selection,_stamp,_surface_variable,_surface_wind,
                              _var,_vertical_range_check,_volume_variable,
                              _write_time_text,destagger_x,
                              destagger_y,destagger_z,earth_wind,rotation_angle,
                              speed_direction)

ROOT=Path(__file__).resolve().parents[1]
CALWRF_CONTROL=Path(__file__).with_name('fixtures')/'calwrf.inp'


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
    assert _stamp('2018031800')==_stamp('2018-03-18_00:00:00')


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
    with xr.open_dataset(output_path) as output:
        assert {name:output.sizes[name] for name in ('time','level','y','x')} == {
            'time':1, 'level':1, 'y':1, 'x':1,
        }
        assert output['surface_pressure'].dims == ('time','y','x')


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
    np.testing.assert_array_equal(_rain_increment([[1.]],[[5.]]),[[-4.]])


def test_rain_increments_continue_across_wrf_file_boundaries_by_default():
    previous=(np.array([[12.,20.]],dtype='f4'),
              np.array([[1.,2.]],dtype='f4'))
    total,grid,convective=_rain_increments(
        np.array([[14.5,20.25]],dtype='f4'),
        np.array([[1.25,2.5]],dtype='f4'),previous)
    np.testing.assert_array_equal(total,[[2.75,.75]])
    np.testing.assert_array_equal(grid,[[2.5,.25]])
    np.testing.assert_array_equal(convective,[[.25,.5]])


def test_legacy_file_rain_reset_only_resets_total_accumulation():
    previous=(np.array([[12.,20.]],dtype='f4'),
              np.array([[1.,2.]],dtype='f4'))
    rainnc=np.array([[14.5,20.25]],dtype='f4')
    rainc=np.array([[1.25,2.5]],dtype='f4')
    total,grid,convective=_rain_increments(
        rainnc,rainc,previous,reset_total=True)
    np.testing.assert_array_equal(total,[[15.75,22.75]])
    np.testing.assert_array_equal(grid,[[2.5,.25]])
    np.testing.assert_array_equal(convective,[[.25,.5]])


def test_legacy_total_rain_reset_requires_an_actual_file_boundary():
    previous=(np.array([[1.]],dtype='f4'),np.array([[0.]],dtype='f4'))
    first=Path('first.nc'); second=Path('second.nc')
    assert not _reset_total_rain_at_file_boundary(False,previous,first,second)
    assert not _reset_total_rain_at_file_boundary(True,None,first,second)
    assert not _reset_total_rain_at_file_boundary(True,previous,first,first)
    assert _reset_total_rain_at_file_boundary(True,previous,first,second)


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
    begin=_stamp('2018-03-18_00:00:00'); end=_stamp('2018-03-18_01:00:00')
    selected,baseline=_select_wrf_times(paths,begin,end)
    assert baseline==(paths[0],0,'2018-03-17_23:00:00')
    assert [(path,index) for path,index,_ in selected]==[(paths[0],1),(paths[1],0)]


def test_calwrf_negative_date_sentinel_selects_all_available_times(tmp_path):
    paths=[]
    for number,times in enumerate((('2018-01-01_01:00:00','2018-01-01_02:00:00'),
                                   ('2018-01-01_03:00:00','2018-01-01_04:00:00'))):
        path=tmp_path/f'wrf-{number}.nc'; paths.append(path)
        with nc.Dataset(path,'w') as dataset:
            dataset.createDimension('Time',len(times)); dataset.createDimension('DateStrLen',19)
            variable=dataset.createVariable('Times','S1',('Time','DateStrLen'))
            variable[:]=np.asarray([list(value) for value in times],dtype='S1')
    begin,end=_requested_interval(paths,'-2018010100','-2018010200')
    assert begin==_stamp('2018-01-01_01:00:00')
    assert end==_stamp('2018-01-01_04:00:00')


def test_pressure_temperature_matches_gnu_real4_bits():
    pressure,temperature=_pressure_temperature(
        [1234.567,-4321.125,.125,18765.5],
        [91345.25,100123.75,54321.875,70111.125],
        [2.3456,-12.75,18.125,-.03125])
    np.testing.assert_array_equal(pressure.view(np.uint32),np.array(
        [0x44677315,0x446f81ae,0x4407ce14,0x445e310a],dtype=np.uint32))
    np.testing.assert_array_equal(temperature.view(np.uint32),np.array(
        [0x4393e087,0x438de001,0x438598f6,0x43910320],dtype=np.uint32))


def test_pressure_temperature_uses_powf_at_formatted_half_ties():
    # Real Zhejiang cells whose temperatures straddle an F6.1 half-tie.
    # Computing REAL**REAL through a float64 detour moves these by 1--2 ulps
    # and writes the neighbouring tenth compared with GNU CALWRF.
    pressure,temperature=_pressure_temperature(
        np.array([0x434b3a80,0x44a7b6c0,0x4210a400,0x450077e0,
                  0x43b26240],dtype=np.uint32).view(np.float32),
        np.array([0x467e9d78,0x47a09ee7,0x460633aa,0x47bc4300,
                  0x46b1e848],dtype=np.uint32).view(np.float32),
        np.array([0x427e9e29,0x3d495fca,0x42c821b5,0xc1b0dd3e,
                  0x41c6da9b],dtype=np.uint32).view(np.float32))
    np.testing.assert_array_equal(temperature.view(np.uint32),np.array(
        [0x43594001,0x438e8667,0x43468ccd,0x438a5334,0x4355c001],
        dtype=np.uint32))
    assert [f'{value:.1f}' for value in temperature]==[
        '217.3','285.1','198.6','276.7','213.8']


def test_destagger_is_vectorized_midpoint():
    a = np.arange(24, dtype="f4").reshape(2, 3, 4)
    np.testing.assert_array_equal(destagger_x(a), (a[..., :-1] + a[..., 1:]) / 2)
    np.testing.assert_array_equal(destagger_y(a), (a[..., :-1, :] + a[..., 1:, :]) / 2)
    np.testing.assert_array_equal(destagger_z(a), (a[:-1] + a[1:]) / 2)


def test_geopotential_height_matches_calwrf_real4_operation_order():
    ph=np.array([1234.567,-4321.125,.125,18765.5,-234.75],dtype=np.float32)
    phb=np.array([91345.25,100123.75,54321.875,70111.125,88765.375],dtype=np.float32)
    actual=_geopotential_height(ph[:,None,None],phb[:,None,None])[:,0,0]
    np.testing.assert_array_equal(actual.view(np.uint32),np.array(
        [0x46160634,0x45ef1ce4,0x45e414d6,0x460d48a7],dtype=np.uint32))


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


def test_lambert_cone_and_rotation_match_gnu_real4_bits():
    dataset=SimpleNamespace(MAP_PROJ=1,TRUELAT1=30.1234,TRUELAT2=60.5678,
                            STAND_LON=170.25)
    latitude=np.array([60.,-60.,30.,-15.],dtype=np.float32)
    longitude=np.array([-170.1,-170.1,160.123,350.5],dtype=np.float32)
    assert _cone(dataset.TRUELAT1,dataset.TRUELAT2).view(np.uint32)==np.uint32(0x3f385b28)
    actual=rotation_angle(dataset,latitude,longitude)
    np.testing.assert_array_equal(actual.view(np.uint32),np.array(
        [0x3e7ce7a8,0xbe7ce7a8,0xbe0256e0,0x40109781],dtype=np.uint32))


def test_zhejiang_lambert_cone_uses_gnu_real_intrinsics():
    assert _cone(30.,60.).view(np.uint32)==np.uint32(0x3f372f61)


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


def test_calwrf_rotation_speed_direction_match_gnu_real4_bits():
    u=np.array([-3.123456,0.,5.,1.e-10],dtype=np.float32)
    v=np.array([4.765432,-2.5,0.,-2.e-10],dtype=np.float32)
    alpha=np.array([.3712345,-.8123456,0.,1.125],dtype=np.float32)
    east,north=earth_wind(u,v,alpha)
    speed,direction=speed_direction(east,north)
    expected=(
        [0xbf974a21,0x3fe849ef,0x40a00000,0xaf17009c],
        [0x40b25d5a,0xbfdc1861,0x00000000,0xaf4205a8],
        [0x40b654a9,0x40200000,0x40a00000,0x2f75dbb8],
        [0x43280719,0x439cba5f,0x43870000,0x43b40000],
    )
    for actual,bits in zip((east,north,speed,direction),expected):
        np.testing.assert_array_equal(actual.view(np.uint32),np.array(bits,dtype=np.uint32))


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


def test_calwrf_hydrometeor_range_check_uses_vertical_neighbours():
    values=np.zeros((3,3,3),dtype='f4')
    values[0,1,1]=.01; values[1,1,1]=.2; values[2,1,1]=.03
    checked=_vertical_range_check(values,0.,.05,slice(0,3),slice(0,3),slice(0,3))
    assert checked[1,1,1]==pytest.approx(.02)
