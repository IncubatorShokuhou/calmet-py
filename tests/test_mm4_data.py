from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

from pycalmet.config import GridConfig
from pycalmet.calmet_solver import build_mm4_step_fields,target_latitude_longitude
from pycalmet.config import CalmetConfig
from pycalmet._calmet import _meteorology_input
from pycalmet._calmet import run as run_calmet
from pycalmet.mm4_data import MM4Data


def _mm4(path,record_hours=(4,)):
    lines=['synthetic MM4'.ljust(36),f'{18:02d}{1:02d}{2:02d}{3:02d}{len(record_hours):4d}{2:4d}{2:4d}{1:4d}{100.:6.1f}',
           f'{4:4d}{5:4d}{2:4d}{2:4d}',f'{.5:6.4f}']
    for j in (5,6):
        for i,terrain in ((4,100),(5,200)):
            lines.append(f'{i:3d}{j:3d}{30.+j-5:7.3f}{120.+i:8.3f}{terrain:5d}{20:3d}')
    for hour in record_hours:
        for j in (5,6):
            for i,terrain in ((4,100),(5,200)):
                lines.append(f'{18:02d}{1:02d}{2:02d}{hour:02d}{i:3d}{j:3d}{1000.:7.1f}{.2:5.2f}{0:2d}')
                for level in range(9):
                    encoded=200 if level==0 else 201
                    lines.append(f'{999.-level*50:5.1f}{terrain+level*100:6.0f}{encoded:4d}{10:2d}'
                                 f'{180.:4.0f}{10.:2.0f}')
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')


def test_mm4_reader_decodes_header_profiles_and_legacy_units(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path)
    assert data.header.start==datetime(2018,1,2,3)
    assert data.header.latitude.shape==(2,2)
    record=data.at_time(datetime(2018,1,2,4))
    assert data.header.latitude.dtype==np.float32
    assert record.pressure_mb.dtype==np.float32
    assert record.temperature_k.dtype==np.float32
    assert record.relative_humidity.dtype==np.float32
    assert record.wind_speed_m_s.dtype==np.float32
    np.testing.assert_allclose(record.precipitation_mm_h,2.)
    np.testing.assert_allclose(record.height_agl_m[:,0,0],np.arange(9)*100.)
    assert record.temperature_k[0,0,0]==pytest.approx(293.15)
    assert record.temperature_k[1,0,0]==pytest.approx(253.05)
    np.testing.assert_allclose(record.wind_speed_m_s,5.1444)
    assert np.all((record.relative_humidity>=0.)&(record.relative_humidity<=100.))
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    # Put the target on the projected midpoint of the two synthetic source
    # points so both contribute symmetrically.
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    fields=data.interpolate_record(record,grid,[50.,150.])
    assert fields.u.shape==(2,1,1)
    assert fields.temperature_k.dtype==np.float32
    assert fields.surface_density.dtype==np.float32
    # RDMM4's rounded REAL*4 DEG2RAD leaves a small deterministic crosswind
    # at a nominal 180-degree direction.
    expected_u=-np.float32(5.1444)*np.sin(np.float32(.0174533)*np.float32(180.))
    wind_factor=np.log(50./.5)/np.log(100./.5)
    np.testing.assert_allclose(fields.u[:,0,0],[expected_u*wind_factor,expected_u],rtol=2.e-5)
    expected=5.1444*np.log(50./.5)/np.log(100./.5)
    np.testing.assert_allclose(fields.v[0],expected,rtol=2.e-5)
    np.testing.assert_allclose(fields.v[1],5.1444,rtol=2.e-5)
    np.testing.assert_allclose(fields.temperature_k[:,0,0],[253.54,253.05])
    np.testing.assert_allclose(fields.surface_temperature_k,254.03,atol=4.e-5)
    np.testing.assert_allclose(fields.profile_temperature_k[0],254.03,atol=4.e-5)
    assert fields.profile_height_m.shape==(10,1,1)
    np.testing.assert_allclose(fields.precipitation_mm_h,2.)
    assert fields.precipitation_code.item()==20
    assert 0.<=fields.cloud_fraction.item()<=1.
    # Source cloud base is 200 m, but RDMM4 stores ICEILG in integer
    # hundreds of feet: INT(200/30.48) * 30.48 = 182.88 m.
    np.testing.assert_allclose(fields.ceiling_height_m,np.float32(6)*np.float32(30.48))
    assert fields.surface_relative_humidity.dtype==np.int32
    assert fields.surface_temperature_k.shape==(1,1)
    assert fields.thermal_state.pt20.shape==(2,2)
    continued=data.interpolate_record(record,grid,[50.,150.],thermal_state=fields.thermal_state)
    np.testing.assert_allclose(continued.thermal_state.pt20,fields.thermal_state.pt20)
    step=build_mm4_step_fields(data,record,grid,[0.,100.,200.])
    assert step.u.shape==(2,1,1)
    assert step.pbl_height.dtype==np.float32
    assert step.sensible_heat_flux.dtype==np.float32
    assert step.friction_velocity.dtype==np.float32
    np.testing.assert_allclose(step.precipitation_rate,2.)
    observed=build_mm4_step_fields(data,record,grid,[0.,100.,200.],
                                   prognostic_observation_rmin_km=.1)
    assert observed.prognostic_observation_u.shape==(2,1,1)
    assert observed.prognostic_observation_strength.shape==(1,1)
    assert np.all(observed.prognostic_observation_strength>0.)
    fields3=data.interpolate_record(record,grid,[50.,150.],cloud_method=3)
    np.testing.assert_allclose(fields3.ceiling_height_m,80.*30.48)

    # MM4 source category 20 follows STULL by default, but ILUOC3D may name
    # any source-grid category as ocean for the nighttime extrapolation.
    land_night=data.interpolate_record(record,grid,[50.,150.],base_timezone=8)
    ocean=data.interpolate_record(record,grid,[50.,150.],base_timezone=8,
                                   prognostic_ocean_category=20)
    assert not np.allclose(ocean.temperature_k,land_night.temperature_k)
    assert not np.allclose(ocean.surface_temperature_k,
                           land_night.surface_temperature_k)
    wrapped=build_mm4_step_fields(data,record,grid,[0.,100.,200.],
                                  base_timezone=8,
                                  prognostic_ocean_category=20)
    np.testing.assert_allclose(wrapped.temperature,ocean.temperature_k)
    np.testing.assert_allclose(wrapped.surface_temperature,
                               ocean.surface_temperature_k)


def test_mm4_reader_rejects_bad_grid_sequence(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    text=path.read_text(encoding='ascii').replace(f'{5:3d}{5:3d}{30.:7.3f}',
                                                  f'{6:3d}{5:3d}{30.:7.3f}',1)
    path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='static grid index'):
        MM4Data(path)


def test_mm4_surface_scalars_and_850_humidity_precede_height_sort(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path); record=data.records[0]
    height=record.height_agl_m.copy(); height[0]=500.; height[1]=0.
    pressure=record.pressure_mb.copy(); pressure[0]=800.; pressure[1]=900.; pressure[2]=860.
    humidity=record.relative_humidity.copy(); humidity[0]=11.; humidity[1]=22.; humidity[2]=33.
    temperature=record.temperature_k.copy(); temperature[0]=300.; temperature[1]=250.
    record=replace(record,height_agl_m=height,pressure_mb=pressure,
                   relative_humidity=humidity,temperature_k=temperature)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    fields=data.interpolate_record(record,grid,[50.],cloud_method=3)
    np.testing.assert_allclose(fields.surface_pressure_mb,800.)
    np.testing.assert_allclose(fields.surface_density,.3484321*800./300.)
    np.testing.assert_array_equal(fields.surface_relative_humidity,11)
    # N=3 is the final record-order pressure above 850 mb, irrespective of
    # the height sort.
    from pycalmet.calmet_numerics import cloud_fraction
    np.testing.assert_allclose(fields.cloud_fraction,cloud_fraction(np.array([[33.]])))


def test_mm4_profile_reproduces_duplicate_deletion_and_removes_below_ground(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path); record=data.records[0]
    height=record.height_agl_m.copy()
    height[:,0,0]=[-50.,0.,100.,100.,200.,300.,400.,500.,600.]
    marker=np.broadcast_to(np.arange(9.)[:,None,None],height.shape).copy()
    profiles=data._sorted_above_ground_profiles(
        height=height,temperature=marker,humidity=marker,pressure=marker,
        direction=marker,speed=marker)
    # Literal RDMM4 behavior: the 100-m duplicate is removed, and the stale
    # tail comparison reduces NLEVOK once more.  The -50-m level is then
    # excluded while packing above-ground levels.
    np.testing.assert_array_equal(profiles['height'][:,0,0],
                                  [0.,100.,200.,300.,400.,500.,500.,500.,500.])
    np.testing.assert_array_equal(profiles['temperature'][:6,0,0],
                                  [1.,2.,4.,5.,6.,7.])


def test_mm4_rejects_calmet_grid_outside_curvilinear_source_domain(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(data.header.longitude,data.header.latitude,grid,'WGS-84',units='km')
    grid.x_origin_km=float(np.max(x)+10.); grid.y_origin_km=float(np.max(y)+10.)
    with pytest.raises(ValueError,match='grid corner 1 is outside prognostic domain'):
        data.interpolate_record(data.records[0],grid,[50.])


def test_native_mm4_adapter_supplies_subhour_step_fields(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,timestep_seconds=900,z_faces_m=(0.,100.,200.))
    start=datetime(2018,1,2,3); end=datetime(2018,1,2,4)
    with _meteorology_input(config,(path,),tmp_path,start,end) as (dataset,provider):
        assert dataset['time_text'].shape==(5,19)
        first=provider(datetime(2018,1,2,3,15))
        last=provider(end)
    assert first is last
    assert first.u.shape==(2,1,1)


def test_native_mm4_adapter_applies_readhd_sigmap_guard(tmp_path,monkeypatch):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,precipitation_station_count=-1,
        precipitation_radius_km=1.,z_faces_m=(0.,100.,200.))
    captured={}
    class Result:
        thermal_state=None
    def fake_build(*args,**kwargs):
        captured.update(kwargs)
        return Result()
    monkeypatch.setattr('pycalmet._calmet.build_mm4_step_fields',fake_build)
    with _meteorology_input(config,(path,),tmp_path,datetime(2018,1,2,3),
                            datetime(2018,1,2,4)) as (_,provider):
        provider(datetime(2018,1,2,4))
    source_x,source_y=project_geographic(data.header.longitude,data.header.latitude,
                                         grid,'WGS-84',units='km')
    source_x=np.asarray(source_x,dtype=np.float32); source_y=np.asarray(source_y,dtype=np.float32)
    dx=np.asarray(source_x[0,0]-source_x[0,1],dtype=np.float32)
    dy=np.asarray(source_y[0,0]-source_y[0,1],dtype=np.float32)
    spacing=np.sqrt(np.asarray(dx*dx+dy*dy,dtype=np.float32))
    expected=np.float32(np.sqrt(np.float32(2.))*spacing*np.float32(1.05))
    assert captured['precipitation_radius_km']==float(expected)


def test_native_mm4_adapter_does_not_fill_current_file_gap_from_later_file(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    _mm4(first_path,(4,6)); _mm4(second_path,(5,6))
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010205',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match=r'record 2018-01-02 06:00:00 is beyond requested time 2018-01-02 05:00:00'):
        with _meteorology_input(config,(first_path,second_path),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,5)):
            pass


def test_native_mm4_adapter_validates_first_header_start(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    text=path.read_text(encoding='ascii').replace('18010203','18010205',1)
    path.write_text(text,encoding='ascii')
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match='header begins at .* after first CALMET ending time'):
        with _meteorology_input(config,(path,),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,4)):
            pass


def test_native_mm4_adapter_ignores_followup_static_header_like_rdmm4(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    clean_path=tmp_path/'MM4-clean.DAT'
    _mm4(first_path,(4,)); _mm4(second_path,(5,)); _mm4(clean_path,(5,))
    text=second_path.read_text(encoding='ascii')
    second_path.write_text(text.replace(f'{100:5d}{20:3d}',f'{100:5d}{16:3d}',1),encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010205',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with _meteorology_input(config,(first_path,second_path),tmp_path,
                            datetime(2018,1,2,3),datetime(2018,1,2,5)) as (_,provider):
        later=provider(datetime(2018,1,2,5))
    # The second header changed category 20 to 19, but RDMM4 skips it and
    # retains first-file ILU4=20.  The resulting field therefore matches a
    # direct first-header interpretation of the second record.
    with _meteorology_input(config,(first_path,clean_path),tmp_path,
                            datetime(2018,1,2,3),datetime(2018,1,2,5)) as (_,provider):
        clean=provider(datetime(2018,1,2,5))
    np.testing.assert_array_equal(later.temperature,clean.temperature)


def test_native_mm4_adapter_advances_thermal_state_when_later_hour_requested_first(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path,(4,5))
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010205',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with _meteorology_input(config,(path,),tmp_path,datetime(2018,1,2,3),
                            datetime(2018,1,2,5)) as (_,provider):
        later=provider(datetime(2018,1,2,5))
        earlier=provider(datetime(2018,1,2,4))
    assert later is not earlier
    assert later.u.shape==earlier.u.shape==(2,1,1)


def test_public_calmet_idiopt4_5_omit_surface_and_upper_files(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=124.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),
                           grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    geo=tmp_path/'geo.nc'; shape=(1,1)
    with Dataset(geo,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        values={'terrain_height':0.,'dominant_landuse':20,'roughness_length':.1,
                'albedo':.2,'bowen_ratio':1.,'soil_heat_flux_parameter':.15,
                'anthropogenic_heat_flux':0.}
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f8'
            dataset.createVariable(name,dtype,('y','x'))[:]=np.full(shape,value)
    diag=tmp_path/'DIAG.DAT'
    surface_line=' '*15+f'{"SFC":<4} '+''.join(f'{value:5.1f}' for value in (1.,4.,0.))
    upper_line=' '*13+f'{"UPR":<4} '+''.join(
        f'{value:5.1f}' for value in (1.,8.,0.,9.,0.))
    upper_last=' '*13+'LAST '+''.join(f'{0.:5.1f}' for _ in range(5))
    diag.write_text('\n'.join((surface_line,' '*15+'LAST',upper_line,upper_last))+'\n',
                    encoding='ascii')
    station_x=grid.x_origin_km+.5; station_y=grid.y_origin_km+.5
    output=tmp_path/'calmet-diag.nc'
    config=CalmetConfig(path,geo,output,'2018010203','2018010204',grid,
        iprog=4,iwfcod=1,noobs=0,run_type=0,itprog=2,irhprog=1,
        surface_data=None,upper_air_files=(),diagnostic_data=diag,
        diagnostic_surface_wind=1,diagnostic_upper_wind=1,
        surface_control_stations=(('SFC',1,station_x,station_y,0.,10.),),
        upper_control_stations=(('UPR',2,station_x,station_y,0.),),
        z_faces_m=(0.,100.,200.),smoothing_passes=(0,0),
        stations_per_level=(1,1),wind_extrapolation_factors=(1.,1.),
        ifradj=0,islope=0,ikine=0,iobr=0,iavet=0,iavezi=0,
        lapse_height_m=100.)
    assert run_calmet(config)==output
    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()


def test_public_calmet_runs_native_mm4_to_netcdf(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=124.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    geo=tmp_path/'geo.nc'; shape=(1,1)
    with Dataset(geo,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        values={'terrain_height':0.,'dominant_landuse':20,'roughness_length':.1,
                'albedo':.2,'bowen_ratio':1.,'soil_heat_flux_parameter':.15,
                'anthropogenic_heat_flux':0.}
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f8'
            dataset.createVariable(name,dtype,('y','x'))[:]=np.full(shape,value)
    output=tmp_path/'calmet.nc'
    config=CalmetConfig(path,geo,output,'2018010203','2018010204',grid,
        iprog=4,iwfcod=1,noobs=2,z_faces_m=(0.,100.,200.),
        smoothing_passes=(0,0),ifradj=0,islope=0,ikine=0,iobr=0,
        iavet=0,iavezi=0,run_type=0,lapse_height_m=100.)
    assert run_calmet(config)==output
    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()

    # COMP skips PREPDI entirely for NOOBS=2.  IDIOPT4/5 still cause
    # OPENOT to require DIAG.DAT, but no wind blocks are read from it.
    diagnostic=tmp_path/'empty-diag.dat'; diagnostic.write_text('',encoding='ascii')
    inert_output=tmp_path/'calmet-inert-diag.nc'
    config.output=inert_output; config.diagnostic_data=diagnostic
    config.diagnostic_surface_wind=1; config.diagnostic_upper_wind=1
    assert run_calmet(config)==inert_output
    with Dataset(inert_output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()
    config.diagnostic_surface_wind=0; config.diagnostic_upper_wind=0
    no_output=tmp_path/'disabled.dat'
    config.output=no_output; config.save_output=False
    assert run_calmet(config)==no_output
    assert not no_output.exists()
    setup_output=tmp_path/'setup.nc'
    config.output=setup_output; config.save_output=True; config.test_mode=1
    assert run_calmet(config)==setup_output
    with Dataset(setup_output) as dataset:
        assert dataset.dimensions['time'].size==0

    pacout=tmp_path/'pacout.dat'
    config.output=pacout; config.test_mode=2; config.legacy_output_format=2
    config.run_type=1; config.z_faces_m=(0.,20.,3500.); config.smoothing_passes=(0,0)
    assert run_calmet(config)==pacout
    from pycalmet.legacy_io import FortranSequentialReader
    with FortranSequentialReader(pacout) as reader:
        header=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(header[:40],dtype='<i4'),
                                      [2018,2,1,0,0,1,1,0,2,4])
        assert reader.read_record()==b''
        for _ in range(3): assert len(reader.read_record())==4
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),
                                      [2018,2,3])
        for _ in range(10): assert len(reader.read_record())==4
        assert len(reader.read_record())==4
        assert reader.read_record() is None

    latitude,longitude=target_latitude_longitude(grid)
    surf=tmp_path/'surf.dat'
    surf.write_text(f"""SURF.DAT        2.1             synthetic station data
0
LL
WGS-84  20200101
DEG
UTC-0000
2018 2 3 0 2018 2 4 0 1
101 STA {latitude.item():.8f} {longitude.item():.8f} 10
2018 2 3 0 2018 2 4 0  1 270 9999 9999 282 61 1000 7
""",encoding='ascii')
    observed=tmp_path/'pacout-observed.dat'
    config.output=observed; config.noobs=1; config.surface_data=surf
    config.stations_per_level=(1,1); config.wind_extrapolation=2
    config.wind_extrapolation_factors=(1.,1.)
    assert run_calmet(config)==observed
    with FortranSequentialReader(observed) as reader:
        header=reader.read_record()
        assert np.frombuffer(header,dtype='<i4',count=4)[3]==1
        assert len(reader.read_record())==8
        for _ in range(3): reader.read_record()
        reader.read_record()
        for _ in range(10): reader.read_record()
        tail=reader.read_record()
        assert len(tail)==20
        assert np.frombuffer(tail,dtype='<f4',count=1,offset=4).item()==282.
        assert np.frombuffer(tail,dtype='<i4',count=2,offset=12).tolist()==[61,7]
