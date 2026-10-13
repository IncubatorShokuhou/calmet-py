from datetime import datetime
import numpy as np
import pytest

from pycalmet.surface_data import (SurfaceData,SurfaceRecord,SurfaceStation,fill_missing_surface,
                                   _nearest_station_grid,_wind_components,component_wind_profiles,
                                   interpolate_surface_scalar,read_unformatted_surface,surface_cloud_fields,
                                   surface_wind_profiles,unpack_surface,wind_profiles)
from pycalmet.legacy_io import FortranSequentialWriter,fixed_string
import struct
from types import SimpleNamespace
from pyproj import Transformer
from pycalmet._calmet import _noobs_step_transform,_surface_step_transform
from pycalmet.calmet_solver import Step1Fields,StepTransformContext
from pycalmet.config import CalmetConfig,GridConfig
from pycalmet.projections import geographic_crs,grid_crs
from pycalmet.sea_data import GriddedSeaData


def _dataset(records):
    return """SURF.DAT        2.1             generated test data
1
comment
LL
WGS-84  20200101
DEG
UTC-0000
2018 1 23 0 2018 2 1 0 2
101 AAA 30.0N 120.0E 10
202 BBB 31.5 -121.25 20
"""+records


def test_surface_data_reads_station_metadata_and_explicit_record(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  1 90 70 5 280 60 1000 0  2 180 80 8 281 70 999 10\n'
        '2018 1 24 0 2018 2 1 0  3 100 60 4 282 61 1001 0  4 190 90 9 283 71 998 20\n'),encoding='ascii')
    data=SurfaceData(path)
    assert data.stations[0].longitude==120. and data.stations[1].longitude==-121.25
    first=data.at_begin(datetime(2018,1,1,23,30))
    np.testing.assert_array_equal(first.ceiling_hundreds_ft,[70,80])
    np.testing.assert_allclose(first.wind_speed,[1,2])
    second=data.at_begin(datetime(2018,1,2))
    np.testing.assert_array_equal(second.cloud_tenths,[4,9])


def test_surface_data_interval_lookup_uses_half_open_records(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  1 90 70 5 280 60 1000 0  2 180 80 8 281 70 999 10\n'
        '2018 1 24 0 2018 2 1 0  3 100 60 4 282 61 1001 0  4 190 90 9 283 71 998 20\n'),encoding='ascii')
    data=SurfaceData(path)
    assert data.at_begin(datetime(2018,1,2)).wind_speed[0]==3.
    with pytest.raises(ValueError,match='no record'):
        data.at_begin(datetime(2018,1,2,1))


def test_surface_data_rejects_more_than_hourly_interval(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 2 1 0  1 90 70 5 280 60 1000 0  2 180 80 8 281 70 999 10\n'),encoding='ascii')
    with pytest.raises(ValueError,match='at most one hour'):
        SurfaceData(path)


def test_surface_missing_values_use_nearest_original_valid_station(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  1 90 9999 9999 9999 9999 9999 0  2 180 80 8 281 70 999 10\n'),encoding='ascii')
    record=SurfaceData(path).records[0]
    result=fill_missing_surface(record,np.array([[0.,0.],[2.,0.]]),cloud_method=2,itprog=0)
    assert result.ceiling_hundreds_ft[0]==80 and result.cloud_tenths[0]==8
    assert result.temperature_k[0]==281 and result.relative_humidity[0]==70
    assert result.pressure_mb[0]==999


def test_surface_missing_ceiling_is_allowed_for_prognostic_cloud(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  1 90 9999 9999 280 60 1000 0  2 180 9999 9999 281 70 999 10\n'),encoding='ascii')
    result=fill_missing_surface(SurfaceData(path).records[0],np.array([[0.,0.],[2.,0.]]),cloud_method=4,itprog=2)
    np.testing.assert_array_equal(result.ceiling_hundreds_ft,[9999,9999])


def test_legacy_surface_hour_ending_records_become_half_open_intervals(tmp_path):
    path=tmp_path/'surf20.dat'
    path.write_text("""SURF.DAT        2.0             legacy test
0
LL
WGS-84  20200101
DEG
2018 1 24 2018 2 1 0 1
101 AAA N30.0 E120.0 10
2018 1 24  1 90 70 5 280 60 1000 0
2018 2 1   2 180 80 8 281 70 999 10
""",encoding='ascii')
    data=SurfaceData(path)
    assert data.start==datetime(2018,1,1,23) and data.end==datetime(2018,1,2,1)
    assert data.at_begin(datetime(2018,1,1,23,30)).wind_speed[0]==1.
    assert data.at_begin(datetime(2018,1,2)).wind_speed[0]==2.


def test_surface_cloud_fields_follow_nearest_station_without_interpolation(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  1 90 70 2 280 60 1000 0  2 180 80 9 281 70 999 10\n'),encoding='ascii')
    nearest=np.array([[0,1],[1,0]])
    cloud,ceiling=surface_cloud_fields(SurfaceData(path).records[0],nearest)
    np.testing.assert_allclose(cloud,[[.2,.9],[.9,.2]])
    np.testing.assert_array_equal(ceiling,[[70,80],[80,70]])
    external=np.full((2,2),.35)
    cloud,ceiling=surface_cloud_fields(SurfaceData(path).records[0],nearest,external)
    np.testing.assert_allclose(cloud,external)
    np.testing.assert_array_equal(ceiling,[[70,80],[80,70]])


def test_surface_scalar_interpolation_uses_radius_station_limit_and_water_groups():
    grid=GridConfig(2,1,1.,1.,x_origin_km=0.,y_origin_km=0.)
    points=np.array([[.5,.5],[1.5,.5],[3.,.5]])
    got=interpolate_surface_scalar([280.,300.,9999.],points,grid,radius_km=.1,maximum_stations=2,power=1)
    np.testing.assert_allclose(got,[[280.,300.]])
    with pytest.raises(ValueError,match='land/water'):
        interpolate_surface_scalar([280.],points[:1],grid,target_water=np.ones((1,2),bool),station_water=[False])
    fallback=interpolate_surface_scalar([280.],points[:1],grid,target_water=np.ones((1,2),bool),
        station_water=[False],empty_fallback=(80.,100.))
    np.testing.assert_allclose(fallback,100.)


def test_calmet_decimal_packed_surface_words_are_vectorized():
    # 280.1 K, code 10, 61%; 1000.2 mb, cloud 8, 270 deg; ceiling 70, 1.25 m/s
    words=np.array([[2801*100000+10*1000+61,
                     10002*100000+8*1000+270,
                     70*10000+125],
                    [9999*100000+99*1000+999,
                     9999*100000+99*1000+999,
                     9999*10000+9999]],dtype=np.int64)
    fields=unpack_surface(words)
    assert fields['temperature_k'][0]==280.1 and fields['precipitation_code'][0]==10
    assert fields['pressure_mb'][0]==1000.2 and fields['cloud_tenths'][0]==8
    assert fields['wind_direction'][0]==270 and fields['wind_speed'][0]==1.25
    assert all(fields[name][1]==9999 for name in fields if name!='ceiling_hundreds_ft')
    assert fields['ceiling_hundreds_ft'][1]==9999


def _binary_surface(path,packing):
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('SURF.DAT',16),fixed_string('2.1',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('LL',8))
        writer.write_record(fixed_string('WGS-84',8),fixed_string('20200101',12))
        writer.write_record(struct.pack('<5i',0,0,0,1,packing))
        writer.write_record(fixed_string('UTC-0000',8))
        writer.write_record(struct.pack('<6i',201800100,0,201800101,0,1,packing))
        writer.write_record(np.array([101],dtype='<i4')); writer.write_record(fixed_string('AAA',4))
        writer.write_record(fixed_string('30N',16)); writer.write_record(fixed_string('120E',16))
        writer.write_record(np.array([10.],dtype='<f4'))
        prefix=struct.pack('<4i',201800100,0,201800101,0)
        if packing:
            words=np.array([2801*100000+10*1000+61,10002*100000+8*1000+270,70*10000+125],dtype='<i4')
            writer.write_record(prefix,words)
        else:
            writer.write_record(prefix,np.array([1.25,270.],dtype='<f4'),np.array([70,8],dtype='<i4'),
                                np.array([280.1],dtype='<f4'),np.array([61],dtype='<i4'),
                                np.array([1000.2],dtype='<f4'),np.array([10],dtype='<i4'))


@pytest.mark.parametrize('packing',[0,1])
def test_unformatted_surface_21_reads_packed_and_unpacked_records(tmp_path,packing):
    path=tmp_path/f'surf-{packing}.dat'; _binary_surface(path,packing)
    data=read_unformatted_surface(path); record=data.at_begin(datetime(2018,1,1))
    assert data.stations[0].identifier==101 and data.stations[0].longitude==120.
    assert record.wind_speed[0]==pytest.approx(1.25) and record.temperature_k[0]==pytest.approx(280.1)
    assert record.ceiling_hundreds_ft[0]==70 and record.cloud_tenths[0]==8


@pytest.mark.parametrize('packing',[0,1])
def test_unformatted_legacy_surface_converts_hour_ending_records(tmp_path,packing):
    path=tmp_path/f'surf-legacy-{packing}.dat'
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('SURF.DAT',16),fixed_string('2.0',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('LL',8))
        writer.write_record(fixed_string('WGS-84',8),fixed_string('20200101',12))
        writer.write_record(struct.pack('<5i',201800101,201800102,0,1,packing))
        writer.write_record(np.array([101],dtype='<i4')); writer.write_record(fixed_string('AAA',4))
        writer.write_record(fixed_string('30N',16)); writer.write_record(fixed_string('120E',16))
        writer.write_record(np.array([10.],dtype='<f4'))
        for end_code in (201800101,201800102):
            prefix=struct.pack('<i',end_code)
            if packing:
                words=np.array([2801*100000+10*1000+61,10002*100000+8*1000+270,
                                70*10000+125],dtype='<i4')
                writer.write_record(prefix,words)
            else:
                writer.write_record(prefix,np.array([1.25,270.],dtype='<f4'),np.array([70,8],dtype='<i4'),
                                    np.array([280.1],dtype='<f4'),np.array([61],dtype='<i4'),
                                    np.array([1000.2],dtype='<f4'),np.array([10],dtype='<i4'))
    data=read_unformatted_surface(path)
    assert data.start==datetime(2018,1,1,0) and data.end==datetime(2018,1,1,2)
    record=data.at_begin(datetime(2018,1,1,1))
    assert record.wind_speed[0]==pytest.approx(1.25)
    assert record.temperature_k[0]==pytest.approx(280.1)


def test_formatted_surface_none_uses_control_station_metadata(tmp_path):
    path=tmp_path/'surf-none.dat'
    path.write_text("""SURF.DAT        2.1             test
0
NONE
UTC-0000
2018 1 0 0 2018 1 1 0 1
101
2018 1 0 0 2018 1 1 0 1.25 270 70 8 280.1 61 1000.2 10
""",encoding='ascii')
    stations=(SurfaceStation(101,'AAA',30.,120.,10.),)
    data=SurfaceData(path,station_metadata=stations,datum='WGS-84')
    assert data.stations==stations and data.datum=='WGS-84'
    assert data.records[0].wind_speed[0]==pytest.approx(1.25)
    with pytest.raises(ValueError,match='IDs'):
        SurfaceData(path,station_metadata=(SurfaceStation(102,'BBB',31.,121.,10.),),datum='WGS-84')


def test_unformatted_surface_none_uses_control_station_metadata(tmp_path):
    path=tmp_path/'surf-none.bin'; stations=(SurfaceStation(101,'AAA',30.,120.,10.),)
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('SURF.DAT',16),fixed_string('2.1',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('NONE',8))
        writer.write_record(fixed_string('UTC-0000',8))
        writer.write_record(struct.pack('<6i',201800100,0,201800101,0,1,0))
        writer.write_record(np.array([101],dtype='<i4'))
        writer.write_record(struct.pack('<4i',201800100,0,201800101,0),
                            np.array([1.25,270.],dtype='<f4'),np.array([70,8],dtype='<i4'),
                            np.array([280.1],dtype='<f4'),np.array([61],dtype='<i4'),
                            np.array([1000.2],dtype='<f4'),np.array([10],dtype='<i4'))
    data=read_unformatted_surface(path,station_metadata=stations,datum='WGS-84')
    assert data.stations==stations
    assert data.records[0].temperature_k[0]==pytest.approx(280.1)


def test_surface_wind_components_and_power_extrapolation(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  2 270 70 2 280 60 1000 0  0 0 80 9 281 70 999 10\n'),encoding='ascii')
    record=SurfaceData(path).records[0]
    u,v=surface_wind_profiles(record,[10.,10.],[10.,40.],[.1,.1],iextrp=2)
    np.testing.assert_allclose(u[:,0],[2.,2.*4.**.143],atol=1.e-12)
    np.testing.assert_allclose(v[:,0],0.,atol=5.e-6)
    np.testing.assert_array_equal(u[:,1],0.); np.testing.assert_array_equal(v[:,1],0.)




def test_component_wind_profiles_do_not_apply_true_north_rotation():
    u,v=component_wind_profiles([-3.],[4.],[10.],[10.,40.],[.1],iextrp=3,
                                factors=[1.,2.])
    np.testing.assert_allclose(u[:,0],[-3.,-6.],atol=1.e-12)
    np.testing.assert_allclose(v[:,0],[4.,8.],atol=1.e-12)


def test_surface_wind_log_and_user_factor_options(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  2 180 70 2 280 60 1000 0  9999 9999 80 9 281 70 999 10\n'),encoding='ascii')
    record=SurfaceData(path).records[0]
    u,v=surface_wind_profiles(record,[20.,10.],[10.,40.],[.1,.1],iextrp=-1)
    expected=2.*(np.log(10.)-np.log(.1))/(np.log(20.)-np.log(.1))
    assert v[0,0]==pytest.approx(expected) and u[0,0]==pytest.approx(0.,abs=3.e-6)
    assert u[1,0]==9999. and v[1,0]==9999.
    u,v=surface_wind_profiles(record,[20.,10.],[10.,40.],[.1,.1],iextrp=3,factors=[.5,2.])
    np.testing.assert_allclose(v[:,0],[1.,4.]); assert np.all(u[:,1]==9999.)


def test_surface_wind_rmin2_skips_profiles_near_valid_upper_station(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  2 180 70 2 280 60 1000 0  0 0 80 9 281 70 999 10\n'),encoding='ascii')
    record=SurfaceData(path).records[0]
    u,v=surface_wind_profiles(record,[20.,10.],[10.,40.],[.1,.1],iextrp=2,
        station_xy=[[0.,0.],[5.,0.]],upper_xy=[[1.9,0.],[20.,0.]],
        upper_valid=[True,False],upper_air_separation_cells=2.,dx_km=1.,dy_km=1.)
    # Strictly-near station 1 retains the raw first-layer observation and no
    # upper extrapolation. Station 2 is calm and remains calm through the column.
    np.testing.assert_allclose(u[:,0],[0.,9999.],atol=3.e-6); np.testing.assert_allclose(v[:,0],[2.,9999.])
    np.testing.assert_array_equal(u[:,1],0.); np.testing.assert_array_equal(v[:,1],0.)


def test_surface_wind_similarity_option_uses_neutral_wind1_profile(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  5 180 70 2 280 60 1000 0  5 180 80 9 281 70 999 10\n'),encoding='ascii')
    record=SurfaceData(path).records[0]
    levels=np.array([10.,200.]); u,v=surface_wind_profiles(
        record,[10.,10.],levels,[.1,.1],iextrp=4,station_latitude=[30.,-30.])
    speed=np.hypot(u,v); expected=5.*np.log(levels/.1)/np.log(10./.1)
    np.testing.assert_allclose(speed[:,0],expected); np.testing.assert_allclose(speed[:,1],expected)
    direction=np.mod(270.-np.degrees(np.arctan2(v,u)),360.)
    assert direction[1,0]>180. and direction[1,1]<180.


def test_surface_true_wind_direction_rotates_to_grid_north(tmp_path):
    path=tmp_path/'surf.dat'
    path.write_text(_dataset(
        '2018 1 23 0 2018 1 24 0  5 180 70 2 280 60 1000 0  5 180 80 9 281 70 999 10\n'),encoding='ascii')
    u,v=surface_wind_profiles(SurfaceData(path).records[0],[10.,10.],[10.],[.1,.1],
                              iextrp=2,grid_convergence_degrees=[10.,-10.])
    direction=np.mod(270.-np.degrees(np.arctan2(v,u)),360.)
    np.testing.assert_allclose(direction[0],[170.,190.],atol=1.e-3)




def test_prepdi_uses_internal_edit_and_editl_missing_thresholds():
    u,v=wind_profiles([998.,999.,-999.],[180.,180.,180.],[10.,10.,10.],
                      [10.],[.1,.1,.1],
                      iextrp=3,factors=[1.])
    assert u[0,0]!=np.float32(9999.) and v[0,0]!=np.float32(9999.)
    np.testing.assert_array_equal(u[0,1:],np.float32(9999.))
    np.testing.assert_array_equal(v[0,1:],np.float32(9999.))
    u,v=component_wind_profiles([900.,901.,999.],[0.,0.,0.],[10.,10.,10.],
                                [10.],[.1,.1,.1],
                                iextrp=3,factors=[1.])
    assert u[0,0]==np.float32(900.) and v[0,0]==np.float32(0.)
    np.testing.assert_array_equal(u[0,1:],np.float32(9999.))
    np.testing.assert_array_equal(v[0,1:],np.float32(9999.))


def test_noobs1_transform_combines_surface_profile_with_inter2():
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([10.]),wind_direction=np.array([270.]),ceiling_hundreds_ft=np.array([70]),
        cloud_tenths=np.array([5]),temperature_k=np.array([280.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',timezone='UTC-0000',at_begin=lambda when:record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,
        surface_data='surf',z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(0.,),wind_extrapolation=2)
    geo={'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])}
    transform,_=_surface_step_transform(config,surface,geo)
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,one*50.,one*0.,one.astype(int))
    result=transform(step,datetime(2018,1,1))
    assert result.u.item()>9. and abs(result.v.item())<3.e-5


@pytest.mark.parametrize(('landuse','target','water_mode','flux_method','warm_layer','itprog'),
                         [(20,'land',0,0,False,2),(20,'land',0,0,False,1),
                          (20,'land',0,10,False,1),(55,'water',0,0,False,2),
                          (55,'waterp',2,0,False,2),(55,'coare',2,10,True,2)])
def test_noobs1_dispatches_iextrp4_by_station_landuse_and_temperature_mode(
        monkeypatch,landuse,target,water_mode,flux_method,warm_layer,itprog):
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),
        geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,
                            anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([10.]),wind_direction=np.array([270.]),
        ceiling_hundreds_ft=np.array([70]),cloud_tenths=np.array([5]),
        temperature_k=np.array([281.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',timezone='UTC-0000',
                            at_begin=lambda when:record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,
        itprog=itprog,surface_data='surf',z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(0.,),wind_extrapolation=4,
        water_temperature_mode=water_mode,imixh=-1,overwater_flux_method=flux_method,
        warm_layer=int(warm_layer))
    one=np.ones((1,1),dtype=np.float32)
    geo={'roughness_length':one*.1,'dominant_landuse':np.full((1,1),landuse),
         'albedo':one*.2,'bowen_ratio':one,'soil_heat_flux_parameter':one*.15,
         'anthropogenic_heat_flux':one*0.}
    transform,_=_surface_step_transform(config,surface,geo)
    level=np.zeros((1,1,1),dtype=np.float32)
    offshore=(None if water_mode==0 else GriddedSeaData(
        one*290.,one*292.,one*80.,one*10.,one*30.,one*.05,
        one*-999.,one*-999.,one*lon))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,
        np.array([[[280.]],[[279.]],[[275.]]],dtype=np.float32),
        np.array([[[0.]],[[100.]],[[400.]]],dtype=np.float32),
        one*50.,one*0.,one.astype(int),overwater_data=offshore)
    context=StepTransformContext(one*.5,one*50.,one*75.,one*.2,one*1.e-4,
                                 gmt_hour=1)
    captured=[]
    def fake_profiles(name):
        def run(*args,**kwargs):
            captured.append((name,args,kwargs))
            state=tuple(np.zeros(1,dtype=np.float32) for _ in range(6))
            return (np.array([[7.]],dtype=np.float32),np.array([[8.]],dtype=np.float32),*state)
        return run
    monkeypatch.setattr('pycalmet._calmet.land_station_similarity_theory',fake_profiles('land'))
    monkeypatch.setattr('pycalmet._calmet.neutral_water_station_profiles',fake_profiles('water'))
    monkeypatch.setattr('pycalmet._calmet.delta_temperature_water_station_profiles',
                        fake_profiles('waterp'))
    def fake_coare(*args,**kwargs):
        captured.append(('coare',args,kwargs))
        state=tuple(np.zeros(1,dtype=np.float32) for _ in range(7))
        warm=kwargs['warm_layer_state']
        warm=type(warm)(*(np.ones(1,dtype=value.dtype)
                          for value in (getattr(warm,name)
                              for name in type(warm).__dataclass_fields__)))
        return (np.array([[7.]],dtype=np.float32),np.array([[8.]],dtype=np.float32),
                *state,warm)
    monkeypatch.setattr('pycalmet._calmet.coare_water_station_profiles',fake_coare)
    result=transform(step,datetime(2018,1,1),context)
    # INTER2 then combines the dispatched station profile with IPROG=14's
    # background; the exact value is intentionally not the raw 7/8 profile.
    np.testing.assert_allclose(result.u,6.930693,rtol=0.,atol=2.e-6)
    np.testing.assert_allclose(result.v,7.920792,rtol=0.,atol=2.e-6)
    expected_calls=(['water',target] if target in {'waterp','coare'} else [target])
    assert [item[0] for item in captured]==expected_calls
    if warm_layer:
        assert context.warm_layer_state.initialized.item()
        assert captured[-1][2]['solar_time_s'].shape==(1,)
    if target=='land':
        np.testing.assert_array_equal(captured[0][1][11],context.previous_convective_height.ravel())
        assert captured[0][2]['mixing_method']==config.imixh
        expected_temperature=281. if itprog<2 else 280.
        np.testing.assert_array_equal(captured[0][1][6],[expected_temperature])


@pytest.mark.parametrize('water_temperature_mode',[0,1,2])
def test_water2_imixh1_preserves_shared_dptt_station_order(monkeypatch,
                                                            water_temperature_mode):
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),
        geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,
                            anemometer_height_m=10.)
    surface_record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([10.]),wind_direction=np.array([270.]),
        ceiling_hundreds_ft=np.array([70]),cloud_tenths=np.array([5]),
        temperature_k=np.array([280.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',timezone='UTC-0000',
                            at_begin=lambda when:surface_record)
    def sea_record(speed):
        return SimpleNamespace(wind_speed_m_s=speed,wind_direction_deg=270.,
            anemometer_height_m=10.,air_sensor_height_m=4.,water_sensor_depth_m=.6,
            air_temperature_k=290.,air_water_delta_k=-2.,relative_humidity=80.,
            mixing_height_m=9999.,gradient_below_k_m=-.0098,
            gradient_above_k_m=-.0045,wave_period_s=-999.,wave_height_m=-999.)
    sea=[]
    for identifier,speed in ((2,5.),(3,6.)):
        record=sea_record(speed)
        sea.append(SimpleNamespace(identifier=identifier,datum='WGS-84',
            point=lambda selected_grid:np.array([.5,.5]),
            longitude_latitude=lambda:(lon,lat),at_begin=lambda when,item=record:item))
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,
        itprog=2,imixh=1,surface_data='surf',z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(0.,),wind_extrapolation=4,
        overwater_flux_method=0,water_temperature_mode=water_temperature_mode)
    one=np.ones((1,1),dtype=np.float32)
    geo={'roughness_length':one*.1,'dominant_landuse':np.full((1,1),20),
         'albedo':one*.2,'bowen_ratio':one,'soil_heat_flux_parameter':one*.15,
         'anthropogenic_heat_flux':one*0.}
    transform,_=_surface_step_transform(config,surface,geo,sea=sea)
    level=np.zeros((1,1,1),dtype=np.float32)
    offshore=(None if water_temperature_mode==0 else GriddedSeaData(
        one*290.,one*292.,one*80.,one*10.,one*30.,one*.05,
        one*-999.,one*-999.,one*lon))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,
        np.array([[[280.]],[[279.]],[[275.]]],dtype=np.float32),
        np.array([[[0.]],[[100.]],[[400.]]],dtype=np.float32),
        one*50.,one*0.,one.astype(int),overwater_data=offshore)
    context=StepTransformContext(one*.5,one*50.,one*75.,one*.2,one*1.e-4)
    monkeypatch.setattr('pycalmet._calmet.land_station_similarity_theory',
        lambda *args,**kwargs:(np.array([[7.]],dtype=np.float32),
                               np.array([[8.]],dtype=np.float32),
                               *(np.zeros(1,dtype=np.float32) for _ in range(6))))
    received=[]; received_keywords=[]
    def fake_water(*args,**kwargs):
        jump=np.asarray(kwargs['previous_temperature_jump_k'],dtype=np.float32)
        received.append(float(jump[0]))
        received_keywords.append(kwargs)
        state=[np.zeros(1,dtype=np.float32) for _ in range(7)]
        state[5]=jump+np.float32(1.)
        return (np.array([[5.]],dtype=np.float32),np.array([[6.]],dtype=np.float32),*state)
    monkeypatch.setattr('pycalmet._calmet.delta_temperature_water_station_profiles',fake_water)
    transform(step,datetime(2018,1,1),context)
    np.testing.assert_allclose(received,[.2,1.2],rtol=0.,atol=1.e-7)
    assert context.previous_temperature_jump[0,0]==pytest.approx(2.2)
    if water_temperature_mode==2:
        assert 'temperature_measurement_height_m' in received_keywords[0]




def test_idiopt4_5_transform_uses_control_metadata_without_observation_files():
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),
        geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    surface_station=SurfaceStation(1,'SFC',lat,lon,10.)
    upper_station=SimpleNamespace(identifier=2,name='UPR',latitude=lat,
                                  longitude=lon,elevation_m=0.)
    diagnostic_record=SimpleNamespace(initial_u_m_s=None,initial_v_m_s=None,
        surface_temperature_k=None,lapse_rate_k_m=None,
        surface_u_m_s=np.array([4.]),surface_v_m_s=np.array([0.]),
        upper_u_m_s=np.array([[8.]]),upper_v_m_s=np.array([[0.]]))
    diagnostic=SimpleNamespace(at_begin=lambda when:diagnostic_record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,
        noobs=0,run_type=0,iprog=14,iwfcod=1,itprog=2,irhprog=1,
        diagnostic_data='diag',diagnostic_surface_wind=1,diagnostic_upper_wind=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation=-1,wind_extrapolation_factors=(1.,))
    geo={'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])}
    transform,nearest=_surface_step_transform(config,None,geo,diagnostic=diagnostic,
        surface_stations=(surface_station,),upper_stations=(upper_station,))
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,
        one*50.,one*0.,one.astype(int))
    result=transform(step,datetime(2018,1,1))
    assert nearest.item()==0
    assert result.u.item()>3.9 and abs(result.v.item())<1.e-6


@pytest.mark.parametrize('method,expected_cloud',[(1,.5),(2,.35)])
def test_noobs1_transform_supplies_observed_cloud_ceiling(method,expected_cloud):
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([10.]),wind_direction=np.array([270.]),ceiling_hundreds_ft=np.array([70]),
        cloud_tenths=np.array([5]),temperature_k=np.array([280.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',timezone='UTC-0000',at_begin=lambda when:record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,cloud_method=method,
        surface_data='surf',cloud_data='cloud' if method==2 else None,z_faces_m=(0.,20.),
        smoothing_passes=(0,),stations_per_level=(1,),wind_extrapolation_factors=(0.,),wind_extrapolation=2)
    transform,_=_surface_step_transform(config,surface,{'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])})
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.35,one*1000.,profile*280.,profile*100.,one*50.,one*0.,one.astype(int))
    result=transform(step,datetime(2018,1,1))
    assert result.cloud_fraction.item()==pytest.approx(expected_cloud)
    assert result.ceiling_height.item()==pytest.approx(70.*30.48)


def test_iprog15_joint_analysis_combines_station_and_raw_prognostic_strength():
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([1.]),wind_direction=np.array([270.]),
        ceiling_hundreds_ft=np.array([70]),cloud_tenths=np.array([5]),temperature_k=np.array([280.]),
        relative_humidity=np.array([60]),pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',at_begin=lambda when:record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,iprog=15,
        weight_data='wt',surface_data='surf',z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(0.,),wind_extrapolation=1)
    weights=np.array([[[.25]]])
    transform,_=_surface_step_transform(config,surface,
        {'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])},weights=weights)
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level+5.,level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,
        one*50.,one*0.,one.astype(int),prognostic_observation_u=level+5.,
        prognostic_observation_v=level,prognostic_observation_strength=one*4.)
    # At the station location RMIN=.1 km: station weight=.25/.1²=25,
    # prognostic weight=(1-.25)*4=3.
    expected=(25.+15.)/28.
    assert transform(step,datetime(2018,1,1)).u.item()==pytest.approx(expected)
    config.iprog=5
    mm4_transform,_=_surface_step_transform(config,surface,
        {'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])},
        weights=weights)
    assert mm4_transform(step,datetime(2018,1,1)).u.item()==pytest.approx(expected)


def test_noobs_transform_applies_initial_guess_and_diagnostic_temperature(monkeypatch):
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,
        noobs=2,iprog=15,use_initial_guess_calmet=1,
        initial_guess_calmet_files=('coarse.dat',),z_faces_m=(0.,20.))
    requested=[]
    def interpolate(source,when,target,centers):
        requested.append((source,when,target,np.asarray(centers)))
        return np.full((1,1,1),7.),np.full((1,1,1),8.)
    monkeypatch.setattr('pycalmet._calmet.interpolate_initial_guess',interpolate)
    one=np.ones((1,1)); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level,level*2.,level*280.,one*280.,one*1000.,one,
        level*50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,
        one*50.,one*0.,one.astype(int))
    source=SimpleNamespace(timezone='UTC-0000')
    diagnostic=SimpleNamespace(at_begin=lambda when:SimpleNamespace(surface_temperature_k=285.))
    result=_noobs_step_transform(config,source,diagnostic)(step,datetime(2018,1,1))
    np.testing.assert_allclose(result.u,7.); np.testing.assert_allclose(result.v,8.)
    assert result.temperature is step.temperature
    np.testing.assert_allclose(result.diagnostic_surface_temperature,285.)
    assert requested[0][1]==datetime(2018,1,1,1)
    np.testing.assert_allclose(requested[0][3],[10.])


def test_iprog14_keeps_prognostic_wind_when_igfmet_is_enabled(monkeypatch):
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,
        noobs=2,iprog=14,use_initial_guess_calmet=1,
        initial_guess_calmet_files=('coarse.dat',),z_faces_m=(0.,20.))
    one=np.ones((1,1)); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level*3.,level*4.,level*280.,one*280.,one*1000.,one,
        level*50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,
        one*50.,one*0.,one.astype(int))
    source=SimpleNamespace(timezone='UTC-0000')
    monkeypatch.setattr('pycalmet._calmet.interpolate_initial_guess',
        lambda *args: (_ for _ in ()).throw(AssertionError('IGF wind used')))
    result=_noobs_step_transform(config,source,None)(step,datetime(2018,1,1))
    np.testing.assert_allclose(result.u,step.u)
    np.testing.assert_allclose(result.v,step.v)


def test_itprog1_water_temperature_uses_sea_station():
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='LAND',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([1.]),wind_direction=np.array([270.]),ceiling_hundreds_ft=np.array([70]),
        cloud_tenths=np.array([5]),temperature_k=np.array([280.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',at_begin=lambda when:record)
    sea_record=SimpleNamespace(air_temperature_k=290.,air_water_delta_k=2.,relative_humidity=85.,wind_speed_m_s=5.,
                               wind_direction_deg=270.,mixing_height_m=120.,
                               anemometer_height_m=10.,air_sensor_height_m=4.,
                               water_sensor_depth_m=.6,wave_period_s=-999.,wave_height_m=-999.)
    sea=SimpleNamespace(point=lambda target:np.array([.5,.5]),longitude_latitude=lambda:(lon,lat),
                        at_begin=lambda when:sea_record)
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,itprog=1,irhprog=0,
        surface_data='surf',sea_files=('sea',),z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(2,),wind_extrapolation_factors=(0.,),wind_extrapolation=2)
    transform,_=_surface_step_transform(config,surface,
        {'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[55]])},sea=(sea,))
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,
        one*50.,one*0.,one.astype(int))
    result=transform(step,datetime(2018,1,1))
    assert result.surface_temperature.item()==290.
    assert result.u.item()>2.9
    assert result.observed_water_mixing_height.item()==120.
    assert result.surface_relative_humidity.item()==85


@pytest.mark.parametrize('npsta,station_code,expected',[(-1,20,10),(0,20,0),(1,20,20),(1,9999,0)])
def test_noobs1_transform_honours_precipitation_station_mode(npsta,station_code,expected):
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='A',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([1.]),wind_direction=np.array([270.]),
        ceiling_hundreds_ft=np.array([70]),cloud_tenths=np.array([5]),temperature_k=np.array([280.]),
        relative_humidity=np.array([60]),pressure_mb=np.array([1000.]),
        precipitation_code=np.array([station_code]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',at_begin=lambda when:record)
    precipitation=(None if npsta<=0 else SimpleNamespace(stations=(station,),datum='WGS-84',
        at_begin=lambda when:SimpleNamespace(rate_mm_h=np.array([2.]))))
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=1,surface_data='surf',
        precipitation_station_count=npsta,precipitation_data='precip' if npsta>0 else None,
        z_faces_m=(0.,20.),smoothing_passes=(0,),
        stations_per_level=(1,),wind_extrapolation_factors=(0,),wind_extrapolation=2)
    transform,_=_surface_step_transform(config,surface,
        {'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])},precipitation=precipitation)
    one=np.ones((1,1)); level=np.zeros((1,1,1)); profile=np.ones((2,1,1))
    step=Step1Fields(level.copy(),level.copy(),level+280.,one*280.,one*1000.,one,
        level+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,one*50.,one*0.,one.astype(int)*10)
    assert transform(step,datetime(2018,1,1)).precipitation_code.item()==expected


def test_noobs0_transform_combines_surface_and_upper_air_profiles():
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    station=SimpleNamespace(identifier=1,name='SFC',latitude=lat,longitude=lon,anemometer_height_m=10.)
    record=SimpleNamespace(begin=datetime(2018,1,1),end=datetime(2018,1,1,1),
        wind_speed=np.array([0.]),wind_direction=np.array([0.]),ceiling_hundreds_ft=np.array([70]),
        cloud_tenths=np.array([5]),temperature_k=np.array([280.]),relative_humidity=np.array([60]),
        pressure_mb=np.array([1000.]),precipitation_code=np.array([0]))
    surface=SimpleNamespace(stations=(station,),datum='WGS-84',at_begin=lambda when:record)
    upper_station=SimpleNamespace(identifier=2,name='UPR',latitude=lat,longitude=lon,elevation_m=0.)
    upper=SimpleNamespace(station=upper_station,datum='WGS-84',
        layer_wind_at=lambda when,faces,angle:(np.array([10.,10.]),np.array([0.,0.])),
        temperature_faces_at=lambda when,faces:np.linspace(280.,278.,len(faces)))
    config=CalmetConfig('met','geo','out','2018010100','2018010101',grid,noobs=0,
        surface_data='surf',upper_air_files=('up',),itprog=0,surface_temperature_option=1,
        upper_temperature_option=1,
        z_faces_m=(0.,20.,40.),smoothing_passes=(0,0),
        stations_per_level=(2,2),wind_extrapolation_factors=(0.,0.),wind_extrapolation=-2)
    transform,_=_surface_step_transform(config,surface,
        {'roughness_length':np.array([[.1]]),'dominant_landuse':np.array([[20]])},(upper,))
    one=np.ones((1,1)); levels=np.zeros((2,1,1)); profile=np.ones((3,1,1))
    prognostic=levels+285.
    step=Step1Fields(levels.copy(),levels.copy(),levels+280.,one*280.,one*1000.,one,
        levels+50.,one*100.,one*10.,one*.1,one*.5,one*1000.,profile*280.,profile*100.,one*50.,one*0.,one.astype(int),
        prognostic_temperature=prognostic)
    result=transform(step,datetime(2018,1,1))
    assert result.u[0].item()==pytest.approx(0.)
    assert result.u[1].item()>4.9
    assert result.diagnostic_surface_temperature.item()==280.
    assert result.diagnostic_lapse_rate.item()==pytest.approx(-.01)
    assert result.diagnostic_lapse_station_index==(0,0)
    np.testing.assert_allclose(result.surface_pressure_mb,1000.)
    np.testing.assert_allclose(result.surface_density,.3484321*1000./280.)
    np.testing.assert_allclose(result.profile_height[:,0,0],[0.,10.,30.])
    np.testing.assert_allclose(result.profile_temperature[:,0,0],[280.,280.,278.5])
    assert result.temperature.dtype==np.float32
    assert result.surface_temperature.dtype==np.float32
    assert result.surface_pressure_mb.dtype==np.float32
    assert result.profile_height.dtype==np.float32
    assert result.profile_temperature.dtype==np.float32
    assert result.prognostic_temperature is prognostic
