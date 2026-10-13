from datetime import datetime

import numpy as np
import pytest

from pycalmet.config import GridConfig
from pycalmet.sea_data import (SeaData,SeaRecord,grid_sea_mixing_height,
                               grid_water_mixing_gradient,nearest_sea_data,
                               nearest_sea_waves)


def _sea_text():
    fields=f'{30.0:<16}{-120.0:<16}{30.0:<16}{60.0:<16}'
    return f"""SEA.DAT         2.11            test overwater station
1
comment
LCC
{fields}
0 0
WGS-84  20200101
KM
UTC-0000
2018 77 0 2018 77 2
4007 buoy station
100 200 10 4 0.6 2018 77 0 2018 77 1 2.25 290 80 100 -0.0098 -0.0045 5 270 8 1.5
100 200 10 4 0.6 2018 77 1 2018 77 2 2.50 291 81 110 -0.0090 -0.0040 6 280 9 1.7
"""


def test_sea_211_header_records_and_interval_lookup(tmp_path):
    path=tmp_path/'sea.dat'; path.write_text(_sea_text(),encoding='ascii')
    data=SeaData(path)
    assert data.identifier==4007 and data.name=='buoy station'
    record=data.at_begin(datetime(2018,3,18,0,30))
    assert record.air_temperature_k==290. and record.air_water_delta_k==2.25
    assert record.air_sensor_height_m==4. and record.water_sensor_depth_m==.6
    assert record.gradient_below_k_m==pytest.approx(-.0098)
    target=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=-120.,lat1=30.,lat2=60.)
    np.testing.assert_allclose(data.point(target),[100.,200.],atol=1.e-8)


def test_sea_rejects_gaps_and_subhourly_versions(tmp_path):
    path=tmp_path/'sea.dat'
    path.write_text(_sea_text().replace('2018 77 1 2018 77 2','2018 77 2 2018 77 3'),encoding='ascii')
    with pytest.raises(ValueError,match='gap'): SeaData(path)
    path.write_text(_sea_text().replace('2.11','2.20'),encoding='ascii')
    with pytest.raises(ValueError,match='earlier than 2.2'): SeaData(path)


def test_sea_21_uses_legacy_record_without_sensor_heights(tmp_path):
    text=_sea_text().replace('2.11','2.10').replace(
        'UTC-0000\n2018 77 0 2018 77 2\n','').replace(
        '100 200 10 4 0.6 2018 77 0 2018 77 1',
        '100 200 10 2018 77 0 2018 77 1').replace(
        '100 200 10 4 0.6 2018 77 1 2018 77 2',
        '100 200 10 2018 77 1 2018 77 2')
    path=tmp_path/'sea21.dat'; path.write_text(text,encoding='ascii')
    data=SeaData(path); record=data.records[0]
    assert data.timezone is None and data.start==datetime(2018,3,18)
    assert record.air_sensor_height_m==9999. and record.water_sensor_depth_m==9999.
    assert record.wave_period_s==8. and record.wave_height_m==1.5


def test_pre_21_sea_record_retains_old_longitude_layout(tmp_path):
    text=_sea_text().replace('2.11','2.00').replace(
        'UTC-0000\n2018 77 0 2018 77 2\n','').replace(
        '100 200 10 4 0.6 2018 77 0 2018 77 1',
        '100 200 120 10 2018 77 0 2018 77 1').replace(
        '100 200 10 4 0.6 2018 77 1 2018 77 2',
        '100 200 120 10 2018 77 1 2018 77 2')
    text=text.replace(' 5 270 8 1.5',' 5 270').replace(' 6 280 9 1.7',' 6 280')
    path=tmp_path/'sea20.dat'; path.write_text(text,encoding='ascii')
    data=SeaData(path); record=data.records[0]
    assert data.version=='2.00' and data.timezone is None
    assert record.x_km==100. and record.y_km==200.
    assert record.anemometer_height_m==10.
    assert record.air_sensor_height_m==9999. and record.water_sensor_depth_m==9999.
    assert record.wave_period_s==9999. and record.wave_height_m==9999.


def test_sea_21_rejects_zero_wave_period_and_converts_missing_waves(tmp_path):
    text=_sea_text().replace('2.11','2.10').replace(
        'UTC-0000\n2018 77 0 2018 77 2\n','').replace(
        '100 200 10 4 0.6 2018 77 0 2018 77 1',
        '100 200 10 2018 77 0 2018 77 1').replace(
        '100 200 10 4 0.6 2018 77 1 2018 77 2',
        '100 200 10 2018 77 1 2018 77 2')
    path=tmp_path/'sea21.dat'
    path.write_text(text.replace(' 5 270 8 1.5',' 5 270 0 1.5'),encoding='ascii')
    with pytest.raises(ValueError,match='period'): SeaData(path)
    path.write_text(text.replace(' 5 270 8 1.5',' 5 270 9999 9999'),encoding='ascii')
    record=SeaData(path).records[0]
    assert record.wave_period_s==-999. and record.wave_height_m==-999.


@pytest.mark.parametrize('projection,header,grid',[
    ('PS',f'{90.:<16}{0.:<16}{60.:<16}',
     GridConfig(1,1,1.,1.,projection='PS',lat0=90.,lon0=0.,lat1=60.)),
    ('EM',f'{0.:<16}{0.:<16}',
     GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.)),
    ('LAZA',f'{30.:<16}{-120.:<16}\n10 20',
     GridConfig(1,1,1.,1.,projection='LAZA',lat0=30.,lon0=-120.,
                false_easting_km=10.,false_northing_km=20.)),
    ('TTM',f'{0.:<16}{120.:<16}\n10 20',
     GridConfig(1,1,1.,1.,projection='TTM',lat0=0.,lon0=120.,
                false_easting_km=10.,false_northing_km=20.)),
])
def test_sea_accepts_all_legacy_projection_headers(tmp_path,projection,header,grid):
    lcc_header=f"LCC\n{30.0:<16}{-120.0:<16}{30.0:<16}{60.0:<16}\n0 0"
    path=tmp_path/f'{projection.lower()}.dat'
    path.write_text(_sea_text().replace(lcc_header,f'{projection}\n{header}'),encoding='ascii')
    data=SeaData(path)
    assert data.projection==projection
    np.testing.assert_allclose(data.point(grid),[100.,200.],atol=1.e-8)


def test_sea_mixing_height_uses_all_valid_stations_inverse_square():
    grid=GridConfig(2,1,1.,1.)
    points=np.array([[.5,.5],[2.5,.5]])
    got=grid_sea_mixing_height([100.,300.],points,grid)
    assert got[0,0]==pytest.approx(100.,abs=1.e-5)
    assert got[0,1]==pytest.approx(200.)
    assert np.isnan(grid_sea_mixing_height([9999.],points[:1],grid)).all()


def test_sea_mixing_height_uses_real4_meters_and_station_order():
    grid=GridConfig(2,1,.7,.7,x_origin_km=100.)
    points=np.array([[100.13,.21],[101.02,.48],[100.61,-.17]],dtype=np.float32)
    values=np.array([123.4567,987.6543,345.6789],dtype=np.float32)
    got=grid_sea_mixing_height(values,points,grid)
    assert got.dtype==np.float32
    # diagnostics/code/water_zi_reference.f90, compiled with GNU Fortran.
    np.testing.assert_array_equal(got.view(np.uint32),
                                  [[0x437A4B34,0x446CE7CF]])


def test_sea_mixing_height_multiple_stations_reject_nonpositive_sum():
    grid=GridConfig(1,1,1.,1.)
    points=np.array([[.25,.5],[.75,.5]])
    assert np.isnan(grid_sea_mixing_height([0.,9999.],points,grid)).all()


def test_water_mixing_gradient_preserves_water_multi_station_positive_sum_test():
    grid=GridConfig(1,1,1.,1.)
    points=np.array([[.25,.5],[.75,.5]])
    # WATER's historical XSUM.GT.0 test rejects ordinary negative lapse rates.
    negative=grid_water_mixing_gradient([-.003,-.006],points,grid,-.0045)
    np.testing.assert_array_equal(negative,np.array([[-.0045]],dtype=np.float32))
    positive=grid_water_mixing_gradient([.003,.009],points,grid,-.0045)
    np.testing.assert_array_equal(positive,np.array([[.006]],dtype=np.float32))
    # The special NOWSTA=1 branch accepts a negative observed gradient.
    single=grid_water_mixing_gradient([-.006],points[:1],grid,-.0045)
    np.testing.assert_array_equal(single,np.array([[-.006]],dtype=np.float32))


def test_nearest_sea_data_uses_only_stations_with_valid_delta_temperature():
    begin=datetime(2018,1,1); end=datetime(2018,1,1,1)
    def record(x,delta,air,sensor,depth):
        return SeaRecord(begin,end,x,.5,10.,sensor,depth,delta,air,80.,100.,
                         -.0098,-.0045,5.,270.,8.,1.5)
    records=(record(.5,9999.,280.,4.,.6),record(2.5,2.,290.,9999.,9999.))
    grid=GridConfig(2,1,1.,1.)
    got=nearest_sea_data(records,np.array([[.5,.5],[2.5,.5]]),[120.,121.],grid)
    np.testing.assert_array_equal(got.air_temperature_k,[[290.,290.]])
    np.testing.assert_array_equal(got.sea_temperature_k,[[288.,288.]])
    np.testing.assert_array_equal(got.air_sensor_height_m,[[10.,10.]])
    np.testing.assert_array_equal(got.water_sensor_depth_m,
                                  np.array([[.6,.6]],dtype=np.float32))
    np.testing.assert_array_equal(got.station_longitude,[[121.,121.]])
    assert nearest_sea_data(records[:1],np.array([[.5,.5]]),[120.],grid) is None


def test_nearest_sea_data_keeps_first_station_on_real4_distance_tie():
    begin=datetime(2018,1,1); end=datetime(2018,1,1,1)
    def record(x,air,wave):
        return SeaRecord(begin,end,x,.5,10.,10.,.6,2.,air,80.,100.,
                         -.0098,-.0045,5.,270.,wave,1.5)
    records=(record(.5,280.,8.),record(1.5,290.,9.))
    got=nearest_sea_data(records,np.array([[.5,.5],[1.5,.5]]),[120.,121.],
                         GridConfig(1,1,1.,1.,x_origin_km=.5))
    np.testing.assert_array_equal(got.air_temperature_k,[[280.]])
    np.testing.assert_array_equal(got.wave_period_s,[[8.]])


def test_nearest_sea_data_does_not_apply_coare_sst_floor_to_deltat():
    begin=datetime(2018,1,1); end=datetime(2018,1,1,1)
    record=SeaRecord(begin,end,.5,.5,10.,10.,.6,20.,280.,80.,100.,
                     -.0098,-.0045,5.,270.,8.,1.5)
    got=nearest_sea_data((record,),np.array([[.5,.5]]),[120.],GridConfig(1,1,1.,1.))
    np.testing.assert_array_equal(got.sea_temperature_k,[[260.]])


def test_waterp_wave_selection_does_not_require_valid_delta_temperature():
    begin=datetime(2018,1,1); end=datetime(2018,1,1,1)
    record=SeaRecord(begin,end,.5,.5,10.,10.,.6,9999.,290.,80.,100.,
                     -.0098,-.0045,5.,270.,9.,2.)
    period,height=nearest_sea_waves((record,),np.array([[.5,.5]]),
                                    GridConfig(1,1,1.,1.))
    np.testing.assert_array_equal(period,[[9.]])
    np.testing.assert_array_equal(height,[[2.]])
