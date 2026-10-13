from datetime import datetime,timedelta
import numpy as np
import pytest

from pycalmet.upper_air import (Sounding,UpperAirData,UpperAirStation,_prepdi_time_interpolate,grid_temperature_faces,
                                layer_average,temperature_faces)


def _text(records):
    return """UP.DAT          2.1             test upper-air data
1
comment
LL
WGS-84  20200101
DEG
UTC+0000
2018 1 0 0 2018 2 0 0 100 3 2
T T T T
1234 ABC 30N 120E 15
"""+records


def test_formatted_upper_air_reads_and_converts_rdup_fields(tmp_path):
    path=tmp_path/'up.dat'
    path.write_text(_text(
        '1234 2018 1 1 0 2\n1000,15,20,270,5,900,1015,10,180,10\n'
        '1234 2018 1 2 0 1\n800 2015 5 999 999\n'),encoding='ascii')
    data=UpperAirData(path)
    assert data.station.latitude==30. and data.station.longitude==120.
    sounding=data.at(datetime(2018,1,1)); height,u,v=sounding.converted(10.)
    np.testing.assert_array_equal(height,[0.,1000.])
    radians=np.float32(.0174533)*np.array([260.,170.],dtype=np.float32)
    np.testing.assert_array_equal(u,-np.array([5.,10.],dtype=np.float32)*
                                  np.sin(radians.astype(np.float64)).astype(np.float32))
    np.testing.assert_array_equal(v,-np.array([5.,10.],dtype=np.float32)*
                                  np.cos(radians.astype(np.float64)).astype(np.float32))
    _,u,v=data.at(datetime(2018,1,2)).converted()
    assert u.item()==999. and v.item()==999.


def test_rdup_wind_conversion_matches_gnu_real4_bits_and_zero_branch():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),np.arange(4.),np.zeros(4),
                      np.array([217.3,0.,10.,999.]),np.array([5.718242,5.,.01,5.]))
    _,u,v=sounding.converted(1.125)
    np.testing.assert_array_equal(u.view(np.uint32),np.array(
        [0x4058039a,0x80000000,0xbae39aaf,0x4479c000],dtype=np.uint32))
    np.testing.assert_array_equal(v.view(np.uint32),np.array(
        [0x4093b525,0xc0a00000,0xbc2159d4,0x4479c000],dtype=np.uint32))


def test_formatted_upper_air_none_uses_control_station_metadata(tmp_path):
    path=tmp_path/'up-none.dat'
    path.write_text("""UP.DAT          2.1             test
0
NONE
UTC+0000
2018 1 0 0 2018 2 0 0 100 3 2
T T T T
1234 2018 1 1 0 2
1000 15 20 270 5 900 1015 10 180 10
1234 2018 1 2 0 2
1000 15 20 270 5 900 1015 10 180 10
""",encoding='ascii')
    station=UpperAirStation(1234,'ABC',30.,120.,0.)
    data=UpperAirData(path,station_metadata=station,datum='WGS-84')
    assert data.station==station and data.datum=='WGS-84'
    assert data.at(datetime(2018,1,1)).wind_speed[0]==5.


def test_upper_air_rejects_bad_station_and_time_order(tmp_path):
    bad=tmp_path/'bad.dat'
    bad.write_text(_text('9 2018 1 1 0 1\n1000 15 20 270 5\n'),encoding='ascii')
    with pytest.raises(ValueError,match='station ID'): UpperAirData(bad)
    order=tmp_path/'order.dat'
    order.write_text(_text(
        '1234 2018 1 2 0 1\n1000 15 20 270 5\n'
        '1234 2018 1 1 0 1\n1000 15 20 270 5\n'),encoding='ascii')
    with pytest.raises(ValueError,match='chronological'): UpperAirData(order)


def test_upper_air_layer_average_matches_vertav_piecewise_linear_integral():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),np.array([100.,200.,300.,400.]),
                      np.zeros(4),np.array([270.,270.,999.,270.]),np.array([0.,10.,999.,30.]))
    u,v=layer_average(sounding,[0.,50.,200.])
    # Valid profile after RDUP is (z,u)=(0,0),(100,10),(300,30).
    np.testing.assert_allclose(u,[2.5,12.5])
    np.testing.assert_allclose(v,0.,atol=1.e-12)
    with pytest.raises(ValueError,match='above top'): layer_average(sounding,[0.,301.])


def test_upper_air_temperature_faces_skip_missing_levels_like_intp():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),np.array([100.,200.,300.,400.]),
                      np.array([290.,999.9,270.,260.]),np.zeros(4),np.zeros(4))
    np.testing.assert_allclose(temperature_faces(sounding,[0.,100.,250.]),[290.,280.,265.])
    with pytest.raises(ValueError,match='above top'): temperature_faces(sounding,[0.,301.])
    missing=Sounding(datetime(2018,1,1),np.ones(2),np.array([0.,100.]),
                     np.array([999.9,280.]),np.zeros(2),np.zeros(2))
    with pytest.raises(ValueError,match='bottom'): temperature_faces(missing,[0.,50.])


def test_facet_intp_matches_gnu_real4_bits_with_missing_layer():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),
        np.array([13.2,115.7,248.9,510.4]),
        np.array([291.2345,999.9,273.4567,260.1234]),np.zeros(4),np.zeros(4))
    got=temperature_faces(sounding,[0.,77.7,199.3,400.2])
    np.testing.assert_array_equal(got.view(np.uint32),np.array(
        [0x43919e04,0x438eafdd,0x438a19e1,0x438488dc],dtype=np.uint32))


def test_upper_air_time_interpolates_layer_averages(tmp_path):
    path=tmp_path/'up.dat'
    path.write_text(_text(
        '1234 2018 1 1 0 2\n1000 15 20 270 0 900 115 10 270 10\n'
        '1234 2018 1 2 0 2\n1000 15 40 270 10 900 115 30 270 20\n'),encoding='ascii')
    data=UpperAirData(path)
    u,v=data.layer_wind_at(datetime(2018,1,1,12),[0.,100.])
    assert u.item()==pytest.approx(10.) and v.item()==pytest.approx(0.,abs=1.e-12)
    left,right=data.bracket(datetime(2018,1,2))
    assert right.timestamp-left.timestamp==timedelta(days=1)
    temperatures=data.temperature_faces_at(datetime(2018,1,1,12),[0.,100.])
    assert temperatures.dtype==np.float32
    np.testing.assert_allclose(temperatures,[30.,20.])


def test_prepdi_time_interpolation_preserves_fortran_real4_order():
    got=_prepdi_time_interpolate([-89.21386],[-23.326223],1,3)
    assert got.dtype==np.float32
    # GNU PREPDI expression BB-(BB-AA)*XFACT; the algebraically equivalent
    # AA+(1-XFACT)*(BB-AA) produces C23527B4 instead.
    assert got.view(np.uint32).item()==np.uint32(0xc23527b3)


def test_vertav_insertion_and_ordered_trapezoids_match_gnu_bits():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),
        np.array([10.,110.,260.,510.]),np.zeros(4),
        np.array([17.,123.,241.,319.]),np.array([2.3,7.8,11.2,5.6]))
    u,v=layer_average(sounding,[0.,75.,225.,400.])
    np.testing.assert_array_equal(
        u.view(np.uint32),np.array([0xc037e600,0xbf3f2400,0x41006f40],dtype=np.uint32))
    np.testing.assert_array_equal(
        v.view(np.uint32),np.array([0x3e5fa000,0x4090c600,0x403bb100],dtype=np.uint32))


@pytest.mark.parametrize('version',['2.0','1.0'])
def test_upper_air_legacy_versions_and_fixed_column_sounding_header(tmp_path,version):
    path=tmp_path/'up-old.dat'
    fixed=' '*9+f'{1234:8d}'+' '*5+f'{18:02d}{1:02d}{1:02d}{0:02d}'+' '*35+f'{2:5d}'
    path.write_text(f"""UP.DAT          {version:<16}legacy
0
LL
WGS-84  20200101
DEG
2018 1 0 2018 2 0 100 3 2
T T T T
1234 ABC 30N 120E 15
"""+fixed+'\n1000 15 20 270 5 900 1015 10 180 10\n',encoding='ascii')
    data=UpperAirData(path)
    assert data.version==version and data.soundings[0].timestamp==datetime(2018,1,1)
    assert data.soundings[0].pressure_mb.size==2


def test_upper_air_22_snapshot_has_explicit_seconds(tmp_path):
    path=tmp_path/'up22.dat'
    path.write_text("""UP.DAT          2.2             snapshot
0
LL
WGS-84  20200101
DEG
UTC+0000
2018 1 0 0 2018 1 1 0 100 3 2
T T T T
1234 ABC 30N 120E 15
1234 2018 1 1 0 1800 2018 1 1 0 1800 2
1000 15 20 270 5 900 1015 10 180 10
""",encoding='ascii')
    data=UpperAirData(path)
    assert data.soundings[0].timestamp==datetime(2018,1,1,0,30)


def test_selected_lapse_rate_uses_saved_convective_hour(tmp_path):
    path=tmp_path/'lapse.dat'
    path.write_text(_text(
        '1234 2018 1 1 0 2\n1000 15 300 270 0 900 215 296 270 0\n'
        '1234 2018 1 1 12 2\n1000 15 300 270 0 900 215 298 270 0\n'),encoding='ascii')
    data=UpperAirData(path); current=datetime(2018,1,1,6)
    assert data.lapse_rate_at(current,200.)==pytest.approx(-.015)
    corrected=data.lapse_rate_at(current,200.,datetime(2018,1,1,3))
    assert np.float32(corrected).view(np.uint32)==np.uint32(0xbc21a7ae)


def test_upper_air_22_time_averaged_record_is_constant_over_interval(tmp_path):
    path=tmp_path/'up22-average.dat'
    text="""UP.DAT          2.2             averaged
0
LL
WGS-84  20200101
DEG
UTC+0000
2018 1 0 0 2018 1 1 0 100 3 2
T T T T
1234 ABC 30N 120E 15
1234 2018 1 1 0 0 2018 1 1 1 0 2
1000 15 20 270 5 900 1015 10 180 10
"""
    path.write_text(text,encoding='ascii')
    data=UpperAirData(path)
    u,v=data.layer_wind_at(datetime(2018,1,1,0,30),[0.,100.])
    assert u.item()==pytest.approx(4.75) and v.item()==pytest.approx(.5)


def test_upper_temperature_horizontal_weights_use_grid_cell_distance():
    from pycalmet.config import GridConfig
    grid=GridConfig(2,1,1.,1.)
    profiles=np.array([[300.,299.,297.],[280.,279.,277.]])
    got=grid_temperature_faces(profiles,np.array([[.5,.5],[10.5,.5]]),grid,power=2)
    np.testing.assert_allclose(got[:,0,0],(profiles[0]*1.+profiles[1]*.01)/1.01)
    np.testing.assert_allclose(got[:,0,1],(profiles[0]*1.+profiles[1]/81.)/(1.+1./81.))
    assert got.dtype==np.float32


def test_temp3d_station_accumulation_matches_gnu_real4_bits():
    from pycalmet.config import GridConfig
    profiles=np.array([
        [290.125,275.875,261.5],
        [301.75,282.25,268.625],
        [287.375,279.5,270.875],
    ])
    stations=np.array([[.2,.4],[2.7,.8],[.9,3.1]])
    got=grid_temperature_faces(profiles,stations,GridConfig(2,1,1.,1.),power=2)
    expected=np.array([
        [0x4391ca44,0x4393abd1],
        [0x438a9c9d,0x438ba0fc],
        [0x4383ca01,0x4384e928],
    ],dtype=np.uint32)
    np.testing.assert_array_equal(got[:,0].view(np.uint32),expected)
