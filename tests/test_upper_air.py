from datetime import datetime,timedelta
import os
import time
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


@pytest.mark.parametrize('wire_timezone',['UTC+0000','UTC-0000'])
def test_upper_air_normalizes_zero_utc_timezone(tmp_path,wire_timezone):
    path=tmp_path/'utc.dat'
    path.write_text(_text(
        '1234 2018 1 1 0 1\n1000 15 20 270 5\n'
    ).replace('UTC+0000',wire_timezone,1),encoding='ascii')

    data=UpperAirData(path)

    assert data.timezone=='UTC+0000'


@pytest.mark.parametrize('wire_timezone',['GMT+0000','UTC-0500','UTC+1260'])
def test_upper_air_rejects_invalid_or_nonzero_timezone(tmp_path,wire_timezone):
    path=tmp_path/'bad-timezone.dat'
    path.write_text(_text(
        '1234 2018 1 1 0 1\n1000 15 20 270 5\n'
    ).replace('UTC+0000',wire_timezone,1),encoding='ascii')

    with pytest.raises(ValueError,match=r'must use UTC\+0000'):
        UpperAirData(path)




def test_upper_air_temperature_faces_skip_missing_levels_like_intp():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),np.array([100.,200.,300.,400.]),
                      np.array([290.,999.9,270.,260.]),np.zeros(4),np.zeros(4))
    np.testing.assert_allclose(temperature_faces(sounding,[0.,100.,250.]),[290.,280.,265.])
    with pytest.raises(ValueError,match='above top'): temperature_faces(sounding,[0.,301.])
    missing=Sounding(datetime(2018,1,1),np.ones(2),np.array([0.,100.]),
                     np.array([999.9,280.]),np.zeros(2),np.zeros(2))
    with pytest.raises(ValueError,match='bottom'): temperature_faces(missing,[0.,50.])


def test_upper_air_layer_average_uses_direct_piecewise_linear_integration():
    sounding=Sounding(
        datetime(2018,1,1),np.ones(3),np.array([100.,200.,400.]),
        np.zeros(3),np.full(3,270.),np.array([1.,3.,7.]))
    u,v=layer_average(sounding,[0.,50.,200.,300.])
    np.testing.assert_allclose(u,[1.5,3.5,6.],rtol=0.,atol=1.e-6)
    np.testing.assert_allclose(v,0.,rtol=0.,atol=1.e-6)
    assert u.dtype==np.float32 and v.dtype==np.float32


def test_upper_air_time_interpolation_has_fp32_output_boundary():
    result=_prepdi_time_interpolate(
        np.array([1.,3.],dtype=np.float64),
        np.array([3.,7.],dtype=np.float64),1800,3600)
    np.testing.assert_array_equal(result,np.array([2.,5.],dtype=np.float32))
    assert result.dtype==np.float32




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


def test_upper_air_21_explicit_interval_is_snapshot_at_ending_time(tmp_path):
    path=tmp_path/'up21-explicit.dat'
    path.write_text("""UP.DAT          2.1             explicit times
0
LL
WGS-84  20200101
DEG
UTC+0000
2018 1 0 0 2018 1 1 0 100 3 2
T T T T
1234 ABC 30N 120E 15
1234 2018 1 1 0 900 2018 1 1 0 1800 2
1000 15 20 270 5 900 1015 10 180 10
""",encoding='ascii')
    data=UpperAirData(path)
    sounding=data.soundings[0]
    assert sounding.timestamp==datetime(2018,1,1,0,30)
    assert sounding.end_timestamp is None
    assert sounding.pressure_mb.size==2


@pytest.mark.parametrize('version',['2.1','2.2'])
@pytest.mark.parametrize(('hour','seconds'),[
    ('-1','0'),('25','0'),('0','-1'),('0','3601'),
    ('0.5','0'),('0','0.5'),
])
def test_upper_air_calendar_soundings_reject_invalid_clock_fields(
        tmp_path,version,hour,seconds):
    path=tmp_path/'invalid-calendar-time.dat'
    path.write_text(f"""UP.DAT          {version:<16}invalid calendar time
0
LL
WGS-84  20200101
DEG
UTC+0000
2018 1 0 0 2018 2 0 0 100 3 2
T T T T
1234 ABC 30N 120E 15
1234 2018 1 1 {hour} {seconds} 2018 1 1 1 0 1
1000 15 20 270 5
""",encoding='ascii')

    with pytest.raises(ValueError):
        UpperAirData(path)


def _sounding(timestamp,temperature=(20.,10.)):
    return Sounding(
        timestamp,
        np.array([1000.,900.]),
        np.array([0.,1000.]),
        np.asarray(temperature),
        np.array([270.,270.]),
        np.array([5.,5.]))


@pytest.mark.skipif(not hasattr(time,'tzset'),reason='requires POSIX TZ handling')
def test_upper_air_bracketing_is_independent_of_host_dst(monkeypatch):
    data=object.__new__(UpperAirData)
    data.soundings=tuple(_sounding(datetime(2021,3,14,hour,30))
                         for hour in (1,2,3))
    original=os.environ.get('TZ')
    try:
        # This spring-forward rule makes naive datetime.timestamp() map 02:30
        # and 03:30 to the same epoch instant on common libc implementations.
        monkeypatch.setenv('TZ','EST5EDT,M3.2.0/2,M11.1.0/2')
        time.tzset()
        left,right=data.bracket(datetime(2021,3,14,3,0))
        assert (left.timestamp,right.timestamp)==(
            datetime(2021,3,14,2,30),datetime(2021,3,14,3,30))
    finally:
        if original is None: monkeypatch.delenv('TZ',raising=False)
        else: monkeypatch.setenv('TZ',original)
        time.tzset()


def test_upper_air_subsecond_interpolation_preserves_timedelta_precision():
    start=datetime(2020,1,1)
    data=object.__new__(UpperAirData)
    data.soundings=(
        _sounding(start,(20.,10.)),
        _sounding(start+timedelta(microseconds=500000),(22.,12.)))
    result=data.temperature_faces_at(
        start+timedelta(microseconds=250000),(0.,1000.))
    np.testing.assert_allclose(result,[21.,11.],rtol=0.,atol=1.e-6)


def test_upper_air_convective_lapse_handles_subsecond_boundaries():
    start=datetime(2020,1,1)
    data=object.__new__(UpperAirData)
    data.soundings=(
        _sounding(start,(20.,10.)),
        _sounding(start+timedelta(seconds=1),(22.,12.)))
    result=data.lapse_rate_at(
        start+timedelta(microseconds=750000),1000.,
        last_convective_time=start+timedelta(microseconds=500000))
    assert result==pytest.approx(-.0099,abs=2.e-7)
