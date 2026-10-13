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




def test_upper_air_temperature_faces_skip_missing_levels_like_intp():
    sounding=Sounding(datetime(2018,1,1),np.ones(4),np.array([100.,200.,300.,400.]),
                      np.array([290.,999.9,270.,260.]),np.zeros(4),np.zeros(4))
    np.testing.assert_allclose(temperature_faces(sounding,[0.,100.,250.]),[290.,280.,265.])
    with pytest.raises(ValueError,match='above top'): temperature_faces(sounding,[0.,301.])
    missing=Sounding(datetime(2018,1,1),np.ones(2),np.array([0.,100.]),
                     np.array([999.9,280.]),np.zeros(2),np.zeros(2))
    with pytest.raises(ValueError,match='bottom'): temperature_faces(missing,[0.,50.])




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
