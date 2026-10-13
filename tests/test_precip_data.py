from datetime import datetime
import numpy as np
import pytest
import struct

from pycalmet.config import GridConfig
from pycalmet.legacy_io import FortranSequentialWriter,fixed_string
from pycalmet.precip_data import (PrecipitationData,grid_precipitation,
                                  read_unformatted_precipitation)
from pycalmet.surface_data import SurfaceStation


def test_formatted_precipitation_reads_explicit_intervals(tmp_path):
    path=tmp_path/'precip.dat'
    path.write_text("""PRECIP.DAT      2.1             test
0
LL
WGS-84  20200101
DEG
UTC-0000
2018 1 0 0 2018 1 2 0 2
1 A 30N 120E
2 B 31N 121E
2018 1 0 0 2018 1 1 0 0.5 9999
2018 1 1 0 2018 1 2 0 1.5 2.5
""",encoding='ascii')
    data=PrecipitationData(path)
    np.testing.assert_array_equal(data.at_begin(datetime(2018,1,1,0,30)).rate_mm_h,[.5,9999.])
    np.testing.assert_array_equal(data.at_begin(datetime(2018,1,1,1)).rate_mm_h,[1.5,2.5])


def test_formatted_legacy_precipitation_converts_hour_ending_records(tmp_path):
    path=tmp_path/'precip-legacy.dat'
    path.write_text("""PRECIP.DAT      2.0             test
0
LL
WGS-84  20200101
DEG
2018 1 1 2018 1 3 0 2
1 A 30N 120E
2 B 31N 121E
2018 1 1 0.5 9999
2018 1 2 1.5 2.5
2018 1 3 0.0 4.0
""",encoding='ascii')
    data=PrecipitationData(path)
    assert data.start==datetime(2018,1,1,0) and data.end==datetime(2018,1,1,3)
    assert data.timezone==0
    np.testing.assert_array_equal(data.at_begin(datetime(2018,1,1,0,30)).rate_mm_h,[.5,9999.])
    np.testing.assert_array_equal(data.at_begin(datetime(2018,1,1,2)).rate_mm_h,[0.,4.])


@pytest.mark.parametrize('method',[1,2,3])
def test_gride_precipitation_vectorized_methods_and_missing(method):
    grid=GridConfig(2,1,1.,1.,x_origin_km=0.,y_origin_km=0.)
    values=np.array([0.,10.,9999.]); points=np.array([[0.,0.],[2.,0.],[1.,0.]])
    result=grid_precipitation(values,points,grid,method,0. if method==3 else 50.,.01)
    assert result.shape==(1,2) and np.all(np.isfinite(result))
    assert result[0,0]<result[0,1] and np.all((result>=0.)&(result<=10.))
    zero=grid_precipitation([9999.],[[0.,0.]],grid,method,50.,.01)
    np.testing.assert_array_equal(zero,0.)


def test_gride_methods_match_gnu_real4_bits():
    grid=GridConfig(2,1,1.,1.)
    rates=[0.,7.25,2.5]
    points=[[.2,.3],[2.4,.6],[.8,2.8]]
    expected={
        1:[0x3fa7bd7c,0x407dddbb],
        2:[0x3e9aed10,0x409699d3],
        3:[0x3e4f928a,0x409df4e7],
    }
    for method in (1,2,3):
        got=grid_precipitation(rates,points,grid,method,3.,.01)
        np.testing.assert_array_equal(
            got.view(np.uint32),np.array([expected[method]],dtype=np.uint32))


@pytest.mark.parametrize('packing',[0,1])
def test_unformatted_precipitation_reads_packed_zero_runs(tmp_path,packing):
    path=tmp_path/f'precip-{packing}.dat'
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('PRECIP.DAT',16),fixed_string('2.1',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('LL',8))
        writer.write_record(fixed_string('WGS-84',8),fixed_string('20200101',12))
        writer.write_record(np.zeros(5,dtype='<i4')); writer.write_record(fixed_string('UTC-0000',8))
        writer.write_record(struct.pack('<6i',201800100,0,201800101,0,2,packing))
        writer.write_record(np.array([1,2],dtype='<i4')); writer.write_record(fixed_string('A',4),fixed_string('B',4))
        writer.write_record(fixed_string('30N',16),fixed_string('31N',16))
        writer.write_record(fixed_string('120E',16),fixed_string('121E',16))
        prefix=struct.pack('<4i',201800100,0,201800101,0)
        if packing:
            writer.write_record(prefix,struct.pack('<i',2)); writer.write_record(np.array([1.5,-1.],dtype='<f4'))
        else:
            writer.write_record(prefix,np.array([1.5,0.],dtype='<f4'))
    data=read_unformatted_precipitation(path)
    np.testing.assert_allclose(data.records[0].rate_mm_h,[1.5,0.])


@pytest.mark.parametrize('packing',[0,1])
def test_unformatted_legacy_precipitation_uses_hour_ending_records(tmp_path,packing):
    path=tmp_path/f'precip-legacy-{packing}.dat'
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('PRECIP.DAT',16),fixed_string('2.0',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('LL',8))
        writer.write_record(fixed_string('WGS-84',8),fixed_string('20200101',12))
        writer.write_record(struct.pack('<5i',201800101,201800102,0,2,packing))
        writer.write_record(np.array([1,2],dtype='<i4')); writer.write_record(fixed_string('A',4),fixed_string('B',4))
        writer.write_record(fixed_string('30N',16),fixed_string('31N',16))
        writer.write_record(fixed_string('120E',16),fixed_string('121E',16))
        for end_code in (201800101,201800102):
            if packing:
                writer.write_record(struct.pack('<2i',end_code,2))
                writer.write_record(np.array([1.5,-1.],dtype='<f4'))
            else:
                writer.write_record(struct.pack('<i',end_code),np.array([1.5,0.],dtype='<f4'))
    data=read_unformatted_precipitation(path)
    assert data.start==datetime(2018,1,1,0) and data.end==datetime(2018,1,1,2)
    np.testing.assert_allclose(data.at_begin(datetime(2018,1,1,1)).rate_mm_h,[1.5,0.])


def test_formatted_precipitation_none_uses_control_station_metadata(tmp_path):
    path=tmp_path/'precip-none.dat'
    path.write_text("""PRECIP.DAT      2.1             test
0
NONE
UTC-0000
2018 1 0 0 2018 1 1 0 1
101
2018 1 0 0 2018 1 1 0 1.5
""",encoding='ascii')
    stations=(SurfaceStation(101,'AAA',30.,120.,0.),)
    data=PrecipitationData(path,station_metadata=stations,datum='WGS-84')
    assert data.stations==stations
    np.testing.assert_array_equal(data.records[0].rate_mm_h,[1.5])


def test_unformatted_precipitation_none_uses_control_station_metadata(tmp_path):
    path=tmp_path/'precip-none.bin'; stations=(SurfaceStation(101,'AAA',30.,120.,0.),)
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('PRECIP.DAT',16),fixed_string('2.1',16),fixed_string('test',64))
        writer.write_record(struct.pack('<i',0)); writer.write_record(fixed_string('NONE',8))
        writer.write_record(fixed_string('UTC-0000',8))
        writer.write_record(struct.pack('<6i',201800100,0,201800101,0,1,0))
        writer.write_record(np.array([101],dtype='<i4'))
        writer.write_record(struct.pack('<4i',201800100,0,201800101,0),np.array([1.5],dtype='<f4'))
    data=read_unformatted_precipitation(path,station_metadata=stations,datum='WGS-84')
    assert data.stations==stations
    np.testing.assert_allclose(data.records[0].rate_mm_h,[1.5])
