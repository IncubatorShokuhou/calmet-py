from datetime import datetime
from types import SimpleNamespace

import numpy as np

from pycalmet.wind_debug import (FinalWindDebugWriter,WindPrintWriter,
                                 format_wind_number_map)


def test_final_wind_debug_writer_matches_outfil_record_order(tmp_path):
    u=np.array([[[1.,2.],[3.,4.]]],dtype=np.float32)
    v=u+10
    w=np.array([[[0.,.1],[.2,.3]],[[.4,.5],[.6,.7]]],dtype=np.float32)
    path=tmp_path/'TEST.OUT'
    with FinalWindDebugWriter(path,2,2,1,1.,2.) as writer:
        writer.append(SimpleNamespace(timestamp=datetime(2018,3,18),
                                      wind=SimpleNamespace(u=u,v=v,w=w)))
        writer.append(SimpleNamespace(timestamp=datetime(2018,3,18,1),
                                      wind=SimpleNamespace(u=u,v=v,w=w)))
    lines=path.read_text(encoding='ascii').splitlines()
    assert lines[:7]==['    0.','   2   2   1','  1.  2.',
                       '   1.00   2.00','   3.00   4.00',
                       '  11.00  12.00','  13.00  14.00']
    assert lines[7:11]==[' 0.0E+00 1.0E-01',' 2.0E-01 3.0E-01',
                         ' 4.0E-01 5.0E-01',' 6.0E-01 7.0E-01']
    assert lines[11]=='  100.'


def test_final_wind_debug_writer_wraps_fortran_59_value_records(tmp_path):
    u=np.arange(60,dtype=np.float32).reshape(1,1,60)
    w=np.zeros((2,1,60),dtype=np.float32)
    path=tmp_path/'wide.out'
    with FinalWindDebugWriter(path,60,1,1,1.,1.) as writer:
        writer.append(SimpleNamespace(timestamp=datetime(2018,3,18),
            wind=SimpleNamespace(u=u,v=u,w=w)))
    lines=path.read_text(encoding='ascii').splitlines()
    assert len(lines[3])==59*7 and lines[4].strip()=='59.00'


def test_wind_number_map_scales_rounds_and_reverses_rows():
    text=format_wind_number_map(np.array([[.11,-.16],[.25,0.]],dtype=np.float32))
    rows=[line for line in text.splitlines() if line.startswith('    1 I') or
          line.startswith('    2 I')]
    assert rows==['    2 I 25  0','    1 I 11-16']
    assert 'ARRAY HAS BEEN SCALED BY  0.1E+03' in text


def test_wind_print_writer_writes_retained_and_final_stages(tmp_path):
    one=np.ones((1,1,1),dtype=np.float32)
    faces=np.zeros((2,1,1),dtype=np.float32)
    stage=(one,one*2,faces)
    wind=SimpleNamespace(u=one,v=one*2,w=faces,
        kinematic_stage=stage,froude_stage=stage,slope_stage=stage)
    hour=SimpleNamespace(timestamp=datetime(2018,3,18),wind=wind)
    path=tmp_path/'TEST.PRT'
    with WindPrintWriter(path,(0,0,0,0,0,1,1,1,1)) as writer:
        writer.append(hour)
    text=path.read_text(encoding='ascii')
    assert 'KINEMATIC EFFECTS' in text
    assert 'FROUDE NUMBER EFFECTS' in text
    assert 'SLOPE EFFECTS' in text
    assert 'FINAL WIND FIELD AT TIME' in text
    assert text.count('X-COMPONENT OF WIND')==4


def test_wind_print_writer_writes_divergence(tmp_path):
    u=np.array([[[0.,1.]],[[1.,0.]]],dtype=np.float32)
    v=np.array([[[0.,0.]],[[1.,-1.]]],dtype=np.float32)
    divergence=np.array([[[.1,.2]],[[.3,.4]]],dtype=np.float32)
    wind=SimpleNamespace(u=u,v=v,w=np.zeros((3,1,2),dtype=np.float32),
                         divergence=divergence)
    hour=SimpleNamespace(timestamp=datetime(2018,3,18),wind=wind)
    path=tmp_path/'TEST.PRT'
    with WindPrintWriter(path,(0,0,0,0,1,0,0,0,0),levels=1) as writer:
        writer.append(hour)
    text=path.read_text(encoding='ascii')
    assert text.count('DIVERGENCE AT LEVEL')==1


def test_wind_print_writer_writes_final_speed_and_direction(tmp_path,monkeypatch):
    u=np.array([[[0.,0.,1.,-1.]],[[3.,4.,5.,6.]]],dtype=np.float32)
    v=np.array([[[0.,-1.,0.,0.]],[[4.,3.,12.,8.]]],dtype=np.float32)
    wind=SimpleNamespace(u=u,v=v,w=np.zeros((3,1,4),dtype=np.float32))
    hour=SimpleNamespace(timestamp=datetime(2018,3,18),wind=wind)
    printed=[]
    monkeypatch.setattr('pycalmet.wind_debug.format_wind_number_map',
                        lambda values: printed.append(np.asarray(values).copy()) or '<map>\n')
    path=tmp_path/'TEST.PRT'
    with WindPrintWriter(path,(0,0,0,1,0,0,0,0,0),levels=1) as writer:
        writer.append(hour)

    text=path.read_text(encoding='ascii')
    assert 'FINAL WIND SPEED AND DIRECTION AT TIME' in text
    assert text.count('WIND SPEED (M/S) AT LEVEL')==1
    assert text.count('WIND DIRECTION (DEGREES) AT LEVEL')==1
    np.testing.assert_allclose(printed[0],[[0.,1.,1.,1.]])
    np.testing.assert_allclose(printed[1],[[0.,360.,270.,90.]])


def test_wind_print_writer_writes_initial_adjusted_and_smoothed_fields(tmp_path):
    u=np.ones((1,1,1),dtype=np.float32)
    v=u*2; w=np.zeros((2,1,1),dtype=np.float32)
    divergence=np.full_like(u,np.float32(1.e-4))
    wind=SimpleNamespace(u=u,v=v,w=w,divergence=divergence,
        interpolated_stage=(u,v),terrain_adjusted_stage=(u,v),
        initial_stage=(u,v,w,divergence))
    hour=SimpleNamespace(timestamp=datetime(2018,3,18),wind=wind)
    path=tmp_path/'TEST.PRT'
    with WindPrintWriter(path,(1,1,1,0,0,0,0,0,0),levels=1) as writer:
        writer.append(hour)
    text=path.read_text(encoding='ascii')
    assert 'INTERPOLATED WIND FIELD' in text
    assert 'TERRAIN ADJUSTED SURFACE WINDS' in text
    assert 'SMOOTHED WIND AND INITIAL DIVERGENCE FIELDS' in text
    assert 'MAXIMUM DIVERGENCE =   0.1000E-03' in text
