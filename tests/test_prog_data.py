import numpy as np
import pytest
from datetime import datetime

from pycalmet.config import GridConfig
from pycalmet.legacy_io import FortranSequentialWriter
from pycalmet.prog_data import ProgData
from pycalmet.calmet_solver import build_prog_step_fields
from pycalmet._calmet import _meteorology_input
from pycalmet.config import CalmetConfig


def _write_prog(path,*,levels=(10.,110.),hours=(1.,)):
    lower=np.array([[100.,200.],[300.,400.]],dtype='<f4')
    upper=lower+1000.
    fields=np.stack((lower,upper)) if len(levels)==2 else lower[None]
    with FortranSequentialWriter(path) as writer:
        for hour in hours:
            writer.write_record(np.array([hour],dtype='<f4'))
            writer.write_record(np.array([2,2,len(levels)],dtype='<i4'))
            writer.write_record(np.array([0.,0.,1.],dtype='<f4'))
            writer.write_record(np.asarray(levels,dtype='<f4'))
            for component in (fields*hour,fields*(2.*hour)):
                for level in component:
                    for row in level:
                        writer.write_record(row)




def test_prog_interpolation_uses_fortran_integer_index_at_west_boundary(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path)
    grid=GridConfig(1,1,1.,1.,x_origin_km=-.2,y_origin_km=.5)
    u,v=ProgData(path).interpolate(1.,grid,[60.])
    # INT(-0.2 + 1)-1 gives the invalid western candidate -1, so only
    # source column zero contributes. Truncating -0.2 before adding one
    # would incorrectly blend both source columns.
    np.testing.assert_allclose(u,[[[7.]]])
    np.testing.assert_allclose(v,[[[14.]]])


def test_prog_data_single_level_is_held_vertically(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,levels=(10.,))
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    u,v=ProgData(path).interpolate(1.,grid,[0.,10.,100.])
    np.testing.assert_allclose(u[:,0,0],2.5)
    np.testing.assert_allclose(v[:,0,0],5.)


def test_prog_data_rejects_truncated_record_sequence(tmp_path):
    path=tmp_path/'PROG.DAT'
    with FortranSequentialWriter(path) as writer:
        writer.write_record(np.array([1.],dtype='<f4'))
    with pytest.raises(ValueError,match='dimensions'):
        ProgData(path)


def test_prog_input_adapter_supplies_chronological_step_fields(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path)
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    config=CalmetConfig(path,'geo','out','2018010100','2018010101',grid,
        iprog=1,prognostic_timestep_seconds=3600,z_faces_m=(0.,120.))
    start=datetime(2018,1,1); end=datetime(2018,1,1,1)
    with _meteorology_input(config,(path,),tmp_path,start,end) as (dataset,provider):
        assert dataset['time_text'].shape==(2,19)
        step=provider(end)
    np.testing.assert_allclose(step.u,7.5)
    np.testing.assert_allclose(step.v,15.)


def test_prog_input_adapter_reproduces_subhour_boundary_cache(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(1.,1.,2.))
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    config=CalmetConfig(path,'geo','out','2018010100','2018010102',grid,
        iprog=2,iwfcod=1,timestep_seconds=900,prognostic_timestep_seconds=3600,
        z_faces_m=(0.,120.))
    start=datetime(2018,1,1); end=datetime(2018,1,1,2)
    with _meteorology_input(config,(path,),tmp_path,start,end) as (dataset,provider):
        assert dataset['time_text'].shape==(9,19)
        first=[provider(datetime(2018,1,1,0,minute)).u.item()
               for minute in (15,30,45)]
        at_one=provider(datetime(2018,1,1,1)).u.item()
        second=provider(datetime(2018,1,1,1,15)).u.item()
        at_two=provider(datetime(2018,1,1,2)).u.item()
    np.testing.assert_allclose(first+[at_one],[7.5]*4)
    assert second==pytest.approx(7.5)
    assert at_two==pytest.approx(15.)


def test_prog_subhour_adapter_reproduces_duplicate_boundary_read(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(1.,2.))
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    config=CalmetConfig(path,'geo','out','2018010100','2018010102',grid,
        iprog=2,iwfcod=1,timestep_seconds=900,prognostic_timestep_seconds=3600,
        z_faces_m=(0.,120.))
    with pytest.raises(ValueError,match='chronological hour sequence'):
        with _meteorology_input(
                config,(path,),tmp_path,datetime(2018,1,1),datetime(2018,1,1,2)):
            pass


def test_prog_forward_scan_skips_only_records_earlier_than_each_target(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(.5,1.,1.5,2.))
    data=ProgData(path)
    selected=data.records_for_hours((1.,2.))
    assert tuple(record.hour for record in selected)==(1.,2.)
    assert data.at_hour(1.).hour==1.

    out_of_order=tmp_path/'out-of-order.dat'
    _write_prog(out_of_order,hours=(2.,1.))
    with pytest.raises(ValueError,match='ending hour'):
        ProgData(out_of_order).at_hour(1.)
    with pytest.raises(ValueError,match='chronological'):
        ProgData(out_of_order).records_for_hours((1.,))


def test_prog_hour_labels_use_fortran_real_to_integer_truncation(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(np.nextafter(np.float32(2.),np.float32(0.)),))
    data=ProgData(path)
    assert int(np.float32(data.records[0].hour)*np.float32(100.))==199
    with pytest.raises(ValueError,match='ending hour'):
        data.at_hour(2.)
