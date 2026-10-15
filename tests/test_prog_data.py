import numpy as np
import pytest
from datetime import datetime,timedelta

from pycalmet.config import GridConfig
from pycalmet.legacy_io import FortranSequentialWriter
from pycalmet.prog_data import ProgData,_hour_label
from pycalmet.legacy._compat.prog_data import ProgData as LegacyProgData
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




def test_prog_interpolation_uses_floor_cell_index_at_west_boundary(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path)
    grid=GridConfig(1,1,1.,1.,x_origin_km=-.2,y_origin_km=.5)
    u,v=ProgData(path).interpolate(1.,grid,[60.])
    np.testing.assert_allclose(u,[[[7.]]])
    np.testing.assert_allclose(v,[[[14.]]])

    outside=GridConfig(1,1,1.,1.,x_origin_km=-1.2,y_origin_km=.5)
    with pytest.raises(ValueError,match='cannot interpolate'):
        ProgData(path).interpolate(1.,outside,[60.])
    legacy_u,_=LegacyProgData(path).interpolate(1.,outside,[60.])
    np.testing.assert_allclose(legacy_u,[[[7.]]])


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


def test_prog_input_adapter_selects_full_datetimes_across_midnight(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(23.,24.,1.,2.))
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    config=CalmetConfig(path,'geo','out','2018010122','2018010202',grid,
        iprog=1,prognostic_timestep_seconds=3600,z_faces_m=(0.,120.))
    start=datetime(2018,1,1,22); end=datetime(2018,1,2,2)
    with _meteorology_input(config,(path,),tmp_path,start,end) as (_,provider):
        values=[provider(timestamp).u.item() for timestamp in (
            datetime(2018,1,1,23),datetime(2018,1,2),
            datetime(2018,1,2,1),datetime(2018,1,2,2))]
    np.testing.assert_allclose(values,np.array([23.,24.,1.,2.])*7.5)


def test_prog_input_adapter_reuses_each_hourly_record_for_its_substeps(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(1.,2.))
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
    assert second==pytest.approx(15.)
    assert at_two==pytest.approx(15.)


def test_prog_subhour_adapter_does_not_require_duplicate_boundary_records(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(1.,2.))
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    config=CalmetConfig(path,'geo','out','2018010100','2018010102',grid,
        iprog=2,iwfcod=1,timestep_seconds=900,prognostic_timestep_seconds=3600,
        z_faces_m=(0.,120.))
    with _meteorology_input(
            config,(path,),tmp_path,datetime(2018,1,1),datetime(2018,1,1,2)) as (_,provider):
        assert provider(datetime(2018,1,1,1,15)).u.item()==pytest.approx(15.)


def test_prog_forward_scan_skips_only_records_earlier_than_each_target(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(.5,1.,1.5,2.))
    data=ProgData(path)
    selected=data.records_for_times((datetime(2018,1,1,1),datetime(2018,1,1,2)))
    assert tuple(record.hour for record in selected)==(1.,2.)
    assert data.at_hour(1.).hour==1.

    out_of_order=tmp_path/'out-of-order.dat'
    _write_prog(out_of_order,hours=(2.,1.))
    with pytest.raises(ValueError,match='out of chronological order'):
        ProgData(out_of_order).at_hour(1.)
    with pytest.raises(ValueError,match='out of chronological order'):
        ProgData(out_of_order).records_for_times((datetime(2018,1,1,1),))


def test_prog_record_selection_unwraps_24_across_midnight(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(23.,24.,1.,2.))
    targets=(datetime(2018,1,1,23),datetime(2018,1,2),
             datetime(2018,1,2,1),datetime(2018,1,2,2))
    selected=ProgData(path).records_for_times(targets)
    assert tuple(record.hour for record in selected)==(23.,24.,1.,2.)


def test_prog_record_selection_treats_zero_as_midnight_alias(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(23.,0.,1.,2.))
    targets=(datetime(2018,1,1,23),datetime(2018,1,2),
             datetime(2018,1,2,1),datetime(2018,1,2,2))
    selected=ProgData(path).records_for_times(targets)
    assert tuple(record.hour for record in selected)==(23.,0.,1.,2.)


def test_prog_record_selection_spans_multiple_days(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(22.,23.,24.,1.,2.,23.,24.,1.))
    targets=(datetime(2018,1,1,23),datetime(2018,1,2),
             datetime(2018,1,2,2),datetime(2018,1,2,23),
             datetime(2018,1,3),datetime(2018,1,3,1))
    selected=ProgData(path).records_for_times(targets)
    assert tuple(record.hour for record in selected)==(23.,24.,2.,23.,24.,1.)


def test_prog_record_selection_reports_missing_datetime(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(23.,24.,2.))
    targets=(datetime(2018,1,1,23),datetime(2018,1,2),datetime(2018,1,2,1))
    with pytest.raises(ValueError,match='2018-01-02 01:00:00'):
        ProgData(path).records_for_times(targets)


@pytest.mark.parametrize('times',[
    (datetime(2018,1,1,2),datetime(2018,1,1,1)),
    (datetime(2018,1,1,1),datetime(2018,1,1,1)),
])
def test_prog_record_selection_rejects_unordered_targets(tmp_path,times):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(1.,2.))
    with pytest.raises(ValueError,match='strictly chronological'):
        ProgData(path).records_for_times(times)


def test_prog_hour_labels_round_binary_float_noise_to_clock_time(tmp_path):
    path=tmp_path/'PROG.DAT'
    _write_prog(path,hours=(np.nextafter(np.float32(2.),np.float32(0.)),))
    data=ProgData(path)
    assert data.at_hour(2.).hour==pytest.approx(2.)
    with pytest.raises(ValueError,match='ending hour'):
        LegacyProgData(path).at_hour(2.)


def test_prog_hour_labels_use_datetime_clock_parsing():
    assert _hour_label(1.30)==timedelta(hours=1,minutes=30)
    assert _hour_label(24.)==timedelta(0)
    with pytest.raises(ValueError,match='invalid PROG.DAT hour'):
        _hour_label(1.60)


@pytest.mark.parametrize('hour',[1.234,24.004,.005])
def test_prog_hour_labels_reject_silent_rounding(hour):
    with pytest.raises(ValueError,match='invalid PROG.DAT hour'):
        _hour_label(hour)
