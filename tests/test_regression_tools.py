from pathlib import Path
import importlib.util
from io import BytesIO

import numpy as np
import pytest


def _module(name):
    path=Path(__file__).parents[1]/'tools'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calwrf_m2d_regression_compares_every_record(monkeypatch):
    from types import SimpleNamespace

    regression=_module('regress_calwrf_m2d')
    header=SimpleNamespace(nx=2,ny=1,nz=1,nhours=1,start=2018031800,
                           variables=('WD10','WS10'),source_nx=4,source_ny=3,
                           source_nz=2,i_start=2,j_start=2,k_start=1)
    reference=(SimpleNamespace(date=2018031800,name='WD10',
                               values=np.array([[1.,2.]],dtype=np.float32)),
               SimpleNamespace(date=2018031800,name='WS10',
                               values=np.array([[3.,4.]],dtype=np.float32)))
    changed=(reference[0],SimpleNamespace(date=2018031800,name='WS10',
             values=np.array([[3.,5.]],dtype=np.float32)))
    records={'reference':reference,'same':reference,'changed':changed}
    class Reader:
        def __init__(self,path): self.path=path
        def records(self): return header,iter(records[self.path])
    monkeypatch.setattr(regression,'M2DReader',Reader)
    rows=regression.compare('reference','same',hours=1)
    assert len(rows)==2 and all(row[2]==1. for row in rows)
    rows=regression.compare('reference','changed',hours=1)
    assert next(row for row in rows if row[1]=='WS10')[2]==.5


def test_calwrf_m2d_byte_regression_reports_first_changed_line(monkeypatch):
    regression=_module('regress_calwrf_m2d')
    # nx=8, ny=1 and one variable means one label plus one grid line per hour.
    streams={
        'reference': BytesIO(b'label\nvalues\n'),
        'same': BytesIO(b'label\nvalues\n'),
        'changed': BytesIO(b'label\nVALUES\n'),
    }
    monkeypatch.setattr(regression,'_data_stream',
        lambda path: (streams[path],8,1,1,1))
    assert regression.compare_record_bytes('reference','same',hours=1)==2
    streams['reference']=BytesIO(b'label\nvalues\n')
    with pytest.raises(ValueError,match='line 2'):
        regression.compare_record_bytes('reference','changed',hours=1)


def test_calwrf_m3d_regression_compares_every_value(monkeypatch):
    from types import SimpleNamespace

    regression=_module('regress_calwrf_m3d')
    array=np.array([[1.]],dtype=np.float32)
    header=SimpleNamespace(projection='LCC',latitude_origin=30.,
        longitude_origin=120.,standard_parallel_1=30.,standard_parallel_2=60.,
        x_origin_km=0.,y_origin_km=0.,spacing_km=3.,nx=1,ny=1,nz=1,
        nhours=1,start=2018031800,sigma=array,latitude=array,longitude=array,
        terrain=array,landuse=np.array([[1]],dtype=np.int16),
        source_nx=2,source_ny=2,source_nz=2,i_start=2,j_start=2,k_start=2)
    cell=SimpleNamespace(date=2018031800,i=1,j=1,
        surface=np.arange(10,dtype=np.float32),
        levels=np.arange(8,dtype=np.float32).reshape(1,8),
        water=np.arange(5,dtype=np.float32).reshape(1,5))
    changed=SimpleNamespace(date=cell.date,i=cell.i,j=cell.j,
        surface=cell.surface.copy(),levels=cell.levels.copy(),water=cell.water.copy())
    changed.water[0,4]=99.
    records={'reference':cell,'same':cell,'changed':changed}
    class Reader:
        def __init__(self,path): self.path=path
        def cells(self): return header,iter((records[self.path],))
    monkeypatch.setattr(regression,'M3DReader',Reader)
    result=regression.compare('reference','same',hours=1)
    assert all(equal==total for equal,total,_maximum in result.values())
    result=regression.compare('reference','changed',hours=1)
    assert result['water'][:2]==(4,5)


def test_calwrf_m3d_byte_regression_reports_first_changed_record(monkeypatch):
    regression=_module('regress_calwrf_m3d')
    streams={
        'reference': BytesIO(b'one\ntwo\nthree\n'),
        'same': BytesIO(b'one\ntwo\nthree\n'),
        'changed': BytesIO(b'one\nTWO\nthree\n'),
    }
    monkeypatch.setattr(regression,'_data_stream',
        lambda path: (streams[path],1,1,2))
    assert regression.compare_record_bytes('reference','same',hours=1)==3
    streams['reference']=BytesIO(b'one\ntwo\nthree\n')
    with pytest.raises(ValueError,match='record 2'):
        regression.compare_record_bytes('reference','changed',hours=1)


def test_terrel_regression_compares_every_cell(monkeypatch):
    from types import SimpleNamespace

    regression=_module('regress_terrel')
    monkeypatch.setattr(regression,'terrel_config',
                        lambda _path: SimpleNamespace(grid=object()))
    fields={'reference':np.array([[1.,2.]],dtype=np.float64),
            'same':np.array([[1.,2.]],dtype=np.float64),
            'changed':np.array([[1.,3.]],dtype=np.float64)}
    monkeypatch.setattr(regression,'_read_terrain',
                        lambda path,_grid: fields[path])
    assert regression.compare('control','reference','same')==(2,2,0.)
    assert regression.compare('control','reference','changed')==(1,2,1.)


@pytest.mark.parametrize('module,helper,message',(
    ('regress_terrel','_terrain_data_lines','TERREL data line 2'),
    ('regress_ctgproc','_landuse_data_lines','CTGPROC data line 2'),
))
def test_preprocessor_byte_regressions_report_changed_line(monkeypatch,module,helper,message):
    regression=_module(module)
    records={'reference':[b'one\n',b'two\n'],'same':[b'one\n',b'two\n'],
             'changed':[b'one\n',b'TWO\n']}
    monkeypatch.setattr(regression,helper,lambda path: records[path])
    assert regression.compare_data_lines('same','reference')==2
    with pytest.raises(ValueError,match=message):
        regression.compare_data_lines('changed','reference')


def test_makegeo_byte_regression_reports_first_changed_data_line(monkeypatch):
    regression=_module('regress_makegeo')
    records={'reference':[b'one\n',b'two\n'],'same':[b'one\n',b'two\n'],
             'changed':[b'one\n',b'TWO\n']}
    monkeypatch.setattr(regression,'_legacy_data_lines',lambda path: records[path])
    assert regression.compare_data_lines('same','reference')==2
    with pytest.raises(ValueError,match='line 2'):
        regression.compare_data_lines('changed','reference')


@pytest.mark.parametrize('module,message',(
    ('regress_calmet','CALMET period record 2'),
    ('regress_aux','CALMET.AUX hour 1 record 2'),
))
def test_calmet_byte_regressions_report_changed_record(monkeypatch,module,message):
    regression=_module(module)
    records={'reference':[[b'one',b'two']],'same':[[b'one',b'two']],
             'changed':[[b'one',b'TWO']]}
    helper='_period_payloads' if module=='regress_calmet' else '_hour_payloads'
    monkeypatch.setattr(regression,helper,lambda path,_limit: records[path]
                        if module=='regress_aux' else records[path][0])
    if module=='regress_calmet':
        assert regression.compare_record_bytes('reference','same',object())==2
        call=lambda: regression.compare_record_bytes('reference','changed',object())
    else:
        assert regression.compare_record_bytes('reference','same',1)==2
        call=lambda: regression.compare_record_bytes('reference','changed',1)
    with pytest.raises(ValueError,match=message): call()


def test_ctgproc_regression_accepts_legacy_candidate(monkeypatch):
    regression=_module('regress_ctgproc')
    reference=np.zeros((2,1,2),dtype=np.float32)
    reference[0,0,0]=1.; reference[1,0,1]=1.
    changed=reference.copy(); changed[:,0,0]=(.4,.6)
    fields={'reference.dat':reference,'same.dat':reference,'changed.dat':changed}
    monkeypatch.setattr(regression,'read_legacy',
                        lambda path,_nx,_ny,_categories: fields[path])
    assert regression.compare('same.dat','reference.dat',2,1,2)==(1.,1.,0.,0.)
    precision,dominant,_mean,maximum=regression.compare(
        'changed.dat','reference.dat',2,1,2)
    assert precision==.5 and dominant==.5 and maximum==pytest.approx(.6)


def test_makegeo_regression_checks_all_fields(monkeypatch):
    regression=_module('regress_makegeo')
    fields={name:np.ones((1,2),dtype=np.float32) for name in regression.NAMES}
    changed={name:value.copy() for name,value in fields.items()}
    changed['leaf_area_index'][0,1]=2.
    monkeypatch.setattr(regression,'_fields',
                        lambda path: changed if path=='changed.dat' else fields)
    rows=regression.compare('same.dat','reference.dat')
    assert all(equal==total for _name,equal,total,*_rest in rows)
    rows=regression.compare('changed.dat','reference.dat')
    leaf=next(row for row in rows if row[0]=='leaf_area_index')
    assert leaf[1:3]==(1,2) and leaf[4]==1.
