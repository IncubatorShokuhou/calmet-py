from datetime import datetime,timedelta
import struct

import numpy as np
import pytest

from pycalmet.calmet_output import LegacyCalmetWriter
from pycalmet.config import GridConfig
from pycalmet.igf_data import (_r2interpi_four,_rdcalmet_rotate,_rdcalmet_vertical,
                               InitialGuessCalmetData,InitialGuessCalmetSeries,
                               interpolate_initial_guess)
from pycalmet.legacy_io import FortranSequentialWriter,fixed_string


def _write_igf(path,*,surface_station=False,complete=False,misordered=False):
    grid=GridConfig(2,2,2.,2.,projection='LCC',lat0=30.,lon0=-120.,
                    lat1=30.,lat2=60.,x_origin_km=10.,y_origin_km=20.)
    geo={'roughness_length':np.array([[.1,.2],[.3,.4]]),
         'dominant_landuse':np.array([[20,55],[30,40]],dtype=np.int32),
         'terrain_height':np.array([[100.,200.],[300.,400.]]),
         'leaf_area_index':np.array([[1.,2.],[3.,4.]])}
    start=datetime(2018,1,1)
    writer=LegacyCalmetWriter(path,grid,[0.,20.,60.],geo,start,
        datetime(2018,1,1,1),run_type=(1 if complete else 0),timestep_seconds=3600,
        include_w_temperature=complete,
        surface_coordinates_km=([(10.5,20.5)] if surface_station else ()),
        nearest_surface_station=(np.ones((2,2),dtype=np.int32)
                                 if surface_station else None))
    writer._field_record('U-LEV  1',[[1.,2.],[3.,4.]],start,start)
    writer._field_record('V-LEV  1',[[5.,6.],[7.,8.]],start,start)
    if complete: writer._field_record('WFACE  1',np.zeros((2,2)),start,start)
    writer._field_record('U-LEV  2',[[9.,10.],[11.,12.]],start,start)
    writer._field_record('V-LEV  2',[[13.,14.],[15.,16.]],start,start)
    if complete:
        writer._field_record('WFACE  2',np.zeros((2,2)),start,start)
        writer._field_record('T-LEV  1',np.full((2,2),290.),start,start)
        writer._field_record('T-LEV  2',np.full((2,2),288.),start,start)
        labels=['IPGT','USTAR','ZI','EL','WSTAR','RMM','TEMPK','RHO','QSW','IRH','IPCODE']
        if misordered: labels[-2:]=reversed(labels[-2:])
        for label in labels:
            writer._field_record(label,np.ones((2,2)),start,start,
                                 integer=label in {'IPGT','IRH','IPCODE'})
    writer.close()
    return start


def _old_field(writer,label,values,end_code=0,integer=False):
    dtype='<i4' if integer else '<f4'
    writer.write_record(fixed_string(label,8),struct.pack('<i',end_code),
                        np.asarray(values,dtype=dtype))


def _old_body(writer,*,modern_2d=False):
    _old_field(writer,'ZFACE',[0.,20.,60.])
    _old_field(writer,'Z0',[.1,.2,.3,.4])
    _old_field(writer,'ILANDU',[20,55,30,40],integer=True)
    _old_field(writer,'ELEV',[100.,200.,300.,400.])
    _old_field(writer,'XLAI',[1.,2.,3.,4.])
    _old_field(writer,'U-LEV  1',[1.,2.,3.,4.],201800101)
    _old_field(writer,'V-LEV  1',[5.,6.,7.,8.],201800101)
    _old_field(writer,'WFACE  1',[0.,0.,0.,0.],201800101)
    _old_field(writer,'U-LEV  2',[9.,10.,11.,12.],201800101)
    _old_field(writer,'V-LEV  2',[13.,14.,15.,16.],201800101)
    _old_field(writer,'WFACE  2',[0.,0.,0.,0.],201800101)
    _old_field(writer,'T-LEV  1',[290.,290.,290.,290.],201800101)
    _old_field(writer,'T-LEV  2',[288.,288.,288.,288.],201800101)
    for label in ('IPGT','USTAR','ZI','EL','WSTAR'):
        _old_field(writer,label,[1.,1.,1.,1.],201800101,integer=label=='IPGT')
    # Pre-2.1 declared CALMET.DAT uses full 2-D arrays; the older three-title
    # format uses NSSTA-length arrays (zero here).
    for label in ('TEMPK','RHO','QSW','IRH'):
        _old_field(writer,label,([1.,1.,1.,1.] if modern_2d else []),201800101,
                   integer=label=='IRH')


def _write_old_igf(path):
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('CALMET.DAT',16),fixed_string('2.0',16),
                            fixed_string('synthetic regression file',64))
        writer.write_record(struct.pack('<i',0))
        control=b''.join((
            struct.pack('<10i',2018,1,1,1,5,1,1,2,2,2),
            struct.pack('<3f',2000.,10000.,20000.),
            struct.pack('<9i',0,0,0,0,0,14,50,55,1),
            fixed_string('LCC',8),fixed_string('WGS-84',8),fixed_string('2018010101',12),
            struct.pack('<2f',0.,0.),fixed_string('N',4),struct.pack('<i',0),
            struct.pack('<4f',30.,-120.,30.,60.)))
        assert len(control)==148
        writer.write_record(control)
        _old_body(writer,modern_2d=True)


def _write_ancient_igf(path):
    with FortranSequentialWriter(path) as writer:
        writer.write_record(*(fixed_string(f'legacy title {number}',80) for number in range(1,4)))
        control=b''.join((
            fixed_string('5.2',8),fixed_string('980304',8),
            struct.pack('<10i',2018,1,1,1,5,1,1,2,2,2),
            struct.pack('<3f',2000.,10000.,20000.),struct.pack('<i',0),
            struct.pack('<9i',0,0,0,0,0,14,50,55,1)))
        assert len(control)==108
        writer.write_record(control)
        writer.write_record(struct.pack('<2fif4f',30.,120.,1,1.,30.,60.,30.,120.))
        _old_body(writer)


def _write_pre_projection_igf(path):
    with FortranSequentialWriter(path) as writer:
        writer.write_record(*(fixed_string(f'legacy title {number}',80)
                              for number in range(1,4)))
        control=b''.join((
            fixed_string('5.1',8),fixed_string('970101',8),
            struct.pack('<10i',2018,1,1,1,5,1,1,2,2,2),
            struct.pack('<3f',2000.,500000.,3000000.),struct.pack('<i',50),
            struct.pack('<9i',0,0,0,0,0,14,50,55,1)))
        assert len(control)==108
        writer.write_record(control)
        # The pre-980304 format proceeds directly to ZFACE: it has no
        # 32-byte projection extension record.
        _old_body(writer)


def test_initial_guess_reader_loads_modern_header_static_and_winds(tmp_path):
    path=tmp_path/'calmet.dat'; start=_write_igf(path)
    data=InitialGuessCalmetData(path)
    assert data.version=='2.1' and data.run_type==0
    assert (data.grid.nx,data.grid.ny,data.grid.dx_km)==(2,2,2.)
    np.testing.assert_allclose(data.z_faces_m,[0.,20.,60.])
    np.testing.assert_allclose(data.z_centers_m,[10.,40.])
    np.testing.assert_allclose(data.roughness_m,[[.1,.2],[.3,.4]],rtol=1.e-6)
    begin,u,v=data.at_end(start.replace(hour=1))
    assert begin==start
    np.testing.assert_allclose(u,[[[1.,2.],[3.,4.]],[[9.,10.],[11.,12.]]])
    np.testing.assert_allclose(v,[[[5.,6.],[7.,8.]],[[13.,14.],[15.,16.]]])
    with pytest.raises(ValueError,match='no field record'):
        data.fields_at_end(start.replace(hour=1))


def test_initial_guess_hour_covers_subhourly_calmet_steps(tmp_path):
    path=tmp_path/'calmet.dat'; start=_write_igf(path)
    data=InitialGuessCalmetData(path)
    begin,u,_=data.at_covering(start+timedelta(minutes=15))
    assert begin==start and u[1,1,1]==12.
    series=InitialGuessCalmetSeries((path,))
    begin,u,_=series.at_covering(start+timedelta(minutes=45))
    assert begin==start and u[0,0,0]==1.
    with pytest.raises(ValueError,match='covering'):
        series.at_covering(start)


def test_initial_guess_reader_consumes_nearest_surface_station_record(tmp_path):
    path=tmp_path/'calmet-with-surface.dat'; _write_igf(path,surface_station=True)
    data=InitialGuessCalmetData(path)
    assert data.surface_station_count==1
    np.testing.assert_array_equal(data.nearest_surface_station,np.ones((2,2),dtype=np.int32))
    _,u,v=data.at_end(datetime(2018,1,1,1))
    np.testing.assert_allclose(u[:,0,0],[1.,9.])
    np.testing.assert_allclose(v[:,1,1],[8.,16.])


def test_initial_guess_reader_validates_complete_lcalgrd_period(tmp_path):
    path=tmp_path/'complete-calmet.dat'; _write_igf(path,complete=True)
    data=InitialGuessCalmetData(path,retain_fields=True)
    assert data.includes_w_temperature
    ending=datetime(2018,1,1,1)
    assert ending in data.records
    begin,fields=data.fields_at_end(ending)
    assert begin==datetime(2018,1,1)
    assert tuple(fields)==(
        'U-LEV  1','V-LEV  1','WFACE  1','U-LEV  2','V-LEV  2','WFACE  2',
        'T-LEV  1','T-LEV  2','IPGT','USTAR','ZI','EL','WSTAR','RMM','TEMPK',
        'RHO','QSW','IRH','IPCODE')
    np.testing.assert_array_equal(fields['T-LEV  1'],np.full((2,2),290.))
    assert fields['IPGT'].dtype==np.dtype('<i4')
    with pytest.raises(ValueError,match='no field record'):
        data.fields_at_end(datetime(2018,1,1,2))

    payload=path.read_bytes()
    size=int.from_bytes(payload[-4:],'little',signed=True)
    path.write_bytes(payload[:-(size+8)])
    with pytest.raises(ValueError,match='expected IPCODE, found <EOF>'):
        InitialGuessCalmetData(path)

    misordered=tmp_path/'misordered-calmet.dat'
    _write_igf(misordered,complete=True,misordered=True)
    with pytest.raises(ValueError,match='expected IRH, found IPCODE'):
        InitialGuessCalmetData(misordered)


def test_initial_guess_reader_loads_pre_21_hour_ending_format(tmp_path):
    path=tmp_path/'old-calmet.dat'; _write_old_igf(path)
    data=InitialGuessCalmetData(path)
    assert data.version=='2.0' and data.timezone=='UTC-0500'
    assert data.start==datetime(2018,1,1) and data.end==datetime(2018,1,1,1)
    assert data.duration_hours==1 and data.includes_w_temperature
    assert data.grid==GridConfig(2,2,2.,2.,projection='LCC',datum='WGS-84',
        lat0=30.,lon0=-120.,lat1=30.,lat2=60.,utm_zone=0,
        x_origin_km=10.,y_origin_km=20.)
    begin,u,v=data.at_end(datetime(2018,1,1,1))
    assert begin==datetime(2018,1,1)
    np.testing.assert_allclose(u,[[[1.,2.],[3.,4.]],[[9.,10.],[11.,12.]]])
    np.testing.assert_allclose(v,[[[5.,6.],[7.,8.]],[[13.,14.],[15.,16.]]])


def test_initial_guess_reader_loads_three_title_legacy_format(tmp_path):
    path=tmp_path/'ancient-calmet.dat'; _write_ancient_igf(path)
    data=InitialGuessCalmetData(path)
    assert data.version=='5.2' and data.grid.projection=='LCC'
    assert data.grid.datum=='NWS-84' and data.grid.lon0==-120.
    assert data.start==datetime(2018,1,1)
    begin,u,v=data.at_end(datetime(2018,1,1,1))
    assert begin==datetime(2018,1,1)
    np.testing.assert_allclose(u[:,0,0],[1.,9.])
    np.testing.assert_allclose(v[:,1,1],[8.,16.])


def test_initial_guess_reader_loads_pre_projection_utm_format(tmp_path):
    path=tmp_path/'pre-projection-calmet.dat'; _write_pre_projection_igf(path)
    data=InitialGuessCalmetData(path)
    assert data.version=='5.1'
    assert data.grid==GridConfig(2,2,2.,2.,projection='UTM',datum='NWS-84',
        lat0=0.,lon0=0.,lat1=0.,lat2=0.,utm_zone=50,hemisphere='N',
        x_origin_km=500.,y_origin_km=3000.)
    begin,u,v=data.at_end(datetime(2018,1,1,1))
    assert begin==datetime(2018,1,1)
    np.testing.assert_allclose(u[:,0,0],[1.,9.])
    np.testing.assert_allclose(v[:,1,1],[8.,16.])


def test_initial_guess_interpolation_matches_vertical_and_four_point_rules(tmp_path,monkeypatch):
    path=tmp_path/'calmet.dat'; start=_write_igf(path)
    data=InitialGuessCalmetData(path)
    target=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=-120.,
        lat1=30.,lat2=60.,x_origin_km=11.5,y_origin_km=21.5)
    def unexpected_remap(*args,**kwargs):
        raise AssertionError('identical IGFMET projections must bypass GLOBE remapping')
    monkeypatch.setattr('pycalmet.igf_data.transform_grid',unexpected_remap)
    u,v=interpolate_initial_guess(data,start.replace(hour=1),target,[5.,25.,80.])
    # The target is at the four-source-cell centroid, so horizontal weights
    # are equal. Layer 5 m uses the per-cell logarithmic roughness profile,
    # layer 25 m is halfway between 10 and 40 m, and 80 m persists the top.
    z0=np.array([.1,.3,.2,.4])
    first_u=np.array([1.,3.,2.,4.]); first_v=np.array([5.,7.,6.,8.])
    ratio=(np.log(5.)-np.log(z0))/(np.log(10.)-np.log(z0))
    assert u[0,0,0]==pytest.approx(np.mean(first_u*ratio))
    assert v[0,0,0]==pytest.approx(np.mean(first_v*ratio))
    assert u[1,0,0]==pytest.approx(np.mean((first_u+np.array([9.,11.,10.,12.]))/2.))
    assert v[1,0,0]==pytest.approx(np.mean((first_v+np.array([13.,15.,14.,16.]))/2.))
    assert u[2,0,0]==pytest.approx(10.5)
    assert v[2,0,0]==pytest.approx(14.5)


def test_initial_guess_reader_rejects_missing_wind_level(tmp_path):
    path=tmp_path/'calmet.dat'; _write_igf(path)
    payload=path.read_bytes()
    # Removing the final complete Fortran record leaves level-two V absent.
    size=int.from_bytes(payload[-4:],'little',signed=True)
    path.write_bytes(payload[:-(size+8)])
    with pytest.raises(ValueError,match='incomplete'): InitialGuessCalmetData(path)
