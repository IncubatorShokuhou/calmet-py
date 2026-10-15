from io import BytesIO
import shutil
from datetime import datetime

import numpy as np
import pytest

from pycalmet.legacy_m3d import _level, _surface, _water
from pycalmet.legacy_calwrf import (_legacy_vertical_velocity,_nint,_r4_divide,
                                    _r4_scale,_relative_humidity,_compressed_water,
                                    write_m3d)
from pycalmet.legacy_m2d import M2DReader
from pycalmet.legacy_m3d import M3DReader
from pycalmet._calmet import _meteorology_dataset
from pycalmet.calmet_solver import dataset_timestamp
import netCDF4 as nc


def _surface_line(timestamp, i):
    return (f'{timestamp}{i:3d}{1:3d}{1000.:7.1f}{0.:5.2f}{0:2d}'
            f'{0.:8.1f}{0.:8.1f}{280.:8.1f}{5.:8.2f}{180.:8.1f}'
            f'{5.:8.1f}{279.:8.1f}\n')


def _level_line():
    return (f'{900:4d}{100:6d}{280.:6.1f}{180:4d}{5.:5.1f}{0.:6.2f}'
            f'{50:3d}{5.:5.2f}{-5.:6.3f}\n')


def _formatted_m3d(hours):
    """Return a tiny two-cell formatted M3D with explicit cell timestamps."""
    lines=['3D.DAT         2.1\n','0\n','1 1 1 1 1 0\n',
           'LCC 30 120 30 60 0 0 1 2 1 1\n','0\n',
           f'{hours[0][0]} {len(hours)} 2 1 1\n','1 1 2 1 1 1\n',
           '.900\n','1 1 30 120 0 21\n','2 1 30 121 0 21\n']
    for stamps in hours:
        for i,stamp in enumerate(stamps,1):
            lines.extend((_surface_line(stamp,i),_level_line()))
    return ''.join(lines)


def _formatted_m2d(timestamps):
    lines=['2D.DAT         2.1\n','3\n','producer\n',
           '1 Number of 2D Variables\n','1 TEMP\n','1 1 1 1 1 1\n',
           'LCC 30 120 30 60 0 0 1 1 1 1\n','0\n',
           f'{timestamps[0]} {len(timestamps)} 1 1 1\n','1 1 1 1 1 1\n',
           '.900\n','1 1 30 120 0 21\n']
    for index,stamp in enumerate(timestamps):
        lines.extend((f'{stamp}  TEMP    \n',f'{280.+index:10.3f}\n'))
    return ''.join(lines)


def test_legacy_unit_scaling_retains_default_real_rounding():
    pressure=np.float32(101015.)
    vapor=np.float32(.002105)
    assert _r4_divide(pressure,100.).view(np.uint32)==np.float32(1010.15).view(np.uint32)
    assert _r4_scale(vapor,1000.).view(np.uint32)==np.float32(2.105).view(np.uint32)


def test_nint_rounds_half_away_from_zero_like_fortran():
    values=np.array([-2.5,-2.49,-.5,-.49,0.,.49,.5,2.49,2.5],dtype=np.float32)
    np.testing.assert_array_equal(_nint(values),[-3,-2,-1,0,0,0,1,2,3])


def test_legacy_vertical_velocity_uses_calwrf_threshold():
    assert _legacy_vertical_velocity(np.float32(-.00498)).view(np.uint32)==0
    retained=_legacy_vertical_velocity(np.float32(-.004994))
    assert retained.view(np.uint32)==np.float32(-.004994).view(np.uint32)


def test_fixed_width_surface_parser():
    line = b"2018031800  1  1 1005.3 0.00 0     0.0     0.0   296.4   16.17   125.6     4.8   296.3\n"
    date, i, j, values = _surface(line)
    assert (date, i, j) == (datetime(2018,3,18), 1, 1)
    np.testing.assert_allclose(values, [1005.3, 0, 0, 0, 0, 296.4, 16.17, 125.6, 4.8, 296.3])


def test_fixed_width_level_parser_ignores_compressed_tail():
    values = _level(b"1010    34 296.4 126  4.8  0.00 9216.13-5.000\n")
    np.testing.assert_allclose(values, [1010, 34, 296.4, 126, 4.8, 0, 92, 16.13])
    np.testing.assert_array_equal(_water(b"1010    34 296.4 126  4.8  0.00 9216.13-5.000\n"),
                                  np.zeros(5))
    np.testing.assert_allclose(_water(b"1010    34 296.4 126  4.8  0.00 9216.13 0.100 0.020\n"),
                               [.1,.02])


def test_formatted_m3d_exposes_datetime_across_year_boundary(tmp_path):
    path=tmp_path/'cross-year.m3d'
    path.write_text(_formatted_m3d([
        ('2018123123','2018123123'),
        ('2019010100','2019010100'),
    ]),encoding='ascii')

    header,cells=M3DReader(path).cells()

    assert header.start==datetime(2018,12,31,23)
    assert header.wire_start=='2018123123'
    records=list(cells)
    assert [record.timestamp for record in records]==[
        datetime(2018,12,31,23),datetime(2018,12,31,23),
        datetime(2019,1,1),datetime(2019,1,1),
    ]
    assert records[-1].wire_date=='2019010100'


def test_formatted_m3d_rejects_mixed_cell_times_within_hour(tmp_path):
    path=tmp_path/'mixed-times.m3d'
    path.write_text(_formatted_m3d([
        ('2018123123','2019010100'),
    ]),encoding='ascii')
    _,cells=M3DReader(path).cells()
    with pytest.raises(ValueError,match='inconsistent within an hour'):
        list(cells)


@pytest.mark.parametrize('timestamp',['2019022900','201901010 ','2019AA0100'])
def test_formatted_m3d_rejects_invalid_fixed_width_timestamp(tmp_path,timestamp):
    path=tmp_path/'invalid-time.m3d'
    path.write_text(_formatted_m3d([(timestamp,timestamp)]),encoding='ascii')
    with pytest.raises(ValueError,match='M3D .* timestamp'):
        M3DReader(path).cells()


def test_formatted_m2d_exposes_datetime_across_year_boundary(tmp_path):
    path=tmp_path/'cross-year.m2d'
    path.write_text(_formatted_m2d(['2018123123','2019010100']),encoding='ascii')

    header,records=M2DReader(path).records()

    assert header.start==datetime(2018,12,31,23)
    assert header.wire_start=='2018123123'
    records=list(records)
    assert [record.timestamp for record in records]==[
        datetime(2018,12,31,23),datetime(2019,1,1)]
    assert records[-1].wire_date=='2019010100'


@pytest.mark.parametrize('timestamp',['2019022900','201901010 ','2019AA0100'])
def test_formatted_m2d_rejects_invalid_fixed_width_timestamp(tmp_path,timestamp):
    path=tmp_path/'invalid-time.m2d'
    path.write_text(_formatted_m2d([timestamp]),encoding='ascii')
    with pytest.raises(ValueError,match='M2D .* timestamp'):
        M2DReader(path).records()


@pytest.mark.parametrize('timestamp',[
    '2018-03-18_01:30:00',
    '2018-03-18_01:00:01',
])
def test_legacy_calwrf_writer_rejects_non_hour_canonical_times(tmp_path,timestamp):
    canonical=tmp_path/'subhourly.nc'
    with nc.Dataset(canonical,'w') as dataset:
        dataset.createDimension('time',1)
        dataset.createDimension('date_strlen',19)
        dataset.createVariable('time_text','S1',('time','date_strlen'))[:]=(
            np.frombuffer(timestamp.encode('ascii'),dtype='S1')[None,:])
    output=tmp_path/'out.m3d'; surface=tmp_path/'out.m2d'

    with pytest.raises(ValueError,match='whole-hour'):
        write_m3d(canonical,output,surface)

    assert not output.exists()
    assert not surface.exists()


def test_legacy_writer_round_trips_canonical_calwrf_fields(tmp_path):
    canonical=tmp_path/'calwrf.nc'
    with nc.Dataset(canonical,'w') as dataset:
        for name,size in [('time',1),('level',1),('y',1),('x',1),('date_strlen',19)]:
            dataset.createDimension(name,size)
        dataset.createVariable('time_text','S1',('time','date_strlen'))[:]=np.frombuffer(
            b'2018-03-18_01:00:00',dtype='S1')[None,:]
        dataset.createVariable('sigma','f4',('level',))[:]=[.9]
        for name,value in [('latitude',30.),('longitude',120.),('terrain',12.),('landuse',21.)]:
            dataset.createVariable(name,'f4',('y','x'))[:]=value
        fields2={'surface_pressure':100000.,'precipitation':2.,'snow_cover':0.,
            'shortwave_down':300.,'longwave_down':200.,'temperature_2m':290.,
            'qvapor_2m':.01,'wind_direction_10m':180.,'wind_speed_10m':4.,'sst':289.,
            'skin_temperature':291.,'pbl_height':500.,'sensible_heat_flux':20.,
            'latent_heat_flux':30.,'friction_velocity':.3,
            'convective_precipitation':.5,'grid_precipitation':1.5}
        for name,value in fields2.items():
            dataset.createVariable(name,'f4',('time','y','x'))[:]=value
        fields3={'pressure':900.,'height':1000.,'temperature':280.,'wind_direction':270.,
            'wind_speed':5.,'vertical_velocity':0.,'qvapor':.005,'qcloud':0.,
            'qrain':0.,'qice':0.,'qsnow':0.,'qgraupel':0.}
        for name,value in fields3.items():
            dataset.createVariable(name,'f4',('time','level','y','x'))[:]=value
        dataset.map_proj=1; dataset.cen_lat=30.; dataset.stand_lon=120.
        dataset.truelat1=30.; dataset.truelat2=60.; dataset.wrf_dx_m=1000.
        dataset.legacy_cloud_component_count=5
        dataset.source_grid_nx=4; dataset.source_grid_ny=5; dataset.source_grid_nz=3
        dataset.source_i_start=3; dataset.source_i_end=3
        dataset.source_j_start=4; dataset.source_j_end=4
        dataset.source_k_start=2; dataset.source_k_end=2
        dataset.source_latitude_1_1=29.; dataset.source_longitude_1_1=119.
    output=tmp_path/'out.m3d'; surface=tmp_path/'out.m2d'
    write_m3d(canonical,output,surface)
    comment=output.read_text(encoding='ascii').splitlines()[2]
    assert comment.startswith('Produced by CALWRF v2.0.2')
    level_at=comment.index('Level: ')+len('Level: ')
    assert int(comment[level_at:level_at+6])==131108
    header,cells=M3DReader(output).cells(); cell=next(iter(cells))
    assert (header.nx,header.ny,header.nz,header.start)==(
        1,1,1,datetime(2018,3,18,1))
    assert (header.source_nx,header.source_ny,header.source_nz)==(4,5,3)
    assert (header.i_start,header.j_start,header.k_start)==(3,4,2)
    assert (cell.i,cell.j)==(3,4)
    np.testing.assert_allclose(cell.surface,[1000.,.2,0.,300.,200.,290.,10.,180.,4.,289.])
    np.testing.assert_allclose(cell.levels[0,:6],[900.,1000.,280.,270.,5.,0.])
    header,records=M2DReader(surface).records(); records=list(records)
    assert (header.nx,header.ny,header.nz,header.start)==(
        1,1,1,datetime(2018,3,18,1))
    assert (header.source_nx,header.source_ny,header.source_nz)==(4,5,3)
    assert (header.i_start,header.j_start,header.k_start)==(3,4,2)
    assert header.variables==('PSFC','RAIN','SNOWC','SWDOWN','GLW','T2','Q2','WD10',
        'WS10','SST','TSK','PBLH','HFX','LH','UST','RAINC','RAINNC')
    assert [(record.timestamp,record.name) for record in records]==[
        (datetime(2018,3,18,1),name) for name in header.variables]
    expected=[1000.,.2,0.,300.,200.,290.,10.,180.,4.,289.,291.,500.,20.,30.,.3,.05,.15]
    np.testing.assert_allclose([record.values.item() for record in records],expected,
        rtol=0,atol=5e-4)
    converted=tmp_path/'converted.nc'; M3DReader(output).to_netcdf(converted)
    with nc.Dataset(converted) as dataset:
        assert nc.chartostring(dataset['time_text'][:]).tolist()==['2018-03-18_01:00:00']
        np.testing.assert_allclose(dataset['surface_pressure'][:],100000.)
        np.testing.assert_allclose(dataset['qvapor'][:],.005)
        np.testing.assert_array_equal(dataset['qcloud'][:],0.)
        np.testing.assert_allclose(dataset['latitude'][:],30.,atol=5e-5)
        assert (dataset.source_grid_nx,dataset.source_grid_ny,dataset.source_grid_nz)==(4,5,3)
        assert (dataset.source_i_start,dataset.source_j_start,dataset.source_k_start)==(3,4,2)
    before=set(tmp_path.iterdir())
    with _meteorology_dataset(output,tmp_path) as dataset:
        assert dataset['pressure'].shape==(1,1,1,1)
    assert set(tmp_path.iterdir())==before
    second_nc=tmp_path/'second.nc'; shutil.copyfile(canonical,second_nc)
    with nc.Dataset(second_nc,'r+') as dataset:
        dataset['time_text'][0]=np.frombuffer(b'2018-03-18_02:00:00',dtype='S1')
    second_m3d=tmp_path/'second.m3d'; write_m3d(second_nc,second_m3d)
    with _meteorology_dataset((output,second_m3d),tmp_path) as dataset:
        assert dataset['time_text'].shape==(2,19)
        assert [dataset_timestamp(dataset,index) for index in range(2)]==[
            datetime(2018,3,18,1),datetime(2018,3,18,2)]
        np.testing.assert_allclose(dataset['pressure'][:,0,0,0],[900.,900.])
    duplicate=tmp_path/'duplicate.m3d'; shutil.copyfile(output,duplicate)
    with _meteorology_dataset((output,duplicate),tmp_path) as dataset:
        assert dataset['time_text'].shape==(1,19)
    conflicting_nc=tmp_path/'conflicting.nc'; shutil.copyfile(canonical,conflicting_nc)
    with nc.Dataset(conflicting_nc,'r+') as dataset: dataset['pressure'][0,0,0,0]=850.
    conflicting=tmp_path/'conflicting.m3d'; write_m3d(conflicting_nc,conflicting)
    with pytest.raises(ValueError,match='duplicate meteorology time has inconsistent variable'):
        with _meteorology_dataset((output,conflicting),tmp_path): pass
    with pytest.raises(ValueError,match='not in chronological order'):
        with _meteorology_dataset((second_m3d,output),tmp_path): pass
    inconsistent=tmp_path/'inconsistent.nc'; shutil.copyfile(second_nc,inconsistent)
    with nc.Dataset(inconsistent,'r+') as dataset: dataset['latitude'][0,0]=31.
    with pytest.raises(ValueError,match='inconsistent static field latitude'):
        with _meteorology_dataset((canonical,inconsistent),tmp_path): pass
