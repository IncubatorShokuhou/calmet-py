from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

from pycalmet.config import GridConfig
from pycalmet.calmet_solver import build_mm4_step_fields,target_latitude_longitude
from pycalmet.config import CalmetConfig
from pycalmet._calmet import _meteorology_input
from pycalmet._calmet import run as run_calmet
from pycalmet.mm4_data import MM4Data
from pycalmet.legacy._compat._calmet import _meteorology_input as legacy_meteorology_input
from pycalmet.legacy._compat.mm4_data import MM4Data as LegacyMM4Data


def _mm4(path,record_hours=(4,),year=18):
    lines=['synthetic MM4'.ljust(36),f'{year:02d}{1:02d}{2:02d}{3:02d}{len(record_hours):4d}{2:4d}{2:4d}{1:4d}{100.:6.1f}',
           f'{4:4d}{5:4d}{2:4d}{2:4d}',f'{.5:6.4f}']
    for j in (5,6):
        for i,terrain in ((4,100),(5,200)):
            lines.append(f'{i:3d}{j:3d}{30.+j-5:7.3f}{120.+i:8.3f}{terrain:5d}{20:3d}')
    for hour in record_hours:
        for j in (5,6):
            for i,terrain in ((4,100),(5,200)):
                lines.append(f'{year:02d}{1:02d}{2:02d}{hour:02d}{i:3d}{j:3d}{1000.:7.1f}{.2:5.2f}{0:2d}')
                for level in range(9):
                    encoded=200 if level==0 else 201
                    lines.append(f'{999.-level*50:5.1f}{terrain+level*100:6.0f}{encoded:4d}{10:2d}'
                                 f'{180.:4.0f}{10.:2.0f}')
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')


def test_mm4_two_digit_year_uses_datetime_standard_window(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path,year=68)
    data=MM4Data(path)
    assert data.header.start==datetime(2068,1,2,3)
    assert data.records[0].timestamp==datetime(2068,1,2,4)




def test_mm4_reader_rejects_bad_grid_sequence(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    text=path.read_text(encoding='ascii').replace(f'{5:3d}{5:3d}{30.:7.3f}',
                                                  f'{6:3d}{5:3d}{30.:7.3f}',1)
    path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='static grid index'):
        MM4Data(path)


def test_mm4_surface_scalars_and_850_humidity_use_physical_coordinates(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path); record=data.records[0]
    height=record.height_agl_m.copy(); height[0]=500.; height[1]=0.
    pressure=record.pressure_mb.copy()
    pressure[0]=800.; pressure[1]=900.; pressure[2]=860.; pressure[3]=840.
    humidity=record.relative_humidity.copy()
    humidity[0]=11.; humidity[1]=20.; humidity[2]=40.; humidity[3]=60.
    temperature=record.temperature_k.copy(); temperature[0]=300.; temperature[1]=250.
    record=replace(record,height_agl_m=height,pressure_mb=pressure,
                   relative_humidity=humidity,temperature_k=temperature)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    fields=data.interpolate_record(record,grid,[50.],cloud_method=3)
    np.testing.assert_allclose(fields.surface_pressure_mb,900.)
    np.testing.assert_allclose(fields.surface_density,.3484321*900./250.)
    np.testing.assert_array_equal(fields.surface_relative_humidity,20.)
    from pycalmet.calmet_numerics import cloud_fraction
    np.testing.assert_allclose(fields.cloud_fraction,cloud_fraction(np.array([[50.]])))

    legacy=LegacyMM4Data(path).interpolate_record(record,grid,[50.],cloud_method=3)
    np.testing.assert_allclose(legacy.surface_pressure_mb,800.)
    np.testing.assert_array_equal(legacy.surface_relative_humidity,11)
    np.testing.assert_allclose(legacy.cloud_fraction,cloud_fraction(np.array([[40.]])))


def test_mm4_profile_removes_duplicates_and_below_ground_levels(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path); record=data.records[0]
    height=record.height_agl_m.copy()
    height[:,0,0]=[-50.,0.,100.,100.,200.,300.,400.,500.,600.]
    marker=np.broadcast_to(np.arange(9.)[:,None,None],height.shape).copy()
    profiles=data._sorted_above_ground_profiles(
        height=height,temperature=marker,humidity=marker,pressure=marker,
        direction=marker,speed=marker)
    np.testing.assert_array_equal(profiles['height'][:,0,0],
                                  [0.,100.,200.,300.,400.,500.,600.,600.,600.])
    np.testing.assert_array_equal(profiles['temperature'][:7,0,0],
                                  [1.,2.,4.,5.,6.,7.,8.])

    legacy=LegacyMM4Data._sorted_above_ground_profiles(
        height=height,temperature=marker,humidity=marker,pressure=marker,
        direction=marker,speed=marker)
    np.testing.assert_array_equal(legacy['height'][:,0,0],
                                  [0.,100.,200.,300.,400.,500.,500.,500.,500.])


def test_mm4_rejects_calmet_grid_outside_curvilinear_source_domain(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(data.header.longitude,data.header.latitude,grid,'WGS-84',units='km')
    grid.x_origin_km=float(np.max(x)+10.); grid.y_origin_km=float(np.max(y)+10.)
    with pytest.raises(ValueError,match='grid corner 1 is outside prognostic domain'):
        data.interpolate_record(data.records[0],grid,[50.])


def test_native_mm4_adapter_supplies_subhour_step_fields(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,timestep_seconds=900,z_faces_m=(0.,100.,200.))
    start=datetime(2018,1,2,3); end=datetime(2018,1,2,4)
    with _meteorology_input(config,(path,),tmp_path,start,end) as (dataset,provider):
        assert dataset['time_text'].shape==(5,19)
        first=provider(datetime(2018,1,2,3,15))
        last=provider(end)
    assert first is last
    assert first.u.shape==(2,1,1)


def test_native_mm4_adapter_applies_fp64_projected_sigmap_guard(tmp_path,monkeypatch):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    data=MM4Data(path)
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,precipitation_station_count=-1,
        precipitation_radius_km=1.,z_faces_m=(0.,100.,200.))
    captured={}
    class Result:
        thermal_state=None
    def fake_build(*args,**kwargs):
        captured.update(kwargs)
        return Result()
    monkeypatch.setattr('pycalmet._calmet.build_mm4_step_fields',fake_build)
    with _meteorology_input(config,(path,),tmp_path,datetime(2018,1,2,3),
                            datetime(2018,1,2,4)) as (_,provider):
        provider(datetime(2018,1,2,4))
    source_x,source_y=project_geographic(data.header.longitude,data.header.latitude,
                                         grid,'WGS-84',units='km')
    source_x=np.asarray(source_x,dtype=np.float64)
    source_y=np.asarray(source_y,dtype=np.float64)
    horizontal=np.hypot(np.diff(source_x,axis=1),np.diff(source_y,axis=1))
    vertical=np.hypot(np.diff(source_x,axis=0),np.diff(source_y,axis=0))
    expected=np.hypot(np.median(horizontal),np.median(vertical))*1.05
    assert captured['precipitation_radius_km']==expected


def test_native_mm4_adapter_indexes_all_files_and_fills_earlier_gap(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    _mm4(first_path,(4,6)); _mm4(second_path,(5,6))
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010206',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    start=datetime(2018,1,2,3); end=datetime(2018,1,2,6)
    with _meteorology_input(config,(first_path,second_path),tmp_path,start,end) as (_,provider):
        assert provider(datetime(2018,1,2,5)).u.shape==(2,1,1)
        # Both files contain an identical 06:00 boundary record.  The modern
        # index deduplicates that ordinary overlap.
        assert provider(end).u.shape==(2,1,1)

    # The exact compatibility API retains RDMM4's forward file scan and
    # therefore cannot repair the first file's 05:00 gap.
    with pytest.raises(ValueError,match=r'record 2018-01-02 06:00:00 is beyond requested time 2018-01-02 05:00:00'):
        with legacy_meteorology_input(config,(first_path,second_path),tmp_path,start,end):
            pass


def test_native_mm4_adapter_rejects_conflicting_overlap(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    _mm4(first_path,(4,)); _mm4(second_path,(4,))
    second_path.write_text(
        second_path.read_text(encoding='ascii').replace(' 1000.0',' 1001.0',1),
        encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match=r'conflicting native MM4 records for 2018-01-02 04:00:00'):
        with _meteorology_input(config,(first_path,second_path),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,4)):
            pass


def test_native_mm4_adapter_reports_all_missing_requested_times(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    _mm4(first_path,(4,)); _mm4(second_path,(6,))
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010206',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(
            ValueError,
            match=r'do not cover requested time\(s\): 2018-01-02 05:00:00'):
        with _meteorology_input(config,(first_path,second_path),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,6)):
            pass


def test_native_mm4_adapter_rejects_header_after_its_earliest_record(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    text=path.read_text(encoding='ascii').replace('18010203','18010205',1)
    path.write_text(text,encoding='ascii')
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match='header.*begins at .* after its earliest record'):
        with _meteorology_input(config,(path,),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,4)):
            pass


def test_native_mm4_adapter_reports_its_hourly_file_cadence(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.)
    config=CalmetConfig(path,'geo','out','2018010203','2018010204',grid,
        iprog=4,iwfcod=1,prognostic_timestep_seconds=7200,
        z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match='native MM4.DAT records are hourly'):
        with _meteorology_input(config,(path,),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,4)):
            pass


def test_native_mm4_adapter_validates_every_static_header(tmp_path):
    first_path=tmp_path/'MM4-1.DAT'; second_path=tmp_path/'MM4-2.DAT'
    _mm4(first_path,(4,)); _mm4(second_path,(5,))
    text=second_path.read_text(encoding='ascii')
    second_path.write_text(text.replace(f'{100:5d}{20:3d}',f'{100:5d}{16:3d}',1),encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(first_path,'geo','out','2018010203','2018010205',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with pytest.raises(ValueError,match=r'static header.*land-use categories'):
        with _meteorology_input(config,(first_path,second_path),tmp_path,
                                datetime(2018,1,2,3),datetime(2018,1,2,5)):
            pass

    # Legacy RDMM4 skips the second static header entirely.
    with legacy_meteorology_input(config,(first_path,second_path),tmp_path,
                                  datetime(2018,1,2,3),datetime(2018,1,2,5)) as (_,provider):
        assert provider(datetime(2018,1,2,5)).u.shape==(2,1,1)


def test_native_mm4_adapter_advances_thermal_state_when_later_hour_requested_first(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path,(4,5))
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.,lat1=0.,lat2=0.)
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    config=CalmetConfig(path,'geo','out','2018010203','2018010205',grid,
        iprog=4,iwfcod=1,z_faces_m=(0.,100.,200.))
    with _meteorology_input(config,(path,),tmp_path,datetime(2018,1,2,3),
                            datetime(2018,1,2,5)) as (_,provider):
        later=provider(datetime(2018,1,2,5))
        earlier=provider(datetime(2018,1,2,4))
    assert later is not earlier
    assert later.u.shape==earlier.u.shape==(2,1,1)


def test_public_calmet_idiopt4_5_omit_surface_and_upper_files(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=124.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),
                           grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    geo=tmp_path/'geo.nc'; shape=(1,1)
    with Dataset(geo,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        values={'terrain_height':0.,'dominant_landuse':20,'roughness_length':.1,
                'albedo':.2,'bowen_ratio':1.,'soil_heat_flux_parameter':.15,
                'anthropogenic_heat_flux':0.}
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f8'
            dataset.createVariable(name,dtype,('y','x'))[:]=np.full(shape,value)
    diag=tmp_path/'DIAG.DAT'
    surface_line=' '*15+f'{"SFC":<4} '+''.join(f'{value:5.1f}' for value in (1.,4.,0.))
    upper_line=' '*13+f'{"UPR":<4} '+''.join(
        f'{value:5.1f}' for value in (1.,8.,0.,9.,0.))
    upper_last=' '*13+'LAST '+''.join(f'{0.:5.1f}' for _ in range(5))
    diag.write_text('\n'.join((surface_line,' '*15+'LAST',upper_line,upper_last))+'\n',
                    encoding='ascii')
    station_x=grid.x_origin_km+.5; station_y=grid.y_origin_km+.5
    output=tmp_path/'calmet-diag.nc'
    config=CalmetConfig(path,geo,output,'2018010203','2018010204',grid,
        iprog=4,iwfcod=1,noobs=0,run_type=0,itprog=2,irhprog=1,
        surface_data=None,upper_air_files=(),diagnostic_data=diag,
        diagnostic_surface_wind=1,diagnostic_upper_wind=1,
        surface_control_stations=(('SFC',1,station_x,station_y,0.,10.),),
        upper_control_stations=(('UPR',2,station_x,station_y,0.),),
        z_faces_m=(0.,100.,200.),smoothing_passes=(0,0),
        stations_per_level=(1,1),wind_extrapolation_factors=(1.,1.),
        ifradj=0,islope=0,ikine=0,iobr=0,iavet=0,iavezi=0,
        lapse_height_m=100.)
    assert run_calmet(config)==output
    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()


def test_full_calmet_run_uses_diag_winds_and_observed_thermodynamics(tmp_path):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=124.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),
                           grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    geo=tmp_path/'geo.nc'
    with Dataset(geo,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        values={'terrain_height':0.,'dominant_landuse':20,'roughness_length':.1,
                'albedo':.2,'bowen_ratio':1.,'soil_heat_flux_parameter':.15,
                'anthropogenic_heat_flux':0.}
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f4'
            dataset.createVariable(name,dtype,('y','x'))[:]=[[value]]
    surface=tmp_path/'SURF.DAT'
    surface.write_text("""SURF.DAT        2.1             full-run test
1
comment
LL
WGS-84  20200101
DEG
UTC-0000
2018 2 3 0 2018 2 4 0 1
1 SFC 30N 124.5E 10
2018 2 3 0 2018 2 4 0 1 270 70 5 330 60 1000 0
""",encoding='ascii')
    upper=tmp_path/'UP.DAT'
    upper.write_text("""UP.DAT          2.1             full-run test
1
comment
LL
WGS-84  20200101
DEG
UTC+0000
2018 2 3 0 2018 2 4 0 100 3 2
T T T T
2 UPR 30N 124.5E 0
2 2018 1 2 3 2
1000 0 20 270 1 900 100 10 270 1
2 2018 1 2 4 2
1000 0 20 270 1 900 100 10 270 1
""",encoding='ascii')
    diag=tmp_path/'DIAG.DAT'
    surface_line=' '*15+f'{"SFC":<4} '+''.join(
        f'{value:5.1f}' for value in (1.,4.,0.))
    upper_line=' '*13+f'{"UPR":<4} '+''.join(
        f'{value:5.1f}' for value in (1.,8.,0.,8.,0.))
    upper_last=' '*13+'LAST '+''.join(f'{0.:5.1f}' for _ in range(5))
    diag.write_text('\n'.join((surface_line,' '*15+'LAST',upper_line,upper_last))+'\n',
                    encoding='ascii')
    output=tmp_path/'calmet-full-diag.nc'
    config=CalmetConfig(path,geo,output,'2018010203','2018010204',grid,
        iprog=4,iwfcod=1,noobs=0,run_type=1,itprog=0,irhprog=0,
        surface_data=surface,upper_air_files=(upper,),diagnostic_data=diag,
        diagnostic_surface_wind=1,diagnostic_upper_wind=1,
        z_faces_m=(0.,20.,40.),smoothing_passes=(0,0),stations_per_level=(2,2),
        wind_extrapolation=-1,wind_extrapolation_factors=(1.,1.),
        ifradj=0,islope=0,ikine=0,iobr=0,iavet=0,iavezi=0,
        lapse_height_m=20.,mixing_lapse_depth_m=20.)

    assert run_calmet(config)==output
    with Dataset(output) as dataset:
        np.testing.assert_allclose(dataset['surface_temperature'][:],330.)
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()


def test_public_calmet_runs_native_mm4_to_netcdf(tmp_path,monkeypatch):
    path=tmp_path/'MM4.DAT'; _mm4(path)
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=30.,lon0=124.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    from pycalmet.projections import project_geographic
    x,y=project_geographic(np.array([124.,125.]),np.array([30.,30.]),grid,'WGS-84',units='km')
    grid.x_origin_km=float(x.mean()-.5); grid.y_origin_km=float(y.mean()-.5)
    geo=tmp_path/'geo.nc'; shape=(1,1)
    with Dataset(geo,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        values={'terrain_height':0.,'dominant_landuse':20,'roughness_length':.1,
                'leaf_area_index':1.,
                'albedo':.2,'bowen_ratio':1.,'soil_heat_flux_parameter':.15,
                'anthropogenic_heat_flux':0.}
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f8'
            dataset.createVariable(name,dtype,('y','x'))[:]=np.full(shape,value)
    output=tmp_path/'calmet.nc'
    config=CalmetConfig(path,geo,output,'2018010203','2018010204',grid,
        iprog=4,iwfcod=1,noobs=2,itprog=2,irhprog=1,wind_extrapolation=-1,
        z_faces_m=(0.,100.,200.),
        smoothing_passes=(0,0),ifradj=0,islope=0,ikine=0,iobr=0,
        iavet=0,iavezi=0,run_type=0,lapse_height_m=100.)
    assert run_calmet(config)==output
    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()

    # IDIOPT4/5 have no consumer without observations, so they neither need
    # nor open a DIAG.DAT file in the modern path.
    monkeypatch.setattr('pycalmet._calmet.DiagnosticData',
        lambda *args,**kwargs: pytest.fail('inactive IDIOPT opened DIAG.DAT'))
    inert_output=tmp_path/'calmet-inert-diag.nc'
    config.output=inert_output; config.diagnostic_data=None
    config.diagnostic_surface_wind=1; config.diagnostic_upper_wind=1
    assert run_calmet(config)==inert_output
    with Dataset(inert_output) as dataset:
        assert dataset.dimensions['time'].size==1
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()
    config.diagnostic_surface_wind=0; config.diagnostic_upper_wind=0
    no_output=tmp_path/'disabled.dat'
    config.output=no_output; config.save_output=False
    assert run_calmet(config)==no_output
    assert not no_output.exists()
    setup_output=tmp_path/'setup.nc'
    config.output=setup_output; config.save_output=True; config.test_mode=1
    assert run_calmet(config)==setup_output
    with Dataset(setup_output) as dataset:
        assert dataset.dimensions['time'].size==0

    pacout=tmp_path/'pacout.dat'
    config.output=pacout; config.test_mode=2; config.legacy_output_format=2
    config.run_type=1; config.z_faces_m=(0.,20.,3500.); config.smoothing_passes=(0,0)
    assert run_calmet(config)==pacout
    from pycalmet.legacy_io import FortranSequentialReader
    with FortranSequentialReader(pacout) as reader:
        header=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(header[:40],dtype='<i4'),
                                      [2018,2,1,0,0,1,1,0,2,4])
        assert reader.read_record()==b''
        for _ in range(3): assert len(reader.read_record())==4
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),
                                      [2018,2,3])
        for _ in range(10): assert len(reader.read_record())==4
        assert len(reader.read_record())==4
        assert reader.read_record() is None

    # Native adapters expose a mapping rather than a NetCDF ``variables``
    # attribute.  Traditional CALMET output still needs to create its auxiliary
    # file and must treat the absent qcloud field as unavailable.
    calmet_dat=tmp_path/'calmet.dat'
    config.output=calmet_dat; config.legacy_output_format=1
    assert run_calmet(config)==calmet_dat
    assert calmet_dat.is_file()
    assert (tmp_path/'calmet.dat.aux').is_file()
    config.legacy_output_format=2

    latitude,longitude=target_latitude_longitude(grid)
    surf=tmp_path/'surf.dat'
    surf.write_text(f"""SURF.DAT        2.1             synthetic station data
0
LL
WGS-84  20200101
DEG
UTC-0000
2018 2 3 0 2018 2 4 0 1
101 STA {latitude.item():.8f} {longitude.item():.8f} 10
2018 2 3 0 2018 2 4 0  1 270 9999 9999 282 61 1000 7
""",encoding='ascii')
    observed=tmp_path/'pacout-observed.dat'
    config.output=observed; config.noobs=1; config.surface_data=surf
    config.stations_per_level=(1,1); config.wind_extrapolation=2
    config.wind_extrapolation_factors=(1.,1.)
    assert run_calmet(config)==observed
    with FortranSequentialReader(observed) as reader:
        header=reader.read_record()
        assert np.frombuffer(header,dtype='<i4',count=4)[3]==1
        assert len(reader.read_record())==8
        for _ in range(3): reader.read_record()
        reader.read_record()
        for _ in range(10): reader.read_record()
        tail=reader.read_record()
        assert len(tail)==20
        assert np.frombuffer(tail,dtype='<f4',count=1,offset=4).item()==282.
        assert np.frombuffer(tail,dtype='<i4',count=2,offset=12).tolist()==[61,7]
