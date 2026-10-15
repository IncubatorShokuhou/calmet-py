from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from pyproj import Transformer
from pycalmet._ctgproc import (SnowGridInfo,_write_legacy as write_landuse_legacy,
                               _write_netcdf as write_landuse_netcdf,
                               run as run_ctgproc)
from pycalmet._makegeo import (SnowReference,_category_sum,_fields,_fortran_exp,_fortran_log,_merge_landuse,_snow_albedo_values,_split_categories,_water_sum,
                               _expand_packed_snow,_read_formatted_snow_header,
                               _fill_snow_spatial,_fill_snow_temporal,
                               _iter_formatted_snow_days,
                               _qa_cell_values,
                               _read_formatted_snow_variable,_read_landuse,_read_snow_reference,
                               _read_terrain,
                               _snow_weighted_fields,_write_legacy,_write_qa_grids,run)
from pycalmet._calmet import _geophysical_dataset
from pycalmet.legacy_geo import GeoReader
from pycalmet.config import CtgprocConfig,GridConfig, MakegeoConfig
from pycalmet.projections import geographic_crs,grid_crs
from pycalmet._terrel import _write_legacy as write_terrain_legacy


@pytest.mark.parametrize('projection',('UTM','LCC','TTM','PS','EM','LAZA'))
def test_makegeo_reads_projection_dependent_terrain_header(tmp_path,projection):
    grid=GridConfig(2,2,1.,1.,projection=projection,utm_zone=50,
                    lat0=27.,lon0=117.5,lat1=30.,lat2=60.)
    terrain=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    path=tmp_path/f'{projection}.dat'
    write_terrain_legacy(path,terrain,grid,grid.datum)
    np.testing.assert_array_equal(_read_terrain(path,grid),terrain)
    wrong=GridConfig(2,2,1.,1.,projection=('UTM' if projection!='UTM' else 'LCC'))
    with pytest.raises(ValueError,match='projection or datum mismatch'):
        _read_terrain(path,wrong)


@pytest.mark.parametrize(('unit','scale'),(('KM',1000.),('FT',.3048),('YD',.9144)))
def test_makegeo_terrain_reader_honors_elevation_units_and_south_north_rows(
        tmp_path,unit,scale):
    grid=GridConfig(2,2,1.,1.,projection='UTM',utm_zone=50)
    path=tmp_path/f'terrain-{unit}.dat'
    write_terrain_legacy(path,np.zeros((2,2)),grid,grid.datum)
    lines=path.read_text(encoding='ascii').splitlines()
    units=next(index for index,line in enumerate(lines) if line.startswith('KM  M'))
    lines[units]=f'KM  {unit:<4}'
    lines[units+1]='W_E S_N '
    lines[units+2:]=['1 2','3 4']
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    np.testing.assert_allclose(_read_terrain(path,grid),
                               np.array([[1.,2.],[3.,4.]])*scale)


def test_makegeo_legacy_landuse_checks_grid_geometry_and_projection_parameters(tmp_path):
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=27.,lon0=117.5,
                    lat1=30.,lat2=60.)
    path=tmp_path/'LU.DAT'
    counts=np.ones((1,1,1),dtype=np.int64)
    write_landuse_legacy(path,np.ones((1,1,1),dtype=np.float32),counts,grid,
                         categories=np.array([10],dtype=np.int16))
    wrong_origin=GridConfig(1,1,1.,1.,projection='LCC',lat0=27.,lon0=117.5,
                            lat1=30.,lat2=60.,x_origin_km=1.)
    with pytest.raises(ValueError,match='land-use grid mismatch: geometry'):
        _read_landuse(path,wrong_origin)
    wrong_parallel=GridConfig(1,1,1.,1.,projection='LCC',lat0=27.,lon0=117.5,
                              lat1=31.,lat2=60.)
    with pytest.raises(ValueError,match='projection parameters'):
        _read_landuse(path,wrong_parallel)


def test_makegeo_legacy_landuse_checks_categories_and_record_order(tmp_path):
    grid=GridConfig(1,1,1.,1.)
    path=tmp_path/'LU.DAT'
    write_landuse_legacy(path,np.ones((1,1,1),dtype=np.float32),
                         np.ones((1,1,1),dtype=np.int64),grid,
                         categories=np.array([10],dtype=np.int16))
    with pytest.raises(ValueError,match='categories do not match'):
        _read_landuse(path,grid,(11,))

    lines=path.read_text(encoding='ascii').splitlines()
    record=next(index for index,line in enumerate(lines) if line.startswith('    1    1'))
    lines[record]=f'{2:5d}{1:5d}'+lines[record][10:]
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    with pytest.raises(ValueError,match='cell order'):
        _read_landuse(path,grid,(10,))


def test_makegeo_reads_pre_2_01_landuse_with_control_categories(tmp_path):
    grid=GridConfig(1,1,1.,1.)
    path=tmp_path/'LU.DAT'
    write_landuse_legacy(path,np.ones((1,1,1),dtype=np.float32),
                         np.ones((1,1,1),dtype=np.int64),grid,
                         categories=np.array([10],dtype=np.int16))
    lines=path.read_text(encoding='ascii').splitlines()
    lines[0]=lines[0][:16]+f'{"2.0":<16}'+lines[0][32:]
    units=next(index for index,line in enumerate(lines) if line.startswith('KM'))
    del lines[units+1]
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    categories,fractions=_read_landuse(path,grid,(10,))
    np.testing.assert_array_equal(categories,[10])
    np.testing.assert_allclose(fractions,1.)


def test_makegeo_validates_netcdf_input_grid_metadata(tmp_path):
    grid=GridConfig(2,1,1.,1.)
    terrain=tmp_path/'terrain.nc'
    with Dataset(terrain,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',2)
        dataset.grid_nx=2; dataset.grid_ny=1; dataset.grid_dx_km=2.
        dataset.projection=grid.projection; dataset.datum=grid.datum
        dataset.createVariable('terrain_height','f4',('y','x'))[:]=1.
    with pytest.raises(ValueError,match='terrain NetCDF grid mismatch: grid_dx_km'):
        _read_terrain(terrain,grid)

    landuse=tmp_path/'landuse.nc'
    with Dataset(landuse,'w') as dataset:
        dataset.createDimension('category',1); dataset.createDimension('y',2)
        dataset.createDimension('x',2)
        dataset.createVariable('landuse_category','i2',('category',))[:]=10
        dataset.createVariable('landuse_fraction','f4',('category','y','x'))[:]=1.
    with pytest.raises(ValueError,match='land-use NetCDF array shape mismatch'):
        _read_landuse(landuse,grid)


def test_makegeo_accepts_equivalent_datum_spellings(tmp_path):
    grid=GridConfig(1,1,1.,1.,datum='WGS-84')
    legacy=tmp_path/'terrain.dat'
    write_terrain_legacy(legacy,np.array([[12.]],dtype=np.float32),grid,'WGS84')
    np.testing.assert_array_equal(_read_terrain(legacy,grid),[[12.]])

    netcdf=tmp_path/'terrain.nc'
    with Dataset(netcdf,'w') as dataset:
        dataset.createDimension('y',1); dataset.createDimension('x',1)
        dataset.datum='WGS84'
        dataset.createVariable('terrain_height','f4',('y','x'))[:]=12.
    np.testing.assert_array_equal(_read_terrain(netcdf,grid),[[12.]])


def test_makegeo_header_accepts_prefix_directions_and_wrapped_longitude(tmp_path):
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=27.,lon0=-117.,
                    lat1=30.,lat2=60.)
    path=tmp_path/'terrain.dat'
    write_terrain_legacy(path,np.array([[12.]],dtype=np.float32),grid,grid.datum)
    lines=path.read_text(encoding='ascii').splitlines()
    projection=lines.index('LCC     ')
    lines[projection+1]=''.join(f'{value:<16}' for value in
                                ('N 27','243E','N 30','N 60'))
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    np.testing.assert_array_equal(_read_terrain(path,grid),[[12.]])

def test_single_category_preserves_its_properties():
    fractions=np.zeros((38,1,1)); fractions[0,0,0]=1
    cfg=MakegeoConfig('t','l','o',GridConfig(1,1,1,1))
    dominant,z0,albedo,bowen,soil,qf,lai=_fields(fractions,cfg)
    assert dominant.item()==10
    np.testing.assert_allclose([z0.item(),albedo.item(),bowen.item(),soil.item(),qf.item(),lai.item()],[.5,.18,1.,.2,0.,1.])






def test_makegeo_qa_cell_uses_one_based_indices_and_exposes_fields():
    config=MakegeoConfig('t','l','o',GridConfig(2,2,1.,1.),qa_cell=(2,1))
    terrain=np.array([[10.,20.],[30.,40.]])
    fields=tuple(np.full((2,2),value) for value in range(7))
    assert _qa_cell_values(terrain,fields,config)==(0.,20.,1.,2.,3.,4.,5.,6.)
    assert _qa_cell_values(terrain,fields,
        MakegeoConfig('t','l','o',config.grid,qa_cell=(3,1))) is None

def test_water_needs_more_than_threshold_to_be_dominant():
    fractions=np.zeros((38,1,1)); fractions[7,0,0]=.5; fractions[18,0,0]=.5
    cfg=MakegeoConfig('t','l','o',GridConfig(1,1,1,1))
    assert _fields(fractions,cfg)[0].item()==20

def test_secondary_landuse_only_replaces_missing_primary_cells():
    primary=np.zeros((38,1,2)); primary[0,0,0]=1.
    secondary=np.zeros_like(primary); secondary[1,0,:]=1.
    merged=_merge_landuse(primary,secondary)
    np.testing.assert_array_equal(merged[:,0,0],primary[:,0,0])
    np.testing.assert_array_equal(merged[:,0,1],secondary[:,0,1])




def test_makegeo_reads_sparse_snow_reference_from_legacy_and_netcdf(tmp_path):
    grid=GridConfig(2,1,1.,1.)
    categories=np.array([11,21],dtype=np.int16)
    fractions=np.array([[[1.,0.]],[[0.,1.]]],dtype=np.float32)
    counts=fractions.astype(np.int64)
    snow=SnowGridInfo.create(grid)
    snow.hits={(0,11,101):2,(0,11,102):1,(1,21,202):3}
    outputs=(tmp_path/'LU.DAT',tmp_path/'land.nc')
    write_landuse_legacy(outputs[0],fractions,counts,grid,categories,snow=snow)
    write_landuse_netcdf(outputs[1],fractions,counts,grid,categories,snow=snow)
    for output in outputs:
        reference=_read_snow_reference(output,grid,categories)
        np.testing.assert_array_equal(reference.cell_index,[0,0,1])
        np.testing.assert_array_equal(reference.category_index,[0,0,1])
        np.testing.assert_array_equal(reference.snow_grid_index,[101,102,202])
        np.testing.assert_allclose(reference.fraction,[2/3,1/3,1.],atol=5e-4)


def test_makegeo_rejects_incomplete_snow_reference(tmp_path):
    source=tmp_path/'LU.DAT'
    source.write_text('SNOW Grid Information:\n   1   1  0\n',encoding='ascii')
    with np.testing.assert_raises_regex(ValueError,'every target cell'):
        _read_snow_reference(source,GridConfig(2,1,1.,1.),np.array([11]))


def test_snow_weighted_fields_select_properties_per_reference_sample():
    grid=GridConfig(2,1,1.,1.)
    categories=np.array([1,2])
    fractions=np.array([[[.6,.4]],[[.4,.6]]])
    reference=SnowReference(
        np.array([0,0,1,1]),np.array([0,1,0,1]),np.array([1,1,2,2]),
        np.array([.25,.75,.5,.5]))
    normal={1:(1.,.1,1.,.2,3.,4.,10),2:(.5,.2,2.,.3,4.,5.,20)}
    snowy={1:(.2,.7,10.,.8,30.,40.,10),2:(.1,.8,20.,.9,40.,50.,20)}
    config=MakegeoConfig('t','l','o',grid,input_properties=normal,
        snow_properties=snowy,snow_roughness_method=2,snow_albedo_method=2,
        water_categories=(),output_categories=(10,20),missing_category=1)
    fields=_snow_weighted_fields(
        fractions,categories,reference,np.array([[0.,.3]]),np.array([[2,2]]),
        config,snow_full_nx=2)
    assert fields[0].tolist()==[[10,20]]
    np.testing.assert_allclose(fields[1],[[.5**.75,np.sqrt(.02)]])
    # MSAL=2 retains the legacy ABI bug: positive snow age selects the snow
    # table even where the depth field is zero.
    np.testing.assert_allclose(fields[2],[[.775,.75]])
    np.testing.assert_allclose(fields[3],[[1.75,15.]])
    np.testing.assert_allclose(fields[4],[[.275,.85]])
    np.testing.assert_allclose(fields[5],[[3.75,35.]])
    np.testing.assert_allclose(fields[6],[[4.75,45.]])


def test_computed_snow_roughness_and_aged_albedo_match_fortran_formulas():
    grid=GridConfig(1,1,1.,1.)
    reference=SnowReference(np.array([0]),np.array([0]),np.array([1]),np.array([1.]))
    normal={1:(1.,.2,1.,.2,0.,1.,10),2:(.05,.7,.5,.1,0.,0.,90)}
    snowy={1:(.1,.8,2.,.3,0.,.5,10),2:(.05,.8,.5,.1,0.,0.,90)}
    config=MakegeoConfig('t','l','o',grid,input_properties=normal,
        snow_properties=snowy,snow_age_parameters={1:(10,.5),2:(10,.1)},
        snow_roughness_method=1,snow_height_scale_m=2.,snow_albedo_method=1,
        snow_depth_minimum_m=.1,water_categories=(),output_categories=(10,90),
        missing_category=1)
    fields=_snow_weighted_fields(np.array([[[1.]],[[0.]]]),np.array([1,2]),reference,
        np.array([[.5]]),np.array([[4]]),config,snow_full_nx=1)
    assert fields[1].item()==.75
    # rdcy=(0.8-0.2)*0.5=0.3; aged snow albedo=0.8*(1-0.3*4/10)
    np.testing.assert_allclose(fields[2].item(),.704)


def test_formatted_snow_header_and_run_length_data_are_streamed():
    text=(f'{"SNODAS":<16}{"1.0":<16}{"synthetic":<64}\n'
          '1\ncomment\nLL      \nWGS-84  GREGORIAN \nM   \n'
          '-124.7 52.8 5 1 0.008333 -0.008333\nUTC+0000\nGREGORIAN_YMDHS\n'
          '-124.729584 52.8704185 6935 3351 0.00833333377 -0.00833333377\n'
          '1 5 1 1 2\n')
    block=lambda year,month,day,hour: f'{year:4d}{month:3d}{day:3d}{hour:3d} {0:04d} '
    variable=(block(2020,1,2,0)+block(2020,1,2,0)+
              f'{"SNOWDEP":<8} {-2:7d}{3:7d}{-32768:7d}{10.:8.1f}{4:10d}\n'
              f'{-3:7d}{0:7d}{1:7d}{-32768:7d}\n')
    stream=StringIO(text+variable)
    header=_read_formatted_snow_header(stream)
    assert (header.nx,header.ny,header.raw_nx,header.variables_per_day)==(5,1,6935,2)
    data=_read_formatted_snow_variable(stream,header)
    assert data.timestamp.isoformat()=='2020-01-02T00:00:00'
    assert data.name=='SNOWDEP'
    np.testing.assert_allclose(data.values[0,:4],[-.2,-.2,-.2,-.1])
    assert np.isnan(data.values[0,4])
    assert _read_formatted_snow_variable(stream,header) is None


def test_packed_snow_decoder_rejects_truncated_runs():
    with np.testing.assert_raises_regex(ValueError,'truncated'):
        _expand_packed_snow([-4],0,-32768,1.,4)


def test_snow_missing_fill_uses_original_spatial_snapshot_and_three_days():
    spatial=_fill_snow_spatial(np.array([[1.,np.nan,np.nan,np.nan,5.]]),radius=2)
    np.testing.assert_allclose(spatial,[[1.,1.,3.,5.,5.]])
    window=np.array([[[1.,np.nan]],[[np.nan,4.]],[[3.,8.]]])
    temporal=_fill_snow_temporal(window)
    np.testing.assert_allclose(temporal,[[2.,4.]])


def test_snow_temporal_fill_preserves_all_day_missing_value():
    result=_fill_snow_temporal(np.full((3,1,1),np.nan))
    assert np.isnan(result.item())


def test_formatted_snow_daily_iterator_keeps_three_day_window_and_age(tmp_path):
    header=(f'{"SNODAS":<16}{"1.0":<16}{"synthetic":<64}\n0\nLL      \n'
            'WGS-84  GREGORIAN \nM   \n-124.7 52.8 1 1 0.008333 -0.008333\n'
            'UTC+0000\nGREGORIAN_YMDHS\n'
            '-124.729584 52.8704185 6935 3351 0.00833333377 -0.00833333377\n'
            '1 1 1 1 2\n')
    block=lambda day: f'{2020:4d}{1:3d}{day:3d}{0:3d} {0:04d} '
    def variable(day,name,value):
        return (block(day)+block(day)+
                f'{name:<8} {0:7d}{100:7d}{-32768:7d}{10.:8.1f}{1:10d}\n'
                f'{value:7d}\n')
    body=''
    for day,depth,precipitation in ((1,10,2),(2,-32768,0),(3,30,0),(4,40,2)):
        body+=variable(day,'SNOWDEP',depth)+variable(day,'PRECIPS',precipitation)
    source=tmp_path/'snow.dat'; source.write_text(header+body,encoding='ascii')
    days=list(_iter_formatted_snow_days(source,20200102,20200103))
    assert [item[0].day for item in days]==[2,3]
    assert days[0][1].item()==2.
    assert days[1][1].item()==3.
    # RDSN returns the next day's already-updated NDSNOW array.
    assert days[0][2].item()==2
    assert days[1][2].item()==0
    assert days[0][3].i_begin==1


@pytest.mark.parametrize('land_name',['land.nc','LU.DAT'])
@pytest.mark.parametrize('output_name',['geo.nc','GEO.DAT'])
def test_makegeo_public_run_writes_one_dated_file_per_snow_day(
        tmp_path,land_name,output_name):
    grid=GridConfig(1,1,1.,1.)
    categories=np.array([11],dtype=np.int16)
    fractions=np.ones((1,1,1),dtype=np.float32)
    snow_reference=SnowGridInfo.create(grid); snow_reference.hits={(0,11,1):1}
    land=tmp_path/land_name
    writer=write_landuse_netcdf if land.suffix=='.nc' else write_landuse_legacy
    writer(land,fractions,fractions.astype(np.int64),grid,categories,
           snow=snow_reference)
    header=(f'{"SNODAS":<16}{"1.0":<16}{"synthetic":<64}\n0\nLL      \n'
            'WGS-84  GREGORIAN \nM   \n-124.7 52.8 1 1 0.008333 -0.008333\n'
            'UTC+0000\nGREGORIAN_YMDHS\n'
            '-124.729584 52.8704185 6935 3351 0.00833333377 -0.00833333377\n'
            '1 1 1 1 2\n')
    block=lambda day: f'{2020:4d}{1:3d}{day:3d}{0:3d} {0:04d} '
    def variable(day,name,value):
        return (block(day)+block(day)+
                f'{name:<8} {0:7d}{100:7d}{-32768:7d}{10.:8.1f}{1:10d}\n'
                f'{value:7d}\n')
    body=''.join(variable(day,'SNOWDEP',depth)+variable(day,'PRECIPS',0)
                 for day,depth in ((1,0),(2,5),(3,5)))
    snow_data=tmp_path/'snow.dat'; snow_data.write_text(header+body,encoding='ascii')
    normal={11:(1.,.2,1.,.2,0.,1.,10)}
    snowy={11:(.1,.8,2.,.3,0.,.5,10)}
    result=run(MakegeoConfig('unused',land,tmp_path/output_name,grid,use_terrain=False,
        input_properties=normal,snow_processing=True,snow_data=snow_data,
        snow_start_date=20200102,snow_end_date=20200102,snow_properties=snowy,
        snow_age_parameters={11:(10,.1)},snow_roughness_method=2,
        snow_albedo_method=2,water_categories=(),output_categories=(10,),
        missing_category=11))
    expected=tmp_path/f'{Path(output_name).stem}20200102{Path(output_name).suffix}'
    assert result.output==expected
    assert result.outputs==(result.output,)
    if result.output.suffix=='.nc':
        with Dataset(result.output) as dataset:
            np.testing.assert_allclose(dataset['roughness_length'][0,0],.1)
            np.testing.assert_allclose(dataset['albedo'][0,0],.8)
            np.testing.assert_allclose(dataset['bowen_ratio'][0,0],2.)
    else:
        text=result.output.read_text(encoding='ascii')
        assert 'gridded z0 field' in text and '0.1000' in text

def test_terrain_disabled_does_not_read_terrain_file(tmp_path,monkeypatch):
    fractions=np.zeros((38,1,1)); fractions[0,0,0]=1.
    monkeypatch.setattr('pycalmet._makegeo._read_landuse',
                        lambda path,grid,expected: (np.array([11]),fractions[:1]))
    result=run(MakegeoConfig(tmp_path/'absent.dat','land.dat',tmp_path/'geo.nc',
               GridConfig(1,1,1.,1.),use_terrain=False))
    assert result.terrain_minimum_m==result.terrain_maximum_m==0.


def test_makegeo_writes_surfer_landuse_and_terrain_qa_grids(tmp_path):
    grid=GridConfig(2,2,1.,2.,x_origin_km=10.,y_origin_km=20.)
    config=MakegeoConfig('t','l','o',grid,landuse_plot_output=tmp_path/'lu.grd',
                         terrain_plot_output=tmp_path/'terrain.grd',
                         color_map_output=tmp_path/'luse.clr',image_mode=0)
    dominant=np.array([[10,-20],[55,90]])
    terrain=np.array([[1.,2.],[3.,4.]])
    _write_qa_grids(terrain,dominant,grid,config)
    land_lines=(tmp_path/'lu.grd').read_text(encoding='ascii').splitlines()
    terrain_lines=(tmp_path/'terrain.grd').read_text(encoding='ascii').splitlines()
    assert land_lines[:2]==['DSAA','           2           2']
    assert [float(value) for value in land_lines[2].split()]==[10.5,11.5]
    assert [float(value) for value in land_lines[3].split()]==[21.,23.]
    assert [int(value) for value in ' '.join(land_lines[5:]).split()]==[10,20,55,90]
    assert terrain_lines[0]=='DSAA'
    np.testing.assert_allclose([float(value) for value in terrain_lines[4].split()],[1.,4.])
    assert (tmp_path/'luse.clr').read_text(encoding='ascii').startswith('ColorMap 1 1\n')


def test_legacy_geo_round_trip_and_transparent_calmet_adapter(tmp_path):
    grid=GridConfig(2,2,1.,1.,projection='LCC',lat0=27.,lon0=117.5,
                    lat1=30.,lat2=60.)
    config=MakegeoConfig('terrain','land','geo',grid,output_categories=(10,20),
                         output_water_min=50,output_water_max=55)
    terrain=np.array([[1.25,2.5],[3.75,4.]])
    fields=(np.array([[10,20],[20,10]],dtype=np.int16),
            np.array([[.1,.2],[.3,.4]]),np.array([[.11,.12],[.13,.14]]),
            np.array([[1.,2.],[3.,4.]]),np.array([[.2,.3],[.4,.5]]),
            np.array([[0.,1.],[2.,3.]]),np.array([[1.,2.],[3.,4.]]))
    legacy=tmp_path/'GEO.DAT'; _write_legacy(legacy,terrain,fields,grid,config)
    assert legacy.read_text(encoding='ascii').splitlines()[3]==f'{config.title:<80}'
    parsed=GeoReader(legacy).read()
    np.testing.assert_allclose(parsed.terrain,terrain)
    for actual,expected in zip(parsed.fields,fields): np.testing.assert_allclose(actual,expected)
    converted=tmp_path/'geo.nc'; GeoReader(legacy).to_netcdf(converted)
    with Dataset(converted) as dataset:
        np.testing.assert_allclose(dataset['roughness_length'][:],fields[1])
        assert (dataset.water_category_min,dataset.water_category_max)==(50,55)
        assert 'utm_zone' not in dataset.ncattrs()
    before=set(tmp_path.iterdir())
    with _geophysical_dataset(legacy,tmp_path) as dataset:
        np.testing.assert_array_equal(dataset['dominant_landuse'][:],fields[0])
    assert set(tmp_path.iterdir())==before


def test_legacy_geo_uses_fortran_format_reversion(tmp_path):
    grid=GridConfig(102,1,1.,1.,lat0=27.,lon0=117.5,lat1=30.,lat2=60.)
    config=MakegeoConfig('terrain','land','geo',grid,output_categories=(10,),
                         output_water_min=50,output_water_max=55)
    terrain=np.arange(102,dtype=np.float32)[None,:]
    fields=(np.full((1,102),10,dtype=np.int16),)+(np.ones((1,102)),)*6
    path=tmp_path/'GEO.DAT'
    _write_legacy(path,terrain,fields,grid,config)
    lines=path.read_text(encoding='ascii').splitlines()
    declaration=next(index for index,line in enumerate(lines) if 'NLU, IWAT1, IWAT2' in line)
    category=lines[declaration+1]
    assert 'new LU categories' not in category
    terrain_header=next(index for index,line in enumerate(lines)
                        if 'TERRAIN heights' in line)
    land_lines=lines[lines.index(category)+1:terrain_header]
    assert [len(line) for line in land_lines]==[80]*10+[15]
    assert land_lines[0].endswith(',') and not land_lines[-1].endswith(',')
    assert len(lines[terrain_header])==81
    terrain_lines=lines[terrain_header+1:terrain_header+3]
    assert [len(line) for line in terrain_lines]==[800,15]
    assert terrain_lines[0].endswith(',') and not terrain_lines[-1].endswith(',')


@pytest.mark.parametrize('projection',('UTM','LCC','TTM','PS','EM','LAZA'))
def test_makegeo_legacy_header_round_trips_every_documented_projection(tmp_path,projection):
    grid=GridConfig(1,1,1.,1.,projection=projection,lat0=-27.,lon0=-117.5,
                    lat1=-30.,lat2=-60.,false_easting_km=12.5,
                    false_northing_km=-8.25,utm_zone=12,hemisphere='S')
    config=MakegeoConfig('terrain','land','geo',grid,output_categories=(10,),
                         output_water_min=50,output_water_max=55)
    fields=(np.array([[10]],dtype=np.int16),)+(np.ones((1,1)),)*6
    path=tmp_path/f'{projection}.dat'
    _write_legacy(path,np.array([[123.]]),fields,grid,config)
    lines=path.read_text(encoding='ascii').splitlines()
    assert len(lines[0])==96
    if projection in {'LCC','LAZA','TTM'}:
        assert '   12.5000000      -8.25000000    ' in lines
    assert f'{grid.datum:<8}02-21-2003  ' in lines
    parsed=GeoReader(path).read()
    assert parsed.projection==projection
    assert parsed.datum==grid.datum
    if projection=='UTM':
        assert (parsed.utm_zone,parsed.hemisphere)==(12,'S')
    else:
        assert (parsed.latitude_origin,parsed.longitude_origin)==(-27.,-117.5)
    if projection in {'LCC','PS'}: assert parsed.standard_parallel_1==-30.
    if projection=='LCC': assert parsed.standard_parallel_2==-60.
    expected_offsets=((12.5,-8.25) if projection in {'LCC','LAZA','TTM'} else (0.,0.))
    assert (parsed.false_easting_km,parsed.false_northing_km)==expected_offsets


def test_legacy_geo_expands_default_and_user_category_tables(tmp_path):
    source=tmp_path/'table-geo.dat'
    source.write_text(
        'GEO.DAT         2.1             synthetic\n   0\nLCC\n'
        '27N             117.5E          30N             60N\n'
        '0 0\nWGS-84\n1 1 0 0 1 1\nKM  M\n'
        '1\n1 55 55\n10\n10\n2.0\n3.0\n'
        '1\n10 0.7\n0\n0\n0\n0\n0\n',encoding='ascii')
    data=GeoReader(source).read()
    assert data.terrain.item()==6.
    expected=(10,.7,.18,1.5,.25,0.,.2)
    np.testing.assert_allclose([field.item() for field in data.fields],expected)


def test_makegeo_removes_constant_terrain_qa_grid(tmp_path):
    target=tmp_path/'terrain.grd'; target.write_text('stale',encoding='ascii')
    config=MakegeoConfig('t','l','o',GridConfig(1,1,1.,1.),
                         terrain_plot_output=target)
    _write_qa_grids(np.zeros((1,1)),np.ones((1,1)),config.grid,config)
    assert not target.exists()

def test_makegeo_accepts_generic_categories_and_control_properties(tmp_path):
    land=tmp_path/'generic.nc'
    with Dataset(land,'w') as dataset:
        dataset.createDimension('category',2); dataset.createDimension('y',1); dataset.createDimension('x',2)
        dataset.createVariable('landuse_category','i2',('category',))[:]=[7,42]
        dataset.createVariable('landuse_fraction','f4',('category','y','x'))[:]=[[[1.,.4]],[[0.,.6]]]
    properties={7:(.5,.2,1.,.2,3.,2.,10),42:(.001,.1,0.,1.,0.,0.,55)}
    output=tmp_path/'geo.nc'
    run(MakegeoConfig('unused',land,output,GridConfig(2,1,1.,1.),use_terrain=False,
        input_properties=properties,water_categories=(42,),output_categories=(10,55),
        output_water_min=55,output_water_max=55,missing_category=7))
    with Dataset(output) as dataset:
        np.testing.assert_array_equal(dataset['dominant_landuse'][:],[[10,55]])
        np.testing.assert_allclose(dataset['anthropogenic_heat_flux'][:],[[3.,1.2]])
        assert (dataset.water_category_min,dataset.water_category_max)==(55,55)
        assert dataset.landuse_categories=='10,55'

def test_makegeo_category_split_percentages_are_applied_before_weighting():
    fractions=np.array([[[1.]],[[0.]],[[0.]]])
    properties={1:(1.,.1,1.,.1,0.,1.,10),2:(.1,.2,2.,.2,0.,2.,20),3:(.01,.3,3.,.3,0.,3.,30)}
    config=MakegeoConfig('t','l','o',GridConfig(1,1,1.,1.),input_properties=properties,
        water_categories=(),output_categories=(10,20,30),missing_category=1,
        category_splits=((1,2,25.),(1,3,75.)))
    dominant,z0,albedo,bowen,soil,qf,lai=_fields(fractions,config,np.array([1,2,3]))
    assert dominant.item()==30
    np.testing.assert_allclose([z0.item(),albedo.item(),bowen.item(),soil.item(),lai.item()],
        [10.**-1.75,.275,2.75,.275,2.75],rtol=2.e-6,atol=1.e-6)



def test_generic_ctgproc_legacy_output_feeds_makegeo(tmp_path):
    grid=GridConfig(1,1,1.,1.,projection='LCC',false_easting_km=12.,
                    false_northing_km=34.,datum='ESR-S')
    lon,lat=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True).transform(.5,.5)
    source=tmp_path/'generic.lu'
    source.write_text('GENERIC.LANDUSE 1.0\n0\nLL\nWGS-84\nDEG\n2\n7 42\n'
                      f'7 {lon:.12f} {lat:.12f}\n42 {lon:.12f} {lat:.12f}\n',encoding='ascii')
    land=tmp_path/'LU.DAT'
    run_ctgproc(CtgprocConfig([source],land,grid,input_type='GEN'))
    land_text=land.read_text(encoding='ascii')
    offsets='   12.0000000       34.0000000    '
    assert offsets in land_text and 'ESR-S' in land_text
    output=tmp_path/'GEO.DAT'
    properties={7:(.5,.2,1.,.2,0.,2.,10),42:(.001,.1,0.,1.,0.,0.,55)}
    result=run(MakegeoConfig('unused',land,output,grid,use_terrain=False,
        input_properties=properties,water_categories=(42,),output_categories=(10,55),
        output_water_min=55,output_water_max=55,missing_category=7))
    assert result.populated_cells==1
    text=output.read_text(encoding='ascii')
    assert '    2  55  55 - NLU, IWAT1, IWAT2' in text
    assert offsets in text and 'ESR-S' in text
