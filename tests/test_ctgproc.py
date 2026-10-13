from pathlib import Path

import numpy as np
import pytest
import tifffile
from netCDF4 import Dataset
from pyproj import CRS,Transformer

from pycalmet._ctgproc import (CORINE_TO_USGS,GLAZ_SPECS,GLAZ_TO_USGS,NZ_TO_USGS,USGS38,
                               QA_FILENAMES,SnowGridInfo,
                               CoastPolygon,_coastline_types,_corine_indices,_geotiff_geometry,_polygon_membership,_read_coastline_bln,
                               _fractions_from_counts,_grid_centers_1d,_grid_indices,_low_hit_threshold,_merge_global_counts,_mesh_coordinates,_mesh_offsets,_select_glaz_markers,_snow_grid_indices,_source_centers,
                               _swap_coastline_categories,_window,_write_legacy,run)
from pycalmet.config import CtgprocConfig,CtgprocSource,GridConfig
from pycalmet.ctg_data import NLCD92_SPECS,ModisSpec,Nlcd92Spec
from pycalmet.projections import geographic_crs,grid_crs


ROOT=Path(__file__).resolve().parents[1]


def test_glaz_translation_is_the_legacy_table():
    assert GLAZ_TO_USGS[1] == 16
    assert GLAZ_TO_USGS[16] == 52
    assert GLAZ_TO_USGS[24] == 91
    assert len(USGS38) == 38
    assert np.all(np.diff(USGS38) > 0)


def test_geotiff_orientation_four_uses_bottom_to_top_rows():
    x0,y0,dx,dy,_=_geotiff_geometry(
        (30.,30.,0.),(0.,0.,0.,500000.,4000000.,0.),1,
        CRS.from_epsg(32650),orientation=4)
    np.testing.assert_allclose([x0,y0,dx,dy],[500.015,4000.015,.03,.03])


def test_pnpoly_preserves_legacy_vertex_edge_and_inside_states():
    polygon=CoastPolygon(1,np.array([0.,2.,2.,0.],dtype=np.float32),
                          np.array([0.,0.,2.,2.],dtype=np.float32))
    actual=_polygon_membership(
        np.array([1.,3.,0.,1.,2.],dtype=np.float32),
        np.array([1.,1.,0.,0.,1.],dtype=np.float32),polygon)
    np.testing.assert_array_equal(actual,[1,-1,0,0,0])


def test_six_continental_glaz_specs_match_fortran_tables():
    assert set(GLAZ_SPECS)=={'GLAZNA','GLAZSA','GLAZEU','GLAZAS','GLAZAF','GLAZAP'}
    assert (GLAZ_SPECS['GLAZNA'].nx,GLAZ_SPECS['GLAZNA'].ny)==(9223,8996)
    assert (GLAZ_SPECS['GLAZAS'].x_origin_km,GLAZ_SPECS['GLAZAS'].longitude_origin)==(-8000.,100.)
    assert GLAZ_SPECS['GLAZAS'].y_origin_km==-5499.
    assert (GLAZ_SPECS['GLAZAP'].y_origin_km,GLAZ_SPECS['GLAZAP'].latitude_origin)==(-3944.891,-15.)
    assert len(NLCD92_SPECS)==52
    assert NLCD92_SPECS['washington_dc']==Nlcd92Spec(662,784,1610.19,1936.68)


def test_snow_grid_information_uses_sparse_sample_hits(tmp_path,monkeypatch):
    grid=GridConfig(1,1,1.,1.)
    snow=SnowGridInfo.create(grid)
    monkeypatch.setattr(
        'pycalmet._ctgproc.unproject_geographic',
        lambda x,y,grid,target_datum,units: (
            -124.729584+np.array([0.,1.,1.])*8.33333377e-3,
            np.full(np.asarray(y).shape,52.8704185)))
    snow.add(np.array([.1,.2,.2]),np.array([.1,.1,.1]),
             np.array([11,11,21]),np.ones(3,dtype=bool))
    fractions=np.zeros((len(USGS38),1,1),dtype=np.float32)
    counts=np.zeros_like(fractions,dtype=np.int64)
    output=tmp_path/'LU.DAT'
    _write_legacy(output,fractions,counts,grid,snow=snow)
    lines=output.read_text(encoding='ascii').splitlines()
    assert f'{grid.datum:<8}02-21-2003  ' in lines
    marker=lines.index('SNOW Grid Information:')
    fields=lines[marker+1].split()
    assert fields[:3]==['1','1','2']
    assert fields[3:9]==['1','2','1','0.333','2','0.333']
    assert fields[9:]==['8','1','2','0.333']


def test_snow_grid_hits_preserve_fortran_jp_index_bug_and_insertion_order(monkeypatch):
    grid=GridConfig(1,1,1.,1.)
    snow=SnowGridInfo.create(grid)
    longitude=-124.729584+np.array([0.,1.,0.,0.])*8.33333377e-3
    monkeypatch.setattr('pycalmet._ctgproc.unproject_geographic',
        lambda x,y,grid,target_datum,units:(longitude,np.full(4,52.8704185)))
    snow.add(np.full(4,.1),np.full(4,.1),np.full(4,11),np.ones(4,dtype=bool))
    grouped,totals=snow.records(USGS38)
    assert grouped[0,1]==[(1,1),(2,3)]
    assert totals.item()==4


def test_snow_processing_requires_final_non_generic_run(tmp_path):
    source=tmp_path/'generic.dat'
    source.write_text('GENERIC.LANDUSE\n0\nLL\nWGS-84\nDEG\n1\n11\n',encoding='ascii')
    grid=GridConfig(1,1,1.,1.)
    with pytest.raises(ValueError,match='final run'):
        run(CtgprocConfig([source],tmp_path/'out',grid,input_type='GEN',
                          snow_processing=True,final_output=False))
    with pytest.raises(ValueError,match='GENERIC'):
        run(CtgprocConfig([source],tmp_path/'out',grid,input_type='GEN',
                          snow_processing=True))


def test_multiple_glaz_files_accumulate_before_fraction_output(tmp_path,monkeypatch):
    files=[tmp_path/'a.img',tmp_path/'b.img']
    calls=[]
    def fake(path,config):
        calls.append(path.name); data=np.zeros((len(USGS38),1,1),dtype=np.int32)
        data[0 if path.name=='a.img' else 1,0,0]=1
        return data,1,1
    monkeypatch.setattr('pycalmet._ctgproc._accumulate_glazas',fake)
    result=run(CtgprocConfig(files,tmp_path/'land.nc',GridConfig(1,1,1.,1.),input_type='GLAZAS'))
    from netCDF4 import Dataset
    with Dataset(result.output) as dataset:
        np.testing.assert_allclose(dataset['landuse_fraction'][:2,0,0],[.5,.5])
    assert calls==['a.img','b.img'] and result.source_cells_read==2


def test_mixed_sources_follow_fortran_type_order_and_own_settings(tmp_path,monkeypatch):
    calls=[]
    def fake_global(path,config,*args,**kwargs):
        calls.append((config.input_type,config.input_datum,config.mesh_density))
        counts=np.zeros((len(USGS38),1,1),dtype=np.int64)
        counts[np.where(USGS38==11)[0].item(),0,0]=1
        return counts,1,1
    def fake_ctg(path,config):
        calls.append((config.input_type,config.input_datum,config.mesh_density))
        counts=np.zeros((len(USGS38),1,1),dtype=np.int64)
        counts[np.where(USGS38==21)[0].item(),0,0]=1
        return counts,1,1
    monkeypatch.setattr('pycalmet._ctgproc._accumulate_global_geotiff',fake_global)
    monkeypatch.setattr('pycalmet._ctgproc._accumulate_ctg',fake_ctg)
    sources=(CtgprocSource(tmp_path/'ctg','CTG','NAS-C',3),
             CtgprocSource(tmp_path/'glc','GLC2K','WGS-84',5))
    result=run(CtgprocConfig([],tmp_path/'mixed.nc',GridConfig(1,1,1.,1.),
                             landuse_sources=sources))
    assert calls==[('GLC2K','WGS-84',5),('CTG','NAS-C',3)]
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        indices=[np.where(categories==value)[0].item() for value in (11,21)]
        np.testing.assert_array_equal(dataset['landuse_fraction'][indices,0,0],[.5,.5])


def test_glaz_selection_prefers_europe_at_europe_asia_overlap():
    selected=_select_glaz_markers(60.,45.,59.,61.,44.,46.)
    assert 'GLAZEU' in selected and 'GLAZAS' not in selected
    assert _select_glaz_markers(120.,30.,119.,121.,29.,31.)=={'GLAZAS'}


def test_multiple_glaz_sources_skip_regions_outside_target(tmp_path,monkeypatch):
    calls=[]
    def fake(path,config):
        calls.append(config.input_type)
        counts=np.zeros((len(USGS38),1,1),dtype=np.int64)
        counts[0,0,0]=1
        return counts,1,1
    monkeypatch.setattr('pycalmet._ctgproc._select_glaz_regions',lambda grid:{'GLAZEU'})
    monkeypatch.setattr('pycalmet._ctgproc._accumulate_glazas',fake)
    sources=(CtgprocSource(tmp_path/'asia','GLAZAS','ESR-S'),
             CtgprocSource(tmp_path/'europe','GLAZEU','ESR-S'))
    run(CtgprocConfig([],tmp_path/'selected.nc',GridConfig(1,1,1.,1.),
                      landuse_sources=sources))
    assert calls==['GLAZEU']


def test_global_merge_drops_water_over_land_and_replaces_water_with_land():
    accumulated=np.zeros((len(USGS38),1,3),dtype=np.int64)
    current=np.zeros_like(accumulated)
    land=np.where(USGS38==11)[0].item(); water=np.where(USGS38==52)[0].item()
    accumulated[land,0,0]=2; current[water,0,0]=3
    accumulated[water,0,1]=4; current[land,0,1]=5
    accumulated[water,0,2]=6; current[water,0,2]=7
    _merge_global_counts(accumulated,current,USGS38)
    np.testing.assert_array_equal(accumulated[[land,water],0,:],
                                  [[2,5,0],[0,0,13]])


def test_global_input_is_rejected_after_a_continuation_grid(tmp_path):
    source=tmp_path/'global.img'; source.write_bytes(b'x')
    with pytest.raises(ValueError,match='first continuation run'):
        run(CtgprocConfig([source],tmp_path/'out.nc',GridConfig(1,1,1.,1.),
                          input_type='GLAZAS',previous_landuse=tmp_path/'previous.nc'))


def test_coastline_fill_only_sets_empty_cells_whose_centers_are_ocean(tmp_path,monkeypatch):
    source=tmp_path/'land.img'; source.write_bytes(b'x')
    coastline=tmp_path/'coast.bln'
    coastline.write_text('5, 1, "1,1,1"\n1,0\n3,0\n3,1\n1,1\n1,0\n',encoding='ascii')
    def fake(path,config):
        counts=np.zeros((len(USGS38),1,3),dtype=np.int64)
        counts[np.where(USGS38==11)[0].item(),0,1]=2
        return counts,1,1
    monkeypatch.setattr('pycalmet._ctgproc._accumulate_glazas',fake)
    config=CtgprocConfig([source],tmp_path/'coast.nc',GridConfig(3,1,1.,1.),
        input_type='GLAZAS',coastline_file=coastline,fill_empty_ocean=True)
    result=run(config)
    with Dataset(result.output) as dataset:
        ocean=np.where(dataset['landuse_category'][:]==55)[0].item()
        np.testing.assert_array_equal(dataset['sample_count'][ocean,0,:],[1,0,0])
        np.testing.assert_array_equal(dataset['landuse_fraction'][ocean,0,:],[1.,0.,0.])
    assert result.populated_cells==2


def test_coastline_nested_types_and_edges_match_iptype(tmp_path):
    coastline=tmp_path/'nested.bln'
    coastline.write_text(
        '5, 1, "1,1,1"\n0,0\n4,0\n4,4\n0,4\n0,0\n'
        '5, 1, "2,1,2"\n1,1\n3,1\n3,3\n1,3\n1,1\n',encoding='ascii')
    polygons=_read_coastline_bln(coastline,GridConfig(4,4,1.,1.))
    result=_coastline_types(np.array([-1.,.5,2.,0.,1.]),
                            np.array([2.,.5,2.,2.,2.]),polygons)
    # ocean, mainland, lake, mainland edge->ocean, lake edge->lake
    np.testing.assert_array_equal(result,[0,1,2,0,2])
    config=CtgprocConfig([],tmp_path/'unused',GridConfig(4,4,1.,1.),
                         coastline_swap=True,ocean_category=55)
    swapped,keep=_swap_coastline_categories(
        np.array([0,11,54,55,11]),np.array([-1.,-1.,-1.,.5,.5]),
        np.full(5,2.),config,polygons)
    np.testing.assert_array_equal(swapped,[0,55,54,55,11])
    np.testing.assert_array_equal(keep,[True,True,True,False,True])


def test_coastline_swap_reclassifies_each_source_sample_before_binning(tmp_path):
    grid=GridConfig(2,1,1.,1.)
    inverse=Transformer.from_crs(grid_crs(grid,units='km'),
                                 geographic_crs('WGS-84'),always_xy=True)
    ocean_lon,ocean_lat=inverse.transform(.5,.5)
    land_lon,land_lat=inverse.transform(1.5,.5)
    source=tmp_path/'nzgen.dat'
    # NZ classes 1, 50, and 1 map to USGS 55, 52, and 55 respectively;
    # class 10 maps to 16.  On ocean, all but 54 become IOCEAN=55.  On
    # land, 54/55 are dropped while ordinary land classes remain.
    source.write_text(
        f'1 10 {ocean_lat} {ocean_lon}\n'
        f'2 50 {ocean_lat} {ocean_lon}\n'
        f'3 1 {land_lat} {land_lon}\n'
        f'4 10 {land_lat} {land_lon}\n',encoding='ascii')
    coastline=tmp_path/'coast.bln'
    coastline.write_text('5, 1, "1,1,1"\n1,0\n2,0\n2,1\n1,1\n1,0\n',encoding='ascii')
    result=run(CtgprocConfig([source],tmp_path/'swap.nc',grid,input_type='NZGEN',
        input_datum='WGS-84',coastline_file=coastline,coastline_swap=True))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        ocean=np.where(categories==55)[0].item(); land=np.where(categories==16)[0].item()
        assert dataset['sample_count'][ocean,0,0]==2
        assert dataset['sample_count'][land,0,1]==1
        assert dataset['sample_count'][:,0,1].sum()==1
    assert result.samples_inside_grid==3


def test_generic_point_format_preserves_categories_and_accumulates_cells(tmp_path):
    grid=GridConfig(2,1,1.,1.,x_origin_km=0.,y_origin_km=0.)
    inverse=Transformer.from_crs(grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True)
    lon0,lat0=inverse.transform(.25,.5); lon1,lat1=inverse.transform(1.25,.5)
    generic=tmp_path/'generic.lu'
    generic.write_text(
        'GENERIC.LANDUSE 1.0             LU, Longitude, Latitude (free-format)\n'
        '2\nPrepared for test\nLongitude east, latitude north\nLL      \n'
        'WGS-84  02-21-2003\nDEG \n3\n7 42 99\n'
        f'7 {lon0:.12f} {lat0:.12f}\n42 {lon0:.12f} {lat0:.12f}\n99 {lon1:.12f} {lat1:.12f}\n',encoding='ascii')
    result=run(CtgprocConfig([generic],tmp_path/'generic.nc',grid,input_type='GEN'))
    with Dataset(result.output) as dataset:
        np.testing.assert_array_equal(dataset['landuse_category'][:],[7,42,99])
        np.testing.assert_allclose(dataset['landuse_fraction'][:,0,0],[.5,.5,0.])
        np.testing.assert_allclose(dataset['landuse_fraction'][:,0,1],[0.,0.,1.])
        assert dataset.average_landuse_hits==1
        assert dataset.low_hit_threshold==1
        assert dataset.low_hit_cell_count==0
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(3,3,2)
    assert (result.average_hits,result.low_hit_threshold,result.low_hit_cells)==(1,1,0)


def test_compressed_ctg_runs_expand_from_utm_anchor(tmp_path):
    source=tmp_path/'sample.ctg'
    header=(' '*50+f'{10:5d}'+' synthetic CTG header\n'+
            'header 2\nheader 3\nheader 4\nheader 5\n')
    anchor=f'{0:4d}{0:3d}{500000:8d}{4000000:9d}\n'
    run_record=f'{2:4d}{11:3d}{0:8d}{0:9d}\n'
    source.write_text(header+anchor+run_record,encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=10,datum='NAS-C',
                    x_origin_km=499.5,y_origin_km=3999.5)
    result=run(CtgprocConfig([source],tmp_path/'ctg.nc',grid,input_type='CTG',
                             input_datum='NAS-C',mesh_density=1))
    with Dataset(result.output) as dataset:
        category=np.where(dataset['landuse_category'][:]==11)[0].item()
        assert dataset['sample_count'][category,0,0]==2
        assert dataset['landuse_fraction'][category,0,0]==1.
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(2,2,1)


def test_output_format_can_override_filename_suffix(tmp_path):
    source=tmp_path/'sample.ctg'
    source.write_text(' '*50+f'{10:5d}'+'\n2\n3\n4\n5\n'+
        f'{0:4d}{0:3d}{500000:8d}{4000000:9d}\n'+
        f'{1:4d}{11:3d}{0:8d}{0:9d}\n',encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=10,datum='NAS-C',
                    x_origin_km=499.5,y_origin_km=3999.5)
    output=tmp_path/'landuse.dat'
    run(CtgprocConfig([source],output,grid,input_type='CTG',input_datum='NAS-C',
                      output_format='netcdf'))
    with Dataset(output) as dataset:
        assert dataset.landuse_file_type=='FRACTION'


def test_output_format_rejects_unknown_value(tmp_path):
    config=CtgprocConfig([],tmp_path/'landuse.dat',GridConfig(1,1,1.,1.),
                         output_format='binary')
    with pytest.raises(ValueError,match='output_format'):
        run(config)


def test_ctg_cell_qa_writes_legacy_named_coordinate_files(tmp_path):
    source=tmp_path/'sample.ctg'
    source.write_text(' '*50+f'{10:5d}'+'\n2\n3\n4\n5\n'+
        f'{0:4d}{0:3d}{500000:8d}{4000000:9d}\n'+
        f'{2:4d}{11:3d}{0:8d}{0:9d}\n',encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=10,datum='NAS-C',
                    x_origin_km=499.5,y_origin_km=3999.5)
    run(CtgprocConfig([source],tmp_path/'ctg.nc',grid,input_type='CTG',
                      input_datum='NAS-C',mesh_density=1,qa_cell_output=True))
    assert {path.name for path in tmp_path.glob('qa*.dat')}==set(QA_FILENAMES)
    mesh=np.loadtxt(tmp_path/'qamesh.dat',ndmin=2)
    cells=np.loadtxt(tmp_path/'qactg.dat',ndmin=2)
    assert mesh.shape==(2,2)
    assert cells.shape==(2,3)
    np.testing.assert_allclose(cells[:,2],11.)
    assert (tmp_path/'qaglaz.dat').stat().st_size==0


def test_compressed_ctg_rejects_nonstandard_in_range_category(tmp_path):
    source=tmp_path/'bad.ctg'
    source.write_text(' '*50+f'{10:5d}'+'\n2\n3\n4\n5\n'+
        f'{0:4d}{0:3d}{500000:8d}{4000000:9d}\n'+
        f'{1:4d}{10:3d}{0:8d}{0:9d}\n',encoding='ascii')
    grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=10,datum='NAS-C',
                    x_origin_km=499.5,y_origin_km=3999.5)
    with pytest.raises(ValueError,match='USGS-38'):
        run(CtgprocConfig([source],tmp_path/'bad.nc',grid,input_type='CTG',input_datum='NAS-C'))


def test_nzgen_points_use_legacy_category_translation(tmp_path):
    assert (NZ_TO_USGS[1],NZ_TO_USGS[10],NZ_TO_USGS[72])==(55,16,74)
    grid=GridConfig(1,1,1.,1.)
    inverse=Transformer.from_crs(grid_crs(grid,units='km'),
                                 geographic_crs('WGS-84'),always_xy=True)
    longitude,latitude=inverse.transform(.5,.5)
    source=tmp_path/'nzgen.dat'
    source.write_text(f'1 0 {latitude} {longitude}\n2 10 {latitude} {longitude}\n',
                      encoding='ascii')
    result=run(CtgprocConfig([source],tmp_path/'nzgen.nc',grid,input_type='NZGEN',
                             input_datum='WGS-84'))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        water=np.where(categories==55)[0].item(); urban=np.where(categories==16)[0].item()
        np.testing.assert_allclose(dataset['landuse_fraction'][[water,urban],0,0],[.5,.5])
    assert (result.source_cells_read,result.samples_inside_grid)==(2,2)


def test_nlcd92_flat_grid_uses_state_geometry_and_class_table(tmp_path,monkeypatch):
    monkeypatch.setitem(NLCD92_SPECS,'test',Nlcd92Spec(2,2,0.,.03))
    source=tmp_path/'test_nlcd.bin'
    np.array([[11,12],[21,0]],dtype='u1').tofile(source)
    grid=GridConfig(1,1,.06,.06,projection='ACEA',lat0=23.,lon0=-96.,
                    lat1=29.5,lat2=45.5,datum='NAR-C',
                    x_origin_km=-.015,y_origin_km=-.015)
    result=run(CtgprocConfig([source],tmp_path/'nlcd.nc',grid,input_type='NLCD92',
                             input_datum='NAR-C'))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        indices=[np.where(categories==value)[0].item() for value in (11,52,91)]
        np.testing.assert_allclose(dataset['landuse_fraction'][indices,0,0],[1/3,1/3,1/3])
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(4,3,1)


@pytest.mark.parametrize(('kind','values','expected'),[
    ('NLCD01',[[22,32],[72,0]],[11,72,82]),
    ('NLCDTF',[[11,12],[21,0]],[52,91,11]),
])
def test_nlcd_geotiff_uses_affine_epsg_and_versioned_table(tmp_path,kind,values,expected):
    source=tmp_path/f'{kind.lower()}.tif'
    keys=(1,1,0,3,1024,0,1,1,1025,0,1,1,3072,0,1,5070)
    tifffile.imwrite(source,np.asarray(values,dtype='u1'),extratags=[
        (33550,'d',3,(30.,30.,0.),False),
        (33922,'d',6,(0.,0.,0.,0.,60.,0.),False),
        (34735,'H',len(keys),keys,False)])
    grid=GridConfig(1,1,.06,.06,projection='ACEA',lat0=23.,lon0=-96.,
                    lat1=29.5,lat2=45.5,datum='NAR-C')
    result=run(CtgprocConfig([source],tmp_path/f'{kind}.nc',grid,input_type=kind))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        indices=[np.where(categories==value)[0].item() for value in expected]
        np.testing.assert_allclose(dataset['landuse_fraction'][indices,0,0],[1/3]*3)
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(4,3,1)


@pytest.mark.parametrize(('kind','values','expected','fractions','inside'),[
    ('GLC2K',[[1,2],[24,3]],[42,41],[1/3,2/3],12),
    ('UMDGLC',[[0,1],[2,14]],[55,42,77],[.25,.5,.25],16),
])
def test_global_geotiff_meshes_pixels_and_translates_classes(
        tmp_path,kind,values,expected,fractions,inside):
    source=tmp_path/f'{kind.lower()}.tif'
    raster_type=2 if kind=='GLC2K' else 1
    keys=(1,1,0,3,1024,0,1,2,1025,0,1,raster_type,2048,0,1,4326)
    tags=[
        (33550,'d',3,(.01,.01,0.),False),
        (33922,'d',6,(0.,0.,0.,-.02,.02,0.),False)]
    if kind!='GLC2K': tags.append((34735,'H',len(keys),keys,False))
    tifffile.imwrite(source,np.asarray(values,dtype='u1'),extratags=tags)
    grid=GridConfig(1,1,2.,2.,projection='LAZA',lat0=0.,lon0=0.,
                    x_origin_km=-2.,y_origin_km=0.)
    result=run(CtgprocConfig([source],tmp_path/f'{kind}.nc',grid,input_type=kind,
                             mesh_density=2))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        indices=[np.where(categories==value)[0].item() for value in expected]
        np.testing.assert_allclose(dataset['landuse_fraction'][indices,0,0],fractions)
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(4,inside,1)


def test_corine_supports_index_class_and_rgb_encodings(tmp_path):
    indexed=_corine_indices(np.array([[1,33,44]],dtype='u1'),3,tmp_path/'index.tif')
    classes=_corine_indices(np.array([[111,334,523]],dtype='u2'),1,tmp_path/'class.tif')
    rgb=_corine_indices(np.array([[[230,0,77],[0,0,0],[230,242,255]]],dtype='u1'),
                         2,tmp_path/'rgb.tif')
    np.testing.assert_array_equal(indexed,[[1,33,44]])
    np.testing.assert_array_equal(classes,[[1,33,44]])
    np.testing.assert_array_equal(rgb,[[1,33,44]])
    np.testing.assert_array_equal(CORINE_TO_USGS[indexed],[[16,74,55]])


def test_corine_geotiff_runs_through_public_accumulator(tmp_path):
    source=tmp_path/'corine.tif'
    keys=(1,1,0,3,1024,0,1,2,1025,0,1,1,2048,0,1,4326)
    tifffile.imwrite(source,np.array([[111,112],[333,0]],dtype='u2'),extratags=[
        (33550,'d',3,(.01,.01,0.),False),
        (33922,'d',6,(0.,0.,0.,-.02,.02,0.),False),
        (34735,'H',len(keys),keys,False)])
    grid=GridConfig(1,1,2.,2.,projection='LAZA',lat0=0.,lon0=0.,
                    x_origin_km=-2.,y_origin_km=0.)
    result=run(CtgprocConfig([source],tmp_path/'corine.nc',grid,input_type='CORINE',
                             mesh_density=1))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        forest=np.where(categories==16)[0].item(); wetland=np.where(categories==83)[0].item()
        np.testing.assert_allclose(dataset['landuse_fraction'][[forest,wetland],0,0],[2/3,1/3])
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(4,3,1)


def test_raw_modis_windows_meshes_and_preserves_itab_overwrite(tmp_path,monkeypatch):
    spec=ModisSpec(2,2,-.015,.015,.01)
    monkeypatch.setattr('pycalmet._ctgproc.MODIS_SPEC',spec)
    source=tmp_path/'modis.bin'; np.array([[1,2],[0,18]],dtype='u1').tofile(source)
    grid=GridConfig(1,1,2.,2.,projection='LAZA',lat0=0.,lon0=0.,datum='ESR-R',
                    x_origin_km=-2.,y_origin_km=0.)
    result=run(CtgprocConfig([source],tmp_path/'modis.nc',grid,input_type='MODIS',
                             input_datum='ESR-R',mesh_density=2))
    with Dataset(result.output) as dataset:
        categories=dataset['landuse_category'][:]
        indices=[np.where(categories==value)[0].item() for value in (16,21,61)]
        np.testing.assert_allclose(dataset['landuse_fraction'][indices,0,0],[1/3]*3)
    assert (result.source_cells_read,result.samples_inside_grid,result.populated_cells)==(4,12,1)


def test_generic_files_require_matching_category_tables(tmp_path):
    header='GENERIC.LANDUSE 1.0\n0\nLL\nWGS-84\nDEG\n1\n{}\n'
    first=tmp_path/'one.lu'; second=tmp_path/'two.lu'
    first.write_text(header.format(1),encoding='ascii'); second.write_text(header.format(2),encoding='ascii')
    with pytest.raises(ValueError,match='identical category'):
        run(CtgprocConfig([first,second],tmp_path/'out.nc',GridConfig(1,1,1.,1.),input_type='GEN'))
    with pytest.raises(ValueError,match='GENERIC'):
        run(CtgprocConfig([first],tmp_path/'coast.nc',GridConfig(1,1,1.,1.),
                          input_type='GEN',coastline_swap=True))


def test_category_map_merges_source_classes_in_declared_order(tmp_path):
    grid=GridConfig(2,1,1.,1.); inverse=Transformer.from_crs(
        grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True)
    lon0,lat0=inverse.transform(.25,.5); lon1,lat1=inverse.transform(1.25,.5)
    source=tmp_path/'mapped.lu'
    source.write_text('GENERIC.LANDUSE 1.0\n0\nLL\nWGS-84\nDEG\n3\n7 42 99\n'
        f'7 {lon0} {lat0}\n42 {lon0} {lat0}\n99 {lon1} {lat1}\n',encoding='ascii')
    result=run(CtgprocConfig([source],tmp_path/'mapped.nc',grid,input_type='GEN',
                             category_map={7:10,42:10,99:20}))
    with Dataset(result.output) as dataset:
        np.testing.assert_array_equal(dataset['landuse_category'][:],[10,20])
        np.testing.assert_array_equal(dataset['sample_count'][:,0,:],[[2,0],[0,1]])
        np.testing.assert_array_equal(dataset['landuse_fraction'][:,0,:],[[1.,0.],[0.,1.]])


@pytest.mark.parametrize('extension',['nc','dat'])
def test_continuation_hits_round_trip_and_accumulate(tmp_path,extension):
    grid=GridConfig(1,1,1.,1.)
    inverse=Transformer.from_crs(grid_crs(grid,units='km'),
                                 geographic_crs('WGS-84'),always_xy=True)
    longitude,latitude=inverse.transform(.5,.5)
    source=tmp_path/'nzgen.dat'
    source.write_text(f'1 1 {latitude} {longitude}\n',encoding='ascii')
    first=tmp_path/f'first.{extension}'
    run(CtgprocConfig([source],first,grid,input_type='NZGEN',input_datum='WGS-84',
                      final_output=False))
    if extension=='nc':
        with Dataset(first,'r+') as dataset: dataset.datum='WGS84'
    else:
        text=first.read_text(encoding='ascii')
        first.write_text(text.replace('WGS-84  ','WGS84   '),encoding='ascii')
    water=np.where(USGS38==55)[0].item()
    if extension=='nc':
        with Dataset(first) as dataset:
            assert dataset.landuse_file_type=='HITS'
            assert dataset['sample_count'][water,0,0]==1
    else:
        assert '\nHITS\n' in first.read_text(encoding='ascii')
    final=tmp_path/'final.nc'
    result=run(CtgprocConfig([source],final,grid,input_type='NZGEN',input_datum='WGS-84',
                             previous_landuse=first,final_output=True))
    with Dataset(final) as dataset:
        assert dataset.landuse_file_type=='FRACTION'
        assert dataset['sample_count'][water,0,0]==2
        assert dataset['landuse_fraction'][water,0,0]==1.
    assert result.populated_cells==1
