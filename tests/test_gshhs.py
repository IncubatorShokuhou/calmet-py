import struct

import numpy as np
from netCDF4 import Dataset
import pytest
from pyproj import Transformer

from pycalmet._ctgproc import _read_coastline_bln,run
from pycalmet.config import CtgprocConfig,GridConfig
from pycalmet.gshhs import (boundary_side,clip_coast_polygon,extract_gshhs,
                            iter_gshhs)
from pycalmet.projections import geographic_crs,grid_crs


def _points(*values):
    return struct.pack(f'>{len(values)}i',*values)


def test_reads_modern_gshhs_v2_flags_and_points(tmp_path):
    path=tmp_path/'gshhs.b'
    flag=1|(10<<8)|(1<<16)|(1<<24)|(1<<25)
    header=(7,2,flag,1000000,2000000,-3000000,4000000,50,51,-1,-1)
    path.write_bytes(struct.pack('>11i',*header)+_points(1000000,-3000000,2000000,4000000))
    polygon,=iter_gshhs(path)
    assert (polygon.header.identifier,polygon.header.level,polygon.header.version)==(7,1,10)
    assert polygon.header.crosses_greenwich and polygon.header.source==1
    assert polygon.header.river_lake and polygon.header.full_area_tenth_km2==51
    np.testing.assert_array_equal(polygon.longitude,[1.,2.])
    np.testing.assert_array_equal(polygon.latitude,[-3.,4.])


@pytest.mark.parametrize('version', [4,5])
def test_reads_packed_gshhs_v14_and_v15(tmp_path,version):
    path=tmp_path/'old.b'; flag=2|(version<<8)|(1<<16)
    path.write_bytes(struct.pack('>8i',3,1,flag,0,1,2,3,4)+_points(500000,250000))
    polygon,=iter_gshhs(path)
    assert polygon.header.version==version and polygon.header.level==2
    assert polygon.header.crosses_greenwich and polygon.header.source==0


def test_reads_gshhs_v13_and_pre_v13_layouts(tmp_path):
    v13=tmp_path/'v13.b'
    v13.write_bytes(struct.pack('>9i2h',1,1,3,0,1,2,3,4,3,1,0)+_points(1,2))
    polygon,=iter_gshhs(v13)
    assert polygon.header.version==3 and polygon.header.crosses_greenwich

    legacy=tmp_path/'legacy.b'
    legacy.write_bytes(struct.pack('>8i2h',2,1,4,0,1,2,3,4,0,1)+_points(3,4))
    polygon,=iter_gshhs(legacy)
    assert polygon.header.identifier==2 and polygon.header.level==4
    assert not polygon.header.crosses_greenwich and polygon.header.source==1


def test_rejects_truncated_points_and_inconsistent_following_version(tmp_path):
    truncated=tmp_path/'truncated.b'; flag=1|(10<<8)
    truncated.write_bytes(struct.pack('>11i',1,2,flag,0,0,0,0,1,1,-1,-1)+_points(0,0))
    with pytest.raises(ValueError,match='truncated GSHHS points'):
        list(iter_gshhs(truncated))

    mixed=tmp_path/'mixed.b'
    first=struct.pack('>11i',1,0,flag,0,0,0,0,1,1,-1,-1)
    second=struct.pack('>11i',2,0,1|(9<<8),0,0,0,0,1,1,-1,-1)
    mixed.write_bytes(first+second)
    with pytest.raises(ValueError,match='inconsistent GSHHS v2'):
        list(iter_gshhs(mixed))


def test_boundary_side_matches_iside_ties_and_inside_rules():
    x=np.array([-2.,.5,2.,.5,.1,.9,-1.,2.])
    y=np.array([.5,2.,.5,-2.,.5,.5,2.,2.])
    np.testing.assert_array_equal(boundary_side(x,y,1.,0.,1.,0.),
                                  [1,2,3,4,1,3,2,2])


def test_clip_coast_polygon_retains_outside_endpoints_and_closes_land():
    longitude=np.array([-.5,.2,.8,1.5,1.5,-.5])
    latitude=np.array([.2,.2,.8,.8,-.5,-.5])
    result=clip_coast_polygon(longitude,latitude,longitude,latitude,1,
                              (0.,1.,0.,1.),(0.,1.,0.,1.))
    np.testing.assert_array_equal(result,np.array([
        [-.5,.2],[.2,.2],[.8,.8],[1.5,.8],[1.,1.],[0.,1.],[0.,0.],[-.5,.2]
    ]))


def test_clip_surrounding_polygon_uses_level_direction():
    longitude=np.array([-2.,-2.,.5,2.,2.,2.,.5,-2.,-2.])
    latitude=np.array([.5,-2.,-2.,-2.,.5,2.,2.,2.,.5])
    # The old surround test also requires observations on all four nearest sides.
    land=clip_coast_polygon(longitude,latitude,longitude,latitude,1,
                            (0.,1.,0.,1.),(0.,1.,0.,1.),(-2.,2.,-2.,2.))
    water=clip_coast_polygon(longitude[::-1],latitude[::-1],longitude[::-1],latitude[::-1],2,
                             (0.,1.,0.,1.),(0.,1.,0.,1.),(-2.,2.,-2.,2.))
    np.testing.assert_array_equal(land,[[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    np.testing.assert_array_equal(water,[[0.,1.],[1.,1.],[1.,0.],[0.,0.],[0.,1.]])


def test_extract_gshhs_projects_closes_and_writes_readable_bln(tmp_path):
    source=tmp_path/'gshhs.b'; flag=1|(10<<8)|(1<<24)
    longitude=np.array([-.02,-.02,0.,.02,.02,.02,0.,-.02,-.02])
    latitude=np.array([0.,-.02,-.02,-.02,0.,.02,.02,.02,0.])
    coordinates=np.column_stack((longitude,latitude))*1.e6
    header=(9,len(longitude),flag,int(coordinates[:,0].min()),int(coordinates[:,0].max()),
            int(coordinates[:,1].min()),int(coordinates[:,1].max()),100,100,-1,-1)
    source.write_bytes(struct.pack('>11i',*header)+
                       struct.pack(f'>{coordinates.size}i',*coordinates.astype(int).ravel()))
    grid=GridConfig(2,2,1.,1.,projection='EM',lat0=0.,lon0=0.,
                    x_origin_km=-1.,y_origin_km=-1.)
    output=tmp_path/'coast.bln'
    assert extract_gshhs(source,output,grid)==1
    polygon,=_read_coastline_bln(output,grid)
    assert polygon.kind==1 and len(polygon.x)==5
    np.testing.assert_allclose([polygon.x.min(),polygon.x.max(),polygon.y.min(),polygon.y.max()],
                               [-1.0001,1.0001,-1.0001,1.0001],atol=1.e-6)


def test_extract_gshhs_skips_disabled_wdbii_and_uses_wvs_datum(tmp_path,monkeypatch):
    source=tmp_path/'mixed-source.b'
    longitude=np.array([-.002,.002,.002,-.002,-.002])
    latitude=np.array([-.002,-.002,.002,.002,-.002])
    coordinates=np.column_stack((longitude,latitude))*1.e6
    payload=[]
    for identifier,source_flag in ((1,0),(2,1)):
        flag=1|(10<<8)|(source_flag<<24)
        header=(identifier,len(longitude),flag,int(coordinates[:,0].min()),
                int(coordinates[:,0].max()),int(coordinates[:,1].min()),
                int(coordinates[:,1].max()),100,100,-1,-1)
        payload.append(struct.pack('>11i',*header))
        payload.append(struct.pack(f'>{coordinates.size}i',
                                   *coordinates.astype(int).ravel()))
    source.write_bytes(b''.join(payload))
    seen=[]
    def shift(lon,lat,*,source,target):
        seen.append(source)
        return np.asarray(lon),np.asarray(lat),np.zeros(np.asarray(lon).shape)
    monkeypatch.setattr('pycalmet.gshhs.datum_shift',shift)
    grid=GridConfig(2,2,1.,1.,projection='EM',lat0=0.,lon0=0.,
                    x_origin_km=-1.,y_origin_km=-1.)
    count=extract_gshhs(source,tmp_path/'coast.bln',grid,source_datum='WGS-84',
                        wdbii_datum='NAR-C')
    assert count==1
    assert seen==['WGS-84']


def test_empty_direct_extraction_represents_an_all_ocean_domain(tmp_path):
    source=tmp_path/'gshhs.b'; source.write_bytes(b'')
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.)
    output=tmp_path/'coast.bln'
    assert extract_gshhs(source,output,grid)==0 and output.read_text()==''
    assert _read_coastline_bln(output,grid,allow_empty=True)==()


def test_ctgproc_direct_gshhs_empty_domain_swaps_samples_to_ocean(tmp_path):
    grid=GridConfig(1,1,1.,1.,projection='EM',lat0=0.,lon0=0.)
    longitude,latitude=Transformer.from_crs(
        grid_crs(grid,units='km'),geographic_crs('WGS-84'),always_xy=True
    ).transform(.5,.5)
    landuse=tmp_path/'nzgen.dat'; landuse.write_text(
        f'1 10 {latitude} {longitude}\n',encoding='ascii')
    gshhs=tmp_path/'gshhs.b'; gshhs.write_bytes(b'')
    output=tmp_path/'landuse.nc'
    run(CtgprocConfig([landuse],output,grid,input_type='NZGEN',
        input_datum='WGS-84',gshhs_file=gshhs,coastline_file=tmp_path/'coast.bln',
        coastline_swap=True,fill_empty_ocean=False))
    with Dataset(output) as dataset:
        categories=np.asarray(dataset['landuse_category'][:])
        fractions=np.asarray(dataset['landuse_fraction'][:])
    assert fractions[np.flatnonzero(categories==55)[0],0,0]==1.
