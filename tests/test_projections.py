import numpy as np
import pytest
from pyproj import CRS,Transformer

from pycalmet.config import GridConfig
from pycalmet.datums import DATUMS,datum_definition,datum_shift
from pycalmet.legacy._compat.config import GridConfig as LegacyGridConfig
from pycalmet.legacy._compat.projections import (
    geographic_crs as legacy_geographic_crs,
    grid_crs as legacy_grid_crs,
)
from pycalmet.projections import (geographic_crs,grid_convergence,grid_crs,
                                  project_geographic,project_glaz_to_lcc,transform_grid,
                                  unproject_geographic)


def test_current_lcc_definition_preserves_kilometre_coordinates():
    grid=GridConfig(2,2,1.,1.,projection='LCC')
    # COORDLIB's NIMA table intentionally carries its 2003 WGS-84 inverse
    # flattening precision, rather than PROJ's newer EPSG precision.
    old=CRS.from_proj4('+proj=lcc +lat_0=27 +lon_0=117.5 +lat_1=30 +lat_2=60 '
                       '+a=6378137 +rf=298.2572 +towgs84=0,0,0 +units=km +no_defs')
    lon=np.array([117.5,120.]); lat=np.array([27.,30.])
    expected=Transformer.from_crs('EPSG:4326',old,always_xy=True).transform(lon,lat)
    actual=Transformer.from_crs('EPSG:4326',grid_crs(grid,units='km'),always_xy=True).transform(lon,lat)
    np.testing.assert_allclose(actual,expected,rtol=0.,atol=1.e-12)


def test_all_calpuff_projection_codes_round_trip():
    cases=[
        GridConfig(1,1,1.,1.,projection='LCC'),
        GridConfig(1,1,1.,1.,projection='TTM'),
        GridConfig(1,1,1.,1.,projection='UTM',utm_zone=50,hemisphere='N'),
        GridConfig(1,1,1.,1.,projection='PS',lat0=90.,lat1=60.),
        GridConfig(1,1,1.,1.,projection='EM'),
        GridConfig(1,1,1.,1.,projection='LAZA'),
        GridConfig(1,1,1.,1.,projection='ACEA',lat0=23.,lon0=-96.,lat1=29.5,lat2=45.5),
    ]
    for grid in cases:
        forward=Transformer.from_crs('EPSG:4326',grid_crs(grid),always_xy=True)
        reverse=Transformer.from_crs(grid_crs(grid),'EPSG:4326',always_xy=True)
        x,y=forward.transform(117.8,28.1); lon,lat=reverse.transform(x,y)
        assert np.isfinite([x,y,lon,lat]).all()
        np.testing.assert_allclose([lon,lat],[117.8,28.1],atol=1.e-8)


def test_ttm_uses_control_scale_factor():
    grid=GridConfig(1,1,1.,1.,projection='TTM',lat0=30.,lon0=120.,
                    ttm_scale=.9996)
    expected=CRS.from_proj4(
        '+proj=tmerc +lat_0=30 +lon_0=120 +k=.9996 +a=6378137 '
        '+rf=298.2572 +towgs84=0,0,0 +units=m +no_defs')
    actual=Transformer.from_crs('EPSG:4326',grid_crs(grid),always_xy=True).transform(121.,31.)
    wanted=Transformer.from_crs('EPSG:4326',expected,always_xy=True).transform(121.,31.)
    np.testing.assert_allclose(actual,wanted,rtol=0.,atol=1.e-8)


@pytest.mark.parametrize(
    ('grid_type','geographic_crs_function','grid_crs_function'),
    ((GridConfig,geographic_crs,grid_crs),
     (LegacyGridConfig,legacy_geographic_crs,legacy_grid_crs)),
    ids=('public','legacy'),
)
@pytest.mark.parametrize(('hemisphere','expected_y'),(('N',0.),('S',10000.)))
def test_spherical_datum_utm_uses_equivalent_transverse_mercator(
        grid_type,geographic_crs_function,grid_crs_function,
        hemisphere,expected_y):
    grid=grid_type(1,1,1.,1.,projection='UTM',utm_zone=50,
                   hemisphere=hemisphere,datum='ESR-S')
    forward=Transformer.from_crs(
        geographic_crs_function('ESR-S'),grid_crs_function(grid,units='km'),
        always_xy=True)
    reverse=Transformer.from_crs(
        grid_crs_function(grid,units='km'),geographic_crs_function('ESR-S'),
        always_xy=True)
    x,y=forward.transform(117.,0.)
    np.testing.assert_allclose([x,y],[500.,expected_y],rtol=0.,atol=1.e-9)
    np.testing.assert_allclose(reverse.transform(x,y),[117.,0.],rtol=0.,atol=1.e-10)


@pytest.mark.parametrize(
    ('projection','projection_parameters','point'),
    (('UTM',{'utm_zone':50,'hemisphere':'N'},(118.,28.)),
     ('PS',{'lat0':70.,'lon0':20.,'lat1':60.},(25.,75.)),
     ('EM',{'lat0':0.,'lon0':20.},(25.,30.))),
)
def test_utm_ps_em_ignore_user_false_offsets_and_match_public_legacy(
        projection,projection_parameters,point):
    results=[]
    for grid_type,geographic_crs_function,grid_crs_function in (
            (GridConfig,geographic_crs,grid_crs),
            (LegacyGridConfig,legacy_geographic_crs,legacy_grid_crs)):
        base=grid_type(1,1,1.,1.,projection=projection,
                       **projection_parameters)
        with_user_offsets=grid_type(
            1,1,1.,1.,projection=projection,
            false_easting_km=123.5,false_northing_km=-456.25,
            **projection_parameters)
        source=geographic_crs_function(base.datum)
        base_transformer=Transformer.from_crs(
            source,grid_crs_function(base,units='km'),always_xy=True)
        offset_transformer=Transformer.from_crs(
            source,grid_crs_function(with_user_offsets,units='km'),always_xy=True)
        base_result=base_transformer.transform(*point)
        offset_result=offset_transformer.transform(*point)
        np.testing.assert_allclose(offset_result,base_result,rtol=0.,atol=1.e-9)
        results.append(base_result)
    np.testing.assert_allclose(results[1],results[0],rtol=0.,atol=1.e-9)


@pytest.mark.parametrize(
    ('grid_type','geographic_crs_function','grid_crs_function'),
    ((GridConfig,geographic_crs,grid_crs),
     (LegacyGridConfig,legacy_geographic_crs,legacy_grid_crs)),
    ids=('public','legacy'),
)
@pytest.mark.parametrize(('lat0','lat1'),((70.,60.),(-70.,-60.)),ids=('north','south'))
def test_polar_stereographic_control_origin_maps_to_zero(
        grid_type,geographic_crs_function,grid_crs_function,lat0,lat1):
    grid=grid_type(
        1,1,1.,1.,projection='PS',lat0=lat0,lon0=20.,lat1=lat1,
        false_easting_km=123.5,false_northing_km=-456.25)
    projected=grid_crs_function(grid,units='km')
    geographic=geographic_crs_function(grid.datum)
    forward=Transformer.from_crs(geographic,projected,always_xy=True)
    reverse=Transformer.from_crs(projected,geographic,always_xy=True)
    x,y=forward.transform(grid.lon0,grid.lat0)
    np.testing.assert_allclose([x,y],[0.,0.],rtol=0.,atol=1.e-9)
    np.testing.assert_allclose(
        reverse.transform(0.,0.),[grid.lon0,grid.lat0],rtol=0.,atol=1.e-10)


def test_documented_calpuff_datums_have_explicit_proj_definitions():
    for datum in ('WGS-84','WGS-96','WGS-G','WGS-72','NAS-C','NAR-C','NWS-84','ESR-S'):
        crs=geographic_crs(datum)
        assert crs.is_geographic
        forward=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
        reverse=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
        lon,lat=forward.transform(*reverse.transform(-100.,40.))
        np.testing.assert_allclose([lon,lat],[-100.,40.],atol=2.e-7)

    shifted=Transformer.from_crs(geographic_crs('NAS-C'),'EPSG:4326',always_xy=True).transform(-100.,40.)
    assert np.hypot(shifted[0]+100.,shifted[1]-40.)>1.e-5


def test_complete_nima_datum_table_can_construct_geographic_crs():
    assert len(DATUMS)==232
    assert datum_definition('NAS-C')==(6378206.,294.9787,-8.,160.,176.)
    assert datum_definition('WGS84')==datum_definition('WGS-84')
    for datum in DATUMS:
        assert geographic_crs(datum).is_geographic


def test_vectorized_proj_datum_shift_stays_within_legacy_coordlib_tolerance():
    longitude=np.array([-100.,-100.]); latitude=np.array([40.,40.]); height=np.array([123.,123.])
    actual=np.asarray(datum_shift(longitude,latitude,height,source='NAS-C',target='WGS-84'))
    legacy=np.array([
        [-100.00041760087252,-100.00041760087252],
        [40.0000094957673,40.0000094957673],
        [87.38544217546216,87.38544217546216],
    ])
    assert actual.dtype==np.float64
    np.testing.assert_allclose(actual[:2],legacy[:2],rtol=0.,atol=2.e-8)
    np.testing.assert_allclose(actual[2],legacy[2],rtol=0.,atol=1.e-4)


def test_wgs72_uses_standard_proj_helmert_and_spheres_do_not_shift():
    actual=np.asarray(datum_shift(-100.,40.,123.,source='WGS-72',target='WGS-84'))
    legacy=np.array([-99.99984611111111,40.000032742747266,125.3754912624908])
    np.testing.assert_allclose(actual[0],legacy[0],rtol=0.,atol=1.e-12)
    np.testing.assert_allclose(actual[1],legacy[1],rtol=0.,atol=2.e-7)
    np.testing.assert_allclose(actual[2],legacy[2],rtol=0.,atol=3.e-3)
    sphere=np.asarray(datum_shift(-100.,40.,123.,source='NWS-84',target='WGS-84'))
    np.testing.assert_array_equal(sphere,[-100.,40.,123.])


def test_proj_datum_shift_round_trip_preserves_coordinates_and_longitude_branch():
    original=np.array([190.,40.,123.],dtype=np.float64)
    shifted=np.asarray(datum_shift(*original,source='NAS-C',target='WGS-84'))
    restored=np.asarray(datum_shift(*shifted,source='WGS-84',target='NAS-C'))
    assert shifted[0]>180.
    np.testing.assert_allclose(restored[:2],original[:2],rtol=0.,atol=1.e-10)
    np.testing.assert_allclose(restored[2],original[2],rtol=0.,atol=1.e-5)


def test_projection_helpers_apply_coordlib_shift_before_projection():
    grid=GridConfig(1,1,1.,1.,projection='LCC',lat0=40.,lon0=-100.,
                    lat1=30.,lat2=60.,datum='NAR-C')
    longitude=np.array([-100.,-99.]); latitude=np.array([40.,41.])
    shifted_lon,shifted_lat,_=datum_shift(longitude,latitude,source='NAS-C',target='NAR-C')
    expected=project_geographic(shifted_lon,shifted_lat,grid,'NAR-C')
    actual=project_geographic(longitude,latitude,grid,'NAS-C')
    np.testing.assert_array_equal(actual,expected)
    restored=unproject_geographic(*actual,grid,'NAS-C')
    # GLOBE exposes projected coordinates through a REAL*4 kilometre value,
    # so an inverse round trip is limited by that legacy interface precision.
    np.testing.assert_allclose(restored,[longitude,latitude],rtol=0.,atol=2.e-7)


def test_projected_grid_transform_uses_source_and_target_datums():
    source=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=14,datum='NAS-C')
    target=GridConfig(1,1,1.,1.,projection='LCC',lat0=40.,lon0=-100.,
                      lat1=30.,lat2=60.,datum='NAR-C')
    x,y=project_geographic(np.array([-100.]),np.array([40.]),source,'NAS-C')
    actual=transform_grid(x,y,source,target)
    expected=project_geographic(np.array([-100.]),np.array([40.]),target,'NAS-C')
    np.testing.assert_allclose(actual,expected,rtol=0.,atol=2.e-8)


def test_identical_datum_shift_is_exact_and_esr_r_alias_is_supported():
    values=np.array([-100.,40.,123.])
    actual=np.asarray(datum_shift(*values,source='NAS-C',target='NAS-C'))
    np.testing.assert_array_equal(actual,values)
    assert datum_definition('ESR-R')==datum_definition('ESR-S')
