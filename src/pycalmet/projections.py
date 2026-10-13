"""Shared CALPUFF map-projection definitions backed by PROJ/pyproj."""
from __future__ import annotations

import ctypes
import math
import numpy as np

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS,Proj,Transformer

from .config import GridConfig
from .datums import datum_definition,datum_shift,normalize_datum


_PROCESS_LIBRARY=ctypes.CDLL(None)


def _real_intrinsic(name,value):
    """Call the C float libm operation used by GNU default REAL intrinsics."""
    try:
        function=getattr(_PROCESS_LIBRARY,name)
    except AttributeError:
        return np.float32(getattr(math,name[:-1])(float(np.float32(value))))
    function.argtypes=(ctypes.c_float,)
    function.restype=ctypes.c_float
    return np.float32(function(ctypes.c_float(value)))


def _earth(datum: str) -> str:
    normalized=normalize_datum(datum)
    # ESR-R is the historical CALPUFF spelling for the ESR-S sphere.
    definition=datum_definition('ESR-S' if normalized=='ESR-R' else normalized)
    if definition.inv_f>=1.e20:
        return f'+R={definition.radius:.15g}'
    return (f'+a={definition.radius:.15g} +rf={definition.inv_f:.15g} '
            f'+towgs84={definition.dx:g},{definition.dy:g},{definition.dz:g}')


def geographic_crs(datum: str) -> CRS:
    """Geographic CRS for a CALPUFF datum/region identifier."""
    return CRS.from_proj4(f'+proj=longlat {_earth(datum)} +no_defs')


def grid_crs(grid: GridConfig, datum: str | None=None, *, units: str='m') -> CRS:
    """Construct the CRS represented by CALPUFF grid-control parameters."""
    projection=grid.projection.upper(); earth=_earth(grid.datum if datum is None else datum)
    offsets=f'+x_0={grid.false_easting_km*1000:g} +y_0={grid.false_northing_km*1000:g}'
    if projection=='LCC':
        definition=(f'+proj=lcc +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+lat_1={grid.lat1:g} +lat_2={grid.lat2:g} {offsets} {earth}')
    elif projection=='TTM':
        definition=(f'+proj=tmerc +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+k={grid.ttm_scale:g} {offsets} {earth}')
    elif projection=='UTM':
        south=' +south' if grid.hemisphere.upper()=='S' else ''
        definition=f'+proj=utm +zone={grid.utm_zone:d}{south} {earth}'
    elif projection=='PS':
        pole=90. if grid.lat0>=0. else -90.
        definition=(f'+proj=stere +lat_0={pole:g} +lat_ts={grid.lat1:g} +lon_0={grid.lon0:g} '
                    f'{offsets} {earth}')
    elif projection=='EM':
        definition=f'+proj=merc +lat_ts=0 +lon_0={grid.lon0:g} {offsets} {earth}'
    elif projection=='LAZA':
        definition=f'+proj=laea +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} {offsets} {earth}'
    elif projection=='ACEA':
        definition=(f'+proj=aea +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+lat_1={grid.lat1:g} +lat_2={grid.lat2:g} {offsets} {earth}')
    else:
        raise NotImplementedError(f'CALPUFF projection {projection!r} is not implemented')
    if units not in {'m','km'}: raise ValueError("projection units must be 'm' or 'km'")
    return CRS.from_proj4(definition+f' +units={units} +no_defs')


def _project_gctp_lcc(longitude_degrees,latitude_degrees,
                      target_grid: GridConfig,*,units='m'):
    """GCTP 2.0.2 Lambert conformal forward projection."""
    if target_grid.projection.upper()!='LCC':
        raise ValueError('target_grid must use the LCC projection')
    if units not in {'m','km'}:
        raise ValueError("projection units must be 'm' or 'km'")
    longitude_degrees,latitude_degrees=np.broadcast_arrays(
        np.asarray(longitude_degrees,dtype=np.float64),
        np.asarray(latitude_degrees,dtype=np.float64))
    shape=longitude_degrees.shape
    radians_per_degree=0.4848136811095359e-5*3600.0
    longitude=longitude_degrees*0.01745329251994330
    latitude=latitude_degrees*0.01745329251994330
    definition=datum_definition(target_grid.datum)
    semimajor=definition.radius
    if normalize_datum(target_grid.datum)=='WGS-84':
        semiminor=6356752.314245
        eccentricity=np.sqrt(1.0-(semiminor/semimajor)**2)
    else:
        eccentricity=np.sqrt(2.0/definition.inv_f-1.0/definition.inv_f**2)
    first=target_grid.lat1*radians_per_degree
    second=target_grid.lat2*radians_per_degree
    origin_latitude=target_grid.lat0*radians_per_degree
    origin_longitude=target_grid.lon0*radians_per_degree

    def ms(phi):
        sine=np.sin(phi)
        return np.cos(phi)/np.sqrt(1.0-eccentricity**2*sine**2)

    def ts(phi):
        sine=np.sin(phi)
        con=((1.0-eccentricity*sine)/(1.0+eccentricity*sine))**(
            0.5*eccentricity)
        return np.tan(0.5*(1.5707963267948966-phi))/con

    ms1,ts1=ms(first),ts(first)
    if abs(first-second)>=1.0e-10:
        cone=np.log(ms1/ms(second))/np.log(ts1/ts(second))
    else:
        cone=np.sin(first)
    factor=ms1/(cone*ts1**cone)
    rho0=semimajor*factor*ts(origin_latitude)**cone
    rho=semimajor*factor*ts(latitude)**cone
    theta=cone*((longitude-origin_longitude+np.pi)%(2.0*np.pi)-np.pi)
    output_x=target_grid.false_easting_km*1000.0+rho*np.sin(theta)
    output_y=target_grid.false_northing_km*1000.0+rho0-rho*np.cos(theta)
    # XMULTO is an implicitly typed REAL*4 in COORDS and GLOBE returns grid
    # coordinates in km.  Terrain readers subsequently multiply that REAL*4
    # result by 1000 when passing metres to SAVEELEV.  Keep both roundings;
    # casting projected metres directly to REAL*4 can differ by one ULP.
    output_x=(output_x*float(np.float32(0.001))).astype(np.float32)
    output_y=(output_y*float(np.float32(0.001))).astype(np.float32)
    if units=='m':
        output_x=(output_x*np.float32(1000.0)).astype(np.float32)
        output_y=(output_y*np.float32(1000.0)).astype(np.float32)
    if not shape: return float(output_x),float(output_y)
    return output_x,output_y


def _unproject_gctp_lcc(x,y,source_grid: GridConfig,*,units='m'):
    """GCTP 2.0.2 Lambert conformal inverse projection."""
    if source_grid.projection.upper()!='LCC':
        raise ValueError('source_grid must use the LCC projection')
    if units not in {'m','km'}:
        raise ValueError("projection units must be 'm' or 'km'")
    x,y=np.broadcast_arrays(np.asarray(x,dtype=np.float64),
                            np.asarray(y,dtype=np.float64))
    shape=x.shape
    scale=1000.0 if units=='km' else 1.0
    x=x*scale-source_grid.false_easting_km*1000.0
    y=y*scale-source_grid.false_northing_km*1000.0
    radians_per_degree=0.4848136811095359e-5*3600.0
    definition=datum_definition(source_grid.datum)
    semimajor=definition.radius
    if normalize_datum(source_grid.datum)=='WGS-84':
        semiminor=6356752.314245
        eccentricity=np.sqrt(1.0-(semiminor/semimajor)**2)
    else:
        eccentricity=np.sqrt(2.0/definition.inv_f-1.0/definition.inv_f**2)
    first=source_grid.lat1*radians_per_degree
    second=source_grid.lat2*radians_per_degree
    origin_latitude=source_grid.lat0*radians_per_degree
    origin_longitude=source_grid.lon0*radians_per_degree

    def ms(phi):
        sine=np.sin(phi)
        return np.cos(phi)/np.sqrt(1.0-eccentricity**2*sine**2)

    def ts(phi):
        sine=np.sin(phi)
        con=((1.0-eccentricity*sine)/(1.0+eccentricity*sine))**(
            0.5*eccentricity)
        return np.tan(0.5*(1.5707963267948966-phi))/con

    ms1,ts1=ms(first),ts(first)
    if abs(first-second)>=1.0e-10:
        cone=np.log(ms1/ms(second))/np.log(ts1/ts(second))
    else:
        cone=np.sin(first)
    factor=ms1/(cone*ts1**cone)
    rho0=semimajor*factor*ts(origin_latitude)**cone
    rho=np.copysign(np.hypot(x,rho0-y),cone)
    t=(rho/(semimajor*factor))**(1.0/cone)
    latitude=np.full_like(t,1.5707963267948966)-2.0*np.arctan(t)
    for _ in range(15):
        con=eccentricity*np.sin(latitude)
        updated=(np.full_like(t,1.5707963267948966)-2.0*np.arctan(
            t*((1.0-con)/(1.0+con))**(0.5*eccentricity)))
        if np.all(np.abs(updated-latitude)<1.0e-10):
            latitude=updated
            break
        latitude=updated
    longitude=origin_longitude+np.arctan2(x,rho0-y)/cone
    longitude=longitude/radians_per_degree
    latitude=latitude/radians_per_degree
    if not shape: return float(longitude),float(latitude)
    return longitude,latitude


def project_geographic(longitude,latitude,grid: GridConfig,source_datum: str,
                       *,units: str='m'):
    """Project geographic points using COORDLIB's datum-shift algorithm."""
    target_datum=grid.datum
    lon,lat,_=datum_shift(longitude,latitude,source=source_datum,target=target_datum)
    if grid.projection.upper()=='LCC':
        return _project_gctp_lcc(lon,lat,grid,units=units)
    transformer=Transformer.from_crs(geographic_crs(target_datum),
                                     grid_crs(grid,units=units),always_xy=True)
    shape=lon.shape
    x,y=transformer.transform(lon.ravel().tolist(),lat.ravel().tolist())
    x=np.asarray(x,dtype=float).reshape(shape); y=np.asarray(y,dtype=float).reshape(shape)
    return (float(x),float(y)) if not shape else (x,y)


def unproject_geographic(x,y,grid: GridConfig,target_datum: str,*,units: str='m'):
    """Unproject grid points and apply COORDLIB's inverse datum shift."""
    source_datum=grid.datum
    x_array,y_array=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    shape=x_array.shape
    if grid.projection.upper()=='LCC':
        lon,lat=_unproject_gctp_lcc(x_array,y_array,grid,units=units)
        lon=np.asarray(lon,dtype=float); lat=np.asarray(lat,dtype=float)
    else:
        transformer=Transformer.from_crs(grid_crs(grid,units=units),
                                         geographic_crs(source_datum),always_xy=True)
        lon,lat=transformer.transform(x_array.ravel().tolist(),y_array.ravel().tolist())
        lon=np.asarray(lon,dtype=float).reshape(shape); lat=np.asarray(lat,dtype=float).reshape(shape)
    shifted_lon,shifted_lat,_=datum_shift(
        lon,lat,source=source_datum,target=target_datum)
    if not shape: return float(shifted_lon),float(shifted_lat)
    return shifted_lon,shifted_lat


def transform_grid(x,y,source_grid: GridConfig,target_grid: GridConfig,
                   *,units: str='m'):
    """Transform projected points through legacy COORDLIB datum semantics."""
    lon,lat=unproject_geographic(x,y,source_grid,source_grid.datum,units=units)
    return project_geographic(lon,lat,target_grid,source_grid.datum,units=units)


def project_from_crs(x,y,source_crs: CRS,source_datum: str,target_grid: GridConfig,
                     *,units: str='m'):
    """Transform an externally defined projection with COORDLIB datum rules."""
    transformer=Transformer.from_crs(source_crs,geographic_crs(source_datum),always_xy=True)
    x_array,y_array=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    shape=x_array.shape
    lon,lat=transformer.transform(x_array.ravel().tolist(),y_array.ravel().tolist())
    lon=np.asarray(lon,dtype=float).reshape(shape); lat=np.asarray(lat,dtype=float).reshape(shape)
    return project_geographic(lon,lat,target_grid,source_datum,units=units)


def project_glaz_to_lcc(x, y, source_grid: GridConfig, target_grid: GridConfig,
                        *, units: str = 'km'):
    """Vectorized GLAZ-to-LCC transform matching legacy ``COORDLIB``.

    GLAZ rasters are the one CTGPROC input for which a mesh point landing a
    few ULPs either side of a target-cell edge can change the dominant land
    category.  PROJ implements mathematically equivalent LAEA/LCC formulas,
    but does not reproduce GCTP 2.0.2's constants and REAL*4 ``GLOBE``
    boundary.  This transcription deliberately preserves those details.
    """
    if source_grid.projection.upper() != 'LAZA':
        raise ValueError('source_grid must use the LAZA projection')
    if target_grid.projection.upper() != 'LCC':
        raise ValueError('target_grid must use the LCC projection')
    if units not in {'m', 'km'}:
        raise ValueError("projection units must be 'm' or 'km'")

    # GLOBE receives and returns REAL*4 coordinates.  COORDS immediately
    # promotes them, while GCTP carries out all projection work in REAL*8.
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=np.float32),
                                 np.asarray(y, dtype=np.float32))
    shape = xx.shape
    scale = 1000.0 if units == 'km' else 1.0
    xx = xx.astype(np.float64) * scale - source_grid.false_easting_km * 1000.0
    yy = yy.astype(np.float64) * scale - source_grid.false_northing_km * 1000.0

    # PAKRZ0 and UNTFZ0 constants from COORDLIB/GCTP 2.0.2.
    radians_per_degree = 0.4848136811095359e-5 * 3600.0
    lon0 = source_grid.lon0 * radians_per_degree
    lat0 = source_grid.lat0 * radians_per_degree
    radius = datum_definition(source_grid.datum).radius
    distance = np.sqrt(xx * xx + yy * yy)
    z = 2.0 * np.arcsin(np.minimum(1.0, distance / (2.0 * radius)))
    sin_z, cos_z = np.sin(z), np.cos(z)
    sin_lat0, cos_lat0 = np.sin(lat0), np.cos(lat0)
    latitude = np.empty_like(distance)
    longitude = np.full_like(distance, lon0)
    away = distance > 1.0e-10
    latitude[away] = np.arcsin(
        cos_z[away] * sin_lat0
        + yy[away] * sin_z[away] * cos_lat0 / distance[away]
    )
    latitude[~away] = lat0
    denominator = distance * (cos_z - sin_lat0 * np.sin(latitude))
    longitude[away] = lon0 + np.arctan2(
        xx[away] * sin_z[away] * cos_lat0, denominator[away]
    )

    # GTPZ0 returns intermediate geographic coordinates in degrees and the
    # following GTPZ0 call converts them back to radians.  Its two tabulated
    # factors are intentionally not replaced by an algebraic cancellation.
    longitude_degrees = longitude * 57.29577951308231
    latitude_degrees = latitude * 57.29577951308231

    # DAT2DAT skips the sphere-to-reference half of a shift, but still
    # performs the reference-to-target-ellipsoid half.
    if normalize_datum(source_grid.datum) != normalize_datum(target_grid.datum):
        longitude_degrees, latitude_degrees, _ = datum_shift(
            longitude_degrees, latitude_degrees,
            source=source_grid.datum, target=target_grid.datum)
    return _project_gctp_lcc(
        longitude_degrees,latitude_degrees,target_grid,units=units)


def unproject_to_crs(x,y,source_grid: GridConfig,target_crs: CRS,target_datum: str,
                     *,units: str='m'):
    """Transform a CALPUFF grid to an external projection via COORDLIB."""
    lon,lat=unproject_geographic(x,y,source_grid,target_datum,units=units)
    transformer=Transformer.from_crs(geographic_crs(target_datum),target_crs,always_xy=True)
    lon_array,lat_array=np.broadcast_arrays(np.asarray(lon,dtype=float),np.asarray(lat,dtype=float))
    shape=lon_array.shape
    px,py=transformer.transform(lon_array.ravel().tolist(),lat_array.ravel().tolist())
    px=np.asarray(px,dtype=float).reshape(shape); py=np.asarray(py,dtype=float).reshape(shape)
    if not shape: return float(px),float(py)
    return px,py


def grid_convergence(grid: GridConfig, longitude, latitude):
    """Meridian convergence in degrees, positive east of the central line."""
    lon=np.asarray(longitude,dtype=float); lat=np.asarray(latitude,dtype=float)
    if grid.projection.upper()=='LCC':
        # READCF stores D2R and CONEC in default REAL.  Wind vectors nearly
        # parallel to the terrain gradient can choose the opposite FRADJ
        # tangent if this is evaluated with modern double-precision constants.
        d2r=np.float32(.0174533)
        first=np.float32(abs(np.float32(grid.lat1)))
        second=np.float32(abs(np.float32(grid.lat2)))
        first_cosine=_real_intrinsic('cosf',np.float32(first*d2r))
        second_cosine=_real_intrinsic('cosf',np.float32(second*d2r))
        numerator=_real_intrinsic('logf',np.float32(first_cosine/second_cosine))
        first_tangent=_real_intrinsic(
            'tanf',np.float32(d2r*np.float32(45.-first/np.float32(2.))))
        second_tangent=_real_intrinsic(
            'tanf',np.float32(d2r*np.float32(45.-second/np.float32(2.))))
        denominator=_real_intrinsic('logf',np.float32(first_tangent/second_tangent))
        cone=np.float32(numerator/denominator)
        return np.asarray(cone*(np.asarray(lon,dtype=np.float32)-
                               np.float32(grid.lon0)),dtype=np.float32)
    return np.asarray(Proj(grid_crs(grid)).get_factors(lon,lat).meridian_convergence)
