"""Shared CALPUFF map-projection definitions backed by PROJ/pyproj."""
from __future__ import annotations

import numpy as np

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS,Proj,Transformer

from .config import GridConfig
from .datums import (datum_definition,datum_proj4_parameters,datum_shift,
                     normalize_datum)


def _earth(datum: str) -> str:
    normalized=normalize_datum(datum)
    # ESR-R is the historical CALPUFF spelling for the ESR-S sphere.
    return datum_proj4_parameters('ESR-S' if normalized=='ESR-R' else normalized)


def geographic_crs(datum: str) -> CRS:
    """Geographic CRS for a CALPUFF datum/region identifier."""
    return CRS.from_proj4(f'+proj=longlat {_earth(datum)} +no_defs')


def grid_crs(grid: GridConfig, datum: str | None=None, *, units: str='m') -> CRS:
    """Construct the CRS represented by CALPUFF grid-control parameters."""
    datum_name=grid.datum if datum is None else datum
    projection=grid.projection.upper(); earth=_earth(datum_name)
    offsets=f'+x_0={grid.false_easting_km*1000:g} +y_0={grid.false_northing_km*1000:g}'
    if projection=='LCC':
        definition=(f'+proj=lcc +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+lat_1={grid.lat1:g} +lat_2={grid.lat2:g} {offsets} {earth}')
    elif projection=='TTM':
        definition=(f'+proj=tmerc +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+k={grid.ttm_scale:g} {offsets} {earth}')
    elif projection=='UTM':
        normalized=normalize_datum(datum_name)
        ellipsoid=datum_definition('ESR-S' if normalized=='ESR-R' else normalized)
        if ellipsoid.inv_f>=1.e20:
            # PROJ rejects ``+proj=utm`` with a sphere even though CALPUFF's
            # COORDLIB supports the combination.  UTM is a fixed transverse
            # Mercator definition, so spell it out for spherical datums.
            central_meridian=grid.utm_zone*6-183
            false_northing=10000000 if grid.hemisphere.upper()=='S' else 0
            definition=(f'+proj=tmerc +lat_0=0 +lon_0={central_meridian:g} '
                        f'+k=0.9996 +x_0=500000 +y_0={false_northing:g} {earth}')
        else:
            south=' +south' if grid.hemisphere.upper()=='S' else ''
            definition=f'+proj=utm +zone={grid.utm_zone:d}{south} {earth}'
    elif projection=='PS':
        pole=90. if grid.lat0>=0. else -90.
        # COORDLIB reports PS coordinates relative to RLON0/RLAT0 and does
        # not apply the user FEAST/FNORTH values to this projection.
        base=(f'+proj=stere +lat_0={pole:g} +lat_ts={grid.lat1:g} '
              f'+lon_0={grid.lon0:g} +x_0=0 +y_0=0 {earth}')
        origin_x,origin_y=Transformer.from_crs(
            geographic_crs(datum_name),
            CRS.from_proj4(base+' +units=m +no_defs'),always_xy=True,
        ).transform(grid.lon0,grid.lat0)
        definition=(f'+proj=stere +lat_0={pole:g} +lat_ts={grid.lat1:g} '
                    f'+lon_0={grid.lon0:g} +x_0={-origin_x:.15g} '
                    f'+y_0={-origin_y:.15g} {earth}')
    elif projection=='EM':
        # CALPUFF fixes the EM latitude of origin at the equator and ignores
        # user FEAST/FNORTH values.
        definition=f'+proj=merc +lat_ts=0 +lon_0={grid.lon0:g} {earth}'
    elif projection=='LAZA':
        definition=f'+proj=laea +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} {offsets} {earth}'
    elif projection=='ACEA':
        definition=(f'+proj=aea +lat_0={grid.lat0:g} +lon_0={grid.lon0:g} '
                    f'+lat_1={grid.lat1:g} +lat_2={grid.lat2:g} {offsets} {earth}')
    else:
        raise NotImplementedError(f'CALPUFF projection {projection!r} is not implemented')
    if units not in {'m','km'}: raise ValueError("projection units must be 'm' or 'km'")
    return CRS.from_proj4(definition+f' +units={units} +no_defs')


def project_geographic(longitude,latitude,grid: GridConfig,source_datum: str,
                       *,units: str='m'):
    """Project geographic points with pyproj."""
    target_datum=grid.datum
    lon,lat,_=datum_shift(longitude,latitude,source=source_datum,target=target_datum)
    transformer=Transformer.from_crs(geographic_crs(target_datum),
                                     grid_crs(grid,units=units),always_xy=True)
    shape=lon.shape
    x,y=transformer.transform(lon.ravel().tolist(),lat.ravel().tolist())
    x=np.asarray(x,dtype=float).reshape(shape); y=np.asarray(y,dtype=float).reshape(shape)
    return (float(x),float(y)) if not shape else (x,y)


def unproject_geographic(x,y,grid: GridConfig,target_datum: str,*,units: str='m'):
    """Unproject grid points with pyproj and apply the requested datum shift."""
    source_datum=grid.datum
    x_array,y_array=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    shape=x_array.shape
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
    """Transform projected points between grids."""
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
    """Transform GLAZ coordinates to the target LCC grid with pyproj."""
    return transform_grid(x, y, source_grid, target_grid, units=units)


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
    return np.asarray(Proj(grid_crs(grid)).get_factors(lon,lat).meridian_convergence)
