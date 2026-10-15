"""Shared CALPUFF map-projection definitions backed by PROJ/pyproj."""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS,Proj,Transformer

from .config import GridConfig
from .datums import (datum_definition,datum_proj4_parameters,datum_shift,
                     normalize_datum)


_ProjectedCoordinates: TypeAlias = tuple[float, float] | tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]


def _earth(datum: str) -> str:
    """Return the PROJ earth-model parameters for a CALPUFF datum.

    Args:
        datum: CALPUFF datum or spherical-earth identifier.

    Returns:
        PROJ ellipsoid, sphere, and datum-shift parameter fragment.

    Raises:
        ValueError: If ``datum`` is unknown.
    """
    normalized=normalize_datum(datum)
    # ESR-R is the historical CALPUFF spelling for the ESR-S sphere.
    return datum_proj4_parameters('ESR-S' if normalized=='ESR-R' else normalized)


def geographic_crs(datum: str) -> CRS:
    """Construct the longitude/latitude CRS for a CALPUFF datum.

    Args:
        datum: CALPUFF datum or spherical-earth identifier.

    Returns:
        Longitude-first pyproj geographic CRS with angular coordinates in
        degrees.

    Raises:
        ValueError: If ``datum`` is unknown.
    """
    return CRS.from_proj4(f'+proj=longlat {_earth(datum)} +no_defs')


def grid_crs(grid: GridConfig, datum: str | None=None, *, units: str='m') -> CRS:
    """Construct a projected CRS from CALPUFF grid controls.

    Args:
        grid: Grid geometry and projection parameters. Supported projections
            are LCC, TTM, UTM, PS, EM, LAZA, and ACEA.
        datum: Optional datum override. ``None`` uses ``grid.datum``.
        units: Linear output units, either ``"m"`` or ``"km"``.

    Returns:
        pyproj CRS matching CALPUFF's projection origin, false offsets,
        ellipsoid, datum, and requested linear units.

    Raises:
        ValueError: If ``datum`` is unknown or ``units`` is not ``"m"`` or
            ``"km"``.
        NotImplementedError: If ``grid.projection`` is unsupported.
    """
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


def project_geographic(
    longitude: npt.ArrayLike,
    latitude: npt.ArrayLike,
    grid: GridConfig,
    source_datum: str,
    *,
    units: str = 'm',
) -> _ProjectedCoordinates:
    """Project geographic coordinates onto a CALPUFF grid.

    Args:
        longitude: Scalar or array-like longitude in degrees east.
        latitude: Scalar or array-like latitude in degrees north.
        grid: Target CALPUFF grid and datum definition.
        source_datum: Datum of ``longitude`` and ``latitude``.
        units: Linear units of the returned coordinates, ``"m"`` or
            ``"km"``.

    Returns:
        Pair ``(x, y)`` in the requested units. Scalar inputs return Python
        floats; broadcast array inputs return ``float64`` arrays with their
        broadcast shape, in the target grid CRS.

    Raises:
        ValueError: If the inputs cannot be broadcast, a datum is unknown, or
            ``units`` is invalid.
        NotImplementedError: If the target projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
    target_datum=grid.datum
    lon,lat,_=datum_shift(longitude,latitude,source=source_datum,target=target_datum)
    transformer=Transformer.from_crs(geographic_crs(target_datum),
                                     grid_crs(grid,units=units),always_xy=True)
    shape=lon.shape
    x,y=transformer.transform(lon.ravel().tolist(),lat.ravel().tolist())
    x=np.asarray(x,dtype=float).reshape(shape); y=np.asarray(y,dtype=float).reshape(shape)
    return (float(x),float(y)) if not shape else (x,y)


def unproject_geographic(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    grid: GridConfig,
    target_datum: str,
    *,
    units: str = 'm',
) -> _ProjectedCoordinates:
    """Convert CALPUFF grid coordinates to geographic coordinates.

    Args:
        x: Scalar or array-like projected easting in ``units``.
        y: Scalar or array-like projected northing in ``units``.
        grid: Source CALPUFF grid and datum definition.
        target_datum: Datum requested for the geographic result.
        units: Linear units of ``x`` and ``y``, ``"m"`` or ``"km"``.

    Returns:
        Pair ``(longitude, latitude)`` in angular degrees on
        ``target_datum``. Scalar inputs return Python floats; broadcast array
        inputs return ``float64`` arrays with their broadcast shape.

    Raises:
        ValueError: If inputs cannot be broadcast, a datum is unknown, or
            ``units`` is invalid.
        NotImplementedError: If the source projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
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


def transform_grid(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    source_grid: GridConfig,
    target_grid: GridConfig,
    *,
    units: str = 'm',
) -> _ProjectedCoordinates:
    """Transform projected coordinates between two CALPUFF grids.

    Args:
        x: Scalar or array-like source-grid easting in ``units``.
        y: Scalar or array-like source-grid northing in ``units``.
        source_grid: Projection and datum of the input coordinates.
        target_grid: Projection and datum of the returned coordinates.
        units: Shared linear units for input and output, ``"m"`` or ``"km"``.

    Returns:
        Pair ``(x, y)`` in the target grid CRS and requested units. Scalar
        inputs return floats; broadcast array inputs return ``float64``
        arrays with their broadcast shape.

    Raises:
        ValueError: If inputs, units, or datum identifiers are invalid.
        NotImplementedError: If either projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
    lon,lat=unproject_geographic(x,y,source_grid,source_grid.datum,units=units)
    return project_geographic(lon,lat,target_grid,source_grid.datum,units=units)


def project_from_crs(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    source_crs: CRS,
    source_datum: str,
    target_grid: GridConfig,
    *,
    units: str = 'm',
) -> _ProjectedCoordinates:
    """Project coordinates from an external CRS to a CALPUFF grid.

    Args:
        x: Scalar or array-like coordinate on ``source_crs``'s x axis.
        y: Scalar or array-like coordinate on ``source_crs``'s y axis.
        source_crs: pyproj CRS describing the input coordinate axes and units.
        source_datum: CALPUFF datum represented by ``source_crs``.
        target_grid: Target CALPUFF grid and datum definition.
        units: Linear units for the returned grid coordinates, ``"m"`` or
            ``"km"``.

    Returns:
        Pair of target-grid coordinates. Scalar inputs return floats;
        broadcast array inputs return ``float64`` arrays with their broadcast
        shape and the requested linear units.

    Raises:
        ValueError: If inputs cannot be broadcast, a datum is unknown, or
            ``units`` is invalid.
        NotImplementedError: If the target projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
    transformer=Transformer.from_crs(source_crs,geographic_crs(source_datum),always_xy=True)
    x_array,y_array=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    shape=x_array.shape
    lon,lat=transformer.transform(x_array.ravel().tolist(),y_array.ravel().tolist())
    lon=np.asarray(lon,dtype=float).reshape(shape); lat=np.asarray(lat,dtype=float).reshape(shape)
    return project_geographic(lon,lat,target_grid,source_datum,units=units)


def project_glaz_to_lcc(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    source_grid: GridConfig,
    target_grid: GridConfig,
    *,
    units: str = 'km',
) -> _ProjectedCoordinates:
    """Transform legacy GLAZ coordinates to a target LCC grid.

    Args:
        x: Scalar or array-like GLAZ-grid easting in ``units``.
        y: Scalar or array-like GLAZ-grid northing in ``units``.
        source_grid: GLAZ source-grid projection and datum.
        target_grid: Target Lambert conformal conic grid definition.
        units: Shared linear units for input and output, ``"m"`` or ``"km"``.

    Returns:
        Pair of target-LCC coordinates. Scalar inputs return floats;
        broadcast array inputs return ``float64`` arrays with their broadcast
        shape and the requested units.

    Raises:
        ValueError: If inputs, units, or datum identifiers are invalid.
        NotImplementedError: If a grid projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
    return transform_grid(x, y, source_grid, target_grid, units=units)


def unproject_to_crs(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    source_grid: GridConfig,
    target_crs: CRS,
    target_datum: str,
    *,
    units: str = 'm',
) -> _ProjectedCoordinates:
    """Transform CALPUFF grid coordinates to an external projected CRS.

    Args:
        x: Scalar or array-like source-grid easting in ``units``.
        y: Scalar or array-like source-grid northing in ``units``.
        source_grid: Source CALPUFF grid and datum definition.
        target_crs: pyproj CRS defining the result axes and their units.
        target_datum: CALPUFF datum represented by ``target_crs``.
        units: Linear units of source ``x`` and ``y``, ``"m"`` or ``"km"``.

    Returns:
        Pair of coordinates in ``target_crs`` units. Scalar inputs return
        floats; broadcast array inputs return ``float64`` arrays with their
        broadcast shape.

    Raises:
        ValueError: If inputs cannot be broadcast, a datum is unknown, or
            ``units`` is invalid.
        NotImplementedError: If the source projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot transform a coordinate.
    """
    lon,lat=unproject_geographic(x,y,source_grid,target_datum,units=units)
    transformer=Transformer.from_crs(geographic_crs(target_datum),target_crs,always_xy=True)
    lon_array,lat_array=np.broadcast_arrays(np.asarray(lon,dtype=float),np.asarray(lat,dtype=float))
    shape=lon_array.shape
    px,py=transformer.transform(lon_array.ravel().tolist(),lat_array.ravel().tolist())
    px=np.asarray(px,dtype=float).reshape(shape); py=np.asarray(py,dtype=float).reshape(shape)
    if not shape: return float(px),float(py)
    return px,py


def grid_convergence(
    grid: GridConfig,
    longitude: npt.ArrayLike,
    latitude: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate meridian convergence for a CALPUFF grid.

    Args:
        grid: Grid projection and datum definition.
        longitude: Scalar or array-like longitude in degrees east on the grid
            datum.
        latitude: Scalar or array-like latitude in degrees north on the grid
            datum.

    Returns:
        ``float64`` NumPy array of meridian-convergence angles in degrees,
        positive east of the projection's central line, with the broadcast
        coordinate shape.

    Raises:
        ValueError: If coordinate shapes are incompatible or the grid datum
            is invalid.
        NotImplementedError: If the grid projection is unsupported.
        pyproj.exceptions.ProjError: If PROJ cannot evaluate the factors.
    """
    lon=np.asarray(longitude,dtype=float); lat=np.asarray(latitude,dtype=float)
    return np.asarray(Proj(grid_crs(grid)).get_factors(lon,lat).meridian_convergence)
