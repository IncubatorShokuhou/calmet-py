"""Small, independently testable numerical kernels for the CALMET model.

Array axes follow the Python convention ``(..., y, x)``.  These routines are
kept free of model-global state so the main solver can compose them directly.
"""
from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coare import WarmLayerState


# Physical fields and model state use single precision throughout the modern
# solver.  Double precision is reserved for coordinates, reductions, and
# nonlinear convergence work where single-precision cancellation can change
# a grid-cell classification or prevent convergence.
_FloatT = TypeVar("_FloatT", np.float32, np.float64)

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
FloatingArray: TypeAlias = Float32Array | Float64Array
FloatingDType: TypeAlias = (
    type[np.float32] | np.dtype[np.float32]
    | type[np.float64] | np.dtype[np.float64]
)
FloatPair: TypeAlias = tuple[Float32Array, Float32Array]
FloatTriple: TypeAlias = tuple[Float32Array, Float32Array, Float32Array]
FloatQuad: TypeAlias = tuple[
    Float32Array, Float32Array, Float32Array, Float32Array
]
FloatSix: TypeAlias = tuple[
    Float32Array, Float32Array, Float32Array, Float32Array, Float32Array,
    Float32Array,
]
FloatEight: TypeAlias = tuple[
    Float32Array, Float32Array, Float32Array, Float32Array, Float32Array,
    Float32Array, Float32Array, Float32Array,
]
FloatNine: TypeAlias = tuple[
    Float32Array, Float32Array, Float32Array, Float32Array, Float32Array,
    Float32Array, Float32Array, Float32Array, Float32Array,
]
CoareStationResult: TypeAlias = tuple[
    Float32Array, Float32Array, Float32Array, Float32Array, Float32Array,
    Float32Array, Float32Array, Float32Array, Float32Array,
    WarmLayerState | None,
]
BoolArray: TypeAlias = NDArray[np.bool_]
Int8Array: TypeAlias = NDArray[np.int8]
Int32Array: TypeAlias = NDArray[np.int32]
IntArray: TypeAlias = NDArray[np.int_]
WarmStateArray: TypeAlias = Float64Array | BoolArray
WaterRange: TypeAlias = tuple[int, int]
GridBoundaries: TypeAlias = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
LakeBreezeRegion: TypeAlias = tuple[
    float, float, float, float, float, float, float, float, Sequence[int]
]
MixingOption: TypeAlias = ArrayLike | float | int | bool | None

MODEL_DTYPE: type[np.float32] = np.float32
ACCUMULATION_DTYPE: type[np.float64] = np.float64
COORDINATE_DTYPE: type[np.float64] = np.float64


def _model_array(value: ArrayLike) -> Float32Array:
    """Convert a numerical result to the model-state precision.

    Args:
        value: Scalar or array-like numerical values in any broadcast shape.

    Returns:
        A NumPy float32 array with the same shape as ``value``. Existing
        float32 arrays may be returned without a copy.
    """
    return np.asarray(value, dtype=MODEL_DTYPE)


def _psiud(
    z_over_l: ArrayLike,
    *,
    dtype: FloatingDType = MODEL_DTYPE,
) -> FloatingArray:
    """Evaluate CALMET's COARE/Dyer momentum stability function.

    Args:
        z_over_l: Scalar or array-like dimensionless height-to-Monin-Obukhov-
            length ratio.
        dtype: NumPy floating dtype used for the calculation and result.

    Returns:
        Array with the shape of ``z_over_l`` and the requested floating dtype;
        values are dimensionless momentum stability corrections.
    """
    return psi_momentum_dyer(z_over_l,dtype=dtype)


def _psiuc(
    height: ArrayLike,
    monin_obukhov: ArrayLike,
    fit: int = 16,
    *,
    dtype: FloatingDType = MODEL_DTYPE,
) -> FloatingArray:
    """Evaluate the PSIUC momentum correction used when ``IPSIFCN=1``.

    Args:
        height: Evaluation height in metres, scalar or array-like.
        monin_obukhov: Monin-Obukhov length in metres, broadcast-compatible
            with ``height``.
        fit: Empirical unstable-fit coefficient; CALMET supports 15 or 16.
        dtype: NumPy floating dtype used for calculation and output.

    Returns:
        Dimensionless array with the broadcast input shape and requested dtype.

    Raises:
        ValueError: If ``fit`` is neither 15 nor 16.
    """
    if fit not in {15,16}: raise ValueError('PSIUC fit must be 15 or 16')
    work_dtype=np.dtype(dtype).type
    height,length=np.broadcast_arrays(np.asarray(height,dtype=work_dtype),
                                      np.asarray(monin_obukhov,dtype=work_dtype))
    output=np.empty(height.shape,dtype=work_dtype)
    stable=length>0.
    if stable.any():
        ratio=-.29*height/length
        output[stable]=-17.*(1.-np.exp(ratio[stable]))
    unstable=~stable
    if unstable.any():
        ratio=fit*height/length
        x=np.power(1.-ratio,.25)
        first=np.log(.5*(1.+x))
        second=np.log(.5*(1.+x*x))
        output[unstable]=(2.*first[unstable]+second[unstable]-
                          2.*np.arctan(x[unstable])+np.pi/2.)
    return cast(FloatingArray, output)


def similarity_wind_profiles(u: ArrayLike,v: ArrayLike,anemometer_height_m: ArrayLike,z_centers_m: ArrayLike,z_faces_m: ArrayLike,
        roughness_m: ArrayLike,monin_obukhov_m: ArrayLike,mixing_height_m: ArrayLike,station_latitude: ArrayLike,*,
        ipsifcn: int=0,minimum_mixing_height_m: float=50.) -> tuple[Float32Array, Float32Array]:
    """Construct vectorized CALMET SIMILT station wind profiles.

    Args:
        u: One-dimensional station eastward winds in m s-1.
        v: One-dimensional station northward winds in m s-1, matching ``u``.
        anemometer_height_m: Wind measurement heights in metres by station.
        z_centers_m: One-dimensional CALMET layer-centre heights in metres.
        z_faces_m: One-dimensional layer-face heights in metres with one more
            element than ``z_centers_m``.
        roughness_m: Surface roughness length in metres by station.
        monin_obukhov_m: Monin-Obukhov length in metres by station.
        mixing_height_m: Mixing height in metres by station.
        station_latitude: Station latitude in degrees north, used to select
            the hemisphere-dependent wind turning direction.
        ipsifcn: Momentum stability function selector: 0 for Dyer/COARE or 1
            for PSIUC.
        minimum_mixing_height_m: Minimum profile depth in metres.

    Returns:
        Pair ``(u_profile, v_profile)`` of float32 arrays with shape
        ``(level, station)`` in m s-1. Layers whose upper face is above the
        station mixing-height limit retain the missing value 9999.

    Raises:
        ValueError: If station arrays or vertical coordinates have
            inconsistent shapes, or ``ipsifcn`` is unsupported.
    """
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    anem=np.asarray(anemometer_height_m,dtype=MODEL_DTYPE)
    roughness=np.asarray(roughness_m,dtype=MODEL_DTYPE)
    ell=np.asarray(monin_obukhov_m,dtype=MODEL_DTYPE)
    mixing=np.asarray(mixing_height_m,dtype=MODEL_DTYPE)
    latitude=np.asarray(station_latitude,dtype=MODEL_DTYPE)
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    if any(value.shape!=u.shape for value in (v,anem,roughness,ell,mixing,latitude)):
        raise ValueError('SIMILT station inputs must have identical shapes')
    if centers.ndim!=1 or faces.shape!=(centers.size+1,):
        raise ValueError('SIMILT vertical coordinates are inconsistent')
    if ipsifcn not in {0,1}: raise ValueError('IPSIFCN must be zero or one')
    adjusted=ell.copy(); short=np.abs(adjusted)<5.*roughness
    adjusted[short]=np.where(adjusted[short]<0.,
                             -5.*roughness[short],5.*roughness[short])
    inverse=np.where(np.abs(adjusted)>=10000.,0.,1./adjusted)
    speed=np.sqrt(u*u+v*v)
    angle=270.-np.rad2deg(np.arctan2(v,u))
    direction=np.fmod(angle,360.)
    direction=np.where(direction==0.,360.,direction)
    log_anem=np.log(anem/roughness)
    neutral=np.abs(inverse)<.00005
    psi_anem=(_psiud(anem/adjusted)
              if ipsifcn==0 else _psiuc(anem,adjusted,16))
    psi_anem=np.where(neutral,0.,psi_anem)
    inverse_table=np.array([-.03333333,-.01,-.0027027,0.,.002857,.007692,
                            .0166667,.05,.1111111],dtype=MODEL_DTYPE)
    angle_table=np.array([12.,10.,9.,12.,18.,28.,35.,38.,39.],dtype=MODEL_DTYPE)
    indices=np.searchsorted(inverse_table,inverse,side='left')
    turn=np.full(u.shape,angle_table[-1],dtype=MODEL_DTYPE)
    low=indices==0; turn[low]=angle_table[0]
    middle=(indices>0)&(indices<inverse_table.size); lower=indices[middle]-1; upper=indices[middle]
    ratio=((inverse[middle]-inverse_table[lower])/
           (inverse_table[upper]-inverse_table[lower]))
    turn[middle]=angle_table[lower]+(angle_table[upper]-angle_table[lower])*ratio
    turn=np.where(neutral,12.,turn)
    decay_anem=np.exp(-anem/200.)
    at_anem=turn*1.58*(1.-decay_anem)
    output_u=np.full((centers.size,u.size),9999.,dtype=MODEL_DTYPE)
    output_v=np.full_like(output_u,9999.)
    limit=np.maximum(mixing+.001,minimum_mixing_height_m)
    for level,center in enumerate(centers):
        active=faces[level+1]<=limit
        if not active.any(): continue
        log_level=np.log(center/roughness)
        psi=(_psiud(center/adjusted)
             if ipsifcn==0 else _psiuc(np.full_like(adjusted,center),adjusted,16))
        psi=np.where(neutral,0.,psi)
        magnitude=speed*(log_level-psi)/(log_anem-psi_anem)
        decay=np.exp(np.full(u.shape,-center/200.,dtype=MODEL_DTYPE))
        shear=turn*1.58*(1.-decay)
        north=direction+shear-at_anem
        north=np.where(north>360.,north-360.,north)
        south=direction-shear+at_anem
        south=np.where(south<=0.,360.+south,south)
        wd=np.where(latitude>0.,north,south)
        radians=np.deg2rad(wd)
        output_u[level,active]=-magnitude[active]*np.sin(radians[active])
        output_v[level,active]=-magnitude[active]*np.cos(radians[active])
    return output_u,output_v

def cloud_liquid_water_profile(
        cloud_water_g_kg: ArrayLike,height_agl_m: ArrayLike,z_faces_m: ArrayLike,threshold_g_kg: float=.005) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Map source cloud liquid water to CALMET layers and above-model fields.

    Args:
        cloud_water_g_kg: Float-like array shaped ``(source_level, ...)`` with
            cloud liquid water mixing ratio in g kg-1.
        height_agl_m: Source mass-level heights above ground in metres with the
            same shape as ``cloud_water_g_kg``.
        z_faces_m: Strictly increasing one-dimensional CALMET layer faces in
            metres above ground.
        threshold_g_kg: Minimum source and reported above-model cloud water in
            g kg-1.

    Returns:
        Tuple ``(mapped, above, bottom, top)`` of float32 arrays. ``mapped`` has
        shape ``(target_level, ...)`` and units g kg-1; the remaining arrays
        have the source profile's trailing shape, with ``above`` in g kg-1 and
        cloud ``bottom`` and ``top`` in metres above ground. Absent cloud uses
        zero. Inputs are not modified.

    Raises:
        ValueError: If cloud and height profiles differ, contain no source
            level, or target faces are not a strictly increasing vector.
    """
    cloud=np.asarray(cloud_water_g_kg,dtype=MODEL_DTYPE)
    height=np.asarray(height_agl_m,dtype=MODEL_DTYPE)
    target=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    if cloud.shape!=height.shape or cloud.ndim<1 or cloud.shape[0]<1:
        raise ValueError('cloud water and height profiles must have identical shapes')
    if target.ndim!=1 or target.size<2 or np.any(np.diff(target)<=0.):
        raise ValueError('CALMET cloud-water faces must be a strictly increasing vector')
    levels=cloud.shape[0]
    source_faces=np.empty((levels+1,*cloud.shape[1:]),dtype=MODEL_DTYPE)
    source_faces[0]=np.where(height[0]<=MODEL_DTYPE(0.),
                             height[0]-MODEL_DTYPE(1.),MODEL_DTYPE(0.))
    if levels>1:
        source_faces[1:levels]=.5*(height[:-1]+height[1:])
    source_faces[-1]=2.*height[-1]-source_faces[-2]

    mapped=np.zeros((target.size-1,*cloud.shape[1:]),dtype=MODEL_DTYPE)
    for output_index,(bottom_face,top_face) in enumerate(zip(target[:-1],target[1:])):
        thickness=top_face-bottom_face
        for source_index in range(levels):
            upper=source_faces[source_index+1]
            lower=np.maximum(source_faces[source_index],0.)
            overlap=(upper>bottom_face)&(lower<top_face)&(upper>0.)
            depth=np.minimum(top_face,upper)-np.maximum(bottom_face,lower)
            contribution=cloud[source_index]*depth/thickness
            mapped[output_index]=np.where(
                overlap,mapped[output_index]+contribution,mapped[output_index])

    threshold=MODEL_DTYPE(threshold_g_kg)
    model_top=target[-1]
    weighted=np.zeros(cloud.shape[1:],dtype=MODEL_DTYPE)
    weight=np.zeros(cloud.shape[1:],dtype=MODEL_DTYPE)
    bottom=source_faces[-1].copy()
    top=np.zeros(cloud.shape[1:],dtype=MODEL_DTYPE)
    for source_index in range(levels):
        lower=source_faces[source_index]; upper=source_faces[source_index+1]
        selected=(upper>model_top)&(cloud[source_index]>=threshold)
        depth=upper-lower
        weighted=np.where(selected,weighted+cloud[source_index]*depth,weighted)
        weight=np.where(selected,weight+depth,weight)
        bottom=np.where(selected,np.minimum(bottom,lower),bottom)
        top=np.where(selected,np.maximum(top,upper),top)
    average=np.divide(weighted,weight,out=np.zeros_like(weighted),where=weight>0.)
    reported=average>=threshold
    return cast(FloatQuad,tuple(_model_array(value) for value in (
        mapped,np.where(reported,average,0.),np.where(reported,bottom,0.),
        np.where(reported,top,0.))))

def density_from_pressure_temperature(pressure_mb: ArrayLike,temperature_k: ArrayLike) -> Float32Array:
    """Calculate dry-air density with CALMET's AIRDEN ideal-gas expression.

    Args:
        pressure_mb: Atmospheric pressure in millibars, scalar or array-like.
        temperature_k: Air temperature in kelvins, broadcast-compatible with
            ``pressure_mb``.

    Returns:
        Float32 density in kg m-3 with the broadcast input shape.
    """
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    return _model_array(.3484321*pressure/temperature)

def coriolis_parameter(latitude_degrees: ArrayLike, configured_value: float=999.) -> Float32Array:
    """Calculate CALMET's absolute Coriolis field with an equatorial floor.

    Args:
        latitude_degrees: Latitude in decimal degrees, scalar or array-like.
        configured_value: Scalar Coriolis parameter in s-1. Values above 998
            request calculation from latitude.

    Returns:
        Float32 nonnegative Coriolis parameter in s-1 with the latitude shape;
        all values are floored at ``2.5e-5`` s-1.
    """
    latitude=np.asarray(latitude_degrees,dtype=np.float64)
    if configured_value>998.:
        omega=2.*np.pi/86400.
        angle=np.deg2rad(latitude)
        value=2.*omega*np.sin(angle)
    else:
        value=np.full_like(latitude,np.float64(configured_value))
    return _model_array(np.maximum(np.abs(value),np.float64(.25e-4)))

def inverse_distance_squared(values: ArrayLike, point_x: ArrayLike, point_y: ArrayLike, sample_x: ArrayLike, sample_y: ArrayLike, *, ignore_zero: bool=False) -> Float32Array:
    """Interpolate samples with CALMET R2INTERP inverse-square weighting.

    Args:
        values: Sample values whose final broadcast axis indexes samples.
        point_x: Target x coordinates in the same units as ``sample_x``.
        point_y: Target y coordinates, matching or broadcast-compatible with
            ``point_x``.
        sample_x: One-dimensional sample x coordinates.
        sample_y: One-dimensional sample y coordinates matching ``sample_x``.
        ignore_zero: If true, exclude sample values below ``1e-6``.

    Returns:
        Float32 interpolated values with the broadcast target shape. A target
        with no retained weight receives zero. Inputs are not modified.
    """
    values=np.asarray(values,dtype=MODEL_DTYPE)
    dx=np.asarray(sample_x,dtype=COORDINATE_DTYPE)-np.asarray(
        point_x,dtype=COORDINATE_DTYPE)[...,None]
    dy=np.asarray(sample_y,dtype=COORDINATE_DTYPE)-np.asarray(
        point_y,dtype=COORDINATE_DTYPE)[...,None]
    weights=1.0/np.maximum(dx*dx+dy*dy,0.001)
    if ignore_zero:
        weights=np.where(values>=MODEL_DTYPE(1.e-6),weights,0.0)
    denominator=weights.sum(axis=-1,dtype=ACCUMULATION_DTYPE)
    numerator=(values*weights).sum(axis=-1,dtype=ACCUMULATION_DTYPE)
    result=np.divide(numerator,denominator,out=np.zeros_like(denominator),
                     where=denominator>1.e-9)
    return _model_array(result)


def lake_breeze_adjust(u: ArrayLike, v: ArrayLike, station_u: ArrayLike, station_v: ArrayLike, station_xy_m: ArrayLike,
                       station_ids: ArrayLike, regions: Iterable[LakeBreezeRegion], dx_m: float, dy_m: float) -> tuple[Float32Array, Float32Array]:
    """Apply LLBREEZ coast-distance interpolation to the lowest wind layer.

    Args:
        u: Wind eastward component shaped ``(level, y, x)`` in m s-1.
        v: Wind northward component with the same shape and units as ``u``.
        station_u: One-dimensional station eastward winds in m s-1.
        station_v: One-dimensional station northward winds in m s-1.
        station_xy_m: Float-like ``(station, 2)`` projected coordinates in m.
        station_ids: One-dimensional integer identifiers corresponding to the
            station arrays.
        regions: Iterable of nine-item regions: grid-line bounds
            ``XG1, XG2, YG1, YG2``, two coastline endpoints in relative metres,
            and an ordered station-ID sequence.
        dx_m: Target grid spacing along x in metres.
        dy_m: Target grid spacing along y in metres.

    Returns:
        Pair of float32 ``(level, y, x)`` copies. Only lowest-layer cells inside
        configured regions are replaced; other cells and higher layers remain
        unchanged.

    Raises:
        ValueError: If wind or station shapes are inconsistent, a region has
            the wrong layout or no stations, or a station ID is unavailable.
    """
    result_u=np.asarray(u,dtype=MODEL_DTYPE).copy()
    result_v=np.asarray(v,dtype=MODEL_DTYPE).copy()
    if result_u.shape!=result_v.shape or result_u.ndim!=3:
        raise ValueError('lake-breeze U/V fields must share (level,y,x) shape')
    ids=np.asarray(station_ids,dtype=np.int64)
    xy=np.asarray(station_xy_m,dtype=np.float64)
    su=np.asarray(station_u,dtype=MODEL_DTYPE)
    sv=np.asarray(station_v,dtype=MODEL_DTYPE)
    if xy.shape!=(ids.size,2) or su.shape!=(ids.size,) or sv.shape!=(ids.size,):
        raise ValueError('lake-breeze station arrays have inconsistent shapes')
    ny,nx=result_u.shape[1:]
    grid_i=np.arange(1,nx+1,dtype=np.float64); grid_j=np.arange(1,ny+1,dtype=np.float64)
    xx=(grid_i-np.float64(.5))*np.float64(dx_m)
    yy=(grid_j-np.float64(.5))*np.float64(dy_m)
    for region in regions:
        if len(region)!=9:
            raise ValueError('lake-breeze region must contain eight coordinates and station IDs')
        xg1,xg2,yg1,yg2,x_begin,y_begin,x_end,y_end=(
            np.float64(value) for value in region[:8])
        selected=region[8]
        selected=cast(Sequence[int],np.asarray(selected,dtype=np.int64))
        if cast(IntArray,selected).size==0:
            raise ValueError('lake-breeze region must contain at least one station')
        lookup: dict[int, int] = {}
        for index,value in enumerate(ids): lookup.setdefault(int(value),index)
        try: indices=np.asarray([lookup[int(value)] for value in cast(IntArray,selected)],dtype=np.intp)
        except KeyError as error:
            raise ValueError(f'lake-breeze station ID {error.args[0]} is not available') from error
        xcd=np.float64(x_begin-x_end); ycd=np.float64(y_begin-y_end)
        slope=np.float64(9.9e19) if xcd==0. else np.float64(ycd/xcd)
        intercept=np.float64(y_begin-slope*x_begin)
        sx=xy[indices,0]; sy=xy[indices,1]
        if xcd==0.:
            station_distance=np.abs(sx-x_end)
        elif ycd==0.:
            station_distance=np.abs(sy-y_end)
        else:
            a=np.abs(sx*slope+intercept-sy)
            b=np.abs((sy-intercept)/slope-sx)
            hypotenuse=np.sqrt(np.asarray(a*a+b*b,dtype=np.float64))
            station_distance=np.divide(np.asarray(a*b,dtype=np.float64),hypotenuse,
                out=np.zeros_like(a),where=hypotenuse>0.)
        station_sign=np.asarray(sy-slope*sx-intercept,dtype=np.float64)
        inside_i=(grid_i>=min(xg1,xg2))&(grid_i<=max(xg1,xg2))
        inside_j=(grid_j>=min(yg1,yg2))&(grid_j<=max(yg1,yg2))
        target_y,target_x=np.nonzero(inside_j[:,None]&inside_i[None,:])
        if target_x.size==0: continue
        gx=xx[target_x]; gy=yy[target_y]
        if xcd==0.:
            grid_distance=np.abs(gx-x_end)
        elif ycd==0.:
            grid_distance=np.abs(gy-y_end)
        else:
            a=np.abs(gx*slope+intercept-gy)
            b=np.abs((gy-intercept)/slope-gx)
            hypotenuse=np.sqrt(np.asarray(a*a+b*b,dtype=np.float64))
            grid_distance=np.divide(np.asarray(a*b,dtype=np.float64),hypotenuse,
                out=np.zeros_like(a),where=hypotenuse>0.)
        grid_sign=np.asarray(gy-slope*gx-intercept,dtype=np.float64)
        same=grid_sign[:,None]*station_sign[None,:]>=0.
        relative=np.asarray(np.where(same,station_distance[None,:]-grid_distance[:,None],
                          station_distance[None,:]+grid_distance[:,None]),dtype=np.float64)
        dpr=np.maximum(np.abs(relative.astype(np.float64)),1.e-10)
        weights=1./(dpr*dpr)
        values_u=su[indices]; values_v=sv[indices]
        valid_u=values_u<990.; valid_v=values_v<990.
        if valid_u.any() and valid_v.any():
            def interpolate(values: Float32Array,valid: BoolArray) -> Float64Array:
                """Interpolate one station component to active region cells.

                Args:
                    values: One-dimensional wind component for selected
                        stations in m s-1.
                    valid: Boolean mask selecting non-missing station values.

                Returns:
                    Float64 array of inverse-square coast-distance interpolated
                    values, one per target cell in the current region.
                """
                top=np.zeros(target_x.size,dtype=np.float64)
                bottom=np.zeros(target_x.size,dtype=np.float64)
                for station in np.flatnonzero(valid):
                    top=np.asarray(top+values[station]*weights[:,station],dtype=np.float64)
                    bottom+=weights[:,station]
                return np.asarray(top/bottom,dtype=np.float64)
            interpolated_u=interpolate(values_u,valid_u)
            interpolated_v=interpolate(values_v,valid_v)
            result_u[0,target_y,target_x]=interpolated_u
            result_v[0,target_y,target_x]=interpolated_v
    return result_u,result_v

def wind_speed_direction(u: ArrayLike,v: ArrayLike) -> tuple[Float32Array, Float32Array]:
    """Convert vector wind to CALMET RTHETA speed and direction.

    Args:
        u: Eastward wind component in m s-1, scalar or array-like.
        v: Northward wind component in m s-1, broadcast-compatible with ``u``.

    Returns:
        Pair ``(speed, direction)`` of float32 arrays with the broadcast input
        shape. Speed is in m s-1 and meteorological direction is degrees
        clockwise from north; calm direction is zero.
    """
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    squared=u*u+v*v
    speed=np.sqrt(squared)
    direction=np.fmod(270.-np.rad2deg(np.arctan2(v,u)),360.)
    direction=np.where(squared<1.e-10,0.,
                       np.where(direction==0.,360.,direction))
    return _model_array(speed),_model_array(direction)

def cloud_fraction(relative_humidity_percent: ArrayLike) -> Float32Array:
    """Diagnose CLOUD3 cloud fraction with the Teixeira relation.

    Args:
        relative_humidity_percent: Relative humidity in percent, scalar or
            array-like.

    Returns:
        Array with the input shape containing dimensionless cloud fraction.
        Computation uses float32 and values at or above 99 percent map to one.
    """
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    rhd=rh/100.; gap=1.-rhd
    result=np.ones_like(rhd)
    active=rh<99.
    root=np.sqrt(1.+100.*gap[active])
    result[active]=.02*(-1.+root)/gap[active]
    return result

def cloud_cover_teixeira_850(relative_humidity_percent: ArrayLike,pressure_mb: ArrayLike) -> Float32Array:
    """Diagnose CLOUD3 cover at the last level at or above 850 mb.

    Args:
        relative_humidity_percent: Float-like ``(level, y, x)`` relative
            humidity in percent.
        pressure_mb: Pressure in millibars with the same shape.

    Returns:
        Float32 dimensionless cloud fraction with shape ``(y, x)``. Columns
        that never cross below 850 mb use zero relative humidity.

    Raises:
        ValueError: If inputs do not share a three-dimensional shape.
    """
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    if rh.shape!=pressure.shape or rh.ndim!=3:
        raise ValueError('cloud profiles must share shape (z,y,x)')
    below=pressure<850.; first=np.argmax(below,axis=0); level=np.maximum(first-1,0)
    yy,xx=np.indices(level.shape)
    rh850=np.where(below.any(axis=0),rh[level,yy,xx],0.)
    return cloud_fraction(rh850)

def cloud_cover_mm5(relative_humidity_percent: ArrayLike, pressure_mb: ArrayLike, height_m: ArrayLike) -> tuple[Float32Array, Float32Array]:
    """Diagnose CLOUD4 low, middle, and high cloud cover and ceiling.

    Args:
        relative_humidity_percent: Float-like ``(level, y, x)`` relative
            humidity in percent.
        pressure_mb: Pressure in millibars with the same shape.
        height_m: Layer heights in metres above ground with the same shape.

    Returns:
        Pair ``(cover, ceiling)`` of float32 ``(y, x)`` arrays. ``cover`` is a
        dimensionless fraction and ``ceiling`` is metres above ground, with
        zero for clear columns. Pressure bands and integer midpoint selection
        follow CALMET CLOUD4 exactly.

    Raises:
        ValueError: If inputs do not share a three-dimensional shape.
    """
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    height=np.asarray(height_m,dtype=MODEL_DTYPE)
    if rh.shape!=pressure.shape or rh.shape!=height.shape or rh.ndim!=3: raise ValueError('cloud profiles must share shape (z,y,x)')
    nz=rh.shape[0]; levels=np.arange(nz)[:,None,None]
    def first_below(threshold: float) -> IntArray:
        """Find the first level whose pressure is below a threshold.

        Args:
            threshold: Pressure threshold in millibars.

        Returns:
            Integer ``(y, x)`` level indices; columns with no crossing receive
            ``nz``, one past the final valid level.
        """
        mask=pressure<threshold
        return np.where(mask.any(axis=0),mask.argmax(axis=0),nz)
    klo,kmid,khi=first_below(970.),first_below(800.),first_below(450.)
    def band_max(start: IntArray,stop: IntArray | None=None) -> Float32Array:
        """Find maximum relative humidity within column-specific level bands.

        Args:
            start: Integer ``(y, x)`` inclusive starting level.
            stop: Optional integer ``(y, x)`` exclusive ending level.

        Returns:
            Float32 ``(y, x)`` maximum relative humidity in percent. An empty
            band contributes zero.
        """
        mask=levels>=start[None,:,:]
        if stop is not None: mask &= levels<stop[None,:,:]
        return np.max(np.where(mask,rh,0.),axis=0)
    low=4.*band_max(klo,kmid)/100.-3.
    middle=4.*band_max(kmid,khi)/100.-3.
    high=2.5*band_max(khi)/100.-1.5
    low=np.clip(low,0.,1.); middle=np.clip(middle,0.,1.)
    high=np.clip(high,0.,1.)
    cover=np.maximum.reduce((low,middle,high))
    low_level=klo+(kmid-klo)//2; middle_level=kmid+(khi-kmid)//2
    ceiling_level=np.where(low>=.5,low_level,np.where(middle>=.5,middle_level,np.where(high>=.5,khi,
        np.where((low>=middle)&(low>=high),low_level,np.where((middle>=low)&(middle>=high),middle_level,khi)))))
    valid=(cover>1.e-9)&(ceiling_level<nz)
    ceiling=np.zeros_like(cover)
    yy,xx=np.indices(cover.shape); ceiling[valid]=height[ceiling_level[valid],yy[valid],xx[valid]]
    return cover,ceiling

def lapse_rate(temperature_k: ArrayLike, previous_convective_height_m: ArrayLike, z_centers_m: ArrayLike, height_m: float=200., *, spatial: bool=True) -> Float32Array:
    """Diagnose the CGAMMA2 near-surface potential-temperature lapse rate.

    Args:
        temperature_k: Float-like temperature shaped ``(level, y, x)`` in K.
        previous_convective_height_m: Previous convective height shaped
            ``(y, x)`` in metres.
        z_centers_m: One-dimensional model layer centres in metres above ground.
        height_m: Diagnostic interpolation height in metres above ground.
        spatial: If true, retain cellwise lapse rates; otherwise replace them
            with their domain mean.

    Returns:
        Float32 ``(y, x)`` lapse-rate field in K m-1. Cells with a previous
        convective height at least ``height_m`` use the dry-adiabatic value
        ``-0.0098`` K m-1.

    Raises:
        ValueError: If profile/grid shapes disagree or ``height_m`` is not
            bracketed above the lowest model midpoint.
    """
    temp=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    zi=np.asarray(previous_convective_height_m,dtype=MODEL_DTYPE)
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    height=MODEL_DTYPE(height_m)
    if temp.ndim!=3 or temp.shape[1:]!=zi.shape or temp.shape[0]!=centers.size: raise ValueError('temperature must be (z,y,x) and match heights/grid')
    upper=np.flatnonzero(centers>=height)
    if not upper.size: raise ValueError('lapse-rate height is above the top model level')
    k=int(upper[0])
    if k==0: raise ValueError('lapse-rate height is below the lowest model midpoint')
    ratio=(centers[k]-height)/(centers[k]-centers[k-1])
    at_height=temp[k]-(temp[k]-temp[k-1])*ratio
    diagnosed=(at_height-temp[0])/(height-centers[0])
    gamma=np.where(zi>=height,-.0098,diagnosed)
    if spatial: return gamma
    average=np.mean(gamma,dtype=ACCUMULATION_DTYPE)
    return np.full_like(gamma,average)

def froude_adjustment(u: ArrayLike, v: ArrayLike, terrain_m: ArrayLike, maximum_terrain_m: ArrayLike, gamma_k_per_m: ArrayLike,
                      temperature_k: ArrayLike, z_centers_m: ArrayLike, critical_froude: ArrayLike,
                      dx_m: float, dy_m: float) -> tuple[Float32Array, Float32Array]:
    """Apply CALMET FRADJ terrain blocking while conserving wind speed.

    Args:
        u: Eastward wind shaped ``(level, y, x)`` in m s-1.
        v: Northward wind with the same shape and units as ``u``.
        terrain_m: Cell terrain shaped ``(y, x)`` in metres.
        maximum_terrain_m: Local maximum terrain shaped ``(y, x)`` in metres.
        gamma_k_per_m: Near-surface temperature lapse shaped ``(y, x)`` in
            K m-1.
        temperature_k: Surface temperature shaped ``(y, x)`` in kelvins.
        z_centers_m: One-dimensional layer-centre heights above ground in m.
        critical_froude: Scalar or broadcast-compatible critical Froude number.
        dx_m: Grid spacing along x in metres.
        dy_m: Grid spacing along y in metres.

    Returns:
        Pair of float32 ``(level, y, x)`` wind arrays in m s-1. Blocked
        upslope flow is rotated tangentially; input arrays are not modified.

    Raises:
        ValueError: If wind, grid, or vertical-coordinate shapes disagree.
    """
    u=np.array(u,dtype=MODEL_DTYPE,copy=True)
    v=np.array(v,dtype=MODEL_DTYPE,copy=True)
    terrain=np.asarray(terrain_m,dtype=MODEL_DTYPE)
    maximum=np.asarray(maximum_terrain_m,dtype=MODEL_DTYPE)
    gamma=np.asarray(gamma_k_per_m,dtype=MODEL_DTYPE)
    temp=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    if u.shape!=v.shape or u.ndim!=3 or u.shape[1:]!=terrain.shape or u.shape[0]!=centers.size: raise ValueError('wind must have shape (z,y,x) matching grid and centers')
    west=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    gx=(east-west)*(.5/dx_m)
    gy=(north-south)*(.5/dy_m)
    flat=(np.abs(gx)<.00015)&(np.abs(gy)<.00015)
    gx=np.where(flat,0.,gx); gy=np.where(flat,0.,gy)
    magnitude=np.hypot(gx,gy)
    drx=np.divide(gx,magnitude,out=np.zeros_like(gx),where=magnitude>0.)
    dry=np.divide(gy,magnitude,out=np.zeros_like(gy),where=magnitude>0.)
    speed=np.sqrt(u*u+v*v)
    obstacle=maximum[None,:,:]-(terrain[None,:,:]+centers[:,None,None])
    gamma2=gamma+.01
    frequency=np.sqrt(np.maximum(0.,9.8*gamma2/temp))
    denominator=frequency[None,:,:]*obstacle
    froude=np.divide(speed,denominator,out=np.full_like(speed,np.inf),where=denominator>0)
    un=np.divide(u,speed,out=np.zeros_like(u),where=speed>0)
    vn=np.divide(v,speed,out=np.zeros_like(v),where=speed>0)
    toward=drx[None,:,:]*un+dry[None,:,:]*vn
    tangent=-dry[None,:,:]*un+drx[None,:,:]*vn
    tangent_sign=np.where(tangent>0.,1.,-1.)
    active=((speed>0)&(obstacle>0)&(gamma2[None,:,:]>0)&
            (froude<=critical_froude)&(toward>0)&(magnitude[None,:,:]>0))
    tangent_u=tangent_sign*(-dry[None,:,:]); tangent_v=tangent_sign*drx[None,:,:]
    u=np.where(active,tangent_u*speed,u)
    v=np.where(active,tangent_v*speed,v)
    return _model_array(u),_model_array(v)

def profile_temperature_layer(
    temperature_k: ArrayLike,
    height_m: ArrayLike,
    bottom_m: ArrayLike,
    depth_m: float=200.,
    *,
    minimum_lapse: float=.001,
    active_mask: ArrayLike | None=None,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Interpolate MIXDT2 layer temperatures and potential lapse.

    Args:
        temperature_k: Temperature profiles shaped ``(level, y, x)`` in K.
        height_m: Profile heights with the same shape in metres above ground.
        bottom_m: Cellwise ``(y, x)`` bottom height in metres.
        depth_m: Layer depth above ``bottom_m`` in metres.
        minimum_lapse: Minimum potential-temperature lapse in K m-1.
        active_mask: Optional boolean ``(y, x)`` mask. Only true columns are
            interpolated and validated; false columns return ``NaN``. This
            lets land-only MIXHT calculations skip water profiles.

    Returns:
        Tuple ``(lower, upper, lapse)`` of float32 ``(y, x)`` arrays. The first
        two are temperatures in K at ``bottom_m`` and ``bottom_m + depth_m``;
        ``lapse`` is potential-temperature lapse in K m-1 after the
        dry-adiabatic conversion.

    Raises:
        ValueError: If profile or mask shapes disagree, requested heights are
            not bracketed in an active column, or active profile heights do
            not increase strictly.
    """
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    height=np.asarray(height_m,dtype=MODEL_DTYPE)
    bottom=np.asarray(bottom_m,dtype=MODEL_DTYPE)
    if temperature.shape!=height.shape or temperature.ndim!=3 or temperature.shape[1:]!=bottom.shape: raise ValueError('profiles must be (z,y,x) and bottom must match their grid')
    active=(np.ones(bottom.shape,dtype=bool) if active_mask is None else
            np.asarray(active_mask,dtype=bool))
    if active.shape!=bottom.shape:
        raise ValueError('active temperature-profile mask must match the grid')
    flat_t=temperature.reshape(temperature.shape[0],-1)
    flat_z=height.reshape(height.shape[0],-1)
    active_columns=np.flatnonzero(active.ravel())
    def interpolate(target: ArrayLike) -> Float32Array:
        """Interpolate all flattened temperature columns at target heights.

        Args:
            target: ``(y, x)`` heights in metres above ground.

        Returns:
            Float32 ``(y, x)`` temperatures in kelvins.

        Raises:
            ValueError: If a target is not bracketed or adjacent profile
                heights fail to increase.
        """
        target=np.asarray(target,dtype=MODEL_DTYPE).ravel()
        output=np.full(target.shape,np.nan,dtype=MODEL_DTYPE)
        if active_columns.size==0:
            return output.reshape(bottom.shape)
        selected_target=target[active_columns]
        selected_height=flat_z[:,active_columns]
        upper=np.sum(selected_height<=selected_target[None,:],axis=0)
        if np.any(upper==0) or np.any(upper>=flat_z.shape[0]): raise ValueError('temperature profile does not bracket mixing-height layer')
        lower=upper-1
        zl=selected_height[lower,np.arange(active_columns.size)]
        zu=selected_height[upper,np.arange(active_columns.size)]
        if np.any(zu<=zl): raise ValueError('temperature profile heights must increase strictly')
        selected_temperature=flat_t[:,active_columns]
        column=np.arange(active_columns.size)
        value=(selected_temperature[upper,column]-
               (selected_temperature[upper,column]-selected_temperature[lower,column])*
               (zu-selected_target)/(zu-zl))
        output[active_columns]=value
        return _model_array(output.reshape(bottom.shape))
    lower=interpolate(bottom); upper=interpolate(bottom+depth_m)
    lapse=np.maximum((upper-lower)/depth_m+.0098,minimum_lapse)
    return cast(FloatTriple,tuple(_model_array(value) for value in (lower,upper,lapse)))

def mixing_height_maul_carson(temperature_profile_k: ArrayLike, profile_height_m: ArrayLike, previous_convective_height_m: ArrayLike,
                               temperature_jump_k: ArrayLike, friction_velocity: ArrayLike, monin_obukhov_m: ArrayLike,
                               sensible_heat_flux: ArrayLike, density: ArrayLike, coriolis: ArrayLike,
                               *, timestep_s: float=3600., depth_m: float=200., minimum_lapse: float=.001,
                               minimum_height_m: float=50., maximum_height_m: float=2500.,
                               threshold_flux_per_m: float=0., conste: float=.15, constn: float=2400.,
                               constb: float=1.41, relaxation: bool=True, relaxation_time_s: float=800., land_mask: ArrayLike | None=None) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate MIXHT2/MIXHMC land mixing heights for ``IMIXH=1``.

    Args:
        temperature_profile_k: Temperature shaped ``(level, y, x)`` in K.
        profile_height_m: Heights with the same shape in metres above ground.
        previous_convective_height_m: Previous convective height in metres.
        temperature_jump_k: Previous inversion temperature jump in K.
        friction_velocity: Friction velocity in m s-1.
        monin_obukhov_m: Monin-Obukhov length in metres.
        sensible_heat_flux: Surface sensible heat flux in W m-2.
        density: Air density in kg m-3.
        coriolis: Absolute Coriolis parameter in s-1.
        timestep_s: Integration time step in seconds.
        depth_m: MIXDT2 layer depth in metres.
        minimum_lapse: Minimum potential-temperature lapse in K m-1.
        minimum_height_m: Lower mixing-height bound in metres.
        maximum_height_m: Upper mixing-height bound in metres.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        conste: Maul-Carson entrainment coefficient.
        constn: Stable mechanical-height coefficient.
        constb: Neutral/convective mechanical-height coefficient.
        relaxation: Whether weak positive heating relaxes toward equilibrium.
        relaxation_time_s: Relaxation time scale in seconds.
        land_mask: Optional boolean ``(y, x)`` mask; false cells return NaN.

    Returns:
        Tuple ``(total, convective, temperature_jump)`` of float32 ``(y, x)``
        arrays. Heights are metres and the updated jump is kelvins.

    Raises:
        ValueError: If the temperature profiles or requested MIXDT2 layer are
            inconsistent.
    """
    old=np.asarray(previous_convective_height_m,dtype=MODEL_DTYPE)
    land: NDArray[np.bool_] | None=(
        None if land_mask is None else np.asarray(land_mask,dtype=bool)
    )
    if land is not None and land.shape!=old.shape:
        raise ValueError('land mask must match the mixing-height grid')
    jump=np.asarray(temperature_jump_k,dtype=MODEL_DTYPE)
    ustar=np.asarray(friction_velocity,dtype=MODEL_DTYPE)
    length=np.asarray(monin_obukhov_m,dtype=MODEL_DTYPE)
    qh=np.asarray(sensible_heat_flux,dtype=MODEL_DTYPE)
    rho=np.asarray(density,dtype=MODEL_DTYPE); f=np.asarray(coriolis,dtype=MODEL_DTYPE)
    active_profile=np.broadcast_to(qh>=0.,old.shape)
    if land is not None:
        active_profile=active_profile&land
    tht,thtp,gamma=profile_temperature_layer(
        temperature_profile_k,profile_height_m,old,depth_m,
        minimum_lapse=minimum_lapse,active_mask=active_profile,
    )
    daytime=qh>=0.
    wt=qh/(rho*996.)
    wto=threshold_flux_per_m*old/(1.2*996.)
    net=wt-wto
    twodte=2.*timestep_s*conste
    onedte=timestep_s*(1.+conste)
    next_jump=np.sqrt(np.maximum(gamma*twodte*net,0.))
    radicand=old*old+2.*(net*onedte-jump*old)/gamma
    convective=np.sqrt(np.maximum(radicand,0.))+next_jump/gamma
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=1.2*996.*wt/threshold_flux_per_m
        relaxation_decay=MODEL_DTYPE(np.exp(-timestep_s/relaxation_time_s))
        relaxed=equilibrium+(old-equilibrium)*relaxation_decay
        relaxing=(wt>0.)&(net<=0.)
        convective=np.where(relaxing,relaxed,convective)
        next_jump=np.where(relaxing,jump,next_jump)
    convective=np.where((wt<=0.) if relaxation else (net<=0.),0.,
                        np.clip(convective,0.,maximum_height_m))
    tave=.5*(tht+thtp)
    bvf=np.power(9.81*gamma/tave,.25)
    cmech=constb/np.sqrt(np.maximum(np.abs(f),1.e-30))
    mechanical=cmech*ustar/bvf
    total=np.clip(np.maximum(mechanical,convective),minimum_height_m,
                  maximum_height_m)
    stable1=constn*np.power(ustar,1.5)
    stable2=.4*np.sqrt(np.maximum(
        ustar*length/np.maximum(np.abs(f),1.e-30),0.))
    total=np.where(daytime,total,np.clip(np.minimum(stable1,stable2),minimum_height_m,maximum_height_m))
    convective=np.where(daytime,convective,0.)
    next_jump=np.where(daytime,next_jump,0.)
    if land is not None:
        total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan); next_jump=np.where(land,next_jump,np.nan)
    return cast(FloatTriple,tuple(_model_array(value) for value in (total,convective,next_jump)))


def convective_height_maul_carson(previous_convective_height_m: ArrayLike,temperature_jump_k: ArrayLike,
                                  buoyancy_flux_k_m_s: ArrayLike,potential_lapse_k_m: ArrayLike,*,
                                  timestep_s: float=3600.,threshold_flux_per_m: float=0.,
                                  maximum_height_m: float=2500.,minimum_lapse: float=.001,conste: float=.15,
                                  relaxation: bool=True,relaxation_time_s: float=800.) -> tuple[Float32Array, Float32Array]:
    """Calculate CALMET MIXHMC from a supplied surface buoyancy flux.

    Args:
        previous_convective_height_m: Previous convective height in metres.
        temperature_jump_k: Previous inversion temperature jump in kelvins.
        buoyancy_flux_k_m_s: Surface kinematic buoyancy flux in K m s-1.
        potential_lapse_k_m: Potential-temperature lapse in K m-1.
        timestep_s: Integration time step in seconds.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        maximum_height_m: Upper convective-height bound in metres.
        minimum_lapse: Minimum lapse used in the integration in K m-1.
        conste: Maul-Carson entrainment coefficient.
        relaxation: Whether weak positive flux relaxes toward equilibrium.
        relaxation_time_s: Relaxation time scale in seconds.

    Returns:
        Pair ``(convective_height, temperature_jump)`` of float32 arrays with
        the broadcast input shape. Height is metres and jump is kelvins.
    """
    old=np.asarray(previous_convective_height_m,dtype=MODEL_DTYPE)
    jump=np.asarray(temperature_jump_k,dtype=MODEL_DTYPE)
    wt=np.asarray(buoyancy_flux_k_m_s,dtype=MODEL_DTYPE)
    gamma=np.maximum(np.asarray(potential_lapse_k_m,dtype=MODEL_DTYPE),minimum_lapse)
    wto=threshold_flux_per_m*old/(1.2*996.)
    net=wt-wto
    twodte=2.*timestep_s*conste
    onedte=timestep_s*(1.+conste)
    next_jump=np.sqrt(np.maximum(gamma*twodte*net,0.))
    radicand=old*old+2.*(net*onedte-jump*old)/gamma
    convective=np.sqrt(np.maximum(radicand,0.))+next_jump/gamma
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=1.2*996.*wt/threshold_flux_per_m
        decay=MODEL_DTYPE(np.exp(-timestep_s/relaxation_time_s))
        relaxed=equilibrium+(old-equilibrium)*decay
        relaxing=(wt>0.)&(net<=0.)
        convective=np.where(relaxing,relaxed,convective)
        next_jump=np.where(relaxing,jump,next_jump)
    active=(wt>0.) if relaxation else (net>0.)
    convective=np.where(active,np.clip(convective,0.,maximum_height_m),0.)
    next_jump=np.where(active,next_jump,0.)
    return _model_array(convective),_model_array(next_jump)


def convective_height_batchvarova_gryning(
        previous_convective_height_m: ArrayLike,buoyancy_flux_k_m_s: ArrayLike,potential_lapse_k_m: ArrayLike,
        friction_velocity: ArrayLike,monin_obukhov_m: ArrayLike,surface_temperature_k: ArrayLike,*,
        timestep_s: float=3600.,minimum_height_m: float=50.,maximum_height_m: float=2500.,
        threshold_flux_per_m: float=0.,minimum_lapse: float=.001,relaxation: bool=True,
        relaxation_time_s: float=800.) -> Float32Array:
    """Solve vectorized CALMET MIXHBG convective heights.

    Args:
        previous_convective_height_m: Previous convective height in metres.
        buoyancy_flux_k_m_s: Surface kinematic buoyancy flux in K m s-1.
        potential_lapse_k_m: Potential-temperature lapse in K m-1.
        friction_velocity: Friction velocity in m s-1.
        monin_obukhov_m: Monin-Obukhov length in metres.
        surface_temperature_k: Surface air temperature in kelvins.
        timestep_s: Integration time step in seconds.
        minimum_height_m: Lower convective-height bound in metres.
        maximum_height_m: Upper convective-height bound in metres.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        minimum_lapse: Minimum lapse in K m-1.
        relaxation: Whether weak positive flux relaxes toward equilibrium.
        relaxation_time_s: Relaxation time scale in seconds.

    Returns:
        Float32 convective height in metres with the broadcast input shape.
        Nonlinear false-position iterations use float64 internally.
    """
    old=np.asarray(previous_convective_height_m,dtype=np.float64)
    wt=np.asarray(buoyancy_flux_k_m_s,dtype=np.float64)
    gamma=np.maximum(np.asarray(potential_lapse_k_m,dtype=np.float64),
                     np.float64(minimum_lapse))
    ustar=np.asarray(friction_velocity,dtype=np.float64)
    length=np.asarray(monin_obukhov_m,dtype=np.float64)
    tk=np.asarray(surface_temperature_k,dtype=np.float64)
    h0=np.maximum(old,np.float64(minimum_height_m))
    wto=np.asarray(np.float64(threshold_flux_per_m)*h0/
                    (np.float64(1.2)*np.float64(1005.)),dtype=np.float64)
    net=np.asarray(wt-wto,dtype=np.float64)
    active=(length<=0.)&(net>0.)
    convective=np.zeros_like(h0,dtype=np.float64)
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=np.asarray(np.float64(1.2)*np.float64(1005.)*wt/
                               np.float64(threshold_flux_per_m),dtype=np.float64)
        decay=np.asarray(np.exp(np.float64(-timestep_s)/
                                np.float64(relaxation_time_s)),dtype=np.float64)
        relaxed=np.asarray(equilibrium+(h0-equilibrium)*decay,dtype=np.float64)
        convective=np.where((length<=0.)&(wt>=0.)&(net<=0.),relaxed,convective)
    elif not relaxation:
        convective=np.where((length<=0.)&(net==0.),old,convective)

    # Solve the Batchvarova--Gryning equation with false-position iterations.
    a=np.where(active,np.asarray(net/gamma,dtype=np.float64),np.float64(1.)).astype(np.float64)
    b=np.asarray(np.float64(8.)*ustar**np.float64(2.)*tk/
                 (gamma*np.float64(9.81)),dtype=np.float64)
    c=np.asarray(np.float64(2.5)*np.float64(.4)*length,dtype=np.float64)
    d=np.float64(1.)+np.float64(.2)+np.float64(.2)
    e=np.float64(1.)+np.float64(.2)
    d3=np.asarray(d*d*d,dtype=np.float64)
    zeta0=np.asarray(d*h0-np.float64(2.)*c,dtype=np.float64)
    zetap0=np.asarray(e*h0-c,dtype=np.float64)
    r1=np.asarray(np.float64(8.)*c,dtype=np.float64)
    r2=np.asarray(r1*c,dtype=np.float64)
    r3=np.asarray(np.float64(2.)*b*d3/e,dtype=np.float64)
    r4=np.asarray(np.float64(1.)/(np.float64(2.)*a*d3),dtype=np.float64)
    r5=np.asarray(e/d,dtype=np.float64)

    def fbg(zeta: Float64Array,zetap: Float64Array) -> Float64Array:
        """Evaluate the Batchvarova-Gryning implicit height function.

        Args:
            zeta: Float64 transformed height array for the main logarithm.
            zetap: Float64 transformed height array for the secondary logarithm.

        Returns:
            Float64 array with the broadcast input shape and the implicit
            function value in the solver's transformed units.
        """
        # Inactive vector lanes use benign placeholders because logarithms
        # require positive arguments even though those lanes are discarded.
        safe_zeta=np.where(zeta>0.,zeta,np.float64(1.)).astype(np.float64)
        safe_zetap=np.where(zetap>0.,zetap,np.float64(1.)).astype(np.float64)
        return np.asarray(safe_zeta**np.float64(2.)+r1*safe_zeta+
                          r2*np.log(safe_zeta)+r3*np.log(safe_zetap),dtype=np.float64)

    time=np.asarray(np.float64(timestep_s)+r4*fbg(zeta0,zetap0),dtype=np.float64)
    weak=length>-500.
    square=np.asarray(h0*h0+np.float64(2.)*d*a*np.float64(timestep_s),dtype=np.float64)
    cube=np.asarray(h0**np.float64(3.)-np.float64(6.)*a*c*np.float64(timestep_s),
                    dtype=np.float64)
    # Evaluate the selected lanes separately so a negative inactive cube does
    # not emit a fractional-power warning.
    h=np.empty_like(h0,dtype=np.float64)
    h[weak]=np.sqrt(square[weak]).astype(np.float64)
    h[~weak]=np.cbrt(cube[~weak])
    zeta1=np.asarray(d*h-np.float64(2.)*c,dtype=np.float64)
    zetap1=np.asarray(e*h-c,dtype=np.float64)
    f1=np.asarray(r4*fbg(zeta1,zetap1)-time,dtype=np.float64)
    converged=np.abs(f1)<=1.
    f0=np.full_like(f1,np.float64(-timestep_s))
    denominator=np.asarray(f1-f0,dtype=np.float64)
    zeta2=np.asarray((f1*zeta0-f0*zeta1)/denominator,dtype=np.float64)
    zetap2=np.asarray(r5*(zeta2+np.float64(2.)*c)-c,dtype=np.float64)
    invalid=(zeta2<=0.)|(zetap2<=0.)
    zeta2=np.where(invalid,np.float64(.5)*(zeta0+zeta1),zeta2).astype(np.float64)
    zetap2=np.asarray(r5*(zeta2+np.float64(2.)*c)-c,dtype=np.float64)
    f2=np.asarray(r4*fbg(zeta2,zetap2)-time,dtype=np.float64)
    h=np.where(~converged,(zeta2+np.float64(2.)*c)/d,h).astype(np.float64)
    converged|=np.abs(f2)<=1.
    failed=np.zeros_like(converged)
    for _ in range(10):
        work=active&~converged&~failed
        denominator=np.asarray(f1-f2,dtype=np.float64)
        stalled=np.abs(denominator)<=np.float64(1.e-10)
        converged|=work&stalled
        safe=np.where(stalled,np.float64(1.),denominator).astype(np.float64)
        zeta3=np.asarray((f1*zeta2-f2*zeta1)/safe,dtype=np.float64)
        zetap3=np.asarray(r5*(zeta3+np.float64(2.)*c)-c,dtype=np.float64)
        bad=(zeta3<=0.)|(zetap3<=0.)
        failed|=work&bad
        f3=np.asarray(r4*fbg(zeta3,zetap3)-time,dtype=np.float64)
        h=np.where(work&~bad,(zeta3+np.float64(2.)*c)/d,h).astype(np.float64)
        zeta1=np.where(work&~bad,zeta2,zeta1)
        zeta2=np.where(work&~bad,zeta3,zeta2)
        f1=np.where(work&~bad,f2,f1)
        f2=np.where(work&~bad,f3,f2)
        converged|=work&~bad&(np.abs(f3)<=1.)
    fallback=active&~converged
    h=np.where(fallback,((zeta1+zeta2)*np.float64(.5)+np.float64(2.)*c)/d,h).astype(np.float64)
    result=np.where(
        active,
        np.clip(h,np.float64(minimum_height_m),np.float64(maximum_height_m)),
        convective)
    return _model_array(result)


def mixing_height_batchvarova_gryning(temperature_profile_k: ArrayLike, profile_height_m: ArrayLike,
                                      previous_convective_height_m: ArrayLike, temperature_jump_k: ArrayLike,
                                      friction_velocity: ArrayLike, monin_obukhov_m: ArrayLike,
                                      sensible_heat_flux: ArrayLike, density: ArrayLike, coriolis: ArrayLike,
                                      *, surface_temperature_k: ArrayLike, timestep_s: float=3600.,
                                      depth_m: float=200., minimum_lapse: float=.001,
                                      minimum_height_m: float=50., maximum_height_m: float=2500.,
                                      threshold_flux_per_m: float=0., constn: float=2400.,
                                      constb: float=1.41, relaxation: bool=True,
                                      relaxation_time_s: float=800., land_mask: ArrayLike | None=None) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate MIXHBG plus common MIXHT2 mechanical mixing heights.

    Args:
        temperature_profile_k: Temperature shaped ``(level, y, x)`` in K.
        profile_height_m: Heights with the same shape in metres above ground.
        previous_convective_height_m: Previous convective height in metres.
        temperature_jump_k: Existing inversion temperature jump in kelvins.
        friction_velocity: Friction velocity in m s-1.
        monin_obukhov_m: Monin-Obukhov length in metres.
        sensible_heat_flux: Surface sensible heat flux in W m-2.
        density: Air density in kg m-3.
        coriolis: Absolute Coriolis parameter in s-1.
        surface_temperature_k: Surface air temperature in kelvins.
        timestep_s: Integration time step in seconds.
        depth_m: MIXDT2 layer depth in metres.
        minimum_lapse: Minimum potential-temperature lapse in K m-1.
        minimum_height_m: Lower mixing-height bound in metres.
        maximum_height_m: Upper mixing-height bound in metres.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        constn: Stable mechanical-height coefficient.
        constb: Neutral/convective mechanical-height coefficient.
        relaxation: Whether weak positive heating relaxes toward equilibrium.
        relaxation_time_s: Relaxation time scale in seconds.
        land_mask: Optional boolean ``(y, x)`` mask; false cells return NaN.

    Returns:
        Tuple ``(total, convective, temperature_jump)`` of float32 ``(y, x)``
        arrays. Heights are metres. MIXHBG preserves the supplied jump in K.

    Raises:
        ValueError: If temperature profile shapes or the requested MIXDT2
            layer are inconsistent.

    Notes:
        The nonlinear equation uses float64 vectorized false-position
        iterations without a horizontal Python loop.
    """
    old=np.asarray(previous_convective_height_m,dtype=np.float64)
    land: NDArray[np.bool_] | None=(
        None if land_mask is None else np.asarray(land_mask,dtype=bool)
    )
    if land is not None and land.shape!=old.shape:
        raise ValueError('land mask must match the mixing-height grid')
    jump=np.asarray(temperature_jump_k,dtype=np.float64)
    ustar=np.asarray(friction_velocity,dtype=np.float64)
    length=np.asarray(monin_obukhov_m,dtype=np.float64)
    qh=np.asarray(sensible_heat_flux,dtype=np.float64)
    rho_field=np.asarray(density,dtype=np.float64)
    f=np.asarray(coriolis,dtype=np.float64)
    tk=np.asarray(surface_temperature_k,dtype=np.float64)
    h0=np.maximum(old,np.float64(minimum_height_m))
    active_profile=np.broadcast_to(qh>=np.float64(0.),old.shape)
    if land is not None:
        active_profile=active_profile&land
    tht,thtp,gamma=profile_temperature_layer(
        temperature_profile_k,profile_height_m,h0,depth_m,
        minimum_lapse=minimum_lapse,active_mask=active_profile,
    )
    wt=np.asarray(qh/(rho_field*np.float64(996.)),dtype=np.float64)
    convective=convective_height_batchvarova_gryning(
        h0,wt,gamma,ustar,length,tk,timestep_s=timestep_s,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
        threshold_flux_per_m=threshold_flux_per_m,minimum_lapse=minimum_lapse,
        relaxation=relaxation,relaxation_time_s=relaxation_time_s)
    daytime=qh>=np.float64(0.)
    tave=np.asarray(np.float64(.5)*(tht+thtp),dtype=np.float64)
    bvf=np.asarray(np.power(
        np.asarray(np.float64(9.81)*gamma/tave,dtype=np.float64),np.float64(.25)),
        dtype=np.float64)
    cmech=np.asarray(np.float64(constb)/np.sqrt(np.maximum(
        np.abs(f),np.float64(1.e-30))),dtype=np.float64)
    mechanical=np.asarray(cmech*ustar/bvf,dtype=np.float64)
    total=np.clip(np.maximum(mechanical,convective),np.float64(minimum_height_m),
                  np.float64(maximum_height_m))
    stable1=np.asarray(np.float64(constn)*np.power(ustar,np.float64(1.5)),
                       dtype=np.float64)
    stable2=np.asarray(np.float64(.4)*np.sqrt(np.maximum(
        ustar*length/np.maximum(np.abs(f),np.float64(1.e-30)),np.float64(0.))),
        dtype=np.float64)
    total=np.where(daytime,total,np.clip(np.minimum(stable1,stable2),
        np.float64(minimum_height_m),np.float64(maximum_height_m))).astype(np.float64)
    convective=np.where(daytime,convective,np.float64(0.)).astype(np.float64)
    if land is not None:
        total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan)
        jump=np.where(land,jump,np.nan)
    return cast(FloatTriple,tuple(_model_array(value) for value in (total,convective,jump)))

def hosker_roughness(speed_m_s: ArrayLike) -> Float32Array:
    """Calculate Hosker wind-dependent water roughness.

    Args:
        speed_m_s: Wind speed in m s-1, scalar or array-like.

    Returns:
        Float32 roughness length in metres with the input shape, floored at
        ``2e-6`` m. Inputs are not modified.
    """
    speed=np.asarray(speed_m_s,dtype=MODEL_DTYPE)
    return _model_array(np.maximum(2.e-6,2.e-6*np.power(speed,2.5)))


def neutral_water_parameters(u10: ArrayLike, v10: ArrayLike, coriolis: ArrayLike, *, constw: float=.16,
                             minimum_height_m: float=50., maximum_height_m: float=2500.) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array]:
    """Calculate WATER's neutral ``NOWSTA=0`` state for NOOBS runs.

    Args:
        u10: Ten-metre eastward wind in m s-1.
        v10: Ten-metre northward wind in m s-1, broadcast-compatible with
            ``u10``.
        coriolis: Absolute Coriolis parameter in s-1, broadcast-compatible with
            the wind.
        constw: OCD overwater mechanical mixing-height coefficient.
        minimum_height_m: Lower mixing-height bound in metres.
        maximum_height_m: Upper mixing-height bound in metres.

    Returns:
        Six float32 arrays with the broadcast input shape:
        ``(roughness_m, friction_velocity_m_s, monin_obukhov_m,
        sensible_heat_flux_w_m2, mixing_height_m, convective_height_m)``.
        The neutral path uses ``9.9e9`` m Monin-Obukhov length and zero heat
        flux and convective height.
    """
    u=np.asarray(u10,dtype=MODEL_DTYPE); v=np.asarray(v10,dtype=MODEL_DTYPE)
    f=np.asarray(coriolis,dtype=MODEL_DTYPE)
    speed=np.sqrt(u*u+v*v)
    roughness=hosker_roughness(speed)
    friction=np.maximum(.4*speed/np.log(10./roughness),1.e-9)
    mixing=water_mechanical_height(friction,f,constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    return (roughness,_model_array(friction),np.full_like(speed,9.9e9),
            np.zeros_like(speed),mixing,np.zeros_like(speed))


def neutral_water_station_profiles(u: ArrayLike,v: ArrayLike,anemometer_height_m: ArrayLike,current_mixing_height_m: ArrayLike,
        coriolis: ArrayLike,z_centers_m: ArrayLike,z_faces_m: ArrayLike,station_latitude: ArrayLike,*,constw: float=.16,
        minimum_height_m: float=50.,maximum_height_m: float=2500.,ipsifcn: int=0) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array]:
    """Run WATER2's neutral station path followed by SIMILT profiles.

    Args:
        u: One-dimensional station eastward wind in m s-1.
        v: One-dimensional station northward wind in m s-1.
        anemometer_height_m: Wind measurement heights in metres by station.
        current_mixing_height_m: Current grid mixing heights in metres sampled
            at station locations.
        coriolis: Absolute Coriolis parameter in s-1 by station.
        z_centers_m: One-dimensional CALMET layer centres in metres.
        z_faces_m: One-dimensional layer faces in metres.
        station_latitude: Station latitude in degrees north.
        constw: OCD overwater mechanical mixing-height coefficient.
        minimum_height_m: Lower overwater mixing-height bound in metres.
        maximum_height_m: Upper overwater mixing-height bound in metres.
        ipsifcn: SIMILT stability function selector, 0 or 1.

    Returns:
        Six float32 arrays ``(u_profile, v_profile, roughness, friction,
        monin_obukhov, mixing_height)``. Profiles have shape
        ``(level, station)`` in m s-1; state arrays have shape ``(station,)``
        and units m, m s-1, m, and m respectively. Missing/calm stations retain
        the observed lowest layer, 9999 aloft, and NaN state.

    Raises:
        ValueError: If station/vertical shapes disagree or ``ipsifcn`` is
            unsupported.
    """
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    values=tuple(np.asarray(value,dtype=MODEL_DTYPE) for value in (
        anemometer_height_m,current_mixing_height_m,coriolis,station_latitude))
    if u.ndim!=1 or v.shape!=u.shape or any(value.shape!=u.shape for value in values):
        raise ValueError('WATER2 station inputs must be matching one-dimensional arrays')
    anemometer,current_mixing,coriolis_value,latitude=values
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    output_u=np.full((centers.size,u.size),9999.,dtype=MODEL_DTYPE)
    output_v=np.full_like(output_u,9999.); output_u[0]=u; output_v[0]=v
    state: list[Float32Array] = [
        np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(4)
    ]
    speed=np.sqrt(u*u+v*v)
    active=(u<=990.)&(v<=990.)&(speed>=.0001)
    if not active.any(): return cast(FloatSix,(output_u,output_v,*state))
    selected_speed=speed[active]; selected_height=anemometer[active]
    roughness=hosker_roughness(selected_speed)
    rescale=np.trunc(selected_height).astype(np.int32)!=10
    speed10=selected_speed.copy()
    speed10[rescale]=(selected_speed[rescale]*
        np.log(10./roughness[rescale])/
        np.log(selected_height[rescale]/roughness[rescale]))
    # WATER2 does not repeat Z0MIN after adjusting a non-10 m observation.
    roughness[rescale]=2.e-6*np.power(speed10[rescale],2.5)
    friction=np.maximum(.4*speed10/np.log(10./roughness),1.e-9)
    length=np.full_like(friction,9.9e9)
    mixing=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    mixing=np.maximum(np.minimum(mixing,current_mixing[active]),minimum_height_m)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],selected_height,centers,z_faces_m,roughness,length,mixing,
        latitude[active],ipsifcn=ipsifcn,minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    for destination,source in zip(state,(roughness,friction,length,mixing),strict=True):
        destination[active]=source
    return cast(FloatSix,(output_u,output_v,*state))


def water_mechanical_height(friction_velocity: ArrayLike,coriolis: ArrayLike,*,constw: float=.16,
                            minimum_height_m: float=50.,maximum_height_m: float=2500.) -> Float32Array:
    """Calculate WATER mechanical mixing height ``CONSTW*USTAR/FCORI``.

    Args:
        friction_velocity: Friction velocity in m s-1.
        coriolis: Absolute Coriolis parameter in s-1, broadcast-compatible with
            ``friction_velocity``.
        constw: Dimensionless overwater mechanical-height coefficient.
        minimum_height_m: Lower returned height bound in metres.
        maximum_height_m: Upper returned height bound in metres.

    Returns:
        Float32 mixing height in metres with the broadcast input shape.
    """
    friction=np.asarray(friction_velocity,dtype=MODEL_DTYPE)
    coriolis=np.asarray(coriolis,dtype=MODEL_DTYPE)
    return _model_array(np.clip(constw*friction/coriolis,
                                minimum_height_m,maximum_height_m))


def saturation_vapor_pressure_lowe(temperature_c: ArrayLike) -> Float32Array:
    """Evaluate the CALMET ESAT Lowe saturation-vapour-pressure polynomial.

    Args:
        temperature_c: Air or water temperature in degrees Celsius, scalar or
            array-like.

    Returns:
        Float32 saturation vapour pressure in millibars with the input shape.
    """
    value=np.asarray(temperature_c,dtype=MODEL_DTYPE)
    coefficients=np.asarray(
        (6.107799961,.4436518521,.01428945805,.0002650648471,
         .000003031240396,.00000002034080948,.00000000006136820929),
        dtype=MODEL_DTYPE)
    result=np.full_like(value,coefficients[-1],dtype=MODEL_DTYPE)
    for coefficient in reversed(coefficients[:-1]):
        result=coefficient+value*result
    return _model_array(result)


def overwater_delta_temperature_state(air_temperature_k: ArrayLike,sea_temperature_k: ArrayLike,
                                      relative_humidity_percent: ArrayLike,
                                      measurement_height_m: ArrayLike) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Prepare WATER virtual-potential-temperature inputs for DELTAT.

    Args:
        air_temperature_k: Measured air temperature in kelvins.
        sea_temperature_k: Sea-surface temperature in kelvins.
        relative_humidity_percent: Measured relative humidity in percent.
        measurement_height_m: Air measurement height above water in metres.

    Returns:
        Four broadcast float32 arrays in kelvins:
        ``(theta, delta_theta_z, delta_theta_10m, theta_10m)``. ``theta`` is
        measurement-height virtual potential temperature; the two deltas are
        air-minus-saturated-surface differences at measurement height and
        10 m; ``theta_10m`` is the corresponding 10-m value.
    """
    air,sea,rh,height=np.broadcast_arrays(
        np.asarray(air_temperature_k,dtype=MODEL_DTYPE),
        np.asarray(sea_temperature_k,dtype=MODEL_DTYPE),
        np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE),
        np.asarray(measurement_height_m,dtype=MODEL_DTYPE))
    air_c=air-273.15
    saturation=saturation_vapor_pressure_lowe(air_c)
    vapor=.01*rh*saturation
    mixing=.622*vapor/(1000.-vapor)
    virtual=air*(1.+.61*mixing)
    theta=virtual+.01*height

    sea_c=sea-273.15
    sea_saturation=saturation_vapor_pressure_lowe(sea_c)
    saturated_mixing=.622*sea_saturation/(1000.-sea_saturation)
    surface_theta=sea*(1.+.61*saturated_mixing)
    difference=theta-surface_theta
    difference_10m=difference*11.512925/(np.log(height)+9.2103404)
    theta_10m=surface_theta+difference_10m
    return cast(FloatQuad,tuple(_model_array(value) for value in
                 (theta,difference,difference_10m,theta_10m)))


def delta_temperature_flux(wind_speed_10m: ArrayLike,measurement_height_m: ArrayLike,
                           measurement_minus_surface_virtual_potential_k: ArrayLike,
                           ten_m_minus_surface_virtual_potential_k: ArrayLike,
                           virtual_potential_temperature_10m_k: ArrayLike,
                           measurement_virtual_potential_temperature_k: ArrayLike,
                           roughness_m: ArrayLike) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Solve CALMET DELTAT surface-layer fluxes by nonlinear iteration.

    Args:
        wind_speed_10m: Ten-metre wind speed in m s-1.
        measurement_height_m: Thermal measurement height in metres.
        measurement_minus_surface_virtual_potential_k: Virtual potential
            temperature difference at measurement height in kelvins.
        ten_m_minus_surface_virtual_potential_k: Corresponding 10-m difference
            in kelvins.
        virtual_potential_temperature_10m_k: Ten-metre virtual potential
            temperature in kelvins.
        measurement_virtual_potential_temperature_k: Measurement-height
            virtual potential temperature in kelvins.
        roughness_m: Water roughness length in metres.

    Returns:
        Tuple ``(monin_obukhov, friction_velocity, buoyancy_heat_flux)`` of
        float32 arrays with the broadcast input shape. Units are metres,
        m s-1, and W m-2 respectively. Iteration uses float64 internally.
    """
    wind,height,difference,difference_10m,theta_10m,theta,z0=np.broadcast_arrays(
        np.asarray(wind_speed_10m,dtype=np.float64),
        np.asarray(measurement_height_m,dtype=np.float64),
        np.asarray(measurement_minus_surface_virtual_potential_k,dtype=np.float64),
        np.asarray(ten_m_minus_surface_virtual_potential_k,dtype=np.float64),
        np.asarray(virtual_potential_temperature_10m_k,dtype=np.float64),
        np.asarray(measurement_virtual_potential_temperature_k,dtype=np.float64),
        np.asarray(roughness_m,dtype=np.float64))
    r=np.float64
    von_karman=r(.4); vkg=r(von_karman*r(9.81))
    log_10=np.asarray(np.log(r(10.)/z0),dtype=np.float64)
    log_measurement=np.asarray(np.log(height/z0),dtype=np.float64)
    neutral_ustar=np.maximum(
        np.asarray(von_karman*wind/log_10,dtype=np.float64),r(1.e-9))
    length=np.full(wind.shape,r(9.9e9),dtype=np.float64)
    ustar=neutral_ustar.copy()
    nonneutral=np.abs(difference_10m)>r(1.e-6)
    drag=np.asarray(r(.75e-3)+r(.067e-3)*wind,dtype=np.float64)
    initial=np.asarray(theta_10m*np.power(drag,r(1.5))*wind**2/
                       (r(5.096e-3)*np.where(nonneutral,difference_10m,r(1.))),
                       dtype=np.float64)
    active=nonneutral&(np.abs(initial)<=r(9999.))
    length=np.where(active,initial,length)
    length=np.where(active&(np.abs(length)<r(5.)),np.copysign(r(5.),length),length)
    old=length.copy()
    for _ in range(3):
        safe_length=np.where(active,length,r(1.))
        wind_ratio=np.asarray(r(10.)/safe_length,dtype=np.float64)
        temperature_ratio=np.asarray(height/safe_length,dtype=np.float64)
        stable=safe_length>r(0.)
        # NumPy evaluates both arms of ``where``. Feed unstable-only
        # expressions benign values on stable or inactive lanes.
        unstable_x_base=np.where(~stable,r(1.)-r(15.)*wind_ratio,r(1.))
        x=np.asarray(unstable_x_base**r(.25),dtype=np.float64)
        psi_u_unstable=np.asarray(
            r(2.)*np.log(r(.5)*(r(1.)+x))+
            np.log(r(.5)*(r(1.)+x*x))-r(2.)*np.arctan(x)+np.pi/r(2.),
            dtype=np.float64)
        unstable_y_base=np.where(~stable,r(1.)-r(9.)*temperature_ratio,r(1.))
        y=np.asarray(np.sqrt(unstable_y_base),dtype=np.float64)
        psi_t_unstable=np.asarray(r(1.48)*np.log(r(.5)*(r(1.)+y)),dtype=np.float64)
        psi_u=np.where(stable,-r(4.7)*wind_ratio,psi_u_unstable)
        psi_t=np.where(stable,-r(4.7)*temperature_ratio,psi_t_unstable)
        candidate_ustar=np.maximum(
            np.asarray(von_karman*wind/(log_10-psi_u),dtype=np.float64),r(1.e-9))
        theta_star=np.asarray(von_karman*difference/
                              (r(.74)*log_measurement-psi_t),dtype=np.float64)
        valid_theta=np.abs(theta_star)>r(1.e-6)
        candidate_length=np.asarray(
            theta*candidate_ustar**2/
            (vkg*np.where(valid_theta,theta_star,r(1.))),dtype=np.float64)
        valid=active&valid_theta&(np.abs(candidate_length)<=r(9999.))
        candidate_length=np.where(np.abs(candidate_length)<r(5.),
                                  np.copysign(r(5.),candidate_length),candidate_length)
        ustar=np.where(valid,candidate_ustar,ustar)
        length=np.where(valid,candidate_length,length)
        converged=valid&(np.abs((length-old)/old)<r(.05))
        neutralized=active&~valid
        ustar=np.where(neutralized,neutral_ustar,ustar)
        length=np.where(neutralized,r(9.9e9),length)
        active&=valid&~converged
        old=np.where(active,length,old)
    buoyancy_flux_w_m2=np.asarray(-r(1195.2)*theta*ustar**3/(vkg*length),
                                  dtype=np.float64)
    return cast(FloatTriple,tuple(_model_array(value) for value in
                 (length,ustar,buoyancy_flux_w_m2)))


def delta_temperature_water_station_profiles(u: ArrayLike,v: ArrayLike,anemometer_height_m: ArrayLike,
        air_temperature_k: ArrayLike,air_minus_sea_temperature_k: ArrayLike,relative_humidity_percent: ArrayLike,
        observed_mixing_height_m: ArrayLike,current_mixing_height_m: ArrayLike,coriolis: ArrayLike,
        potential_lapse_k_m: ArrayLike,previous_temperature_jump_k: ArrayLike,z_centers_m: ArrayLike,z_faces_m: ArrayLike,
        station_latitude: ArrayLike,*,mixing_method: int=1,constw: float=.16,minimum_height_m: float=50.,
        maximum_height_m: float=2500.,threshold_flux_per_m: float=.05,ipsifcn: int=0,
        temperature_measurement_height_m: ArrayLike | None=None,sea_temperature_k: ArrayLike | None=None,**mixing_options: MixingOption) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array]:
    """Run WATER2's DELTAT station branch through mixing and SIMILT.

    Args:
        u: One-dimensional station eastward wind in m s-1.
        v: One-dimensional station northward wind in m s-1.
        anemometer_height_m: Wind measurement heights in metres.
        air_temperature_k: Station air temperatures in kelvins.
        air_minus_sea_temperature_k: Air-minus-sea temperature in kelvins.
        relative_humidity_percent: Station relative humidity in percent.
        observed_mixing_height_m: Observed mixing heights in metres; values at
            least 990 indicate missing observations.
        current_mixing_height_m: Current grid mixing heights in metres.
        coriolis: Absolute Coriolis parameter in s-1.
        potential_lapse_k_m: Potential-temperature lapse in K m-1.
        previous_temperature_jump_k: Previous inversion jump in kelvins.
        z_centers_m: One-dimensional CALMET layer centres in metres.
        z_faces_m: One-dimensional CALMET layer faces in metres.
        station_latitude: Station latitude in degrees north.
        mixing_method: Mixing-height selector ``-2``, ``-1``, 1, or 2.
        constw: OCD overwater mechanical-height coefficient.
        minimum_height_m: Lower overwater mixing-height bound in metres.
        maximum_height_m: Upper overwater mixing-height bound in metres.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        ipsifcn: SIMILT stability function selector, 0 or 1.
        temperature_measurement_height_m: Optional thermal sensor heights in
            metres; anemometer heights are used when omitted.
        sea_temperature_k: Optional explicit sea temperature in kelvins;
            otherwise it is derived from air temperature and delta.
        **mixing_options: Additional keyword options passed to the selected
            convective-height routine.

    Returns:
        Nine float32 arrays: ``(u_profile, v_profile, roughness, friction,
        monin_obukhov, mixing_height, convective_height, temperature_jump,
        buoyancy_heat_flux)``. Profiles are ``(level, station)`` in m s-1;
        state arrays are ``(station,)`` with units m, m s-1, m, m, m, K, and
        W m-2. Missing/calm winds use the neutral fallback state.

    Raises:
        ValueError: If station shapes disagree, ``mixing_method`` or
            ``ipsifcn`` is unsupported, or downstream profile inputs are
            invalid.
    """
    base=neutral_water_station_profiles(u,v,anemometer_height_m,
        current_mixing_height_m,coriolis,z_centers_m,z_faces_m,station_latitude,
        constw=constw,minimum_height_m=minimum_height_m,
        maximum_height_m=maximum_height_m,ipsifcn=ipsifcn)
    output_u,output_v=(base[0].copy(),base[1].copy())
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    station_inputs=tuple(np.asarray(value,dtype=MODEL_DTYPE) for value in (
        anemometer_height_m,air_temperature_k,air_minus_sea_temperature_k,
        relative_humidity_percent,observed_mixing_height_m,current_mixing_height_m,
        coriolis,potential_lapse_k_m,previous_temperature_jump_k,station_latitude))
    if any(value.shape!=u.shape for value in station_inputs):
        raise ValueError('WATER2 DeltaT station inputs must have identical shapes')
    (anemometer,air,delta,rh,observed,current,coriolis_value,lapse,
     previous_jump,latitude)=station_inputs
    speed=np.sqrt(u*u+v*v)
    active=(u<=990.)&(v<=990.)&(speed>=.0001)&(delta<990.)
    state: list[Float32Array] = [np.array(value,copy=True) for value in base[2:]]
    # DeltaT-specific state: convective height, jump, and buoyancy heat flux.
    state.extend(np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(3))
    if not active.any(): return cast(FloatNine,(output_u,output_v,*state))
    height=anemometer[active]
    thermal_height=(height if temperature_measurement_height_m is None else
        np.broadcast_to(np.asarray(temperature_measurement_height_m,dtype=MODEL_DTYPE),u.shape)[active])
    selected_air=np.where(air[active]<990.,air[active],288.7)
    selected_rh=np.where(rh[active]<990.,rh[active],100.)
    sea=(selected_air-delta[active]
         if sea_temperature_k is None else
         np.broadcast_to(np.asarray(sea_temperature_k,dtype=MODEL_DTYPE),u.shape)[active])
    theta,difference,difference_10m,theta_10m=overwater_delta_temperature_state(
        selected_air,sea,selected_rh,thermal_height)
    initial_z0=hosker_roughness(speed[active])
    rescale=np.trunc(height).astype(np.int32)!=10
    speed10=speed[active].copy()
    speed10[rescale]=(speed10[rescale]*
        np.log(10./initial_z0[rescale])/
        np.log(height[rescale]/initial_z0[rescale]))
    roughness=initial_z0.copy()
    roughness[rescale]=2.e-6*np.power(speed10[rescale],2.5)
    length,friction,flux=delta_temperature_flux(
        speed10,thermal_height,difference,difference_10m,theta_10m,theta,roughness)
    mechanical=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    has_observed=observed[active]<990.
    initial=np.where(has_observed,observed[active],mechanical)
    initial=np.maximum(np.minimum(initial,current[active]),minimum_height_m)
    convective=np.zeros_like(initial); next_jump=previous_jump[active].copy()
    buoyancy=flux/1195.2
    options: dict[str, MixingOption] = dict(mixing_options)
    if int(mixing_method)==1:
        convective,next_jump=cast(Callable[..., FloatPair],
            convective_height_maul_carson)(
            initial,previous_jump[active],buoyancy,lapse[active],
            threshold_flux_per_m=threshold_flux_per_m,
            maximum_height_m=maximum_height_m,**options)
    elif int(mixing_method)==2:
        convective=cast(Callable[..., Float32Array],
            convective_height_batchvarova_gryning)(
            initial,buoyancy,lapse[active],friction,length,selected_air,
            minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
            threshold_flux_per_m=threshold_flux_per_m,**options)
    elif int(mixing_method) not in {-2,-1}:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    calculated=np.clip(np.maximum(mechanical,convective),minimum_height_m,
                       maximum_height_m)
    mixing=np.where(has_observed,initial,calculated)
    convective=np.where(has_observed,np.nan,convective)
    next_jump=np.where(has_observed,np.nan,next_jump)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],height,z_centers_m,z_faces_m,roughness,length,mixing,
        latitude[active],ipsifcn=ipsifcn,minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    sources=(roughness,friction,length,mixing,convective,next_jump,flux)
    for destination,source in zip(state,sources,strict=True): destination[active]=source
    return cast(FloatNine,(output_u,output_v,*state))


def coare_water_station_profiles(u: ArrayLike,v: ArrayLike,anemometer_height_m: ArrayLike,air_sensor_height_m: ArrayLike,
        air_temperature_k: ArrayLike,air_minus_sea_temperature_k: ArrayLike,relative_humidity_percent: ArrayLike,
        observed_mixing_height_m: ArrayLike,current_mixing_height_m: ArrayLike,coriolis: ArrayLike,
        potential_lapse_k_m: ArrayLike,previous_temperature_jump_k: ArrayLike,z_centers_m: ArrayLike,z_faces_m: ArrayLike,
        station_latitude: ArrayLike,shortwave_down_w_m2: ArrayLike,longwave_down_w_m2: ArrayLike,rainfall_mm_h: ArrayLike,
        *,mixing_method: int=1,coare_method: int=10,wave_period_s: ArrayLike=-999.,wave_height_m: ArrayLike=-999.,
        coastal_shelf_length_km: ArrayLike=0.,distance_to_coast_km: ArrayLike=0.,cool_skin: bool=False,
        constw: float=.16,minimum_height_m: float=50.,maximum_height_m: float=2500.,
        threshold_flux_per_m: float=.05,ipsifcn: int=0,sea_temperature_k: ArrayLike | None=None,
        warm_layer_state: WarmLayerState | None=None,solar_time_s: ArrayLike | None=None,water_sensor_depth_m: ArrayLike=.6,
        **mixing_options: MixingOption) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, WarmLayerState | None]:
    """Run WATER2's COARE branch through mixing-height and SIMILT steps.

    Args:
        u: One-dimensional station eastward wind in m s-1.
        v: One-dimensional station northward wind in m s-1.
        anemometer_height_m: Wind measurement heights in metres.
        air_sensor_height_m: Air temperature/humidity sensor heights in metres.
        air_temperature_k: Station air temperatures in kelvins.
        air_minus_sea_temperature_k: Air-minus-sea temperature in kelvins.
        relative_humidity_percent: Station relative humidity in percent.
        observed_mixing_height_m: Observed mixing heights in metres; at least
            990 denotes missing.
        current_mixing_height_m: Current grid mixing heights in metres.
        coriolis: Absolute Coriolis parameter in s-1.
        potential_lapse_k_m: Potential-temperature lapse in K m-1.
        previous_temperature_jump_k: Previous inversion jump in kelvins.
        z_centers_m: One-dimensional CALMET layer centres in metres.
        z_faces_m: One-dimensional CALMET layer faces in metres.
        station_latitude: Station latitude in degrees north.
        shortwave_down_w_m2: Downward short-wave radiation in W m-2.
        longwave_down_w_m2: Downward long-wave radiation in W m-2.
        rainfall_mm_h: Rainfall rate in mm h-1.
        mixing_method: Mixing-height selector ``-2``, ``-1``, 1, or 2.
        coare_method: COARE selector ``+/-10``, ``+/-11``, or ``+/-12``.
        wave_period_s: Wave period in seconds for negative method selectors.
        wave_height_m: Significant wave height in metres for negative methods.
        coastal_shelf_length_km: Shelf length supplied to COARE in kilometres.
        distance_to_coast_km: Distance to coast in kilometres, scalar or by
            station.
        cool_skin: Whether to enable COARE cool-skin physics.
        constw: OCD overwater mechanical-height coefficient.
        minimum_height_m: Lower overwater mixing-height bound in metres.
        maximum_height_m: Upper overwater mixing-height bound in metres.
        threshold_flux_per_m: Convective threshold coefficient in W m-3.
        ipsifcn: SIMILT stability function selector, 0 or 1.
        sea_temperature_k: Optional explicit sea temperature in kelvins.
        warm_layer_state: Optional COARE warm-layer state, either station-sized
            or already selected.
        solar_time_s: Local solar seconds required when warm-layer state is
            supplied.
        water_sensor_depth_m: Water temperature sensor depth in metres.
        **mixing_options: Additional keyword options passed to the selected
            convective-height routine.

    Returns:
        Ten items: nine arrays matching the DELTAT return layout
        ``(u_profile, v_profile, roughness, friction, monin_obukhov,
        mixing_height, convective_height, temperature_jump,
        buoyancy_heat_flux)`` followed by updated warm-layer state or ``None``.
        Arrays are float32; profiles are ``(level, station)`` and state fields
        are ``(station,)`` in their documented physical units.

    Raises:
        ValueError: If ``coare_method`` or ``mixing_method`` is unsupported,
            station shapes disagree, warm-layer time is missing, or downstream
            COARE/SIMILT inputs are invalid.
    """
    from .coare import (WarmLayerState,advance_warm_layer,surface_flux,
                        update_warm_layer_fluxes)

    if abs(int(coare_method)) not in {10,11,12}:
        raise ValueError('COARE station method must be +/-10, +/-11, or +/-12')
    base=neutral_water_station_profiles(u,v,anemometer_height_m,
        current_mixing_height_m,coriolis,z_centers_m,z_faces_m,station_latitude,
        constw=constw,minimum_height_m=minimum_height_m,
        maximum_height_m=maximum_height_m,ipsifcn=ipsifcn)
    output_u,output_v=base[0].copy(),base[1].copy()
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    named=(anemometer_height_m,air_sensor_height_m,air_temperature_k,
        air_minus_sea_temperature_k,relative_humidity_percent,
        observed_mixing_height_m,current_mixing_height_m,coriolis,
        potential_lapse_k_m,previous_temperature_jump_k,station_latitude,
        shortwave_down_w_m2,longwave_down_w_m2,rainfall_mm_h)
    values=tuple(np.asarray(value,dtype=MODEL_DTYPE) for value in named)
    if any(value.shape!=u.shape for value in values):
        raise ValueError('COARE WATER2 station inputs must have identical shapes')
    (anemometer,air_height,air,delta,rh,observed,current,coriolis_value,lapse,
     previous_jump,latitude,shortwave,longwave,rainfall)=values
    speed=np.sqrt(u*u+v*v)
    active=(u<=990.)&(v<=990.)&(speed>=.0001)&(delta<990.)
    state: list[Float32Array] = [np.array(value,copy=True) for value in base[2:]]
    state.extend(np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(3))
    if not active.any():
        return cast("CoareStationResult",(output_u,output_v,*state,warm_layer_state))
    selected_air=np.where(air[active]<990.,air[active],288.7)
    selected_rh=np.where(rh[active]<990.,rh[active],100.)
    sea=(selected_air-delta[active]
         if sea_temperature_k is None else
         np.broadcast_to(np.asarray(sea_temperature_k,dtype=MODEL_DTYPE),u.shape)[active])
    sea=np.maximum(sea,270.15)
    next_warm_state: WarmLayerState | None = warm_layer_state
    if warm_layer_state is not None:
        if solar_time_s is None:
            raise ValueError('COARE warm-layer station profiles require solar_time_s')
        full_warm_state=np.shape(warm_layer_state.initialized)==u.shape
        selected_warm_state: WarmLayerState = (WarmLayerState(*(
            np.asarray(getattr(warm_layer_state,name))[active]
            for name in WarmLayerState.__dataclass_fields__))
            if full_warm_state else warm_layer_state)
        solar=np.broadcast_to(np.asarray(solar_time_s,dtype=float),u.shape)[active]
        depth=np.broadcast_to(np.asarray(water_sensor_depth_m,dtype=float),u.shape)[active]
        sea,selected_warm_state,_=advance_warm_layer(
            selected_warm_state,solar,sea,depth,
            shortwave[active],longwave[active])
    saturation=saturation_vapor_pressure_lowe(selected_air-273.15)
    vapor=.01*selected_rh*saturation
    ratio=.622*vapor/(1000.-vapor)
    humidity=ratio/(1.+ratio)*1000.
    neutral_friction=base[3][active]
    mechanical=water_mechanical_height(neutral_friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    has_observed=observed[active]<990.
    initial=np.where(has_observed,observed[active],mechanical)
    initial=np.maximum(np.minimum(initial,current[active]),minimum_height_m)
    flux=surface_flux(speed[active],selected_air,sea,humidity,
        wind_height_m=anemometer[active],temperature_height_m=air_height[active],
        pressure_mb=1008.,boundary_layer_height_m=initial,
        shortwave_down_w_m2=shortwave[active],longwave_down_w_m2=longwave[active],
        rainfall_mm_h=rainfall[active],wave_method=abs(int(coare_method))-10,
        wave_period_s=(np.broadcast_to(wave_period_s,u.shape)[active]
                       if int(coare_method)<0 else -999.),
        wave_height_m=(np.broadcast_to(wave_height_m,u.shape)[active]
                       if int(coare_method)<0 else -999.),
        coastal_shelf_length_km=coastal_shelf_length_km,
        distance_to_coast_km=np.broadcast_to(distance_to_coast_km,u.shape)[active],
        cool_skin=cool_skin)
    if next_warm_state is not None:
        selected_warm_state=update_warm_layer_fluxes(selected_warm_state,flux)
        if full_warm_state:
            merged: list[WarmStateArray] = []
            for name in WarmLayerState.__dataclass_fields__:
                field_values: WarmStateArray = np.array(
                    getattr(warm_layer_state,name),copy=True)
                field_values[active]=getattr(selected_warm_state,name)
                merged.append(field_values)
            next_warm_state=cast(Callable[..., WarmLayerState],WarmLayerState)(*merged)
        else:
            next_warm_state=selected_warm_state
    roughness=np.asarray(flux.roughness_m,dtype=MODEL_DTYPE)
    friction=np.asarray(flux.friction_velocity_m_s,dtype=MODEL_DTYPE)
    length=np.asarray(flux.monin_obukhov_m,dtype=MODEL_DTYPE)
    length=np.where(np.abs(length)<5.,np.where(length<0.,-5.,5.),length)
    buoyancy=np.asarray(flux.buoyancy_flux_k_m_s,dtype=MODEL_DTYPE)
    mechanical=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    convective=np.zeros_like(initial); next_jump=previous_jump[active].copy()
    options: dict[str, MixingOption] = dict(mixing_options)
    if int(mixing_method)==1:
        convective,next_jump=cast(Callable[..., FloatPair],
            convective_height_maul_carson)(
            initial,previous_jump[active],buoyancy,lapse[active],
            threshold_flux_per_m=threshold_flux_per_m,
            maximum_height_m=maximum_height_m,**options)
    elif int(mixing_method)==2:
        convective=cast(Callable[..., Float32Array],
            convective_height_batchvarova_gryning)(
            initial,buoyancy,lapse[active],friction,length,selected_air,
            minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
            threshold_flux_per_m=threshold_flux_per_m,**options)
    elif int(mixing_method) not in {-2,-1}:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    calculated=np.clip(np.maximum(mechanical,convective),minimum_height_m,
                       maximum_height_m)
    mixing=np.where(has_observed,initial,calculated)
    convective=np.where(has_observed,np.nan,convective)
    next_jump=np.where(has_observed,np.nan,next_jump)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],anemometer[active],z_centers_m,z_faces_m,roughness,length,
        mixing,latitude[active],ipsifcn=ipsifcn,
        minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    buoyancy_flux=buoyancy*1195.2
    for destination,source in zip(state,(roughness,friction,length,mixing,convective,
            next_jump,buoyancy_flux),strict=True): destination[active]=source
    return cast("CoareStationResult",(output_u,output_v,*state,next_warm_state))


def upwind_mixing_average(u: ArrayLike, v: ArrayLike, mixing_height: ArrayLike, convective_height: ArrayLike, dx_m: float | None=None,
                          dy_m: float | None=None, *, grid_spacing_m: float | None=None,
                          minimum_radius_cells: int=10, half_angle_degrees: float=30.,
                          timestep_s: float=3600.) -> tuple[Float32Array, Float32Array]:
    """Apply AVEMIX upwind-cone averaging to mixing-height grids.

    Args:
        u: Two-dimensional eastward transport wind in m s-1.
        v: Two-dimensional northward transport wind in m s-1.
        mixing_height: Two-dimensional total mixing height in metres.
        convective_height: Two-dimensional convective mixing height in metres.
        dx_m: Positive grid spacing along x in metres. When omitted,
            ``grid_spacing_m`` supplies the legacy square-cell spacing.
        dy_m: Positive grid spacing along y in metres. Defaults to ``dx_m``.
        grid_spacing_m: Legacy alias for ``dx_m``; mutually exclusive with it.
        minimum_radius_cells: Nonnegative square half-width always included
            around each target, in grid cells.
        half_angle_degrees: Upwind cone half-angle in ``[0, 90)`` degrees.
        timestep_s: Positive trajectory duration in seconds.

    Returns:
        Pair of float32 ``(y, x)`` arrays containing averaged total and
        convective mixing heights in metres. Inputs are not modified.

    Raises:
        TypeError: If both ``dx_m`` and ``grid_spacing_m`` are supplied.
        ValueError: If field shapes disagree, spacings or time step are not
            positive finite values, the cone/radius is invalid, or winds are
            non-finite.

    Notes:
        Candidate geometry is evaluated in physical metres, supporting
        rectangular cells. The target-cell loop is retained because each wind
        vector defines a different search window.
    """
    u=np.asarray(u,dtype=np.float64); v=np.asarray(v,dtype=np.float64)
    zi=np.asarray(mixing_height,dtype=np.float64); zic=np.asarray(convective_height,dtype=np.float64)
    if u.shape!=v.shape or u.shape!=zi.shape or u.shape!=zic.shape or u.ndim!=2: raise ValueError('AVEMIX fields must share a two-dimensional grid')
    ny,nx=u.shape; result=np.empty_like(zi); result_convective=np.empty_like(zic)
    if dx_m is None:
        dx_m=grid_spacing_m
    elif grid_spacing_m is not None:
        raise TypeError('pass dx_m or grid_spacing_m, not both')
    if dy_m is None: dy_m=dx_m
    dx_m=float(cast(float,dx_m)); dy_m=float(cast(float,dy_m)); timestep_s=float(timestep_s)
    if (not np.isfinite(timestep_s) or timestep_s<=0. or
            not np.isfinite(dx_m) or dx_m<=0. or
            not np.isfinite(dy_m) or dy_m<=0.):
        raise ValueError('AVEMIX timestep and grid spacing must be finite and positive')
    if (not np.isfinite(half_angle_degrees) or half_angle_degrees<0. or
            half_angle_degrees>=90.):
        raise ValueError('AVEMIX half angle must be finite and in [0, 90) degrees')
    if minimum_radius_cells<0:
        raise ValueError('AVEMIX minimum radius must be nonnegative')
    radius=int(minimum_radius_cells)
    tangent=np.tan(np.deg2rad(half_angle_degrees))
    # The area-equivalent cell width supplies the one-cell softening distance.
    # For square cells every weight is the old cell-unit weight times the same
    # constant, so normalized square-grid results retain their scale.
    softening_m=np.sqrt(dx_m)*np.sqrt(dy_m)
    @lru_cache(maxsize=4096)
    def window_geometry(left: int,right: int,down: int,up: int) -> tuple[Float64Array, Float64Array, BoolArray, BoolArray]:
        """Build and cache relative coordinates for a clipped search window.

        Args:
            left: Number of candidate columns left of the target.
            right: Number of candidate columns right of the target.
            down: Number of candidate rows below the target.
            up: Number of candidate rows above the target.

        Returns:
            Tuple ``(rx, ry, local, square)``. ``rx`` and ``ry`` are broadcast
            float64 offsets in metres; ``local`` marks the target cell and
            ``square`` marks the always-included minimum-radius window.
        """
        rx_integer=np.arange(left,-right-1,-1,dtype=np.int64)[None,:]
        ry_integer=np.arange(down,-up-1,-1,dtype=np.int64)[:,None]
        rx=rx_integer.astype(COORDINATE_DTYPE)*dx_m
        ry=ry_integer.astype(COORDINATE_DTYPE)*dy_m
        local=(rx_integer==0)&(ry_integer==0)
        square=(np.abs(rx_integer)<=radius)&(np.abs(ry_integer)<=radius)
        return rx,ry,local,square
    for i in range(nx):
        for j in range(ny):
            etax=u[j,i]*timestep_s; etay=v[j,i]*timestep_s
            if not np.isfinite(etax) or not np.isfinite(etay):
                raise ValueError('AVEMIX wind fields must be finite')
            eta2=etax*etax+etay*etay
            eta=np.sqrt(eta2); inverse_eta=1./eta if eta2>0. else 0.
            # Candidate displacement is target minus source.  The cone is
            # bounded by the wind trajectory and its two perpendicular end
            # points; convert only the window bounds back to array indices.
            perpendicular_x=-etay*tangent
            perpendicular_y=etax*tangent
            end_ax=etax+perpendicular_x; end_bx=etax-perpendicular_x
            end_ay=etay+perpendicular_y; end_by=etay-perpendicular_y
            candidate_x=(i,i-end_ax/dx_m,i-end_bx/dx_m)
            candidate_y=(j,j-end_ay/dy_m,j-end_by/dy_m)
            xlo=max(0,min(int(np.floor(min(candidate_x))),i-radius))
            xhi=min(nx-1,max(int(np.ceil(max(candidate_x))),i+radius))
            ylo=max(0,min(int(np.floor(min(candidate_y))),j-radius))
            yhi=min(ny-1,max(int(np.ceil(max(candidate_y))),j+radius))
            # The window is rectangular, so relative one-dimensional axes
            # broadcast to every expression below.  Constructing ``mgrid``
            # here formerly allocated two full candidate arrays for every
            # one of ~300k targets and dominated the model runtime.
            # Cache geometry shared by targets with the same clipped window
            # extents; only wind-dependent cone and weight expressions remain
            # per target.
            geometry=window_geometry(i-xlo,xhi-i,j-ylo,yhi-j)
            rx,ry,local,square=geometry
            distance2=rx*rx+ry*ry
            if eta>0.:
                projected=(rx*etax+ry*etay)*inverse_eta
                across=np.abs(rx*etay-ry*etax)*inverse_eta
                tolerance=np.finfo(np.float64).eps*max(eta,dx_m,dy_m)*16.
                cone=((projected>=-tolerance)&
                      (distance2<=eta2+tolerance*max(eta,1.))&
                      (across<=projected*tangent+tolerance))
                along=np.where(projected<0.,eta-projected,projected)
            else:
                cone=np.zeros_like(local)
                along=np.sqrt(distance2)
                across=np.zeros_like(along)
            include=local|square|cone
            displaced=softening_m+across
            denominator=along*along+displaced*displaced
            weight=1./denominator
            weight=np.where(local,1./(softening_m*softening_m),weight)
            weight=np.where(include,weight,0.)
            denominator=np.sum(weight,dtype=ACCUMULATION_DTYPE)
            result[j,i]=np.sum(
                weight*zi[ylo:yhi+1,xlo:xhi+1],dtype=ACCUMULATION_DTYPE)/denominator
            result_convective[j,i]=np.sum(
                weight*zic[ylo:yhi+1,xlo:xhi+1],
                dtype=ACCUMULATION_DTYPE)/denominator
    return _model_array(result),_model_array(result_convective)


def mesopac_layer_average(u: ArrayLike, v: ArrayLike, mixing_height_m: ArrayLike, z_faces_m: ArrayLike, *,
                          upper_top_m: float=3000., minimum_upper_depth_m: float=300.) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Vectorized PACAVE lower- and upper-layer wind averages.

    The lower layer extends from the ground to ``max(ZI, 10 m)``.  The
    upper layer extends from ZI to the greater of 3000 m or ZI + 300 m,
    matching the fixed arguments used by CALMET's OUTPC path.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        mixing_height_m: Mixing height with shape ``(y, x)`` in metres.
        z_faces_m: Increasing layer-face heights with shape ``(level + 1,)``
            in metres above ground level.
        upper_top_m: Minimum top of the upper averaging layer in metres.
        minimum_upper_depth_m: Minimum upper-layer depth above the local
            mixing height in metres.

    Returns:
        Tuple of four float32 arrays ``(lower_u, lower_v, upper_u, upper_v)``.
        Every array has shape ``(y, x)`` and units m s-1. Inputs are not
        modified.

    Raises:
        ValueError: If wind, mixing-height, or face dimensions disagree, if
            the faces are not strictly increasing, or if either averaging
            interval extends above the highest face.
    """
    u=np.asarray(u,dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)
    htmix=np.asarray(mixing_height_m,dtype=np.float64)
    zi=np.maximum(htmix,np.float64(10.))
    faces=np.asarray(z_faces_m,dtype=np.float64)
    if u.shape!=v.shape or u.ndim!=3 or zi.shape!=u.shape[1:]:
        raise ValueError('PACAVE winds must be (level,y,x) and match mixing height')
    if faces.ndim!=1 or faces.size!=u.shape[0]+1 or np.any(np.diff(faces)<=0.):
        raise ValueError('PACAVE requires one increasing face above and below each wind layer')
    top=np.maximum.reduce((np.full_like(htmix,np.float64(upper_top_m)),
                           np.full_like(htmix,np.float64(20.)),
                           htmix+np.float64(minimum_upper_depth_m)))
    if np.any(zi>faces[-1]):
        raise ValueError('PACAVE mixing height exceeds the top CALMET face')
    if np.any(top>faces[-1]):
        raise ValueError('PACAVE upper averaging height exceeds the top CALMET face')
    bottom=faces[:-1,None,None]
    ceiling=faces[1:,None,None]
    lower_depth=np.maximum(np.minimum(ceiling,zi)-bottom,np.float64(0.))
    upper_depth=np.maximum(np.minimum(ceiling,top)-np.maximum(bottom,zi),np.float64(0.))

    def average(values: Float64Array,depth: Float64Array) -> Float64Array:
        """Average one wind component using per-layer overlap depths.

        Args:
            values: Wind values with shape ``(level, y, x)`` in m s-1.
            depth: Included depth per layer with shape ``(level, y, x)`` in
                metres.

        Returns:
            A float64 array with shape ``(y, x)`` containing the
            depth-weighted mean wind in m s-1. Neither input is modified.
        """
        numerator=np.cumsum(np.asarray(values*depth,dtype=np.float64),
                            axis=0,dtype=np.float64)[-1]
        denominator=np.cumsum(depth,axis=0,dtype=np.float64)[-1]
        return np.asarray(numerator/denominator,dtype=np.float64)

    lower_u=average(u,lower_depth); lower_v=average(v,lower_depth)
    upper_u=average(u,upper_depth); upper_v=average(v,upper_depth)
    # PACAVE bypasses arithmetic when ZI and ZZTOP lie in one model layer.
    # ``side='left'`` reproduces its ``ZFACE(K+1) .LT. ZI`` comparison.
    ksav=np.searchsorted(faces[1:],zi,side='left')
    direct=np.take(faces[1:],ksav)>top
    layer_u=np.take_along_axis(u,ksav[None,:,:],axis=0)[0]
    layer_v=np.take_along_axis(v,ksav[None,:,:],axis=0)[0]
    upper_u=np.where(direct,layer_u,upper_u).astype(np.float64)
    upper_v=np.where(direct,layer_v,upper_v).astype(np.float64)
    return cast(FloatQuad,tuple(_model_array(value) for value in
                 (lower_u,lower_v,upper_u,upper_v)))

def prognostic_temperature_3d(prognostic_temperature_k: ArrayLike, surface_temperature_k: ArrayLike,
                              convective_height_m: ArrayLike, z_faces_m: ArrayLike) -> Float32Array:
    """Blend prognostic temperatures through the convective-layer boundary.

    The lowest CALMET layer remains anchored to the final SURFVAR/AVETMP
    temperature.  Each layer aloft receives a dry-adiabatic temperature in
    proportion to the layer thickness below the convective-layer top.  A
    layer wholly below the top is therefore fully mixed, a layer wholly above
    it retains its prognostic value, and a layer intersected by the top varies
    continuously as that top moves through the layer.

    Args:
        prognostic_temperature_k: Prognostic temperature with shape
            ``(level, y, x)`` in kelvins.
        surface_temperature_k: Final surface temperature with shape ``(y, x)``
            in kelvins.
        convective_height_m: Convective mixing height with shape ``(y, x)``
            in metres above ground level.
        z_faces_m: Increasing layer-face heights with shape ``(level + 1,)``
            in metres above ground level.

    Returns:
        A new float32 array with shape ``(level, y, x)`` in kelvins. The
        lowest layer equals ``surface_temperature_k`` and higher intersected
        layers contain an overlap-weighted blend.

    Raises:
        ValueError: If the fields do not share CALMET grid dimensions or the
            face vector is not finite and strictly increasing.
    """
    source=np.asarray(prognostic_temperature_k,dtype=MODEL_DTYPE)
    surface=np.asarray(surface_temperature_k,dtype=MODEL_DTYPE)
    zic=np.asarray(convective_height_m,dtype=MODEL_DTYPE)
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    if (source.ndim!=3 or source.shape[0]<1 or surface.ndim!=2 or
            source.shape[1:]!=surface.shape or surface.shape!=zic.shape):
        raise ValueError('TEMP3D inputs do not share CALMET dimensions')
    if (faces.ndim!=1 or faces.size!=source.shape[0]+1 or
            not np.all(np.isfinite(faces)) or np.any(np.diff(faces)<=0.)):
        raise ValueError(
            'TEMP3D requires one increasing face above and below each layer')
    result=source.copy(); result[0]=surface
    if source.shape[0]>1:
        centers=(faces[:-1]+faces[1:])*np.float32(.5)
        lower=faces[1:-1,None,None]
        upper=faces[2:,None,None]
        fraction=np.clip(
            (zic[None,:,:]-lower)/(upper-lower),np.float32(0.),np.float32(1.))
        adiabatic=(surface[None,:,:]-np.float32(.0098)*
                   (centers[1:,None,None]-centers[0]))
        blended=(fraction*adiabatic+(np.float32(1.)-fraction)*source[1:])
        result[1:]=np.where(fraction<=0.,source[1:],
                            np.where(fraction>=1.,adiabatic,blended))
    return result

def pgt_stability(u10: ArrayLike, v10: ArrayLike, solar_elevation_sine: ArrayLike, cloud_fraction_field: ArrayLike,
                  ceiling_height_m: ArrayLike, landuse: ArrayLike, monin_obukhov_m: ArrayLike, water_range: WaterRange=(50,55)) -> Int8Array:
    """Diagnose Pasquill-Gifford-Turner stability over land and water.

    Args:
        u10: Eastward 10 m wind in m s-1, broadcastable to the surface grid.
        v10: Northward 10 m wind in m s-1, broadcastable to the surface grid.
        solar_elevation_sine: Sine of solar elevation, dimensionless and
            broadcastable to the surface grid.
        cloud_fraction_field: Cloud fraction from 0 to 1, broadcastable to
            the surface grid.
        ceiling_height_m: Cloud-ceiling height in metres, broadcastable to
            the surface grid.
        landuse: Integer land-use categories on the surface grid.
        monin_obukhov_m: Monin-Obukhov length in metres, broadcastable to the
            surface grid and used for water cells.
        water_range: Inclusive pair of land-use category numbers treated as
            water.

    Returns:
        An int8 array on the broadcast surface-grid shape. Values 1 through 6
        represent stability classes A through F. Inputs are not modified.
    """
    u=np.asarray(u10,dtype=MODEL_DTYPE); v=np.asarray(v10,dtype=MODEL_DTYPE)
    sun=np.asarray(solar_elevation_sine,dtype=MODEL_DTYPE)
    cloud=np.asarray(cloud_fraction_field,dtype=MODEL_DTYPE)
    ceiling=np.asarray(ceiling_height_m,dtype=MODEL_DTYPE)
    lu=np.asarray(landuse); length=np.asarray(monin_obukhov_m,dtype=MODEL_DTYPE)
    jcc=np.trunc(10.*cloud+.5).astype(int)
    jceil=np.trunc(ceiling/30.48).astype(int)
    nighttime=sun<=0.; insolation=np.select(
        (sun<=.258819,sun<=.5735764,sun<=.8660254),
        (1,2,3),default=4)
    category=np.where(nighttime,np.where(jcc<=4,-2,np.where((jcc==10)&(jceil<70),0,-1)),insolation)
    heavy=(~nighttime)&(jcc>5)
    opaque=jcc==10
    adjusted_opaque=np.where(jceil<70,0,np.where(jceil<160,np.maximum(insolation-2,1),np.maximum(insolation-1,1)))
    adjusted_other=np.where(jceil<70,np.maximum(insolation-2,1),np.where(jceil<160,np.maximum(insolation-1,1),insolation))
    category=np.where(heavy,np.where(opaque,adjusted_opaque,adjusted_other),category)
    table=np.array([6,6,6,6,6,6,5,5,5,5,4,4, 6,6,6,5,5,5,4,4,4,4,4,4,
        4,4,4,4,4,4,4,4,4,4,4,4, 3,3,3,4,4,4,4,4,4,4,4,4,
        2,2,2,3,3,3,3,3,3,4,4,4, 1,2,2,2,2,2,2,3,3,3,3,4,
        1,1,1,1,1,2,2,2,2,3,3,3],dtype=np.int8).reshape((12,7),order='F')
    speed=np.sqrt(u*u+v*v)
    knots=np.clip(np.trunc(1.9438*speed+.5).astype(int),1,12)
    result=table[knots-1,category+2]
    water=(lu>=water_range[0])&(lu<=water_range[1])
    water_class=np.where(np.abs(length)>25.,4,
        np.where(length<-10.,3,np.where(length<0.,2,
        np.where(length<=10.,6,5))))
    return np.where(water,water_class,result).astype(np.int8)

def convective_velocity_scale(convective_height_m: ArrayLike, sensible_heat_flux: ArrayLike, temperature_k: ArrayLike,
                              density: ArrayLike, landuse: ArrayLike, water_range: WaterRange=(50,55)) -> Float32Array:
    """Calculate the convective velocity scale used by WATER/WSTARR.

    Args:
        convective_height_m: Convective mixing height with shape ``(y, x)``
            in metres.
        sensible_heat_flux: Surface heat flux with shape ``(y, x)``. Land
            values are in W m-2; water values use CALMET's stored
            ``1195.2 * w'theta_v'`` convention in W m-2.
        temperature_k: Surface air temperature, broadcastable to ``(y, x)``,
            in kelvins.
        density: Air density, broadcastable to ``(y, x)``, in kg m-3.
        landuse: Integer land-use categories with shape ``(y, x)``.
        water_range: Inclusive pair of land-use category numbers treated as
            water.

    Returns:
        A float32 array with shape ``(y, x)`` containing convective velocity
        scale in m s-1. Cells with non-positive heat flux or height are zero;
        inputs are not modified.

    Raises:
        ValueError: If ``landuse`` and ``convective_height_m`` have different
            shapes.
    """
    zi=np.asarray(convective_height_m,dtype=MODEL_DTYPE)
    qh=np.asarray(sensible_heat_flux,dtype=MODEL_DTYPE)
    temp=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    rho=np.asarray(density,dtype=MODEL_DTYPE)
    lu=np.asarray(landuse)
    if lu.shape!=zi.shape:
        raise ValueError('WSTARR landuse does not match the surface fields')
    water=(lu>=water_range[0])&(lu<=water_range[1])
    land_argument=9.81*qh*zi/(temp*rho*996.)
    # WATER stores RH0CP*<w'theta_v'> in QH and forms WSTAR before WSTARR,
    # which deliberately skips water cells.  RH0CP is the fixed 1195.2.
    water_argument=9.81*(qh/1195.2)*zi/temp
    argument=np.where(water,water_argument,land_argument)
    value=np.cbrt(np.maximum(argument,0.))
    return _model_array(np.where((qh>0.)&(zi>0.),value,0.))

def solar_sine(julian_day: int, base_timezone: float, latitude: ArrayLike, longitude_east: ArrayLike) -> Float32Array:
    """Calculate CALMET's half-hour solar-elevation sine fields.

    Args:
        julian_day: One-based day of year.
        base_timezone: Base time-zone offset used by CALMET, in hours west of
            UTC.
        latitude: Latitude field with shape ``(y, x)`` in degrees north.
        longitude_east: Longitude field with shape ``(y, x)`` in degrees east.

    Returns:
        A float32 array with shape ``(26, y, x)`` containing the dimensionless
        sine of solar elevation at half-hour samples from 00:30 before through
        00:30 after the modeled day. Inputs are not modified.
    """
    lat=np.asarray(latitude,dtype=COORDINATE_DTYPE)
    lon=np.asarray(longitude_east,dtype=COORDINATE_DTYPE)
    day=COORDINATE_DTYPE(julian_day)
    timezone=COORDINATE_DTYPE(base_timezone)
    d=(day-1.)*.9856479
    radd=np.deg2rad(d)
    sd=np.sin(radd); cd=np.cos(radd)
    rad2d=2.*radd
    s2=np.sin(rad2d); c2=np.cos(rad2d)
    em=12.+.12357*sd-.004289*cd+.153809*s2+.060783*c2
    sigma=279.9348+d+1.914827*sd-.079525*cd+.019938*s2-.00162*c2
    sincd=.39784989*np.sin(np.deg2rad(sigma))
    capd=np.arcsin(sincd); coscd=np.cos(capd)
    radlat=np.deg2rad(lat)
    sinlat=np.sin(radlat); coslat=np.cos(radlat)
    hours=np.arange(26,dtype=COORDINATE_DTYPE)-.5+timezone
    solha=15.*(hours[:,None,None]-em)+lon[None,:,:]
    result=sinlat[None,:,:]*sincd+coslat[None,:,:]*coscd*np.cos(
        np.deg2rad(solha))
    return _model_array(result)

def terrain_adjust_surface_wind(u: ArrayLike, v: ArrayLike, terrain_m: ArrayLike, top_height_m: float, dx_m: float, dy_m: float,
                                *, maximum_iterations: int=75, tolerance: float=.02,
                                relaxation: float=1.5) -> tuple[Float32Array, Float32Array]:
    """ADJUST terrain-potential correction for objective-analysis winds.

    CALMET applies this Poisson solve only to the first wind layer when
    ``IWFCOD=0``.  The four alternating Gauss-Seidel sweeps are deliberately
    retained because every sweep immediately reuses updated neighbours.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        terrain_m: Terrain elevation with shape ``(y, x)`` in metres.
        top_height_m: Model-domain top height in metres.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.
        maximum_iterations: Maximum number of four-sweep solver iterations.
        tolerance: Dimensionless maximum relative potential-change criterion.
        relaxation: Dimensionless successive-over-relaxation factor.

    Returns:
        Tuple ``(u_adjusted, v_adjusted)`` of new float32 arrays with shape
        ``(level, y, x)`` in m s-1. Only level zero is adjusted; the input
        arrays are not modified.

    Raises:
        ValueError: If the wind arrays are not matching ``(level, y, x)``
            fields, terrain does not match their horizontal grid, or a height
            or grid spacing is non-positive.
    """
    work_dtype=ACCUMULATION_DTYPE
    u=np.array(u,dtype=work_dtype,copy=True); v=np.array(v,dtype=work_dtype,copy=True)
    terrain=np.asarray(terrain_m,dtype=work_dtype)
    if u.shape!=v.shape or u.ndim!=3 or u.shape[1:]!=terrain.shape:
        raise ValueError('wind must have shape (z,y,x) matching terrain')
    if top_height_m<=0. or dx_m<=0. or dy_m<=0.:
        raise ValueError('ADJUST heights and grid spacing must be positive')
    west_h=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east_h=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south_h=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north_h=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    fx=work_dtype(.5)/work_dtype(dx_m)/work_dtype(top_height_m)
    fy=work_dtype(.5)/work_dtype(dy_m)/work_dtype(top_height_m)
    gx=np.asarray((east_h-west_h)*fx,dtype=work_dtype)
    gy=np.asarray((north_h-south_h)*fy,dtype=work_dtype)
    forcing=np.asarray(u[0]*gx+v[0]*gy,dtype=work_dtype)
    phi=np.zeros_like(terrain,dtype=work_dtype)
    west=forcing[:,0].copy(); east=forcing[:,-1].copy()
    south=forcing[0].copy(); north=forcing[-1].copy()
    dx2=np.float64(dx_m)**2
    dy2=np.float64(dy_m)**2
    area=dx2*dy2
    factor=1./(2.*(dx2+dy2))
    relax=np.float64(relaxation)
    one_minus=1.-relax
    ny,nx=terrain.shape
    orders=((range(ny),range(nx)),(range(ny),range(nx-1,-1,-1)),
            (range(ny-1,-1,-1),range(nx)),
            (range(ny-1,-1,-1),range(nx-1,-1,-1)))
    for _ in range(maximum_iterations):
        error=0.
        for rows,columns in orders:
            for j in rows:
                for i in columns:
                    old=phi[j,i]
                    left=phi[j,i-1] if i else west[j]
                    right=phi[j,i+1] if i+1<nx else east[j]
                    below=phi[j-1,i] if j else south[i]
                    above=phi[j+1,i] if j+1<ny else north[i]
                    xx=dy2*(right+left)
                    yy=dx2*(above+below)
                    phi[j,i]=(one_minus*old+relax*factor*
                              (xx+yy-area*forcing[j,i]))
                    if i==0: west[j]=phi[j,i]
                    if i==nx-1: east[j]=phi[j,i]
                    if j==0: south[i]=phi[j,i]
                    if j==ny-1: north[i]=phi[j,i]
                    if abs(old)>=1.e-10:
                        error=max(error,abs((phi[j,i]-old)/old))
        if error<=tolerance: break
    left=np.concatenate((west[:,None],phi[:,:-1]),axis=1)
    right=np.concatenate((phi[:,1:],east[:,None]),axis=1)
    below=np.concatenate((south[None,:],phi[:-1]),axis=0)
    above=np.concatenate((phi[1:],north[None,:]),axis=0)
    dxi=.5/np.float64(dx_m)
    dyi=.5/np.float64(dy_m)
    u[0]=(right-left)*dxi+u[0]
    v[0]=(above-below)*dyi+v[0]
    return _model_array(u),_model_array(v)

def terrain_extrema(terrain_m: ArrayLike, radius_km: float, dx_m: float, dy_m: float) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    """Return local terrain extrema and their nearest physical distances.

    The circular search and all comparisons use double precision.  When an
    extremum occurs at more than one cell, its nearest occurrence is used;
    completely equidistant occurrences are resolved by row-major source
    index.  This makes the result independent of the order in which offsets
    are visited.  Returned model fields remain single precision.

    Args:
        terrain_m: Terrain elevation with shape ``(y, x)`` in metres.
        radius_km: Circular search radius in kilometres.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.

    Returns:
        Tuple ``(highest, lowest, distance_high, distance_low)`` of float32
        arrays with shape ``(y, x)``. The first two contain elevations in
        metres; the last two contain nearest-extremum distances in metres.
        The input is not modified.

    Raises:
        ValueError: If terrain is not two-dimensional, either grid spacing is
            non-finite or non-positive, or the radius is non-finite or
            negative.
    """
    h=np.asarray(terrain_m,dtype=COORDINATE_DTYPE)
    if h.ndim!=2:
        raise ValueError('terrain must be a two-dimensional array')
    dx=np.float64(dx_m); dy=np.float64(dy_m)
    radius=np.float64(radius_km)*np.float64(1000.)
    if not np.isfinite(dx) or dx<=0. or not np.isfinite(dy) or dy<=0.:
        raise ValueError('terrain grid spacing must be finite and positive')
    if not np.isfinite(radius) or radius<0.:
        raise ValueError('terrain search radius must be finite and nonnegative')

    ny,nx=h.shape
    highest=h.copy(); lowest=h.copy()
    distance_high=np.zeros_like(h); distance_low=np.zeros_like(h)
    source_indices=np.arange(ny*nx,dtype=np.int64).reshape(ny,nx)
    source_high=source_indices.copy(); source_low=source_indices.copy()
    irange=int(np.ceil(radius/dx))
    jrange=int(np.ceil(radius/dy))
    for dj in range(-jrange,jrange+1):
        for di in range(-irange,irange+1):
            distance=np.hypot(np.float64(di)*dx,np.float64(dj)*dy)
            if distance>radius: continue
            target_y=slice(max(0,-dj),min(ny,ny-dj)); target_x=slice(max(0,-di),min(nx,nx-di))
            source_y=slice(max(0,dj),min(ny,ny+dj)); source_x=slice(max(0,di),min(nx,nx+di))
            candidate=h[source_y,source_x]
            candidate_source=source_indices[source_y,source_x]

            hi=highest[target_y,target_x]
            dhi=distance_high[target_y,target_x]
            shi=source_high[target_y,target_x]
            update=(candidate>hi)|(
                (candidate==hi)&((distance<dhi)|(
                    (distance==dhi)&(candidate_source<shi))))
            hi[update]=candidate[update]
            dhi[update]=distance
            shi[update]=candidate_source[update]

            lo=lowest[target_y,target_x]
            dlo=distance_low[target_y,target_x]
            slo=source_low[target_y,target_x]
            update=(candidate<lo)|(
                (candidate==lo)&((distance<dlo)|(
                    (distance==dlo)&(candidate_source<slo))))
            lo[update]=candidate[update]
            dlo[update]=distance
            slo[update]=candidate_source[update]
    return cast(FloatQuad,tuple(_model_array(value) for value in (
        highest,lowest,distance_high,distance_low)))

def topographic_vertical_velocity(u: ArrayLike,v: ArrayLike,terrain_m: ArrayLike,z_faces_m: ArrayLike,gamma_k_per_m: ArrayLike,temperature_k: ArrayLike,alpha: float,dx_m: float,dy_m: float) -> Float32Array:
    """Calculate TOPOF2 terrain-following vertical velocity.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        terrain_m: Terrain elevation with shape ``(y, x)`` in metres.
        z_faces_m: Layer-face heights with shape ``(level + 1,)`` in metres
            above ground level.
        gamma_k_per_m: Potential-temperature lapse rate with shape ``(y, x)``
            in K m-1.
        temperature_k: Near-surface temperature with shape ``(y, x)`` in
            kelvins.
        alpha: Dimensionless empirical multiplier for the vertical-gradient
            correction.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.

    Returns:
        A float32 array with shape ``(level + 1, y, x)`` containing vertical
        velocity at layer faces in m s-1. Inputs are not modified.
    """
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    terrain=np.asarray(terrain_m,dtype=MODEL_DTYPE)
    west=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    gx=(east-west)*(.5/dx_m)
    gy=(north-south)*(.5/dy_m)
    gamma2=np.asarray(gamma_k_per_m,dtype=MODEL_DTYPE)+.01
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    stable=np.sqrt(np.maximum(9.8*gamma2/temperature,0.))
    stability=np.where(gamma2<0.,-1.,stable)
    speed=np.maximum(np.sqrt(u[-1]*u[-1]+v[-1]*v[-1]),1.e-6)
    wave=np.where(stability<=0.,2./500.,stability/speed)
    wtopo=u[0]*gx+v[0]*gy; nz=u.shape[0]
    decay=np.empty((nz+1,)+wtopo.shape,dtype=MODEL_DTYPE); decay[0]=wtopo
    w=np.empty_like(decay); w[0]=wtopo
    for k in range(nz):
        bkz=np.minimum(wave*faces[k+1],50.)
        decay[k+1]=wtopo*np.exp(-bkz)
        dzi=1./(faces[k+1]-faces[k])
        dwdz1=(decay[k+1]-decay[k])*dzi
        w[k+1]=alpha*dwdz1/dzi+w[k]
    w[0]=w[0]-u[0]*gx-v[0]*gy
    w[1:]=w[1:]-u*gx[None,:,:]-v*gy[None,:,:]
    return _model_array(w)

def slope_flow(terrain_m: ArrayLike,highest_m: ArrayLike,lowest_m: ArrayLike,distance_high_m: ArrayLike,distance_low_m: ArrayLike,sensible_heat_flux: ArrayLike,temperature_k: ArrayLike,landuse: ArrayLike,z_faces_m: ArrayLike,dx_m: float,dy_m: float,water_range: WaterRange=(50,55)) -> tuple[Float32Array, Float32Array]:
    """Calculate Mahrt/Horst-Doran terrain-slope wind components.

    Args:
        terrain_m: Terrain elevation with shape ``(y, x)`` in metres.
        highest_m: Highest terrain in the search neighbourhood with shape
            ``(y, x)`` in metres.
        lowest_m: Lowest terrain in the search neighbourhood with shape
            ``(y, x)`` in metres.
        distance_high_m: Distance to the nearest highest point with shape
            ``(y, x)`` in metres.
        distance_low_m: Distance to the nearest lowest point with shape
            ``(y, x)`` in metres.
        sensible_heat_flux: Surface sensible heat flux with shape ``(y, x)``
            in W m-2.
        temperature_k: Surface air temperature with shape ``(y, x)`` in
            kelvins.
        landuse: Integer land-use categories with shape ``(y, x)``.
        z_faces_m: Layer-face heights with shape ``(level + 1,)`` in metres
            above ground level.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.
        water_range: Inclusive pair of land-use category numbers treated as
            water and excluded from slope flow.

    Returns:
        Tuple ``(u_slope, v_slope)`` of float32 arrays with shape
        ``(level, y, x)`` in m s-1. Values are additive wind components;
        inputs are not modified.
    """
    h=np.asarray(terrain_m,dtype=MODEL_DTYPE)
    qh=np.asarray(sensible_heat_flux,dtype=MODEL_DTYPE)
    temp=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    west=np.concatenate((h[:,:1],h[:,:-1]),axis=1)
    east=np.concatenate((h[:,1:],h[:,-1:]),axis=1)
    south=np.concatenate((h[:1,:],h[:-1,:]),axis=0)
    north=np.concatenate((h[1:,:],h[-1:,:]),axis=0)
    gx=(east-west)*(.5/dx_m)
    gy=(north-south)*(.5/dy_m)
    magnitude=np.hypot(gx,gy)
    angle=np.arctan(magnitude)
    active=(np.abs(angle)>=.009)
    active &= ~((np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1]))
    down=qh<0.
    highest=np.asarray(highest_m,dtype=MODEL_DTYPE)
    lowest=np.asarray(lowest_m,dtype=MODEL_DTYPE)
    relief=np.where(down,highest-h,h-lowest)
    hd=.05*relief
    distance=np.where(down,np.asarray(distance_high_m,dtype=MODEL_DTYPE),
                      np.asarray(distance_low_m,dtype=MODEL_DTYPE))
    dcrest=np.sqrt(relief*relief+distance*distance)
    sinalf=np.sin(np.abs(angle))
    bounded=np.divide(relief,dcrest,out=np.zeros_like(relief),where=dcrest>0)
    sinalf=np.where(down,np.minimum(sinalf,bounded),sinalf)
    exponent=np.maximum(-dcrest*.08/np.maximum(hd,1.e-30),-50.)
    one_third=MODEL_DTYPE(1./3.)
    downslope_base=sinalf*(9.81/temp)*np.abs(qh)*dcrest/1229.9/.08
    decay=1.-np.exp(exponent)
    downslope=-np.power(downslope_base,one_third)*np.power(decay,one_third)
    upslope_base=(9.81/temp)*np.abs(qh)*relief/1229.9
    upslope=np.power(upslope_base,one_third)
    velocity=np.where(down,downslope,upslope); velocity=np.where(active&(hd>.05),velocity,0.)
    unit_x=np.divide(gx,magnitude,out=np.zeros_like(gx),where=magnitude>0.)
    unit_y=np.divide(gy,magnitude,out=np.zeros_like(gy),where=magnitude>0.)
    ux=unit_x*velocity
    vy=unit_y*velocity
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    ratios=np.clip((hd[None,:,:]-faces[:-1,None,None])/
                   np.diff(faces)[:,None,None],0.,1.)
    return _model_array(ratios*ux[None,:,:]),_model_array(ratios*vy[None,:,:])

def vertical_layer_average(values: ArrayLike, heights_m: ArrayLike, faces_m: ArrayLike, *, missing: float=998.99) -> Float32Array:
    """VERTAV exact piecewise-linear layer mean for arbitrary profiles.

    The final axis is the source sounding level. All preceding axes identify
    independent profiles. ``heights_m`` may be one-dimensional or match the
    values array.

    Args:
        values: Source values with shape ``(..., source_level)``. Units are
            preserved in the output.
        heights_m: Source heights in metres, either shape ``(source_level,)``
            or broadcastable to ``values``.
        faces_m: Target layer-face heights with shape ``(target_level + 1,)``
            in metres.
        missing: Values greater than or equal to this threshold are omitted.

    Returns:
        A float32 array with shape ``(..., target_level)`` and the same units
        as ``values``. The target-layer axis is last, broadcasting is applied
        only to source heights, and inputs are not modified.

    Raises:
        ValueError: If fewer than two source levels or faces are supplied, or
            valid adjacent source segments do not fully cover a target layer.
    """
    values=np.asarray(values,dtype=ACCUMULATION_DTYPE)
    heights=np.broadcast_to(np.asarray(heights_m,dtype=ACCUMULATION_DTYPE),values.shape)
    faces=np.asarray(faces_m,dtype=ACCUMULATION_DTYPE)
    if values.shape[-1]<2 or faces.ndim!=1 or faces.size<2: raise ValueError('at least two source levels and two faces are required')
    z0,z1=heights[...,:-1],heights[...,1:]; v0,v1=values[...,:-1],values[...,1:]
    valid=(v0<missing)&(v1<missing)&(z1>z0)
    output: list[Float64Array] = []
    for bottom,top in zip(faces[:-1],faces[1:]):
        lo=np.maximum(z0,bottom); hi=np.minimum(z1,top); width=np.maximum(hi-lo,0.); use=valid&(width>0.)
        fraction_lo=np.divide(lo-z0,z1-z0,out=np.zeros_like(lo),where=z1>z0)
        fraction_hi=np.divide(hi-z0,z1-z0,out=np.zeros_like(hi),where=z1>z0)
        vlo=v0+(v1-v0)*fraction_lo; vhi=v0+(v1-v0)*fraction_hi
        integral=np.sum(np.where(use,.5*(vlo+vhi)*width,0.),axis=-1)
        coverage=np.sum(np.where(use,width,0.),axis=-1)
        if np.any(np.abs(coverage-(top-bottom))>1.e-4): raise ValueError(f'sounding does not fully cover layer {bottom:g}-{top:g} m')
        output.append(integral/(top-bottom))
    return _model_array(np.stack(output,axis=-1))

def vertical_interpolate(values: ArrayLike, heights_m: ArrayLike, targets_m: ArrayLike, *, missing: float=998.99) -> Float32Array:
    """Interpolate profiles linearly while skipping missing source levels.

    Args:
        values: Source values with shape ``(..., source_level)``. Units are
            preserved in the output.
        heights_m: Source heights in metres, either shape ``(source_level,)``
            or broadcastable to ``values``.
        targets_m: Target heights of any shape in metres.
        missing: Values greater than or equal to this threshold are omitted.

    Returns:
        A float32 array with shape ``values.shape[:-1] + targets_m.shape`` and
        the same units as ``values``. Inputs are not modified.

    Raises:
        ValueError: If a profile has fewer than two valid source levels or its
            valid height range does not bracket every target.
    """
    values=np.asarray(values,dtype=ACCUMULATION_DTYPE)
    heights=np.broadcast_to(np.asarray(heights_m,dtype=ACCUMULATION_DTYPE),values.shape)
    targets=np.asarray(targets_m,dtype=ACCUMULATION_DTYPE)
    leading=values.shape[:-1]
    result=np.empty(leading+targets.shape,dtype=ACCUMULATION_DTYPE)
    for index in np.ndindex(leading):
        good=values[index]<missing; z=heights[index][good]; v=values[index][good]
        if z.size<2 or targets.min()<z.min() or targets.max()>z.max(): raise ValueError('valid sounding does not bracket interpolation targets')
        result[index]=np.interp(targets,z,v)
    return _model_array(result)

def smooth_wind(u: ArrayLike,v: ArrayLike,passes: ArrayLike) -> tuple[Float32Array, Float32Array]:
    """Apply CALMET's sequential Gauss-Seidel wind-smoothing stencil.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        passes: Number of smoothing passes, either a scalar or one integer per
            model level.

    Returns:
        Tuple ``(u_smoothed, v_smoothed)`` of new float32 arrays with shape
        ``(level, y, x)`` in m s-1. Each pass updates the copies sequentially;
        the supplied arrays are not modified.

    Raises:
        ValueError: If the wind arrays differ in shape, are not
            three-dimensional, or ``passes`` cannot broadcast over levels.
    """
    u=np.array(u,dtype=MODEL_DTYPE,copy=True); v=np.array(v,dtype=MODEL_DTYPE,copy=True)
    if u.shape!=v.shape or u.ndim!=3: raise ValueError('wind arrays must have shape (z,y,x)')
    counts=np.broadcast_to(np.asarray(passes,dtype=int), (u.shape[0],))
    ub_left=u[:,:,0].copy(); ub_right=u[:,:,-1].copy()
    vb_bottom=v[:,0,:].copy(); vb_top=v[:,-1,:].copy()
    for k,count in enumerate(counts):
        for _ in range(count):
            for j in range(u.shape[1]):
                for i in range(u.shape[2]):
                    uc=u[k,j,i]
                    uw=ub_left[k,j] if i==0 else u[k,j,i-1]
                    ue=ub_right[k,j] if i+1==u.shape[2] else u[k,j,i+1]
                    us=uc if j==0 else u[k,j-1,i]
                    un=uc if j+1==u.shape[1] else u[k,j+1,i]
                    total=ue+uw+un+us
                    u[k,j,i]=MODEL_DTYPE(.5)*uc+MODEL_DTYPE(.125)*total

                    vc=v[k,j,i]
                    vw=vc if i==0 else v[k,j,i-1]
                    ve=vc if i+1==v.shape[2] else v[k,j,i+1]
                    vs=vb_bottom[k,i] if j==0 else v[k,j-1,i]
                    vn=vb_top[k,i] if j+1==v.shape[1] else v[k,j+1,i]
                    total=ve+vw+vn+vs
                    v[k,j,i]=MODEL_DTYPE(.5)*vc+MODEL_DTYPE(.125)*total
    return u,v

def divergence_and_vertical_velocity(u: ArrayLike,v: ArrayLike,z_faces_m: ArrayLike,dx_m: float,dy_m: float,*,obrien: bool=False,boundaries: GridBoundaries | None=None) -> tuple[Float32Array, Float32Array]:
    """Calculate horizontal divergence and integrate vertical velocity.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        z_faces_m: Layer-face heights with shape ``(level + 1,)`` in metres.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.
        obrien: Whether to apply the O'Brien correction that forces zero
            vertical velocity at the domain top.
        boundaries: Optional tuple ``(u_west, u_east, v_south, v_north)``.
            The u boundaries have shape ``(level, y)`` and the v boundaries
            have shape ``(level, x)``; omitted boundaries repeat edge winds.

    Returns:
        Tuple ``(divergence, w)``. ``divergence`` is a float32 array with
        shape ``(level, y, x)`` in s-1, and ``w`` is a float32 array with
        shape ``(level + 1, y, x)`` in m s-1. Inputs are not modified.
    """
    work_dtype=ACCUMULATION_DTYPE
    u=np.asarray(u,dtype=work_dtype); v=np.asarray(v,dtype=work_dtype); faces=np.asarray(z_faces_m,dtype=work_dtype)
    if boundaries is None:
        u_west,u_east,v_south,v_north=u[:,:,0],u[:,:,-1],v[:,0,:],v[:,-1,:]
    else:
        u_west,u_east,v_south,v_north=(np.asarray(value,dtype=work_dtype) for value in boundaries)
    up=np.concatenate((u_west[:,:,None],u,u_east[:,:,None]),axis=2)
    vp=np.concatenate((v_south[:,None,:],v,v_north[:,None,:]),axis=1)
    dxi=work_dtype(1.)/(work_dtype(2.)*work_dtype(dx_m)); dyi=work_dtype(1.)/(work_dtype(2.)*work_dtype(dy_m))
    divergence=np.asarray((up[:,:,2:]-up[:,:,:-2])*dxi+
                          (vp[:,2:,:]-vp[:,:-2,:])*dyi,dtype=work_dtype)
    w=np.zeros((u.shape[0]+1,)+u.shape[1:],dtype=work_dtype)
    # Integrate successively because every face depends on the one below it.
    for k in range(u.shape[0]):
        w[k+1]=np.asarray(-((faces[k+1]-faces[k])*divergence[k])+w[k],dtype=work_dtype)
    if obrien:
        top=w[-1].copy()
        for k in range(u.shape[0]):
            ratio=work_dtype(faces[k+1]/faces[-1])
            w[k+1]=np.asarray(w[k+1]-ratio*top,dtype=work_dtype)
    return _model_array(divergence),_model_array(w)

def minimize_divergence(u: ArrayLike,v: ArrayLike,w: ArrayLike,z_faces_m: ArrayLike,dx_m: float,dy_m: float,*,boundaries: GridBoundaries | None=None,
                        maximum_iterations: int=50,limit: float=5.e-6) -> tuple[Float32Array, Float32Array, Float32Array, Int32Array]:
    """Adjust horizontal winds to reduce three-dimensional divergence.

    Args:
        u: Eastward wind with shape ``(level, y, x)`` in m s-1.
        v: Northward wind with shape ``(level, y, x)`` in m s-1.
        w: Layer-face vertical velocity with shape ``(level + 1, y, x)`` in
            m s-1.
        z_faces_m: Layer-face heights with shape ``(level + 1,)`` in metres.
        dx_m: East-west grid spacing in metres.
        dy_m: South-north grid spacing in metres.
        boundaries: Optional tuple ``(u_west, u_east, v_south, v_north)``.
            The u boundaries have shape ``(level, y)`` and the v boundaries
            have shape ``(level, x)``; omitted values start from wind edges.
        maximum_iterations: Maximum four-sweep iterations per model level.
        limit: Absolute convergence limit for divergence in s-1.

    Returns:
        Tuple ``(u_adjusted, v_adjusted, divergence, iterations)``. The first
        two float32 arrays have shape ``(level, y, x)`` in m s-1, divergence
        is float32 with that shape in s-1, and iterations is int32 with shape
        ``(level,)``. Input arrays are not modified.

    Raises:
        ValueError: If u and v shapes differ or w does not have one more
            vertical face than the horizontal-wind fields.
    """
    work_dtype=ACCUMULATION_DTYPE
    u=np.array(u,dtype=work_dtype,copy=True); v=np.array(v,dtype=work_dtype,copy=True)
    w=np.asarray(w,dtype=work_dtype); faces=np.asarray(z_faces_m,dtype=work_dtype)
    if u.shape!=v.shape or w.shape!=(u.shape[0]+1,*u.shape[1:]):
        raise ValueError('MINIM wind arrays have inconsistent dimensions')
    if boundaries is None:
        boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    west,east,south,north=(np.array(value,dtype=work_dtype,copy=True) for value in boundaries)
    divergence=np.empty_like(u); iterations=np.zeros(u.shape[0],dtype=np.int32)

    def layer_divergence(k: int) -> Float64Array:
        """Evaluate total divergence for one model layer.

        Args:
            k: Zero-based model-layer index.

        Returns:
            A float64 array with shape ``(y, x)`` containing divergence in
            s-1. The enclosing wind and boundary arrays are read in place.
        """
        up=np.concatenate((west[k,:,None],u[k],east[k,:,None]),axis=1)
        vp=np.concatenate((south[k,None,:],v[k],north[k,None,:]),axis=0)
        dxi=work_dtype(1.)/(work_dtype(2.)*work_dtype(dx_m)); dyi=work_dtype(1.)/(work_dtype(2.)*work_dtype(dy_m))
        horizontal=(up[:,2:]-up[:,:-2])*dxi+(vp[2:]-vp[:-2])*dyi
        return np.asarray(horizontal+(w[k+1]-w[k])/(faces[k+1]-faces[k]),dtype=work_dtype)

    for k in range(u.shape[0]):
        divergence[k]=layer_divergence(k)
        if np.max(np.abs(divergence[k]))<=limit: continue
        for iteration in range(1,maximum_iterations+1):
            for direction in range(4):
                div=divergence[k]
                irange=(range(u.shape[2]) if direction in {0,2}
                        else range(u.shape[2]-1,-1,-1))
                jrange=(range(u.shape[1]) if direction in {0,1}
                        else range(u.shape[1]-1,-1,-1))
                for j in jrange:
                    for i in irange:
                        if div[j,i]==work_dtype(0.): continue
                        ut=work_dtype(-work_dtype(2.)*(div[j,i]*work_dtype(dx_m))/work_dtype(2.))
                        vt=work_dtype(-work_dtype(2.)*(div[j,i]*work_dtype(dy_m))/work_dtype(2.))
                        if i>0: u[k,j,i-1]=work_dtype(u[k,j,i-1]-work_dtype(.5)*ut)
                        if i+1<u.shape[2]: u[k,j,i+1]=work_dtype(u[k,j,i+1]+work_dtype(.5)*ut)
                        if j>0: v[k,j-1,i]=work_dtype(v[k,j-1,i]-work_dtype(.5)*vt)
                        if j+1<u.shape[1]: v[k,j+1,i]=work_dtype(v[k,j+1,i]+work_dtype(.5)*vt)
                west[k]=u[k,:,0]; east[k]=u[k,:,-1]
                south[k]=v[k,0,:]; north[k]=v[k,-1,:]
                divergence[k]=layer_divergence(k)
            iterations[k]=iteration
            if np.max(np.abs(divergence[k]))<=limit: break
    return (_model_array(u),_model_array(v),_model_array(divergence),iterations)

def heat_flux(solar_sine_field: ArrayLike,cloud_fraction_field: ArrayLike,temperature_k: ArrayLike,landuse: ArrayLike,albedo: ArrayLike,
              bowen: ArrayLike,soil_heat_flux: ArrayLike,anthropogenic_heat: ArrayLike,water_range: WaterRange=(50,55),*,
              turbidity_coefficient_1: float=990.,turbidity_coefficient_2: float=-30.,
              cloudiness_coefficient_1: float=-.75,cloudiness_coefficient_2: float=3.4,
              net_radiation_coefficient_1: float=5.31e-13,
              net_radiation_coefficient_2: float=60.,net_radiation_coefficient_3: float=.12) -> tuple[Float32Array, Float32Array]:
    """Calculate short-wave radiation and land sensible heat flux.

    Args:
        solar_sine_field: Sine of solar elevation on the surface grid,
            dimensionless.
        cloud_fraction_field: Cloud fraction on the surface grid from 0 to 1.
        temperature_k: Surface air temperature on the surface grid in kelvins.
        landuse: Integer land-use categories on the surface grid.
        albedo: Surface albedo fraction, broadcastable to the surface grid.
        bowen: Bowen ratio, broadcastable to the surface grid.
        soil_heat_flux: Fraction of net radiation stored as soil heat,
            broadcastable to the surface grid.
        anthropogenic_heat: Anthropogenic heat flux, broadcastable to the
            surface grid, in W m-2.
        water_range: Inclusive pair of land-use category numbers treated as
            water.
        turbidity_coefficient_1: Clear-sky solar coefficient in W m-2.
        turbidity_coefficient_2: Clear-sky solar offset in W m-2.
        cloudiness_coefficient_1: Dimensionless cloud attenuation coefficient.
        cloudiness_coefficient_2: Dimensionless cloud-fraction exponent.
        net_radiation_coefficient_1: Temperature-sixth-power coefficient in
            W m-2 K-6.
        net_radiation_coefficient_2: Cloud radiation coefficient in W m-2.
        net_radiation_coefficient_3: Dimensionless net-radiation denominator
            coefficient.

    Returns:
        Tuple ``(sensible_heat, shortwave)`` of float32 arrays on the
        broadcast surface-grid shape in W m-2. Water sensible heat is zero;
        nighttime land sensible heat is -0.1 W m-2. Inputs are not modified.
    """
    sun=np.asarray(solar_sine_field,dtype=MODEL_DTYPE)
    cloud=np.asarray(cloud_fraction_field,dtype=MODEL_DTYPE)
    temp=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    albedo=np.asarray(albedo,dtype=MODEL_DTYPE); bowen=np.asarray(bowen,dtype=MODEL_DTYPE)
    soil=np.asarray(soil_heat_flux,dtype=MODEL_DTYPE)
    anthropogenic=np.asarray(anthropogenic_heat,dtype=MODEL_DTYPE)
    cloud_power=np.power(cloud,cloudiness_coefficient_2)
    qsw=np.maximum(
        (turbidity_coefficient_1*sun+turbidity_coefficient_2)*
        (1.+cloudiness_coefficient_1*cloud_power),0.)
    temp2=temp*temp
    temp3=temp2*temp
    temp4=temp2*temp2
    temp6=temp3*temp3
    qstar=((1.-albedo)*qsw+
        net_radiation_coefficient_1*temp6-
        5.67e-8*temp4+net_radiation_coefficient_2*cloud)/(
        net_radiation_coefficient_3+1.)
    qh=bowen*(qstar*(1.-soil)+anthropogenic)/(1.+bowen)
    qh=np.where(sun<=0.,-.1,qh)
    water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
    return _model_array(np.where(water,0.,qh)),_model_array(qsw)

def psi_momentum_dyer(
    z_over_l: ArrayLike,
    *,
    dtype: FloatingDType = MODEL_DTYPE,
) -> FloatingArray:
    """Return the Dyer momentum stability correction.

    Model calls default to single precision.  Nonlinear convergence routines
    can request double precision without changing the returned model fields.

    Args:
        z_over_l: Dimensionless height-to-Monin-Obukhov-length ratio of any
            shape.
        dtype: NumPy-compatible floating dtype used for the calculation and
            returned array.

    Returns:
        An array with the same shape as ``z_over_l`` and the requested dtype,
        containing the dimensionless momentum stability correction. The
        input is not modified.
    """
    work_dtype=np.dtype(dtype).type
    zl=cast(FloatingArray,np.asarray(z_over_l,dtype=work_dtype))
    result=np.empty_like(zl,dtype=work_dtype)
    unstable=zl<0.
    zu=zl[unstable]
    x=np.power(1.-16.*zu,.25)
    half_pi=work_dtype(np.pi/2.)
    psik=2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-2.*np.arctan(x)+half_pi
    y=np.cbrt(1.-10.15*zu)
    sqrt_three=work_dtype(np.sqrt(3.))
    pi_over_sqrt=work_dtype(np.pi)/sqrt_three
    psic=(1.5*np.log((1.+y+y*y)/3.)-
          sqrt_three*np.arctan((1.+2.*y)/sqrt_three)+
          pi_over_sqrt)
    f=zu*zu/(1.+zu*zu); result[unstable]=(1.-f)*psik+f*psic
    stable=~unstable
    c=np.minimum(50.,.35*zl[stable])
    result[stable]=-((1.+zl[stable])+(2./3.)*(
        zl[stable]-14.28)/np.exp(c)+8.525)
    return cast(FloatingArray, result)

def surface_similarity(u: ArrayLike,v: ArrayLike,roughness_m: ArrayLike,height_m: float,density: ArrayLike,temperature_k: ArrayLike,
                       cloud_fraction_field: ArrayLike,sensible_heat_flux: ArrayLike,landuse: ArrayLike,
                       water_range: WaterRange=(50,55),*,wind_measurement_height_m: ArrayLike | None=None,
                       ipsifcn: int=0,minimum_stable_wind_m_s: float=.5) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate ELUSTR surface-layer state over land cells.

    Args:
        u: Eastward wind at ``height_m``, in m s-1 and broadcastable to the
            surface grid.
        v: Northward wind at ``height_m``, in m s-1 and broadcastable to the
            surface grid.
        roughness_m: Aerodynamic roughness length in metres, broadcastable to
            the surface grid.
        height_m: Height of the supplied model wind in metres.
        density: Air density in kg m-3, broadcastable to the surface grid.
        temperature_k: Surface air temperature in kelvins, broadcastable to
            the surface grid.
        cloud_fraction_field: Cloud fraction from 0 to 1, broadcastable to
            the surface grid.
        sensible_heat_flux: Initial sensible heat flux in W m-2,
            broadcastable to the surface grid. Stable cells are recomputed.
        landuse: Integer land-use categories on the surface grid.
        water_range: Inclusive pair of land-use category numbers treated as
            water.
        wind_measurement_height_m: Optional wind-observation height in metres,
            scalar or broadcastable to the surface grid. ``height_m`` is used
            when omitted.
        ipsifcn: Momentum stability-function selector, 0 for Dyer or 1 for
            CALMET's convective fit.
        minimum_stable_wind_m_s: Lower wind-speed bound for stable conditions,
            in m s-1.

    Returns:
        Tuple ``(friction_velocity, monin_obukhov_length, sensible_heat)`` of
        float32 arrays on the broadcast surface-grid shape, with units m s-1,
        m, and W m-2 respectively. Water cells are zero. The sensible-heat
        input is copied and all supplied arrays remain unchanged.

    Raises:
        ValueError: If ``ipsifcn`` is not 0 or 1, or input arrays cannot be
            broadcast by the numerical operations.
    """
    u=np.asarray(u,dtype=np.float64); v=np.asarray(v,dtype=np.float64)
    z0=np.asarray(roughness_m,dtype=np.float64); rho=np.asarray(density,dtype=np.float64)
    temp=np.asarray(temperature_k,dtype=np.float64)
    cloud=np.asarray(cloud_fraction_field,dtype=np.float64)
    qh=np.array(sensible_heat_flux,dtype=np.float64,copy=True)
    height=np.float64(height_m)
    measurement=(height if wind_measurement_height_m is None else
                 np.asarray(wind_measurement_height_m,dtype=np.float64))
    if ipsifcn not in {0,1}: raise ValueError('IPSIFCN must be zero or one')
    squared=np.asarray(u*u+v*v,dtype=np.float64)
    speed=np.maximum(np.sqrt(squared),np.float64(.001))
    logz=np.asarray(np.log(np.asarray(height/z0,dtype=np.float64)),
                    dtype=np.float64)
    ustar=np.zeros_like(speed); length=np.zeros_like(speed)
    daytime=qh>=0.; trial=np.maximum(np.float64(.4)*speed/logz,np.float64(.05))
    xl=np.full_like(speed,np.float64(-9.9e9))
    active=daytime&(qh!=0.)
    for _ in range(5):
        old=xl
        trial_cubed=trial*trial*trial
        new=np.where(active,np.asarray(-np.float64(253.8226)*rho*temp*
            trial_cubed/np.where(qh==0.,np.float64(1.),qh),dtype=np.float64),xl)
        if ipsifcn==0:
            psizl=psi_momentum_dyer(
                measurement/new,dtype=ACCUMULATION_DTYPE)
            psiz0l=psi_momentum_dyer(
                z0/new,dtype=ACCUMULATION_DTYPE)
        else:
            psizl=_psiuc(
                measurement,new,15,dtype=ACCUMULATION_DTYPE)
            psiz0l=_psiuc(z0,new,15,dtype=ACCUMULATION_DTYPE)
        denominator=np.asarray(logz-psizl+psiz0l,dtype=np.float64)
        refined=np.maximum(np.asarray(np.float64(.4)*speed/denominator,dtype=np.float64),
                           np.float64(.05))
        converged=active&(np.abs(np.asarray((new-old)/new,dtype=np.float64))<np.float64(.005))
        xl=np.where(active,new,xl).astype(np.float64)
        trial=np.where(active,refined,trial).astype(np.float64)
        active &= ~converged
    ustar=np.where(daytime,trial,ustar); length=np.where(daytime,xl,length)
    night=~daytime
    stable_speed=np.maximum(speed,np.float64(minimum_stable_wind_m_s))
    jcc=np.trunc(np.float64(10.)*cloud+np.float64(.5)).astype(np.int32)
    cloud_index_squared=np.asarray(jcc*jcc,dtype=np.float64)
    theta1=np.asarray(np.float64(.09)*(np.float64(1.)-
        np.float64(5.e-3)*cloud_index_squared),dtype=np.float64)
    cdn=np.asarray(np.float64(.4)/logz,dtype=np.float64)
    stable_speed_squared=np.asarray(stable_speed*stable_speed,dtype=np.float64)
    theta2=np.asarray(temp*cdn*stable_speed_squared/
                      (np.float64(184.428)*height),dtype=np.float64)
    theta=np.maximum(np.minimum(theta1,theta2),np.float64(1.e-9))
    u02=np.asarray(np.float64(46.107)*height*theta/temp,dtype=np.float64)
    stable_u=np.maximum(np.asarray(np.float64(.5)*cdn*(stable_speed+np.sqrt(np.maximum(
        np.float64(0.),stable_speed*stable_speed-np.float64(4.)*u02/cdn))),dtype=np.float64),
        np.float64(.05))
    # ELUSTR enters this refinement only when the initial product exceeds
    # XCRIT.  Once inside, its early-exit threshold is 1.05*XCRIT; testing
    # against XCRIT again performs extra iterations and is not equivalent.
    adjust=(stable_u*theta)>np.float64(.05)
    tolerance=np.float64(1.05)*np.float64(.05)
    for _ in range(3):
        theta=np.where(adjust,np.float64(.05)/stable_u,theta).astype(np.float64)
        u02=np.asarray(np.float64(46.107)*height*theta/temp,dtype=np.float64)
        updated=np.maximum(np.asarray(np.float64(.5)*cdn*(stable_speed+np.sqrt(np.maximum(
            np.float64(0.),stable_speed*stable_speed-np.float64(4.)*u02/cdn))),dtype=np.float64),
            np.float64(.05))
        stable_u=np.where(adjust,updated,stable_u).astype(np.float64)
        adjust &= (stable_u*theta)>tolerance
    stable_qh=np.asarray(-np.float64(996.)*rho*stable_u*theta,dtype=np.float64)
    stable_u_cubed=stable_u*stable_u*stable_u
    stable_l=np.asarray(-np.float64(253.8226)*rho*temp*stable_u_cubed/stable_qh,
                        dtype=np.float64)
    ustar=np.where(night,stable_u,ustar); qh=np.where(night,stable_qh,qh); length=np.where(night,stable_l,length)
    water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
    return cast(FloatTriple,tuple(_model_array(value) for value in (
        np.where(water,0.,ustar),np.where(water,0.,length),np.where(water,0.,qh))))


def land_station_surface_similarity(u: ArrayLike,v: ArrayLike,roughness_m: ArrayLike,anemometer_height_m: ArrayLike,
                                    first_level_height_m: float,density: ArrayLike,temperature_k: ArrayLike,
                                    cloud_fraction: ArrayLike,sensible_heat_flux: ArrayLike,*,ipsifcn: int=0) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate ELUSTR2 surface-layer state for land stations.

    Args:
        u: Eastward station wind with shape ``(station,)`` in m s-1.
        v: Northward station wind with shape ``(station,)`` in m s-1.
        roughness_m: Station aerodynamic roughness with shape ``(station,)``
            in metres.
        anemometer_height_m: Wind measurement heights with shape ``(station,)``
            in metres.
        first_level_height_m: Height of the first CALMET layer centre in
            metres.
        density: Station air density with shape ``(station,)`` in kg m-3.
        temperature_k: Station air temperature with shape ``(station,)`` in
            kelvins.
        cloud_fraction: Station cloud fraction with shape ``(station,)`` from
            0 to 1.
        sensible_heat_flux: Station sensible heat flux with shape
            ``(station,)`` in W m-2.
        ipsifcn: Momentum stability-function selector, 0 for Dyer or 1 for
            CALMET's convective fit.

    Returns:
        Tuple ``(friction_velocity, monin_obukhov_length, sensible_heat)`` of
        float32 arrays with shape ``(station,)`` and units m s-1, m, and
        W m-2 respectively. Inputs are not modified.

    Raises:
        ValueError: If ``ipsifcn`` is not 0 or 1, or station inputs cannot be
            broadcast together.
    """
    return surface_similarity(u,v,roughness_m,first_level_height_m,density,
        temperature_k,cloud_fraction,sensible_heat_flux,
        np.zeros(np.shape(u),dtype=np.int16),wind_measurement_height_m=anemometer_height_m,
        ipsifcn=ipsifcn,minimum_stable_wind_m_s=0.)


def land_station_mixing_height(temperature_profile_k: ArrayLike,profile_height_m: ArrayLike,
        previous_convective_height_m: ArrayLike,temperature_jump_k: ArrayLike,friction_velocity: ArrayLike,
        monin_obukhov_m: ArrayLike,sensible_heat_flux: ArrayLike,density: ArrayLike,coriolis: ArrayLike,*,mixing_method: int=1,
        surface_temperature_k: ArrayLike | None=None,**options: MixingOption) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Calculate MIXHT2ST mixing heights for land-station columns.

    Args:
        temperature_profile_k: Prognostic temperature profiles with shape
            ``(source_level, station)`` in kelvins.
        profile_height_m: Heights matching ``temperature_profile_k`` in
            metres.
        previous_convective_height_m: Previous convective heights with shape
            ``(station,)`` in metres.
        temperature_jump_k: Previous inversion temperature jumps with shape
            ``(station,)`` in kelvins.
        friction_velocity: Station friction velocity with shape ``(station,)``
            in m s-1.
        monin_obukhov_m: Station Monin-Obukhov length with shape ``(station,)``
            in metres.
        sensible_heat_flux: Station sensible heat flux with shape
            ``(station,)`` in W m-2.
        density: Station air density with shape ``(station,)`` in kg m-3.
        coriolis: Absolute Coriolis parameter with shape ``(station,)`` in
            s-1.
        mixing_method: Mixing-height method ``+/-1`` for Maul-Carson or
            ``+/-2`` for Batchvarova-Gryning.
        surface_temperature_k: Surface temperature with shape ``(station,)``
            in kelvins, required for method ``+/-2``.
        **options: Additional keyword arguments passed to the chosen gridded
            mixing-height routine.

    Returns:
        Tuple ``(mixing_height, convective_height, temperature_jump)`` of
        float32 arrays with shape ``(station,)`` and units m, m, and K.
        Inputs are not modified.

    Raises:
        ValueError: If profile arrays are not matching
            ``(source_level, station)`` fields, the method is unsupported, or
            method ``+/-2`` lacks surface temperature.
        TypeError: If ``options`` contains an unsupported keyword for the
            selected mixing-height routine.
    """
    temperature=np.asarray(temperature_profile_k,dtype=MODEL_DTYPE)
    height=np.asarray(profile_height_m,dtype=MODEL_DTYPE)
    if temperature.ndim!=2 or height.shape!=temperature.shape:
        raise ValueError('station temperature profiles must have shape (level, station)')
    shaped=lambda value: np.asarray(value,dtype=MODEL_DTYPE).reshape(1,-1)
    arguments=(temperature[:,None,:],height[:,None,:],
        shaped(previous_convective_height_m),shaped(temperature_jump_k),
        shaped(friction_velocity),shaped(monin_obukhov_m),
        shaped(sensible_heat_flux),shaped(density),shaped(coriolis))
    if abs(int(mixing_method))==1:
        result=cast(Callable[..., FloatTriple],
            mixing_height_maul_carson)(*arguments,**options)
    elif abs(int(mixing_method))==2:
        if surface_temperature_k is None:
            raise ValueError('MIXHBG station profiles require surface temperature')
        result=cast(Callable[..., FloatTriple],mixing_height_batchvarova_gryning)(
            *arguments,surface_temperature_k=shaped(surface_temperature_k),**options)
    else:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    return cast(FloatTriple,tuple(_model_array(value[0]) for value in result))


def land_station_similarity_theory(u: ArrayLike,v: ArrayLike,roughness_m: ArrayLike,anemometer_height_m: ArrayLike,
        first_level_height_m: float,density: ArrayLike,temperature_k: ArrayLike,cloud_fraction: ArrayLike,
        sensible_heat_flux: ArrayLike,temperature_profile_k: ArrayLike,profile_height_m: ArrayLike,
        previous_convective_height_m: ArrayLike,temperature_jump_k: ArrayLike,coriolis: ArrayLike,
        z_centers_m: ArrayLike,z_faces_m: ArrayLike,station_latitude: ArrayLike,*,mixing_method: int=1,ipsifcn: int=0,
        minimum_mixing_height_m: float=50.,**mixing_options: MixingOption) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array]:
    """Vectorized land branch of STHEOR: ELUSTR2, MIXHT2ST, then SIMILT.

    Missing and calm station winds retain their first-level observation and
    remain missing aloft, matching ELUSTR2's early return before SIMILT.

    Args:
        u: Eastward station wind with shape ``(station,)`` in m s-1.
        v: Northward station wind with shape ``(station,)`` in m s-1.
        roughness_m: Station aerodynamic roughness with shape ``(station,)``
            in metres.
        anemometer_height_m: Wind measurement heights with shape ``(station,)``
            in metres.
        first_level_height_m: Height of the first CALMET layer centre in
            metres.
        density: Station air density with shape ``(station,)`` in kg m-3.
        temperature_k: Station surface temperature with shape ``(station,)``
            in kelvins.
        cloud_fraction: Station cloud fraction with shape ``(station,)`` from
            0 to 1.
        sensible_heat_flux: Station sensible heat flux with shape
            ``(station,)`` in W m-2.
        temperature_profile_k: Prognostic temperature with shape
            ``(source_level, station)`` in kelvins.
        profile_height_m: Heights matching ``temperature_profile_k`` in
            metres.
        previous_convective_height_m: Previous convective heights with shape
            ``(station,)`` in metres.
        temperature_jump_k: Previous inversion jumps with shape ``(station,)``
            in kelvins.
        coriolis: Absolute Coriolis parameter with shape ``(station,)`` in
            s-1.
        z_centers_m: CALMET layer-centre heights with shape ``(level,)`` in
            metres above ground level.
        z_faces_m: CALMET layer-face heights with shape ``(level + 1,)`` in
            metres above ground level.
        station_latitude: Station latitude with shape ``(station,)`` in
            degrees north.
        mixing_method: Mixing-height method ``+/-1`` for Maul-Carson or
            ``+/-2`` for Batchvarova-Gryning.
        ipsifcn: Momentum stability-function selector, 0 or 1.
        minimum_mixing_height_m: Lower profile mixing-height bound in metres.
        **mixing_options: Additional keyword arguments passed to the selected
            mixing-height routine.

    Returns:
        Eight float32 arrays ``(u_profile, v_profile, friction_velocity,
        monin_obukhov_length, sensible_heat, mixing_height,
        convective_height, temperature_jump)``. Wind profiles have shape
        ``(level, station)`` in m s-1; state arrays have shape ``(station,)``
        with units m s-1, m, W m-2, m, m, and K. Inputs are not modified.

    Raises:
        ValueError: If station or profile shapes disagree, ``mixing_method``
            or ``ipsifcn`` is unsupported, or required downstream inputs are
            missing.
        TypeError: If ``mixing_options`` contains an unsupported keyword for
            the selected mixing-height routine.
    """
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    if u.ndim!=1 or v.shape!=u.shape:
        raise ValueError('STHEOR station winds must be matching one-dimensional arrays')
    station_values=(roughness_m,anemometer_height_m,density,temperature_k,
                    cloud_fraction,sensible_heat_flux,
                    previous_convective_height_m,temperature_jump_k,coriolis,
                    station_latitude)
    converted=tuple(np.asarray(value,dtype=MODEL_DTYPE) for value in station_values)
    if any(value.shape!=u.shape for value in converted):
        raise ValueError('STHEOR station inputs must have identical shapes')
    (roughness,anemometer,rho,temperature,cloud,heat,previous,jump,
     coriolis_value,latitude)=converted
    profile_temperature=np.asarray(temperature_profile_k,dtype=MODEL_DTYPE)
    profile_height=np.asarray(profile_height_m,dtype=MODEL_DTYPE)
    if profile_temperature.shape!=profile_height.shape or profile_temperature.shape[1:]!=u.shape:
        raise ValueError('STHEOR temperature profiles must have shape (level, station)')
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    output_u=np.full((centers.size,u.size),9999.,dtype=MODEL_DTYPE)
    output_v=np.full_like(output_u,9999.)
    output_u[0]=u; output_v[0]=v
    state=[np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(6)]
    speed=np.sqrt(u*u+v*v)
    active=(u<=990.)&(v<=990.)&(speed>=.001)
    if not active.any():
        return cast(FloatEight,(output_u,output_v,*state))
    ustar,length,station_heat=land_station_surface_similarity(
        u[active],v[active],roughness[active],anemometer[active],
        first_level_height_m,rho[active],temperature[active],cloud[active],
        heat[active],ipsifcn=ipsifcn)
    mix_options: dict[str, MixingOption] = dict(mixing_options)
    if abs(int(mixing_method))==2:
        mix_options.setdefault('surface_temperature_k',temperature[active])
    mixing,convective,next_jump=land_station_mixing_height(
        profile_temperature[:,active],profile_height[:,active],previous[active],
        jump[active],ustar,length,station_heat,rho[active],coriolis_value[active],
        mixing_method=mixing_method,**mix_options)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],anemometer[active],centers,z_faces_m,
        roughness[active],length,mixing,latitude[active],ipsifcn=ipsifcn,
        minimum_mixing_height_m=minimum_mixing_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    for destination,source in zip(
            state,(ustar,length,station_heat,mixing,convective,next_jump),strict=True):
        destination[active]=source
    return cast(FloatEight,(output_u,output_v,*state))


def objective_analysis_component(station_values: ArrayLike,station_xy: ArrayLike,first_guess: ArrayLike,landuse: ArrayLike,dx_km: float,dy_km: float,
                                 *,rmin_km: float=.1,rmax_surface_km: float=50.,rmax_upper_km: float=100.,
                                 rmax_water_km: float=100.,surface_scale_km: float=10.,upper_scale_km: float=20.,
                                 stations_per_level: ArrayLike=5,observation_weight: ArrayLike=1.,vary_radius: bool=True,
                                 water_range: WaterRange=(50,55),barriers: ArrayLike=(),barrier_top_level: int=0,
                                 background_weight: ArrayLike | None=None,block_size: int=32768) -> Float32Array:
    """Analyze one wind component with CALMET's INTER2 interpolation.

    Args:
        station_values: Observed component values with shape
            ``(level, station)``. Values of 900 or greater are missing; units
            are preserved in the output.
        station_xy: Station coordinates with shape ``(station, 2)`` in
            kilometres in the CALMET grid coordinate system.
        first_guess: Background component with shape ``(level, y, x)`` and
            the same units as ``station_values``.
        landuse: Integer land-use categories with shape ``(y, x)``.
        dx_km: East-west grid spacing in kilometres.
        dy_km: South-north grid spacing in kilometres.
        rmin_km: Minimum station distance used in inverse-square weights, in
            kilometres.
        rmax_surface_km: Maximum land-station radius at level zero in
            kilometres.
        rmax_upper_km: Maximum land-station radius above level zero in
            kilometres.
        rmax_water_km: Maximum station radius over water in kilometres.
        surface_scale_km: Background weighting length at level zero in
            kilometres.
        upper_scale_km: Background weighting length above level zero in
            kilometres.
        stations_per_level: Number of nearest observations retained, either a
            scalar or an integer with shape ``(level,)``.
        observation_weight: Dimensionless observation multiplier, scalar or
            broadcastable to ``(level, y, x)``.
        vary_radius: Whether a target with no station inside its maximum
            radius falls back to the nearest valid station.
        water_range: Inclusive pair of land-use category numbers treated as
            water.
        barriers: Barrier endpoints with shape ``(barrier, 4)`` ordered as
            ``(x_begin, y_begin, x_end, y_end)`` in kilometres.
        barrier_top_level: Number of lowest model levels at which barriers
            can block observations.
        background_weight: Optional explicit background weight in km-2,
            scalar or broadcastable to ``(level, y, x)``. Scale-derived
            weights are used when omitted.
        block_size: Number of flattened target cells processed per block.

    Returns:
        A new float32 array with shape ``(level, y, x)`` containing the
        analyzed component in the same units as ``station_values`` and
        ``first_guess``.

    Raises:
        ValueError: If observation, coordinate, background, or land-use
            dimensions disagree; a station count is non-positive; barriers
            are malformed; or a required level or target has no valid,
            unblocked station.
    """
    work_dtype=ACCUMULATION_DTYPE
    values=np.asarray(station_values,dtype=work_dtype); points=np.asarray(station_xy,dtype=work_dtype)
    guess=np.asarray(first_guess,dtype=work_dtype); lu=np.asarray(landuse)
    if values.ndim!=2 or points.shape!=(values.shape[1],2):
        raise ValueError('station values must be (level,station) with matching XY coordinates')
    if guess.ndim!=3 or lu.shape!=guess.shape[1:]: raise ValueError('first guess and landuse shapes do not match')
    if guess.shape[0]!=values.shape[0]: raise ValueError('station and first-guess level counts differ')
    nz,ny,nx=guess.shape
    counts=np.broadcast_to(np.asarray(stations_per_level,dtype=int),(nz,))
    if np.any(counts<1): raise ValueError('stations_per_level/NINTRP must be positive')
    wo=np.broadcast_to(np.asarray(observation_weight,dtype=work_dtype),guess.shape)
    background_grid: Float64Array | None = (
        None if background_weight is None else
        np.broadcast_to(np.asarray(background_weight,dtype=work_dtype),guess.shape)
    )
    x=np.asarray((np.arange(nx,dtype=work_dtype)+work_dtype(.5))*work_dtype(dx_km),dtype=work_dtype)
    y=np.asarray((np.arange(ny,dtype=work_dtype)+work_dtype(.5))*work_dtype(dy_km),dtype=work_dtype)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    water=((lu>=water_range[0])&(lu<=water_range[1])).ravel(); output=np.empty_like(guess)
    # INTER2 tests STDAT1 .LT. EDITL; COMP defines EDITL as 900.
    valid=values<work_dtype(900.)
    barrier_array=np.asarray(barriers,dtype=float)
    if barrier_array.size==0: barrier_array=np.empty((0,4))
    if barrier_array.ndim!=2 or barrier_array.shape[1]!=4:
        raise ValueError('barriers must contain (x_begin,y_begin,x_end,y_end) rows')
    if vary_radius and background_grid is None and np.any(~valid.any(axis=1)):
        raise ValueError('INTER2 has a level with no valid station data')
    for start in range(0,targets.shape[0],block_size):
        stop=min(start+block_size,targets.shape[0])
        delta=np.asarray(targets[start:stop,None,:]-points[None,:,:],dtype=work_dtype)
        distance=np.sqrt(np.asarray(delta[:,:,0]**2+delta[:,:,1]**2,dtype=work_dtype))
        safe=np.maximum(distance,work_dtype(rmin_km))
        blocked=_barrier_blocked(targets[start:stop],points,barrier_array)
        for level in range(nz):
            if not valid[level].any() and background_grid is not None:
                output[level].ravel()[start:stop]=guess[level].ravel()[start:stop]
                continue
            radius=np.where(water[start:stop],work_dtype(rmax_water_km),
                            work_dtype(rmax_surface_km if level==0 else rmax_upper_km))
            eligible=valid[level][None,:]&(distance<=radius[:,None])
            if level<barrier_top_level: eligible &= ~blocked
            order=np.argsort(np.where(eligible,distance,np.inf),axis=1,kind='stable')[:,:counts[level]]
            selected=np.take_along_axis(eligible,order,axis=1)
            if vary_radius:
                empty=~selected.any(axis=1)
                if empty.any():
                    fallback_valid=np.broadcast_to(valid[level],distance.shape)
                    if level<barrier_top_level: fallback_valid=fallback_valid&~blocked
                    fallback_distance=np.where(fallback_valid,distance,np.inf)
                    if np.any(~np.isfinite(np.min(fallback_distance[empty],axis=1))):
                        raise ValueError('INTER2 has no unblocked valid station for a target cell')
                    nearest=np.argmin(fallback_distance,axis=1)
                    order[empty,0]=nearest[empty]; selected[empty,0]=True
            selected_distance=np.take_along_axis(safe,order,axis=1)
            selected_values=values[level,order]
            obs_factor=np.asarray(wo[level].ravel()[start:stop,None]/
                                  np.asarray(selected_distance**2,dtype=work_dtype),dtype=work_dtype)
            obs_factor=np.where(selected,obs_factor,work_dtype(0.)).astype(work_dtype)
            numerator=np.sum(
                selected_values*obs_factor,axis=1,dtype=ACCUMULATION_DTYPE)
            denominator=np.sum(
                obs_factor,axis=1,dtype=ACCUMULATION_DTYPE)
            scale=work_dtype(surface_scale_km if level==0 else upper_scale_km)
            background=(work_dtype(1.)/(scale*scale) if background_grid is None else
                        background_grid[level].ravel()[start:stop])
            output[level].ravel()[start:stop]=np.asarray(
                (numerator+guess[level].ravel()[start:stop]*background)/
                (denominator+background),dtype=work_dtype)
    return _model_array(output)


def wind1_initial_guess(station_u: ArrayLike,station_v: ArrayLike,station_xy: ArrayLike,surface_station_count: int,
                        nx: int,ny: int,dx_km: float,dy_km: float,*,bias: ArrayLike=0.,extrapolate_calm: bool=True,
                        barriers: ArrayLike=(),barrier_top_level: int=0,block_size: int=32768) -> tuple[Float32Array, Float32Array]:
    """Vectorized WIND1 all-station first-guess interpolation.

    Surface (including SEA) profiles precede upper-air profiles. Distances use
    a one-grid-cell floor before vertically varying group bias and
    per-level missing/calm/barrier removal.

    Args:
        station_u: Eastward station-wind profiles with shape
            ``(level, station)`` in m s-1. Values above 900 are missing.
        station_v: Northward station-wind profiles matching ``station_u`` in
            m s-1.
        station_xy: Station coordinates with shape ``(station, 2)`` in
            kilometres in the CALMET grid coordinate system.
        surface_station_count: Number of leading stations treated as surface
            or overwater stations; remaining stations are upper-air sites.
        nx: Number of grid columns.
        ny: Number of grid rows.
        dx_km: East-west grid spacing in kilometres.
        dy_km: South-north grid spacing in kilometres.
        bias: Relative surface-versus-upper-air weight in ``[-1, 1]``, either
            scalar or shape ``(level,)``. Positive values favor upper-air
            stations and negative values favor surface stations.
        extrapolate_calm: Whether calm surface observations contribute above
            the first model level.
        barriers: Barrier endpoints with shape ``(barrier, 4)`` ordered as
            ``(x_begin, y_begin, x_end, y_end)`` in kilometres.
        barrier_top_level: Number of lowest model levels at which barriers
            can block observations.
        block_size: Number of flattened target cells processed per block.

    Returns:
        Tuple ``(u_initial, v_initial)`` of new float32 arrays with shape
        ``(level, ny, nx)`` in m s-1. Inputs are not modified.

    Raises:
        ValueError: If profile or coordinate shapes disagree, the surface
            count or bias is invalid, grid spacing is non-positive, barriers
            are malformed, a positive bias lacks upper-air stations, or no
            valid unblocked observation remains for a target.
    """
    work_dtype=ACCUMULATION_DTYPE
    u=np.asarray(station_u,dtype=work_dtype); v=np.asarray(station_v,dtype=work_dtype)
    points=np.asarray(station_xy,dtype=work_dtype)
    if u.ndim!=2 or v.shape!=u.shape or points.shape!=(u.shape[1],2):
        raise ValueError('WIND1 station profiles and coordinates are inconsistent')
    nz,nstation=u.shape; nsurface=int(surface_station_count)
    if not 0<=nsurface<=nstation: raise ValueError('invalid WIND1 surface-station count')
    if (not np.isfinite(dx_km) or not np.isfinite(dy_km) or
            dx_km<=0. or dy_km<=0.):
        raise ValueError('WIND1 grid spacing must be finite and positive')
    layer_bias=np.broadcast_to(np.asarray(bias,dtype=work_dtype),(nz,))
    if np.any((layer_bias < work_dtype(-1.))|(layer_bias > work_dtype(1.))):
        raise ValueError('WIND1 BIAS must lie in [-1,1]')
    barrier_array=np.asarray(barriers,dtype=float)
    if barrier_array.size==0: barrier_array=np.empty((0,4))
    if barrier_array.ndim!=2 or barrier_array.shape[1]!=4:
        raise ValueError('barriers must contain (x_begin,y_begin,x_end,y_end) rows')
    cell_x=np.asarray(np.arange(nx,dtype=work_dtype)+work_dtype(.5),dtype=work_dtype)
    cell_y=np.asarray(np.arange(ny,dtype=work_dtype)+work_dtype(.5),dtype=work_dtype)
    grid_x,grid_y=np.meshgrid(cell_x,cell_y)
    grid_targets=np.column_stack((grid_x.ravel(),grid_y.ravel()))
    x=np.asarray(cell_x*work_dtype(dx_km),dtype=work_dtype)
    y=np.asarray(cell_y*work_dtype(dy_km),dtype=work_dtype)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    # Interpolation distance is expressed in grid-cell units, so a station
    # one column or one row away receives the same base weight even when the
    # cells are rectangular.  Barrier intersections continue to use the
    # physical kilometre coordinates in ``points`` and ``targets`` below.
    normalized_points=np.column_stack((
        points[:,0]/work_dtype(dx_km),
        points[:,1]/work_dtype(dy_km),
    ))
    output_u=np.empty((nz,ny*nx),dtype=work_dtype); output_v=np.empty_like(output_u)
    # WIND1 operates on internal US/VS arrays: COMP sets EDITL=900 and
    # every source check uses .GT. EDITL, not the external 9998/9999 code.
    surface_valid=(u[0,:nsurface]<=work_dtype(900.))&(v[0,:nsurface]<=work_dtype(900.))
    valid=(u<=work_dtype(900.))&(v<=work_dtype(900.))
    calm=np.sqrt(np.asarray(u[0,:nsurface]**2+v[0,:nsurface]**2,dtype=work_dtype))<work_dtype(1.e-4)
    def stable_sum(items: Float64Array) -> Float64Array:
        """Reduce station contributions in accumulation precision.

        Args:
            items: Numeric array with shape ``(target, station)`` and units
                inherited from the quantity being accumulated.

        Returns:
            A float64 array with shape ``(target,)`` and the same units as
            ``items``. The input is not modified.
        """
        return np.sum(items,axis=1,dtype=ACCUMULATION_DTYPE)
    def discard_weights(weights: Float64Array,remove: BoolArray) -> None:
        """Remove unavailable observations and renormalize target weights.

        Args:
            weights: Float64 interpolation weights with shape
                ``(target, station)``. This array is modified in place.
            remove: Boolean mask with the same shape; true entries are set to
                zero before affected rows are normalized.

        Returns:
            None. ``weights`` is updated in place.

        Raises:
            ValueError: If an affected target has no positive weight after
                removal.
        """
        affected=np.any(remove,axis=1)
        if not affected.any():
            return
        weights[remove]=work_dtype(0.)
        denominator=stable_sum(weights)
        if np.any(denominator[affected]<=work_dtype(0.)):
            raise ValueError('WIND1 has no valid unblocked observation')
        weights[affected]/=denominator[affected,None]
    for start in range(0,targets.shape[0],block_size):
        stop=min(start+block_size,targets.shape[0]); local=targets[start:stop]
        normalized_local=grid_targets[start:stop]
        delta=np.asarray(normalized_points[None,:,:]-normalized_local[:,None,:],dtype=work_dtype)
        distance2=np.asarray(delta[:,:,0]**2+delta[:,:,1]**2,dtype=work_dtype)
        base=np.asarray(work_dtype(1.)/np.maximum(distance2,work_dtype(1.)),dtype=work_dtype)
        base[:,:nsurface]*=surface_valid[None,:]
        total=stable_sum(base)[:,None]
        if np.any(total<=0.): raise ValueError('WIND1 has no station with valid surface wind')
        base=np.asarray(base/total,dtype=work_dtype)
        surface_sum=stable_sum(base[:,:nsurface])
        upper_sum=stable_sum(base[:,nsurface:])
        blocked=_barrier_blocked(local,points,barrier_array)
        for level in range(nz):
            weights=base.copy(); value=layer_bias[level]
            if value>=0.:
                weights[:,:nsurface]*=work_dtype(1.)-value
                desired=np.asarray(value+(work_dtype(1.)-value)*upper_sum,dtype=work_dtype)
                if nsurface<nstation:
                    weights[:,nsurface:]*=np.divide(desired,upper_sum,
                        out=np.zeros_like(desired),where=upper_sum>0.)[:,None]
                elif value>0.: raise ValueError('positive WIND1 BIAS requires upper-air stations')
            else:
                desired=np.asarray(-value+(work_dtype(1.)+value)*surface_sum,dtype=work_dtype)
                if nsurface:
                    weights[:,:nsurface]*=np.divide(desired,surface_sum,
                        out=np.zeros_like(desired),where=surface_sum>0.)[:,None]
                weights[:,nsurface:]*=work_dtype(1.)+value
            if not extrapolate_calm and level>0 and nsurface:
                calm_remove=np.zeros_like(weights,dtype=bool)
                calm_remove[:,:nsurface]=calm[None,:]&surface_valid[None,:]
                discard_weights(weights,calm_remove)
            if nsurface<nstation and value>work_dtype(-.9999):
                missing=np.zeros_like(weights,dtype=bool)
                missing[:,nsurface:]=~valid[level,nsurface:][None,:]
                discard_weights(weights,missing)
            if level<barrier_top_level:
                discard_weights(weights,blocked)
            output_u[level,start:stop]=stable_sum(
                np.asarray(weights*u[level][None,:],dtype=work_dtype))
            output_v[level,start:stop]=stable_sum(
                np.asarray(weights*v[level][None,:],dtype=work_dtype))
    return (_model_array(output_u.reshape(nz,ny,nx)),
            _model_array(output_v.reshape(nz,ny,nx)))


def _barrier_blocked(targets: ArrayLike,stations: ArrayLike,barriers: ArrayLike) -> BoolArray:
    """Test every target-to-station segment against CALMET barriers.

    Args:
        targets: Target coordinates with shape ``(target, 2)``.
        stations: Station coordinates with shape ``(station, 2)``.
        barriers: Barrier endpoints with shape ``(barrier, 4)`` ordered as
            ``(x_begin, y_begin, x_end, y_end)``. All coordinates must use
            the same units.

    Returns:
        A boolean array with shape ``(target, station)``. True marks a segment
        blocked by at least one barrier; inputs are not modified.
    """
    work_dtype=ACCUMULATION_DTYPE
    target=np.asarray(targets,dtype=work_dtype); station=np.asarray(stations,dtype=work_dtype)
    blocked=np.zeros((target.shape[0],station.shape[0]),dtype=bool)
    for xb,yb,xe,ye in np.asarray(barriers,dtype=work_dtype):
        dx=xb-xe; slope=9.9e9 if dx==0. else (yb-ye)/dx
        intercept=yb-slope*xb
        d_station=np.asarray(station[:,1]-slope*station[:,0]-intercept,dtype=work_dtype)
        d_target=np.asarray(target[:,1]-slope*target[:,0]-intercept,dtype=work_dtype)
        opposite=np.asarray(d_target[:,None]*d_station[None,:],dtype=work_dtype)<work_dtype(0.)
        a=np.column_stack((np.full(target.shape[0],xb,dtype=work_dtype)-target[:,0],np.full(target.shape[0],yb,dtype=work_dtype)-target[:,1]))
        b=np.column_stack((np.full(target.shape[0],xe,dtype=work_dtype)-target[:,0],np.full(target.shape[0],ye,dtype=work_dtype)-target[:,1]))
        c=station[None,:,:]-target[:,None,:]
        def cosine(left: Float64Array,right: Float64Array) -> Float64Array:
            """Calculate cosines between broadcast two-component vectors.

            Args:
                left: Vectors with final axis length 2.
                right: Vectors with final axis length 2 and broadcastable
                    leading axes.

            Returns:
                A dimensionless float64 array over the broadcast leading
                shape. Zero-length vector pairs return zero; inputs are not
                modified.
            """
            numerator=np.asarray(left[...,0]*right[...,0]+left[...,1]*right[...,1],dtype=work_dtype)
            left2=np.asarray(left[...,0]**2+left[...,1]**2,dtype=work_dtype)
            right2=np.asarray(right[...,0]**2+right[...,1]**2,dtype=work_dtype)
            denominator=np.sqrt(np.asarray(left2*right2,dtype=work_dtype))
            return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator!=work_dtype(0.))
        cos_ab=cosine(a,b)
        cos_ac=cosine(a[:,None,:],c)
        cos_bc=cosine(b[:,None,:],c)
        blocked |= opposite&(cos_ac>=cos_ab[:,None])&(cos_bc>=cos_ab[:,None])
    return blocked


def barrier_blocked_pairs(targets: ArrayLike, stations: ArrayLike, barriers: ArrayLike) -> BoolArray:
    """Return BARIER flags for target-specific candidate points.

    ``stations`` has shape ``(target, candidate, 2)``.  This is the layout
    used by CALMET's four prognostic neighbours and avoids constructing the
    prohibitively large target-by-full-prognostic-grid matrix.

    Args:
        targets: Target coordinates with shape ``(target, 2)``.
        stations: Target-specific candidate coordinates with shape
            ``(target, candidate, 2)``.
        barriers: Barrier endpoints with shape ``(barrier, 4)`` ordered as
            ``(x_begin, y_begin, x_end, y_end)``. All coordinates must use
            the same units.

    Returns:
        A boolean array with shape ``(target, candidate)``. True marks a
        target-candidate segment blocked by at least one barrier; inputs are
        not modified.

    Raises:
        ValueError: If target or candidate coordinate arrays have incompatible
            dimensions.
    """
    work_dtype=ACCUMULATION_DTYPE
    target=np.asarray(targets,dtype=work_dtype); station=np.asarray(stations,dtype=work_dtype)
    if target.ndim!=2 or target.shape[1]!=2:
        raise ValueError('targets must have shape (target, 2)')
    if station.ndim!=3 or station.shape[0]!=target.shape[0] or station.shape[2]!=2:
        raise ValueError('stations must have shape (target, candidate, 2)')
    blocked=np.zeros(station.shape[:2],dtype=bool)
    for xb,yb,xe,ye in np.asarray(barriers,dtype=work_dtype):
        dx=xb-xe; slope=9.9e9 if dx==0. else (yb-ye)/dx
        intercept=yb-slope*xb
        d_station=np.asarray(station[...,1]-slope*station[...,0]-intercept,dtype=work_dtype)
        d_target=np.asarray(target[:,1]-slope*target[:,0]-intercept,dtype=work_dtype)
        opposite=np.asarray(d_target[:,None]*d_station,dtype=work_dtype)<work_dtype(0.)
        a=np.column_stack((np.full(target.shape[0],xb,dtype=work_dtype)-target[:,0],
                           np.full(target.shape[0],yb,dtype=work_dtype)-target[:,1]))
        b=np.column_stack((np.full(target.shape[0],xe,dtype=work_dtype)-target[:,0],
                           np.full(target.shape[0],ye,dtype=work_dtype)-target[:,1]))
        c=station-target[:,None,:]
        def cosine(left: Float64Array,right: Float64Array) -> Float64Array:
            """Calculate cosines between broadcast two-component vectors.

            Args:
                left: Vectors with final axis length 2.
                right: Vectors with final axis length 2 and broadcastable
                    leading axes.

            Returns:
                A dimensionless float64 array over the broadcast leading
                shape. Zero-length vector pairs return zero; inputs are not
                modified.
            """
            numerator=np.asarray(left[...,0]*right[...,0]+left[...,1]*right[...,1],dtype=work_dtype)
            left2=np.asarray(left[...,0]**2+left[...,1]**2,dtype=work_dtype)
            right2=np.asarray(right[...,0]**2+right[...,1]**2,dtype=work_dtype)
            denominator=np.sqrt(np.asarray(left2*right2,dtype=work_dtype))
            return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator!=work_dtype(0.))
        cos_ab=cosine(a,b)
        cos_ac=cosine(a[:,None,:],c)
        cos_bc=cosine(b[:,None,:],c)
        blocked |= opposite&(cos_ac>=cos_ab[:,None])&(cos_bc>=cos_ab[:,None])
    return blocked
