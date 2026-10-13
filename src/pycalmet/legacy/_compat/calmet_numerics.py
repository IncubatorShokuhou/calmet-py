"""Small, independently testable numerical kernels from CALMET 6.5.

Array axes follow the Python convention ``(..., y, x)``.  These routines are
kept free of model-global state so the main solver can compose and regress
them one Fortran subroutine at a time.
"""
from __future__ import annotations
import ctypes
from functools import lru_cache
import math
import numpy as np


_PROCESS_LIBRARY=ctypes.CDLL(None)


def _real4_math(function,value):
    """Scalar libm operation followed by a Fortran REAL assignment."""
    array=np.asarray(value,dtype=np.float32)
    if array.ndim==0:
        return np.float32(function(float(array)))
    return np.frompyfunc(function,1,1)(array).astype(np.float32)


def _real4_libm(name,value):
    """GNU default-REAL libm intrinsic, including its float approximation."""
    array=np.asarray(value,dtype=np.float32)
    try:
        function=getattr(_PROCESS_LIBRARY,name)
    except AttributeError:
        return _real4_math(getattr(math,name[:-1]),array)
    function.argtypes=(ctypes.c_float,)
    function.restype=ctypes.c_float
    scalar=lambda item: function(ctypes.c_float(item))
    if array.ndim==0:
        return np.float32(scalar(array))
    return np.frompyfunc(scalar,1,1)(array).astype(np.float32)


def _real4_atan2(y,x):
    left,right=np.broadcast_arrays(np.asarray(y,dtype=np.float32),
                                   np.asarray(x,dtype=np.float32))
    function=getattr(_PROCESS_LIBRARY,'atan2f')
    function.argtypes=(ctypes.c_float,ctypes.c_float)
    function.restype=ctypes.c_float
    scalar=lambda a,b: function(ctypes.c_float(a),ctypes.c_float(b))
    if left.ndim==0: return np.float32(scalar(left,right))
    return np.frompyfunc(scalar,2,1)(left,right).astype(np.float32)


def _real4_power(base,exponent):
    """GNU default-REAL ``powf`` used for non-integral REAL powers."""
    left,right=np.broadcast_arrays(np.asarray(base,dtype=np.float32),
                                   np.asarray(exponent,dtype=np.float32))
    try:
        function=getattr(_PROCESS_LIBRARY,'powf')
    except AttributeError:
        function=None
    if function is not None:
        function.argtypes=(ctypes.c_float,ctypes.c_float)
        function.restype=ctypes.c_float
        scalar=lambda base,power: function(ctypes.c_float(base),ctypes.c_float(power))
    else:
        scalar=math.pow
    if left.ndim==0:
        return np.float32(scalar(left,right))
    return np.frompyfunc(scalar,2,1)(left,right).astype(np.float32)


def _psiud(z_over_l):
    """CALMET's double-precision COARE/Dyer wind stability function."""
    return psi_momentum_dyer(z_over_l)


def _psiuc(height,monin_obukhov,fit=16):
    """Legacy REAL PSIUC stability function used when IPSIFCN=1."""
    if fit not in {15,16}: raise ValueError('PSIUC fit must be 15 or 16')
    f32=np.float32
    height,length=np.broadcast_arrays(np.asarray(height,dtype=f32),
                                      np.asarray(monin_obukhov,dtype=f32))
    output=np.empty(height.shape,dtype=f32); stable=length>f32(0.)
    if stable.any():
        ratio=np.asarray(np.asarray(-f32(.29)*height,dtype=f32)/length,dtype=f32)
        output[stable]=np.asarray(-f32(17.)*np.asarray(
            f32(1.)-_real4_libm('expf',ratio[stable]),dtype=f32),dtype=f32)
    unstable=~stable
    if unstable.any():
        ratio=np.asarray(np.asarray(f32(fit)*height,dtype=f32)/length,dtype=f32)
        x=_real4_power(np.asarray(f32(1.)-ratio,dtype=f32),f32(.25))
        first=_real4_libm('logf',np.asarray(f32(.5)*np.asarray(f32(1.)+x,dtype=f32),dtype=f32))
        second=_real4_libm('logf',np.asarray(f32(.5)*np.asarray(
            f32(1.)+np.asarray(x*x,dtype=f32),dtype=f32),dtype=f32))
        output[unstable]=np.asarray(
            np.asarray(f32(2.)*first[unstable],dtype=f32)+second[unstable]-
            np.asarray(f32(2.)*_real4_libm('atanf',x[unstable]),dtype=f32)+
            f32(1.5707963),dtype=f32)
    return output


def similarity_wind_profiles(u,v,anemometer_height_m,z_centers_m,z_faces_m,
        roughness_m,monin_obukhov_m,mixing_height_m,station_latitude,*,
        ipsifcn=0,minimum_mixing_height_m=50.):
    """Vectorized SIMILT station profiles with CALMET REAL precision."""
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    anem=np.asarray(anemometer_height_m,dtype=f32)
    roughness=np.asarray(roughness_m,dtype=f32); ell=np.asarray(monin_obukhov_m,dtype=f32)
    mixing=np.asarray(mixing_height_m,dtype=f32); latitude=np.asarray(station_latitude,dtype=f32)
    centers=np.asarray(z_centers_m,dtype=f32); faces=np.asarray(z_faces_m,dtype=f32)
    if any(value.shape!=u.shape for value in (v,anem,roughness,ell,mixing,latitude)):
        raise ValueError('SIMILT station inputs must have identical shapes')
    if centers.ndim!=1 or faces.shape!=(centers.size+1,):
        raise ValueError('SIMILT vertical coordinates are inconsistent')
    if ipsifcn not in {0,1}: raise ValueError('IPSIFCN must be zero or one')
    adjusted=ell.copy(); short=np.abs(adjusted)<np.asarray(f32(5.)*roughness,dtype=f32)
    adjusted[short]=np.where(adjusted[short]<f32(0.),
        np.asarray(-f32(5.)*roughness[short],dtype=f32),
        np.asarray(f32(5.)*roughness[short],dtype=f32))
    inverse=np.where(np.abs(adjusted)>=f32(10000.),f32(0.),
                     np.asarray(f32(1.)/adjusted,dtype=f32)).astype(f32)
    speed=_real4_libm('sqrtf',np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32))
    angle=np.asarray(f32(270.)-np.asarray(
        _real4_atan2(v,u)*f32(57.2957795),dtype=f32),dtype=f32)
    direction=np.asarray(np.fmod(angle,f32(360.)),dtype=f32)
    direction=np.where(direction==f32(0.),f32(360.),direction).astype(f32)
    log_anem=_real4_libm('logf',np.asarray(anem/roughness,dtype=f32))
    neutral=np.abs(inverse)<f32(.00005)
    psi_anem=(np.asarray(_psiud(np.asarray(anem/adjusted,dtype=f32).astype(np.float64)),dtype=f32)
              if ipsifcn==0 else _psiuc(anem,adjusted,16))
    psi_anem=np.where(neutral,f32(0.),psi_anem).astype(f32)
    inverse_table=np.array([-.03333333,-.01,-.0027027,0.,.002857,.007692,
                            .0166667,.05,.1111111],dtype=f32)
    angle_table=np.array([12.,10.,9.,12.,18.,28.,35.,38.,39.],dtype=f32)
    indices=np.searchsorted(inverse_table,inverse,side='left')
    turn=np.full(u.shape,angle_table[-1],dtype=f32); low=indices==0; turn[low]=angle_table[0]
    middle=(indices>0)&(indices<inverse_table.size); lower=indices[middle]-1; upper=indices[middle]
    ratio=np.asarray(np.asarray(inverse[middle]-inverse_table[lower],dtype=f32)/
                     np.asarray(inverse_table[upper]-inverse_table[lower],dtype=f32),dtype=f32)
    turn[middle]=np.asarray(angle_table[lower]+np.asarray(
        np.asarray(angle_table[upper]-angle_table[lower],dtype=f32)*ratio,dtype=f32),dtype=f32)
    turn=np.where(neutral,f32(12.),turn).astype(f32)
    decay_anem=_real4_libm('expf',np.asarray(-anem/f32(200.),dtype=f32))
    at_anem=np.asarray(np.asarray(turn*f32(1.58),dtype=f32)*
                       np.asarray(f32(1.)-decay_anem,dtype=f32),dtype=f32)
    output_u=np.full((centers.size,u.size),f32(9999.),dtype=f32)
    output_v=np.full_like(output_u,f32(9999.))
    limit=np.maximum(np.asarray(mixing+f32(.001),dtype=f32),f32(minimum_mixing_height_m))
    for level,center in enumerate(centers):
        active=faces[level+1]<=limit
        if not active.any(): continue
        log_level=_real4_libm('logf',np.asarray(center/roughness,dtype=f32))
        psi=(np.asarray(_psiud(np.asarray(center/adjusted,dtype=f32).astype(np.float64)),dtype=f32)
             if ipsifcn==0 else _psiuc(np.full_like(adjusted,center),adjusted,16))
        psi=np.where(neutral,f32(0.),psi).astype(f32)
        numerator=np.asarray(speed*np.asarray(log_level-psi,dtype=f32),dtype=f32)
        magnitude=np.asarray(numerator/np.asarray(log_anem-psi_anem,dtype=f32),dtype=f32)
        decay=_real4_libm('expf',np.full(u.shape,-center/f32(200.),dtype=f32))
        shear=np.asarray(np.asarray(turn*f32(1.58),dtype=f32)*
                         np.asarray(f32(1.)-decay,dtype=f32),dtype=f32)
        north=np.asarray(np.asarray(direction+shear,dtype=f32)-at_anem,dtype=f32)
        north=np.where(north>f32(360.),np.asarray(north-f32(360.),dtype=f32),north)
        south=np.asarray(np.asarray(direction-shear,dtype=f32)+at_anem,dtype=f32)
        south=np.where(south<=f32(0.),np.asarray(f32(360.)+south,dtype=f32),south)
        wd=np.where(latitude>f32(0.),north,south).astype(f32)
        radians=np.asarray(f32(.0174533)*wd,dtype=f32)
        output_u[level,active]=np.asarray(-magnitude[active]*_real4_libm('sinf',radians[active]),dtype=f32)
        output_v[level,active]=np.asarray(-magnitude[active]*_real4_libm('cosf',radians[active]),dtype=f32)
    return output_u,output_v

def cloud_liquid_water_profile(
        cloud_water_g_kg,height_agl_m,z_faces_m,threshold_g_kg=.005):
    """Vectorized Z2FACE/PRFVAR mapping used by CALMET.AUX output."""
    cloud=np.asarray(cloud_water_g_kg,dtype=np.float32)
    height=np.asarray(height_agl_m,dtype=np.float32)
    target=np.asarray(z_faces_m,dtype=np.float32)
    if cloud.shape!=height.shape or cloud.ndim<1 or cloud.shape[0]<1:
        raise ValueError('cloud water and height profiles must have identical shapes')
    if target.ndim!=1 or target.size<2 or np.any(np.diff(target)<=0.):
        raise ValueError('CALMET cloud-water faces must be a strictly increasing vector')
    levels=cloud.shape[0]
    source_faces=np.empty((levels+1,*cloud.shape[1:]),dtype=np.float32)
    source_faces[0]=np.where(height[0]<=np.float32(0.),
                             height[0]-np.float32(1.),np.float32(0.))
    if levels>1:
        source_faces[1:levels]=np.asarray(
            np.float32(.5)*np.asarray(height[:-1]+height[1:],dtype=np.float32),
            dtype=np.float32)
    source_faces[-1]=np.asarray(
        np.float32(2.)*height[-1]-source_faces[-2],dtype=np.float32)

    mapped=np.zeros((target.size-1,*cloud.shape[1:]),dtype=np.float32)
    for output_index,(bottom_face,top_face) in enumerate(zip(target[:-1],target[1:])):
        thickness=np.float32(top_face-bottom_face)
        for source_index in range(levels):
            upper=source_faces[source_index+1]
            lower=np.maximum(source_faces[source_index],np.float32(0.))
            overlap=(upper>bottom_face)&(lower<top_face)&(upper>np.float32(0.))
            depth=np.asarray(
                np.minimum(top_face,upper)-np.maximum(bottom_face,lower),
                dtype=np.float32)
            contribution=np.asarray(
                cloud[source_index]*depth/thickness,dtype=np.float32)
            mapped[output_index]=np.where(
                overlap,np.asarray(mapped[output_index]+contribution,dtype=np.float32),
                mapped[output_index])

    threshold=np.float32(threshold_g_kg)
    model_top=target[-1]
    weighted=np.zeros(cloud.shape[1:],dtype=np.float32)
    weight=np.zeros(cloud.shape[1:],dtype=np.float32)
    bottom=np.asarray(source_faces[-1],dtype=np.float32).copy()
    top=np.zeros(cloud.shape[1:],dtype=np.float32)
    for source_index in range(levels):
        lower=source_faces[source_index]; upper=source_faces[source_index+1]
        selected=(upper>model_top)&(cloud[source_index]>=threshold)
        depth=np.asarray(upper-lower,dtype=np.float32)
        weighted=np.where(selected,np.asarray(
            weighted+np.asarray(cloud[source_index]*depth,dtype=np.float32),
            dtype=np.float32),weighted)
        weight=np.where(selected,np.asarray(weight+depth,dtype=np.float32),weight)
        bottom=np.where(selected,np.minimum(bottom,lower),bottom)
        top=np.where(selected,np.maximum(top,upper),top)
    average=np.divide(weighted,weight,out=np.zeros_like(weighted),where=weight>0.)
    reported=average>=threshold
    return (mapped,np.where(reported,average,0.).astype(np.float32),
            np.where(reported,bottom,0.).astype(np.float32),
            np.where(reported,top,0.).astype(np.float32))

def air_density(pressure_mb, temperature_k, nearest_station):
    """AIRDen: ideal-gas density using the nearest surface station."""
    pressure=np.asarray(pressure_mb,dtype=np.float32)
    nearest=np.asarray(nearest_station,dtype=np.intp)
    return density_from_pressure_temperature(pressure[nearest],temperature_k)

def density_from_pressure_temperature(pressure_mb,temperature_k):
    """AIRDEN's left-to-right REAL*4 ideal-gas expression."""
    numerator=np.asarray(np.float32(.3484321)*np.asarray(pressure_mb,dtype=np.float32),
                         dtype=np.float32)
    return np.asarray(numerator/np.asarray(temperature_k,dtype=np.float32),dtype=np.float32)

def coriolis_parameter(latitude_degrees, configured_value=999.):
    """MICROI absolute Coriolis field, including its equatorial floor."""
    latitude=np.asarray(latitude_degrees,dtype=np.float32)
    if configured_value>998.:
        twopi=np.float32(np.float32(2.)*np.float32(3.141592))
        omega=np.asarray(twopi/np.float32(86400.),dtype=np.float32)
        angle=np.asarray(latitude*twopi/np.float32(360.),dtype=np.float32)
        value=np.asarray(np.float32(2.)*omega*np.sin(angle),dtype=np.float32)
    else:
        value=np.full_like(latitude,np.float32(configured_value))
    return np.maximum(np.abs(value),np.float32(.25e-4)).astype(np.float32)

def inverse_distance_squared(values, point_x, point_y, sample_x, sample_y, *, ignore_zero=False):
    """R2INTERP/CINTERP, vectorised over any leading target dimensions."""
    values=np.asarray(values); dx=np.asarray(sample_x)-np.asarray(point_x)[...,None]; dy=np.asarray(sample_y)-np.asarray(point_y)[...,None]
    weights=1.0/np.maximum(dx*dx+dy*dy,0.001)
    if ignore_zero: weights=np.where(values>=1.e-6,weights,0.0)
    denominator=weights.sum(axis=-1)
    return np.divide((values*weights).sum(axis=-1),denominator,out=np.zeros_like(denominator,dtype=np.result_type(values,float)),where=denominator>1.e-9)


def lake_breeze_adjust(u, v, station_u, station_v, station_xy_m,
                       station_ids, regions, dx_m, dy_m):
    """LLBREEZ/INTERB lowest-layer interpolation by coast distance.

    Region tuples contain ``XG1,XG2,YG1,YG2`` grid-line bounds, two
    coastline endpoints in relative metres, and the ordered station IDs.
    """
    result_u=np.asarray(u).copy()
    result_v=np.asarray(v).copy()
    if result_u.shape!=result_v.shape or result_u.ndim!=3:
        raise ValueError('lake-breeze U/V fields must share (level,y,x) shape')
    ids=np.asarray(station_ids,dtype=np.int64)
    xy=np.asarray(station_xy_m,dtype=np.float32)
    su=np.asarray(station_u,dtype=np.float32); sv=np.asarray(station_v,dtype=np.float32)
    if xy.shape!=(ids.size,2) or su.shape!=(ids.size,) or sv.shape!=(ids.size,):
        raise ValueError('lake-breeze station arrays have inconsistent shapes')
    ny,nx=result_u.shape[1:]
    grid_i=np.arange(1,nx+1,dtype=np.float32); grid_j=np.arange(1,ny+1,dtype=np.float32)
    xx=(grid_i-np.float32(.5))*np.float32(dx_m)
    yy=(grid_j-np.float32(.5))*np.float32(dy_m)
    for region in regions:
        if len(region)!=9:
            raise ValueError('lake-breeze region must contain eight coordinates and station IDs')
        xg1,xg2,yg1,yg2,x_begin,y_begin,x_end,y_end=(
            np.float32(value) for value in region[:8])
        selected=region[8]
        selected=np.asarray(selected,dtype=np.int64)
        if selected.size==0:
            raise ValueError('lake-breeze region must contain at least one station')
        lookup={}
        for index,value in enumerate(ids): lookup.setdefault(int(value),index)
        try: indices=np.asarray([lookup[int(value)] for value in selected],dtype=np.intp)
        except KeyError as error:
            raise ValueError(f'lake-breeze station ID {error.args[0]} is not available') from error
        xcd=np.float32(x_begin-x_end); ycd=np.float32(y_begin-y_end)
        slope=np.float32(9.9e19) if xcd==0. else np.float32(ycd/xcd)
        intercept=np.float32(y_begin-slope*x_begin)
        sx=xy[indices,0]; sy=xy[indices,1]
        if xcd==0.:
            station_distance=np.abs(sx-x_end)
        elif ycd==0.:
            station_distance=np.abs(sy-y_end)
        else:
            a=np.abs(sx*slope+intercept-sy)
            b=np.abs((sy-intercept)/slope-sx)
            hypotenuse=np.sqrt(np.asarray(a*a+b*b,dtype=np.float32))
            station_distance=np.divide(np.asarray(a*b,dtype=np.float32),hypotenuse,
                out=np.zeros_like(a),where=hypotenuse>0.)
        station_sign=np.asarray(sy-slope*sx-intercept,dtype=np.float32)
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
            hypotenuse=np.sqrt(np.asarray(a*a+b*b,dtype=np.float32))
            grid_distance=np.divide(np.asarray(a*b,dtype=np.float32),hypotenuse,
                out=np.zeros_like(a),where=hypotenuse>0.)
        grid_sign=np.asarray(gy-slope*gx-intercept,dtype=np.float32)
        same=grid_sign[:,None]*station_sign[None,:]>=0.
        relative=np.asarray(np.where(same,station_distance[None,:]-grid_distance[:,None],
                          station_distance[None,:]+grid_distance[:,None]),dtype=np.float32)
        dpr=np.maximum(np.abs(relative.astype(np.float64)),1.e-10)
        weights=1./(dpr*dpr)
        values_u=su[indices]; values_v=sv[indices]
        valid_u=values_u<990.; valid_v=values_v<990.
        if valid_u.any() and valid_v.any():
            def interpolate(values,valid):
                top=np.zeros(target_x.size,dtype=np.float32)
                bottom=np.zeros(target_x.size,dtype=np.float64)
                for station in np.flatnonzero(valid):
                    top=np.asarray(top+values[station]*weights[:,station],dtype=np.float32)
                    bottom+=weights[:,station]
                return np.asarray(top/bottom,dtype=np.float32)
            interpolated_u=interpolate(values_u,valid_u)
            interpolated_v=interpolate(values_v,valid_v)
            result_u[0,target_y,target_x]=interpolated_u
            result_v[0,target_y,target_x]=interpolated_v
    return result_u,result_v

def wind_speed_direction(u,v):
    """RTHETA meteorological speed and direction (calm direction is zero)."""
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    squared=np.asarray(u*u+v*v,dtype=f32); speed=np.sqrt(squared)
    factor=f32(f32(180.)/f32(3.141592654))
    angle=np.asarray(f32(270.)-np.asarray(np.arctan2(v,u)*factor,dtype=f32),dtype=f32)
    direction=np.fmod(angle,f32(360.)).astype(f32)
    direction=np.where(squared<f32(1.e-10),f32(0.),
                       np.where(direction==f32(0.),f32(360.),direction)).astype(f32)
    return speed,direction

def cloud_fraction(relative_humidity_percent):
    """CLOUD3 Teixeira diagnostic cloud fraction."""
    f32=np.float32
    rh=np.asarray(relative_humidity_percent,dtype=f32)
    rhd=np.asarray(rh/f32(100.),dtype=f32); gap=np.asarray(f32(1.)-rhd,dtype=f32)
    result=np.ones_like(rhd,dtype=f32)
    active=rh<f32(99.)
    root=_real4_libm('sqrtf',np.asarray(f32(1.)+
        np.asarray(f32(100.)*gap[active],dtype=f32),dtype=f32))
    result[active]=np.asarray(f32(.02)*np.asarray(f32(-1.)+root,dtype=f32)/gap[active],dtype=f32)
    return result

def cloud_cover_teixeira_850(relative_humidity_percent,pressure_mb):
    """CLOUD3 cover using the last prognostic level at or above 850 mb."""
    rh=np.asarray(relative_humidity_percent,dtype=np.float32)
    pressure=np.asarray(pressure_mb,dtype=np.float32)
    if rh.shape!=pressure.shape or rh.ndim!=3:
        raise ValueError('cloud profiles must share shape (z,y,x)')
    below=pressure<850.; first=np.argmax(below,axis=0); level=np.maximum(first-1,0)
    yy,xx=np.indices(level.shape)
    rh850=np.where(below.any(axis=0),rh[level,yy,xx],0.)
    return cloud_fraction(rh850)

def cloud_cover_mm5(relative_humidity_percent, pressure_mb, height_m):
    """CLOUD4 low/middle/high RH cloud cover and ceiling height.

    Inputs use ``(z, y, x)`` axes.  The pressure thresholds and integer
    midpoint choice intentionally match CALMET's original CLOUD4 routine.
    """
    f32=np.float32
    rh=np.asarray(relative_humidity_percent,dtype=f32)
    pressure=np.asarray(pressure_mb,dtype=f32)
    height=np.asarray(height_m,dtype=f32)
    if rh.shape!=pressure.shape or rh.shape!=height.shape or rh.ndim!=3: raise ValueError('cloud profiles must share shape (z,y,x)')
    nz=rh.shape[0]; levels=np.arange(nz)[:,None,None]
    def first_below(threshold):
        mask=pressure<threshold
        return np.where(mask.any(axis=0),mask.argmax(axis=0),nz)
    klo,kmid,khi=first_below(970.),first_below(800.),first_below(450.)
    def band_max(start,stop=None):
        mask=levels>=start[None,:,:]
        if stop is not None: mask &= levels<stop[None,:,:]
        return np.max(np.where(mask,rh,f32(0.)),axis=0)
    low=np.asarray(f32(4.)*band_max(klo,kmid)/f32(100.)-f32(3.),dtype=f32)
    middle=np.asarray(f32(4.)*band_max(kmid,khi)/f32(100.)-f32(3.),dtype=f32)
    high=np.asarray(f32(2.5)*band_max(khi)/f32(100.)-f32(1.5),dtype=f32)
    low=np.clip(low,f32(0.),f32(1.)); middle=np.clip(middle,f32(0.),f32(1.))
    high=np.clip(high,f32(0.),f32(1.))
    cover=np.maximum.reduce((low,middle,high))
    low_level=klo+(kmid-klo)//2; middle_level=kmid+(khi-kmid)//2
    ceiling_level=np.where(low>=.5,low_level,np.where(middle>=.5,middle_level,np.where(high>=.5,khi,
        np.where((low>=middle)&(low>=high),low_level,np.where((middle>=low)&(middle>=high),middle_level,khi)))))
    valid=(cover>1.e-9)&(ceiling_level<nz)
    ceiling=np.zeros_like(cover,dtype=f32)
    yy,xx=np.indices(cover.shape); ceiling[valid]=height[ceiling_level[valid],yy[valid],xx[valid]]
    return cover,ceiling

def lapse_rate(temperature_k, previous_convective_height_m, z_centers_m, height_m=200., *, spatial=True):
    """CGAMMA2 prognostic temperature lapse rate (K/m)."""
    f32=np.float32
    temp=np.asarray(temperature_k,dtype=f32)
    zi=np.asarray(previous_convective_height_m,dtype=f32)
    centers=np.asarray(z_centers_m,dtype=f32); height=f32(height_m)
    if temp.ndim!=3 or temp.shape[1:]!=zi.shape or temp.shape[0]!=centers.size: raise ValueError('temperature must be (z,y,x) and match heights/grid')
    upper=np.flatnonzero(centers>=height)
    if not upper.size: raise ValueError('lapse-rate height is above the top model level')
    k=int(upper[0])
    if k==0: raise ValueError('lapse-rate height is below the lowest model midpoint')
    ratio=f32(f32(centers[k]-height)/f32(centers[k]-centers[k-1]))
    at_height=np.asarray(temp[k]-np.asarray((temp[k]-temp[k-1])*ratio,dtype=f32),dtype=f32)
    diagnosed=np.asarray((at_height-temp[0])/f32(height-centers[0]),dtype=f32)
    gamma=np.where(zi>=height,f32(-.0098),diagnosed).astype(f32)
    if spatial: return gamma
    # CGAMMA2 accumulates AVGAMMA with I outer and J inner.  NumPy's pairwise
    # reduction has a different REAL*4 rounding tree, so preserve this rare
    # domain-average option's source order explicitly.
    average=f32(0.)
    for i in range(gamma.shape[1]):
        for j in range(gamma.shape[0]): average=f32(average+gamma[j,i])
    average=f32(average/f32(gamma.size))
    return np.full_like(gamma,average,dtype=f32)

def froude_adjustment(u, v, terrain_m, maximum_terrain_m, gamma_k_per_m,
                      temperature_k, z_centers_m, critical_froude,
                      dx_m, dy_m):
    """FRADJ terrain-blocking rotation with wind speed conserved."""
    u=np.array(u,dtype=np.float32,copy=True)
    v=np.array(v,dtype=np.float32,copy=True)
    f32=np.float32
    terrain=np.asarray(terrain_m,dtype=f32); maximum=np.asarray(maximum_terrain_m,dtype=f32)
    gamma=np.asarray(gamma_k_per_m,dtype=f32); temp=np.asarray(temperature_k,dtype=f32)
    centers=np.asarray(z_centers_m,dtype=f32)
    if u.shape!=v.shape or u.ndim!=3 or u.shape[1:]!=terrain.shape or u.shape[0]!=centers.size: raise ValueError('wind must have shape (z,y,x) matching grid and centers')
    west=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    gx=np.asarray((east-west)*(f32(.5)/f32(dx_m)),dtype=f32)
    gy=np.asarray((north-south)*(f32(.5)/f32(dy_m)),dtype=f32)
    flat=(np.abs(gx)<f32(.00015))&(np.abs(gy)<f32(.00015))
    gx=np.where(flat,f32(0.),gx).astype(f32); gy=np.where(flat,f32(0.),gy).astype(f32)
    magnitude=np.sqrt(np.asarray(gx*gx+gy*gy,dtype=f32))
    # FRADJ converts the gradient through degrees using two rounded REAL*4
    # constants (DPR and RPD), rather than normalising the vector directly.
    # They are not exact inverses; preserving the angle path determines the
    # tangent side for winds that are almost perfectly parallel to a slope.
    fx=np.asarray(gx,dtype=np.float32); fy=np.asarray(gy,dtype=np.float32)
    thetp=np.asarray(_real4_libm('atanf',np.divide(
        fy,fx,out=np.zeros_like(fy),where=fx!=0.))*np.float32(57.296),dtype=np.float32)
    theta=np.zeros_like(fx)
    theta=np.where((fx==0.)&(fy<0.),np.float32(270.),theta)
    theta=np.where((fx==0.)&(fy>0.),np.float32(90.),theta)
    theta=np.where(fx<0.,thetp+np.float32(180.),theta)
    theta=np.where((fx>0.)&(fy>0.),thetp,theta)
    theta=np.where((fx>0.)&(fy<=0.),thetp+np.float32(360.),theta)
    thetad=np.where((theta>=0.)&(theta<=90.),np.float32(90.)-theta,
                    np.float32(450.)-theta)
    angle=np.asarray((np.float32(270.)-thetad)*np.float32(.017453),
                     dtype=np.float32)
    drx=np.asarray(-_real4_libm('cosf',angle),dtype=np.float32)
    dry=np.asarray(-_real4_libm('sinf',angle),dtype=np.float32)
    drx=np.where(magnitude>0.,drx,0.); dry=np.where(magnitude>0.,dry,0.)
    # The legacy source spells both square roots as ``**0.5``.  GFortran
    # therefore calls powf rather than sqrtf; the two correctly rounded
    # float results can differ by one ulp and the tangent projection carries
    # that bit directly into both adjusted wind components.
    speed=np.asarray(_real4_power(np.asarray(u*u+v*v,dtype=f32),f32(.5)),
                     dtype=f32)
    obstacle=np.asarray(maximum[None,:,:]-
                        (terrain[None,:,:]+centers[:,None,None]),dtype=f32)
    gamma2=np.asarray(gamma+f32(.01),dtype=f32)
    frequency=np.asarray(_real4_power(np.maximum(
        f32(0.),np.asarray(f32(9.8)*gamma2/temp,dtype=f32)),f32(.5)),dtype=f32)
    denominator=np.asarray(frequency[None,:,:]*obstacle,dtype=f32)
    froude=np.divide(speed,denominator,out=np.full_like(speed,np.inf),where=denominator>0)
    un=np.divide(u,speed,out=np.zeros_like(u),where=speed>0)
    vn=np.divide(v,speed,out=np.zeros_like(v),where=speed>0)
    toward=np.asarray(drx[None,:,:]*un+dry[None,:,:]*vn,dtype=f32)
    tangent=np.asarray(-dry[None,:,:]*un+drx[None,:,:]*vn,dtype=f32)
    tangent_sign=np.where(tangent>f32(0.),f32(1.),f32(-1.))
    active=((speed>0)&(obstacle>0)&(gamma2[None,:,:]>0)&
            (froude<=f32(critical_froude))&(toward>0)&(magnitude[None,:,:]>0))
    tangent_u=tangent_sign*(-dry[None,:,:]); tangent_v=tangent_sign*drx[None,:,:]
    u=np.where(active,np.asarray(tangent_u*speed,dtype=f32),u).astype(f32)
    v=np.where(active,np.asarray(tangent_v*speed,dtype=f32),v).astype(f32)
    return u,v

def profile_temperature_layer(temperature_k, height_m, bottom_m, depth_m=200., *, minimum_lapse=.001):
    """MIXDT2 temperatures and potential-temperature lapse above ``bottom``.

    Profiles use ``(z, y, x)`` axes and must bracket both requested heights.
    Unlike a generic interpolator this reproduces MIXDT2's linear interpolation
    and its dry-adiabatic conversion to potential-temperature lapse.
    """
    temperature=np.asarray(temperature_k,dtype=np.float32)
    height=np.asarray(height_m,dtype=np.float32)
    bottom=np.asarray(bottom_m,dtype=np.float32)
    if temperature.shape!=height.shape or temperature.ndim!=3 or temperature.shape[1:]!=bottom.shape: raise ValueError('profiles must be (z,y,x) and bottom must match their grid')
    flat_t=temperature.reshape(temperature.shape[0],-1); flat_z=height.reshape(height.shape[0],-1); columns=np.arange(flat_t.shape[1])
    def interpolate(target):
        target=np.asarray(target,dtype=np.float32).ravel()
        upper=np.sum(flat_z<=target[None,:],axis=0)
        if np.any(upper==0) or np.any(upper>=flat_z.shape[0]): raise ValueError('temperature profile does not bracket mixing-height layer')
        lower=upper-1; zl=flat_z[lower,columns]; zu=flat_z[upper,columns]
        if np.any(zu<=zl): raise ValueError('temperature profile heights must increase strictly')
        # Literal MIXDT2 expression: starting from the upper temperature is
        # observably different from the algebraically equivalent lower-point
        # form after each default-REAL operation.
        value=np.asarray(flat_t[upper,columns]-
            (flat_t[upper,columns]-flat_t[lower,columns])*(zu-target)/(zu-zl),
            dtype=np.float32)
        return value.reshape(bottom.shape)
    depth=np.float32(depth_m)
    lower=interpolate(bottom); upper=interpolate(np.asarray(bottom+depth,dtype=np.float32))
    lapse=np.maximum(np.asarray((upper-lower)/depth+np.float32(.0098),dtype=np.float32),
                     np.float32(minimum_lapse))
    return lower,upper,lapse

def mixing_height_maul_carson(temperature_profile_k, profile_height_m, previous_convective_height_m,
                               temperature_jump_k, friction_velocity, monin_obukhov_m,
                               sensible_heat_flux, density, coriolis,
                               *, timestep_s=3600., depth_m=200., minimum_lapse=.001,
                               minimum_height_m=50., maximum_height_m=2500.,
                               threshold_flux_per_m=0., conste=.15, constn=2400.,
                               constb=1.41, relaxation=True, relaxation_time_s=800., land_mask=None):
    """MIXHT2 + MIXHMC for CALMET's ``IMIXH=1`` land-cell path.

    Returns total height, convective height, and the updated temperature jump.
    The implementation is array-vectorised across the horizontal grid.
    """
    old=np.asarray(previous_convective_height_m,dtype=np.float32)
    jump=np.asarray(temperature_jump_k,dtype=np.float32)
    ustar=np.asarray(friction_velocity,dtype=np.float32)
    length=np.asarray(monin_obukhov_m,dtype=np.float32)
    qh=np.asarray(sensible_heat_flux,dtype=np.float32)
    rho=np.asarray(density,dtype=np.float32); f=np.asarray(coriolis,dtype=np.float32)
    tht,thtp,gamma=profile_temperature_layer(temperature_profile_k,profile_height_m,old,depth_m,minimum_lapse=minimum_lapse)
    daytime=qh>=0.
    wt=np.asarray(qh/(rho*np.float32(996.)),dtype=np.float32)
    wto=np.asarray(np.float32(threshold_flux_per_m)*old/
                   (np.float32(1.2)*np.float32(996.)),dtype=np.float32)
    net=np.asarray(wt-wto,dtype=np.float32)
    twodte=np.float32(np.float32(2.)*np.float32(timestep_s)*np.float32(conste))
    onedte=np.float32(np.float32(timestep_s)*(np.float32(1.)+np.float32(conste)))
    next_jump=_real4_libm('sqrtf',np.maximum(
        np.asarray(gamma*twodte*net,dtype=np.float32),np.float32(0.)))
    radicand=np.asarray(old*old+np.float32(2.)*
        (net*onedte-jump*old)/gamma,dtype=np.float32)
    convective=np.asarray(_real4_libm('sqrtf',np.maximum(radicand,np.float32(0.)))+
                          next_jump/gamma,dtype=np.float32)
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=1.2*996.*wt/threshold_flux_per_m
        relaxed=equilibrium+(old-equilibrium)*np.exp(-timestep_s/relaxation_time_s)
        relaxing=(wt>0.)&(net<=0.)
        convective=np.where(relaxing,relaxed,convective)
        next_jump=np.where(relaxing,jump,next_jump)
    convective=np.where((wt<=0.) if relaxation else (net<=0.),np.float32(0.),
                        np.clip(convective,np.float32(0.),np.float32(maximum_height_m))).astype(np.float32)
    tave=np.asarray(np.float32(.5)*(tht+thtp),dtype=np.float32)
    bvf=np.asarray(_real4_power(
        np.asarray(np.float32(9.81)*gamma/tave,dtype=np.float32),np.float32(.25)),
        dtype=np.float32)
    cmech=np.asarray(np.float32(constb)/_real4_libm('sqrtf',np.maximum(
        np.abs(f),np.float32(1.e-30))),dtype=np.float32)
    mechanical=np.asarray(cmech*ustar/bvf,dtype=np.float32)
    total=np.clip(np.maximum(mechanical,convective),np.float32(minimum_height_m),
                  np.float32(maximum_height_m))
    stable1=np.asarray(np.float32(constn)*_real4_power(ustar,np.float32(1.5)),
                       dtype=np.float32)
    stable2=np.asarray(np.float32(.4)*_real4_libm('sqrtf',np.maximum(
        ustar*length/np.maximum(np.abs(f),np.float32(1.e-30)),np.float32(0.))),
        dtype=np.float32)
    total=np.where(daytime,total,np.clip(np.minimum(stable1,stable2),minimum_height_m,maximum_height_m))
    convective=np.where(daytime,convective,np.float32(0.)).astype(np.float32)
    next_jump=np.where(daytime,next_jump,np.float32(0.)).astype(np.float32)
    if land_mask is not None:
        land=np.asarray(land_mask,dtype=bool); total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan); next_jump=np.where(land,next_jump,np.nan)
    return total,convective,next_jump


def convective_height_maul_carson(previous_convective_height_m,temperature_jump_k,
                                  buoyancy_flux_k_m_s,potential_lapse_k_m,*,
                                  timestep_s=3600.,threshold_flux_per_m=0.,
                                  maximum_height_m=2500.,minimum_lapse=.001,conste=.15,
                                  relaxation=True,relaxation_time_s=800.):
    """CALMET MIXHMC from an already computed surface buoyancy flux.

    WATER supplies ``WT`` directly, unlike the land wrapper which derives it
    from sensible heat flux and density.  Arrays are evaluated simultaneously.
    """
    f32=np.float32
    old=np.asarray(previous_convective_height_m,dtype=f32)
    jump=np.asarray(temperature_jump_k,dtype=f32)
    wt=np.asarray(buoyancy_flux_k_m_s,dtype=f32)
    gamma=np.maximum(np.asarray(potential_lapse_k_m,dtype=f32),f32(minimum_lapse))
    wto=np.asarray(f32(threshold_flux_per_m)*old/(f32(1.2)*f32(996.)),dtype=f32)
    net=np.asarray(wt-wto,dtype=f32)
    twodte=f32(f32(2.)*f32(timestep_s)*f32(conste))
    onedte=f32(f32(timestep_s)*f32(f32(1.)+f32(conste)))
    next_jump=_real4_libm('sqrtf',np.maximum(
        np.asarray(gamma*twodte*net,dtype=f32),f32(0.)))
    radicand=np.asarray(old*old+f32(2.)*(net*onedte-jump*old)/gamma,dtype=f32)
    convective=np.asarray(_real4_libm('sqrtf',np.maximum(radicand,f32(0.)))+
                          next_jump/gamma,dtype=f32)
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=np.asarray(f32(1.2)*f32(996.)*wt/f32(threshold_flux_per_m),dtype=f32)
        decay=f32(np.exp(f32(-f32(timestep_s)/f32(relaxation_time_s))))
        relaxed=np.asarray(equilibrium+(old-equilibrium)*decay,dtype=f32)
        relaxing=(wt>f32(0.))&(net<=f32(0.))
        convective=np.where(relaxing,relaxed,convective)
        next_jump=np.where(relaxing,jump,next_jump)
    active=(wt>f32(0.)) if relaxation else (net>f32(0.))
    convective=np.where(active,np.clip(convective,f32(0.),f32(maximum_height_m)),f32(0.))
    next_jump=np.where(active,next_jump,f32(0.))
    return convective.astype(f32),next_jump.astype(f32)


def convective_height_batchvarova_gryning(
        previous_convective_height_m,buoyancy_flux_k_m_s,potential_lapse_k_m,
        friction_velocity,monin_obukhov_m,surface_temperature_k,*,
        timestep_s=3600.,minimum_height_m=50.,maximum_height_m=2500.,
        threshold_flux_per_m=0.,minimum_lapse=.001,relaxation=True,
        relaxation_time_s=800.):
    """Vectorised CALMET ``MIXHBG`` driven by a surface buoyancy flux."""
    old=np.asarray(previous_convective_height_m,dtype=np.float32)
    wt=np.asarray(buoyancy_flux_k_m_s,dtype=np.float32)
    gamma=np.maximum(np.asarray(potential_lapse_k_m,dtype=np.float32),
                     np.float32(minimum_lapse))
    ustar=np.asarray(friction_velocity,dtype=np.float32)
    length=np.asarray(monin_obukhov_m,dtype=np.float32)
    tk=np.asarray(surface_temperature_k,dtype=np.float32)
    h0=np.maximum(old,np.float32(minimum_height_m))
    wto=np.asarray(np.float32(threshold_flux_per_m)*h0/
                    (np.float32(1.2)*np.float32(1005.)),dtype=np.float32)
    net=np.asarray(wt-wto,dtype=np.float32)
    active=(length<=0.)&(net>0.)
    convective=np.zeros_like(h0,dtype=np.float32)
    if relaxation and threshold_flux_per_m>0.:
        equilibrium=np.asarray(np.float32(1.2)*np.float32(1005.)*wt/
                               np.float32(threshold_flux_per_m),dtype=np.float32)
        decay=np.asarray(np.exp(np.float32(-timestep_s)/
                                np.float32(relaxation_time_s)),dtype=np.float32)
        relaxed=np.asarray(equilibrium+(h0-equilibrium)*decay,dtype=np.float32)
        convective=np.where((length<=0.)&(wt>=0.)&(net<=0.),relaxed,convective)
    elif not relaxation:
        convective=np.where((length<=0.)&(net==0.),old,convective)

    # Constants and false-position iterations are literal MIXHBG/FBG translations.
    a=np.where(active,np.asarray(net/gamma,dtype=np.float32),np.float32(1.)).astype(np.float32)
    b=np.asarray(np.float32(8.)*ustar**np.float32(2.)*tk/
                 (gamma*np.float32(9.81)),dtype=np.float32)
    c=np.asarray(np.float32(2.5)*np.float32(.4)*length,dtype=np.float32)
    d=np.float32(1.)+np.float32(.2)+np.float32(.2)
    e=np.float32(1.)+np.float32(.2)
    d3=np.asarray(d*d*d,dtype=np.float32)
    zeta0=np.asarray(d*h0-np.float32(2.)*c,dtype=np.float32)
    zetap0=np.asarray(e*h0-c,dtype=np.float32)
    r1=np.asarray(np.float32(8.)*c,dtype=np.float32)
    r2=np.asarray(r1*c,dtype=np.float32)
    r3=np.asarray(np.float32(2.)*b*d3/e,dtype=np.float32)
    r4=np.asarray(np.float32(1.)/(np.float32(2.)*a*d3),dtype=np.float32)
    r5=np.asarray(e/d,dtype=np.float32)

    def fbg(zeta,zetap):
        # Scalar Fortran never calls FBG after a non-positive-argument test;
        # inactive vector lanes use benign placeholders to preserve that flow.
        safe_zeta=np.where(zeta>0.,zeta,np.float32(1.)).astype(np.float32)
        safe_zetap=np.where(zetap>0.,zetap,np.float32(1.)).astype(np.float32)
        return np.asarray(safe_zeta**np.float32(2.)+r1*safe_zeta+
                          r2*np.log(safe_zeta)+r3*np.log(safe_zetap),dtype=np.float32)

    time=np.asarray(np.float32(timestep_s)+r4*fbg(zeta0,zetap0),dtype=np.float32)
    weak=length>-500.
    square=np.asarray(h0*h0+np.float32(2.)*d*a*np.float32(timestep_s),dtype=np.float32)
    cube=np.asarray(h0**np.float32(3.)-np.float32(6.)*a*c*np.float32(timestep_s),
                    dtype=np.float32)
    # np.where evaluates both branches; only evaluate the Fortran-selected lanes
    # so a negative inactive CUBE does not emit a fractional-power warning.
    h=np.empty_like(h0,dtype=np.float32)
    h[weak]=np.sqrt(square[weak]).astype(np.float32)
    h[~weak]=(cube[~weak]**np.float32(.333)).astype(np.float32)
    zeta1=np.asarray(d*h-np.float32(2.)*c,dtype=np.float32)
    zetap1=np.asarray(e*h-c,dtype=np.float32)
    f1=np.asarray(r4*fbg(zeta1,zetap1)-time,dtype=np.float32)
    converged=np.abs(f1)<=1.
    f0=np.full_like(f1,np.float32(-timestep_s))
    denominator=np.asarray(f1-f0,dtype=np.float32)
    zeta2=np.asarray((f1*zeta0-f0*zeta1)/denominator,dtype=np.float32)
    zetap2=np.asarray(r5*(zeta2+np.float32(2.)*c)-c,dtype=np.float32)
    invalid=(zeta2<=0.)|(zetap2<=0.)
    zeta2=np.where(invalid,np.float32(.5)*(zeta0+zeta1),zeta2).astype(np.float32)
    zetap2=np.asarray(r5*(zeta2+np.float32(2.)*c)-c,dtype=np.float32)
    f2=np.asarray(r4*fbg(zeta2,zetap2)-time,dtype=np.float32)
    h=np.where(~converged,(zeta2+np.float32(2.)*c)/d,h).astype(np.float32)
    converged|=np.abs(f2)<=1.
    failed=np.zeros_like(converged)
    for _ in range(10):
        work=active&~converged&~failed
        denominator=np.asarray(f1-f2,dtype=np.float32)
        stalled=np.abs(denominator)<=np.float32(1.e-10)
        converged|=work&stalled
        safe=np.where(stalled,np.float32(1.),denominator).astype(np.float32)
        zeta3=np.asarray((f1*zeta2-f2*zeta1)/safe,dtype=np.float32)
        zetap3=np.asarray(r5*(zeta3+np.float32(2.)*c)-c,dtype=np.float32)
        bad=(zeta3<=0.)|(zetap3<=0.)
        failed|=work&bad
        f3=np.asarray(r4*fbg(zeta3,zetap3)-time,dtype=np.float32)
        h=np.where(work&~bad,(zeta3+np.float32(2.)*c)/d,h).astype(np.float32)
        zeta1=np.where(work&~bad,zeta2,zeta1)
        zeta2=np.where(work&~bad,zeta3,zeta2)
        f1=np.where(work&~bad,f2,f1)
        f2=np.where(work&~bad,f3,f2)
        converged|=work&~bad&(np.abs(f3)<=1.)
    fallback=active&~converged
    h=np.where(fallback,((zeta1+zeta2)*np.float32(.5)+np.float32(2.)*c)/d,h).astype(np.float32)
    # Preserve MIXHBG's historical overwrite: MAX(H,ZIMN) replaces the
    # preceding MIN(H,ZIMX), so the convective result itself is not capped.
    return np.where(active,np.maximum(h,np.float32(minimum_height_m)),convective).astype(np.float32)


def mixing_height_batchvarova_gryning(temperature_profile_k, profile_height_m,
                                      previous_convective_height_m, temperature_jump_k,
                                      friction_velocity, monin_obukhov_m,
                                      sensible_heat_flux, density, coriolis,
                                      *, surface_temperature_k, timestep_s=3600.,
                                      depth_m=200., minimum_lapse=.001,
                                      minimum_height_m=50., maximum_height_m=2500.,
                                      threshold_flux_per_m=0., constn=2400.,
                                      constb=1.41, relaxation=True,
                                      relaxation_time_s=800., land_mask=None):
    """Vectorised CALMET ``MIXHBG`` plus the common ``MIXHT2`` mechanics.

    The nonlinear Batchvarova--Gryning equation follows the legacy false-image
    secant solver.  Its ten iterations operate on whole arrays; no horizontal
    Python loop is used.  The temperature jump is unchanged because MIXHBG,
    unlike MIXHMC, does not update CALMET's ``DPTT`` state.
    """
    old=np.asarray(previous_convective_height_m,dtype=np.float32)
    jump=np.asarray(temperature_jump_k,dtype=np.float32)
    ustar=np.asarray(friction_velocity,dtype=np.float32)
    length=np.asarray(monin_obukhov_m,dtype=np.float32)
    qh=np.asarray(sensible_heat_flux,dtype=np.float32)
    rho_field=np.asarray(density,dtype=np.float32)
    f=np.asarray(coriolis,dtype=np.float32)
    tk=np.asarray(surface_temperature_k,dtype=np.float32)
    h0=np.maximum(old,np.float32(minimum_height_m))
    tht,thtp,gamma=profile_temperature_layer(temperature_profile_k,profile_height_m,h0,depth_m,minimum_lapse=minimum_lapse)
    wt=np.asarray(qh/(rho_field*np.float32(996.)),dtype=np.float32)
    convective=convective_height_batchvarova_gryning(
        h0,wt,gamma,ustar,length,tk,timestep_s=timestep_s,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
        threshold_flux_per_m=threshold_flux_per_m,minimum_lapse=minimum_lapse,
        relaxation=relaxation,relaxation_time_s=relaxation_time_s)
    daytime=qh>=np.float32(0.)
    tave=np.asarray(np.float32(.5)*(tht+thtp),dtype=np.float32)
    bvf=np.asarray(_real4_power(
        np.asarray(np.float32(9.81)*gamma/tave,dtype=np.float32),np.float32(.25)),
        dtype=np.float32)
    cmech=np.asarray(np.float32(constb)/_real4_libm('sqrtf',np.maximum(
        np.abs(f),np.float32(1.e-30))),dtype=np.float32)
    mechanical=np.asarray(cmech*ustar/bvf,dtype=np.float32)
    total=np.clip(np.maximum(mechanical,convective),np.float32(minimum_height_m),
                  np.float32(maximum_height_m))
    stable1=np.asarray(np.float32(constn)*_real4_power(ustar,np.float32(1.5)),
                       dtype=np.float32)
    stable2=np.asarray(np.float32(.4)*_real4_libm('sqrtf',np.maximum(
        ustar*length/np.maximum(np.abs(f),np.float32(1.e-30)),np.float32(0.))),
        dtype=np.float32)
    total=np.where(daytime,total,np.clip(np.minimum(stable1,stable2),
        np.float32(minimum_height_m),np.float32(maximum_height_m))).astype(np.float32)
    convective=np.where(daytime,convective,np.float32(0.)).astype(np.float32)
    if land_mask is not None:
        land=np.asarray(land_mask,dtype=bool)
        total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan)
    return total,convective,jump.copy()

def hosker_roughness(speed_m_s):
    """Hosker wind-dependent water roughness with REAL*4 power semantics."""
    f32=np.float32
    speed=np.asarray(speed_m_s,dtype=f32)
    powered=np.asarray(_real4_power(speed,f32(2.5)),dtype=f32)
    return np.maximum(f32(2.e-6),np.asarray(f32(2.e-6)*powered,dtype=f32)).astype(f32)


def neutral_water_parameters(u10, v10, coriolis, *, constw=.16,
                             minimum_height_m=50., maximum_height_m=2500.):
    """WATER's ``NOWSTA=0`` neutral overwater path used by NOOBS runs.

    It applies the Hosker wind-dependent roughness, neutral DeltaT friction
    velocity, and the OCD mechanical mixing height.  With no SEA.DAT station
    the legacy routine has zero buoyancy flux and therefore zero convective
    mixing height even when ``IMIXH=1``.
    """
    u=np.asarray(u10,dtype=np.float32); v=np.asarray(v10,dtype=np.float32)
    f=np.asarray(coriolis,dtype=np.float32)
    speed=np.sqrt(np.asarray(u**np.float32(2.)+v**np.float32(2.),dtype=np.float32))
    roughness=hosker_roughness(speed)
    friction=np.maximum(np.asarray(np.float32(.4)*speed/
        _real4_libm('logf',np.asarray(np.float32(10.)/roughness,dtype=np.float32)),
        dtype=np.float32),
        np.float32(1.e-9)).astype(np.float32)
    mixing=water_mechanical_height(friction,f,constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    return (roughness,friction,np.full_like(speed,np.float32(9.9e9)),
            np.zeros_like(speed),mixing,np.zeros_like(speed))


def neutral_water_station_profiles(u,v,anemometer_height_m,current_mixing_height_m,
        coriolis,z_centers_m,z_faces_m,station_latitude,*,constw=.16,
        minimum_height_m=50.,maximum_height_m=2500.,ipsifcn=0):
    """WATER2 neutral/missing-sea-data station path followed by SIMILT."""
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    values=tuple(np.asarray(value,dtype=f32) for value in (
        anemometer_height_m,current_mixing_height_m,coriolis,station_latitude))
    if u.ndim!=1 or v.shape!=u.shape or any(value.shape!=u.shape for value in values):
        raise ValueError('WATER2 station inputs must be matching one-dimensional arrays')
    anemometer,current_mixing,coriolis_value,latitude=values
    centers=np.asarray(z_centers_m,dtype=f32)
    output_u=np.full((centers.size,u.size),f32(9999.),dtype=f32)
    output_v=np.full_like(output_u,f32(9999.)); output_u[0]=u; output_v[0]=v
    state=[np.full(u.shape,np.nan,dtype=f32) for _ in range(4)]
    speed=_real4_libm('sqrtf',np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32))
    active=(u<=f32(990.))&(v<=f32(990.))&(speed>=f32(.0001))
    if not active.any(): return (output_u,output_v,*state,warm_layer_state)
    selected_speed=speed[active]; selected_height=anemometer[active]
    roughness=hosker_roughness(selected_speed)
    rescale=np.trunc(selected_height).astype(np.int32)!=10
    speed10=selected_speed.copy()
    speed10[rescale]=np.asarray(selected_speed[rescale]*
        _real4_libm('logf',np.asarray(f32(10.)/roughness[rescale],dtype=f32))/
        _real4_libm('logf',np.asarray(selected_height[rescale]/roughness[rescale],dtype=f32)),
        dtype=f32)
    # WATER2 does not repeat Z0MIN after adjusting a non-10 m observation.
    roughness[rescale]=np.asarray(f32(2.e-6)*
        _real4_power(speed10[rescale],f32(2.5)),dtype=f32)
    friction=np.maximum(np.asarray(f32(.4)*speed10/
        _real4_libm('logf',np.asarray(f32(10.)/roughness,dtype=f32)),dtype=f32),
        f32(1.e-9)).astype(f32)
    length=np.full_like(friction,f32(9.9e9))
    mixing=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    mixing=np.maximum(np.minimum(mixing,current_mixing[active]),
                      f32(minimum_height_m)).astype(f32)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],selected_height,centers,z_faces_m,roughness,length,mixing,
        latitude[active],ipsifcn=ipsifcn,minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    for destination,source in zip(state,(roughness,friction,length,mixing),strict=True):
        destination[active]=source
    return (output_u,output_v,*state)


def water_mechanical_height(friction_velocity,coriolis,*,constw=.16,
                            minimum_height_m=50.,maximum_height_m=2500.):
    """WATER's REAL*4 ``ZIMECH=CONSTW*USTAR/FCORI`` calculation."""
    value=np.asarray(np.float32(constw)*np.asarray(friction_velocity,dtype=np.float32)/
                     np.asarray(coriolis,dtype=np.float32),dtype=np.float32)
    return np.clip(value,np.float32(minimum_height_m),
                   np.float32(maximum_height_m)).astype(np.float32)


def saturation_vapor_pressure_lowe(temperature_c):
    """CALMET ``ESAT`` Lowe polynomial in the source's REAL*4 precision."""
    value=np.asarray(temperature_c,dtype=np.float32)
    coefficients=tuple(map(np.float32,
        (6.107799961,.4436518521,.01428945805,.0002650648471,
         .000003031240396,.00000002034080948,.00000000006136820929)))
    result=np.full_like(value,coefficients[-1],dtype=np.float32)
    for coefficient in reversed(coefficients[:-1]):
        result=np.asarray(coefficient+value*result,dtype=np.float32)
    return result.astype(np.float32)


def overwater_delta_temperature_state(air_temperature_k,sea_temperature_k,
                                      relative_humidity_percent,
                                      measurement_height_m):
    """Prepare WATER's virtual-potential-temperature inputs to ``DELTAT``.

    Every assignment follows the default-REAL statements in ``WATER``.  The
    returned tuple is ``(theta, delta_theta_z, delta_theta_10m, theta_10m)``.
    """
    air,sea,rh,height=np.broadcast_arrays(
        np.asarray(air_temperature_k,dtype=np.float32),
        np.asarray(sea_temperature_k,dtype=np.float32),
        np.asarray(relative_humidity_percent,dtype=np.float32),
        np.asarray(measurement_height_m,dtype=np.float32))
    r=np.float32
    air_c=np.asarray(air-r(273.15),dtype=np.float32)
    saturation=saturation_vapor_pressure_lowe(air_c)
    vapor=np.asarray(r(.01)*rh*saturation,dtype=np.float32)
    mixing=np.asarray(r(.622)*vapor/(r(1000.)-vapor),dtype=np.float32)
    virtual=np.asarray(air*(r(1.)+r(.61)*mixing),dtype=np.float32)
    theta=np.asarray(virtual+r(.01)*height,dtype=np.float32)

    sea_c=np.asarray(sea-r(273.15),dtype=np.float32)
    sea_saturation=saturation_vapor_pressure_lowe(sea_c)
    saturated_mixing=np.asarray(
        r(.622)*sea_saturation/(r(1000.)-sea_saturation),dtype=np.float32)
    surface_theta=np.asarray(sea*(r(1.)+r(.61)*saturated_mixing),dtype=np.float32)
    difference=np.asarray(theta-surface_theta,dtype=np.float32)
    difference_10m=np.asarray(
        difference*r(11.512925)/(np.log(height)+r(9.2103404)),dtype=np.float32)
    theta_10m=np.asarray(surface_theta+difference_10m,dtype=np.float32)
    return theta,difference,difference_10m,theta_10m


def delta_temperature_flux(wind_speed_10m,measurement_height_m,
                           measurement_minus_surface_virtual_potential_k,
                           ten_m_minus_surface_virtual_potential_k,
                           virtual_potential_temperature_10m_k,
                           measurement_virtual_potential_temperature_k,
                           roughness_m):
    """Vectorized CALMET ``DELTAT`` with its original REAL*4 arithmetic."""
    wind,height,difference,difference_10m,theta_10m,theta,z0=np.broadcast_arrays(
        np.asarray(wind_speed_10m,dtype=np.float32),
        np.asarray(measurement_height_m,dtype=np.float32),
        np.asarray(measurement_minus_surface_virtual_potential_k,dtype=np.float32),
        np.asarray(ten_m_minus_surface_virtual_potential_k,dtype=np.float32),
        np.asarray(virtual_potential_temperature_10m_k,dtype=np.float32),
        np.asarray(measurement_virtual_potential_temperature_k,dtype=np.float32),
        np.asarray(roughness_m,dtype=np.float32))
    r=np.float32
    von_karman=r(.4); vkg=r(von_karman*r(9.81))
    log_10=np.asarray(np.log(r(10.)/z0),dtype=np.float32)
    log_measurement=np.asarray(np.log(height/z0),dtype=np.float32)
    neutral_ustar=np.maximum(
        np.asarray(von_karman*wind/log_10,dtype=np.float32),r(1.e-9))
    length=np.full(wind.shape,r(9.9e9),dtype=np.float32)
    ustar=neutral_ustar.copy()
    nonneutral=np.abs(difference_10m)>r(1.e-6)
    drag=np.asarray(r(.75e-3)+r(.067e-3)*wind,dtype=np.float32)
    initial=np.asarray(theta_10m*np.power(drag,r(1.5))*wind**2/
                       (r(5.096e-3)*np.where(nonneutral,difference_10m,r(1.))),
                       dtype=np.float32)
    active=nonneutral&(np.abs(initial)<=r(9999.))
    length=np.where(active,initial,length)
    length=np.where(active&(np.abs(length)<r(5.)),np.copysign(r(5.),length),length)
    old=length.copy()
    for _ in range(3):
        safe_length=np.where(active,length,r(1.))
        wind_ratio=np.asarray(r(10.)/safe_length,dtype=np.float32)
        temperature_ratio=np.asarray(height/safe_length,dtype=np.float32)
        stable=safe_length>r(0.)
        # NumPy evaluates both arms of ``where``.  Feed the unstable-only
        # intrinsics benign values on stable/inactive lanes, mirroring the
        # Fortran IF without spurious invalid-power warnings.
        unstable_x_base=np.where(~stable,r(1.)-r(15.)*wind_ratio,r(1.))
        x=np.asarray(unstable_x_base**r(.25),dtype=np.float32)
        psi_u_unstable=np.asarray(
            r(2.)*np.log(r(.5)*(r(1.)+x))+
            np.log(r(.5)*(r(1.)+x*x))-r(2.)*np.arctan(x)+r(1.5707963),
            dtype=np.float32)
        unstable_y_base=np.where(~stable,r(1.)-r(9.)*temperature_ratio,r(1.))
        y=np.asarray(np.sqrt(unstable_y_base),dtype=np.float32)
        psi_t_unstable=np.asarray(r(1.48)*np.log(r(.5)*(r(1.)+y)),dtype=np.float32)
        psi_u=np.where(stable,-r(4.7)*wind_ratio,psi_u_unstable)
        psi_t=np.where(stable,-r(4.7)*temperature_ratio,psi_t_unstable)
        candidate_ustar=np.maximum(
            np.asarray(von_karman*wind/(log_10-psi_u),dtype=np.float32),r(1.e-9))
        theta_star=np.asarray(von_karman*difference/
                              (r(.74)*log_measurement-psi_t),dtype=np.float32)
        valid_theta=np.abs(theta_star)>r(1.e-6)
        candidate_length=np.asarray(
            theta*candidate_ustar**2/
            (vkg*np.where(valid_theta,theta_star,r(1.))),dtype=np.float32)
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
                                  dtype=np.float32)
    return (length.astype(np.float32),ustar.astype(np.float32),
            buoyancy_flux_w_m2)


def delta_temperature_water_station_profiles(u,v,anemometer_height_m,
        air_temperature_k,air_minus_sea_temperature_k,relative_humidity_percent,
        observed_mixing_height_m,current_mixing_height_m,coriolis,
        potential_lapse_k_m,previous_temperature_jump_k,z_centers_m,z_faces_m,
        station_latitude,*,mixing_method=1,constw=.16,minimum_height_m=50.,
        maximum_height_m=2500.,threshold_flux_per_m=.05,ipsifcn=0,
        temperature_measurement_height_m=None,sea_temperature_k=None,**mixing_options):
    """WATER2 station DeltaT branch, with neutral fallback, through SIMILT."""
    f32=np.float32
    base=neutral_water_station_profiles(u,v,anemometer_height_m,
        current_mixing_height_m,coriolis,z_centers_m,z_faces_m,station_latitude,
        constw=constw,minimum_height_m=minimum_height_m,
        maximum_height_m=maximum_height_m,ipsifcn=ipsifcn)
    output_u,output_v=(base[0].copy(),base[1].copy())
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    station_inputs=tuple(np.asarray(value,dtype=f32) for value in (
        anemometer_height_m,air_temperature_k,air_minus_sea_temperature_k,
        relative_humidity_percent,observed_mixing_height_m,current_mixing_height_m,
        coriolis,potential_lapse_k_m,previous_temperature_jump_k,station_latitude))
    if any(value.shape!=u.shape for value in station_inputs):
        raise ValueError('WATER2 DeltaT station inputs must have identical shapes')
    (anemometer,air,delta,rh,observed,current,coriolis_value,lapse,
     previous_jump,latitude)=station_inputs
    speed=_real4_libm('sqrtf',np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32))
    active=(u<=f32(990.))&(v<=f32(990.))&(speed>=f32(.0001))&(delta<f32(990.))
    state=[np.array(value,copy=True) for value in base[2:]]
    # DeltaT-specific state: convective height, jump, and buoyancy heat flux.
    state.extend(np.full(u.shape,np.nan,dtype=f32) for _ in range(3))
    if not active.any(): return (output_u,output_v,*state)
    height=anemometer[active]
    thermal_height=(height if temperature_measurement_height_m is None else
        np.broadcast_to(np.asarray(temperature_measurement_height_m,dtype=f32),u.shape)[active])
    selected_air=np.where(air[active]<f32(990.),air[active],f32(288.7)).astype(f32)
    selected_rh=np.where(rh[active]<f32(990.),rh[active],f32(100.)).astype(f32)
    sea=(np.asarray(selected_air-delta[active],dtype=f32)
         if sea_temperature_k is None else
         np.broadcast_to(np.asarray(sea_temperature_k,dtype=f32),u.shape)[active])
    theta,difference,difference_10m,theta_10m=overwater_delta_temperature_state(
        selected_air,sea,selected_rh,thermal_height)
    initial_z0=hosker_roughness(speed[active])
    rescale=np.trunc(height).astype(np.int32)!=10
    speed10=speed[active].copy()
    speed10[rescale]=np.asarray(speed10[rescale]*
        _real4_libm('logf',np.asarray(f32(10.)/initial_z0[rescale],dtype=f32))/
        _real4_libm('logf',np.asarray(height[rescale]/initial_z0[rescale],dtype=f32)),
        dtype=f32)
    roughness=initial_z0.copy()
    roughness[rescale]=np.asarray(f32(2.e-6)*
        _real4_power(speed10[rescale],f32(2.5)),dtype=f32)
    length,friction,flux=delta_temperature_flux(
        speed10,thermal_height,difference,difference_10m,theta_10m,theta,roughness)
    mechanical=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    has_observed=observed[active]<f32(990.)
    initial=np.where(has_observed,observed[active],mechanical).astype(f32)
    initial=np.maximum(np.minimum(initial,current[active]),f32(minimum_height_m)).astype(f32)
    convective=np.zeros_like(initial); next_jump=previous_jump[active].copy()
    buoyancy=np.asarray(flux/f32(1195.2),dtype=f32)
    options=dict(mixing_options)
    if int(mixing_method)==1:
        convective,next_jump=convective_height_maul_carson(
            initial,previous_jump[active],buoyancy,lapse[active],
            threshold_flux_per_m=threshold_flux_per_m,
            maximum_height_m=maximum_height_m,**options)
    elif int(mixing_method)==2:
        convective=convective_height_batchvarova_gryning(
            initial,buoyancy,lapse[active],friction,length,selected_air,
            minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
            threshold_flux_per_m=threshold_flux_per_m,**options)
    elif int(mixing_method) not in {-2,-1}:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    calculated=np.clip(np.maximum(mechanical,convective),f32(minimum_height_m),
                       f32(maximum_height_m)).astype(f32)
    mixing=np.where(has_observed,initial,calculated).astype(f32)
    convective=np.where(has_observed,np.nan,convective).astype(f32)
    next_jump=np.where(has_observed,np.nan,next_jump).astype(f32)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],height,z_centers_m,z_faces_m,roughness,length,mixing,
        latitude[active],ipsifcn=ipsifcn,minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    sources=(roughness,friction,length,mixing,convective,next_jump,flux)
    for destination,source in zip(state,sources,strict=True): destination[active]=source
    return (output_u,output_v,*state)


def coare_water_station_profiles(u,v,anemometer_height_m,air_sensor_height_m,
        air_temperature_k,air_minus_sea_temperature_k,relative_humidity_percent,
        observed_mixing_height_m,current_mixing_height_m,coriolis,
        potential_lapse_k_m,previous_temperature_jump_k,z_centers_m,z_faces_m,
        station_latitude,shortwave_down_w_m2,longwave_down_w_m2,rainfall_mm_h,
        *,mixing_method=1,coare_method=10,wave_period_s=-999.,wave_height_m=-999.,
        coastal_shelf_length_km=0.,distance_to_coast_km=0.,cool_skin=False,
        constw=.16,minimum_height_m=50.,maximum_height_m=2500.,
        threshold_flux_per_m=.05,ipsifcn=0,sea_temperature_k=None,
        warm_layer_state=None,solar_time_s=None,water_sensor_depth_m=.6,
        **mixing_options):
    """Stateless COARE branch of WATER2 followed by its mixing and SIMILT steps."""
    from .coare import (WarmLayerState,advance_warm_layer,surface_flux,
                        update_warm_layer_fluxes)

    if abs(int(coare_method)) not in {10,11,12}:
        raise ValueError('COARE station method must be +/-10, +/-11, or +/-12')
    f32=np.float32
    base=neutral_water_station_profiles(u,v,anemometer_height_m,
        current_mixing_height_m,coriolis,z_centers_m,z_faces_m,station_latitude,
        constw=constw,minimum_height_m=minimum_height_m,
        maximum_height_m=maximum_height_m,ipsifcn=ipsifcn)
    output_u,output_v=base[0].copy(),base[1].copy()
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    named=(anemometer_height_m,air_sensor_height_m,air_temperature_k,
        air_minus_sea_temperature_k,relative_humidity_percent,
        observed_mixing_height_m,current_mixing_height_m,coriolis,
        potential_lapse_k_m,previous_temperature_jump_k,station_latitude,
        shortwave_down_w_m2,longwave_down_w_m2,rainfall_mm_h)
    values=tuple(np.asarray(value,dtype=f32) for value in named)
    if any(value.shape!=u.shape for value in values):
        raise ValueError('COARE WATER2 station inputs must have identical shapes')
    (anemometer,air_height,air,delta,rh,observed,current,coriolis_value,lapse,
     previous_jump,latitude,shortwave,longwave,rainfall)=values
    speed=_real4_libm('sqrtf',np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32))
    active=(u<=f32(990.))&(v<=f32(990.))&(speed>=f32(.0001))&(delta<f32(990.))
    state=[np.array(value,copy=True) for value in base[2:]]
    state.extend(np.full(u.shape,np.nan,dtype=f32) for _ in range(3))
    if not active.any(): return (output_u,output_v,*state)
    selected_air=np.where(air[active]<f32(990.),air[active],f32(288.7)).astype(f32)
    selected_rh=np.where(rh[active]<f32(990.),rh[active],f32(100.)).astype(f32)
    sea=(np.asarray(selected_air-delta[active],dtype=f32)
         if sea_temperature_k is None else
         np.broadcast_to(np.asarray(sea_temperature_k,dtype=f32),u.shape)[active])
    sea=np.maximum(sea,f32(270.15))
    next_warm_state=warm_layer_state
    if warm_layer_state is not None:
        if solar_time_s is None:
            raise ValueError('COARE warm-layer station profiles require solar_time_s')
        full_warm_state=np.shape(warm_layer_state.initialized)==u.shape
        selected_warm_state=(WarmLayerState(*(
            np.asarray(getattr(warm_layer_state,name))[active]
            for name in WarmLayerState.__dataclass_fields__))
            if full_warm_state else warm_layer_state)
        solar=np.broadcast_to(np.asarray(solar_time_s,dtype=float),u.shape)[active]
        depth=np.broadcast_to(np.asarray(water_sensor_depth_m,dtype=float),u.shape)[active]
        sea,selected_warm_state,_=advance_warm_layer(
            selected_warm_state,solar,sea,depth,
            shortwave[active],longwave[active])
    saturation=saturation_vapor_pressure_lowe(selected_air-f32(273.15))
    vapor=np.asarray(f32(.01)*selected_rh*saturation,dtype=f32)
    ratio=np.asarray(f32(.622)*vapor/(f32(1000.)-vapor),dtype=f32)
    humidity=np.asarray(ratio/(f32(1.)+ratio)*f32(1000.),dtype=f32)
    neutral_friction=base[3][active]
    mechanical=water_mechanical_height(neutral_friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    has_observed=observed[active]<f32(990.)
    initial=np.where(has_observed,observed[active],mechanical).astype(f32)
    initial=np.maximum(np.minimum(initial,current[active]),f32(minimum_height_m)).astype(f32)
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
            merged=[]
            for name in WarmLayerState.__dataclass_fields__:
                values=np.array(getattr(warm_layer_state,name),copy=True)
                values[active]=getattr(selected_warm_state,name)
                merged.append(values)
            next_warm_state=WarmLayerState(*merged)
        else:
            next_warm_state=selected_warm_state
    roughness=np.asarray(flux.roughness_m,dtype=f32)
    friction=np.asarray(flux.friction_velocity_m_s,dtype=f32)
    length=np.asarray(flux.monin_obukhov_m,dtype=f32)
    length=np.where(np.abs(length)<f32(5.),np.where(length<f32(0.),f32(-5.),f32(5.)),
                    length).astype(f32)
    buoyancy=np.asarray(flux.buoyancy_flux_k_m_s,dtype=f32)
    mechanical=water_mechanical_height(friction,coriolis_value[active],constw=constw,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m)
    convective=np.zeros_like(initial); next_jump=previous_jump[active].copy()
    options=dict(mixing_options)
    if int(mixing_method)==1:
        convective,next_jump=convective_height_maul_carson(
            initial,previous_jump[active],buoyancy,lapse[active],
            threshold_flux_per_m=threshold_flux_per_m,
            maximum_height_m=maximum_height_m,**options)
    elif int(mixing_method)==2:
        convective=convective_height_batchvarova_gryning(
            initial,buoyancy,lapse[active],friction,length,selected_air,
            minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
            threshold_flux_per_m=threshold_flux_per_m,**options)
    elif int(mixing_method) not in {-2,-1}:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    calculated=np.clip(np.maximum(mechanical,convective),f32(minimum_height_m),
                       f32(maximum_height_m)).astype(f32)
    mixing=np.where(has_observed,initial,calculated).astype(f32)
    convective=np.where(has_observed,np.nan,convective).astype(f32)
    next_jump=np.where(has_observed,np.nan,next_jump).astype(f32)
    profile_u,profile_v=similarity_wind_profiles(
        u[active],v[active],anemometer[active],z_centers_m,z_faces_m,roughness,length,
        mixing,latitude[active],ipsifcn=ipsifcn,
        minimum_mixing_height_m=minimum_height_m)
    output_u[:,active]=profile_u; output_v[:,active]=profile_v
    buoyancy_flux=np.asarray(buoyancy*f32(1195.2),dtype=f32)
    for destination,source in zip(state,(roughness,friction,length,mixing,convective,
            next_jump,buoyancy_flux),strict=True): destination[active]=source
    return (output_u,output_v,*state,next_warm_state)


def upwind_mixing_average(u, v, mixing_height, convective_height, grid_spacing_m,
                          *, minimum_radius_cells=10, half_angle_degrees=30., timestep_s=3600.):
    """AVEMIX upwind-cone spatial averaging of ``ZI`` and ``ZICONV``.

    The target-cell loop is retained because every target has a differently
    oriented and sized search window; all candidate-cell calculations inside
    that window are NumPy-vectorised.  This follows the legacy integer vertex,
    angular inclusion, boundary clipping, and displaced inverse-square weight.
    """
    u=np.asarray(u,dtype=np.float32); v=np.asarray(v,dtype=np.float32)
    zi=np.asarray(mixing_height,dtype=np.float32); zic=np.asarray(convective_height,dtype=np.float32)
    if u.shape!=v.shape or u.shape!=zi.shape or u.shape!=zic.shape or u.ndim!=2: raise ValueError('AVEMIX fields must share a two-dimensional grid')
    ny,nx=u.shape; result=np.empty_like(zi); result_convective=np.empty_like(zic)
    degree=np.float32(.01745329)
    # AVEMIX/AVETMP declare DTIME as DATA /3600./ rather than receiving the
    # model output step.  Keep the keyword for API compatibility, but preserve
    # that legacy one-hour trajectory even for sub-hour CALMET runs.
    del timestep_s
    scale=np.float32(np.float32(3600.)/np.float32(grid_spacing_m))
    tangent=np.asarray(np.tan(np.float32(degree*np.float32(half_angle_degrees))),
                       dtype=np.float32)
    @lru_cache(maxsize=4096)
    def window_geometry(left,right,down,up):
        rx_integer=np.arange(left,-right-1,-1,dtype=np.int64)[None,:]
        ry_integer=np.arange(down,-up-1,-1,dtype=np.int64)[:,None]
        rx=rx_integer.astype(np.float32); ry=ry_integer.astype(np.float32)
        local=(rx_integer==0)&(ry_integer==0)
        square=(np.abs(rx_integer)<=minimum_radius_cells)&(
            np.abs(ry_integer)<=minimum_radius_cells)
        beta=np.fmod(np.float32(270.)-np.arctan2(
            ry.astype(np.float32),rx.astype(np.float32))/degree,np.float32(360.))
        return rx_integer,ry_integer,rx,ry,local,square,beta
    for i in range(nx):
        for j in range(ny):
            etax=np.float32(u[j,i]*scale); etay=np.float32(v[j,i]*scale)
            eta2=np.float32(etax*etax+etay*etay)
            eta=np.sqrt(eta2); inverse_eta=np.float32(1.)/eta if eta2>0. else np.float32(0.)
            ix=int(etax); iy=int(etay); cross_x=int(etay*tangent); cross_y=int(etax*tangent)
            upwind_x=i-ix; upwind_y=j-iy
            ax=upwind_x+cross_x; bx=upwind_x-cross_x; ay=upwind_y-cross_y; by=upwind_y+cross_y
            def boundary_angle(px,py):
                if i!=px:
                    return np.fmod(np.float32(270.)-np.arctan2(
                        np.float32(j-py),np.float32(i-px))/degree,np.float32(360.))
                return np.float32(180.) if j>=py else np.float32(0.)
            alpha_a=boundary_angle(ax,ay); alpha_b=boundary_angle(bx,by)
            xlo=max(0,min(ax,bx,i-minimum_radius_cells)); xhi=min(nx-1,max(ax,bx,i+minimum_radius_cells))
            ylo=max(0,min(ay,by,j-minimum_radius_cells)); yhi=min(ny-1,max(ay,by,j+minimum_radius_cells))
            # The window is rectangular, so relative one-dimensional axes
            # broadcast to every expression below.  Constructing ``mgrid``
            # here formerly allocated two full candidate arrays for every
            # one of ~300k targets and dominated the model runtime.
            # In AVEMIX/AVETMP the INTEGER offsets are converted to default
            # REAL before each product with ETAX/ETAY.  Cache geometry shared
            # by targets with the same clipped window extents; only wind-
            # dependent cone and weight expressions remain per target.
            geometry=window_geometry(i-xlo,xhi-i,j-ylo,yhi-j)
            rx_integer,ry_integer,rx,ry,local,square,beta=geometry
            cone=(beta>=alpha_a)&(beta<=alpha_b)&(
                (rx_integer*rx_integer+ry_integer*ry_integer)<=eta2)
            include=local|square|cone
            if eta>0.:
                along=np.asarray(np.asarray(
                    np.asarray(rx*etax,dtype=np.float32)+
                    np.asarray(ry*etay,dtype=np.float32),dtype=np.float32)*inverse_eta,
                    dtype=np.float32)
                along=np.where(along<0.,eta-along,along).astype(np.float32)
                across=np.asarray(np.abs(np.asarray(
                    np.asarray(rx*etay,dtype=np.float32)-
                    np.asarray(ry*etax,dtype=np.float32),dtype=np.float32))*inverse_eta,
                    dtype=np.float32)
            else:
                along=np.hypot(rx,ry)
                across=np.zeros_like(along,dtype=np.float32)
            # Both powers are integer default-REAL expressions in AVEMIX and
            # AVETMP.  NumPy 2 promotes ``float32 ** np.int32(2)`` to
            # float64, shifting the weights enough to perturb the carried
            # convective-height state.  Spell out gfortran's two REAL*4
            # multiplications so this arithmetic is stable across NumPy
            # releases.
            along_squared=np.asarray(along*along,dtype=np.float32)
            displaced=np.asarray(np.float32(1.)+np.abs(across),dtype=np.float32)
            displaced_squared=np.asarray(displaced*displaced,dtype=np.float32)
            denominator=np.asarray(along_squared+displaced_squared,dtype=np.float32)
            weight=np.asarray(np.float32(1.)/denominator,dtype=np.float32)
            weight=np.where(local,np.float32(1.),weight)
            weight=np.where(include,weight,np.float32(0.)).astype(np.float32)
            # Fortran visits KI (x) outside KJ (y); transpose before C-order
            # CUMSUM to retain that ordered REAL*4 accumulation without a
            # Python loop over candidate cells.
            ordered_weight=weight.T.ravel()
            denominator=np.cumsum(ordered_weight,dtype=np.float32)[-1]
            ordered_zi=np.asarray((weight*zi[ylo:yhi+1,xlo:xhi+1]).T.ravel(),
                                  dtype=np.float32)
            ordered_zic=np.asarray((weight*zic[ylo:yhi+1,xlo:xhi+1]).T.ravel(),
                                   dtype=np.float32)
            result[j,i]=np.float32(np.cumsum(ordered_zi,dtype=np.float32)[-1]/denominator)
            result_convective[j,i]=np.float32(
                np.cumsum(ordered_zic,dtype=np.float32)[-1]/denominator)
    return result,result_convective


def mesopac_layer_average(u, v, mixing_height_m, z_faces_m, *,
                          upper_top_m=3000., minimum_upper_depth_m=300.):
    """Vectorized PACAVE lower- and upper-layer wind averages.

    The lower layer extends from the ground to ``max(ZI, 10 m)``.  The
    upper layer extends from ZI to the greater of 3000 m or ZI + 300 m,
    matching the fixed arguments used by CALMET's OUTPC path.
    """
    u=np.asarray(u,dtype=np.float32)
    v=np.asarray(v,dtype=np.float32)
    htmix=np.asarray(mixing_height_m,dtype=np.float32)
    zi=np.maximum(htmix,np.float32(10.))
    faces=np.asarray(z_faces_m,dtype=np.float32)
    if u.shape!=v.shape or u.ndim!=3 or zi.shape!=u.shape[1:]:
        raise ValueError('PACAVE winds must be (level,y,x) and match mixing height')
    if faces.ndim!=1 or faces.size!=u.shape[0]+1 or np.any(np.diff(faces)<=0.):
        raise ValueError('PACAVE requires one increasing face above and below each wind layer')
    top=np.maximum.reduce((np.full_like(htmix,np.float32(upper_top_m)),
                           np.full_like(htmix,np.float32(20.)),
                           htmix+np.float32(minimum_upper_depth_m)))
    if np.any(zi>faces[-1]):
        raise ValueError('PACAVE mixing height exceeds the top CALMET face')
    if np.any(top>faces[-1]):
        raise ValueError('PACAVE upper averaging height exceeds the top CALMET face')
    bottom=faces[:-1,None,None]
    ceiling=faces[1:,None,None]
    lower_depth=np.maximum(np.minimum(ceiling,zi)-bottom,np.float32(0.))
    upper_depth=np.maximum(np.minimum(ceiling,top)-np.maximum(bottom,zi),np.float32(0.))

    def average(values,depth):
        # PACAVE accumulates K from the ground upward in REAL*4.
        numerator=np.cumsum(np.asarray(values*depth,dtype=np.float32),
                            axis=0,dtype=np.float32)[-1]
        denominator=np.cumsum(depth,axis=0,dtype=np.float32)[-1]
        return np.asarray(numerator/denominator,dtype=np.float32)

    lower_u=average(u,lower_depth); lower_v=average(v,lower_depth)
    upper_u=average(u,upper_depth); upper_v=average(v,upper_depth)
    # PACAVE bypasses arithmetic when ZI and ZZTOP lie in one model layer.
    # ``side='left'`` reproduces its ``ZFACE(K+1) .LT. ZI`` comparison.
    ksav=np.searchsorted(faces[1:],zi,side='left')
    direct=np.take(faces[1:],ksav)>top
    layer_u=np.take_along_axis(u,ksav[None,:,:],axis=0)[0]
    layer_v=np.take_along_axis(v,ksav[None,:,:],axis=0)[0]
    upper_u=np.where(direct,layer_u,upper_u).astype(np.float32)
    upper_v=np.where(direct,layer_v,upper_v).astype(np.float32)
    return lower_u,lower_v,upper_u,upper_v

def prognostic_temperature_3d(prognostic_temperature_k, surface_temperature_k,
                              convective_height_m, z_centers_m):
    """TEMP3D's current ``ITPROG=2`` branch.

    Level one is the final SURFVAR/AVETMP temperature.  Prognostic values are
    retained aloft except levels inside the convective layer, where CALMET
    imposes a dry-adiabatic profile relative to the first model midpoint.
    """
    source=np.asarray(prognostic_temperature_k,dtype=np.float32)
    surface=np.asarray(surface_temperature_k,dtype=np.float32)
    zic=np.asarray(convective_height_m,dtype=np.float32)
    centers=np.asarray(z_centers_m,dtype=np.float32)
    if source.ndim!=3 or source.shape[1:]!=surface.shape or surface.shape!=zic.shape or source.shape[0]!=centers.size: raise ValueError('TEMP3D inputs do not share CALMET dimensions')
    result=source.copy(); result[0]=surface
    if centers.size>1:
        adiabatic=np.asarray(surface[None,:,:]-np.float32(.0098)*
            (centers[1:,None,None]-centers[0]),dtype=np.float32)
        result[1:]=np.where(centers[1:,None,None]<=zic[None,:,:],adiabatic,result[1:])
    return result

def pgt_stability(u10, v10, solar_elevation_sine, cloud_fraction_field,
                  ceiling_height_m, landuse, monin_obukhov_m, water_range=(50,55)):
    """PGTSTB Turner land classes plus WATER's Monin-Obukhov classes."""
    u=np.asarray(u10,dtype=np.float32); v=np.asarray(v10,dtype=np.float32)
    sun=np.asarray(solar_elevation_sine,dtype=np.float32)
    cloud=np.asarray(cloud_fraction_field,dtype=np.float32)
    ceiling=np.asarray(ceiling_height_m,dtype=np.float32)
    lu=np.asarray(landuse); length=np.asarray(monin_obukhov_m,dtype=np.float32)
    jcc=np.trunc(np.float32(10.)*cloud+np.float32(.5)).astype(int)
    jceil=np.trunc(ceiling/np.float32(30.48)).astype(int)
    nighttime=sun<=np.float32(0.); insolation=np.select(
        (sun<=np.float32(.258819),sun<=np.float32(.5735764),sun<=np.float32(.8660254)),
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
    speed=np.sqrt(np.asarray(u*u+v*v,dtype=np.float32))
    knots=np.clip(np.trunc(np.float32(1.9438)*speed+np.float32(.5)).astype(int),1,12)
    result=table[knots-1,category+2]
    water=(lu>=water_range[0])&(lu<=water_range[1])
    water_class=np.where(np.abs(length)>np.float32(25.),4,
        np.where(length<np.float32(-10.),3,np.where(length<np.float32(0.),2,
        np.where(length<=np.float32(10.),6,5))))
    return np.where(water,water_class,result).astype(np.int8)

def convective_velocity_scale(convective_height_m, sensible_heat_flux, temperature_k,
                              density, landuse, water_range=(50,55)):
    """WATER/WSTARR convective velocity scale; zero for non-positive flux."""
    zi=np.asarray(convective_height_m,dtype=np.float32)
    qh=np.asarray(sensible_heat_flux,dtype=np.float32)
    temp=np.asarray(temperature_k,dtype=np.float32)
    rho=np.asarray(density,dtype=np.float32)
    lu=np.asarray(landuse)
    if lu.shape!=zi.shape:
        raise ValueError('WSTARR landuse does not match the surface fields')
    water=(lu>=water_range[0])&(lu<=water_range[1])
    land_argument=np.asarray(np.float32(9.81)*qh*zi/
                             (temp*rho*np.float32(996.)),dtype=np.float32)
    # WATER stores RH0CP*<w'theta_v'> in QH and forms WSTAR before WSTARR,
    # which deliberately skips water cells.  RH0CP is the fixed 1195.2.
    water_argument=np.asarray(np.float32(9.81)*(qh/np.float32(1195.2))*zi/temp,
                              dtype=np.float32)
    argument=np.where(water,water_argument,land_argument).astype(np.float32)
    value=np.asarray(_real4_power(np.maximum(argument,np.float32(0.)),
                                  np.float32(.3333333)),dtype=np.float32)
    return np.where((qh>0.)&(zi>0.),value,np.float32(0.)).astype(np.float32)

def solar_sine(julian_day, base_timezone, latitude, longitude_east):
    """SOLAR: 26 half-hour solar-elevation sine fields."""
    lat=np.asarray(latitude,dtype=np.float32)
    lon=np.asarray(longitude_east,dtype=np.float32)
    d=np.float32((np.float32(julian_day)-np.float32(1.))*np.float32(.9856479))
    radd=np.float32(np.float32(.0174533)*d)
    sd=_real4_libm('sinf',radd); cd=_real4_libm('cosf',radd)
    rad2d=np.float32(np.float32(2.)*radd)
    s2=_real4_libm('sinf',rad2d); c2=_real4_libm('cosf',rad2d)
    em=np.float32(np.float32(12.)+np.float32(.12357)*sd-
        np.float32(.004289)*cd+np.float32(.153809)*s2+np.float32(.060783)*c2)
    sigma=np.float32(np.float32(279.9348)+d+np.float32(1.914827)*sd-
        np.float32(.079525)*cd+np.float32(.019938)*s2-np.float32(.00162)*c2)
    sincd=np.float32(np.float32(.39784989)*_real4_libm(
        'sinf',np.float32(.0174533)*sigma))
    capd=_real4_libm('asinf',sincd); coscd=_real4_libm('cosf',capd)
    radlat=np.asarray(np.float32(.0174533)*lat,dtype=np.float32)
    sinlat=_real4_libm('sinf',radlat); coslat=_real4_libm('cosf',radlat)
    hours=np.asarray(np.arange(26,dtype=np.float32)-np.float32(.5)+
                     np.float32(base_timezone),dtype=np.float32)
    solha=np.asarray(np.float32(15.)*(hours[:,None,None]-em)+lon[None,:,:],
                     dtype=np.float32)
    return np.asarray(sinlat[None,:,:]*sincd+coslat[None,:,:]*coscd*
                      _real4_libm('cosf',np.float32(.0174533)*solha),dtype=np.float32)

def terrain_gradients(terrain_m, dx_m, dy_m):
    """CALMET one-sided-at-boundary, centred-interior terrain gradients."""
    h=np.asarray(terrain_m,dtype=float)
    west=np.concatenate((h[:,:1],h[:,:-1]),axis=1); east=np.concatenate((h[:,1:],h[:,-1:]),axis=1)
    south=np.concatenate((h[:1,:],h[:-1,:]),axis=0); north=np.concatenate((h[1:,:],h[-1:,:]),axis=0)
    return (east-west)*(0.5/dx_m),(north-south)*(0.5/dy_m)

def terrain_adjust_surface_wind(u, v, terrain_m, top_height_m, dx_m, dy_m,
                                *, maximum_iterations=75, tolerance=.02,
                                relaxation=1.5):
    """ADJUST terrain-potential correction for objective-analysis winds.

    CALMET applies this Poisson solve only to the first wind layer when
    ``IWFCOD=0``.  The four alternating Gauss-Seidel sweeps are deliberately
    retained because every sweep immediately reuses updated neighbours.
    """
    f32=np.float32
    u=np.array(u,dtype=f32,copy=True); v=np.array(v,dtype=f32,copy=True)
    terrain=np.asarray(terrain_m,dtype=f32)
    if u.shape!=v.shape or u.ndim!=3 or u.shape[1:]!=terrain.shape:
        raise ValueError('wind must have shape (z,y,x) matching terrain')
    if top_height_m<=0. or dx_m<=0. or dy_m<=0.:
        raise ValueError('ADJUST heights and grid spacing must be positive')
    west_h=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east_h=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south_h=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north_h=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    fx=f32(.5)/f32(dx_m)/f32(top_height_m)
    fy=f32(.5)/f32(dy_m)/f32(top_height_m)
    gx=np.asarray((east_h-west_h)*fx,dtype=f32)
    gy=np.asarray((north_h-south_h)*fy,dtype=f32)
    forcing=np.asarray(u[0]*gx+v[0]*gy,dtype=f32)
    phi=np.zeros_like(terrain,dtype=f32)
    west=forcing[:,0].copy(); east=forcing[:,-1].copy()
    south=forcing[0].copy(); north=forcing[-1].copy()
    dx2=f32(f32(dx_m)*f32(dx_m)); dy2=f32(f32(dy_m)*f32(dy_m))
    area=f32(dx2*dy2); factor=f32(f32(1.)/(f32(2.)*f32(dx2+dy2)))
    relax=f32(relaxation); one_minus=f32(f32(1.)-relax)
    ny,nx=terrain.shape
    orders=((range(ny),range(nx)),(range(ny),range(nx-1,-1,-1)),
            (range(ny-1,-1,-1),range(nx)),
            (range(ny-1,-1,-1),range(nx-1,-1,-1)))
    for _ in range(maximum_iterations):
        for rows,columns in orders:
            # ADJUST resets ERROR for every IDIR and tests the value left by
            # the fourth sweep, rather than the maximum over all four.
            error=f32(-1.e9)
            for j in rows:
                for i in columns:
                    old=phi[j,i]
                    left=phi[j,i-1] if i else west[j]
                    right=phi[j,i+1] if i+1<nx else east[j]
                    below=phi[j-1,i] if j else south[i]
                    above=phi[j+1,i] if j+1<ny else north[i]
                    xx=f32(dy2*f32(right+left)); yy=f32(dx2*f32(above+below))
                    phi[j,i]=f32(one_minus*old+f32(relax*factor)*
                                 f32(f32(xx+yy)-f32(area*forcing[j,i])))
                    if i==0: west[j]=phi[j,i]
                    if i==nx-1: east[j]=phi[j,i]
                    if j==0: south[i]=phi[j,i]
                    if j==ny-1: north[i]=phi[j,i]
                    if abs(old)>=1.e-10:
                        error=max(error,f32(abs(f32(phi[j,i]-old)/old)))
        if error<=tolerance: break
    left=np.concatenate((west[:,None],phi[:,:-1]),axis=1)
    right=np.concatenate((phi[:,1:],east[:,None]),axis=1)
    below=np.concatenate((south[None,:],phi[:-1]),axis=0)
    above=np.concatenate((phi[1:],north[None,:]),axis=0)
    dxi=f32(.5)/f32(dx_m); dyi=f32(.5)/f32(dy_m)
    u[0]=np.asarray((right-left)*dxi+u[0],dtype=f32)
    v[0]=np.asarray((above-below)*dyi+v[0],dtype=f32)
    return u,v

def terrain_extrema(terrain_m, radius_km, dx_m, dy_m):
    """TERSET crest/valley heights and distances, preserving scan order."""
    f32=np.float32
    h=np.asarray(terrain_m,dtype=f32); ny,nx=h.shape
    highest=h.copy(); lowest=h.copy()
    distance_high=np.zeros_like(h,dtype=f32); distance_low=np.zeros_like(h,dtype=f32)
    dx_km=f32(f32(dx_m)*f32(.001)); dy_km=f32(f32(dy_m)*f32(.001))
    radius=f32(radius_km)
    irange=int(np.floor(f32(radius/dx_km)+f32(.5)))
    jrange=int(np.floor(f32(radius/dy_km)+f32(.5)))
    for dj in range(-jrange,jrange+1):
        for di in range(-irange,irange+1):
            xdistance=f32(f32(di)*dx_km); ydistance=f32(f32(dj)*dy_km)
            distance=f32(np.sqrt(f32(f32(xdistance*xdistance)+f32(ydistance*ydistance))))
            if distance>radius: continue
            target_y=slice(max(0,-dj),min(ny,ny-dj)); target_x=slice(max(0,-di),min(nx,nx-di))
            source_y=slice(max(0,dj),min(ny,ny+dj)); source_x=slice(max(0,di),min(nx,nx+di))
            candidate=h[source_y,source_x]
            distance_m=f32(distance*f32(1000.))
            hi=highest[target_y,target_x]; update=candidate>hi
            hi[update]=candidate[update]; distance_high[target_y,target_x][update]=distance_m
            lo=lowest[target_y,target_x]; update=candidate<lo
            lo[update]=candidate[update]; distance_low[target_y,target_x][update]=distance_m
    return highest,lowest,distance_high,distance_low

def topographic_vertical_velocity(u,v,terrain_m,z_faces_m,gamma_k_per_m,temperature_k,alpha,dx_m,dy_m):
    """TOPOF2 terrain-following vertical velocity; wind axes are (z,y,x)."""
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32); faces=np.asarray(z_faces_m,dtype=f32)
    terrain=np.asarray(terrain_m,dtype=f32)
    west=np.concatenate((terrain[:,:1],terrain[:,:-1]),axis=1)
    east=np.concatenate((terrain[:,1:],terrain[:,-1:]),axis=1)
    south=np.concatenate((terrain[:1,:],terrain[:-1,:]),axis=0)
    north=np.concatenate((terrain[1:,:],terrain[-1:,:]),axis=0)
    gx=np.asarray((east-west)*(f32(.5)/f32(dx_m)),dtype=f32)
    gy=np.asarray((north-south)*(f32(.5)/f32(dy_m)),dtype=f32)
    gamma2=np.asarray(np.asarray(gamma_k_per_m,dtype=f32)-f32(-.01),dtype=f32)
    stable=np.sqrt(np.maximum(
        np.asarray(f32(9.8)*gamma2/np.asarray(temperature_k,dtype=f32),dtype=f32),f32(0.)))
    stability=np.where(gamma2<f32(0.),f32(-1.),stable).astype(f32)
    speed=np.sqrt(np.asarray(u[-1]*u[-1]+v[-1]*v[-1],dtype=f32))
    speed=np.maximum(speed,f32(1.e-6))
    wave=np.where(stability<=f32(0.),f32(2.)/f32(500.),stability/speed).astype(f32)
    wtopo=np.asarray(u[0]*gx+v[0]*gy,dtype=f32); nz=u.shape[0]
    decay=np.empty((nz+1,)+wtopo.shape,dtype=f32); decay[0]=wtopo
    w=np.empty_like(decay); w[0]=wtopo
    for k in range(nz):
        bkz=np.minimum(np.asarray(wave*faces[k+1],dtype=f32),f32(50.))
        decay[k+1]=np.asarray(wtopo*np.exp(-bkz),dtype=f32)
        dzi=f32(1.)/f32(faces[k+1]-faces[k])
        dwdz1=np.asarray((decay[k+1]-decay[k])*dzi,dtype=f32)
        dwdz=np.asarray(f32(alpha)*dwdz1,dtype=f32)
        w[k+1]=np.asarray(dwdz/dzi+w[k],dtype=f32)
    w[0]=np.asarray(w[0]-u[0]*gx-v[0]*gy,dtype=f32)
    w[1:]=np.asarray(w[1:]-u*gx[None,:,:]-v*gy[None,:,:],dtype=f32)
    return w

def slope_flow(terrain_m,highest_m,lowest_m,distance_high_m,distance_low_m,sensible_heat_flux,temperature_k,landuse,z_faces_m,dx_m,dy_m,water_range=(50,55)):
    """SLOPE Mahrt/Horst-Doran shooting-flow parameterisation."""
    h=np.asarray(terrain_m,dtype=np.float32)
    qh=np.asarray(sensible_heat_flux,dtype=np.float32)
    temp=np.asarray(temperature_k,dtype=np.float32)
    dx_inverse=np.float32(.5)/np.float32(dx_m)
    dy_inverse=np.float32(.5)/np.float32(dy_m)
    west=np.concatenate((h[:,:1],h[:,:-1]),axis=1)
    east=np.concatenate((h[:,1:],h[:,-1:]),axis=1)
    south=np.concatenate((h[:1,:],h[:-1,:]),axis=0)
    north=np.concatenate((h[1:,:],h[-1:,:]),axis=0)
    gx=np.asarray((east-west)*dx_inverse,dtype=np.float32)
    gy=np.asarray((north-south)*dy_inverse,dtype=np.float32)
    # SLOPE spells this as ``(... )**0.5`` in default REAL.  GNU therefore
    # calls powf; sqrtf can differ by one bit and perturb the shooting flow.
    magnitude=np.asarray(_real4_power(
        np.asarray(gx*gx+gy*gy,dtype=np.float32),np.float32(.5)),dtype=np.float32)
    angle=_real4_libm('atanf',magnitude)
    active=(np.abs(angle)>=np.float32(.009))
    active &= ~((np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1]))
    down=qh<0.
    relief=np.where(down,np.asarray(highest_m,dtype=np.float32)-h,
                    h-np.asarray(lowest_m,dtype=np.float32)).astype(np.float32)
    hd=np.asarray(np.float32(.05)*relief,dtype=np.float32)
    distance=np.where(down,np.asarray(distance_high_m,dtype=np.float32),
                      np.asarray(distance_low_m,dtype=np.float32))
    dcrest=np.sqrt(np.asarray(relief*relief+distance*distance,dtype=np.float32))
    sinalf=_real4_libm('sinf',np.abs(angle))
    bounded=np.divide(relief,dcrest,out=np.zeros_like(relief),where=dcrest>0)
    sinalf=np.where(down,np.minimum(sinalf,bounded),sinalf)
    exponent=np.maximum(np.asarray(-dcrest*np.float32(.08)/
        np.maximum(hd,np.float32(1.e-30)),dtype=np.float32),np.float32(-50.))
    one_third=np.float32(1.)/np.float32(3.)
    downslope_base=np.asarray(sinalf*(np.float32(9.81)/temp)*np.abs(qh)*dcrest/
                              np.float32(1229.9)/np.float32(.08),dtype=np.float32)
    decay=np.asarray(np.float32(1.)-_real4_libm('expf',exponent),dtype=np.float32)
    downslope=-np.asarray(_real4_power(downslope_base,one_third)*
                          _real4_power(decay,one_third),dtype=np.float32)
    upslope_base=np.asarray((np.float32(9.81)/temp)*np.abs(qh)*relief/
                            np.float32(1229.9),dtype=np.float32)
    upslope=_real4_power(upslope_base,one_third)
    velocity=np.where(down,downslope,upslope); velocity=np.where(active&(hd>.05),velocity,0.)
    # SLOPE deliberately goes through rounded degrees and radians rather
    # than normalising the gradient directly (the same legacy angle chain
    # used by FRADJ).  It matters when a slope component nearly cancels the
    # background wind.
    theta_p=np.asarray(_real4_libm('atanf',np.divide(
        gy,gx,out=np.zeros_like(gy),where=gx!=0.))*
                       np.float32(57.296),dtype=np.float32)
    theta=np.where(gx<0.,theta_p+np.float32(180.),theta_p)
    theta=np.where((gx>0.)&(gy<=0.),theta_p+np.float32(360.),theta)
    theta=np.where((gx==0.)&(gy<0.),np.float32(270.),theta)
    theta=np.where((gx==0.)&(gy>0.),np.float32(90.),theta)
    theta_d=np.where((theta>=0.)&(theta<=90.),np.float32(90.)-theta,
                     np.float32(450.)-theta)
    direction=np.asarray((np.float32(270.)-theta_d)*np.float32(.017453),
                         dtype=np.float32)
    ux=np.asarray(-_real4_libm('cosf',direction)*velocity,dtype=np.float32)
    vy=np.asarray(-_real4_libm('sinf',direction)*velocity,dtype=np.float32)
    ux=np.where(magnitude>0.,ux,np.float32(0.)); vy=np.where(magnitude>0.,vy,np.float32(0.))
    faces=np.asarray(z_faces_m,dtype=np.float32)
    ratios=np.clip((hd[None,:,:]-faces[:-1,None,None])/
                   np.diff(faces)[:,None,None],np.float32(0.),np.float32(1.))
    return np.asarray(ratios*ux[None,:,:],dtype=np.float32),\
        np.asarray(ratios*vy[None,:,:],dtype=np.float32)

def vertical_layer_average(values, heights_m, faces_m, *, missing=998.99):
    """VERTAV exact piecewise-linear layer mean for arbitrary profiles.

    The final axis is the source sounding level. All preceding axes identify
    independent profiles. ``heights_m`` may be one-dimensional or match the
    values array.
    """
    values=np.asarray(values,dtype=float); heights=np.broadcast_to(np.asarray(heights_m,dtype=float),values.shape)
    faces=np.asarray(faces_m,dtype=float)
    if values.shape[-1]<2 or faces.ndim!=1 or faces.size<2: raise ValueError('at least two source levels and two faces are required')
    z0,z1=heights[...,:-1],heights[...,1:]; v0,v1=values[...,:-1],values[...,1:]
    valid=(v0<missing)&(v1<missing)&(z1>z0)
    output=[]
    for bottom,top in zip(faces[:-1],faces[1:]):
        lo=np.maximum(z0,bottom); hi=np.minimum(z1,top); width=np.maximum(hi-lo,0.); use=valid&(width>0.)
        fraction_lo=np.divide(lo-z0,z1-z0,out=np.zeros_like(lo),where=z1>z0)
        fraction_hi=np.divide(hi-z0,z1-z0,out=np.zeros_like(hi),where=z1>z0)
        vlo=v0+(v1-v0)*fraction_lo; vhi=v0+(v1-v0)*fraction_hi
        integral=np.sum(np.where(use,.5*(vlo+vhi)*width,0.),axis=-1)
        coverage=np.sum(np.where(use,width,0.),axis=-1)
        if np.any(np.abs(coverage-(top-bottom))>1.e-4): raise ValueError(f'sounding does not fully cover layer {bottom:g}-{top:g} m')
        output.append(integral/(top-bottom))
    return np.stack(output,axis=-1)

def vertical_interpolate(values, heights_m, targets_m, *, missing=998.99):
    """FACET/INTP piecewise-linear interpolation with missing-level skips."""
    values=np.asarray(values,dtype=float); heights=np.broadcast_to(np.asarray(heights_m,dtype=float),values.shape)
    targets=np.asarray(targets_m,dtype=float); leading=values.shape[:-1]; result=np.empty(leading+targets.shape)
    for index in np.ndindex(leading):
        good=values[index]<missing; z=heights[index][good]; v=values[index][good]
        if z.size<2 or targets.min()<z.min() or targets.max()>z.max(): raise ValueError('valid sounding does not bracket interpolation targets')
        result[index]=np.interp(targets,z,v)
    return result

def smooth_wind(u,v,passes):
    """SMOOTH's in-place Gauss-Seidel stencil, including fixed boundaries.

    Both axes remain sequential because Fortran immediately reuses updated
    western and southern neighbours.  Each addition in the four-neighbour
    sum also rounds to REAL*4; regrouping it as a linear recurrence changes
    that association and can produce one-ulp wind errors.
    """
    # U/V and their boundary arrays are REAL*4 in SMOOTH.  Keeping the
    # container itself single precision is important because the updated
    # southern row is consumed immediately by the next row.
    u=np.array(u,dtype=np.float32,copy=True); v=np.array(v,dtype=np.float32,copy=True)
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
                    total=np.float32(np.float32(np.float32(ue+uw)+un)+us)
                    u[k,j,i]=np.float32(np.float32(np.float32(.5)*uc)+
                                        np.float32(np.float32(.125)*total))

                    vc=v[k,j,i]
                    vw=vc if i==0 else v[k,j,i-1]
                    ve=vc if i+1==v.shape[2] else v[k,j,i+1]
                    vs=vb_bottom[k,i] if j==0 else v[k,j-1,i]
                    vn=vb_top[k,i] if j+1==v.shape[1] else v[k,j+1,i]
                    total=np.float32(np.float32(np.float32(ve+vw)+vn)+vs)
                    v[k,j,i]=np.float32(np.float32(np.float32(.5)*vc)+
                                        np.float32(np.float32(.125)*total))
    return u,v

def divergence_and_vertical_velocity(u,v,z_faces_m,dx_m,dy_m,*,obrien=False,boundaries=None):
    """DIVCEL plus vertical integration and optional O'Brien correction."""
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32); faces=np.asarray(z_faces_m,dtype=f32)
    if boundaries is None:
        u_west,u_east,v_south,v_north=u[:,:,0],u[:,:,-1],v[:,0,:],v[:,-1,:]
    else:
        u_west,u_east,v_south,v_north=(np.asarray(value,dtype=f32) for value in boundaries)
    up=np.concatenate((u_west[:,:,None],u,u_east[:,:,None]),axis=2)
    vp=np.concatenate((v_south[:,None,:],v,v_north[:,None,:]),axis=1)
    dxi=f32(1.)/(f32(2.)*f32(dx_m)); dyi=f32(1.)/(f32(2.)*f32(dy_m))
    divergence=np.asarray((up[:,:,2:]-up[:,:,:-2])*dxi+
                          (vp[:,2:,:]-vp[:,:-2,:])*dyi,dtype=f32)
    w=np.zeros((u.shape[0]+1,)+u.shape[1:],dtype=f32)
    # The vertical recurrence rounds after every model layer in DIAGNO.
    for k in range(u.shape[0]):
        w[k+1]=np.asarray(-((faces[k+1]-faces[k])*divergence[k])+w[k],dtype=f32)
    if obrien:
        top=w[-1].copy()
        for k in range(u.shape[0]):
            ratio=f32(faces[k+1]/faces[-1])
            w[k+1]=np.asarray(w[k+1]-ratio*top,dtype=f32)
    return divergence,w

def minimize_divergence(u,v,w,z_faces_m,dx_m,dy_m,*,boundaries=None,
                        maximum_iterations=50,limit=5.e-6):
    """MINIM four-sweep horizontal mass-consistency adjustment."""
    # MINIM is a REAL*4 routine.  Each interior face receives two additions
    # per directional sweep, in the II/JJ traversal order.  Those additions
    # are mathematically commutative but not bitwise commutative, so retain
    # the four legacy scans rather than collapsing them into one reduction.
    f32=np.float32
    u=np.array(u,dtype=f32,copy=True); v=np.array(v,dtype=f32,copy=True)
    w=np.asarray(w,dtype=f32); faces=np.asarray(z_faces_m,dtype=f32)
    if u.shape!=v.shape or w.shape!=(u.shape[0]+1,*u.shape[1:]):
        raise ValueError('MINIM wind arrays have inconsistent dimensions')
    if boundaries is None:
        boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    west,east,south,north=(np.array(value,dtype=f32,copy=True) for value in boundaries)
    divergence=np.empty_like(u); iterations=np.zeros(u.shape[0],dtype=np.int32)

    def layer_divergence(k):
        up=np.concatenate((west[k,:,None],u[k],east[k,:,None]),axis=1)
        vp=np.concatenate((south[k,None,:],v[k],north[k,None,:]),axis=0)
        dxi=f32(1.)/(f32(2.)*f32(dx_m)); dyi=f32(1.)/(f32(2.)*f32(dy_m))
        horizontal=(up[:,2:]-up[:,:-2])*dxi+(vp[2:]-vp[:-2])*dyi
        return np.asarray(horizontal+(w[k+1]-w[k])/(faces[k+1]-faces[k]),dtype=f32)

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
                        if div[j,i]==f32(0.): continue
                        ut=f32(-f32(2.)*(div[j,i]*f32(dx_m))/f32(2.))
                        vt=f32(-f32(2.)*(div[j,i]*f32(dy_m))/f32(2.))
                        if i>0: u[k,j,i-1]=f32(u[k,j,i-1]-f32(.5)*ut)
                        if i+1<u.shape[2]: u[k,j,i+1]=f32(u[k,j,i+1]+f32(.5)*ut)
                        if j>0: v[k,j-1,i]=f32(v[k,j-1,i]-f32(.5)*vt)
                        if j+1<u.shape[1]: v[k,j+1,i]=f32(v[k,j+1,i]+f32(.5)*vt)
                west[k]=u[k,:,0]; east[k]=u[k,:,-1]
                south[k]=v[k,0,:]; north[k]=v[k,-1,:]
                divergence[k]=layer_divergence(k)
            iterations[k]=iteration
            if np.max(np.abs(divergence[k]))<=limit: break
    return u,v,divergence,iterations

def heat_flux(solar_sine_field,cloud_fraction_field,temperature_k,landuse,albedo,
              bowen,soil_heat_flux,anthropogenic_heat,water_range=(50,55),*,
              turbidity_coefficient_1=990.,turbidity_coefficient_2=-30.,
              cloudiness_coefficient_1=-.75,cloudiness_coefficient_2=3.4,
              net_radiation_coefficient_1=5.31e-13,
              net_radiation_coefficient_2=60.,net_radiation_coefficient_3=.12):
    """HEATFX short-wave radiation and land sensible heat flux."""
    sun=np.asarray(solar_sine_field,dtype=np.float32)
    cloud=np.asarray(cloud_fraction_field,dtype=np.float32)
    temp=np.asarray(temperature_k,dtype=np.float32)
    albedo=np.asarray(albedo,dtype=np.float32); bowen=np.asarray(bowen,dtype=np.float32)
    soil=np.asarray(soil_heat_flux,dtype=np.float32)
    anthropogenic=np.asarray(anthropogenic_heat,dtype=np.float32)
    ha1=np.float32(turbidity_coefficient_1); ha2=np.float32(turbidity_coefficient_2)
    hb1=np.float32(cloudiness_coefficient_1); hb2=np.float32(cloudiness_coefficient_2)
    hc1=np.float32(net_radiation_coefficient_1)
    hc2=np.float32(net_radiation_coefficient_2)
    hc3=np.float32(net_radiation_coefficient_3)
    cloud_power=np.asarray(_real4_power(cloud,hb2),dtype=np.float32)
    qsw=np.maximum((ha1*sun+ha2)*
        (np.float32(1.)+hb1*cloud_power),np.float32(0.))
    # gfortran's optimized REAL**4/REAL**6 expansion forms x², then x⁴=x²*x²
    # and x⁶=(x²*x)*(x²*x).  NumPy's integer-power ufunc uses a different
    # rounding path for float32 operands.
    temp2=np.asarray(temp*temp,dtype=np.float32)
    temp3=np.asarray(temp2*temp,dtype=np.float32)
    temp4=np.asarray(temp2*temp2,dtype=np.float32)
    temp6=np.asarray(temp3*temp3,dtype=np.float32)
    qstar=np.asarray(((np.float32(1.)-albedo)*qsw+
        hc1*temp6-
        np.float32(5.67e-8)*temp4+hc2*cloud)/
        np.asarray(hc3+np.float32(1.),dtype=np.float32),dtype=np.float32)
    qh=np.asarray(bowen*(qstar*(np.float32(1.)-soil)+anthropogenic)/
                  (np.float32(1.)+bowen),dtype=np.float32)
    qh=np.where(sun<=0.,np.float32(-.1),qh).astype(np.float32)
    water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
    return np.where(water,0.,qh),qsw

def psi_momentum_dyer(z_over_l):
    """PSIUD CALMET/COARE momentum stability correction."""
    zl=np.asarray(z_over_l,dtype=float); result=np.empty_like(zl); unstable=zl<0.
    # PSIUD declares its variables REAL*8, but the unsuffixed constants in
    # the legacy source are default REAL.  Literal-only expressions such as
    # SQRT(3.) and 4.*ATAN(1.) are consequently evaluated in REAL*4 before
    # promotion.  Preserve that mixed-kind arithmetic: using ideal double
    # constants shifts ELUSTR's Monin-Obukhov iteration by several ulps.
    one=np.float64(np.float32(1.)); two=np.float64(np.float32(2.))
    zu=zl[unstable]
    x=(one-np.float64(np.float32(16.))*zu)**np.float64(np.float32(.25))
    atan_one=np.float32(_real4_libm('atanf',np.float32(1.)))
    two_atan_one=np.float64(np.float32(np.float32(2.)*atan_one))
    psik=two*np.log((one+x)/two)+np.log((one+x*x)/two)-two*np.arctan(x)+two_atan_one
    y=(one-np.float64(np.float32(10.15))*zu)**np.float64(np.float32(.3333))
    sqrt_three=np.float32(_real4_libm('sqrtf',np.float32(3.)))
    four_atan_over_sqrt=np.float64(np.float32(
        np.float32(np.float32(4.)*atan_one)/sqrt_three))
    psic=np.float64(np.float32(1.5))*np.log((one+y+y*y)/np.float64(np.float32(3.)))-np.float64(sqrt_three)*np.arctan(
        (one+two*y)/np.float64(sqrt_three))+four_atan_over_sqrt
    f=zu*zu/(one+zu*zu); result[unstable]=(one-f)*psik+f*psic
    stable=~unstable
    c=np.minimum(np.float64(50.),np.float64(np.float32(.35))*zl[stable])
    result[stable]=-((one+zl[stable])+np.float64(np.float32(.6667))*(
        zl[stable]-np.float64(np.float32(14.28)))/np.exp(c)+np.float64(np.float32(8.525)))
    return result

def surface_similarity(u,v,roughness_m,height_m,density,temperature_k,
                       cloud_fraction_field,sensible_heat_flux,landuse,
                       water_range=(50,55),*,wind_measurement_height_m=None,
                       ipsifcn=0,minimum_stable_wind_m_s=.5):
    """ELUSTR land-cell friction velocity, heat flux and M-O length."""
    u=np.asarray(u,dtype=np.float32); v=np.asarray(v,dtype=np.float32)
    z0=np.asarray(roughness_m,dtype=np.float32); rho=np.asarray(density,dtype=np.float32)
    temp=np.asarray(temperature_k,dtype=np.float32)
    cloud=np.asarray(cloud_fraction_field,dtype=np.float32)
    qh=np.array(sensible_heat_flux,dtype=np.float32,copy=True)
    height=np.float32(height_m)
    measurement=(height if wind_measurement_height_m is None else
                 np.asarray(wind_measurement_height_m,dtype=np.float32))
    if ipsifcn not in {0,1}: raise ValueError('IPSIFCN must be zero or one')
    squared=np.asarray(u*u+v*v,dtype=np.float32)
    speed=np.maximum(np.sqrt(squared),np.float32(.001))
    # ALOG on a default REAL expression is GNU libm's logf.  NumPy's
    # float32 log ufunc is normally within one ulp, but that last bit is
    # amplified by ELUSTR's short, fixed Monin-Obukhov iteration.
    logz=np.asarray(_real4_libm('logf',np.asarray(height/z0,dtype=np.float32)),
                    dtype=np.float32)
    ustar=np.zeros_like(speed); length=np.zeros_like(speed)
    daytime=qh>=0.; trial=np.maximum(np.float32(.4)*speed/logz,np.float32(.05))
    xl=np.full_like(speed,np.float32(-9.9e9))
    active=daytime&(qh!=0.)
    for _ in range(5):
        old=xl
        # gfortran lowers REAL**3 to two rounded REAL*4 multiplications.
        # NumPy's power ufunc can differ by one ulp, which is enough to move
        # ELUSTR across its 0.5 percent convergence boundary.
        trial_cubed=np.asarray(np.asarray(trial*trial,dtype=np.float32)*trial,
                               dtype=np.float32)
        new=np.where(active,np.asarray(-np.float32(253.8226)*rho*temp*
            trial_cubed/np.where(qh==0.,np.float32(1.),qh),dtype=np.float32),xl)
        # PSIUD alone is REAL*8.  ELUSTR forms its argument in REAL*4 and
        # immediately assigns the double result back to a REAL correction.
        if ipsifcn==0:
            psizl=np.asarray(psi_momentum_dyer(
                np.asarray(measurement/new,dtype=np.float32).astype(np.float64)),dtype=np.float32)
            psiz0l=np.asarray(psi_momentum_dyer(
                np.asarray(z0/new,dtype=np.float32).astype(np.float64)),dtype=np.float32)
        else:
            psizl=_psiuc(measurement,new,15)
            psiz0l=_psiuc(z0,new,15)
        denominator=np.asarray(logz-psizl+psiz0l,dtype=np.float32)
        refined=np.maximum(np.asarray(np.float32(.4)*speed/denominator,dtype=np.float32),
                           np.float32(.05))
        converged=active&(np.abs(np.asarray((new-old)/new,dtype=np.float32))<np.float32(.005))
        xl=np.where(active,new,xl).astype(np.float32)
        trial=np.where(active,refined,trial).astype(np.float32)
        active &= ~converged
    ustar=np.where(daytime,trial,ustar); length=np.where(daytime,xl,length)
    night=~daytime
    stable_speed=np.maximum(speed,np.float32(minimum_stable_wind_m_s))
    jcc=np.trunc(np.float32(10.)*cloud+np.float32(.5)).astype(np.int32)
    # Fortran converts JCC**2 to default REAL before the surrounding REAL*4
    # products.  NumPy promotes a float32 scalar multiplied by an int32 array
    # to float64, so convert the integer square explicitly before evaluating
    # the THETA1 expression.
    cloud_index_squared=np.asarray(jcc*jcc,dtype=np.float32)
    theta1=np.asarray(np.float32(.09)*(np.float32(1.)-
        np.float32(5.e-3)*cloud_index_squared),dtype=np.float32)
    cdn=np.asarray(np.float32(.4)/logz,dtype=np.float32)
    # NumPy 2 promotes ``float32_array ** np.int32(2)`` to float64, whereas
    # GNU evaluates the default-REAL ``WS ** 2`` as one rounded REAL*4
    # multiplication.  Spell out that multiplication so the result is also
    # stable across NumPy's pre-2.0 and 2.x promotion rules.
    stable_speed_squared=np.asarray(stable_speed*stable_speed,dtype=np.float32)
    theta2=np.asarray(temp*cdn*stable_speed_squared/
                      (np.float32(184.428)*height),dtype=np.float32)
    theta=np.maximum(np.minimum(theta1,theta2),np.float32(1.e-9))
    u02=np.asarray(np.float32(46.107)*height*theta/temp,dtype=np.float32)
    stable_u=np.maximum(np.asarray(np.float32(.5)*cdn*(stable_speed+np.sqrt(np.maximum(
        np.float32(0.),stable_speed*stable_speed-np.float32(4.)*u02/cdn))),dtype=np.float32),
        np.float32(.05))
    # ELUSTR enters this refinement only when the initial product exceeds
    # XCRIT.  Once inside, its early-exit threshold is 1.05*XCRIT; testing
    # against XCRIT again performs extra iterations and is not equivalent.
    adjust=(stable_u*theta)>np.float32(.05)
    tolerance=np.float32(1.05)*np.float32(.05)
    for _ in range(3):
        theta=np.where(adjust,np.float32(.05)/stable_u,theta).astype(np.float32)
        u02=np.asarray(np.float32(46.107)*height*theta/temp,dtype=np.float32)
        updated=np.maximum(np.asarray(np.float32(.5)*cdn*(stable_speed+np.sqrt(np.maximum(
            np.float32(0.),stable_speed*stable_speed-np.float32(4.)*u02/cdn))),dtype=np.float32),
            np.float32(.05))
        stable_u=np.where(adjust,updated,stable_u).astype(np.float32)
        adjust &= (stable_u*theta)>tolerance
    stable_qh=np.asarray(-np.float32(996.)*rho*stable_u*theta,dtype=np.float32)
    stable_u_cubed=np.asarray(np.asarray(stable_u*stable_u,dtype=np.float32)*stable_u,
                              dtype=np.float32)
    stable_l=np.asarray(-np.float32(253.8226)*rho*temp*stable_u_cubed/stable_qh,
                        dtype=np.float32)
    ustar=np.where(night,stable_u,ustar); qh=np.where(night,stable_qh,qh); length=np.where(night,stable_l,length)
    water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
    return np.where(water,0.,ustar),np.where(water,0.,length),np.where(water,0.,qh)


def land_station_surface_similarity(u,v,roughness_m,anemometer_height_m,
                                    first_level_height_m,density,temperature_k,
                                    cloud_fraction,sensible_heat_flux,*,ipsifcn=0):
    """ELUSTR2 surface-layer state for vectorized land stations."""
    return surface_similarity(u,v,roughness_m,first_level_height_m,density,
        temperature_k,cloud_fraction,sensible_heat_flux,
        np.zeros(np.shape(u),dtype=np.int16),wind_measurement_height_m=anemometer_height_m,
        ipsifcn=ipsifcn,minimum_stable_wind_m_s=0.)


def land_station_mixing_height(temperature_profile_k,profile_height_m,
        previous_convective_height_m,temperature_jump_k,friction_velocity,
        monin_obukhov_m,sensible_heat_flux,density,coriolis,*,mixing_method=1,
        surface_temperature_k=None,**options):
    """MIXHT2ST over prognostic columns for a vector of land stations."""
    temperature=np.asarray(temperature_profile_k,dtype=np.float32)
    height=np.asarray(profile_height_m,dtype=np.float32)
    if temperature.ndim!=2 or height.shape!=temperature.shape:
        raise ValueError('station temperature profiles must have shape (level, station)')
    shaped=lambda value: np.asarray(value,dtype=np.float32).reshape(1,-1)
    arguments=(temperature[:,None,:],height[:,None,:],
        shaped(previous_convective_height_m),shaped(temperature_jump_k),
        shaped(friction_velocity),shaped(monin_obukhov_m),
        shaped(sensible_heat_flux),shaped(density),shaped(coriolis))
    if abs(int(mixing_method))==1:
        result=mixing_height_maul_carson(*arguments,**options)
    elif abs(int(mixing_method))==2:
        if surface_temperature_k is None:
            raise ValueError('MIXHBG station profiles require surface temperature')
        result=mixing_height_batchvarova_gryning(
            *arguments,surface_temperature_k=shaped(surface_temperature_k),**options)
    else:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    return tuple(np.asarray(value[0],dtype=np.float32) for value in result)


def land_station_similarity_theory(u,v,roughness_m,anemometer_height_m,
        first_level_height_m,density,temperature_k,cloud_fraction,
        sensible_heat_flux,temperature_profile_k,profile_height_m,
        previous_convective_height_m,temperature_jump_k,coriolis,
        z_centers_m,z_faces_m,station_latitude,*,mixing_method=1,ipsifcn=0,
        minimum_mixing_height_m=50.,**mixing_options):
    """Vectorized land branch of STHEOR: ELUSTR2, MIXHT2ST, then SIMILT.

    Missing and calm station winds retain their first-level observation and
    remain missing aloft, matching ELUSTR2's early return before SIMILT.
    """
    f32=np.float32
    u=np.asarray(u,dtype=f32); v=np.asarray(v,dtype=f32)
    if u.ndim!=1 or v.shape!=u.shape:
        raise ValueError('STHEOR station winds must be matching one-dimensional arrays')
    station_values=(roughness_m,anemometer_height_m,density,temperature_k,
                    cloud_fraction,sensible_heat_flux,
                    previous_convective_height_m,temperature_jump_k,coriolis,
                    station_latitude)
    converted=tuple(np.asarray(value,dtype=f32) for value in station_values)
    if any(value.shape!=u.shape for value in converted):
        raise ValueError('STHEOR station inputs must have identical shapes')
    (roughness,anemometer,rho,temperature,cloud,heat,previous,jump,
     coriolis_value,latitude)=converted
    profile_temperature=np.asarray(temperature_profile_k,dtype=f32)
    profile_height=np.asarray(profile_height_m,dtype=f32)
    if profile_temperature.shape!=profile_height.shape or profile_temperature.shape[1:]!=u.shape:
        raise ValueError('STHEOR temperature profiles must have shape (level, station)')
    centers=np.asarray(z_centers_m,dtype=f32)
    output_u=np.full((centers.size,u.size),f32(9999.),dtype=f32)
    output_v=np.full_like(output_u,f32(9999.))
    output_u[0]=u; output_v[0]=v
    state=[np.full(u.shape,np.nan,dtype=f32) for _ in range(6)]
    speed=_real4_libm('sqrtf',np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32))
    active=(u<=f32(990.))&(v<=f32(990.))&(speed>=f32(.001))
    if not active.any():
        return (output_u,output_v,*state)
    ustar,length,station_heat=land_station_surface_similarity(
        u[active],v[active],roughness[active],anemometer[active],
        first_level_height_m,rho[active],temperature[active],cloud[active],
        heat[active],ipsifcn=ipsifcn)
    mix_options=dict(mixing_options)
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
    return (output_u,output_v,*state)


def objective_analysis_component(station_values,station_xy,first_guess,landuse,dx_km,dy_km,
                                 *,rmin_km=.1,rmax_surface_km=50.,rmax_upper_km=100.,
                                 rmax_water_km=100.,surface_scale_km=10.,upper_scale_km=20.,
                                 stations_per_level=5,observation_weight=1.,vary_radius=True,
                                 water_range=(50,55),barriers=(),barrier_top_level=0,
                                 background_weight=None,block_size=32768):
    """Vectorized no-barrier INTER2 component analysis for IPROG 0/2/4/14."""
    f32=np.float32
    values=np.asarray(station_values,dtype=f32); points=np.asarray(station_xy,dtype=f32)
    guess=np.asarray(first_guess,dtype=f32); lu=np.asarray(landuse)
    if values.ndim!=2 or points.shape!=(values.shape[1],2):
        raise ValueError('station values must be (level,station) with matching XY coordinates')
    if guess.ndim!=3 or lu.shape!=guess.shape[1:]: raise ValueError('first guess and landuse shapes do not match')
    if guess.shape[0]!=values.shape[0]: raise ValueError('station and first-guess level counts differ')
    nz,ny,nx=guess.shape
    counts=np.broadcast_to(np.asarray(stations_per_level,dtype=int),(nz,))
    if np.any(counts<1): raise ValueError('stations_per_level/NINTRP must be positive')
    wo=np.broadcast_to(np.asarray(observation_weight,dtype=f32),guess.shape)
    background_grid=(None if background_weight is None else
                     np.broadcast_to(np.asarray(background_weight,dtype=f32),guess.shape))
    x=np.asarray((np.arange(nx,dtype=f32)+f32(.5))*f32(dx_km),dtype=f32)
    y=np.asarray((np.arange(ny,dtype=f32)+f32(.5))*f32(dy_km),dtype=f32)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    water=((lu>=water_range[0])&(lu<=water_range[1])).ravel(); output=np.empty_like(guess)
    # INTER2 tests STDAT1 .LT. EDITL; COMP defines EDITL as 900.
    valid=values<f32(900.)
    barrier_array=np.asarray(barriers,dtype=float)
    if barrier_array.size==0: barrier_array=np.empty((0,4))
    if barrier_array.ndim!=2 or barrier_array.shape[1]!=4:
        raise ValueError('barriers must contain (x_begin,y_begin,x_end,y_end) rows')
    if vary_radius and background_grid is None and np.any(~valid.any(axis=1)):
        raise ValueError('INTER2 has a level with no valid station data')
    for start in range(0,targets.shape[0],block_size):
        stop=min(start+block_size,targets.shape[0])
        delta=np.asarray(targets[start:stop,None,:]-points[None,:,:],dtype=f32)
        distance=np.sqrt(np.asarray(delta[:,:,0]**2+delta[:,:,1]**2,dtype=f32))
        safe=np.maximum(distance,f32(rmin_km))
        blocked=_barrier_blocked(targets[start:stop],points,barrier_array)
        for level in range(nz):
            if not valid[level].any() and background_grid is not None:
                output[level].ravel()[start:stop]=guess[level].ravel()[start:stop]
                continue
            radius=np.where(water[start:stop],f32(rmax_water_km),
                            f32(rmax_surface_km if level==0 else rmax_upper_km))
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
                                  np.asarray(selected_distance**2,dtype=f32),dtype=f32)
            obs_factor=np.where(selected,obs_factor,f32(0.)).astype(f32)
            numerator=np.zeros(stop-start,dtype=f32)
            denominator=np.zeros(stop-start,dtype=f32)
            for station in range(order.shape[1]):
                numerator=np.asarray(numerator+selected_values[:,station]*obs_factor[:,station],dtype=f32)
                denominator=np.asarray(denominator+obs_factor[:,station],dtype=f32)
            scale=f32(surface_scale_km if level==0 else upper_scale_km)
            background=(f32(f32(1.)/f32(scale*scale)) if background_grid is None else
                        background_grid[level].ravel()[start:stop])
            output[level].ravel()[start:stop]=np.asarray(
                (numerator+guess[level].ravel()[start:stop]*background)/
                (denominator+background),dtype=f32)
    return output


def wind1_initial_guess(station_u,station_v,station_xy,surface_station_count,
                        nx,ny,dx_km,dy_km,*,bias=0.,extrapolate_calm=True,
                        barriers=(),barrier_top_level=0,block_size=32768):
    """Vectorized WIND1 all-station first-guess interpolation.

    Surface (including SEA) profiles precede upper-air profiles. Distances use
    the legacy one-grid-cell floor before vertically varying group bias and
    per-level missing/calm/barrier removal.
    """
    f32=np.float32
    u=np.asarray(station_u,dtype=f32); v=np.asarray(station_v,dtype=f32)
    points=np.asarray(station_xy,dtype=f32)
    if u.ndim!=2 or v.shape!=u.shape or points.shape!=(u.shape[1],2):
        raise ValueError('WIND1 station profiles and coordinates are inconsistent')
    nz,nstation=u.shape; nsurface=int(surface_station_count)
    if not 0<=nsurface<=nstation: raise ValueError('invalid WIND1 surface-station count')
    if dx_km!=dy_km: raise ValueError('WIND1 requires CALMET square grid cells')
    layer_bias=np.broadcast_to(np.asarray(bias,dtype=f32),(nz,))
    if np.any((layer_bias < f32(-1.))|(layer_bias > f32(1.))):
        raise ValueError('WIND1 BIAS must lie in [-1,1]')
    barrier_array=np.asarray(barriers,dtype=float)
    if barrier_array.size==0: barrier_array=np.empty((0,4))
    if barrier_array.ndim!=2 or barrier_array.shape[1]!=4:
        raise ValueError('barriers must contain (x_begin,y_begin,x_end,y_end) rows')
    cell_x=np.asarray(np.arange(nx,dtype=f32)+f32(.5),dtype=f32)
    cell_y=np.asarray(np.arange(ny,dtype=f32)+f32(.5),dtype=f32)
    grid_x,grid_y=np.meshgrid(cell_x,cell_y)
    grid_targets=np.column_stack((grid_x.ravel(),grid_y.ravel()))
    x=np.asarray(cell_x*f32(dx_km),dtype=f32)
    y=np.asarray(cell_y*f32(dy_km),dtype=f32)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    # WIND1 stores station coordinates and DGRID in metres, forms DGRIDI
    # first, and only then converts each coordinate to grid-cell units.
    points_m=np.asarray(points*f32(1000.),dtype=f32)
    dgrid_m=np.asarray(f32(dx_km)*f32(1000.),dtype=f32)
    dgridi=np.asarray(f32(1.)/dgrid_m,dtype=f32)
    normalized_points=np.asarray(points_m*dgridi,dtype=f32)
    output_u=np.empty((nz,ny*nx),dtype=f32); output_v=np.empty_like(output_u)
    # WIND1 operates on internal US/VS arrays: COMP sets EDITL=900 and
    # every source check uses .GT. EDITL, not the external 9998/9999 code.
    surface_valid=(u[0,:nsurface]<=f32(900.))&(v[0,:nsurface]<=f32(900.))
    valid=(u<=f32(900.))&(v<=f32(900.))
    calm=np.sqrt(np.asarray(u[0,:nsurface]**2+v[0,:nsurface]**2,dtype=f32))<f32(1.e-4)
    def ordered_sum(items):
        total=np.zeros(items.shape[0],dtype=f32)
        for station in range(items.shape[1]):
            total=np.asarray(total+items[:,station],dtype=f32)
        return total
    def discard_in_station_order(weights,remove):
        """Apply WIND1's order-dependent w/(1-w_removed) updates."""
        for station in range(nstation):
            rows=remove[:,station]&(weights[:,station]!=f32(0.))
            if not rows.any(): continue
            removed_weight=weights[rows,station].copy()
            if np.any(removed_weight==f32(1.)):
                raise ValueError('WIND1 has no valid unblocked observation')
            denominator=np.asarray(f32(1.)-removed_weight,dtype=f32)
            weights[rows]=np.asarray(weights[rows]/denominator[:,None],dtype=f32)
            weights[rows,station]=f32(0.)
    for start in range(0,targets.shape[0],block_size):
        stop=min(start+block_size,targets.shape[0]); local=targets[start:stop]
        normalized_local=grid_targets[start:stop]
        delta=np.asarray(normalized_points[None,:,:]-normalized_local[:,None,:],dtype=f32)
        distance2=np.asarray(delta[:,:,0]**2+delta[:,:,1]**2,dtype=f32)
        base=np.asarray(f32(1.)/np.maximum(distance2,f32(1.)),dtype=f32)
        base[:,:nsurface]*=surface_valid[None,:]
        total=ordered_sum(base)[:,None]
        if np.any(total<=0.): raise ValueError('WIND1 has no station with valid surface wind')
        base=np.asarray(base/total,dtype=f32)
        surface_sum=ordered_sum(base[:,:nsurface])
        upper_sum=ordered_sum(base[:,nsurface:])
        blocked=_barrier_blocked(local,points,barrier_array)
        for level in range(nz):
            weights=base.copy(); value=layer_bias[level]
            if value>=0.:
                weights[:,:nsurface]*=f32(1.)-value
                desired=np.asarray(value+(f32(1.)-value)*upper_sum,dtype=f32)
                if nsurface<nstation:
                    weights[:,nsurface:]*=np.divide(desired,upper_sum,
                        out=np.zeros_like(desired),where=upper_sum>0.)[:,None]
                elif value>0.: raise ValueError('positive WIND1 BIAS requires upper-air stations')
            else:
                desired=np.asarray(-value+(f32(1.)+value)*surface_sum,dtype=f32)
                if nsurface:
                    weights[:,:nsurface]*=np.divide(desired,surface_sum,
                        out=np.zeros_like(desired),where=surface_sum>0.)[:,None]
                weights[:,nsurface:]*=f32(1.)+value
            if not extrapolate_calm and level>0 and nsurface:
                calm_remove=np.zeros_like(weights,dtype=bool)
                calm_remove[:,:nsurface]=calm[None,:]&surface_valid[None,:]
                # Preserve WIND1's special near-unit calm branch, which
                # rebuilds only upper-air weights from RSQWT.
                for station in range(nsurface):
                    rows=calm_remove[:,station]&(weights[:,station]>=f32(.9999))
                    if rows.any():
                        denominator=np.asarray(f32(1.)-base[rows,station],dtype=f32)
                        if np.any(denominator==f32(0.)):
                            raise ValueError('WIND1 calm station leaves no upper-air observation')
                        weights[rows,nsurface:]=np.asarray(
                            base[rows,nsurface:]/denominator[:,None],dtype=f32)
                        weights[rows,station]=f32(0.)
                        calm_remove[rows,station]=False
                discard_in_station_order(weights,calm_remove)
            if nsurface<nstation and value>f32(-.9999):
                missing=np.zeros_like(weights,dtype=bool)
                missing[:,nsurface:]=~valid[level,nsurface:][None,:]
                discard_in_station_order(weights,missing)
            if level<barrier_top_level:
                discard_in_station_order(weights,blocked)
            output_u[level,start:stop]=ordered_sum(
                np.asarray(weights*u[level][None,:],dtype=f32))
            output_v[level,start:stop]=ordered_sum(
                np.asarray(weights*v[level][None,:],dtype=f32))
    return output_u.reshape(nz,ny,nx),output_v.reshape(nz,ny,nx)


def _barrier_blocked(targets,stations,barriers):
    """Vectorized BARIER result for target × station pairs."""
    f32=np.float32
    target=np.asarray(targets,dtype=f32); station=np.asarray(stations,dtype=f32)
    blocked=np.zeros((target.shape[0],station.shape[0]),dtype=bool)
    for xb,yb,xe,ye in np.asarray(barriers,dtype=f32):
        dx=f32(xb-xe); slope=f32(9.9e9) if dx==f32(0.) else f32(f32(yb-ye)/dx)
        intercept=f32(yb-f32(slope*xb))
        d_station=np.asarray(station[:,1]-slope*station[:,0]-intercept,dtype=f32)
        d_target=np.asarray(target[:,1]-slope*target[:,0]-intercept,dtype=f32)
        opposite=np.asarray(d_target[:,None]*d_station[None,:],dtype=f32)<f32(0.)
        a=np.column_stack((np.full(target.shape[0],xb,dtype=f32)-target[:,0],np.full(target.shape[0],yb,dtype=f32)-target[:,1]))
        b=np.column_stack((np.full(target.shape[0],xe,dtype=f32)-target[:,0],np.full(target.shape[0],ye,dtype=f32)-target[:,1]))
        c=station[None,:,:]-target[:,None,:]
        def cosine(left,right):
            numerator=np.asarray(left[...,0]*right[...,0]+left[...,1]*right[...,1],dtype=f32)
            left2=np.asarray(left[...,0]**2+left[...,1]**2,dtype=f32)
            right2=np.asarray(right[...,0]**2+right[...,1]**2,dtype=f32)
            denominator=np.sqrt(np.asarray(left2*right2,dtype=f32))
            return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator!=f32(0.))
        cos_ab=cosine(a,b)
        cos_ac=cosine(a[:,None,:],c)
        cos_bc=cosine(b[:,None,:],c)
        blocked |= opposite&(cos_ac>=cos_ab[:,None])&(cos_bc>=cos_ab[:,None])
    return blocked


def barrier_blocked_pairs(targets, stations, barriers):
    """Return BARIER flags for target-specific candidate points.

    ``stations`` has shape ``(target, candidate, 2)``.  This is the layout
    used by CALMET's four prognostic neighbours and avoids constructing the
    prohibitively large target-by-full-prognostic-grid matrix.
    """
    f32=np.float32
    target=np.asarray(targets,dtype=f32); station=np.asarray(stations,dtype=f32)
    if target.ndim!=2 or target.shape[1]!=2:
        raise ValueError('targets must have shape (target, 2)')
    if station.ndim!=3 or station.shape[0]!=target.shape[0] or station.shape[2]!=2:
        raise ValueError('stations must have shape (target, candidate, 2)')
    blocked=np.zeros(station.shape[:2],dtype=bool)
    for xb,yb,xe,ye in np.asarray(barriers,dtype=f32):
        dx=f32(xb-xe); slope=f32(9.9e9) if dx==f32(0.) else f32(f32(yb-ye)/dx)
        intercept=f32(yb-f32(slope*xb))
        d_station=np.asarray(station[...,1]-slope*station[...,0]-intercept,dtype=f32)
        d_target=np.asarray(target[:,1]-slope*target[:,0]-intercept,dtype=f32)
        opposite=np.asarray(d_target[:,None]*d_station,dtype=f32)<f32(0.)
        a=np.column_stack((np.full(target.shape[0],xb,dtype=f32)-target[:,0],
                           np.full(target.shape[0],yb,dtype=f32)-target[:,1]))
        b=np.column_stack((np.full(target.shape[0],xe,dtype=f32)-target[:,0],
                           np.full(target.shape[0],ye,dtype=f32)-target[:,1]))
        c=station-target[:,None,:]
        def cosine(left,right):
            numerator=np.asarray(left[...,0]*right[...,0]+left[...,1]*right[...,1],dtype=f32)
            left2=np.asarray(left[...,0]**2+left[...,1]**2,dtype=f32)
            right2=np.asarray(right[...,0]**2+right[...,1]**2,dtype=f32)
            denominator=np.sqrt(np.asarray(left2*right2,dtype=f32))
            return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator!=f32(0.))
        cos_ab=cosine(a,b)
        cos_ac=cosine(a[:,None,:],c)
        cos_bc=cosine(b[:,None,:],c)
        blocked |= opposite&(cos_ac>=cos_ab[:,None])&(cos_bc>=cos_ab[:,None])
    return blocked
