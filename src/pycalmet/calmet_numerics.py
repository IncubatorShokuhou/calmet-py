"""Small, independently testable numerical kernels for the CALMET model.

Array axes follow the Python convention ``(..., y, x)``.  These routines are
kept free of model-global state so the main solver can compose them directly.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np


# Physical fields and model state use single precision throughout the modern
# solver.  Double precision is reserved for coordinates, reductions, and
# nonlinear convergence work where single-precision cancellation can change
# a grid-cell classification or prevent convergence.
MODEL_DTYPE = np.float32
ACCUMULATION_DTYPE = np.float64
COORDINATE_DTYPE = np.float64


def _model_array(value):
    """Return a model-state array at the public numerical boundary."""
    return np.asarray(value, dtype=MODEL_DTYPE)


def _psiud(z_over_l,*,dtype=MODEL_DTYPE):
    """CALMET's COARE/Dyer wind stability function."""
    return psi_momentum_dyer(z_over_l,dtype=dtype)


def _psiuc(height,monin_obukhov,fit=16,*,dtype=MODEL_DTYPE):
    """PSIUC stability function used when ``IPSIFCN=1``."""
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
    return output


def similarity_wind_profiles(u,v,anemometer_height_m,z_centers_m,z_faces_m,
        roughness_m,monin_obukhov_m,mixing_height_m,station_latitude,*,
        ipsifcn=0,minimum_mixing_height_m=50.):
    """Vectorized SIMILT station profiles in the model's default precision."""
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
        cloud_water_g_kg,height_agl_m,z_faces_m,threshold_g_kg=.005):
    """Vectorized Z2FACE/PRFVAR mapping used by CALMET.AUX output."""
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
    return tuple(_model_array(value) for value in (
        mapped,np.where(reported,average,0.),np.where(reported,bottom,0.),
        np.where(reported,top,0.)))

def air_density(pressure_mb, temperature_k, nearest_station):
    """AIRDen: ideal-gas density using the nearest surface station."""
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    nearest=np.asarray(nearest_station,dtype=np.intp)
    return density_from_pressure_temperature(pressure[nearest],temperature_k)

def density_from_pressure_temperature(pressure_mb,temperature_k):
    """AIRDEN ideal-gas density expression."""
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    return _model_array(.3484321*pressure/temperature)

def coriolis_parameter(latitude_degrees, configured_value=999.):
    """MICROI absolute Coriolis field, including its equatorial floor."""
    latitude=np.asarray(latitude_degrees,dtype=np.float64)
    if configured_value>998.:
        omega=2.*np.pi/86400.
        angle=np.deg2rad(latitude)
        value=2.*omega*np.sin(angle)
    else:
        value=np.full_like(latitude,np.float64(configured_value))
    return _model_array(np.maximum(np.abs(value),np.float64(.25e-4)))

def inverse_distance_squared(values, point_x, point_y, sample_x, sample_y, *, ignore_zero=False):
    """R2INTERP/CINTERP, vectorised over any leading target dimensions."""
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


def lake_breeze_adjust(u, v, station_u, station_v, station_xy_m,
                       station_ids, regions, dx_m, dy_m):
    """LLBREEZ/INTERB lowest-layer interpolation by coast distance.

    Region tuples contain ``XG1,XG2,YG1,YG2`` grid-line bounds, two
    coastline endpoints in relative metres, and the ordered station IDs.
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
        selected=np.asarray(selected,dtype=np.int64)
        if selected.size==0:
            raise ValueError('lake-breeze region must contain at least one station')
        lookup={}
        for index,value in enumerate(ids): lookup.setdefault(int(value),index)
        try: indices=np.asarray([lookup[int(value)] for value in selected],dtype=np.intp)
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
            def interpolate(values,valid):
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

def wind_speed_direction(u,v):
    """RTHETA meteorological speed and direction (calm direction is zero)."""
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    squared=u*u+v*v
    speed=np.sqrt(squared)
    direction=np.fmod(270.-np.rad2deg(np.arctan2(v,u)),360.)
    direction=np.where(squared<1.e-10,0.,
                       np.where(direction==0.,360.,direction))
    return _model_array(speed),_model_array(direction)

def cloud_fraction(relative_humidity_percent):
    """CLOUD3 Teixeira diagnostic cloud fraction."""
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    rhd=rh/100.; gap=1.-rhd
    result=np.ones_like(rhd)
    active=rh<99.
    root=np.sqrt(1.+100.*gap[active])
    result[active]=.02*(-1.+root)/gap[active]
    return result

def cloud_cover_teixeira_850(relative_humidity_percent,pressure_mb):
    """CLOUD3 cover using the last prognostic level at or above 850 mb."""
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
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
    rh=np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    height=np.asarray(height_m,dtype=MODEL_DTYPE)
    if rh.shape!=pressure.shape or rh.shape!=height.shape or rh.ndim!=3: raise ValueError('cloud profiles must share shape (z,y,x)')
    nz=rh.shape[0]; levels=np.arange(nz)[:,None,None]
    def first_below(threshold):
        mask=pressure<threshold
        return np.where(mask.any(axis=0),mask.argmax(axis=0),nz)
    klo,kmid,khi=first_below(970.),first_below(800.),first_below(450.)
    def band_max(start,stop=None):
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

def lapse_rate(temperature_k, previous_convective_height_m, z_centers_m, height_m=200., *, spatial=True):
    """CGAMMA2 prognostic temperature lapse rate (K/m)."""
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

def froude_adjustment(u, v, terrain_m, maximum_terrain_m, gamma_k_per_m,
                      temperature_k, z_centers_m, critical_froude,
                      dx_m, dy_m):
    """FRADJ terrain-blocking rotation with wind speed conserved."""
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

def profile_temperature_layer(temperature_k, height_m, bottom_m, depth_m=200., *, minimum_lapse=.001):
    """MIXDT2 temperatures and potential-temperature lapse above ``bottom``.

    Profiles use ``(z, y, x)`` axes and must bracket both requested heights.
    Unlike a generic interpolator this reproduces MIXDT2's linear interpolation
    and its dry-adiabatic conversion to potential-temperature lapse.
    """
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    height=np.asarray(height_m,dtype=MODEL_DTYPE)
    bottom=np.asarray(bottom_m,dtype=MODEL_DTYPE)
    if temperature.shape!=height.shape or temperature.ndim!=3 or temperature.shape[1:]!=bottom.shape: raise ValueError('profiles must be (z,y,x) and bottom must match their grid')
    flat_t=temperature.reshape(temperature.shape[0],-1); flat_z=height.reshape(height.shape[0],-1); columns=np.arange(flat_t.shape[1])
    def interpolate(target):
        target=np.asarray(target,dtype=MODEL_DTYPE).ravel()
        upper=np.sum(flat_z<=target[None,:],axis=0)
        if np.any(upper==0) or np.any(upper>=flat_z.shape[0]): raise ValueError('temperature profile does not bracket mixing-height layer')
        lower=upper-1; zl=flat_z[lower,columns]; zu=flat_z[upper,columns]
        if np.any(zu<=zl): raise ValueError('temperature profile heights must increase strictly')
        value=(flat_t[upper,columns]-
               (flat_t[upper,columns]-flat_t[lower,columns])*(zu-target)/(zu-zl))
        return value.reshape(bottom.shape)
    lower=interpolate(bottom); upper=interpolate(bottom+depth_m)
    lapse=np.maximum((upper-lower)/depth_m+.0098,minimum_lapse)
    return tuple(_model_array(value) for value in (lower,upper,lapse))

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
    old=np.asarray(previous_convective_height_m,dtype=MODEL_DTYPE)
    jump=np.asarray(temperature_jump_k,dtype=MODEL_DTYPE)
    ustar=np.asarray(friction_velocity,dtype=MODEL_DTYPE)
    length=np.asarray(monin_obukhov_m,dtype=MODEL_DTYPE)
    qh=np.asarray(sensible_heat_flux,dtype=MODEL_DTYPE)
    rho=np.asarray(density,dtype=MODEL_DTYPE); f=np.asarray(coriolis,dtype=MODEL_DTYPE)
    tht,thtp,gamma=profile_temperature_layer(temperature_profile_k,profile_height_m,old,depth_m,minimum_lapse=minimum_lapse)
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
    if land_mask is not None:
        land=np.asarray(land_mask,dtype=bool); total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan); next_jump=np.where(land,next_jump,np.nan)
    return tuple(_model_array(value) for value in (total,convective,next_jump))


def convective_height_maul_carson(previous_convective_height_m,temperature_jump_k,
                                  buoyancy_flux_k_m_s,potential_lapse_k_m,*,
                                  timestep_s=3600.,threshold_flux_per_m=0.,
                                  maximum_height_m=2500.,minimum_lapse=.001,conste=.15,
                                  relaxation=True,relaxation_time_s=800.):
    """CALMET MIXHMC from an already computed surface buoyancy flux.

    WATER supplies ``WT`` directly, unlike the land wrapper which derives it
    from sensible heat flux and density.  Arrays are evaluated simultaneously.
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
        previous_convective_height_m,buoyancy_flux_k_m_s,potential_lapse_k_m,
        friction_velocity,monin_obukhov_m,surface_temperature_k,*,
        timestep_s=3600.,minimum_height_m=50.,maximum_height_m=2500.,
        threshold_flux_per_m=0.,minimum_lapse=.001,relaxation=True,
        relaxation_time_s=800.):
    """Vectorised CALMET ``MIXHBG`` driven by a surface buoyancy flux."""
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

    def fbg(zeta,zetap):
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

    The nonlinear Batchvarova--Gryning equation uses a false-position secant
    solver.  Its ten iterations operate on whole arrays; no horizontal
    Python loop is used.  The temperature jump is unchanged because MIXHBG,
    unlike MIXHMC, does not update CALMET's ``DPTT`` state.
    """
    old=np.asarray(previous_convective_height_m,dtype=np.float64)
    jump=np.asarray(temperature_jump_k,dtype=np.float64)
    ustar=np.asarray(friction_velocity,dtype=np.float64)
    length=np.asarray(monin_obukhov_m,dtype=np.float64)
    qh=np.asarray(sensible_heat_flux,dtype=np.float64)
    rho_field=np.asarray(density,dtype=np.float64)
    f=np.asarray(coriolis,dtype=np.float64)
    tk=np.asarray(surface_temperature_k,dtype=np.float64)
    h0=np.maximum(old,np.float64(minimum_height_m))
    tht,thtp,gamma=profile_temperature_layer(temperature_profile_k,profile_height_m,h0,depth_m,minimum_lapse=minimum_lapse)
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
    if land_mask is not None:
        land=np.asarray(land_mask,dtype=bool)
        total=np.where(land,total,np.nan); convective=np.where(land,convective,np.nan)
    return tuple(_model_array(value) for value in (total,convective,jump))

def hosker_roughness(speed_m_s):
    """Hosker wind-dependent water roughness with floating-point power semantics."""
    speed=np.asarray(speed_m_s,dtype=MODEL_DTYPE)
    return _model_array(np.maximum(2.e-6,2.e-6*np.power(speed,2.5)))


def neutral_water_parameters(u10, v10, coriolis, *, constw=.16,
                             minimum_height_m=50., maximum_height_m=2500.):
    """WATER's ``NOWSTA=0`` neutral overwater path used by NOOBS runs.

    It applies the Hosker wind-dependent roughness, neutral DeltaT friction
    velocity, and the OCD mechanical mixing height.  With no SEA.DAT station
    the legacy routine has zero buoyancy flux and therefore zero convective
    mixing height even when ``IMIXH=1``.
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


def neutral_water_station_profiles(u,v,anemometer_height_m,current_mixing_height_m,
        coriolis,z_centers_m,z_faces_m,station_latitude,*,constw=.16,
        minimum_height_m=50.,maximum_height_m=2500.,ipsifcn=0):
    """WATER2 neutral/missing-sea-data station path followed by SIMILT."""
    u=np.asarray(u,dtype=MODEL_DTYPE); v=np.asarray(v,dtype=MODEL_DTYPE)
    values=tuple(np.asarray(value,dtype=MODEL_DTYPE) for value in (
        anemometer_height_m,current_mixing_height_m,coriolis,station_latitude))
    if u.ndim!=1 or v.shape!=u.shape or any(value.shape!=u.shape for value in values):
        raise ValueError('WATER2 station inputs must be matching one-dimensional arrays')
    anemometer,current_mixing,coriolis_value,latitude=values
    centers=np.asarray(z_centers_m,dtype=MODEL_DTYPE)
    output_u=np.full((centers.size,u.size),9999.,dtype=MODEL_DTYPE)
    output_v=np.full_like(output_u,9999.); output_u[0]=u; output_v[0]=v
    state=[np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(4)]
    speed=np.sqrt(u*u+v*v)
    active=(u<=990.)&(v<=990.)&(speed>=.0001)
    if not active.any(): return (output_u,output_v,*state)
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
    return (output_u,output_v,*state)


def water_mechanical_height(friction_velocity,coriolis,*,constw=.16,
                            minimum_height_m=50.,maximum_height_m=2500.):
    """WATER's floating-point ``ZIMECH=CONSTW*USTAR/FCORI`` calculation."""
    friction=np.asarray(friction_velocity,dtype=MODEL_DTYPE)
    coriolis=np.asarray(coriolis,dtype=MODEL_DTYPE)
    return _model_array(np.clip(constw*friction/coriolis,
                                minimum_height_m,maximum_height_m))


def saturation_vapor_pressure_lowe(temperature_c):
    """CALMET ``ESAT`` Lowe polynomial in model precision."""
    value=np.asarray(temperature_c,dtype=MODEL_DTYPE)
    coefficients=np.asarray(
        (6.107799961,.4436518521,.01428945805,.0002650648471,
         .000003031240396,.00000002034080948,.00000000006136820929),
        dtype=MODEL_DTYPE)
    result=np.full_like(value,coefficients[-1],dtype=MODEL_DTYPE)
    for coefficient in reversed(coefficients[:-1]):
        result=coefficient+value*result
    return _model_array(result)


def overwater_delta_temperature_state(air_temperature_k,sea_temperature_k,
                                      relative_humidity_percent,
                                      measurement_height_m):
    """Prepare WATER's virtual-potential-temperature inputs to ``DELTAT``.

    The returned tuple is ``(theta, delta_theta_z, delta_theta_10m,
    theta_10m)``.
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
    return tuple(_model_array(value) for value in
                 (theta,difference,difference_10m,theta_10m))


def delta_temperature_flux(wind_speed_10m,measurement_height_m,
                           measurement_minus_surface_virtual_potential_k,
                           ten_m_minus_surface_virtual_potential_k,
                           virtual_potential_temperature_10m_k,
                           measurement_virtual_potential_temperature_k,
                           roughness_m):
    """Vectorized CALMET ``DELTAT`` with double-precision nonlinear iteration."""
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
    return tuple(_model_array(value) for value in
                 (length,ustar,buoyancy_flux_w_m2))


def delta_temperature_water_station_profiles(u,v,anemometer_height_m,
        air_temperature_k,air_minus_sea_temperature_k,relative_humidity_percent,
        observed_mixing_height_m,current_mixing_height_m,coriolis,
        potential_lapse_k_m,previous_temperature_jump_k,z_centers_m,z_faces_m,
        station_latitude,*,mixing_method=1,constw=.16,minimum_height_m=50.,
        maximum_height_m=2500.,threshold_flux_per_m=.05,ipsifcn=0,
        temperature_measurement_height_m=None,sea_temperature_k=None,**mixing_options):
    """WATER2 station DeltaT branch, with neutral fallback, through SIMILT."""
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
    state=[np.array(value,copy=True) for value in base[2:]]
    # DeltaT-specific state: convective height, jump, and buoyancy heat flux.
    state.extend(np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(3))
    if not active.any(): return (output_u,output_v,*state)
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
    state=[np.array(value,copy=True) for value in base[2:]]
    state.extend(np.full(u.shape,np.nan,dtype=MODEL_DTYPE) for _ in range(3))
    if not active.any(): return (output_u,output_v,*state,warm_layer_state)
    selected_air=np.where(air[active]<990.,air[active],288.7)
    selected_rh=np.where(rh[active]<990.,rh[active],100.)
    sea=(selected_air-delta[active]
         if sea_temperature_k is None else
         np.broadcast_to(np.asarray(sea_temperature_k,dtype=MODEL_DTYPE),u.shape)[active])
    sea=np.maximum(sea,270.15)
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
            merged=[]
            for name in WarmLayerState.__dataclass_fields__:
                values=np.array(getattr(warm_layer_state,name),copy=True)
                values[active]=getattr(selected_warm_state,name)
                merged.append(values)
            next_warm_state=WarmLayerState(*merged)
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
    return (output_u,output_v,*state,next_warm_state)


def upwind_mixing_average(u, v, mixing_height, convective_height, dx_m=None,
                          dy_m=None, *, grid_spacing_m=None,
                          minimum_radius_cells=10, half_angle_degrees=30.,
                          timestep_s=3600.):
    """AVEMIX upwind-cone spatial averaging of ``ZI`` and ``ZICONV``.

    The target-cell loop is retained because every target has a differently
    oriented and sized search window; all candidate-cell calculations inside
    that window are NumPy-vectorised.  Trajectories, cone angles, and distance
    weights are evaluated in physical metres, which supports rectangular grid
    cells.  Omitting ``dy_m`` retains the former square-grid calling form;
    ``grid_spacing_m`` remains an alias for its old keyword spelling.
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
    dx_m=float(dx_m); dy_m=float(dy_m); timestep_s=float(timestep_s)
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
    def window_geometry(left,right,down,up):
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


def mesopac_layer_average(u, v, mixing_height_m, z_faces_m, *,
                          upper_top_m=3000., minimum_upper_depth_m=300.):
    """Vectorized PACAVE lower- and upper-layer wind averages.

    The lower layer extends from the ground to ``max(ZI, 10 m)``.  The
    upper layer extends from ZI to the greater of 3000 m or ZI + 300 m,
    matching the fixed arguments used by CALMET's OUTPC path.
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

    def average(values,depth):
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
    return tuple(_model_array(value) for value in
                 (lower_u,lower_v,upper_u,upper_v))

def prognostic_temperature_3d(prognostic_temperature_k, surface_temperature_k,
                              convective_height_m, z_faces_m):
    """Blend prognostic temperatures through the convective-layer boundary.

    The lowest CALMET layer remains anchored to the final SURFVAR/AVETMP
    temperature.  Each layer aloft receives a dry-adiabatic temperature in
    proportion to the layer thickness below the convective-layer top.  A
    layer wholly below the top is therefore fully mixed, a layer wholly above
    it retains its prognostic value, and a layer intersected by the top varies
    continuously as that top moves through the layer.
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

def pgt_stability(u10, v10, solar_elevation_sine, cloud_fraction_field,
                  ceiling_height_m, landuse, monin_obukhov_m, water_range=(50,55)):
    """PGTSTB Turner land classes plus WATER's Monin-Obukhov classes."""
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

def convective_velocity_scale(convective_height_m, sensible_heat_flux, temperature_k,
                              density, landuse, water_range=(50,55)):
    """WATER/WSTARR convective velocity scale; zero for non-positive flux."""
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

def solar_sine(julian_day, base_timezone, latitude, longitude_east):
    """SOLAR: 26 half-hour solar-elevation sine fields."""
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

def terrain_gradients(terrain_m, dx_m, dy_m):
    """CALMET one-sided-at-boundary, centred-interior terrain gradients."""
    h=np.asarray(terrain_m,dtype=MODEL_DTYPE)
    west=np.concatenate((h[:,:1],h[:,:-1]),axis=1); east=np.concatenate((h[:,1:],h[:,-1:]),axis=1)
    south=np.concatenate((h[:1,:],h[:-1,:]),axis=0); north=np.concatenate((h[1:,:],h[-1:,:]),axis=0)
    return (_model_array((east-west)*(.5/dx_m)),
            _model_array((north-south)*(.5/dy_m)))

def terrain_adjust_surface_wind(u, v, terrain_m, top_height_m, dx_m, dy_m,
                                *, maximum_iterations=75, tolerance=.02,
                                relaxation=1.5):
    """ADJUST terrain-potential correction for objective-analysis winds.

    CALMET applies this Poisson solve only to the first wind layer when
    ``IWFCOD=0``.  The four alternating Gauss-Seidel sweeps are deliberately
    retained because every sweep immediately reuses updated neighbours.
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

def terrain_extrema(terrain_m, radius_km, dx_m, dy_m):
    """Return local terrain extrema and their nearest physical distances.

    The circular search and all comparisons use double precision.  When an
    extremum occurs at more than one cell, its nearest occurrence is used;
    completely equidistant occurrences are resolved by row-major source
    index.  This makes the result independent of the order in which offsets
    are visited.  Returned model fields remain single precision.
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
    return tuple(_model_array(value) for value in (
        highest,lowest,distance_high,distance_low))

def topographic_vertical_velocity(u,v,terrain_m,z_faces_m,gamma_k_per_m,temperature_k,alpha,dx_m,dy_m):
    """TOPOF2 terrain-following vertical velocity; wind axes are (z,y,x)."""
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

def slope_flow(terrain_m,highest_m,lowest_m,distance_high_m,distance_low_m,sensible_heat_flux,temperature_k,landuse,z_faces_m,dx_m,dy_m,water_range=(50,55)):
    """SLOPE Mahrt/Horst-Doran shooting-flow parameterisation."""
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

def vertical_layer_average(values, heights_m, faces_m, *, missing=998.99):
    """VERTAV exact piecewise-linear layer mean for arbitrary profiles.

    The final axis is the source sounding level. All preceding axes identify
    independent profiles. ``heights_m`` may be one-dimensional or match the
    values array.
    """
    values=np.asarray(values,dtype=ACCUMULATION_DTYPE)
    heights=np.broadcast_to(np.asarray(heights_m,dtype=ACCUMULATION_DTYPE),values.shape)
    faces=np.asarray(faces_m,dtype=ACCUMULATION_DTYPE)
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
    return _model_array(np.stack(output,axis=-1))

def vertical_interpolate(values, heights_m, targets_m, *, missing=998.99):
    """FACET/INTP piecewise-linear interpolation with missing-level skips."""
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

def smooth_wind(u,v,passes):
    """Apply the in-place Gauss-Seidel wind-smoothing stencil."""
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

def divergence_and_vertical_velocity(u,v,z_faces_m,dx_m,dy_m,*,obrien=False,boundaries=None):
    """DIVCEL plus vertical integration and optional O'Brien correction."""
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

def minimize_divergence(u,v,w,z_faces_m,dx_m,dy_m,*,boundaries=None,
                        maximum_iterations=50,limit=5.e-6):
    """MINIM four-sweep horizontal mass-consistency adjustment."""
    work_dtype=ACCUMULATION_DTYPE
    u=np.array(u,dtype=work_dtype,copy=True); v=np.array(v,dtype=work_dtype,copy=True)
    w=np.asarray(w,dtype=work_dtype); faces=np.asarray(z_faces_m,dtype=work_dtype)
    if u.shape!=v.shape or w.shape!=(u.shape[0]+1,*u.shape[1:]):
        raise ValueError('MINIM wind arrays have inconsistent dimensions')
    if boundaries is None:
        boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    west,east,south,north=(np.array(value,dtype=work_dtype,copy=True) for value in boundaries)
    divergence=np.empty_like(u); iterations=np.zeros(u.shape[0],dtype=np.int32)

    def layer_divergence(k):
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

def heat_flux(solar_sine_field,cloud_fraction_field,temperature_k,landuse,albedo,
              bowen,soil_heat_flux,anthropogenic_heat,water_range=(50,55),*,
              turbidity_coefficient_1=990.,turbidity_coefficient_2=-30.,
              cloudiness_coefficient_1=-.75,cloudiness_coefficient_2=3.4,
              net_radiation_coefficient_1=5.31e-13,
              net_radiation_coefficient_2=60.,net_radiation_coefficient_3=.12):
    """HEATFX short-wave radiation and land sensible heat flux."""
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

def psi_momentum_dyer(z_over_l,*,dtype=MODEL_DTYPE):
    """Return the Dyer momentum stability correction.

    Model calls default to single precision.  Nonlinear convergence routines
    can request double precision without changing the returned model fields.
    """
    work_dtype=np.dtype(dtype).type
    zl=np.asarray(z_over_l,dtype=work_dtype)
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
    return result

def surface_similarity(u,v,roughness_m,height_m,density,temperature_k,
                       cloud_fraction_field,sensible_heat_flux,landuse,
                       water_range=(50,55),*,wind_measurement_height_m=None,
                       ipsifcn=0,minimum_stable_wind_m_s=.5):
    """ELUSTR land-cell friction velocity, heat flux and M-O length."""
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
    return tuple(_model_array(value) for value in (
        np.where(water,0.,ustar),np.where(water,0.,length),np.where(water,0.,qh)))


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
        result=mixing_height_maul_carson(*arguments,**options)
    elif abs(int(mixing_method))==2:
        if surface_temperature_k is None:
            raise ValueError('MIXHBG station profiles require surface temperature')
        result=mixing_height_batchvarova_gryning(
            *arguments,surface_temperature_k=shaped(surface_temperature_k),**options)
    else:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    return tuple(_model_array(value[0]) for value in result)


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
    background_grid=(None if background_weight is None else
                     np.broadcast_to(np.asarray(background_weight,dtype=work_dtype),guess.shape))
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


def wind1_initial_guess(station_u,station_v,station_xy,surface_station_count,
                        nx,ny,dx_km,dy_km,*,bias=0.,extrapolate_calm=True,
                        barriers=(),barrier_top_level=0,block_size=32768):
    """Vectorized WIND1 all-station first-guess interpolation.

    Surface (including SEA) profiles precede upper-air profiles. Distances use
    a one-grid-cell floor before vertically varying group bias and
    per-level missing/calm/barrier removal.
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
    def stable_sum(items):
        return np.sum(items,axis=1,dtype=ACCUMULATION_DTYPE)
    def discard_weights(weights,remove):
        """Remove unavailable observations and normalize the remaining weights."""
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


def _barrier_blocked(targets,stations,barriers):
    """Vectorized BARIER result for target × station pairs."""
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
        def cosine(left,right):
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


def barrier_blocked_pairs(targets, stations, barriers):
    """Return BARIER flags for target-specific candidate points.

    ``stations`` has shape ``(target, candidate, 2)``.  This is the layout
    used by CALMET's four prognostic neighbours and avoids constructing the
    prohibitively large target-by-full-prognostic-grid matrix.
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
        def cosine(left,right):
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
