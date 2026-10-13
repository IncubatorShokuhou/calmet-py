"""Reader for the fixed-column IWAQM ``MM4.DAT`` format."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path

import numpy as np


def _fields(line: str, widths, converters, label: str):
    values=[]; begin=0
    try:
        for width,convert in zip(widths,converters):
            values.append(convert(line[begin:begin+width]))
            begin+=width
    except ValueError as error:
        raise ValueError(f'invalid MM4.DAT {label} record: {line!r}') from error
    return values


def _year(value: int) -> int:
    """Apply the legacy YR4 two-digit-year expansion."""
    if value>=100: return value
    return value+2000 if value<50 else value+1900


@dataclass(frozen=True,slots=True)
class MM4Header:
    label: str
    start: datetime
    hours: int
    model_nx: int
    model_ny: int
    sigma_levels: np.ndarray
    top_pressure_mb: float
    i_start: int
    j_start: int
    latitude: np.ndarray
    longitude: np.ndarray
    terrain_height_m: np.ndarray
    landuse: np.ndarray


@dataclass(frozen=True,slots=True)
class MM4Record:
    timestamp: datetime
    sea_level_pressure_mb: np.ndarray
    precipitation_mm_h: np.ndarray
    snow_code: np.ndarray
    pressure_mb: np.ndarray
    height_agl_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity: np.ndarray
    wind_direction_deg: np.ndarray
    wind_speed_m_s: np.ndarray


@dataclass(frozen=True,slots=True)
class MM4Interpolated:
    u: np.ndarray
    v: np.ndarray
    temperature_k: np.ndarray
    relative_humidity: np.ndarray
    pressure_mb: np.ndarray
    profile_height_m: np.ndarray
    profile_temperature_k: np.ndarray
    surface_density: np.ndarray
    precipitation_mm_h: np.ndarray
    surface_pressure_mb: np.ndarray
    surface_relative_humidity: np.ndarray
    cloud_fraction: np.ndarray
    ceiling_height_m: np.ndarray
    precipitation_code: np.ndarray
    surface_temperature_k: np.ndarray
    thermal_state: object
    prognostic_observation_u: np.ndarray | None = None
    prognostic_observation_v: np.ndarray | None = None
    prognostic_observation_strength: np.ndarray | None = None


class MM4Data:
    """Load native formatted MM4 data as vectorized hourly arrays."""

    def __init__(self,path):
        self.path=Path(path)
        with self.path.open(encoding='ascii') as stream:
            self.header=self._read_header(stream)
            self.records=self._read_records(stream)
        if not self.records: raise ValueError('MM4.DAT contains no meteorological records')

    @staticmethod
    def _line(stream,label):
        line=stream.readline()
        if not line: raise ValueError(f'truncated MM4.DAT before {label}')
        return line.rstrip('\r\n')

    @classmethod
    def _read_header(cls,stream):
        label=cls._line(stream,'creation label')[:36]
        values=_fields(cls._line(stream,'grid header'),(2,2,2,2,4,4,4,4,6),
                       (int,int,int,int,int,int,int,int,float),'grid header')
        year,month,day,hour,hours,model_nx,model_ny,nzp,top=values
        i_start,j_start,nx,ny=_fields(cls._line(stream,'subdomain header'),(4,4,4,4),
                                     (int,int,int,int),'subdomain header')
        if min(hours,model_nx,model_ny,nzp,nx,ny)<=0:
            raise ValueError('MM4.DAT header dimensions and duration must be positive')
        sigma=np.array([float(cls._line(stream,f'sigma level {index+1}')[:6])
                        for index in range(nzp)],dtype=np.float32)
        latitude=np.empty((ny,nx),dtype=np.float32)
        longitude=np.empty((ny,nx),dtype=np.float32)
        terrain=np.empty((ny,nx),dtype=np.float32)
        landuse=np.empty((ny,nx),dtype=np.int32)
        for j in range(ny):
            for i in range(nx):
                values=_fields(cls._line(stream,'static grid'),(3,3,7,8,5,3),
                               (int,int,float,float,int,int),'static grid')
                ii,jj,latitude[j,i],longitude[j,i],terrain[j,i],landuse[j,i]=values
                if (ii,jj)!=(i_start+i,j_start+j):
                    raise ValueError(f'MM4.DAT static grid index {(ii,jj)} does not match '
                                     f'expected {(i_start+i,j_start+j)}')
        return MM4Header(label,datetime(_year(year),month,day,hour),hours,model_nx,
                         model_ny,sigma,top,i_start,j_start,latitude,longitude,
                         terrain,landuse)

    def _read_records(self,stream):
        header=self.header; ny,nx=header.latitude.shape; levels=8+header.sigma_levels.size
        records=[]
        while True:
            first=stream.readline()
            if not first: break
            surface=np.empty((ny,nx,3),dtype=np.float32)
            pressure=np.empty((levels,ny,nx),dtype=np.float32); height=np.empty_like(pressure)
            encoded_temperature=np.empty_like(pressure); depression=np.empty_like(pressure)
            direction=np.empty_like(pressure); speed=np.empty_like(pressure)
            timestamp=None
            for j in range(ny):
                for i in range(nx):
                    line=first.rstrip('\r\n') if i==0 and j==0 else self._line(stream,'surface data')
                    values=_fields(line,(2,2,2,2,3,3,7,5,2),
                                   (int,int,int,int,int,int,float,float,int),'surface data')
                    year,month,day,hour,ii,jj,pmsl,rain,snow=values
                    current=datetime(_year(year),month,day,hour)
                    if timestamp is None: timestamp=current
                    if current!=timestamp: raise ValueError('MM4.DAT grid cells have inconsistent timestamps')
                    expected=(header.i_start+i,header.j_start+j)
                    if (ii,jj)!=expected:
                        raise ValueError(f'MM4.DAT data grid index {(ii,jj)} does not match expected {expected}')
                    surface[j,i]=(pmsl,rain*10.,snow)
                    for level in range(levels):
                        values=_fields(self._line(stream,'vertical data'),(5,6,4,2,4,2),
                                       (float,float,int,int,float,float),'vertical data')
                        pressure[level,j,i],height[level,j,i],encoded_temperature[level,j,i],\
                            depression[level,j,i],direction[level,j,i],speed[level,j,i]=values
            negative=(encoded_temperature.astype(np.int64)%2)!=0
            f32=np.float32
            temperature_c=np.asarray(np.where(negative,-encoded_temperature,
                                                encoded_temperature)/f32(10.),dtype=f32)
            dew_depression=np.asarray(np.where(depression<f32(56.),depression/f32(10.),
                                                depression-f32(50.)),dtype=f32)
            dry_f=np.asarray(f32(1.8)*temperature_c+f32(32.),dtype=f32)
            dew_f=np.asarray(f32(1.8)*np.asarray(temperature_c-dew_depression,dtype=f32)+
                             f32(32.),dtype=f32)
            numerator=np.asarray(f32(173.)-f32(.1)*dry_f+dew_f,dtype=f32)
            denominator=np.asarray(f32(173.)+f32(.9)*dry_f,dtype=f32)
            relative_humidity=np.minimum(f32(100.),np.asarray(
                (numerator/denominator)**8*f32(100.),dtype=f32))
            records.append(MM4Record(timestamp,surface[...,0],surface[...,1],
                surface[...,2].astype(np.int32),pressure,height-header.terrain_height_m,
                np.asarray(temperature_c+f32(273.15),dtype=f32),relative_humidity,direction,
                np.asarray(speed*f32(.51444),dtype=f32)))
        return tuple(records)

    def at_time(self,timestamp):
        for record in self.records:
            if record.timestamp==timestamp: return record
        raise ValueError(f'MM4.DAT has no record for {timestamp}')

    @staticmethod
    def _sorted_above_ground_profiles(**fields):
        """Apply QCKSRT3 and RDMM4's literal duplicate/ground filtering.

        The legacy duplicate loop continues to ``NLEV-1`` after reducing
        ``NLEVOK``.  Consequently the stale final pair can remove one more
        effective level.  Repeating each column's final valid value gives a
        rectangular NumPy representation without changing persistence above
        the Fortran effective top.
        """
        height=np.asarray(fields['height'])
        order=np.argsort(height,axis=0,kind='stable')
        sorted_fields={name:np.take_along_axis(np.asarray(values),order,axis=0).copy()
                       for name,values in fields.items()}
        levels=height.shape[0]
        valid=np.full(height.shape[1:],levels,dtype=np.int32)
        z=sorted_fields['height']
        for level in range(levels-1):
            duplicate=z[level]==z[level+1]
            if not np.any(duplicate): continue
            for destination in range(level+1,levels-1):
                move=duplicate & (destination<valid-1)
                if not np.any(move): continue
                for values in sorted_fields.values():
                    values[destination][move]=values[destination+1][move]
            valid[duplicate]-=1
        # RDMM4 packs only Z >= -0.0001 into UP/VP/TP/ZP.  Do the same per
        # source column, then persist the last valid value through the unused
        # rectangular tail.
        level_index=np.arange(levels,dtype=np.int32)[:,None,None]
        keep=(level_index<valid[None]) & (z>=-0.0001)
        count=keep.sum(axis=0)
        if np.any(count==0):
            j,i=np.argwhere(count==0)[0]
            raise ValueError(f'MM4.DAT profile {(i,j)} has no above-ground level')
        # Kept indices are already height ordered.  Sorting a sentinel-filled
        # index cube packs them along axis 0; clipped ranks then repeat the
        # final effective level throughout the rectangular tail.
        source_index=np.sort(np.where(keep,level_index,levels),axis=0)
        rank=np.minimum(level_index,count[None]-1)
        source_index=np.take_along_axis(source_index,rank,axis=0)
        packed={name:np.take_along_axis(values,source_index,axis=0)
                for name,values in sorted_fields.items()}
        return packed

    def interpolate_record(self,record,grid,target_height_m,*,cloud_method=4,
                           precipitation_method=2,precipitation_radius_km=50.,
                           precipitation_cutoff=.01,base_timezone=0,
                           thermal_state=None,prognostic_observation_rmin_km=None,
                           prognostic_observation_barriers=(),
                           prognostic_observation_barrier_top_level=0,
                           prognostic_ocean_category=16):
        """Apply RDHD4/RDMM4 vertical and four-neighbour interpolation."""
        from .calmet_solver import (PrognosticMapper,SourceThermalState,
                                    _prognostic_wind_components,_stull_temperature,_vertical_centers,
                                    solar_at_step_midpoint)
        from .calmet_numerics import (_real4_libm,_real4_power,cloud_cover_mm5,cloud_fraction,
                                      density_from_pressure_temperature,
                                      saturation_vapor_pressure_lowe)
        from .projections import grid_convergence

        mapper=PrognosticMapper.build(self.header.latitude,self.header.longitude,grid,
                                       validate_domain=True)
        profiles=self._sorted_above_ground_profiles(
            height=record.height_agl_m,temperature=record.temperature_k,
            humidity=record.relative_humidity,pressure=record.pressure_mb,
            direction=record.wind_direction_deg,speed=record.wind_speed_m_s)
        height=profiles['height']; temperature=profiles['temperature']
        humidity=profiles['humidity']; pressure=profiles['pressure']
        direction=profiles['direction']; speed=profiles['speed']
        convergence=grid_convergence(grid,self.header.longitude,self.header.latitude)
        source_u,source_v=_prognostic_wind_components(
            speed,direction,convergence[None,:,:])
        centers=np.asarray(target_height_m,dtype=np.float32)
        if centers.ndim!=1 or centers.size<1:
            raise ValueError('target MM4 heights must be a non-empty vector')
        def wind_centers(field):
            base=_vertical_centers(field,height,centers)
            flat=np.asarray(field).reshape(field.shape[0],-1)
            zflat=height.reshape(height.shape[0],-1)
            columns=np.arange(flat.shape[1]); anchor=np.where(zflat[0]<1.,1,0)
            anchor=np.minimum(anchor,field.shape[0]-1)
            anchor_z=zflat[anchor,columns]; anchor_value=flat[anchor,columns]
            for level,center in enumerate(centers):
                below=center<anchor_z
                log_half=np.float32(_real4_libm('logf',np.float32(.5)))
                numerator=np.float32(_real4_libm('logf',center)-log_half)
                denominator=np.asarray(_real4_libm('logf',np.maximum(
                    anchor_z,np.float32(.500001)))-log_half,dtype=np.float32)
                factor=np.divide(numerator,denominator,
                    out=np.ones_like(anchor_z,dtype=np.float32),
                    where=anchor_z>np.float32(.500001))
                base[level]=np.where(below.reshape(base.shape[1:]),
                    (factor*anchor_value).reshape(base.shape[1:]),base[level])
            return base
        source_u_centers=wind_centers(source_u); source_v_centers=wind_centers(source_v)
        u=mapper.interpolate(source_u_centers); v=mapper.interpolate(source_v_centers)
        observation_u=observation_v=observation_strength=None
        if prognostic_observation_rmin_km is not None:
            observation_u,observation_strength=mapper.interpolate_observations(
                source_u_centers,prognostic_observation_rmin_km,
                barriers=prognostic_observation_barriers,
                barrier_top_level=prognostic_observation_barrier_top_level)
            observation_v,_=mapper.interpolate_observations(
                source_v_centers,prognostic_observation_rmin_km,
                barriers=prognostic_observation_barriers,
                barrier_top_level=prognostic_observation_barrier_top_level)
        flat_shape=height.shape[1:]; columns=np.arange(height[0].size)
        zflat=height.reshape(height.shape[0],-1)
        anchor=np.where(zflat[0]<1.,1,0); anchor=np.minimum(anchor,height.shape[0]-1)
        following=np.minimum(anchor+1,height.shape[0]-1)
        for level in range(2,height.shape[0]):
            duplicate=zflat[following,columns]==zflat[anchor,columns]
            following=np.where(duplicate,np.minimum(following+1,height.shape[0]-1),following)
        def gather(values,index): return np.asarray(values).reshape(values.shape[0],-1)[index,columns].reshape(flat_shape)
        anchor_z=gather(height,anchor); following_z=gather(height,following)
        anchor_temperature=gather(temperature,anchor); following_temperature=gather(temperature,following)
        f32=np.float32
        temperature=np.asarray(temperature,dtype=f32)
        humidity=np.asarray(humidity,dtype=f32); pressure=np.asarray(pressure,dtype=f32)
        vapor=np.asarray(np.asarray(f32(.01)*humidity,dtype=f32)*
                         saturation_vapor_pressure_lowe(np.asarray(
                             temperature-f32(273.15),dtype=f32)),dtype=f32)
        mixing=np.asarray(np.asarray(f32(.622)*vapor,dtype=f32)/
                          np.asarray(pressure-vapor,dtype=f32),dtype=f32)
        specific=np.asarray(mixing/np.asarray(f32(1.)+mixing,dtype=f32),dtype=f32)
        potential=np.asarray(temperature*_real4_power(
            np.asarray(f32(1000.)/pressure,dtype=f32),f32(.286)),dtype=f32)
        virtual_potential=np.asarray(potential*np.asarray(
            f32(1.)+np.asarray(f32(.61)*specific,dtype=f32),dtype=f32),dtype=f32)
        theta2=gather(virtual_potential,anchor); theta3=gather(virtual_potential,following)
        if thermal_state is None:
            scale=np.asarray(following_z/f32(5.),dtype=f32)
            upper_exp=_real4_libm('expf',np.asarray(-f32(.77)*following_z/scale,dtype=f32))
            lower_exp=_real4_libm('expf',np.asarray(-f32(.77)*anchor_z/scale,dtype=f32))
            denominator=np.asarray(upper_exp-lower_exp,dtype=f32)
            drop=np.divide(np.asarray(theta3-theta2,dtype=f32),denominator,
                           out=np.zeros_like(theta2,dtype=f32),where=denominator!=f32(0.))
            drop=np.maximum(np.minimum(drop,f32(0.)),f32(-15.))
            theta30=np.asarray(theta3-np.asarray(drop*upper_exp,dtype=f32),dtype=f32)
            theta20=theta30.copy()
        else:
            theta20=np.asarray(thermal_state.pt20,dtype=f32)
            theta30=np.asarray(thermal_state.pt30,dtype=f32)
            if theta20.shape!=flat_shape or theta30.shape!=flat_shape:
                raise ValueError('MM4 thermal state does not match source grid')
        sun=solar_at_step_midpoint(record.timestamp-timedelta(hours=1),3600.,base_timezone,
                                   self.header.latitude,self.header.longitude)
        daylight=sun>0.; theta20=np.where(daylight,theta2,theta20); theta30=np.where(daylight,theta3,theta30)
        next_thermal_state=SourceThermalState(theta20.copy(),theta30.copy())
        source_temperature_centers=_vertical_centers(temperature,height,centers)
        immediate=np.minimum(anchor+1,height.shape[0]-1)
        immediate_z=gather(height,immediate); immediate_temperature=gather(temperature,immediate)
        immediate_theta=gather(virtual_potential,immediate)
        logarithm_argument=np.asarray(
            np.asarray(anchor_temperature*immediate_theta,dtype=f32)/
            np.asarray(immediate_temperature*theta2,dtype=f32),dtype=f32)
        coefficient=np.divide(_real4_libm('logf',logarithm_argument),
            np.asarray(anchor_z-immediate_z,dtype=f32),out=np.zeros_like(anchor_z,dtype=f32),
            where=anchor_z!=immediate_z).astype(f32)
        multiplier=np.asarray(np.asarray(_real4_libm('expf',np.asarray(
            -coefficient*anchor_z,dtype=f32))*anchor_temperature,dtype=f32)/theta2,dtype=f32)
        ocean=self.header.landuse==prognostic_ocean_category
        for level,center in enumerate(centers):
            below=center<anchor_z
            daytime=np.asarray(anchor_temperature+np.asarray(
                f32(.0098)*np.asarray(anchor_z-center,dtype=f32),dtype=f32),dtype=f32)
            fraction=np.divide(np.asarray(center-anchor_z,dtype=f32),
                np.asarray(following_z-anchor_z,dtype=f32),out=np.zeros_like(anchor_z,dtype=f32),
                where=following_z!=anchor_z)
            ocean_theta=np.asarray(theta2+np.asarray(
                fraction*np.asarray(theta3-theta2,dtype=f32),dtype=f32),dtype=f32)
            land_theta=_stull_temperature(theta2,anchor_z,theta3,following_z,
                                           theta20,theta30,center)
            night_theta=np.where(ocean,ocean_theta,land_theta)
            nighttime=np.asarray(np.asarray(night_theta*_real4_libm(
                'expf',np.asarray(coefficient*center,dtype=f32)),dtype=f32)*multiplier,dtype=f32)
            replacement=np.where(daylight,daytime,nighttime)
            source_temperature_centers[level]=np.where(below,replacement,
                                                        source_temperature_centers[level])
        surface_day=np.asarray(anchor_temperature+np.asarray(
            f32(.0098)*anchor_z,dtype=f32),dtype=f32)
        ocean_surface=np.asarray(theta2-np.divide(np.asarray(
            anchor_z*np.asarray(theta3-theta2,dtype=f32),dtype=f32),
            np.asarray(following_z-anchor_z,dtype=f32),out=np.zeros_like(anchor_z,dtype=f32),
            where=following_z!=anchor_z),dtype=f32)
        land_surface=_stull_temperature(theta2,anchor_z,theta3,following_z,
                                         theta20,theta30,0.)
        surface_night=np.asarray(np.where(ocean,ocean_surface,land_surface)*multiplier,dtype=f32)
        source_surface_temperature=np.where(daylight,surface_day,surface_night)
        mapped_temperature=mapper.interpolate(source_temperature_centers)
        mapped_humidity=mapper.interpolate(_vertical_centers(humidity,height,centers))
        mapped_pressure=mapper.interpolate(_vertical_centers(pressure,height,centers))
        nearest_height=mapper.nearest(height); nearest_temperature=mapper.nearest(temperature)
        # RDMM4 saves the surface scalars while reading N=1, before QCKSRT3
        # sorts the profile by height.  Preserve that distinction for files
        # containing below-ground or otherwise non-monotonic levels.
        source_surface_pressure=np.asarray(record.pressure_mb[0])
        raw_surface_temperature=np.asarray(record.temperature_k[0])
        source_surface_humidity=np.asarray(record.relative_humidity[0])
        surface_density=mapper.interpolate(density_from_pressure_temperature(
            source_surface_pressure,raw_surface_temperature))
        surface_pressure=mapper.interpolate(source_surface_pressure)
        surface_humidity=np.trunc(mapper.interpolate(source_surface_humidity)).astype(np.int32)
        if cloud_method==4:
            source_cloud,source_ceiling=cloud_cover_mm5(humidity,pressure,height)
            cloud=mapper.interpolate(source_cloud)
            mapped_ceiling=mapper.interpolate(source_ceiling,ignore_zero=True)
            # ICEILG is stored as integer hundreds of feet.  Expose metres
            # to the Python solver, but only after RDMM4's truncation.
            ceiling=np.asarray(np.trunc(mapped_ceiling/f32(30.48))*f32(30.48),dtype=f32)
        elif cloud_method==3:
            # The legacy loop retains the last record-order level whose
            # pressure is above 850 mb (and always initializes from N=1).
            source_pressure=np.asarray(record.pressure_mb)
            source_humidity=np.asarray(record.relative_humidity)
            eligible=source_pressure>850.; eligible[0]=True
            reverse=np.argmax(eligible[::-1],axis=0)
            level=eligible.shape[0]-1-reverse; yy,xx=np.indices(level.shape)
            rh850=source_humidity[level,yy,xx]
            cloud=cloud_fraction(mapper.interpolate(rh850))
            ceiling=np.full((grid.ny,grid.nx),80.*30.48)
        else:
            raise ValueError('MM4 cloud_method must be MCLOUD 3 or 4')
        precipitation=mapper.radius_interpolate(record.precipitation_mm_h,
            precipitation_radius_km,method=precipitation_method,cutoff=precipitation_cutoff)
        precipitation_code=np.where(precipitation>0.,
            np.where(mapped_temperature[0]<273.15,20,10),0).astype(np.int32)
        return MM4Interpolated(u,v,mapped_temperature,mapped_humidity,mapped_pressure,
            np.concatenate((np.zeros((1,grid.ny,grid.nx)),nearest_height)),
            np.concatenate((mapper.nearest(source_surface_temperature)[None],nearest_temperature)),
            surface_density,precipitation,surface_pressure,surface_humidity,cloud,ceiling,
            precipitation_code,mapper.interpolate(source_surface_temperature),next_thermal_state,
            observation_u,observation_v,observation_strength)
