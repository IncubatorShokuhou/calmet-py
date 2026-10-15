"""Reader for the fixed-column IWAQM ``MM4.DAT`` format."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path

import numpy as np

from .datetime_utils import expand_year


MODEL_DTYPE=np.float32


def _fields(line: str, widths, converters, label: str):
    values=[]; begin=0
    try:
        for width,convert in zip(widths,converters):
            values.append(convert(line[begin:begin+width]))
            begin+=width
    except ValueError as error:
        raise ValueError(f'invalid MM4.DAT {label} record: {line!r}') from error
    return values


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
                        for index in range(nzp)],dtype=MODEL_DTYPE)
        latitude=np.empty((ny,nx),dtype=np.float64)
        longitude=np.empty((ny,nx),dtype=np.float64)
        terrain=np.empty((ny,nx),dtype=MODEL_DTYPE)
        landuse=np.empty((ny,nx),dtype=np.int32)
        for j in range(ny):
            for i in range(nx):
                values=_fields(cls._line(stream,'static grid'),(3,3,7,8,5,3),
                               (int,int,float,float,int,int),'static grid')
                ii,jj,latitude[j,i],longitude[j,i],terrain[j,i],landuse[j,i]=values
                if (ii,jj)!=(i_start+i,j_start+j):
                    raise ValueError(f'MM4.DAT static grid index {(ii,jj)} does not match '
                                     f'expected {(i_start+i,j_start+j)}')
        return MM4Header(label,datetime(expand_year(year),month,day,hour),hours,model_nx,
                         model_ny,sigma,top,i_start,j_start,latitude,longitude,
                         terrain,landuse)

    def _read_records(self,stream):
        header=self.header; ny,nx=header.latitude.shape; levels=8+header.sigma_levels.size
        records=[]
        while True:
            first=stream.readline()
            if not first: break
            surface=np.empty((ny,nx,3),dtype=MODEL_DTYPE)
            pressure=np.empty((levels,ny,nx),dtype=MODEL_DTYPE); height=np.empty_like(pressure)
            encoded_temperature=np.empty_like(pressure); depression=np.empty_like(pressure)
            direction=np.empty_like(pressure); speed=np.empty_like(pressure)
            timestamp=None
            for j in range(ny):
                for i in range(nx):
                    line=first.rstrip('\r\n') if i==0 and j==0 else self._line(stream,'surface data')
                    values=_fields(line,(2,2,2,2,3,3,7,5,2),
                                   (int,int,int,int,int,int,float,float,int),'surface data')
                    year,month,day,hour,ii,jj,pmsl,rain,snow=values
                    current=datetime(expand_year(year),month,day,hour)
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
            temperature_c=(np.where(negative,-encoded_temperature,
                                    encoded_temperature)/MODEL_DTYPE(10.))
            dew_depression=np.where(
                depression<MODEL_DTYPE(56.),
                depression/MODEL_DTYPE(10.),
                depression-MODEL_DTYPE(50.),
            )
            dry_f=MODEL_DTYPE(1.8)*temperature_c+MODEL_DTYPE(32.)
            dew_f=(MODEL_DTYPE(1.8)*(temperature_c-dew_depression)+
                   MODEL_DTYPE(32.))
            numerator=MODEL_DTYPE(173.)-MODEL_DTYPE(.1)*dry_f+dew_f
            denominator=MODEL_DTYPE(173.)+MODEL_DTYPE(.9)*dry_f
            relative_humidity=np.minimum(
                MODEL_DTYPE(100.),
                (numerator/denominator)**8*MODEL_DTYPE(100.),
            )
            records.append(MM4Record(timestamp,surface[...,0],surface[...,1],
                surface[...,2].astype(np.int32),pressure,height-header.terrain_height_m,
                temperature_c+MODEL_DTYPE(273.15),relative_humidity,direction,
                speed*MODEL_DTYPE(.51444)))
        return tuple(records)

    def at_time(self,timestamp):
        for record in self.records:
            if record.timestamp==timestamp: return record
        raise ValueError(f'MM4.DAT has no record for {timestamp}')

    @staticmethod
    def _sorted_above_ground_profiles(**fields):
        """Sort profiles, discard underground levels, and remove duplicates."""
        height=np.asarray(fields['height'])
        if any(np.asarray(values).shape!=height.shape for values in fields.values()):
            raise ValueError('MM4.DAT profile fields must have identical shapes')
        levels,ny,nx=height.shape
        packed={name:np.empty_like(np.asarray(values))
                for name,values in fields.items()}
        for j in range(ny):
            for i in range(nx):
                order=np.argsort(height[:,j,i],kind='stable')
                ordered_height=height[order,j,i]
                above=ordered_height>=-0.0001
                distinct=np.concatenate((np.array([True]),
                                         np.diff(ordered_height)!=0.))
                selected=order[above & distinct]
                if selected.size==0:
                    raise ValueError(f'MM4.DAT profile {(i,j)} has no above-ground level')
                source=np.pad(selected,(0,levels-selected.size),mode='edge')
                for name,values in fields.items():
                    packed[name][:,j,i]=np.asarray(values)[source,j,i]
        return packed

    @staticmethod
    def _interpolate_pressure_level(values,pressure_mb,target_pressure_mb):
        """Interpolate columns on pressure, using the nearest endpoint outside range."""
        values=np.asarray(values)
        pressure=np.asarray(pressure_mb)
        if values.shape!=pressure.shape or values.ndim!=3:
            raise ValueError('MM4 pressure interpolation fields must share a 3-D shape')
        result=np.empty(values.shape[1:],dtype=MODEL_DTYPE)
        for j in range(values.shape[1]):
            for i in range(values.shape[2]):
                column_pressure=np.asarray(pressure[:,j,i],dtype=np.float64)
                column_values=np.asarray(values[:,j,i],dtype=np.float64)
                valid=(np.isfinite(column_pressure)&np.isfinite(column_values)&
                       (column_pressure>0.))
                if not np.any(valid):
                    raise ValueError(
                        f'MM4 profile {(i,j)} has no valid pressure level')
                column_pressure=column_pressure[valid]
                column_values=column_values[valid]
                order=np.argsort(column_pressure,kind='stable')
                column_pressure=column_pressure[order]
                column_values=column_values[order]
                unique,inverse=np.unique(column_pressure,return_inverse=True)
                totals=np.bincount(inverse,weights=column_values)
                counts=np.bincount(inverse)
                averaged=totals/counts
                result[j,i]=np.interp(float(target_pressure_mb),unique,averaged)
        return result

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
        from .calmet_numerics import (cloud_cover_mm5,cloud_fraction,
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
        centers=np.asarray(target_height_m,dtype=MODEL_DTYPE)
        if centers.ndim!=1 or centers.size<1:
            raise ValueError('target MM4 heights must be a non-empty vector')
        def wind_centers(field):
            base=_vertical_centers(field,height,centers)
            flat=np.asarray(field).reshape(field.shape[0],-1)
            zflat=height.reshape(height.shape[0],-1)
            columns=np.arange(flat.shape[1]); anchor=np.where(zflat[0]<1.,1,0)
            anchor=np.minimum(anchor,field.shape[0]-1)
            anchor_z=zflat[anchor,columns]; anchor_value=flat[anchor,columns]
            log_half=np.log(MODEL_DTYPE(.5))
            for level,center in enumerate(centers):
                below=center<anchor_z
                numerator=np.log(center)-log_half
                denominator=(np.log(np.maximum(anchor_z,MODEL_DTYPE(.500001)))-
                             log_half)
                factor=np.divide(numerator,denominator,
                    out=np.ones_like(anchor_z,dtype=MODEL_DTYPE),
                    where=anchor_z>MODEL_DTYPE(.500001))
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
        temperature=np.asarray(temperature,dtype=MODEL_DTYPE)
        humidity=np.asarray(humidity,dtype=MODEL_DTYPE)
        pressure=np.asarray(pressure,dtype=MODEL_DTYPE)
        vapor=(MODEL_DTYPE(.01)*humidity*
               saturation_vapor_pressure_lowe(temperature-MODEL_DTYPE(273.15)))
        mixing=MODEL_DTYPE(.622)*vapor/(pressure-vapor)
        specific=mixing/(MODEL_DTYPE(1.)+mixing)
        potential=temperature*np.power(
            MODEL_DTYPE(1000.)/pressure,MODEL_DTYPE(.286))
        virtual_potential=potential*(MODEL_DTYPE(1.)+MODEL_DTYPE(.61)*specific)
        theta2=gather(virtual_potential,anchor); theta3=gather(virtual_potential,following)
        if thermal_state is None:
            scale=following_z/MODEL_DTYPE(5.)
            upper_exp=np.exp(-MODEL_DTYPE(.77)*following_z/scale)
            lower_exp=np.exp(-MODEL_DTYPE(.77)*anchor_z/scale)
            denominator=upper_exp-lower_exp
            drop=np.divide(theta3-theta2,denominator,
                           out=np.zeros_like(theta2),where=denominator!=0.)
            drop=np.clip(drop,MODEL_DTYPE(-15.),MODEL_DTYPE(0.))
            theta30=theta3-drop*upper_exp
            theta20=theta30.copy()
        else:
            theta20=np.asarray(thermal_state.pt20,dtype=MODEL_DTYPE)
            theta30=np.asarray(thermal_state.pt30,dtype=MODEL_DTYPE)
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
        logarithm_argument=(anchor_temperature*immediate_theta)/(
            immediate_temperature*theta2)
        coefficient=np.divide(np.log(logarithm_argument),
            anchor_z-immediate_z,out=np.zeros_like(anchor_z),
            where=anchor_z!=immediate_z)
        multiplier=np.exp(-coefficient*anchor_z)*anchor_temperature/theta2
        ocean=self.header.landuse==prognostic_ocean_category
        for level,center in enumerate(centers):
            below=center<anchor_z
            daytime=anchor_temperature+MODEL_DTYPE(.0098)*(anchor_z-center)
            fraction=np.divide(center-anchor_z,following_z-anchor_z,
                out=np.zeros_like(anchor_z),
                where=following_z!=anchor_z)
            ocean_theta=theta2+fraction*(theta3-theta2)
            land_theta=_stull_temperature(theta2,anchor_z,theta3,following_z,
                                           theta20,theta30,center)
            night_theta=np.where(ocean,ocean_theta,land_theta)
            nighttime=night_theta*np.exp(coefficient*center)*multiplier
            replacement=np.where(daylight,daytime,nighttime)
            source_temperature_centers[level]=np.where(below,replacement,
                                                        source_temperature_centers[level])
        surface_day=anchor_temperature+MODEL_DTYPE(.0098)*anchor_z
        ocean_surface=theta2-np.divide(
            anchor_z*(theta3-theta2),following_z-anchor_z,
            out=np.zeros_like(anchor_z),where=following_z!=anchor_z)
        land_surface=_stull_temperature(theta2,anchor_z,theta3,following_z,
                                         theta20,theta30,0.)
        surface_night=np.where(ocean,ocean_surface,land_surface)*multiplier
        source_surface_temperature=np.where(daylight,surface_day,surface_night)
        mapped_temperature=mapper.interpolate(source_temperature_centers)
        mapped_humidity=mapper.interpolate(_vertical_centers(humidity,height,centers))
        mapped_pressure=mapper.interpolate(_vertical_centers(pressure,height,centers))
        nearest_height=mapper.nearest(height); nearest_temperature=mapper.nearest(temperature)
        # The lowest valid physical layer is the surface reference even when
        # records arrive with non-monotonic or below-ground levels.
        source_surface_pressure=np.asarray(pressure[0])
        raw_surface_temperature=np.asarray(temperature[0])
        source_surface_humidity=np.asarray(humidity[0])
        surface_density=mapper.interpolate(density_from_pressure_temperature(
            source_surface_pressure,raw_surface_temperature))
        surface_pressure=mapper.interpolate(source_surface_pressure)
        surface_humidity=np.asarray(
            mapper.interpolate(source_surface_humidity),dtype=MODEL_DTYPE)
        if cloud_method==4:
            source_cloud,source_ceiling=cloud_cover_mm5(humidity,pressure,height)
            cloud=mapper.interpolate(source_cloud)
            mapped_ceiling=mapper.interpolate(source_ceiling,ignore_zero=True)
            ceiling=np.asarray(mapped_ceiling,dtype=MODEL_DTYPE)
        elif cloud_method==3:
            rh850=self._interpolate_pressure_level(humidity,pressure,850.)
            cloud=cloud_fraction(mapper.interpolate(rh850))
            ceiling=np.full((grid.ny,grid.nx),80.*30.48,dtype=MODEL_DTYPE)
        else:
            raise ValueError('MM4 cloud_method must be MCLOUD 3 or 4')
        precipitation=mapper.radius_interpolate(record.precipitation_mm_h,
            precipitation_radius_km,method=precipitation_method,cutoff=precipitation_cutoff)
        precipitation_code=np.where(precipitation>0.,
            np.where(mapped_temperature[0]<273.15,20,10),0).astype(np.int32)
        def as_model(value):
            return np.asarray(value,dtype=MODEL_DTYPE)
        profile_height=np.concatenate((
            np.zeros((1,grid.ny,grid.nx),dtype=MODEL_DTYPE),
            as_model(nearest_height)))
        profile_temperature=np.concatenate((
            as_model(mapper.nearest(source_surface_temperature))[None],
            as_model(nearest_temperature)))
        return MM4Interpolated(
            as_model(u),as_model(v),as_model(mapped_temperature),
            as_model(mapped_humidity),as_model(mapped_pressure),profile_height,
            profile_temperature,as_model(surface_density),as_model(precipitation),
            as_model(surface_pressure),surface_humidity,as_model(cloud),
            as_model(ceiling),precipitation_code,
            as_model(mapper.interpolate(source_surface_temperature)),next_thermal_state,
            None if observation_u is None else as_model(observation_u),
            None if observation_v is None else as_model(observation_v),
            None if observation_strength is None else as_model(observation_strength))
