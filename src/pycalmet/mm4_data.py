"""Reader for the fixed-column IWAQM ``MM4.DAT`` format."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Literal, SupportsFloat, TextIO, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._calmet_types import SourceThermalState
from .config import GridConfig
from .datetime_utils import expand_year


MODEL_DTYPE: type[np.float32]=np.float32
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]
IndexArray: TypeAlias = NDArray[np.intp]
MM4Scalar: TypeAlias = int | float
MM4Converter: TypeAlias = Callable[[str], MM4Scalar]



def _fields(
    line: str,
    widths: Sequence[int],
    converters: Sequence[MM4Converter],
    label: str,
) -> list[MM4Scalar]:
    """Decode adjacent fixed-width fields from one MM4.DAT line.

    Args:
        line: Raw text record without interpretation.
        widths: Field widths in characters.
        converters: One conversion callable per width.
        label: Record name included in parse errors.

    Returns:
        Converted field values in source order.

    Raises:
        ValueError: If any field cannot be converted.
    """
    values: list[MM4Scalar]=[]; begin=0
    try:
        for width,convert in zip(widths,converters):
            values.append(convert(line[begin:begin+width]))
            begin+=width
    except ValueError as error:
        raise ValueError(f'invalid MM4.DAT {label} record: {line!r}') from error
    return values


@dataclass(frozen=True,slots=True)
class MM4Header:
    """Static model and subdomain metadata from MM4.DAT.

    Attributes:
        label: File creation label.
        start: First modeled timestamp.
        hours: Declared duration in hours.
        model_nx: Full-model east-west dimension.
        model_ny: Full-model north-south dimension.
        sigma_levels: Native sigma-coordinate levels.
        top_pressure_mb: Model-top pressure in millibars.
        i_start: One-based source subdomain start index in X.
        j_start: One-based source subdomain start index in Y.
        latitude: Subdomain latitudes in degrees north, shape ``(y, x)``.
        longitude: Subdomain longitudes in degrees east, shape ``(y, x)``.
        terrain_height_m: Terrain elevation in metres, shape ``(y, x)``.
        landuse: Integer land-use categories, shape ``(y, x)``.
    """
    label: str
    start: datetime
    hours: int
    model_nx: int
    model_ny: int
    sigma_levels: Float32Array
    top_pressure_mb: float
    i_start: int
    j_start: int
    latitude: Float64Array
    longitude: Float64Array
    terrain_height_m: Float32Array
    landuse: Int32Array


@dataclass(frozen=True,slots=True)
class MM4Record:
    """One hourly MM4 meteorological record.

    Attributes:
        timestamp: Beginning or label time carried by the source record.
        sea_level_pressure_mb: Sea-level pressure on ``(y, x)`` in millibars.
        precipitation_mm_h: Hourly precipitation rate on ``(y, x)``.
        snow_code: Integer surface snow code on ``(y, x)``.
        pressure_mb: Pressure on ``(level, y, x)`` in millibars.
        height_agl_m: Height above local terrain on ``(level, y, x)`` in metres.
        temperature_k: Temperature on ``(level, y, x)`` in kelvin.
        relative_humidity: Relative humidity on ``(level, y, x)`` in percent.
        wind_direction_deg: Meteorological direction in degrees.
        wind_speed_m_s: Wind speed in metres per second.
    """
    timestamp: datetime
    sea_level_pressure_mb: Float32Array
    precipitation_mm_h: Float32Array
    snow_code: Int32Array
    pressure_mb: Float32Array
    height_agl_m: Float32Array
    temperature_k: Float32Array
    relative_humidity: Float32Array
    wind_direction_deg: Float32Array
    wind_speed_m_s: Float32Array


@dataclass(frozen=True,slots=True)
class MM4Interpolated:
    """MM4 fields interpolated onto the CALMET grid and vertical levels.

    Attributes:
        u: Eastward wind in metres per second, shape ``(level, ny, nx)``.
        v: Northward wind in metres per second, shape ``(level, ny, nx)``.
        temperature_k: Layer temperature in kelvin.
        relative_humidity: Layer relative humidity in percent.
        pressure_mb: Layer pressure in millibars.
        profile_height_m: Native profile heights including the surface.
        profile_temperature_k: Native profile temperatures including surface.
        surface_density: Air density at the surface in kilograms per cubic
            metre.
        precipitation_mm_h: Surface precipitation rate.
        surface_pressure_mb: Surface pressure in millibars.
        surface_relative_humidity: Surface relative humidity in percent.
        cloud_fraction: Cloud fraction from zero to one.
        ceiling_height_m: Cloud-ceiling height in metres.
        precipitation_code: Integer CALMET precipitation type code.
        surface_temperature_k: Diagnosed surface temperature in kelvin.
        thermal_state: Source-grid thermal memory for the next timestep.
        prognostic_observation_u: Optional observation-form eastward wind.
        prognostic_observation_v: Optional observation-form northward wind.
        prognostic_observation_strength: Optional interpolation weight sum.

    Notes:
        Unless stated otherwise, gridded numeric fields use ``float32`` and
        horizontal shape ``(ny, nx)``.
    """
    u: Float32Array
    v: Float32Array
    temperature_k: Float32Array
    relative_humidity: Float32Array
    pressure_mb: Float32Array
    profile_height_m: Float32Array
    profile_temperature_k: Float32Array
    surface_density: Float32Array
    precipitation_mm_h: Float32Array
    surface_pressure_mb: Float32Array
    surface_relative_humidity: Float32Array
    cloud_fraction: Float32Array
    ceiling_height_m: Float32Array
    precipitation_code: Int32Array
    surface_temperature_k: Float32Array
    thermal_state: SourceThermalState
    prognostic_observation_u: Float32Array | None = None
    prognostic_observation_v: Float32Array | None = None
    prognostic_observation_strength: Float32Array | None = None


class MM4Data:
    """Load native formatted MM4 data as vectorized hourly arrays.

    Attributes:
        path: Path to the decoded MM4.DAT file.
        header: Static grid and model metadata.
        records: Ordered hourly meteorological records.
    """

    def __init__(self,path: str | Path) -> None:
        """Read and validate a fixed-column MM4.DAT file.

        Args:
            path: ASCII MM4.DAT file path.

        Raises:
            OSError: If the file cannot be opened or read.
            UnicodeError: If the source is not ASCII.
            ValueError: If the header or a meteorological record is malformed.
        """
        self.path: Path=Path(path)
        with self.path.open(encoding='ascii') as stream:
            self.header: MM4Header=self._read_header(stream)
            self.records: tuple[MM4Record,...]=self._read_records(stream)
        if not self.records: raise ValueError('MM4.DAT contains no meteorological records')

    @staticmethod
    def _line(stream: TextIO,label: str) -> str:
        """Read one required text line without its newline.

        Args:
            stream: Open MM4.DAT text stream.
            label: Expected record name included in truncation errors.

        Returns:
            Source line with trailing CR/LF removed.

        Raises:
            ValueError: If the file ends before the line is available.
        """
        line=stream.readline()
        if not line: raise ValueError(f'truncated MM4.DAT before {label}')
        return line.rstrip('\r\n')

    @classmethod
    def _read_header(cls,stream: TextIO) -> MM4Header:
        """Decode the MM4 model header and static subdomain grid.

        Args:
            stream: Open stream positioned at the first header record.

        Returns:
            Validated static MM4 header.

        Raises:
            ValueError: If dimensions, dates, fields, or grid indices are
                invalid.
        """
        label=cls._line(stream,'creation label')[:36]
        grid_fields=cast(tuple[int,int,int,int,int,int,int,int,float],tuple(
            _fields(cls._line(stream,'grid header'),(2,2,2,2,4,4,4,4,6),
                    (int,int,int,int,int,int,int,int,float),'grid header')))
        year,month,day,hour,hours,model_nx,model_ny,nzp,top=grid_fields
        subdomain_fields=cast(tuple[int,int,int,int],tuple(
            _fields(cls._line(stream,'subdomain header'),(4,4,4,4),
                    (int,int,int,int),'subdomain header')))
        i_start,j_start,nx,ny=subdomain_fields
        if min(hours,model_nx,model_ny,nzp,nx,ny)<=0:
            raise ValueError('MM4.DAT header dimensions and duration must be positive')
        sigma: Float32Array=np.array(
            [float(cls._line(stream,f'sigma level {index+1}')[:6])
             for index in range(nzp)],dtype=MODEL_DTYPE)
        latitude: Float64Array=np.empty((ny,nx),dtype=np.float64)
        longitude: Float64Array=np.empty((ny,nx),dtype=np.float64)
        terrain: Float32Array=np.empty((ny,nx),dtype=MODEL_DTYPE)
        landuse: Int32Array=np.empty((ny,nx),dtype=np.int32)
        for j in range(ny):
            for i in range(nx):
                static_fields=cast(tuple[int,int,float,float,int,int],tuple(
                    _fields(cls._line(stream,'static grid'),(3,3,7,8,5,3),
                            (int,int,float,float,int,int),'static grid')))
                ii,jj,latitude[j,i],longitude[j,i],terrain[j,i],landuse[j,i]=static_fields
                if (ii,jj)!=(i_start+i,j_start+j):
                    raise ValueError(f'MM4.DAT static grid index {(ii,jj)} does not match '
                                     f'expected {(i_start+i,j_start+j)}')
        return MM4Header(label,datetime(expand_year(year),month,day,hour),hours,model_nx,
                         model_ny,sigma,top,i_start,j_start,latitude,longitude,
                         terrain,landuse)

    def _read_records(self,stream: TextIO) -> tuple[MM4Record,...]:
        """Decode all remaining hourly surface and vertical records.

        Args:
            stream: Open stream positioned after the static header.

        Returns:
            Tuple of vectorized hourly records.

        Raises:
            ValueError: If a record is truncated, inconsistent, or has an
                unexpected grid index.
        """
        header=self.header; ny,nx=header.latitude.shape; levels=8+header.sigma_levels.size
        records: list[MM4Record]=[]
        while True:
            first=stream.readline()
            if not first: break
            surface: Float32Array=np.empty((ny,nx,3),dtype=MODEL_DTYPE)
            pressure: Float32Array=np.empty((levels,ny,nx),dtype=MODEL_DTYPE); height: Float32Array=np.empty_like(pressure)
            encoded_temperature: Float32Array=np.empty_like(pressure); depression: Float32Array=np.empty_like(pressure)
            direction: Float32Array=np.empty_like(pressure); speed: Float32Array=np.empty_like(pressure)
            timestamp: datetime | None=None
            for j in range(ny):
                for i in range(nx):
                    line=first.rstrip('\r\n') if i==0 and j==0 else self._line(stream,'surface data')
                    surface_fields=cast(
                        tuple[int,int,int,int,int,int,float,float,int],
                        tuple(_fields(line,(2,2,2,2,3,3,7,5,2),
                            (int,int,int,int,int,int,float,float,int),'surface data')))
                    year,month,day,hour,ii,jj,pmsl,rain,snow=surface_fields
                    current=datetime(expand_year(year),month,day,hour)
                    if timestamp is None: timestamp=current
                    if current!=timestamp: raise ValueError('MM4.DAT grid cells have inconsistent timestamps')
                    expected=(header.i_start+i,header.j_start+j)
                    if (ii,jj)!=expected:
                        raise ValueError(f'MM4.DAT data grid index {(ii,jj)} does not match expected {expected}')
                    surface[j,i]=(pmsl,rain*10.,snow)
                    for level in range(levels):
                        vertical_fields=cast(tuple[float,float,int,int,float,float],tuple(
                            _fields(self._line(stream,'vertical data'),(5,6,4,2,4,2),
                                    (float,float,int,int,float,float),'vertical data')))
                        pressure[level,j,i],height[level,j,i],encoded_temperature[level,j,i],\
                            depression[level,j,i],direction[level,j,i],speed[level,j,i]=vertical_fields
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
            records.append(MM4Record(cast(datetime,timestamp),surface[...,0],surface[...,1],
                surface[...,2].astype(np.int32),pressure,height-header.terrain_height_m,
                cast(Float32Array,temperature_c+MODEL_DTYPE(273.15)),relative_humidity,direction,
                speed*MODEL_DTYPE(.51444)))
        return tuple(records)

    def at_time(self,timestamp: datetime) -> MM4Record:
        """Select the record with an exact timestamp.

        Args:
            timestamp: Naive hourly model timestamp.

        Returns:
            Matching MM4 record.

        Raises:
            ValueError: If the file has no record at ``timestamp``.
        """
        for record in self.records:
            if record.timestamp==timestamp: return record
        raise ValueError(f'MM4.DAT has no record for {timestamp}')

    @staticmethod
    def _sorted_above_ground_profiles(
        **fields: ArrayLike,
    ) -> dict[str,NDArray[np.generic]]:
        """Sort profiles, discard underground levels, and remove duplicates.

        Args:
            **fields: Named three-dimensional arrays with common shape
                ``(level, y, x)``; a ``height`` field in metres is required.

        Returns:
            Dictionary of packed arrays in ascending height order. Repeated
            tail values pad columns containing fewer unique valid levels.

        Raises:
            ValueError: If field shapes differ or a column has no above-ground
                level.
        """
        height=np.asarray(fields['height'])
        if any(np.asarray(values).shape!=height.shape for values in fields.values()):
            raise ValueError('MM4.DAT profile fields must have identical shapes')
        levels,ny,nx=height.shape
        packed: dict[str,NDArray[np.generic]]={
            name:np.empty_like(np.asarray(values)) for name,values in fields.items()
        }
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
    def _interpolate_pressure_level(
        values: ArrayLike,
        pressure_mb: ArrayLike,
        target_pressure_mb: SupportsFloat,
    ) -> Float32Array:
        """Interpolate columns to pressure using nearest endpoint extrapolation.

        Args:
            values: Source values with shape ``(level, y, x)``.
            pressure_mb: Matching source pressures in millibars.
            target_pressure_mb: Scalar target pressure in millibars.

        Returns:
            ``float32`` field with shape ``(y, x)``.

        Raises:
            ValueError: If shapes differ, inputs are not three-dimensional, or
                a column has no finite positive pressure level.
        """
        values=np.asarray(values)
        pressure=np.asarray(pressure_mb)
        if values.shape!=pressure.shape or values.ndim!=3:
            raise ValueError('MM4 pressure interpolation fields must share a 3-D shape')
        result: Float32Array=np.empty(values.shape[1:],dtype=MODEL_DTYPE)
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

    def interpolate_record(
        self,
        record: MM4Record,
        grid: GridConfig,
        target_height_m: ArrayLike,
        *,
        cloud_method: int=4,
        precipitation_method: Literal[1,2,3]=2,
        precipitation_radius_km: float=50.,
        precipitation_cutoff: float=.01,
        base_timezone: int=0,
        thermal_state: SourceThermalState | None=None,
        prognostic_observation_rmin_km: float | None=None,
        prognostic_observation_barriers: Sequence[tuple[float,float,float,float]]=(),
        prognostic_observation_barrier_top_level: int=0,
        prognostic_ocean_category: int=16,
    ) -> MM4Interpolated:
        """Apply RDHD4/RDMM4 vertical and four-neighbour interpolation.

        Args:
            record: Hourly source field to interpolate.
            grid: Target CALMET grid.
            target_height_m: Non-empty vector of CALMET layer-centre heights
                above ground in metres.
            cloud_method: CALMET ``MCLOUD`` option: 3 uses 850-mb humidity and
                4 uses the MM5 column method.
            precipitation_method: Radius-interpolation method for precipitation.
            precipitation_radius_km: Precipitation influence radius or method-3
                scale in kilometres.
            precipitation_cutoff: Rates below this value become zero, in
                millimetres per hour.
            base_timezone: CALMET base timezone offset used for solar time.
            thermal_state: Optional source-grid nocturnal thermal state from
                the preceding timestep.
            prognostic_observation_rmin_km: Optional radius that enables a
                second interpolation treating model cells as observations.
            prognostic_observation_barriers: Barrier line segments in grid
                kilometres for observation-form interpolation.
            prognostic_observation_barrier_top_level: Highest zero-based level
                at which barriers apply.
            prognostic_ocean_category: MM4 land-use code treated as ocean.

        Returns:
            Interpolated winds, thermodynamic fields, precipitation, clouds,
            and next-step thermal state on the target grid.

        Raises:
            ValueError: If the target grid lies outside the source domain,
                target heights or thermal state are invalid, or
                ``cloud_method`` is outside the supported values 3 and 4.
        """
        from .calmet_solver import (PrognosticMapper,
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
        height=cast(Float32Array,profiles['height'])
        temperature=cast(Float32Array,profiles['temperature'])
        humidity=cast(Float32Array,profiles['humidity'])
        pressure=cast(Float32Array,profiles['pressure'])
        direction=cast(Float32Array,profiles['direction'])
        speed=cast(Float32Array,profiles['speed'])
        convergence=grid_convergence(grid,self.header.longitude,self.header.latitude)
        source_u,source_v=_prognostic_wind_components(
            speed,direction,convergence[None,:,:])
        centers: Float32Array=np.asarray(target_height_m,dtype=MODEL_DTYPE)
        if centers.ndim!=1 or centers.size<1:
            raise ValueError('target MM4 heights must be a non-empty vector')
        def wind_centers(field: Float32Array) -> Float32Array:
            """Interpolate a wind component vertically with surface log law.

            Args:
                field: Source component on ``(level, y, x)``.

            Returns:
                Component at target centres on ``(target_level, y, x)``.
            """
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
        observation_u: Float32Array | None
        observation_v: Float32Array | None
        observation_strength: Float32Array | None
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
        def gather(
            values: Float32Array,
            index: IndexArray,
        ) -> Float32Array:
            """Gather a different vertical index in every source column.

            Args:
                values: Source array with shape ``(level, y, x)``.
                index: Flat integer level index for every horizontal column.

            Returns:
                Gathered two-dimensional source-grid field.
            """
            return np.asarray(values).reshape(values.shape[0],-1)[index,columns].reshape(flat_shape)
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
        def as_model(value: ArrayLike) -> Float32Array:
            """Cast a result field to the model's ``float32`` dtype.

            Args:
                value: Array-like result field.

            Returns:
                NumPy ``float32`` view or copy.
            """
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
