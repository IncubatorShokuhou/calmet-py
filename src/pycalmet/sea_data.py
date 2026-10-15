"""Formatted CALMET SEA.DAT 2.1/2.11 overwater observations."""
from __future__ import annotations

from typing import TypeAlias, TypedDict, cast

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import GridConfig
from .datetime_utils import parse_ordinal
from .projections import transform_grid,unproject_geographic
from .surface_data import _nearest_indices


COMPUTE_DTYPE: type[np.float32]=np.float32
SPATIAL_DTYPE: type[np.float64]=np.float64
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]


class _GridParameters(TypedDict,total=False):
    """Optional projection keywords parsed from a SEA.DAT header."""

    utm_zone: int
    hemisphere: str
    lat0: float
    lon0: float
    lat1: float
    lat2: float
    false_easting_km: float
    false_northing_km: float



def _julian(
    year: int | float | str,
    day: int | float | str,
    hour: int | float | str,
) -> datetime:
    """Convert SEA.DAT ordinal time fields to a timestamp.

    Args:
        year: Four-digit calendar year.
        day: One-based day of year.
        hour: Hour value accepted by :func:`parse_ordinal`.

    Returns:
        Naive Gregorian timestamp represented by the three fields.

    Raises:
        ValueError: If the ordinal fields do not form a valid timestamp.
    """
    return parse_ordinal(
        cast(int | str,year),cast(int | str,day),cast(int | str,hour))


def _angle(text: str) -> float:
    """Parse a signed projection angle with an optional compass suffix.

    Args:
        text: Numeric angle optionally followed by ``N``, ``S``, ``E``, or
            ``W``.

    Returns:
        Angle in signed decimal degrees; south and west are negative.

    Raises:
        ValueError: If no numeric angle can be parsed.
    """
    match=re.search(r'([-+]?\d+(?:\.\d*)?)\s*([NSEW]?)',text.strip(),re.I)
    if not match: raise ValueError(f'invalid SEA.DAT projection angle: {text!r}')
    value=float(match.group(1))
    return -abs(value) if match.group(2).upper() in {'S','W'} else value


@dataclass(frozen=True,slots=True)
class SeaRecord:
    """One time-bounded overwater observation from SEA.DAT.

    Attributes:
        begin: Inclusive start of the observation interval.
        end: Exclusive end of the observation interval.
        x_km: Station easting in source-projection kilometres.
        y_km: Station northing in source-projection kilometres.
        anemometer_height_m: Wind sensor height in metres.
        air_sensor_height_m: Air temperature/humidity sensor height in metres.
        water_sensor_depth_m: Water temperature sensor depth in metres.
        air_water_delta_k: Air minus water temperature in kelvin.
        air_temperature_k: Air temperature in kelvin.
        relative_humidity: Relative humidity in percent.
        mixing_height_m: Observed overwater mixing height in metres.
        gradient_below_k_m: Temperature gradient below the mixing height in
            kelvin per metre.
        gradient_above_k_m: Temperature gradient above the mixing height in
            kelvin per metre.
        wind_speed_m_s: Wind speed in metres per second.
        wind_direction_deg: Meteorological wind direction in degrees.
        wave_period_s: Wave period in seconds, or a negative missing sentinel.
        wave_height_m: Wave height in metres, or a negative missing sentinel.
    """
    begin: datetime
    end: datetime
    x_km: float
    y_km: float
    anemometer_height_m: float
    air_sensor_height_m: float
    water_sensor_depth_m: float
    air_water_delta_k: float
    air_temperature_k: float
    relative_humidity: float
    mixing_height_m: float
    gradient_below_k_m: float
    gradient_above_k_m: float
    wind_speed_m_s: float
    wind_direction_deg: float
    wave_period_s: float
    wave_height_m: float


@dataclass(frozen=True,slots=True)
class GriddedSeaData:
    """Nearest valid SEA.DAT record fields on the CALMET grid.

    Attributes:
        air_temperature_k: Air temperature in kelvin.
        sea_temperature_k: Derived sea temperature in kelvin.
        relative_humidity: Relative humidity in percent.
        anemometer_height_m: Wind sensor height in metres.
        air_sensor_height_m: Temperature sensor height in metres.
        water_sensor_depth_m: Water sensor depth in metres.
        wave_period_s: Wave period in seconds.
        wave_height_m: Wave height in metres.
        station_longitude: Longitude of the station selected for each cell in
            degrees east.

    Notes:
        All fields have shape ``(ny, nx)``; numeric meteorological arrays use
        ``float32`` and station longitude uses spatial ``float64`` precision.
    """
    air_temperature_k: Float32Array
    sea_temperature_k: Float32Array
    relative_humidity: Float32Array
    anemometer_height_m: Float32Array
    air_sensor_height_m: Float32Array
    water_sensor_depth_m: Float32Array
    wave_period_s: Float32Array
    wave_height_m: Float32Array
    station_longitude: Float64Array


class SeaData:
    """Read one station per formatted SEA.DAT, as RDHDOW/RDOW do.

    Attributes:
        path: Path to the parsed SEA.DAT file.
        version: Dataset version text from the file header.
        projection: CALMET source projection code.
        datum: Source horizontal datum name.
        datum_date: Datum metadata date copied from the header.
        timezone: Version 2.11 timezone label, otherwise ``None``.
        start: Beginning of the declared or inferred record period.
        end: End of the declared or inferred record period.
        identifier: Integer station identifier.
        name: Station name.
        source_grid: One-cell grid carrying the station projection metadata.
        records: Chronologically contiguous :class:`SeaRecord` values.
    """

    version: str
    projection: str
    datum: str
    datum_date: str
    identifier: int
    name: str

    def __init__(self,path: str | Path) -> None:
        """Parse and validate a formatted SEA.DAT file.

        Args:
            path: ASCII SEA.DAT path using a supported pre-2.2 layout.

        Raises:
            OSError: If the file cannot be read.
            UnicodeError: If the file is not strict ASCII.
            ValueError: If the header, projection, time span, record width, or
                record chronology is invalid.
        """
        self.path: Path=Path(path)
        lines: list[str]=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty SEA.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='SEA.DAT': raise ValueError('invalid SEA.DAT dataset label')
        self.version=first[16:32].strip()
        try: version=float(self.version[:4])
        except ValueError as error: raise ValueError('invalid SEA.DAT version') from error
        if version<=0. or version>=2.2:
            raise ValueError('SEA.DAT version must be positive and earlier than 2.2')
        index=1
        comments=int(lines[index].split()[0]); index+=1+comments
        self.projection=lines[index][:8].strip().upper(); index+=1
        params: _GridParameters={}
        if self.projection=='UTM':
            fields=lines[index].split(); index+=1
            params['utm_zone']=int(fields[0]); params['hemisphere']=fields[1].upper()
        elif self.projection=='LCC':
            line=lines[index].ljust(64); index+=1
            params['lat0']=_angle(line[:16]); params['lon0']=_angle(line[16:32])
            params['lat1']=_angle(line[32:48]); params['lat2']=_angle(line[48:64])
            false=lines[index].split(); index+=1
            params['false_easting_km']=float(false[0]); params['false_northing_km']=float(false[1])
        elif self.projection=='PS':
            line=lines[index].ljust(48); index+=1
            params['lat0']=_angle(line[:16]); params['lon0']=_angle(line[16:32])
            params['lat1']=_angle(line[32:48])
        elif self.projection=='EM':
            line=lines[index].ljust(32); index+=1
            params['lat0']=_angle(line[:16]); params['lon0']=_angle(line[16:32])
        elif self.projection in {'LAZA','TTM'}:
            line=lines[index].ljust(32); index+=1
            params['lat0']=_angle(line[:16]); params['lon0']=_angle(line[16:32])
            false=lines[index].split(); index+=1
            params['false_easting_km']=float(false[0]); params['false_northing_km']=float(false[1])
        else:
            raise ValueError('SEA.DAT projection must be UTM, LCC, PS, EM, LAZA, or TTM')
        datum_line=lines[index]; index+=1
        self.datum=datum_line[:8].strip(); self.datum_date=datum_line[8:18].strip()
        units=lines[index][:4].strip().upper(); index+=1
        if units!='KM': raise ValueError('SEA.DAT coordinates must use KM')
        modern=version>=2.10999
        self.timezone: str | None
        self.start: datetime | None
        self.end: datetime | None
        if modern:
            self.timezone=lines[index][:8].strip(); index+=1
            span=lines[index].split(); index+=1
            if len(span)!=6: raise ValueError('SEA.DAT time span must contain six integers')
            self.start=_julian(*span[:3]); self.end=_julian(*span[3:])
        else:
            self.timezone=None; self.start=None; self.end=None
        station=lines[index].split(maxsplit=1); index+=1
        if len(station)!=2: raise ValueError('invalid SEA.DAT station identification')
        self.identifier=int(station[0]); self.name=station[1].strip()
        self.source_grid: GridConfig=GridConfig(1,1,1.,1.,projection=self.projection,datum=self.datum,**params)
        records: list[SeaRecord]=[]
        for line in lines[index:]:
            if not line.strip(): continue
            values=line.replace(',',' ').split()
            number=[float(item) for item in values]
            if modern:
                if len(number)!=21: raise ValueError('SEA.DAT 2.11 record must contain 21 fields')
                begin=_julian(*number[5:8]); end=_julian(*number[8:11])
                arguments=(*number[:5],*number[11:])
            elif version>=2.099:
                if len(number)!=19: raise ValueError('SEA.DAT 2.1 record must contain 19 fields')
                begin=_julian(*number[3:6]); end=_julian(*number[6:9])
                wave_period,wave_height=number[-2:]
                if wave_period==0.: raise ValueError('SEA.DAT wave period must not be zero')
                wave_period=-999. if wave_period>9998. else wave_period
                wave_height=-999. if wave_height>9998. else wave_height
                arguments=(*number[:3],9999.,9999.,*number[9:-2],wave_period,wave_height)
            else:
                if len(number)!=18: raise ValueError('legacy SEA.DAT record must contain 18 fields')
                begin=_julian(*number[4:7]); end=_julian(*number[7:10])
                # The old west-longitude field was superseded in v5.6; CALMET
                # derives longitude from the projected coordinates for all versions.
                arguments=(number[0],number[1],number[3],9999.,9999.,
                           *number[10:],9999.,9999.)
            if not begin<end: raise ValueError('SEA.DAT record interval must advance')
            records.append(SeaRecord(begin,end,*arguments))
        if not records: raise ValueError('SEA.DAT contains no meteorological records')
        for left,right in zip(records[:-1],records[1:]):
            if left.end!=right.begin: raise ValueError('SEA.DAT records contain a gap or overlap')
        if modern and (records[0].begin<cast(datetime,self.start) or
                       records[-1].end>cast(datetime,self.end)):
            raise ValueError('SEA.DAT records fall outside the declared header period')
        if not modern: self.start,self.end=records[0].begin,records[-1].end
        self.records: tuple[SeaRecord,...]=tuple(records)

    def at_begin(self,timestamp: datetime) -> SeaRecord:
        """Select the record whose half-open interval contains a timestamp.

        Args:
            timestamp: Naive timestamp in the SEA.DAT time basis.

        Returns:
            The covering station record.

        Raises:
            ValueError: If no record covers ``timestamp``.
        """
        record: SeaRecord | None
        record=next((item for item in self.records if item.begin<=timestamp<item.end),None)
        if record is None: raise ValueError(f'SEA.DAT has no record covering {timestamp}')
        return record

    def point(self,target_grid: GridConfig) -> Float64Array:
        """Transform the station coordinate to a target grid.

        Args:
            target_grid: Grid whose projection and datum define the result.

        Returns:
            Two-element ``float64`` array ``(x_km, y_km)``.
        """
        record=self.records[0]
        return np.asarray(transform_grid(record.x_km,record.y_km,
                                         self.source_grid,target_grid,units='km'),
                          dtype=SPATIAL_DTYPE)

    def longitude_latitude(self) -> tuple[float,float]:
        """Derive the station's WGS-84 longitude and latitude.

        Returns:
            Tuple ``(longitude_deg_east, latitude_deg_north)``.
        """
        record=self.records[0]
        return cast(
            tuple[float,float],
            unproject_geographic(record.x_km,record.y_km,self.source_grid,
                                 'WGS-84',units='km'),
        )


def grid_sea_mixing_height(
    values: ArrayLike,
    station_xy: ArrayLike,
    grid: GridConfig,
    block_size: int=65536,
) -> Float32Array:
    """Interpolate observed overwater mixing height from valid stations.

    Args:
        values: One mixing-height value per station in metres; values at or
            above 9998 are missing.
        station_xy: ``(station, 2)`` source coordinates in grid kilometres.
        grid: Target CALMET grid.
        block_size: Number of target cells processed per distance block.

    Returns:
        ``float32`` mixing-height array in metres with shape ``(ny, nx)``.
        The array is all NaN when no station is valid.

    Raises:
        ValueError: If station coordinates do not match ``values``.
    """
    data: Float32Array=np.asarray(values,dtype=COMPUTE_DTYPE)
    points: Float64Array=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    valid=data<COMPUTE_DTYPE(9998.)
    if points.shape!=(data.size,2): raise ValueError('SEA mixing-height coordinates do not match values')
    if not valid.any(): return np.full((grid.ny,grid.nx),np.nan,dtype=COMPUTE_DTYPE)
    # WATER has a separate exact-value branch only when NOWSTA itself is one;
    # with several stations it still divides the accumulators even if only one
    # station has a valid ZIOW.
    if data.size==1: return np.full((grid.ny,grid.nx),data[0],dtype=COMPUTE_DTYPE)
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=SPATIAL_DTYPE)
    relative=(points-origin)*1000.
    tx=(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dx_km)*1000.)
    ty=(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dy_km)*1000.)
    xx,yy=np.meshgrid(tx,ty); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output: Float32Array=np.empty(targets.shape[0],dtype=COMPUTE_DTYPE)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        # These all-station sums can span many magnitudes, so accumulate in
        # double precision and cast only the returned model field.
        xsum=np.zeros(end-begin,dtype=np.float64)
        ysum=np.zeros(end-begin,dtype=np.float64)
        for station in np.flatnonzero(valid):
            delta=relative[station]-targets[begin:end]
            distance2=np.maximum(np.sum(delta*delta,axis=1),1.e-30)
            xsum+=np.float64(data[station])/distance2
            ysum+=1./distance2
        output[begin:end]=np.divide(xsum,ysum,out=np.full_like(xsum,np.nan),
                                    where=xsum>0.).astype(COMPUTE_DTYPE)
    return output.reshape(grid.ny,grid.nx)


def grid_water_mixing_gradient(
    values: ArrayLike,
    station_xy: ArrayLike,
    grid: GridConfig,
    default: float,
    block_size: int=65536,
) -> Float32Array:
    """Interpolate the above-mixing-height gradient from SEA stations.

    Args:
        values: One temperature gradient per station in kelvin per metre;
            values at or above 9998 are missing.
        station_xy: ``(station, 2)`` coordinates in grid kilometres.
        grid: Target CALMET grid.
        default: Gradient used where no valid observation is available, in
            kelvin per metre.
        block_size: Number of target cells processed per distance block.

    Returns:
        ``float32`` gradient array with shape ``(ny, nx)``.

    Raises:
        ValueError: If station coordinates do not match ``values``.
    """
    data: Float32Array=np.asarray(values,dtype=COMPUTE_DTYPE)
    points: Float64Array=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    if points.shape!=(data.size,2):
        raise ValueError('SEA gradient coordinates do not match values')
    fallback=COMPUTE_DTYPE(default)
    valid=data<COMPUTE_DTYPE(9998.)
    if data.size==1:
        value=data[0] if valid[0] else fallback
        return np.full((grid.ny,grid.nx),value,dtype=COMPUTE_DTYPE)
    if not valid.any():
        return np.full((grid.ny,grid.nx),fallback,dtype=COMPUTE_DTYPE)
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=SPATIAL_DTYPE)
    relative=(points-origin)*1000.
    tx=(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dx_km)*1000.)
    ty=(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dy_km)*1000.)
    xx,yy=np.meshgrid(tx,ty); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output: Float32Array=np.empty(targets.shape[0],dtype=COMPUTE_DTYPE)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        xsum=np.zeros(end-begin,dtype=np.float64)
        ysum=np.zeros(end-begin,dtype=np.float64)
        for station in np.flatnonzero(valid):
            delta=relative[station]-targets[begin:end]
            distance2=np.maximum(np.sum(delta*delta,axis=1),1.e-30)
            xsum+=np.float64(data[station])/distance2
            ysum+=1./distance2
        output[begin:end]=np.where(xsum>0.,xsum/ysum,fallback).astype(COMPUTE_DTYPE)
    return output.reshape(grid.ny,grid.nx)


def _nearest_sea_indices(
    points: ArrayLike,
    valid: ArrayLike,
    grid: GridConfig,
) -> IndexArray:
    """Select the nearest valid SEA station for every target cell.

    Args:
        points: ``(station, 2)`` coordinates in grid kilometres.
        valid: Boolean station-validity mask.
        grid: Target CALMET grid.

    Returns:
        Flat integer array of original station indices with ``nx * ny``
        elements in row-major grid order.

    Raises:
        ValueError: If shapes are inconsistent or no candidate is valid.
    """
    points=np.asarray(points,dtype=SPATIAL_DTYPE)
    valid=np.asarray(valid,dtype=bool)
    if points.ndim!=2 or points.shape[1]!=2 or valid.shape!=(points.shape[0],):
        raise ValueError('SEA station coordinates and validity mask do not match')
    if not valid.any(): raise ValueError('nearest SEA station requires a valid candidate')
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=SPATIAL_DTYPE)
    relative=(points-origin)*1000.
    target_x=(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dx_km)*1000.)
    target_y=(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*(float(grid.dy_km)*1000.)
    xx,yy=np.meshgrid(target_x,target_y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    selected=_nearest_indices(relative[valid],targets)
    return np.flatnonzero(valid)[selected]


def nearest_sea_waves(
    records: Iterable[SeaRecord],
    station_xy: ArrayLike,
    grid: GridConfig,
) -> tuple[Float32Array,Float32Array]:
    """Grid WATERP's independently selected valid wave observations.

    Args:
        records: Station records for the current timestep.
        station_xy: ``(station, 2)`` coordinates in grid kilometres.
        grid: Target CALMET grid.

    Returns:
        Tuple of ``float32`` wave-period and wave-height arrays, each with
        shape ``(ny, nx)`` and units of seconds and metres respectively.
    """
    records=tuple(records); points=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    valid=np.array([(item.wave_period_s>=0.) and (item.wave_height_m>=0.)
                    for item in records])
    chosen=(_nearest_sea_indices(points,valid,grid) if valid.any() else
            np.zeros(grid.nx*grid.ny,dtype=np.intp))
    def field(name: str) -> Float32Array:
        """Gather one wave attribute at the selected station indices.

        Args:
            name: :class:`SeaRecord` wave-field name.

        Returns:
            ``float32`` field with shape ``(ny, nx)``.
        """
        return np.asarray([getattr(records[index],name) for index in chosen],
                          dtype=COMPUTE_DTYPE).reshape(grid.ny,grid.nx)
    return field('wave_period_s'),field('wave_height_m')


def nearest_sea_data(
    records: Iterable[SeaRecord],
    station_xy: ArrayLike,
    longitudes: ArrayLike,
    grid: GridConfig,
) -> GriddedSeaData | None:
    """Select WATER fields from the nearest station with a valid delta.

    Args:
        records: Station records for the current timestep.
        station_xy: ``(station, 2)`` coordinates in grid kilometres.
        longitudes: WGS-84 station longitudes in degrees east.
        grid: Target CALMET grid.

    Returns:
        Gridded overwater fields, or ``None`` when every station lacks a valid
        air-water temperature difference.

    Raises:
        ValueError: If station coordinates or longitudes do not match the
            record count.
    """
    records=tuple(records); points=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    longitude=np.asarray(longitudes,dtype=SPATIAL_DTYPE)
    if points.shape!=(len(records),2) or longitude.shape!=(len(records),):
        raise ValueError('SEA station metadata do not match records')
    valid=np.array([item.air_water_delta_k<9998. for item in records])
    if not valid.any(): return None
    chosen=_nearest_sea_indices(points,valid,grid)
    def field(name: str) -> Float32Array:
        """Gather one scalar observation at the primary selected stations.

        Args:
            name: :class:`SeaRecord` attribute name.

        Returns:
            ``float32`` field with shape ``(ny, nx)``.
        """
        return np.asarray([getattr(records[index],name) for index in chosen],
                          dtype=COMPUTE_DTYPE).reshape(grid.ny,grid.nx)
    air=field('air_temperature_k'); delta=field('air_water_delta_k')
    anemometer=field('anemometer_height_m')
    anemometer=np.where(anemometer<COMPUTE_DTYPE(9998.),anemometer,
                        COMPUTE_DTYPE(10.)).astype(COMPUTE_DTYPE)
    sensor=field('air_sensor_height_m')
    sensor=np.where(sensor<COMPUTE_DTYPE(9998.),sensor,anemometer).astype(COMPUTE_DTYPE)
    depth=field('water_sensor_depth_m')
    depth=np.where(depth<COMPUTE_DTYPE(9998.),depth,COMPUTE_DTYPE(.6)).astype(COMPUTE_DTYPE)
    humidity=field('relative_humidity')
    humidity=np.where(humidity<COMPUTE_DTYPE(9998.99),humidity,
                      COMPUTE_DTYPE(100.)).astype(COMPUTE_DTYPE)
    wave_valid=np.array([(item.wave_period_s>=0.) and (item.wave_height_m>=0.)
                         for item in records])
    if wave_valid.any():
        wave_chosen=_nearest_sea_indices(points,wave_valid,grid)
    else:
        # WATER leaves NOW=1 when every station has missing wave data.
        wave_chosen=np.zeros(chosen.size,dtype=np.intp)
    def wave_field(name: str) -> Float32Array:
        """Gather one wave observation at independently selected stations.

        Args:
            name: :class:`SeaRecord` wave-field name.

        Returns:
            ``float32`` field with shape ``(ny, nx)``.
        """
        return np.asarray([getattr(records[index],name) for index in wave_chosen],
                          dtype=COMPUTE_DTYPE).reshape(grid.ny,grid.nx)
    return GriddedSeaData(
        np.where(air<COMPUTE_DTYPE(9998.),air,
                 COMPUTE_DTYPE(288.7)).astype(COMPUTE_DTYPE),
        np.asarray(np.where(air<COMPUTE_DTYPE(9998.),air,
                            COMPUTE_DTYPE(288.7))-delta,dtype=COMPUTE_DTYPE),
        humidity,anemometer,sensor,depth,wave_field('wave_period_s'),wave_field('wave_height_m'),
        longitude[chosen].reshape(grid.ny,grid.nx))
