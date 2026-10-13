"""Formatted CALMET SEA.DAT 2.1/2.11 overwater observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np

from .config import GridConfig
from .datetime_utils import parse_ordinal
from .projections import transform_grid,unproject_geographic


def _julian(year,day,hour):
    return parse_ordinal(year,day,hour)


def _angle(text):
    match=re.search(r'([-+]?\d+(?:\.\d*)?)\s*([NSEW]?)',text.strip(),re.I)
    if not match: raise ValueError(f'invalid SEA.DAT projection angle: {text!r}')
    value=float(match.group(1))
    return -abs(value) if match.group(2).upper() in {'S','W'} else value


@dataclass(frozen=True,slots=True)
class SeaRecord:
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
    """Nearest valid SEA.DAT record fields on the CALMET grid."""
    air_temperature_k: np.ndarray
    sea_temperature_k: np.ndarray
    relative_humidity: np.ndarray
    anemometer_height_m: np.ndarray
    air_sensor_height_m: np.ndarray
    water_sensor_depth_m: np.ndarray
    wave_period_s: np.ndarray
    wave_height_m: np.ndarray
    station_longitude: np.ndarray


class SeaData:
    """Read one station per modern formatted SEA.DAT, as RDHDOW/RDOW do."""
    def __init__(self,path):
        self.path=Path(path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
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
        params={}
        if self.projection=='UTM':
            fields=lines[index].split(); index+=1
            params.update(utm_zone=int(fields[0]),hemisphere=fields[1].upper())
        elif self.projection=='LCC':
            line=lines[index].ljust(64); index+=1
            params.update(lat0=_angle(line[:16]),lon0=_angle(line[16:32]),
                          lat1=_angle(line[32:48]),lat2=_angle(line[48:64]))
            false=lines[index].split(); index+=1
            params.update(false_easting_km=float(false[0]),false_northing_km=float(false[1]))
        elif self.projection=='PS':
            line=lines[index].ljust(48); index+=1
            params.update(lat0=_angle(line[:16]),lon0=_angle(line[16:32]),
                          lat1=_angle(line[32:48]))
        elif self.projection=='EM':
            line=lines[index].ljust(32); index+=1
            params.update(lat0=_angle(line[:16]),lon0=_angle(line[16:32]))
        elif self.projection in {'LAZA','TTM'}:
            line=lines[index].ljust(32); index+=1
            params.update(lat0=_angle(line[:16]),lon0=_angle(line[16:32]))
            false=lines[index].split(); index+=1
            params.update(false_easting_km=float(false[0]),false_northing_km=float(false[1]))
        else:
            raise ValueError('SEA.DAT projection must be UTM, LCC, PS, EM, LAZA, or TTM')
        datum_line=lines[index]; index+=1
        self.datum=datum_line[:8].strip(); self.datum_date=datum_line[8:18].strip()
        units=lines[index][:4].strip().upper(); index+=1
        if units!='KM': raise ValueError('SEA.DAT coordinates must use KM')
        modern=version>=2.10999
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
        self.source_grid=GridConfig(1,1,1.,1.,projection=self.projection,datum=self.datum,**params)
        records=[]
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
        if modern and (records[0].begin<self.start or records[-1].end>self.end):
            raise ValueError('SEA.DAT records fall outside the declared header period')
        if not modern: self.start,self.end=records[0].begin,records[-1].end
        self.records=tuple(records)

    def at_begin(self,timestamp):
        record=next((item for item in self.records if item.begin<=timestamp<item.end),None)
        if record is None: raise ValueError(f'SEA.DAT has no record covering {timestamp}')
        return record

    def point(self,target_grid):
        """Transform the current record-independent station coordinate to target km."""
        record=self.records[0]
        return np.asarray(transform_grid(record.x_km,record.y_km,
                                         self.source_grid,target_grid,units='km'),dtype=float)

    def longitude_latitude(self):
        """Return station longitude/latitude derived from its projected coordinate."""
        record=self.records[0]
        return unproject_geographic(record.x_km,record.y_km,self.source_grid,
                                    'WGS-84',units='km')


def grid_sea_mixing_height(values,station_xy,grid,block_size=65536):
    """Reproduce WATER's all-station 1/R**2 interpolation of observed ZIOW."""
    f32=np.float64
    data=np.asarray(values,dtype=f32); points=np.asarray(station_xy,dtype=f32)
    valid=data<f32(9998.)
    if points.shape!=(data.size,2): raise ValueError('SEA mixing-height coordinates do not match values')
    if not valid.any(): return np.full((grid.ny,grid.nx),np.nan,dtype=f32)
    # WATER has a separate exact-value branch only when NOWSTA itself is one;
    # with several stations it still divides the accumulators even if only one
    # station has a valid ZIOW.
    if data.size==1: return np.full((grid.ny,grid.nx),data[0],dtype=f32)
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=f32)
    relative=np.asarray((points-origin)*f32(1000.),dtype=f32)
    tx=np.asarray((np.arange(grid.nx,dtype=f32)+f32(.5))*
                  (f32(grid.dx_km)*f32(1000.)),dtype=f32)
    ty=np.asarray((np.arange(grid.ny,dtype=f32)+f32(.5))*
                  (f32(grid.dy_km)*f32(1000.)),dtype=f32)
    xx,yy=np.meshgrid(tx,ty); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty(targets.shape[0],dtype=f32)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        xsum=np.zeros(end-begin,dtype=f32)
        ysum=np.zeros(end-begin,dtype=f32)
        for station in np.flatnonzero(valid):
            delta=np.asarray(relative[station]-targets[begin:end],dtype=f32)
            distance2=np.asarray(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1],
                                 dtype=f32)
            distance2=np.maximum(distance2,f32(1.e-30))
            xsum=np.asarray(xsum+data[station]/distance2,dtype=f32)
            ysum=np.asarray(ysum+f32(1.)/distance2,dtype=f32)
        output[begin:end]=np.divide(xsum,ysum,out=np.full_like(xsum,np.nan),
                                    where=xsum>f32(0.))
    return output.reshape(grid.ny,grid.nx)


def grid_water_mixing_gradient(values,station_xy,grid,default,block_size=65536):
    """Reproduce WATER's station handling for the above-ZI lapse rate."""
    f32=np.float64
    data=np.asarray(values,dtype=f32); points=np.asarray(station_xy,dtype=f32)
    if points.shape!=(data.size,2):
        raise ValueError('SEA gradient coordinates do not match values')
    fallback=f32(default)
    valid=data<f32(9998.)
    if data.size==1:
        value=data[0] if valid[0] else fallback
        return np.full((grid.ny,grid.nx),value,dtype=f32)
    if not valid.any():
        return np.full((grid.ny,grid.nx),fallback,dtype=f32)
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=f32)
    relative=np.asarray((points-origin)*f32(1000.),dtype=f32)
    tx=np.asarray((np.arange(grid.nx,dtype=f32)+f32(.5))*
                  (f32(grid.dx_km)*f32(1000.)),dtype=f32)
    ty=np.asarray((np.arange(grid.ny,dtype=f32)+f32(.5))*
                  (f32(grid.dy_km)*f32(1000.)),dtype=f32)
    xx,yy=np.meshgrid(tx,ty); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty(targets.shape[0],dtype=f32)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        xsum=np.zeros(end-begin,dtype=f32); ysum=np.zeros(end-begin,dtype=f32)
        for station in np.flatnonzero(valid):
            delta=np.asarray(relative[station]-targets[begin:end],dtype=f32)
            distance2=np.asarray(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1],
                                 dtype=f32)
            distance2=np.maximum(distance2,f32(1.e-30))
            xsum=np.asarray(xsum+data[station]/distance2,dtype=f32)
            ysum=np.asarray(ysum+f32(1.)/distance2,dtype=f32)
        output[begin:end]=np.where(xsum>f32(0.),
                                   np.asarray(xsum/ysum,dtype=f32),fallback)
    return output.reshape(grid.ny,grid.nx)


def _nearest_sea_indices(points,valid,grid):
    """Return WATER's first-on-tie nearest indices into a valid subset."""
    f32=np.float64
    points=np.asarray(points,dtype=f32); valid=np.asarray(valid,dtype=bool)
    origin=np.asarray((grid.x_origin_km,grid.y_origin_km),dtype=f32)
    relative=np.asarray((points-origin)*f32(1000.),dtype=f32)
    target_x=np.asarray((np.arange(grid.nx,dtype=f32)+f32(.5))*
                        f32(f32(grid.dx_km)*f32(1000.)),dtype=f32)
    target_y=np.asarray((np.arange(grid.ny,dtype=f32)+f32(.5))*
                        f32(f32(grid.dy_km)*f32(1000.)),dtype=f32)
    xx,yy=np.meshgrid(target_x,target_y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    candidates=relative[valid]
    selected=np.empty(targets.shape[0],dtype=np.intp)
    for begin in range(0,targets.shape[0],65536):
        delta=np.asarray(candidates[None,:,:]-targets[begin:begin+65536,None,:],dtype=f32)
        distance2=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)
        selected[begin:begin+65536]=np.argmin(distance2,axis=1)
    return np.flatnonzero(valid)[selected]


def nearest_sea_waves(records,station_xy,grid):
    """Grid WATERP's independently selected valid SEA wave observations."""
    records=tuple(records); points=np.asarray(station_xy,dtype=np.float64)
    valid=np.array([(item.wave_period_s>=0.) and (item.wave_height_m>=0.)
                    for item in records])
    chosen=(_nearest_sea_indices(points,valid,grid) if valid.any() else
            np.zeros(grid.nx*grid.ny,dtype=np.intp))
    def field(name):
        return np.asarray([getattr(records[index],name) for index in chosen],
                          dtype=np.float64).reshape(grid.ny,grid.nx)
    return field('wave_period_s'),field('wave_height_m')


def nearest_sea_data(records,station_xy,longitudes,grid):
    """Reproduce WATER's nearest station having a valid air-water delta."""
    records=tuple(records); points=np.asarray(station_xy,dtype=np.float64)
    longitude=np.asarray(longitudes,dtype=np.float64)
    if points.shape!=(len(records),2) or longitude.shape!=(len(records),):
        raise ValueError('SEA station metadata do not match records')
    valid=np.array([item.air_water_delta_k<9998. for item in records])
    if not valid.any(): return None
    f32=np.float64
    chosen=_nearest_sea_indices(points,valid,grid)
    def field(name):
        return np.asarray([getattr(records[index],name) for index in chosen],
                          dtype=f32).reshape(grid.ny,grid.nx)
    air=field('air_temperature_k'); delta=field('air_water_delta_k')
    anemometer=field('anemometer_height_m')
    anemometer=np.where(anemometer<9998.,anemometer,10.)
    sensor=field('air_sensor_height_m')
    sensor=np.where(sensor<9998.,sensor,anemometer)
    depth=field('water_sensor_depth_m'); depth=np.where(depth<9998.,depth,.6)
    humidity=field('relative_humidity')
    humidity=np.where(humidity<9998.99,humidity,100.)
    wave_valid=np.array([(item.wave_period_s>=0.) and (item.wave_height_m>=0.)
                         for item in records])
    if wave_valid.any():
        wave_chosen=_nearest_sea_indices(points,wave_valid,grid)
    else:
        # WATER leaves NOW=1 when every station has missing wave data.
        wave_chosen=np.zeros(chosen.size,dtype=np.intp)
    def wave_field(name):
        return np.asarray([getattr(records[index],name) for index in wave_chosen],
                          dtype=f32).reshape(grid.ny,grid.nx)
    return GriddedSeaData(
        np.where(air<f32(9998.),air,f32(288.7)),
        np.asarray(np.where(air<f32(9998.),air,f32(288.7))-delta,dtype=f32),
        humidity,anemometer,sensor,depth,wave_field('wave_period_s'),wave_field('wave_height_m'),
        longitude[chosen].reshape(grid.ny,grid.nx))
