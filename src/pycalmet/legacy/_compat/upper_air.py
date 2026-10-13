"""Formatted CALMET UP.DAT sounding reader."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import re
import numpy as np

from .surface_data import _coordinate


def _instant(year,day,hour,seconds=0):
    year=int(year); day=int(day); hour=int(hour); seconds=int(seconds)
    if year<100: year+=2000 if year<50 else 1900
    if not 1<=day<=366 or hour<0 or seconds<0: raise ValueError('invalid UP.DAT timestamp')
    return datetime(year,1,1)+timedelta(days=day-1,hours=hour,seconds=seconds)


def _year(value):
    value=int(value)
    return value+2000 if value<50 else (value+1900 if value<100 else value)


def _sounding_header(line):
    fields=line.replace(',',' ').split()
    if len(fields)>=6: return tuple(map(int,fields[:6]))
    padded=line.ljust(70)
    try:
        return (int(padded[9:17]),int(padded[22:24]),int(padded[24:26]),
                int(padded[26:28]),int(padded[28:30]),int(padded[65:70]))
    except ValueError as error: raise ValueError('invalid fixed-column UP.DAT sounding header') from error


def _sounding_header22(line):
    fields=line.replace(',',' ').split()
    if len(fields)>=12: return tuple(map(int,fields[:12]))
    padded=line.ljust(76)
    try:
        return (int(padded[9:17]),int(padded[21:25]),int(padded[25:29]),int(padded[29:32]),
                int(padded[32:35]),int(padded[35:40]),int(padded[44:48]),int(padded[48:52]),
                int(padded[52:55]),int(padded[55:58]),int(padded[58:63]),int(padded[71:76]))
    except ValueError as error: raise ValueError('invalid fixed-column UP.DAT 2.2 sounding header') from error


def _calendar_instant(year,month,day,hour,seconds=0):
    return datetime(_year(year),int(month),int(day))+timedelta(hours=int(hour),seconds=int(seconds))


@dataclass(frozen=True,slots=True)
class UpperAirStation:
    identifier: int
    name: str
    latitude: float
    longitude: float
    elevation_m: float


@dataclass(frozen=True,slots=True)
class Sounding:
    timestamp: datetime
    pressure_mb: np.ndarray
    height_msl: np.ndarray
    temperature_c: np.ndarray
    wind_direction: np.ndarray
    wind_speed: np.ndarray
    end_timestamp: datetime | None = None

    def converted(self,convergence_degrees=0.):
        """Apply RDUP MSL-to-LGL and meteorological wind conversion."""
        f32=np.float32
        if self.height_msl.size==0: raise ValueError('cannot convert an empty upper-air sounding')
        if np.any(self.height_msl>=9998.): raise ValueError('upper-air sounding contains missing heights')
        height=np.asarray(np.asarray(self.height_msl,dtype=f32)-
                          f32(self.height_msl[0]),dtype=f32)
        direction=np.asarray(self.wind_direction,dtype=f32)
        speed=np.asarray(self.wind_speed,dtype=f32)
        missing=np.asarray(f32(999.)-f32(.01),dtype=f32)
        valid=(direction<missing)&(speed<missing)
        # RDUP applies map convergence only when both encoded direction and
        # speed exceed 0.01. Preserve its zero-direction/calm branch.
        rotate=valid&(direction>f32(.01))&(speed>f32(.01))
        adjusted=np.asarray(direction-f32(convergence_degrees),dtype=f32)
        adjusted=np.where(adjusted<f32(0.),np.asarray(adjusted+f32(360.),dtype=f32),adjusted)
        adjusted=np.where(adjusted>f32(360.),np.asarray(adjusted-f32(360.),dtype=f32),adjusted)
        direction=np.where(rotate,adjusted,direction).astype(f32)
        radians=np.asarray(f32(.0174533)*direction,dtype=f32)
        sine=np.asarray(np.sin(radians.astype(np.float64)),dtype=f32)
        cosine=np.asarray(np.cos(radians.astype(np.float64)),dtype=f32)
        u=np.where(valid,np.asarray(-speed*sine,dtype=f32),f32(999.))
        v=np.where(valid,np.asarray(-speed*cosine,dtype=f32),f32(999.))
        return height,u,v


def layer_average(sounding,z_faces_m,convergence_degrees=0.):
    """Reproduce VERTAV's piecewise-linear layer wind integration."""
    f32=np.float32
    faces=np.asarray(z_faces_m,dtype=f32)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('upper-air layer faces must increase strictly from zero')
    height,u,v=sounding.converted(convergence_degrees)
    valid=(u<998.99)&(v<998.99)
    height=height[valid]; u=u[valid]; v=v[valid]
    if height.size<2 or np.any(np.diff(height)<=0.):
        raise ValueError('upper-air sounding needs two strictly increasing valid wind levels')
    if faces[-1]>height[-1]: raise ValueError('CALMET cell face is above top sounding level')
    # VERTAV makes all components positive, inserts each face in order, and
    # accumulates trapezoids sequentially in REAL*4.
    u=np.asarray(u+f32(1000.),dtype=f32); v=np.asarray(v+f32(1000.),dtype=f32)
    z=np.asarray(height,dtype=f32)
    for face in faces:
        if face==f32(0.) or np.any(z==face):
            continue
        upper=int(np.searchsorted(z,face,side='left'))
        if upper==0: raise ValueError('lowest valid upper-air wind is above a CALMET cell face')
        if upper>=z.size: raise ValueError('CALMET cell face is above top sounding level')
        ratio=np.asarray((z[upper]-face)/(z[upper]-z[upper-1]),dtype=f32)
        ubound=np.asarray(u[upper]-np.asarray((u[upper]-u[upper-1])*ratio,dtype=f32),dtype=f32)
        vbound=np.asarray(v[upper]-np.asarray((v[upper]-v[upper-1])*ratio,dtype=f32),dtype=f32)
        z=np.insert(z,upper,face); u=np.insert(u,upper,ubound); v=np.insert(v,upper,vbound)
    boundaries=[]
    start=0
    for face in faces:
        matches=np.flatnonzero(z[start:]==face)
        if not matches.size: raise ValueError('CALMET cell face is absent from upper-air profile')
        found=start+int(matches[0]); boundaries.append(found); start=found+1
    output_u=np.empty(faces.size-1,dtype=f32); output_v=np.empty_like(output_u)
    for layer,(bottom,top) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        sum_u=f32(0.); sum_v=f32(0.)
        for point in range(bottom+1,top+1):
            previous=point-1; dz=np.asarray(z[point]-z[previous],dtype=f32)
            uheight=np.asarray(np.minimum(u[previous],u[point])+
                f32(.5)*np.abs(np.asarray(u[point]-u[previous],dtype=f32)),dtype=f32)
            vheight=np.asarray(np.minimum(v[previous],v[point])+
                f32(.5)*np.abs(np.asarray(v[point]-v[previous],dtype=f32)),dtype=f32)
            sum_u=np.asarray(sum_u+np.asarray(uheight*dz,dtype=f32),dtype=f32)
            sum_v=np.asarray(sum_v+np.asarray(vheight*dz,dtype=f32),dtype=f32)
        inverse=np.asarray(f32(1.)/np.asarray(faces[layer+1]-faces[layer],dtype=f32),dtype=f32)
        output_u[layer]=np.asarray(np.asarray(sum_u*inverse,dtype=f32)-f32(1000.),dtype=f32)
        output_v[layer]=np.asarray(np.asarray(sum_v*inverse,dtype=f32)-f32(1000.),dtype=f32)
    return output_u,output_v


def temperature_faces(sounding,z_faces_m):
    """Reproduce FACET/INTP temperature interpolation at CALMET faces."""
    f32=np.float32
    faces=np.asarray(z_faces_m,dtype=f32)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('upper-air temperature faces must increase strictly from zero')
    raw_height=np.asarray(sounding.height_msl,dtype=f32)
    height=np.asarray(raw_height-f32(raw_height[0]),dtype=f32)
    temperature=np.asarray(sounding.temperature_c,dtype=f32)
    output=np.empty(faces.size,dtype=f32)
    missing=np.asarray(f32(999.9)-f32(.01),dtype=f32)
    for index,face in enumerate(faces):
        upper=next((j for j in range(1,height.size)
                    if height[j]>=face and temperature[j]<missing),None)
        if upper is None: raise ValueError('temperature interpolation face is above top valid sounding level')
        lower=next((j for j in range(upper-1,-1,-1) if temperature[j]<missing),None)
        if lower is None: raise ValueError('bottom of upper-air temperature sounding is missing')
        if height[lower]>face: raise ValueError('sounding below temperature interpolation face is missing')
        depth=np.asarray(height[upper]-height[lower],dtype=f32)
        if abs(depth)<f32(1.e-5):
            output[index]=np.asarray(f32(.5)*np.asarray(
                temperature[upper]+temperature[lower],dtype=f32),dtype=f32)
        else:
            ratio=np.asarray((height[upper]-face)/depth,dtype=f32)
            difference=np.asarray(temperature[upper]-temperature[lower],dtype=f32)
            output[index]=np.asarray(temperature[upper]-np.asarray(
                difference*ratio,dtype=f32),dtype=f32)
    return output


def grid_temperature_faces(station_faces,station_xy,grid,power=1,block_size=65536):
    """Horizontally interpolate TEMP3D upper-air faces from every station."""
    f32=np.float32
    values=np.asarray(station_faces,dtype=f32)
    points=np.asarray(station_xy,dtype=f32)
    if values.ndim!=2 or points.shape!=(values.shape[0],2) or values.shape[0]<1:
        raise ValueError('upper-air temperature stations and face profiles do not match')
    if power not in {1,2}: raise ValueError('upper-air temperature power must be 1 or 2')
    tx=np.asarray(f32(grid.x_origin_km)+(np.arange(grid.nx,dtype=f32)+f32(.5))*
                  f32(grid.dx_km),dtype=f32)
    ty=np.asarray(f32(grid.y_origin_km)+(np.arange(grid.ny,dtype=f32)+f32(.5))*
                  f32(grid.dy_km),dtype=f32)
    xx,yy=np.meshgrid(tx,ty)
    targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty((values.shape[1],targets.shape[0]),dtype=f32)
    scale=np.array([grid.dx_km,grid.dy_km],dtype=f32)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        delta=np.asarray((targets[begin:end,None,:]-points[None,:,:])/scale,dtype=f32)
        metric=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)
        if power==1: metric=np.asarray(np.sqrt(metric),dtype=f32)
        weights=np.asarray(f32(1.)/np.maximum(metric,f32(1.)),dtype=f32)
        denominator=np.zeros(end-begin,dtype=f32)
        for station in range(points.shape[0]):
            denominator=np.asarray(denominator+weights[:,station],dtype=f32)
        inverse=np.asarray(f32(1.)/denominator,dtype=f32)
        result=np.zeros((values.shape[1],end-begin),dtype=f32)
        for station in range(points.shape[0]):
            weight=np.asarray(weights[:,station]*inverse,dtype=f32)
            result=np.asarray(result+values[station,:,None]*weight[None,:],dtype=f32)
        output[:,begin:end]=result
    return output.reshape((values.shape[1],grid.ny,grid.nx))


def _prepdi_time_interpolate(left,right,seconds_to_right,total_seconds):
    """PREPDI's ``BB-(BB-AA)*XFACT`` REAL*4 interpolation."""
    f32=np.float32
    first=np.asarray(left,dtype=f32); second=np.asarray(right,dtype=f32)
    remaining=np.asarray(f32(seconds_to_right)/f32(total_seconds),dtype=f32)
    delta=np.asarray(second-first,dtype=f32)
    weighted=np.asarray(delta*remaining,dtype=f32)
    return np.asarray(second-weighted,dtype=f32)


class UpperAirData:
    """Read one modern formatted UP.DAT station file (RDHDU/RDUP)."""
    def __init__(self,path,*,station_metadata=None,datum=None):
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty UP.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='UP.DAT': raise ValueError('invalid UP.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version in {'2.1','2.2'}
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid UP.DAT comment header') from error
        try: projection=lines[index][:8].strip().upper(); index+=1
        except IndexError as error: raise ValueError('incomplete UP.DAT header') from error
        if projection not in {'LL','NONE'}: raise ValueError('UP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('upper-air station location is present in both control file and UP.DAT')
            datum_line=lines[index]; index+=1; self.datum=datum_line[:8].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('UP.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('UP.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum)
        if explicit:
            self.timezone=lines[index][:8].strip(); index+=1
            if self.timezone.upper() not in {'UTC+0000','UTC-0000'}:
                raise ValueError('UP.DAT 2.1/2.2 sounding times must use UTC+0000')
        else: self.timezone='UTC+0000'
        span=lines[index].replace(',',' ').split(); index+=1
        expected=11 if explicit else 9
        if len(span)!=expected: raise ValueError(f'UP.DAT time-span header must contain {expected} values')
        if explicit:
            self.start=_instant(*span[:4]); self.end=_instant(*span[4:8]); offset=8
        else:
            self.start=_instant(span[0],span[1],span[2]); self.end=_instant(span[3],span[4],span[5]); offset=6
        self.top_pressure_mb=float(span[offset]); self.source_type=int(span[offset+1]); self.record_format=int(span[offset+2])
        if self.record_format not in {1,2}: raise ValueError('UP.DAT record format must be 1 or 2')
        flags=re.findall(r'[TtFf]',lines[index]); index+=1
        if len(flags)!=4: raise ValueError('UP.DAT data-selection flags must contain four logical values')
        self.required_fields=tuple(value.upper()=='T' for value in flags)
        if projection=='LL':
            station=lines[index].replace(',',' ').split(); index+=1
            if len(station)<5: raise ValueError('invalid UP.DAT station metadata')
            self.station=UpperAirStation(int(station[0]),station[1],_coordinate(station[2],True),
                                         _coordinate(station[3],False),float(station[4]))
        else:
            self.station=station_metadata
        soundings=[]
        while index<len(lines):
            if not lines[index].strip(): index+=1; continue
            if self.version=='2.2':
                values=_sounding_header22(lines[index])
                identifier=values[0]; begin=_calendar_instant(*values[1:6]); end=_calendar_instant(*values[6:11]); levels=values[11]
                timestamp=begin
            else:
                identifier,year,month,day,hour,levels=_sounding_header(lines[index])
                timestamp=datetime(_year(year),month,day,hour)
            index+=1
            if identifier!=self.station.identifier: raise ValueError('UP.DAT sounding station ID does not match header')
            if levels<0: raise ValueError('UP.DAT sounding level count cannot be negative')
            values=[]
            while len(values)<5*levels and index<len(lines):
                values.extend(float(item) for item in lines[index].replace('/',' ').replace(',',' ').split())
                index+=1
            if len(values)!=5*levels: raise ValueError('UP.DAT sounding has an incomplete or overfull level list')
            table=np.asarray(values,dtype=float).reshape(levels,5)
            if timestamp<self.start or timestamp>self.end: raise ValueError('UP.DAT sounding lies outside declared time span')
            soundings.append(Sounding(timestamp,*(table[:,column].copy() for column in range(5)),
                                      end if self.version=='2.2' and end!=begin else None))
        if not soundings: raise ValueError('UP.DAT contains no soundings')
        if any(right.timestamp<=left.timestamp for left,right in zip(soundings[:-1],soundings[1:])):
            raise ValueError('UP.DAT soundings must be strictly chronological')
        averaged=[item.end_timestamp is not None for item in soundings]
        if any(averaged) and not all(averaged): raise ValueError('UP.DAT 2.2 cannot mix snapshot and time-averaged soundings')
        if all(averaged) and any(item.end_timestamp<=item.timestamp for item in soundings):
            raise ValueError('UP.DAT averaged sounding interval must be positive')
        self.soundings=tuple(soundings)

    def at(self,timestamp):
        matches=[item for item in self.soundings if item.timestamp==timestamp]
        if not matches: raise ValueError(f'UP.DAT has no sounding at {timestamp!s}')
        return matches[0]

    def bracket(self,timestamp):
        """Return RDUP's two soundings bracketing an ending timestamp."""
        times=np.asarray([item.timestamp.timestamp() for item in self.soundings])
        target=timestamp.timestamp(); right=int(np.searchsorted(times,target,side='left'))
        if right>=len(self.soundings): raise ValueError('UP.DAT does not extend beyond requested time')
        if times[right]==target:
            if right==0: raise ValueError('UP.DAT has no earlier sounding for requested time')
            left=right-1
        else:
            left=right-1
            if left<0: raise ValueError('UP.DAT does not begin before requested time')
        return self.soundings[left],self.soundings[right]

    def layer_wind_at(self,timestamp,z_faces_m,convergence_degrees=0.):
        """VERTAV each bracketing profile and apply PREPDI time interpolation."""
        if self.soundings[0].end_timestamp is not None:
            # RDUPN2 exposes an averaged profile unchanged at both interval
            # endpoints; at a shared boundary the earlier record is BB.
            sounding=next((item for item in self.soundings
                           if item.timestamp<timestamp<=item.end_timestamp),None)
            if sounding is None: raise ValueError('UP.DAT averaged records do not cover requested time')
            return layer_average(sounding,z_faces_m,convergence_degrees)
        left,right=self.bracket(timestamp)
        total=int((right.timestamp-left.timestamp).total_seconds())
        remaining=int((right.timestamp-timestamp).total_seconds())
        ua,va=layer_average(left,z_faces_m,convergence_degrees)
        ub,vb=layer_average(right,z_faces_m,convergence_degrees)
        return (_prepdi_time_interpolate(ua,ub,remaining,total),
                _prepdi_time_interpolate(va,vb,remaining,total))

    def temperature_faces_at(self,timestamp,z_faces_m):
        """FACET each bracketing profile and apply TEMP3D time interpolation."""
        if self.soundings[0].end_timestamp is not None:
            sounding=next((item for item in self.soundings
                           if item.timestamp<timestamp<=item.end_timestamp),None)
            if sounding is None: raise ValueError('UP.DAT averaged records do not cover requested time')
            return temperature_faces(sounding,z_faces_m)
        left,right=self.bracket(timestamp)
        first=temperature_faces(left,z_faces_m); second=temperature_faces(right,z_faces_m)
        total=int((right.timestamp-left.timestamp).total_seconds())
        remaining=int((right.timestamp-timestamp).total_seconds())
        return _prepdi_time_interpolate(first,second,remaining,total)

    def lapse_rate_at(self,timestamp,height_m,last_convective_time=None):
        """CGAMMA lapse rate, including its saved last-convective-hour correction."""
        f32=np.float32
        height=f32(height_m)
        if height<=0.: raise ValueError('CGAMMA height must be positive')
        current=self.temperature_faces_at(timestamp,(0.,height))
        ordinary=np.asarray(np.asarray(current[1]-current[0],dtype=f32)/height,dtype=f32)
        if last_convective_time is None or self.soundings[0].end_timestamp is not None:
            return float(ordinary)
        left,right=self.bracket(timestamp)
        if not (left.timestamp<last_convective_time<timestamp<right.timestamp): return float(ordinary)
        first=temperature_faces(left,(0.,height)); second=temperature_faces(right,(0.,height))
        total=int((right.timestamp-left.timestamp).total_seconds())
        convective_to_right=int((right.timestamp-last_convective_time).total_seconds())
        convective_surface=_prepdi_time_interpolate(
            first[0],second[0],convective_to_right,total)
        convective_upper=np.asarray(convective_surface-
            np.asarray(f32(.0098)*height,dtype=f32),dtype=f32)
        current_to_right=int((right.timestamp-timestamp).total_seconds())
        upper=_prepdi_time_interpolate(
            convective_upper,second[1],current_to_right,convective_to_right)
        return float(np.asarray(np.asarray(upper-current[0],dtype=f32)/height,dtype=f32))
