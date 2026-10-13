"""Formatted CALMET SURF.DAT 2.1 reader and interval lookup."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import re
import shlex
import struct
import numpy as np

from .projections import project_geographic
from .legacy_io import FortranSequentialReader


@dataclass(frozen=True,slots=True)
class SurfaceStation:
    identifier: int
    name: str
    latitude: float
    longitude: float
    anemometer_height_m: float


@dataclass(frozen=True,slots=True)
class SurfaceRecord:
    begin: datetime
    end: datetime
    wind_speed: np.ndarray
    wind_direction: np.ndarray
    ceiling_hundreds_ft: np.ndarray
    cloud_tenths: np.ndarray
    temperature_k: np.ndarray
    relative_humidity: np.ndarray
    pressure_mb: np.ndarray
    precipitation_code: np.ndarray


def unpack_surface(words):
    """Vectorized CALMET UNPCKS decimal-field decoding (3 words/station)."""
    packed=np.asarray(words,dtype=np.int64)
    if packed.ndim!=2 or packed.shape[1]!=3 or np.any(packed<0):
        raise ValueError('packed surface data must have shape (station,3) and be non-negative')
    word1,word2,word3=packed.T
    temperature=word1//100000
    precipitation=(word1//1000)-temperature*100
    humidity=word1-temperature*100000-precipitation*1000
    pressure=word2//100000
    cloud=(word2//1000)-pressure*100
    direction=word2-pressure*100000-cloud*1000
    ceiling=word3//10000
    speed=word3-ceiling*10000
    return dict(
        wind_speed=np.where(speed==9999,9999.,speed/100.),
        wind_direction=np.where(direction==999,9999.,direction.astype(float)),
        ceiling_hundreds_ft=ceiling.astype(np.int32),
        cloud_tenths=np.where(cloud==99,9999,cloud).astype(np.int32),
        temperature_k=np.where(temperature==9999,9999.,temperature/10.),
        relative_humidity=np.where(humidity==999,9999,humidity).astype(np.int32),
        pressure_mb=np.where(pressure==9999,9999.,pressure/10.),
        precipitation_code=np.where(precipitation==99,9999,precipitation).astype(np.int32),
    )


def _instant(year,day,hour,seconds):
    year=int(year); year=year+2000 if year<50 else (year+1900 if year<100 else year)
    day,hour,seconds=int(day),int(hour),int(seconds)
    if not 1<=day<=366 or hour<0 or seconds<0: raise ValueError('invalid SURF.DAT timestamp')
    return datetime(year,1,1)+timedelta(days=day-1,hours=hour,seconds=seconds)


def _coded_instant(code,seconds=0):
    text=str(int(code)); year_digits=len(text)-5
    if year_digits not in {2,4}: raise ValueError(f'invalid SURF.DAT coded time {code!r}')
    return _instant(text[:year_digits],text[year_digits:year_digits+3],text[-2:],seconds)


def _coordinate(text,latitude):
    value=re.sub(r'\s+','',text.upper())
    match=re.fullmatch(r'([NSEW]?)([+-]?\d+(?:\.\d*)?)([NSEW]?)',value)
    if not match: raise ValueError(f'unsupported SURF.DAT coordinate {text!r}')
    if match.group(1) and match.group(3): raise ValueError(f'multiple SURF.DAT hemispheres: {text!r}')
    number=float(match.group(2)); hemisphere=match.group(1) or match.group(3)
    if hemisphere in {'S','W'}: number=-abs(number)
    elif hemisphere in {'N','E'}: number=abs(number)
    limit=90. if latitude else 180.
    if not -limit<=number<=limit: raise ValueError(f'SURF.DAT coordinate outside valid range: {text!r}')
    return number


class SurfaceData:
    """Read a formatted, unpacked SURF.DAT dataset."""
    def __init__(self,path,*,station_metadata=None,datum=None):
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty SURF.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='SURF.DAT': raise ValueError('invalid SURF.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version=='2.1'
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid SURF.DAT comment header') from error
        try: projection=lines[index][:8].strip().upper(); index+=1
        except IndexError as error: raise ValueError('incomplete SURF.DAT header') from error
        if projection not in {'LL','NONE'}: raise ValueError('SURF.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('surface station locations are present in both control file and SURF.DAT')
            datum_line=lines[index]; index+=1
            self.datum=datum_line[:8].strip(); self.datum_date=datum_line[8:20].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('SURF.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('SURF.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum); self.datum_date=''
        if explicit:
            self.timezone=lines[index][:8].strip(); index+=1
        header=shlex.split(lines[index]); index+=1
        values=[int(item) for item in header]
        if explicit:
            if len(values)!=9: raise ValueError('SURF.DAT 2.1 time-span header must contain 9 integers')
            self.start=_instant(*values[:4]); self.end=_instant(*values[4:8]); count=values[8]
        else:
            if len(values)!=8: raise ValueError('legacy SURF.DAT time-span header must contain 8 integers')
            first_end=_instant(values[0],values[1],values[2],0)
            self.start=first_end-timedelta(hours=1)
            self.end=_instant(values[3],values[4],values[5],0)
            self.timezone=values[6]; count=values[7]
        if count<1: raise ValueError('SURF.DAT must contain at least one station')
        stations=[]
        if projection=='LL':
            for _ in range(count):
                try: fields=shlex.split(lines[index]); index+=1
                except IndexError as error: raise ValueError('incomplete SURF.DAT station list') from error
                if len(fields)!=5: raise ValueError('SURF.DAT station row must contain ID, name, latitude, longitude, and height')
                stations.append(SurfaceStation(int(fields[0]),fields[1],_coordinate(fields[2],True),
                                               _coordinate(fields[3],False),float(fields[4])))
        else:
            stations=list(station_metadata)
            if len(stations)!=count: raise ValueError('SURF.DAT station count does not match control-file metadata')
            identifiers=[]
            while len(identifiers)<count and index<len(lines):
                identifiers.extend(int(item) for item in shlex.split(lines[index])); index+=1
            if len(identifiers)!=count: raise ValueError('SURF.DAT NONE station ID list has wrong length')
            if identifiers!=[item.identifier for item in stations]:
                raise ValueError('SURF.DAT station IDs do not match control-file metadata')
        if len({station.identifier for station in stations})!=count: raise ValueError('duplicate SURF.DAT station ID')
        self.stations=tuple(stations)
        tokens=shlex.split('\n'.join(lines[index:]),comments=False)
        width=(8 if explicit else 3)+8*count
        if len(tokens)%width: raise ValueError('SURF.DAT contains an incomplete meteorological record')
        self.records=[]
        for offset in range(0,len(tokens),width):
            row=tokens[offset:offset+width]
            if explicit:
                begin=_instant(*row[:4]); end=_instant(*row[4:8]); value_offset=8
                if not begin<end or end-begin>timedelta(hours=1):
                    raise ValueError('SURF.DAT record interval must advance by at most one hour')
            else:
                end=_instant(row[0],row[1],row[2],0); begin=end-timedelta(hours=1); value_offset=3
            raw=np.asarray(row[value_offset:],dtype=float).reshape(count,8)
            self.records.append(SurfaceRecord(begin,end,raw[:,0].copy(),raw[:,1].copy(),
                raw[:,2].astype(np.int32),raw[:,3].astype(np.int32),raw[:,4].copy(),
                raw[:,5].astype(np.int32),raw[:,6].copy(),raw[:,7].astype(np.int32)))
        if not self.records: raise ValueError('SURF.DAT contains no meteorological records')
        for left,right in zip(self.records[:-1],self.records[1:]):
            if left.end!=right.begin: raise ValueError('SURF.DAT records contain a gap or overlap')
        if self.records[0].begin<self.start or self.records[-1].end>self.end:
            raise ValueError('SURF.DAT records fall outside the declared header period')

    def at_begin(self,when):
        for record in self.records:
            if record.begin<=when<record.end: return record
        raise ValueError(f'SURF.DAT has no record covering {when}')


def read_unformatted_surface(path,*,station_metadata=None,datum=None):
    """Read legacy or 2.1 GNU-Fortran sequential-unformatted SURF.DAT."""
    source=Path(path); result=object.__new__(SurfaceData); result.path=source
    with FortranSequentialReader(source) as reader:
        def record(size=None,label='record'):
            payload=reader.read_record()
            if payload is None: raise ValueError(f'truncated unformatted SURF.DAT {label}')
            if size is not None and len(payload)!=size: raise ValueError(f'unformatted SURF.DAT {label} has wrong size')
            return payload
        head=record(96,'identity')
        if head[:16].decode('ascii').strip().upper()!='SURF.DAT': raise ValueError('invalid SURF.DAT dataset label')
        result.version=head[16:32].decode('ascii').strip()
        explicit=result.version=='2.1'
        comments=struct.unpack('<i',record(4,'comment count'))[0]
        if comments<0: raise ValueError('negative SURF.DAT comment count')
        for _ in range(comments): record(80,'comment')
        projection=record(8,'projection').decode('ascii').strip().upper()
        if projection not in {'LL','NONE'}: raise ValueError('SURF.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('surface station locations are present in both control file and SURF.DAT')
            datum_record=record(20,'datum'); result.datum=datum_record[:8].decode('ascii').strip(); result.datum_date=datum_record[8:].decode('ascii').strip()
            # RDHD reads this five-integer record for every unformatted LL file.
            compatibility=record(20,'legacy compatibility header')
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',compatibility)
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        else:
            if station_metadata is None or datum is None:
                raise ValueError('SURF.DAT NONE requires control-file station metadata and datum')
            result.datum=str(datum); result.datum_date=''
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',record(20,'time span'))
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        if count<1 or packing not in {0,1}: raise ValueError('invalid SURF.DAT station count or packing code')
        identifiers=np.frombuffer(record(4*count,'station IDs'),dtype='<i4')
        if projection=='LL':
            names=record(4*count,'station names')
            latitudes=record(16*count,'station latitudes'); longitudes=record(16*count,'station longitudes')
            heights=np.frombuffer(record(4*count,'anemometer heights'),dtype='<f4')
            result.stations=tuple(SurfaceStation(int(identifiers[i]),names[4*i:4*i+4].decode('ascii').strip(),
                _coordinate(latitudes[16*i:16*i+16].decode('ascii'),True),
                _coordinate(longitudes[16*i:16*i+16].decode('ascii'),False),float(heights[i])) for i in range(count))
        else:
            result.stations=tuple(station_metadata)
            if len(result.stations)!=count or identifiers.tolist()!=[item.identifier for item in result.stations]:
                raise ValueError('SURF.DAT station IDs do not match control-file metadata')
        result.records=[]
        while (payload:=reader.read_record()) is not None:
            prefix_size=16 if explicit else 4
            expected=prefix_size+(12 if packing else 32)*count
            if len(payload)!=expected: raise ValueError('unformatted SURF.DAT meteorological record has wrong size')
            if explicit:
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[:16])
                begin,end=_coded_instant(begin_code,begin_sec),_coded_instant(end_code,end_sec)
            else:
                end_code=struct.unpack('<i',payload[:4])[0]
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            if not begin<end or end-begin>timedelta(hours=1):
                raise ValueError('SURF.DAT record interval must advance by at most one hour')
            if packing:
                fields=unpack_surface(np.frombuffer(payload,dtype='<i4',offset=prefix_size).reshape(3,count).T)
            else:
                offset=prefix_size
                def take(dtype):
                    nonlocal offset
                    values=np.frombuffer(payload,dtype=dtype,count=count,offset=offset).copy(); offset+=4*count
                    return values
                fields=dict(wind_speed=take('<f4'),wind_direction=take('<f4'),
                    ceiling_hundreds_ft=take('<i4'),cloud_tenths=take('<i4'),temperature_k=take('<f4'),
                    relative_humidity=take('<i4'),pressure_mb=take('<f4'),precipitation_code=take('<i4'))
            result.records.append(SurfaceRecord(begin,end,**fields))
    if not result.records: raise ValueError('SURF.DAT contains no meteorological records')
    for left,right in zip(result.records[:-1],result.records[1:]):
        if left.end!=right.begin: raise ValueError('SURF.DAT records contain a gap or overlap')
    if result.records[0].begin<result.start or result.records[-1].end>result.end:
        raise ValueError('SURF.DAT records fall outside the declared header period')
    return result


def load_surface(path,format=2,*,station_metadata=None,datum=None):
    """Load SURF.DAT according to CALMET IFORMS (1=binary, 2=text)."""
    if format==1: return read_unformatted_surface(path,station_metadata=station_metadata,datum=datum)
    if format==2: return SurfaceData(path,station_metadata=station_metadata,datum=datum)
    raise ValueError('SURF.DAT format/IFORMS must be 1 or 2')


def station_grid_mapping(stations,grid,datum):
    """Return station map coordinates and CALMET's nearest-station grid."""
    sx,sy=project_geographic([item.longitude for item in stations],
                             [item.latitude for item in stations],grid,datum,units='km')
    points=np.asarray(np.column_stack((sx,sy)),dtype=np.float32)
    return points,_nearest_station_grid(points,grid)


def _nearest_station_grid(points,grid,block_size=65536):
    """SETCOM's REAL*4 nearest-station scan, returned as zero-based IDs."""
    f32=np.float32
    points=np.asarray(points,dtype=f32)
    if points.ndim!=2 or points.shape[1]!=2 or points.shape[0]<1:
        raise ValueError('nearest-station coordinates must have shape (station, 2)')
    origin=np.array([grid.x_origin_km,grid.y_origin_km],dtype=f32)
    relative=np.asarray(points-origin,dtype=f32)
    x=np.asarray((np.arange(grid.nx,dtype=f32)+f32(.5))*f32(grid.dx_km),dtype=f32)
    y=np.asarray((np.arange(grid.ny,dtype=f32)+f32(.5))*f32(grid.dy_km),dtype=f32)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    nearest=np.empty(targets.shape[0],dtype=np.int32)
    for start in range(0,targets.shape[0],block_size):
        delta=np.asarray(relative[None,:,:]-targets[start:start+block_size,None,:],dtype=f32)
        distance=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)
        # argmin retains the first station on a tie, matching strict .LT.
        nearest[start:start+block_size]=np.argmin(distance,axis=1)
    return nearest.reshape(grid.ny,grid.nx)


def fill_missing_surface(record,station_xy,*,cloud_method=1,itprog=0):
    """Vectorized MISSFC nearest-valid-station replacement."""
    f32=np.float32
    coordinates=np.asarray(station_xy,dtype=f32)
    count=record.wind_speed.size
    if coordinates.shape!=(count,2): raise ValueError('surface station coordinates have wrong shape')
    delta=np.asarray(coordinates[:,None,:]-coordinates[None,:,:],dtype=f32)
    distance=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)

    def filled(values,required,integer=False):
        source=np.asarray(values) if integer else np.asarray(values,dtype=f32)
        missing=(source>=np.asarray(f32(9999.)-f32(.01),dtype=f32)
                 if not integer else source>=9999)
        if not required or not missing.any(): return source.copy()
        valid=~missing
        if not valid.any(): raise ValueError('surface variable is missing at every station')
        candidates=np.where(valid[None,:],distance,np.inf)
        replacement=source[np.argmin(candidates,axis=1)]
        return np.where(missing,replacement,source)

    return SurfaceRecord(record.begin,record.end,record.wind_speed.copy(),record.wind_direction.copy(),
        filled(record.ceiling_hundreds_ft,cloud_method<3,True).astype(np.int32),
        filled(record.cloud_tenths,cloud_method<3,True).astype(np.int32),
        filled(record.temperature_k,itprog!=2),
        filled(record.relative_humidity,True,True).astype(np.int32),
        filled(record.pressure_mb,True),record.precipitation_code.copy())


def surface_cloud_fields(record,nearest_station,external_cloud=None):
    """Vectorized OUTCLD/PGTSTB nearest-station cloud and ceiling fields."""
    nearest=np.asarray(nearest_station)
    if nearest.ndim!=2 or np.any((nearest<0)|(nearest>=record.cloud_tenths.size)):
        raise ValueError('nearest surface-station index is invalid')
    ceiling=np.asarray(record.ceiling_hundreds_ft)[nearest]
    if external_cloud is None:
        cloud=np.float32(.1)*np.asarray(record.cloud_tenths,dtype='f4')[nearest]
    else:
        cloud=np.asarray(external_cloud,dtype='f4')
        if cloud.shape!=nearest.shape: raise ValueError('external cloud field does not match CALMET grid')
    return cloud.astype('f4'),ceiling.astype(np.int32)


def interpolate_surface_scalar(values,station_xy,grid,*,radius_km=500.,maximum_stations=5,
                               power=1,target_water=None,station_water=None,empty_fallback=None):
    """Vectorized SURFVAR radius/nearest interpolation for T and RH."""
    f32=np.float32
    values=np.asarray(values,dtype=f32); points=np.asarray(station_xy,dtype=f32)
    if points.shape!=(values.size,2): raise ValueError('surface scalar station coordinates have wrong shape')
    if power not in {1,2} or radius_km<0. or maximum_stations<1:
        raise ValueError('surface scalar interpolation requires power 1/2, non-negative radius and positive station count')
    water=np.zeros(values.size,dtype=bool) if station_water is None else np.asarray(station_water,dtype=bool)
    if water.shape!=values.shape: raise ValueError('station water mask has wrong shape')
    target_mask=np.zeros((grid.ny,grid.nx),dtype=bool) if target_water is None else np.asarray(target_water,dtype=bool)
    if target_mask.shape!=(grid.ny,grid.nx): raise ValueError('target water mask has wrong shape')
    x=np.asarray(f32(grid.x_origin_km)+(np.arange(grid.nx,dtype=f32)+f32(.5))*
                 f32(grid.dx_km),dtype=f32)
    y=np.asarray(f32(grid.y_origin_km)+(np.arange(grid.ny,dtype=f32)+f32(.5))*
                 f32(grid.dy_km),dtype=f32)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty(targets.shape[0],dtype=f32); flat_water=target_mask.ravel()
    valid=values<np.asarray(f32(9999.)-f32(.01),dtype=f32)
    normalized_radius=np.asarray(f32(radius_km)/f32(grid.dx_km),dtype=f32)
    for start in range(0,targets.shape[0],65536):
        stop=min(start+65536,targets.shape[0]); group=flat_water[start:stop]
        delta=np.asarray((targets[start:stop,None,:]-points[None,:,:])/
                         f32(grid.dx_km),dtype=f32)
        distance=np.asarray(np.sqrt(np.asarray(
            delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)),dtype=f32)
        eligible=valid[None,:]&(water[None,:]==group[:,None]); available=eligible.any(axis=1)
        if np.any(~available) and empty_fallback is None:
            raise ValueError('grid cell has no valid land/water surface temperature station')
        if not available.any():
            fallback=np.asarray(empty_fallback,dtype=f32)
            output[start:stop]=np.where(group,fallback[1],fallback[0]); continue
        # Give unavailable rows a harmless valid index; their result is replaced below.
        eligible[~available,0]=True
        ordered=np.argsort(np.where(eligible,distance,np.inf),axis=1,kind='stable')
        nearest=ordered[:,0]; nearest_distance=np.take_along_axis(distance,nearest[:,None],axis=1)[:,0]
        selected=ordered[:,:min(maximum_stations,values.size)]
        selected_distance=np.take_along_axis(distance,selected,axis=1)
        selected_valid=np.take_along_axis(eligible,selected,axis=1)&(selected_distance<=normalized_radius)
        denominator=np.zeros(stop-start,dtype=f32); numerator=np.zeros_like(denominator)
        for rank in range(selected.shape[1]):
            metric=selected_distance[:,rank]
            if power==2: metric=np.asarray(metric*metric,dtype=f32)
            weight=np.where(selected_valid[:,rank],
                np.asarray(f32(1.)/np.maximum(metric,f32(1.)),dtype=f32),f32(0.))
            denominator=np.asarray(denominator+weight,dtype=f32)
            numerator=np.asarray(numerator+np.asarray(
                weight*values[selected[:,rank]],dtype=f32),dtype=f32)
        weighted=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator!=0.)
        result=np.where(nearest_distance>normalized_radius,values[nearest],weighted)
        if np.any(~available):
            fallback=np.asarray(empty_fallback,dtype=f32)
            if fallback.shape!=(2,): raise ValueError('empty_fallback must contain land and water values')
            result[~available]=np.where(group[~available],fallback[1],fallback[0])
        output[start:stop]=result
    return output.reshape(grid.ny,grid.nx)


def _near_upper_air(station_count,station_xy,upper_xy,upper_valid,
                    separation_cells,dx_km,dy_km):
    """Vectorized DIAGNO RMIN2 test with legacy REAL*4 arithmetic."""
    f32=np.float32
    skip=np.zeros(station_count,dtype=bool)
    if separation_cells<0. or upper_xy is None:
        return skip
    if dx_km<=0. or dy_km<=0.:
        raise ValueError('RMIN2 grid spacing must be positive')
    surface_points=np.asarray(station_xy,dtype=f32)
    upper_points=np.asarray(upper_xy,dtype=f32)
    available=np.asarray(upper_valid,dtype=bool)
    if surface_points.shape!=(station_count,2) or upper_points.ndim!=2 or upper_points.shape[1]!=2:
        raise ValueError('RMIN2 station coordinates are inconsistent')
    if available.shape!=(upper_points.shape[0],):
        raise ValueError('RMIN2 upper validity has wrong shape')
    if available.any():
        delta=np.asarray(upper_points[None,:,:]-surface_points[:,None,:],dtype=f32)
        rsx=np.asarray(delta[...,0]/f32(dx_km),dtype=f32)
        rsy=np.asarray(delta[...,1]/f32(dy_km),dtype=f32)
        squared=np.asarray(np.asarray(rsx*rsx,dtype=f32)+
                           np.asarray(rsy*rsy,dtype=f32),dtype=f32)
        distance=np.asarray(np.sqrt(squared),dtype=f32)
        skip=np.min(distance[:,available],axis=1)<f32(separation_cells)
    return skip


def _wind_components(wind_speed,wind_direction,grid_convergence_degrees=None):
    """PREPDI/RDS true-north wind conversion in legacy REAL*4 order."""
    f32=np.float32
    speed=np.asarray(wind_speed,dtype=f32)
    direction=np.asarray(wind_direction,dtype=f32)
    if speed.shape!=direction.shape:
        raise ValueError('wind speed and direction have different shapes')
    convergence=(np.zeros_like(direction) if grid_convergence_degrees is None else
                 np.asarray(grid_convergence_degrees,dtype=f32))
    if convergence.shape!=direction.shape:
        raise ValueError('grid convergence must have one value per station')
    valid=(np.abs(speed)<f32(999.))&(np.abs(direction)<f32(999.))
    rotated=np.asarray(direction-convergence,dtype=f32)
    rotated=np.where(rotated<f32(0.),np.asarray(rotated+f32(360.),dtype=f32),rotated)
    rotated=np.where(rotated>f32(360.),np.asarray(rotated-f32(360.),dtype=f32),rotated)
    direction=np.where(valid,rotated,direction).astype(f32)
    angle=np.asarray(f32(.0174533)*direction,dtype=f32)
    sine=np.asarray(np.sin(angle.astype(np.float64)),dtype=f32)
    cosine=np.asarray(np.cos(angle.astype(np.float64)),dtype=f32)
    u=np.where(valid,np.asarray(-speed*sine,dtype=f32),f32(9999.))
    v=np.where(valid,np.asarray(-speed*cosine,dtype=f32),f32(9999.))
    return u.astype(f32),v.astype(f32),valid,direction


def wind_profiles(wind_speed,wind_direction,anemometer_height_m,z_centers_m,roughness_m,
                  iextrp=-1,factors=None,station_latitude=None,
                  grid_convergence_degrees=None,station_xy=None,upper_xy=None,
                  upper_valid=None,upper_air_separation_cells=-1.,dx_km=1.,dy_km=1.,
                  power_exponent=.143):
    """Convert land/overwater wind observations and apply IEXTRP 1--4."""
    f32=np.float32
    speed=np.asarray(wind_speed,dtype=f32)
    anem=np.asarray(anemometer_height_m,dtype=f32)
    roughness=np.asarray(roughness_m,dtype=f32); levels=np.asarray(z_centers_m,dtype=f32)
    if anem.shape!=speed.shape or roughness.shape!=speed.shape or levels.ndim!=1:
        raise ValueError('surface wind profile inputs have incompatible shapes')
    if np.any(anem<=0.) or np.any(roughness<=0.) or np.any(levels<=0.):
        raise ValueError('wind heights and roughness must be positive')
    u,v,valid,direction=_wind_components(speed,wind_direction,grid_convergence_degrees)
    output_u=np.full((levels.size,speed.size),f32(9999.),dtype=f32)
    output_v=np.full_like(output_u,f32(9999.))
    skip=_near_upper_air(speed.size,station_xy,upper_xy,upper_valid,
                         upper_air_separation_cells,dx_km,dy_km)
    output_u[0,skip&valid]=u[skip&valid]; output_v[0,skip&valid]=v[skip&valid]
    calm=valid&(speed<1.e-4)&~skip; output_u[:,calm]=0.; output_v[:,calm]=0.
    active=valid&~calm&~skip; option=abs(int(iextrp))
    if option==1:
        log_level=np.asarray(np.log(np.float64(levels[0])),dtype=f32)
        log_roughness=np.asarray(np.log(roughness.astype(np.float64)),dtype=f32)
        log_anemometer=np.asarray(np.log(anem.astype(np.float64)),dtype=f32)
        numerator=np.asarray(log_level-log_roughness,dtype=f32)
        denominator=np.asarray(log_anemometer-log_roughness,dtype=f32)
        coefficient=np.asarray(numerator/denominator,dtype=f32)
        output_u[0,active]=np.asarray(u[active]*coefficient[active],dtype=f32)
        output_v[0,active]=np.asarray(v[active]*coefficient[active],dtype=f32)
    elif option==2:
        exponent=np.broadcast_to(np.asarray(power_exponent,dtype=f32),speed.shape)
        # DIAGNO advances K=2..NZ from the already rounded preceding layer;
        # only K=2 is referenced directly to the anemometer height.
        for level in range(1,levels.size):
            reference=anem if level==1 else np.full_like(anem,levels[level-1])
            ratio=np.asarray(levels[level]/reference,dtype=f32)
            coefficient=np.asarray(ratio.astype(np.float64)**exponent.astype(np.float64),dtype=f32)
            source_u=u if level==1 else output_u[level-1]
            source_v=v if level==1 else output_v[level-1]
            output_u[level,active]=np.asarray(source_u[active]*coefficient[active],dtype=f32)
            output_v[level,active]=np.asarray(source_v[active]*coefficient[active],dtype=f32)
        ratio=np.asarray(anem/levels[0],dtype=f32)
        coefficient=np.asarray(ratio.astype(np.float64)**exponent.astype(np.float64),dtype=f32)
        output_u[0,active]=np.asarray(u[active]/coefficient[active],dtype=f32)
        output_v[0,active]=np.asarray(v[active]/coefficient[active],dtype=f32)
    elif option==3:
        scale=np.asarray(factors,dtype=f32)
        if scale.shape!=(levels.size,): raise ValueError('IEXTRP=3 requires one FEXTR2 factor per layer')
        output_u[:,active]=np.asarray(u[None,active]*scale[:,None],dtype=f32)
        output_v[:,active]=np.asarray(v[None,active]*scale[:,None],dtype=f32)
    elif option==4:
        latitude=np.asarray(station_latitude,dtype=f32)
        if latitude.shape!=speed.shape: raise ValueError('IEXTRP=4 requires one latitude per station')
        # WIND1 calls SIMILT with neutral L=ZI=10000 m.  Preserve every
        # REAL*4 assignment in SIMILT rather than evaluating the equivalent
        # log profile and turning-angle equations in NumPy float64.
        ws1=np.asarray(np.sqrt(np.asarray(
            np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32)),dtype=f32)
        factor=f32(57.2957795)
        recovered=np.asarray(f32(270.)-np.asarray(
            np.asarray(np.arctan2(v,u),dtype=f32)*factor,
            dtype=f32),dtype=f32)
        wd1=np.asarray(np.fmod(recovered,f32(360.)),dtype=f32)
        wd1=np.where(wd1==f32(0.),f32(360.),wd1).astype(f32)
        height_ratio=np.asarray(anem/roughness,dtype=f32)
        denominator=np.asarray(np.log(height_ratio.astype(np.float64)),dtype=f32)
        if np.any(np.abs(denominator[active])<f32(1.e-12)):
            raise ValueError('anemometer height must differ from roughness')
        level_ratio=np.asarray(levels[:,None]/roughness[None,:],dtype=f32)
        logarithm=np.asarray(np.log(level_ratio.astype(np.float64)),dtype=f32)
        magnitude=np.asarray(np.asarray(ws1[None,:]*logarithm,dtype=f32)/denominator[None,:],dtype=f32)
        dh_d1=np.asarray(f32(12.)*f32(1.58),dtype=f32)
        anem_exponent=np.asarray(np.asarray(-f32(1.)*anem,dtype=f32)/f32(200.),dtype=f32)
        anem_decay=np.asarray(np.exp(anem_exponent.astype(np.float64)),dtype=f32)
        at_anem=np.asarray(dh_d1*np.asarray(f32(1.)-anem_decay,dtype=f32),dtype=f32)
        level_exponent=np.asarray(np.asarray(-f32(1.)*levels,dtype=f32)/f32(200.),dtype=f32)
        level_decay=np.asarray(np.exp(level_exponent.astype(np.float64)),dtype=f32)
        at_level=np.asarray(dh_d1*np.asarray(f32(1.)-level_decay,dtype=f32),dtype=f32)
        north=np.asarray(np.asarray(wd1[None,:]+at_level[:,None],dtype=f32)-at_anem[None,:],dtype=f32)
        north=np.where(north>f32(360.),np.asarray(north-f32(360.),dtype=f32),north)
        south=np.asarray(np.asarray(wd1[None,:]-at_level[:,None],dtype=f32)+at_anem[None,:],dtype=f32)
        south=np.where(south<=f32(0.),np.asarray(f32(360.)+south,dtype=f32),south)
        adjusted_direction=np.where(latitude[None,:]>f32(0.),north,south).astype(f32)
        radians=np.asarray(f32(.0174533)*adjusted_direction,dtype=f32)
        profile_sine=np.asarray(np.sin(radians.astype(np.float64)),dtype=f32)
        profile_cosine=np.asarray(np.cos(radians.astype(np.float64)),dtype=f32)
        output_u[:,active]=np.asarray(-magnitude[:,active]*profile_sine[:,active],dtype=f32)
        output_v[:,active]=np.asarray(-magnitude[:,active]*profile_cosine[:,active],dtype=f32)
    else:
        raise ValueError('IEXTRP must be one of +/-1, +/-2, +/-3, or +/-4')
    return output_u,output_v


def surface_wind_profiles(record,anemometer_height_m,z_centers_m,roughness_m,iextrp=-1,
                          factors=None,station_latitude=None,grid_convergence_degrees=None,
                          station_xy=None,upper_xy=None,upper_valid=None,
                          upper_air_separation_cells=-1.,dx_km=1.,dy_km=1.,
                          power_exponent=.143):
    """Convert a SURF.DAT record and apply PREPDI IEXTRP 1--4."""
    return wind_profiles(record.wind_speed,record.wind_direction,anemometer_height_m,
        z_centers_m,roughness_m,iextrp,factors,station_latitude,
        grid_convergence_degrees,station_xy,upper_xy,upper_valid,
        upper_air_separation_cells,dx_km,dy_km,power_exponent)


def component_wind_profiles(u_component,v_component,anemometer_height_m,z_centers_m,
                            roughness_m,iextrp=-1,factors=None,station_latitude=None,
                            station_xy=None,upper_xy=None,upper_valid=None,
                            upper_air_separation_cells=-1.,dx_km=1.,dy_km=1.,
                            power_exponent=.143):
    """Profile grid-oriented DIAG.DAT U/V surface observations."""
    f32=np.float32
    u=np.asarray(u_component,dtype=f32); v=np.asarray(v_component,dtype=f32)
    if u.shape!=v.shape: raise ValueError('surface U/V components have different shapes')
    anem=np.asarray(anemometer_height_m,dtype=f32)
    roughness=np.asarray(roughness_m,dtype=f32); levels=np.asarray(z_centers_m,dtype=f32)
    if anem.shape!=u.shape or roughness.shape!=u.shape or levels.ndim!=1:
        raise ValueError('surface component profile inputs have incompatible shapes')
    if np.any(anem<=0.) or np.any(roughness<=0.) or np.any(levels<=0.):
        raise ValueError('wind heights and roughness must be positive')
    # US/VS are initialized to EDIT=999 and DIAGNO treats values above
    # EDITL=900 as missing before vertical extrapolation.
    valid=(u<=f32(900.))&(v<=f32(900.))
    profiled_u=np.full((levels.size,u.size),f32(9999.),dtype=f32)
    profiled_v=np.full_like(profiled_u,f32(9999.))
    skip=_near_upper_air(u.size,station_xy,upper_xy,upper_valid,
                         upper_air_separation_cells,dx_km,dy_km)
    profiled_u[0,skip&valid]=u[skip&valid]
    profiled_v[0,skip&valid]=v[skip&valid]
    speed=np.asarray(np.sqrt(np.asarray(
        np.asarray(u*u,dtype=f32)+np.asarray(v*v,dtype=f32),dtype=f32)),dtype=f32)
    calm=valid&(speed<f32(1.e-4))&~skip
    profiled_u[:,calm]=f32(0.); profiled_v[:,calm]=f32(0.)
    active=valid&~calm&~skip
    option=abs(int(iextrp))
    if option==1:
        log_level=np.asarray(np.log(np.float64(levels[0])),dtype=f32)
        log_roughness=np.asarray(np.log(roughness.astype(np.float64)),dtype=f32)
        log_anemometer=np.asarray(np.log(anem.astype(np.float64)),dtype=f32)
        coefficient=np.asarray(np.asarray(log_level-log_roughness,dtype=f32)/
                               np.asarray(log_anemometer-log_roughness,dtype=f32),dtype=f32)
        profiled_u[0,active]=np.asarray(u[active]*coefficient[active],dtype=f32)
        profiled_v[0,active]=np.asarray(v[active]*coefficient[active],dtype=f32)
    elif option==2:
        exponent=np.broadcast_to(np.asarray(power_exponent,dtype=f32),u.shape)
        for level in range(1,levels.size):
            reference=anem if level==1 else np.full_like(anem,levels[level-1])
            ratio=np.asarray(levels[level]/reference,dtype=f32)
            coefficient=np.asarray(ratio.astype(np.float64)**exponent.astype(np.float64),dtype=f32)
            source_u=u if level==1 else profiled_u[level-1]
            source_v=v if level==1 else profiled_v[level-1]
            profiled_u[level,active]=np.asarray(source_u[active]*coefficient[active],dtype=f32)
            profiled_v[level,active]=np.asarray(source_v[active]*coefficient[active],dtype=f32)
        ratio=np.asarray(anem/levels[0],dtype=f32)
        coefficient=np.asarray(ratio.astype(np.float64)**exponent.astype(np.float64),dtype=f32)
        profiled_u[0,active]=np.asarray(u[active]/coefficient[active],dtype=f32)
        profiled_v[0,active]=np.asarray(v[active]/coefficient[active],dtype=f32)
    elif option==3:
        scale=np.asarray(factors,dtype=f32)
        if scale.shape!=(levels.size,): raise ValueError('IEXTRP=3 requires one FEXTR2 factor per layer')
        profiled_u[:,active]=np.asarray(u[None,active]*scale[:,None],dtype=f32)
        profiled_v[:,active]=np.asarray(v[None,active]*scale[:,None],dtype=f32)
    elif option==4:
        latitude=np.asarray(station_latitude,dtype=f32)
        if latitude.shape!=u.shape: raise ValueError('IEXTRP=4 requires one latitude per station')
        factor=f32(57.2957795)
        recovered=np.asarray(f32(270.)-np.asarray(
            np.asarray(np.arctan2(v,u),dtype=f32)*factor,
            dtype=f32),dtype=f32)
        wd1=np.asarray(np.fmod(recovered,f32(360.)),dtype=f32)
        wd1=np.where(wd1==f32(0.),f32(360.),wd1).astype(f32)
        denominator=np.asarray(np.log(np.asarray(anem/roughness,dtype=f32).astype(np.float64)),dtype=f32)
        if np.any(np.abs(denominator[active])<f32(1.e-12)):
            raise ValueError('anemometer height must differ from roughness')
        logarithm=np.asarray(np.log(np.asarray(
            levels[:,None]/roughness[None,:],dtype=f32).astype(np.float64)),dtype=f32)
        magnitude=np.asarray(np.asarray(speed[None,:]*logarithm,dtype=f32)/denominator[None,:],dtype=f32)
        dh_d1=np.asarray(f32(12.)*f32(1.58),dtype=f32)
        anem_decay=np.asarray(np.exp(np.asarray(-anem/f32(200.),dtype=f32).astype(np.float64)),dtype=f32)
        level_decay=np.asarray(np.exp(np.asarray(-levels/f32(200.),dtype=f32).astype(np.float64)),dtype=f32)
        at_anem=np.asarray(dh_d1*np.asarray(f32(1.)-anem_decay,dtype=f32),dtype=f32)
        at_level=np.asarray(dh_d1*np.asarray(f32(1.)-level_decay,dtype=f32),dtype=f32)
        north=np.asarray(np.asarray(wd1[None,:]+at_level[:,None],dtype=f32)-at_anem[None,:],dtype=f32)
        north=np.where(north>f32(360.),np.asarray(north-f32(360.),dtype=f32),north)
        south=np.asarray(np.asarray(wd1[None,:]-at_level[:,None],dtype=f32)+at_anem[None,:],dtype=f32)
        south=np.where(south<=f32(0.),np.asarray(f32(360.)+south,dtype=f32),south)
        direction=np.where(latitude[None,:]>f32(0.),north,south).astype(f32)
        radians=np.asarray(f32(.0174533)*direction,dtype=f32)
        sine=np.asarray(np.sin(radians.astype(np.float64)),dtype=f32)
        cosine=np.asarray(np.cos(radians.astype(np.float64)),dtype=f32)
        profiled_u[:,active]=np.asarray(-magnitude[:,active]*sine[:,active],dtype=f32)
        profiled_v[:,active]=np.asarray(-magnitude[:,active]*cosine[:,active],dtype=f32)
    else:
        raise ValueError('IEXTRP must be one of +/-1, +/-2, +/-3, or +/-4')
    return profiled_u,profiled_v
