"""Formatted PRECIP.DAT observations and CALMET GRIDE interpolation."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import struct
import numpy as np

from .legacy_io import FortranSequentialReader
from .surface_data import SurfaceStation,_coded_instant,_coordinate,_instant


@dataclass(frozen=True,slots=True)
class PrecipitationRecord:
    begin: datetime
    end: datetime
    rate_mm_h: np.ndarray


class PrecipitationData:
    """Read formatted, unpacked PRECIP.DAT 2.1 records."""
    def __init__(self,path,*,station_metadata=None,datum=None):
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty PRECIP.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='PRECIP.DAT': raise ValueError('invalid PRECIP.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version=='2.1'
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid PRECIP.DAT comment header') from error
        projection=lines[index][:8].strip().upper(); index+=1
        if projection not in {'LL','NONE'}: raise ValueError('PRECIP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('precipitation station locations are present in both control file and PRECIP.DAT')
            datum_line=lines[index]; index+=1; self.datum=datum_line[:8].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('PRECIP.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('PRECIP.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum)
        if explicit:
            self.timezone=lines[index][:8].strip(); index+=1
        span=[int(item) for item in lines[index].replace(',',' ').split()]; index+=1
        expected=9 if explicit else 8
        if len(span)!=expected: raise ValueError(f'PRECIP.DAT time-span header must contain {expected} integers')
        if explicit:
            self.start=_instant(*span[:4]); self.end=_instant(*span[4:8]); count=span[8]
        else:
            first_end=_instant(*span[:3],0)
            self.start=first_end-timedelta(hours=1)
            self.end=_instant(*span[3:6],0); self.timezone=span[6]; count=span[7]
        if count<1: raise ValueError('PRECIP.DAT must contain at least one station')
        stations=[]
        if projection=='LL':
            for _ in range(count):
                fields=lines[index].replace(',',' ').split(); index+=1
                if len(fields)<4: raise ValueError('invalid PRECIP.DAT station metadata')
                stations.append(SurfaceStation(int(fields[0]),fields[1],_coordinate(fields[2],True),
                                               _coordinate(fields[3],False),0.))
        else:
            stations=list(station_metadata)
            if len(stations)!=count: raise ValueError('PRECIP.DAT station count does not match control-file metadata')
            identifiers=[]
            while len(identifiers)<count and index<len(lines):
                identifiers.extend(int(item) for item in lines[index].replace(',',' ').split()); index+=1
            if len(identifiers)!=count or identifiers!=[item.identifier for item in stations]:
                raise ValueError('PRECIP.DAT station IDs do not match control-file metadata')
        self.stations=tuple(stations)
        records=[]
        while index<len(lines):
            if not lines[index].strip(): index+=1; continue
            fields=lines[index].replace(',',' ').split(); index+=1
            time_width=8 if explicit else 3
            if len(fields)!=time_width+count: raise ValueError('PRECIP.DAT record has wrong field count')
            times=[int(item) for item in fields[:time_width]]
            if explicit:
                begin=_instant(*times[:4]); end=_instant(*times[4:])
            else:
                end=_instant(*times,0); begin=end-timedelta(hours=1)
            if not begin<end or (end-begin).total_seconds()>3600: raise ValueError('PRECIP.DAT record interval must be positive and at most one hour')
            records.append(PrecipitationRecord(begin,end,np.asarray(fields[time_width:],dtype=float)))
        if not records: raise ValueError('PRECIP.DAT contains no records')
        if records[0].begin<self.start or records[-1].end>self.end: raise ValueError('PRECIP.DAT records lie outside declared span')
        if any(left.end!=right.begin for left,right in zip(records[:-1],records[1:])): raise ValueError('PRECIP.DAT records are not contiguous')
        self.records=tuple(records)

    def at_begin(self,timestamp):
        for record in self.records:
            if record.begin<=timestamp<record.end: return record
        raise ValueError(f'PRECIP.DAT has no record covering {timestamp!s}')


def unpack_precipitation(words,count):
    """Expand CALMET PACK zero runs (negative word means that many zeros)."""
    output=[]
    for word in np.asarray(words,dtype=float):
        if word>=0.: output.append(float(word))
        else: output.extend([0.]*int(abs(word)+.1))
    if len(output)!=count: raise ValueError('packed PRECIP.DAT record expands to wrong station count')
    return np.asarray(output,dtype=float)


def read_unformatted_precipitation(path,*,station_metadata=None,datum=None):
    """Read GNU-Fortran sequential-unformatted PRECIP.DAT 2.1."""
    source=Path(path); result=object.__new__(PrecipitationData); result.path=source
    with FortranSequentialReader(source) as reader:
        def record(size=None,label='record'):
            payload=reader.read_record()
            if payload is None: raise ValueError(f'truncated unformatted PRECIP.DAT {label}')
            if size is not None and len(payload)!=size: raise ValueError(f'unformatted PRECIP.DAT {label} has wrong size')
            return payload
        head=record(96,'identity')
        if head[:16].decode('ascii').strip().upper()!='PRECIP.DAT': raise ValueError('invalid PRECIP.DAT dataset label')
        result.version=head[16:32].decode('ascii').strip()
        explicit=result.version=='2.1'
        comments=struct.unpack('<i',record(4,'comment count'))[0]
        if comments<0: raise ValueError('negative PRECIP.DAT comment count')
        for _ in range(comments): record(80,'comment')
        projection=record(8,'projection').decode('ascii').strip().upper()
        if projection not in {'LL','NONE'}: raise ValueError('PRECIP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('precipitation station locations are present in both control file and PRECIP.DAT')
            datum_record=record(20,'datum'); result.datum=datum_record[:8].decode('ascii').strip()
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
                raise ValueError('PRECIP.DAT NONE requires control-file station metadata and datum')
            result.datum=str(datum)
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',record(20,'time span'))
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        if count<1 or packing not in {0,1}: raise ValueError('invalid PRECIP.DAT station count or packing code')
        identifiers=np.frombuffer(record(4*count,'station IDs'),dtype='<i4')
        if projection=='LL':
            names=record(4*count,'station names'); latitudes=record(16*count,'station latitudes'); longitudes=record(16*count,'station longitudes')
            result.stations=tuple(SurfaceStation(int(identifiers[i]),names[4*i:4*i+4].decode('ascii').strip(),
                _coordinate(latitudes[16*i:16*i+16].decode('ascii'),True),
                _coordinate(longitudes[16*i:16*i+16].decode('ascii'),False),0.) for i in range(count))
        else:
            result.stations=tuple(station_metadata)
            if len(result.stations)!=count or identifiers.tolist()!=[item.identifier for item in result.stations]:
                raise ValueError('PRECIP.DAT station IDs do not match control-file metadata')
        records=[]
        while (payload:=reader.read_record()) is not None:
            if explicit and packing:
                if len(payload)!=20: raise ValueError('packed PRECIP.DAT record header has wrong size')
                begin_code,begin_sec,end_code,end_sec,nwords=struct.unpack('<5i',payload)
                if nwords<0: raise ValueError('packed PRECIP.DAT word count cannot be negative')
                rates=(np.zeros(count) if nwords==0 else
                       unpack_precipitation(np.frombuffer(record(4*nwords,'packed values'),dtype='<f4'),count))
            elif explicit:
                if len(payload)!=16+4*count: raise ValueError('PRECIP.DAT record has wrong size')
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[:16])
                rates=np.frombuffer(payload,dtype='<f4',count=count,offset=16).copy()
            elif packing:
                if len(payload)!=8: raise ValueError('packed legacy PRECIP.DAT record header has wrong size')
                end_code,nwords=struct.unpack('<2i',payload)
                if nwords<0: raise ValueError('packed PRECIP.DAT word count cannot be negative')
                rates=(np.zeros(count) if nwords==0 else
                       unpack_precipitation(np.frombuffer(record(4*nwords,'packed values'),dtype='<f4'),count))
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            else:
                if len(payload)!=4+4*count: raise ValueError('legacy PRECIP.DAT record has wrong size')
                end_code=struct.unpack('<i',payload[:4])[0]
                rates=np.frombuffer(payload,dtype='<f4',count=count,offset=4).copy()
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            if explicit:
                begin,end=_coded_instant(begin_code,begin_sec),_coded_instant(end_code,end_sec)
            if not begin<end or (end-begin).total_seconds()>3600: raise ValueError('PRECIP.DAT record interval must be positive and at most one hour')
            records.append(PrecipitationRecord(begin,end,rates))
    if not records: raise ValueError('PRECIP.DAT contains no records')
    if records[0].begin<result.start or records[-1].end>result.end: raise ValueError('PRECIP.DAT records lie outside declared span')
    if any(left.end!=right.begin for left,right in zip(records[:-1],records[1:])): raise ValueError('PRECIP.DAT records are not contiguous')
    result.records=tuple(records); return result


def load_precipitation(path,format=2,*,station_metadata=None,datum=None):
    if format==1: return read_unformatted_precipitation(path,station_metadata=station_metadata,datum=datum)
    if format==2: return PrecipitationData(path,station_metadata=station_metadata,datum=datum)
    raise ValueError('PRECIP.DAT format/IFORMP must be 1 or 2')


def grid_precipitation(values,station_xy,grid,method=2,radius_km=50.,cutoff=.01):
    """Vectorized/blockwise CALMET GRIDE methods 1, 2 and 3."""
    f32=np.float32
    rates=np.asarray(values,dtype=f32)
    points=np.asarray(np.asarray(station_xy,dtype=f32)*f32(1000.),dtype=f32)
    if points.shape!=(rates.size,2): raise ValueError('precipitation station coordinates have wrong shape')
    valid=rates<9998.
    if not valid.any(): return np.zeros((grid.ny,grid.nx),dtype=f32)
    points=points[valid]; rates=rates[valid]
    if method not in {1,2,3}: raise ValueError('precipitation interpolation method must be 1, 2, or 3')
    sigma=np.asarray(f32(1000.)*f32(radius_km),dtype=f32)
    if method==3 and sigma==0.:
        wet=rates>0.; pairs=wet[:,None]!=wet[None,:]
        delta=np.asarray(points[:,None,:]-points[None,:,:],dtype=f32)
        pair_distance=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)
        if pairs.any(): sigma=np.asarray(f32(.5)*np.sqrt(np.min(pair_distance[pairs])),dtype=f32)
        else: sigma=np.asarray(f32(.5)*np.sqrt(f32(1.e37)),dtype=f32)
    x=np.asarray((np.arange(grid.nx,dtype=f32)+f32(.5))*f32(grid.dx_km)*f32(1000.),dtype=f32)
    y=np.asarray((np.arange(grid.ny,dtype=f32)+f32(.5))*f32(grid.dy_km)*f32(1000.),dtype=f32)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty(targets.shape[0],dtype=f32)
    for start in range(0,targets.shape[0],65536):
        delta=np.asarray(targets[start:start+65536,None,:]-points[None,:,:],dtype=f32)
        d2=np.asarray(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1],dtype=f32)
        if method in {1,2}:
            sigma2=np.asarray(sigma*sigma,dtype=f32); inside=d2<=sigma2
            denominator=np.zeros(d2.shape[0],dtype=f32); numerator=np.zeros_like(denominator)
            for station in range(rates.size):
                distance=(np.asarray(np.sqrt(d2[:,station]),dtype=f32)
                          if method==1 else d2[:,station])
                safe=np.maximum(distance,f32(.01))
                weight=np.where(inside[:,station],np.asarray(f32(1.)/safe,dtype=f32),f32(0.))
                denominator=np.asarray(denominator+weight,dtype=f32)
                numerator=np.asarray(numerator+np.asarray(
                    rates[station]*weight,dtype=f32),dtype=f32)
            weighted=np.divide(numerator,denominator,out=np.zeros_like(denominator),where=denominator!=0.)
            nearest=rates[np.argmin(d2,axis=1)]
            result=np.where(denominator==f32(0.),nearest,
                            np.where(weighted<f32(cutoff),f32(0.),weighted))
        else:
            sigma2=np.asarray(sigma*sigma,dtype=f32)
            denominator=np.zeros(d2.shape[0],dtype=f32); numerator=np.zeros_like(denominator)
            for station in range(rates.size):
                exponent=np.asarray(d2[:,station]/sigma2,dtype=f32)
                decay=np.where(exponent>f32(60.),f32(0.),
                               np.asarray(np.exp((-exponent).astype(np.float64)),dtype=f32))
                safe=np.maximum(d2[:,station],f32(.01))
                weight=np.asarray(decay/safe,dtype=f32)
                denominator=np.asarray(denominator+weight,dtype=f32)
                numerator=np.asarray(numerator+np.asarray(
                    np.asarray(rates[station]*decay,dtype=f32)/safe,dtype=f32),dtype=f32)
            result=np.where(numerator<f32(1.e-30),f32(0.),
                            np.asarray(numerator/denominator,dtype=f32))
            result=np.where(result<f32(cutoff),f32(0.),result)
        output[start:start+d2.shape[0]]=result
    return output.reshape(grid.ny,grid.nx)
