"""Reader and vectorized interpolator for legacy CALMET ``PROG.DAT``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GridConfig
from .legacy_io import FortranSequentialReader


def _array(payload, dtype, count, label):
    values=np.frombuffer(payload,dtype=dtype)
    if values.size!=count:
        raise ValueError(f'PROG.DAT {label} record contains {values.size} values; expected {count}')
    return values.copy()


@dataclass(frozen=True, slots=True)
class ProgRecord:
    """One hour-ending CSUMM prognostic wind field."""

    hour: float
    x_origin_km: float
    y_origin_km: float
    spacing_km: float
    height_m: np.ndarray
    u_cm_s: np.ndarray
    v_cm_s: np.ndarray


class ProgData:
    """Legacy sequential-unformatted PROG.DAT used by ``IPROG=1/2``."""

    def __init__(self,path):
        self.path=Path(path)
        records=[]
        with FortranSequentialReader(self.path) as reader:
            while (time_payload:=reader.read_record()) is not None:
                hour=float(_array(time_payload,'<f4',1,'time')[0])
                dimensions=_array(self._required(reader,'dimensions'),'<i4',3,'dimensions')
                nx,ny,nz=(int(value) for value in dimensions)
                if nx<1 or ny<1 or nz<1:
                    raise ValueError('PROG.DAT dimensions must be positive')
                geometry=_array(self._required(reader,'geometry'),'<f4',3,'geometry')
                height=_array(self._required(reader,'heights'),'<f4',nz,'heights')
                if np.any(np.diff(height)<=0.):
                    raise ValueError('PROG.DAT heights must increase strictly')
                u=self._read_component(reader,nx,ny,nz,'U')
                v=self._read_component(reader,nx,ny,nz,'V')
                records.append(ProgRecord(hour,float(geometry[0]),float(geometry[1]),
                                          float(geometry[2]),height,u,v))
        if not records:
            raise ValueError('PROG.DAT contains no prognostic records')
        if any(record.spacing_km<=0. for record in records):
            raise ValueError('PROG.DAT grid spacing must be positive')
        self.records=tuple(records)

    @staticmethod
    def _required(reader,label):
        payload=reader.read_record()
        if payload is None:
            raise ValueError(f'truncated PROG.DAT before {label} record')
        return payload

    @classmethod
    def _read_component(cls,reader,nx,ny,nz,label):
        result=np.empty((nz,ny,nx),dtype=np.float64)
        for level in range(nz):
            for row in range(ny):
                result[level,row]=_array(
                    cls._required(reader,f'{label} level {level+1} row {row+1}'),
                    '<f4',nx,label)
        return result

    def at_hour(self,hour):
        """Apply PROGRD's forward scan for one legacy ending-hour label."""
        wanted=int(np.float64(hour)*np.float64(100.))%2400
        for record in self.records:
            label=int(np.float64(record.hour)*np.float64(100.))
            if label<wanted: continue
            if label==wanted: return record
            break
        raise ValueError(f'PROG.DAT has no record for ending hour {float(hour):g}')

    def interpolate(self,hour,grid: GridConfig,target_height_m):
        """Reproduce ``PROGRD`` vertical and four-corner 1/R² interpolation."""
        return self.interpolate_record(self.at_hour(hour),grid,target_height_m)

    @staticmethod
    def interpolate_record(record,grid: GridConfig,target_height_m):
        """Interpolate one explicitly selected record without hour ambiguity."""
        f32=np.float64
        target_height=np.asarray(target_height_m,dtype=f32)
        if target_height.ndim!=1 or target_height.size<1:
            raise ValueError('target PROG.DAT heights must be a non-empty vector')
        if record.height_m.size==1:
            u=np.broadcast_to(record.u_cm_s,(target_height.size,*record.u_cm_s.shape[1:]))
            v=np.broadcast_to(record.v_cm_s,(target_height.size,*record.v_cm_s.shape[1:]))
        else:
            upper=np.searchsorted(record.height_m,target_height,side='right')
            upper=np.clip(upper,1,record.height_m.size-1)
            lower=upper-1
            denominator=np.asarray(record.height_m[upper]-record.height_m[lower],dtype=f32)
            fraction=np.asarray(
                np.asarray(target_height-record.height_m[lower],dtype=f32)/denominator,
                dtype=f32)
            fraction=np.where(target_height<record.height_m[0],f32(0.),fraction)
            fraction=np.where(target_height>=record.height_m[-1],f32(1.),fraction)
            u=np.asarray(record.u_cm_s[lower]+np.asarray(
                fraction[:,None,None]*np.asarray(
                    record.u_cm_s[upper]-record.u_cm_s[lower],dtype=f32),dtype=f32),dtype=f32)
            v=np.asarray(record.v_cm_s[lower]+np.asarray(
                fraction[:,None,None]*np.asarray(
                    record.v_cm_s[upper]-record.v_cm_s[lower],dtype=f32),dtype=f32),dtype=f32)

        target_x=np.asarray(f32(grid.x_origin_km)+
                            np.arange(grid.nx,dtype=f32)*f32(grid.dx_km),dtype=f32)
        target_y=np.asarray(f32(grid.y_origin_km)+
                            np.arange(grid.ny,dtype=f32)*f32(grid.dy_km),dtype=f32)
        xx,yy=np.meshgrid(target_x,target_y)
        local_x=np.asarray(xx-f32(record.x_origin_km),dtype=f32)
        local_y=np.asarray(yy-f32(record.y_origin_km),dtype=f32)
        spacing=f32(record.spacing_km)
        # PROGRD assigns ``X/DXKP + 1`` to a one-based INTEGER first.
        # Moving the +1 after truncation changes cells just west/south of
        # the source origin because Fortran truncates toward zero.
        ix0=np.trunc(np.asarray(local_x/spacing+f32(1.),dtype=f32)).astype(np.int64)-1
        iy0=np.trunc(np.asarray(local_y/spacing+f32(1.),dtype=f32)).astype(np.int64)-1
        ix=np.stack((ix0,ix0+1,ix0+1,ix0),axis=-1)
        iy=np.stack((iy0,iy0,iy0+1,iy0+1),axis=-1)
        valid=(ix>=0)&(ix<record.u_cm_s.shape[2])&(iy>=0)&(iy<record.u_cm_s.shape[1])
        source_x=np.asarray(ix.astype(f32)*spacing,dtype=f32)
        source_y=np.asarray(iy.astype(f32)*spacing,dtype=f32)
        rsx=np.asarray(local_x[...,None]-source_x,dtype=f32)
        rsy=np.asarray(local_y[...,None]-source_y,dtype=f32)
        distance=np.maximum(np.sqrt(np.asarray(
            np.asarray(rsx*rsx,dtype=f32)+np.asarray(rsy*rsy,dtype=f32),dtype=f32)),
            f32(.001))
        weights=np.where(valid,np.asarray(
            f32(1.)/np.asarray(distance*distance,dtype=f32),dtype=f32),f32(0.))
        total=np.zeros(valid.shape[:-1],dtype=f32)
        sampled_u=u[:,iy.clip(0,u.shape[1]-1),ix.clip(0,u.shape[2]-1)]
        sampled_v=v[:,iy.clip(0,v.shape[1]-1),ix.clip(0,v.shape[2]-1)]
        numerator_u=np.zeros(sampled_u.shape[:-1],dtype=f32)
        numerator_v=np.zeros(sampled_v.shape[:-1],dtype=f32)
        for corner in range(4):
            weight=weights[...,corner]
            total=np.asarray(total+weight,dtype=f32)
            numerator_u=np.asarray(numerator_u+np.asarray(
                sampled_u[...,corner]*weight[None],dtype=f32),dtype=f32)
            numerator_v=np.asarray(numerator_v+np.asarray(
                sampled_v[...,corner]*weight[None],dtype=f32),dtype=f32)
        if np.any(total==0.):
            raise ValueError('PROG.DAT cannot interpolate one or more CALMET grid points')
        output_u=np.asarray(numerator_u/total[None],dtype=f32)
        output_v=np.asarray(numerator_v/total[None],dtype=f32)
        return (np.asarray(output_u*f32(.01),dtype=f32),
                np.asarray(output_v*f32(.01),dtype=f32))

    def records_for_hours(self,hours):
        """Reproduce PROGRD's stateful scan, skipping earlier extra records."""
        wanted=[int(np.float64(hour)*np.float64(100.))%2400 for hour in hours]
        labels=[int(np.float64(record.hour)*np.float64(100.)) for record in self.records]
        selected=[]; cursor=0
        for target in wanted:
            while cursor<len(labels) and labels[cursor]<target:
                cursor+=1
            if cursor>=len(labels) or labels[cursor]!=target:
                raise ValueError(
                    'PROG.DAT does not contain the requested chronological hour sequence')
            selected.append(self.records[cursor]); cursor+=1
        return tuple(selected)
