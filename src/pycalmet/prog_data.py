"""Reader and vectorized interpolator for legacy CALMET ``PROG.DAT``."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import GridConfig
from .legacy_io import FortranSequentialReader


MODEL_DTYPE=np.float32
_DAY=timedelta(days=1)
_NOON=timedelta(hours=12)


def _parse_hour(value):
    """Parse a PROG.DAT decimal ``HH.MM`` value at the file boundary."""
    hour=float(value)
    if not np.isfinite(hour):
        raise ValueError('PROG.DAT hour must be finite')
    text=f'{hour:05.2f}'
    canonical=float(text)
    tolerance=max(float(np.spacing(np.float32(canonical)))*2.,
                  float(np.finfo(np.float32).eps))
    if abs(hour-canonical)>tolerance:
        raise ValueError(f'invalid PROG.DAT hour {hour:g}')
    if text=='24.00':
        return timedelta(0),True
    try:
        clock=datetime.strptime(text,'%H.%M')
    except ValueError as error:
        raise ValueError(f'invalid PROG.DAT hour {hour:g}') from error
    return timedelta(hours=clock.hour,minutes=clock.minute),False


def _hour_label(value):
    """Return the normalized time-of-day for a decimal ``HH.MM`` value."""
    return _parse_hour(value)[0]


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
        result=np.empty((nz,ny,nx),dtype=MODEL_DTYPE)
        for level in range(nz):
            for row in range(ny):
                result[level,row]=_array(
                    cls._required(reader,f'{label} level {level+1} row {row+1}'),
                    '<f4',nx,label)
        return result

    def at_hour(self,hour):
        """Return the first record matching a normalized ending-hour label."""
        wanted=_hour_label(hour)
        self._record_offsets()
        for record in self.records:
            label=_hour_label(record.hour)
            if label==wanted: return record
        raise ValueError(f'PROG.DAT has no record for ending hour {float(hour):g}')

    def interpolate(self,hour,grid: GridConfig,target_height_m):
        """Apply vertical and four-corner inverse-distance interpolation."""
        return self.interpolate_record(self.at_hour(hour),grid,target_height_m)

    @staticmethod
    def interpolate_record(record,grid: GridConfig,target_height_m):
        """Interpolate one explicitly selected record without hour ambiguity."""
        target_height=np.asarray(target_height_m,dtype=MODEL_DTYPE)
        source_height=np.asarray(record.height_m,dtype=MODEL_DTYPE)
        if target_height.ndim!=1 or target_height.size<1:
            raise ValueError('target PROG.DAT heights must be a non-empty vector')
        if source_height.size==1:
            u=np.broadcast_to(record.u_cm_s,(target_height.size,*record.u_cm_s.shape[1:]))
            v=np.broadcast_to(record.v_cm_s,(target_height.size,*record.v_cm_s.shape[1:]))
        else:
            upper=np.searchsorted(source_height,target_height,side='right')
            upper=np.clip(upper,1,source_height.size-1)
            lower=upper-1
            fraction=(target_height-source_height[lower]) / (
                source_height[upper]-source_height[lower]
            )
            fraction=np.where(target_height<source_height[0],MODEL_DTYPE(0.),fraction)
            fraction=np.where(target_height>=source_height[-1],MODEL_DTYPE(1.),fraction)
            u=(record.u_cm_s[lower]+fraction[:,None,None]*
               (record.u_cm_s[upper]-record.u_cm_s[lower]))
            v=(record.v_cm_s[lower]+fraction[:,None,None]*
               (record.v_cm_s[upper]-record.v_cm_s[lower]))

        target_x=(MODEL_DTYPE(grid.x_origin_km)+np.arange(grid.nx,dtype=MODEL_DTYPE)*
                  MODEL_DTYPE(grid.dx_km))
        target_y=(MODEL_DTYPE(grid.y_origin_km)+np.arange(grid.ny,dtype=MODEL_DTYPE)*
                  MODEL_DTYPE(grid.dy_km))
        xx,yy=np.meshgrid(target_x,target_y)
        local_x=xx-MODEL_DTYPE(record.x_origin_km)
        local_y=yy-MODEL_DTYPE(record.y_origin_km)
        spacing=MODEL_DTYPE(record.spacing_km)
        ix0=np.floor(local_x/spacing).astype(np.int64)
        iy0=np.floor(local_y/spacing).astype(np.int64)
        ix=np.stack((ix0,ix0+1,ix0+1,ix0),axis=-1)
        iy=np.stack((iy0,iy0,iy0+1,iy0+1),axis=-1)
        valid=(ix>=0)&(ix<record.u_cm_s.shape[2])&(iy>=0)&(iy<record.u_cm_s.shape[1])
        source_x=ix.astype(MODEL_DTYPE)*spacing
        source_y=iy.astype(MODEL_DTYPE)*spacing
        rsx=local_x[...,None]-source_x
        rsy=local_y[...,None]-source_y
        distance=np.maximum(np.hypot(rsx,rsy),MODEL_DTYPE(.001))
        weights=np.where(valid,MODEL_DTYPE(1.)/(distance*distance),MODEL_DTYPE(0.))
        total=np.sum(weights,axis=-1,dtype=np.float64)
        sampled_u=u[:,iy.clip(0,u.shape[1]-1),ix.clip(0,u.shape[2]-1)]
        sampled_v=v[:,iy.clip(0,v.shape[1]-1),ix.clip(0,v.shape[2]-1)]
        weight64=weights.astype(np.float64)
        numerator_u=np.sum(sampled_u.astype(np.float64)*weight64[None],
                           axis=-1,dtype=np.float64)
        numerator_v=np.sum(sampled_v.astype(np.float64)*weight64[None],
                           axis=-1,dtype=np.float64)
        if np.any(total==0.):
            raise ValueError('PROG.DAT cannot interpolate one or more CALMET grid points')
        output_u=(numerator_u/total[None])*0.01
        output_v=(numerator_v/total[None])*0.01
        return (np.asarray(output_u,dtype=MODEL_DTYPE),
                np.asarray(output_v,dtype=MODEL_DTYPE))

    def _record_offsets(self):
        """Unwrap record clocks into an ordered, date-independent timeline."""
        offsets=[]
        day=timedelta(0)
        previous_offset=None
        previous_clock=None
        for index,record in enumerate(self.records):
            clock,is_24=_parse_hour(record.hour)
            if is_24:
                offset=day+_DAY
                day+=_DAY
            else:
                offset=day+clock
                if previous_offset is not None and offset<previous_offset:
                    # A large backwards clock jump is an implicit midnight
                    # (``00.00`` is the conventional alias for ``24.00``).
                    # A small jump is an out-of-order record, not a new day.
                    if previous_clock-clock>=_NOON:
                        day+=_DAY
                        offset=day+clock
                    else:
                        raise ValueError(
                            'PROG.DAT records are out of chronological order at '
                            f'record {index+1} ({record.hour:g})')
            if previous_offset is not None and offset<previous_offset:
                raise ValueError(
                    'PROG.DAT records are out of chronological order at '
                    f'record {index+1} ({record.hour:g})')
            offsets.append(offset)
            previous_offset=offset
            previous_clock=clock
        return tuple(offsets)

    def records_for_times(self,times):
        """Select records for ordered ending datetimes across one or more days.

        Dates are supplied by the caller because ``PROG.DAT`` stores only a
        decimal clock.  Record order supplies day rollovers; leading and
        intervening records earlier than a requested time are skipped.
        """
        targets=tuple(times)
        if not targets:
            return ()
        if any(not isinstance(target,datetime) for target in targets):
            raise TypeError('PROG.DAT target times must be datetime values')
        if any(target.second or target.microsecond for target in targets):
            raise ValueError('PROG.DAT target times must have whole-minute precision')
        try:
            ordered=all(left<right for left,right in zip(targets,targets[1:]))
        except TypeError as error:
            raise ValueError('PROG.DAT target times must use compatible time zones') from error
        if not ordered:
            raise ValueError('PROG.DAT target times must be strictly chronological')

        offsets=self._record_offsets()
        first_clock=timedelta(hours=targets[0].hour,minutes=targets[0].minute)
        candidates=[index for index,record in enumerate(self.records)
                    if _hour_label(record.hour)==first_clock]
        best_count=-1
        missing=targets[0]
        for first_index in candidates:
            origin=targets[0]-offsets[first_index]
            selected=[]
            cursor=first_index
            for target in targets:
                requested=target-origin
                while cursor<len(offsets) and offsets[cursor]<requested:
                    cursor+=1
                if cursor>=len(offsets) or offsets[cursor]!=requested:
                    if len(selected)>best_count:
                        best_count=len(selected)
                        missing=target
                    break
                selected.append(self.records[cursor])
                cursor+=1
            else:
                return tuple(selected)
        raise ValueError(
            'PROG.DAT has no record for requested ending time '
            f'{missing.isoformat(sep=" ")}')
