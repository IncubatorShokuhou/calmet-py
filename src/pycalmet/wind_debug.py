"""Legacy wind-module disk debug output."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fortran_exponential(value,digits,width):
    value=float(value)
    if value==0.:
        exponent=0; mantissa=0.
    else:
        exponent=int(np.floor(np.log10(abs(value))))+1
        mantissa=value/(10.**exponent)
        if abs(round(mantissa,digits))>=1.:
            mantissa/=10.; exponent+=1
    return f'{mantissa:.{digits}f}E{exponent:+03d}'.rjust(width)


class FinalWindDebugWriter:
    """Reproduce ``OUTFIL`` records written to ``TEST.OUT`` for ``IPR8``."""

    def __init__(self,path,nx,ny,nz,dx_km,dy_km):
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        self.stream=target.open('w',encoding='ascii')
        self.nx=int(nx); self.ny=int(ny); self.nz=int(nz)
        self.dx_km=float(dx_km); self.dy_km=float(dy_km)
        self.first_hour=None

    @staticmethod
    def _rows(values,format_spec,width):
        array=np.asarray(values)
        for row in array:
            for first in range(0,row.size,width):
                yield ''.join(format(value,format_spec) for value in row[first:first+width])

    def append(self,hour):
        self.append_fields(hour.timestamp,hour.wind.u,hour.wind.v,hour.wind.w)

    def append_fields(self,timestamp,u,v,w):
        current=timestamp.replace(minute=0,second=0,microsecond=0)
        if self.first_hour is None: self.first_hour=current
        nh=int((current-self.first_hour).total_seconds()//3600)+1
        u=np.asarray(u); v=np.asarray(v); w=np.asarray(w)
        if u.shape!=(self.nz,self.ny,self.nx) or v.shape!=u.shape:
            raise ValueError('TEST.OUT U/V fields do not match the configured grid')
        if w.shape!=(self.nz+1,self.ny,self.nx):
            raise ValueError('TEST.OUT W fields do not match the configured grid')
        stream=self.stream
        stream.write(f'{float((nh-1)*100):5.0f}.\n')
        stream.write(f'{self.nx:4d}{self.ny:4d}{self.nz:4d}\n')
        stream.write(f'{self.dx_km:3.0f}.{self.dy_km:3.0f}.\n')
        for line in self._rows(u.reshape(-1,self.nx),'7.2f',59): stream.write(line+'\n')
        for line in self._rows(v.reshape(-1,self.nx),'7.2f',59): stream.write(line+'\n')
        for line in self._rows(w.reshape(-1,self.nx),'8.1E',59): stream.write(line+'\n')
        stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream=None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


def format_wind_number_map(values):
    """Reproduce the legacy ``WNDLPT`` two-digit scaled number map."""
    data=np.asarray(values,dtype=np.float32)
    if data.ndim!=2: raise ValueError('WNDLPT fields must be two-dimensional')
    maximum=float(np.max(np.abs(data)))
    if maximum==0.:
        return "\n         FIELD CONTAINS ALL ZEROS - PRINTING SUPPRESSED\n"
    if maximum<10.:
        factor=next(np.float32(10.)**power for power in range(1,21)
                    if int((np.float32(10.)**power)*maximum)>=10)
    elif maximum>=100.:
        factor=next(np.float32(10.)**(-power) for power in range(1,21)
                    if int((np.float32(10.)**(-power))*maximum)<100)
    else:
        factor=np.float32(1.)
    scaled=np.asarray(data*factor,dtype=np.float32)
    mapped=np.where(scaled>0.,np.trunc(scaled+np.float32(.5)),
                    np.where(scaled<0.,np.trunc(scaled-np.float32(.5)),0.)).astype(int)
    lines=[]; ny=min(data.shape[0],36)
    for first in range(0,min(data.shape[1],72),36):
        last=min(first+36,data.shape[1],72)
        lines.append('')
        lines.append('       '+''.join(f'{index:3d}' for index in range(first+1,last+1)))
        lines.append('      I'+'--+'*(last-first))
        for row in range(ny-1,-1,-1):
            lines.append(f'   {row+1:2d} I'+''.join(
                f'{value:3d}' for value in mapped[row,first:last]))
    scale_text=_fortran_exponential(factor,1,9)
    lines.extend(('',f'         ARRAY HAS BEEN SCALED BY{scale_text}  FOR PRINTING'))
    return '\n'.join(lines)+'\n'


class WindPrintWriter:
    """Write the requested wind-module diagnostic maps to ``TEST.PRT``."""

    def __init__(self,path,fields,levels=None):
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        self.stream=target.open('w',encoding='ascii')
        self.fields=tuple(fields); self.levels=levels; self.first_hour=None

    def _windpr(self,u,v,w,include_w):
        for level in range(u.shape[0]):
            if include_w and level:
                self.stream.write(
                    f'\n     Z-COMPONENT OF WIND (W) BETWEEN LEVELS{level:4d} & {level+1:2d}'
                    '     (M/SEC)\n     '+'-'*59+'\n')
                self.stream.write(format_wind_number_map(w[level]))
            self.stream.write(
                f'\n     X-COMPONENT OF WIND (U) AT LEVEL = {level+1:4d}     (M/SEC)\n'
                '     '+'-'*49+'\n')
            self.stream.write(format_wind_number_map(u[level]))
            self.stream.write(
                f'\n     Y-COMPONENT OF WIND (V) AT LEVEL = {level+1:4d}     (M/SEC)\n'
                '     '+'-'*49+'\n')
            self.stream.write(format_wind_number_map(v[level]))

    def _speed_direction(self,u,v):
        """Print final horizontal-wind speed and meteorological direction."""
        u=np.asarray(u); v=np.asarray(v)
        levels=u.shape[0] if self.levels is None else min(int(self.levels),u.shape[0])
        for level in range(levels):
            speed=np.hypot(u[level],v[level])
            direction=np.mod(
                270.-np.rad2deg(np.arctan2(v[level],u[level])),360.)
            direction=np.where(
                speed<1.e-5,0.,np.where(direction==0.,360.,direction))
            self.stream.write(
                f'\n\n     WIND SPEED (M/S) AT LEVEL = {level+1:4d}\n'
                '     '+'-'*26+'\n')
            self.stream.write(format_wind_number_map(speed))
            self.stream.write(
                f'\n\n     WIND DIRECTION (DEGREES) AT LEVEL = {level+1:4d}\n'
                '     '+'-'*34+'\n')
            self.stream.write(format_wind_number_map(direction))

    def append(self,hour):
        current=hour.timestamp.replace(minute=0,second=0,microsecond=0)
        if self.first_hour is None: self.first_hour=current
        nh=int((current-self.first_hour).total_seconds()//3600)+1
        stages=((5,'KINEMATIC EFFECTS',17,'kinematic_stage',True),
                (6,'FROUDE NUMBER EFFECTS',21,'froude_stage',False),
                (7,'SLOPE EFFECTS',13,'slope_stage',False))
        for index,title,dashes,attribute,include_w in stages:
            if self.fields[index]:
                stage=getattr(hour.wind,attribute)
                if stage is None: raise RuntimeError(f'missing retained wind stage {attribute}')
                self.stream.write(f'\n          {title}\n          '+dashes*'-'+'\n')
                self._windpr(*stage,include_w=include_w)
        if self.fields[0]:
            stage=hour.wind.interpolated_stage
            if stage is None: raise RuntimeError('missing retained interpolated wind stage')
            self.stream.write('           INTERPOLATED WIND FIELD\n           '+23*'-'+'\n')
            self._windpr(*stage,np.empty(0),include_w=False)
        if self.fields[1]:
            stage=hour.wind.terrain_adjusted_stage
            if stage is None: raise RuntimeError('missing retained terrain-adjusted wind stage')
            self.stream.write('\n          TERRAIN ADJUSTED SURFACE WINDS\n          '+30*'-'+'\n')
            self._windpr(*stage,np.empty(0),include_w=False)
        if self.fields[2]:
            stage=hour.wind.initial_stage
            if stage is None: raise RuntimeError('missing retained initial wind stage')
            u,v,w,divergence=stage
            self.stream.write('\n          SMOOTHED WIND AND INITIAL DIVERGENCE FIELDS\n'
                              '          '+43*'-'+'\n')
            self._windpr(u,v,w,include_w=True)
            maximum=float(np.max(np.abs(divergence)))
            self.stream.write('\n\n                    MAXIMUM DIVERGENCE = '+
                              _fortran_exponential(maximum,4,12)+'\n')
            for level in range(min(int(self.levels),divergence.shape[0])):
                self.stream.write(
                    f'\n\n     DIVERGENCE AT LEVEL = {level+1:4d}\n'
                    '     '+'-'*24+'\n')
                self.stream.write(format_wind_number_map(divergence[level]))
        if self.fields[8]:
            time=float((nh-1)*100)
            self.stream.write(
                f'          FINAL WIND FIELD AT TIME = {time:6.0f}.'
                f'     (SIMULATION HOUR NO.{nh:3d} )\n          '+63*'-'+'\n')
            self._windpr(hour.wind.u,hour.wind.v,hour.wind.w,include_w=True)
        levels=hour.wind.u.shape[0] if self.levels is None else min(
            int(self.levels),hour.wind.u.shape[0])
        if self.fields[4]:
            self.stream.write('\n          FINAL DIVERGENCE FIELD AT TIME =\n')
            for level in range(levels):
                self.stream.write(
                    f'\n\n     DIVERGENCE AT LEVEL = {level+1:4d}\n'
                    '     '+'-'*24+'\n')
                self.stream.write(format_wind_number_map(hour.wind.divergence[level]))
        if self.fields[3]:
            time=float((nh-1)*100)
            self.stream.write(
                f'\n          FINAL WIND SPEED AND DIRECTION AT TIME = {time:7.0f}.'
                f'     (SIMULATION HOUR NO.{nh:3d} )\n          '+72*'-'+'\n')
            self._speed_direction(hour.wind.u,hour.wind.v)
        self.stream.flush()

    def close(self):
        if self.stream is not None: self.stream.close(); self.stream=None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
