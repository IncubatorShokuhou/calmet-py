"""Sequential scalar records from legacy CALMET DIAG.DAT files."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    surface_temperature_k: float | None = None
    lapse_rate_k_m: float | None = None
    initial_u_m_s: float | None = None
    initial_v_m_s: float | None = None
    surface_u_m_s: np.ndarray | None = None
    surface_v_m_s: np.ndarray | None = None
    upper_u_m_s: np.ndarray | None = None
    upper_v_m_s: np.ndarray | None = None


def _fixed_value(line: str, label: str, width: int) -> float:
    """Read the value following the ten columns skipped by formats 1010/1030."""
    if len(line) < 10+width:
        raise ValueError(f'DIAG.DAT {label} record is shorter than ten columns')
    try:
        return float(line[10:10+width])
    except ValueError as error:
        raise ValueError(f'invalid DIAG.DAT {label} record') from error


class DiagnosticData:
    """Index supported PREPDI DIAG.DAT records by CALMET interval start."""

    def __init__(self, path, start: datetime, count: int, timestep_seconds: int,
                 *, surface_temperature=False, lapse_rate=False, initial_wind=False,
                 surface_wind=False,upper_wind=False,surface_wind_names=(),
                 upper_wind_names=(),levels=0,iextrp=-1):
        self.path=Path(path)
        lines=iter(self.path.read_text(encoding='ascii',errors='strict').splitlines())

        def take(label,width):
            try: return _fixed_value(next(lines),label,width)
            except StopIteration as error:
                raise ValueError(f'DIAG.DAT ends before {label} record') from error

        def line(label):
            try: return next(lines)
            except StopIteration as error:
                raise ValueError(f'DIAG.DAT ends before {label} record') from error

        def names(values,label):
            normalized=tuple(str(item).strip().upper()[:4] for item in values)
            if len(set(normalized))!=len(normalized):
                raise ValueError(f'DIAG.DAT {label} station names are not unique in A4 format')
            return normalized

        surface_names=names(surface_wind_names,'surface')
        upper_names=names(upper_wind_names,'upper-air')
        read_surface_wind=bool(surface_wind or surface_names)
        read_upper_wind=bool(upper_wind or upper_names)
        nz=int(levels)
        if nz<0: raise ValueError('DIAG.DAT level count must be non-negative')

        def fields(text,start,number,label):
            if len(text)<start+5*number:
                raise ValueError(f'DIAG.DAT {label} record is too short')
            try: return [float(text[start+5*i:start+5*(i+1)]) for i in range(number)]
            except ValueError as error: raise ValueError(f'invalid DIAG.DAT {label} record') from error

        def surface_block():
            u=np.full(len(surface_names),9999.); v=np.full(len(surface_names),9999.)
            lookup={name:index for index,name in enumerate(surface_names)}
            while True:
                text=line('surface wind')
                if len(text)<19: raise ValueError('DIAG.DAT surface wind record is too short')
                name=text[15:19].strip().upper()
                if name=='LAST': break
                if not name: continue
                if name not in lookup: raise ValueError(f'unrecognized DIAG.DAT surface station {name!r}')
                _,u_value,v_value=fields(text,20,3,'surface wind')
                index=lookup[name]
                u[index]=9999. if u_value==999. else u_value
                v[index]=9999. if v_value==999. else v_value
            return u,v

        def upper_block():
            u=np.full((nz,len(upper_names)),9999.); v=np.full_like(u,9999.)
            lookup={name:index for index,name in enumerate(upper_names)}
            total=2*nz+1; first=min(total,11)
            while True:
                text=line('upper-air wind')
                if len(text)<17: raise ValueError('DIAG.DAT upper-air wind record is too short')
                name=text[13:17].strip().upper()
                values=fields(text,18,first,'upper-air wind')
                if total>11:
                    values+=fields(line('upper-air wind continuation'),20,total-11,
                                   'upper-air wind continuation')
                if name=='LAST': break
                if name not in lookup: raise ValueError(f'unrecognized DIAG.DAT upper-air station {name!r}')
                index=lookup[name]
                for level in range(nz):
                    u_value,v_value=values[1+2*level:3+2*level]
                    if level==0 and (iextrp<=0 or name in surface_names): continue
                    if u_value!=999. and v_value!=999.:
                        u[level,index]=u_value; v[level,index]=v_value
            return u,v

        records={}
        for index in range(int(count)):
            temperature=take('surface temperature',6) if surface_temperature else None
            # GAMDIA is read in K/km and converted to K/m by PREPDI.
            gamma=take('lapse rate',5)/1000. if lapse_rate else None
            u=take('initial U wind',5) if initial_wind else None
            v=take('initial V wind',5) if initial_wind else None
            surface_u,surface_v=(surface_block() if read_surface_wind else (None,None))
            upper_u,upper_v=(upper_block() if read_upper_wind else (None,None))
            when=start+timedelta(seconds=index*timestep_seconds)
            records[when]=DiagnosticRecord(temperature,gamma,u,v,
                surface_u,surface_v,upper_u,upper_v)
        self.records=records

    def at_begin(self, when: datetime) -> DiagnosticRecord:
        try: return self.records[when]
        except KeyError as error:
            raise ValueError(f'DIAG.DAT has no record for {when}') from error
