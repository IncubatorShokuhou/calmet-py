"""Reader for CALMET 6 CLOUD.DAT gridded cloud-fraction records."""
from __future__ import annotations

from datetime import datetime,timedelta
from pathlib import Path
import shlex
import struct
import numpy as np

from .legacy_io import FortranSequentialReader
from .legacy_io import FortranSequentialWriter,fixed_string
from .datetime_utils import format_ordinal_code,parse_ordinal_code


def _datetime(code: int, seconds: int) -> datetime:
    try: return parse_ordinal_code(code,seconds)
    except ValueError as error: raise ValueError(f'invalid CLOUD.DAT timestamp {code}/{seconds}') from error


class CloudData:
    """In-memory indexed CLOUD.DAT records, keyed by interval ending time."""
    def __init__(self,path,nx,ny,format=2):
        self.path=Path(path); self.nx=int(nx); self.ny=int(ny)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        if format==2: records=self._formatted()
        elif format==1: records=self._unformatted()
        else: raise ValueError('CLOUD.DAT format/IFORMC must be 1 or 2')
        self.records={}
        for begin,end,field in records:
            if end in self.records: raise ValueError(f'duplicate CLOUD.DAT ending time: {end}')
            if begin>=end: raise ValueError(f'CLOUD.DAT interval does not advance: {begin}--{end}')
            if not np.isfinite(field).all() or np.any((field<0.)|(field>1.)):
                raise ValueError(f'CLOUD.DAT cloud fraction outside 0..1 at {end}')
            self.records[end]=field
        if not self.records: raise ValueError('CLOUD.DAT contains no records')

    def _formatted(self):
        tokens=shlex.split(self.path.read_text(encoding='ascii',errors='strict'),comments=False)
        width=5+self.nx*self.ny
        if len(tokens)%width: raise ValueError('formatted CLOUD.DAT has an incomplete record')
        for offset in range(0,len(tokens),width):
            row=tokens[offset:offset+width]
            if row[0].upper()!='CLOUDFRA': raise ValueError(f'invalid CLOUD.DAT label: {row[0]!r}')
            try:
                begin=_datetime(int(row[1]),int(row[2])); end=_datetime(int(row[3]),int(row[4]))
                values=np.asarray(row[5:],dtype='f4').reshape(self.ny,self.nx)
            except ValueError as error: raise ValueError('invalid formatted CLOUD.DAT record') from error
            yield begin,end,values

    def _unformatted(self):
        expected=24+4*self.nx*self.ny
        with FortranSequentialReader(self.path) as reader:
            while (payload:=reader.read_record()) is not None:
                if len(payload)!=expected: raise ValueError('unformatted CLOUD.DAT record has wrong grid size')
                label=payload[:8].decode('ascii').rstrip()
                if label!='CLOUDFRA': raise ValueError(f'invalid CLOUD.DAT label: {label!r}')
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[8:24])
                # Fortran X(NX,NY): I varies fastest, identical to C (NY,NX).
                values=np.frombuffer(payload,dtype='<f4',offset=24).reshape(self.ny,self.nx).copy()
                yield _datetime(begin_code,begin_sec),_datetime(end_code,end_sec),values

    def at_end(self,when):
        try: return self.records[when]
        except KeyError as error: raise ValueError(f'CLOUD.DAT has no record ending at {when}') from error


class CloudDataWriter:
    """Write OUTCLD's formatted or sequential-unformatted records."""
    def __init__(self,path,nx,ny,format=2):
        self.path=Path(path); self.nx=int(nx); self.ny=int(ny); self.format=int(format)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.format==1: self.stream=FortranSequentialWriter(self.path)
        elif self.format==2: self.stream=self.path.open('w',encoding='ascii')
        else: raise ValueError('CLOUD.DAT format/IFORMC must be 1 or 2')

    @staticmethod
    def _code(value):
        return format_ordinal_code(value),value.minute*60+value.second

    def append(self,begin,end,field):
        values=np.asarray(field,dtype='<f4')
        if values.shape!=(self.ny,self.nx):
            raise ValueError('CLOUD.DAT field does not match configured grid')
        if not np.isfinite(values).all() or np.any((values<0.)|(values>1.)):
            raise ValueError('CLOUD.DAT cloud fraction outside 0..1')
        begin_code,begin_seconds=self._code(begin); end_code,end_seconds=self._code(end)
        if self.format==1:
            self.stream.write_record(fixed_string('CLOUDFRA',8),
                struct.pack('<4i',begin_code,begin_seconds,end_code,end_seconds),
                values.ravel(order='C'))
        else:
            numbers=' '.join(format(float(value),'.9g') for value in values.ravel(order='C'))
            self.stream.write(f'"CLOUDFRA" {begin_code} {begin_seconds} '
                              f'{end_code} {end_seconds} {numbers}\n')
            self.stream.flush()

    def close(self):
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
