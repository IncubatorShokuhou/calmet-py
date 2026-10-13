"""Reader for CALMET.AUX 1.1 cloud-liquid-water files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from .legacy_io import FortranSequentialReader


@dataclass(frozen=True,slots=True)
class AuxiliaryHeader:
    comments: tuple[str,...]
    start: tuple[int,int,int,int,int]
    end: tuple[int,int,int,int,int]
    timezone: str
    run_hours: int
    nx: int
    ny: int
    nz: int
    spacing_m: float
    x_origin_m: float
    y_origin_m: float
    projection: str
    datum: str
    datum_date: str
    false_easting_km: float
    false_northing_km: float
    hemisphere: str
    utm_zone: int
    latitude_origin: float
    longitude_origin: float
    standard_parallel_1: float
    standard_parallel_2: float
    z_faces_m: np.ndarray


@dataclass(frozen=True,slots=True)
class AuxiliaryCloudHour:
    begin_code: int
    begin_seconds: int
    end_code: int
    end_seconds: int
    cloud_water_above_g_kg: np.ndarray
    cloud_bottom_above_m: np.ndarray
    cloud_top_above_m: np.ndarray
    cloud_water_3d_g_kg: np.ndarray


class AuxiliaryData:
    """Stream and validate a GNU sequential-unformatted CALMET.AUX file."""
    _fields_2d=('CLDMRUP','ZCLDBOT','ZCLDTOP')

    def __init__(self,path: str | Path):
        self.path=Path(path)
        self.reader=FortranSequentialReader(self.path)
        self.header=self._read_header()

    @staticmethod
    def _text(value): return value.decode('ascii').rstrip()

    def _record(self,description):
        record=self.reader.read_record()
        if record is None: raise ValueError(f'truncated CALMET.AUX {description}')
        return record

    def _read_header(self):
        declaration=self._record('declaration')
        if (len(declaration)!=96 or self._text(declaration[:16])!='CALMET.AUX' or
                self._text(declaration[16:32])!='1.1'):
            raise ValueError('unsupported CALMET.AUX declaration')
        count_record=self._record('comment count')
        if len(count_record)!=4: raise ValueError('invalid CALMET.AUX comment count')
        count=struct.unpack('<i',count_record)[0]
        if count<1: raise ValueError('CALMET.AUX must contain a producer comment')
        comments=tuple(self._text(self._record('comment')) for _ in range(count))
        run=self._record('run control')
        if len(run)!=144: raise ValueError('invalid CALMET.AUX run-control record')
        dates=struct.unpack('<10i',run[:40])
        timezone=self._text(run[40:48])
        run_hours,nx,ny,nz=struct.unpack('<4i',run[48:64])
        spacing,x_origin,y_origin=struct.unpack('<3f',run[64:76])
        projection=self._text(run[76:84]); datum=self._text(run[84:92])
        datum_date=self._text(run[92:104])
        false_easting,false_northing=struct.unpack('<2f',run[104:112])
        hemisphere=self._text(run[112:116]); utm_zone=struct.unpack('<i',run[116:120])[0]
        angles=struct.unpack('<4f',run[120:136])
        naux2d,naux3d=struct.unpack('<2i',run[136:144])
        if (nx<=0 or ny<=0 or nz<=0 or naux2d!=3 or naux3d!=1):
            raise ValueError('unsupported CALMET.AUX grid or variable counts')
        zface=self._record('ZFACE')
        if len(zface)!=24+4*(nz+1) or self._text(zface[:8])!='ZFACE':
            raise ValueError('invalid CALMET.AUX ZFACE record')
        faces=np.frombuffer(zface[24:],dtype='<f4').copy()
        if faces[0]!=0. or np.any(np.diff(faces)<=0.):
            raise ValueError('invalid CALMET.AUX vertical faces')
        expected_2d=(b'2D_VARS '+b''.join(name.encode().ljust(8,b' ') for name in self._fields_2d)+
            b'G/KG    MAGL    MAGL    RC4 RC4 RC4 ')
        expected_3d=b'3D_VARS CLDMR3D G/KG    RC4 '
        if self._record('2-D variables')!=expected_2d:
            raise ValueError('unsupported CALMET.AUX 2-D variable table')
        if self._record('3-D variables')!=expected_3d:
            raise ValueError('unsupported CALMET.AUX 3-D variable table')
        return AuxiliaryHeader(comments,dates[:5],dates[5:],timezone,run_hours,nx,ny,nz,
            spacing,x_origin,y_origin,projection,datum,datum_date,false_easting,
            false_northing,hemisphere,utm_zone,*angles,faces)

    def _timestamp(self):
        record=self._record('timestamp')
        if len(record)!=16: raise ValueError('invalid CALMET.AUX timestamp record')
        return struct.unpack('<4i',record)

    def _compressed(self,label):
        count_record=self._record(f'{label} compressed count')
        if len(count_record)!=4: raise ValueError('invalid CALMET.AUX compressed count')
        count=struct.unpack('<i',count_record)[0]
        record=self._record(label)
        if len(record)!=12+4*count or self._text(record[:12])!=label:
            raise ValueError(f'invalid CALMET.AUX {label} record')
        encoded=np.frombuffer(record[12:],dtype='<f4')
        negative=encoded<0.
        lengths=np.where(negative,np.trunc(-encoded).astype(np.int64),1)
        ends=np.cumsum(lengths)
        size=self.header.nx*self.header.ny
        if count<1 or ends[-1]!=size:
            raise ValueError(f'CALMET.AUX {label} compression length mismatch')
        values=np.zeros(size,dtype=np.float32)
        values[ends[~negative]-1]=encoded[~negative]
        return values.reshape(self.header.ny,self.header.nx)

    def read_hour(self):
        first=self.reader.read_record()
        if first is None: return None
        if len(first)!=16: raise ValueError('invalid CALMET.AUX timestamp record')
        stamp=struct.unpack('<4i',first)
        fields=[]
        for index,label in enumerate(self._fields_2d):
            if index and self._timestamp()!=stamp:
                raise ValueError('CALMET.AUX 2-D timestamps are inconsistent')
            fields.append(self._compressed(label))
        if self._timestamp()!=stamp:
            raise ValueError('CALMET.AUX 3-D timestamp is inconsistent')
        layers=np.stack([self._compressed(f'CLDMR3D {level:04d}')
                         for level in range(1,self.header.nz+1)])
        return AuxiliaryCloudHour(*stamp,*fields,layers)

    def close(self): self.reader.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
