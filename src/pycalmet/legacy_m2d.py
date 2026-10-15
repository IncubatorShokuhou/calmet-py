"""Streaming reader for CALWRF 2.1 formatted M2D files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True,slots=True)
class M2DHeader:
    nx: int
    ny: int
    nz: int
    nhours: int
    start: datetime
    variables: tuple[str,...]
    source_nx: int = 0
    source_ny: int = 0
    source_nz: int = 0
    i_start: int = 1
    j_start: int = 1
    k_start: int = 1

    @property
    def wire_start(self) -> str:
        """Ten-column timestamp stored in the formatted M2D header."""
        return self.start.strftime('%Y%m%d%H')


@dataclass(frozen=True,slots=True)
class M2DRecord:
    timestamp: datetime
    name: str
    values: np.ndarray

    @property
    def date(self) -> datetime:
        """Compatibility name for the structured record timestamp."""
        return self.timestamp

    @property
    def wire_date(self) -> str:
        """Ten-column timestamp stored in the formatted variable record."""
        return self.timestamp.strftime('%Y%m%d%H')


def _timestamp(field: bytes,description: str) -> datetime:
    """Decode one exact-width ``YYYYMMDDHH`` field at the file boundary."""
    try:
        text=field.decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError(f'invalid M2D {description} timestamp') from error
    if len(text)!=10 or not text.isdecimal():
        raise ValueError(f'invalid M2D {description} timestamp {text!r}')
    try:
        return datetime.strptime(text,'%Y%m%d%H')
    except ValueError as error:
        raise ValueError(f'invalid M2D {description} timestamp {text!r}') from error


class M2DReader:
    def __init__(self,path):
        self.path=Path(path)

    def records(self) -> tuple[M2DHeader,Iterable[M2DRecord]]:
        stream=self.path.open('rb',buffering=1024*1024)
        stream.readline()
        comment_count=int(stream.readline())
        comments=[stream.readline().decode('ascii') for _ in range(comment_count)]
        try:
            variable_count=int(comments[1].split()[0])
            variables=tuple(line.split()[1] for line in comments[2:2+variable_count])
        except (IndexError,ValueError) as error:
            stream.close(); raise ValueError('invalid M2D variable comment block') from error
        stream.readline()
        grid=stream.readline().split(); source_nx,source_ny,source_nz=map(int,grid[8:11])
        stream.readline()
        period=stream.readline().split()
        if len(period)<5:
            stream.close(); raise ValueError('M2D period record is incomplete')
        try:
            start=_timestamp(period[0],'header')
            nhours,nx,ny,nz=map(int,period[1:5])
        except ValueError:
            stream.close(); raise
        bounds=stream.readline().split()
        if len(bounds)<6:
            stream.close(); raise ValueError('M2D extraction-bounds record is incomplete')
        i_start,j_start,i_end,j_end,k_start,k_end=map(int,bounds[:6])
        if ((i_end-i_start+1,j_end-j_start+1,k_end-k_start+1)!=(nx,ny,nz)):
            stream.close(); raise ValueError(
                'M2D extraction bounds do not match subset dimensions')
        for _ in range(nz+nx*ny): stream.readline()
        header=M2DHeader(nx,ny,nz,nhours,start,variables,
                         source_nx,source_ny,source_nz,i_start,j_start,k_start)

        def generate():
            try:
                for _ in range(nhours*variable_count):
                    line=stream.readline()
                    if not line: raise EOFError('M2D ended before a variable record')
                    timestamp=_timestamp(line[:10],'variable-record')
                    name=line[12:20].decode('ascii').strip()
                    rows=[]
                    for _row in range(ny):
                        values=[]
                        while len(values)<nx:
                            data=stream.readline().rstrip(b'\r\n')
                            if not data: raise EOFError('M2D ended inside a gridded row')
                            values.extend(float(data[index:index+10])
                                          for index in range(0,len(data),10)
                                          if data[index:index+10].strip())
                        if len(values)!=nx: raise ValueError('M2D row contains extra values')
                        rows.append(values)
                    yield M2DRecord(timestamp,name,np.asarray(rows[::-1],dtype=np.float32))
            finally:
                stream.close()
        return header,generate()
