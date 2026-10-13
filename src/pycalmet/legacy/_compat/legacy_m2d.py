"""Streaming reader for CALWRF 2.1 formatted M2D files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True,slots=True)
class M2DHeader:
    nx: int
    ny: int
    nz: int
    nhours: int
    start: int
    variables: tuple[str,...]
    source_nx: int = 0
    source_ny: int = 0
    source_nz: int = 0
    i_start: int = 1
    j_start: int = 1
    k_start: int = 1


@dataclass(frozen=True,slots=True)
class M2DRecord:
    date: int
    name: str
    values: np.ndarray


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
        period=[int(value) for value in stream.readline().split()]
        if len(period)<5:
            stream.close(); raise ValueError('M2D period record is incomplete')
        start,nhours,nx,ny,nz=period[:5]
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
                    date=int(line[:10]); name=line[12:20].decode('ascii').strip()
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
                    yield M2DRecord(date,name,np.asarray(rows[::-1],dtype=np.float32))
            finally:
                stream.close()
        return header,generate()
