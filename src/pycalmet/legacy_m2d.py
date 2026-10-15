"""Streaming reader for CALWRF 2.1 formatted M2D files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True,slots=True)
class M2DHeader:
    """Static metadata from a formatted CALWRF M2D file.

    Attributes:
        nx: Extracted east-west cell count.
        ny: Extracted north-south cell count.
        nz: Extracted vertical-level count recorded in the header.
        nhours: Number of hourly periods.
        start: First source timestamp.
        variables: Variable labels in record order.
        source_nx: Full source-grid east-west dimension.
        source_ny: Full source-grid north-south dimension.
        source_nz: Full source-grid vertical dimension.
        i_start: One-based west extraction index.
        j_start: One-based south extraction index.
        k_start: One-based bottom extraction index.
    """
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
        """Return the timestamp stored in the M2D header.

        Returns:
            Ten-character ``YYYYMMDDHH`` timestamp.
        """
        return self.start.strftime('%Y%m%d%H')


@dataclass(frozen=True,slots=True)
class M2DRecord:
    """One timestamped two-dimensional M2D variable.

    Attributes:
        timestamp: Timestamp attached to the source record.
        name: Eight-character variable label with padding removed.
        values: ``float32`` array with shape ``(ny, nx)`` in south-to-north
            row order.
    """
    timestamp: datetime
    name: str
    values: npt.NDArray[np.float32]

    @property
    def date(self) -> datetime:
        """Expose the timestamp through the historical ``date`` name.

        Returns:
            Structured record timestamp.
        """
        return self.timestamp

    @property
    def wire_date(self) -> str:
        """Return the timestamp stored in the variable record.

        Returns:
            Ten-character ``YYYYMMDDHH`` timestamp.
        """
        return self.timestamp.strftime('%Y%m%d%H')


def _timestamp(field: bytes,description: str) -> datetime:
    """Decode one exact-width ``YYYYMMDDHH`` boundary field.

    Args:
        field: Exactly ten ASCII timestamp bytes.
        description: Field role included in validation errors.

    Returns:
        Naive hourly timestamp.

    Raises:
        ValueError: If the bytes are non-ASCII, wrong-width, nonnumeric, or do
            not form a valid calendar timestamp.
    """
    try:
        text: str=field.decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError(f'invalid M2D {description} timestamp') from error
    if len(text)!=10 or not text.isdecimal():
        raise ValueError(f'invalid M2D {description} timestamp {text!r}')
    try:
        return datetime.strptime(text,'%Y%m%d%H')
    except ValueError as error:
        raise ValueError(f'invalid M2D {description} timestamp {text!r}') from error


class M2DReader:
    """Stream structured records from a formatted CALWRF M2D file.

    Attributes:
        path: Source M2D file path.
    """
    def __init__(self,path: str | Path) -> None:
        """Store the M2D source path without opening it.

        Args:
            path: Existing formatted M2D path.
        """
        self.path: Path=Path(path)

    def records(self) -> tuple[M2DHeader,Iterator[M2DRecord]]:
        """Read the header and return a lazy variable-record iterator.

        Returns:
            Tuple containing parsed header metadata and a single-pass iterator
            of two-dimensional records. The iterator owns and closes the file.

        Raises:
            OSError: If the file cannot be opened or read.
            UnicodeError: If required text records are not ASCII.
            ValueError: If comments, dimensions, bounds, or numeric rows are
                malformed.
            EOFError: If iteration reaches a truncated variable record.
        """
        stream: BinaryIO=self.path.open('rb',buffering=1024*1024)
        stream.readline()
        comment_count: int=int(stream.readline())
        comments: list[str]=[stream.readline().decode('ascii') for _ in range(comment_count)]
        try:
            variable_count: int=int(comments[1].split()[0])
            variables: tuple[str,...]=tuple(line.split()[1] for line in comments[2:2+variable_count])
        except (IndexError,ValueError) as error:
            stream.close(); raise ValueError('invalid M2D variable comment block') from error
        stream.readline()
        grid: list[bytes]=stream.readline().split()
        source_nx: int; source_ny: int; source_nz: int
        source_nx,source_ny,source_nz=map(int,grid[8:11])
        stream.readline()
        period: list[bytes]=stream.readline().split()
        if len(period)<5:
            stream.close(); raise ValueError('M2D period record is incomplete')
        try:
            start: datetime=_timestamp(period[0],'header')
            nhours: int; nx: int; ny: int; nz: int
            nhours,nx,ny,nz=map(int,period[1:5])
        except ValueError:
            stream.close(); raise
        bounds: list[bytes]=stream.readline().split()
        if len(bounds)<6:
            stream.close(); raise ValueError('M2D extraction-bounds record is incomplete')
        i_start: int; j_start: int; i_end: int; j_end: int
        k_start: int; k_end: int
        i_start,j_start,i_end,j_end,k_start,k_end=map(int,bounds[:6])
        if ((i_end-i_start+1,j_end-j_start+1,k_end-k_start+1)!=(nx,ny,nz)):
            stream.close(); raise ValueError(
                'M2D extraction bounds do not match subset dimensions')
        for _ in range(nz+nx*ny): stream.readline()
        header: M2DHeader=M2DHeader(nx,ny,nz,nhours,start,variables,
                                   source_nx,source_ny,source_nz,
                                   i_start,j_start,k_start)

        def generate() -> Iterator[M2DRecord]:
            """Yield decoded variable records and close the owned stream.

            Yields:
                Timestamped ``float32`` M2D grids in file order.

            Raises:
                EOFError: If a variable header or grid row is truncated.
                ValueError: If a row has the wrong number of values.
            """
            try:
                for _ in range(nhours*variable_count):
                    line: bytes=stream.readline()
                    if not line: raise EOFError('M2D ended before a variable record')
                    timestamp: datetime=_timestamp(line[:10],'variable-record')
                    name: str=line[12:20].decode('ascii').strip()
                    rows: list[list[float]]=[]
                    for _row in range(ny):
                        values: list[float]=[]
                        while len(values)<nx:
                            data: bytes=stream.readline().rstrip(b'\r\n')
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
