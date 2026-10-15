"""Reader for CALMET 6 CLOUD.DAT gridded cloud-fraction records."""
from __future__ import annotations

from datetime import datetime,timedelta
from pathlib import Path
import shlex
import struct
from types import TracebackType
from collections.abc import Iterator
from typing import TextIO, TypeAlias, cast
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .legacy_io import FortranSequentialReader
from .legacy_io import FortranSequentialWriter,fixed_string
from .datetime_utils import format_ordinal_code,parse_ordinal_code


Float32Array: TypeAlias = NDArray[np.float32]


def _datetime(code: int, seconds: int) -> datetime:
    """Decode a CLOUD.DAT ordinal timestamp.

    Args:
        code: Integer ``YYYYJJJHH`` ordinal-date code.
        seconds: Seconds after the encoded hour.

    Returns:
        Decoded naive :class:`datetime` in the file's declared time basis.

    Raises:
        ValueError: If the code or seconds do not form a valid timestamp.
    """
    try: return parse_ordinal_code(code,seconds)
    except ValueError as error: raise ValueError(f'invalid CLOUD.DAT timestamp {code}/{seconds}') from error


class CloudData:
    """In-memory indexed CLOUD.DAT records, keyed by interval ending time.

    Attributes:
        path: Path to the CLOUD.DAT input file.
        nx: Number of grid columns.
        ny: Number of grid rows.
        records: Mapping from interval-ending datetimes to float32 cloud
            fractions of shape ``(ny, nx)`` and range 0--1.
    """

    def __init__(self,path: str | Path,nx: int,ny: int,format: int=2) -> None:
        """Read and validate every cloud-fraction record.

        Args:
            path: Path to CLOUD.DAT.
            nx: Number of grid columns expected in every field.
            ny: Number of grid rows expected in every field.
            format: CALMET ``IFORMC`` value: ``1`` for sequential-unformatted
                binary or ``2`` for formatted text.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the format selector, timestamps, record lengths,
                labels, duplicate ending times, or cloud fractions are invalid.
        """
        self.path: Path=Path(path); self.nx: int=int(nx); self.ny: int=int(ny)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        if format==2: records=self._formatted()
        elif format==1: records=self._unformatted()
        else: raise ValueError('CLOUD.DAT format/IFORMC must be 1 or 2')
        self.records: dict[datetime,Float32Array]={}
        for begin,end,field in records:
            if end in self.records: raise ValueError(f'duplicate CLOUD.DAT ending time: {end}')
            if begin>=end: raise ValueError(f'CLOUD.DAT interval does not advance: {begin}--{end}')
            if not np.isfinite(field).all() or np.any((field<0.)|(field>1.)):
                raise ValueError(f'CLOUD.DAT cloud fraction outside 0..1 at {end}')
            self.records[end]=field
        if not self.records: raise ValueError('CLOUD.DAT contains no records')

    def _formatted(self) -> Iterator[tuple[datetime,datetime,Float32Array]]:
        """Iterate over records decoded from formatted CLOUD.DAT text.

        Yields:
            Tuples ``(begin, end, values)``. ``values`` is a float32 array of
            shape ``(ny, nx)`` containing cloud fraction from 0 to 1.

        Raises:
            UnicodeDecodeError: If the file is not ASCII.
            ValueError: If a record is incomplete or contains an invalid label,
                timestamp, or numeric field.
        """
        tokens=shlex.split(self.path.read_text(encoding='ascii',errors='strict'),comments=False)
        width=5+self.nx*self.ny
        if len(tokens)%width: raise ValueError('formatted CLOUD.DAT has an incomplete record')
        for offset in range(0,len(tokens),width):
            row=tokens[offset:offset+width]
            if row[0].upper()!='CLOUDFRA': raise ValueError(f'invalid CLOUD.DAT label: {row[0]!r}')
            try:
                begin=_datetime(int(row[1]),int(row[2])); end=_datetime(int(row[3]),int(row[4]))
                values: Float32Array=np.asarray(row[5:],dtype='f4').reshape(self.ny,self.nx)
            except ValueError as error: raise ValueError('invalid formatted CLOUD.DAT record') from error
            yield begin,end,values

    def _unformatted(self) -> Iterator[tuple[datetime,datetime,Float32Array]]:
        """Iterate over sequential-unformatted CLOUD.DAT records.

        Yields:
            Tuples ``(begin, end, values)``. ``values`` is a float32 array of
            shape ``(ny, nx)`` containing cloud fraction from 0 to 1.

        Raises:
            ValueError: If a record has the wrong byte length, label, or
                timestamp encoding.
        """
        expected=24+4*self.nx*self.ny
        with FortranSequentialReader(self.path) as reader:
            while (payload:=reader.read_record()) is not None:
                if len(payload)!=expected: raise ValueError('unformatted CLOUD.DAT record has wrong grid size')
                label=payload[:8].decode('ascii').rstrip()
                if label!='CLOUDFRA': raise ValueError(f'invalid CLOUD.DAT label: {label!r}')
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[8:24])
                # Fortran X(NX,NY): I varies fastest, identical to C (NY,NX).
                values: Float32Array=np.frombuffer(payload,dtype='<f4',offset=24).reshape(self.ny,self.nx).copy()
                yield _datetime(begin_code,begin_sec),_datetime(end_code,end_sec),values

    def at_end(self,when: datetime) -> Float32Array:
        """Return the cloud field for an exact interval-ending time.

        Args:
            when: Interval-ending datetime to retrieve.

        Returns:
            Float32 cloud-fraction array of shape ``(ny, nx)``.

        Raises:
            ValueError: If no record ends at ``when``.
        """
        try: return self.records[when]
        except KeyError as error: raise ValueError(f'CLOUD.DAT has no record ending at {when}') from error


class CloudDataWriter:
    """Write OUTCLD formatted or sequential-unformatted records.

    Attributes:
        path: Destination CLOUD.DAT path.
        nx: Number of grid columns required by :meth:`append`.
        ny: Number of grid rows required by :meth:`append`.
        format: CALMET ``IFORMC`` value, either ``1`` or ``2``.
        stream: Open formatted or Fortran sequential output stream, or
            ``None`` after :meth:`close`.
    """

    def __init__(self,path: str | Path,nx: int,ny: int,format: int=2) -> None:
        """Open a CLOUD.DAT destination.

        Args:
            path: Output path; missing parent directories are created.
            nx: Number of grid columns.
            ny: Number of grid rows.
            format: ``1`` for sequential-unformatted binary or ``2`` for
                formatted ASCII text.

        Raises:
            OSError: If the destination cannot be created.
            ValueError: If ``format`` is not 1 or 2.
        """
        self.path: Path=Path(path); self.nx: int=int(nx); self.ny: int=int(ny); self.format: int=int(format)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream: FortranSequentialWriter | TextIO | None
        if self.format==1: self.stream=FortranSequentialWriter(self.path)
        elif self.format==2: self.stream=self.path.open('w',encoding='ascii')
        else: raise ValueError('CLOUD.DAT format/IFORMC must be 1 or 2')

    @staticmethod
    def _code(value: datetime) -> tuple[int,int]:
        """Encode a datetime for a CLOUD.DAT interval header.

        Args:
            value: Datetime to encode.

        Returns:
            Tuple ``(ordinal_code, seconds)`` where ``seconds`` is the offset
            after the encoded hour.

        Raises:
            ValueError: If ``value`` cannot be represented as a CALMET ordinal
                code.
        """
        return format_ordinal_code(value),value.minute*60+value.second

    def append(self,begin: datetime,end: datetime,field: ArrayLike) -> None:
        """Append one cloud-fraction grid and flush formatted output.

        Args:
            begin: Beginning datetime of the output interval.
            end: Ending datetime of the output interval.
            field: Array-like cloud fraction with shape ``(ny, nx)`` and
                finite values in the closed interval 0--1.

        Raises:
            ValueError: If ``field`` has the wrong shape, contains invalid
                values, or either timestamp cannot be encoded.
            OSError: If writing to the destination fails.
        """
        values: Float32Array=np.asarray(field,dtype='<f4')
        if values.shape!=(self.ny,self.nx):
            raise ValueError('CLOUD.DAT field does not match configured grid')
        if not np.isfinite(values).all() or np.any((values<0.)|(values>1.)):
            raise ValueError('CLOUD.DAT cloud fraction outside 0..1')
        begin_code,begin_seconds=self._code(begin); end_code,end_seconds=self._code(end)
        if self.format==1:
            cast(FortranSequentialWriter,self.stream).write_record(fixed_string('CLOUDFRA',8),
                struct.pack('<4i',begin_code,begin_seconds,end_code,end_seconds),
                values.ravel(order='C'))
        else:
            numbers=' '.join(format(float(value),'.9g') for value in values.ravel(order='C'))
            stream=cast(TextIO,self.stream)
            stream.write(f'"CLOUDFRA" {begin_code} {begin_seconds} '
                         f'{end_code} {end_seconds} {numbers}\n')
            stream.flush()

    def close(self) -> None:
        """Close the output stream; repeated calls are harmless."""
        if self.stream is not None: self.stream.close(); self.stream=None

    def __enter__(self) -> CloudDataWriter:
        """Return this writer for use as a context manager.

        Returns:
            The open :class:`CloudDataWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer when leaving a context-manager block.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()
