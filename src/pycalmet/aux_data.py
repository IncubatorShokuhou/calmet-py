"""Reader for CALMET.AUX 1.1 cloud-liquid-water files."""
from __future__ import annotations

from typing import ClassVar, TypeAlias, cast

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import struct
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from .legacy_io import FortranSequentialReader
from .datetime_utils import parse_calendar,parse_ordinal_code


Float32Array: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
Int64Array: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True,slots=True)
class AuxiliaryHeader:
    """Static metadata decoded from the CALMET.AUX header.

    Attributes:
        comments: Producer comments stored in the file header.
        start: Declared beginning of the model run.
        end: Declared end of the model run.
        wire_start: Original ``(year, month, day, hour, seconds)`` start fields.
        wire_end: Original ``(year, month, day, hour, seconds)`` end fields.
        timezone: CALMET time-zone label.
        run_hours: Declared run duration in hours.
        nx: Number of grid columns.
        ny: Number of grid rows.
        nz: Number of vertical layers.
        spacing_m: Horizontal grid spacing in metres.
        x_origin_m: Grid x origin in metres.
        y_origin_m: Grid y origin in metres.
        projection: CALPUFF projection identifier.
        datum: CALPUFF horizontal datum identifier.
        datum_date: Datum database date written by the producer.
        false_easting_km: Projection false easting in kilometres.
        false_northing_km: Projection false northing in kilometres.
        hemisphere: UTM hemisphere code.
        utm_zone: UTM zone number.
        latitude_origin: Projection latitude of origin in degrees north.
        longitude_origin: Projection longitude of origin in degrees east.
        standard_parallel_1: First standard parallel in degrees north.
        standard_parallel_2: Second standard parallel in degrees north.
        z_faces_m: Float32 array of shape ``(nz + 1,)`` containing layer-face
            heights in metres above ground level.
    """
    comments: tuple[str,...]
    start: datetime
    end: datetime
    wire_start: tuple[int,int,int,int,int]
    wire_end: tuple[int,int,int,int,int]
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
    z_faces_m: Float32Array


@dataclass(frozen=True,slots=True)
class AuxiliaryCloudHour:
    """Cloud-water fields for one CALMET.AUX output interval.

    Attributes:
        begin: Inclusive interval beginning.
        end: Interval ending.
        wire_time_bounds: Original ordinal-date codes and seconds as
            ``(begin_code, begin_seconds, end_code, end_seconds)``.
        cloud_water_above_g_kg: Float32 ``(ny, nx)`` cloud-water mixing ratio
            above the model top, in grams per kilogram.
        cloud_bottom_above_m: Float32 ``(ny, nx)`` cloud-bottom height above
            the model top, in metres.
        cloud_top_above_m: Float32 ``(ny, nx)`` cloud-top height above the
            model top, in metres.
        cloud_water_3d_g_kg: Float32 ``(nz, ny, nx)`` in-model cloud-water
            mixing ratio, in grams per kilogram.
    """
    begin: datetime
    end: datetime
    wire_time_bounds: tuple[int,int,int,int]
    cloud_water_above_g_kg: Float32Array
    cloud_bottom_above_m: Float32Array
    cloud_top_above_m: Float32Array
    cloud_water_3d_g_kg: Float32Array


class AuxiliaryData:
    """Stream and validate a GNU sequential-unformatted CALMET.AUX file.

    Attributes:
        path: Path to the input file.
        reader: Sequential-unformatted record reader positioned after the
            most recently consumed record.
        header: Validated static file metadata.
    """
    _fields_2d: ClassVar[tuple[str, str, str]] = (
        'CLDMRUP', 'ZCLDBOT', 'ZCLDTOP'
    )

    def __init__(self,path: str | Path) -> None:
        """Open a CALMET.AUX file and read its static header.

        Args:
            path: Path to the sequential-unformatted CALMET.AUX file.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If the declaration, grid metadata, variable table, or
                vertical coordinates are invalid or unsupported.
        """
        self.path: Path=Path(path)
        self.reader: FortranSequentialReader=FortranSequentialReader(self.path)
        self.header: AuxiliaryHeader=self._read_header()

    @staticmethod
    def _text(value: bytes) -> str:
        """Decode one fixed-width ASCII field and remove trailing spaces.

        Args:
            value: Bytes containing an ASCII field.

        Returns:
            Decoded text with trailing whitespace removed.

        Raises:
            UnicodeDecodeError: If ``value`` is not valid ASCII.
        """
        return value.decode('ascii').rstrip()

    def _record(self,description: str) -> bytes:
        """Read one required Fortran record.

        Args:
            description: Human-readable record name used in error messages.

        Returns:
            Raw record payload as bytes.

        Raises:
            ValueError: If the file ends before the record is available.
        """
        record=self.reader.read_record()
        if record is None: raise ValueError(f'truncated CALMET.AUX {description}')
        return record

    def _read_header(self) -> AuxiliaryHeader:
        """Decode and validate the declaration and static metadata records.

        Returns:
            An :class:`AuxiliaryHeader` describing the grid and run period.

        Raises:
            ValueError: If a required record is missing, malformed, or uses
                unsupported CALMET.AUX dimensions or variables.
        """
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
        try:
            start=parse_calendar(*dates[:5]); end=parse_calendar(*dates[5:])
        except ValueError as error:
            raise ValueError('invalid CALMET.AUX run-control timestamp') from error
        if end<start:
            raise ValueError('CALMET.AUX run-control end precedes start')
        timezone=self._text(run[40:48])
        run_hours,nx,ny,nz=struct.unpack('<4i',run[48:64])
        spacing,x_origin,y_origin=struct.unpack('<3f',run[64:76])
        projection=self._text(run[76:84]); datum=self._text(run[84:92])
        datum_date=self._text(run[92:104])
        false_easting,false_northing=struct.unpack('<2f',run[104:112])
        hemisphere=self._text(run[112:116]); utm_zone=struct.unpack('<i',run[116:120])[0]
        angles=cast(tuple[float,float,float,float],struct.unpack('<4f',run[120:136]))
        naux2d,naux3d=struct.unpack('<2i',run[136:144])
        if (nx<=0 or ny<=0 or nz<=0 or naux2d!=3 or naux3d!=1):
            raise ValueError('unsupported CALMET.AUX grid or variable counts')
        zface=self._record('ZFACE')
        if len(zface)!=24+4*(nz+1) or self._text(zface[:8])!='ZFACE':
            raise ValueError('invalid CALMET.AUX ZFACE record')
        faces: Float32Array=np.frombuffer(zface[24:],dtype='<f4').copy()
        if faces[0]!=0. or np.any(np.diff(faces)<=0.):
            raise ValueError('invalid CALMET.AUX vertical faces')
        expected_2d=(b'2D_VARS '+b''.join(name.encode().ljust(8,b' ') for name in self._fields_2d)+
            b'G/KG    MAGL    MAGL    RC4 RC4 RC4 ')
        expected_3d=b'3D_VARS CLDMR3D G/KG    RC4 '
        if self._record('2-D variables')!=expected_2d:
            raise ValueError('unsupported CALMET.AUX 2-D variable table')
        if self._record('3-D variables')!=expected_3d:
            raise ValueError('unsupported CALMET.AUX 3-D variable table')
        return AuxiliaryHeader(comments,start,end,
            cast(tuple[int,int,int,int,int],dates[:5]),
            cast(tuple[int,int,int,int,int],dates[5:]),timezone,
            run_hours,nx,ny,nz,
            spacing,x_origin,y_origin,projection,datum,datum_date,false_easting,
            false_northing,hemisphere,utm_zone,*angles,faces)

    @staticmethod
    def _decode_timestamp(record: bytes) -> tuple[datetime,datetime,tuple[int,int,int,int]]:
        """Decode one CALMET.AUX interval timestamp record.

        Args:
            record: Sixteen-byte record containing four little-endian int32
                values: begin code, begin seconds, end code, and end seconds.

        Returns:
            Tuple ``(begin, end, wire_bounds)`` where the first two elements
            are datetimes and ``wire_bounds`` preserves the four encoded
            integer values.

        Raises:
            ValueError: If the record size, encoded dates, or interval order
                is invalid.
        """
        if len(record)!=16: raise ValueError('invalid CALMET.AUX timestamp record')
        begin_code,begin_seconds,end_code,end_seconds=struct.unpack('<4i',record)
        try:
            begin=parse_ordinal_code(begin_code,begin_seconds)
            end=parse_ordinal_code(end_code,end_seconds)
        except ValueError as error:
            raise ValueError('invalid CALMET.AUX timestamp') from error
        if begin>=end:
            raise ValueError('CALMET.AUX interval does not advance')
        return begin,end,(begin_code,begin_seconds,end_code,end_seconds)

    def _timestamp(self) -> tuple[datetime,datetime]:
        """Read the next timestamp record.

        Returns:
            Tuple ``(begin, end)`` of decoded datetimes.

        Raises:
            ValueError: If the timestamp record is missing or invalid.
        """
        return self._decode_timestamp(self._record('timestamp'))[:2]

    def _compressed(self,label: str) -> Float32Array:
        """Decode one CALMET zero-run-compressed two-dimensional field.

        Args:
            label: Expected twelve-character field label.

        Returns:
            Float32 array of shape ``(ny, nx)``. Negative encoded values
            expand to zero runs; nonnegative values are retained.

        Raises:
            ValueError: If record lengths, labels, or the expanded grid size
                do not match the header.
        """
        count_record=self._record(f'{label} compressed count')
        if len(count_record)!=4: raise ValueError('invalid CALMET.AUX compressed count')
        count=struct.unpack('<i',count_record)[0]
        record=self._record(label)
        if len(record)!=12+4*count or self._text(record[:12])!=label:
            raise ValueError(f'invalid CALMET.AUX {label} record')
        encoded: Float32Array=np.frombuffer(record[12:],dtype='<f4')
        negative: BoolArray=encoded<0.
        lengths: Int64Array=np.where(negative,np.trunc(-encoded).astype(np.int64),1)
        ends: Int64Array=np.cumsum(lengths)
        size=self.header.nx*self.header.ny
        if count<1 or ends[-1]!=size:
            raise ValueError(f'CALMET.AUX {label} compression length mismatch')
        values: Float32Array=np.zeros(size,dtype=np.float32)
        values[ends[~negative]-1]=encoded[~negative]
        return values.reshape(self.header.ny,self.header.nx)

    def read_hour(self) -> AuxiliaryCloudHour | None:
        """Read the next complete cloud-water interval.

        Returns:
            An :class:`AuxiliaryCloudHour` containing three float32
            ``(ny, nx)`` fields and one float32 ``(nz, ny, nx)`` field, or
            ``None`` at end of file.

        Raises:
            ValueError: If a field is malformed or its timestamp differs from
                the other fields in the interval.
        """
        first=self.reader.read_record()
        if first is None: return None
        begin,end,wire_time_bounds=self._decode_timestamp(first)
        stamp=begin,end
        fields: list[Float32Array]=[]
        for index,label in enumerate(self._fields_2d):
            if index and self._timestamp()!=stamp:
                raise ValueError('CALMET.AUX 2-D timestamps are inconsistent')
            fields.append(self._compressed(label))
        if self._timestamp()!=stamp:
            raise ValueError('CALMET.AUX 3-D timestamp is inconsistent')
        layers: Float32Array=np.stack([self._compressed(f'CLDMR3D {level:04d}')
                                              for level in range(1,self.header.nz+1)])
        typed_fields=cast(
            tuple[Float32Array,Float32Array,Float32Array],fields)
        return AuxiliaryCloudHour(begin,end,wire_time_bounds,*typed_fields,layers)

    def close(self) -> None:
        """Close the underlying sequential-record stream."""
        self.reader.close()

    def __enter__(self) -> AuxiliaryData:
        """Return this reader for use as a context manager.

        Returns:
            The open :class:`AuxiliaryData` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the reader when leaving a context-manager block.

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
