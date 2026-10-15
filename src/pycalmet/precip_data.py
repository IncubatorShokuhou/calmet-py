"""Formatted PRECIP.DAT observations and CALMET GRIDE interpolation."""
from __future__ import annotations

from typing import TypeAlias, cast
from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import struct
from collections.abc import Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import GridConfig
from .legacy_io import FortranSequentialReader
from .surface_data import SurfaceStation,_coded_instant,_coordinate,_instant


MODEL_DTYPE: type[np.float32]=np.float32
ACCUM_DTYPE: type[np.float64]=np.float64

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True,slots=True)
class PrecipitationRecord:
    """Station precipitation rates for one observation interval.

    Attributes:
        begin: Inclusive interval beginning.
        end: Exclusive interval ending.
        rate_mm_h: Float32 array of shape ``(station,)`` containing
            precipitation rates in millimetres per hour. Values near 9999
            represent missing observations.
    """
    begin: datetime
    end: datetime
    rate_mm_h: Float32Array


class PrecipitationData:
    """Read formatted, unpacked PRECIP.DAT records.

    Attributes:
        path: Source PRECIP.DAT path.
        version: File-format version string.
        datum: Datum identifier for station coordinates.
        timezone: Time-zone field from the file header.
        start: Declared beginning of the observation period.
        end: Declared end of the observation period.
        stations: Tuple of :class:`~pycalmet.surface_data.SurfaceStation`
            metadata in record order.
        records: Chronological tuple of :class:`PrecipitationRecord` objects.
    """

    version: str
    datum: str
    start: datetime
    end: datetime

    def __init__(
        self,
        path: str | Path,
        *,
        station_metadata: Sequence[SurfaceStation] | None=None,
        datum: str | None=None,
    ) -> None:
        """Parse formatted PRECIP.DAT text and validate its timeline.

        Args:
            path: Path to formatted PRECIP.DAT.
            station_metadata: Station metadata supplied by the control file
                when the file declares projection ``NONE``. Must be ``None``
                when station coordinates are embedded in the file.
            datum: Control-file datum used with projection ``NONE``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            UnicodeDecodeError: If the file is not ASCII.
            ValueError: If header metadata, station definitions, timestamps,
                record widths, or interval continuity are invalid.
        """
        self.path: Path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty PRECIP.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='PRECIP.DAT': raise ValueError('invalid PRECIP.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version=='2.1'
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid PRECIP.DAT comment header') from error
        projection=lines[index][:8].strip().upper(); index+=1
        if projection not in {'LL','NONE'}: raise ValueError('PRECIP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('precipitation station locations are present in both control file and PRECIP.DAT')
            datum_line=lines[index]; index+=1; self.datum=datum_line[:8].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('PRECIP.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('PRECIP.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum)
        self.timezone: str | int
        if explicit:
            self.timezone=lines[index][:8].strip(); index+=1
        span=[int(item) for item in lines[index].replace(',',' ').split()]; index+=1
        expected=9 if explicit else 8
        if len(span)!=expected: raise ValueError(f'PRECIP.DAT time-span header must contain {expected} integers')
        if explicit:
            self.start=_instant(span[0],span[1],span[2],span[3])
            self.end=_instant(span[4],span[5],span[6],span[7]); count=span[8]
        else:
            first_end=_instant(span[0],span[1],span[2],0)
            self.start=first_end-timedelta(hours=1)
            self.end=_instant(span[3],span[4],span[5],0); self.timezone=span[6]; count=span[7]
        if count<1: raise ValueError('PRECIP.DAT must contain at least one station')
        stations: list[SurfaceStation]=[]
        if projection=='LL':
            for _ in range(count):
                fields=lines[index].replace(',',' ').split(); index+=1
                if len(fields)<4: raise ValueError('invalid PRECIP.DAT station metadata')
                stations.append(SurfaceStation(int(fields[0]),fields[1],_coordinate(fields[2],True),
                                               _coordinate(fields[3],False),0.))
        else:
            stations=list(cast(Sequence[SurfaceStation],station_metadata))
            if len(stations)!=count: raise ValueError('PRECIP.DAT station count does not match control-file metadata')
            identifiers: list[int]=[]
            while len(identifiers)<count and index<len(lines):
                identifiers.extend(int(item) for item in lines[index].replace(',',' ').split()); index+=1
            if len(identifiers)!=count or identifiers!=[item.identifier for item in stations]:
                raise ValueError('PRECIP.DAT station IDs do not match control-file metadata')
        self.stations: tuple[SurfaceStation,...]=tuple(stations)
        records: list[PrecipitationRecord]=[]
        while index<len(lines):
            if not lines[index].strip(): index+=1; continue
            fields=lines[index].replace(',',' ').split(); index+=1
            time_width=8 if explicit else 3
            if len(fields)!=time_width+count: raise ValueError('PRECIP.DAT record has wrong field count')
            times=[int(item) for item in fields[:time_width]]
            if explicit:
                begin=_instant(times[0],times[1],times[2],times[3])
                end=_instant(times[4],times[5],times[6],times[7])
            else:
                end=_instant(times[0],times[1],times[2],0); begin=end-timedelta(hours=1)
            if not begin<end or (end-begin).total_seconds()>3600: raise ValueError('PRECIP.DAT record interval must be positive and at most one hour')
            records.append(PrecipitationRecord(
                begin,end,np.asarray(fields[time_width:],dtype=MODEL_DTYPE)))
        if not records: raise ValueError('PRECIP.DAT contains no records')
        if records[0].begin<self.start or records[-1].end>self.end: raise ValueError('PRECIP.DAT records lie outside declared span')
        if any(left.end!=right.begin for left,right in zip(records[:-1],records[1:])): raise ValueError('PRECIP.DAT records are not contiguous')
        self.records: tuple[PrecipitationRecord,...]=tuple(records)

    def at_begin(self,timestamp: datetime) -> PrecipitationRecord:
        """Return the record whose half-open interval contains a timestamp.

        Args:
            timestamp: Datetime satisfying ``record.begin <= timestamp <
                record.end``.

        Returns:
            The covering :class:`PrecipitationRecord`.

        Raises:
            ValueError: If no record covers ``timestamp``.
        """
        for record in self.records:
            if record.begin<=timestamp<record.end: return record
        raise ValueError(f'PRECIP.DAT has no record covering {timestamp!s}')


def unpack_precipitation(words: ArrayLike,count: int) -> Float32Array:
    """Expand CALMET PACK zero runs.

    Args:
        words: One-dimensional encoded values. A nonnegative word is a data
            value; a negative word encodes that many consecutive zeros.
        count: Required number of decoded station values.

    Returns:
        Float32 array of shape ``(count,)`` containing precipitation rates in
        millimetres per hour.

    Raises:
        ValueError: If the encoded sequence does not expand to ``count``
            values.
    """
    output: list[float]=[]
    for word in np.asarray(words,dtype=MODEL_DTYPE):
        if word>=0.: output.append(float(word))
        else: output.extend([0.]*int(abs(word)+.1))
    if len(output)!=count: raise ValueError('packed PRECIP.DAT record expands to wrong station count')
    return np.asarray(output,dtype=MODEL_DTYPE)


def read_unformatted_precipitation(
    path: str | Path,
    *,
    station_metadata: Sequence[SurfaceStation] | None=None,
    datum: str | None=None,
) -> PrecipitationData:
    """Read a GNU-Fortran sequential-unformatted PRECIP.DAT file.

    Args:
        path: Path to a legacy or version 2.1 binary file.
        station_metadata: Control-file station metadata required when the file
            declares projection ``NONE``.
        datum: Control-file datum required with projection ``NONE``.

    Returns:
        A fully populated :class:`PrecipitationData` instance. Each record's
        ``rate_mm_h`` is a float32 ``(station,)`` array.

    Raises:
        OSError: If the input file cannot be opened.
        ValueError: If record framing, metadata, packing, timestamps, station
            identifiers, or interval continuity are invalid.
    """
    source=Path(path); result=object.__new__(PrecipitationData); result.path=source
    with FortranSequentialReader(source) as reader:
        def record(size: int | None=None,label: str='record') -> bytes:
            """Read and optionally size-check one required binary record.

            Args:
                size: Expected payload length in bytes, or ``None`` to accept
                    any length.
                label: Record description used in validation errors.

            Returns:
                Raw record payload as bytes.

            Raises:
                ValueError: If the file ends early or the payload length is
                    not ``size``.
            """
            payload=reader.read_record()
            if payload is None: raise ValueError(f'truncated unformatted PRECIP.DAT {label}')
            if size is not None and len(payload)!=size: raise ValueError(f'unformatted PRECIP.DAT {label} has wrong size')
            return payload
        head=record(96,'identity')
        if head[:16].decode('ascii').strip().upper()!='PRECIP.DAT': raise ValueError('invalid PRECIP.DAT dataset label')
        result.version=head[16:32].decode('ascii').strip()
        explicit=result.version=='2.1'
        comments=struct.unpack('<i',record(4,'comment count'))[0]
        if comments<0: raise ValueError('negative PRECIP.DAT comment count')
        for _ in range(comments): record(80,'comment')
        projection=record(8,'projection').decode('ascii').strip().upper()
        if projection not in {'LL','NONE'}: raise ValueError('PRECIP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('precipitation station locations are present in both control file and PRECIP.DAT')
            datum_record=record(20,'datum'); result.datum=datum_record[:8].decode('ascii').strip()
            compatibility=record(20,'legacy compatibility header')
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',compatibility)
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        else:
            if station_metadata is None or datum is None:
                raise ValueError('PRECIP.DAT NONE requires control-file station metadata and datum')
            result.datum=str(datum)
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',record(20,'time span'))
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        if count<1 or packing not in {0,1}: raise ValueError('invalid PRECIP.DAT station count or packing code')
        identifiers=np.frombuffer(record(4*count,'station IDs'),dtype='<i4')
        if projection=='LL':
            names=record(4*count,'station names'); latitudes=record(16*count,'station latitudes'); longitudes=record(16*count,'station longitudes')
            result.stations=tuple(SurfaceStation(int(identifiers[i]),names[4*i:4*i+4].decode('ascii').strip(),
                _coordinate(latitudes[16*i:16*i+16].decode('ascii'),True),
                _coordinate(longitudes[16*i:16*i+16].decode('ascii'),False),0.) for i in range(count))
        else:
            result.stations=tuple(cast(Sequence[SurfaceStation],station_metadata))
            if len(result.stations)!=count or identifiers.tolist()!=[item.identifier for item in result.stations]:
                raise ValueError('PRECIP.DAT station IDs do not match control-file metadata')
        records: list[PrecipitationRecord]=[]
        while (payload:=reader.read_record()) is not None:
            if explicit and packing:
                if len(payload)!=20: raise ValueError('packed PRECIP.DAT record header has wrong size')
                begin_code,begin_sec,end_code,end_sec,nwords=struct.unpack('<5i',payload)
                if nwords<0: raise ValueError('packed PRECIP.DAT word count cannot be negative')
                rates=(np.zeros(count,dtype=MODEL_DTYPE) if nwords==0 else
                       unpack_precipitation(np.frombuffer(record(4*nwords,'packed values'),dtype='<f4'),count))
            elif explicit:
                if len(payload)!=16+4*count: raise ValueError('PRECIP.DAT record has wrong size')
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[:16])
                rates=np.frombuffer(payload,dtype='<f4',count=count,offset=16).copy()
            elif packing:
                if len(payload)!=8: raise ValueError('packed legacy PRECIP.DAT record header has wrong size')
                end_code,nwords=struct.unpack('<2i',payload)
                if nwords<0: raise ValueError('packed PRECIP.DAT word count cannot be negative')
                rates=(np.zeros(count,dtype=MODEL_DTYPE) if nwords==0 else
                       unpack_precipitation(np.frombuffer(record(4*nwords,'packed values'),dtype='<f4'),count))
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            else:
                if len(payload)!=4+4*count: raise ValueError('legacy PRECIP.DAT record has wrong size')
                end_code=struct.unpack('<i',payload[:4])[0]
                rates=np.frombuffer(payload,dtype='<f4',count=count,offset=4).copy()
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            if explicit:
                begin,end=_coded_instant(begin_code,begin_sec),_coded_instant(end_code,end_sec)
            if not begin<end or (end-begin).total_seconds()>3600: raise ValueError('PRECIP.DAT record interval must be positive and at most one hour')
            records.append(PrecipitationRecord(begin,end,rates))
    if not records: raise ValueError('PRECIP.DAT contains no records')
    if records[0].begin<result.start or records[-1].end>result.end: raise ValueError('PRECIP.DAT records lie outside declared span')
    if any(left.end!=right.begin for left,right in zip(records[:-1],records[1:])): raise ValueError('PRECIP.DAT records are not contiguous')
    result.records=tuple(records); return result


def load_precipitation(
    path: str | Path,
    format: int=2,
    *,
    station_metadata: Sequence[SurfaceStation] | None=None,
    datum: str | None=None,
) -> PrecipitationData:
    """Load PRECIP.DAT using the CALMET ``IFORMP`` selector.

    Args:
        path: Path to PRECIP.DAT.
        format: ``1`` for sequential-unformatted binary or ``2`` for formatted
            text.
        station_metadata: Control-file station metadata for projection
            ``NONE``.
        datum: Control-file datum for projection ``NONE``.

    Returns:
        Parsed :class:`PrecipitationData`.

    Raises:
        ValueError: If ``format`` is not 1 or 2, or the selected reader rejects
            the file.
    """
    if format==1: return read_unformatted_precipitation(path,station_metadata=station_metadata,datum=datum)
    if format==2: return PrecipitationData(path,station_metadata=station_metadata,datum=datum)
    raise ValueError('PRECIP.DAT format/IFORMP must be 1 or 2')


def grid_precipitation(
    values: ArrayLike,
    station_xy: ArrayLike,
    grid: GridConfig,
    method: int=2,
    radius_km: float=50.,
    cutoff: float=.01,
) -> Float32Array:
    """Interpolate station precipitation with CALMET GRIDE methods 1--3.

    Args:
        values: Station precipitation rates in millimetres per hour. Values
            greater than or equal to 9998 are treated as missing.
        station_xy: Float-like array of shape ``(station, 2)`` containing
            station x/y coordinates in kilometres relative to the CALMET grid
            origin.
        grid: Grid configuration providing ``nx``, ``ny``, ``dx_km``, and
            ``dy_km``.
        method: GRIDE method: ``1`` for inverse distance, ``2`` for inverse
            squared distance, or ``3`` for exponentially damped inverse
            squared distance.
        radius_km: Search radius for methods 1 and 2. For method 3, zero asks
            the routine to infer a wet/dry decorrelation scale.
        cutoff: Interpolated rates below this value, in millimetres per hour,
            are set to zero.

    Returns:
        Float32 precipitation-rate array of shape ``(grid.ny, grid.nx)`` in
        millimetres per hour. An all-missing station vector produces zeros.

    Raises:
        ValueError: If station coordinates have the wrong shape or ``method``
            is not 1, 2, or 3.
    """
    f32=MODEL_DTYPE
    rates: Float32Array=np.asarray(values,dtype=f32)
    # Spatial distances and weighted station accumulations are the deliberate
    # FP64 boundary; precipitation fields enter and leave as FP32.
    points: Float64Array=np.asarray(station_xy,dtype=ACCUM_DTYPE)*ACCUM_DTYPE(1000.)
    if points.shape!=(rates.size,2): raise ValueError('precipitation station coordinates have wrong shape')
    valid=rates<9998.
    if not valid.any(): return np.zeros((grid.ny,grid.nx),dtype=f32)
    points=points[valid]; rates=rates[valid]
    if method not in {1,2,3}: raise ValueError('precipitation interpolation method must be 1, 2, or 3')
    sigma=ACCUM_DTYPE(1000.)*ACCUM_DTYPE(radius_km)
    if method==3 and sigma==0.:
        wet=rates>0.; pairs=wet[:,None]!=wet[None,:]
        delta=points[:,None,:]-points[None,:,:]
        pair_distance=delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1]
        if pairs.any(): sigma=ACCUM_DTYPE(.5)*np.sqrt(np.min(pair_distance[pairs]))
        else: sigma=ACCUM_DTYPE(.5)*np.sqrt(ACCUM_DTYPE(1.e37))
    x=(np.arange(grid.nx,dtype=ACCUM_DTYPE)+ACCUM_DTYPE(.5))*ACCUM_DTYPE(
        grid.dx_km)*ACCUM_DTYPE(1000.)
    y=(np.arange(grid.ny,dtype=ACCUM_DTYPE)+ACCUM_DTYPE(.5))*ACCUM_DTYPE(
        grid.dy_km)*ACCUM_DTYPE(1000.)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output: Float32Array=np.empty(targets.shape[0],dtype=f32)
    for start in range(0,targets.shape[0],65536):
        delta=targets[start:start+65536,None,:]-points[None,:,:]
        d2=delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1]
        if method in {1,2}:
            sigma2=sigma*sigma; inside=d2<=sigma2
            denominator=np.zeros(d2.shape[0],dtype=ACCUM_DTYPE)
            numerator=np.zeros_like(denominator)
            for station in range(rates.size):
                distance=(np.sqrt(d2[:,station])
                          if method==1 else d2[:,station])
                safe=np.maximum(distance,ACCUM_DTYPE(.01))
                weight=np.where(inside[:,station],ACCUM_DTYPE(1.)/safe,
                                ACCUM_DTYPE(0.))
                denominator+=weight
                numerator+=ACCUM_DTYPE(rates[station])*weight
            weighted=np.divide(numerator,denominator,out=np.zeros_like(denominator),where=denominator!=0.)
            nearest=rates[np.argmin(d2,axis=1)]
            result=np.where(denominator==f32(0.),nearest,
                            np.where(weighted<ACCUM_DTYPE(cutoff),0.,weighted))
        else:
            sigma2=sigma*sigma
            denominator=np.zeros(d2.shape[0],dtype=ACCUM_DTYPE)
            numerator=np.zeros_like(denominator)
            for station in range(rates.size):
                exponent=d2[:,station]/sigma2
                decay=np.where(exponent>60.,0.,np.exp(-exponent))
                safe=np.maximum(d2[:,station],ACCUM_DTYPE(.01))
                weight=decay/safe
                denominator+=weight
                numerator+=ACCUM_DTYPE(rates[station])*weight
            result=np.where(numerator<1.e-30,0.,numerator/denominator)
            result=np.where(result<ACCUM_DTYPE(cutoff),0.,result)
        output[start:start+d2.shape[0]]=np.asarray(result,dtype=MODEL_DTYPE)
    return output.reshape(grid.ny,grid.nx)
