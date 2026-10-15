"""Formatted CALMET SURF.DAT 2.1 reader and interval lookup."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timedelta
from pathlib import Path
import re
import shlex
import struct
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypedDict, cast, overload
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.spatial import cKDTree

from .config import GridConfig
from .projections import project_geographic
from .legacy_io import FortranSequentialReader
from .datetime_utils import parse_ordinal,parse_ordinal_code


COMPUTE_DTYPE: type[np.float32]=np.float32
SPATIAL_DTYPE: type[np.float64]=np.float64
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]


class _SurfaceFields(TypedDict):
    """Typed field mapping used to construct a decoded surface record."""

    wind_speed: Float32Array
    wind_direction: Float32Array
    ceiling_hundreds_ft: Int32Array
    cloud_tenths: Int32Array
    temperature_k: Float32Array
    relative_humidity: Int32Array
    pressure_mb: Float32Array
    precipitation_code: Int32Array



@dataclass(frozen=True,slots=True)
class SurfaceStation:
    """Surface observation-station metadata.

    Attributes:
        identifier: Integer station identifier used by SURF.DAT.
        name: Station name, commonly limited to four characters by CALMET.
        latitude: Geographic latitude in decimal degrees north.
        longitude: Geographic longitude in decimal degrees east.
        anemometer_height_m: Wind-sensor height in metres above local ground.
    """
    identifier: int
    name: str
    latitude: float
    longitude: float
    anemometer_height_m: float


@dataclass(frozen=True,slots=True)
class SurfaceRecord:
    """Meteorological observations for one SURF.DAT interval.

    Attributes:
        begin: Inclusive interval beginning.
        end: Exclusive interval ending.
        wind_speed: Float32 ``(station,)`` wind speed in metres per second.
        wind_direction: Float32 ``(station,)`` meteorological direction in
            degrees from north.
        ceiling_hundreds_ft: Int32 ``(station,)`` ceiling height in hundreds
            of feet.
        cloud_tenths: Int32 ``(station,)`` CALMET ``ICC`` opaque sky cover in
            tenths. Observation converters may fall back to total sky cover
            when opaque cover is unavailable.
        temperature_k: Float32 ``(station,)`` air temperature in kelvin.
        relative_humidity: Int32 ``(station,)`` relative humidity in percent.
        pressure_mb: Float32 ``(station,)`` station pressure in millibars.
        precipitation_code: Int32 ``(station,)`` CALMET precipitation code.

    Notes:
        Individual missing observations use the SURF.DAT sentinel 9999.
    """
    begin: datetime
    end: datetime
    wind_speed: Float32Array
    wind_direction: Float32Array
    ceiling_hundreds_ft: Int32Array
    cloud_tenths: Int32Array
    temperature_k: Float32Array
    relative_humidity: Int32Array | Float32Array
    pressure_mb: Float32Array
    precipitation_code: Int32Array


def unpack_surface(
    words: ArrayLike,
) -> _SurfaceFields:
    """Decode CALMET UNPCKS decimal fields for multiple stations.

    Args:
        words: Nonnegative integer-like array of shape ``(station, 3)``.

    Returns:
        Dictionary of eight arrays accepted by :class:`SurfaceRecord`.
        Continuous fields are float32 and coded fields are int32, each with
        shape ``(station,)`` and the units documented by ``SurfaceRecord``.
        Encoded missing values are normalized to 9999.

    Raises:
        ValueError: If ``words`` is not a nonnegative ``(station, 3)`` array.
    """
    packed=np.asarray(words,dtype=np.int64)
    if packed.ndim!=2 or packed.shape[1]!=3 or np.any(packed<0):
        raise ValueError('packed surface data must have shape (station,3) and be non-negative')
    word1,word2,word3=packed.T
    temperature=word1//100000
    precipitation=(word1//1000)-temperature*100
    humidity=word1-temperature*100000-precipitation*1000
    pressure=word2//100000
    cloud=(word2//1000)-pressure*100
    direction=word2-pressure*100000-cloud*1000
    ceiling=word3//10000
    speed=word3-ceiling*10000
    speed_values=speed.astype(COMPUTE_DTYPE)
    direction_values=direction.astype(COMPUTE_DTYPE)
    temperature_values=temperature.astype(COMPUTE_DTYPE)
    pressure_values=pressure.astype(COMPUTE_DTYPE)
    return dict(
        wind_speed=cast(Float32Array,np.where(speed==9999,9999.,speed_values/100.)),
        wind_direction=cast(Float32Array,np.where(direction==999,9999.,direction_values)),
        ceiling_hundreds_ft=ceiling.astype(np.int32),
        cloud_tenths=np.where(cloud==99,9999,cloud).astype(np.int32),
        temperature_k=cast(Float32Array,np.where(temperature==9999,9999.,temperature_values/10.)),
        relative_humidity=np.where(humidity==999,9999,humidity).astype(np.int32),
        pressure_mb=cast(Float32Array,np.where(pressure==9999,9999.,pressure_values/10.)),
        precipitation_code=np.where(precipitation==99,9999,precipitation).astype(np.int32),
    )


def _instant(
    year: int | str,
    day: int | str,
    hour: int | str,
    seconds: int | str,
) -> datetime:
    """Build a SURF.DAT datetime from ordinal-date components.

    Args:
        year: Four- or two-digit year accepted by the CALMET parser.
        day: One-based day of year.
        hour: Hour of day.
        seconds: Seconds after the hour.

    Returns:
        Decoded naive :class:`datetime`.

    Raises:
        ValueError: If the ordinal timestamp is invalid.
    """
    try: return parse_ordinal(year,day,hour,seconds)
    except ValueError as error: raise ValueError('invalid SURF.DAT timestamp') from error


def _coded_instant(code: int | str,seconds: int | str=0) -> datetime:
    """Decode a compact SURF.DAT ordinal timestamp.

    Args:
        code: Integer ``YYYYJJJHH`` ordinal-date code.
        seconds: Seconds after the encoded hour.

    Returns:
        Decoded naive :class:`datetime`.

    Raises:
        ValueError: If the encoded timestamp is invalid.
    """
    try: return parse_ordinal_code(code,seconds)
    except ValueError as error: raise ValueError(f'invalid SURF.DAT coded time {code!r}') from error


def _coordinate(text: str,latitude: bool) -> float:
    """Parse a signed or hemisphere-qualified geographic coordinate.

    Args:
        text: Coordinate such as ``30.5``, ``30.5N``, or ``W120``.
        latitude: ``True`` to enforce the latitude range; ``False`` for
            longitude.

    Returns:
        Coordinate in signed decimal degrees, positive north or east.

    Raises:
        ValueError: If the syntax, hemisphere combination, or numeric range is
            invalid.
    """
    value=re.sub(r'\s+','',text.upper())
    match=re.fullmatch(r'([NSEW]?)([+-]?\d+(?:\.\d*)?)([NSEW]?)',value)
    if not match: raise ValueError(f'unsupported SURF.DAT coordinate {text!r}')
    if match.group(1) and match.group(3): raise ValueError(f'multiple SURF.DAT hemispheres: {text!r}')
    number=float(match.group(2)); hemisphere=match.group(1) or match.group(3)
    if hemisphere in {'S','W'}: number=-abs(number)
    elif hemisphere in {'N','E'}: number=abs(number)
    limit=90. if latitude else 180.
    if not -limit<=number<=limit: raise ValueError(f'SURF.DAT coordinate outside valid range: {text!r}')
    return number


class SurfaceData:
    """Read a formatted, unpacked SURF.DAT dataset.

    Attributes:
        path: Source SURF.DAT path.
        version: File-format version string.
        datum: Datum identifier for station coordinates.
        datum_date: Datum database date, or an empty string for projection
            ``NONE``.
        timezone: File time-zone label or legacy numeric field.
        start: Declared beginning of the observation period.
        end: Declared end of the observation period.
        stations: Tuple of :class:`SurfaceStation` metadata in record order.
        records: Chronological list of :class:`SurfaceRecord` objects.
    """

    version: str
    datum: str
    datum_date: str
    start: datetime
    end: datetime

    def __init__(
        self,
        path: str | Path,
        *,
        station_metadata: Sequence[SurfaceStation] | None=None,
        datum: str | None=None,
    ) -> None:
        """Parse formatted SURF.DAT text and validate its timeline.

        Args:
            path: Path to formatted SURF.DAT.
            station_metadata: Control-file stations required when the file
                declares projection ``NONE``. Must be ``None`` when station
                coordinates are embedded in the file.
            datum: Control-file datum required with projection ``NONE``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            UnicodeDecodeError: If the file is not ASCII.
            ValueError: If header metadata, station definitions, timestamps,
                meteorological fields, or interval continuity are invalid.
        """
        self.path: Path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty SURF.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='SURF.DAT': raise ValueError('invalid SURF.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version=='2.1'
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid SURF.DAT comment header') from error
        try: projection=lines[index][:8].strip().upper(); index+=1
        except IndexError as error: raise ValueError('incomplete SURF.DAT header') from error
        if projection not in {'LL','NONE'}: raise ValueError('SURF.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('surface station locations are present in both control file and SURF.DAT')
            datum_line=lines[index]; index+=1
            self.datum=datum_line[:8].strip(); self.datum_date=datum_line[8:20].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('SURF.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('SURF.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum); self.datum_date=''
        self.timezone: str | int
        if explicit:
            self.timezone=lines[index][:8].strip(); index+=1
        header=shlex.split(lines[index]); index+=1
        values=[int(item) for item in header]
        if explicit:
            if len(values)!=9: raise ValueError('SURF.DAT 2.1 time-span header must contain 9 integers')
            self.start=_instant(*values[:4]); self.end=_instant(*values[4:8]); count=values[8]
        else:
            if len(values)!=8: raise ValueError('legacy SURF.DAT time-span header must contain 8 integers')
            first_end=_instant(values[0],values[1],values[2],0)
            self.start=first_end-timedelta(hours=1)
            self.end=_instant(values[3],values[4],values[5],0)
            self.timezone=values[6]; count=values[7]
        if count<1: raise ValueError('SURF.DAT must contain at least one station')
        stations: list[SurfaceStation]=[]
        if projection=='LL':
            for _ in range(count):
                try: fields=shlex.split(lines[index]); index+=1
                except IndexError as error: raise ValueError('incomplete SURF.DAT station list') from error
                if len(fields)!=5: raise ValueError('SURF.DAT station row must contain ID, name, latitude, longitude, and height')
                stations.append(SurfaceStation(int(fields[0]),fields[1],_coordinate(fields[2],True),
                                               _coordinate(fields[3],False),float(fields[4])))
        else:
            stations=list(cast(Sequence[SurfaceStation],station_metadata))
            if len(stations)!=count: raise ValueError('SURF.DAT station count does not match control-file metadata')
            identifiers: list[int]=[]
            while len(identifiers)<count and index<len(lines):
                identifiers.extend(int(item) for item in shlex.split(lines[index])); index+=1
            if len(identifiers)!=count: raise ValueError('SURF.DAT NONE station ID list has wrong length')
            if identifiers!=[item.identifier for item in stations]:
                raise ValueError('SURF.DAT station IDs do not match control-file metadata')
        if len({station.identifier for station in stations})!=count: raise ValueError('duplicate SURF.DAT station ID')
        self.stations: tuple[SurfaceStation,...]=tuple(stations)
        tokens=shlex.split('\n'.join(lines[index:]),comments=False)
        width=(8 if explicit else 3)+8*count
        if len(tokens)%width: raise ValueError('SURF.DAT contains an incomplete meteorological record')
        self.records: list[SurfaceRecord]=[]
        for offset in range(0,len(tokens),width):
            row=tokens[offset:offset+width]
            if explicit:
                begin=_instant(*row[:4]); end=_instant(*row[4:8]); value_offset=8
                if not begin<end or end-begin>timedelta(hours=1):
                    raise ValueError('SURF.DAT record interval must advance by at most one hour')
            else:
                end=_instant(row[0],row[1],row[2],0); begin=end-timedelta(hours=1); value_offset=3
            raw=np.asarray(row[value_offset:],dtype=COMPUTE_DTYPE).reshape(count,8)
            self.records.append(SurfaceRecord(begin,end,raw[:,0].copy(),raw[:,1].copy(),
                raw[:,2].astype(np.int32),raw[:,3].astype(np.int32),raw[:,4].copy(),
                raw[:,5].astype(np.int32),raw[:,6].copy(),raw[:,7].astype(np.int32)))
        if not self.records: raise ValueError('SURF.DAT contains no meteorological records')
        for left,right in zip(self.records[:-1],self.records[1:]):
            if left.end!=right.begin: raise ValueError('SURF.DAT records contain a gap or overlap')
        if self.records[0].begin<self.start or self.records[-1].end>self.end:
            raise ValueError('SURF.DAT records fall outside the declared header period')

    def at_begin(self,when: datetime) -> SurfaceRecord:
        """Return the record whose half-open interval contains a timestamp.

        Args:
            when: Datetime satisfying ``record.begin <= when < record.end``.

        Returns:
            Covering :class:`SurfaceRecord`.

        Raises:
            ValueError: If no record covers ``when``.
        """
        for record in self.records:
            if record.begin<=when<record.end: return record
        raise ValueError(f'SURF.DAT has no record covering {when}')


def read_unformatted_surface(
    path: str | Path,
    *,
    station_metadata: Sequence[SurfaceStation] | None=None,
    datum: str | None=None,
) -> SurfaceData:
    """Read legacy or 2.1 GNU-Fortran sequential-unformatted SURF.DAT.

    Args:
        path: Path to the binary SURF.DAT file.
        station_metadata: Control-file stations required for projection
            ``NONE``.
        datum: Control-file datum required for projection ``NONE``.

    Returns:
        A populated :class:`SurfaceData` instance whose observation fields are
        float32 or int32 arrays of shape ``(station,)``.

    Raises:
        OSError: If the input file cannot be opened.
        ValueError: If record framing, metadata, packing, timestamps, station
            identifiers, field sizes, or interval continuity are invalid.
    """
    source=Path(path); result=object.__new__(SurfaceData); result.path=source
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
            if payload is None: raise ValueError(f'truncated unformatted SURF.DAT {label}')
            if size is not None and len(payload)!=size: raise ValueError(f'unformatted SURF.DAT {label} has wrong size')
            return payload
        head=record(96,'identity')
        if head[:16].decode('ascii').strip().upper()!='SURF.DAT': raise ValueError('invalid SURF.DAT dataset label')
        result.version=head[16:32].decode('ascii').strip()
        explicit=result.version=='2.1'
        comments=struct.unpack('<i',record(4,'comment count'))[0]
        if comments<0: raise ValueError('negative SURF.DAT comment count')
        for _ in range(comments): record(80,'comment')
        projection=record(8,'projection').decode('ascii').strip().upper()
        if projection not in {'LL','NONE'}: raise ValueError('SURF.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('surface station locations are present in both control file and SURF.DAT')
            datum_record=record(20,'datum'); result.datum=datum_record[:8].decode('ascii').strip(); result.datum_date=datum_record[8:].decode('ascii').strip()
            # RDHD reads this five-integer record for every unformatted LL file.
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
                raise ValueError('SURF.DAT NONE requires control-file station metadata and datum')
            result.datum=str(datum); result.datum_date=''
            if explicit:
                result.timezone=record(8,'timezone').decode('ascii').strip()
                begin_code,begin_sec,end_code,end_sec,count,packing=struct.unpack('<6i',record(24,'time span'))
                result.start=_coded_instant(begin_code,begin_sec); result.end=_coded_instant(end_code,end_sec)
            else:
                begin_code,end_code,result.timezone,count,packing=struct.unpack('<5i',record(20,'time span'))
                result.start=_coded_instant(begin_code)-timedelta(hours=1)
                result.end=_coded_instant(end_code)
        if count<1 or packing not in {0,1}: raise ValueError('invalid SURF.DAT station count or packing code')
        identifiers=np.frombuffer(record(4*count,'station IDs'),dtype='<i4')
        if projection=='LL':
            names=record(4*count,'station names')
            latitudes=record(16*count,'station latitudes'); longitudes=record(16*count,'station longitudes')
            heights=np.frombuffer(record(4*count,'anemometer heights'),dtype='<f4')
            result.stations=tuple(SurfaceStation(int(identifiers[i]),names[4*i:4*i+4].decode('ascii').strip(),
                _coordinate(latitudes[16*i:16*i+16].decode('ascii'),True),
                _coordinate(longitudes[16*i:16*i+16].decode('ascii'),False),float(heights[i])) for i in range(count))
        else:
            result.stations=tuple(cast(Sequence[SurfaceStation],station_metadata))
            if len(result.stations)!=count or identifiers.tolist()!=[item.identifier for item in result.stations]:
                raise ValueError('SURF.DAT station IDs do not match control-file metadata')
        decoded_records: list[SurfaceRecord]=[]
        result.records=decoded_records
        while (payload:=reader.read_record()) is not None:
            prefix_size=16 if explicit else 4
            expected=prefix_size+(12 if packing else 32)*count
            if len(payload)!=expected: raise ValueError('unformatted SURF.DAT meteorological record has wrong size')
            if explicit:
                begin_code,begin_sec,end_code,end_sec=struct.unpack('<4i',payload[:16])
                begin,end=_coded_instant(begin_code,begin_sec),_coded_instant(end_code,end_sec)
            else:
                end_code=struct.unpack('<i',payload[:4])[0]
                end=_coded_instant(end_code); begin=end-timedelta(hours=1)
            if not begin<end or end-begin>timedelta(hours=1):
                raise ValueError('SURF.DAT record interval must advance by at most one hour')
            if packing:
                fields=unpack_surface(np.frombuffer(payload,dtype='<i4',offset=prefix_size).reshape(3,count).T)
            else:
                offset=prefix_size
                def take(dtype: DTypeLike) -> NDArray[Any]:
                    """Consume one station-vector field from a record payload.

                    Args:
                        dtype: NumPy dtype describing the little-endian int32
                            or float32 field.

                    Returns:
                        Independent array of shape ``(station,)`` with the
                        requested dtype.
                    """
                    nonlocal offset
                    values=np.frombuffer(payload,dtype=dtype,count=count,offset=offset).copy(); offset+=4*count
                    return values
                fields=cast(_SurfaceFields,dict(
                    wind_speed=take('<f4'),wind_direction=take('<f4'),
                    ceiling_hundreds_ft=take('<i4'),cloud_tenths=take('<i4'),
                    temperature_k=take('<f4'),relative_humidity=take('<i4'),
                    pressure_mb=take('<f4'),precipitation_code=take('<i4')))
            result.records.append(SurfaceRecord(begin,end,**fields))
    if not result.records: raise ValueError('SURF.DAT contains no meteorological records')
    for left,right in zip(result.records[:-1],result.records[1:]):
        if left.end!=right.begin: raise ValueError('SURF.DAT records contain a gap or overlap')
    if result.records[0].begin<result.start or result.records[-1].end>result.end:
        raise ValueError('SURF.DAT records fall outside the declared header period')
    return result


def load_surface(
    path: str | Path,
    format: int=2,
    *,
    station_metadata: Sequence[SurfaceStation] | None=None,
    datum: str | None=None,
) -> SurfaceData:
    """Load SURF.DAT according to CALMET ``IFORMS``.

    Args:
        path: Path to SURF.DAT.
        format: ``1`` for sequential-unformatted binary or ``2`` for formatted
            text.
        station_metadata: Control-file stations for projection ``NONE``.
        datum: Control-file datum for projection ``NONE``.

    Returns:
        Parsed :class:`SurfaceData` instance.

    Raises:
        ValueError: If ``format`` is not 1 or 2, or the selected reader rejects
            the file.
    """
    if format==1: return read_unformatted_surface(path,station_metadata=station_metadata,datum=datum)
    if format==2: return SurfaceData(path,station_metadata=station_metadata,datum=datum)
    raise ValueError('SURF.DAT format/IFORMS must be 1 or 2')


def station_grid_mapping(
    stations: Sequence[SurfaceStation],
    grid: GridConfig,
    datum: str,
) -> tuple[Float64Array,Int32Array]:
    """Project stations and build CALMET's nearest-station grid.

    Args:
        stations: Sequence of :class:`SurfaceStation` objects.
        grid: Target CALMET grid configuration.
        datum: Datum identifier of the station longitude/latitude coordinates.

    Returns:
        Tuple ``(points, nearest)``. ``points`` is a float64 array of shape
        ``(station, 2)`` containing projected x/y coordinates in kilometres;
        ``nearest`` is an int32 array of shape ``(grid.ny, grid.nx)`` holding
        zero-based station indices.

    Raises:
        ValueError: If projection metadata or the station coordinate array is
            invalid.
    """
    sx,sy=project_geographic([item.longitude for item in stations],
                             [item.latitude for item in stations],grid,datum,units='km')
    points: Float64Array=np.asarray(np.column_stack((sx,sy)),dtype=SPATIAL_DTYPE)
    return points,_nearest_station_grid(points,grid)


def _nearest_indices(
    points: ArrayLike,
    targets: ArrayLike,
    block_size: int=65536,
) -> IndexArray:
    """Query the nearest source point for each target.

    Args:
        points: Float-like source coordinates of shape ``(source, 2)``.
        targets: Float-like query coordinates of shape ``(target, 2)`` in the
            same units and coordinate system.
        block_size: Maximum targets passed to the k-d tree per query block.

    Returns:
        Platform-integer array of shape ``(target,)`` containing zero-based
        source indices.

    Raises:
        ValueError: If coordinate shapes are invalid, no source exists, or
            ``block_size`` is not positive.
    """
    source=np.asarray(points,dtype=SPATIAL_DTYPE)
    query=np.asarray(targets,dtype=SPATIAL_DTYPE)
    if source.ndim!=2 or source.shape[1]!=2 or source.shape[0]<1:
        raise ValueError('nearest-station coordinates must have shape (station, 2)')
    if query.ndim!=2 or query.shape[1]!=2:
        raise ValueError('nearest-station targets must have shape (target, 2)')
    if block_size<1: raise ValueError('nearest-station block size must be positive')
    tree=cKDTree(source)
    selected: IndexArray=np.empty(query.shape[0],dtype=np.intp)
    for start in range(0,query.shape[0],block_size):
        stop=min(start+block_size,query.shape[0])
        selected[start:stop]=np.asarray(
            tree.query(query[start:stop],k=1)[1],dtype=np.intp)
    return selected


def _nearest_neighbors(
    points: ArrayLike,
    targets: ArrayLike,
    count: int,
    block_size: int=65536,
) -> tuple[Float64Array,IndexArray]:
    """Return the closest source points and their distances.

    Args:
        points: Float-like source coordinates of shape ``(source, 2)``.
        targets: Float-like query coordinates of shape ``(target, 2)`` in the
            same units.
        count: Requested neighbor count, capped at the number of sources.
        block_size: Maximum targets queried in one k-d tree call.

    Returns:
        Tuple ``(distances, indices)`` with shape ``(target, neighbor)``.
        Distances are float64 in the coordinate units and indices use the
        platform integer dtype.

    Raises:
        ValueError: If coordinate shapes, ``count``, or ``block_size`` are
            invalid.
    """
    source=np.asarray(points,dtype=SPATIAL_DTYPE)
    query=np.asarray(targets,dtype=SPATIAL_DTYPE)
    if source.ndim!=2 or source.shape[1]!=2 or source.shape[0]<1:
        raise ValueError('nearest-station coordinates must have shape (station, 2)')
    if query.ndim!=2 or query.shape[1]!=2:
        raise ValueError('nearest-station targets must have shape (target, 2)')
    if block_size<1: raise ValueError('nearest-station block size must be positive')
    count=min(int(count),source.shape[0])
    if count<1: raise ValueError('nearest-station count must be positive')
    tree=cKDTree(source)
    chosen: IndexArray=np.empty((query.shape[0],count),dtype=np.intp)
    distances: Float64Array=np.empty((query.shape[0],count),dtype=SPATIAL_DTYPE)
    for start in range(0,query.shape[0],block_size):
        stop=min(start+block_size,query.shape[0])
        block_distance,block_chosen=tree.query(query[start:stop],k=count)
        if count==1:
            block_distance=block_distance[:,None]
            block_chosen=block_chosen[:,None]
        chosen[start:stop]=np.asarray(block_chosen,dtype=np.intp)
        distances[start:stop]=np.asarray(block_distance,dtype=SPATIAL_DTYPE)
    return distances,chosen


def _nearest_station_grid(
    points: ArrayLike,
    grid: GridConfig,
    block_size: int=65536,
) -> Int32Array:
    """Return zero-based nearest-station IDs for every grid cell.

    Args:
        points: Float-like ``(station, 2)`` projected coordinates in kilometres.
        grid: Target grid configuration; station coordinates may include the
            grid origin while cell targets are evaluated relative to it.
        block_size: Maximum grid-cell targets queried per block.

    Returns:
        Int32 array of shape ``(grid.ny, grid.nx)`` containing zero-based
        nearest-station indices.

    Raises:
        ValueError: If station coordinates are invalid or ``block_size`` is not
            positive.
    """
    points=np.asarray(points,dtype=SPATIAL_DTYPE)
    if points.ndim!=2 or points.shape[1]!=2 or points.shape[0]<1:
        raise ValueError('nearest-station coordinates must have shape (station, 2)')
    origin=np.array([grid.x_origin_km,grid.y_origin_km],dtype=SPATIAL_DTYPE)
    relative=points-origin
    x=(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*float(grid.dx_km)
    y=(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*float(grid.dy_km)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    nearest=_nearest_indices(relative,targets,block_size).astype(np.int32,copy=False)
    return nearest.reshape(grid.ny,grid.nx)


def fill_missing_surface(
    record: SurfaceRecord,
    station_xy: ArrayLike,
    *,
    cloud_method: int=1,
    itprog: int=0,
) -> SurfaceRecord:
    """Fill required missing observations from each variable's nearest station.

    Args:
        record: :class:`SurfaceRecord` whose station vectors use 9999 as the
            missing sentinel.
        station_xy: Float-like array of shape ``(station, 2)`` containing
            projected station coordinates in a common distance unit.
        cloud_method: CALMET cloud method. Ceiling and cloud cover are filled
            only for methods below 3.
        itprog: Temperature-processing mode. Temperature is not filled when
            ``itprog == 2``.

    Returns:
        A new :class:`SurfaceRecord`. Required temperature, humidity, pressure,
        cloud, and ceiling fields have nearest valid values substituted while
        winds and precipitation codes are copied unchanged. Float arrays are
        float32 and coded arrays are int32.

    Raises:
        ValueError: If station coordinates do not match the record or a
            required variable is missing at every station.
    """
    coordinates=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    count=record.wind_speed.size
    if coordinates.shape!=(count,2): raise ValueError('surface station coordinates have wrong shape')

    @overload
    def filled(
        values: ArrayLike,
        required: bool,
        integer: Literal[False]=False,
    ) -> Float32Array:
        """Type floating-point nearest-station filling.

        Args:
            values: One-dimensional station field.
            required: Whether missing values must be filled.
            integer: Literal selecting floating-point output.

        Returns:
            Filled float32 station values.
        """
        ...

    @overload
    def filled(
        values: ArrayLike,
        required: bool,
        integer: Literal[True],
    ) -> Int32Array:
        """Type integer nearest-station filling.

        Args:
            values: One-dimensional station field.
            required: Whether missing values must be filled.
            integer: Literal selecting integer output.

        Returns:
            Filled int32 station values.
        """
        ...

    def filled(
        values: ArrayLike,
        required: bool,
        integer: bool=False,
    ) -> Float32Array | Int32Array:
        """Fill one station vector from its nearest valid observations.

        Args:
            values: One-dimensional station field.
            required: Whether missing values must be filled.
            integer: Whether to return int32 and apply the integer missing
                sentinel; otherwise return float32.

        Returns:
            Independent array with the same shape as ``values``. Required
            missing elements contain the nearest valid station value.

        Raises:
            ValueError: If filling is required and no station has a valid
                value.
        """
        source=np.asarray(values,dtype=np.int32 if integer else COMPUTE_DTYPE)
        missing=(source>=9998.99
                 if not integer else source>=9999)
        if not required or not missing.any():
            return cast(Float32Array | Int32Array,source.copy())
        valid=~missing
        if not valid.any(): raise ValueError('surface variable is missing at every station')
        valid_indices=np.flatnonzero(valid)
        nearest=_nearest_indices(coordinates[valid],coordinates[missing])
        result=source.copy()
        result[missing]=source[valid_indices[nearest]]
        return cast(Float32Array | Int32Array,result)

    return SurfaceRecord(record.begin,record.end,
        np.asarray(record.wind_speed,dtype=COMPUTE_DTYPE).copy(),
        np.asarray(record.wind_direction,dtype=COMPUTE_DTYPE).copy(),
        filled(record.ceiling_hundreds_ft,cloud_method<3,True).astype(np.int32),
        filled(record.cloud_tenths,cloud_method<3,True).astype(np.int32),
        filled(record.temperature_k,itprog!=2),
        filled(record.relative_humidity,True),
        filled(record.pressure_mb,True),
        np.asarray(record.precipitation_code,dtype=np.int32).copy())


def surface_cloud_fields(
    record: SurfaceRecord,
    nearest_station: ArrayLike,
    external_cloud: ArrayLike | None=None,
) -> tuple[Float32Array,Int32Array]:
    """Map OUTCLD/PGTSTB cloud cover and ceiling to the model grid.

    Args:
        record: :class:`SurfaceRecord` containing station cloud and ceiling
            observations.
        nearest_station: Integer array of shape ``(ny, nx)`` with zero-based
            station indices.
        external_cloud: Optional cloud-fraction array of shape ``(ny, nx)``.
            When omitted, station cloud tenths are converted to fractions.

    Returns:
        Tuple ``(cloud, ceiling)``. ``cloud`` is float32 with shape ``(ny, nx)``
        and unitless fraction; ``ceiling`` is int32 with the same shape and
        units of hundreds of feet.

    Raises:
        ValueError: If station indices are invalid or ``external_cloud`` does
            not match the target grid.
    """
    nearest=np.asarray(nearest_station)
    if nearest.ndim!=2 or np.any((nearest<0)|(nearest>=record.cloud_tenths.size)):
        raise ValueError('nearest surface-station index is invalid')
    ceiling=np.asarray(record.ceiling_hundreds_ft)[nearest]
    if external_cloud is None:
        cloud=.1*np.asarray(record.cloud_tenths,dtype=COMPUTE_DTYPE)[nearest]
    else:
        cloud=np.asarray(external_cloud,dtype=COMPUTE_DTYPE)
        if cloud.shape!=nearest.shape: raise ValueError('external cloud field does not match CALMET grid')
    return cloud.astype(COMPUTE_DTYPE,copy=False),ceiling.astype(np.int32)


def interpolate_surface_scalar(
    values: ArrayLike,
    station_xy: ArrayLike,
    grid: GridConfig,
    *,
    radius_km: float=500.,
    maximum_stations: int=5,
    power: int=1,
    target_water: ArrayLike | None=None,
    station_water: ArrayLike | None=None,
    empty_fallback: ArrayLike | None=None,
) -> Float32Array:
    """Interpolate a surface scalar with IDW and land/water separation.

    Args:
        values: Float-like ``(station,)`` values, such as temperature in kelvin
            or relative humidity in percent. Values at least 9998.99 are
            missing.
        station_xy: Float-like ``(station, 2)`` projected coordinates in
            kilometres.
        grid: Target grid configuration.
        radius_km: Maximum distance for weighted neighbors. If every selected
            neighbor is farther away, the nearest valid station is used.
        maximum_stations: Maximum nearest stations included for each cell.
        power: Inverse-distance exponent, either 1 or 2.
        target_water: Optional boolean ``(grid.ny, grid.nx)`` water mask.
        station_water: Optional boolean ``(station,)`` water-station mask.
        empty_fallback: Optional two-value ``(land, water)`` fallback used when
            a target category has no valid station.

    Returns:
        Float32 array of shape ``(grid.ny, grid.nx)`` in the same units as
        ``values``.

    Raises:
        ValueError: If shapes or interpolation controls are invalid, or a land
            or water target has no usable station and no fallback.
    """
    values=np.asarray(values,dtype=COMPUTE_DTYPE)
    points=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    if points.shape!=(values.size,2): raise ValueError('surface scalar station coordinates have wrong shape')
    if power not in {1,2} or radius_km<0. or maximum_stations<1:
        raise ValueError('surface scalar interpolation requires power 1/2, non-negative radius and positive station count')
    water=np.zeros(values.size,dtype=bool) if station_water is None else np.asarray(station_water,dtype=bool)
    if water.shape!=values.shape: raise ValueError('station water mask has wrong shape')
    target_mask=np.zeros((grid.ny,grid.nx),dtype=bool) if target_water is None else np.asarray(target_water,dtype=bool)
    if target_mask.shape!=(grid.ny,grid.nx): raise ValueError('target water mask has wrong shape')
    x=float(grid.x_origin_km)+(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*float(grid.dx_km)
    y=float(grid.y_origin_km)+(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*float(grid.dy_km)
    xx,yy=np.meshgrid(x,y); targets=np.column_stack((xx.ravel(),yy.ravel()))
    output: Float32Array=np.empty(targets.shape[0],dtype=COMPUTE_DTYPE)
    flat_water=target_mask.ravel()
    valid=values<9998.99
    radius=float(radius_km)
    fallback: Float32Array | None=None
    if empty_fallback is not None:
        fallback=np.asarray(empty_fallback,dtype=COMPUTE_DTYPE)
        if fallback.shape!=(2,): raise ValueError('empty_fallback must contain land and water values')
    for is_water in (False,True):
        target_indices=np.flatnonzero(flat_water==is_water)
        if target_indices.size==0: continue
        station_indices=np.flatnonzero(valid&(water==is_water))
        if station_indices.size==0:
            if fallback is None:
                raise ValueError('grid cell has no valid land/water surface temperature station')
            output[target_indices]=fallback[int(is_water)]
            continue
        distance,local_indices=_nearest_neighbors(
            points[station_indices],targets[target_indices],maximum_stations)
        selected=station_indices[local_indices]
        selected_valid=distance<=radius
        exact=selected_valid&(distance==0.)
        metric=distance if power==1 else distance*distance
        weights=np.zeros_like(distance,dtype=SPATIAL_DTYPE)
        np.divide(1.,metric,out=weights,where=selected_valid&~exact)
        # Summation in double precision keeps many-station weighted means stable;
        # the model field itself remains FP32.
        denominator=np.sum(weights,dtype=np.float64,axis=1)
        numerator=np.sum(weights*values[selected],dtype=np.float64,axis=1)
        weighted=np.divide(numerator,denominator,out=np.zeros_like(numerator),
                           where=denominator!=0.)
        exact_count=np.sum(exact,axis=1)
        exact_total=np.sum(np.where(exact,values[selected],0.),dtype=np.float64,axis=1)
        exact_value=np.divide(exact_total,exact_count,out=np.zeros_like(exact_total),
                              where=exact_count!=0)
        result=np.where(exact_count!=0,exact_value,weighted)
        result=np.where(distance[:,0]>radius,values[selected[:,0]],result)
        output[target_indices]=result.astype(COMPUTE_DTYPE)
    return output.reshape(grid.ny,grid.nx)


def _near_upper_air(
    station_count: int,
    station_xy: ArrayLike | None,
    upper_xy: ArrayLike | None,
    upper_valid: ArrayLike | None,
    separation_cells: float,
    dx_km: float,
    dy_km: float,
) -> BoolArray:
    """Identify surface stations within RMIN2 of valid upper-air data.

    Args:
        station_count: Number of surface stations.
        station_xy: Float-like ``(station_count, 2)`` surface coordinates in
            kilometres.
        upper_xy: Float-like ``(upper_station, 2)`` upper-air coordinates in
            kilometres, or ``None`` to disable filtering.
        upper_valid: Boolean ``(upper_station,)`` mask of usable upper-air
            observations.
        separation_cells: RMIN2 distance in grid-cell units. A negative value
            disables filtering.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.

    Returns:
        Boolean array of shape ``(station_count,)``; ``True`` marks surface
        stations whose extrapolated profiles should be skipped.

    Raises:
        ValueError: If enabled coordinates or masks have inconsistent shapes,
            or grid spacing is not positive.
    """
    skip=np.zeros(station_count,dtype=bool)
    if separation_cells<0. or upper_xy is None:
        return skip
    if dx_km<=0. or dy_km<=0.:
        raise ValueError('RMIN2 grid spacing must be positive')
    surface_points=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    upper_points=np.asarray(upper_xy,dtype=SPATIAL_DTYPE)
    available=np.asarray(upper_valid,dtype=bool)
    if surface_points.shape!=(station_count,2) or upper_points.ndim!=2 or upper_points.shape[1]!=2:
        raise ValueError('RMIN2 station coordinates are inconsistent')
    if available.shape!=(upper_points.shape[0],):
        raise ValueError('RMIN2 upper validity has wrong shape')
    if available.any():
        scale=np.asarray((dx_km,dy_km),dtype=SPATIAL_DTYPE)
        tree=cKDTree(upper_points[available]/scale)
        distance,_=tree.query(surface_points/scale,k=1)
        skip=distance<float(separation_cells)
    return skip


def _wind_components(
    wind_speed: ArrayLike,
    wind_direction: ArrayLike,
    grid_convergence_degrees: ArrayLike | None=None,
) -> tuple[Float32Array,Float32Array,BoolArray,Float32Array]:
    """Convert meteorological wind observations to grid-oriented components.

    Args:
        wind_speed: Station wind speeds in metres per second.
        wind_direction: Meteorological directions in degrees from north, with
            the same shape as ``wind_speed``.
        grid_convergence_degrees: Optional map-meridian convergence in degrees
            for every observation.

    Returns:
        Tuple ``(u, v, valid, direction)``. Components and adjusted directions
        are float32 arrays with the input shape; ``u`` and ``v`` are metres per
        second and use 9999 for missing values. ``valid`` is boolean.

    Raises:
        ValueError: If speed, direction, and convergence shapes differ.
    """
    speed=np.asarray(wind_speed,dtype=COMPUTE_DTYPE)
    direction=np.asarray(wind_direction,dtype=COMPUTE_DTYPE)
    if speed.shape!=direction.shape:
        raise ValueError('wind speed and direction have different shapes')
    convergence=(np.zeros_like(direction) if grid_convergence_degrees is None else
                 np.asarray(grid_convergence_degrees,dtype=COMPUTE_DTYPE))
    if convergence.shape!=direction.shape:
        raise ValueError('grid convergence must have one value per station')
    valid=(np.abs(speed)<999.)&(np.abs(direction)<999.)
    rotated=np.mod(direction-convergence,360.)
    direction=np.where(valid,rotated,direction).astype(COMPUTE_DTYPE,copy=False)
    angle=np.deg2rad(direction)
    u=np.where(valid,-speed*np.sin(angle),9999.).astype(COMPUTE_DTYPE,copy=False)
    v=np.where(valid,-speed*np.cos(angle),9999.).astype(COMPUTE_DTYPE,copy=False)
    return u,v,valid,direction


def _apply_wind_extrapolation(
    u: Float32Array,
    v: Float32Array,
    valid: BoolArray,
    calm_speed: Float32Array,
    anem: Float32Array,
    levels: Float32Array,
    roughness: Float32Array,
    *,
    iextrp: int,
    factors: ArrayLike | None,
    station_latitude: ArrayLike | None,
    station_xy: ArrayLike | None,
    upper_xy: ArrayLike | None,
    upper_valid: ArrayLike | None,
    upper_air_separation_cells: float,
    dx_km: float,
    dy_km: float,
    power_exponent: ArrayLike,
) -> tuple[Float32Array,Float32Array]:
    """Apply one IEXTRP vertical profile to each surface observation.

    Args:
        u: Float-like ``(station,)`` grid-oriented U wind in metres per second.
        v: Float-like ``(station,)`` grid-oriented V wind in metres per second.
        valid: Boolean ``(station,)`` observation-validity mask.
        calm_speed: Float-like ``(station,)`` speed used to identify calms.
        anem: Float-like ``(station,)`` anemometer heights in metres.
        levels: Float-like ``(level,)`` target heights in metres above ground.
        roughness: Float-like ``(station,)`` roughness lengths in metres.
        iextrp: Signed PREPDI option; its absolute value selects method 1--4.
        factors: Per-level FEXTR2 factors required by method 3.
        station_latitude: Station latitudes in degrees, required by method 4.
        station_xy: ``(station, 2)`` projected coordinates in kilometres.
        upper_xy: ``(upper_station, 2)`` upper-air coordinates, or ``None``.
        upper_valid: Boolean validity mask for ``upper_xy``.
        upper_air_separation_cells: RMIN2 exclusion distance in grid cells.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.
        power_exponent: Scalar or ``(station,)`` exponent for method 2.

    Returns:
        Tuple ``(u_profile, v_profile)`` of float32 arrays with shape
        ``(level, station)`` and units of metres per second. Missing values are
        9999.

    Raises:
        ValueError: If RMIN2 inputs are inconsistent, the option is unsupported,
            method-specific inputs have wrong shapes, or method 4 encounters
            an anemometer height equal to roughness length.
    """
    output_u=np.full((levels.size,u.size),9999.,dtype=COMPUTE_DTYPE)
    output_v=np.full_like(output_u,9999.)
    skip=_near_upper_air(u.size,station_xy,upper_xy,upper_valid,
                         upper_air_separation_cells,dx_km,dy_km)
    output_u[0,skip&valid]=u[skip&valid]
    output_v[0,skip&valid]=v[skip&valid]
    calm=valid&(calm_speed<1.e-4)&~skip
    output_u[:,calm]=0.
    output_v[:,calm]=0.
    active=valid&~calm&~skip
    option=abs(int(iextrp))
    if option==1:
        coefficient=(np.log(levels[0])-np.log(roughness))/(
            np.log(anem)-np.log(roughness))
        output_u[0,active]=u[active]*coefficient[active]
        output_v[0,active]=v[active]*coefficient[active]
    elif option==2:
        exponent=np.broadcast_to(
            np.asarray(power_exponent,dtype=COMPUTE_DTYPE),u.shape)
        # Every target level is referenced directly to the observation height.
        # This is the ordinary power-law profile and does not make one layer's
        # result depend on rounding performed while computing the layer below.
        coefficient=(levels[:,None]/anem[None,:])**exponent[None,:]
        output_u[:,active]=u[None,active]*coefficient[:,active]
        output_v[:,active]=v[None,active]*coefficient[:,active]
    elif option==3:
        scale=np.asarray(factors,dtype=COMPUTE_DTYPE)
        if scale.shape!=(levels.size,):
            raise ValueError('IEXTRP=3 requires one FEXTR2 factor per layer')
        output_u[:,active]=u[None,active]*scale[:,None]
        output_v[:,active]=v[None,active]*scale[:,None]
    elif option==4:
        latitude=np.asarray(station_latitude,dtype=COMPUTE_DTYPE)
        if latitude.shape!=u.shape:
            raise ValueError('IEXTRP=4 requires one latitude per station')
        speed=np.hypot(u,v)
        direction=np.mod(270.-np.rad2deg(np.arctan2(v,u)),360.)
        direction=np.where(direction==0.,360.,direction)
        denominator=np.log(anem/roughness)
        if np.any(np.abs(denominator[active])<1.e-12):
            raise ValueError('anemometer height must differ from roughness')
        magnitude=(speed[None,:]*np.log(levels[:,None]/roughness[None,:])/
                   denominator[None,:])
        turning_scale=12.*1.58
        at_anem=turning_scale*(1.-np.exp(-anem/200.))
        at_level=turning_scale*(1.-np.exp(-levels/200.))
        turning=at_level[:,None]-at_anem[None,:]
        adjusted=np.where(latitude[None,:]>0.,
                          direction[None,:]+turning,
                          direction[None,:]-turning)
        radians=np.deg2rad(np.mod(adjusted,360.))
        output_u[:,active]=-magnitude[:,active]*np.sin(radians[:,active])
        output_v[:,active]=-magnitude[:,active]*np.cos(radians[:,active])
    else:
        raise ValueError('IEXTRP must be one of +/-1, +/-2, +/-3, or +/-4')
    return output_u,output_v


def wind_profiles(
    wind_speed: ArrayLike,
    wind_direction: ArrayLike,
    anemometer_height_m: ArrayLike,
    z_centers_m: ArrayLike,
    roughness_m: ArrayLike,
    iextrp: int=-1,
    factors: ArrayLike | None=None,
    station_latitude: ArrayLike | None=None,
    grid_convergence_degrees: ArrayLike | None=None,
    station_xy: ArrayLike | None=None,
    upper_xy: ArrayLike | None=None,
    upper_valid: ArrayLike | None=None,
    upper_air_separation_cells: float=-1.,
    dx_km: float=1.,
    dy_km: float=1.,
    power_exponent: ArrayLike=.143,
) -> tuple[Float32Array,Float32Array]:
    """Convert observed winds and apply PREPDI IEXTRP methods 1--4.

    Args:
        wind_speed: ``(station,)`` wind speed in metres per second.
        wind_direction: ``(station,)`` meteorological direction in degrees.
        anemometer_height_m: ``(station,)`` sensor heights in metres.
        z_centers_m: ``(level,)`` target layer-center heights in metres.
        roughness_m: ``(station,)`` roughness lengths in metres.
        iextrp: Signed extrapolation option; absolute values 1--4 are supported.
        factors: ``(level,)`` FEXTR2 factors used by method 3.
        station_latitude: ``(station,)`` latitude in degrees for method 4.
        grid_convergence_degrees: Optional ``(station,)`` map convergence in
            degrees.
        station_xy: Optional ``(station, 2)`` projected coordinates in
            kilometres for RMIN2 filtering.
        upper_xy: Optional ``(upper_station, 2)`` upper-air coordinates.
        upper_valid: Optional boolean ``(upper_station,)`` availability mask.
        upper_air_separation_cells: RMIN2 exclusion distance in grid cells; a
            negative value disables it.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.
        power_exponent: Scalar or ``(station,)`` exponent for method 2.

    Returns:
        Tuple of float32 U and V arrays, each shaped ``(level, station)`` in
        metres per second and using 9999 for missing values.

    Raises:
        ValueError: If input shapes, heights, roughness, RMIN2 controls, or the
            selected IEXTRP method are invalid.
    """
    speed=np.asarray(wind_speed,dtype=COMPUTE_DTYPE)
    anem=np.asarray(anemometer_height_m,dtype=COMPUTE_DTYPE)
    roughness=np.asarray(roughness_m,dtype=COMPUTE_DTYPE)
    levels=np.asarray(z_centers_m,dtype=COMPUTE_DTYPE)
    if (anem.shape!=speed.shape or roughness.shape!=speed.shape or
            levels.ndim!=1 or levels.size==0):
        raise ValueError('surface wind profile inputs have incompatible shapes')
    if np.any(anem<=0.) or np.any(roughness<=0.) or np.any(levels<=0.):
        raise ValueError('wind heights and roughness must be positive')
    u,v,valid,_=_wind_components(speed,wind_direction,grid_convergence_degrees)
    return _apply_wind_extrapolation(
        u,v,valid,speed,anem,levels,roughness,iextrp=iextrp,factors=factors,
        station_latitude=station_latitude,station_xy=station_xy,upper_xy=upper_xy,
        upper_valid=upper_valid,upper_air_separation_cells=upper_air_separation_cells,
        dx_km=dx_km,dy_km=dy_km,power_exponent=power_exponent)


def surface_wind_profiles(
    record: SurfaceRecord,
    anemometer_height_m: ArrayLike,
    z_centers_m: ArrayLike,
    roughness_m: ArrayLike,
    iextrp: int=-1,
    factors: ArrayLike | None=None,
    station_latitude: ArrayLike | None=None,
    grid_convergence_degrees: ArrayLike | None=None,
    station_xy: ArrayLike | None=None,
    upper_xy: ArrayLike | None=None,
    upper_valid: ArrayLike | None=None,
    upper_air_separation_cells: float=-1.,
    dx_km: float=1.,
    dy_km: float=1.,
    power_exponent: ArrayLike=.143,
) -> tuple[Float32Array,Float32Array]:
    """Apply PREPDI wind extrapolation to one SURF.DAT record.

    Args:
        record: :class:`SurfaceRecord` supplying wind speed and direction.
        anemometer_height_m: ``(station,)`` sensor heights in metres.
        z_centers_m: ``(level,)`` target layer-center heights in metres.
        roughness_m: ``(station,)`` roughness lengths in metres.
        iextrp: Signed extrapolation option; absolute values 1--4 are supported.
        factors: ``(level,)`` factors required by method 3.
        station_latitude: ``(station,)`` latitude in degrees for method 4.
        grid_convergence_degrees: Optional ``(station,)`` map convergence in
            degrees.
        station_xy: Optional ``(station, 2)`` coordinates in kilometres.
        upper_xy: Optional ``(upper_station, 2)`` upper-air coordinates.
        upper_valid: Optional boolean ``(upper_station,)`` validity mask.
        upper_air_separation_cells: RMIN2 exclusion distance in grid cells.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.
        power_exponent: Scalar or ``(station,)`` exponent for method 2.

    Returns:
        Tuple of float32 U and V arrays with shape ``(level, station)`` and
        units of metres per second.

    Raises:
        ValueError: If profile inputs or the selected method are invalid.
    """
    return wind_profiles(record.wind_speed,record.wind_direction,anemometer_height_m,
        z_centers_m,roughness_m,iextrp,factors,station_latitude,
        grid_convergence_degrees,station_xy,upper_xy,upper_valid,
        upper_air_separation_cells,dx_km,dy_km,power_exponent)


def component_wind_profiles(
    u_component: ArrayLike,
    v_component: ArrayLike,
    anemometer_height_m: ArrayLike,
    z_centers_m: ArrayLike,
    roughness_m: ArrayLike,
    iextrp: int=-1,
    factors: ArrayLike | None=None,
    station_latitude: ArrayLike | None=None,
    station_xy: ArrayLike | None=None,
    upper_xy: ArrayLike | None=None,
    upper_valid: ArrayLike | None=None,
    upper_air_separation_cells: float=-1.,
    dx_km: float=1.,
    dy_km: float=1.,
    power_exponent: ArrayLike=.143,
) -> tuple[Float32Array,Float32Array]:
    """Profile grid-oriented DIAG.DAT U/V surface observations.

    Args:
        u_component: ``(station,)`` U wind in metres per second.
        v_component: ``(station,)`` V wind in metres per second.
        anemometer_height_m: ``(station,)`` sensor heights in metres.
        z_centers_m: ``(level,)`` target layer-center heights in metres.
        roughness_m: ``(station,)`` roughness lengths in metres.
        iextrp: Signed extrapolation option; absolute values 1--4 are supported.
        factors: ``(level,)`` factors required by method 3.
        station_latitude: ``(station,)`` latitude in degrees for method 4.
        station_xy: Optional ``(station, 2)`` coordinates in kilometres.
        upper_xy: Optional ``(upper_station, 2)`` upper-air coordinates.
        upper_valid: Optional boolean ``(upper_station,)`` validity mask.
        upper_air_separation_cells: RMIN2 exclusion distance in grid cells.
        dx_km: Grid-cell width in kilometres.
        dy_km: Grid-cell height in kilometres.
        power_exponent: Scalar or ``(station,)`` exponent for method 2.

    Returns:
        Tuple of float32 U and V arrays with shape ``(level, station)`` and
        units of metres per second. Components above 900 are treated as
        missing.

    Raises:
        ValueError: If component shapes, profile inputs, RMIN2 controls, or the
            selected extrapolation method are invalid.
    """
    u=np.asarray(u_component,dtype=COMPUTE_DTYPE)
    v=np.asarray(v_component,dtype=COMPUTE_DTYPE)
    if u.shape!=v.shape: raise ValueError('surface U/V components have different shapes')
    anem=np.asarray(anemometer_height_m,dtype=COMPUTE_DTYPE)
    roughness=np.asarray(roughness_m,dtype=COMPUTE_DTYPE)
    levels=np.asarray(z_centers_m,dtype=COMPUTE_DTYPE)
    if (anem.shape!=u.shape or roughness.shape!=u.shape or
            levels.ndim!=1 or levels.size==0):
        raise ValueError('surface component profile inputs have incompatible shapes')
    if np.any(anem<=0.) or np.any(roughness<=0.) or np.any(levels<=0.):
        raise ValueError('wind heights and roughness must be positive')
    # US/VS are initialized to EDIT=999 and DIAGNO treats values above
    # EDITL=900 as missing before vertical extrapolation.
    valid=(u<=900.)&(v<=900.)
    speed=np.hypot(u,v).astype(COMPUTE_DTYPE,copy=False)
    return _apply_wind_extrapolation(
        u,v,valid,speed,anem,levels,roughness,iextrp=iextrp,factors=factors,
        station_latitude=station_latitude,station_xy=station_xy,upper_xy=upper_xy,
        upper_valid=upper_valid,upper_air_separation_cells=upper_air_separation_cells,
        dx_km=dx_km,dy_km=dy_km,power_exponent=power_exponent)
