"""Formatted CALMET UP.DAT sounding reader."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import numpy as np

from .surface_data import _coordinate
from .datetime_utils import parse_calendar,parse_ordinal,parse_timezone_label


COMPUTE_DTYPE=np.float32
SPATIAL_DTYPE=np.float64
DRY_ADIABATIC_LAPSE_RATE=COMPUTE_DTYPE(.0098)


def _instant(year,day,hour,seconds=0):
    """Build an UP.DAT datetime from ordinal-date components.

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
    except ValueError as error: raise ValueError('invalid UP.DAT timestamp') from error


def _sounding_header(line):
    """Decode a snapshot sounding header in delimited or fixed-column form.

    Args:
        line: UP.DAT sounding-header text.

    Returns:
        Tuple ``(station_id, year, month, day, hour, levels)``. Calendar fields
        retain their input strings for the shared datetime parser; the station
        identifier and level count are integers.

    Raises:
        ValueError: If integer fields cannot be decoded.
    """
    fields=line.replace(',',' ').split()
    try:
        if len(fields)>=6:
            return (int(fields[0]),*fields[1:5],int(fields[5]))
        padded=line.ljust(70)
        return (int(padded[9:17]),padded[22:24],padded[24:26],
                padded[26:28],padded[28:30],int(padded[65:70]))
    except ValueError as error: raise ValueError('invalid fixed-column UP.DAT sounding header') from error


def _sounding_header22(line):
    """Decode an UP.DAT 2.2 interval sounding header.

    Args:
        line: Delimited or fixed-column header text.

    Returns:
        Twelve-value tuple containing station ID, five begin-time fields, five
        end-time fields, and the integer level count.

    Raises:
        ValueError: If integer fields cannot be decoded.
    """
    fields=line.replace(',',' ').split()
    try:
        if len(fields)>=12:
            return (int(fields[0]),*fields[1:11],int(fields[11]))
        padded=line.ljust(76)
        return (int(padded[9:17]),padded[21:25],padded[25:29],padded[29:32],
                padded[32:35],padded[35:40],padded[44:48],padded[48:52],
                padded[52:55],padded[55:58],padded[58:63],int(padded[71:76]))
    except ValueError as error: raise ValueError('invalid fixed-column UP.DAT 2.2 sounding header') from error


def _calendar_instant(year,month,day,hour,seconds=0):
    """Build an UP.DAT datetime from calendar components.

    Args:
        year: Calendar year.
        month: Calendar month.
        day: Day of month.
        hour: Hour of day.
        seconds: Seconds after the hour.

    Returns:
        Decoded naive :class:`datetime`.

    Raises:
        ValueError: If the calendar timestamp is invalid.
    """
    try: return parse_calendar(year,month,day,hour,seconds)
    except ValueError as error: raise ValueError('invalid UP.DAT calendar timestamp') from error


@dataclass(frozen=True,slots=True)
class UpperAirStation:
    """Upper-air station metadata.

    Attributes:
        identifier: Integer station identifier used in sounding headers.
        name: Station name.
        latitude: Geographic latitude in decimal degrees north.
        longitude: Geographic longitude in decimal degrees east.
        elevation_m: Station elevation in metres above mean sea level.
    """
    identifier: int
    name: str
    latitude: float
    longitude: float
    elevation_m: float


@dataclass(frozen=True,slots=True)
class Sounding:
    """One vertical upper-air profile.

    Attributes:
        timestamp: Snapshot time or beginning of an averaged interval.
        pressure_mb: Float32 ``(source_level,)`` pressure in millibars.
        height_msl: Float32 ``(source_level,)`` height in metres above mean sea
            level.
        temperature_c: Float32 ``(source_level,)`` temperature in degrees
            Celsius.
        wind_direction: Float32 ``(source_level,)`` meteorological wind
            direction in degrees from north.
        wind_speed: Float32 ``(source_level,)`` wind speed in metres per second.
        end_timestamp: End of a version 2.2 averaged interval, or ``None`` for
            an instantaneous sounding.
    """
    timestamp: datetime
    pressure_mb: np.ndarray
    height_msl: np.ndarray
    temperature_c: np.ndarray
    wind_direction: np.ndarray
    wind_speed: np.ndarray
    end_timestamp: datetime | None = None

    def converted(self,convergence_degrees=0.):
        """Convert MSL heights and meteorological winds to local U/V profiles.

        Args:
            convergence_degrees: Scalar or ``(source_level,)`` map convergence
                in degrees, subtracted from valid meteorological directions.

        Returns:
            Tuple ``(height, u, v)`` of float32 arrays with shape
            ``(source_level,)``. Height is metres above the first sounding
            level; U and V are metres per second and use 999 for missing winds.

        Raises:
            ValueError: If the sounding is empty or contains missing heights.
        """
        if self.height_msl.size==0: raise ValueError('cannot convert an empty upper-air sounding')
        if np.any(self.height_msl>=9998.): raise ValueError('upper-air sounding contains missing heights')
        height=np.asarray(self.height_msl,dtype=COMPUTE_DTYPE)
        height=height-height[0]
        direction=np.asarray(self.wind_direction,dtype=COMPUTE_DTYPE)
        speed=np.asarray(self.wind_speed,dtype=COMPUTE_DTYPE)
        valid=(direction<998.99)&(speed<998.99)
        # RDUP applies map convergence only when both encoded direction and
        # speed exceed 0.01. Preserve its zero-direction/calm branch.
        rotate=valid&(direction>.01)&(speed>.01)
        convergence=np.asarray(convergence_degrees,dtype=COMPUTE_DTYPE)
        adjusted=np.mod(direction-convergence,360.)
        direction=np.where(rotate,adjusted,direction)
        radians=np.deg2rad(direction)
        u=np.where(valid,-speed*np.sin(radians),999.)
        v=np.where(valid,-speed*np.cos(radians),999.)
        return (height.astype(COMPUTE_DTYPE,copy=False),
                u.astype(COMPUTE_DTYPE,copy=False),
                v.astype(COMPUTE_DTYPE,copy=False))


def layer_average(sounding,z_faces_m,convergence_degrees=0.):
    """Integrate a piecewise-linear sounding wind over model layers.

    Args:
        sounding: :class:`Sounding` to convert and vertically average.
        z_faces_m: Strictly increasing layer-face heights in metres, beginning
            at zero, with shape ``(model_level + 1,)``.
        convergence_degrees: Scalar or source-level map convergence in degrees.

    Returns:
        Tuple ``(u, v)`` of float32 layer-mean wind arrays with shape
        ``(model_level,)`` and units of metres per second.

    Raises:
        ValueError: If faces are invalid, fewer than two valid wind levels
            remain, sounding heights are not strictly increasing, or the model
            layer range extends beyond valid sounding winds.
    """
    faces=np.asarray(z_faces_m,dtype=COMPUTE_DTYPE)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('upper-air layer faces must increase strictly from zero')
    height,u,v=sounding.converted(convergence_degrees)
    valid=(u<998.99)&(v<998.99)
    height=height[valid]; u=u[valid]; v=v[valid]
    if height.size<2 or np.any(np.diff(height)<=0.):
        raise ValueError('upper-air sounding needs two strictly increasing valid wind levels')
    if faces[0]<height[0]:
        raise ValueError('lowest valid upper-air wind is above a CALMET cell face')
    if faces[-1]>height[-1]: raise ValueError('CALMET cell face is above top sounding level')
    # Layer integration is a reduction, so accumulate the piecewise-linear
    # trapezoids in FP64 and cast only the resulting model profile to FP32.
    z=np.asarray(height,dtype=np.float64)
    profile_u=np.asarray(u,dtype=np.float64)
    profile_v=np.asarray(v,dtype=np.float64)
    boundaries=np.asarray(faces,dtype=np.float64)
    output_u=np.empty(faces.size-1,dtype=COMPUTE_DTYPE)
    output_v=np.empty_like(output_u)
    for layer,(bottom,top) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        interior=(z>bottom)&(z<top)
        layer_z=np.concatenate(([bottom],z[interior],[top]))
        layer_u=np.concatenate((
            [np.interp(bottom,z,profile_u)],profile_u[interior],
            [np.interp(top,z,profile_u)]))
        layer_v=np.concatenate((
            [np.interp(bottom,z,profile_v)],profile_v[interior],
            [np.interp(top,z,profile_v)]))
        dz=np.diff(layer_z)
        output_u[layer]=np.sum(.5*(layer_u[:-1]+layer_u[1:])*dz,
                               dtype=np.float64)/(top-bottom)
        output_v[layer]=np.sum(.5*(layer_v[:-1]+layer_v[1:])*dz,
                               dtype=np.float64)/(top-bottom)
    return output_u,output_v


def temperature_faces(sounding,z_faces_m):
    """Interpolate sounding temperatures at CALMET layer faces.

    Args:
        sounding: :class:`Sounding` containing height and temperature profiles.
        z_faces_m: Strictly increasing face heights in metres above the first
            sounding level, beginning at zero.

    Returns:
        Float32 array of shape ``(face,)`` containing temperatures in degrees
        Celsius at the requested faces.

    Raises:
        ValueError: If faces are invalid, a face lies above the top valid
            temperature, or a valid bracketing value is unavailable.
    """
    faces=np.asarray(z_faces_m,dtype=COMPUTE_DTYPE)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('upper-air temperature faces must increase strictly from zero')
    raw_height=np.asarray(sounding.height_msl,dtype=COMPUTE_DTYPE)
    height=raw_height-raw_height[0]
    temperature=np.asarray(sounding.temperature_c,dtype=COMPUTE_DTYPE)
    output=np.empty(faces.size,dtype=COMPUTE_DTYPE)
    for index,face in enumerate(faces):
        upper=next((j for j in range(1,height.size)
                    if height[j]>=face and temperature[j]<999.89),None)
        if upper is None: raise ValueError('temperature interpolation face is above top valid sounding level')
        lower=next((j for j in range(upper-1,-1,-1) if temperature[j]<999.89),None)
        if lower is None: raise ValueError('bottom of upper-air temperature sounding is missing')
        if height[lower]>face: raise ValueError('sounding below temperature interpolation face is missing')
        depth=height[upper]-height[lower]
        if abs(depth)<1.e-5:
            output[index]=.5*(temperature[upper]+temperature[lower])
        else:
            ratio=(height[upper]-face)/depth
            output[index]=temperature[upper]-(temperature[upper]-temperature[lower])*ratio
    return output


def grid_temperature_faces(station_faces,station_xy,grid,power=1,block_size=65536):
    """Horizontally interpolate TEMP3D face temperatures from stations.

    Args:
        station_faces: Float-like array of shape ``(station, face)`` containing
            temperature in degrees Celsius.
        station_xy: Float-like array of shape ``(station, 2)`` containing
            projected coordinates in kilometres.
        grid: Target CALMET grid configuration.
        power: Inverse-distance exponent, either 1 or 2. Distances are measured
            in grid-cell units and floored at one cell.
        block_size: Maximum grid cells processed in one interpolation block.

    Returns:
        Float32 array of shape ``(face, grid.ny, grid.nx)`` containing
        temperatures in degrees Celsius.

    Raises:
        ValueError: If station arrays do not match or ``power`` is not 1 or 2.
    """
    values=np.asarray(station_faces,dtype=COMPUTE_DTYPE)
    points=np.asarray(station_xy,dtype=SPATIAL_DTYPE)
    if values.ndim!=2 or points.shape!=(values.shape[0],2) or values.shape[0]<1:
        raise ValueError('upper-air temperature stations and face profiles do not match')
    if power not in {1,2}: raise ValueError('upper-air temperature power must be 1 or 2')
    tx=float(grid.x_origin_km)+(np.arange(grid.nx,dtype=SPATIAL_DTYPE)+.5)*float(grid.dx_km)
    ty=float(grid.y_origin_km)+(np.arange(grid.ny,dtype=SPATIAL_DTYPE)+.5)*float(grid.dy_km)
    xx,yy=np.meshgrid(tx,ty)
    targets=np.column_stack((xx.ravel(),yy.ravel()))
    output=np.empty((values.shape[1],targets.shape[0]),dtype=COMPUTE_DTYPE)
    scale=np.array([grid.dx_km,grid.dy_km],dtype=SPATIAL_DTYPE)
    reduction_values=values.astype(np.float64,copy=False)
    for begin in range(0,targets.shape[0],block_size):
        end=min(begin+block_size,targets.shape[0])
        delta=(targets[begin:end,None,:]-points[None,:,:])/scale
        metric=np.sum(delta*delta,axis=2)
        if power==1: metric=np.sqrt(metric)
        weights=1./np.maximum(metric,1.)
        denominator=np.sum(weights,dtype=np.float64,axis=1)
        # Station accumulation remains FP64 to avoid losing small weights;
        # the returned temperature field is FP32.
        result=reduction_values.T@weights.T
        result/=denominator[None,:]
        output[:,begin:end]=result.astype(COMPUTE_DTYPE)
    return output.reshape((values.shape[1],grid.ny,grid.nx))


def _prepdi_time_interpolate(left,right,seconds_to_right,total_seconds):
    """Linearly interpolate fields between two sounding timestamps.

    Args:
        left: Value or array at the earlier sounding time.
        right: Value or array at the later sounding time.
        seconds_to_right: Seconds from the requested time to ``right``.
        total_seconds: Seconds from ``left`` to ``right``.

    Returns:
        Float32 array broadcast from ``left`` and ``right`` in their original
        physical units.

    Raises:
        ValueError: If the input arrays cannot be broadcast together.
        ZeroDivisionError: If ``total_seconds`` is zero.
    """
    first=np.asarray(left,dtype=COMPUTE_DTYPE)
    second=np.asarray(right,dtype=COMPUTE_DTYPE)
    fraction=float(seconds_to_right)/float(total_seconds)
    return np.asarray(second-(second-first)*fraction,dtype=COMPUTE_DTYPE)


class UpperAirData:
    """Read one formatted UP.DAT station file using RDHDU/RDUP semantics.

    Attributes:
        path: Source UP.DAT path.
        version: File-format version string.
        datum: Datum identifier for the station coordinates.
        timezone: Normalized time-zone label. Modern files must use UTC+0000.
        start: Declared beginning of the sounding period.
        end: Declared end of the sounding period.
        top_pressure_mb: Header top-pressure threshold in millibars.
        source_type: CALMET upper-air source-type code.
        record_format: Sounding record-format selector, 1 or 2.
        required_fields: Four booleans selecting pressure, height, temperature,
            and wind fields.
        station: :class:`UpperAirStation` described in the file or supplied by
            the control file.
        soundings: Strictly chronological tuple of :class:`Sounding` objects.
    """

    def __init__(self,path,*,station_metadata=None,datum=None):
        """Parse and validate a formatted UP.DAT station file.

        Args:
            path: Path to formatted UP.DAT.
            station_metadata: Control-file :class:`UpperAirStation` required
                when the file declares projection ``NONE``. Must be ``None``
                when the file embeds station coordinates.
            datum: Control-file datum required with projection ``NONE``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            UnicodeDecodeError: If the file is not ASCII.
            ValueError: If header fields, station metadata, time-zone rules,
                sounding rows, level counts, time ranges, or chronology are
                invalid.
        """
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        lines=self.path.read_text(encoding='ascii',errors='strict').splitlines()
        if not lines: raise ValueError('empty UP.DAT')
        first=lines[0].ljust(32)
        if first[:16].strip().upper()!='UP.DAT': raise ValueError('invalid UP.DAT dataset label')
        self.version=first[16:32].strip()
        explicit=self.version in {'2.1','2.2'}
        index=1
        try: comments=int(lines[index].split()[0]); index+=1+comments
        except (IndexError,ValueError) as error: raise ValueError('invalid UP.DAT comment header') from error
        try: projection=lines[index][:8].strip().upper(); index+=1
        except IndexError as error: raise ValueError('incomplete UP.DAT header') from error
        if projection not in {'LL','NONE'}: raise ValueError('UP.DAT projection must be NONE or LL')
        if projection=='LL':
            if station_metadata is not None:
                raise ValueError('upper-air station location is present in both control file and UP.DAT')
            datum_line=lines[index]; index+=1; self.datum=datum_line[:8].strip()
            units=lines[index][:4].strip().upper(); index+=1
            if units not in {'DEG','DEGS'}: raise ValueError('UP.DAT LL coordinates must use degree units')
        else:
            if station_metadata is None or datum is None:
                raise ValueError('UP.DAT NONE requires control-file station metadata and datum')
            self.datum=str(datum)
        if explicit:
            raw_timezone=lines[index][:8].strip(); index+=1
            try:
                timezone_offset,self.timezone=parse_timezone_label(raw_timezone)
            except (TypeError,ValueError) as error:
                raise ValueError(
                    'UP.DAT 2.1/2.2 sounding times must use UTC+0000'
                ) from error
            if timezone_offset!=0:
                raise ValueError('UP.DAT 2.1/2.2 sounding times must use UTC+0000')
        else: self.timezone='UTC+0000'
        span=lines[index].replace(',',' ').split(); index+=1
        expected=11 if explicit else 9
        if len(span)!=expected: raise ValueError(f'UP.DAT time-span header must contain {expected} values')
        if explicit:
            self.start=_instant(*span[:4]); self.end=_instant(*span[4:8]); offset=8
        else:
            self.start=_instant(span[0],span[1],span[2]); self.end=_instant(span[3],span[4],span[5]); offset=6
        self.top_pressure_mb=float(span[offset]); self.source_type=int(span[offset+1]); self.record_format=int(span[offset+2])
        if self.record_format not in {1,2}: raise ValueError('UP.DAT record format must be 1 or 2')
        flags=re.findall(r'[TtFf]',lines[index]); index+=1
        if len(flags)!=4: raise ValueError('UP.DAT data-selection flags must contain four logical values')
        self.required_fields=tuple(value.upper()=='T' for value in flags)
        if projection=='LL':
            station=lines[index].replace(',',' ').split(); index+=1
            if len(station)<5: raise ValueError('invalid UP.DAT station metadata')
            self.station=UpperAirStation(int(station[0]),station[1],_coordinate(station[2],True),
                                         _coordinate(station[3],False),float(station[4]))
        else:
            self.station=station_metadata
        soundings=[]
        while index<len(lines):
            if not lines[index].strip(): index+=1; continue
            header_fields=lines[index].replace(',',' ').split()
            # UP.DAT 2.1 and 2.2 share the 12-field record header with
            # explicit begin/end calendar times and seconds.  Some existing
            # 2.1 files use the older six-field snapshot header, so retain
            # that representation when the field count identifies it.
            explicit_record=(self.version=='2.2' or
                             (self.version=='2.1' and len(header_fields)==12))
            if explicit_record:
                values=_sounding_header22(lines[index])
                identifier=values[0]; begin=_calendar_instant(*values[1:6]); end=_calendar_instant(*values[6:11]); levels=values[11]
                # Version 2.1 always represents an instantaneous sounding at
                # its ending time, even when its two encoded times differ.
                # Version 2.2 alone assigns interval semantics to begin/end.
                timestamp=end if self.version=='2.1' else begin
            else:
                identifier,year,month,day,hour,levels=_sounding_header(lines[index])
                timestamp=_calendar_instant(year,month,day,hour)
            index+=1
            if identifier!=self.station.identifier: raise ValueError('UP.DAT sounding station ID does not match header')
            if levels<0: raise ValueError('UP.DAT sounding level count cannot be negative')
            values=[]
            while len(values)<5*levels and index<len(lines):
                values.extend(float(item) for item in lines[index].replace('/',' ').replace(',',' ').split())
                index+=1
            if len(values)!=5*levels: raise ValueError('UP.DAT sounding has an incomplete or overfull level list')
            table=np.asarray(values,dtype=COMPUTE_DTYPE).reshape(levels,5)
            if timestamp<self.start or timestamp>self.end: raise ValueError('UP.DAT sounding lies outside declared time span')
            soundings.append(Sounding(timestamp,*(table[:,column].copy() for column in range(5)),
                                      end if self.version=='2.2' and end!=begin else None))
        if not soundings: raise ValueError('UP.DAT contains no soundings')
        if any(right.timestamp<=left.timestamp for left,right in zip(soundings[:-1],soundings[1:])):
            raise ValueError('UP.DAT soundings must be strictly chronological')
        averaged=[item.end_timestamp is not None for item in soundings]
        if any(averaged) and not all(averaged): raise ValueError('UP.DAT 2.2 cannot mix snapshot and time-averaged soundings')
        if all(averaged) and any(item.end_timestamp<=item.timestamp for item in soundings):
            raise ValueError('UP.DAT averaged sounding interval must be positive')
        self.soundings=tuple(soundings)

    def at(self,timestamp):
        """Return the sounding at an exact stored timestamp.

        Args:
            timestamp: Snapshot time or averaged-interval beginning.

        Returns:
            Matching :class:`Sounding`.

        Raises:
            ValueError: If no sounding starts at ``timestamp``.
        """
        matches=[item for item in self.soundings if item.timestamp==timestamp]
        if not matches: raise ValueError(f'UP.DAT has no sounding at {timestamp!s}')
        return matches[0]

    def bracket(self,timestamp):
        """Return RDUP's two snapshot soundings bracketing a timestamp.

        Args:
            timestamp: Requested interpolation time.

        Returns:
            Tuple ``(left, right)`` of consecutive :class:`Sounding` objects.
            If ``timestamp`` exactly equals ``right.timestamp``, ``left`` is
            still the preceding sounding.

        Raises:
            ValueError: If the file has no sounding strictly before the request
                or does not extend to the requested time.
        """
        times=tuple(item.timestamp for item in self.soundings)
        right=bisect_left(times,timestamp)
        if right>=len(self.soundings): raise ValueError('UP.DAT does not extend beyond requested time')
        if times[right]==timestamp:
            if right==0: raise ValueError('UP.DAT has no earlier sounding for requested time')
            left=right-1
        else:
            left=right-1
            if left<0: raise ValueError('UP.DAT does not begin before requested time')
        return self.soundings[left],self.soundings[right]

    def layer_wind_at(self,timestamp,z_faces_m,convergence_degrees=0.):
        """VERTAV bracketing profiles and interpolate winds in time.

        Args:
            timestamp: Requested model time.
            z_faces_m: Strictly increasing model-layer faces in metres,
                beginning at zero.
            convergence_degrees: Scalar or source-level map convergence in
                degrees.

        Returns:
            Tuple of float32 U and V arrays with shape ``(model_level,)`` and
            units of metres per second. Version 2.2 averaged soundings are
            returned unchanged for times inside their interval.

        Raises:
            ValueError: If averaged records do not cover the time, snapshot
                soundings do not bracket it, or vertical averaging fails.
        """
        if self.soundings[0].end_timestamp is not None:
            # RDUPN2 exposes an averaged profile unchanged at both interval
            # endpoints; at a shared boundary the earlier record is BB.
            sounding=next((item for item in self.soundings
                           if item.timestamp<timestamp<=item.end_timestamp),None)
            if sounding is None: raise ValueError('UP.DAT averaged records do not cover requested time')
            return layer_average(sounding,z_faces_m,convergence_degrees)
        left,right=self.bracket(timestamp)
        total=(right.timestamp-left.timestamp).total_seconds()
        remaining=(right.timestamp-timestamp).total_seconds()
        ua,va=layer_average(left,z_faces_m,convergence_degrees)
        ub,vb=layer_average(right,z_faces_m,convergence_degrees)
        return (_prepdi_time_interpolate(ua,ub,remaining,total),
                _prepdi_time_interpolate(va,vb,remaining,total))

    def temperature_faces_at(self,timestamp,z_faces_m):
        """FACET bracketing profiles and interpolate temperatures in time.

        Args:
            timestamp: Requested model time.
            z_faces_m: Strictly increasing layer-face heights in metres,
                beginning at zero.

        Returns:
            Float32 ``(face,)`` temperature array in degrees Celsius. Version
            2.2 averaged soundings are returned unchanged within their interval.

        Raises:
            ValueError: If averaged records do not cover the time, snapshot
                soundings do not bracket it, or face interpolation fails.
        """
        if self.soundings[0].end_timestamp is not None:
            sounding=next((item for item in self.soundings
                           if item.timestamp<timestamp<=item.end_timestamp),None)
            if sounding is None: raise ValueError('UP.DAT averaged records do not cover requested time')
            return temperature_faces(sounding,z_faces_m)
        left,right=self.bracket(timestamp)
        first=temperature_faces(left,z_faces_m); second=temperature_faces(right,z_faces_m)
        total=(right.timestamp-left.timestamp).total_seconds()
        remaining=(right.timestamp-timestamp).total_seconds()
        return _prepdi_time_interpolate(first,second,remaining,total)

    def lapse_rate_at(self,timestamp,height_m,last_convective_time=None):
        """Compute the CGAMMA lapse rate with the convective-hour correction.

        Args:
            timestamp: Requested model time.
            height_m: Positive height above the surface at which to evaluate
                the temperature difference, in metres.
            last_convective_time: Optional saved time of the most recent
                convective transition. When it lies between the left sounding
                and ``timestamp``, the upper temperature follows a dry-adiabatic
                transition before relaxing toward the right sounding.

        Returns:
            Temperature lapse rate in kelvin per metre as a Python ``float``.

        Raises:
            ValueError: If ``height_m`` is not positive, soundings do not cover
                ``timestamp``, or temperatures cannot be interpolated.
        """
        height=COMPUTE_DTYPE(height_m)
        if height<=0.: raise ValueError('CGAMMA height must be positive')
        current=self.temperature_faces_at(timestamp,(0.,height))
        ordinary=(current[1]-current[0])/height
        if last_convective_time is None or self.soundings[0].end_timestamp is not None:
            return float(ordinary)
        left,right=self.bracket(timestamp)
        if not (left.timestamp<last_convective_time<timestamp<right.timestamp):
            return float(ordinary)
        first=temperature_faces(left,(0.,height)); second=temperature_faces(right,(0.,height))
        total=(right.timestamp-left.timestamp).total_seconds()
        convective_to_right=(right.timestamp-last_convective_time).total_seconds()
        convective_surface=_prepdi_time_interpolate(
            first[0],second[0],convective_to_right,total)
        convective_upper=convective_surface-DRY_ADIABATIC_LAPSE_RATE*height
        current_to_right=(right.timestamp-timestamp).total_seconds()
        upper=_prepdi_time_interpolate(
            convective_upper,second[1],current_to_right,convective_to_right)
        return float((upper-current[0])/height)
