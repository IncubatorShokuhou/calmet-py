"""Reader for CALMET.DAT winds used by the IGFMET nested-grid option."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree

from .config import GridConfig
from .legacy_io import FortranSequentialReader
from .datetime_utils import (format_timezone_label,parse_calendar,
                             parse_ordinal_code,parse_timezone_label)
from .projections import (grid_convergence,grid_crs,transform_grid,
                          unproject_geographic)


MODEL_DTYPE=np.float32


def _coded_time(code: int, seconds: int) -> datetime:
    """Decode a CALMET ordinal timestamp and label parse failures.

    Args:
        code: Integer ``YYYYJJJHH`` ordinal date/hour code.
        seconds: Additional seconds within the encoded hour.

    Returns:
        Naive decoded timestamp.

    Raises:
        ValueError: If the code and seconds do not represent a valid time.
    """
    try: return parse_ordinal_code(code,seconds)
    except ValueError as error: raise ValueError(f'invalid CALMET.DAT timestamp {code}/{seconds}') from error


def _field(payload: bytes, size: int, *, mtver=1, integer=False):
    """Decode one CALMET.DAT labeled field record.

    Args:
        payload: Raw Fortran sequential-record payload.
        size: Required number of grid or station values.
        mtver: Whether to parse the CALMET 2.1 interval header.
        integer: Whether field values are signed 32-bit integers instead of
            32-bit floats.

    Returns:
        Tuple containing label, begin date/seconds, end date/seconds, and a
        writable one-dimensional value array.

    Raises:
        UnicodeError: If the field label is not ASCII.
        ValueError: If the payload has the wrong byte length.
    """
    prefix=24 if mtver else 12
    if len(payload)!=prefix+4*size: raise ValueError('CALMET.DAT field record has wrong size')
    label=payload[:8].decode('ascii').rstrip()
    if mtver:
        begin_code,begin_seconds,end_code,end_seconds=struct.unpack('<4i',payload[8:24])
    else:
        end_code=struct.unpack_from('<i',payload,8)[0]
        begin_code=begin_seconds=end_seconds=0
    dtype='<i4' if integer else '<f4'
    values=np.frombuffer(payload,dtype=dtype,count=size,offset=prefix).copy()
    return label,begin_code,begin_seconds,end_code,end_seconds,values


def _run_control_time(year,month,day,hour,seconds=0):
    """Decode a calendar timestamp from a CALMET run-control record.

    Args:
        year: Four-digit calendar year.
        month: Calendar month.
        day: Day of month.
        hour: Hour of day.
        seconds: Additional seconds within the hour.

    Returns:
        Naive decoded timestamp.

    Raises:
        ValueError: If the fields do not represent a valid timestamp.
    """
    try: return parse_calendar(year,month,day,hour,seconds)
    except ValueError as error:
        raise ValueError('invalid CALMET.DAT run-control timestamp') from error


def _run_control_timezone(value):
    """Normalize a CALMET run-control timezone to hours west of UTC.

    Args:
        value: CALMET timezone label or numeric representation.

    Returns:
        Integer or float timezone offset in hours west of UTC.

    Raises:
        ValueError: If the timezone label is unsupported.
    """
    try: return parse_timezone_label(value)[1]
    except (TypeError,ValueError) as error:
        raise ValueError('invalid CALMET.DAT run-control timezone') from error


def _covering_record(records,when,label):
    """Find the wind interval satisfying ``begin < when <= end``.

    Args:
        records: Mapping from interval end to ``(begin, u, v)`` record.
        when: Requested timestamp.
        label: Data-source name included in validation errors.

    Returns:
        The uniquely covering wind record.

    Raises:
        ValueError: If no record covers ``when`` or intervals overlap there.
    """
    if when in records: return records[when]
    matches=[record for end,record in records.items()
             if record[0]<when<end]
    if len(matches)==1: return matches[0]
    if len(matches)>1: raise ValueError(f'{label} has overlapping wind records at {when}')
    raise ValueError(f'{label} has no wind record covering {when}')


class InitialGuessCalmetData:
    """Read declared CALMET.DAT static metadata and hourly U/V fields.

    Attributes:
        path: Source CALMET.DAT path.
        version: Declared file-format version.
        start: Beginning of the modeled time period.
        end: End of the modeled time period.
        timezone: Normalized CALMET timezone label.
        grid: Source horizontal grid.
        duration_hours: Declared run duration in hours.
        run_type: CALMET run type stored in the source header.
        iwfcod: Source diagnostic-wind formulation code.
        surface_station_count: Number of surface stations declared by the
            source file.
        upper_station_count: Number of upper-air stations.
        precipitation_station_count: Number of precipitation stations.
        sea_station_count: Number of overwater stations.
        landuse_category_count: Number of land-use categories.
        water_category_min: First water category code in the inclusive range.
        water_category_max: Last water category code in the inclusive range.
        includes_w_temperature: Whether vertical velocity and layer
            temperature fields are present.
        z_faces_m: Vertical cell-face heights in metres.
        roughness_m: Surface roughness in metres on ``(ny, nx)``.
        landuse: Integer land-use categories on ``(ny, nx)``.
        terrain_m: Terrain elevation in metres on ``(ny, nx)``.
        leaf_area_index: Leaf-area index on ``(ny, nx)``.
        nearest_surface_station: Optional integer nearest-station field on
            ``(ny, nx)``, or ``None`` when no surface stations are declared.
        records: Wind records keyed by ending timestamp.
        fields: Optionally retained complete periods keyed by ending timestamp.
    """

    def __init__(self,path,*,retain_fields=False,field_endings=None,
                 record_endings=None):
        """Read and validate one CALMET.DAT initial-guess file.

        Args:
            path: Sequential-unformatted CALMET.DAT path.
            retain_fields: Keep every field for selected periods in addition
                to the U/V records required by IGFMET.
            field_endings: Optional ending times whose complete fields are
                retained; ``None`` retains all when enabled.
            record_endings: Optional ending times whose U/V records are kept;
                ``None`` keeps all.

        Raises:
            OSError: If the file cannot be opened or read.
            UnicodeError: If fixed-width text fields are not ASCII.
            ValueError: If metadata, record sizes/order, timestamps, or wind
                level coverage are invalid.
        """
        self.path=Path(path)
        selected_field_endings=(None if field_endings is None else
                                frozenset(field_endings))
        selected_record_endings=(None if record_endings is None else
                                 frozenset(record_endings))
        with FortranSequentialReader(self.path) as reader:
            first_record=reader.read_record()
            if first_record is None: raise ValueError('empty CALMET.DAT file')
            declared=(len(first_record)==96 and
                      first_record[:16].decode('ascii').strip()=='CALMET.DAT')
            if declared:
                self.version=first_record[16:32].decode('ascii').strip()
                mtver=int(self.version=='2.1')
                comment_record=reader.read_record()
                if comment_record is None or len(comment_record)!=4:
                    raise ValueError('invalid CALMET.DAT comment count')
                comment_count=struct.unpack('<i',comment_record)[0]
                if comment_count<0: raise ValueError('negative CALMET.DAT comment count')
                for _ in range(comment_count):
                    comment=reader.read_record()
                    if comment is None or len(comment)!=132:
                        raise ValueError('invalid CALMET.DAT comment record')
                control=reader.read_record()
                expected=176 if mtver else 148
                if control is None or len(control)!=expected:
                    raise ValueError(f'invalid CALMET.DAT {self.version} run-control record')
                if mtver:
                    first=struct.unpack_from('<10i',control,0)
                    self.start=_run_control_time(*first[:5])
                    self.end=_run_control_time(*first[5:])
                    self.timezone=_run_control_timezone(
                        control[40:48].decode('ascii').strip())
                    hours,self.run_type,nx,ny,nz=struct.unpack_from('<5i',control,48)
                    dx_m,x_origin_m,y_origin_m=struct.unpack_from('<3f',control,68)
                    station_offset=80; projection_offset=116; false_offset=144
                else:
                    year,month,day,hour,hours_west,hours,self.run_type,nx,ny,nz=struct.unpack_from('<10i',control,0)
                    # Pre-2.1 headers store the first hour-ending time.
                    self.start=_run_control_time(year,month,day,hour)-timedelta(hours=1)
                    self.end=self.start+timedelta(hours=hours)
                    self.timezone=format_timezone_label(hours_west)
                    dx_m,x_origin_m,y_origin_m=struct.unpack_from('<3f',control,40)
                    station_offset=52; projection_offset=88; false_offset=116
                (self.iwfcod,self.surface_station_count,self.upper_station_count,
                 self.precipitation_station_count,self.sea_station_count,
                 self.landuse_category_count,self.water_category_min,
                 self.water_category_max,lcalgrd)=struct.unpack_from('<9i',control,station_offset)
                projection=control[projection_offset:projection_offset+8].decode('ascii').strip()
                datum=control[projection_offset+8:projection_offset+16].decode('ascii').strip()
                false_easting,false_northing=struct.unpack_from('<2f',control,false_offset)
                hemisphere=control[false_offset+8:false_offset+12].decode('ascii').strip()
                utm_zone=struct.unpack_from('<i',control,false_offset+12)[0]
                lat0,lon0,lat1,lat2=struct.unpack_from('<4f',control,false_offset+16)
            else:
                if len(first_record)!=240:
                    raise ValueError('invalid legacy CALMET.DAT title record')
                control=reader.read_record()
                if control is None or len(control)!=108:
                    raise ValueError('invalid legacy CALMET.DAT run-control record')
                self.version=control[:8].decode('ascii').strip()
                level=control[8:16].decode('ascii').strip()
                year,month,day,hour,hours_west,hours,self.run_type,nx,ny,nz=struct.unpack_from('<10i',control,16)
                self.start=_run_control_time(year,month,day,hour)-timedelta(hours=1)
                self.end=self.start+timedelta(hours=hours)
                self.timezone=format_timezone_label(hours_west)
                dx_m,x_origin_m,y_origin_m=struct.unpack_from('<3f',control,56)
                utm_zone=struct.unpack_from('<i',control,68)[0]
                (self.iwfcod,self.surface_station_count,self.upper_station_count,
                 self.precipitation_station_count,self.sea_station_count,
                 self.landuse_category_count,self.water_category_min,
                 self.water_category_max,lcalgrd)=struct.unpack_from('<9i',control,72)
                try:
                    has_projection=int(level[:6])>=980304 or float(self.version[:6])>5.1
                except ValueError as error:
                    raise ValueError('invalid legacy CALMET version or level') from error
                if has_projection:
                    projection_record=reader.read_record()
                    if projection_record is None or len(projection_record)!=32:
                        raise ValueError('invalid legacy CALMET projection record')
                    lat0,west_lon0,llconf,_=struct.unpack_from('<2fif',projection_record,0)
                    lat1,lat2,_,_=struct.unpack_from('<4f',projection_record,16)
                    lon0=-west_lon0
                    projection='LCC' if llconf else 'UTM'
                else:
                    # Before the 980304 header extension CALMET supported
                    # only UTM here; the zone is already in record 2 and the
                    # next record is ZFACE.  RDHDMET likewise assumes the
                    # historical NWS-84 datum for this format.
                    projection='UTM'
                    lat0=lon0=lat1=lat2=0.
                datum='NWS-84'; false_easting=false_northing=0.
                hemisphere='S' if utm_zone<0 else 'N'
                utm_zone=abs(utm_zone); mtver=0
            self.grid=GridConfig(nx,ny,dx_m/1000.,dx_m/1000.,projection=projection,
                datum=datum,x_origin_km=x_origin_m/1000.,y_origin_km=y_origin_m/1000.,
                false_easting_km=false_easting,false_northing_km=false_northing,
                hemisphere=hemisphere,utm_zone=utm_zone,lat0=lat0,lon0=lon0,
                lat1=lat1,lat2=lat2)
            self.duration_hours=hours
            self.includes_w_temperature=bool(lcalgrd)
            count=nx*ny

            def take(label,size,integer=False):
                """Read and validate the next labeled static field.

                Args:
                    label: Expected eight-character CALMET field label.
                    size: Required scalar value count.
                    integer: Whether values use the integer field encoding.

                Returns:
                    One-dimensional decoded value array.

                Raises:
                    ValueError: If the file ends, label differs, or record size
                        is invalid.
                """
                payload=reader.read_record()
                if payload is None: raise ValueError(f'CALMET.DAT ends before {label}')
                actual,*_,values=_field(payload,size,mtver=mtver,integer=integer)
                if actual!=label: raise ValueError(f'expected CALMET.DAT {label}, found {actual}')
                return values

            self.z_faces_m=take('ZFACE',nz+1)
            if self.surface_station_count:
                take('XSSTA',self.surface_station_count); take('YSSTA',self.surface_station_count)
            if self.upper_station_count:
                take('XUSTA',self.upper_station_count); take('YUSTA',self.upper_station_count)
            if self.precipitation_station_count>0:
                take('XPSTA',self.precipitation_station_count); take('YPSTA',self.precipitation_station_count)
            self.roughness_m=take('Z0',count).reshape(ny,nx)
            self.landuse=take('ILANDU',count,True).reshape(ny,nx)
            self.terrain_m=take('ELEV',count).reshape(ny,nx)
            self.leaf_area_index=take('XLAI',count).reshape(ny,nx)
            self.nearest_surface_station=(
                take('NEARS',count,True).reshape(ny,nx)
                if self.surface_station_count else None)

            partial={}; labels_by_period={}; fields_by_period={}
            while (payload:=reader.read_record()) is not None:
                if len(payload)<8: raise ValueError('CALMET.DAT field record is truncated')
                raw_label=payload[:8].decode('ascii').rstrip()
                station_field=(not declared and raw_label in {
                    'TEMPK','RHO','QSW','IRH','IPCODE'})
                field_size=self.surface_station_count if station_field else count
                integer=raw_label in {'IPGT','IRH','IPCODE'}
                label,begin_code,begin_seconds,end_code,end_seconds,values=_field(
                    payload,field_size,mtver=mtver,integer=integer)
                end=_coded_time(end_code,end_seconds)
                begin=_coded_time(begin_code,begin_seconds) if mtver else end-timedelta(hours=1)
                if begin>=end: raise ValueError('CALMET field interval does not advance')
                labels_by_period.setdefault((begin,end),[]).append(label)
                keep_fields=(retain_fields and
                             (selected_field_endings is None or
                              end in selected_field_endings))
                if keep_fields:
                    period_fields=fields_by_period.setdefault((begin,end),{})
                    if label in period_fields:
                        raise ValueError(f'duplicate CALMET field {label} at {end}')
                    period_fields[label]=(values.reshape(ny,nx)
                                          if field_size==count else values)
                if not (label.startswith('U-LEV') or label.startswith('V-LEV')): continue
                if (selected_record_endings is not None and
                        end not in selected_record_endings):
                    continue
                try: level=int(label[5:])-1
                except ValueError as error: raise ValueError(f'invalid CALMET wind label {label!r}') from error
                if not 0<=level<nz: raise ValueError(f'CALMET wind level is out of range: {label!r}')
                item=partial.setdefault((begin,end),[
                    np.full((nz,ny,nx),np.nan,dtype=MODEL_DTYPE),
                    np.full((nz,ny,nx),np.nan,dtype=MODEL_DTYPE)])
                component=0 if label.startswith('U-') else 1
                if np.isfinite(item[component][level]).any():
                    raise ValueError(f'duplicate CALMET wind record {label} at {end}')
                item[component][level]=values.reshape(ny,nx)
        self.records={}; self.fields={}
        for (begin,end),(u,v) in partial.items():
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError(f'incomplete CALMET wind records for {begin}--{end}')
            expected=[]
            for level in range(1,nz+1):
                expected.extend((f'U-LEV{level:3d}',f'V-LEV{level:3d}'))
                if self.includes_w_temperature: expected.append(f'WFACE{level:3d}')
            if self.run_type==1:
                if self.includes_w_temperature:
                    expected.extend(f'T-LEV{level:3d}' for level in range(1,nz+1))
                expected.extend(('IPGT','USTAR','ZI','EL','WSTAR'))
                if self.precipitation_station_count!=0: expected.append('RMM')
                expected.extend(('TEMPK','RHO','QSW','IRH'))
                if self.precipitation_station_count!=0: expected.append('IPCODE')
            actual=labels_by_period[(begin,end)]
            if actual!=expected:
                mismatch=next((index for index,(left,right) in
                    enumerate(zip(actual,expected)) if left!=right),
                    min(len(actual),len(expected)))
                found=actual[mismatch] if mismatch<len(actual) else '<EOF>'
                wanted=expected[mismatch] if mismatch<len(expected) else '<end of period>'
                raise ValueError(f'invalid CALMET record order for {begin}--{end}: '
                                 f'expected {wanted}, found {found}')
            if end in self.records: raise ValueError(f'duplicate CALMET ending time {end}')
            self.records[end]=(begin,u,v)
            if retain_fields and (selected_field_endings is None or
                                  end in selected_field_endings):
                self.fields[end]=(begin,fields_by_period[(begin,end)])
        if not self.records: raise ValueError('CALMET.DAT contains no wind records')

    @property
    def z_centers_m(self):
        """Return vertical cell-centre heights.

        Returns:
            One-dimensional ``float32`` heights in metres.
        """
        return np.asarray(.5*(self.z_faces_m[:-1]+self.z_faces_m[1:]),
                          dtype=MODEL_DTYPE)

    def at_end(self,when):
        """Return the wind record with an exact ending timestamp.

        Args:
            when: Requested interval end.

        Returns:
            Tuple ``(begin, u, v)`` with wind arrays on ``(level, y, x)``.

        Raises:
            ValueError: If the file has no record ending at ``when``.
        """
        try: return self.records[when]
        except KeyError as error: raise ValueError(f'CALMET.DAT has no wind record ending at {when}') from error

    def fields_at_end(self,when):
        """Return all retained fields for one complete output period.

        Args:
            when: Requested interval end.

        Returns:
            Tuple containing interval begin and a label-to-array mapping.

        Raises:
            ValueError: If the period was not retained or is unavailable.
        """
        try: return self.fields[when]
        except KeyError as error: raise ValueError(
            f'CALMET.DAT has no field record ending at {when}') from error

    def at_covering(self,when):
        """Return the RDMET2 period satisfying ``begin < when <= end``.

        Args:
            when: Timestamp within the requested wind interval.

        Returns:
            Tuple ``(begin, u, v)`` for the unique covering period.

        Raises:
            ValueError: If no period covers the time or periods overlap.
        """
        return _covering_record(self.records,when,'CALMET.DAT')


class InitialGuessCalmetSeries:
    """Present consecutive compatible IGFDAT files as one time-indexed source.

    Attributes:
        files: Ordered source-file readers.
        grid: Common source horizontal grid.
        z_faces_m: Common vertical cell-face heights in metres.
        roughness_m: Common surface roughness array in metres.
        timezone: Common CALMET timezone label.
        includes_w_temperature: Whether every file includes W and temperature.
        records: Combined non-overlapping wind periods by ending time.
        fields: Combined retained complete periods by ending time.
    """

    def __init__(self,paths,*,retain_fields=False,field_endings=None,
                 record_endings=None):
        """Load a compatible, non-overlapping sequence of CALMET.DAT files.

        Args:
            paths: One or more CALMET.DAT paths in any order.
            retain_fields: Keep complete fields as well as winds.
            field_endings: Optional ending times whose complete fields are
                retained.
            record_endings: Optional ending times whose wind records are kept.

        Raises:
            ValueError: If no path is supplied, files use incompatible grids,
                layers, timezones, or roughness, or their periods overlap.
        """
        files=tuple(InitialGuessCalmetData(
                        path,retain_fields=retain_fields,
                        field_endings=field_endings,
                        record_endings=record_endings)
                    for path in paths)
        if not files: raise ValueError('IGFMET requires at least one CALMET.DAT file')
        first=files[0]
        for item in files[1:]:
            if item.grid!=first.grid:
                raise ValueError('IGFDAT files use different grids')
            if not np.array_equal(item.z_faces_m,first.z_faces_m):
                raise ValueError('IGFDAT files use different vertical layers')
            if item.timezone!=first.timezone:
                raise ValueError('IGFDAT files use different time zones')
            if not np.array_equal(item.roughness_m,first.roughness_m):
                raise ValueError('IGFDAT files use different roughness fields')
        self.files=files
        self.grid=first.grid; self.z_faces_m=first.z_faces_m
        self.roughness_m=first.roughness_m; self.timezone=first.timezone
        self.includes_w_temperature=all(item.includes_w_temperature for item in files)
        self.records={}; self.fields={}
        for item in files:
            overlap=self.records.keys()&item.records.keys()
            if overlap: raise ValueError(f'IGFDAT files overlap at {min(overlap)}')
            self.records.update(item.records)
            self.fields.update(item.fields)

    @property
    def z_centers_m(self):
        """Return common vertical cell-centre heights.

        Returns:
            One-dimensional ``float32`` heights in metres.
        """
        return np.asarray(.5*(self.z_faces_m[:-1]+self.z_faces_m[1:]),
                          dtype=MODEL_DTYPE)

    def at_end(self,when):
        """Return the series wind record with an exact ending timestamp.

        Args:
            when: Requested interval end.

        Returns:
            Tuple ``(begin, u, v)`` with wind arrays on ``(level, y, x)``.

        Raises:
            ValueError: If the series has no record ending at ``when``.
        """
        try: return self.records[when]
        except KeyError as error: raise ValueError(f'IGFDAT series has no wind record ending at {when}') from error

    def fields_at_end(self,when):
        """Return retained complete fields for an exact ending timestamp.

        Args:
            when: Requested interval end.

        Returns:
            Tuple containing interval begin and a label-to-array mapping.

        Raises:
            ValueError: If the period was not retained or is unavailable.
        """
        try: return self.fields[when]
        except KeyError as error: raise ValueError(
            f'IGFDAT series has no field record ending at {when}') from error

    def at_covering(self,when):
        """Return the RDMET2 period satisfying ``begin < when <= end``.

        Args:
            when: Timestamp within the requested wind interval.

        Returns:
            Tuple ``(begin, u, v)`` for the unique covering period.

        Raises:
            ValueError: If no period covers the time or periods overlap.
        """
        return _covering_record(self.records,when,'IGFDAT series')


def _r2interpi_four(source_points,target_points,values,*,block_size=32768):
    """Interpolate from the four nearest cells with inverse-square weights.

    Args:
        source_points: Source coordinates with shape ``(source, 2)``.
        target_points: Target coordinates with shape ``(target, 2)``.
        values: Source data with shape ``(field, source)``.
        block_size: Number of targets queried from the spatial tree per block.

    Returns:
        ``float32`` array with shape ``(field, target)``.

    Raises:
        ValueError: If coordinate/value shapes are inconsistent or fewer than
            four source cells are available.
    """
    source=np.asarray(source_points,dtype=np.float64)
    targets=np.asarray(target_points,dtype=np.float64)
    fields=np.asarray(values,dtype=MODEL_DTYPE)
    if source.ndim!=2 or source.shape[1]!=2 or targets.ndim!=2 or targets.shape[1]!=2:
        raise ValueError('R2INTERPI coordinates must be (point,2) arrays')
    if fields.ndim!=2 or fields.shape[1]!=source.shape[0]:
        raise ValueError('R2INTERPI values must be (field,source point)')
    if source.shape[0]<4: raise ValueError('R2INTERPI requires at least four source cells')
    output=np.empty((fields.shape[0],targets.shape[0]),dtype=MODEL_DTYPE)
    tree=cKDTree(source)
    for start in range(0,targets.shape[0],block_size):
        stop=min(start+block_size,targets.shape[0])
        distances,nearest=tree.query(targets[start:stop],k=4)
        inverse=1./np.maximum(distances**2,.001)
        selected=fields[:,np.asarray(nearest,dtype=np.intp)]
        numerator=np.sum(selected*inverse[None,:,:],axis=-1,dtype=np.float64)
        denominator=np.sum(inverse,axis=-1,dtype=np.float64)
        output[:,start:stop]=np.asarray(
            numerator/denominator[None,:],dtype=MODEL_DTYPE)
    return output


def _rdcalmet_vertical(values,source_centers_m,target_centers_m,roughness_m):
    """Apply RDCALMET vertical persistence and interpolation.

    Below the first source level the routine applies logarithmic wind-profile
    scaling with local roughness. Above the top source level it persists the
    top value; interior targets use linear height interpolation.

    Args:
        values: Source field with shape ``(source_level, y, x)``.
        source_centers_m: Strictly increasing source heights in metres.
        target_centers_m: Positive target heights in metres.
        roughness_m: Surface roughness array in metres with shape ``(y, x)``.

    Returns:
        ``float32`` array with shape ``(target_level, y, x)``.

    Raises:
        ValueError: If shapes differ, heights are invalid, or roughness cannot
            support logarithmic extrapolation.
    """
    field=np.asarray(values,dtype=MODEL_DTYPE)
    source=np.asarray(source_centers_m,dtype=MODEL_DTYPE)
    target=np.asarray(target_centers_m,dtype=MODEL_DTYPE)
    roughness=np.asarray(roughness_m,dtype=MODEL_DTYPE)
    if field.ndim!=3 or field.shape[0]!=source.size:
        raise ValueError('RDCALMET values must be (source level,y,x)')
    if roughness.shape!=field.shape[1:]:
        raise ValueError('RDCALMET roughness must match the horizontal grid')
    if target.ndim!=1 or np.any(target<=MODEL_DTYPE(0.)):
        raise ValueError('IGFMET target layer centers must be a positive vector')
    if source.ndim!=1 or np.any(np.diff(source)<=MODEL_DTYPE(0.)):
        raise ValueError('IGFMET source levels are not increasing')
    if np.any(roughness<=MODEL_DTYPE(0.)) or np.any(roughness==source[0]):
        raise ValueError('IGFMET roughness is invalid for logarithmic extrapolation')
    result=np.empty((target.size,*field.shape[1:]),dtype=MODEL_DTYPE)
    below=target<source[0]
    if np.any(below):
        log_z0=np.log(roughness)
        log_source=np.log(source[0])
        for index in np.flatnonzero(below):
            log_target=np.log(target[index])
            ratio=(log_target-log_z0)/(log_source-log_z0)
            result[index]=ratio*field[0]
    above=target>=source[-1]
    result[above]=field[-1]
    middle=~(below|above)
    for index in np.flatnonzero(middle):
        upper=int(np.searchsorted(source,target[index],side='right'))
        lower=upper-1
        ratio=(target[index]-source[lower])/(source[upper]-source[lower])
        result[index]=field[lower]+(field[upper]-field[lower])*ratio
    return result


def _rdcalmet_rotate(u,v,direction_adjustment_degrees):
    """Rotate horizontal winds by a grid-direction adjustment.

    Args:
        u: Eastward wind component in metres per second.
        v: Northward wind component in metres per second.
        direction_adjustment_degrees: Broadcast-compatible clockwise direction
            adjustment in degrees.

    Returns:
        Tuple of rotated eastward and northward ``float32`` components with the
        common broadcast shape.
    """
    east,north,adjustment=np.broadcast_arrays(
        np.asarray(u,dtype=MODEL_DTYPE),np.asarray(v,dtype=MODEL_DTYPE),
        np.asarray(direction_adjustment_degrees,dtype=MODEL_DTYPE))
    speed=np.hypot(east,north)
    active=speed>1.e-9
    direction=np.mod(
        270.-np.rad2deg(np.arctan2(north,east))+adjustment,360.)
    radians=np.deg2rad(direction)
    rotated_u=np.where(active,-speed*np.sin(radians),east).astype(MODEL_DTYPE)
    rotated_v=np.where(active,-speed*np.cos(radians),north).astype(MODEL_DTYPE)
    return rotated_u,rotated_v


def interpolate_initial_guess(data, when: datetime,
                              target_grid: GridConfig, target_centers_m):
    """Apply RDCALMET rotation, vertical interpolation, and R2INTERPI.

    Args:
        data: Single CALMET.DAT reader or compatible series exposing grid,
            layers, roughness, and wind lookup methods.
        when: Target timestamp covered by the requested source period.
        target_grid: CALMET grid receiving the nested initial guess.
        target_centers_m: Target layer-centre heights in metres.

    Returns:
        Tuple of eastward and northward ``float32`` winds in metres per second,
        each with shape ``(target_level, ny, nx)``.

    Raises:
        ValueError: If the time is unavailable or source geometry cannot
            support vertical or four-neighbour interpolation.
    """
    lookup=getattr(data,'at_covering',data.at_end)
    _,source_u,source_v=lookup(when)
    source_grid=data.grid
    r=np.float64
    sx=np.asarray(r(source_grid.x_origin_km)+
                  (np.arange(source_grid.nx,dtype=np.float64)+r(.5))*r(source_grid.dx_km),
                  dtype=np.float64)
    sy=np.asarray(r(source_grid.y_origin_km)+
                  (np.arange(source_grid.ny,dtype=np.float64)+r(.5))*r(source_grid.dy_km),
                  dtype=np.float64)
    source_x,source_y=np.meshgrid(sx,sy)
    source_crs=grid_crs(source_grid,units='km')
    target_crs=grid_crs(target_grid,units='km')
    if source_crs==target_crs:
        transformed_x,transformed_y=source_x,source_y
    else:
        transformed_x,transformed_y=(np.asarray(value,dtype=np.float64) for value in
            transform_grid(source_x,source_y,source_grid,target_grid,units='km'))

    if source_crs!=target_crs and ({source_grid.projection.upper(),
                                    target_grid.projection.upper()}&{'LCC','PS'}):
        source_longitude,source_latitude=unproject_geographic(
            source_x,source_y,source_grid,source_grid.datum,units='km')
        target_longitude,target_latitude=unproject_geographic(
            transformed_x,transformed_y,target_grid,target_grid.datum,units='km')
        source_angle=grid_convergence(source_grid,source_longitude,source_latitude)
        target_angle=grid_convergence(target_grid,target_longitude,target_latitude)
        rotated_u,rotated_v=_rdcalmet_rotate(
            source_u,source_v,source_angle-target_angle)
    else:
        rotated_u=source_u; rotated_v=source_v

    source_levels=np.asarray(data.z_centers_m,dtype=MODEL_DTYPE)
    target_levels=np.asarray(target_centers_m,dtype=MODEL_DTYPE)
    vertical_u=_rdcalmet_vertical(rotated_u,source_levels,target_levels,
                                  data.roughness_m)
    vertical_v=_rdcalmet_vertical(rotated_v,source_levels,target_levels,
                                  data.roughness_m)

    source_points=np.column_stack((transformed_x.T.ravel(),transformed_y.T.ravel()))
    tx=np.asarray(r(target_grid.x_origin_km)+
                  (np.arange(target_grid.nx,dtype=np.float64)+r(.5))*r(target_grid.dx_km),
                  dtype=np.float64)
    ty=np.asarray(r(target_grid.y_origin_km)+
                  (np.arange(target_grid.ny,dtype=np.float64)+r(.5))*r(target_grid.dy_km),
                  dtype=np.float64)
    target_x,target_y=np.meshgrid(tx,ty)
    targets=np.column_stack((target_x.ravel(),target_y.ravel()))
    source_u_flat=vertical_u.swapaxes(1,2).reshape(target_levels.size,-1)
    source_v_flat=vertical_v.swapaxes(1,2).reshape(target_levels.size,-1)
    output_u=_r2interpi_four(source_points,targets,source_u_flat)
    output_v=_r2interpi_four(source_points,targets,source_v_flat)
    shape=(target_levels.size,target_grid.ny,target_grid.nx)
    return output_u.reshape(shape),output_v.reshape(shape)
