"""Vectorised MAKEGEO implementation for the standard USGS-38 scheme."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from datetime import datetime,timedelta
from pathlib import Path
import re
import tempfile
import numpy as np
from netCDF4 import Dataset
from .config import MakegeoConfig
from ._ctgproc import USGS38
from .datums import NIMA_DATABASE_DATE,normalize_datum
from .legacy_text import gfortran_real_pair

OUTCAT=np.array([10,20,-20,30,40,51,54,55,60,61,62,70,80,90],dtype=np.int16)
MAPCAT=np.array([10]*7+[20]*4+[30]*3+[40]*3+[51]*3+[54,55,61,62]+[70]*7+[80]*5+[90]*2,dtype=np.int16)
Z0=np.array([.5]+[1.]*6+[.25]*4+[.05]*3+[1.]*3+[.001]*5+[1.,.2]+[.05]*7+[.2]*5+[.05]*2)
ALBEDO=np.array([.18]*7+[.15]*4+[.25]*3+[.1]*3+[.1]*5+[.1,.1]+[.3]*7+[.3]*5+[.7]*2)
BOWEN=np.array([1.]+[1.5]*6+[1.]*4+[1.]*3+[1.]*3+[0.]*5+[.5,.1]+[1.]*7+[.5]*5+[.5]*2)
SOIL=np.array([.2]+[.25]*6+[.15]*4+[.15]*3+[.15]*3+[1.]*5+[.25,.25]+[.15]*7+[.15]*5+[.15]*2)
LAI=np.array([1.]+[.2]*6+[3.]*4+[.5]*3+[7.]*3+[0.]*5+[2.,1.]+[.05]*7+[0.]*5+[0.]*2)
STANDARD_PROPERTIES={int(category):(float(Z0[i]),float(ALBEDO[i]),float(BOWEN[i]),
    float(SOIL[i]),0.,float(LAI[i]),int(MAPCAT[i])) for i,category in enumerate(USGS38)}


def _category_sum(values, mask=None):
    """Sum categories in MAKEGEO's K order while vectorizing grid cells."""
    data=np.asarray(values,dtype=np.float32)
    selected=(np.ones(data.shape[0],dtype=bool) if mask is None else
              np.asarray(mask,dtype=bool))
    total=np.zeros(data.shape[1:],dtype=np.float32)
    for category in np.flatnonzero(selected):
        total=np.asarray(total+data[category],dtype=np.float32)
    return total


def _water_sum(fractions, categories, water_categories):
    """Accumulate water fractions in MAKEGEO's IWAT control-file order."""
    data=np.asarray(fractions,dtype=np.float32)
    categories=np.asarray(categories)
    total=np.zeros(data.shape[1:],dtype=np.float32)
    for category in water_categories:
        found=np.flatnonzero(categories==int(category))
        if found.size:
            total=np.asarray(total+data[int(found[0])],dtype=np.float32)
    return total


def _fortran_log(values):
    """GNU default-REAL ALOG rounding for vector inputs."""
    data=np.asarray(values,dtype=np.float32)
    return np.asarray(np.log(data.astype(np.float64)),dtype=np.float32)


def _fortran_exp(values):
    """GNU default-REAL EXP rounding for vector inputs."""
    data=np.asarray(values,dtype=np.float32)
    return np.asarray(np.exp(data.astype(np.float64)),dtype=np.float32)


def _snow_albedo_values(normal, snowy, reduction, maximum_age, age):
    """Evaluate WTSWALB's snow-age branch at default-REAL boundaries."""
    f32=np.float32
    normal=np.asarray(normal,dtype=f32); snowy=np.asarray(snowy,dtype=f32)
    reduction=np.asarray(reduction,dtype=f32)
    maximum_age=np.asarray(maximum_age,dtype=np.int32)
    age=np.asarray(age,dtype=np.int64)
    difference=np.asarray(snowy-normal,dtype=f32)
    decay=np.asarray(difference*reduction,dtype=f32)
    clipped=np.minimum(age,maximum_age)
    ratio=np.asarray(clipped.astype(f32)/maximum_age.astype(f32),dtype=f32)
    multiplier=np.asarray(f32(1.)-np.asarray(decay*ratio,dtype=f32),dtype=f32)
    return np.asarray(snowy*multiplier,dtype=f32)

def _legacy_round(values,digits):
    """Positive-field decimal rounding used by formatted Fortran output."""
    scale=10.**digits; data=np.asarray(values,dtype=float)
    return np.copysign(np.floor(np.abs(data)*scale+.500001)/scale,data)

@dataclass(frozen=True,slots=True)
class GeophysicalResult:
    output: Path
    populated_cells: int
    terrain_minimum_m: float
    terrain_maximum_m: float
    outputs: tuple[Path,...] = ()
    qa_cell: tuple[int,int] | None = None
    qa_values: tuple[float,...] | None = None


def _qa_cell_values(terrain, fields, config):
    """Return MAKEGEO's list-file QA fields using one-based IXQA/IYQA."""
    if config.qa_cell is None:
        return None
    ix,iy=config.qa_cell
    if not 1<=ix<=config.grid.nx or not 1<=iy<=config.grid.ny:
        return None
    return tuple(float(field[iy-1,ix-1]) for field in (fields[0],terrain,*fields[1:]))


@dataclass(frozen=True,slots=True)
class SnowReference:
    """Sparse CTGPROC-to-SNODAS cross-reference used by MAKEGEO."""

    cell_index: np.ndarray
    category_index: np.ndarray
    snow_grid_index: np.ndarray
    fraction: np.ndarray


@dataclass(frozen=True,slots=True)
class SnowDataHeader:
    dataset: str
    version: str
    modification: str
    projection: str
    datum: str
    date_system: str
    units: str
    longitude_origin: float
    latitude_origin: float
    nx: int
    ny: int
    dx: float
    dy: float
    timezone: str
    datetime_type: str
    raw_longitude_origin: float
    raw_latitude_origin: float
    raw_nx: int
    raw_ny: int
    raw_dx: float
    raw_dy: float
    i_begin: int
    i_end: int
    j_begin: int
    j_end: int
    variables_per_day: int


@dataclass(frozen=True,slots=True)
class SnowVariable:
    timestamp: datetime
    name: str
    values: np.ndarray


def _expand_packed_snow(packed,minimum,missing,scale,size):
    """Expand RDSN's signed-int16 run stream and restore physical values."""
    packed=np.asarray(packed,dtype=np.int64).ravel()
    if scale==0.:
        raise ValueError('packed snow scale must be non-zero')
    output=np.empty(int(size),dtype=np.int64)
    source=target=0
    while source<packed.size:
        value=int(packed[source]); source+=1
        if value>=0 or value==missing:
            count=1
        else:
            count=-value
            if source>=packed.size:
                raise ValueError('truncated packed snow run')
            value=int(packed[source]); source+=1
        if target+count>output.size:
            raise ValueError('packed snow stream expands beyond its grid')
        output[target:target+count]=value
        target+=count
    if target!=output.size:
        raise ValueError('packed snow stream does not fill its grid')
    f32=np.float32
    result=np.asarray(np.asarray(output,dtype=f32)+f32(minimum),dtype=f32)
    result=np.asarray(result/f32(scale),dtype=f32)
    result[output==missing]=np.nan
    return result


def _read_formatted_snow_header(stream):
    first=stream.readline()
    if len(first)<32:
        raise ValueError('SNOW.DAT header record 1 is truncated')
    dataset=first[:16].strip(); version=first[16:32].strip(); modification=first[32:96].strip()
    try:
        comment_count=int(stream.readline().strip())
    except ValueError as error:
        raise ValueError('SNOW.DAT comment count is invalid') from error
    if not 0<=comment_count<=100:
        raise ValueError('SNOW.DAT comment count must be 0..100')
    for _ in range(comment_count):
        if not stream.readline():
            raise ValueError('SNOW.DAT comments are truncated')
    projection=stream.readline()[:8].strip()
    datum_line=stream.readline(); datum=datum_line[:8].strip(); date_system=datum_line[8:18].strip()
    units=stream.readline()[:4].strip()
    try:
        grid=stream.readline().split()
        lon,lat=float(grid[0]),float(grid[1]); nx,ny=int(grid[2]),int(grid[3])
        dx,dy=float(grid[4]),float(grid[5])
        timezone=stream.readline()[:8].strip()
        datetime_type=stream.readline()[:16].strip()
        raw=stream.readline().split()
        raw_lon,raw_lat=float(raw[0]),float(raw[1]); raw_nx,raw_ny=int(raw[2]),int(raw[3])
        raw_dx,raw_dy=float(raw[4]),float(raw[5])
        window=[int(value) for value in stream.readline().split()]
    except (IndexError,ValueError) as error:
        raise ValueError('SNOW.DAT grid header is invalid') from error
    if len(window)!=5 or nx<1 or ny<1 or raw_nx<1 or raw_ny<1:
        raise ValueError('SNOW.DAT grid dimensions/window are invalid')
    return SnowDataHeader(dataset,version,modification,projection,datum,date_system,
        units,lon,lat,nx,ny,dx,dy,timezone,datetime_type,raw_lon,raw_lat,
        raw_nx,raw_ny,raw_dx,raw_dy,*window)


def _read_formatted_snow_variable(stream,header):
    line=stream.readline()
    if not line:
        return None
    line=line.rstrip('\n').ljust(86)
    try:
        year=int(line[0:4]); month=int(line[4:7]); day=int(line[7:10]); hour=int(line[10:13])
        name=line[38:46].strip()
        minimum=int(line[47:54]); maximum=int(line[54:61]); missing=int(line[61:68])
        scale=float(line[68:76]); packed_count=int(line[76:86])
    except ValueError as error:
        raise ValueError('SNOW.DAT variable header is invalid') from error
    if maximum-minimum>32767 or packed_count<0:
        raise ValueError('SNOW.DAT packed range/count is invalid')
    packed=[]
    while len(packed)<packed_count:
        data=stream.readline()
        if not data:
            raise ValueError('SNOW.DAT packed data are truncated')
        try:
            packed.extend(int(data[index:index+7]) for index in range(0,len(data.rstrip('\n')),7)
                          if data[index:index+7].strip())
        except ValueError as error:
            raise ValueError('SNOW.DAT packed integer is invalid') from error
    if len(packed)!=packed_count:
        raise ValueError('SNOW.DAT packed record contains extra integers')
    values=_expand_packed_snow(packed,minimum,missing,scale,header.nx*header.ny)
    return SnowVariable(datetime(year,month,day,hour),name,
                        values.reshape(header.ny,header.nx))


def _fill_snow_spatial(values,radius=2):
    """Vectorized FILLS square-window search using the original snapshot."""
    source=np.asarray(values,dtype=np.float32)
    if source.ndim!=2 or radius<0:
        raise ValueError('spatial snow fill requires a 2-D field and non-negative radius')
    result=source.copy(); missing=np.isnan(source)
    valid=np.isfinite(source)
    ny,nx=source.shape
    for distance in range(1,radius+1):
        sums=np.zeros(source.shape,dtype=np.float32)
        counts=np.zeros(source.shape,dtype=np.int32)
        # FILLS scans JJ outside II.  Each offset still updates the whole
        # target grid at once; only the order-dependent 5x5 window is looped.
        for dy in range(-distance,distance+1):
            target_y=slice(max(0,-dy),min(ny,ny-dy))
            source_y=slice(max(0,dy),min(ny,ny+dy))
            for dx in range(-distance,distance+1):
                target_x=slice(max(0,-dx),min(nx,nx-dx))
                source_x=slice(max(0,dx),min(nx,nx+dx))
                good=valid[source_y,source_x]
                region=sums[target_y,target_x]
                region[:]=np.asarray(region+np.where(
                    good,source[source_y,source_x],np.float32(0.)),dtype=np.float32)
                counts[target_y,target_x]+=good
        use=missing&np.isnan(result)&(counts>=1)
        averaged=np.divide(sums,counts.astype(np.float32),
                           out=np.zeros_like(sums),where=counts>0)
        result[use]=averaged[use]
    return result


def _fill_snow_temporal(values,current=1):
    """Vectorized FILLT: use current day, else mean all available window days."""
    window=np.asarray(values,dtype=np.float32)
    if window.ndim!=3 or not 0<=current<window.shape[0]:
        raise ValueError('temporal snow fill requires (day,y,x) and a valid current index')
    result=window[current].copy()
    missing=np.isnan(result)
    valid=np.isfinite(window)
    counts=np.zeros(result.shape,dtype=np.int32)
    totals=np.zeros(result.shape,dtype=np.float32)
    for day in range(window.shape[0]):
        totals=np.asarray(totals+np.where(
            valid[day],window[day],np.float32(0.)),dtype=np.float32)
        counts+=valid[day]
    replacement=np.divide(totals,counts.astype(np.float32),
                          out=np.full(result.shape,np.nan,dtype=np.float32),where=counts>0)
    result[missing]=replacement[missing]
    return result


def _snow_control_date(value):
    try:
        return datetime.strptime(str(int(value)),'%Y%m%d')
    except (TypeError,ValueError) as error:
        raise ValueError(f'invalid MAKEGEO snow date: {value!r}') from error


def _iter_formatted_snow_days(path,start_date,end_date):
    """Stream RDSN's rolling three-day window as daily depth/age fields."""
    start=_snow_control_date(start_date); end=_snow_control_date(end_date)
    if start>end:
        raise ValueError('snow_start_date/IDATEBEG must not follow snow_end_date/IDATEEND')
    cutoff=start-timedelta(days=2)
    ages=None; window=deque(maxlen=3); yielded=[]
    with Path(path).open('r',encoding='ascii',errors='strict') as stream:
        header=_read_formatted_snow_header(stream)
        if header.raw_nx!=6935 or header.raw_ny!=3351:
            raise ValueError('SNOW.DAT raw grid dimensions do not match SNODAS')
        if (abs(header.raw_longitude_origin+124.729584)>1.e-3 or
                abs(header.raw_latitude_origin-52.8704185)>1.e-3 or
                abs(header.raw_dx-8.33333377e-3)>1.e-5 or
                abs(header.raw_dy+8.33333377e-3)>1.e-5):
            raise ValueError('SNOW.DAT raw grid geometry does not match SNODAS')
        ages=np.full((header.ny,header.nx),14,dtype=np.int32)
        while True:
            records=[]
            for _ in range(header.variables_per_day):
                record=_read_formatted_snow_variable(stream,header)
                if record is None:
                    if records:
                        raise ValueError('SNOW.DAT ends inside a daily variable group')
                    break
                records.append(record)
            if not records:
                break
            timestamps={record.timestamp for record in records}
            if len(timestamps)!=1:
                raise ValueError('SNOW.DAT variables in one group have different timestamps')
            timestamp=records[0].timestamp.replace(hour=0,minute=0,second=0,microsecond=0)
            if timestamp<cutoff:
                continue
            variables={record.name:record.values for record in records}
            if 'SNOWDEP' not in variables or 'PRECIPS' not in variables:
                raise ValueError('SNOW.DAT day lacks SNOWDEP or PRECIPS')
            precipitation=variables['PRECIPS']
            ages=np.where(precipitation>=.1,0,ages+1).astype(np.int32)
            depth=_fill_snow_spatial(variables['SNOWDEP'],radius=2)
            if window and timestamp-window[-1][0]!=timedelta(days=1):
                raise ValueError('SNOW.DAT daily timestamps are not consecutive')
            window.append((timestamp,depth,ages.copy()))
            if len(window)<3:
                continue
            current=window[1][0]
            if start<=current<=end:
                depth_current=_fill_snow_temporal(
                    np.stack([item[1] for item in window]),current=1)
                yielded.append(current)
                # RDSN has already processed the next day's PRECIPS when it
                # returns the middle day's snow depth; NDSNOW is not rolled.
                yield current,depth_current,window[-1][2],header
            if current>=end:
                break
    expected=(end-start).days+1
    if len(yielded)!=expected:
        raise ValueError(f'SNOW.DAT supplies {len(yielded)} of {expected} requested days')


def _read_snow_reference(path,g,categories):
    """Read CTGPROC's SNOWREF arrays without materialising legacy 4-D arrays."""
    source=Path(path)
    if source.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(source) as dataset:
            required=('snow_cell_index','snow_landuse_index','snow_grid_index',
                      'snow_fraction')
            missing=[name for name in required if name not in dataset.variables]
            if missing:
                raise ValueError(f'land-use file has no snow-grid reference: {missing}')
            cell=np.asarray(dataset['snow_cell_index'][:],dtype=np.int64)-1
            land=np.asarray(dataset['snow_landuse_index'][:],dtype=np.int64)-1
            snow=np.asarray(dataset['snow_grid_index'][:],dtype=np.int64)
            fraction=np.asarray(dataset['snow_fraction'][:],dtype=float)
    else:
        lines=source.read_text(encoding='ascii',errors='strict').splitlines()
        try:
            marker=lines.index('SNOW Grid Information:')
        except ValueError as error:
            raise ValueError('land-use file has no SNOW Grid Information section') from error
        records=lines[marker+1:]
        if len(records)!=g.nx*g.ny:
            raise ValueError('snow-grid reference does not contain every target cell')
        cells=[]; lands=[]; snow_grids=[]; fractions=[]
        seen=set()
        for expected,line in enumerate(records):
            fields=line.split()
            try:
                i,j,type_count=(int(value) for value in fields[:3])
            except (ValueError,TypeError) as error:
                raise ValueError(f'invalid snow-grid reference record {expected+1}') from error
            if (i,j)!=(expected%g.nx+1,expected//g.nx+1):
                raise ValueError('snow-grid reference cell order does not match the target grid')
            if not 0<=type_count<=10:
                raise ValueError('snow-grid reference has more than 10 land-use types in a cell')
            position=3
            for _ in range(type_count):
                if position+2>len(fields):
                    raise ValueError('truncated snow-grid land-use reference')
                land=int(fields[position]); count=int(fields[position+1]); position+=2
                if not 1<=land<=len(categories):
                    raise ValueError('snow-grid land-use index is outside the category table')
                if not 0<=count<=10:
                    raise ValueError('snow-grid reference has more than 10 snow grids per land-use type')
                if (expected,land) in seen:
                    raise ValueError('duplicate land-use type in snow-grid target cell')
                seen.add((expected,land))
                for _ in range(count):
                    if position+2>len(fields):
                        raise ValueError('truncated snow-grid index/fraction pair')
                    cells.append(expected); lands.append(land-1)
                    snow_grids.append(int(fields[position])); fractions.append(float(fields[position+1]))
                    position+=2
            if position!=len(fields):
                raise ValueError('extra fields in snow-grid reference record')
        cell=np.asarray(cells,dtype=np.int64)
        land=np.asarray(lands,dtype=np.int64)
        snow=np.asarray(snow_grids,dtype=np.int64)
        fraction=np.asarray(fractions,dtype=float)
    size=g.nx*g.ny
    if not (cell.shape==land.shape==snow.shape==fraction.shape):
        raise ValueError('snow-grid reference arrays have inconsistent lengths')
    if np.any((cell<0)|(cell>=size)) or np.any((land<0)|(land>=len(categories))):
        raise ValueError('snow-grid reference index is outside its declared dimensions')
    if np.any(snow<1) or np.any(~np.isfinite(fraction)) or np.any(fraction<0.):
        raise ValueError('snow-grid reference contains invalid index or fraction values')
    return SnowReference(cell,land,snow,fraction)

def _validate_netcdf_grid(dataset,g,label):
    expected={'grid_nx':g.nx,'grid_ny':g.ny,'grid_dx_km':g.dx_km,
              'grid_dy_km':g.dy_km,'x_origin_km':g.x_origin_km,
              'y_origin_km':g.y_origin_km}
    differences=[]
    for name,wanted in expected.items():
        if name not in dataset.ncattrs(): continue
        actual=getattr(dataset,name)
        equal=(int(actual)==wanted if isinstance(wanted,int) else
               np.isclose(float(actual),wanted,rtol=1.e-4,atol=0.))
        if not equal: differences.append(name)
    if ('projection' in dataset.ncattrs() and
            str(dataset.projection).strip().upper()!=g.projection.upper()):
        differences.append('projection')
    if ('datum' in dataset.ncattrs() and
            normalize_datum(str(dataset.datum))!=normalize_datum(g.datum)):
        differences.append('datum')
    if differences:
        raise ValueError(f'{label} NetCDF grid mismatch: '+', '.join(differences))


def _legacy_angle(value):
    token=re.sub(r'\s+','',str(value))
    if not token:
        raise ValueError('empty directional angle')
    directions=re.findall(r'[NSEW]',token,re.I)
    if len(directions)>1:
        raise ValueError('invalid directional angle')
    direction=directions[0].upper() if directions else ''
    if direction: token=re.sub(r'[NSEW]','',token,flags=re.I)
    number=float(token)
    if direction in {'S','W'}: number=-abs(number)
    elif direction in {'N','E'}: number=abs(number)
    if direction in {'E','W'}:
        if number>180.: number-=360.
        elif number < -180.: number+=360.
    return number


def _legacy_grid_header(lines,index,g,label,*,landuse=False):
    """Read and validate the shared WRTHEAD metadata used by MAKEGEO.

    The legacy reader checks the full projection, datum, and grid geometry for
    each terrain and land-use input.  Return the first line following the grid
    record together with that record's fields.
    """
    try:
        projection=lines[index][:8].strip().upper(); index+=1
        parameter_line=lines[index]; index+=1
        false_easting=false_northing=0.
        if projection in {'LCC','LAZA','TTM'}:
            false_fields=lines[index].split(); index+=1
            if len(false_fields)<2:
                raise ValueError('incomplete false easting/northing')
            false_easting,false_northing=map(float,false_fields[:2])
        datum=lines[index][:8].strip().upper(); index+=1
        fields=lines[index].split(); index+=1
        required=7 if landuse else 6
        if len(fields)<required:
            raise ValueError('incomplete grid record')
        nx,ny=int(fields[0]),int(fields[1])
        geometry=tuple(float(value) for value in fields[2:6])
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid {label} grid header') from error

    differences=[]
    if projection!=g.projection.upper(): differences.append('projection')
    if normalize_datum(datum)!=normalize_datum(g.datum): differences.append('datum')
    if (nx,ny)!=(g.nx,g.ny): differences.append('dimensions')
    expected=(g.x_origin_km,g.y_origin_km,g.dx_km,g.dy_km)
    if not np.allclose(geometry,expected,rtol=1.e-4,atol=0.):
        differences.append('geometry')

    try:
        if projection=='UTM':
            match=re.fullmatch(r'\s*([+-]?\d+)\s*([NSns])\s*',parameter_line)
            if match is None:
                raise ValueError('invalid UTM parameters')
            if int(match.group(1))!=g.utm_zone or match.group(2).upper()!=g.hemisphere.upper():
                differences.append('UTM zone/hemisphere')
        else:
            count={'LCC':4,'PS':3,'EM':2,'LAZA':2,'TTM':2}.get(projection)
            if count is None:
                raise ValueError('invalid projection parameters')
            fixed=[parameter_line[offset:offset+16].strip()
                   for offset in range(0,16*count,16)]
            parameters=(fixed if len(parameter_line)>=16*count and all(fixed)
                        else parameter_line.split())
            if len(parameters)<count:
                raise ValueError('invalid projection parameters')
            actual=tuple(_legacy_angle(value) for value in parameters[:count])
            wanted=(g.lat0,g.lon0,g.lat1,g.lat2)[:count]
            if not np.allclose(actual,wanted,rtol=1.e-4,atol=0.):
                differences.append('projection parameters')
        if projection in {'LCC','LAZA','TTM'} and not np.allclose(
                (false_easting,false_northing),
                (g.false_easting_km,g.false_northing_km),rtol=1.e-4,atol=0.):
            differences.append('false easting/northing')
    except ValueError as error:
        raise ValueError(f'invalid {label} projection header') from error
    if differences:
        prefix=(f'{label} projection or datum mismatch; '
                if {'projection','datum'}&set(differences) else '')
        raise ValueError(prefix+f'{label} grid mismatch: '+', '.join(differences))
    return index,fields


def _read_terrain(path,g):
    p=Path(path)
    if p.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(p) as ds:
            _validate_netcdf_grid(ds,g,'terrain')
            terrain=np.asarray(ds['terrain_height'][:]).astype(np.float64)
        if terrain.shape!=(g.ny,g.nx):
            raise ValueError(f'terrain NetCDF array shape mismatch in {p}')
        return terrain
    lines=p.read_text(encoding='ascii').splitlines()
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid terrain header in {p}') from error
    index=2+comments
    index,_=_legacy_grid_header(lines,index,g,'terrain')
    try:
        unit_line=lines[index]; index+=1
        xy_unit=unit_line[:4].strip().upper()
        z_unit=unit_line[4:8].strip().upper()
        order=lines[index].split(); index+=1
    except (IndexError,ValueError) as error:
        raise ValueError(f'invalid terrain grid header in {p}') from error
    if xy_unit!='KM': raise ValueError(f'terrain horizontal units must be KM in {p}')
    scales={'M':1.,'KM':1000.,'FT':.3048,'YD':.9144}
    if z_unit not in scales: raise ValueError(f'unsupported terrain elevation units in {p}: {z_unit}')
    if len(order)<2 or order[0].upper()!='W_E' or order[1].upper() not in {'N_S','S_N'}:
        raise ValueError(f'unsupported terrain record order in {p}')
    values=np.fromstring(' '.join(lines[index:]),sep=' ')
    if values.size != g.nx*g.ny: raise ValueError(f"terrain size mismatch in {p}")
    terrain=values.reshape(g.ny,g.nx)*scales[z_unit]
    return terrain[::-1] if order[1].upper()=='N_S' else terrain

def _read_landuse(path,g,expected_categories=None):
    p=Path(path)
    expected=(None if expected_categories is None else
              np.asarray(tuple(expected_categories),dtype=np.int64))
    if p.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(p) as ds:
            _validate_netcdf_grid(ds,g,'land-use')
            cats=np.asarray(ds['landuse_category'][:]); frac=np.asarray(ds['landuse_fraction'][:])
        if frac.ndim!=3 or frac.shape[1:]!=(g.ny,g.nx):
            raise ValueError(f'land-use NetCDF array shape mismatch in {p}')
        if cats.size!=frac.shape[0]:
            raise ValueError(f'land-use NetCDF category dimension mismatch in {p}')
        if expected is not None and not np.array_equal(cats.astype(np.int64),expected):
            raise ValueError(f'land-use categories do not match MAKEGEO input categories in {p}')
        return cats.astype(np.int64),frac.astype(np.float64)
    lines=p.read_text(encoding='ascii').splitlines()
    try:
        version=float(lines[0][16:32].strip())
        comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error: raise ValueError(f'invalid land-use header in {p}') from error
    index=2+comments
    if lines[index].strip().upper()!='FRACTION':
        raise ValueError('MAKEGEO requires a FRACTION land-use file')
    index+=1
    index,fields=_legacy_grid_header(lines,index,g,'land-use',landuse=True)
    try: xy_unit=lines[index][:4].strip().upper(); index+=1
    except IndexError as error: raise ValueError(f'invalid land-use header in {p}') from error
    if xy_unit!='KM': raise ValueError(f'land-use horizontal units must be KM in {p}')
    category_count=int(fields[6])
    if version>2.01:
        categories=np.fromstring(lines[index],sep=' ',dtype=np.int64); index+=1
        if categories.size!=category_count:
            raise ValueError(f'land-use category header mismatch in {p}')
    else:
        if expected is None:
            raise ValueError(f'land-use version {version:g} requires MAKEGEO input categories')
        categories=expected.copy()
        if categories.size!=category_count:
            raise ValueError(f'land-use category header mismatch in {p}')
    if expected is not None and not np.array_equal(categories,expected):
        raise ValueError(f'land-use categories do not match MAKEGEO input categories in {p}')
    record_count=g.nx*g.ny
    records=lines[index:index+record_count]
    result=np.empty((categories.size,g.ny,g.nx),dtype=np.float64)
    if len(records)!=record_count: raise ValueError(f"land-use record count mismatch in {p}")
    tail=lines[index+record_count:]
    if tail and tail[0]!='SNOW Grid Information:':
        raise ValueError(f'unrecognized records after land-use grid in {p}')
    for record,line in enumerate(records):
        values=np.fromstring(line[:10+6*categories.size],sep=' ')
        if values.size!=categories.size+2:
            raise ValueError(f'invalid land-use fraction record in {p}')
        i,j=int(values[0]),int(values[1])
        expected_index=(record%g.nx+1,record//g.nx+1)
        if (i,j)!=expected_index:
            raise ValueError(f'land-use cell order does not match the target grid in {p}')
        result[:,j-1,i-1]=values[2:]
    return categories,result

def _split_categories(categories,frac,splits):
    if not splits: return categories,frac
    categories=np.asarray(categories,dtype=np.int64)
    result=np.array(frac,dtype=np.float32,copy=True)
    lookup={int(value):index for index,value in enumerate(categories)}
    grouped={}
    for source,target,percent in splits: grouped.setdefault(int(source),[]).append((int(target),float(percent)))
    for source,targets in grouped.items():
        if source not in lookup or any(target not in lookup for target,_ in targets):
            raise ValueError('split source and target categories must exist in the input category list')
        source_index=lookup[source]
        percentage_sum=np.float32(0.)
        # MAKEGEO 3.2 labels PREC as percent but uses it directly, without
        # dividing by 100. Preserve that behavior in the bit-compatible API.
        for target,percent in targets:
            fraction=np.float32(percent)
            contribution=np.asarray(fraction*result[source_index],dtype=np.float32)
            target_index=lookup[target]
            result[target_index]=np.asarray(
                result[target_index]+contribution,dtype=np.float32)
            percentage_sum=np.asarray(percentage_sum+fraction,dtype=np.float32)
        result[source_index]=np.asarray(
            result[source_index]-np.asarray(
                percentage_sum*result[source_index],dtype=np.float32),dtype=np.float32)
    return categories,result

def _fields(frac,config,categories=USGS38):
    categories,frac=_split_categories(categories,frac,config.category_splits)
    frac=np.asarray(frac,dtype=np.float32)
    properties=config.input_properties or STANDARD_PROPERTIES
    missing_properties=[int(category) for category in categories if int(category) not in properties]
    if missing_properties: raise ValueError(f'missing MAKEGEO properties for categories: {missing_properties}')
    table=np.asarray([properties[int(category)][:6] for category in categories],
                     dtype=np.float32)
    mapped=np.asarray([properties[int(category)][6] for category in categories],dtype=np.int16)
    if any(int(value) not in set(config.output_categories) for value in mapped):
        raise ValueError('mapped land-use category is absent from output_categories')
    totals=_category_sum(frac)
    missing=totals < .001
    if np.any((totals>.001)&(totals<config.minimum_coverage)):
        raise ValueError('land-use coverage lies between replacement and FLUMIN thresholds')
    if missing.any():
        found=np.flatnonzero(categories==config.missing_category)
        if not found.size: raise ValueError('missing_category is absent from input categories')
        k=int(found[0]); frac=frac.copy()
        frac[k,missing]=np.asarray(frac[k,missing]+np.asarray(
            np.float32(1.)-totals[missing],dtype=np.float32),dtype=np.float32)
        totals=_category_sum(frac)
    water_mask=np.isin(categories,config.water_categories)
    water=_water_sum(frac,categories,config.water_categories)>np.float32(
        np.float32(config.water_threshold)+np.float32(.0001))
    eligible=np.where(water[None,:,:],water_mask[:,None,None],~water_mask[:,None,None])
    selected=np.argmax(np.where(eligible,frac,-np.inf),axis=0)
    dominant=mapped[selected]
    def weighted(values,*,logarithmic=False):
        reference=(_fortran_log(values) if logarithmic else values)
        numerator=np.zeros(totals.shape,dtype=np.float32)
        denominator=np.zeros(totals.shape,dtype=np.float32)
        # WT is sequential in K but independent over the horizontal grid.
        # Preserve its REAL*4 accumulation boundary while vectorizing every
        # grid cell in each category step.
        for category in range(categories.size):
            numerator=np.asarray(
                numerator+frac[category]*reference[category],dtype=np.float32)
            denominator=np.asarray(
                denominator+frac[category],dtype=np.float32)
        result=np.asarray(numerator/denominator,dtype=np.float32)
        return _fortran_exp(result) if logarithmic else result
    roughness=weighted(table[:,0],logarithmic=True)
    fields=tuple(weighted(table[:,index]) for index in range(1,6))
    return dominant,roughness,*fields


def _snow_weighted_fields(frac,categories,reference,snow_depth,snow_age,config,
                          *,snow_i_offset=0,snow_j_offset=0,snow_full_nx=6935):
    """Vectorized WTSWZ0/WTSW/WTSWALB daily surface-property kernel."""
    categories,frac=_split_categories(categories,frac,config.category_splits)
    normal=config.input_properties or STANDARD_PROPERTIES
    snowy=config.snow_properties
    if set(int(value) for value in categories)-set(normal):
        raise ValueError('normal land-use properties do not cover the input categories')
    if set(int(value) for value in categories)-set(snowy):
        raise ValueError('snow land-use properties do not cover the input categories')
    frac=np.asarray(frac,dtype=np.float32)
    normal_table=np.asarray([normal[int(category)][:6] for category in categories],
                            dtype=np.float32)
    snow_table=np.asarray([snowy[int(category)][:6] for category in categories],
                          dtype=np.float32)
    mapped=np.asarray([normal[int(category)][6] for category in categories],dtype=np.int16)
    if any(snowy[int(category)][6]!=mapped[index]
           for index,category in enumerate(categories)):
        raise ValueError('snow and no-snow output category maps must match')
    depth=np.asarray(snow_depth,dtype=np.float32)
    age=np.asarray(snow_age,dtype=np.int64)
    if depth.shape!=age.shape or depth.ndim!=2:
        raise ValueError('snow depth and age must be matching two-dimensional arrays')
    global_index=reference.snow_grid_index-1
    local_i=global_index%snow_full_nx-int(snow_i_offset)
    local_j=global_index//snow_full_nx-int(snow_j_offset)
    if np.any((local_i<0)|(local_i>=depth.shape[1])|
              (local_j<0)|(local_j>=depth.shape[0])):
        raise ValueError('snow-grid reference lies outside the supplied snow-data window')
    sample_depth=depth[local_j,local_i]
    sample_age=age[local_j,local_i]
    if np.any(~np.isfinite(sample_depth)):
        raise ValueError('snow depth contains missing values at referenced cells')
    land=reference.category_index
    weights=np.asarray(reference.fraction,dtype=np.float32)
    cell=reference.cell_index
    cell_count=config.grid.nx*config.grid.ny
    denominator=np.zeros(cell_count,dtype=np.float32)
    np.add.at(denominator,cell,weights)
    if np.any(denominator<=0.):
        raise ValueError('snow-grid reference has no weight for one or more target cells')

    def aggregate(values):
        numerator=np.zeros(cell_count,dtype=np.float32)
        np.add.at(numerator,cell,
                  np.asarray(weights*np.asarray(values,dtype=np.float32),
                             dtype=np.float32))
        return np.asarray(numerator/denominator,dtype=np.float32).reshape(
            config.grid.ny,config.grid.nx)

    has_snow=sample_depth>0.
    if config.snow_roughness_method==1:
        if config.snow_height_scale_m<=0.:
            raise ValueError('snow_height_scale_m/HSCL must be positive')
        scale=np.float32(config.snow_height_scale_m)
        obstacle=np.asarray(normal_table[:,0]*scale,dtype=np.float32)
        minimum=obstacle[-1]
        roughness=np.where(
            sample_depth<config.snow_depth_minimum_m,
            normal_table[land,0],
            np.asarray(np.maximum(obstacle[land]-sample_depth,minimum)/scale,
                       dtype=np.float32))
    elif config.snow_roughness_method==2:
        roughness=np.where(has_snow,snow_table[land,0],normal_table[land,0])
    else:
        raise ValueError('snow_roughness_method/MSRL must be 1 or 2')
    roughness=_fortran_exp(aggregate(_fortran_log(roughness)))

    if config.snow_albedo_method==1:
        parameters=config.snow_age_parameters
        if set(int(value) for value in categories)-set(parameters):
            raise ValueError('snow age parameters do not cover the input categories')
        raw_times=np.asarray([parameters[int(category)][0] for category in categories],
                             dtype=np.float32)
        reductions=np.asarray([parameters[int(category)][1] for category in categories],
                              dtype=np.float32)
        times=np.floor(np.asarray(raw_times+np.float32(.5),dtype=np.float32)).astype(np.int32)
        if np.any(times<=0):
            raise ValueError('snow albedo age time scales must be positive')
        snow_albedo=_snow_albedo_values(
            normal_table[land,1],snow_table[land,1],reductions[land],
            times[land],sample_age)
        albedo=np.where(sample_depth<config.snow_depth_minimum_m,
                        normal_table[land,1],snow_albedo)
    elif config.snow_albedo_method==2:
        # MAKEGEO 3.2 passes integer NDSNOW where WTSW declares real XSNOWC.
        # With its implicit Fortran interface, zero remains zero and positive
        # ages become positive subnormal reals.  Preserve that effective test.
        albedo=np.where(sample_age>0,snow_table[land,1],normal_table[land,1])
    else:
        raise ValueError('snow_albedo_method/MSAL must be 1 or 2')
    fields=[aggregate(albedo)]
    for column in range(2,6):
        values=np.where(has_snow,snow_table[land,column],normal_table[land,column])
        fields.append(aggregate(values))

    dominant=_fields(frac,config,categories)[0]
    return (dominant,roughness,*fields)

def _merge_landuse(primary,secondary):
    """LLU2 replacement: supplement only cells missing from the primary."""
    primary=np.asarray(primary,dtype=np.float32); secondary=np.asarray(secondary,dtype=np.float32)
    if primary.shape!=secondary.shape: raise ValueError('primary and secondary land-use grids do not match')
    missing=_category_sum(primary)<=np.float32(.001)
    return np.where(missing[None,:,:],secondary,primary)

def _write_netcdf(path,terrain,fields,g,config):
    names=('dominant_landuse','roughness_length','albedo','bowen_ratio','soil_heat_flux_parameter','anthropogenic_heat_flux','leaf_area_index')
    # GEO.DAT is the numerical interface used by legacy CALMET.  Keep its
    # F7.2/F7.4 quantisation in NetCDF so bypassing text I/O changes no values.
    terrain=_legacy_round(terrain,2)
    fields=(fields[0],_legacy_round(fields[1],4),*(_legacy_round(value,2) for value in fields[2:]))
    with Dataset(path,'w',format='NETCDF4') as ds:
        ds.createDimension('y',g.ny); ds.createDimension('x',g.nx); ds.Conventions='CF-1.8'; ds.source='PyCALMET MAKEGEO NumPy port'
        ds.projection=g.projection; ds.datum=g.datum; ds.false_easting_km=g.false_easting_km; ds.false_northing_km=g.false_northing_km
        ds.grid_nx=g.nx; ds.grid_ny=g.ny; ds.grid_dx_km=g.dx_km; ds.grid_dy_km=g.dy_km
        ds.x_origin_km=g.x_origin_km; ds.y_origin_km=g.y_origin_km
        ds.latitude_origin=g.lat0; ds.longitude_origin=g.lon0; ds.standard_parallel_1=g.lat1; ds.standard_parallel_2=g.lat2
        ds.ttm_scale=g.ttm_scale
        ds.utm_zone=g.utm_zone; ds.hemisphere=g.hemisphere
        ds.water_category_min=int(config.output_water_min); ds.water_category_max=int(config.output_water_max)
        ds.landuse_categories=','.join(str(int(value)) for value in config.output_categories)
        ds.createVariable('terrain_height','f4',('y','x'),zlib=True)[:]=terrain
        for name,data in zip(names,fields): ds.createVariable(name,'i2' if name=='dominant_landuse' else 'f4',('y','x'),zlib=True)[:]=data

def _comma_rows(stream,data,*,width,decimals,values_per_record):
    """Write Fortran repeated ``F/I,a1`` records with format reversion."""
    for row in np.asarray(data)[::-1]:
        fields=[(f'{int(value):{width}d}' if decimals is None else
                 f'{float(value):{width}.{decimals}f}')
                for value in row]
        for start in range(0,len(fields),values_per_record):
            stop=min(start+values_per_record,len(fields))
            stream.write(''.join(
                value+(',' if index<len(fields)-1 else '')
                for index,value in enumerate(fields[start:stop],start))+'\n')


def _rows(stream,data,fmt):
    decimals=4 if fmt=='.4f' else 2
    _comma_rows(stream,data,width=7,decimals=decimals,values_per_record=100)

def _write_legacy(path,terrain,fields,g,config):
    dominant,z0,alb,bow,soil,qf,lai=fields
    with path.open('w',encoding='ascii') as s:
        s.write(f"{'GEO.DAT':<16}{'2.1':<16}"
                f"{'Header structure with coordinate parameters':<64}\n"
                "   2\nProduced by PyCALMET MAKEGEO NumPy implementation\n"
                f"{config.title[:80]:<80}\n")
        projection=g.projection.upper()
        latitude=lambda value: f'{abs(value):g}{"N" if value>=0 else "S"}'
        longitude=lambda value: f'{abs(value):g}{"E" if value>=0 else "W"}'
        s.write(f'{projection:<8}\n')
        if projection=='UTM':
            s.write(f'{g.utm_zone:4d}{g.hemisphere:<4}\n')
        else:
            parameters=[latitude(g.lat0),longitude(g.lon0)]
            if projection in {'LCC','PS'}: parameters.append(latitude(g.lat1))
            if projection=='LCC': parameters.append(latitude(g.lat2))
            s.write(''.join(f'{value:<16}' for value in parameters)+'\n')
        if projection in {'LCC','LAZA','TTM'}:
            s.write(gfortran_real_pair(g.false_easting_km,g.false_northing_km))
        s.write(f'{g.datum:<8}{NIMA_DATABASE_DATE:<12}\n')
        s.write(f'{g.nx:8d}{g.ny:8d}{g.x_origin_km:12.3f}{g.y_origin_km:12.3f}{g.dx_km:12.3f}{g.dy_km:12.3f}\nKM  M   \n')
        s.write(f' 1          - LAND USE DATA - (1=new categories)\n{len(config.output_categories):5d}{config.output_water_min:4d}{config.output_water_max:4d} - NLU, IWAT1, IWAT2\n '+''.join(f' {v:3d}' for v in config.output_categories)+'\n')
        _comma_rows(s,dominant,width=7,decimals=None,values_per_record=10)
        s.write(f' 1.0000  - {"TERRAIN heights - HTFAC (Conversion to meters)":<70}\n'); _rows(s,terrain,'.2f')
        for label,data,fmt in [('gridded z0 field',z0,'.4f'),('gridded albedo field',alb,'.2f'),('gridded Bowen ratio field',bow,'.2f'),('gridded soil heat flux parameters',soil,'.2f'),('gridded anthropogenic heat flux field',qf,'.2f'),('gridded leaf area index field',lai,'.2f')]:
            s.write(f' 2    - {label:<70}\n'); _rows(s,data,fmt)


def _write_qa_grids(terrain,dominant,g,config):
    if config.image_mode not in {0,1}:
        raise ValueError('MAKEGEO IMAGE must be zero or one')
    if config.landuse_plot_output is not None:
        path=Path(config.landuse_plot_output); path.parent.mkdir(parents=True,exist_ok=True)
        if config.image_mode==0:
            xlo=g.x_origin_km+.5*g.dx_km; xhi=g.x_origin_km+(g.nx-.5)*g.dx_km
            ylo=g.y_origin_km+.5*g.dy_km; yhi=g.y_origin_km+(g.ny-.5)*g.dy_km
        else:
            xlo=g.x_origin_km; xhi=g.x_origin_km+g.nx*g.dx_km
            ylo=g.y_origin_km; yhi=g.y_origin_km+g.ny*g.dy_km
        with path.open('w',encoding='ascii') as stream:
            stream.write('DSAA\n')
            stream.write(f'{g.nx:12d}{g.ny:12d}\n{xlo:12.4f}{xhi:12.4f}\n')
            stream.write(f'{ylo:12.4f}{yhi:12.4f}\n{10:12d}{100:12d}\n')
            for row in np.asarray(dominant):
                stream.write(''.join(f'{abs(int(value)):3d}  ' for value in row)+'\n')
    if config.terrain_plot_output is not None:
        path=Path(config.terrain_plot_output); path.parent.mkdir(parents=True,exist_ok=True)
        minimum=float(np.min(terrain)); maximum=float(np.max(terrain))
        if maximum==minimum:
            path.unlink(missing_ok=True)
        else:
            xlo=g.x_origin_km+.5*g.dx_km; xhi=g.x_origin_km+(g.nx-.5)*g.dx_km
            ylo=g.y_origin_km+.5*g.dy_km; yhi=g.y_origin_km+(g.ny-.5)*g.dy_km
            with path.open('w',encoding='ascii') as stream:
                stream.write('DSAA\n')
                stream.write(f'{g.nx:12d}{g.ny:12d}\n{xlo:12.4f}{xhi:12.4f}\n')
                stream.write(f'{ylo:12.4f}{yhi:12.4f}\n{minimum:12.4E}{maximum:12.4E}\n')
                for row in np.asarray(terrain):
                    stream.write(''.join(f'{value:11.4E} ' for value in row)+'\n')
    if config.color_map_output is not None:
        path=Path(config.color_map_output); path.parent.mkdir(parents=True,exist_ok=True)
        colors='''ColorMap 1 1
    0.000000 255 255   0
   11.000000 255 255   0
   11.000000 204 255 102
   22.100000 204 255 102
   22.100000 160 255 160
   33.100000 160 255 160
   33.100000  80 255  80
   44.200000  80 255  80
   44.200000 153 255 255
   50.000000 153 255 255
   50.000000  60 204 255
   55.400000  60 204 255
   55.400000 204 153 204
   66.500000 204 153 204
   66.500000 255 204 153
   77.600000 255 204 153
   77.600000 255 255 204
   88.700000 255 255 204
   88.700000 255 255 255
   99.029557 255 255 255
   99.029557 255   0   0
  100.000000 255   0   0
'''
        path.write_text(colors,encoding='ascii')

def run(config: MakegeoConfig) -> GeophysicalResult:
    terrain=(_read_terrain(config.terrain,config.grid) if config.use_terrain else
             np.zeros((config.grid.ny,config.grid.nx),dtype=float))
    expected_categories=tuple((config.input_properties or STANDARD_PROPERTIES).keys())
    categories,frac=_read_landuse(config.landuse,config.grid,expected_categories)
    if config.secondary_landuse is not None:
        secondary_categories,secondary=_read_landuse(
            config.secondary_landuse,config.grid,expected_categories)
        if not np.array_equal(categories,secondary_categories): raise ValueError('primary and secondary land-use categories do not match')
        frac=_merge_landuse(frac,secondary)
    dominant=_fields(frac,config,categories)[0]
    _write_qa_grids(terrain,dominant,config.grid,config)
    output=Path(config.output); output.parent.mkdir(parents=True,exist_ok=True)
    if config.snow_processing:
        reference=_read_snow_reference(config.landuse,config.grid,categories)
        if config.snow_data is None:
            raise ValueError('snow_processing requires snow_data/SNOWDAT')
        if config.snow_start_date is None or config.snow_end_date is None:
            raise ValueError('snow processing requires IDATEBEG and IDATEEND')
        if not output.suffix:
            raise ValueError('dated MAKEGEO snow output requires a filename extension')
        outputs=[]
        for date,depth,age,header in _iter_formatted_snow_days(
                config.snow_data,config.snow_start_date,config.snow_end_date):
            fields=_snow_weighted_fields(
                frac,categories,reference,depth,age,config,
                snow_i_offset=header.i_begin-1,snow_j_offset=header.j_begin-1,
                snow_full_nx=header.raw_nx)
            dated=output.with_name(f'{output.stem}{date:%Y%m%d}{output.suffix}')
            with tempfile.NamedTemporaryFile(dir=dated.parent,prefix=dated.name+'.',
                                             delete=False) as tmp:
                temporary=Path(tmp.name)
            try:
                if dated.suffix.lower() in {'.nc','.nc4','.cdf'}:
                    _write_netcdf(temporary,terrain,fields,config.grid,config)
                else:
                    _write_legacy(temporary,terrain,fields,config.grid,config)
                temporary.replace(dated)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(dated)
        first=outputs[0]
        qa_values=_qa_cell_values(terrain,fields,config)
        return GeophysicalResult(first,int(frac.sum(axis=0).astype(bool).sum()),
            float(np.min(terrain)),float(np.max(terrain)),tuple(outputs),
            config.qa_cell if qa_values is not None else None,qa_values)
    fields=_fields(frac,config,categories)
    qa_values=_qa_cell_values(terrain,fields,config)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',delete=False) as tmp: temporary=Path(tmp.name)
    try:
        if output.suffix.lower() in {'.nc','.nc4','.cdf'}: _write_netcdf(temporary,terrain,fields,config.grid,config)
        else: _write_legacy(temporary,terrain,fields,config.grid,config)
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    return GeophysicalResult(output,int(frac.sum(axis=0).astype(bool).sum()),
        float(np.min(terrain)),float(np.max(terrain)),(output,),
        config.qa_cell if qa_values is not None else None,qa_values)
