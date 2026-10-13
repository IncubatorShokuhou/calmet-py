"""NumPy CTGPROC implementation for USGS Global LAZA rasters."""
from __future__ import annotations

from dataclasses import dataclass,replace
from pathlib import Path
import re
import tempfile
import numpy as np
from netCDF4 import Dataset

from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS, Transformer
import tifffile
from .config import CtgprocConfig,CtgprocSource,GridConfig
from .ctg_data import MODIS_SPEC,NLCD92_SPECS,ModisSpec,Nlcd92Spec
from .datums import NIMA_DATABASE_DATE,normalize_datum
from .gshhs import extract_gshhs
from .legacy_text import gfortran_real_pair
from .projections import (geographic_crs,grid_crs,project_geographic,
                          project_from_crs,project_glaz_to_lcc,transform_grid,unproject_geographic,
                          unproject_to_crs)

USGS38 = np.array([11,12,13,14,15,16,17,21,22,23,24,31,32,33,41,42,43,51,52,53,54,55,61,62,71,72,73,74,75,76,77,81,82,83,84,85,91,92], dtype=np.int16)
GLOBAL_MERGE_TYPES={'GLAZNA','GLAZSA','GLAZEU','GLAZAS','GLAZAF','GLAZAP',
                    'GLC2K','UMDGLC','MODIS'}
PROCESS_ORDER={'GLC2K':0,'UMDGLC':1,'CTG':2,'NZGEN':3,
               'GLAZNA':4,'GLAZSA':4,'GLAZEU':4,'GLAZAS':4,'GLAZAF':4,'GLAZAP':4,
               'NLCDTF':5,'NLCD92':6,'NLCD01':7,'CORINE':8,'MODIS':9,'GEN':10}
GLAZ_LIMITS={
    'GLAZNA':(90.,8.5,-20.,-180.),'GLAZSA':(13.,-60.,-32.,-100.),
    'GLAZEU':(90.,8.8,60.,-25.),'GLAZAS':(90.,-6.,200.,60.),
    'GLAZAF':(46.,-39.,60.,-30.),'GLAZAP':(22.,-51.,175.,88.),
}
GLAZ_TO_USGS = np.array([0,16,21,21,21,21,21,31,32,33,33,41,41,42,42,43,52,62,61,77,82,81,85,83,91], dtype=np.int16)
NZ_TO_USGS = np.array([
    0,55,0,0,0,0,0,0,0,0,16,14,0,0,0,0,0,0,0,0,21,22,0,0,0,0,0,0,0,0,
    43,43,0,0,0,0,0,0,0,0,32,0,0,61,0,0,0,0,0,0,52,0,0,0,0,0,0,0,0,0,
    62,62,0,0,0,0,0,0,0,0,75,72,74],dtype=np.int16)
NLCD92_TO_USGS=np.zeros(256,dtype=np.int16)
NLCD92_TO_USGS[[11,12,21,22,23,31,32,33,41,42,43,51,61,71,81,82,83,84,85,91,92]]=(
    52,91,11,11,13,74,75,76,41,42,43,32,22,31,21,21,21,21,21,61,62)
NLCD01_TO_USGS=np.zeros(256,dtype=np.int16)
NLCD01_TO_USGS[[11,12,21,22,23,24,31,32,41,42,43,51,52,71,72,73,74,81,82,
                91,92,93,94,95,96,97,98,99,100]]=(
    52,91,30,11,11,13,74,72,41,42,43,32,32,31,82,82,82,21,21,
    61,61,61,61,61,62,62,62,62,62)
GLC2K_TO_USGS=np.array([0,42,41,41,42,41,43,61,61,43,33,32,32,21,31,
                         62,21,22,22,77,55,91,11,55,0],dtype=np.int16)
UMDGLC_TO_USGS=np.array([55,42,42,41,41,43,32,33,32,32,31,21,21,16,77],
                         dtype=np.int16)
CORINE_TO_USGS=np.array([0,16,16,13,14,14,14,75,75,75,22,22,24,24,21,22,
    22,22,21,21,21,21,43,41,42,43,31,82,32,81,72,74,83,74,91,61,61,62,
    62,72,51,52,54,54,55],dtype=np.int16)
CORINE_CLASSES=np.array([111,112,121,122,123,124,131,132,133,141,142,211,
    212,213,221,222,223,231,241,242,243,244,311,312,313,321,322,323,324,
    331,332,333,334,335,411,412,421,422,423,511,512,521,522,523],dtype=np.int32)
CORINE_RGB=np.array([230000077,255000000,204077242,204000000,230204204,
    230204230,166000204,166077000,255077255,255166255,255230255,255255168,
    255255000,230230000,230128000,242166077,230166000,230230077,255230166,
    255230077,230204077,242204166,128255000,166000,77255000,204242077,
    166255128,166230077,166242000,230230230,204204204,204255204,0,
    166230204,166166255,77077255,204204255,230230255,166166230,204242,
    128242230,255166,166255230,230242255],dtype=np.int32)
QA_FILENAMES=('qamesh.dat','qactg.dat','qaglaz.dat','qanzgen.dat',
              'qanlcd.dat','qacorine.dat','qagcl2k.dat','qaumdglc.dat','qamodis.dat',
              'qageneric.dat')

@dataclass(frozen=True,slots=True)
class GlazSpec:
    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float
    latitude_origin: float
    longitude_origin: float

GLAZ_SPECS={
    'GLAZNA':GlazSpec(9223,8996,-4487.,-4515.,50.,-100.),
    'GLAZSA':GlazSpec(6000,8000,-3000.,-4899.,-15.,-60.),
    'GLAZEU':GlazSpec(13000,13000,-3000.,-4999.,55.,20.),
    'GLAZAS':GlazSpec(13000,12000,-8000.,-5499.,45.,100.),
    'GLAZAF':GlazSpec(8350,9276,-4458.,-4795.,5.,20.),
    'GLAZAP':GlazSpec(9300,8000,-5000.,-3944.891,-15.,135.),
}

@dataclass(frozen=True, slots=True)
class LanduseResult:
    output: Path
    source_cells_read: int
    samples_inside_grid: int
    populated_cells: int
    average_hits: int
    low_hit_threshold: int
    low_hit_cells: int


@dataclass(frozen=True,slots=True)
class CoastPolygon:
    kind: int
    x: np.ndarray
    y: np.ndarray


def _grid_indices(x, y, grid):
    """Vectorized LUGRID indices with floating-point subtraction/division."""
    f32=np.float64
    x=np.asarray(x,dtype=f32); y=np.asarray(y,dtype=f32)
    delx=np.asarray(x-f32(grid.x_origin_km),dtype=f32)
    dely=np.asarray(y-f32(grid.y_origin_km),dtype=f32)
    qx=np.asarray(delx/f32(grid.dx_km),dtype=f32)
    qy=np.asarray(dely/f32(grid.dy_km),dtype=f32)
    # Fortran first forms one-based I/J with truncation toward zero and only
    # adds one on the non-negative side.  Return Python zero-based indices.
    ix=np.trunc(qx).astype(np.int64)+(delx>=f32(0.)).astype(np.int64)-1
    iy=np.trunc(qy).astype(np.int64)+(dely>=f32(0.)).astype(np.int64)-1
    return ix,iy


def _inside_grid(x, y, grid):
    ix,iy=_grid_indices(x,y,grid)
    return ix,iy,(ix>=0)&(ix<grid.nx)&(iy>=0)&(iy<grid.ny)


def _grid_centers_1d(origin, spacing, count):
    """Construct CTGPROC cell centers in its left-associated REAL arithmetic."""
    f32=np.float64
    index=np.arange(1,int(count)+1,dtype=np.float64)
    offset=np.asarray(index-f32(.5),dtype=f32)
    distance=np.asarray(f32(spacing)*offset,dtype=f32)
    return np.asarray(f32(origin)+distance,dtype=f32)


def _snow_grid_indices(longitude, latitude):
    """Map GLOBE's REAL coordinates to CTGPROC's fixed SNODAS grid."""
    f32=np.float64
    longitude=np.asarray(longitude,dtype=f32); latitude=np.asarray(latitude,dtype=f32)
    lon_delta=np.asarray(longitude-f32(-124.729584),dtype=f32)
    lat_delta=np.asarray(latitude-f32(52.8704185),dtype=f32)
    lon_ratio=np.asarray(lon_delta/f32(8.33333377e-3),dtype=f32)
    lat_ratio=np.asarray(lat_delta/f32(-8.33333377e-3),dtype=f32)
    # The fixed SNODAS extent makes both ratios non-negative, so NINT is
    # floor(x + .5), with the addition itself rounded to default REAL.
    isn=np.floor(np.asarray(lon_ratio+f32(.5),dtype=f32)).astype(np.int64)+1
    jsn=np.floor(np.asarray(lat_ratio+f32(.5),dtype=f32)).astype(np.int64)+1
    return isn,jsn


@dataclass(slots=True)
class SnowGridInfo:
    """Sparse equivalent of CTGPROC's NGSNOW/IPSNOW/RSNOW arrays."""

    grid: GridConfig
    hits: dict[tuple[int, int, int], int]
    order: dict[tuple[int, int], list[int]]

    @classmethod
    def create(cls, grid):
        return cls(grid, {}, {})

    def add(self, x, y, categories, good):
        selected=np.flatnonzero(good)
        if selected.size==0:
            return
        x=np.asarray(x)[selected]; y=np.asarray(y)[selected]
        category=np.asarray(categories,dtype=np.int64)[selected]
        ix,iy=_grid_indices(x,y,self.grid)
        cell=iy*self.grid.nx+ix
        lon,lat=unproject_geographic(x,y,self.grid,'WGS-84',units='km')
        isn,jsn=_snow_grid_indices(lon,lat)
        snow=(jsn-1)*6935+isn
        # SETSNOW is record-order dependent and contains a historical JP/K
        # indexing bug: a repeat of any prior snow grid increments the last
        # grid added for that cell/land pair.  This short loop is therefore
        # intentionally not replaced by an unordered group-by reduction.
        for raw_cell,raw_category,raw_snow in zip(cell,category,snow,strict=True):
            group=(int(raw_cell),int(raw_category)); snow_grid=int(raw_snow)
            grids=self.order.setdefault(group,[])
            if snow_grid not in grids:
                if len(grids)>=10:
                    raise ValueError('legacy CTGPROC permits at most 10 snow grids per land-use type')
                grids.append(snow_grid)
                target=snow_grid
            else:
                target=grids[-1]
            key=(group[0],group[1],target)
            self.hits[key]=self.hits.get(key,0)+1

    def records(self, categories):
        lookup={int(value):index+1 for index,value in enumerate(categories)}
        grouped={}
        totals=np.zeros(self.grid.nx*self.grid.ny,dtype=np.int64)
        for (cell,category,snow),count in self.hits.items():
            if category not in lookup:
                raise ValueError(f'snow land-use category {category} is absent from output categories')
            totals[cell]+=count
        if self.order:
            for (cell,category),grids in self.order.items():
                land=lookup[category]
                grouped[cell,land]=[(snow,self.hits[cell,category,snow]) for snow in grids]
        else:
            # Compatibility for callers that populated ``hits`` directly.
            for (cell,category,snow),count in self.hits.items():
                grouped.setdefault((cell,lookup[category]),[]).append((snow,count))
        return grouped,totals


def _bin_samples(counts,columns,x,y,categories,good,config,snow=None):
    """Vectorized LUGRID count update plus optional sample-level snow QA."""
    if snow is not None:
        snow.add(x,y,categories,good)
    ix,iy=_grid_indices(np.asarray(x)[good],np.asarray(y)[good],config.grid)
    cells=iy*config.grid.nx+ix
    selected_categories=np.asarray(categories)[good]
    for category in np.unique(selected_categories):
        selected=selected_categories==category
        counts[columns[int(category)]]+=np.bincount(
            cells[selected],minlength=counts.shape[1]).astype(counts.dtype,copy=False)


def _read_with_optional_snow(function,*args,snow=None,**kwargs):
    """Keep the established reader call contract when snow QA is disabled."""
    if snow is None:
        return function(*args,**kwargs)
    return function(*args,snow=snow,**kwargs)


def _qa_directory(config):
    return (Path(config.qa_output_directory) if config.qa_output_directory is not None
            else Path(config.output).parent)


def _qa_write(config,name,*columns):
    if not config.qa_cell_output:
        return
    values=np.column_stack(tuple(np.asarray(column).ravel() for column in columns))
    if values.size==0:
        return
    with (_qa_directory(config)/name).open('ab') as stream:
        np.savetxt(stream,values,fmt='%.8g')


def _read_coastline_bln(path,grid,*,allow_empty=False):
    """Read COASTRD's pre-extracted BLN polygon representation."""
    source=Path(path)
    lines=source.read_text(encoding='ascii',errors='strict').splitlines()
    polygons=[]; index=0
    while index<len(lines):
        if not lines[index].strip():
            index+=1; continue
        header=re.fullmatch(r'\s*(\d+)\s*(?:,\s*|\s+)([-+]?\d+)\s*(?:,\s*|\s+)(.*?)\s*',
                            lines[index])
        index+=1
        if header is None: raise ValueError(f'invalid coastline BLN header at {source}:{index}')
        try:
            count=int(header.group(1))
            identifiers=[int(value) for value in re.findall(r'-?\d+',header.group(3))]
        except ValueError as error:
            raise ValueError(f'invalid coastline BLN header at {source}:{index}') from error
        if count<3 or not identifiers or identifiers[0] not in {1,2,3,4}:
            raise ValueError(f'invalid coastline polygon at {source}:{index}')
        if index+count>len(lines): raise ValueError(f'truncated coastline polygon at {source}:{index}')
        points=[]
        for number,line in enumerate(lines[index:index+count],start=index+1):
            try: point=tuple(float(value) for value in line.replace(',',' ').split()[:2])
            except ValueError as error:
                raise ValueError(f'invalid coastline point at {source}:{number}') from error
            if len(point)!=2: raise ValueError(f'invalid coastline point at {source}:{number}')
            points.append(point)
        index+=count
        coordinates=np.asarray(points,dtype=np.float64)
        polygons.append(CoastPolygon(identifiers[0],coordinates[:,0],coordinates[:,1]))
    if not polygons:
        if allow_empty: return ()
        raise ValueError(f'no coastline polygons found in {source}')
    xmin=min(np.min(item.x) for item in polygons); xmax=max(np.max(item.x) for item in polygons)
    ymin=min(np.min(item.y) for item in polygons); ymax=max(np.max(item.y) for item in polygons)
    grid_xmax=grid.x_origin_km+grid.nx*grid.dx_km
    grid_ymax=grid.y_origin_km+grid.ny*grid.dy_km
    if xmax<grid.x_origin_km or xmin>grid_xmax or ymax<grid.y_origin_km or ymin>grid_ymax:
        raise ValueError('coastline BLN limits are not consistent with the CTGPROC grid')
    return tuple(polygons)


def _polygon_membership(px,py,polygon):
    """Vectorized PNPOLY result: -1 outside, 0 edge, 1 inside."""
    f32=np.float64
    px=np.asarray(px,dtype=f32); py=np.asarray(py,dtype=f32)
    x=np.asarray(polygon.x,dtype=f32); y=np.asarray(polygon.y,dtype=f32)
    x2=np.roll(x,-1); y2=np.roll(y,-1)
    result=np.empty(px.shape,dtype=np.int8)
    # Bound each broadcast to roughly two million point-edge pairs.  GSHHS
    # shorelines can contain many thousands of vertices.
    chunk=max(1,2_000_000//max(1,x.size))
    for begin in range(0,px.size,chunk):
        end=min(begin+chunk,px.size)
        local_x=px[begin:end,None]; local_y=py[begin:end,None]
        xi=np.asarray(x-local_x,dtype=f32); yi=np.asarray(y-local_y,dtype=f32)
        xj=np.asarray(x2-local_x,dtype=f32); yj=np.asarray(y2-local_y,dtype=f32)
        ix=xi>=f32(0.); iy=yi>=f32(0.); jx=xj>=f32(0.); jy=yj>=f32(0.)
        xor_x=ix!=jx; xor_y=iy!=jy
        vertex=(xi==f32(0.))&(yi==f32(0.))
        zero_length=(xi==xj)&(yi==yj)
        vertical=(xi==f32(0.))&(xj==f32(0.))&xor_y
        horizontal=(yi==f32(0.))&(yj==f32(0.))&xor_x
        eligible=(iy|jy)&xor_x&~zero_length
        obvious=eligible&iy&jy
        remaining=eligible&~obvious
        numerator=np.asarray(np.asarray(yi*xj,dtype=f32)-
                             np.asarray(xi*yj,dtype=f32),dtype=f32)
        denominator=np.asarray(xj-xi,dtype=f32)
        quotient=np.divide(numerator,denominator,out=np.zeros_like(numerator),
                           where=remaining)
        edge=np.any(vertex|vertical|horizontal|(remaining&(quotient==f32(0.))),axis=1)
        toggles=np.count_nonzero(obvious|(remaining&(quotient>f32(0.))),axis=1)
        inside=toggles%2==1
        result[begin:end]=np.where(edge,0,np.where(inside,1,-1))
    return result


def _coastline_types(x,y,polygons):
    """Return IPTYPE's highest nested polygon type for each point."""
    result=np.zeros(np.asarray(x).shape,dtype=np.int8)
    flat=result.ravel(); px=np.asarray(x).ravel(); py=np.asarray(y).ravel()
    for polygon in polygons:
        bounds=((px>=np.min(polygon.x))&(px<=np.max(polygon.x))&
                (py>=np.min(polygon.y))&(py<=np.max(polygon.y))&
                (flat<polygon.kind))
        if not bounds.any(): continue
        selected=np.flatnonzero(bounds)
        membership=_polygon_membership(px[selected],py[selected],polygon)
        inside=membership==1; flat[selected[inside]]=np.maximum(flat[selected[inside]],polygon.kind)
        edge=membership==0
        flat[selected[edge]]={1:0,2:2,3:2,4:4}[polygon.kind]
    return result


def _fill_empty_ocean(counts,categories,config):
    if config.coastline_file is None:
        raise ValueError('fill_empty_ocean requires coastline_file/COASTBLN')
    if config.input_type.upper()=='GEN':
        raise ValueError('legacy CTGPROC does not support coastline processing for GENERIC data')
    matches=np.flatnonzero(np.asarray(categories)==config.ocean_category)
    if matches.size!=1:
        raise ValueError('ocean_category/IOCEAN must be in the USGS-38 category table')
    polygons=_read_coastline_bln(config.coastline_file,config.grid,
                                  allow_empty=config.gshhs_file is not None)
    x=_grid_centers_1d(config.grid.x_origin_km,config.grid.dx_km,config.grid.nx)
    y=_grid_centers_1d(config.grid.y_origin_km,config.grid.dy_km,config.grid.ny)
    xx,yy=np.meshgrid(x,y,indexing='xy')
    ocean=(counts.sum(axis=0)==0)&(_coastline_types(xx,yy,polygons)==0)
    counts[matches[0],ocean]=1
    return int(ocean.sum())


def _coastline_polygons(config):
    if not config.coastline_swap:
        return ()
    if config.coastline_file is None:
        raise ValueError('coastline_swap/LMARSWAP requires coastline_file/COASTBLN')
    if config.input_type.upper()=='GEN':
        raise ValueError('legacy CTGPROC does not support coastline processing for GENERIC data')
    if config.ocean_category not in USGS38:
        raise ValueError('ocean_category/IOCEAN must be in the USGS-38 category table')
    return _read_coastline_bln(config.coastline_file,config.grid,
                               allow_empty=config.gshhs_file is not None)


def _swap_coastline_categories(categories,x,y,config,polygons):
    """Apply LUGRID's LMARSWAP rule to projected source samples."""
    values=np.asarray(categories,dtype=np.int16)
    if not polygons and config.coastline_file is None:
        return values,np.ones(values.shape,dtype=bool)
    point_type=_coastline_types(x,y,polygons)
    present=values>0
    marine=point_type==0
    keep=~present|marine|~np.isin(values,(54,55))
    swapped=np.where(present&marine&(values!=54),config.ocean_category,values)
    return swapped,keep

def _remap_categories(categories,counts,mapping):
    """Apply a source-to-output class map and merge repeated targets."""
    source=np.asarray(categories,dtype=np.int16)
    unknown=sorted(set(mapping)-set(int(value) for value in source))
    if unknown: raise ValueError(f'category_map contains absent source categories: {unknown}')
    targets=np.asarray([mapping.get(int(value),int(value)) for value in source],dtype=np.int16)
    if np.any(targets<=0): raise ValueError('category_map target categories must be positive integers')
    output=np.asarray(list(dict.fromkeys(int(value) for value in targets)),dtype=np.int16)
    lookup={int(value):index for index,value in enumerate(output)}
    merged=np.zeros((output.size,*counts.shape[1:]),dtype=counts.dtype)
    for source_index,target in enumerate(targets): merged[lookup[int(target)]]+=counts[source_index]
    return output,merged


def _merge_global_counts(accumulated,current,categories,excluded_water=52):
    """Vectorized GMERGE handling for continent-exclusion water cells."""
    matches=np.flatnonzero(np.asarray(categories)==excluded_water)
    if matches.size!=1: raise ValueError('GMERGE excluded-water category is absent')
    water=int(matches[0])
    accumulated_total=accumulated.sum(axis=0,dtype=np.int64)
    current_total=current.sum(axis=0,dtype=np.int64)
    accumulated_land=accumulated_total>accumulated[water]
    current_land=current_total>current[water]
    replace=~accumulated_land&current_land
    add=(accumulated_land&current_land)|(~accumulated_land&~current_land)
    accumulated[:,replace]=current[:,replace]
    accumulated[:,add]+=current[:,add]


def _select_glaz_markers(center_lon,center_lat,left_lon,right_lon,
                         lower_lat,upper_lat):
    """GSELECT overlap tests after its longitude-branch normalization."""
    f32=np.float64
    center_lon=f32(center_lon); center_lat=f32(center_lat)
    left_lon=f32(left_lon); right_lon=f32(right_lon)
    lower_lat=f32(lower_lat); upper_lat=f32(upper_lat)
    keep=set()
    for kind,limits in GLAZ_LIMITS.items():
        north,south,east,west=(f32(value) for value in limits)
        center_hits=int(south<=center_lat<=north)+int(west<=center_lon<=east)
        corner_hits=(int(south<=upper_lat<=north)+int(south<=lower_lat<=north)+
                     int(west<left_lon<=east)+int(west<right_lon<=east))
        for shift in (f32(360.),f32(-360.)):
            shifted_left=np.asarray(left_lon+shift,dtype=f32)
            shifted_right=np.asarray(right_lon+shift,dtype=f32)
            corner_hits+=int(west<shifted_left<=east)
            corner_hits+=int(west<shifted_right<=east)
        if (center_hits==2 and corner_hits==4) or center_hits==2 or corner_hits==3:
            keep.add(kind)
    if {'GLAZEU','GLAZAS'}<=keep: keep.remove('GLAZAS')
    return keep


def _select_glaz_regions(grid):
    """Transform target markers and reproduce GSELECT's regional screen."""
    f32=np.float64
    x0=f32(grid.x_origin_km); y0=f32(grid.y_origin_km)
    x_width=np.asarray(f32(grid.nx)*f32(grid.dx_km),dtype=f32)
    y_width=np.asarray(f32(grid.ny)*f32(grid.dy_km),dtype=f32)
    x1=np.asarray(x0+x_width,dtype=f32); y1=np.asarray(y0+y_width,dtype=f32)
    xc=np.asarray(f32(.5)*np.asarray(x0+x1,dtype=f32),dtype=f32)
    yc=np.asarray(f32(.5)*np.asarray(y1+y0,dtype=f32),dtype=f32)
    x=np.array([xc,x0,x0,x1,x1],dtype=f32)
    y=np.array([yc,y0,y1,y1,y0],dtype=f32)
    transform=Transformer.from_crs(_lcc(grid),geographic_crs(grid.datum),always_xy=True)
    longitude,latitude=transform.transform(x,y)
    longitude=np.asarray(longitude,dtype=f32); latitude=np.asarray(latitude,dtype=f32)
    center=longitude[0]; corners=longitude[1:].copy()
    if np.asarray(center-corners[0],dtype=f32)<f32(0.): corners[0]=f32(corners[0]-f32(360.))
    if np.asarray(center-corners[1],dtype=f32)<f32(0.): corners[1]=f32(corners[1]-f32(360.))
    if np.asarray(corners[2]-center,dtype=f32)<f32(0.): corners[2]=f32(corners[2]+f32(360.))
    if np.asarray(corners[3]-center,dtype=f32)<f32(0.): corners[3]=f32(corners[3]+f32(360.))
    left=min(corners[0],corners[1]); right=max(corners[2],corners[3])
    if left<f32(-180.): left=f32(left+f32(360.))
    if left>f32(180.): left=f32(left-f32(360.))
    if right<f32(-180.): right=f32(right+f32(360.))
    if right>f32(180.): right=f32(right-f32(360.))
    lower=min(latitude[1],latitude[4]); upper=max(latitude[2],latitude[3])
    if center<f32(-180.): center=f32(center+f32(360.))
    if center>f32(180.): center=f32(center-f32(360.))
    return _select_glaz_markers(center,latitude[0],left,right,lower,upper)

def _lcc(g: GridConfig) -> CRS:
    return grid_crs(g,units='km')

def _glaz_crs(spec: GlazSpec) -> CRS:
    return CRS.from_proj4(f"+proj=laea +lat_0={spec.latitude_origin:g} +lon_0={spec.longitude_origin:g} +R=6370997 +units=km +no_defs")

def _window(g, reverse, spec):
    f32=np.float64
    x0=f32(g.x_origin_km); y0=f32(g.y_origin_km)
    x_width=np.asarray(f32(g.nx)*f32(g.dx_km),dtype=f32)
    y_width=np.asarray(f32(g.ny)*f32(g.dy_km),dtype=f32)
    x1=np.asarray(x0+x_width,dtype=f32); y1=np.asarray(y0+y_width,dtype=f32)
    x=np.array([x0,x1,x1,x0],dtype=f32)
    y=np.array([y0,y0,y1,y1],dtype=f32)
    px,py=reverse(x,y)
    px=np.asarray(px,dtype=f32); py=np.asarray(py,dtype=f32)
    x_delta=np.asarray(px-f32(spec.x_origin_km),dtype=f32)
    y_delta=np.asarray(py-f32(spec.y_origin_km),dtype=f32)
    ix0,ix1=int(np.min(x_delta)),int(np.max(x_delta))
    iy0,iy1=int(np.min(y_delta)),int(np.max(y_delta))
    buf=max((ix1-ix0)//10+1,(iy1-iy0)//10+1)
    return range(max(0,ix0-buf),min(spec.nx,ix1+buf+1)),range(max(0,iy0-buf),min(spec.ny,iy1+buf+1))


def _mesh_offsets(mesh,cell_size=1.):
    """Return LOADOTHER's REAL mesh offsets in IMESH order."""
    f32=np.float64
    step=np.asarray(f32(cell_size)/f32(mesh),dtype=f32)
    half=np.asarray(f32(cell_size)/f32(2.),dtype=f32)
    index=np.arange(1,int(mesh)+1,dtype=f32)
    centered=np.asarray(index-f32(.5),dtype=f32)
    return np.asarray(np.asarray(centered*step,dtype=f32)-half,dtype=f32)


def _mesh_coordinates(centers,mesh,cell_size=1.):
    """Vectorize Fortran ``(center-half)+(index-.5)*step`` arithmetic."""
    f32=np.float64
    center=np.asarray(centers,dtype=f32)
    step=np.asarray(f32(cell_size)/f32(mesh),dtype=f32)
    half=np.asarray(f32(cell_size)/f32(2.),dtype=f32)
    index=np.arange(1,int(mesh)+1,dtype=f32)
    delta=np.asarray(np.asarray(index-f32(.5),dtype=f32)*step,dtype=f32)
    lower=np.asarray(center-half,dtype=f32)
    return np.asarray(lower[...,None]+delta,dtype=f32)


def _source_centers(origin,spacing,indices):
    """Evaluate Fortran ``origin+spacing*(index-1)`` in default REAL."""
    f32=np.float64
    index=np.asarray(indices,dtype=f32)
    product=np.asarray(f32(spacing)*index,dtype=f32)
    return np.asarray(f32(origin)+product,dtype=f32)


def _geotiff_geometry(scale,tie,raster_type,crs,*,orientation=1,
                      extra_half_pixel=False):
    """Decode GET_IFD's origin and spacing in its REAL working units."""
    f32=np.float64
    source_crs=CRS.from_user_input(crs)
    if source_crs.is_geographic:
        unit_scale=f32(1.)
        native_per_work=1.
    else:
        axes=source_crs.axis_info
        metres_per_native=float(axes[0].unit_conversion_factor) if axes else 1.
        unit_scale=f32(metres_per_native/1000.)
        native_per_work=1000./metres_per_native
    dx=np.asarray(f32(scale[0])*unit_scale,dtype=f32)
    dy=np.asarray(f32(scale[1])*unit_scale,dtype=f32)
    if dx<=0. or dy<=0.:
        raise ValueError('GeoTIFF pixel scale must be positive')
    if orientation not in {1,4}:
        raise ValueError(f'unsupported GeoTIFF orientation {orientation}')
    if orientation==1: dy=np.asarray(-dy,dtype=f32)
    x0=np.asarray(f32(tie[3])*unit_scale,dtype=f32)
    y0=np.asarray(f32(tie[4])*unit_scale,dtype=f32)
    if int(raster_type)==1:
        x0=np.asarray(x0+np.asarray(dx/f32(2.),dtype=f32),dtype=f32)
        y0=np.asarray(y0+np.asarray(dy/f32(2.),dtype=f32),dtype=f32)
    ix=int(float(tie[0])); iy=int(float(tie[1]))
    if ix or iy:
        x0=np.asarray(x0-np.asarray(dx*f32(ix),dtype=f32),dtype=f32)
        y0=np.asarray(y0-np.asarray(dy*f32(iy),dtype=f32),dtype=f32)
    if extra_half_pixel:
        x0=np.asarray(x0+np.asarray(dx/f32(2.),dtype=f32),dtype=f32)
        y0=np.asarray(y0+np.asarray(dy/f32(2.),dtype=f32),dtype=f32)
    return x0,y0,dx,dy,native_per_work


def _accumulate_glazas(path, config, snow=None):
    kind=config.input_type.upper(); spec=GLAZ_SPECS.get(kind)
    if spec is None: raise NotImplementedError(f"unsupported GLAZ input type: {kind}")
    if path.stat().st_size != spec.nx*spec.ny:
        raise ValueError(f"{kind} raster must contain {spec.nx}x{spec.ny} bytes: {path}")
    mesh=config.mesh_density
    if mesh < 1: raise ValueError("mesh_density must be positive")
    g=config.grid; source=np.memmap(path,dtype="u1",mode="r",shape=(spec.ny,spec.nx))
    source_crs=_glaz_crs(spec)
    source_grid=GridConfig(1,1,1.,1.,projection='LAZA',
                           lat0=spec.latitude_origin,lon0=spec.longitude_origin,
                           datum=config.input_datum)
    def reverse(x,y):
        return unproject_to_crs(x,y,g,source_crs,config.input_datum,units='km')
    irange,jrange=_window(g,reverse,spec)
    counts=np.zeros((USGS38.size,g.ny*g.nx),dtype=np.int32)
    columns={int(v):i for i,v in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    read=inside=0
    source_index=np.arange(irange.start,irange.stop,dtype=np.float64)
    xs=np.asarray(np.float64(spec.x_origin_km)+source_index,dtype=np.float64)
    for j in jrange:
        raw=np.asarray(source[spec.ny-1-j,irange.start:irange.stop])
        valid=(raw>=1)&(raw<GLAZ_TO_USGS.size)
        if not valid.any(): continue
        base_x=xs[valid]; classes=GLAZ_TO_USGS[raw[valid]]
        source_y=np.asarray(np.float64(spec.y_origin_km)+np.float64(j),dtype=np.float64)
        mesh_x=_mesh_coordinates(base_x,mesh)
        mesh_y=_mesh_coordinates(source_y,mesh)
        # Fortran loops IMESH outside JMESH, which matters to SETSNOW's
        # historical JP/K state.
        sx=np.broadcast_to(mesh_x[:,:,None],(base_x.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(base_x.size,mesh,mesh)).ravel()
        gx,gy=project_glaz_to_lcc(sx,sy,source_grid,g,units='km')
        center_x,center_y=project_glaz_to_lcc(
            base_x,np.full(base_x.size,source_y,dtype=np.float64),
            source_grid,g,units='km')
        _qa_write(config,'qaglaz.dat',center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,g)
        repeated=np.repeat(classes,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=int(valid.sum()); inside+=int(good.sum())
    return counts.reshape(USGS38.size,g.ny,g.nx),read,inside


def _ctg_record(line,path,number):
    fields=line.split()
    try:
        if len(fields)==4: return tuple(int(value) for value in fields)
        padded=line.ljust(24)
        return (int(padded[0:4]),int(padded[4:7]),
                int(padded[7:15]),int(padded[15:24]))
    except ValueError as error:
        raise ValueError(f'invalid CTG record at {path}:{number}') from error


def _accumulate_ctg(path,config,snow=None):
    """Expand USGS 200 m CTG run records and bin their mesh samples."""
    lines=path.read_text(encoding='ascii',errors='strict').splitlines()
    if len(lines)<6: raise ValueError(f'compressed CTG file is truncated: {path}')
    try: zone=int(lines[0].ljust(55)[50:55])
    except ValueError as error: raise ValueError(f'invalid CTG UTM zone: {path}') from error
    if not 1<=zone<=60: raise ValueError(f'CTG UTM zone must be 1..60: {path}')
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    source_grid=GridConfig(1,1,1.,1.,projection='UTM',utm_zone=zone,
                           hemisphere='N',datum=config.input_datum)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    x_anchor=y_anchor=None; read=inside=0
    for number,line in enumerate(lines[5:],start=6):
        if not line.strip(): continue
        run,category,x_value,y_value=_ctg_record(line,path,number)
        if run<0: raise ValueError(f'negative CTG run length at {path}:{number}')
        if run==0:
            x_anchor=x_value-200; y_anchor=y_value
            continue
        if x_value!=0: continue
        if x_anchor is None: raise ValueError(f'CTG data precede coordinate anchor at {path}:{number}')
        x_m=x_anchor+200*np.arange(1,run+1,dtype=np.int64); x_anchor=int(x_m[-1])
        read+=run
        if not 1<=category<=92: continue
        if category not in columns:
            raise ValueError(f'CTG category {category} is not in the USGS-38 table at {path}:{number}')
        centers_x=np.asarray(np.asarray(x_m,dtype=np.float64)*np.float64(.001),dtype=np.float64)
        center_y=np.asarray(np.float64(y_anchor)*np.float64(.001),dtype=np.float64)
        mesh_x=_mesh_coordinates(centers_x,mesh,np.float64(.2))
        mesh_y=_mesh_coordinates(center_y,mesh,np.float64(.2))
        # IMESH is the outer loop and JMESH the inner loop in LOADOTHER.
        sx=np.broadcast_to(mesh_x[:,:,None],(run,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(run,mesh,mesh)).ravel()
        gx,gy=transform_grid(sx,sy,source_grid,config.grid,units='km')
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        categories=np.full(gx.shape,category,dtype=np.int16)
        categories,coastal=_swap_coastline_categories(categories,gx,gy,config,polygons)
        good=coastal&in_grid
        _qa_write(config,'qamesh.dat',gx[good],gy[good])
        used=good.reshape(run,mesh*mesh).any(axis=1)
        if used.any():
            center_x,center_y=transform_grid(centers_x[used],
                np.full(np.count_nonzero(used),center_y,dtype=np.float64),source_grid,
                config.grid,units='km')
            _qa_write(config,'qactg.dat',center_x,center_y,
                      np.full(np.count_nonzero(used),category))
        _bin_samples(counts,columns,gx,gy,categories,good,config,snow)
        inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside

def _read_generic(path,config):
    """Read CTGPROC ``GENERIC.LANDUSE`` point records and bin them by cell."""
    lines=path.read_text(encoding='ascii',errors='strict').splitlines()
    if not lines or lines[0][:16].strip().upper()!='GENERIC.LANDUSE':
        raise ValueError(f'generic land-use dataset header is invalid: {path}')
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error: raise ValueError(f'invalid generic comment count: {path}') from error
    index=2+comments
    try:
        projection=lines[index][:8].strip().upper(); index+=1
        datum=lines[index][:8].strip(); index+=1
        units=lines[index][:4].strip().upper(); index+=1
        count=int(lines[index].split()[0]); index+=1
    except (IndexError,ValueError) as error:
        raise ValueError(f'incomplete generic land-use header: {path}') from error
    if projection!='LL' or units!='DEG':
        raise ValueError('generic land-use coordinates must use LL projection and DEG units')
    categories=[]
    while len(categories)<count and index<len(lines):
        categories.extend(int(item) for item in lines[index].split()); index+=1
    if len(categories)!=count or len(set(categories))!=count:
        raise ValueError('generic land-use category list is incomplete or contains duplicates')
    records=[]
    for number,line in enumerate(lines[index:],start=index+1):
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=3: raise ValueError(f'invalid generic record at {path}:{number}')
        category=int(fields[0]); longitude=float(fields[1]); latitude=float(fields[2])
        if category not in categories: raise ValueError(f'undeclared land-use category {category} at {path}:{number}')
        if not -90.<=latitude<=90.: raise ValueError(f'invalid latitude at {path}:{number}')
        records.append((category,longitude,latitude))
    counts=np.zeros((count,config.grid.ny,config.grid.nx),dtype=np.int64)
    if not records: return np.asarray(categories,dtype=np.int16),counts,0,0
    values=np.asarray(records,dtype=float)
    # pyproj currently routes one-element ndarrays through its scalar path;
    # explicit sequences keep the vector API stable across NumPy versions.
    x,y=project_geographic(values[:,1].tolist(),values[:,2].tolist(),
                           config.grid,datum,units='km')
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    _qa_write(config,'qageneric.dat',x,y,values[:,0])
    ix,iy,good=_inside_grid(x,y,config.grid)
    lookup={category:i for i,category in enumerate(categories)}
    rows=np.fromiter((lookup[int(value)] for value in values[:,0]),dtype=np.int64,count=len(records))
    np.add.at(counts,(rows[good],iy[good],ix[good]),1)
    return np.asarray(categories,dtype=np.int16),counts,len(records),int(good.sum())


def _read_nzgen(path,config,snow=None):
    """Read New Zealand free-format points and apply the legacy NZCAT table."""
    records=[]
    for number,line in enumerate(path.read_text(encoding='ascii',errors='strict').splitlines(),start=1):
        if not line.strip(): continue
        fields=line.replace(',',' ').split()
        if len(fields)!=4: raise ValueError(f'invalid NZGEN record at {path}:{number}')
        try: _,category,latitude,longitude=(float(value) for value in fields)
        except ValueError as error: raise ValueError(f'invalid NZGEN record at {path}:{number}') from error
        category=int(category)
        if category==0: category=1
        if not 1<=category<NZ_TO_USGS.size or NZ_TO_USGS[category]==0:
            raise ValueError(f'unmapped NZGEN category {category} at {path}:{number}')
        if not -90.<=latitude<=90.: raise ValueError(f'invalid NZGEN latitude at {path}:{number}')
        records.append((int(NZ_TO_USGS[category]),longitude,latitude))
    counts=np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64)
    if not records: return counts,0,0
    values=np.asarray(records,dtype=float)
    x,y=project_geographic(values[:,1].tolist(),values[:,2].tolist(),
                           config.grid,config.input_datum,units='km')
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    _qa_write(config,'qanzgen.dat',x,y)
    ix,iy,in_grid=_inside_grid(x,y,config.grid)
    polygons=_coastline_polygons(config)
    categories,coastal=_swap_coastline_categories(values[:,0],x,y,config,polygons)
    good=coastal&in_grid
    lookup={int(value):index for index,value in enumerate(USGS38)}
    _bin_samples(counts.reshape(USGS38.size,-1),lookup,x,y,categories,good,config,snow)
    return counts,len(records),int(good.sum())


def _nlcd92_state(path):
    match=re.search(r'^(.+?)[_-]nlcd',path.name,re.IGNORECASE)
    if match is None or match.group(1).lower() not in NLCD92_SPECS:
        raise ValueError(f'NLCD92 filename does not identify a legacy state: {path.name}')
    return match.group(1).lower()


def _accumulate_nlcd92(path,config,spec: Nlcd92Spec | None=None,snow=None):
    """Read a headerless state NLCD92 byte grid using the legacy geometry table."""
    spec=NLCD92_SPECS[_nlcd92_state(path)] if spec is None else spec
    if path.stat().st_size!=spec.nx*spec.ny:
        raise ValueError(f'NLCD92 raster must contain {spec.nx}x{spec.ny} bytes: {path}')
    source_grid=GridConfig(1,1,1.,1.,projection='ACEA',lat0=23.,lon0=-96.,
                           lat1=29.5,lat2=45.5,datum=config.input_datum)
    corners_x=np.array([config.grid.x_origin_km,
        config.grid.x_origin_km+config.grid.nx*config.grid.dx_km]*2)
    corners_y=np.repeat([config.grid.y_origin_km,
        config.grid.y_origin_km+config.grid.ny*config.grid.dy_km],2)
    sx,sy=transform_grid(corners_x,corners_y,config.grid,source_grid,units='km')
    ix0=int(np.floor((np.min(sx)-spec.x_origin_km)/.03)); ix1=int(np.ceil((np.max(sx)-spec.x_origin_km)/.03))
    iy0=int(np.floor((spec.y_origin_km-np.max(sy))/.03)); iy1=int(np.ceil((spec.y_origin_km-np.min(sy))/.03))
    buffer=max(abs(ix1-ix0)//10+1,abs(iy1-iy0)//10+1)
    ix0=max(0,ix0-buffer); ix1=min(spec.nx-1,ix1+buffer)
    iy0=max(0,iy0-buffer); iy1=min(spec.ny-1,iy1+buffer)
    if ix0>ix1 or iy0>iy1:
        return np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64),0,0
    source=np.memmap(path,dtype='u1',mode='r',shape=(spec.ny,spec.nx))
    columns={int(value):index for index,value in enumerate(USGS38)}
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    polygons=_coastline_polygons(config)
    x=spec.x_origin_km+.03*np.arange(ix0,ix1+1); read=inside=0
    for j in range(iy0,iy1+1):
        raw=np.asarray(source[j,ix0:ix1+1]); translated=NLCD92_TO_USGS[raw]
        gx,gy=transform_grid(x,np.full(x.size,spec.y_origin_km-.03*j),
                             source_grid,config.grid,units='km')
        _qa_write(config,'qanlcd.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        translated,coastal=_swap_coastline_categories(translated,gx,gy,config,polygons)
        good=(translated>0)&coastal&in_grid
        invalid=(raw>0)&(translated==0)
        if invalid.any(): raise ValueError(f'unmapped NLCD92 category {int(raw[invalid][0])} in {path}')
        _bin_samples(counts,columns,gx,gy,translated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_nlcd_geotiff(path,config,translation,snow=None):
    """Read an NLCD GeoTIFF using its standard CRS and affine tags."""
    with tifffile.TiffFile(path) as source:
        metadata=source.geotiff_metadata
        if metadata is None: raise ValueError(f'NLCD file has no GeoTIFF metadata: {path}')
        epsg=metadata.get('ProjectedCSTypeGeoKey') or metadata.get('GeographicTypeGeoKey')
        if epsg is None: raise NotImplementedError('NLCD GeoTIFF requires an EPSG CRS key')
        scale=metadata.get('ModelPixelScale'); tie=metadata.get('ModelTiepoint')
        if scale is None or tie is None or len(scale)<2 or len(tie)<6:
            raise ValueError(f'NLCD GeoTIFF lacks pixel scale or tiepoint: {path}')
        data=np.asarray(source.asarray())
        raster_type=int(metadata.get('GTRasterTypeGeoKey',1))
        orientation=int(source.pages[0].tags[274].value) if 274 in source.pages[0].tags else 1
    if data.ndim!=2: raise ValueError('NLCD GeoTIFF must contain one two-dimensional band')
    if data.dtype.kind not in {'u','i'}: raise ValueError('NLCD GeoTIFF categories must be integers')
    source_crs=CRS.from_epsg(int(epsg))
    x0,y0,dx,dy,native_per_work=_geotiff_geometry(
        scale,tie,raster_type,source_crs,orientation=orientation)
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    x=_source_centers(x0,dx,np.arange(data.shape[1])); read=inside=0
    for row,raw in enumerate(data):
        if np.any((raw<0)|(raw>=len(translation))):
            raise ValueError(f'NLCD GeoTIFF category is outside the byte lookup table: {path}')
        translated=translation[raw.astype(np.int64,copy=False)]
        invalid=(raw!=0)&(translated==0)
        if invalid.any(): raise ValueError(f'unmapped NLCD category {int(raw[invalid][0])} in {path}')
        nonstandard=~np.isin(translated,[0,*columns])
        if nonstandard.any():
            raise ValueError(f'NLCD category maps outside the USGS-38 table: {int(translated[nonstandard][0])}')
        y=_source_centers(y0,dy,row)
        gx,gy=transformer.transform(
            x*native_per_work,np.full(x.size,y,dtype=np.float64)*native_per_work)
        _qa_write(config,'qanlcd.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        translated,coastal=_swap_coastline_categories(translated,gx,gy,config,polygons)
        good=(translated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,translated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _corine_indices(data,photometric,path):
    """Return one-based CORINE classes for index, class-code, or RGB TIFFs."""
    raw=np.asarray(data)
    if photometric==2:
        if raw.ndim==3 and raw.shape[-1] in {3,4}: rgb=raw[...,:3]
        elif raw.ndim==3 and raw.shape[0] in {3,4}: rgb=np.moveaxis(raw[:3],0,-1)
        else: raise ValueError(f'CORINE RGB GeoTIFF must have three color bands: {path}')
        if rgb.dtype.kind not in {'u','i'} or np.any((rgb<0)|(rgb>255)):
            raise ValueError(f'CORINE RGB values must be bytes: {path}')
        values=(rgb[...,0].astype(np.int32)*1_000_000+
                rgb[...,1].astype(np.int32)*1_000+rgb[...,2].astype(np.int32))
        indices=np.zeros(values.shape,dtype=np.int16)
        for index,value in enumerate(CORINE_RGB,1): indices[values==value]=index
        invalid=indices==0
    else:
        if raw.ndim!=2 or raw.dtype.kind not in {'u','i'}:
            raise ValueError(f'CORINE GeoTIFF must contain one integer category band: {path}')
        values=raw.astype(np.int64,copy=False)
        if photometric==3:
            indices=values.astype(np.int16,copy=False)
            invalid=(values<0)|(values>=len(CORINE_TO_USGS))
        else:
            order=np.argsort(CORINE_CLASSES); sorted_classes=CORINE_CLASSES[order]
            positions=np.searchsorted(sorted_classes,values)
            found=(positions<sorted_classes.size)&(
                sorted_classes[np.minimum(positions,sorted_classes.size-1)]==values)
            indices=np.zeros(values.shape,dtype=np.int16)
            indices[found]=order[positions[found]]+1
            invalid=(values!=0)&~found
    if np.any(invalid):
        value=(values[invalid][0] if values.ndim==2 else values.reshape(-1)[0])
        raise ValueError(f'unmapped CORINE category {value!r} in {path}')
    return indices


def _accumulate_corine(path,config,snow=None):
    """Mesh-sample all three CORINE GeoTIFF category encodings."""
    with tifffile.TiffFile(path) as source:
        metadata=source.geotiff_metadata or {}
        epsg=metadata.get('ProjectedCSTypeGeoKey') or metadata.get('GeographicTypeGeoKey')
        page=source.pages[0]
        scale=(metadata.get('ModelPixelScale') or
               (page.tags.get(33550).value if page.tags.get(33550) else None))
        tie=(metadata.get('ModelTiepoint') or
             (page.tags.get(33922).value if page.tags.get(33922) else None))
        if epsg is None or scale is None or tie is None:
            raise ValueError(f'CORINE GeoTIFF has incomplete georeferencing: {path}')
        photometric=int(page.photometric)
        indices=_corine_indices(source.asarray(),photometric,path)
        raster_type=int(metadata.get('GTRasterTypeGeoKey',1))
        orientation=int(page.tags[274].value) if 274 in page.tags else 1
    source_crs=CRS.from_epsg(int(epsg))
    x0,y0,dx,dy,native_per_work=_geotiff_geometry(
        scale,tie,raster_type,source_crs,orientation=orientation)
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config); read=inside=0
    centers=_source_centers(x0,dx,np.arange(indices.shape[1]))
    for row,source_indices in enumerate(indices):
        translated=CORINE_TO_USGS[source_indices]
        center_y=_source_centers(y0,dy,row)
        mesh_x=_mesh_coordinates(centers,mesh,dx)
        mesh_y=_mesh_coordinates(center_y,mesh,dy)
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=transformer.transform(sx*native_per_work,sy*native_per_work)
        qa_x,qa_y=transformer.transform(
            centers*native_per_work,
            np.full(centers.size,center_y,dtype=np.float64)*native_per_work)
        _qa_write(config,'qacorine.dat',qa_x,qa_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=source_indices.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_global_geotiff(path,config,translation,*,offset=0,missing=(),
                               glc_shift=False,fallback_epsg=None,snow=None):
    """Mesh-sample a global categorical GeoTIFF as CTGPROC does."""
    with tifffile.TiffFile(path) as source:
        metadata=source.geotiff_metadata or {}
        epsg=(metadata.get('ProjectedCSTypeGeoKey') or
              metadata.get('GeographicTypeGeoKey') or fallback_epsg)
        page=source.pages[0]
        scale=(metadata.get('ModelPixelScale') or
               (page.tags.get(33550).value if page.tags.get(33550) else None))
        tie=(metadata.get('ModelTiepoint') or
             (page.tags.get(33922).value if page.tags.get(33922) else None))
        if epsg is None or scale is None or tie is None:
            raise ValueError(f'global land-use GeoTIFF has incomplete georeferencing: {path}')
        data=np.asarray(source.asarray())
        raster_type=int(metadata.get('GTRasterTypeGeoKey',2 if glc_shift else 1))
        orientation=int(page.tags[274].value) if 274 in page.tags else 1
    if data.ndim!=2 or data.dtype.kind not in {'u','i'}:
        raise ValueError('global land-use GeoTIFF must be one integer band')
    source_crs=CRS.from_epsg(int(epsg))
    x0,y0,dx,dy,native_per_work=_geotiff_geometry(
        scale,tie,raster_type,source_crs,orientation=orientation,
        extra_half_pixel=glc_shift)
    transformer=Transformer.from_crs(source_crs,_lcc(config.grid),always_xy=True)
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    read=inside=0
    for row,raw in enumerate(data):
        raw=np.asarray(raw,dtype=np.int64); source_category=raw+offset
        absent=np.isin(raw,tuple(missing))
        invalid=(source_category<0)|(source_category>=len(translation))
        if np.any(invalid&~absent):
            raise ValueError(f'global land-use category is outside its lookup table: {path}')
        translated=np.zeros(raw.shape,dtype=np.int16)
        valid=~invalid&~absent; translated[valid]=translation[source_category[valid]]
        if np.any((translated==0)&valid):
            raise ValueError(f'unmapped global land-use category in {path}')
        centers=_source_centers(x0,dx,np.arange(data.shape[1]))
        center_y=_source_centers(y0,dy,row)
        mesh_x=_mesh_coordinates(centers,mesh,dx)
        mesh_y=_mesh_coordinates(center_y,mesh,dy)
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=transformer.transform(sx*native_per_work,sy*native_per_work)
        center_x,center_y=transformer.transform(
            centers*native_per_work,
            np.full(centers.size,center_y,dtype=np.float64)*native_per_work)
        qa_name='qagcl2k.dat' if config.input_type.upper()=='GLC2K' else 'qaumdglc.dat'
        _qa_write(config,qa_name,center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside


def _accumulate_modis(path,config,spec: ModisSpec | None=None,snow=None):
    """Window and mesh-sample the raw BU MODIS global byte grid."""
    spec=MODIS_SPEC if spec is None else spec
    if path.stat().st_size!=spec.nx*spec.ny:
        raise ValueError(f'MODIS raster must contain {spec.nx}x{spec.ny} bytes: {path}')
    corners_x=np.array([config.grid.x_origin_km,
        config.grid.x_origin_km+config.grid.nx*config.grid.dx_km]*2)
    corners_y=np.repeat([config.grid.y_origin_km,
        config.grid.y_origin_km+config.grid.ny*config.grid.dy_km],2)
    lon,lat=unproject_geographic(corners_x,corners_y,config.grid,
                                 config.input_datum,units='km')
    ix0=int(np.floor((np.min(lon)-spec.x_origin)/spec.spacing))
    ix1=int(np.ceil((np.max(lon)-spec.x_origin)/spec.spacing))
    iy0=int(np.floor((spec.y_origin-np.max(lat))/spec.spacing))
    iy1=int(np.ceil((spec.y_origin-np.min(lat))/spec.spacing))
    buffer=max(abs(ix1-ix0)//10+1,abs(iy1-iy0)//10+1)
    ix0=max(0,ix0-buffer); ix1=min(spec.nx-1,ix1+buffer)
    iy0=max(0,iy0-buffer); iy1=min(spec.ny-1,iy1+buffer)
    counts=np.zeros((USGS38.size,config.grid.ny*config.grid.nx),dtype=np.int64)
    if ix0>ix1 or iy0>iy1: return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),0,0
    source=np.memmap(path,dtype='u1',mode='r',shape=(spec.ny,spec.nx))
    mesh=config.mesh_density
    if mesh<1: raise ValueError('mesh_density must be positive')
    columns={int(value):index for index,value in enumerate(USGS38)}
    polygons=_coastline_polygons(config)
    centers=_source_centers(spec.x_origin,spec.spacing,
                            np.arange(ix0,ix1+1)); read=inside=0
    for row in range(iy0,iy1+1):
        raw=np.asarray(source[row,ix0:ix1+1]); valid=(raw>=1)&(raw<=18)
        translated=np.zeros(raw.shape,dtype=np.int16); translated[valid]=GLAZ_TO_USGS[raw[valid]]
        mesh_x=_mesh_coordinates(centers,mesh,np.float64(spec.spacing))
        center_y=_source_centers(spec.y_origin,-spec.spacing,row)
        mesh_y=_mesh_coordinates(center_y,mesh,np.float64(-spec.spacing))
        sx=np.broadcast_to(mesh_x[:,:,None],(centers.size,mesh,mesh)).ravel()
        sy=np.broadcast_to(mesh_y[None,None,:],(centers.size,mesh,mesh)).ravel()
        gx,gy=project_geographic(sx,sy,config.grid,config.input_datum,units='km')
        center_x,center_y=project_geographic(
            centers,np.full(centers.size,center_y,dtype=np.float64),
            config.grid,config.input_datum,units='km')
        # CTGPROC 7.0 writes MODIS centers to unit 42 (qaglaz.dat), while
        # qamodis.dat is opened but remains empty.
        _qa_write(config,'qaglaz.dat',center_x,center_y)
        _qa_write(config,'qamesh.dat',gx,gy)
        ix,iy,in_grid=_inside_grid(gx,gy,config.grid)
        repeated=np.repeat(translated,mesh*mesh)
        repeated,coastal=_swap_coastline_categories(repeated,gx,gy,config,polygons)
        good=(repeated>0)&coastal&in_grid
        _bin_samples(counts,columns,gx,gy,repeated,good,config,snow)
        read+=raw.size; inside+=int(good.sum())
    return counts.reshape(USGS38.size,config.grid.ny,config.grid.nx),read,inside

def _read_previous_landuse(path,grid):
    """Read RDGRID continuation counts from NetCDF or legacy LU.DAT."""
    source=Path(path)
    if source.suffix.lower() in {'.nc','.nc4','.cdf'}:
        with Dataset(source) as dataset:
            file_type=str(getattr(dataset,'landuse_file_type','')).upper()
            if file_type!='HITS': raise ValueError('previous CTGPROC NetCDF must have HITS file type')
            categories=np.asarray(dataset['landuse_category'][:],dtype=np.int16)
            counts=np.asarray(dataset['sample_count'][:],dtype=np.int64)
            shape=(grid.ny,grid.nx)
            if counts.ndim!=3 or counts.shape!=(categories.size,*shape):
                raise ValueError('previous land-use grid dimensions do not match')
            for name,wanted in {'grid_nx':grid.nx,'grid_ny':grid.ny}.items():
                if name in dataset.ncattrs() and int(getattr(dataset,name))!=wanted:
                    raise ValueError('previous land-use grid dimensions do not match')
            geometry={'grid_dx_km':grid.dx_km,'grid_dy_km':grid.dy_km,
                      'x_origin_km':grid.x_origin_km,'y_origin_km':grid.y_origin_km}
            if any(name in dataset.ncattrs() and
                   not np.isclose(float(getattr(dataset,name)),wanted,rtol=1.e-4,atol=0.)
                   for name,wanted in geometry.items()):
                raise ValueError('previous land-use grid geometry does not match')
            if ('projection' in dataset.ncattrs() and
                    str(dataset.projection).strip().upper()!=grid.projection.upper()):
                raise ValueError('previous land-use projection does not match')
            if ('datum' in dataset.ncattrs() and
                    normalize_datum(str(dataset.datum))!=normalize_datum(grid.datum)):
                raise ValueError('previous land-use datum does not match')
            if np.any(counts<0): raise ValueError('previous land-use counts must be non-negative')
        return categories,counts
    lines=source.read_text(encoding='ascii',errors='strict').splitlines()
    if len(lines)<6 or not lines[0][:16].strip().upper().startswith('LU.DAT'):
        raise ValueError(f'previous CTGPROC file has an invalid header: {source}')
    try: comments=int(lines[1].split()[0])
    except (IndexError,ValueError) as error:
        raise ValueError(f'previous CTGPROC file has an invalid comment count: {source}') from error
    index=2+comments
    if index>=len(lines) or lines[index].strip().upper()!='HITS':
        raise ValueError('previous CTGPROC LU.DAT must have HITS file type')
    index+=1
    projection=lines[index][:8].strip().upper(); index+=1
    if projection!=grid.projection.upper(): raise ValueError('previous land-use projection does not match')
    index+=1  # projection parameters
    if projection in {'LCC','LAZA','TTM'}: index+=1
    datum=lines[index][:8].strip().upper(); index+=1
    if normalize_datum(datum)!=normalize_datum(grid.datum):
        raise ValueError('previous land-use datum does not match')
    fields=lines[index].split(); index+=1
    if len(fields)<7: raise ValueError('previous land-use grid header is incomplete')
    nx,ny=int(fields[0]),int(fields[1]); category_count=int(fields[6])
    if (nx,ny)!=(grid.nx,grid.ny): raise ValueError('previous land-use grid dimensions do not match')
    expected=(grid.x_origin_km,grid.y_origin_km,grid.dx_km,grid.dy_km)
    if not np.allclose([float(value) for value in fields[2:6]],expected,rtol=1.e-4,atol=0.):
        raise ValueError('previous land-use grid geometry does not match')
    index+=1  # KM units
    categories=np.fromstring(lines[index],sep=' ',dtype=np.int16); index+=1
    if categories.size!=category_count: raise ValueError('previous land-use category list is incomplete')
    counts=np.zeros((category_count,grid.ny,grid.nx),dtype=np.int64)
    seen=np.zeros((grid.ny,grid.nx),dtype=bool)
    for number,line in enumerate(lines[index:],start=index+1):
        values=np.fromstring(line[:10+5*category_count],sep=' ',dtype=np.int64)
        if values.size<2+category_count:
            raise ValueError(f'invalid previous land-use record at {source}:{number}')
        i,j=int(values[0])-1,int(values[1])-1
        if not (0<=i<grid.nx and 0<=j<grid.ny) or seen[j,i]:
            raise ValueError(f'invalid previous land-use grid index at {source}:{number}')
        counts[:,j,i]=values[2:2+category_count]; seen[j,i]=True
    if not seen.all(): raise ValueError('previous land-use file does not contain every grid cell')
    if np.any(counts<0): raise ValueError('previous land-use counts must be non-negative')
    return categories,counts


def _snow_rows(snow,categories):
    if snow is None:
        return (),np.empty(0,dtype=np.int64)
    grouped,totals=snow.records(categories)
    rows=[]
    for (cell,land),items in sorted(grouped.items()):
        for snow_grid,count in items:
            fraction=np.asarray(np.float64(count)/np.float64(totals[cell]),dtype=np.float64)
            rows.append((cell,land,snow_grid,fraction))
    return rows,totals


def _write_netcdf(path,fractions,counts,g,categories=USGS38,*,final_output=True,
                  snow=None,qa_statistics=None):
    with Dataset(path,"w",format="NETCDF4") as ds:
        ds.createDimension("category",len(categories)); ds.createDimension("y",g.ny); ds.createDimension("x",g.nx)
        ds.Conventions="CF-1.8"; ds.source="PyCALMET CTGPROC NumPy port"; ds.projection=g.projection; ds.datum=g.datum
        ds.landuse_file_type='FRACTION' if final_output else 'HITS'
        ds.grid_nx=g.nx; ds.grid_ny=g.ny; ds.grid_dx_km=g.dx_km; ds.grid_dy_km=g.dy_km
        ds.x_origin_km=g.x_origin_km; ds.y_origin_km=g.y_origin_km
        ds.false_easting_km=g.false_easting_km; ds.false_northing_km=g.false_northing_km
        ds.ttm_scale=g.ttm_scale
        ds.createVariable("landuse_category","i2",("category",))[:]=categories
        ds.createVariable("landuse_fraction","f4",("category","y","x"),zlib=True)[:]=fractions
        ds.createVariable("sample_count","i4",("category","y","x"),zlib=True)[:]=counts
        if qa_statistics is not None:
            average,threshold,low=qa_statistics
            ds.average_landuse_hits=int(average)
            ds.low_hit_threshold=int(threshold)
            ds.low_hit_cell_count=int(low)
        rows,_=_snow_rows(snow,categories)
        if snow is not None:
            ds.createDimension('snow_record',len(rows))
            ds.snow_grid_information='CTGPROC sparse SNODAS mapping'
            values=np.asarray(rows,dtype=float).reshape((-1,4))
            ds.createVariable('snow_cell_index','i4',('snow_record',))[:]=values[:,0].astype(np.int32)+1
            ds.createVariable('snow_landuse_index','i2',('snow_record',))[:]=values[:,1].astype(np.int16)
            ds.createVariable('snow_grid_index','i4',('snow_record',))[:]=values[:,2].astype(np.int32)
            ds.createVariable('snow_fraction','f4',('snow_record',))[:]=values[:,3]

def _write_legacy(path,fractions,counts,g,categories=USGS38,*,final_output=True,
                  snow=None):
    with path.open("w",encoding="ascii") as s:
        file_type='FRACTION' if final_output else 'HITS'
        s.write(f"{'LU.DAT':<16}{'2.1':<16}"
                f"{'Coordinate parameters and LandUse categories':<64}\n"
                "   2\nProduced by PyCALMET CTGPROC NumPy implementation\n"
                "Internal Coordinate Transformations  ---  pyproj\n"+file_type+"\n")
        projection=g.projection.upper()
        s.write(f"{projection:<8}\n")
        latitude=lambda value: f'{abs(value):g}{"N" if value>=0 else "S"}'
        longitude=lambda value: f'{abs(value):g}{"E" if value>=0 else "W"}'
        if projection=='UTM':
            s.write(f'{g.utm_zone:4d}{g.hemisphere:<4}\n')
        else:
            parameters=[latitude(g.lat0),longitude(g.lon0)]
            if projection in {'LCC','PS'}: parameters.append(latitude(g.lat1))
            if projection=='LCC': parameters.append(latitude(g.lat2))
            s.write(''.join(f'{value:<16}' for value in parameters)+'\n')
        if projection in {'LCC','LAZA','TTM'}:
            s.write(gfortran_real_pair(g.false_easting_km,g.false_northing_km))
        s.write(f"{g.datum:<8}{NIMA_DATABASE_DATE:<12}\n")
        s.write(f"{g.nx:8d}{g.ny:8d}{g.x_origin_km:12.3f}{g.y_origin_km:12.3f}{g.dx_km:12.3f}{g.dy_km:12.3f}{len(categories):6d}\nKM  \n")
        width=6 if final_output else 5
        s.write("          "+"".join(f"{v:{width}d}" for v in categories)+"\n")
        for j in range(g.ny):
            for i in range(g.nx):
                v=fractions[:,j,i]; flag="   .                " if v.sum() else "   Missing          "
                if final_output:
                    fields="".join(f"{x:6.3f}" for x in v)
                else:
                    fields="".join(f"{int(x):5d}" for x in counts[:,j,i])
                s.write(f"{i+1:5d}{j+1:5d}"+fields+flag+"\n")
        if snow is not None:
            grouped,totals=snow.records(categories)
            s.write('SNOW Grid Information:\n')
            for j in range(g.ny):
                for i in range(g.nx):
                    cell=j*g.nx+i
                    lands=sorted(land for candidate,land in grouped if candidate==cell)
                    fields=f'{i+1:4d}{j+1:4d}{len(lands):3d}'
                    for land_number,land in enumerate(lands):
                        items=grouped[cell,land]
                        separator='' if land_number==0 else ' '
                        fields+=f'{separator}{land:3d}{len(items):4d}'
                        for snow_grid,count in items:
                            fields+=f' {snow_grid:9d}{count/totals[cell]:6.3f}'
                    s.write(fields+'\n')


def _fractions_from_counts(counts):
    """Reproduce CTGPROC's NGRID/FLOAT(ISUM) floating-point division."""
    hits=np.asarray(counts,dtype=np.int64)
    totals=hits.sum(axis=0,dtype=np.int64)
    numerator=np.asarray(hits,dtype=np.float64)
    denominator=np.asarray(totals,dtype=np.float64)
    fractions=np.zeros(hits.shape,dtype=np.float64)
    np.divide(numerator,denominator,out=fractions,where=totals>0)
    return totals,fractions


def _low_hit_threshold(average_hits,threshold_percent):
    """Reproduce ``MAX(1,INT(REAL(IAVG*ITHRESH)/100.))``."""
    product=int(average_hits)*int(threshold_percent)
    fraction=np.asarray(
        np.float64(product)/np.float64(100.),dtype=np.float64)
    return max(1,int(fraction))


def run(config: CtgprocConfig) -> LanduseResult:
    if not 0<=config.qa_hit_threshold_percent<=100:
        raise ValueError('qa_hit_threshold_percent/ITHRESH must be between 0 and 100')
    output_format=config.output_format.lower()
    if output_format not in {'auto','legacy','netcdf'}:
        raise ValueError("output_format must be 'auto', 'legacy', or 'netcdf'")
    if config.snow_processing and not config.final_output:
        raise ValueError('legacy CTGPROC permits LSNOW only in the final run')
    if config.snow_processing and config.category_map:
        raise ValueError('snow processing cannot be combined with category_map')
    if config.qa_cell_output:
        directory=_qa_directory(config)
        directory.mkdir(parents=True,exist_ok=True)
        for name in QA_FILENAMES:
            with (directory/name).open('wb'):
                pass
    if config.gshhs_file is not None:
        if config.coastline_file is None:
            raise ValueError('GSHHS extraction requires coastline_file/COASTBLN output')
        extract_gshhs(config.gshhs_file,config.coastline_file,config.grid,
                      source_datum=config.coastline_source_datum,
                      wdbii_datum=config.coastline_wdbii_datum)
    supported={*GLAZ_SPECS,'CTG','NZGEN','NLCD92','NLCD01','NLCDTF',
               'CORINE','GLC2K','UMDGLC','MODIS','GEN'}
    if config.landuse_sources:
        sources=tuple(config.landuse_sources)
    else:
        sources=tuple(CtgprocSource(path,config.input_type,config.input_datum,
                                    config.mesh_density)
                      for path in config.landuse_files)
    if not sources: raise ValueError('CTGPROC requires at least one land-use data file')
    sources=tuple(sorted(enumerate(sources),key=lambda item:
        (PROCESS_ORDER.get(item[1].input_type.upper(),999),item[0])))
    sources=tuple(item[1] for item in sources)
    kinds={source.input_type.upper() for source in sources}
    unknown=sorted(kinds-supported)
    if unknown: raise NotImplementedError(f'unsupported CTGPROC input types: {unknown}')
    glaz_count=sum(source.input_type.upper() in GLAZ_SPECS for source in sources)
    if glaz_count>1:
        selected=_select_glaz_regions(config.grid)
        sources=tuple(source for source in sources
                      if source.input_type.upper() not in GLAZ_SPECS or
                      source.input_type.upper() in selected)
    if 'GEN' in kinds and (config.coastline_swap or config.fill_empty_ocean):
        raise ValueError('legacy CTGPROC does not support coastline processing for GENERIC data')
    if config.snow_processing and 'GEN' in kinds:
        raise ValueError('legacy CTGPROC does not support snow processing for GENERIC data')
    if config.previous_landuse is not None and kinds&({*GLOBAL_MERGE_TYPES,'GEN'}):
        raise ValueError('legacy CTGPROC requires global and GENERIC data in the first continuation run')
    if config.coastline_file is not None and not config.fill_empty_ocean:
        if not config.coastline_swap:
            raise ValueError('coastline processing requires LMARFILL or LMARSWAP')
    if config.previous_landuse is not None:
        categories,counts=_read_previous_landuse(config.previous_landuse,config.grid)
    else:
        categories=counts=None
    if not sources and categories is None:
        categories=USGS38
        counts=np.zeros((USGS38.size,config.grid.ny,config.grid.nx),dtype=np.int64)
    read=inside=0
    snow=SnowGridInfo.create(config.grid) if config.snow_processing else None
    for source in sources:
        kind=source.input_type.upper(); item=Path(source.path)
        active=replace(config,input_type=kind,input_datum=source.input_datum,
                       mesh_density=source.mesh_density,landuse_sources=())
        if kind=='GEN':
            file_categories,partial,file_read,file_inside=_read_generic(item,active)
            if categories is None: categories=file_categories; counts=np.zeros_like(partial)
            elif not np.array_equal(categories,file_categories): raise ValueError('generic files must declare identical category lists')
        elif kind=='NZGEN':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _read_nzgen,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='NLCD92':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_nlcd92,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind in {'NLCD01','NLCDTF'}:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38
            table=NLCD01_TO_USGS if kind=='NLCD01' else NLCD92_TO_USGS
            partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_nlcd_geotiff,item,active,table,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='CORINE':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_corine,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind in {'GLC2K','UMDGLC'}:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38
            if kind=='GLC2K':
                    partial,file_read,file_inside=_read_with_optional_snow(
                        _accumulate_global_geotiff,item,active,GLC2K_TO_USGS,
                        missing=(24,),glc_shift=True,fallback_epsg=4326,snow=snow)
            else:
                    partial,file_read,file_inside=_read_with_optional_snow(
                        _accumulate_global_geotiff,item,active,UMDGLC_TO_USGS,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='MODIS':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_modis,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        elif kind=='CTG':
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_ctg,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        else:
            if categories is not None and not np.array_equal(categories,USGS38):
                raise ValueError('previous land-use categories do not match USGS-38 input')
            categories=USGS38; partial,file_read,file_inside=_read_with_optional_snow(
                _accumulate_glazas,item,active,snow=snow)
            if counts is None: counts=np.zeros(partial.shape,dtype=np.int64)
        if kind in GLOBAL_MERGE_TYPES:
            _merge_global_counts(counts,partial,categories)
        else:
            counts+=partial
        read+=file_read; inside+=file_inside
    raw_totals=counts.sum(axis=0,dtype=np.int64)
    average_hits=int(raw_totals.sum(dtype=np.int64)//(config.grid.nx*config.grid.ny))
    low_hit_threshold=_low_hit_threshold(
        average_hits,config.qa_hit_threshold_percent)
    low_hit_cells=int(np.count_nonzero(raw_totals<low_hit_threshold))
    if config.fill_empty_ocean and config.final_output:
        _fill_empty_ocean(counts,categories,config)
    if config.category_map:
        categories,counts=_remap_categories(categories,counts,config.category_map)
    totals,fractions=_fractions_from_counts(counts)
    output=Path(config.output); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+".",delete=False) as tmp: temporary=Path(tmp.name)
    try:
        netcdf=(output_format=='netcdf' or
                (output_format=='auto' and output.suffix.lower() in {'.nc','.nc4','.cdf'}))
        (_write_netcdf(temporary,fractions,counts,config.grid,categories,
                       final_output=config.final_output,snow=snow,
                       qa_statistics=(average_hits,low_hit_threshold,low_hit_cells))
         if netcdf
         else _write_legacy(temporary,fractions,counts,config.grid,categories,
                            final_output=config.final_output,snow=snow))
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    return LanduseResult(output,read,inside,int((totals>0).sum()),average_hits,
                         low_hit_threshold,low_hit_cells)
