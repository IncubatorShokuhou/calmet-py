"""Reader and canonical-NetCDF adapter for CALMET GEO.DAT 2.1 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import netCDF4 as nc
import numpy as np


_NUMBER=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?')
_DEFAULT_CATEGORIES=np.array([10,20,-20,30,40,50,54,55,60,61,62,70,80,90],dtype=np.int16)
_DEFAULT_FIELDS=(
    (1.,.25,.25,.05,1.,.001,.001,.001,1.,1.,.20,.05,.20,.05),
    (.18,.15,.15,.25,.10,.10,.10,.10,.10,.10,.10,.30,.30,.70),
    (1.5,1.,.5,1.,1.,0.,0.,0.,.5,.5,.1,1.,.5,.5),
    (.25,.15,.15,.15,.15,1.,1.,1.,.25,.25,.25,.15,.15,.15),
    (0.,)*14,
    (.2,3.,3.,.5,7.,0.,0.,0.,2.,2.,1.,.05,0.,0.),
)


@dataclass(frozen=True,slots=True)
class GeoData:
    projection: str
    datum: str
    nx: int
    ny: int
    x_origin_km: float
    y_origin_km: float
    dx_km: float
    dy_km: float
    latitude_origin: float
    longitude_origin: float
    standard_parallel_1: float
    standard_parallel_2: float
    false_easting_km: float
    false_northing_km: float
    utm_zone: int
    hemisphere: str
    water_min: int
    water_max: int
    categories: np.ndarray
    terrain: np.ndarray
    fields: tuple[np.ndarray,...]


def _numbers(line):
    return [float(value.replace('D','E').replace('d','e')) for value in _NUMBER.findall(line)]


def _angle(text):
    token=text.strip().upper()
    if not token: return 0.
    sign=-1. if token[-1:] in {'S','W'} else 1.
    return sign*float(token.rstrip('NSEW'))


def _values(stream,count):
    values=[]
    while len(values)<count:
        line=stream.readline()
        if not line: raise EOFError('GEO.DAT ended inside a numeric record')
        values.extend(_numbers(line))
    if len(values)!=count: raise ValueError('GEO.DAT numeric record contains extra values')
    return values


def _grid(stream,nx,ny,dtype=float):
    rows=[_values(stream,nx) for _ in range(ny)]
    return np.asarray(rows[::-1],dtype=dtype)


class GeoReader:
    def __init__(self,path): self.path=Path(path)

    def read(self):
        with self.path.open('r',encoding='ascii',errors='strict') as stream:
            if stream.readline()[:16].strip().upper()!='GEO.DAT':
                raise ValueError('geophysical input is not a GEO.DAT file')
            comments=int(stream.readline())
            for _ in range(comments): stream.readline()
            projection=stream.readline().strip().upper()
            lat0=lon0=lat1=lat2=0.; zone=0; hemisphere='N'
            if projection=='UTM':
                line=stream.readline()
                # WRTHEAD uses FORMAT(I4,A4), so the zone and hemisphere are
                # adjacent (for example ``"  12S   "``), not whitespace
                # separated fields.
                zone=int(line[:4]); hemisphere=line[4:8].strip()[:1].upper()
                if not hemisphere: raise ValueError('GEO.DAT UTM hemisphere is missing')
            else:
                line=stream.readline()
                widths=[line[index:index+16] for index in range(0,64,16)]
                lat0=_angle(widths[0]); lon0=_angle(widths[1])
                if projection in {'LCC','PS'}: lat1=_angle(widths[2])
                if projection=='LCC': lat2=_angle(widths[3])
            if projection in {'LCC','LAZA','TTM'}:
                false_easting,false_northing=_numbers(stream.readline())[:2]
            else: false_easting=false_northing=0.
            datum=stream.readline()[:8].strip().upper()
            grid=_numbers(stream.readline())
            if len(grid)<6: raise ValueError('GEO.DAT grid record is incomplete')
            nx,ny=int(grid[0]),int(grid[1]); x0,y0,dx,dy=grid[2:6]
            if stream.readline()[:2].strip().upper()!='KM':
                raise ValueError('GEO.DAT horizontal units must be KM')
            land_option=int(_numbers(stream.readline())[0])
            if land_option==0:
                categories=_DEFAULT_CATEGORIES.copy(); water_min=water_max=55
            elif land_option==1:
                values=_numbers(stream.readline())
                if len(values)<3: raise ValueError('GEO.DAT land-use declaration is incomplete')
                count,water_min,water_max=map(int,values[:3])
                categories=np.asarray(_values(stream,count),dtype=np.int16)
            else: raise ValueError(f'unsupported GEO.DAT land-use option {land_option}')
            landuse=_grid(stream,nx,ny,np.int16)
            terrain_line=_numbers(stream.readline())
            if not terrain_line: raise ValueError('GEO.DAT lacks HTFAC')
            terrain=_grid(stream,nx,ny)*terrain_line[0]
            outputs=[]
            for defaults in _DEFAULT_FIELDS:
                option_line=_numbers(stream.readline())
                if not option_line: raise ValueError('GEO.DAT lacks a geophysical option')
                option=int(option_line[0])
                if option==2: field=_grid(stream,nx,ny)
                else:
                    table=np.zeros(categories.size,dtype=float)
                    if option==0: table[:min(len(defaults),table.size)]=defaults[:table.size]
                    elif option==1:
                        for index,category in enumerate(categories):
                            record=_values(stream,2)
                            if int(record[0])!=category:
                                raise ValueError('GEO.DAT category table does not match land-use categories')
                            table[index]=record[1]
                    else: raise ValueError(f'invalid GEO.DAT geophysical option {option}')
                    lookup={int(category):table[index] for index,category in enumerate(categories)}
                    try: field=np.vectorize(lookup.__getitem__,otypes=[float])(landuse)
                    except KeyError as error: raise ValueError(f'GEO.DAT land-use category {error.args[0]} is absent from table') from error
                outputs.append(np.asarray(field,dtype=np.float32))
        return GeoData(projection,datum,nx,ny,x0,y0,dx,dy,lat0,lon0,lat1,lat2,
                       false_easting,false_northing,zone,hemisphere,water_min,water_max,
                       categories,np.asarray(terrain,dtype=np.float32),
                       (np.asarray(landuse,dtype=np.int16),*outputs))

    def to_netcdf(self,path,*,compression=1):
        data=self.read(); output=Path(path); output.parent.mkdir(parents=True,exist_ok=True)
        names=('dominant_landuse','roughness_length','albedo','bowen_ratio',
               'soil_heat_flux_parameter','anthropogenic_heat_flux','leaf_area_index')
        with nc.Dataset(output,'w',format='NETCDF4') as dataset:
            dataset.createDimension('y',data.ny); dataset.createDimension('x',data.nx)
            dataset.projection=data.projection; dataset.datum=data.datum
            dataset.grid_nx=data.nx; dataset.grid_ny=data.ny
            dataset.grid_dx_km=data.dx_km; dataset.grid_dy_km=data.dy_km
            dataset.x_origin_km=data.x_origin_km; dataset.y_origin_km=data.y_origin_km
            dataset.latitude_origin=data.latitude_origin; dataset.longitude_origin=data.longitude_origin
            dataset.standard_parallel_1=data.standard_parallel_1
            dataset.standard_parallel_2=data.standard_parallel_2
            dataset.false_easting_km=data.false_easting_km
            dataset.false_northing_km=data.false_northing_km
            if data.projection=='UTM':
                dataset.utm_zone=data.utm_zone; dataset.hemisphere=data.hemisphere
            dataset.water_category_min=data.water_min; dataset.water_category_max=data.water_max
            dataset.landuse_categories=','.join(str(int(value)) for value in data.categories)
            dataset.createVariable('terrain_height','f4',('y','x'),zlib=compression>0,
                                   complevel=compression)[:]=data.terrain
            for name,values in zip(names,data.fields):
                kind='i2' if name=='dominant_landuse' else 'f4'
                dataset.createVariable(name,kind,('y','x'),zlib=compression>0,
                                       complevel=compression)[:]=values
        return output
