"""Reader for CALWRF 2.0 formatted 3D.DAT/m3d regression files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import netCDF4 as nc
import numpy as np


@dataclass(frozen=True, slots=True)
class M3DHeader:
    projection: str
    latitude_origin: float
    longitude_origin: float
    standard_parallel_1: float
    standard_parallel_2: float
    x_origin_km: float
    y_origin_km: float
    spacing_km: float
    nx: int
    ny: int
    nz: int
    nhours: int
    start: int
    sigma: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    terrain: np.ndarray
    landuse: np.ndarray
    source_nx: int = 0
    source_ny: int = 0
    source_nz: int = 0
    i_start: int = 1
    j_start: int = 1
    k_start: int = 1


@dataclass(frozen=True, slots=True)
class M3DCell:
    date: int
    i: int
    j: int
    surface: np.ndarray
    levels: np.ndarray
    water: np.ndarray


def _ints(line: bytes) -> list[int]:
    return [int(x) for x in line.split()]


def _read_header(stream: BinaryIO) -> M3DHeader:
    stream.readline()  # format/version
    ncomment = int(stream.readline())
    for _ in range(ncomment):
        stream.readline()
    stream.readline()  # six output switches
    grid = stream.readline().split()
    projection = grid[0].decode("ascii")
    lat0,lon0,lat1,lat2,x0,y0,spacing=map(float,grid[1:8])
    source_nx, source_ny, source_nz = map(int, grid[8:11])
    stream.readline()  # physics flags
    period = _ints(stream.readline())
    if len(period)<5:
        raise ValueError('M3D period record is incomplete')
    start, nhours, nx, ny, nz = period[:5]
    bounds=stream.readline().split()
    if len(bounds)<6:
        raise ValueError('M3D extraction-bounds record is incomplete')
    i_start,j_start,i_end,j_end,k_start,k_end=map(int,bounds[:6])
    if ((i_end-i_start+1,j_end-j_start+1,k_end-k_start+1)!=(nx,ny,nz)):
        raise ValueError('M3D extraction bounds do not match subset dimensions')
    sigma = np.array([float(stream.readline()) for _ in range(nz)], dtype="f4")
    latitude=np.empty((ny,nx),dtype='f4'); longitude=np.empty_like(latitude)
    terrain=np.empty_like(latitude); landuse=np.empty((ny,nx),dtype='i2')
    for _ in range(nx * ny):
        fields=stream.readline().split()
        if len(fields)<6: raise ValueError('invalid M3D static-cell record')
        i,j=int(fields[0])-i_start,int(fields[1])-j_start
        if not (0<=i<nx and 0<=j<ny): raise ValueError('M3D static-cell index is outside grid')
        latitude[j,i]=float(fields[2]); longitude[j,i]=float(fields[3])
        terrain[j,i]=float(fields[4]); landuse[j,i]=int(fields[5])
    return M3DHeader(projection,lat0,lon0,lat1,lat2,x0,y0,spacing,
                     nx,ny,nz,nhours,start,sigma,latitude,longitude,terrain,landuse,
                     source_nx,source_ny,source_nz,i_start,j_start,k_start)


def _surface(line: bytes) -> tuple[int, int, int, np.ndarray]:
    # Fortran 221: i10,2i3,f7.1,f5.2,i2,3f8.1,f8.2,3f8.1
    date = int(line[0:10]); i = int(line[10:13]); j = int(line[13:16])
    values = np.array([
        float(line[16:23]), float(line[23:28]), float(line[28:30]),
        float(line[30:38]), float(line[38:46]), float(line[46:54]),
        float(line[54:62]), float(line[62:70]), float(line[70:78]),
        float(line[78:86]),
    ], dtype="f4")
    return date, i, j, values


def _level(line: bytes) -> np.ndarray:
    # Basic section of Fortran 222: i4,i6,f6.1,i4,f5.1,f6.2,i3,f5.2.
    # Remaining compressed cloud-water fields are intentionally not parsed;
    # their textual encoding is lossy and varies with NCMP.
    return np.array([
        float(line[0:4]), float(line[4:10]), float(line[10:16]),
        float(line[16:20]), float(line[20:25]), float(line[25:31]),
        float(line[31:34]), float(line[34:39]),
    ], dtype="f4")


def _water(line: bytes) -> np.ndarray:
    """Decode WRTCMP's F6.3 water fields, including negative run count."""
    tail=line.rstrip(b'\r\n')[39:]
    values=np.array([float(tail[index:index+6]) for index in range(0,len(tail),6)
                     if tail[index:index+6].strip()],dtype='f4')
    if values.size==1 and values[0]<0.:
        return np.zeros(int(round(-float(values[0]))),dtype='f4')
    return values


class M3DReader:
    """Streaming reader; cells are yielded in time, y, x order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def cells(self) -> tuple[M3DHeader, Iterable[M3DCell]]:
        stream = self.path.open("rb", buffering=1024 * 1024)
        header = _read_header(stream)

        def generate():
            try:
                for _time in range(header.nhours):
                    for _j in range(header.ny):
                        for _i in range(header.nx):
                            line = stream.readline()
                            if not line:
                                raise EOFError("m3d ended inside a surface record")
                            date, i, j, surface = _surface(line)
                            lines=[stream.readline() for _ in range(header.nz)]
                            if any(not line for line in lines):
                                raise EOFError('m3d ended inside a level record')
                            levels=np.stack([_level(line) for line in lines])
                            water=np.zeros((header.nz,5),dtype='f4')
                            for level,line in enumerate(lines):
                                values=_water(line)
                                if values.size>5: raise ValueError('M3D water record has more than five fields')
                                water[level,:values.size]=values
                            yield M3DCell(date,i,j,surface,levels,water)
            finally:
                stream.close()

        return header, generate()

    def to_netcdf(self,path: str | Path,*,compression=1) -> Path:
        """Stream this formatted M3D file into the canonical CALWRF schema."""
        output=Path(path); output.parent.mkdir(parents=True,exist_ok=True)
        header,cells=self.cells()
        with nc.Dataset(output,'w',format='NETCDF4') as dataset:
            for name,size in (('time',header.nhours),('level',header.nz),
                              ('y',header.ny),('x',header.nx),('date_strlen',19)):
                dataset.createDimension(name,size)
            time_text=dataset.createVariable('time_text','S1',('time','date_strlen'))
            dataset.createVariable('sigma','f4',('level',))[:]=header.sigma
            for name,values in (('latitude',header.latitude),('longitude',header.longitude),
                                ('terrain',header.terrain),('landuse',header.landuse)):
                kind='i2' if name=='landuse' else 'f4'
                dataset.createVariable(name,kind,('y','x'))[:]=values
            level_names=('pressure','height','temperature','wind_direction','wind_speed',
                         'vertical_velocity','relative_humidity','qvapor','qcloud','qrain',
                         'qice','qsnow','qgraupel')
            level_vars={name:dataset.createVariable(name,'f4',('time','level','y','x'),
                zlib=compression>0,complevel=compression) for name in level_names}
            surface_names=('surface_pressure','precipitation','snow_cover','shortwave_down',
                'longwave_down','temperature_2m','qvapor_2m','wind_direction_10m',
                'wind_speed_10m','sst')
            surface_vars={name:dataset.createVariable(name,'f4',('time','y','x'),
                zlib=compression>0,complevel=compression) for name in surface_names}
            for name in ('pbl_height','sensible_heat_flux','friction_velocity'):
                surface_vars[name]=dataset.createVariable(name,'f4',('time','y','x'),
                    zlib=compression>0,complevel=compression,fill_value=False)
                surface_vars[name][:]=0.
            for time_index in range(header.nhours):
                surface=np.empty((10,header.ny,header.nx),dtype='f4')
                levels=np.empty((13,header.nz,header.ny,header.nx),dtype='f4')
                date=None
                for _ in range(header.nx*header.ny):
                    cell=next(cells); date=cell.date
                    j=cell.j-header.j_start; i=cell.i-header.i_start
                    surface[:,j,i]=cell.surface
                    levels[:8,:,j,i]=cell.levels.T
                    levels[8:,:,j,i]=cell.water.T
                stamp=f'{date:010d}'
                text=f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}_{stamp[8:]}:00:00'
                time_text[time_index]=np.frombuffer(text.encode('ascii'),dtype='S1')
                scales=(100.,10.,1.,1.,1.,1.,.001,1.,1.,1.)
                for index,name in enumerate(surface_names): surface_vars[name][time_index]=surface[index]*scales[index]
                for index,name in enumerate(level_names):
                    scale=.001 if name in {'qvapor','qcloud','qrain','qice','qsnow','qgraupel'} else 1.
                    level_vars[name][time_index]=levels[index]*scale
            dataset.source_format='CALWRF M3D 2.1'
            dataset.pycalmet_stage='calwrf'
            dataset.map_proj={'LCC':1,'PS':2,'EM':3}.get(header.projection,1)
            dataset.moad_cen_lat=header.latitude_origin
            dataset.cen_lat=header.latitude_origin; dataset.stand_lon=header.longitude_origin
            dataset.truelat1=header.standard_parallel_1; dataset.truelat2=header.standard_parallel_2
            dataset.wrf_dx_m=header.spacing_km*1000.; dataset.wrf_dy_m=header.spacing_km*1000.
            dataset.source_grid_nx=header.source_nx
            dataset.source_grid_ny=header.source_ny
            dataset.source_grid_nz=header.source_nz
            dataset.source_i_start=header.i_start
            dataset.source_i_end=header.i_start+header.nx-1
            dataset.source_j_start=header.j_start
            dataset.source_j_end=header.j_start+header.ny-1
            dataset.source_k_start=header.k_start
            dataset.source_k_end=header.k_start+header.nz-1
            dataset.source_x_origin_km=header.x_origin_km
            dataset.source_y_origin_km=header.y_origin_km
        return output


def selected_cells(path: str | Path, wanted: set[tuple[int, int, int]]) -> dict[tuple[int, int, int], M3DCell]:
    """Read selected ``(time_index, j, i)`` zero-based cells in one pass."""
    reader = M3DReader(path)
    header, cells = reader.cells()
    result = {}
    per_time = header.nx * header.ny
    for linear, cell in enumerate(cells):
        time_index, within = divmod(linear, per_time)
        j, i = divmod(within, header.nx)
        key = (time_index, j, i)
        if key in wanted:
            result[key] = cell
            if len(result) == len(wanted):
                cells.close()
                break
    missing = wanted - result.keys()
    if missing:
        raise KeyError(f"m3d cells not found: {sorted(missing)}")
    return result
