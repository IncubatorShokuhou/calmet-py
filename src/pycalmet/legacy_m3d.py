"""Reader for CALWRF 2.0 formatted 3D.DAT/m3d regression files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable

import netCDF4 as nc
import numpy as np


@dataclass(frozen=True, slots=True)
class M3DHeader:
    """Metadata and static fields decoded from a formatted CALWRF M3D file.

    Attributes:
        projection: Legacy projection code, normally ``LCC``, ``PS``, or
            ``EM``.
        latitude_origin: Projection latitude of origin in degrees north.
        longitude_origin: Projection longitude of origin in degrees east.
        standard_parallel_1: First standard parallel in degrees north.
        standard_parallel_2: Second standard parallel in degrees north.
        x_origin_km: Source-grid easting origin in km.
        y_origin_km: Source-grid northing origin in km.
        spacing_km: Horizontal grid spacing in km.
        nx: Number of extracted west-to-east cells.
        ny: Number of extracted south-to-north cells.
        nz: Number of extracted vertical levels.
        nhours: Number of hourly records declared in the file.
        start: Naive datetime represented by the M3D header timestamp.
        sigma: Dimensionless ``float32`` sigma coordinates with shape
            ``(nz,)``.
        latitude: ``float32`` latitude in degrees north with shape
            ``(ny, nx)``.
        longitude: ``float32`` longitude in degrees east with shape
            ``(ny, nx)``.
        terrain: ``float32`` terrain elevation in metres with shape
            ``(ny, nx)``.
        landuse: ``int16`` land-use category grid with shape ``(ny, nx)``.
        source_nx: Number of cells in the unextracted source grid.
        source_ny: Number of cells in the unextracted source grid.
        source_nz: Number of levels in the unextracted source grid.
        i_start: One-based source-grid index of the first extracted x cell.
        j_start: One-based source-grid index of the first extracted y cell.
        k_start: One-based source-grid index of the first extracted level.
    """
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
    start: datetime
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

    @property
    def wire_start(self) -> str:
        """Format the header start time for the M3D wire representation.

        Returns:
            A ten-character ``YYYYMMDDHH`` timestamp.
        """
        return self.start.strftime("%Y%m%d%H")


@dataclass(frozen=True, slots=True)
class M3DCell:
    """One horizontal cell from one formatted M3D hour.

    Attributes:
        timestamp: Naive datetime decoded from the surface record.
        i: One-based source-grid x index stored in the file.
        j: One-based source-grid y index stored in the file.
        surface: ``float32`` array with shape ``(10,)`` containing surface
            pressure (hPa), precipitation (cm), snow cover, downward shortwave
            and longwave fluxes (W/m2), 2 m temperature (K), 2 m water-vapour
            mixing ratio (g/kg), 10 m wind direction (degrees), 10 m wind speed
            (m/s), and sea-surface temperature (K), in that order.
        levels: ``float32`` array with shape ``(nz, 8)`` containing pressure
            (hPa), height (m), temperature (K), wind direction (degrees), wind
            speed (m/s), vertical velocity (m/s), relative humidity (percent),
            and water-vapour mixing ratio (g/kg).
        water: ``float32`` cloud-component array with shape ``(nz, 5)`` in
            ``qcloud``, ``qrain``, ``qice``, ``qsnow``, ``qgraupel`` order,
            expressed in g/kg. Missing trailing components are zero-filled.
    """
    timestamp: datetime
    i: int
    j: int
    surface: np.ndarray
    levels: np.ndarray
    water: np.ndarray

    @property
    def date(self) -> datetime:
        """Return the structured timestamp under its compatibility name.

        Returns:
            The same naive :class:`datetime.datetime` as ``timestamp``.
        """
        return self.timestamp

    @property
    def wire_date(self) -> str:
        """Format the cell timestamp for the M3D wire representation.

        Returns:
            A ten-character ``YYYYMMDDHH`` timestamp.
        """
        return self.timestamp.strftime("%Y%m%d%H")


def _timestamp(field: bytes,description: str) -> datetime:
    """Decode one exact-width ``YYYYMMDDHH`` field at the file boundary.

    Args:
        field: Exactly ten ASCII bytes containing a decimal timestamp.
        description: Record label included in validation errors.

    Returns:
        A naive :class:`datetime.datetime` at whole-hour precision.

    Raises:
        ValueError: If the field is not ten ASCII digits or does not represent
            a valid calendar hour.
    """
    try:
        text=field.decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError(f'invalid M3D {description} timestamp') from error
    if len(text)!=10 or not text.isdecimal():
        raise ValueError(f'invalid M3D {description} timestamp {text!r}')
    try:
        return datetime.strptime(text,'%Y%m%d%H')
    except ValueError as error:
        raise ValueError(f'invalid M3D {description} timestamp {text!r}') from error


def _read_header(stream: BinaryIO) -> M3DHeader:
    """Read an M3D header and its static horizontal-grid records.

    Args:
        stream: Binary input stream positioned at the M3D format/version line.

    Returns:
        An :class:`M3DHeader`. Static latitude, longitude, and terrain arrays
        are ``float32`` with shape ``(ny, nx)``; land use is ``int16`` with the
        same shape; sigma is ``float32`` with shape ``(nz,)``. The stream is
        left at the first hourly surface record.

    Raises:
        OSError: If the stream cannot be read.
        UnicodeError: If the projection code is not ASCII.
        ValueError: If timestamps, dimensions, extraction bounds, static-cell
            records, or numeric fields are invalid or inconsistent.
        IndexError: If a required projection/grid field is absent from a
            truncated header record.
    """
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
    period = stream.readline().split()
    if len(period)<5:
        raise ValueError('M3D period record is incomplete')
    start=_timestamp(period[0],'header')
    try:
        nhours,nx,ny,nz=map(int,period[1:5])
    except ValueError as error:
        raise ValueError('invalid M3D period dimensions') from error
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


def _surface(line: bytes) -> tuple[datetime, int, int, np.ndarray]:
    """Decode one fixed-width M3D surface record.

    Args:
        line: Surface-record bytes using CALWRF format 221.

    Returns:
        A tuple of ``(timestamp, i, j, values)``. ``i`` and ``j`` are the
        one-based source-grid indices. ``values`` is ``float32`` with shape
        ``(10,)`` and uses the field order and units documented by
        :class:`M3DCell`.

    Raises:
        ValueError: If the timestamp or any fixed-width numeric field is
            invalid or the record is truncated.
    """
    # Fortran 221: i10,2i3,f7.1,f5.2,i2,3f8.1,f8.2,3f8.1
    timestamp=_timestamp(line[0:10],'surface-record')
    i = int(line[10:13]); j = int(line[13:16])
    values = np.array([
        float(line[16:23]), float(line[23:28]), float(line[28:30]),
        float(line[30:38]), float(line[38:46]), float(line[46:54]),
        float(line[54:62]), float(line[62:70]), float(line[70:78]),
        float(line[78:86]),
    ], dtype="f4")
    return timestamp, i, j, values


def _level(line: bytes) -> np.ndarray:
    """Decode the eight basic fields from one fixed-width M3D level record.

    Args:
        line: Level-record bytes using the basic portion of CALWRF format 222.
            Any cloud-water fields after column 39 are ignored here.

    Returns:
        A ``float32`` array with shape ``(8,)`` containing pressure (hPa),
        height (m), temperature (K), wind direction (degrees), wind speed
        (m/s), vertical velocity (m/s), relative humidity (percent), and
        water-vapour mixing ratio (g/kg).

    Raises:
        ValueError: If a fixed-width numeric field is invalid or truncated.
    """
    # Basic section of Fortran 222: i4,i6,f6.1,i4,f5.1,f6.2,i3,f5.2.
    # Remaining compressed cloud-water fields are intentionally not parsed;
    # their textual encoding is lossy and varies with NCMP.
    return np.array([
        float(line[0:4]), float(line[4:10]), float(line[10:16]),
        float(line[16:20]), float(line[20:25]), float(line[25:31]),
        float(line[31:34]), float(line[34:39]),
    ], dtype="f4")


def _water(line: bytes) -> np.ndarray:
    """Decode WRTCMP's cloud-water fields, including negative run counts.

    Args:
        line: Complete M3D level-record bytes. Columns after the eight basic
            fields are interpreted as repeated ``F6.3`` values in g/kg.

    Returns:
        A one-dimensional ``float32`` array. A single negative value ``-N``
        expands to ``N`` zeros; otherwise each encoded field is returned in
        wire order. The caller pads missing trailing cloud components.

    Raises:
        ValueError: If a nonblank six-column field is not a valid float.
    """
    tail=line.rstrip(b'\r\n')[39:]
    values=np.array([float(tail[index:index+6]) for index in range(0,len(tail),6)
                     if tail[index:index+6].strip()],dtype='f4')
    if values.size==1 and values[0]<0.:
        return np.zeros(int(round(-float(values[0]))),dtype='f4')
    return values


class M3DReader:
    """Stream formatted CALWRF M3D records without loading the full file.

    Attributes:
        path: Path to the formatted M3D input.
    """

    def __init__(self, path: str | Path):
        """Create a reader for one formatted M3D file.

        Args:
            path: Path-like input location. The file is opened by
                :meth:`cells` or :meth:`to_netcdf`.
        """
        self.path = Path(path)

    def cells(self) -> tuple[M3DHeader, Iterable[M3DCell]]:
        """Read the header and create a streaming cell iterator.

        Returns:
            A ``(header, cells)`` tuple. ``header`` is an :class:`M3DHeader`;
            ``cells`` yields :class:`M3DCell` objects in time, south-to-north
            y, then west-to-east x order. The underlying binary stream remains
            open until the iterator is exhausted or explicitly closed.

        Raises:
            OSError: If the input cannot be opened or its header cannot be
                read.
            UnicodeError: If header text is not ASCII.
            ValueError: If the M3D header is malformed or inconsistent.

        Notes:
            Truncation and hourly-record validation errors are raised while
            consuming the returned iterator rather than by this method call.
        """
        stream = self.path.open("rb", buffering=1024 * 1024)
        header = _read_header(stream)

        def generate():
            """Yield cells from the open M3D stream and close it afterward.

            Yields:
                :class:`M3DCell` objects. ``surface`` has shape ``(10,)``,
                ``levels`` has shape ``(nz, 8)``, and ``water`` has shape
                ``(nz, 5)``; all three arrays use ``float32``.

            Raises:
                EOFError: If the file ends inside a surface or level record.
                ValueError: If timestamps within an hour disagree, a
                    fixed-width field is invalid, or a level contains more
                    than five cloud-water components.
                OSError: If the open stream fails while reading.
            """
            try:
                for _time in range(header.nhours):
                    hour_timestamp=None
                    for _j in range(header.ny):
                        for _i in range(header.nx):
                            line = stream.readline()
                            if not line:
                                raise EOFError("m3d ended inside a surface record")
                            timestamp, i, j, surface = _surface(line)
                            if hour_timestamp is None:
                                hour_timestamp=timestamp
                            elif timestamp!=hour_timestamp:
                                raise ValueError(
                                    'M3D cell timestamps are inconsistent within an hour')
                            lines=[stream.readline() for _ in range(header.nz)]
                            if any(not line for line in lines):
                                raise EOFError('m3d ended inside a level record')
                            levels=np.stack([_level(line) for line in lines])
                            water=np.zeros((header.nz,5),dtype='f4')
                            for level,line in enumerate(lines):
                                values=_water(line)
                                if values.size>5: raise ValueError('M3D water record has more than five fields')
                                water[level,:values.size]=values
                            yield M3DCell(timestamp,i,j,surface,levels,water)
            finally:
                stream.close()

        return header, generate()

    def to_netcdf(self,path: str | Path,*,compression=1) -> Path:
        """Stream this formatted M3D file into canonical CALWRF NetCDF.

        Args:
            path: Destination NetCDF path. Parent directories are created when
                needed.
            compression: NetCDF4 deflate level for time-varying surface and
                level fields; zero disables compression.

        Returns:
            The destination as a :class:`pathlib.Path`. Output variables use
            ``(time, y, x)`` or ``(time, level, y, x)`` dimensions. Mixing
            ratios are converted from g/kg to kg/kg, pressure from hPa to Pa,
            and precipitation from cm to mm.

        Raises:
            OSError: If the source cannot be read or the destination cannot be
                created or written. A failure after creation can leave a
                partial NetCDF file.
            UnicodeError: If required M3D text is not ASCII.
            EOFError: If the M3D file ends inside an hourly record.
            ValueError: If header or cell records are malformed,
                inconsistent, or contain too many cloud-water components.
            RuntimeError: If the NetCDF4 backend rejects an output definition
                or write operation, including an unsupported compression
                setting.
        """
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
                timestamp=None
                for _ in range(header.nx*header.ny):
                    cell=next(cells); timestamp=cell.timestamp
                    j=cell.j-header.j_start; i=cell.i-header.i_start
                    surface[:,j,i]=cell.surface
                    levels[:8,:,j,i]=cell.levels.T
                    levels[8:,:,j,i]=cell.water.T
                text=timestamp.strftime('%Y-%m-%d_%H:%M:%S')
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
    """Read selected zero-based M3D cells in one streaming pass.

    Args:
        path: Path to the formatted M3D input.
        wanted: Set of ``(time_index, j, i)`` keys. These indices describe the
            extracted M3D array and are zero-based, independent of the
            one-based source indices stored on each :class:`M3DCell`.

    Returns:
        A dictionary mapping every requested key to its decoded
        :class:`M3DCell`. Each cell contains ``float32`` surface, level, and
        cloud-water arrays.

    Raises:
        KeyError: If one or more requested keys are outside the available M3D
            records.
        OSError: If the input cannot be opened or read.
        UnicodeError: If header text is not ASCII.
        EOFError: If the file ends inside a surface or level record.
        ValueError: If header or cell records are malformed or inconsistent.
    """
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
