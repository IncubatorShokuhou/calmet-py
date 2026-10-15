"""Reader and canonical-NetCDF adapter for CALMET GEO.DAT 2.1 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, TextIO, TypeAlias, cast

import netCDF4 as nc
import numpy as np
import numpy.typing as npt


_NUMBER: re.Pattern[str]=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?')
_DEFAULT_CATEGORIES: npt.NDArray[np.int16]=np.array(
    [10,20,-20,30,40,50,54,55,60,61,62,70,80,90],dtype=np.int16
)
_DEFAULT_FIELDS: tuple[tuple[float,...],...]=(
    (1.,.25,.25,.05,1.,.001,.001,.001,1.,1.,.20,.05,.20,.05),
    (.18,.15,.15,.25,.10,.10,.10,.10,.10,.10,.10,.30,.30,.70),
    (1.5,1.,.5,1.,1.,0.,0.,0.,.5,.5,.1,1.,.5,.5),
    (.25,.15,.15,.15,.15,1.,1.,1.,.25,.25,.25,.15,.15,.15),
    (0.,)*14,
    (.2,3.,3.,.5,7.,0.,0.,0.,2.,2.,1.,.05,0.,0.),
)
CompressionLevel: TypeAlias = Literal[0,1,2,3,4,5,6,7,8,9]


@dataclass(frozen=True,slots=True)
class GeoData:
    """Decoded CALMET GEO.DAT grid and geophysical fields.

    Attributes:
        projection: CALMET projection code, such as ``UTM`` or ``LCC``.
        datum: Uppercase CALMET datum identifier.
        nx: Number of west-to-east grid cells.
        ny: Number of south-to-north grid cells.
        x_origin_km: Grid southwest-corner easting in km.
        y_origin_km: Grid southwest-corner northing in km.
        dx_km: West-to-east grid spacing in km.
        dy_km: South-to-north grid spacing in km.
        latitude_origin: Projection latitude of origin in degrees north.
        longitude_origin: Projection longitude of origin in degrees east.
        standard_parallel_1: First standard parallel in degrees north.
        standard_parallel_2: Second standard parallel in degrees north.
        false_easting_km: Projection false easting in km.
        false_northing_km: Projection false northing in km.
        utm_zone: UTM zone, or zero for a non-UTM projection.
        hemisphere: UTM hemisphere code, ``N`` or ``S``.
        water_min: Lowest land-use category treated as water.
        water_max: Highest land-use category treated as water.
        categories: One-dimensional ``int16`` array of declared land-use
            categories.
        terrain: ``float32`` terrain elevation in metres with shape
            ``(ny, nx)``.
        fields: Seven ``(ny, nx)`` arrays in this order: dominant land-use
            category (``int16``), roughness length in metres, albedo fraction,
            Bowen ratio, soil heat-flux parameter, anthropogenic heat flux in
            W/m2, and dimensionless leaf-area index. Physical fields are
            ``float32``.
    """
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
    categories: npt.NDArray[np.int16]
    terrain: npt.NDArray[np.float32]
    fields: tuple[npt.NDArray[np.int16] | npt.NDArray[np.float32],...]


def _numbers(line: str) -> list[float]:
    """Extract Fortran-style real numbers from one text record.

    Args:
        line: Text record that may contain decimal or exponent-form numbers.
            Both ``D`` and ``E`` exponent markers are accepted.

    Returns:
        A list of Python ``float`` values in lexical order. Non-numeric text is
        ignored.
    """
    return [float(value.replace('D','E').replace('d','e')) for value in _NUMBER.findall(line)]


def _angle(text: str) -> float:
    """Parse a CALMET angle with an optional cardinal suffix.

    Args:
        text: Numeric angle optionally ending in ``N``, ``S``, ``E``, or
            ``W``. Empty text denotes zero.

    Returns:
        Signed decimal degrees, with south and west represented as negative.

    Raises:
        ValueError: If the nonempty numeric portion is not a valid float.
    """
    token: str=text.strip().upper()
    if not token: return 0.
    sign: float=-1. if token[-1:] in {'S','W'} else 1.
    return sign*float(token.rstrip('NSEW'))


def _values(stream: TextIO,count: int) -> list[float]:
    """Read exactly ``count`` numeric values from consecutive text records.

    Args:
        stream: Open GEO.DAT text stream positioned at the first numeric row.
        count: Required number of values.

    Returns:
        A list of exactly ``count`` Python ``float`` values.

    Raises:
        EOFError: If the stream ends before enough values are available.
        ValueError: If the final record causes the result to exceed ``count``.
        OSError: If the input stream cannot be read.
    """
    values: list[float]=[]
    while len(values)<count:
        line: str=stream.readline()
        if not line: raise EOFError('GEO.DAT ended inside a numeric record')
        values.extend(_numbers(line))
    if len(values)!=count: raise ValueError('GEO.DAT numeric record contains extra values')
    return values


def _grid(
    stream: TextIO,
    nx: int,
    ny: int,
    dtype: npt.DTypeLike = float,
) -> npt.NDArray[np.float64] | npt.NDArray[np.int16]:
    """Read one south-up CALMET grid and convert it to NumPy row order.

    Args:
        stream: Open GEO.DAT text stream positioned at the grid values.
        nx: Number of values in each formatted row.
        ny: Number of formatted rows.
        dtype: NumPy-compatible output data type.

    Returns:
        An array with shape ``(ny, nx)``. GEO.DAT's north-to-south records are
        reversed so index zero is the southern row used by PyCALMET.

    Raises:
        EOFError: If a grid row is truncated.
        ValueError: If a numeric row contains too many values or cannot be
            converted to ``dtype``.
        OSError: If the input stream cannot be read.
    """
    rows: list[list[float]]=[_values(stream,nx) for _ in range(ny)]
    return np.asarray(rows[::-1],dtype=dtype)


class GeoReader:
    """Read formatted CALMET GEO.DAT 2.1 files.

    Attributes:
        path: Path to the formatted GEO.DAT input.
    """

    def __init__(self,path: str | Path) -> None:
        """Create a reader for one formatted geophysical file.

        Args:
            path: Path-like GEO.DAT input location. The file is opened only
                when :meth:`read` or :meth:`to_netcdf` is called.
        """
        self.path: Path=Path(path)

    def read(self) -> GeoData:
        """Decode the GEO.DAT header, terrain, land use, and physical fields.

        Returns:
            A :class:`GeoData` instance. Grids have shape ``(ny, nx)``;
            terrain and continuous physical fields use ``float32``, while
            land-use categories use ``int16``.

        Raises:
            OSError: If the input file cannot be opened or read.
            UnicodeError: If the file is not strict ASCII text.
            EOFError: If a numeric grid or category table ends prematurely.
            ValueError: If the dataset identifier, projection metadata,
                horizontal units, land-use declaration, category table, or
                geophysical option records are invalid or unsupported.
        """
        with self.path.open('r',encoding='ascii',errors='strict') as stream:
            if stream.readline()[:16].strip().upper()!='GEO.DAT':
                raise ValueError('geophysical input is not a GEO.DAT file')
            comments: int=int(stream.readline())
            for _ in range(comments): stream.readline()
            projection: str=stream.readline().strip().upper()
            lat0: float; lon0: float; lat1: float; lat2: float
            zone: int; hemisphere: str
            lat0=lon0=lat1=lat2=0.; zone=0; hemisphere='N'
            if projection=='UTM':
                line: str=stream.readline()
                # WRTHEAD uses FORMAT(I4,A4), so the zone and hemisphere are
                # adjacent (for example ``"  12S   "``), not whitespace
                # separated fields.
                zone=int(line[:4]); hemisphere=line[4:8].strip()[:1].upper()
                if not hemisphere: raise ValueError('GEO.DAT UTM hemisphere is missing')
            else:
                line=stream.readline()
                widths: list[str]=[
                    line[index:index+16] for index in range(0,64,16)
                ]
                lat0=_angle(widths[0]); lon0=_angle(widths[1])
                if projection in {'LCC','PS'}: lat1=_angle(widths[2])
                if projection=='LCC': lat2=_angle(widths[3])
            if projection in {'LCC','LAZA','TTM'}:
                false_easting: float; false_northing: float
                false_easting,false_northing=_numbers(stream.readline())[:2]
            else: false_easting=false_northing=0.
            datum: str=stream.readline()[:8].strip().upper()
            grid: list[float]=_numbers(stream.readline())
            if len(grid)<6: raise ValueError('GEO.DAT grid record is incomplete')
            nx: int; ny: int
            nx,ny=int(grid[0]),int(grid[1])
            x0: float; y0: float; dx: float; dy: float
            x0,y0,dx,dy=grid[2:6]
            if stream.readline()[:2].strip().upper()!='KM':
                raise ValueError('GEO.DAT horizontal units must be KM')
            land_option: int=int(_numbers(stream.readline())[0])
            water_min: int; water_max: int
            if land_option==0:
                categories: npt.NDArray[np.int16]=_DEFAULT_CATEGORIES.copy()
                water_min=water_max=55
            elif land_option==1:
                values: list[float]=_numbers(stream.readline())
                if len(values)<3: raise ValueError('GEO.DAT land-use declaration is incomplete')
                count: int
                count,water_min,water_max=map(int,values[:3])
                categories=np.asarray(_values(stream,count),dtype=np.int16)
            else: raise ValueError(f'unsupported GEO.DAT land-use option {land_option}')
            landuse: npt.NDArray[np.int16]=cast(
                npt.NDArray[np.int16],_grid(stream,nx,ny,np.int16)
            )
            terrain_line: list[float]=_numbers(stream.readline())
            if not terrain_line: raise ValueError('GEO.DAT lacks HTFAC')
            terrain: npt.NDArray[np.float64]=cast(
                npt.NDArray[np.float64],_grid(stream,nx,ny)
            )*terrain_line[0]
            outputs: list[npt.NDArray[np.float32]]=[]
            for defaults in _DEFAULT_FIELDS:
                option_line: list[float]=_numbers(stream.readline())
                if not option_line: raise ValueError('GEO.DAT lacks a geophysical option')
                option: int=int(option_line[0])
                field: npt.NDArray[np.float64] | npt.NDArray[np.int16]
                if option==2: field=_grid(stream,nx,ny)
                else:
                    table: npt.NDArray[np.float64]=np.zeros(
                        categories.size,dtype=float
                    )
                    if option==0: table[:min(len(defaults),table.size)]=defaults[:table.size]
                    elif option==1:
                        for index,category in enumerate(categories):
                            record: list[float]=_values(stream,2)
                            if int(record[0])!=category:
                                raise ValueError('GEO.DAT category table does not match land-use categories')
                            table[index]=record[1]
                    else: raise ValueError(f'invalid GEO.DAT geophysical option {option}')
                    lookup: dict[int,np.float64]={
                        int(category):table[index]
                        for index,category in enumerate(categories)
                    }
                    try: field=np.vectorize(lookup.__getitem__,otypes=[float])(landuse)
                    except KeyError as error: raise ValueError(f'GEO.DAT land-use category {error.args[0]} is absent from table') from error
                outputs.append(np.asarray(field,dtype=np.float32))
        return GeoData(projection,datum,nx,ny,x0,y0,dx,dy,lat0,lon0,lat1,lat2,
                       false_easting,false_northing,zone,hemisphere,water_min,water_max,
                       categories,np.asarray(terrain,dtype=np.float32),
                       (np.asarray(landuse,dtype=np.int16),*outputs))

    def to_netcdf(
        self,path: str | Path,*,compression: CompressionLevel=1
    ) -> Path:
        """Convert the formatted GEO.DAT input to canonical NetCDF.

        Args:
            path: Destination NetCDF path. Parent directories are created when
                needed.
            compression: NetCDF4 deflate level applied to each two-dimensional
                output variable; zero disables compression.

        Returns:
            The destination as a :class:`pathlib.Path`.

        Raises:
            OSError: If the GEO.DAT input cannot be read or the destination
                cannot be created or written. A failure after creation can
                leave a partial NetCDF file.
            UnicodeError: If the input is not strict ASCII text.
            EOFError: If a numeric input record is truncated.
            ValueError: If GEO.DAT contents are malformed or unsupported.
            RuntimeError: If the NetCDF4 backend rejects an output definition
                or write operation, including an unsupported compression
                setting.
        """
        data: GeoData=self.read()
        output: Path=Path(path); output.parent.mkdir(parents=True,exist_ok=True)
        names: tuple[str,...]=('dominant_landuse','roughness_length','albedo','bowen_ratio',
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
                kind: Literal['i2','f4']='i2' if name=='dominant_landuse' else 'f4'
                dataset.createVariable(name,kind,('y','x'),zlib=compression>0,
                                       complevel=compression)[:]=values
        return output
