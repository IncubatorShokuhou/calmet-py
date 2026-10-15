"""Streaming output for the NumPy CALMET solver.

The canonical NetCDF file preserves floating-point solver fields without the
rounding in CALMET's legacy sequential-unformatted interchange file.  A
separate legacy writer can therefore be regression-tested against this source.
"""
from __future__ import annotations
from collections.abc import Iterable,Mapping,Sequence
from pathlib import Path
from datetime import datetime,timedelta
from os import PathLike
import struct
from types import TracebackType
from typing import TYPE_CHECKING,Any,Literal,TypeAlias,cast
import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset,Variable,date2num
from .datetime_utils import format_ordinal_code
from .calmet_numerics import convective_velocity_scale,mesopac_layer_average,pgt_stability
from .calmet_solver import HourFields,target_latitude_longitude
from .config import GridConfig
from .legacy_io import FortranSequentialWriter,fixed_string


FilePath: TypeAlias=str | PathLike[str]
NumericArray: TypeAlias=npt.NDArray[np.generic]
if TYPE_CHECKING:
    ArraySource: TypeAlias=npt.ArrayLike | Variable[Any]
else:
    ArraySource: TypeAlias=npt.ArrayLike | Variable
GeophysicalSource: TypeAlias=Mapping[str,ArraySource] | Dataset
IntegerScalar: TypeAlias=int | np.integer
RealScalar: TypeAlias=int | float | np.integer | np.floating
CompressionLevel: TypeAlias=Literal[0,1,2,3,4,5,6,7,8,9]
NetCDFDType: TypeAlias=Literal['f8','f4','i2','i1']


def _array_data(
    value: ArraySource,
    dtype: npt.DTypeLike | None=None,
) -> NumericArray:
    """Read a netCDF variable or array and apply one dtype conversion.

    Args:
        value: netCDF variable or array-like value. Objects exposing
            ``dimensions`` are materialized with ``value[:]``.
        dtype: Optional NumPy dtype for the returned data.

    Returns:
        NumPy array with the source shape and requested dtype.
    """
    if isinstance(value,Variable):
        return np.asarray(value[:],dtype=dtype)
    return np.asarray(value,dtype=dtype)


def _whole_run_hours(start: datetime,end: datetime,output_name: str) -> int:
    """Return an exact whole-hour duration for an integer output header.

    Args:
        start: Run beginning datetime.
        end: Run ending datetime.
        output_name: Product name included in validation errors.

    Returns:
        Nonnegative integer run duration in hours.

    Raises:
        ValueError: If the interval is negative or is not a whole number of
            hours.
    """
    seconds=(end-start).total_seconds()
    if seconds<0 or seconds%3600:
        raise ValueError(f'{output_name} output requires a whole-hour run period')
    return int(seconds/3600)

def _convective_velocity(
    hour: HourFields,
    landuse: npt.ArrayLike,
    water_range: tuple[int,int],
) -> npt.NDArray[np.float32]:
    """Reproduce WATER-before-AVEMIX and land-only WSTARR sequencing.

    Args:
        hour: Solver fields for one output time.
        landuse: Integer land-use array of shape ``(ny, nx)``.
        water_range: Inclusive ``(minimum, maximum)`` water-category codes.

    Returns:
        Float32 convective velocity scale with shape ``(ny, nx)`` and units of
        metres per second. Retained overwater values replace the land formula
        within water categories when available.

    Raises:
        ValueError: If the numerical inputs cannot be broadcast to the common
            model grid.
    """
    surface=hour.surface
    value: npt.NDArray[np.float32]=convective_velocity_scale(surface.convective_mixing_height,
        surface.sensible_heat_flux,hour.temperature[0],surface.density,landuse,
        water_range=water_range)
    if surface.water_convective_velocity_scale is not None:
        categories=cast(npt.NDArray[np.integer],np.asarray(landuse))
        water: npt.NDArray[np.bool_]=(categories>=water_range[0])&(categories<=water_range[1])
        value=np.where(water,surface.water_convective_velocity_scale,value)
    return np.asarray(value,dtype=np.float32)

class CalmetNetCDFWriter:
    """Append :class:`HourFields` to a CF-style NetCDF file one step at a time.

    Attributes:
        path: Destination NetCDF path.
        dataset: Open :class:`netCDF4.Dataset`, or ``None`` after closing.
        water_range: Inclusive land-use category range treated as water.
        run_type: CALMET run type; zero writes only horizontal winds.
    """

    path: Path
    dataset: Dataset | None
    water_range: tuple[int,int]
    run_type: int
    _index: int

    def __init__(
        self,
        path: FilePath,
        grid: GridConfig,
        z_faces_m: npt.ArrayLike,
        geophysical: GeophysicalSource,
        *,
        compression: int=1,
        water_range: Sequence[int]=(50,55),
        run_type: int=1,
    ) -> None:
        """Create the NetCDF dimensions, coordinates, and output variables.

        Args:
            path: Destination path; missing parent directories are created.
            grid: Horizontal CALMET grid definition.
            z_faces_m: Float-like layer-face heights in metres with shape
                ``(level + 1,)``.
            geophysical: Mapping containing ``terrain_height`` in metres and
                integer ``dominant_landuse``, each shaped ``(grid.ny, grid.nx)``.
            compression: NetCDF4 deflate level passed as ``complevel``.
            water_range: Inclusive water-category code range.
            run_type: Zero for wind-only output; nonzero for the complete
                meteorological product.

        Raises:
            OSError: If the destination or parent directory cannot be created.
            ValueError: If projection, geophysical, vertical-coordinate, or
                NetCDF compression inputs are invalid.

        Notes:
            Construction creates and opens ``path`` immediately and writes all
            static coordinate and geophysical variables.
        """
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._index=0
        self.dataset=Dataset(self.path,'w',format='NETCDF4')
        self.water_range=cast(tuple[int,int],tuple(water_range))
        self.run_type=int(run_type)
        ds=self.dataset
        faces: npt.NDArray[np.float32]=np.asarray(z_faces_m,dtype=np.float32)
        centers: npt.NDArray[np.float32]=np.float32(.5)*(faces[:-1]+faces[1:])
        ds.createDimension('time',None); ds.createDimension('level',centers.size); ds.createDimension('face',faces.size); ds.createDimension('y',grid.ny); ds.createDimension('x',grid.nx)
        ds.Conventions='CF-1.8'; ds.source='PyCALMET NumPy solver'; ds.pycalmet_stage='calmet'; ds.projection=grid.projection
        ds.water_category_min=int(self.water_range[0]); ds.water_category_max=int(self.water_range[1])
        time: Variable[np.float64]=ds.createVariable('time','f8',('time',)); time.units='seconds since 1970-01-01 00:00:00'; time.calendar='standard'
        for coordinate_name,coordinate_data,coordinate_units in (
                ('level_height',centers,'m'),('face_height',faces,'m')):
            variable=ds.createVariable(
                coordinate_name,'f4',
                ('level' if coordinate_name=='level_height' else 'face',))
            variable.units=coordinate_units
            variable[:]=np.asarray(coordinate_data,dtype=np.float32)
        latitude: npt.NDArray[np.float64]
        longitude: npt.NDArray[np.float64]
        latitude,longitude=target_latitude_longitude(grid)
        static: tuple[tuple[str,NumericArray,str,NetCDFDType],...]=(
                ('latitude',np.asarray(latitude,dtype=np.float64),'degrees_north','f8'),
                ('longitude',np.asarray(longitude,dtype=np.float64),'degrees_east','f8'),
                ('terrain_height',_array_data(geophysical['terrain_height'],np.float32),'m','f4'),
                ('landuse',_array_data(geophysical['dominant_landuse'],np.int16),'1','i2'))
        compression_level=cast(CompressionLevel,compression)
        for static_name,static_data,static_units,static_dtype in static:
            variable=ds.createVariable(
                static_name,static_dtype,('y','x'),zlib=True,
                complevel=compression_level)
            variable.units=static_units; variable[:]=static_data
        variables: tuple[
            tuple[str,NetCDFDType,tuple[str,...],str],...
        ]=(
            ('wind_u','f4',('time','level','y','x'),'m s-1'),('wind_v','f4',('time','level','y','x'),'m s-1'),
            *(() if self.run_type==0 else (
            ('vertical_velocity','f4',('time','face','y','x'),'m s-1'),('temperature','f4',('time','level','y','x'),'K'),
            ('relative_humidity','f4',('time','level','y','x'),'%'),('cloud_fraction','f4',('time','y','x'),'1'),
            ('surface_relative_humidity','f4',('time','y','x'),'%'),
            ('precipitation_rate','f4',('time','y','x'),'mm h-1'),('precipitation_code','i1',('time','y','x'),'1'),
            ('ceiling_height','f4',('time','y','x'),'m'),('surface_temperature','f4',('time','y','x'),'K'),
            ('surface_density','f4',('time','y','x'),'kg m-3'),('shortwave_flux','f4',('time','y','x'),'W m-2'),
            ('sensible_heat_flux','f4',('time','y','x'),'W m-2'),('friction_velocity','f4',('time','y','x'),'m s-1'),
            ('monin_obukhov_length','f4',('time','y','x'),'m'),('mixing_height','f4',('time','y','x'),'m'),
            ('convective_mixing_height','f4',('time','y','x'),'m'),('convective_velocity_scale','f4',('time','y','x'),'m s-1'),
            ('pgt_stability_class','i1',('time','y','x'),'1'))))
        for name,dtype,dims,units in variables:
            variable=ds.createVariable(
                name,dtype,dims,zlib=True,complevel=compression_level)
            variable.units=units

    def append(
        self,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
        landuse: npt.ArrayLike,
    ) -> None:
        """Write and synchronize one time record.

        Args:
            hour: Solver output for one model timestamp. Three-dimensional
                level fields use ``(level, ny, nx)`` and face fields use
                ``(face, ny, nx)``.
            solar_elevation_sine: ``(ny, nx)`` sine of solar elevation used for
                PGT stability classification.
            landuse: Integer ``(ny, nx)`` land-use categories.

        Raises:
            IndexError: If required level or face arrays are incomplete.
            ValueError: If a field cannot be assigned to its configured NetCDF
                dimensions.
            RuntimeError: If the underlying NetCDF dataset cannot be written.

        Notes:
            Fields are converted to float32 except time, latitude, longitude,
            and integer codes. The dataset is flushed after every append.
        """
        ds=cast(Dataset,self.dataset)
        i=self._index; step=hour.step1; surface=hour.surface
        ds['time'][i]=date2num(hour.timestamp,ds['time'].units,ds['time'].calendar)
        if self.run_type==0:
            ds['wind_u'][i]=np.asarray(hour.wind.u,dtype=np.float32)
            ds['wind_v'][i]=np.asarray(hour.wind.v,dtype=np.float32)
            ds.sync(); self._index+=1; return
        values: dict[str,npt.ArrayLike]={'wind_u':hour.wind.u,'wind_v':hour.wind.v,'vertical_velocity':hour.wind.w,
            'temperature':hour.temperature,'relative_humidity':step.relative_humidity,
            'cloud_fraction':step.cloud_fraction,'ceiling_height':step.ceiling_height,
            'surface_relative_humidity':step.surface_relative_humidity,
            'precipitation_rate':step.precipitation_rate,'precipitation_code':step.precipitation_code,
            'surface_temperature':hour.temperature[0],'surface_density':surface.density,
            'shortwave_flux':surface.shortwave_flux,'sensible_heat_flux':surface.sensible_heat_flux,
            'friction_velocity':surface.friction_velocity,'monin_obukhov_length':surface.monin_obukhov_length,
            'mixing_height':surface.mixing_height,'convective_mixing_height':surface.convective_mixing_height}
        integer_fields: set[str]={'precipitation_code'}
        for name,value in values.items():
            dtype=np.int8 if name in integer_fields else np.float32
            ds[name][i]=np.asarray(value,dtype=dtype)
        ds['convective_velocity_scale'][i]=_convective_velocity(hour,landuse,self.water_range)
        ds['pgt_stability_class'][i]=np.asarray(
            pgt_stability(hour.wind.u[0],hour.wind.v[0],solar_elevation_sine,
                          step.cloud_fraction,step.ceiling_height,landuse,
                          surface.monin_obukhov_length,water_range=self.water_range),
            dtype=np.int8,
        )
        ds.sync(); self._index+=1

    def close(self) -> None:
        """Flush and close the NetCDF dataset; repeated calls are harmless."""
        if self.dataset is not None: self.dataset.close(); self.dataset=None

    def __enter__(self) -> CalmetNetCDFWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`CalmetNetCDFWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the dataset on context-manager exit.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()


class MesopacWriter:
    """Write CALMET's revised MESOPAC-II sequential-unformatted layout.

    Attributes:
        writer: Open GNU-style Fortran sequential-record writer.
        grid: Horizontal CALMET grid definition.
        faces: Float32 layer-face heights in metres.
        landuse: Integer-like ``(ny, nx)`` dominant land-use field.
        surface_coordinates: Float32 ``(surface_station, 2)`` projected
            coordinates in metres.
        upper_coordinates: Float32 ``(upper_station, 2)`` projected coordinates
            in metres.
        water_range: Inclusive land-use category range treated as water.
    """

    writer: FortranSequentialWriter
    grid: GridConfig
    faces: npt.NDArray[np.float32]
    landuse: NumericArray
    surface_coordinates: npt.NDArray[np.float32]
    upper_coordinates: npt.NDArray[np.float32]
    water_range: tuple[int,int]

    def __init__(
        self,
        path: FilePath,
        grid: GridConfig,
        z_faces_m: npt.ArrayLike,
        geophysical: GeophysicalSource,
        start: datetime,
        end: datetime,
        *,
        timezone_offset: int=0,
        surface_coordinates_m: npt.ArrayLike=(),
        upper_coordinates_m: npt.ArrayLike=(),
        nearest_surface_station: npt.ArrayLike | None=None,
        water_range: Sequence[int]=(50,55),
    ) -> None:
        """Open a MESOPAC file and write its static header records.

        Args:
            path: Destination sequential-unformatted file.
            grid: Horizontal CALMET grid; MESOPAC requires square cells.
            z_faces_m: Layer-face heights in metres with shape ``(level + 1,)``.
            geophysical: Mapping containing ``dominant_landuse`` and
                ``roughness_length`` arrays shaped ``(grid.ny, grid.nx)``.
            start: Run beginning datetime.
            end: Run ending datetime; the interval must contain whole hours.
            timezone_offset: Integer time-zone offset stored in the header.
            surface_coordinates_m: ``(surface_station, 2)`` projected station
                coordinates in metres.
            upper_coordinates_m: ``(upper_station, 2)`` projected station
                coordinates in metres.
            nearest_surface_station: Optional integer ``(ny, nx)`` nearest-
                station grid. Omitted values are written as zeros.
            water_range: Inclusive water-category code range.

        Raises:
            OSError: If the destination cannot be created or written.
            ValueError: If the run duration, cell geometry, land-use shape,
                nearest-station shape, or coordinate shapes are invalid.

        Notes:
            Construction writes the header, optional station records,
            roughness, nearest-station, and land-use records immediately.
        """
        hours=_whole_run_hours(start,end,'MESOPAC')
        self.writer=FortranSequentialWriter(Path(path)); self.grid=grid
        self.faces=np.asarray(z_faces_m,dtype=np.float32)
        self.landuse=self._array(geophysical['dominant_landuse'])
        self.surface_coordinates=np.asarray(surface_coordinates_m,dtype=np.float32).reshape(-1,2)
        self.upper_coordinates=np.asarray(upper_coordinates_m,dtype=np.float32).reshape(-1,2)
        self.water_range=cast(tuple[int,int],tuple(water_range))
        shape=(grid.ny,grid.nx)
        if self.landuse.shape!=shape:
            raise ValueError('MESOPAC landuse does not match the CALMET grid')
        nearest: NumericArray=(np.zeros(shape,dtype=np.int32) if nearest_surface_station is None
                              else np.asarray(nearest_surface_station))
        if nearest.shape!=shape:
            raise ValueError('MESOPAC nearest-station grid does not match the CALMET grid')
        spacing=np.float32(grid.dx_km*1000.)
        if grid.dx_km!=grid.dy_km:
            raise ValueError('MESOPAC output requires square horizontal cells')
        header=self._i(start.year,start.timetuple().tm_yday,hours,
                       len(self.surface_coordinates),len(self.upper_coordinates),
                       grid.nx,grid.ny,int(timezone_offset),2,4)+self._f(spacing,.4)
        self.writer.write_record(header)
        self.writer.write_record(self._station_grid_coordinates(self.surface_coordinates).T.ravel())
        if len(self.upper_coordinates):
            self.writer.write_record(self._station_grid_coordinates(self.upper_coordinates).T.ravel())
        self.writer.write_record(self._array(geophysical['roughness_length']).astype('<f4',copy=False).ravel(order='C'))
        self.writer.write_record(nearest.astype('<i4',copy=False).ravel(order='C'))
        self.writer.write_record(self.landuse.astype('<i4',copy=False).ravel(order='C'))

    @staticmethod
    def _array(value: ArraySource) -> NumericArray:
        """Materialize a netCDF variable or array-like value.

        Args:
            value: netCDF variable or array-like object.

        Returns:
            NumPy array preserving the source shape and dtype.
        """
        return _array_data(value)

    @staticmethod
    def _i(*values: IntegerScalar) -> bytes:
        """Pack values as little-endian signed 32-bit integers.

        Args:
            *values: Integer-compatible scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as int32.
        """
        return struct.pack('<'+'i'*len(values),*values)

    @staticmethod
    def _f(*values: RealScalar) -> bytes:
        """Pack values as little-endian IEEE float32 scalars.

        Args:
            *values: Real scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as float32.
        """
        return struct.pack('<'+'f'*len(values),*values)

    def _station_grid_coordinates(
        self,
        coordinates: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Convert projected station metres to MESOPAC grid coordinates.

        Args:
            coordinates: Float-like array of shape ``(station, 2)`` in metres.

        Returns:
            Little-endian float32 array of shape ``(station, 2)`` containing
            one-based fractional grid coordinates referenced to cell centres.
        """
        shift=np.float32(self.grid.dx_km*500.)
        return np.asarray((coordinates-shift)/np.float32(self.grid.dx_km*1000.)+np.float32(1.),dtype='<f4')

    def _grid_record(self,value: npt.ArrayLike,integer: bool=False) -> None:
        """Write one two-dimensional MESOPAC grid record.

        Args:
            value: Array-like field with shape ``(grid.ny, grid.nx)``.
            integer: Write little-endian int32 when true; otherwise float32.

        Raises:
            ValueError: If ``value`` does not match the configured grid.
            OSError: If writing the Fortran record fails.

        Notes:
            Values are flattened in C order before the record is written.
        """
        array: NumericArray=np.asarray(value)
        if array.shape!=(self.grid.ny,self.grid.nx):
            raise ValueError('MESOPAC hourly field does not match the CALMET grid')
        self.writer.write_record(array.astype('<i4' if integer else '<f4',copy=False).ravel(order='C'))

    def append(
        self,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
        *,
        station_temperature_k: npt.ArrayLike,
        station_shortwave_flux: npt.ArrayLike,
        station_relative_humidity: npt.ArrayLike,
        station_precipitation_code: npt.ArrayLike,
    ) -> None:
        """Append one MESOPAC hourly record group.

        Args:
            hour: Solver fields for one timestamp.
            solar_elevation_sine: ``(ny, nx)`` sine of solar elevation.
            station_temperature_k: Float-like ``(surface_station,)`` air
                temperature in kelvin.
            station_shortwave_flux: Float-like ``(surface_station,)`` shortwave
                flux in watts per square metre.
            station_relative_humidity: Integer-compatible
                ``(surface_station,)`` relative humidity in percent.
            station_precipitation_code: Integer-compatible
                ``(surface_station,)`` precipitation codes.

        Raises:
            ValueError: If station vectors or hourly grid fields do not match
                their configured dimensions.
            OSError: If a sequential record cannot be written.

        Notes:
            The method writes a timestamp, ten two-dimensional grid records,
            then one record containing mean density and station data.
        """
        ns=len(self.surface_coordinates)
        station: tuple[NumericArray,NumericArray,NumericArray,NumericArray]=(
            np.asarray(station_temperature_k),np.asarray(station_shortwave_flux),
            np.asarray(station_relative_humidity),np.asarray(station_precipitation_code))
        if any(value.shape!=(ns,) for value in station):
            raise ValueError('MESOPAC station arrays must match the surface-station count')
        wind=hour.wind; surface=hour.surface; step=hour.step1
        ul,vl,uup,vup=mesopac_layer_average(wind.u,wind.v,surface.mixing_height,self.faces)
        wstar=_convective_velocity(hour,self.landuse,self.water_range)
        pgt=pgt_stability(wind.u[0],wind.v[0],solar_elevation_sine,step.cloud_fraction,
                          step.ceiling_height,self.landuse,surface.monin_obukhov_length,
                          water_range=self.water_range)
        stamp=self._i(hour.timestamp.year,hour.timestamp.timetuple().tm_yday,hour.timestamp.hour)
        self.writer.write_record(stamp)
        for value,integer in ((ul,False),(vl,False),(uup,False),(vup,False),
                              (surface.mixing_height,False),(surface.friction_velocity,False),
                              (wstar,False),(surface.monin_obukhov_length,False),(pgt,True),
                              (step.precipitation_rate,False)):
            self._grid_record(value,integer)
        density: npt.NDArray[np.float32]=np.asarray(
            surface.density,dtype=np.float32).ravel(order='C')
        average_density: np.float32=(
            np.cumsum(density,dtype=np.float32)[-1]/np.float32(density.size))
        self.writer.write_record(self._f(average_density),
            station[0].astype('<f4',copy=False),station[1].astype('<f4',copy=False),
            station[2].astype('<i4',copy=False),station[3].astype('<i4',copy=False))

    def close(self) -> None:
        """Close the sequential-unformatted output stream."""
        self.writer.close()

    def __enter__(self) -> MesopacWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`MesopacWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer on context-manager exit.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()


class AuxiliaryCalmetWriter:
    """Write CALMET.AUX 1.1 cloud-water sequential-unformatted records.

    Attributes:
        writer: Open GNU-style Fortran sequential-record writer.
        grid: Horizontal CALMET grid definition.
        faces: Float32 layer-face heights in metres.
        available: Whether cloud-water data are available and records should be
            written.
        timestep_seconds: Duration represented by each appended field group.
    """
    _fields_2d: tuple[tuple[str,str,str],...]=(
        ('CLDMRUP','G/KG','RC4'),('ZCLDBOT','MAGL','RC4'),
        ('ZCLDTOP','MAGL','RC4'))
    _fields_3d: tuple[tuple[str,str,str],...]=(('CLDMR3D','G/KG','RC4'),)
    writer: FortranSequentialWriter
    grid: GridConfig
    faces: npt.NDArray[np.float32]
    available: bool
    timestep_seconds: int

    def __init__(
        self,
        path: FilePath,
        grid: GridConfig,
        z_faces_m: npt.ArrayLike,
        start: datetime,
        end: datetime,
        *,
        control_lines: Iterable[str]=(),
        base_timezone: str='UTC-0000',
        available: bool=True,
        timestep_seconds: int=3600,
    ) -> None:
        """Open CALMET.AUX and optionally write its version 1.1 header.

        Args:
            path: Destination sequential-unformatted file.
            grid: Horizontal CALMET grid definition.
            z_faces_m: Layer-face heights in metres with shape ``(level + 1,)``.
            start: Declared run beginning datetime.
            end: Declared run ending datetime.
            control_lines: Control-file text copied into 132-byte comment
                records.
            base_timezone: Eight-character CALMET time-zone label.
            available: If false, create an empty file and make :meth:`append`
                a no-op.
            timestep_seconds: Seconds from each appended beginning time to its
                ending time.

        Raises:
            OSError: If the destination cannot be created or written.
            ValueError: If available output has a negative or non-whole-hour
                declared run interval.
            AssertionError: If the encoded fixed header no longer has its
                required 144-byte layout.

        Notes:
            When ``available`` is true, construction writes all declaration,
            comment, grid, ZFACE, and variable-table records immediately.
        """
        run_hours=(_whole_run_hours(start,end,'CALMET.AUX')
                   if available else None)
        self.writer=FortranSequentialWriter(Path(path))
        self.grid=grid
        self.faces=np.asarray(z_faces_m,dtype=np.float32)
        self.available=bool(available)
        self.timestep_seconds=int(timestep_seconds)
        if self.available:
            self._write_header(
                start,end,list(control_lines),base_timezone,
                cast(int,run_hours))

    @staticmethod
    def _i(*values: IntegerScalar) -> bytes:
        """Pack values as little-endian signed 32-bit integers.

        Args:
            *values: Integer-compatible scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as int32.
        """
        return struct.pack('<'+'i'*len(values),*values)

    @staticmethod
    def _f(*values: RealScalar) -> bytes:
        """Pack values as little-endian IEEE float32 scalars.

        Args:
            *values: Real scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as float32.
        """
        return struct.pack('<'+'f'*len(values),*values)

    @staticmethod
    def _stamp(value: datetime) -> int:
        """Encode a datetime as CALMET ``YYYYJJJHH``.

        Args:
            value: Datetime to encode.

        Returns:
            Integer ordinal-date code.

        Raises:
            ValueError: If the datetime cannot be represented.
        """
        return format_ordinal_code(value)

    @staticmethod
    def _date_parts(value: datetime) -> tuple[int,int,int,int,int]:
        """Split a datetime into CALMET calendar header fields.

        Args:
            value: Datetime to split.

        Returns:
            Tuple ``(year, month, day, hour, seconds_after_hour)``.
        """
        return value.year,value.month,value.day,value.hour,value.minute*60+value.second

    @classmethod
    def _time_bounds(cls,begin: datetime,end: datetime) -> bytes:
        """Pack one CALMET.AUX interval timestamp record.

        Args:
            begin: Interval beginning datetime.
            end: Interval ending datetime.

        Returns:
            Sixteen bytes containing begin/end ordinal codes and seconds as
            four little-endian int32 values.

        Raises:
            ValueError: If either datetime cannot be encoded.
            struct.error: If an encoded value is outside int32 range.
        """
        return cls._i(cls._stamp(begin),begin.minute*60+begin.second,
                      cls._stamp(end),end.minute*60+end.second)

    def _write_header(
        self,
        start: datetime,
        end: datetime,
        control_lines: Sequence[str],
        base_timezone: str,
        hours: int,
    ) -> None:
        """Write the complete CALMET.AUX declaration and static header.

        Args:
            start: Declared run beginning datetime.
            end: Declared run ending datetime.
            control_lines: Control text to store as comment records.
            base_timezone: CALMET time-zone label.
            hours: Integer run duration in hours.

        Raises:
            AssertionError: If the fixed run-control payload is not 144 bytes.
            OSError: If writing any sequential record fails.

        Notes:
            This writes declaration, comments, run control, float32 ZFACE, and
            the fixed two- and three-dimensional variable tables.
        """
        writer=self.writer; grid=self.grid
        writer.write_record(fixed_string('CALMET.AUX',16),fixed_string('1.1',16),
            fixed_string('Generic file structure with compression',64))
        comments=['Produced by CALMET Version: 6.5.0  Level: 150223',*control_lines]
        writer.write_record(self._i(len(comments)))
        for line in comments:
            writer.write_record(fixed_string(line.rstrip('\n'),132))
        run=(self._i(*self._date_parts(start),*self._date_parts(end))+
             fixed_string(base_timezone,8)+self._i(hours,grid.nx,grid.ny,self.faces.size-1)+
             self._f(grid.dx_km*1000.,grid.x_origin_km*1000.,grid.y_origin_km*1000.)+
             fixed_string(grid.projection.upper(),8)+fixed_string(grid.datum,8)+
             fixed_string('02-21-2003',12)+
             self._f(grid.false_easting_km,grid.false_northing_km)+
             fixed_string(grid.hemisphere.upper(),4)+self._i(grid.utm_zone)+
             self._f(grid.lat0,grid.lon0,grid.lat1,grid.lat2)+
             self._i(len(self._fields_2d),len(self._fields_3d)))
        if len(run)!=144: raise AssertionError('CALMET.AUX run-control layout changed')
        writer.write_record(run)
        writer.write_record(fixed_string('ZFACE',8),self._i(0,0,0,0),
                            self.faces.astype('<f4',copy=False))
        writer.write_record(fixed_string('2D_VARS',8),
            *(fixed_string(item[0],8) for item in self._fields_2d),
            *(fixed_string(item[1],8) for item in self._fields_2d),
            *(fixed_string(item[2],4) for item in self._fields_2d))
        writer.write_record(fixed_string('3D_VARS',8),
            *(fixed_string(item[0],8) for item in self._fields_3d),
            *(fixed_string(item[1],8) for item in self._fields_3d),
            *(fixed_string(item[2],4) for item in self._fields_3d))

    def _compressed(self,label: str,data: npt.ArrayLike) -> None:
        """Zero-run encode and write one two-dimensional auxiliary field.

        Args:
            label: Twelve-character record label.
            data: Nonnegative field with shape ``(grid.ny, grid.nx)``.

        Raises:
            ValueError: If the field shape is wrong or a value is negative.
            OSError: If the count or data record cannot be written.

        Notes:
            Values are encoded as little-endian float32 in C order. Negative
            encoded words represent zero-run lengths; physical negative values
            therefore cannot be represented.
        """
        values: npt.NDArray[np.float32]=np.asarray(data,dtype=np.float32)
        if values.shape!=(self.grid.ny,self.grid.nx):
            raise ValueError('CALMET.AUX field does not match the CALMET grid')
        flat: npt.NDArray[np.float32]=values.ravel(order='C')
        if np.any(flat<0.):
            raise ValueError('CALMET.AUX compression cannot encode negative values')
        nonzero: npt.NDArray[np.intp]=np.flatnonzero(flat!=np.float32(0.))
        if nonzero.size:
            previous: npt.NDArray[np.intp]=np.r_[-1,nonzero[:-1]]
            gaps: npt.NDArray[np.intp]=nonzero-previous-1
            has_gap: npt.NDArray[np.bool_]=gaps>0
            value_positions: npt.NDArray[np.intp]=np.arange(nonzero.size)+np.cumsum(has_gap)
            trailing=int(flat.size-nonzero[-1]-1)
            encoded: npt.NDArray[np.float32]=np.empty(
                nonzero.size+int(has_gap.sum())+int(trailing>0),dtype='<f4')
            encoded[value_positions]=flat[nonzero]
            encoded[value_positions[has_gap]-1]=-(
                gaps[has_gap].astype(np.float32)+np.float32(.0001))
            if trailing:
                encoded[-1]=-(np.float32(trailing)+np.float32(.0001))
        else:
            encoded=np.asarray([-(np.float32(flat.size)+np.float32(.0001))],dtype='<f4')
        self.writer.write_record(self._i(encoded.size))
        self.writer.write_record(fixed_string(label,12),encoded)

    def append(self,hour: HourFields) -> None:
        """Append one timestamped CALMET.AUX cloud-water field group.

        Args:
            hour: Solver output whose step-one fields provide three float32
                ``(ny, nx)`` above-model fields and one ``(level, ny, nx)``
                in-model cloud-water field. Mixing ratios use grams per
                kilogram and cloud heights use metres above ground level.

        Raises:
            ValueError: If required cloud-water fields are absent, have the
                wrong shape, or contain negative values.
            OSError: If a timestamp or compressed data record cannot be written.

        Notes:
            No records are written when ``available`` is false. Otherwise each
            two-dimensional variable and the three-dimensional variable group
            receives its own interval timestamp record.
        """
        if not self.available: return
        step=hour.step1
        fields=(step.cloud_liquid_water_above,
                step.cloud_liquid_water_above_bottom,
                step.cloud_liquid_water_above_top)
        if any(field is None for field in fields) or step.cloud_liquid_water_3d is None:
            raise ValueError('CALMET.AUX output requires prognostic cloud-water fields')
        available_fields=cast(
            tuple[npt.ArrayLike,npt.ArrayLike,npt.ArrayLike],fields)
        begin=hour.timestamp
        end=begin+timedelta(seconds=self.timestep_seconds)
        timestamp=self._time_bounds(begin,end)
        for (name,_,_),field in zip(self._fields_2d,available_fields):
            self.writer.write_record(timestamp)
            self._compressed(name,field)
        cloud: NumericArray=np.asarray(step.cloud_liquid_water_3d)
        if cloud.shape!=(self.faces.size-1,self.grid.ny,self.grid.nx):
            raise ValueError('CALMET.AUX 3-D cloud water does not match the CALMET grid')
        self.writer.write_record(timestamp)
        for index,field in enumerate(cloud,1):
            self._compressed(f'CLDMR3D {index:04d}',field)

    def close(self) -> None:
        """Close the sequential-unformatted output stream."""
        self.writer.close()

    def __enter__(self) -> AuxiliaryCalmetWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`AuxiliaryCalmetWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer on context-manager exit.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()

class LegacyCalmetWriter:
    """Write CALMET.DAT v2.1 OUTHD/OUTHR sequential records.

    Attributes:
        writer: Open GNU-style Fortran sequential-record writer.
        grid: Horizontal CALMET grid definition.
        faces: Little-endian float32 layer-face heights in metres.
        include_w_temperature: Whether W-face and temperature-layer records are
            included.
        water_range: Inclusive water-category code range.
        landuse_category_count: Number of land-use categories in the header.
        run_type: Zero for wind-only output; nonzero for complete output.
        timestep_seconds: Duration represented by each appended record group.
        precipitation_station_count: NPSTA value written to the header.
        iwfcod: Wind-field output-format code, zero or one.
        sea_station_count: NOWSTA value written to the header.
        surface_coordinates: Float64 ``(surface_station, 2)`` coordinates in
            kilometres.
        upper_coordinates: Float64 ``(upper_station, 2)`` coordinates in
            kilometres.
        precipitation_coordinates: Float64 ``(precipitation_station, 2)``
            coordinates in kilometres.
        nearest_surface_station: Optional int32 ``(ny, nx)`` grid of one-based
            station indices.
        landuse: Dominant land-use field used for water physics.
    """

    writer: FortranSequentialWriter
    grid: GridConfig
    faces: npt.NDArray[np.float32]
    include_w_temperature: bool
    water_range: tuple[int,int]
    landuse_category_count: int
    run_type: int
    timestep_seconds: int
    precipitation_station_count: int
    iwfcod: int
    sea_station_count: int
    surface_coordinates: npt.NDArray[np.float64]
    upper_coordinates: npt.NDArray[np.float64]
    precipitation_coordinates: npt.NDArray[np.float64]
    nearest_surface_station: npt.NDArray[np.int32] | None
    landuse: NumericArray

    def __init__(
        self,
        path: FilePath,
        grid: GridConfig,
        z_faces_m: npt.ArrayLike,
        geophysical: GeophysicalSource,
        start: datetime,
        end: datetime,
        *,
        control_lines: Iterable[str]=(),
        base_timezone: str='UTC-0000',
        include_w_temperature: bool=False,
        water_range: Sequence[int]=(50,55),
        landuse_category_count: int=14,
        run_type: int=1,
        timestep_seconds: int=3600,
        precipitation_station_count: int=-1,
        surface_coordinates_km: npt.ArrayLike=(),
        upper_coordinates_km: npt.ArrayLike=(),
        precipitation_coordinates_km: npt.ArrayLike=(),
        nearest_surface_station: npt.ArrayLike | None=None,
        iwfcod: int=1,
        sea_station_count: int=0,
    ) -> None:
        """Open CALMET.DAT and write all static version 2.1 records.

        Args:
            path: Destination sequential-unformatted CALMET.DAT file.
            grid: Horizontal CALMET grid definition.
            z_faces_m: Layer-face heights in metres with shape ``(level + 1,)``.
            geophysical: Mapping containing ``roughness_length``,
                ``dominant_landuse``, ``terrain_height``, and ``leaf_area_index``
                fields shaped ``(grid.ny, grid.nx)``.
            start: Declared run beginning datetime.
            end: Declared run ending datetime; the interval must contain whole
                hours.
            control_lines: Control-file text copied into comment records.
            base_timezone: Eight-character CALMET time-zone label.
            include_w_temperature: Write vertical velocity and temperature
                records when true.
            water_range: Inclusive water-category code range.
            landuse_category_count: Number of land-use categories.
            run_type: Zero for wind-only output; nonzero for complete fields.
            timestep_seconds: Seconds represented by each append operation.
            precipitation_station_count: NPSTA header value. Zero suppresses
                precipitation rate and code records.
            surface_coordinates_km: ``(surface_station, 2)`` station coordinates
                in kilometres.
            upper_coordinates_km: ``(upper_station, 2)`` station coordinates in
                kilometres.
            precipitation_coordinates_km: ``(precipitation_station, 2)``
                coordinates in kilometres.
            nearest_surface_station: Optional one-based int-like ``(ny, nx)``
                station grid. It is derived when omitted and surface stations
                exist.
            iwfcod: Wind-field format code, either 0 or 1.
            sea_station_count: Nonnegative NOWSTA header value.

        Raises:
            OSError: If the destination cannot be created or written.
            ValueError: If run duration, control codes, coordinate counts,
                nearest-station indices, or array shapes are invalid.
            AssertionError: If the encoded run-control record is not exactly
                176 bytes.

        Notes:
            Construction writes the declaration, embedded control comments,
            run control, vertical faces, station coordinates, and static
            geophysical records immediately.
        """
        run_hours=_whole_run_hours(start,end,'CALMET')
        self.writer=FortranSequentialWriter(Path(path)); self.grid=grid; self.faces=np.asarray(z_faces_m,dtype='<f4'); self.include_w_temperature=include_w_temperature
        self.water_range=cast(tuple[int,int],tuple(water_range)); self.landuse_category_count=int(landuse_category_count)
        self.run_type=int(run_type)
        self.timestep_seconds=int(timestep_seconds)
        self.precipitation_station_count=int(precipitation_station_count)
        self.iwfcod=int(iwfcod); self.sea_station_count=int(sea_station_count)
        if self.iwfcod not in {0,1}: raise ValueError('IWFCOD must be zero or one')
        if self.sea_station_count<0: raise ValueError('NOWSTA cannot be negative')
        self.surface_coordinates=np.asarray(surface_coordinates_km,dtype=float).reshape(-1,2)
        self.upper_coordinates=np.asarray(upper_coordinates_km,dtype=float).reshape(-1,2)
        self.precipitation_coordinates=np.asarray(precipitation_coordinates_km,dtype=float).reshape(-1,2)
        shape=(grid.ny,grid.nx)
        if nearest_surface_station is None and len(self.surface_coordinates):
            x: npt.NDArray[np.float64]=np.asarray(
                grid.x_origin_km+(np.arange(grid.nx)+.5)*grid.dx_km,
                dtype=np.float64)
            y: npt.NDArray[np.float64]=np.asarray(
                grid.y_origin_km+(np.arange(grid.ny)+.5)*grid.dy_km,
                dtype=np.float64)
            gx: npt.NDArray[np.float64]
            gy: npt.NDArray[np.float64]
            gx,gy=np.meshgrid(x,y)
            distance: npt.NDArray[np.float64]=(
                (gx[...,None]-self.surface_coordinates[:,0])**2+
                (gy[...,None]-self.surface_coordinates[:,1])**2)
            nearest_surface_station=np.argmin(distance,axis=2)+1
        self.nearest_surface_station=(None if nearest_surface_station is None else
            np.asarray(nearest_surface_station,dtype=np.int32))
        if self.nearest_surface_station is not None:
            if self.nearest_surface_station.shape!=shape:
                raise ValueError('nearest surface-station grid does not match CALMET grid')
            if (np.any(self.nearest_surface_station<1) or
                np.any(self.nearest_surface_station>len(self.surface_coordinates))):
                raise ValueError('nearest surface-station index is out of range')
        if self.precipitation_station_count>0 and len(self.precipitation_coordinates)!=self.precipitation_station_count:
            raise ValueError('precipitation coordinates must match positive NPSTA')
        self.landuse=self._array(geophysical['dominant_landuse']); self._write_header(
            geophysical,start,end,list(control_lines),base_timezone,run_hours)

    @staticmethod
    def _array(value: ArraySource) -> NumericArray:
        """Materialize a netCDF variable or array-like value.

        Args:
            value: netCDF variable or array-like object.

        Returns:
            NumPy array preserving the source shape and dtype.
        """
        return _array_data(value)

    @staticmethod
    def _date_parts(value: datetime) -> tuple[int,int,int,int,int]:
        """Split a datetime into CALMET calendar header fields.

        Args:
            value: Datetime to split.

        Returns:
            Tuple ``(year, month, day, hour, seconds_after_hour)``.
        """
        return (value.year,value.month,value.day,value.hour,value.minute*60+value.second)

    @staticmethod
    def _stamp(value: datetime) -> int:
        """Encode a datetime as CALMET ``YYYYJJJHH``.

        Args:
            value: Datetime to encode.

        Returns:
            Integer ordinal-date code.

        Raises:
            ValueError: If the datetime cannot be represented.
        """
        return format_ordinal_code(value)

    @staticmethod
    def _i(*values: IntegerScalar) -> bytes:
        """Pack values as little-endian signed 32-bit integers.

        Args:
            *values: Integer-compatible scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as int32.
        """
        return struct.pack('<'+'i'*len(values),*values)

    @staticmethod
    def _f(*values: RealScalar) -> bytes:
        """Pack values as little-endian IEEE float32 scalars.

        Args:
            *values: Real scalar values.

        Returns:
            Packed byte string containing four bytes per value.

        Raises:
            struct.error: If a value cannot be represented as float32.
        """
        return struct.pack('<'+'f'*len(values),*values)

    def _field_record(
        self,
        label: str,
        data: ArraySource,
        begin: datetime | None=None,
        end: datetime | None=None,
        integer: bool=False,
    ) -> None:
        """Write one labeled static or time-bounded CALMET field record.

        Args:
            label: Field label stored in an eight-byte fixed string.
            data: Array-like data flattened in C order.
            begin: Optional interval beginning. Both time arguments must be
                supplied together.
            end: Optional interval ending.
            integer: Write little-endian int32 when true; otherwise float32.

        Raises:
            ValueError: If only one interval boundary is supplied or a datetime
                cannot be encoded.
            OSError: If the sequential record cannot be written.

        Notes:
            Static records receive four zero int32 time fields. Timed records
            receive ordinal begin/end codes and seconds after the hour.
        """
        if begin is None:
            if end is not None:
                raise ValueError('CALMET field end time requires a begin time')
            times: bytes=self._i(0,0,0,0)
        else:
            if end is None:
                raise ValueError('CALMET field begin time requires an end time')
            times=self._i(self._stamp(begin),begin.minute*60+begin.second,
                          self._stamp(end),end.minute*60+end.second)
        dtype: str='<i4' if integer else '<f4'; array: NumericArray=self._array(data).astype(dtype,copy=False)
        self.writer.write_record(fixed_string(label,8),times,array.ravel(order='C'))

    def _write_header(
        self,
        geo: GeophysicalSource,
        start: datetime,
        end: datetime,
        control_lines: Sequence[str],
        base_timezone: str,
        run_hours: int,
    ) -> None:
        """Write declaration, run control, coordinates, and static fields.

        Args:
            geo: Geophysical mapping used by the constructor.
            start: Declared run beginning datetime.
            end: Declared run ending datetime.
            control_lines: Embedded control-file comment lines.
            base_timezone: CALMET time-zone label.
            run_hours: Integer run duration in hours.

        Raises:
            AssertionError: If the fixed run-control payload is not 176 bytes.
            ValueError: If a static field or timestamp cannot be encoded.
            OSError: If writing any sequential record fails.
        """
        w=self.writer; g=self.grid
        w.write_record(fixed_string('CALMET.DAT',16),fixed_string('2.1',16),fixed_string('No-Obs file structure with embedded control file',64))
        comments: list[str]=[
            'Produced by PyCALMET NumPy implementation',
            'Internal Coordinate Transformations: pyproj/PROJ',
        ]+list(control_lines)
        w.write_record(self._i(len(comments)))
        for line in comments: w.write_record(fixed_string(line.rstrip('\n'),132))
        run=self._i(*self._date_parts(start),*self._date_parts(end))+fixed_string(base_timezone,8)
        run+=self._i(run_hours,self.run_type,g.nx,g.ny,self.faces.size-1)
        run+=self._f(g.dx_km*1000.,g.x_origin_km*1000.,g.y_origin_km*1000.)
        run+=self._i(self.iwfcod,len(self.surface_coordinates),len(self.upper_coordinates),
                     self.precipitation_station_count,self.sea_station_count,self.landuse_category_count,
                     *self.water_range,int(self.include_w_temperature))
        run+=fixed_string(g.projection.upper(),8)+fixed_string(g.datum,8)+fixed_string('02-21-2003',12)
        run+=self._f(g.false_easting_km,g.false_northing_km)+fixed_string(g.hemisphere.upper(),4)+self._i(g.utm_zone)+self._f(g.lat0,g.lon0,g.lat1,g.lat2)
        if len(run)!=176: raise AssertionError('CALMET run-control record layout changed')
        w.write_record(run)
        self._field_record('ZFACE',self.faces)
        if len(self.surface_coordinates):
            self._field_record('XSSTA',self.surface_coordinates[:,0]*1000.)
            self._field_record('YSSTA',self.surface_coordinates[:,1]*1000.)
        if len(self.upper_coordinates):
            self._field_record('XUSTA',self.upper_coordinates[:,0]*1000.)
            self._field_record('YUSTA',self.upper_coordinates[:,1]*1000.)
        if len(self.precipitation_coordinates):
            self._field_record('XPSTA',self.precipitation_coordinates[:,0]*1000.)
            self._field_record('YPSTA',self.precipitation_coordinates[:,1]*1000.)
        self._field_record('Z0',self._array(geo['roughness_length']))
        self._field_record('ILANDU',geo['dominant_landuse'],integer=True)
        self._field_record('ELEV',self._array(geo['terrain_height']))
        self._field_record('XLAI',self._array(geo['leaf_area_index']))
        if len(self.surface_coordinates):
            self._field_record(
                'NEARS',cast(npt.NDArray[np.int32],
                             self.nearest_surface_station),integer=True)

    def append(
        self,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
        *,
        surface_relative_humidity: npt.ArrayLike | None=None,
        precipitation_rate: npt.ArrayLike | None=None,
        precipitation_code: npt.ArrayLike | None=None,
    ) -> None:
        """Append one OUTHR time-bounded field group.

        Args:
            hour: Solver output for the interval beginning at ``hour.timestamp``.
            solar_elevation_sine: ``(ny, nx)`` sine of solar elevation used for
                PGT stability classification.
            surface_relative_humidity: Optional ``(ny, nx)`` percent field;
                defaults to the step-one result.
            precipitation_rate: Optional ``(ny, nx)`` rate in millimetres per
                hour; defaults to the step-one result.
            precipitation_code: Optional integer ``(ny, nx)`` precipitation
                code; defaults to the step-one result.

        Raises:
            ValueError: If a timestamp cannot be encoded or a supplied field
                cannot be converted to the required record dtype.
            OSError: If writing any sequential record fails.

        Notes:
            U/V records are always written. Run type zero returns after winds;
            other runs add optional W/T records, stability and surface fields.
            Precipitation fields are omitted when NPSTA is zero.
        """
        begin=hour.timestamp; end=begin+timedelta(seconds=self.timestep_seconds); wind=hour.wind; surface=hour.surface
        if surface_relative_humidity is None: surface_relative_humidity=hour.step1.surface_relative_humidity
        if precipitation_rate is None: precipitation_rate=hour.step1.precipitation_rate
        if precipitation_code is None: precipitation_code=hour.step1.precipitation_code
        for k in range(wind.u.shape[0]):
            self._field_record(f'U-LEV{k+1:3d}',wind.u[k],begin,end); self._field_record(f'V-LEV{k+1:3d}',wind.v[k],begin,end)
            if self.include_w_temperature: self._field_record(f'WFACE{k+1:3d}',wind.w[k+1],begin,end)
        if self.run_type==0: return
        if self.include_w_temperature:
            for k in range(hour.temperature.shape[0]): self._field_record(f'T-LEV{k+1:3d}',hour.temperature[k],begin,end)
        pgt=pgt_stability(wind.u[0],wind.v[0],solar_elevation_sine,hour.step1.cloud_fraction,hour.step1.ceiling_height,self.landuse,surface.monin_obukhov_length,water_range=self.water_range)
        wstar=_convective_velocity(hour,self.landuse,self.water_range)
        fields: list[tuple[str,ArraySource,bool]]=[
            ('IPGT',pgt,True),('USTAR',surface.friction_velocity,False),
            ('ZI',surface.mixing_height,False),
            ('EL',surface.monin_obukhov_length,False),('WSTAR',wstar,False)]
        if self.precipitation_station_count!=0:
            fields.append(('RMM',precipitation_rate,False))
        fields.extend((('TEMPK',hour.temperature[0],False),('RHO',surface.density,False),
                       ('QSW',surface.shortwave_flux,False),('IRH',surface_relative_humidity,True)))
        if self.precipitation_station_count!=0:
            fields.append(('IPCODE',precipitation_code,True))
        for label,data,integer in fields: self._field_record(label,data,begin,end,integer)

    def close(self) -> None:
        """Close the sequential-unformatted output stream."""
        self.writer.close()

    def __enter__(self) -> LegacyCalmetWriter:
        """Return this open writer for context-manager use.

        Returns:
            The :class:`LegacyCalmetWriter` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer on context-manager exit.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.close()
