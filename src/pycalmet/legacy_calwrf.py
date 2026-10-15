"""Writers for CALWRF 2.0.2 formatted M3D/M2D interchange files."""

from __future__ import annotations

from .datetime_utils import parse_timestamp
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO, cast

import netCDF4 as nc
import numpy as np
import numpy.typing as npt

from ._common import fortran_nint
from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS,Transformer


def _nint(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Round values with the CALWRF default-REAL ``NINT`` convention.

    Args:
        values: Scalar or array-like numeric values. Inputs are rounded to
            single precision before applying half-away-from-zero rounding.

    Returns:
        A NumPy ``int64`` array with the same shape as ``values``.
    """
    return fortran_nint(values)


def _r4_scale(
    value: npt.ArrayLike,scale: float
) -> np.float32 | npt.NDArray[np.float32]:
    """Apply a CALWRF default-REAL multiplication without promotion.

    Args:
        value: Scalar or array-like values to scale.
        scale: Scalar conversion factor.

    Returns:
        A NumPy ``float32`` scalar or array, matching the shape of ``value``.
    """
    return np.float32(np.float32(cast(float,value))*np.float32(scale))


def _r4_divide(
    value: npt.ArrayLike,divisor: float
) -> np.float32 | npt.NDArray[np.float32]:
    """Apply a CALWRF default-REAL division without promotion.

    Args:
        value: Scalar or array-like dividend.
        divisor: Scalar divisor.

    Returns:
        A NumPy ``float32`` scalar or array, matching the shape of ``value``.
    """
    return np.float32(np.float32(cast(float,value))/np.float32(divisor))


def _relative_humidity(
    qvapor: npt.ArrayLike,
    pressure_hpa: npt.ArrayLike,
    temperature_k: npt.ArrayLike,
) -> np.float32 | npt.NDArray[np.float32]:
    """Reproduce CALWRF's single-precision relative-humidity diagnostic.

    Args:
        qvapor: Water-vapour mixing ratio in kg/kg. The input may be any shape
            broadcastable with ``pressure_hpa`` and ``temperature_k``.
        pressure_hpa: Atmospheric pressure in hPa.
        temperature_k: Air temperature in K.

    Returns:
        A ``float32`` array of relative humidity in percent, clipped to the
        inclusive range 0--100 and broadcast to the common input shape.
    """
    f32: type[np.float32]=np.float32
    temperature: npt.NDArray[np.float32]=np.asarray(temperature_k,dtype=f32)
    pressure: npt.NDArray[np.float32]=np.asarray(pressure_hpa,dtype=f32)
    vapor: npt.NDArray[np.float32]=np.asarray(qvapor,dtype=f32)
    # GETRH1 receives QQ after the caller has converted kg/kg to g/kg,
    # then converts it back.  Both assignments round to default REAL.
    qq: npt.NDArray[np.float32]=np.asarray(vapor*f32(1000.),dtype=f32)
    qq2: npt.NDArray[np.float32]=np.asarray(qq/f32(1000.),dtype=f32)
    numerator: npt.NDArray[np.float32]=np.asarray(f32(17.67)*np.asarray(
        temperature-f32(273.15),dtype=f32),dtype=f32)
    denominator: npt.NDArray[np.float32]=np.asarray(
        temperature-f32(29.65),dtype=f32
    )
    exponent: npt.NDArray[np.float32]=np.asarray(
        numerator/denominator,dtype=f32
    )
    exponential: npt.NDArray[np.float32]=np.asarray(
        np.exp(exponent.astype(np.float64)),dtype=f32
    )
    coefficient: npt.NDArray[np.float32]=np.asarray(
        f32(.622)*f32(6.112),dtype=f32
    )
    saturation: npt.NDArray[np.float32]=np.asarray(
        np.asarray(coefficient/pressure,dtype=f32)*exponential,dtype=f32
    )
    humidity: npt.NDArray[np.float32]=np.asarray(
        np.asarray(f32(100.)*qq2,dtype=f32)/saturation,dtype=f32
    )
    return np.minimum(np.maximum(humidity,f32(0.)),f32(100.)).astype(f32)


def _date_number(text: str) -> int:
    """Encode a canonical whole-hour timestamp as ``YYYYMMDDHH``.

    Args:
        text: Timestamp string accepted by
            :func:`pycalmet.datetime_utils.parse_timestamp`.

    Returns:
        The ten-digit CALWRF timestamp as an integer.

    Raises:
        ValueError: If ``text`` is invalid or contains nonzero minutes,
            seconds, or microseconds that the legacy format cannot represent.
    """
    timestamp: datetime=parse_timestamp(text)
    if timestamp.minute or timestamp.second or timestamp.microsecond:
        raise ValueError(
            'legacy CALWRF M2D/M3D output requires whole-hour canonical WRF times'
        )
    return int(timestamp.strftime('%Y%m%d%H'))


def _compressed_water(values: npt.ArrayLike) -> str:
    """Format one M3D row of cloud and precipitation mixing ratios.

    Args:
        values: One-dimensional cloud-water component values in g/kg. Negative
            values are written as zero.

    Returns:
        Concatenated six-column ``F6.3`` fields. If more than one component is
        below 0.00049 g/kg, a single negative run-count field is returned.
    """
    values=np.maximum(np.asarray(values,dtype=np.float32),np.float32(0.))
    if values.size>1 and np.all(values<np.float32(.00049)):
        return f'{-float(values.size):6.3f}'
    return ''.join(f'{value:6.3f}' for value in values)


def _header(
    stream: TextIO,
    dataset: nc.Dataset,
    name: str,
    variables_2d: Sequence[tuple[str,str,str]] = (),
) -> None:
    """Write the common CALWRF file identifier and comment records.

    Args:
        stream: Open text stream positioned at the beginning of an output
            file.
        dataset: Canonical CALWRF NetCDF dataset. It is accepted alongside the
            other header writers for a consistent internal interface.
        name: Legacy dataset identifier, normally ``"3D.DAT"`` or
            ``"2D.DAT"``.
        variables_2d: Sequence of ``(name, units, description)`` metadata
            triples included in an M2D header.

    Returns:
        None. Header records are appended to ``stream``.

    Raises:
        OSError: If the output stream cannot be written.
    """
    version: str='2.1'
    modification: str='WRF_ARW         Python          Created by PyCALMET'
    stream.write(f'{name:<16}{version:<16}{modification:<64}\n')
    # RDHD53 does not merely display this comment: when it sees ``CALWRF`` it
    # parses the six columns after ``Level: `` as an integer and applies a
    # historical precipitation QA guard.  Keep the original producer grammar
    # so a PyCALMET M3D remains readable by GNU CALMET as well as by Python.
    comments: list[str]=[
        f'Produced by CALWRF {"v2.0.2":<12}  Level: {"131108":<8}'
    ]
    if variables_2d:
        comments.extend([f'{len(variables_2d):3d}  Number of 2D Variables',
                         *(f'{index:3d}  {name:<10} {units:<11} {description}'
                           for index,(name,units,description) in enumerate(variables_2d,1))])
    stream.write(f'{len(comments):4d}\n')
    for comment in comments:
        stream.write(f'{comment:<132}\n')


def _common_header(
    stream: TextIO,
    dataset: nc.Dataset,
    times: Sequence[str],
    iout: Sequence[int],
) -> None:
    """Write the projection, period, vertical, and static-grid records.

    Args:
        stream: Open legacy-format text stream positioned after its comments.
        dataset: Canonical CALWRF NetCDF dataset containing ``latitude``,
            ``longitude``, ``sigma``, ``terrain``, and ``landuse`` variables,
            together with ``time``, ``level``, ``y``, and ``x`` dimensions.
        times: Ordered canonical timestamp strings; the first timestamp and
            sequence length define the legacy output period.
        iout: Six integer output switches written as three-column fields.

    Returns:
        None. Grid metadata and ``(y, x)`` static-cell records are appended to
        ``stream``. Coordinates are degrees, origins and spacing are km, and
        terrain is rounded to integer metres.

    Raises:
        KeyError: If a required NetCDF dimension or variable is missing.
        ValueError: If the map projection is unsupported, source-grid bounds
            are inconsistent, or the first timestamp is invalid.
        OSError: If records cannot be written to ``stream``.
    """
    map_projection: int=int(getattr(dataset,'map_proj',1))
    try:
        projection: str={1:'LCC',2:'PS',3:'EM'}[map_projection]
    except KeyError as error:
        raise ValueError(
            f'legacy CALWRF output does not support MAP_PROJ={map_projection}'
        ) from error
    ny: int=len(dataset.dimensions['y'])
    nx: int=len(dataset.dimensions['x'])
    nz: int=len(dataset.dimensions['level'])
    source_nx: int=int(getattr(dataset,'source_grid_nx',nx))
    source_ny: int=int(getattr(dataset,'source_grid_ny',ny))
    source_nz: int=int(getattr(dataset,'source_grid_nz',nz))
    i_start: int=int(getattr(dataset,'source_i_start',1))
    j_start: int=int(getattr(dataset,'source_j_start',1))
    k_start: int=int(getattr(dataset,'source_k_start',1))
    i_end: int=int(getattr(dataset,'source_i_end',i_start+nx-1))
    j_end: int=int(getattr(dataset,'source_j_end',j_start+ny-1))
    k_end: int=int(getattr(dataset,'source_k_end',k_start+nz-1))
    if ((i_end-i_start+1,j_end-j_start+1,k_end-k_start+1)!=(nx,ny,nz) or
            not (1<=i_start<=i_end<=source_nx and
                 1<=j_start<=j_end<=source_ny and
                 1<=k_start<=k_end<=source_nz)):
        raise ValueError('canonical CALWRF source-grid bounds are inconsistent')
    stream.write(''.join(f'{value:3d}' for value in iout)+'\n')
    lat0: float=float(getattr(dataset,'moad_cen_lat',getattr(dataset,'cen_lat',
                                                     getattr(dataset,'latitude_origin',0.))))
    lon0: float=float(getattr(dataset,'stand_lon',getattr(dataset,'longitude_origin',0.)))
    lat1: float=float(getattr(dataset,'truelat1',0.))
    lat2: float=float(getattr(dataset,'truelat2',0.))
    dx: float=float(getattr(dataset,'wrf_dx_m',1000.))/1000.
    lat: npt.NDArray[np.generic]=np.asarray(dataset['latitude'][:])
    lon: npt.NDArray[np.generic]=np.asarray(dataset['longitude'][:])
    projected: CRS
    if projection=='LCC':
        projected=CRS.from_proj4(f'+proj=lcc +lat_0={lat0} +lon_0={lon0} '
            f'+lat_1={lat1} +lat_2={lat2} +R=6370000 +units=km +no_defs')
    elif projection=='PS':
        pole: float=90. if lat0>=0. else -90.
        projected=CRS.from_proj4(f'+proj=stere +lat_0={pole} +lon_0={lon0} '
            f'+lat_ts={lat1} +R=6370000 +units=km +no_defs')
    else:
        projected=CRS.from_proj4(f'+proj=merc +lon_0={lon0} +lat_ts={lat1} '
            f'+R=6370000 +units=km +no_defs')
    if hasattr(dataset,'source_x_origin_km') and hasattr(dataset,'source_y_origin_km'):
        x1: float=float(dataset.source_x_origin_km)
        y1: float=float(dataset.source_y_origin_km)
    else:
        source_latitude: float=float(getattr(dataset,'source_latitude_1_1',lat[0,0]))
        source_longitude: float=float(getattr(dataset,'source_longitude_1_1',lon[0,0]))
        x1,y1=Transformer.from_crs(CRS.from_proj4('+proj=longlat +R=6370000 +no_defs'),
                                    projected,always_xy=True).transform(
                                        source_longitude,source_latitude)
    stream.write(f'{projection:<4}{lat0:9.4f}{lon0:10.4f}{lat1:7.2f}{lat2:7.2f}'
                 f'{x1:10.3f}{y1:10.3f}{dx:8.3f}'
                 f'{source_nx:4d}{source_ny:4d}{source_nz:3d}\n')
    physics: list[int]=[1,int(getattr(dataset,'mp_physics',0))+10,
             int(getattr(dataset,'cu_physics',0))+10,
             int(getattr(dataset,'bl_pbl_physics',0))+10,
             int(getattr(dataset,'ra_sw_physics',0))+10,
             int(getattr(dataset,'sf_surface_physics',0))+10,
             int(getattr(dataset,'grid_fdda',0)),int(getattr(dataset,'obs_nudge_opt',0)),
             *([0]*12),25]
    stream.write(''.join(f'{value:3d}' for value in physics)+'\n')
    start: int=_date_number(times[0])
    stream.write(f'{start:10d}{len(times):5d}{nx:4d}{ny:4d}{nz:4d}\n')
    stream.write(f'{i_start:4d}{j_start:4d}{i_end:4d}{j_end:4d}{k_start:4d}{k_end:4d}'
                 f'{lon.min():10.4f}{lon.max():10.4f}{lat.min():9.4f}{lat.max():9.4f}\n')
    for sigma in dataset['sigma'][:]:
        stream.write(f'{float(sigma):6.3f}\n')
    terrain: npt.NDArray[np.generic]=np.asarray(dataset['terrain'][:])
    landuse: npt.NDArray[np.generic]=np.asarray(dataset['landuse'][:])
    for j in range(ny):
        lines: list[str]=[]
        for i in range(nx):
            lines.append(f'{i+i_start:4d}{j+j_start:4d}{lat[j,i]:9.4f}{lon[j,i]:10.4f}'
                         f'{int(_nint(terrain[j,i])):5d}{int(_nint(landuse[j,i])):3d} '
                         f'{-999.:9.4f}{-999.:10.4f}{-999:5d}\n')
        stream.writelines(lines)


def write_m3d(
    canonical_path: str | Path,
    m3d_path: str | Path,
    m2d_path: str | Path | None = None,
) -> None:
    """Convert canonical CALWRF NetCDF to formatted M3D and optional M2D.

    Args:
        canonical_path: Path to the canonical CALWRF NetCDF input.
        m3d_path: Destination for formatted ``3D.DAT`` records. Parent
            directories are created when needed.
        m2d_path: Optional destination for formatted ``2D.DAT`` records.
            Parent directories are created when needed.

    Returns:
        None. The function writes ``m3d_path`` and, when requested,
        ``m2d_path``. Existing files at those paths are replaced.

    Raises:
        KeyError: If a required canonical dimension or variable is absent.
        ValueError: If a timestamp is not an exact hour, projection metadata
            is unsupported, or source-grid bounds are inconsistent.
        OSError: If an input cannot be opened or an output cannot be created
            or written. A failure after output creation can leave a partial
            legacy file.
    """
    m3d_path=Path(m3d_path); m3d_path.parent.mkdir(parents=True,exist_ok=True)
    with nc.Dataset(canonical_path) as dataset:
        times: list[str]=[
            str(value) for value in nc.chartostring(dataset['time_text'][:]).tolist()
        ]
        # M2D/M3D timestamps have hour precision.  Validate every canonical
        # time before opening either output so minutes or seconds are never
        # silently discarded by the fixed-width legacy representation.
        for text in times: _date_number(text)
        cloud_count: int=int(getattr(dataset,'legacy_cloud_component_count',5))
        iout: tuple[int,...]=(1,1,int(cloud_count>=1),int(cloud_count>=4),
                                  int(cloud_count>=5),int(m2d_path is not None))
        with m3d_path.open('w',encoding='ascii') as stream:
            _header(stream,dataset,'3D.DAT')
            _common_header(stream,dataset,times,iout)
            nt: int=len(times)
            ny: int=len(dataset.dimensions['y'])
            nx: int=len(dataset.dimensions['x'])
            for time_index in range(nt):
                date: int=_date_number(times[time_index])
                arrays: dict[str,npt.NDArray[np.generic]]={
                    name:np.asarray(dataset[name][time_index]) for name in (
                    'surface_pressure','precipitation','snow_cover','shortwave_down',
                    'longwave_down','temperature_2m','qvapor_2m','wind_direction_10m',
                    'wind_speed_10m','sst','pressure','height','temperature','wind_direction',
                    'wind_speed','vertical_velocity','qvapor','qcloud','qrain',
                    'qice','qsnow','qgraupel')}
                rh: np.float32 | npt.NDArray[np.float32]=_relative_humidity(
                    arrays['qvapor'],arrays['pressure'],arrays['temperature']
                )
                surface_pressure: npt.NDArray[np.float32]=cast(
                    npt.NDArray[np.float32],_r4_divide(
                        arrays['surface_pressure'],100.
                    )
                )
                surface_rain: npt.NDArray[np.float32]=cast(
                    npt.NDArray[np.float32],_r4_divide(
                        arrays['precipitation'],10.
                    )
                )
                surface_snow: npt.NDArray[np.int64]=_nint(arrays['snow_cover'])
                surface_q: npt.NDArray[np.float32]=np.maximum(
                    _r4_scale(arrays['qvapor_2m'],1000.),np.float32(0.)
                )
                surface_direction: npt.NDArray[np.generic]=np.where(
                    arrays['wind_direction_10m']<np.float32(.05),np.float32(360.),
                    arrays['wind_direction_10m'])
                level_pressure: npt.NDArray[np.int64]=_nint(arrays['pressure'])
                level_height: npt.NDArray[np.int64]=_nint(arrays['height'])
                level_direction: npt.NDArray[np.int64]=_nint(arrays['wind_direction'])
                level_direction=np.where(level_direction==0,360,level_direction)
                level_vertical: npt.NDArray[np.float32]=np.where(
                    np.abs(arrays['vertical_velocity'])<np.float32(.00499),np.float32(0.),
                    arrays['vertical_velocity']).astype(np.float32)
                level_rh: npt.NDArray[np.int64]=np.clip(_nint(rh),1,100)
                level_q: npt.NDArray[np.float32]=np.maximum(
                    _r4_scale(arrays['qvapor'],1000.),np.float32(0.)
                )
                level_water: npt.NDArray[np.float32]=np.stack(tuple(_r4_scale(arrays[name],1000.) for name in
                    ('qcloud','qrain','qice','qsnow','qgraupel')),axis=-1)[...,:cloud_count]
                for j in range(ny):
                    lines: list[str]=[]
                    for i in range(nx):
                        i_source: int=i+int(getattr(dataset,'source_i_start',1))
                        j_source: int=j+int(getattr(dataset,'source_j_start',1))
                        lines.append(f'{date:10d}{i_source:3d}{j_source:3d}'
                            f'{surface_pressure[j,i]:7.1f}'
                            f'{surface_rain[j,i]:5.2f}'
                            f'{int(surface_snow[j,i]):2d}'
                            f'{arrays["shortwave_down"][j,i]:8.1f}'
                            f'{arrays["longwave_down"][j,i]:8.1f}'
                            f'{arrays["temperature_2m"][j,i]:8.1f}'
                            f'{surface_q[j,i]:8.2f}'
                            f'{surface_direction[j,i]:8.1f}{arrays["wind_speed_10m"][j,i]:8.1f}'
                            f'{arrays["sst"][j,i]:8.1f}\n')
                        for k in range(arrays['pressure'].shape[0]):
                            basic: str=(f'{int(level_pressure[k,j,i]):4d}'
                                   f'{int(level_height[k,j,i]):6d}'
                                   f'{arrays["temperature"][k,j,i]:6.1f}'
                                   f'{int(level_direction[k,j,i]):4d}'
                                   f'{arrays["wind_speed"][k,j,i]:5.1f}'
                                   f'{level_vertical[k,j,i]:6.2f}'
                                   f'{int(level_rh[k,j,i]):3d}'
                                   f'{level_q[k,j,i]:5.2f}')
                            lines.append(basic+_compressed_water(
                                level_water[k,j,i])+'\n')
                    stream.writelines(lines)
        if m2d_path is not None:
            _write_m2d(dataset,times,Path(m2d_path),iout)


def _write_m2d(
    dataset: nc.Dataset,
    times: Sequence[str],
    path: Path,
    iout: Sequence[int],
) -> None:
    """Write canonical surface variables as a formatted CALWRF M2D file.

    Args:
        dataset: Open canonical CALWRF NetCDF dataset containing all seventeen
            surface variables emitted by CALWRF.
        times: Ordered canonical timestamp strings, one per ``time`` record.
        path: Destination ``2D.DAT`` path.
        iout: Six integer output switches shared with the companion M3D file.

    Returns:
        None. The destination parent is created and ``path`` is replaced with
        fixed-width rows ordered by time, variable, and north-to-south grid
        row.

    Raises:
        KeyError: If a required surface variable is missing.
        ValueError: If a timestamp is invalid or cannot be represented at
            whole-hour precision.
        OSError: If the destination directory or file cannot be created or
            written. A write failure can leave a partial file.
    """
    variables: tuple[tuple[str,str,str,str,float],...]=(
        ('PSFC','hPa','SURFACE PRESSURE','surface_pressure',.01),
        ('RAIN','cm','TOTAL PRECIPITATION','precipitation',.1),
        ('SNOWC','none','FRACTIONAL SNOW COVER','snow_cover',1.),
        ('SWDOWN','W','DOWNWARD SHORT WAVE FLUX','shortwave_down',1.),
        ('GLW','W','DOWNWARD LONG WAVE FLUX','longwave_down',1.),
        ('T2','K','TEMPERATURE AT 2 M','temperature_2m',1.),
        ('Q2','g kg-1','WATER VAPOR MIXING RATIO AT 2 M','qvapor_2m',1000.),
        ('WD10','degree','WIND DIRECTION AT 10 M','wind_direction_10m',1.),
        ('WS10','m','WIND SPEED AT 10 M','wind_speed_10m',1.),
        ('SST','K','SEA SURFACE TEMPERATURE','sst',1.),
        ('TSK','K','SURFACE SKIN TEMPERATURE','skin_temperature',1.),
        ('PBLH','m','PBL HEIGHT','pbl_height',1.),
        ('HFX','W','UPWARD HEAT FLUX AT SURFACE','sensible_heat_flux',1.),
        ('LH','W','LATENT HEAT FLUX AT SURFACE','latent_heat_flux',1.),
        ('UST','m','U* IN SIMILARITY THEORY','friction_velocity',1.),
        ('RAINC','cm','CUMULUS PRECIPITATION','convective_precipitation',.1),
        ('RAINNC','cm','GRID SCALE PRECIPITATION','grid_precipitation',.1))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='ascii') as stream:
        _header(stream,dataset,'2D.DAT',tuple(item[:3] for item in variables))
        _common_header(stream,dataset,times,iout)
        for time_index,text in enumerate(times):
            date: int=_date_number(text)
            for label,_units,_description,name,scale in variables:
                stream.write(f'{date:010d}  {label:<8}\n')
                values: npt.NDArray[np.generic]=np.asarray(
                    dataset[name][time_index]
                )*scale
                for row in values[::-1]:
                    stream.writelines(
                        ''.join(f'{value:10.3f}' for value in row[start:start+8])+'\n'
                        for start in range(0,row.size,8))
