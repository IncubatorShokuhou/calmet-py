"""Vectorized WRF extraction corresponding to CALWRF 2.0.2."""

from __future__ import annotations

from datetime import datetime,timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np

from ._common import fortran_inclusive_slice
from ._calwrf_listing import write_calwrf_listing
from ._path_safety import validate_calwrf_paths

from .config import CalwrfConfig
from .legacy_calwrf import write_m3d
from .datetime_utils import parse_timestamp

_LIMITS_3D = {
    "pressure": (10.0, 1100.0), "temperature": (150.0, 330.0),
    "wind_direction": (0.0, 360.0), "wind_speed": (0.0, 150.0),
    "vertical_velocity": (-50.0, 50.0), "qvapor": (0.0, 0.05),
    "qcloud": (0.0, 0.05), "qrain": (0.0, 0.05),
    "qice": (0.0, 0.05), "qsnow": (0.0, 0.05), "qgraupel": (0.0, 0.05),
}
_LIMITS_2D = {
    "surface_pressure": (50000.0, 110000.0), "precipitation": (0.0, 500.0),
    "snow_cover": (0.0, 1.1), "shortwave_down": (0.0, 1500.0),
    "longwave_down": (0.0, 1000.0), "temperature_2m": (210.0, 330.0),
    "qvapor_2m": (0.0, 0.05), "wind_direction_10m": (0.0, 360.0),
    "wind_speed_10m": (0.0, 50.0), "sst": (0.0, 350.0),
}


def _horizontal_range_check(values, low, high, ys, xs):
    """Apply CALWRF INTERPH to a selected horizontal domain."""
    original = np.asarray(values, dtype="f4")
    result = original[..., ys, xs].copy()
    ny, nx = original.shape[-2:]
    jb, je, ib, ie = ys.start + 1, ys.stop, xs.start + 1, xs.stop
    jb_edge, je_edge = min(jb, 2), max(je, ny - 1)
    ib_edge, ie_edge = min(ib, 2), max(ie, nx - 1)
    jj, ii = np.arange(jb, je + 1), np.arange(ib, ie + 1)
    j1 = np.maximum(jj - 1, jb_edge) - 1
    j2 = np.minimum(jj + 1, je_edge) - 1
    i1 = np.maximum(ii - 1, ib_edge) - 1
    i2 = np.minimum(ii + 1, ie_edge) - 1
    neighbours = np.stack((
        original[..., jj[:, None] - 1, i1[None, :]],
        original[..., jj[:, None] - 1, i2[None, :]],
        original[..., j1[:, None], ii[None, :] - 1],
        original[..., j2[:, None], ii[None, :] - 1],
    ), axis=0)
    good = (neighbours >= low) & (neighbours <= high)
    count = good.sum(axis=0)
    total = np.where(good, neighbours, 0.0).sum(axis=0, dtype="f4")
    fallback = np.where(result < low, low, high).astype("f4")
    replacement = np.divide(total, count, out=fallback, where=count > 0)
    return np.where((result < low) | (result > high), replacement, result).astype("f4")


def _vertical_range_check(values, low, high, zs, ys, xs):
    """Apply CALWRF's two-level INTERPH rule to hydrometeors."""
    original = np.asarray(values, dtype="f4")
    result = original[zs, ys, xs].copy()
    kk = np.arange(zs.start, zs.stop)
    neighbours = np.stack((
        original[np.maximum(kk - 1, 0), ys, xs],
        original[np.minimum(kk + 1, original.shape[0] - 1), ys, xs],
    ), axis=0)
    good = (neighbours >= low) & (neighbours <= high)
    count = good.sum(axis=0)
    total = np.where(good, neighbours, 0.0).sum(axis=0, dtype="f4")
    fallback = np.where(result < low, low, high).astype("f4")
    replacement = np.divide(total, count, out=fallback, where=count > 0)
    return np.where((result < low) | (result > high), replacement, result).astype("f4")


def _times(ds: nc.Dataset) -> list[str]:
    values = nc.chartostring(ds.variables["Times"][:])
    return [str(x) for x in np.asarray(values).tolist()]


def _wrf_timestamp(value: str) -> datetime:
    try: return parse_timestamp(value)
    except ValueError as error: raise ValueError(f"invalid CALWRF timestamp: {value!r}") from error


def _write_time_text(variable,index,text):
    """Write one WRF-style timestamp without netCDF4 scalar-string coercion."""
    encoded=str(text).encode('ascii')
    if len(encoded)!=19:
        raise ValueError('CALWRF time_text values must contain 19 ASCII bytes')
    variable[index]=np.frombuffer(encoded,dtype='S1')

def _selection(value,size,label):
    """Convert CALWRF's inclusive one-based subset to a Python slice."""
    return fortran_inclusive_slice(value,size,label=f'CALWRF {label}')


def destagger_x(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[..., :-1] + values[..., 1:])).astype(
        np.float32, copy=False,
    )


def destagger_y(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[..., :-1, :] + values[..., 1:, :])).astype(
        np.float32, copy=False,
    )


def destagger_z(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a, dtype=np.float32)
    return (np.float32(0.5) * (values[:-1, ...] + values[1:, ...])).astype(
        np.float32, copy=False,
    )


def _geopotential_height(ph: np.ndarray, phb: np.ndarray) -> np.ndarray:
    """Convert staggered geopotential to mass-level height."""
    geopotential = np.asarray(ph, dtype=np.float32) + np.asarray(
        phb, dtype=np.float32,
    )
    return destagger_z(geopotential / np.float32(9.81))


def _cone(truelat1: float, truelat2: float) -> float:
    """Return the Lambert conformal cone parameter."""
    first=float(truelat1); second=float(truelat2)
    if abs(first-second)<=0.1:
        return float(np.sin(np.deg2rad(abs(first))))
    numerator=np.log(np.cos(np.deg2rad(first))/np.cos(np.deg2rad(second)))
    denominator=np.log(
        np.tan(np.deg2rad(45.-abs(first)/2.)) /
        np.tan(np.deg2rad(45.-abs(second)/2.)))
    return float(numerator/denominator)


def rotation_angle(ds: nc.Dataset, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """CALWRF map-to-earth rotation angle in radians."""
    map_projection=int(getattr(ds,"MAP_PROJ",0))
    # CALWRF sets OP_ROT=.FALSE. for every WRF projection code >= 3, not
    # only Mercator (3).  This also covers global/rotated lat-lon products
    # using later WRF projection identifiers.
    if map_projection>=3:
        return np.zeros_like(lon, dtype="f4")
    if map_projection not in {1,2}:
        raise ValueError(f"CALWRF does not support WRF MAP_PROJ={map_projection}")
    # Keep coordinate arithmetic in double precision, then cross into the
    # model's FP32 boundary once the angle has been calculated.
    latitude=np.asarray(lat,dtype=np.float64)
    longitude=np.asarray(lon,dtype=np.float64)
    stand_lon=float(getattr(ds,"STAND_LON"))
    cone=1.0 if map_projection==2 else _cone(
        getattr(ds,"TRUELAT1"),getattr(ds,"TRUELAT2"),
    )
    delta=longitude-stand_lon
    delta=np.where(delta>180.0,delta-360.0,delta)
    delta=np.where(delta<-180.0,delta+360.0,delta)
    alpha=np.deg2rad(delta*cone)
    return np.asarray(np.where(latitude<0.0,-alpha,alpha),dtype=np.float32)


def earth_wind(u: np.ndarray, v: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate WRF grid-relative U/V to eastward/northward components."""
    u=np.asarray(u,dtype=np.float32)
    v=np.asarray(v,dtype=np.float32)
    angle=np.asarray(alpha,dtype=np.float32)
    cosine=np.cos(angle)
    sine=np.sin(angle)
    east=v*sine+u*cosine
    north=v*cosine-u*sine
    return (np.asarray(east,dtype=np.float32),
            np.asarray(north,dtype=np.float32))


def speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u=np.asarray(u,dtype=np.float32)
    v=np.asarray(v,dtype=np.float32)
    speed=np.hypot(u,v).astype(np.float32,copy=False)
    calculated=np.float32(270.0)-np.rad2deg(np.arctan2(v,u))
    direction=np.where(speed>=np.float32(1.e-9),calculated,np.float32(0.0))
    direction=np.mod(direction,np.float32(360.0))
    direction=np.where(direction<=np.float32(0.0),
                       direction+np.float32(360.0),direction)
    return speed,np.asarray(direction,dtype=np.float32)


def _var(
    dst: nc.Dataset, name: str, dims: tuple[str, ...], units: str,
    compression=1, dtype="f4",
) -> nc.Variable:
    var=dst.createVariable(name,dtype,dims,
                           zlib=compression>0,complevel=compression,
                           shuffle=compression>0)
    var.units = units
    return var


def _mass_coordinate(source: nc.Dataset,name: str,time_index=0) -> np.ndarray:
    """Read WRF mass-point coordinates using CALWRF's alias fallback."""
    for candidate in (name,f'{name}_M'):
        if candidate in source.variables:
            return np.asarray(source[candidate][time_index],dtype='f8')
    raise ValueError(f'WRF file has neither {name} nor {name}_M')


def _rain_increment(current,previous):
    """Return an accumulation increment, detecting a counter reset."""
    values=np.asarray(current,dtype='f4')
    if previous is None:
        return values.copy()
    delta=values-np.asarray(previous,dtype='f4')
    return np.where(delta>=0.,delta,values).astype('f4',copy=False)


def _rain_increments(rainnc,rainc,previous):
    """Return reset-aware total, grid, and convective rain increments."""
    nonconvective=np.asarray(rainnc,dtype='f4')
    convective=np.asarray(rainc,dtype='f4')
    previous_nonconvective=None if previous is None else previous[0]
    previous_convective=None if previous is None else previous[1]
    grid=_rain_increment(nonconvective,previous_nonconvective)
    convective_increment=_rain_increment(convective,previous_convective)
    return np.asarray(grid+convective_increment,dtype='f4'),grid,convective_increment


def _surface_variable(source,index,name,shape,*,required=False):
    """Read one CALWRF 2-D field, retaining its missing-field policy."""
    if name in source.variables:
        return np.asarray(source[name][index],dtype='f4')
    if required:
        raise ValueError(f'WRF file is missing required 2-D variable {name}')
    # GET2D clears X2D before each time record and skips absent optional
    # variables, so their stored field remains exactly zero before QA.
    return np.zeros(shape,dtype='f4')


def _volume_variable(source,index,name,shape,*,required=False):
    """Read one CALWRF 3-D field with its required/optional policy."""
    if name in source.variables:
        return np.asarray(source[name][index],dtype='f4')
    if required:
        raise ValueError(f'WRF file is missing required 3-D variable {name}')
    return np.zeros(shape,dtype='f4')


def _surface_wind(source,index,shape,alpha):
    """Return CALWRF WD10/WS10, including its missing-U10 shortcut."""
    if 'U10' not in source.variables:
        zeros=np.zeros(shape,dtype='f4')
        return zeros.copy(),zeros
    u10=_surface_variable(source,index,'U10',shape)
    v10=_surface_variable(source,index,'V10',shape)
    east,north=earth_wind(u10,v10,alpha)
    speed,direction=speed_direction(east,north)
    return speed,direction


def _pressure_temperature(perturbation_pressure,base_pressure,perturbation_theta):
    """CALWRF pressure (hPa) and potential-temperature conversion."""
    pressure=(np.asarray(perturbation_pressure,dtype=np.float32)+
              np.asarray(base_pressure,dtype=np.float32))*np.float32(0.01)
    theta=np.asarray(perturbation_theta,dtype=np.float32)+np.float32(300.0)
    exponent=np.float32(287.04/1004.0)
    temperature=theta*np.power(pressure/np.float32(1000.0),exponent)
    return (np.asarray(pressure,dtype=np.float32),
            np.asarray(temperature,dtype=np.float32))


def _select_wrf_times(paths,begin,end):
    """Select output records and CALWRF's optional preceding rain record."""
    selected_by_time={}
    baseline=None
    baseline_time=begin-timedelta(hours=1)
    for path in paths:
        with nc.Dataset(path) as dataset:
            for index,text in enumerate(_times(dataset)):
                when=_wrf_timestamp(text)
                canonical=when.strftime('%Y-%m-%d_%H:%M:%S')
                if when==baseline_time and baseline is None:
                    baseline=(path,index,canonical)
                if begin<=when<=end:
                    selected_by_time.setdefault(when,(path,index,canonical))
    selected=[selected_by_time[when] for when in sorted(selected_by_time)]
    return selected,baseline


def _copy_static(src: nc.Dataset,dst: nc.Dataset,ys,xs,zs,compression=1) -> None:
    for name, units in (("terrain", "m"),):
        _var(dst,name,("y","x"),units,compression)
    for name,units in (("latitude","degrees_north"),("longitude","degrees_east")):
        _var(dst,name,("y","x"),units,compression,dtype="f8")
    _var(dst,"landuse",("y","x"),"category",compression,dtype="i2")
    dst["latitude"][:] = _mass_coordinate(src,"XLAT")[ys,xs]
    dst["longitude"][:] = _mass_coordinate(src,"XLONG")[ys,xs]
    dst["terrain"][:] = np.asarray(src["HGT"][0,ys,xs], dtype="f4")
    dst["landuse"][:] = np.asarray(src["LU_INDEX"][0,ys,xs], dtype="i2")
    _var(dst,"sigma",("level",),"1",compression)[:]=np.asarray(src["ZNU"][0,zs],dtype="f4")


def _create_output(path: Path,src: nc.Dataset,nt: int,ny: int,nx: int,nz: int,ys,xs,zs,*,compression=1) -> nc.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    dst = nc.Dataset(path, "w", format="NETCDF4")
    dst.createDimension("time", nt)
    dst.createDimension("level", nz)
    dst.createDimension("y", ny)
    dst.createDimension("x", nx)
    dst.createDimension("date_strlen", 19)
    dst.createVariable("time_text", "S1", ("time", "date_strlen"))
    _copy_static(src,dst,ys,xs,zs,compression)
    fields3 = {
        "pressure": "hPa", "height": "m", "temperature": "K",
        "wind_speed": "m s-1", "wind_direction": "degree",
        "vertical_velocity": "m s-1", "qvapor": "kg kg-1",
        "qcloud": "kg kg-1", "qrain": "kg kg-1", "qice": "kg kg-1",
        "qsnow": "kg kg-1", "qgraupel": "kg kg-1",
    }
    fields2 = {
        "surface_pressure": "Pa", "precipitation": "mm", "snow_cover": "1",
        "shortwave_down": "W m-2", "longwave_down": "W m-2",
        "temperature_2m": "K", "qvapor_2m": "kg kg-1",
        "wind_speed_10m": "m s-1", "wind_direction_10m": "degree",
        "sst": "K", "skin_temperature": "K", "pbl_height": "m",
        "sensible_heat_flux": "W m-2", "latent_heat_flux": "W m-2",
        "friction_velocity": "m s-1", "convective_precipitation": "mm",
        "grid_precipitation": "mm",
    }
    for name, units in fields3.items():
        _var(dst,name,("time","level","y","x"),units,compression)
    # IOSRF only controls the additional 2D.DAT file.  GET2D still runs and
    # WRTOUT embeds the surface record in 3D.DAT for every grid point.
    for name,units in fields2.items():
        _var(dst,name,("time","y","x"),units,compression)
    dst.source_format = "WRF NetCDF"
    dst.pycalmet_stage = "calwrf"
    dst.wrf_dx_m = np.float64(getattr(src, "DX"))
    dst.wrf_dy_m = np.float64(getattr(src, "DY"))
    dst.source_i_start=xs.start+1; dst.source_i_end=xs.stop
    dst.source_j_start=ys.start+1; dst.source_j_end=ys.stop
    dst.source_k_start=zs.start+1; dst.source_k_end=zs.stop
    dst.source_grid_nx=len(src.dimensions["west_east"])
    dst.source_grid_ny=len(src.dimensions["south_north"])
    dst.source_grid_nz=len(src.dimensions["bottom_top"])
    source_latitude=_mass_coordinate(src,"XLAT")
    source_longitude=_mass_coordinate(src,"XLONG")
    dst.source_latitude_1_1=np.float64(source_latitude[0,0])
    dst.source_longitude_1_1=np.float64(source_longitude[0,0])
    for attr in ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
                 "MOAD_CEN_LAT"):
        if hasattr(src, attr):
            dst.setncattr(attr.lower(), getattr(src, attr))
    for attr in ('MP_PHYSICS','CU_PHYSICS','BL_PBL_PHYSICS','RA_SW_PHYSICS',
                 'SF_SURFACE_PHYSICS','GRID_FDDA','OBS_NUDGE_OPT'):
        if hasattr(src,attr): dst.setncattr(attr.lower(),getattr(src,attr))
    present=[name in src.variables for name in ('QCLOUD','QRAIN','QICE','QSNOW','QGRAUP')]
    dst.legacy_cloud_component_count=(5 if present[4] else 4 if present[2] and present[3]
                                      else 2 if present[1] else 1 if present[0] else 1)
    return dst


def run(config: CalwrfConfig) -> Path:
    active_surface_output=(config.surface_output
                           if config.create_surface and config.legacy_output is not None
                           else None)
    validate_calwrf_paths(
        config.wrf_files, config.output, config.legacy_output,
        active_surface_output, config.listing_output,
    )
    paths = [Path(p) for p in config.wrf_files]
    if not paths:
        raise ValueError("wrf_files is empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    begin,end=config.start,config.end
    if end < begin:
        raise ValueError("end precedes start")
    if not 0<=config.compression<=9:
        raise ValueError('compression must be between zero and nine')

    selected,rain_baseline=_select_wrf_times(paths,begin,end)
    if not selected:
        raise ValueError("no WRF times fall inside requested interval")
    selected_times=[_wrf_timestamp(item[2]) for item in selected]
    if selected_times[0]!=begin or selected_times[-1]!=end or any(
            right-left!=timedelta(hours=1) for left,right in zip(selected_times[:-1],selected_times[1:])):
        raise ValueError('selected CALWRF output times must cover the requested interval hourly')

    with nc.Dataset(selected[0][0]) as first:
        source_ny=len(first.dimensions["south_north"]); source_nx=len(first.dimensions["west_east"])
        source_nz=len(first.dimensions["bottom_top"])
        xs=_selection(config.i_range,source_nx,'I'); ys=_selection(config.j_range,source_ny,'J')
        zs=_selection(config.k_range,source_nz,'K')
        nx=xs.stop-xs.start; ny=ys.stop-ys.start; nz=zs.stop-zs.start
        dst=_create_output(Path(config.output),first,len(selected),ny,nx,nz,ys,xs,zs,
                           compression=config.compression)
        dst.title=config.title
    previous_rain=None
    if rain_baseline is not None:
        baseline_path,baseline_index,_=rain_baseline
        with nc.Dataset(baseline_path) as baseline_source:
            baseline_shape=(len(baseline_source.dimensions['south_north']),
                            len(baseline_source.dimensions['west_east']))
            rainnc=_surface_variable(baseline_source,baseline_index,'RAINNC',baseline_shape,
                                     required=True)
            rainc=_surface_variable(baseline_source,baseline_index,'RAINC',baseline_shape)
            previous_rain=(rainnc,rainc)
    try:
        for out_index, (path, time_index, text) in enumerate(selected):
            with nc.Dataset(path) as src:
                take = lambda name: np.asarray(src[name][time_index], dtype="f4")
                crop2=lambda values: np.asarray(values)[ys,xs]
                crop3=lambda values: np.asarray(values)[zs,ys,xs]
                lat,lon=_mass_coordinate(src,"XLAT",time_index),_mass_coordinate(
                    src,"XLONG",time_index)
                alpha = rotation_angle(src, lat, lon)
                pressure_hpa,temperature=_pressure_temperature(take("P"),take("PB"),take("T"))
                height = _geopotential_height(take("PH"), take("PHB"))
                vertical = destagger_z(take("W"))
                u = destagger_x(take("U")); v = destagger_y(take("V"))
                ue, vn = earth_wind(u, v, alpha[None, ...])
                speed, direction = speed_direction(ue, vn)
                surface_shape=alpha.shape
                speed10,direction10=_surface_wind(
                    src,time_index,surface_shape,alpha)

                _write_time_text(dst["time_text"],out_index,text)
                fields3={"pressure":pressure_hpa, "temperature":temperature,
                         "wind_speed":speed, "wind_direction":direction,
                         "vertical_velocity":vertical}
                fields3["qvapor"]=_volume_variable(
                    src,time_index,"QVAPOR",pressure_hpa.shape,required=True)
                for source, target in (("QCLOUD", "qcloud"), ("QRAIN", "qrain"),
                                       ("QICE", "qice"),
                                       ("QSNOW", "qsnow"), ("QGRAUP", "qgraupel")):
                    fields3[target]=_volume_variable(
                        src,time_index,source,pressure_hpa.shape)
                dst["pressure"][out_index] = _horizontal_range_check(fields3["pressure"],*_LIMITS_3D["pressure"],ys,xs)[zs]
                dst["height"][out_index] = crop3(height)
                for target in ("temperature","wind_speed","wind_direction","vertical_velocity","qvapor"):
                    dst[target][out_index]=_horizontal_range_check(fields3[target],*_LIMITS_3D[target],ys,xs)[zs]
                for target in ("qcloud","qrain","qice","qsnow","qgraupel"):
                    dst[target][out_index]=_vertical_range_check(fields3[target],*_LIMITS_3D[target],zs,ys,xs)

                rainnc=_surface_variable(src,time_index,'RAINNC',surface_shape,required=True)
                rainc=_surface_variable(src,time_index,'RAINC',surface_shape)
                increment,grid_increment,convective_increment=_rain_increments(
                    rainnc,rainc,previous_rain)
                previous_rain=(rainnc.copy(),rainc.copy())
                surface={
                    "surface_pressure":_surface_variable(
                        src,time_index,'PSFC',surface_shape,required=True),
                    "precipitation":increment,
                    "grid_precipitation":grid_increment,
                    "convective_precipitation":convective_increment,
                    "snow_cover":_surface_variable(
                        src,time_index,'SNOWC',surface_shape,required=True),
                    "shortwave_down":_surface_variable(src,time_index,'SWDOWN',surface_shape),
                    "longwave_down":_surface_variable(src,time_index,'GLW',surface_shape),
                    "temperature_2m":_surface_variable(src,time_index,'T2',surface_shape),
                    "qvapor_2m":_surface_variable(src,time_index,'Q2',surface_shape),
                    "wind_speed_10m":speed10,"wind_direction_10m":direction10,
                    "sst":_surface_variable(src,time_index,'SST',surface_shape),
                    "skin_temperature":_surface_variable(src,time_index,'TSK',surface_shape),
                    "pbl_height":_surface_variable(src,time_index,'PBLH',surface_shape),
                    "sensible_heat_flux":_surface_variable(src,time_index,'HFX',surface_shape),
                    "latent_heat_flux":_surface_variable(src,time_index,'LH',surface_shape),
                    "friction_velocity":_surface_variable(src,time_index,'UST',surface_shape),
                }
                for target, values in surface.items():
                    if target in _LIMITS_2D:
                        values=_horizontal_range_check(values,*_LIMITS_2D[target],ys,xs)
                    else:
                        values=crop2(values)
                    dst[target][out_index] = values
    finally:
        dst.close()
    if config.legacy_output is not None:
        write_m3d(config.output,config.legacy_output,
                  config.surface_output if config.create_surface else None)
    if config.listing_output is not None:
        write_calwrf_listing(
            config.listing_output,title=config.title,wrf_files=paths,
            requested_start=begin,requested_end=end,output=config.output,
            legacy_output=config.legacy_output,
            surface_output=active_surface_output,written_times=selected_times,
            nx=nx,ny=ny,nz=nz,
        )
    return Path(config.output)
