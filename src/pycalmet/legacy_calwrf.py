"""Writers for CALWRF 2.0.2 formatted M3D/M2D interchange files."""

from __future__ import annotations

from .datetime_utils import parse_timestamp
from pathlib import Path

import netCDF4 as nc
import numpy as np

from ._common import fortran_nint
from ._proj_data import configure_proj_data

configure_proj_data()

from pyproj import CRS,Transformer


def _nint(values):
    return fortran_nint(values)


def _r4_scale(value,scale):
    """Apply a CALWRF default-REAL unit conversion without scalar promotion."""
    return np.float32(np.float32(value)*np.float32(scale))


def _r4_divide(value,divisor):
    """Apply a CALWRF default-REAL division without scalar promotion."""
    return np.float32(np.float32(value)/np.float32(divisor))


def _legacy_vertical_velocity(value):
    value=np.float32(value)
    return np.float32(0.) if abs(value)<np.float32(.00499) else value


def _relative_humidity(qvapor,pressure_hpa,temperature_k):
    f32=np.float32
    temperature=np.asarray(temperature_k,dtype=f32)
    pressure=np.asarray(pressure_hpa,dtype=f32)
    vapor=np.asarray(qvapor,dtype=f32)
    # GETRH1 receives QQ after the caller has converted kg/kg to g/kg,
    # then converts it back.  Both assignments round to default REAL.
    qq=np.asarray(vapor*f32(1000.),dtype=f32)
    qq2=np.asarray(qq/f32(1000.),dtype=f32)
    numerator=np.asarray(f32(17.67)*np.asarray(
        temperature-f32(273.15),dtype=f32),dtype=f32)
    denominator=np.asarray(temperature-f32(29.65),dtype=f32)
    exponent=np.asarray(numerator/denominator,dtype=f32)
    exponential=np.asarray(np.exp(exponent.astype(np.float64)),dtype=f32)
    coefficient=np.asarray(f32(.622)*f32(6.112),dtype=f32)
    saturation=np.asarray(np.asarray(coefficient/pressure,dtype=f32)*exponential,dtype=f32)
    humidity=np.asarray(np.asarray(f32(100.)*qq2,dtype=f32)/saturation,dtype=f32)
    return np.minimum(np.maximum(humidity,f32(0.)),f32(100.)).astype(f32)


def _date_number(text):
    timestamp=parse_timestamp(text)
    if timestamp.minute or timestamp.second or timestamp.microsecond:
        raise ValueError(
            'legacy CALWRF M2D/M3D output requires whole-hour canonical WRF times'
        )
    return int(timestamp.strftime('%Y%m%d%H'))


def _compressed_water(values):
    values=np.maximum(np.asarray(values,dtype=np.float32),np.float32(0.))
    if values.size>1 and np.all(values<np.float32(.00049)):
        return f'{-float(values.size):6.3f}'
    return ''.join(f'{value:6.3f}' for value in values)


def _header(stream,dataset,name,variables_2d=()):
    version='2.1'; modification='WRF_ARW         Python          Created by PyCALMET'
    stream.write(f'{name:<16}{version:<16}{modification:<64}\n')
    # RDHD53 does not merely display this comment: when it sees ``CALWRF`` it
    # parses the six columns after ``Level: `` as an integer and applies a
    # historical precipitation QA guard.  Keep the original producer grammar
    # so a PyCALMET M3D remains readable by GNU CALMET as well as by Python.
    comments=[f'Produced by CALWRF {"v2.0.2":<12}  Level: {"131108":<8}']
    if variables_2d:
        comments.extend([f'{len(variables_2d):3d}  Number of 2D Variables',
                         *(f'{index:3d}  {name:<10} {units:<11} {description}'
                           for index,(name,units,description) in enumerate(variables_2d,1))])
    stream.write(f'{len(comments):4d}\n')
    for comment in comments:
        stream.write(f'{comment:<132}\n')


def _common_header(stream,dataset,times,iout):
    map_projection=int(getattr(dataset,'map_proj',1))
    try:
        projection={1:'LCC',2:'PS',3:'EM'}[map_projection]
    except KeyError as error:
        raise ValueError(
            f'legacy CALWRF output does not support MAP_PROJ={map_projection}'
        ) from error
    ny=len(dataset.dimensions['y']); nx=len(dataset.dimensions['x']); nz=len(dataset.dimensions['level'])
    source_nx=int(getattr(dataset,'source_grid_nx',nx))
    source_ny=int(getattr(dataset,'source_grid_ny',ny))
    source_nz=int(getattr(dataset,'source_grid_nz',nz))
    i_start=int(getattr(dataset,'source_i_start',1))
    j_start=int(getattr(dataset,'source_j_start',1))
    k_start=int(getattr(dataset,'source_k_start',1))
    i_end=int(getattr(dataset,'source_i_end',i_start+nx-1))
    j_end=int(getattr(dataset,'source_j_end',j_start+ny-1))
    k_end=int(getattr(dataset,'source_k_end',k_start+nz-1))
    if ((i_end-i_start+1,j_end-j_start+1,k_end-k_start+1)!=(nx,ny,nz) or
            not (1<=i_start<=i_end<=source_nx and
                 1<=j_start<=j_end<=source_ny and
                 1<=k_start<=k_end<=source_nz)):
        raise ValueError('canonical CALWRF source-grid bounds are inconsistent')
    stream.write(''.join(f'{value:3d}' for value in iout)+'\n')
    lat0=float(getattr(dataset,'moad_cen_lat',getattr(dataset,'cen_lat',
                                                     getattr(dataset,'latitude_origin',0.))))
    lon0=float(getattr(dataset,'stand_lon',getattr(dataset,'longitude_origin',0.)))
    lat1=float(getattr(dataset,'truelat1',0.)); lat2=float(getattr(dataset,'truelat2',0.))
    dx=float(getattr(dataset,'wrf_dx_m',1000.))/1000.
    lat=np.asarray(dataset['latitude'][:]); lon=np.asarray(dataset['longitude'][:])
    if projection=='LCC':
        projected=CRS.from_proj4(f'+proj=lcc +lat_0={lat0} +lon_0={lon0} '
            f'+lat_1={lat1} +lat_2={lat2} +R=6370000 +units=km +no_defs')
    elif projection=='PS':
        pole=90. if lat0>=0. else -90.
        projected=CRS.from_proj4(f'+proj=stere +lat_0={pole} +lon_0={lon0} '
            f'+lat_ts={lat1} +R=6370000 +units=km +no_defs')
    else:
        projected=CRS.from_proj4(f'+proj=merc +lon_0={lon0} +lat_ts={lat1} '
            f'+R=6370000 +units=km +no_defs')
    if hasattr(dataset,'source_x_origin_km') and hasattr(dataset,'source_y_origin_km'):
        x1=float(dataset.source_x_origin_km); y1=float(dataset.source_y_origin_km)
    else:
        source_latitude=float(getattr(dataset,'source_latitude_1_1',lat[0,0]))
        source_longitude=float(getattr(dataset,'source_longitude_1_1',lon[0,0]))
        x1,y1=Transformer.from_crs(CRS.from_proj4('+proj=longlat +R=6370000 +no_defs'),
                                    projected,always_xy=True).transform(
                                        source_longitude,source_latitude)
    stream.write(f'{projection:<4}{lat0:9.4f}{lon0:10.4f}{lat1:7.2f}{lat2:7.2f}'
                 f'{x1:10.3f}{y1:10.3f}{dx:8.3f}'
                 f'{source_nx:4d}{source_ny:4d}{source_nz:3d}\n')
    physics=[1,int(getattr(dataset,'mp_physics',0))+10,
             int(getattr(dataset,'cu_physics',0))+10,
             int(getattr(dataset,'bl_pbl_physics',0))+10,
             int(getattr(dataset,'ra_sw_physics',0))+10,
             int(getattr(dataset,'sf_surface_physics',0))+10,
             int(getattr(dataset,'grid_fdda',0)),int(getattr(dataset,'obs_nudge_opt',0)),
             *([0]*12),25]
    stream.write(''.join(f'{value:3d}' for value in physics)+'\n')
    start=_date_number(times[0])
    stream.write(f'{start:10d}{len(times):5d}{nx:4d}{ny:4d}{nz:4d}\n')
    stream.write(f'{i_start:4d}{j_start:4d}{i_end:4d}{j_end:4d}{k_start:4d}{k_end:4d}'
                 f'{lon.min():10.4f}{lon.max():10.4f}{lat.min():9.4f}{lat.max():9.4f}\n')
    for sigma in dataset['sigma'][:]:
        stream.write(f'{float(sigma):6.3f}\n')
    terrain=np.asarray(dataset['terrain'][:]); landuse=np.asarray(dataset['landuse'][:])
    for j in range(ny):
        lines=[]
        for i in range(nx):
            lines.append(f'{i+i_start:4d}{j+j_start:4d}{lat[j,i]:9.4f}{lon[j,i]:10.4f}'
                         f'{int(_nint(terrain[j,i])):5d}{int(_nint(landuse[j,i])):3d} '
                         f'{-999.:9.4f}{-999.:10.4f}{-999:5d}\n')
        stream.writelines(lines)


def write_m3d(canonical_path,m3d_path,m2d_path=None):
    """Stream canonical CALWRF NetCDF into legacy formatted files."""
    m3d_path=Path(m3d_path); m3d_path.parent.mkdir(parents=True,exist_ok=True)
    with nc.Dataset(canonical_path) as dataset:
        times=[str(value) for value in nc.chartostring(dataset['time_text'][:]).tolist()]
        # M2D/M3D timestamps have hour precision.  Validate every canonical
        # time before opening either output so minutes or seconds are never
        # silently discarded by the fixed-width legacy representation.
        for text in times: _date_number(text)
        cloud_count=int(getattr(dataset,'legacy_cloud_component_count',5))
        iout=(1,1,int(cloud_count>=1),int(cloud_count>=4),int(cloud_count>=5),int(m2d_path is not None))
        with m3d_path.open('w',encoding='ascii') as stream:
            _header(stream,dataset,'3D.DAT')
            _common_header(stream,dataset,times,iout)
            nt=len(times); ny=len(dataset.dimensions['y']); nx=len(dataset.dimensions['x'])
            for time_index in range(nt):
                date=_date_number(times[time_index])
                arrays={name:np.asarray(dataset[name][time_index]) for name in (
                    'surface_pressure','precipitation','snow_cover','shortwave_down',
                    'longwave_down','temperature_2m','qvapor_2m','wind_direction_10m',
                    'wind_speed_10m','sst','pressure','height','temperature','wind_direction',
                    'wind_speed','vertical_velocity','qvapor','qcloud','qrain','qice','qsnow','qgraupel')}
                rh=_relative_humidity(arrays['qvapor'],arrays['pressure'],arrays['temperature'])
                surface_pressure=_r4_divide(arrays['surface_pressure'],100.)
                surface_rain=_r4_divide(arrays['precipitation'],10.)
                surface_snow=_nint(arrays['snow_cover'])
                surface_q=np.maximum(_r4_scale(arrays['qvapor_2m'],1000.),np.float32(0.))
                surface_direction=np.where(
                    arrays['wind_direction_10m']<np.float32(.05),np.float32(360.),
                    arrays['wind_direction_10m'])
                level_pressure=_nint(arrays['pressure'])
                level_height=_nint(arrays['height'])
                level_direction=_nint(arrays['wind_direction'])
                level_direction=np.where(level_direction==0,360,level_direction)
                level_vertical=np.where(
                    np.abs(arrays['vertical_velocity'])<np.float32(.00499),np.float32(0.),
                    arrays['vertical_velocity']).astype(np.float32)
                level_rh=np.clip(_nint(rh),1,100)
                level_q=np.maximum(_r4_scale(arrays['qvapor'],1000.),np.float32(0.))
                level_water=np.stack(tuple(_r4_scale(arrays[name],1000.) for name in
                    ('qcloud','qrain','qice','qsnow','qgraupel')),axis=-1)[...,:cloud_count]
                for j in range(ny):
                    lines=[]
                    for i in range(nx):
                        i_source=i+int(getattr(dataset,'source_i_start',1))
                        j_source=j+int(getattr(dataset,'source_j_start',1))
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
                            basic=(f'{int(level_pressure[k,j,i]):4d}'
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


def _write_m2d(dataset,times,path,iout):
    variables=(('PSFC','hPa','SURFACE PRESSURE','surface_pressure',.01),
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
            date=_date_number(text)
            for label,_units,_description,name,scale in variables:
                stream.write(f'{date:010d}  {label:<8}\n')
                values=np.asarray(dataset[name][time_index])*scale
                for row in values[::-1]:
                    stream.writelines(
                        ''.join(f'{value:10.3f}' for value in row[start:start+8])+'\n'
                        for start in range(0,row.size,8))
