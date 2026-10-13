"""Streaming output for the NumPy CALMET solver.

The canonical NetCDF file preserves floating-point solver fields without the
rounding in CALMET's legacy sequential-unformatted interchange file.  A
separate legacy writer can therefore be regression-tested against this source.
"""
from __future__ import annotations
from pathlib import Path
from datetime import timedelta
import struct
import numpy as np
from netCDF4 import Dataset,date2num
from .datetime_utils import format_ordinal_code
from .calmet_numerics import convective_velocity_scale,mesopac_layer_average,pgt_stability
from .calmet_solver import HourFields,target_latitude_longitude
from .config import GridConfig
from .legacy_io import FortranSequentialWriter,fixed_string
from ._makegeo import _legacy_round

def _convective_velocity(hour: HourFields,landuse,water_range):
    """Reproduce WATER-before-AVEMIX and land-only WSTARR sequencing."""
    surface=hour.surface
    value=convective_velocity_scale(surface.convective_mixing_height,
        surface.sensible_heat_flux,hour.temperature[0],surface.density,landuse,
        water_range=water_range)
    if surface.water_convective_velocity_scale is not None:
        categories=np.asarray(landuse)
        water=(categories>=water_range[0])&(categories<=water_range[1])
        value=np.where(water,surface.water_convective_velocity_scale,value)
    return np.asarray(value,dtype=np.float32)

class CalmetNetCDFWriter:
    """Append :class:`HourFields` to a CF-style NetCDF file one hour at a time."""
    def __init__(self,path,grid: GridConfig,z_faces_m,geophysical,*,compression=1,water_range=(50,55),run_type=1):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._index=0
        self.dataset=Dataset(self.path,'w',format='NETCDF4'); self.water_range=tuple(water_range); self.run_type=int(run_type)
        ds=self.dataset; faces=np.asarray(z_faces_m,dtype=float); centers=.5*(faces[:-1]+faces[1:])
        ds.createDimension('time',None); ds.createDimension('level',centers.size); ds.createDimension('face',faces.size); ds.createDimension('y',grid.ny); ds.createDimension('x',grid.nx)
        ds.Conventions='CF-1.8'; ds.source='PyCALMET NumPy solver'; ds.pycalmet_stage='calmet'; ds.projection=grid.projection
        ds.water_category_min=int(self.water_range[0]); ds.water_category_max=int(self.water_range[1])
        time=ds.createVariable('time','f8',('time',)); time.units='seconds since 1970-01-01 00:00:00'; time.calendar='standard'
        for name,data,units in (('level_height',centers,'m'),('face_height',faces,'m')):
            variable=ds.createVariable(name,'f4',('level' if name=='level_height' else 'face',)); variable.units=units; variable[:]=data
        latitude,longitude=target_latitude_longitude(grid)
        static=(('latitude',latitude,'degrees_north','f4'),('longitude',longitude,'degrees_east','f4'),
                ('terrain_height',np.asarray(geophysical['terrain_height']),'m','f4'),('landuse',np.asarray(geophysical['dominant_landuse']),'1','i2'))
        for name,data,units,dtype in static:
            variable=ds.createVariable(name,dtype,('y','x'),zlib=True,complevel=compression); variable.units=units; variable[:]=data
        variables=(
            ('wind_u','f4',('time','level','y','x'),'m s-1'),('wind_v','f4',('time','level','y','x'),'m s-1'),
            *(() if self.run_type==0 else (
            ('vertical_velocity','f4',('time','face','y','x'),'m s-1'),('temperature','f4',('time','level','y','x'),'K'),
            ('relative_humidity','f4',('time','level','y','x'),'%'),('cloud_fraction','f4',('time','y','x'),'1'),
            ('surface_relative_humidity','i1',('time','y','x'),'%'),
            ('precipitation_rate','f4',('time','y','x'),'mm h-1'),('precipitation_code','i1',('time','y','x'),'1'),
            ('ceiling_height','f4',('time','y','x'),'m'),('surface_temperature','f4',('time','y','x'),'K'),
            ('surface_density','f4',('time','y','x'),'kg m-3'),('shortwave_flux','f4',('time','y','x'),'W m-2'),
            ('sensible_heat_flux','f4',('time','y','x'),'W m-2'),('friction_velocity','f4',('time','y','x'),'m s-1'),
            ('monin_obukhov_length','f4',('time','y','x'),'m'),('mixing_height','f4',('time','y','x'),'m'),
            ('convective_mixing_height','f4',('time','y','x'),'m'),('convective_velocity_scale','f4',('time','y','x'),'m s-1'),
            ('pgt_stability_class','i1',('time','y','x'),'1'))))
        for name,dtype,dims,units in variables:
            variable=ds.createVariable(name,dtype,dims,zlib=True,complevel=compression); variable.units=units

    def append(self,hour: HourFields,solar_elevation_sine,landuse):
        ds=self.dataset; i=self._index; step=hour.step1; surface=hour.surface
        ds['time'][i]=date2num(hour.timestamp,ds['time'].units,ds['time'].calendar)
        ds['wind_u'][i]=hour.wind.u; ds['wind_v'][i]=hour.wind.v
        if self.run_type==0: ds.sync(); self._index+=1; return
        values={'wind_u':hour.wind.u,'wind_v':hour.wind.v,'vertical_velocity':hour.wind.w,
            'temperature':hour.temperature,'relative_humidity':step.relative_humidity,
            'cloud_fraction':step.cloud_fraction,'ceiling_height':step.ceiling_height,
            'surface_relative_humidity':step.surface_relative_humidity,
            'precipitation_rate':step.precipitation_rate,'precipitation_code':step.precipitation_code,
            'surface_temperature':hour.temperature[0],'surface_density':surface.density,
            'shortwave_flux':surface.shortwave_flux,'sensible_heat_flux':surface.sensible_heat_flux,
            'friction_velocity':surface.friction_velocity,'monin_obukhov_length':surface.monin_obukhov_length,
            'mixing_height':surface.mixing_height,'convective_mixing_height':surface.convective_mixing_height}
        for name,value in values.items(): ds[name][i]=np.asarray(value)
        ds['convective_velocity_scale'][i]=_convective_velocity(hour,landuse,self.water_range)
        ds['pgt_stability_class'][i]=pgt_stability(hour.wind.u[0],hour.wind.v[0],solar_elevation_sine,step.cloud_fraction,step.ceiling_height,landuse,surface.monin_obukhov_length,water_range=self.water_range)
        ds.sync(); self._index+=1

    def close(self):
        if self.dataset is not None: self.dataset.close(); self.dataset=None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


class MesopacWriter:
    """GNU sequential-unformatted writer for CALMET's revised MESOPAC-II layout."""
    def __init__(self,path,grid: GridConfig,z_faces_m,geophysical,start,end,*,
                 timezone_offset=0,surface_coordinates_m=(),upper_coordinates_m=(),
                 nearest_surface_station=None,water_range=(50,55)):
        self.writer=FortranSequentialWriter(path); self.grid=grid
        self.faces=np.asarray(z_faces_m,dtype=np.float32)
        self.landuse=self._array(geophysical['dominant_landuse'])
        self.surface_coordinates=np.asarray(surface_coordinates_m,dtype=np.float32).reshape(-1,2)
        self.upper_coordinates=np.asarray(upper_coordinates_m,dtype=np.float32).reshape(-1,2)
        self.water_range=tuple(water_range)
        shape=(grid.ny,grid.nx)
        if self.landuse.shape!=shape:
            raise ValueError('MESOPAC landuse does not match the CALMET grid')
        nearest=np.zeros(shape,dtype=np.int32) if nearest_surface_station is None else np.asarray(nearest_surface_station)
        if nearest.shape!=shape:
            raise ValueError('MESOPAC nearest-station grid does not match the CALMET grid')
        hours=int((end-start).total_seconds()//3600)
        if end<start or end-start!=timedelta(hours=hours):
            raise ValueError('MESOPAC output requires a whole-hour run period')
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
    def _array(value): return np.asarray(value[:] if hasattr(value,'dimensions') else value)
    @staticmethod
    def _i(*values): return struct.pack('<'+'i'*len(values),*values)
    @staticmethod
    def _f(*values): return struct.pack('<'+'f'*len(values),*values)

    def _station_grid_coordinates(self,coordinates):
        shift=np.float32(self.grid.dx_km*500.)
        return np.asarray((coordinates-shift)/np.float32(self.grid.dx_km*1000.)+np.float32(1.),dtype='<f4')

    def _grid_record(self,value,integer=False):
        array=np.asarray(value)
        if array.shape!=(self.grid.ny,self.grid.nx):
            raise ValueError('MESOPAC hourly field does not match the CALMET grid')
        self.writer.write_record(array.astype('<i4' if integer else '<f4',copy=False).ravel(order='C'))

    def append(self,hour: HourFields,solar_elevation_sine,*,station_temperature_k,
               station_shortwave_flux,station_relative_humidity,station_precipitation_code):
        ns=len(self.surface_coordinates)
        station=(np.asarray(station_temperature_k),np.asarray(station_shortwave_flux),
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
        density=np.asarray(surface.density,dtype=np.float32).ravel(order='C')
        average_density=np.cumsum(density,dtype=np.float32)[-1]/np.float32(density.size)
        self.writer.write_record(self._f(average_density),
            station[0].astype('<f4',copy=False),station[1].astype('<f4',copy=False),
            station[2].astype('<i4',copy=False),station[3].astype('<i4',copy=False))

    def close(self): self.writer.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


class AuxiliaryCalmetWriter:
    """CALMET.AUX 1.1 cloud-liquid-water sequential-unformatted writer."""
    _fields_2d=(('CLDMRUP','G/KG','RC4'),('ZCLDBOT','MAGL','RC4'),
                ('ZCLDTOP','MAGL','RC4'))
    _fields_3d=(('CLDMR3D','G/KG','RC4'),)

    def __init__(self,path,grid: GridConfig,z_faces_m,start,end,*,
                 control_lines=(),base_timezone='UTC-0000',available=True,
                 timestep_seconds=3600):
        self.writer=FortranSequentialWriter(path)
        self.grid=grid
        self.faces=np.asarray(z_faces_m,dtype=np.float32)
        self.available=bool(available)
        self.timestep_seconds=int(timestep_seconds)
        if self.available:
            self._write_header(start,end,list(control_lines),base_timezone)

    @staticmethod
    def _i(*values): return struct.pack('<'+'i'*len(values),*values)

    @staticmethod
    def _f(*values): return struct.pack('<'+'f'*len(values),*values)

    @staticmethod
    def _stamp(value):
        return format_ordinal_code(value)

    @staticmethod
    def _date_parts(value):
        return value.year,value.month,value.day,value.hour,value.minute*60+value.second

    def _write_header(self,start,end,control_lines,base_timezone):
        writer=self.writer; grid=self.grid
        writer.write_record(fixed_string('CALMET.AUX',16),fixed_string('1.1',16),
            fixed_string('Generic file structure with compression',64))
        comments=['Produced by CALMET Version: 6.5.0  Level: 150223',*control_lines]
        writer.write_record(self._i(len(comments)))
        for line in comments:
            writer.write_record(fixed_string(line.rstrip('\n'),132))
        hours=int((end-start).total_seconds()//3600)
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

    def _compressed(self,label,data):
        values=np.asarray(data,dtype=np.float32)
        if values.shape!=(self.grid.ny,self.grid.nx):
            raise ValueError('CALMET.AUX field does not match the CALMET grid')
        flat=values.ravel(order='C')
        if np.any(flat<0.):
            raise ValueError('CALMET.AUX compression cannot encode negative values')
        nonzero=np.flatnonzero(flat!=np.float32(0.))
        if nonzero.size:
            previous=np.r_[-1,nonzero[:-1]]
            gaps=nonzero-previous-1
            has_gap=gaps>0
            value_positions=np.arange(nonzero.size)+np.cumsum(has_gap)
            trailing=int(flat.size-nonzero[-1]-1)
            encoded=np.empty(nonzero.size+int(has_gap.sum())+int(trailing>0),dtype='<f4')
            encoded[value_positions]=flat[nonzero]
            encoded[value_positions[has_gap]-1]=-(
                gaps[has_gap].astype(np.float32)+np.float32(.0001))
            if trailing:
                encoded[-1]=-(np.float32(trailing)+np.float32(.0001))
        else:
            encoded=np.asarray([-(np.float32(flat.size)+np.float32(.0001))],dtype='<f4')
        self.writer.write_record(self._i(encoded.size))
        self.writer.write_record(fixed_string(label,12),encoded)

    def append(self,hour: HourFields):
        if not self.available: return
        step=hour.step1
        fields=(step.cloud_liquid_water_above,
                step.cloud_liquid_water_above_bottom,
                step.cloud_liquid_water_above_top)
        if any(field is None for field in fields) or step.cloud_liquid_water_3d is None:
            raise ValueError('CALMET.AUX output requires prognostic cloud-water fields')
        begin=hour.timestamp
        begin_seconds=begin.minute*60+begin.second
        timestamp=self._i(self._stamp(begin),begin_seconds,
                          self._stamp(begin),begin_seconds+self.timestep_seconds)
        for (name,_,_),field in zip(self._fields_2d,fields):
            self.writer.write_record(timestamp)
            self._compressed(name,field)
        cloud=np.asarray(step.cloud_liquid_water_3d)
        if cloud.shape!=(self.faces.size-1,self.grid.ny,self.grid.nx):
            raise ValueError('CALMET.AUX 3-D cloud water does not match the CALMET grid')
        self.writer.write_record(timestamp)
        for index,field in enumerate(cloud,1):
            self._compressed(f'CLDMR3D {index:04d}',field)

    def close(self): self.writer.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

class LegacyCalmetWriter:
    """CALMET.DAT v2.1 OUTHD/OUTHR writer for the current NOOBS layout."""
    def __init__(self,path,grid: GridConfig,z_faces_m,geophysical,start,end,*,
                 control_lines=(),base_timezone='UTC-0000',include_w_temperature=False,
                 water_range=(50,55),landuse_category_count=14,run_type=1,timestep_seconds=3600,
                 precipitation_station_count=-1,surface_coordinates_km=(),upper_coordinates_km=(),
                 precipitation_coordinates_km=(),nearest_surface_station=None,
                 iwfcod=1,sea_station_count=0):
        self.writer=FortranSequentialWriter(path); self.grid=grid; self.faces=np.asarray(z_faces_m,dtype='<f4'); self.include_w_temperature=include_w_temperature
        self.water_range=tuple(water_range); self.landuse_category_count=int(landuse_category_count)
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
            x=grid.x_origin_km+(np.arange(grid.nx)+.5)*grid.dx_km
            y=grid.y_origin_km+(np.arange(grid.ny)+.5)*grid.dy_km
            gx,gy=np.meshgrid(x,y)
            distance=((gx[...,None]-self.surface_coordinates[:,0])**2+
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
        self.landuse=self._array(geophysical['dominant_landuse']); self._write_header(geophysical,start,end,list(control_lines),base_timezone)

    @staticmethod
    def _array(value): return np.asarray(value[:] if hasattr(value,'dimensions') else value)

    @staticmethod
    def _date_parts(value): return (value.year,value.month,value.day,value.hour,value.minute*60+value.second)
    @staticmethod
    def _stamp(value): return format_ordinal_code(value)
    @staticmethod
    def _i(*values): return struct.pack('<'+'i'*len(values),*values)
    @staticmethod
    def _f(*values): return struct.pack('<'+'f'*len(values),*values)

    def _field_record(self,label,data,begin=None,end=None,integer=False):
        if begin is None:
            times=self._i(0,0,0,0)
        else:
            begin_seconds=begin.minute*60+begin.second
            times=self._i(self._stamp(begin),begin_seconds,
                          self._stamp(begin),begin_seconds+self.timestep_seconds)
        dtype='<i4' if integer else '<f4'; array=self._array(data).astype(dtype,copy=False)
        self.writer.write_record(fixed_string(label,8),times,array.ravel(order='C'))

    def _write_header(self,geo,start,end,control_lines,base_timezone):
        w=self.writer; g=self.grid
        w.write_record(fixed_string('CALMET.DAT',16),fixed_string('2.1',16),fixed_string('No-Obs file structure with embedded control file',64))
        comments=['Produced by PyCALMET NumPy implementation','Internal Coordinate Transformations: pyproj/PROJ']+control_lines
        w.write_record(self._i(len(comments)))
        for line in comments: w.write_record(fixed_string(line.rstrip('\n'),132))
        run=self._i(*self._date_parts(start),*self._date_parts(end))+fixed_string(base_timezone,8)
        run+=self._i(int((end-start).total_seconds()//3600),self.run_type,g.nx,g.ny,self.faces.size-1)
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
        self._field_record('Z0',_legacy_round(self._array(geo['roughness_length']),4))
        self._field_record('ILANDU',geo['dominant_landuse'],integer=True)
        self._field_record('ELEV',_legacy_round(self._array(geo['terrain_height']),2))
        self._field_record('XLAI',_legacy_round(self._array(geo['leaf_area_index']),2))
        if len(self.surface_coordinates):
            self._field_record('NEARS',self.nearest_surface_station,integer=True)

    def append(self,hour: HourFields,solar_elevation_sine,*,surface_relative_humidity=None,
               precipitation_rate=None,precipitation_code=None):
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
        fields=[('IPGT',pgt,True),('USTAR',surface.friction_velocity,False),('ZI',surface.mixing_height,False),
                ('EL',surface.monin_obukhov_length,False),('WSTAR',wstar,False)]
        if self.precipitation_station_count!=0: fields.append(('RMM',precipitation_rate,False))
        fields.extend((('TEMPK',hour.temperature[0],False),('RHO',surface.density,False),
                       ('QSW',surface.shortwave_flux,False),('IRH',surface_relative_humidity,True)))
        if self.precipitation_station_count!=0: fields.append(('IPCODE',precipitation_code,True))
        for label,data,integer in fields: self._field_record(label,data,begin,end,integer)

    def close(self): self.writer.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
