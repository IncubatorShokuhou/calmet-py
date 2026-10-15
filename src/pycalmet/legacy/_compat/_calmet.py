"""Public orchestration for the NumPy CALMET diagnostic solver."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timedelta
from contextlib import ExitStack,contextmanager
from pathlib import Path
import re
import tempfile
import numpy as np
from netCDF4 import Dataset
from .config import CalmetConfig,GridConfig
from .datums import normalize_datum
from .calmet_numerics import (coare_water_station_profiles,
                              delta_temperature_water_station_profiles,
                              density_from_pressure_temperature,heat_flux,hosker_roughness,
                              lake_breeze_adjust,land_station_similarity_theory,
                              neutral_water_station_profiles,objective_analysis_component,
                              profile_temperature_layer,terrain_extrema,wind1_initial_guess)
from .calmet_output import (AuxiliaryCalmetWriter,CalmetNetCDFWriter,
                            LegacyCalmetWriter,MesopacWriter)
from .calmet_listing import CalmetListingWriter
from .calmet_solver import (PrognosticMapper,_domain_mean_real4,build_mm4_step_fields,
                            _prognostic_local_timestamp,build_prog_step_fields,
                            dataset_timestamp,iter_hours,
                            solar_at_step_midpoint,target_latitude_longitude)
from .cloud_data import CloudData,CloudDataWriter
from .coare import (WarmLayerState,distance_to_coast,downward_longwave,
                    initial_warm_layer_state)
from .diagnostic_data import DiagnosticData
from .igf_data import InitialGuessCalmetSeries,interpolate_initial_guess
from .projections import grid_convergence,unproject_geographic
from .precip_data import grid_precipitation,load_precipitation
from .prog_data import ProgData
from .mm4_data import MM4Data
from .sea_data import (SeaData,grid_sea_mixing_height,grid_water_mixing_gradient,
                       nearest_sea_data,nearest_sea_waves)
from .surface_data import (SurfaceRecord,SurfaceStation,_near_upper_air,fill_missing_surface,load_surface,station_grid_mapping,
                           _wind_components,component_wind_profiles,interpolate_surface_scalar,surface_cloud_fields,
                           wind_profiles)
from .upper_air import UpperAirData,UpperAirStation,grid_temperature_faces
from .weight_data import read_weight_data
from .wind_debug import FinalWindDebugWriter,WindPrintWriter


@contextmanager
def _single_meteorology_dataset(path,temporary_directory):
    """Open one canonical NetCDF or transparently stage one formatted M3D."""
    with Path(path).open('rb') as stream: signature=stream.read(16)
    if signature.startswith(b'3D.DAT'):
        from .legacy_m3d import M3DReader
        with tempfile.NamedTemporaryFile(dir=temporary_directory,suffix='.nc',delete=False) as handle:
            staged=Path(handle.name)
        try:
            M3DReader(path).to_netcdf(staged)
            with Dataset(staged) as dataset: yield dataset
        finally:
            staged.unlink(missing_ok=True)
    else:
        with Dataset(path) as dataset: yield dataset


@contextmanager
def _meteorology_input(config,paths,temporary_directory,start,end):
    """Open canonical meteorology or adapt native MM4/PROG records."""
    if config.iprog in {3,4,5}:
        with Path(paths[0]).open('rb') as stream: sample=stream.read(256)
        native_mm4=(not sample.startswith((b'CDF',b'\x89HDF',b'3D.DAT')) and
                    len(sample.splitlines())>=2 and
                    re.fullmatch(rb'\s*\d{8}\s*\d+\s+\d+\s+\d+\s+\d+\s*[-+0-9.]*\s*',
                                 sample.splitlines()[1]) is not None)
        if native_mm4:
            sources=[MM4Data(path) for path in paths]
            reference=sources[0].header
            for source in sources[1:]:
                header=source.header
                if ((header.i_start,header.j_start)!=(reference.i_start,reference.j_start) or
                    header.latitude.shape!=reference.latitude.shape or
                    header.sigma_levels.shape!=reference.sigma_levels.shape):
                    raise ValueError('native MM4.DAT files do not share the first-file record layout')
            first_ending=start+timedelta(seconds=config.timestep_seconds)
            header_start=reference.start-timedelta(hours=config.base_timezone)
            if header_start>first_ending:
                raise ValueError(f'MM4.DAT header begins at {header_start}, after first CALMET ending time {first_ending}')
            native_mapper=PrognosticMapper.build(reference.latitude,reference.longitude,
                                                  config.grid,validate_domain=True)
            precipitation_radius=_prognostic_precipitation_radius(
                native_mapper,config.precipitation_radius_km,
                config.precipitation_station_count)
            boundaries=[]; current=start+timedelta(seconds=config.timestep_seconds)
            while current<=end:
                boundaries.append(current); current+=timedelta(seconds=config.timestep_seconds)
            targets=[]
            for boundary in boundaries:
                target=(boundary if boundary.minute==0 and boundary.second==0 else
                        boundary.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
                if not targets or target!=targets[-1]: targets.append(target)
            # RDMM4 consumes MM4.DAT records strictly forwards.  It skips
            # records older than the requested hour, fails as soon as the
            # next record is newer, and opens another MM4 file only at EOF.
            # A global timestamp index would incorrectly let a later file
            # repair a gap inside the current one.
            reference_source=sources[0]
            def records_in_file_order():
                for source in sources:
                    terrain_delta=np.asarray(source.header.terrain_height_m-
                                             reference.terrain_height_m,dtype=np.float32)
                    for record in source.records:
                        # RDMM4 skips later headers and retains all static
                        # fields from RDHD4's first file.  The reader formed
                        # AGL with the local header, so undo that subtraction.
                        if source is not reference_source:
                            record=replace(record,height_agl_m=np.asarray(
                                record.height_agl_m+terrain_delta[None],dtype=np.float32))
                        yield (reference_source,record,
                               record.timestamp-timedelta(hours=config.base_timezone))
            records=records_in_file_order()
            selected={}
            for target in targets:
                for source,record,local in records:
                    if local<target: continue
                    if local>target:
                        raise ValueError(f'MM4.DAT record {local} is beyond requested time {target}')
                    selected[target]=(source,record)
                    break
                else:
                    raise ValueError(f'MM4.DAT has no record for {target}')
            all_times=[start,*boundaries]
            text=np.array([list(value.strftime('%Y-%m-%d_%H:%M:%S'))
                           for value in all_times],dtype='S1')
            state=None; cache={}; built=-1
            def provider(ending):
                nonlocal state,built
                target=(ending if ending.minute==0 and ending.second==0 else
                        ending.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
                try: requested=targets.index(target)
                except ValueError as error:
                    raise ValueError(f'MM4.DAT has no record for {target}') from error
                # Thermal history is also sequential in RDMM4.  Materialize
                # intervening hours if a caller asks for a later step first.
                while built<requested:
                    built+=1; build_target=targets[built]
                    source,record=selected[build_target]
                    cache[build_target]=build_mm4_step_fields(source,record,config.grid,
                        config.z_faces_m,cloud_method=config.cloud_method,
                        precipitation_interpolation=config.precipitation_interpolation,
                        precipitation_radius_km=precipitation_radius,
                        precipitation_cutoff=config.precipitation_cutoff,
                        base_timezone=config.base_timezone,thermal_state=state,
                        prognostic_observation_rmin_km=(config.minimum_station_radius_km
                                                        if config.iprog==5 else None),
                        prognostic_observation_barriers=config.barriers,
                        prognostic_observation_barrier_top_level=config.barrier_top_level,
                        prognostic_ocean_category=config.prognostic_ocean_category)
                    state=cache[build_target].thermal_state
                return cache[target]
            yield {'time_text':text},provider
            return
    if config.iprog not in {1,2}:
        with _meteorology_dataset(paths,temporary_directory) as dataset:
            yield dataset,None
        return
    if len(paths)!=1:
        raise ValueError('IPROG=1/2 requires exactly one PRGDAT file')
    if config.prognostic_timestep_seconds!=3600:
        raise ValueError('legacy PROG.DAT records are hourly; ISTEPPGS must be 3600')
    prog=ProgData(paths[0])
    boundaries=[]
    current=start+timedelta(seconds=config.timestep_seconds)
    while current<=end:
        boundaries.append(current)
        current+=timedelta(seconds=config.timestep_seconds)
    # PROGRD reads on its first call and whenever NSECE is 0 or 3600.  For
    # sub-hourly runs this means another read at the ending-hour boundary,
    # using the same target hour as the first substep of that hour.  Preserve
    # that historical behavior, including its requirement for duplicate
    # labels in a sub-hourly PROG.DAT sequence.
    read_targets=[]
    for index,boundary in enumerate(boundaries):
        if index==0 or (boundary.minute==0 and boundary.second==0):
            target=(boundary if boundary.minute==0 and boundary.second==0 else
                    boundary.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
            read_targets.append(target)
    records=iter(prog.records_for_hours([target.hour for target in read_targets]))
    selected={}; current_record=None
    for index,boundary in enumerate(boundaries):
        if index==0 or (boundary.minute==0 and boundary.second==0):
            current_record=next(records)
        selected[boundary]=current_record
    all_times=[start,*boundaries]
    text=np.array([list(value.strftime('%Y-%m-%d_%H:%M:%S')) for value in all_times],dtype='S1')
    dataset={'time_text':text}
    def provider(ending):
        try: record=selected[ending]
        except KeyError as error: raise ValueError(f'PROG.DAT has no record for {ending}') from error
        return build_prog_step_fields(prog,record.hour,config.grid,config.z_faces_m,record=record)
    yield dataset,provider


class _ConcatenatedVariable:
    def __init__(self,variables,records):
        self._variables=tuple(variables)
        self._records=tuple(records)
        self.shape=(len(self._records),*variables[0].shape[1:])
        self.dimensions=variables[0].dimensions
        self.dtype=variables[0].dtype

    def __getitem__(self,key):
        first=key[0] if isinstance(key,tuple) else key
        rest=key[1:] if isinstance(key,tuple) else ()
        if isinstance(first,(int,np.integer)):
            index=int(first)
            if index<0: index+=self.shape[0]
            if not 0<=index<self.shape[0]: raise IndexError(index)
            source,local=self._records[index]
            return self._variables[source][(local,*rest)] if rest else self._variables[source][local]
        if isinstance(first,slice):
            indices=range(*first.indices(self.shape[0]))
            values=[self[(index,*rest)] if rest else self[index] for index in indices]
            if values: return np.stack(values)
            return np.asarray(self._variables[0][(slice(0,0),*rest)])
        raise TypeError('concatenated meteorology supports integer or slice time indices')


class _ConcatenatedMeteorology:
    """Minimal read-only netCDF-like view over sequential CALWRF datasets."""
    def __init__(self,datasets):
        self._datasets=tuple(datasets)
        if len(self._datasets)<2: raise ValueError('concatenated meteorology needs multiple files')
        required=('time_text','latitude','longitude','terrain','landuse')
        for name in required:
            if any(name not in dataset.variables for dataset in self._datasets):
                raise ValueError(f'meteorology file lacks required variable {name}')
        for name in required[1:]:
            reference=np.asarray(self._datasets[0][name][:])
            for dataset in self._datasets[1:]:
                candidate=np.asarray(dataset[name][:])
                if reference.shape!=candidate.shape or not np.allclose(reference,candidate,
                        rtol=0.,atol=5.e-5,equal_nan=True):
                    raise ValueError(f'meteorology files have inconsistent static field {name}')
        common=set(self._datasets[0].variables)
        for dataset in self._datasets[1:]: common.intersection_update(dataset.variables)
        records=[]; stamps=[]
        for source,dataset in enumerate(self._datasets):
            source_stamps=[dataset_timestamp(dataset,index)
                           for index in range(dataset['time_text'].shape[0])]
            if not source_stamps:
                raise ValueError('meteorology file contains no time records')
            if any(right<=left for left,right in zip(source_stamps[:-1],source_stamps[1:])):
                raise ValueError('meteorology records must be strictly increasing within each file')
            for local,stamp in enumerate(source_stamps):
                if stamps and stamp<stamps[-1]:
                    raise ValueError('meteorology files are not in chronological order')
                if stamps and stamp==stamps[-1]:
                    previous_source,previous_local=records[-1]
                    for name in common-{'time_text'}:
                        variables=[item[name] for item in self._datasets]
                        if not variables[0].dimensions or variables[0].dimensions[0]!='time':
                            continue
                        previous=np.asarray(variables[previous_source][previous_local])
                        candidate=np.asarray(variables[source][local])
                        if previous.shape!=candidate.shape or not np.allclose(
                                previous,candidate,rtol=1.e-6,atol=1.e-6,equal_nan=True):
                            raise ValueError(
                                f'duplicate meteorology time has inconsistent variable {name}')
                    continue
                records.append((source,local)); stamps.append(stamp)
        self.variables={}
        for name in common:
            variables=[dataset[name] for dataset in self._datasets]
            if variables[0].dimensions and variables[0].dimensions[0]=='time':
                if any(variable.dimensions!=variables[0].dimensions or
                       variable.shape[1:]!=variables[0].shape[1:] for variable in variables[1:]):
                    raise ValueError(f'meteorology files have inconsistent variable {name}')
                self.variables[name]=_ConcatenatedVariable(variables,records)
            else: self.variables[name]=variables[0]

    def __getitem__(self,name): return self.variables[name]


@contextmanager
def _meteorology_dataset(paths,temporary_directory):
    """Open and concatenate canonical NetCDF and/or formatted M3D files."""
    if isinstance(paths,(str,Path)): paths=(Path(paths),)
    else: paths=tuple(Path(path) for path in paths)
    if not paths: raise ValueError('meteorology file list is empty')
    with ExitStack() as stack:
        datasets=[stack.enter_context(_single_meteorology_dataset(path,temporary_directory))
                  for path in paths]
        yield datasets[0] if len(datasets)==1 else _ConcatenatedMeteorology(datasets)


@contextmanager
def _geophysical_dataset(path,temporary_directory):
    """Open canonical NetCDF or transparently stage a formatted GEO.DAT."""
    with Path(path).open('rb') as stream: signature=stream.read(16)
    if signature.strip().upper().startswith(b'GEO.DAT'):
        from .legacy_geo import GeoReader
        with tempfile.NamedTemporaryFile(dir=temporary_directory,suffix='.nc',delete=False) as handle:
            staged=Path(handle.name)
        try:
            GeoReader(path).to_netcdf(staged)
            with Dataset(staged) as dataset: yield dataset
        finally:
            staged.unlink(missing_ok=True)
    else:
        with Dataset(path) as dataset: yield dataset

def _stamp(value: str) -> datetime:
    for form in ('%Y-%m-%d_%H:%M:%S','%Y%m%d%H'):
        try: return datetime.strptime(value,form)
        except ValueError: pass
    raise ValueError(f'invalid CALMET timestamp: {value!r}')

def _timezone_label(offset: int) -> str:
    """Return CALMET's eight-character UTC offset label."""
    sign = '-' if offset >= 0 else '+'
    return f'UTC{sign}{abs(offset):02d}00'


def _timezone_offset(label: str) -> int:
    """Convert UTC label using CALMET's positive-west integer convention."""
    match=re.fullmatch(r'UTC([+-])(\d{2})(\d{2})',label.strip().upper())
    if match is None: raise ValueError(f'invalid CALMET timezone {label!r}')
    hours=int(match.group(2))+int(match.group(3))/60.
    return int(hours if match.group(1)=='-' else -hours)


def _validate_input_timezone(value,base_timezone: int,dataset: str):
    """Compare legacy integer or modern UTC data timezone with CALMET IBTZ."""
    try:
        offset=_timezone_offset(value) if isinstance(value,str) else int(value)
    except (TypeError,ValueError) as error:
        raise ValueError(f'invalid {dataset} timezone {value!r}') from error
    if offset!=base_timezone:
        raise ValueError(
            f'{dataset} timezone {value!r} does not match '
            f'CALMET base timezone {base_timezone}')


def _validate_input_span(data,start: datetime,end: datetime,dataset: str):
    """Reproduce READHD's declared begin/end coverage checks."""
    if data.start>start or data.end<end:
        raise ValueError(
            f'{dataset} declared period {data.start}..{data.end} does not cover '
            f'CALMET run {start}..{end}')


def _validate_upper_span(data,start: datetime,end: datetime,base_timezone: int):
    """Reproduce READHD's per-station UTC UP.DAT coverage QA."""
    run_start=start+timedelta(hours=base_timezone)
    run_end=end+timedelta(hours=base_timezone)
    start_hour=data.start.replace(minute=0,second=0,microsecond=0)
    run_start_hour=run_start.replace(minute=0,second=0,microsecond=0)
    start_seconds=data.start.minute*60+data.start.second
    run_start_seconds=run_start.minute*60+run_start.second
    if (start_hour>run_start_hour or
            (start_hour==run_start_hour and start_seconds>run_start_seconds)):
        raise ValueError('UP.DAT declared period does not cover CALMET run beginning')
    end_hour=data.end.replace(minute=0,second=0,microsecond=0)
    run_end_hour=run_end.replace(minute=0,second=0,microsecond=0)
    # READHD 6.5.0 compares IBSECU with IBSEC in its equal-ending-hour
    # condition (rather than IESECU with IESEC). Preserve that published
    # behavior for byte-for-byte control-flow equivalence.
    if (end_hour<run_end_hour or
            (end_hour==run_end_hour and start_seconds<run_start_seconds)):
        raise ValueError('UP.DAT declared period does not cover CALMET run ending')


def _validate_sea_metadata(config, stations):
    """Apply RDHDOW checks that depend on CALMET run controls."""
    reference=None
    for data in stations:
        if config.overwater_flux_method<0 and float(data.version[:4])<2.1:
            raise ValueError('negative ICOARE requires SEA.DAT version 2.1 or newer')
        if data.timezone is not None:
            actual=data.timezone.upper()
            try:
                sea_timezone=_timezone_offset(actual)
            except ValueError as error:
                raise ValueError(f'invalid SEA.DAT timezone {data.timezone!r}')
            # RDHDOW assigns the real offset to INTEGER, truncating fractional
            # zones toward zero before comparing it with IBTZ.
            if sea_timezone!=config.base_timezone:
                raise ValueError(
                    f'SEA.DAT timezone {data.timezone!r} does not match '
                    f'CALMET base timezone {config.base_timezone}')
        grid=data.source_grid
        metadata=(str(data.projection).strip().upper(),
                  str(data.datum).strip().upper(),str(data.datum_date).strip(),
                  int(grid.utm_zone),str(grid.hemisphere).strip().upper(),
                  float(grid.lat0),float(grid.lon0),float(grid.lat1),float(grid.lat2),
                  float(grid.false_easting_km),float(grid.false_northing_km))
        if reference is None:
            reference=metadata
            continue
        text_equal=metadata[:5]==reference[:5]
        angles_equal=all(_legacy_close(left,right)
                         for left,right in zip(metadata[5:9],reference[5:9]))
        offsets_equal=metadata[9:]==reference[9:]
        if not (text_equal and angles_equal and offsets_equal):
            raise ValueError('SEA.DAT files do not share projection and datum metadata')


def _surface_control_metadata(config):
    """Convert Group-7 projected station rows to the LL representation."""
    rows=tuple(config.surface_control_stations)
    if not rows: return None
    zones=np.asarray([item[4] for item in rows],dtype=float)
    if np.any(zones!=config.base_timezone):
        raise ValueError('surface control-station time zone does not match CALMET base time zone')
    longitude,latitude=unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km')
    return tuple(SurfaceStation(item[1],item[0],float(lat),float(lon),item[5])
                 for item,lat,lon in zip(rows,latitude,longitude))


def _precipitation_control_metadata(config):
    rows=tuple(config.precipitation_control_stations)
    if not rows: return None
    longitude,latitude=unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km')
    return tuple(SurfaceStation(item[1],item[0],float(lat),float(lon),0.)
                 for item,lat,lon in zip(rows,latitude,longitude))


def _upper_control_metadata(config):
    rows=tuple(config.upper_control_stations)
    if not rows: return None
    zones=np.asarray([item[4] for item in rows],dtype=float)
    if np.any(zones!=config.base_timezone):
        raise ValueError('upper-air control-station time zone does not match CALMET base time zone')
    longitude,latitude=unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km')
    return tuple(UpperAirStation(item[1],item[0],float(lat),float(lon),0.)
                 for item,lat,lon in zip(rows,latitude,longitude))

def _legacy_close(first,second,tolerance=1.e-4):
    if first==second: return True
    difference=abs(first-second); average=.5*abs(first+second)
    return difference<average and difference/average<=tolerance

def _validate_geophysical_grid(dataset,grid):
    """READGE grid/header compatibility checks for canonical NetCDF GEO."""
    expected={
        'grid_nx':grid.nx,'grid_ny':grid.ny,'grid_dx_km':grid.dx_km,'grid_dy_km':grid.dy_km,
        'x_origin_km':grid.x_origin_km,'y_origin_km':grid.y_origin_km,
    }
    projection=grid.projection.upper()
    if projection=='UTM':
        expected['utm_zone']=grid.utm_zone
    if projection in {'TTM','LCC','LAZA'}:
        expected.update(false_easting_km=grid.false_easting_km,
                        false_northing_km=grid.false_northing_km)
    if projection=='TTM':
        expected['ttm_scale']=grid.ttm_scale
    if projection in {'TTM','LCC','PS','EM','LAZA'}:
        expected.update(latitude_origin=grid.lat0,longitude_origin=grid.lon0)
    if projection in {'LCC','PS'}:
        expected['standard_parallel_1']=grid.lat1
    if projection=='LCC':
        expected['standard_parallel_2']=grid.lat2
    differences=[]
    for name,wanted in expected.items():
        if name not in dataset.ncattrs(): continue
        actual=getattr(dataset,name)
        if isinstance(wanted,int): equal=int(actual)==wanted
        else: equal=_legacy_close(float(actual),float(wanted))
        if not equal: differences.append(f'{name}={actual!r} (expected {wanted!r})')
    text_expected={'projection':grid.projection}
    if projection=='UTM': text_expected['hemisphere']=grid.hemisphere
    for name,wanted in text_expected.items():
        if name in dataset.ncattrs() and str(getattr(dataset,name)).strip().upper()!=str(wanted).strip().upper():
            differences.append(f'{name}={getattr(dataset,name)!r} (expected {wanted!r})')
    if ('datum' in dataset.ncattrs() and
            normalize_datum(str(dataset.datum))!=normalize_datum(grid.datum)):
        differences.append(f'datum={dataset.datum!r} (expected {grid.datum!r})')
    if differences: raise ValueError('geophysical grid does not match CALMET control: '+', '.join(differences))

def _prognostic_precipitation_radius(mapper,requested_km,station_count):
    """Apply READHD's SIGMAP guard for gridded prognostic precipitation."""
    if station_count!=-1 or mapper is None: return requested_km
    # RDHD4/RDHD5 derives DXMM5 from projected (1,1) and (2,1), rather
    # than trusting the nominal WRF grid spacing.  The projected spacing
    # can differ by enough metres to change INTERPQR's inclusive boundary.
    stride=mapper.source_shape[0]
    first=np.asarray(mapper.source_coordinates[0],dtype=np.float32)
    second=np.asarray(mapper.source_coordinates[stride],dtype=np.float32)
    delta=np.asarray(first-second,dtype=np.float32)
    spacing=np.sqrt(np.asarray(delta[0]*delta[0]+delta[1]*delta[1],dtype=np.float32))
    maximum=np.float32(np.sqrt(np.float32(2.))*spacing*np.float32(1.05))
    requested=np.float32(requested_km)
    return float(maximum if requested<spacing or requested>maximum else requested)

def _output_path(config: CalmetConfig) -> Path:
    """Select OPENOT's METDAT/PACDAT target for the requested format."""
    if config.legacy_output_format==2 and config.mesopac_output is not None:
        return Path(config.mesopac_output)
    return Path(config.output)

def _write_distance_to_coast_grid(
        path: str | Path,landuse,grid: GridConfig,
        water_range: tuple[int,int]) -> np.ndarray:
    """Write SETCOAST's optional Surfer ASCII distance grid."""
    distance=distance_to_coast(landuse,water_range,grid.dx_km,grid.dy_km)
    xb=grid.x_origin_km+grid.dx_km
    yb=grid.y_origin_km+grid.dy_km
    xe=xb+(grid.nx-1)*grid.dx_km
    ye=yb+(grid.ny-1)*grid.dy_km
    lines=['DSAA',f'{grid.nx:12d}{grid.ny:12d}',
           f'{xb:12.4f}{xe:12.4f}',f'{yb:12.4f}{ye:12.4f}',
           f'{0.:12.4E}{float(np.max(distance)):12.4E}']
    lines.extend(''.join(f'{float(value):10.4E}  ' for value in row).rstrip()
                 for row in distance)
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return distance

def _validate(config: CalmetConfig):
    projection=config.grid.projection.upper()
    if projection not in {'UTM','LCC','TTM','PS','EM','LAZA'}:
        raise ValueError('CALMET PMAP must be UTM, LCC, TTM, PS, EM, or LAZA')
    if projection=='UTM':
        if not 1<=config.grid.utm_zone<=60: raise ValueError('CALMET UTM zone must be 1..60')
        if config.grid.hemisphere.upper() not in {'N','S'}:
            raise ValueError('CALMET UTM hemisphere must be N or S')
    else:
        if not -90.<=config.grid.lat0<=90. or not -180.<=config.grid.lon0<=180.:
            raise ValueError('CALMET projection origin latitude/longitude is out of range')
        if projection=='LCC' and not (-90.<=config.grid.lat1<=90. and
                                      -90.<=config.grid.lat2<=90.):
            raise ValueError('CALMET LCC matching latitude is out of range')
        if projection=='PS' and not -90.<=config.grid.lat1<=90.:
            raise ValueError('CALMET PS matching latitude is out of range')
    if config.grid.nx<=0 or config.grid.ny<=0:
        raise ValueError('CALMET NX and NY must be positive')
    if str(config.grid.datum).upper()=='UNKNOWN':
        raise ValueError('CALMET DATUM cannot be UNKNOWN')
    if not -12<=config.base_timezone<=12:
        raise ValueError('CALMET base timezone must be between -12 and 12')
    if (config.base_timezone_label is not None and
            _timezone_offset(config.base_timezone_label)!=config.base_timezone):
        raise ValueError('CALMET base timezone label does not match its integer offset')
    if config.use_initial_guess_calmet not in {0,1}:
        raise ValueError('IGFMET must be zero or one')
    if config.regulatory_mode not in {0,1}:
        raise ValueError('MREG must be zero or one')
    if config.legacy_output_format not in {1,2}:
        raise ValueError('IFORMO must be 1 (CALMET) or 2 (MESOPUFF-II)')
    if config.test_mode not in {1,2}:
        raise ValueError('ITEST must be 1 or 2')
    level_output=(config.print_uv_levels,config.print_w_levels,
                  config.print_temperature_levels)
    if any(values and len(values)!=len(config.z_faces_m)-1 for values in level_output):
        raise ValueError('IUVOUT/IWOUT/ITOUT must contain one value per CALMET layer')
    output_switches=tuple(value for values in level_output for value in values)
    output_switches+=tuple(config.print_surface_fields)+tuple(config.wind_debug_fields)
    if any(value not in {0,1} for value in output_switches):
        raise ValueError('CALMET print and debug field switches must be zero or one')
    if config.print_interval_hours<1:
        raise ValueError('IPRINF must be a positive number of hours')
    if config.debug_first_step<1 or config.debug_last_step<config.debug_first_step:
        raise ValueError('NN1/NN2 must define a positive increasing debug-step range')
    if config.wind_debug_output not in {0,1}:
        raise ValueError('IOUTD must be zero or one')
    if config.wind_debug_levels<0 or config.wind_debug_levels>len(config.z_faces_m)-1:
        raise ValueError('NZPRN2 must be between zero and the number of CALMET layers')
    if config.wind_debug_fields[3]:
        raise ValueError('legacy CALMET IPR3 crashes because FORMAT 2820 cannot consume NH')
    if (config.micrometeorology_debug and config.noobs in {0,1} and
            config.surface_temperature_option==-1):
        raise ValueError('LDB with NOOBS=0/1 and ISURFT=-1 prints undefined TINF in legacy CALMET')
    if config.iprog not in {0,1,2,3,4,5,13,14,15}:
        raise ValueError('IPROG must be one of 0, 1, 2, 3, 4, 5, 13, 14, or 15')
    if config.iwfcod not in {0,1}:
        raise ValueError('IWFCOD must be zero or one')
    if config.use_initial_guess_calmet:
        if not config.initial_guess_calmet_files:
            raise ValueError('IGFMET=1 requires at least one IGFDAT file')
        if config.iprog in {1,3,13}:
            raise ValueError(f'legacy CALMET does not allow IGFMET=1 with IPROG={config.iprog}')
    wind_mode=(config.iprog,config.iwfcod)
    if wind_mode not in {(0,0),(0,1),(1,0),(2,1),(3,0),(4,1),(5,0),(5,1),
                         (13,0),(14,1),(15,0),(15,1)}:
        raise ValueError(f'legacy CALMET does not allow IPROG={config.iprog} with IWFCOD={config.iwfcod}')
    if config.iprog in {0,1,2} and config.noobs!=0:
        raise ValueError(f'legacy CALMET requires NOOBS=0 for IPROG={config.iprog}')
    if config.iprog in {3,13} and config.noobs!=2:
        if config.weight_data is None: raise ValueError(f'observed IPROG={config.iprog} requires weight_data/WTDAT')
    if config.iprog in {1,3,13} and config.prognostic_radius_km==0.:
        raise ValueError(f'IPROG={config.iprog} requires non-zero RPROG')
    if config.iprog in {5,15} and config.noobs!=2 and config.weight_data is None:
        raise ValueError(f'observed IPROG={config.iprog} requires weight_data/WTDAT')
    if config.noobs not in {0,1,2}:
        raise ValueError('noobs/NOOBS must be 0, 1, or 2')
    if config.lake_breeze_regions and config.noobs==2:
        raise ValueError('LLBREZE requires meteorological station observations')
    for region in config.lake_breeze_regions:
        if len(region)!=9 or len(region[-1])<1 or not np.isfinite(region[:8]).all():
            raise ValueError('invalid lake_breeze_regions entry')
    diagnostic_options=(config.diagnostic_surface_temperature,config.diagnostic_lapse_rate,
                        config.diagnostic_initial_wind,config.diagnostic_surface_wind,
                        config.diagnostic_upper_wind)
    if any(value not in {0,1} for value in diagnostic_options):
        raise ValueError('IDIOPT1..5 must be zero or one')
    if any(diagnostic_options) and config.diagnostic_data is None:
        raise ValueError('enabled IDIOPT option requires diagnostic_data/DIADAT')
    if (config.diagnostic_surface_wind or config.diagnostic_upper_wind) and config.run_type!=0:
        raise ValueError('legacy CALMET allows IDIOPT4/5 only when IRTYPE=0')
    if (config.iprog in {5,15} and config.iwfcod==1 and config.noobs>=1 and
            not config.diagnostic_initial_wind):
        raise ValueError(f'legacy CALMET requires IDIOPT3=1 for IPROG={config.iprog} with NOOBS=1 or 2')
    if config.diagnostic_surface_temperature and config.surface_temperature_option==-1:
        raise ValueError('legacy CALMET does not allow ISURFT=-1 with IDIOPT1=1')
    if (config.noobs in {0,1} and config.surface_data is None and
            not config.diagnostic_surface_wind):
        raise ValueError('NOOBS=0/1 requires surface_data/SRFDAT')
    if (config.noobs in {0,1} and config.diagnostic_surface_wind and
            not config.surface_control_stations):
        raise ValueError('IDIOPT4 requires surface_control_stations/SS records when SURF.DAT is omitted')
    if config.noobs==0 and not config.upper_air_files and not config.diagnostic_upper_wind:
        raise ValueError('NOOBS=0 requires upper_air_files/UPDAT')
    if config.noobs==0 and config.diagnostic_upper_wind and not config.upper_control_stations:
        raise ValueError('IDIOPT5 requires upper_control_stations/US records when UPDAT is omitted')
    upper_station_count=(len(config.upper_control_stations)
                         if config.diagnostic_upper_wind else len(config.upper_air_files))
    if (config.noobs==0 and not config.diagnostic_initial_wind and
            config.initial_guess_upper_station!=-1 and
            not 1<=config.initial_guess_upper_station<=upper_station_count):
        raise ValueError('IUPWND must be -1 or identify an UPDAT station')
    if (len(config.initial_guess_height_range_m)!=2 or
            config.initial_guess_height_range_m[0]>=config.initial_guess_height_range_m[1]):
        raise ValueError('ZUPWND must contain increasing bottom and top heights')
    if config.itprog not in {0,1,2}:
        raise ValueError('ITPROG must be 0, 1, or 2')
    if config.iprog<3 and config.itprog!=0:
        raise ValueError('legacy CALMET requires ITPROG=0 when IPROG<3')
    if config.itprog==0 and config.noobs!=0:
        raise ValueError('ITPROG=0 requires NOOBS=0 upper-air observations')
    if config.itprog==1 and config.noobs==2:
        raise ValueError('ITPROG=1 requires NOOBS=0/1 surface observations')
    if config.irhprog not in {0,1}:
        raise ValueError('IRHPROG must be zero or one')
    if config.iprog<3 and config.irhprog!=0:
        raise ValueError('legacy CALMET requires IRHPROG=0 when IPROG<3')
    if config.noobs==2 and config.irhprog!=1:
        config.irhprog=1
    if config.temperature_interpolation_power not in {1,2}:
        raise ValueError('temperature_interpolation_power/IRAD must be 1 or 2')
    if config.temperature_radius_km<0. or config.temperature_station_limit<1:
        raise ValueError('TRADKM must be non-negative and NUMTS must be positive')
    if config.water_temperature_category_min>config.water_temperature_category_max:
        raise ValueError('JWAT1 must not exceed JWAT2')
    if config.barrier_count!=len(config.barriers):
        raise ValueError('barrier_count/NBAR does not match barrier coordinates')
    if any(value not in {0,1} for value in (config.ifradj,config.ikine,config.islope,config.iobr,config.iavet,config.iavezi)):
        raise ValueError('diagnostic effect and spatial-averaging switches must be either zero or one')
    if config.imixh not in {-2,-1,1,2}:
        raise ValueError('imixh must be one of -2, -1, 1, or 2')
    if config.water_temperature_mode not in {0,1,2}:
        raise ValueError('ITWPROG must be 0, 1, or 2')
    if config.water_temperature_mode==1 and config.iprog<3:
        raise ValueError('legacy CALMET requires ITWPROG=0 when IPROG<3')
    if config.water_temperature_mode==2 and config.iprog<13:
        raise ValueError('legacy CALMET allows ITWPROG=2 only when IPROG>=13')
    if config.land_flux_threshold<0. or config.water_flux_threshold<0.:
        raise ValueError('THRESHL and THRESHW must be non-negative')
    if config.overwater_flux_method not in {0,10,11,-11,12,-12}:
        raise ValueError('ICOARE must be 0, 10, +/-11, or +/-12')
    if config.coastal_shelf_length_km<0.:
        raise ValueError('DSHELF must be non-negative')
    if config.warm_layer not in {0,1} or config.cool_skin not in {0,1}:
        raise ValueError('IWARM and ICOOL must be zero or one')
    if config.regulatory_mode==1:
        regulatory=(config.overwater_flux_method==0 and config.land_flux_threshold==0. and
                    config.imixh==-1 and config.convective_relaxation==0 and
                    (config.itprog!=2 or config.surface_temperature_option==-2) and
                    (config.itprog<=0 or config.upper_temperature_option==-2))
        if not regulatory:
            raise ValueError('CALMET configuration fails MREG=1 regulatory checks')
    if config.run_type not in {0,1}:
        raise ValueError('run_type/IRTYPE must be zero or one')
    if config.timestep_seconds<=0 or 3600%config.timestep_seconds:
        raise ValueError('timestep_seconds/NSECDT must be a positive integer divisor of 3600')
    if config.legacy_output_format==2 and config.timestep_seconds!=3600:
        raise ValueError('MESOPAC output requires a one-hour NSECDT')
    if config.prognostic_timestep_seconds<=0 or config.prognostic_timestep_seconds%config.timestep_seconds:
        raise ValueError('prognostic_timestep_seconds/ISTEPPGS must be a positive multiple of NSECDT')
    if config.prognostic_timestep_seconds>43200:
        raise ValueError('prognostic_timestep_seconds/ISTEPPGS must not exceed 12 hours')
    if config.iprog<13 and config.prognostic_timestep_seconds!=3600:
        raise ValueError('legacy CALMET requires ISTEPPGS=3600 when IPROG<13')
    if config.cloud_method not in {1,2,3,4}:
        raise ValueError('cloud_method/MCLOUD must be 1, 2, 3, or 4')
    if config.cloud_output not in {0,1}:
        raise ValueError('cloud_output/ICLDOUT must be zero or one')
    if config.cloud_method==1 and config.noobs==2:
        raise ValueError('MCLOUD=1 requires NOOBS=0/1 surface observations')
    if config.cloud_method==2 and config.cloud_data is None:
        raise ValueError('MCLOUD=2 requires cloud_data/CLDDAT')
    if config.cloud_output and config.cloud_data is None:
        raise ValueError('ICLDOUT=1 requires cloud_data/CLDDAT')
    if config.cloud_output and config.cloud_method==2:
        raise ValueError('legacy CALMET does not allow ICLDOUT=1 with MCLOUD=2')
    if config.cloud_data_format not in {1,2}:
        raise ValueError('cloud_data_format/IFORMC must be 1 or 2')
    if config.surface_data_format not in {1,2}:
        raise ValueError('surface_data_format/IFORMS must be 1 or 2')
    if config.precipitation_station_count < -1:
        raise ValueError('precipitation_station_count/NPSTA must be -1, 0, or positive')
    if config.precipitation_station_count==-1 and config.iprog<3:
        raise ValueError('legacy CALMET allows NPSTA=-1 only when IPROG>=3')
    if config.precipitation_station_count>0 and config.precipitation_data is None:
        raise ValueError('positive NPSTA requires precipitation_data/PRCDAT')
    if config.precipitation_data_format not in {1,2}:
        raise ValueError('precipitation_data_format/IFORMP must be 1 or 2')
    if config.convective_relaxation not in {0,1}:
        raise ValueError('convective_relaxation must be zero or one')
    if config.convective_relaxation_time_s<1.:
        raise ValueError('convective_relaxation_time_s/TZICRLX must be at least one second')
    if config.precipitation_interpolation not in {1,2,3}:
        raise ValueError('precipitation_interpolation must be 1, 2, or 3')
    if config.precipitation_radius_km<0. or config.precipitation_cutoff<0.:
        raise ValueError('precipitation radius and cutoff must be non-negative')
    if config.precipitation_interpolation in {1,2} and config.precipitation_radius_km==0.:
        raise ValueError('precipitation radius must be positive for interpolation methods 1 and 2')
    if config.surface_temperature_option==0 or config.surface_temperature_option < -2:
        raise ValueError('ISURFT must be -2, -1, or a positive surface-station number')
    if config.itprog>0 and config.surface_temperature_option not in {-2,-1}:
        raise ValueError('ITPROG>0 requires ISURFT=-1 or -2')
    if config.itprog==0 and config.surface_temperature_option==-2:
        raise ValueError(
            'ITPROG=0 with ISURFT=-2 enters undefined TEMPK(-2) access in legacy CALMET')
    if config.upper_temperature_option==0 or config.upper_temperature_option < -2:
        raise ValueError('IUPT must be -2, -1, or a positive upper-air station number')
    if config.itprog>0 and config.upper_temperature_option not in {-2,-1}:
        raise ValueError('ITPROG>0 requires IUPT=-1 or -2')
    if (config.itprog==0 and config.upper_temperature_option==-2 and
            not config.diagnostic_lapse_rate):
        raise ValueError('ITPROG=0 allows IUPT=-2 only when IDIOPT2 supplies the lapse rate')
    if config.upper_temperature_option>0 and config.noobs!=0:
        raise ValueError('positive IUPT requires NOOBS=0 upper-air observations')
    if config.lapse_height_m<=0.:
        raise ValueError('lapse_height_m must be positive')
    faces=np.asarray(config.z_faces_m,dtype=float)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('z_faces_m must be strictly increasing and begin at zero')
    first_center=np.asarray(
        np.float32(.5)*np.asarray(
            np.float32(faces[0])+np.float32(faces[1]),dtype=np.float32),
        dtype=np.float32)
    if config.run_type!=0 and not np.float32(9.)<=first_center<=np.float32(11.):
        raise ValueError(
            'full CALMET runs require the first layer midpoint to be between 9 and 11 m')
    if config.barrier_top_level is None:
        config.barrier_top_level=faces.size-1
    if config.barrier_count and not 1<=config.barrier_top_level<faces.size:
        raise ValueError('barrier_top_level/KBAR must identify a CALMET layer')
    if config.grid.dx_km!=config.grid.dy_km:
        raise ValueError('legacy CALMET DGRID requires equal X and Y grid spacing')
    if len(config.smoothing_passes)!=faces.size-1:
        raise ValueError('smoothing_passes must contain one value per CALMET layer')
    if config.noobs!=2:
        if len(config.stations_per_level)!=faces.size-1 or any(value<1 for value in config.stations_per_level):
            raise ValueError('stations_per_level/NINTR2 must contain one positive value per CALMET layer')
        if len(config.wind_extrapolation_factors)!=faces.size-1:
            raise ValueError('wind_extrapolation_factors/FEXTR2 must contain one value per CALMET layer')
        if abs(config.wind_extrapolation) not in {1,2,3,4}:
            raise ValueError('wind_extrapolation/IEXTRP must be +/-1, +/-2, +/-3, or +/-4')
        if len(config.wind_bias)<faces.size-1 or any(abs(value)>1. for value in config.wind_bias[:faces.size-1]):
            raise ValueError('wind_bias/BIAS must provide one value in [-1,1] per CALMET layer')
        if config.extrapolate_calm not in {0,1}:
            raise ValueError('extrapolate_calm/ICALM must be zero or one')
    elif abs(config.wind_extrapolation)!=1:
        raise ValueError('legacy CALMET requires |IEXTRP|=1 when NOOBS=2')
    if config.iavezi and not 1<=config.mixing_wind_level<faces.size:
        raise ValueError('mixing_wind_level must identify a CALMET layer using one-based indexing')
    begin=_stamp(config.start); finish=_stamp(config.end)
    # READCF converts an explicit beginning time to its first hour-ending
    # internal timestamp before applying ``IBHR > 5``.  Consequently a
    # user-facing 05:00 start is already too late; 04:xx is the last hour.
    if config.run_type!=0 and begin.hour>=5:
        raise ValueError('CALMET prognostic mixing-height runs must start before 05 LST')
    for label,value in (('start',begin),('end',finish)):
        seconds=value.minute*60+value.second
        if seconds%config.timestep_seconds:
            raise ValueError(f'CALMET {label} time must align with NSECDT')
    return faces


def _initial_guess_at(config,initial_guess,timestamp,centers):
    try: igf_timezone=_timezone_offset(initial_guess.timezone)
    except ValueError as error:
        raise ValueError(f'unsupported IGFDAT timezone: {initial_guess.timezone!r}') from error
    igf_end=timestamp+timedelta(seconds=config.timestep_seconds,
                               hours=config.base_timezone-igf_timezone)
    return interpolate_initial_guess(initial_guess,igf_end,config.grid,centers)


def _noobs_step_transform(config,initial_guess,diagnostic,precipitation=None,
                          precipitation_points=None,sea=()):
    centers=.5*(np.asarray(config.z_faces_m[:-1])+np.asarray(config.z_faces_m[1:]))
    sea=tuple(sea)
    sea_points=(np.stack([data.point(config.grid) for data in sea])
                if sea else np.empty((0,2)))
    sea_longitudes=[data.longitude_latitude()[0] for data in sea]
    def transform(step,timestamp):
        changes={}
        # DIAG gives the prognostic Step-1 field precedence for IPROG=4/14.
        # run() still opens and validates IGFDAT, just as RDHDMET does.
        if initial_guess is not None and config.iprog not in {4,14}:
            changes['u'],changes['v']=_initial_guess_at(
                config,initial_guess,timestamp,centers)
        if diagnostic is not None:
            record=diagnostic.at_begin(timestamp)
            if record.surface_temperature_k is not None:
                changes['diagnostic_surface_temperature']=np.full_like(
                    step.surface_temperature,record.surface_temperature_k,dtype=float)
        if precipitation is not None:
            record=precipitation.at_begin(timestamp)
            if not (config.run_type==0 and config.diagnostic_lapse_rate):
                changes['precipitation_rate']=grid_precipitation(
                    record.rate_mm_h,precipitation_points,config.grid,
                    config.precipitation_interpolation,config.precipitation_radius_km,
                    config.precipitation_cutoff)
        if sea:
            records=[data.at_begin(timestamp) for data in sea]
            changes['observed_water_mixing_height']=grid_sea_mixing_height(
                [item.mixing_height_m for item in records],sea_points,config.grid)
            observed_water=nearest_sea_data(records,sea_points,sea_longitudes,config.grid)
            if config.water_temperature_mode==2:
                if step.overwater_data is None:
                    raise ValueError('ITWPROG=2 requires prognostic offshore data')
                wave_period,wave_height=nearest_sea_waves(records,sea_points,config.grid)
                changes['overwater_data']=replace(step.overwater_data,
                    wave_period_s=wave_period,wave_height_m=wave_height)
            else:
                changes['overwater_data']=observed_water
            if config.water_temperature_mode==0:
                below=np.asarray([item.gradient_below_k_m for item in records])
                above=np.asarray([item.gradient_above_k_m for item in records])
                below=np.where(below<9998.,below,config.water_temperature_gradient_below)
                above=np.where(above<9998.,above,config.water_temperature_gradient_above)
                interpolation=dict(radius_km=config.temperature_radius_km,
                    maximum_stations=config.temperature_station_limit,
                    power=config.temperature_interpolation_power)
                changes['water_gradient_below']=interpolate_surface_scalar(
                    below,sea_points,config.grid,**interpolation)
                changes['water_gradient_above']=interpolate_surface_scalar(
                    above,sea_points,config.grid,**interpolation)
                changes['water_mixing_gradient_above']=grid_water_mixing_gradient(
                    [getattr(item,'gradient_above_k_m',9999.) for item in records],
                    sea_points,config.grid,
                    config.water_temperature_gradient_above)
        return replace(step,**changes)
    return transform


def _surface_step_transform(config,surface,geo,upper_air=(),precipitation=None,sea=(),weights=None,
                            diagnostic=None,initial_guess=None,surface_stations=None,
                            upper_stations=None):
    """Compose PREPDI observations with WIND1/INTER2 for supported modes."""
    surface_stations=tuple(surface.stations if surface is not None else
                           (() if surface_stations is None else surface_stations))
    if not surface_stations:
        raise ValueError('surface station metadata is required for NOOBS=0/1')
    surface_datum=(surface.datum if surface is not None else config.grid.datum)
    points,nearest=station_grid_mapping(surface_stations,config.grid,surface_datum)
    relative=points-np.array([config.grid.x_origin_km,config.grid.y_origin_km])
    station_ix=np.clip(np.floor(relative[:,0]/config.grid.dx_km).astype(int),0,config.grid.nx-1)
    station_iy=np.clip(np.floor(relative[:,1]/config.grid.dy_km).astype(int),0,config.grid.ny-1)
    roughness=np.asarray(geo['roughness_length'])[station_iy,station_ix]
    latitude=np.array([station.latitude for station in surface_stations])
    longitude=np.array([station.longitude for station in surface_stations])
    convergence=grid_convergence(config.grid,longitude,latitude)
    anemometer=np.array([station.anemometer_height_m for station in surface_stations])
    centers=.5*(np.asarray(config.z_faces_m[:-1])+np.asarray(config.z_faces_m[1:]))
    landuse=np.asarray(geo['dominant_landuse'])
    upper_air=tuple(upper_air)
    upper_stations=tuple((data.station for data in upper_air) if upper_stations is None
                         else upper_stations)
    if upper_air and len(upper_air)!=len(upper_stations):
        raise ValueError('upper-air data and station metadata counts do not match')
    sea=tuple(sea)
    sea_points=np.stack([data.point(config.grid) for data in sea]) if sea else np.empty((0,2))
    sea_locations=[data.longitude_latitude() for data in sea]
    sea_convergence=(grid_convergence(config.grid,
        [item[0] for item in sea_locations],[item[1] for item in sea_locations]) if sea else np.empty(0))
    upper_points=[]; upper_convergence=[]
    for index,station in enumerate(upper_stations):
        datum=upper_air[index].datum if upper_air else config.grid.datum
        point,_=station_grid_mapping((station,),config.grid,datum)
        upper_points.append(point[0])
        upper_convergence.append(float(grid_convergence(
            config.grid,station.longitude,station.latitude)))
    coordinate_groups=[relative]
    if sea_points.size:
        coordinate_groups.append(sea_points-np.array([config.grid.x_origin_km,config.grid.y_origin_km]))
    if upper_points:
        coordinate_groups.append(np.asarray(upper_points)-np.array([config.grid.x_origin_km,config.grid.y_origin_km]))
    all_relative=np.vstack(coordinate_groups)
    all_station_ids=(np.asarray(
        [station.identifier for station in surface_stations]
        +[data.identifier for data in sea]
        +[station.identifier for station in upper_stations],dtype=np.int64)
        if config.lake_breeze_regions else None)
    precip_points=None
    if precipitation is not None:
        precip_points,_=station_grid_mapping(precipitation.stations,config.grid,precipitation.datum)
        precip_points-=np.array([config.grid.x_origin_km,config.grid.y_origin_km])
    common=dict(rmin_km=config.minimum_station_radius_km,
        rmax_surface_km=config.maximum_surface_radius_km,rmax_upper_km=config.maximum_upper_radius_km,
        rmax_water_km=config.maximum_water_radius_km,surface_scale_km=config.surface_background_radius_km,
        upper_scale_km=config.upper_background_radius_km,stations_per_level=config.stations_per_level,
        vary_radius=config.varying_radius,water_range=(int(getattr(geo,'water_category_min',50)),
                                                       int(getattr(geo,'water_category_max',55))),
        barriers=config.barriers,barrier_top_level=config.barrier_top_level)
    radiation_options=dict(
        turbidity_coefficient_1=config.turbidity_coefficient_1,
        turbidity_coefficient_2=config.turbidity_coefficient_2,
        cloudiness_coefficient_1=config.cloudiness_coefficient_1,
        cloudiness_coefficient_2=config.cloudiness_coefficient_2,
        net_radiation_coefficient_1=config.net_radiation_coefficient_1,
        net_radiation_coefficient_2=config.net_radiation_coefficient_2,
        net_radiation_coefficient_3=config.net_radiation_coefficient_3)
    if config.iprog==1:
        common.update(rmax_upper_km=config.maximum_surface_radius_km,
            rmax_water_km=config.maximum_surface_radius_km,observation_weight=.5,
            background_weight=.5/config.prognostic_radius_km**2)
    elif config.iprog in {3,13}:
        common.update(rmax_upper_km=config.maximum_surface_radius_km,
            rmax_water_km=config.maximum_surface_radius_km,observation_weight=weights,
            background_weight=(1.-weights)/config.prognostic_radius_km**2)
    def transform(step,timestamp,physical_context=None):
        diagnostic_record=None if diagnostic is None else diagnostic.at_begin(timestamp)
        if surface is None:
            # OPENOT/READHD deliberately omit SURF.DAT for IDIOPT4.  PREPDI
            # still needs the Group-7 locations, while every surface wind is
            # supplied by DIAG.DAT.  Sample the already available Step-1
            # fields only as inert thermodynamic scaffolding for IRTYPE=0.
            sample=(station_iy,station_ix)
            count=len(surface_stations)
            raw_record=SurfaceRecord(
                timestamp,timestamp+timedelta(seconds=config.timestep_seconds),
                np.full(count,9999.,dtype=np.float32),
                np.full(count,9999.,dtype=np.float32),
                np.trunc(np.asarray(step.ceiling_height)[sample]/30.48).astype(np.int32),
                np.trunc(np.asarray(step.cloud_fraction)[sample]*10.).astype(np.int32),
                np.asarray(step.surface_temperature,dtype=np.float32)[sample],
                np.trunc(np.asarray(step.surface_relative_humidity)[sample]).astype(np.int32),
                np.asarray(step.surface_pressure_mb,dtype=np.float32)[sample],
                np.asarray(step.precipitation_code,dtype=np.int32)[sample])
        else:
            raw_record=surface.at_begin(timestamp)
        record=fill_missing_surface(raw_record,points,cloud_method=config.cloud_method,itprog=config.itprog)
        sea_records=[data.at_begin(timestamp) for data in sea]
        observed_surface_temperature=None
        observed_temperature_faces=None
        observed_profile_temperature=None
        observed_profile_height=None
        if config.itprog in {0,1} and surface is None:
            observed_surface_temperature=np.asarray(step.surface_temperature,dtype=np.float32)
        elif config.itprog in {0,1}:
            water_min=config.water_temperature_category_min
            water_max=config.water_temperature_category_max
            temperature_values=np.concatenate((np.asarray(raw_record.temperature_k),
                                               [item.air_temperature_k for item in sea_records]))
            temperature_points=np.vstack((points,sea_points))
            observed_surface_temperature=np.asarray(interpolate_surface_scalar(
                temperature_values,temperature_points,config.grid,
                radius_km=config.temperature_radius_km,
                maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power,
                target_water=(landuse>=water_min)&(landuse<=water_max),
                station_water=np.r_[np.zeros(len(points),dtype=bool),
                                    np.ones(len(sea),dtype=bool)]),dtype=np.float32)
        if config.itprog==0 and upper_air:
            end=timestamp+timedelta(seconds=config.timestep_seconds,
                                    hours=config.base_timezone)
            station_faces=np.stack(
                [data.temperature_faces_at(end,config.z_faces_m) for data in upper_air])
            observed_temperature_faces=grid_temperature_faces(
                station_faces,np.asarray(upper_points),config.grid,
                config.temperature_interpolation_power)
            layer_temperature=np.asarray(np.float32(.5)*np.asarray(
                observed_temperature_faces[:-1]+observed_temperature_faces[1:],
                dtype=np.float32),dtype=np.float32)
            layer_temperature[0]=observed_surface_temperature
            faces=np.asarray(config.z_faces_m,dtype=np.float32)
            profile_centers=np.asarray(np.float32(.5)*np.asarray(
                faces[:-1]+faces[1:],dtype=np.float32),dtype=np.float32)
            observed_profile_height=np.broadcast_to(
                np.r_[np.float32(0.),profile_centers][:,None,None],
                (profile_centers.size+1,config.grid.ny,config.grid.nx))
            observed_profile_temperature=np.concatenate(
                (observed_surface_temperature[None],layer_temperature),axis=0)
        elif config.itprog==0:
            # IDIOPT5 omits every sounding, including its temperature profile.
            # IRTYPE=0 never writes temperature, but the Python hour object still
            # carries a well-shaped field through the wind solve.
            faces=np.asarray(config.z_faces_m,dtype=np.float32)
            target_temperature=np.asarray(step.temperature,dtype=np.float32)
            observed_temperature_faces=np.empty(
                (faces.size,config.grid.ny,config.grid.nx),dtype=np.float32)
            observed_temperature_faces[0]=observed_surface_temperature
            for level in range(target_temperature.shape[0]):
                observed_temperature_faces[level+1]=np.asarray(
                    np.float32(2.)*target_temperature[level]-
                    observed_temperature_faces[level],dtype=np.float32)
            layer_temperature=np.asarray(np.float32(.5)*np.asarray(
                observed_temperature_faces[:-1]+observed_temperature_faces[1:],
                dtype=np.float32),dtype=np.float32)
            layer_temperature[0]=observed_surface_temperature
            profile_centers=np.asarray(np.float32(.5)*np.asarray(
                faces[:-1]+faces[1:],dtype=np.float32),dtype=np.float32)
            observed_profile_height=np.broadcast_to(
                np.r_[np.float32(0.),profile_centers][:,None,None],
                (profile_centers.size+1,config.grid.ny,config.grid.nx))
            observed_profile_temperature=np.concatenate(
                (observed_surface_temperature[None],layer_temperature),axis=0)
        upper_u=upper_v=None
        if upper_stations:
            end=timestamp+timedelta(seconds=config.timestep_seconds,hours=config.base_timezone)
            if upper_air:
                upper_profiles=[data.layer_wind_at(end,config.z_faces_m,angle)
                                for data,angle in zip(upper_air,upper_convergence)]
                upper_u=np.column_stack([item[0] for item in upper_profiles])
                upper_v=np.column_stack([item[1] for item in upper_profiles])
            else:
                upper_profiles=()
            if diagnostic_record is not None and diagnostic_record.upper_u_m_s is not None:
                upper_u=np.asarray(diagnostic_record.upper_u_m_s,dtype=float).copy()
                upper_v=np.asarray(diagnostic_record.upper_v_m_s,dtype=float).copy()
            elif not upper_air:
                raise ValueError('IDIOPT5 requires upper-air wind records in DIAG.DAT')
            upper_valid=np.any((upper_u<900.)|(upper_v<900.),axis=0)
            if config.micrometeorology_debug and upper_air:
                debug_upper=[]
                for data,angle,(layer_u,layer_v) in zip(
                        upper_air,upper_convergence,upper_profiles):
                    if data.soundings[0].end_timestamp is not None:
                        soundings=tuple(item for item in data.soundings
                            if item.timestamp<end<=item.end_timestamp)
                    else:
                        soundings=data.bracket(end)
                    debug_upper.append((data.station,soundings,float(angle),
                                        np.asarray(layer_u,dtype=np.float32).copy(),
                                        np.asarray(layer_v,dtype=np.float32).copy()))
                debug_upper=tuple(debug_upper)
            else:
                debug_upper=()
        else:
            upper_valid=np.empty(0,dtype=bool)
            debug_upper=()
        sea_speed=np.asarray([item.wind_speed_m_s for item in sea_records],dtype=float)
        sea_direction=np.asarray([item.wind_direction_deg for item in sea_records],dtype=float)
        sea_anemometer=np.asarray([item.anemometer_height_m for item in sea_records],dtype=float)
        sea_anemometer=np.where(sea_anemometer>9998.,10.,sea_anemometer)
        sea_roughness=hosker_roughness(np.where(
            (sea_speed>0.)&(sea_speed<9998.),sea_speed,0.))
        wind_speed=np.concatenate((np.asarray(record.wind_speed),sea_speed))
        wind_direction=np.concatenate((np.asarray(record.wind_direction),sea_direction))
        wind_anemometer=np.concatenate((anemometer,sea_anemometer))
        wind_roughness=np.concatenate((roughness,sea_roughness))
        wind_latitude=np.concatenate((latitude,np.asarray(
            [item[1] for item in sea_locations],dtype=float)))
        wind_convergence=np.concatenate((convergence,sea_convergence))
        surface_relative=np.vstack((relative,sea_points-np.array(
            [config.grid.x_origin_km,config.grid.y_origin_km]))) if sea else relative
        power_exponent=np.r_[np.full(len(record.wind_speed),.143),
                             np.full(len(sea_records),.286)]
        profile_options=dict(station_xy=surface_relative,
            upper_xy=(np.asarray(upper_points)-np.array(
                [config.grid.x_origin_km,config.grid.y_origin_km]) if upper_points else None),
            upper_valid=upper_valid,
            upper_air_separation_cells=config.upper_air_separation_cells,
            dx_km=config.grid.dx_km,dy_km=config.grid.dy_km,
            power_exponent=power_exponent)
        if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
            observed_u,observed_v=component_wind_profiles(
                diagnostic_record.surface_u_m_s,diagnostic_record.surface_v_m_s,
                wind_anemometer,centers,wind_roughness,config.wind_extrapolation,
                config.wind_extrapolation_factors,wind_latitude,**profile_options)
        else:
            observed_u,observed_v=wind_profiles(wind_speed,wind_direction,wind_anemometer,
                centers,wind_roughness,config.wind_extrapolation,
                config.wind_extrapolation_factors,wind_latitude,wind_convergence,
                **profile_options)
        if abs(config.wind_extrapolation)==4:
            if physical_context is None:
                raise RuntimeError('IEXTRP=4 requires the STHEOR runtime context')
            station_landuse=landuse[station_iy,station_ix]
            water_min,water_max=common['water_range']
            station_count=len(record.wind_speed)
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                raw_u=np.asarray(diagnostic_record.surface_u_m_s,dtype=np.float32)
                raw_v=np.asarray(diagnostic_record.surface_v_m_s,dtype=np.float32)
            else:
                raw_u,raw_v,_,_=_wind_components(
                    wind_speed,wind_direction,wind_convergence)
            station_cloud=(np.asarray(step.cloud_fraction,dtype=np.float32)[station_iy,station_ix]
                           if config.cloud_method>1 else
                           np.asarray(record.cloud_tenths,dtype=np.float32)*np.float32(.1))
            station_temperature=(
                np.asarray(record.temperature_k,dtype=np.float32)
                if config.overwater_flux_method==0 and config.itprog<2 else
                observed_surface_temperature[station_iy,station_ix]
                if config.itprog<2 else
                np.asarray(step.temperature[0],dtype=np.float32)[station_iy,station_ix])
            station_heat,_=heat_flux(
                physical_context.solar_elevation_sine[station_iy,station_ix],station_cloud,
                station_temperature,station_landuse,
                np.asarray(geo['albedo'])[station_iy,station_ix],
                np.asarray(geo['bowen_ratio'])[station_iy,station_ix],
                np.asarray(geo['soil_heat_flux_parameter'])[station_iy,station_ix],
                np.asarray(geo['anthropogenic_heat_flux'])[station_iy,station_ix],
                water_range=common['water_range'],**radiation_options)
            mixing_options=dict(timestep_s=config.timestep_seconds,
                depth_m=config.mixing_lapse_depth_m,
                minimum_lapse=config.minimum_lapse_rate,
                minimum_height_m=config.minimum_mixing_height_m,
                maximum_height_m=config.maximum_mixing_height_m,
                threshold_flux_per_m=config.land_flux_threshold,
                constb=config.constb,constn=config.constn,
                relaxation=bool(config.convective_relaxation),
                relaxation_time_s=config.convective_relaxation_time_s)
            if abs(config.imixh)==1: mixing_options['conste']=config.conste
            ipsifcn=1 if config.overwater_flux_method==0 else 0
            def ordered_water_profiles(function,arguments,keywords,grid_iy=None,grid_ix=None):
                warm=(config.warm_layer and function is coare_water_station_profiles)
                if config.imixh!=1 and not warm:
                    return function(*arguments,**keywords)
                if warm:
                    if grid_iy is None or grid_ix is None:
                        raise RuntimeError('COARE warm-layer stations require grid indices')
                    if physical_context.warm_layer_state is None:
                        physical_context.warm_layer_state=initial_warm_layer_state(landuse.shape)
                shared_jump=np.float32(physical_context.previous_temperature_jump[0,0])
                profile_u=[]; profile_v=[]; count=len(arguments[0])
                for station_index in range(count):
                    station_arguments=tuple(np.asarray(value)[station_index:station_index+1]
                                            for value in arguments)
                    station_keywords={}
                    for name,value in keywords.items():
                        array=np.asarray(value)
                        station_keywords[name]=(array[station_index:station_index+1]
                            if array.shape==(count,) else value)
                    station_keywords['previous_temperature_jump_k']=np.array(
                        [shared_jump],dtype=np.float32)
                    if warm:
                        iy=int(grid_iy[station_index]); ix=int(grid_ix[station_index])
                        state=physical_context.warm_layer_state
                        station_keywords['warm_layer_state']=WarmLayerState(*(
                            np.asarray(getattr(state,name))[iy,ix].reshape(1)
                            for name in WarmLayerState.__dataclass_fields__))
                    station_profiles=function(*station_arguments,**station_keywords)
                    profile_u.append(station_profiles[0][:,0])
                    profile_v.append(station_profiles[1][:,0])
                    next_jump=np.asarray(station_profiles[7],dtype=np.float32)[0]
                    if np.isfinite(next_jump): shared_jump=next_jump
                    if warm:
                        next_state=station_profiles[-1]
                        for name in WarmLayerState.__dataclass_fields__:
                            np.asarray(getattr(physical_context.warm_layer_state,name))[iy,ix]=(
                                np.asarray(getattr(next_state,name)).reshape(-1)[0])
                physical_context.previous_temperature_jump[0,0]=shared_jump
                return np.column_stack(profile_u),np.column_stack(profile_v)

            surface_water=(station_landuse>=water_min)&(station_landuse<=water_max)
            surface_land=~surface_water
            if surface_land.any():
                selected=np.flatnonzero(surface_land)
                land_profiles=land_station_similarity_theory(
                    raw_u[selected],raw_v[selected],roughness[selected],anemometer[selected],
                    centers[0],np.asarray(step.surface_density)[station_iy[selected],station_ix[selected]],
                    station_temperature[selected],station_cloud[selected],station_heat[selected],
                    np.asarray(observed_profile_temperature
                        if config.itprog==0 else step.profile_temperature
                        )[:,station_iy[selected],station_ix[selected]],
                    np.asarray(observed_profile_height
                        if config.itprog==0 else step.profile_height
                        )[:,station_iy[selected],station_ix[selected]],
                    physical_context.previous_convective_height[station_iy[selected],station_ix[selected]],
                    physical_context.previous_temperature_jump[station_iy[selected],station_ix[selected]],
                    physical_context.coriolis[station_iy[selected],station_ix[selected]],centers,
                    config.z_faces_m,latitude[selected],mixing_method=config.imixh,
                    ipsifcn=ipsifcn,minimum_mixing_height_m=config.minimum_mixing_height_m,
                    **mixing_options)
                observed_u[:,selected]=land_profiles[0]; observed_v[:,selected]=land_profiles[1]
            if surface_water.any():
                selected=np.flatnonzero(surface_water)
                if config.water_temperature_mode!=2:
                    water_profiles=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],anemometer[selected],
                        physical_context.previous_mixing_height[station_iy[selected],station_ix[selected]],
                        physical_context.coriolis[station_iy[selected],station_ix[selected]],centers,
                        config.z_faces_m,latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                else:
                    if step.overwater_data is None:
                        raise ValueError('ITWPROG=2 requires prognostic offshore data')
                    iy=station_iy[selected]; ix=station_ix[selected]; offshore=step.overwater_data
                    air=np.asarray(offshore.air_temperature_k)[iy,ix]
                    sea_temperature=np.asarray(offshore.sea_temperature_k)[iy,ix]
                    delta=np.asarray(air-sea_temperature,dtype=np.float32)
                    humidity=np.asarray(offshore.relative_humidity)[iy,ix]
                    thermal_height=np.asarray(offshore.air_sensor_height_m)[iy,ix]
                    preliminary=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],anemometer[selected],
                        physical_context.previous_mixing_height[iy,ix],
                        physical_context.coriolis[iy,ix],centers,config.z_faces_m,
                        latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                    _,_,potential_lapse=profile_temperature_layer(
                        np.asarray(step.profile_temperature)[:,iy,ix][:,None,:],
                        np.asarray(step.profile_height)[:,iy,ix][:,None,:],
                        preliminary[5][None,:],config.mixing_lapse_depth_m,
                        minimum_lapse=config.minimum_lapse_rate)
                    water_common=dict(observed_mixing_height_m=np.full(selected.size,9999.,dtype=np.float32),
                        current_mixing_height_m=physical_context.previous_mixing_height[iy,ix],
                        coriolis=physical_context.coriolis[iy,ix],
                        potential_lapse_k_m=potential_lapse[0],
                        previous_temperature_jump_k=physical_context.previous_temperature_jump[iy,ix],
                        z_centers_m=centers,z_faces_m=config.z_faces_m,
                        station_latitude=latitude[selected],mixing_method=config.imixh,
                        constw=config.constw,minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,
                        threshold_flux_per_m=config.water_flux_threshold,ipsifcn=ipsifcn,
                        timestep_s=config.timestep_seconds,minimum_lapse=config.minimum_lapse_rate,
                        relaxation=bool(config.convective_relaxation),
                        relaxation_time_s=config.convective_relaxation_time_s)
                    if config.overwater_flux_method==0:
                        function=delta_temperature_water_station_profiles
                        arguments=(raw_u[selected],raw_v[selected],anemometer[selected],
                                   air,delta,humidity)
                        keywords=dict(water_common,
                            temperature_measurement_height_m=thermal_height,
                            sea_temperature_k=sea_temperature)
                    else:
                        cloud=np.asarray(step.cloud_fraction)[iy,ix]
                        _,shortwave=heat_flux(
                            physical_context.solar_elevation_sine[iy,ix],cloud,
                            np.asarray(step.temperature[0])[iy,ix],landuse[iy,ix],
                            np.asarray(geo['albedo'])[iy,ix],np.asarray(geo['bowen_ratio'])[iy,ix],
                            np.asarray(geo['soil_heat_flux_parameter'])[iy,ix],
                            np.asarray(geo['anthropogenic_heat_flux'])[iy,ix],
                            water_range=common['water_range'],**radiation_options)
                        longwave=downward_longwave(np.asarray(step.temperature[0])[iy,ix],
                            np.asarray(step.surface_relative_humidity)[iy,ix],cloud,
                            np.asarray(step.ceiling_height)[iy,ix])
                        coast=distance_to_coast(landuse,common['water_range'],
                                                config.grid.dx_km,config.grid.dy_km)
                        function=coare_water_station_profiles
                        arguments=(raw_u[selected],raw_v[selected],anemometer[selected],
                                   thermal_height,air,delta,humidity)
                        keywords=dict(water_common,sea_temperature_k=sea_temperature,
                            shortwave_down_w_m2=shortwave,longwave_down_w_m2=longwave,
                            rainfall_mm_h=np.asarray(step.precipitation_rate)[iy,ix],
                            coare_method=config.overwater_flux_method,
                            coastal_shelf_length_km=config.coastal_shelf_length_km,
                            distance_to_coast_km=coast[iy,ix],cool_skin=bool(config.cool_skin),
                            solar_time_s=np.mod(np.asarray(offshore.station_longitude)[iy,ix]/15.
                                                +(physical_context.gmt_hour or 0)+24.,24.)*3600.,
                            water_sensor_depth_m=.05)
                    water_profiles=ordered_water_profiles(function,arguments,keywords,iy,ix)
                observed_u[:,selected]=water_profiles[0]; observed_v[:,selected]=water_profiles[1]
            if sea_records:
                if config.water_temperature_mode not in {0,1,2}:
                    raise ValueError('ITWPROG must be 0, 1, or 2')
                origin=np.array([config.grid.x_origin_km,config.grid.y_origin_km])
                sea_relative=sea_points-origin
                sea_ix=np.clip(np.floor(sea_relative[:,0]/config.grid.dx_km).astype(int),0,config.grid.nx-1)
                sea_iy=np.clip(np.floor(sea_relative[:,1]/config.grid.dy_km).astype(int),0,config.grid.ny-1)
                selected=np.arange(station_count,station_count+len(sea_records))
                thermal_height=None
                if config.water_temperature_mode in {0,1}:
                    sea_air=np.asarray([item.air_temperature_k for item in sea_records],dtype=np.float32)
                    sea_delta=np.asarray([item.air_water_delta_k for item in sea_records],dtype=np.float32)
                    sea_rh=np.asarray([item.relative_humidity for item in sea_records],dtype=np.float32)
                    sea_observed=np.asarray([item.mixing_height_m for item in sea_records],dtype=np.float32)
                    gradient=np.asarray([item.gradient_above_k_m for item in sea_records],dtype=np.float32)
                    gradient=np.where(gradient<np.float32(9998.),gradient,
                                      np.float32(config.water_temperature_gradient_above))
                    potential_lapse=gradient+np.float32(.0098)
                    if config.water_temperature_mode==1:
                        preliminary=neutral_water_station_profiles(
                            raw_u[selected],raw_v[selected],sea_anemometer,
                            physical_context.previous_mixing_height[sea_iy,sea_ix],
                            physical_context.coriolis[sea_iy,sea_ix],centers,
                            config.z_faces_m,wind_latitude[selected],constw=config.constw,
                            minimum_height_m=config.minimum_water_mixing_height_m,
                            maximum_height_m=config.maximum_water_mixing_height_m,
                            ipsifcn=ipsifcn)
                        _,_,potential_lapse=profile_temperature_layer(
                            np.asarray(step.profile_temperature)[:,sea_iy,sea_ix][:,None,:],
                            np.asarray(step.profile_height)[:,sea_iy,sea_ix][:,None,:],
                            preliminary[5][None,:],config.mixing_lapse_depth_m,
                            minimum_lapse=config.minimum_lapse_rate)
                        potential_lapse=potential_lapse[0]
                else:
                    if step.overwater_data is None:
                        raise ValueError('ITWPROG=2 requires prognostic offshore data')
                    offshore=step.overwater_data
                    sea_air=np.asarray(offshore.air_temperature_k)[sea_iy,sea_ix]
                    sea_temperature=np.asarray(offshore.sea_temperature_k)[sea_iy,sea_ix]
                    sea_delta=np.asarray(sea_air-sea_temperature,dtype=np.float32)
                    sea_rh=np.asarray(offshore.relative_humidity)[sea_iy,sea_ix]
                    sea_observed=np.full(len(sea_records),np.float32(9999.))
                    thermal_height=np.asarray(offshore.air_sensor_height_m)[sea_iy,sea_ix]
                    preliminary=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],sea_anemometer,
                        physical_context.previous_mixing_height[sea_iy,sea_ix],
                        physical_context.coriolis[sea_iy,sea_ix],centers,
                        config.z_faces_m,wind_latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                    _,_,potential_lapse=profile_temperature_layer(
                        np.asarray(step.profile_temperature)[:,sea_iy,sea_ix][:,None,:],
                        np.asarray(step.profile_height)[:,sea_iy,sea_ix][:,None,:],
                        preliminary[5][None,:],config.mixing_lapse_depth_m,
                        minimum_lapse=config.minimum_lapse_rate)
                    potential_lapse=potential_lapse[0]
                common_water=dict(
                    observed_mixing_height_m=sea_observed,
                    current_mixing_height_m=physical_context.previous_mixing_height[sea_iy,sea_ix],
                    coriolis=physical_context.coriolis[sea_iy,sea_ix],
                    potential_lapse_k_m=potential_lapse,
                    previous_temperature_jump_k=physical_context.previous_temperature_jump[sea_iy,sea_ix],
                    z_centers_m=centers,z_faces_m=config.z_faces_m,
                    station_latitude=wind_latitude[selected],mixing_method=config.imixh,
                    constw=config.constw,minimum_height_m=config.minimum_water_mixing_height_m,
                    maximum_height_m=config.maximum_water_mixing_height_m,
                    threshold_flux_per_m=config.water_flux_threshold,ipsifcn=ipsifcn,
                    timestep_s=config.timestep_seconds,minimum_lapse=config.minimum_lapse_rate,
                    relaxation=bool(config.convective_relaxation),
                    relaxation_time_s=config.convective_relaxation_time_s)
                if config.overwater_flux_method==0:
                    profile_function=delta_temperature_water_station_profiles
                    profile_arguments=(raw_u[selected],raw_v[selected],sea_anemometer,
                                       sea_air,sea_delta,sea_rh)
                    profile_keywords=(common_water if thermal_height is None else
                        dict(common_water,temperature_measurement_height_m=thermal_height,
                             sea_temperature_k=sea_temperature))
                else:
                    cloud=np.asarray(step.cloud_fraction)[sea_iy,sea_ix]
                    _,shortwave=heat_flux(
                        physical_context.solar_elevation_sine[sea_iy,sea_ix],cloud,
                        np.asarray(step.temperature[0])[sea_iy,sea_ix],landuse[sea_iy,sea_ix],
                        np.asarray(geo['albedo'])[sea_iy,sea_ix],
                        np.asarray(geo['bowen_ratio'])[sea_iy,sea_ix],
                        np.asarray(geo['soil_heat_flux_parameter'])[sea_iy,sea_ix],
                        np.asarray(geo['anthropogenic_heat_flux'])[sea_iy,sea_ix],
                        water_range=common['water_range'],**radiation_options)
                    longwave=downward_longwave(
                        np.asarray(step.temperature[0])[sea_iy,sea_ix],
                        np.asarray(step.surface_relative_humidity)[sea_iy,sea_ix],cloud,
                        np.asarray(step.ceiling_height)[sea_iy,sea_ix])
                    coast=distance_to_coast(landuse,common['water_range'],
                                            config.grid.dx_km,config.grid.dy_km)
                    if thermal_height is None:
                        air_height=np.asarray([item.air_sensor_height_m for item in sea_records],dtype=np.float32)
                        air_height=np.where(air_height<np.float32(9998.),air_height,sea_anemometer)
                    else:
                        air_height=thermal_height
                    profile_function=coare_water_station_profiles
                    profile_arguments=(raw_u[selected],raw_v[selected],sea_anemometer,
                                       air_height,sea_air,sea_delta,sea_rh)
                    profile_keywords=dict(common_water,
                        **({} if thermal_height is None else
                           {'sea_temperature_k':sea_temperature}),
                        shortwave_down_w_m2=shortwave,longwave_down_w_m2=longwave,
                        rainfall_mm_h=np.asarray(step.precipitation_rate)[sea_iy,sea_ix],
                        coare_method=config.overwater_flux_method,
                        wave_period_s=np.asarray([item.wave_period_s for item in sea_records]),
                        wave_height_m=np.asarray([item.wave_height_m for item in sea_records]),
                        coastal_shelf_length_km=config.coastal_shelf_length_km,
                        distance_to_coast_km=coast[sea_iy,sea_ix],cool_skin=bool(config.cool_skin),
                        solar_time_s=np.mod((np.asarray(offshore.station_longitude)[sea_iy,sea_ix]
                            if thermal_height is not None else
                            np.asarray([item[0] for item in sea_locations]))/15.
                            +(physical_context.gmt_hour or 0)+24.,24.)*3600.,
                        water_sensor_depth_m=(.05 if thermal_height is not None else
                            np.where(np.asarray([item.water_sensor_depth_m
                                for item in sea_records])<9998.,
                                np.asarray([item.water_sensor_depth_m
                                    for item in sea_records]),.6)),
                    )
                sea_profiles=ordered_water_profiles(
                    profile_function,profile_arguments,profile_keywords,sea_iy,sea_ix)
                observed_u[:,selected]=sea_profiles[0]; observed_v[:,selected]=sea_profiles[1]
            skip=_near_upper_air(len(wind_speed),surface_relative,
                profile_options['upper_xy'],upper_valid,config.upper_air_separation_cells,
                config.grid.dx_km,config.grid.dy_km)
            neutral_u,neutral_v=wind_profiles(wind_speed,wind_direction,wind_anemometer,
                centers,wind_roughness,config.wind_extrapolation,
                config.wind_extrapolation_factors,wind_latitude,wind_convergence,
                **profile_options)
            observed_u[:,:len(wind_speed)]=np.where(
                skip[None,:],neutral_u,observed_u[:,:len(wind_speed)])
            observed_v[:,:len(wind_speed)]=np.where(
                skip[None,:],neutral_v,observed_v[:,:len(wind_speed)])
        if config.micrometeorology_debug:
            if config.surface_temperature_option>0:
                tinf=float(record.temperature_k[config.surface_temperature_option-1])
            elif config.surface_temperature_option==-2:
                tinf=float(_domain_mean_real4(step.temperature[0]))
            else:
                tinf=None
            if (diagnostic_record is not None and
                    diagnostic_record.initial_u_m_s is not None):
                domain_u=float(diagnostic_record.initial_u_m_s)
                domain_v=float(diagnostic_record.initial_v_m_s)
            elif config.noobs==0 and config.initial_guess_upper_station>0:
                selected=config.initial_guess_upper_station-1
                if upper_air:
                    domain_profile=upper_air[selected].layer_wind_at(
                        end,config.initial_guess_height_range_m,
                        upper_convergence[selected])
                    domain_u=float(domain_profile[0][0])
                    domain_v=float(domain_profile[1][0])
                else:
                    valid=(upper_u[:,selected]<900.)&(upper_v[:,selected]<900.)
                    domain_u=float(np.mean(upper_u[valid,selected],dtype=np.float32))
                    domain_v=float(np.mean(upper_v[valid,selected],dtype=np.float32))
            else:
                domain_u=domain_v=0.
            debug_wind=dict(
                debug_station_names=tuple(
                    [item.name for item in surface_stations]+[item.identifier for item in sea]),
                debug_station_u=np.asarray(observed_u[0],dtype=np.float32).copy(),
                debug_station_v=np.asarray(observed_v[0],dtype=np.float32).copy(),
                debug_domain_temperature=tinf,
                debug_domain_u=domain_u,
                debug_domain_v=domain_v,
                debug_upper_data=debug_upper)
        if config.iprog in {3,13} and config.noobs==1:
            observed_u[1:]=9999.; observed_v[1:]=9999.
        if sea_records:
            changes_water_mixing=grid_sea_mixing_height(
                [item.mixing_height_m for item in sea_records],sea_points,config.grid)
        if upper_stations:
            surface_names={str(station.name).strip().upper() for station in surface_stations}
            omit=np.array([config.wind_extrapolation<=0 or
                           station.name.strip().upper() in surface_names
                           for station in upper_stations])
            upper_u[0,omit]=9999.; upper_v[0,omit]=9999.
            observed_u=np.hstack((observed_u,upper_u)); observed_v=np.hstack((observed_v,upper_v))
        analysis_common=common
        if config.iprog in {5,15}:
            if step.prognostic_observation_strength is None:
                raise ValueError(f'IPROG={config.iprog} requires prognostic observation strength')
            prognostic_strength=np.broadcast_to(
                step.prognostic_observation_strength,step.u.shape)
            analysis_common={**common,'observation_weight':weights,
                'background_weight':(1.-weights)*prognostic_strength}
        background_u,background_v=step.u,step.v
        if initial_guess is not None and config.iprog not in {4,14}:
            background_u,background_v=_initial_guess_at(
                config,initial_guess,timestamp,centers)
        elif (config.iprog==0 and not config.diagnostic_initial_wind and
              config.initial_guess_upper_station>0):
            station_index=config.initial_guess_upper_station-1
            if upper_air:
                end=timestamp+timedelta(seconds=config.timestep_seconds,
                                        hours=config.base_timezone)
                initial_u,initial_v=upper_air[station_index].layer_wind_at(
                    end,config.initial_guess_height_range_m,
                    upper_convergence[station_index])
                initial_u=float(initial_u[0]); initial_v=float(initial_v[0])
            else:
                selected_u=np.asarray(upper_u[:,station_index],dtype=np.float32)
                selected_v=np.asarray(upper_v[:,station_index],dtype=np.float32)
                bottom,top=config.initial_guess_height_range_m
                selected=(centers>=bottom)&(centers<=top)&(selected_u<900.)&(selected_v<900.)
                if not selected.any():
                    selected=(selected_u<900.)&(selected_v<900.)
                if not selected.any():
                    raise ValueError('DIAG.DAT has no valid upper-air wind for IUPWND')
                initial_u=float(np.mean(selected_u[selected],dtype=np.float32))
                initial_v=float(np.mean(selected_v[selected],dtype=np.float32))
            background_u=np.full_like(step.u,initial_u,dtype=float)
            background_v=np.full_like(step.v,initial_v,dtype=float)
        elif config.iprog==0 and not config.diagnostic_initial_wind:
            wind1_roughness=wind_roughness.copy()
            land_count=len(record.wind_speed)
            station_landuse=landuse[station_iy,station_ix]
            land_on_water=((station_landuse>=common['water_range'][0])&
                           (station_landuse<=common['water_range'][1]))
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                land_u=np.asarray(diagnostic_record.surface_u_m_s[:land_count],dtype=np.float32)
                land_v=np.asarray(diagnostic_record.surface_v_m_s[:land_count],dtype=np.float32)
                land_speed=np.asarray(np.sqrt(np.asarray(
                    land_u*land_u+land_v*land_v,dtype=np.float32)),dtype=np.float32)
                valid_land=(land_u<=900.)&(land_v<=900.)&(land_speed>0.)
            else:
                land_u,land_v,valid_land,_=_wind_components(
                    record.wind_speed,record.wind_direction,convergence)
                land_speed=np.asarray(np.sqrt(np.asarray(
                    land_u*land_u+land_v*land_v,dtype=np.float32)),dtype=np.float32)
                valid_land&=land_speed>0.
            water_wind=land_on_water&valid_land
            dynamic_z0=hosker_roughness(land_speed[water_wind])
            land_roughness=wind1_roughness[:land_count]
            land_roughness[water_wind]=dynamic_z0
            roughness_override=np.asarray(geo['roughness_length']).copy()
            for x_index,y_index,value in zip(station_ix[water_wind],station_iy[water_wind],dynamic_z0):
                roughness_override[y_index,x_index]=value
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                initial_u,initial_v=component_wind_profiles(
                    diagnostic_record.surface_u_m_s,diagnostic_record.surface_v_m_s,
                    wind_anemometer,centers,wind1_roughness,config.wind_extrapolation,
                    config.wind_extrapolation_factors,wind_latitude,
                    power_exponent=power_exponent)
            else:
                initial_u,initial_v=wind_profiles(wind_speed,wind_direction,wind_anemometer,
                    centers,wind1_roughness,config.wind_extrapolation,
                    config.wind_extrapolation_factors,wind_latitude,wind_convergence,
                    power_exponent=power_exponent)
            if upper_stations:
                initial_u=np.hstack((initial_u,upper_u)); initial_v=np.hstack((initial_v,upper_v))
            bias=np.asarray(config.wind_bias[:centers.size],dtype=float).copy()
            if config.wind_extrapolation<0.: bias[0]=-1.
            if abs(config.wind_extrapolation)==1: bias[1:]=1.
            background_u,background_v=wind1_initial_guess(initial_u,initial_v,all_relative,
                len(wind_speed),config.grid.nx,config.grid.ny,config.grid.dx_km,
                config.grid.dy_km,bias=bias,
                extrapolate_calm=bool(config.extrapolate_calm),barriers=config.barriers,
                barrier_top_level=config.barrier_top_level)
        elif config.iprog==0:
            background_u=np.full_like(step.u,diagnostic_record.initial_u_m_s,dtype=float)
            background_v=np.full_like(step.v,diagnostic_record.initial_v_m_s,dtype=float)
        u=objective_analysis_component(observed_u,all_relative,background_u,landuse,
            config.grid.dx_km,config.grid.dy_km,**analysis_common)
        v=objective_analysis_component(observed_v,all_relative,background_v,landuse,
            config.grid.dx_km,config.grid.dy_km,**analysis_common)
        if config.lake_breeze_regions:
            u,v=lake_breeze_adjust(
                u,v,observed_u[0],observed_v[0],all_relative*1000.,all_station_ids,
                config.lake_breeze_regions,config.grid.dx_km*1000.,config.grid.dy_km*1000.)
        changes=dict(u=u,v=v)
        if config.micrometeorology_debug:
            changes.update(debug_surface_record=raw_record,
                           debug_surface_stations=surface_stations,
                           **debug_wind)
        if (config.iprog==0 and not config.diagnostic_initial_wind and initial_guess is None and
                config.initial_guess_upper_station==-1):
            changes['roughness_override']=roughness_override
        if diagnostic_record is not None:
            if diagnostic_record.surface_temperature_k is not None:
                changes['diagnostic_surface_temperature']=np.full_like(
                    step.surface_temperature,diagnostic_record.surface_temperature_k,dtype=float)
            if diagnostic_record.lapse_rate_k_m is not None:
                changes['diagnostic_lapse_rate']=np.full_like(
                    step.surface_temperature,diagnostic_record.lapse_rate_k_m,dtype=float)
        if config.surface_temperature_option>0 and not config.diagnostic_surface_temperature:
            index=config.surface_temperature_option-1
            if index>=len(surface_stations): raise ValueError('ISURFT exceeds the SURF.DAT station count')
            changes['diagnostic_surface_temperature']=np.full_like(
                step.surface_temperature,float(record.temperature_k[index]))
        if (config.upper_temperature_option>0 and not config.diagnostic_lapse_rate and
                upper_air):
            index=config.upper_temperature_option-1
            if index>=len(upper_air): raise ValueError('IUPT exceeds the UPDAT station count')
            end=timestamp+timedelta(seconds=config.timestep_seconds,hours=config.base_timezone)
            temperatures=upper_air[index].temperature_faces_at(end,(0.,config.lapse_height_m))
            gamma=(temperatures[1]-temperatures[0])/config.lapse_height_m
            point=np.asarray(upper_points[index])-np.array(
                [config.grid.x_origin_km,config.grid.y_origin_km])
            ix=int(np.clip(np.floor(point[0]/config.grid.dx_km),0,config.grid.nx-1))
            iy=int(np.clip(np.floor(point[1]/config.grid.dy_km),0,config.grid.ny-1))
            changes['diagnostic_lapse_rate']=np.full_like(step.surface_temperature,gamma,dtype=float)
            changes['diagnostic_lapse_station_index']=(iy,ix)
            changes['diagnostic_lapse_data']=upper_air[index]
            changes['diagnostic_lapse_timestamp']=end
        if sea_records:
            changes['observed_water_mixing_height']=changes_water_mixing
            observed_water=nearest_sea_data(
                sea_records,sea_points,[item[0] for item in sea_locations],config.grid)
            if config.water_temperature_mode==2:
                if step.overwater_data is None:
                    raise ValueError('ITWPROG=2 requires prognostic offshore data')
                wave_period,wave_height=nearest_sea_waves(
                    sea_records,sea_points,config.grid)
                changes['overwater_data']=replace(step.overwater_data,
                    wave_period_s=wave_period,wave_height_m=wave_height)
            else:
                changes['overwater_data']=observed_water
        if config.itprog in {0,1}:
            changes['surface_temperature']=observed_surface_temperature
            pressure=np.asarray(record.pressure_mb,dtype=np.float32)[nearest]
            changes['surface_pressure_mb']=pressure
            changes['surface_density']=density_from_pressure_temperature(
                pressure,changes['surface_temperature'])
        if config.irhprog==0:
            water_min=config.water_temperature_category_min
            water_max=config.water_temperature_category_max
            sea_humidity=np.asarray([item.relative_humidity for item in sea_records])
            sea_humidity=np.where(sea_humidity<9998.99,sea_humidity,100.)
            sea_humidity=np.trunc(sea_humidity)
            humidity_values=np.concatenate((np.asarray(raw_record.relative_humidity),sea_humidity))
            humidity_points=np.vstack((points,sea_points))
            humidity=interpolate_surface_scalar(humidity_values,humidity_points,config.grid,
                radius_km=config.temperature_radius_km,maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power,
                target_water=(landuse>=water_min)&(landuse<=water_max),
                station_water=np.r_[np.zeros(len(points),dtype=bool),np.ones(len(sea),dtype=bool)],
                empty_fallback=(80.,100.))
            changes['surface_relative_humidity']=np.trunc(humidity).astype(np.int32)
        if sea and config.water_temperature_mode==0:
            below=np.array([getattr(item,'gradient_below_k_m',9999.) for item in sea_records])
            above=np.array([getattr(item,'gradient_above_k_m',9999.) for item in sea_records])
            below=np.where(below<9998.,below,config.water_temperature_gradient_below)
            above=np.where(above<9998.,above,config.water_temperature_gradient_above)
            changes['water_gradient_below']=interpolate_surface_scalar(below,sea_points,config.grid,
                radius_km=config.temperature_radius_km,maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power)
            changes['water_gradient_above']=interpolate_surface_scalar(above,sea_points,config.grid,
                radius_km=config.temperature_radius_km,maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power)
            changes['water_mixing_gradient_above']=grid_water_mixing_gradient(
                [getattr(item,'gradient_above_k_m',9999.) for item in sea_records],
                sea_points,config.grid,
                config.water_temperature_gradient_above)
        if config.itprog==0:
            changes.update(temperature=observed_profile_temperature[1:],
                           observed_temperature_faces=observed_temperature_faces,
                           profile_temperature=observed_profile_temperature,
                           profile_height=observed_profile_height)
        precip_record=(None if precipitation is None else
                       precipitation.at_begin(timestamp))
        if config.precipitation_station_count==0:
            changes['precipitation_code']=np.zeros_like(step.precipitation_code,dtype=np.int32)
        elif (config.precipitation_station_count>0 and
              not (config.run_type==0 and config.diagnostic_lapse_rate)):
            station_code=np.asarray(record.precipitation_code)
            changes['precipitation_code']=np.where(station_code[nearest]<9999,
                                                   station_code[nearest],0).astype(np.int32)
            changes['precipitation_rate']=grid_precipitation(precip_record.rate_mm_h,precip_points,
                config.grid,config.precipitation_interpolation,config.precipitation_radius_km,
                config.precipitation_cutoff)
        if config.cloud_method in {1,2}:
            external=None if config.cloud_method==1 else step.cloud_fraction
            cloud,ceiling=surface_cloud_fields(record,nearest,external)
            changes.update(cloud_fraction=cloud,ceiling_height=ceiling.astype(float)*30.48)
        return replace(step,**changes)
    transform.uses_physical_context=True
    return transform,nearest

def run(config: CalmetConfig) -> Path:
    """Run a supported CALMET diagnostic/objective-analysis path."""
    faces=_validate(config)
    meteorology=((Path(config.meteorology),) if isinstance(config.meteorology,(str,Path)) else
                 tuple(Path(path) for path in config.meteorology))
    if not meteorology: raise ValueError('meteorology file list is empty')
    for path in meteorology:
        if not path.is_file(): raise FileNotFoundError(path)
    geophysical=Path(config.geophysical)
    if not geophysical.is_file(): raise FileNotFoundError(geophysical)
    start,end=_stamp(config.start),_stamp(config.end)
    if end<=start: raise ValueError('CALMET end must be later than start')
    output=_output_path(config); output.parent.mkdir(parents=True,exist_ok=True)
    with (_meteorology_input(config,meteorology,output.parent,start,end) as meteorology_input,
          _geophysical_dataset(geophysical,output.parent) as geo):
        met,step_provider=meteorology_input
        target_shape=(config.grid.ny,config.grid.nx)
        if geo['terrain_height'].shape!=target_shape:
            raise ValueError('geophysical grid does not match the configured CALMET grid')
        _validate_geophysical_grid(geo,config.grid)
        times=[(dataset_timestamp(met,index) if step_provider is not None else
                _prognostic_local_timestamp(met,index,config.base_timezone))
               for index in range(met['time_text'].shape[0])]
        source_step=config.prognostic_timestep_seconds
        timeline_step=(config.timestep_seconds if step_provider is not None else source_step)
        if any((right-left).total_seconds()!=timeline_step for left,right in zip(times[:-1],times[1:])):
            raise ValueError('CALWRF record spacing does not match configured ISTEPPGS')
        duration=(end-start).total_seconds()
        if duration%config.timestep_seconds:
            raise ValueError('CALMET run duration must be an integer number of NSECDT timesteps')
        if not times or start<times[0] or end>times[-1]:
            raise ValueError('CALWRF data do not bracket the requested CALMET period')
        indices=None; time_samples=None
        if config.timestep_seconds==timeline_step:
            indices=[index for index,time in enumerate(times) if start<time<=end]
            if not indices or indices[0]==0 or times[indices[0]-1]!=start or times[indices[-1]]!=end:
                raise ValueError('CALWRF data must contain every hourly CALMET boundary')
        else:
            time_samples=[]; right=1
            boundary=start+timedelta(seconds=config.timestep_seconds)
            while boundary<=end:
                while right<len(times) and times[right]<boundary: right+=1
                if right>=len(times): raise ValueError('CALWRF data end before the requested CALMET period')
                left=right-1
                fraction=(boundary-times[left]).total_seconds()/timeline_step
                time_samples.append((left,right,fraction,boundary))
                boundary+=timedelta(seconds=config.timestep_seconds)
        geo_fields={name:np.asarray(geo[name][:]) for name in
                    ('terrain_height','dominant_landuse','roughness_length','albedo','bowen_ratio',
                     'soil_heat_flux_parameter','anthropogenic_heat_flux')}
        diagnostic=None
        initial_guess=(InitialGuessCalmetSeries(config.initial_guess_calmet_files)
                       if config.use_initial_guess_calmet else None)
        if initial_guess is not None and not initial_guess.includes_w_temperature:
            raise ValueError('IGFMET requires IGFDAT written with LCALGRD=T')
        water_range=(int(getattr(geo,'water_category_min',50)),int(getattr(geo,'water_category_max',55)))
        if water_range[0]>water_range[1]: raise ValueError('geophysical water-category range is invalid')
        if config.distance_to_coast_output is not None:
            _write_distance_to_coast_grid(config.distance_to_coast_output,
                geo_fields['dominant_landuse'],config.grid,water_range)
        category_text=str(getattr(geo,'landuse_categories',''))
        landuse_category_count=len([item for item in category_text.split(',') if item]) or 14
        latitude,longitude=target_latitude_longitude(config.grid)
        options=dict(base_timezone=config.base_timezone,terrain_radius_km=config.terrain_radius_km,
            prognostic_ocean_category=config.prognostic_ocean_category,
                     minimum_average_radius=config.mnmdav,half_average_angle=config.half_average_angle,
                     critical_froude=config.critical_froude,smoothing_passes=config.smoothing_passes,
                     use_froude=bool(config.ifradj),use_slope=bool(config.islope),use_obrien=bool(config.iobr),
                     use_kinematic=bool(config.ikine),kinematic_alpha=config.kinematic_alpha,
                     average_temperature=bool(config.iavet),average_mixing_height=bool(config.iavezi),
                     mixing_wind_level=config.mixing_wind_level,
                     mixing_method=config.imixh,minimum_lapse=config.minimum_lapse_rate,
                     lapse_depth_m=config.mixing_lapse_depth_m,
                     minimum_mixing_height_m=config.minimum_mixing_height_m,
                     maximum_mixing_height_m=config.maximum_mixing_height_m,
                     minimum_water_mixing_height_m=config.minimum_water_mixing_height_m,
                     maximum_water_mixing_height_m=config.maximum_water_mixing_height_m,
                     land_flux_threshold=config.land_flux_threshold,constb=config.constb,
                     conste=config.conste,constn=config.constn,constw=config.constw,
                     coriolis_value=config.coriolis,convective_relaxation=bool(config.convective_relaxation),
                     convective_relaxation_time_s=config.convective_relaxation_time_s,
                     precipitation_interpolation=config.precipitation_interpolation,
                     precipitation_radius_km=config.precipitation_radius_km,
                     precipitation_cutoff=config.precipitation_cutoff,
                     turbidity_coefficient_1=config.turbidity_coefficient_1,
                     turbidity_coefficient_2=config.turbidity_coefficient_2,
                     cloudiness_coefficient_1=config.cloudiness_coefficient_1,
                     cloudiness_coefficient_2=config.cloudiness_coefficient_2,
                     net_radiation_coefficient_1=config.net_radiation_coefficient_1,
                     net_radiation_coefficient_2=config.net_radiation_coefficient_2,
                     net_radiation_coefficient_3=config.net_radiation_coefficient_3,
                     surface_temperature_option=config.surface_temperature_option,
                     upper_temperature_option=config.upper_temperature_option,
                     lapse_height_m=config.lapse_height_m,
                     temperature_mode=config.itprog,
                     water_range=water_range,
                     overwater_flux_method=config.overwater_flux_method,
                     coastal_shelf_length_km=config.coastal_shelf_length_km,
                     warm_layer=bool(config.warm_layer),
                     cool_skin=bool(config.cool_skin),
                     water_flux_threshold=config.water_flux_threshold,
                     water_temperature_mode=config.water_temperature_mode,
                     default_water_gradient_above=config.water_temperature_gradient_above,
                     cloud_method=config.cloud_method,
                     diagnostic_wind=config.iwfcod==1,
                     wind_only=(config.run_type==0 and
                                bool(config.diagnostic_lapse_rate)),
                     prognostic_observation_mode=config.iprog in {5,15},
                     prognostic_observation_rmin_km=config.minimum_station_radius_km,
                     prognostic_observation_barriers=config.barriers,
                     prognostic_observation_barrier_top_level=config.barrier_top_level,
                     prognostic_timestep_s=float(source_step),
                     timestep_s=float(config.timestep_seconds),
                     divergence_limit=config.divergence_limit,divergence_iterations=config.divergence_iterations)
        options['retain_debug_stages']=bool(any(config.wind_debug_fields[:3]) or
                                            any(config.wind_debug_fields[5:8]))
        options['retain_micrometeorology_debug']=bool(config.micrometeorology_debug)
        surface_coordinates=(); upper_coordinates=(); precipitation_coordinates=()
        surface=None; surface_points=None; nearest_surface=None
        sea=tuple(SeaData(path) for path in config.sea_files)
        _validate_sea_metadata(config,sea)
        precipitation=(load_precipitation(config.precipitation_data,
                           config.precipitation_data_format,
                           station_metadata=_precipitation_control_metadata(config),
                           datum=config.grid.datum)
                       if config.precipitation_station_count>0 else None)
        precipitation_points=None
        if precipitation is not None:
            if len(precipitation.stations)!=config.precipitation_station_count:
                raise ValueError('NPSTA does not match PRECIP.DAT station count')
            _validate_input_timezone(
                precipitation.timezone,config.base_timezone,'PRECIP.DAT')
            _validate_input_span(precipitation,start,end,'PRECIP.DAT')
            precipitation_points=station_grid_mapping(
                precipitation.stations,config.grid,precipitation.datum)[0]
            precipitation_points-=np.array(
                [config.grid.x_origin_km,config.grid.y_origin_km])
            precipitation_coordinates=precipitation_points
        if config.noobs in {0,1}:
            surface_metadata=_surface_control_metadata(config)
            if config.diagnostic_surface_wind:
                surface_stations=surface_metadata
                surface_datum=config.grid.datum
            else:
                surface=load_surface(config.surface_data,config.surface_data_format,
                                     station_metadata=surface_metadata,
                                     datum=config.grid.datum)
                _validate_input_timezone(surface.timezone,config.base_timezone,'SURF.DAT')
                _validate_input_span(surface,start,end,'SURF.DAT')
                surface_stations=surface.stations
                surface_datum=surface.datum
            upper_metadata=_upper_control_metadata(config)
            if (config.noobs==0 and not config.diagnostic_upper_wind and
                    upper_metadata is not None and
                    len(upper_metadata)!=len(config.upper_air_files)):
                raise ValueError('upper-air control-station count does not match UPDAT files')
            if config.noobs==0 and config.diagnostic_upper_wind:
                upper=()
                upper_stations=upper_metadata
            else:
                upper=(tuple(UpperAirData(path,
                                         station_metadata=(None if upper_metadata is None else upper_metadata[index]),
                                         datum=config.grid.datum)
                             for index,path in enumerate(config.upper_air_files))
                       if config.noobs==0 else ())
                upper_stations=tuple(data.station for data in upper)
            for data in upper:
                _validate_upper_span(data,start,end,config.base_timezone)
            active_diagnostic=any((config.diagnostic_surface_temperature,
                config.diagnostic_lapse_rate,config.diagnostic_initial_wind,
                config.diagnostic_surface_wind,config.diagnostic_upper_wind))
            if active_diagnostic:
                diagnostic=DiagnosticData(config.diagnostic_data,start,
                    int(duration/config.timestep_seconds),config.timestep_seconds,
                    surface_temperature=bool(config.diagnostic_surface_temperature and
                                             config.surface_temperature_option!=-1),
                    lapse_rate=bool(config.diagnostic_lapse_rate and config.noobs==0),
                    initial_wind=bool(config.diagnostic_initial_wind and config.noobs==0),
                    surface_wind=bool(config.diagnostic_surface_wind),
                    # PREPDI branches to its return before IDIOPT5 when
                    # NOOBS>=1, so the upper-wind block is present only in
                    # the full-observation path.
                    upper_wind=bool(config.diagnostic_upper_wind and config.noobs==0),
                    surface_wind_names=([item.name for item in surface_stations]+
                        [item.name for item in sea] if config.diagnostic_surface_wind else ()),
                    upper_wind_names=([item.name for item in upper_stations]
                        if config.diagnostic_upper_wind and config.noobs==0 else ()),
                    levels=faces.size-1,iextrp=config.wind_extrapolation)
            weights=(read_weight_data(config.weight_data,config.grid.nx,config.grid.ny,
                .5*(faces[:-1]+faces[1:])) if config.iprog in {3,5,13,15} and config.noobs!=2 else None)
            options['step_transform'],nearest_surface= _surface_step_transform(
                config,surface,geo,upper,precipitation,sea,weights,diagnostic,initial_guess,
                surface_stations=surface_stations,upper_stations=upper_stations)
            points,_=station_grid_mapping(surface_stations,config.grid,surface_datum)
            surface_points=points
            surface_coordinates=points-np.array([config.grid.x_origin_km,config.grid.y_origin_km])
            if upper_stations:
                upper_coordinates=np.vstack([
                    station_grid_mapping((station,),config.grid,
                        upper[index].datum if upper else config.grid.datum)[0][0]
                    for index,station in enumerate(upper_stations)
                ])-np.array([config.grid.x_origin_km,config.grid.y_origin_km])
        else:
            active_diagnostic=any((config.diagnostic_surface_temperature,
                config.diagnostic_lapse_rate,config.diagnostic_initial_wind,
                config.diagnostic_surface_wind,config.diagnostic_upper_wind))
            if active_diagnostic:
                diagnostic=DiagnosticData(config.diagnostic_data,start,
                    int(duration/config.timestep_seconds),config.timestep_seconds,
                    surface_temperature=bool(config.diagnostic_surface_temperature and
                                             config.surface_temperature_option!=-1),
                    # COMP does not call PREPDI at all for NOOBS=2.  OPENOT
                    # still requires DIAG.DAT when an IDIOPT flag is set,
                    # but no surface/upper wind records are consumed.
                    surface_wind=False,upper_wind=False,levels=faces.size-1,
                    iextrp=config.wind_extrapolation)
            if initial_guess is not None or diagnostic is not None or precipitation is not None:
                options['step_transform']=_noobs_step_transform(
                    config,initial_guess,diagnostic,precipitation,
                    precipitation_points,sea)
        if config.cloud_method==2:
            cloud=CloudData(config.cloud_data,config.grid.nx,config.grid.ny,config.cloud_data_format)
            options['cloud_provider']=cloud.at_end
        if step_provider is None:
            options['mapper']=PrognosticMapper.build(
                np.asarray(met['latitude'][:]),np.asarray(met['longitude'][:]),config.grid,
                validate_domain=True)
            options['precipitation_radius_km']=_prognostic_precipitation_radius(
                options['mapper'],config.precipitation_radius_km,
                config.precipitation_station_count)
        else:
            options['step_provider']=step_provider
        options['terrain_context']=terrain_extrema(geo_fields['terrain_height'],config.terrain_radius_km,
                                                   config.grid.dx_km*1000.,config.grid.dy_km*1000.)
        options['target_coordinates']=(latitude,longitude)
        hours=iter_hours(met,config.grid,faces,geo_fields,time_indices=indices,time_samples=time_samples,**options)
        cloud_writer=(CloudDataWriter(config.cloud_data,config.grid.nx,config.grid.ny,
                                      config.cloud_data_format)
                      if config.cloud_output else None)
        def append_cloud(hour):
            if cloud_writer is not None:
                cloud_writer.append(hour.timestamp,
                    hour.timestamp+timedelta(seconds=config.timestep_seconds),
                    hour.step1.cloud_fraction)
        def open_listing():
            if not config.print_fields and not config.micrometeorology_debug:
                return None
            target=(Path(config.listing_output) if config.listing_output is not None
                    else output.with_suffix('.lst'))
            return CalmetListingWriter(target,config,
                geo_fields['dominant_landuse'],water_range)
        def open_wind_debug():
            if not config.wind_debug_output:
                return []
            specifications=(
                (5,'kinematic_stage',config.kinematic_wind_debug_output,'test.kin'),
                (6,'froude_stage',config.froude_wind_debug_output,'test.frd'),
                (7,'slope_stage',config.slope_wind_debug_output,'test.slp'),
                (8,None,config.final_wind_debug_output,'test.out'),
            )
            writers=[]
            for index,stage,path,default in specifications:
                if config.wind_debug_fields[index]:
                    target=Path(path) if path is not None else output.parent/default
                    writer=FinalWindDebugWriter(target,config.grid.nx,config.grid.ny,
                        faces.size-1,config.grid.dx_km,config.grid.dy_km)
                    writers.append((stage,writer))
            return writers
        def append_wind_debug(writers,hour):
            for stage,writer in writers:
                if stage is None:
                    writer.append(hour)
                else:
                    fields=getattr(hour.wind,stage)
                    if fields is None:
                        raise RuntimeError(f'missing retained wind debug stage {stage}')
                    writer.append_fields(hour.timestamp,*fields)
        def open_wind_print():
            if not any(config.wind_debug_fields[:3]) and not any(
                    config.wind_debug_fields[4:9]):
                return None
            target=(Path(config.wind_print_output) if config.wind_print_output is not None
                    else output.parent/'test.prt')
            return WindPrintWriter(target,config.wind_debug_fields,
                                   config.wind_debug_levels)
        if not config.save_output:
            # LSAVE only controls gridded disk output; CALMET still executes
            # COMP and advances every stateful hour unless ITEST stops after SETUP.
            listing=open_listing()
            wind_debug=open_wind_debug()
            wind_print=open_wind_print()
            try:
                if config.test_mode==2:
                    for hour in hours:
                        append_cloud(hour)
                        if listing is not None:
                            sun=solar_at_step_midpoint(hour.timestamp,
                                config.timestep_seconds,config.base_timezone,
                                latitude,longitude)
                            listing.append(hour,sun)
                        append_wind_debug(wind_debug,hour)
                        if wind_print is not None: wind_print.append(hour)
            finally:
                if cloud_writer is not None: cloud_writer.close()
                if listing is not None: listing.close()
                for _,debug_writer in wind_debug: debug_writer.close()
                if wind_print is not None: wind_print.close()
            return output
        mesopac=config.legacy_output_format==2
        legacy=output.suffix.lower() not in {'.nc','.nc4','.cdf'}
        if mesopac:
            writer=MesopacWriter(output,config.grid,faces,geo,start,end,
                timezone_offset=config.base_timezone,
                surface_coordinates_m=np.asarray(surface_coordinates)*1000.,
                upper_coordinates_m=np.asarray(upper_coordinates)*1000.,
                nearest_surface_station=(None if nearest_surface is None else nearest_surface+1),
                water_range=water_range)
        else:
            writer=(LegacyCalmetWriter(output,config.grid,faces,geo,start,end,
                    control_lines=config.control_lines,
                    base_timezone=(config.base_timezone_label or
                                   _timezone_label(config.base_timezone)),
                    include_w_temperature=config.include_w_temperature,water_range=water_range,
                    landuse_category_count=landuse_category_count,run_type=config.run_type,
                    timestep_seconds=config.timestep_seconds,
                    precipitation_station_count=config.precipitation_station_count,
                    surface_coordinates_km=surface_coordinates,upper_coordinates_km=upper_coordinates,
                    precipitation_coordinates_km=precipitation_coordinates,
                    nearest_surface_station=(None if nearest_surface is None else
                                             nearest_surface+1),iwfcod=config.iwfcod,
                    sea_station_count=len(sea))
                if legacy else CalmetNetCDFWriter(output,config.grid,faces,geo,compression=config.compression,
                                                   water_range=water_range,run_type=config.run_type))
        auxiliary=(AuxiliaryCalmetWriter(str(output)+'.aux',config.grid,faces,start,end,
                    control_lines=config.control_lines,
                    base_timezone=(config.base_timezone_label or
                                   _timezone_label(config.base_timezone)),
                    available='qcloud' in met.variables,
                    timestep_seconds=config.timestep_seconds)
                   if legacy and not mesopac else None)
        with ExitStack() as outputs:
            outputs.enter_context(writer)
            if cloud_writer is not None: outputs.enter_context(cloud_writer)
            if auxiliary is not None: outputs.enter_context(auxiliary)
            listing=open_listing()
            if listing is not None: outputs.enter_context(listing)
            wind_debug=open_wind_debug()
            for _,debug_writer in wind_debug: outputs.enter_context(debug_writer)
            wind_print=open_wind_print()
            if wind_print is not None: outputs.enter_context(wind_print)
            if config.test_mode==2:
                for hour in hours:
                    append_cloud(hour)
                    sun=solar_at_step_midpoint(hour.timestamp,config.timestep_seconds,config.base_timezone,latitude,longitude)
                    if listing is not None: listing.append(hour,sun)
                    append_wind_debug(wind_debug,hour)
                    if wind_print is not None: wind_print.append(hour)
                    if mesopac:
                        if surface is None:
                            station_temperature=np.empty(0,dtype=np.float32)
                            station_radiation=np.empty(0,dtype=np.float32)
                            station_humidity=np.empty(0,dtype=np.int32)
                            station_precipitation=np.empty(0,dtype=np.int32)
                        else:
                            record=fill_missing_surface(surface.at_begin(hour.timestamp),surface_points,
                                cloud_method=config.cloud_method,itprog=config.itprog)
                            station_temperature=record.temperature_k
                            station_humidity=record.relative_humidity
                            station_precipitation=record.precipitation_code
                            cell=np.trunc(np.asarray(surface_coordinates)/np.array(
                                [config.grid.dx_km,config.grid.dy_km])).astype(int)
                            ix=np.clip(cell[:,0],0,config.grid.nx-1)
                            iy=np.clip(cell[:,1],0,config.grid.ny-1)
                            station_radiation=hour.surface.shortwave_flux[iy,ix]
                        writer.append(hour,sun,station_temperature_k=station_temperature,
                            station_shortwave_flux=station_radiation,
                            station_relative_humidity=station_humidity,
                            station_precipitation_code=station_precipitation)
                    elif legacy: writer.append(hour,sun)
                    else: writer.append(hour,sun,np.asarray(geo['dominant_landuse'][:]))
                    if auxiliary is not None: auxiliary.append(hour)
    return output
