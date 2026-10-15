"""Public orchestration for the NumPy CALMET diagnostic solver."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timedelta
from contextlib import ExitStack,contextmanager
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset,Variable
from .datetime_utils import (exact_integer,format_timezone_label,
                             parse_timezone_label)
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
from .calmet_solver import (HourFields,PrognosticMapper,
                            SourceThermalState,Step1Fields,StepTransformContext,
                            _domain_mean,build_mm4_step_fields,
                            _prognostic_local_timestamp,build_prog_step_fields,
                            DatasetLike,dataset_timestamp,iter_hours,
                            solar_at_step_midpoint,target_latitude_longitude)
from .cloud_data import CloudData,CloudDataWriter
from .coare import (WarmLayerState,distance_to_coast,downward_longwave,
                    initial_warm_layer_state)
from .diagnostic_data import DiagnosticData,DiagnosticRecord
from .igf_data import InitialGuessCalmetSeries,interpolate_initial_guess
from .projections import grid_convergence,unproject_geographic
from .precip_data import (PrecipitationData,PrecipitationRecord,
                          grid_precipitation,load_precipitation)
from .prog_data import ProgData,ProgRecord
from .mm4_data import MM4Data,MM4Header,MM4Record
from .sea_data import (GriddedSeaData,SeaData,SeaRecord,
                       grid_sea_mixing_height,grid_water_mixing_gradient,
                       nearest_sea_data,nearest_sea_waves)
from .surface_data import (SurfaceData,SurfaceRecord,SurfaceStation,_near_upper_air,_nearest_station_grid,fill_missing_surface,load_surface,station_grid_mapping,
                           _wind_components,component_wind_profiles,interpolate_surface_scalar,surface_cloud_fields,
                           wind_profiles)
from .upper_air import (UpperAirData,UpperAirStation,grid_temperature_faces,
                        mixing_temperature_profile)
from .weight_data import read_weight_data
from .wind_debug import FinalWindDebugWriter,WindPrintWriter


if TYPE_CHECKING:
    NetCDFVariable: TypeAlias = Variable[object]
else:
    # netCDF4's stubs expose Variable as generic, but its C-extension class is
    # not subscriptable at runtime.  Keeping the runtime alias concrete also
    # lets typing.get_type_hints() resolve annotations in this module.
    NetCDFVariable: TypeAlias = Variable


Float32Array: TypeAlias = npt.NDArray[np.float32]
Float64Array: TypeAlias = npt.NDArray[np.float64]
IntpArray: TypeAlias = npt.NDArray[np.intp]
Int32Array: TypeAlias = npt.NDArray[np.int32]
ByteArray: TypeAlias = npt.NDArray[np.bytes_]
MeteorologyProvider: TypeAlias = Callable[[datetime],Step1Fields]
StepTransform: TypeAlias = Callable[[Step1Fields,datetime],Step1Fields]
PhysicalStepTransform: TypeAlias = Callable[
    [Step1Fields,datetime,StepTransformContext | None],Step1Fields
]


@contextmanager
def _single_meteorology_dataset(
    path: str | Path,temporary_directory: str | Path
) -> Iterator[Dataset]:
    """Open one canonical NetCDF or transparently stage one formatted M3D.

    Args:
        path: Canonical CALWRF NetCDF or traditional formatted ``3D.DAT``
            input path.
        temporary_directory: Directory in which to create a temporary NetCDF
            conversion when ``path`` is a formatted M3D file.

    Yields:
        Open read-only :class:`netCDF4.Dataset`. Variables retain their stored
        dimensions and dtypes; meteorological model fields are normally
        indexed as ``(time, level, y, x)`` or ``(time, y, x)``.

    Raises:
        OSError: If the source cannot be read or the staged NetCDF cannot be
            created.
        ValueError: If a traditional M3D source is malformed.

    Notes:
        A staged NetCDF file is deleted when the context exits, including
        exceptional exits.
    """
    with Path(path).open('rb') as stream: signature: bytes=stream.read(16)
    if signature.startswith(b'3D.DAT'):
        from .legacy_m3d import M3DReader
        with tempfile.NamedTemporaryFile(dir=temporary_directory,suffix='.nc',delete=False) as handle:
            staged: Path=Path(handle.name)
        try:
            M3DReader(path).to_netcdf(staged)
            with Dataset(staged) as dataset: yield dataset
        finally:
            staged.unlink(missing_ok=True)
    else:
        with Dataset(path) as dataset: yield dataset


def _validate_mm4_static_headers(sources: Sequence[MM4Data]) -> None:
    """Require every native MM4 file to describe the same source grid.

    Args:
        sources: Nonempty sequence of native ``MM4Data`` readers. Their grid
            dimensions, sigma levels, coordinates, terrain, land use, and top
            pressure are compared with the first source.

    Raises:
        ValueError: If any later source has incompatible scalar metadata or
            an array with different shape or values.
    """
    reference: MM4Data=sources[0]
    expected: MM4Header=reference.header
    scalar_fields: dict[str,object]={
        'model grid dimensions':(expected.model_nx,expected.model_ny),
        'subdomain origin':(expected.i_start,expected.j_start),
        'top pressure':expected.top_pressure_mb,
    }
    array_fields: dict[str,npt.NDArray[np.generic]]={
        'sigma levels':expected.sigma_levels,
        'latitude grid':expected.latitude,
        'longitude grid':expected.longitude,
        'terrain heights':expected.terrain_height_m,
        'land-use categories':expected.landuse,
    }
    for source in sources[1:]:
        header: MM4Header=source.header
        actual_scalars: dict[str,object]={
            'model grid dimensions':(header.model_nx,header.model_ny),
            'subdomain origin':(header.i_start,header.j_start),
            'top pressure':header.top_pressure_mb,
        }
        actual_arrays: dict[str,npt.NDArray[np.generic]]={
            'sigma levels':header.sigma_levels,
            'latitude grid':header.latitude,
            'longitude grid':header.longitude,
            'terrain heights':header.terrain_height_m,
            'land-use categories':header.landuse,
        }
        differences: list[str]=[
            name for name,expected_value in scalar_fields.items()
            if actual_scalars[name]!=expected_value
        ]
        for name,expected_values in array_fields.items():
            actual_values=actual_arrays[name]
            if (actual_values.shape!=expected_values.shape or
                    not np.array_equal(actual_values,expected_values,equal_nan=True)):
                differences.append(name)
        if differences:
            raise ValueError(
                f'native MM4 static header in {source.path} is incompatible with '
                f'{reference.path}: '+', '.join(differences))


def _same_mm4_record(left: MM4Record,right: MM4Record) -> bool:
    """Return whether overlapping native MM4 records contain identical data.

    Args:
        left: First native MM4 hourly record.
        right: Second native MM4 hourly record for the same timestamp.

    Returns:
        ``True`` when all surface and three-dimensional meteorological arrays
        have identical shapes and values, treating collocated NaNs as equal.
    """
    fields: tuple[str,...]=('sea_level_pressure_mb','precipitation_mm_h','snow_code','pressure_mb',
            'height_agl_m','temperature_k','relative_humidity','wind_direction_deg',
            'wind_speed_m_s')
    return all(np.array_equal(getattr(left,name),getattr(right,name),equal_nan=True)
               for name in fields)


def _observation_only_meteorology(
    config: CalmetConfig,
    start: datetime,
    end: datetime,
) -> tuple[MeteorologyInput,MeteorologyProvider]:
    """Create the timeline and inert Step-1 scaffold for ``IPROG=0``.

    Args:
        config: Pure-observation CALMET configuration. Its grid, vertical
            faces, and computational time step define every scaffold array.
        start: Inclusive local beginning of the CALMET integration.
        end: Inclusive interval-ending boundary of the integration.

    Returns:
        Pair ``(dataset, provider)``. ``dataset`` contains only a local
        ``time_text`` boundary axis. ``provider`` returns physically benign
        standard-atmosphere ``float32`` fields with horizontal shape
        ``(ny, nx)`` and vertical shape ``(nz, ny, nx)``. The existing
        surface/upper-air transform replaces winds, temperature, pressure,
        surface humidity, and observed cloud fields before they enter the
        solver.

    Raises:
        ValueError: If the solver requests a boundary outside the configured
            pure-observation timeline.

    Notes:
        CALMET still represents one diagnostic step as :class:`Step1Fields`,
        even when no prognostic file exists. The scaffold provides only that
        structural contract; it is not a hidden meteorological background.
    """
    boundaries: list[datetime]=[start]
    current: datetime=start+timedelta(seconds=config.timestep_seconds)
    while current<=end:
        boundaries.append(current)
        current+=timedelta(seconds=config.timestep_seconds)
    text: ByteArray=np.array(
        [list(value.strftime('%Y-%m-%d_%H:%M:%S')) for value in boundaries],
        dtype='S1',
    )
    available: frozenset[datetime]=frozenset(boundaries[1:])
    ny: int=config.grid.ny
    nx: int=config.grid.nx
    faces: Float32Array=np.asarray(config.z_faces_m,dtype=np.float32)
    centers: Float32Array=np.asarray(.5*(faces[:-1]+faces[1:]),dtype=np.float32)
    horizontal_shape: tuple[int,int]=(ny,nx)
    vertical_shape: tuple[int,int,int]=(centers.size,ny,nx)
    profile_shape: tuple[int,int,int]=(centers.size+1,ny,nx)

    def provider(ending: datetime) -> Step1Fields:
        """Return one observation-replaced scaffold at ``ending``.

        Args:
            ending: Local interval-ending timestamp requested by the solver.

        Returns:
            Fresh ``float32`` :class:`Step1Fields` arrays for the target grid.

        Raises:
            ValueError: If ``ending`` is not a configured boundary.
        """
        if ending not in available:
            raise ValueError(
                f'pure-observation timeline has no boundary for {ending}'
            )
        surface_temperature: Float32Array=np.full(
            horizontal_shape,np.float32(288.15),dtype=np.float32
        )
        layer_temperature: Float32Array=np.broadcast_to(
            np.asarray(288.15-.0065*centers,dtype=np.float32)[:,None,None],
            vertical_shape,
        ).copy()
        profile_temperature: Float32Array=np.empty(
            profile_shape,dtype=np.float32
        )
        profile_temperature[0]=surface_temperature
        profile_temperature[1:]=layer_temperature
        profile_height: Float32Array=np.broadcast_to(
            np.concatenate((np.array([0.],dtype=np.float32),centers))[:,None,None],
            profile_shape,
        ).copy()
        surface_pressure: Float32Array=np.full(
            horizontal_shape,np.float32(1013.25),dtype=np.float32
        )
        density: Float32Array=density_from_pressure_temperature(
            surface_pressure,surface_temperature
        )
        zeros_2d: Float32Array=np.zeros(horizontal_shape,dtype=np.float32)
        zeros_3d: Float32Array=np.zeros(vertical_shape,dtype=np.float32)
        humidity_2d: Float32Array=np.full(
            horizontal_shape,np.float32(50.),dtype=np.float32
        )
        humidity_3d: Float32Array=np.full(
            vertical_shape,np.float32(50.),dtype=np.float32
        )
        return Step1Fields(
            zeros_3d.copy(),zeros_3d.copy(),layer_temperature,
            surface_temperature,surface_pressure,density,humidity_3d,
            np.full(horizontal_shape,np.float32(500.),dtype=np.float32),
            zeros_2d.copy(),
            np.full(horizontal_shape,np.float32(.1),dtype=np.float32),
            zeros_2d.copy(),zeros_2d.copy(),profile_temperature,
            profile_height,humidity_2d,zeros_2d.copy(),
            np.zeros(horizontal_shape,dtype=np.int32),
        )

    return {'time_text':text},provider


@contextmanager
def _meteorology_input(
    config: CalmetConfig,
    paths: Sequence[str | Path],
    temporary_directory: str | Path,
    start: datetime,
    end: datetime,
) -> Iterator[
    tuple[
        MeteorologyInput,
        MeteorologyProvider | None,
    ]
]:
    """Open canonical meteorology or adapt native MM4/PROG records.

    Args:
        config: Validated CALMET configuration controlling the prognostic
            format, model grid, time steps, and interpolation options.
        paths: Ordered meteorological input paths. An empty sequence selects
            the pure-observation ``IPROG=0`` timeline.
        temporary_directory: Directory used for temporary M3D-to-NetCDF
            conversions.
        start: Inclusive local beginning of the CALMET integration.
        end: Inclusive meteorological boundary at the end of the integration.

    Yields:
        Pair ``(dataset, provider)``. ``dataset`` is a NetCDF-like object with
        a ``time_text`` variable. ``provider`` is ``None`` for canonical
        CALWRF input, or a callable returning a ``Step1Fields`` object for an
        interval-ending datetime when native MM4 or PROG input is adapted.
        Step fields use ``float32`` and horizontal ``(y, x)`` or vertical
        ``(level, y, x)`` layouts.

    Raises:
        ValueError: If native records are inconsistent, have unsupported time
            spacing, conflict at an overlap, or do not cover the requested
            interval.
        OSError: If an input cannot be opened or a temporary conversion fails.
    """
    dataset: MeteorologyInput
    if not paths:
        if config.iprog!=0:
            raise ValueError(f'IPROG={config.iprog} requires meteorology input')
        yield _observation_only_meteorology(config,start,end)
        return
    if config.iprog in {3,4,5}:
        with Path(paths[0]).open('rb') as stream: sample: bytes=stream.read(256)
        native_mm4: bool=(not sample.startswith((b'CDF',b'\x89HDF',b'3D.DAT')) and
                    len(sample.splitlines())>=2 and
                    re.fullmatch(rb'\s*\d{8}\s*\d+\s+\d+\s+\d+\s+\d+\s*[-+0-9.]*\s*',
                                 sample.splitlines()[1]) is not None)
        if native_mm4:
            if config.prognostic_timestep_seconds!=3600:
                raise ValueError(
                    'native MM4.DAT records are hourly; ISTEPPGS must be 3600')
            sources: list[MM4Data]=[MM4Data(path) for path in paths]
            reference: MM4Header=sources[0].header
            _validate_mm4_static_headers(sources)
            for source in sources:
                earliest: datetime=min(record.timestamp for record in source.records)
                if source.header.start>earliest:
                    raise ValueError(
                        f'native MM4 header in {source.path} begins at '
                        f'{source.header.start}, after its earliest record {earliest}')
            boundaries: list[datetime]=[]
            current: datetime=start+timedelta(seconds=config.timestep_seconds)
            while current<=end:
                boundaries.append(current); current+=timedelta(seconds=config.timestep_seconds)
            targets: list[datetime]=[]
            for boundary in boundaries:
                target=(boundary if boundary.minute==0 and boundary.second==0 else
                        boundary.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
                if not targets or target!=targets[-1]: targets.append(target)
            # Index all files by their actual timestamps.  Overlap is common
            # at file boundaries, so identical records are deduplicated while
            # contradictory records are rejected explicitly.
            record_index: dict[datetime,tuple[MM4Data,MM4Record]]={}
            record_origins: dict[datetime,Path]={}
            for source in sources:
                for record in source.records:
                    local: datetime=record.timestamp-timedelta(hours=config.base_timezone)
                    if local in record_index:
                        _,existing=record_index[local]
                        if not _same_mm4_record(existing,record):
                            raise ValueError(
                                f'conflicting native MM4 records for {local} in '
                                f'{record_origins[local]} and {source.path}')
                        continue
                    record_index[local]=(source,record)
                    record_origins[local]=source.path
            missing: list[datetime]=[
                target for target in targets if target not in record_index
            ]
            if missing:
                values: str=', '.join(
                    value.isoformat(sep=' ') for value in missing
                )
                raise ValueError(f'native MM4 files do not cover requested time(s): {values}')
            selected: dict[
                datetime,tuple[MM4Data,MM4Record] | ProgRecord
            ]={
                target:record_index[target] for target in targets
            }
            target_positions: dict[datetime,int]={
                target:index for index,target in enumerate(targets)
            }
            native_mapper: PrognosticMapper=PrognosticMapper.build(reference.latitude,reference.longitude,
                                                  config.grid,validate_domain=True)
            precipitation_radius: float=_prognostic_precipitation_radius(
                native_mapper,config.precipitation_radius_km,
                config.precipitation_station_count)
            all_times: list[datetime]=[start,*boundaries]
            text: ByteArray=np.array([list(value.strftime('%Y-%m-%d_%H:%M:%S'))
                           for value in all_times],dtype='S1')
            state: SourceThermalState | None=None
            cache: dict[datetime,Step1Fields]={}
            built: int=-1
            def mm4_provider(ending: datetime) -> Step1Fields:
                """Materialize the native MM4 step ending at ``ending``.

                Args:
                    ending: Naive local interval-ending datetime on the
                        CALMET computational timeline.

                Returns:
                    ``Step1Fields`` whose two-dimensional fields have shape
                    ``(y, x)`` and three-dimensional fields have shape
                    ``(level, y, x)``. Physical arrays use ``float32`` and SI
                    units except explicitly named pressure and precipitation
                    fields.

                Raises:
                    ValueError: If no indexed MM4 record covers the requested
                        ending time.

                Notes:
                    Earlier hourly records are built first so the mutable
                    thermal state and cache evolve deterministically.
                """
                nonlocal state,built
                target: datetime=(ending if ending.minute==0 and ending.second==0 else
                        ending.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
                try: requested: int=target_positions[target]
                except KeyError as error:
                    raise ValueError(f'MM4.DAT has no record for {target}') from error
                # Materialize intervening hours so the thermal state is
                # deterministic even when callers request steps out of order.
                while built<requested:
                    built+=1; build_target: datetime=targets[built]
                    source,record=cast(
                        tuple[MM4Data,MM4Record],selected[build_target])
                    cache[build_target]=build_mm4_step_fields(source,record,config.grid,
                        config.z_faces_m,
                        cloud_method=cast(Literal[3,4],config.cloud_method),
                        precipitation_interpolation=cast(
                            Literal[1,2,3],config.precipitation_interpolation),
                        precipitation_radius_km=precipitation_radius,
                        precipitation_cutoff=config.precipitation_cutoff,
                        base_timezone=config.base_timezone,thermal_state=state,
                        prognostic_observation_rmin_km=(config.minimum_station_radius_km
                                                        if config.iprog==5 else None),
                        prognostic_observation_barriers=config.barriers,
                        prognostic_observation_barrier_top_level=cast(
                            int,config.barrier_top_level),
                        prognostic_ocean_category=config.prognostic_ocean_category)
                    state=cache[build_target].thermal_state
                return cache[target]
            yield {'time_text':text},mm4_provider
            return
    if config.iprog not in {1,2}:
        with _meteorology_dataset(paths,temporary_directory) as dataset:
            yield dataset,None
        return
    if len(paths)!=1:
        raise ValueError('IPROG=1/2 requires exactly one PRGDAT file')
    if config.prognostic_timestep_seconds!=3600:
        raise ValueError('legacy PROG.DAT records are hourly; ISTEPPGS must be 3600')
    prog: ProgData=ProgData(paths[0])
    boundaries=[]
    current=start+timedelta(seconds=config.timestep_seconds)
    while current<=end:
        boundaries.append(current)
        current+=timedelta(seconds=config.timestep_seconds)
    targets=[]
    for boundary in boundaries:
        target=(boundary if boundary.minute==0 and boundary.second==0 else
                boundary.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
        if not targets or target!=targets[-1]:
            targets.append(target)
    records: tuple[ProgRecord,...]=prog.records_for_times(targets)
    record_by_target: dict[datetime,ProgRecord]=dict(zip(targets,records))
    selected={}
    for boundary in boundaries:
        target=(boundary if boundary.minute==0 and boundary.second==0 else
                boundary.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1))
        selected[boundary]=record_by_target[target]
    all_times=[start,*boundaries]
    text=np.array(
        [list(value.strftime('%Y-%m-%d_%H:%M:%S')) for value in all_times],
        dtype='S1',
    )
    dataset={'time_text':text}
    def prog_provider(ending: datetime) -> Step1Fields:
        """Build a target-grid step from the selected PROG record.

        Args:
            ending: Naive local interval-ending datetime on the CALMET
                computational timeline.

        Returns:
            ``Step1Fields`` mapped to the CALMET grid, with horizontal fields
            shaped ``(y, x)``, vertical fields shaped ``(level, y, x)``, and
            physical arrays represented in ``float32``.

        Raises:
            ValueError: If the requested ending time has no selected PROG
                record.
        """
        try: record: ProgRecord=cast(ProgRecord,selected[ending])
        except KeyError as error: raise ValueError(f'PROG.DAT has no record for {ending}') from error
        return build_prog_step_fields(prog,record.hour,config.grid,config.z_faces_m,record=record)
    yield dataset,prog_provider


class _ConcatenatedVariable:
    """Read-only time-axis view over equivalent variables from several files.

    Attributes:
        _variables: Source variable objects in meteorology-file order.
        _records: ``(source_index, local_time_index)`` pairs defining the
            deduplicated global time axis.
        shape: Concatenated shape ``(time, *source_shape[1:])``.
        dimensions: NetCDF dimension names copied from the first source.
        dtype: NumPy dtype reported by the first source variable.
    """

    def __init__(
        self,
        variables: Sequence[NetCDFVariable],
        records: Sequence[tuple[int,int]],
    ) -> None:
        """Create a concatenated variable view.

        Args:
            variables: Equivalent NetCDF-like variables, one per source file.
            records: Ordered ``(source_index, local_time_index)`` pairs to
                expose along the concatenated time axis.
        """
        self._variables: tuple[NetCDFVariable,...]=tuple(variables)
        self._records: tuple[tuple[int,int],...]=tuple(records)
        self.shape: tuple[int,...]=(len(self._records),*variables[0].shape[1:])
        self.dimensions: tuple[str,...]=variables[0].dimensions
        self.dtype: np.dtype[np.generic]=variables[0].dtype

    def __getitem__(self,key: object) -> npt.ArrayLike:
        """Read one time record or a slice of concatenated records.

        Args:
            key: Integer time index, time slice, or tuple whose first element
                is a time integer/slice and remaining elements index the
                source variable's trailing axes.

        Returns:
            Source scalar or NumPy array for an integer index; a stacked array
            with time as axis 0 for a slice. Shape, dtype, and physical units
            otherwise follow the source variable.

        Raises:
            IndexError: If an integer time index lies outside the concatenated
                time axis.
            TypeError: If the time selector is neither an integer nor a slice.
        """
        first: object=key[0] if isinstance(key,tuple) else key
        rest: tuple[object,...]=key[1:] if isinstance(key,tuple) else ()
        if isinstance(first,(int,np.integer)):
            index: int=int(first)
            if index<0: index+=self.shape[0]
            if not 0<=index<self.shape[0]: raise IndexError(index)
            source: int; local: int
            source,local=self._records[index]
            value: object=(
                self._variables[source][(local,*rest)]
                if rest else self._variables[source][local]
            )
            return cast(npt.ArrayLike,value)
        if isinstance(first,slice):
            indices: range=range(*first.indices(self.shape[0]))
            values: list[npt.ArrayLike]=[
                self[(index,*rest)] if rest else self[index]
                for index in indices
            ]
            if values: return np.stack(values)
            return np.asarray(self._variables[0][(slice(0,0),*rest)])
        raise TypeError('concatenated meteorology supports integer or slice time indices')


class _ConcatenatedMeteorology:
    """Minimal read-only NetCDF-like view over sequential CALWRF datasets.

    Attributes:
        _datasets: Source datasets in chronological file order.
        variables: Mapping of common variable names to concatenated time
            variables or the first dataset's static variable.
    """

    def __init__(self,datasets: Sequence[Dataset]) -> None:
        """Validate and concatenate several canonical meteorology datasets.

        Args:
            datasets: Sequence of at least two open NetCDF-like CALWRF
                datasets in chronological order.

        Raises:
            ValueError: If required variables are missing, static fields or
                variable layouts differ, timestamps are unordered, a file is
                empty, or duplicate timestamps contain different data.
        """
        self._datasets: tuple[Dataset,...]=tuple(datasets)
        if len(self._datasets)<2: raise ValueError('concatenated meteorology needs multiple files')
        required: tuple[str,...]=(
            'time_text','latitude','longitude','terrain','landuse'
        )
        for name in required:
            if any(name not in dataset.variables for dataset in self._datasets):
                raise ValueError(f'meteorology file lacks required variable {name}')
        for name in required[1:]:
            reference: npt.NDArray[np.generic]=np.asarray(
                self._datasets[0][name][:]
            )
            for dataset in self._datasets[1:]:
                candidate: npt.NDArray[np.generic]=np.asarray(dataset[name][:])
                if reference.shape!=candidate.shape or not np.allclose(reference,candidate,
                        rtol=0.,atol=5.e-5,equal_nan=True):
                    raise ValueError(f'meteorology files have inconsistent static field {name}')
        common: set[str]=set(self._datasets[0].variables)
        for dataset in self._datasets[1:]: common.intersection_update(dataset.variables)
        records: list[tuple[int,int]]=[]
        stamps: list[datetime]=[]
        for source,dataset in enumerate(self._datasets):
            source_stamps: list[datetime]=[
                dataset_timestamp(cast(DatasetLike,dataset),index)
                for index in range(dataset['time_text'].shape[0])
            ]
            if not source_stamps:
                raise ValueError('meteorology file contains no time records')
            if any(right<=left for left,right in zip(source_stamps[:-1],source_stamps[1:])):
                raise ValueError('meteorology records must be strictly increasing within each file')
            for local,stamp in enumerate(source_stamps):
                if stamps and stamp<stamps[-1]:
                    raise ValueError('meteorology files are not in chronological order')
                if stamps and stamp==stamps[-1]:
                    previous_source: int; previous_local: int
                    previous_source,previous_local=records[-1]
                    for name in common-{'time_text'}:
                        variables: list[NetCDFVariable]=[
                            item[name] for item in self._datasets
                        ]
                        if not variables[0].dimensions or variables[0].dimensions[0]!='time':
                            continue
                        previous: npt.NDArray[np.generic]=np.asarray(
                            variables[previous_source][previous_local]
                        )
                        candidate=np.asarray(
                            variables[source][local]
                        )
                        if previous.shape!=candidate.shape or not np.allclose(
                                previous,candidate,rtol=1.e-6,atol=1.e-6,equal_nan=True):
                            raise ValueError(
                                f'duplicate meteorology time has inconsistent variable {name}')
                    continue
                records.append((source,local)); stamps.append(stamp)
        self.variables: dict[str,NetCDFVariable | _ConcatenatedVariable]={}
        for name in common:
            variables=[
                dataset[name] for dataset in self._datasets
            ]
            if variables[0].dimensions and variables[0].dimensions[0]=='time':
                if any(variable.dimensions!=variables[0].dimensions or
                       variable.shape[1:]!=variables[0].shape[1:] for variable in variables[1:]):
                    raise ValueError(f'meteorology files have inconsistent variable {name}')
                self.variables[name]=_ConcatenatedVariable(variables,records)
            else: self.variables[name]=variables[0]

    def __getitem__(self,name: str) -> NetCDFVariable | _ConcatenatedVariable:
        """Return a common static or concatenated variable by name.

        Args:
            name: NetCDF variable name.

        Returns:
            The stored source variable or ``_ConcatenatedVariable`` view.

        Raises:
            KeyError: If the variable is not common to all input datasets.
        """
        return self.variables[name]


MeteorologyInput: TypeAlias = (
    Dataset | _ConcatenatedMeteorology | Mapping[str,ByteArray]
)


def _has_meteorology_variable(dataset: MeteorologyInput,name: str) -> bool:
    """Return whether a meteorology input exposes a named variable.

    Native MM4 and PROG adapters expose their synthetic timeline as a mapping,
    while canonical and concatenated NetCDF inputs expose a ``variables``
    mapping.  This helper keeps capability checks independent of that storage
    detail.

    Args:
        dataset: Native timeline mapping or NetCDF-like meteorology dataset.
        name: Variable name to look up.

    Returns:
        ``True`` when ``name`` is available from the input.
    """
    variables: Mapping[str,object]=(
        dataset if isinstance(dataset,Mapping) else dataset.variables
    )
    return name in variables


@contextmanager
def _meteorology_dataset(
    paths: str | Path | Sequence[str | Path],
    temporary_directory: str | Path,
) -> Iterator[Dataset | _ConcatenatedMeteorology]:
    """Open and concatenate canonical NetCDF and/or formatted M3D files.

    Args:
        paths: One path or an ordered iterable of meteorology paths.
        temporary_directory: Directory used for temporary formatted-M3D
            conversions.

    Yields:
        One open NetCDF dataset for a single source, or a read-only
        ``_ConcatenatedMeteorology`` view for multiple sources. Time-dependent
        variables use time as axis 0 and retain their source dtype and units.

    Raises:
        ValueError: If no paths are supplied or multiple datasets cannot be
            concatenated consistently.
        OSError: If an input or temporary staged file cannot be opened.
    """
    if isinstance(paths,(str,Path)): paths=(Path(paths),)
    else: paths=tuple(Path(path) for path in paths)
    if not paths: raise ValueError('meteorology file list is empty')
    with ExitStack() as stack:
        datasets: list[Dataset]=[
            stack.enter_context(
                _single_meteorology_dataset(path,temporary_directory)
            )
            for path in paths
        ]
        yield datasets[0] if len(datasets)==1 else _ConcatenatedMeteorology(datasets)


@contextmanager
def _geophysical_dataset(
    path: str | Path,temporary_directory: str | Path
) -> Iterator[Dataset]:
    """Open canonical NetCDF or transparently stage a formatted GEO.DAT.

    Args:
        path: Canonical geophysical NetCDF or traditional formatted
            ``GEO.DAT`` path.
        temporary_directory: Directory in which to stage a temporary NetCDF
            conversion.

    Yields:
        Open read-only :class:`netCDF4.Dataset` containing geophysical
        ``(y, x)`` grids such as terrain in metres and surface properties.

    Raises:
        OSError: If the source cannot be read or the staged NetCDF cannot be
            created.
        ValueError: If a traditional GEO source is malformed.

    Notes:
        Any staged NetCDF file is removed when the context exits.
    """
    with Path(path).open('rb') as stream: signature: bytes=stream.read(16)
    if signature.strip().upper().startswith(b'GEO.DAT'):
        from .legacy_geo import GeoReader
        with tempfile.NamedTemporaryFile(dir=temporary_directory,suffix='.nc',delete=False) as handle:
            staged: Path=Path(handle.name)
        try:
            GeoReader(path).to_netcdf(staged)
            with Dataset(staged) as dataset: yield dataset
        finally:
            staged.unlink(missing_ok=True)
    else:
        with Dataset(path) as dataset: yield dataset

def _timezone_offset(label: str) -> int:
    """Convert a UTC label to CALMET's positive-west integer convention.

    Args:
        label: Timezone label accepted by ``parse_timezone_label``, such as
            ``UTC+0800`` or ``UTC-0500``.

    Returns:
        Whole-hour offset using CALMET's convention: positive values are west
        of UTC and negative values are east of UTC.

    Raises:
        ValueError: If the label is malformed or contains a nonintegral-hour
            offset.
    """
    try:
        return parse_timezone_label(label)[0]
    except (TypeError,ValueError) as error:
        raise ValueError(f'invalid CALMET timezone {label!r}: {error}') from error


def _validate_input_timezone(
    value: str | float | int,base_timezone: int,dataset: str
) -> None:
    """Compare an input timezone with the configured CALMET base timezone.

    Args:
        value: Legacy numeric positive-west offset or modern UTC label.
        base_timezone: CALMET positive-west base offset in whole hours.
        dataset: Input name used in validation errors.

    Raises:
        ValueError: If the input timezone is malformed, nonintegral, or does
            not equal ``base_timezone``.
    """
    try:
        offset: int=(_timezone_offset(value) if isinstance(value,str) else
                exact_integer(value,f'{dataset} timezone'))
    except (TypeError,ValueError) as error:
        raise ValueError(f'invalid {dataset} timezone {value!r}: {error}') from error
    if offset!=base_timezone:
        raise ValueError(
            f'{dataset} timezone {value!r} does not match '
            f'CALMET base timezone {base_timezone}')


def _validate_input_span(
    data: SurfaceData | PrecipitationData,
    start: datetime,
    end: datetime,
    dataset: str,
) -> None:
    """Reproduce READHD's declared begin/end coverage checks.

    Args:
        data: Input reader exposing declared ``start`` and ``end`` datetimes.
        start: Required CALMET interval beginning in local model time.
        end: Required CALMET interval ending in local model time.
        dataset: Input name used in validation errors.

    Raises:
        ValueError: If the declared input period does not contain the complete
            requested CALMET interval.
    """
    if data.start>start or data.end<end:
        raise ValueError(
            f'{dataset} declared period {data.start}..{data.end} does not cover '
            f'CALMET run {start}..{end}')


def _validate_upper_span(
    data: UpperAirData,start: datetime,end: datetime,base_timezone: int
) -> None:
    """Validate that one UTC UP.DAT station covers the requested run.

    Args:
        data: Upper-air reader exposing UTC ``start`` and ``end`` datetimes.
        start: CALMET run beginning in naive local model time.
        end: CALMET run ending in naive local model time.
        base_timezone: CALMET positive-west offset used to convert the run to
            UTC.

    Raises:
        ValueError: If the sounding file begins too late or ends too early.
    """
    run_start: datetime=start+timedelta(hours=base_timezone)
    run_end: datetime=end+timedelta(hours=base_timezone)
    if data.start>run_start:
        raise ValueError('UP.DAT declared period does not cover CALMET run beginning')
    if data.end<run_end:
        raise ValueError('UP.DAT declared period does not cover CALMET run ending')


def _validate_sea_metadata(
    config: CalmetConfig, stations: Sequence[SeaData]
) -> None:
    """Apply RDHDOW checks that depend on CALMET run controls.

    Args:
        config: CALMET configuration supplying timezone, COARE method, and
            grid controls.
        stations: Iterable of loaded ``SeaData`` station readers.

    Raises:
        ValueError: If a SEA file version cannot support the requested COARE
            mode, its timezone differs from CALMET, or station files disagree
            on projection or datum metadata.
    """
    reference: tuple[str,str,str,int,str,float,float,float,float,float,float] | None=None
    for data in stations:
        if config.overwater_flux_method<0 and float(data.version[:4])<2.1:
            raise ValueError('negative ICOARE requires SEA.DAT version 2.1 or newer')
        if data.timezone is not None:
            actual: str=data.timezone.upper()
            try:
                sea_timezone: int=_timezone_offset(actual)
            except ValueError as error:
                raise ValueError(
                    f'invalid SEA.DAT timezone {data.timezone!r}: {error}',
                ) from error
            if sea_timezone!=config.base_timezone:
                raise ValueError(
                    f'SEA.DAT timezone {data.timezone!r} does not match '
                    f'CALMET base timezone {config.base_timezone}')
        grid=data.source_grid
        metadata: tuple[str,str,str,int,str,float,float,float,float,float,float]=(
                  str(data.projection).strip().upper(),
                  str(data.datum).strip().upper(),str(data.datum_date).strip(),
                  int(grid.utm_zone),str(grid.hemisphere).strip().upper(),
                  float(grid.lat0),float(grid.lon0),float(grid.lat1),float(grid.lat2),
                  float(grid.false_easting_km),float(grid.false_northing_km))
        if reference is None:
            reference=metadata
            continue
        text_equal: bool=metadata[:5]==reference[:5]
        angles_equal: bool=all(_legacy_close(left,right)
                         for left,right in zip(metadata[5:9],reference[5:9]))
        offsets_equal: bool=metadata[9:]==reference[9:]
        if not (text_equal and angles_equal and offsets_equal):
            raise ValueError('SEA.DAT files do not share projection and datum metadata')


def _surface_control_metadata(
    config: CalmetConfig,
) -> tuple[SurfaceStation,...] | None:
    """Convert Group-7 projected surface-station rows to geographic metadata.

    Args:
        config: CALMET configuration containing projected surface control
            rows and the target grid definition.

    Returns:
        Tuple of ``SurfaceStation`` objects with latitude/longitude in degrees
        and anemometer height in metres, or ``None`` when no rows are supplied.

    Raises:
        ValueError: If a station timezone differs from the CALMET base
            timezone or coordinates cannot be transformed.
    """
    rows=tuple(config.surface_control_stations)
    if not rows: return None
    zones: Float64Array=np.asarray([item[4] for item in rows],dtype=np.float64)
    if np.any(zones!=config.base_timezone):
        raise ValueError('surface control-station time zone does not match CALMET base time zone')
    longitude,latitude=cast(tuple[Float64Array,Float64Array],
        unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km'))
    return tuple(SurfaceStation(item[1],item[0],float(lat),float(lon),item[5])
                 for item,lat,lon in zip(rows,latitude,longitude))


def _precipitation_control_metadata(
    config: CalmetConfig,
) -> tuple[SurfaceStation,...] | None:
    """Convert projected precipitation-control rows to station metadata.

    Args:
        config: CALMET configuration containing projected precipitation rows
            and the target grid definition.

    Returns:
        Tuple of ``SurfaceStation`` objects with latitude/longitude in degrees
        and zero anemometer height, or ``None`` when no rows are supplied.

    Raises:
        ValueError: If the configured coordinates cannot be transformed.
    """
    rows=tuple(config.precipitation_control_stations)
    if not rows: return None
    longitude,latitude=cast(tuple[Float64Array,Float64Array],
        unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km'))
    return tuple(SurfaceStation(item[1],item[0],float(lat),float(lon),0.)
                 for item,lat,lon in zip(rows,latitude,longitude))


def _upper_control_metadata(
    config: CalmetConfig,
) -> tuple[UpperAirStation,...] | None:
    """Convert projected upper-air control rows to station metadata.

    Args:
        config: CALMET configuration containing projected upper-air rows and
            the target grid definition.

    Returns:
        Tuple of ``UpperAirStation`` objects with latitude/longitude in degrees
        and zero elevation, or ``None`` when no rows are supplied.

    Raises:
        ValueError: If a station timezone differs from the CALMET base
            timezone or coordinates cannot be transformed.
    """
    rows=tuple(config.upper_control_stations)
    if not rows: return None
    zones: Float64Array=np.asarray([item[4] for item in rows],dtype=np.float64)
    if np.any(zones!=config.base_timezone):
        raise ValueError('upper-air control-station time zone does not match CALMET base time zone')
    longitude,latitude=cast(tuple[Float64Array,Float64Array],
        unproject_geographic([item[2] for item in rows],
        [item[3] for item in rows],config.grid,config.grid.datum,units='km'))
    return tuple(UpperAirStation(item[1],item[0],float(lat),float(lon),0.)
                 for item,lat,lon in zip(rows,latitude,longitude))

def _legacy_close(
    first: float,second: float,tolerance: float=1.e-4
) -> bool:
    """Compare scalars with the relative tolerance used by legacy headers.

    Args:
        first: First numeric scalar.
        second: Second numeric scalar.
        tolerance: Maximum difference divided by the absolute pair mean.

    Returns:
        ``True`` for exact equality or a relative difference no greater than
        ``tolerance``. Values whose difference is not below their absolute
        mean are rejected before division.
    """
    if first==second: return True
    difference: float=abs(first-second); average: float=.5*abs(first+second)
    return difference<average and difference/average<=tolerance

def _validate_geophysical_grid(dataset: Dataset,grid: GridConfig) -> None:
    """Apply READGE grid/header compatibility checks to canonical NetCDF GEO.

    Args:
        dataset: Open geophysical NetCDF dataset whose optional global
            attributes describe the grid and datum.
        grid: Expected CALMET grid configuration; distances are in kilometres
            and angular projection parameters in degrees.

    Raises:
        ValueError: If a present numeric or text attribute differs from the
            configured grid beyond the legacy relative tolerance.
    """
    expected: dict[str,Any]={
        'grid_nx':grid.nx,'grid_ny':grid.ny,'grid_dx_km':grid.dx_km,'grid_dy_km':grid.dy_km,
        'x_origin_km':grid.x_origin_km,'y_origin_km':grid.y_origin_km,
    }
    projection: str=grid.projection.upper()
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
    differences: list[str]=[]
    for name,wanted in expected.items():
        if name not in dataset.ncattrs(): continue
        actual: Any=getattr(dataset,name)
        if isinstance(wanted,int): equal: bool=int(actual)==wanted
        else: equal=_legacy_close(float(actual),float(wanted))
        if not equal: differences.append(f'{name}={actual!r} (expected {wanted!r})')
    text_expected: dict[str,str]={'projection':grid.projection}
    if projection=='UTM': text_expected['hemisphere']=grid.hemisphere
    for name,wanted in text_expected.items():
        if name in dataset.ncattrs() and str(getattr(dataset,name)).strip().upper()!=str(wanted).strip().upper():
            differences.append(f'{name}={getattr(dataset,name)!r} (expected {wanted!r})')
    if ('datum' in dataset.ncattrs() and
            normalize_datum(str(dataset.datum))!=normalize_datum(grid.datum)):
        differences.append(f'datum={dataset.datum!r} (expected {grid.datum!r})')
    if differences: raise ValueError('geophysical grid does not match CALMET control: '+', '.join(differences))

def _prognostic_precipitation_radius(
    mapper: PrognosticMapper | None,requested_km: float,station_count: int
) -> float:
    """Bound gridded precipitation radius from both projected grid spacings.

    The longer median cardinal spacing is the smallest useful radius.  The
    corresponding cell diagonal, enlarged by five percent, is the automatic
    radius used when the request falls outside the useful interval.

    Args:
        mapper: ``PrognosticMapper`` whose source coordinates are flattened
            ``(point, 2)`` x/y positions in kilometres, or ``None``.
        requested_km: User-requested precipitation interpolation radius in
            kilometres.
        station_count: Precipitation source selector; ``-1`` enables the
            prognostic automatic-radius rule.

    Returns:
        Requested radius unchanged when the automatic rule is inactive or the
        grid spacing cannot be derived; otherwise the requested radius when it
        lies in the useful interval, or 105 percent of the source-cell
        diagonal in kilometres.

    Raises:
        ValueError: If mapper coordinates do not match ``source_shape``.
    """
    if station_count!=-1 or mapper is None: return requested_km
    ny: int; nx: int
    ny,nx=mapper.source_shape
    coordinates: Float64Array=np.asarray(
        mapper.source_coordinates,dtype=np.float64
    )
    if coordinates.shape!=(ny*nx,2):
        raise ValueError('prognostic source coordinates do not match source grid shape')
    grid: Float64Array=coordinates.reshape((ny,nx,2),order='F')
    axis_spacing: list[float]=[]
    if nx>1:
        adjacent: Float64Array=np.linalg.norm(np.diff(grid,axis=1),axis=2)
        valid: Float64Array=adjacent[np.isfinite(adjacent)&(adjacent>0.)]
        if valid.size: axis_spacing.append(float(np.median(valid)))
    if ny>1:
        adjacent=np.linalg.norm(np.diff(grid,axis=0),axis=2)
        valid=adjacent[np.isfinite(adjacent)&(adjacent>0.)]
        if valid.size: axis_spacing.append(float(np.median(valid)))
    if not axis_spacing: return requested_km
    if len(axis_spacing)==1: axis_spacing*=2
    minimum: float=max(axis_spacing)
    maximum: float=float(np.hypot(*cast(tuple[float,float],axis_spacing))*1.05)
    requested: float=float(requested_km)
    return maximum if requested<minimum or requested>maximum else requested

def _output_path(config: CalmetConfig) -> Path:
    """Select OPENOT's METDAT/PACDAT target for the requested format.

    Args:
        config: CALMET configuration containing primary and optional MESOPAC
            output paths.

    Returns:
        ``mesopac_output`` as a :class:`Path` for output format 2 when it is
        provided; otherwise the primary ``output`` path.
    """
    if config.legacy_output_format==2 and config.mesopac_output is not None:
        return Path(config.mesopac_output)
    return Path(config.output)

def _write_distance_to_coast_grid(
        path: str | Path,landuse: npt.ArrayLike,grid: GridConfig,
        water_range: tuple[int,int]) -> Float32Array:
    """Write SETCOAST's optional Surfer ASCII distance grid.

    Args:
        path: Destination Surfer ASCII grid path; missing parents are created.
        landuse: Integer dominant-land-use grid with shape ``(ny, nx)``.
        grid: CALMET grid geometry and spacings in kilometres.
        water_range: Inclusive pair of land-use categories treated as water.

    Returns:
        ``float`` array with shape ``(ny, nx)`` containing distance to the
        nearest coast in kilometres.

    Raises:
        OSError: If the destination directory or ASCII file cannot be written.

    Notes:
        The file is replaced with a DSAA Surfer grid using CALMET cell-centre
        coordinates in kilometres.
    """
    distance: Float32Array=distance_to_coast(
        landuse,water_range,grid.dx_km,grid.dy_km
    )
    xb: float=grid.x_origin_km+grid.dx_km
    yb: float=grid.y_origin_km+grid.dy_km
    xe: float=xb+(grid.nx-1)*grid.dx_km
    ye: float=yb+(grid.ny-1)*grid.dy_km
    lines: list[str]=['DSAA',f'{grid.nx:12d}{grid.ny:12d}',
           f'{xb:12.4f}{xe:12.4f}',f'{yb:12.4f}{ye:12.4f}',
           f'{0.:12.4E}{float(np.max(distance)):12.4E}']
    lines.extend(''.join(f'{float(value):10.4E}  ' for value in row).rstrip()
                 for row in distance)
    target: Path=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return distance


def _diagnostic_input_options(config: CalmetConfig) -> dict[str,bool]:
    """Return only DIAG.DAT fields that the selected modern path consumes.

    Args:
        config: CALMET configuration containing NOOBS, diagnostic switches,
            and temperature controls.

    Returns:
        Boolean mapping for ``surface_temperature``, ``lapse_rate``,
        ``initial_wind``, ``surface_wind``, and ``upper_wind``. A value is
        true only when the active observation mode consumes that field.
    """
    has_surface_observations: bool=config.noobs in {0,1}
    has_upper_observations: bool=config.noobs==0
    return {
        'surface_temperature':bool(config.diagnostic_surface_temperature and
                                   config.surface_temperature_option!=-1),
        'lapse_rate':bool(config.diagnostic_lapse_rate and has_upper_observations),
        'initial_wind':bool(config.diagnostic_initial_wind and has_upper_observations),
        'surface_wind':bool(config.diagnostic_surface_wind and
                            has_surface_observations),
        'upper_wind':bool(config.diagnostic_upper_wind and has_upper_observations),
    }


def _validate(config: CalmetConfig) -> Float32Array:
    """Validate coupled CALMET controls and derive the vertical faces array.

    Args:
        config: Fully normalized CALMET configuration. Its grid, observation,
            prognostic, output, cloud, precipitation, temperature, wind, and
            mixing-height controls are checked together.

    Returns:
        One-dimensional ``float32`` array of vertical cell-face heights in
        metres with shape ``(nz + 1,)``.

    Raises:
        ValueError: If any control is outside its supported domain, required
            inputs or per-level values are absent, coupled options conflict,
            timestamps do not align with the computational step, or the run
            violates vertical-grid and initialization constraints.

    Notes:
        When ``config.barrier_top_level`` is ``None``, validation sets it to
        the number of CALMET layers. No other model fields are created.
    """
    projection: str=config.grid.projection.upper()
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
    if (not np.isfinite(config.grid.dx_km) or not np.isfinite(config.grid.dy_km) or
            config.grid.dx_km<=0. or config.grid.dy_km<=0.):
        raise ValueError('CALMET DX and DY grid spacing must be finite and positive')
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
    level_output: tuple[Sequence[int],...]=(config.print_uv_levels,config.print_w_levels,
                  config.print_temperature_levels)
    if any(values and len(values)!=len(config.z_faces_m)-1 for values in level_output):
        raise ValueError('IUVOUT/IWOUT/ITOUT must contain one value per CALMET layer')
    output_switches: tuple[int,...]=tuple(
        value for values in level_output for value in values
    )
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
    if config.iprog not in {0,1,2,3,4,5,13,14,15}:
        raise ValueError('IPROG must be one of 0, 1, 2, 3, 4, 5, 13, 14, or 15')
    if config.iwfcod not in {0,1}:
        raise ValueError('IWFCOD must be zero or one')
    if config.use_initial_guess_calmet:
        if not config.initial_guess_calmet_files:
            raise ValueError('IGFMET=1 requires at least one IGFDAT file')
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
    diagnostic_options: tuple[int,...]=(config.diagnostic_surface_temperature,config.diagnostic_lapse_rate,
                        config.diagnostic_initial_wind,config.diagnostic_surface_wind,
                        config.diagnostic_upper_wind)
    if any(value not in {0,1} for value in diagnostic_options):
        raise ValueError('IDIOPT1..5 must be zero or one')
    effective_diagnostic: dict[str,bool]=_diagnostic_input_options(config)
    if any(effective_diagnostic.values()) and config.diagnostic_data is None:
        raise ValueError('enabled IDIOPT option requires diagnostic_data/DIADAT')
    if (config.noobs in {0,1} and config.surface_data is None and
            (not config.diagnostic_surface_wind or config.run_type!=0)):
        raise ValueError('NOOBS=0/1 requires surface_data/SRFDAT')
    if (config.noobs in {0,1} and config.diagnostic_surface_wind and
            config.surface_data is None and
            not config.surface_control_stations):
        raise ValueError('IDIOPT4 requires surface_control_stations/SS records when SURF.DAT is omitted')
    if (config.noobs==0 and not config.upper_air_files and
            (not config.diagnostic_upper_wind or config.run_type!=0)):
        raise ValueError('NOOBS=0 requires upper_air_files/UPDAT')
    if (config.noobs==0 and config.diagnostic_upper_wind and
            not config.upper_air_files and not config.upper_control_stations):
        raise ValueError('IDIOPT5 requires upper_control_stations/US records when UPDAT is omitted')
    upper_station_count: int=(len(config.upper_control_stations)
                         if config.diagnostic_upper_wind and not config.upper_air_files
                         else len(config.upper_air_files))
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
        regulatory: bool=(config.overwater_flux_method==0 and config.land_flux_threshold==0. and
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
    if config.cloud_method not in {1,2,3,4}:
        raise ValueError('cloud_method/MCLOUD must be 1, 2, 3, or 4')
    if (not np.isfinite(config.cloud_condensate_threshold_g_kg) or
            config.cloud_condensate_threshold_g_kg<=0.):
        raise ValueError('cloud_condensate_threshold_g_kg must be finite and positive')
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
    faces: Float32Array=np.asarray(config.z_faces_m,dtype=np.float32)
    if faces.ndim!=1 or faces.size<2 or faces[0]!=0. or np.any(np.diff(faces)<=0.):
        raise ValueError('z_faces_m must be strictly increasing and begin at zero')
    first_center: np.float32=.5*(faces[0]+faces[1])
    if config.run_type!=0 and not 9.<=first_center<=11.:
        raise ValueError(
            'full CALMET runs require the first layer midpoint to be between 9 and 11 m')
    if config.barrier_top_level is None:
        config.barrier_top_level=faces.size-1
    if config.barrier_count and not 1<=config.barrier_top_level<faces.size:
        raise ValueError('barrier_top_level/KBAR must identify a CALMET layer')
    if len(cast(Sequence[int],config.smoothing_passes))!=faces.size-1:
        raise ValueError('smoothing_passes must contain one value per CALMET layer')
    if config.noobs!=2:
        if (len(cast(Sequence[int],config.stations_per_level))!=faces.size-1 or
                any(value<1 for value in cast(Sequence[int],config.stations_per_level))):
            raise ValueError('stations_per_level/NINTR2 must contain one positive value per CALMET layer')
        if len(cast(Sequence[float],config.wind_extrapolation_factors))!=faces.size-1:
            raise ValueError('wind_extrapolation_factors/FEXTR2 must contain one value per CALMET layer')
        if abs(config.wind_extrapolation) not in {1,2,3,4}:
            raise ValueError('wind_extrapolation/IEXTRP must be +/-1, +/-2, +/-3, or +/-4')
        if (len(cast(Sequence[float],config.wind_bias))<faces.size-1 or
                any(abs(value)>1. for value in
                    cast(Sequence[float],config.wind_bias)[:faces.size-1])):
            raise ValueError('wind_bias/BIAS must provide one value in [-1,1] per CALMET layer')
        if config.extrapolate_calm not in {0,1}:
            raise ValueError('extrapolate_calm/ICALM must be zero or one')
    if config.iavezi and not 1<=config.mixing_wind_level<faces.size:
        raise ValueError('mixing_wind_level must identify a CALMET layer using one-based indexing')
    begin: datetime=config.start; finish: datetime=config.end
    # Full runs initialize their stateful convective-height history from zero;
    # a pre-dawn interval is required before daytime surface heating begins.
    if config.run_type!=0 and begin.hour>=5:
        raise ValueError('CALMET prognostic mixing-height runs must start before 05 LST')
    for label,value in (('start',begin),('end',finish)):
        hour: datetime=value.replace(minute=0,second=0,microsecond=0)
        seconds: float=(value-hour).total_seconds()
        if seconds%config.timestep_seconds:
            raise ValueError(f'CALMET {label} time must align with NSECDT')
    return faces


def _initial_guess_at(
    config: CalmetConfig,
    initial_guess: InitialGuessCalmetSeries,
    timestamp: datetime,
    centers: npt.ArrayLike,
) -> tuple[Float32Array,Float32Array]:
    """Interpolate coarse CALMET initial-guess winds at one target time.

    Args:
        config: CALMET configuration supplying timestep, base timezone, and
            target grid.
        initial_guess: ``InitialGuessCalmetSeries`` containing coarse fields
            and its source timezone.
        timestamp: Naive local beginning of the target CALMET interval.
        centers: One-dimensional target layer-centre heights in metres with
            shape ``(nz,)``.

    Returns:
        Pair ``(u, v)`` of ``float32`` wind-component arrays in metres per
        second, each shaped ``(nz, ny, nx)`` on the target CALMET grid.

    Raises:
        ValueError: If the initial-guess timezone is unsupported or its
            records cannot cover/interpolate the requested interval ending.
    """
    try: igf_timezone: int=_timezone_offset(initial_guess.timezone)
    except ValueError as error:
        raise ValueError(f'unsupported IGFDAT timezone: {initial_guess.timezone!r}') from error
    igf_end: datetime=timestamp+timedelta(seconds=config.timestep_seconds,
                               hours=config.base_timezone-igf_timezone)
    return interpolate_initial_guess(initial_guess,igf_end,config.grid,centers)


def _noobs_step_transform(
    config: CalmetConfig,
    initial_guess: InitialGuessCalmetSeries | None,
    diagnostic: DiagnosticData | None,
    precipitation: PrecipitationData | None = None,
    precipitation_points: npt.ArrayLike | None = None,
    sea: Sequence[SeaData] = (),
) -> StepTransform:
    """Build a prognostic-only step transformer for optional auxiliary data.

    Args:
        config: CALMET configuration controlling time, temperature, sea, and
            precipitation handling.
        initial_guess: Optional coarse ``InitialGuessCalmetSeries``.
        diagnostic: Optional ``DiagnosticData`` supplying surface temperature.
        precipitation: Optional station precipitation reader.
        precipitation_points: ``(station, 2)`` station x/y coordinates in
            kilometres relative to the CALMET origin.
        sea: Iterable of ``SeaData`` station readers.

    Returns:
        Callable ``transform(step, timestamp)`` that returns a replaced
        ``Step1Fields`` object. Updated physical grids retain ``float32`` and
        horizontal ``(ny, nx)`` or vertical ``(nz, ny, nx)`` layouts.

    Raises:
        ValueError: During transformation, if prognostic offshore data needed
            by water-temperature mode 2 is unavailable.
    """
    faces: Float32Array=np.asarray(config.z_faces_m,dtype=np.float32)
    centers: Float32Array=.5*(faces[:-1]+faces[1:])
    sea=tuple(sea)
    sea_points: Float64Array=(np.stack([data.point(config.grid) for data in sea])
                if sea else np.empty((0,2)))
    sea_longitudes: list[float]=[
        data.longitude_latitude()[0] for data in sea
    ]
    def transform(step: Step1Fields,timestamp: datetime) -> Step1Fields:
        """Merge optional prognostic-only inputs into one Step-1 state.

        Args:
            step: Source ``Step1Fields`` for the current interval.
            timestamp: Naive local beginning of the CALMET interval.

        Returns:
            New ``Step1Fields`` with selected initial winds, diagnostic
            temperature, precipitation, SEA fields, and water gradients.
            Horizontal fields have shape ``(ny, nx)`` and model arrays use
            ``float32`` except integer precipitation codes.

        Raises:
            ValueError: If water-temperature mode 2 requires missing
                prognostic offshore fields.
        """
        changes: dict[str,Any]={}
        record: DiagnosticRecord | PrecipitationRecord
        if initial_guess is not None:
            changes['u'],changes['v']=_initial_guess_at(
                config,initial_guess,timestamp,centers)
        if diagnostic is not None:
            record=diagnostic.at_begin(timestamp)
            if record.surface_temperature_k is not None:
                changes['diagnostic_surface_temperature']=np.full_like(
                    step.surface_temperature,record.surface_temperature_k,
                    dtype=np.float32)
        if precipitation is not None:
            record=precipitation.at_begin(timestamp)
            if not (config.run_type==0 and config.diagnostic_lapse_rate):
                changes['precipitation_rate']=grid_precipitation(
                    record.rate_mm_h,
                    cast(npt.ArrayLike,precipitation_points),config.grid,
                    config.precipitation_interpolation,config.precipitation_radius_km,
                    config.precipitation_cutoff)
        if sea:
            records: list[SeaRecord]=[
                data.at_begin(timestamp) for data in sea
            ]
            changes['observed_water_mixing_height']=grid_sea_mixing_height(
                [item.mixing_height_m for item in records],sea_points,config.grid)
            observed_water: GriddedSeaData | None=nearest_sea_data(
                records,sea_points,sea_longitudes,config.grid
            )
            if config.water_temperature_mode==2:
                if step.overwater_data is None:
                    raise ValueError('ITWPROG=2 requires prognostic offshore data')
                wave_period: Float32Array; wave_height: Float32Array
                wave_period,wave_height=nearest_sea_waves(
                    records,sea_points,config.grid
                )
                changes['overwater_data']=replace(step.overwater_data,
                    wave_period_s=wave_period,wave_height_m=wave_height)
            else:
                changes['overwater_data']=observed_water
            if config.water_temperature_mode==0:
                below: Float32Array=np.asarray(
                    [item.gradient_below_k_m for item in records],dtype=np.float32
                )
                above: Float32Array=np.asarray(
                    [item.gradient_above_k_m for item in records],dtype=np.float32
                )
                below=np.where(below<9998.,below,config.water_temperature_gradient_below)
                above=np.where(above<9998.,above,config.water_temperature_gradient_above)
                interpolation: dict[str,Any]=dict(radius_km=config.temperature_radius_km,
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


def _observed_mixing_profiles(
    upper_air: Sequence[UpperAirData],
    station_xy: npt.ArrayLike,
    grid: GridConfig,
    timestamp: datetime,
    upper_temperature_option: int,
) -> tuple[Float32Array,Float32Array]:
    """Map the raw sounding profiles selected by observed-mode ``MIXHT``.

    Args:
        upper_air: Upper-air files in the same order as the Group-7 stations.
        station_xy: Projected station coordinates in kilometres, shaped
            ``(station, 2)``.
        grid: CALMET target grid.
        timestamp: Interval-ending UTC time used to choose the latest-past
            sounding at each selected station.
        upper_temperature_option: CALMET ``IUPT`` value. ``-1`` selects the
            nearest station independently at each grid cell; a positive value
            selects that one-based station over the whole domain.

    Returns:
        Tuple ``(temperature, height)`` of float32 arrays shaped
        ``(source_level, grid.ny, grid.nx)``. Temperatures are kelvins and
        heights are metres AGL. Profiles retain their raw sounding levels.

    Raises:
        ValueError: If station metadata do not match, ``IUPT`` is unsupported
            or out of range, or a selected raw sounding is invalid.

    Notes:
        Soundings with fewer valid temperature levels are padded by repeating
        their terminal level. The repeated terminal height makes a request at
        or above that sounding top fail the same bracketing check as an
        unpadded profile, while every interpolation below the top remains on
        the original observed levels.
    """
    points: Float64Array=np.asarray(station_xy,dtype=np.float64)
    if points.shape!=(len(upper_air),2) or not upper_air:
        raise ValueError(
            'upper-air files and projected station coordinates do not match'
        )
    if upper_temperature_option==-1:
        source_indices: tuple[int,...]=tuple(range(len(upper_air)))
        selected: Int32Array=_nearest_station_grid(points,grid)
    elif upper_temperature_option>0:
        fixed: int=upper_temperature_option-1
        if fixed>=len(upper_air):
            raise ValueError('IUPT exceeds the UPDAT station count')
        source_indices=(fixed,)
        selected=np.zeros((grid.ny,grid.nx),dtype=np.int32)
    else:
        raise ValueError('observed MIXHT requires IUPT=-1 or a positive station')
    profiles: list[tuple[Float32Array,Float32Array]]=[
        mixing_temperature_profile(upper_air[index].mixing_sounding_at(timestamp))
        for index in source_indices
    ]
    levels: int=max(profile[0].size for profile in profiles)
    station_temperature: Float32Array=np.empty(
        (len(profiles),levels),dtype=np.float32,
    )
    station_height: Float32Array=np.empty_like(station_temperature)
    for index,(temperature,height) in enumerate(profiles):
        count: int=temperature.size
        station_temperature[index,:count]=temperature
        station_height[index,:count]=height
        station_temperature[index,count:]=temperature[-1]
        station_height[index,count:]=height[-1]
    temperature_grid: Float32Array=np.asarray(
        np.moveaxis(station_temperature[selected],-1,0),dtype=np.float32,
    )
    height_grid: Float32Array=np.asarray(
        np.moveaxis(station_height[selected],-1,0),dtype=np.float32,
    )
    return temperature_grid,height_grid


def _surface_step_transform(
    config: CalmetConfig,
    surface: SurfaceData | None,
    geo: Dataset,
    upper_air: Sequence[UpperAirData] = (),
    precipitation: PrecipitationData | None = None,
    sea: Sequence[SeaData] = (),
    weights: Float32Array | None = None,
    diagnostic: DiagnosticData | None = None,
    initial_guess: InitialGuessCalmetSeries | None = None,
    surface_stations: Sequence[SurfaceStation] | None = None,
    upper_stations: Sequence[UpperAirStation] | None = None,
) -> tuple[PhysicalStepTransform,Int32Array]:
    """Compose PREPDI observations with WIND1/INTER2 for supported modes.

    Args:
        config: CALMET configuration controlling observations, interpolation,
            wind analysis, temperature, precipitation, and overwater physics.
        surface: Loaded ``SurfaceData`` or ``None`` when DIAG.DAT supplies the
            permitted wind-only surface observations.
        geo: Open geophysical dataset containing ``(ny, nx)`` terrain and
            surface-property grids.
        upper_air: Iterable of loaded ``UpperAirData`` station readers.
        precipitation: Optional loaded station precipitation data.
        sea: Iterable of loaded ``SeaData`` overwater stations.
        weights: Optional observation-weight array shaped ``(nz, ny, nx)``.
        diagnostic: Optional ``DiagnosticData`` reader.
        initial_guess: Optional coarse ``InitialGuessCalmetSeries``.
        surface_stations: Optional explicit surface metadata used when no
            surface file is opened.
        upper_stations: Optional explicit upper-air metadata used when no
            sounding files are opened.

    Returns:
        Pair ``(transform, nearest_surface)``. ``transform`` replaces a
        ``Step1Fields`` object for one interval; ``nearest_surface`` is an
        integer ``(ny, nx)`` array of zero-based nearest surface-station
        indices.

    Raises:
        ValueError: If required station metadata is absent, station/data counts
            differ, or configured coordinates and observation inputs cannot be
            prepared.
    """
    surface_stations=tuple(surface.stations if surface is not None else
                           (() if surface_stations is None else surface_stations))
    if not surface_stations:
        raise ValueError('surface station metadata is required for NOOBS=0/1')
    surface_datum: str=(
        surface.datum if surface is not None else config.grid.datum
    )
    points: Float64Array; nearest: Int32Array
    points,nearest=station_grid_mapping(
        surface_stations,config.grid,surface_datum
    )
    relative: Float64Array=points-np.array(
        [config.grid.x_origin_km,config.grid.y_origin_km]
    )
    station_ix: IntpArray=np.clip(
        np.floor(relative[:,0]/config.grid.dx_km).astype(int),
        0,config.grid.nx-1,
    )
    station_iy: IntpArray=np.clip(
        np.floor(relative[:,1]/config.grid.dy_km).astype(int),
        0,config.grid.ny-1,
    )
    roughness: Float32Array=np.asarray(
        geo['roughness_length'][:],dtype=np.float32
    )[station_iy,station_ix]
    latitude: Float64Array=np.array(
        [station.latitude for station in surface_stations]
    )
    longitude: Float64Array=np.array(
        [station.longitude for station in surface_stations]
    )
    convergence: Float64Array=grid_convergence(
        config.grid,longitude,latitude
    )
    anemometer: Float32Array=np.asarray(
        [station.anemometer_height_m for station in surface_stations],dtype=np.float32,
    )
    faces: Float32Array=np.asarray(config.z_faces_m,dtype=np.float32)
    centers: Float32Array=.5*(faces[:-1]+faces[1:])
    landuse: npt.NDArray[np.int16]=np.asarray(
        geo['dominant_landuse'][:],dtype=np.int16
    )
    upper_air=tuple(upper_air)
    upper_stations=tuple((data.station for data in upper_air) if upper_stations is None
                         else upper_stations)
    if upper_air and len(upper_air)!=len(upper_stations):
        raise ValueError('upper-air data and station metadata counts do not match')
    sea=tuple(sea)
    sea_points: Float64Array=(
        np.stack([data.point(config.grid) for data in sea])
        if sea else np.empty((0,2))
    )
    sea_locations: list[tuple[float,float]]=[
        data.longitude_latitude() for data in sea
    ]
    sea_convergence: Float64Array=(grid_convergence(config.grid,
        [item[0] for item in sea_locations],[item[1] for item in sea_locations]) if sea else np.empty(0))
    upper_points: list[Float64Array]=[]
    upper_convergence: list[float]=[]
    for index,station in enumerate(upper_stations):
        datum: str=upper_air[index].datum if upper_air else config.grid.datum
        point: Float64Array
        point,_=station_grid_mapping(
            cast(Sequence[SurfaceStation],(station,)),config.grid,datum)
        upper_points.append(point[0])
        upper_convergence.append(float(grid_convergence(
            config.grid,station.longitude,station.latitude)))
    coordinate_groups: list[Float64Array]=[relative]
    if sea_points.size:
        coordinate_groups.append(sea_points-np.array([config.grid.x_origin_km,config.grid.y_origin_km]))
    if upper_points:
        coordinate_groups.append(np.asarray(upper_points)-np.array([config.grid.x_origin_km,config.grid.y_origin_km]))
    all_relative: Float64Array=np.vstack(coordinate_groups)
    all_station_ids: npt.NDArray[np.int64] | None=(np.asarray(
        [station.identifier for station in surface_stations]
        +[data.identifier for data in sea]
        +[station.identifier for station in upper_stations],dtype=np.int64)
        if config.lake_breeze_regions else None)
    precip_points: Float64Array | None=None
    if precipitation is not None:
        precip_points,_=station_grid_mapping(precipitation.stations,config.grid,precipitation.datum)
        precip_points-=np.array([config.grid.x_origin_km,config.grid.y_origin_km])
    common: dict[str,Any]=dict(rmin_km=config.minimum_station_radius_km,
        rmax_surface_km=config.maximum_surface_radius_km,rmax_upper_km=config.maximum_upper_radius_km,
        rmax_water_km=config.maximum_water_radius_km,surface_scale_km=config.surface_background_radius_km,
        upper_scale_km=config.upper_background_radius_km,stations_per_level=config.stations_per_level,
        vary_radius=config.varying_radius,water_range=(int(getattr(geo,'water_category_min',50)),
                                                       int(getattr(geo,'water_category_max',55))),
        barriers=config.barriers,barrier_top_level=config.barrier_top_level)
    observation_only: bool=(config.iprog==0 and
        (config.meteorology is None or
         (not isinstance(config.meteorology,(str,Path)) and
          len(config.meteorology)==0)))
    radiation_options: dict[str,float]=dict(
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
            background_weight=(1.-cast(Float32Array,weights))/config.prognostic_radius_km**2)
    def transform(
        step: Step1Fields,
        timestamp: datetime,
        physical_context: StepTransformContext | None = None,
    ) -> Step1Fields:
        """Analyze observations and replace fields for one model interval.

        Args:
            step: Prognostic ``Step1Fields`` background. Horizontal fields use
                ``(ny, nx)`` and vertical fields ``(nz, ny, nx)``.
            timestamp: Naive local beginning of the CALMET interval.
            physical_context: Optional mutable ``StepTransformContext`` with
                preceding mixing heights, thermal jumps, Coriolis values,
                solar forcing, GMT hour, and COARE warm-layer state.

        Returns:
            New ``Step1Fields`` containing objective-analysis winds in metres
            per second and any selected observed temperature, humidity,
            precipitation, cloud, and overwater fields. Model floating arrays
            are ``float32``; precipitation and cloud codes remain integral.

        Raises:
            ValueError: If a required observation, offshore field, diagnostic
                wind, station selection, or prognostic strength is missing.
            RuntimeError: If similarity-theory extrapolation lacks its physical
                context or warm-layer grid indices.

        Notes:
            Similarity-theory paths may initialize or sample warm-layer state
            on ``physical_context``. Debug fields are copied into the returned
            step when requested.
        """
        diagnostic_record: DiagnosticRecord | None=(
            None if diagnostic is None else diagnostic.at_begin(timestamp)
        )
        if surface is None:
            # OPENOT/READHD deliberately omit SURF.DAT for IDIOPT4.  PREPDI
            # still needs the Group-7 locations, while every surface wind is
            # supplied by DIAG.DAT.  Sample the already available Step-1
            # fields only as inert thermodynamic scaffolding for IRTYPE=0.
            sample: tuple[IntpArray,IntpArray]=(station_iy,station_ix)
            count: int=len(surface_stations)
            raw_record: SurfaceRecord=SurfaceRecord(
                timestamp,timestamp+timedelta(seconds=config.timestep_seconds),
                np.full(count,9999.,dtype=np.float32),
                np.full(count,9999.,dtype=np.float32),
                np.trunc(np.asarray(step.ceiling_height)[sample]/30.48).astype(np.int32),
                np.trunc(np.asarray(step.cloud_fraction)[sample]*10.).astype(np.int32),
                np.asarray(step.surface_temperature,dtype=np.float32)[sample],
                np.asarray(step.surface_relative_humidity,dtype=np.float32)[sample],
                np.asarray(step.surface_pressure_mb,dtype=np.float32)[sample],
                np.asarray(step.precipitation_code,dtype=np.int32)[sample])
        else:
            raw_record=surface.at_begin(timestamp)
        record: SurfaceRecord=fill_missing_surface(
            raw_record,points,cloud_method=config.cloud_method,itprog=config.itprog
        )
        sea_records: list[SeaRecord]=[
            data.at_begin(timestamp) for data in sea
        ]
        observed_surface_temperature: Float32Array | None=None
        observed_temperature_faces: Float32Array | None=None
        observed_profile_temperature: Float32Array | None=None
        observed_profile_height: Float32Array | None=None
        mixing_profile_temperature: Float32Array | None=None
        mixing_profile_height: Float32Array | None=None
        if config.itprog in {0,1} and surface is None:
            observed_surface_temperature=np.asarray(step.surface_temperature,dtype=np.float32)
        elif config.itprog in {0,1}:
            water_min: int=config.water_temperature_category_min
            water_max: int=config.water_temperature_category_max
            temperature_values: Float32Array=np.concatenate((
                np.asarray(raw_record.temperature_k,dtype=np.float32),
                np.asarray([item.air_temperature_k for item in sea_records],dtype=np.float32),
            ))
            temperature_points: Float64Array=np.vstack((points,sea_points))
            observed_surface_temperature=np.asarray(interpolate_surface_scalar(
                temperature_values,temperature_points,config.grid,
                radius_km=config.temperature_radius_km,
                maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power,
                target_water=(landuse>=water_min)&(landuse<=water_max),
                station_water=np.r_[np.zeros(len(points),dtype=bool),
                                    np.ones(len(sea),dtype=bool)]),dtype=np.float32)
        if config.itprog==0 and upper_air:
            end: datetime=timestamp+timedelta(seconds=config.timestep_seconds,
                                    hours=config.base_timezone)
            station_faces: Float32Array=np.asarray(np.stack(
                [data.temperature_faces_at(end,config.z_faces_m) for data in upper_air]),
                dtype=np.float32,
            )
            observed_temperature_faces=grid_temperature_faces(
                station_faces,np.asarray(upper_points),config.grid,
                config.temperature_interpolation_power)
            layer_temperature=.5*(
                observed_temperature_faces[:-1]+observed_temperature_faces[1:])
            layer_temperature[0]=cast(Float32Array,observed_surface_temperature)
            observed_profile_height=np.broadcast_to(
                np.r_[np.float32(0.),centers][:,None,None],
                (centers.size+1,config.grid.ny,config.grid.nx))
            observed_profile_temperature=np.concatenate(
                (cast(Float32Array,observed_surface_temperature)[None],
                 layer_temperature),axis=0)
            if config.upper_temperature_option!=-2:
                mixing_profile_temperature,mixing_profile_height=(
                    _observed_mixing_profiles(
                        upper_air,np.asarray(upper_points),config.grid,end,
                        config.upper_temperature_option,
                    )
                )
        elif config.itprog==0:
            # IDIOPT5 omits every sounding, including its temperature profile.
            # IRTYPE=0 never writes temperature, but the Python hour object still
            # carries a well-shaped field through the wind solve.
            target_temperature: Float32Array=np.asarray(
                step.temperature,dtype=np.float32
            )
            observed_temperature_faces=np.empty(
                (faces.size,config.grid.ny,config.grid.nx),dtype=np.float32)
            observed_temperature_faces[0]=cast(
                Float32Array,observed_surface_temperature)
            for level in range(target_temperature.shape[0]):
                observed_temperature_faces[level+1]=(
                    2.*target_temperature[level]-observed_temperature_faces[level])
            layer_temperature=.5*(
                observed_temperature_faces[:-1]+observed_temperature_faces[1:])
            layer_temperature[0]=cast(Float32Array,observed_surface_temperature)
            observed_profile_height=np.broadcast_to(
                np.r_[np.float32(0.),centers][:,None,None],
                (centers.size+1,config.grid.ny,config.grid.nx))
            observed_profile_temperature=np.concatenate(
                (cast(Float32Array,observed_surface_temperature)[None],
                 layer_temperature),axis=0)
        observation_changes: dict[str,Any]={}
        if config.itprog in {0,1}:
            observation_changes['surface_temperature']=cast(
                Float32Array,observed_surface_temperature)
            pressure: Float32Array=np.asarray(
                record.pressure_mb,dtype=np.float32
            )[nearest]
            observation_changes['surface_pressure_mb']=pressure
            observation_changes['surface_density']=density_from_pressure_temperature(
                pressure,observation_changes['surface_temperature'])
        if config.irhprog==0:
            water_min=config.water_temperature_category_min
            water_max=config.water_temperature_category_max
            sea_humidity: Float32Array=np.asarray(
                [item.relative_humidity for item in sea_records],dtype=np.float32,
            )
            sea_humidity=np.where(sea_humidity<9998.99,sea_humidity,100.)
            humidity_values: Float32Array=np.concatenate((
                np.asarray(raw_record.relative_humidity,dtype=np.float32),sea_humidity,
            ))
            humidity_points: Float64Array=np.vstack((points,sea_points))
            surface_humidity=interpolate_surface_scalar(
                humidity_values,humidity_points,config.grid,
                radius_km=config.temperature_radius_km,
                maximum_stations=config.temperature_station_limit,
                power=config.temperature_interpolation_power,
                target_water=(landuse>=water_min)&(landuse<=water_max),
                station_water=np.r_[
                    np.zeros(len(points),dtype=bool),np.ones(len(sea),dtype=bool)
                ],
                empty_fallback=(80.,100.),
            )
            observation_changes['surface_relative_humidity']=np.asarray(
                surface_humidity,dtype=np.float32
            )
            if observation_only:
                # SURF.DAT supplies one near-surface humidity observation,
                # not a vertical profile.  Expose it in the lowest CALMET
                # layer and mark the unobserved upper layers as missing so
                # neither the inert scaffold nor a fabricated profile leaks
                # into the modern NetCDF output.
                observed_humidity: Float32Array=np.full_like(
                    step.relative_humidity,np.nan,dtype=np.float32,
                )
                observed_humidity[0]=observation_changes[
                    'surface_relative_humidity'
                ]
                observation_changes['relative_humidity']=observed_humidity
        if config.itprog==0:
            observation_changes.update(
                temperature=cast(Float32Array,observed_profile_temperature)[1:],
                observed_temperature_faces=observed_temperature_faces,
                profile_temperature=observed_profile_temperature,
                profile_height=observed_profile_height,
                mixing_profile_temperature=mixing_profile_temperature,
                mixing_profile_height=mixing_profile_height,
            )
        precip_record: PrecipitationRecord | None=(
            None if precipitation is None else precipitation.at_begin(timestamp)
        )
        if config.precipitation_station_count==0:
            observation_changes['precipitation_code']=np.zeros_like(
                step.precipitation_code,dtype=np.int32
            )
            if observation_only:
                observation_changes['precipitation_rate']=np.zeros_like(
                    step.precipitation_rate,dtype=np.float32
                )
        elif (config.precipitation_station_count>0 and
              not (config.run_type==0 and config.diagnostic_lapse_rate)):
            station_code: Int32Array=cast(
                Int32Array,np.asarray(record.precipitation_code)
            )
            observation_changes['precipitation_code']=np.where(
                station_code[nearest]<9999,station_code[nearest],0
            ).astype(np.int32)
            observation_changes['precipitation_rate']=grid_precipitation(
                cast(PrecipitationRecord,precip_record).rate_mm_h,
                cast(npt.ArrayLike,precip_points),config.grid,
                config.precipitation_interpolation,
                config.precipitation_radius_km,config.precipitation_cutoff,
            )
        if config.cloud_method in {1,2}:
            external: Float32Array | None=(
                None if config.cloud_method==1 else step.cloud_fraction
            )
            surface_ceiling: npt.NDArray[np.generic]
            surface_cloud,surface_ceiling=surface_cloud_fields(
                record,nearest,external
            )
            observation_changes.update(
                cloud_fraction=np.asarray(surface_cloud,dtype=np.float32),
                ceiling_height=(
                    np.asarray(surface_ceiling,dtype=np.float32)*np.float32(30.48)
                ),
            )
        # Similarity-theory station profiles are part of the observation
        # transform, so they must see the same analyzed thermodynamic fields
        # that the transform ultimately returns.  In particular, an IPROG=0
        # scaffold must never act as a hidden physical background.
        physics_step: Step1Fields=replace(step,**observation_changes)
        upper_u: Float32Array | None
        upper_v: Float32Array | None
        upper_u=upper_v=None
        if upper_stations:
            end=timestamp+timedelta(
                seconds=config.timestep_seconds,hours=config.base_timezone
            )
            if upper_air:
                upper_profiles: Sequence[tuple[Float32Array,Float32Array]]=[
                    data.layer_wind_at(end,config.z_faces_m,angle)
                    for data,angle in zip(upper_air,upper_convergence)
                ]
                upper_u=np.column_stack([item[0] for item in upper_profiles])
                upper_v=np.column_stack([item[1] for item in upper_profiles])
            else:
                upper_profiles=()
            if diagnostic_record is not None and diagnostic_record.upper_u_m_s is not None:
                upper_u=np.asarray(diagnostic_record.upper_u_m_s,dtype=np.float32).copy()
                upper_v=np.asarray(diagnostic_record.upper_v_m_s,dtype=np.float32).copy()
            elif not upper_air:
                raise ValueError('IDIOPT5 requires upper-air wind records in DIAG.DAT')
            upper_valid: npt.NDArray[np.bool_]=np.any(
                (cast(Float32Array,upper_u)<900.)|
                (cast(Float32Array,upper_v)<900.),axis=0
            )
            if config.micrometeorology_debug and upper_air:
                debug_upper: list[object] | tuple[object,...]=[]
                for data,angle,(layer_u,layer_v) in zip(
                        upper_air,upper_convergence,upper_profiles):
                    if data.soundings[0].end_timestamp is not None:
                        soundings=tuple(item for item in data.soundings
                            if item.timestamp<end<=cast(datetime,item.end_timestamp))
                    else:
                        soundings=data.bracket(end)
                    cast(list[object],debug_upper).append((
                        data.station,soundings,float(angle),
                        np.asarray(layer_u,dtype=np.float32).copy(),
                        np.asarray(layer_v,dtype=np.float32).copy(),
                    ))
                debug_upper=tuple(debug_upper)
            else:
                debug_upper=()
        else:
            upper_valid=np.empty(0,dtype=bool)
            debug_upper=()
        sea_speed: Float32Array=np.asarray(
            [item.wind_speed_m_s for item in sea_records],dtype=np.float32
        )
        sea_direction: Float32Array=np.asarray(
            [item.wind_direction_deg for item in sea_records],dtype=np.float32
        )
        sea_anemometer: Float32Array=np.asarray(
            [item.anemometer_height_m for item in sea_records],dtype=np.float32
        )
        sea_anemometer=np.where(sea_anemometer>9998.,10.,sea_anemometer)
        sea_roughness: Float32Array=hosker_roughness(np.where(
            (sea_speed>0.)&(sea_speed<9998.),sea_speed,0.))
        wind_speed: Float32Array=np.concatenate((
            np.asarray(record.wind_speed,dtype=np.float32),sea_speed,
        )).astype(np.float32,copy=False)
        wind_direction: Float32Array=np.concatenate((
            np.asarray(record.wind_direction,dtype=np.float32),sea_direction,
        )).astype(np.float32,copy=False)
        wind_anemometer: Float32Array=np.concatenate((anemometer,sea_anemometer)).astype(
            np.float32,copy=False,
        )
        wind_roughness: Float32Array=np.concatenate((
            roughness,np.asarray(sea_roughness,dtype=np.float32),
        )).astype(np.float32,copy=False)
        wind_latitude: Float64Array=np.concatenate((latitude,np.asarray(
            [item[1] for item in sea_locations],dtype=float)))
        wind_convergence: Float64Array=np.concatenate((convergence,sea_convergence))
        surface_relative: Float64Array=np.vstack((relative,sea_points-np.array(
            [config.grid.x_origin_km,config.grid.y_origin_km]))) if sea else relative
        power_exponent: Float32Array=np.r_[np.full(len(record.wind_speed),.143,dtype=np.float32),
                             np.full(len(sea_records),.286,dtype=np.float32)]
        profile_options: dict[str,Any]=dict(station_xy=surface_relative,
            upper_xy=(np.asarray(upper_points)-np.array(
                [config.grid.x_origin_km,config.grid.y_origin_km]) if upper_points else None),
            upper_valid=upper_valid,
            upper_air_separation_cells=config.upper_air_separation_cells,
            dx_km=config.grid.dx_km,dy_km=config.grid.dy_km,
            power_exponent=power_exponent)
        if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
            observed_u: Float32Array; observed_v: Float32Array
            observed_u,observed_v=component_wind_profiles(
                diagnostic_record.surface_u_m_s,
                cast(npt.ArrayLike,diagnostic_record.surface_v_m_s),
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
            station_landuse: npt.NDArray[np.int16]=landuse[station_iy,station_ix]
            water_min,water_max=common['water_range']
            station_count: int=len(record.wind_speed)
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                raw_u: Float32Array=np.asarray(
                    diagnostic_record.surface_u_m_s,dtype=np.float32
                )
                raw_v: Float32Array=np.asarray(
                    diagnostic_record.surface_v_m_s,dtype=np.float32
                )
            else:
                raw_u,raw_v,_,_=_wind_components(
                    wind_speed,wind_direction,wind_convergence)
            station_cloud: Float32Array=(np.asarray(
                physics_step.cloud_fraction,dtype=np.float32
            )[station_iy,station_ix]
                           if config.cloud_method>1 else
                           np.asarray(record.cloud_tenths,dtype=np.float32)*np.float32(.1))
            station_temperature: Float32Array=(
                np.asarray(record.temperature_k,dtype=np.float32)
                if config.overwater_flux_method==0 and config.itprog<2 else
                cast(Float32Array,observed_surface_temperature)[station_iy,station_ix]
                if config.itprog<2 else
                np.asarray(
                    physics_step.surface_temperature,dtype=np.float32
                )[station_iy,station_ix])
            station_heat: Float32Array
            station_heat,_=heat_flux(
                physical_context.solar_elevation_sine[station_iy,station_ix],station_cloud,
                station_temperature,station_landuse,
                np.asarray(geo['albedo'][:],dtype=np.float32)[station_iy,station_ix],
                np.asarray(geo['bowen_ratio'][:],dtype=np.float32)[station_iy,station_ix],
                np.asarray(geo['soil_heat_flux_parameter'][:],dtype=np.float32)[station_iy,station_ix],
                np.asarray(geo['anthropogenic_heat_flux'][:],dtype=np.float32)[station_iy,station_ix],
                water_range=common['water_range'],**radiation_options)
            mixing_options: dict[str,Any]=dict(timestep_s=config.timestep_seconds,
                depth_m=config.mixing_lapse_depth_m,
                minimum_lapse=config.minimum_lapse_rate,
                minimum_height_m=config.minimum_mixing_height_m,
                maximum_height_m=config.maximum_mixing_height_m,
                threshold_flux_per_m=config.land_flux_threshold,
                constb=config.constb,constn=config.constn,
                relaxation=bool(config.convective_relaxation),
                relaxation_time_s=config.convective_relaxation_time_s)
            if abs(config.imixh)==1: mixing_options['conste']=config.conste
            ipsifcn: int=1 if config.overwater_flux_method==0 else 0
            def ordered_water_profiles(
                function: Callable[...,tuple[Any,...]],
                arguments: tuple[object,...],
                keywords: Mapping[str,object],
                grid_iy: npt.ArrayLike | None = None,
                grid_ix: npt.ArrayLike | None = None,
            ) -> tuple[Float32Array,Float32Array]:
                """Evaluate water-station profiles with isolated warm-layer state.

                Args:
                    function: Water-profile solver, either the delta-temperature
                        implementation or the COARE implementation.
                    arguments: Positional arguments forwarded to ``function``.
                    keywords: Keyword arguments forwarded to ``function``.
                    grid_iy: Optional integer y indices of the water stations.
                    grid_ix: Optional integer x indices of the water stations.

                Returns:
                    Pair ``(u, v)`` taken from the solver result. Each is a
                    ``float32`` wind-component array in metres per second with
                    shape ``(nz, station)``.

                Raises:
                    RuntimeError: If COARE warm-layer processing is enabled
                        without station grid indices.

                Notes:
                    The routine may initialize the grid warm-layer state on
                    ``physical_context``. Station solvers receive copies of
                    the selected state so they cannot advance the grid state
                    before the later full-grid surface solve.
                """
                warm: bool | int=(
                    config.warm_layer and function is coare_water_station_profiles
                )
                station_keywords: dict[str,Any]=dict(keywords)
                if warm:
                    if grid_iy is None or grid_ix is None:
                        raise RuntimeError('COARE warm-layer stations require grid indices')
                    state: WarmLayerState | None=physical_context.warm_layer_state
                    if state is None:
                        state=initial_warm_layer_state(landuse.shape)
                        physical_context.warm_layer_state=state
                    iy: IntpArray=np.asarray(grid_iy,dtype=np.intp)
                    ix: IntpArray=np.asarray(grid_ix,dtype=np.intp)
                    station_keywords['warm_layer_state']=WarmLayerState(*(
                        np.array(np.asarray(getattr(state,name))[iy,ix],copy=True)
                        for name in WarmLayerState.__dataclass_fields__))
                    # Station profiles sample the previous grid state.  The
                    # later full-grid surface solve advances that state once.
                station_profiles: tuple[Float32Array,...]=function(
                    *arguments,**station_keywords
                )
                return station_profiles[0],station_profiles[1]

            surface_water: npt.NDArray[np.bool_]=(
                (station_landuse>=water_min)&(station_landuse<=water_max)
            )
            surface_land: npt.NDArray[np.bool_]=~surface_water
            if surface_land.any():
                selected: int | IntpArray | npt.NDArray[np.bool_]=np.flatnonzero(
                    surface_land)
                station_mixing_temperature: Float32Array=np.asarray(
                    physics_step.profile_temperature
                    if physics_step.mixing_profile_temperature is None else
                    physics_step.mixing_profile_temperature,
                    dtype=np.float32,
                )
                station_mixing_height: Float32Array=np.asarray(
                    physics_step.profile_height
                    if physics_step.mixing_profile_height is None else
                    physics_step.mixing_profile_height,
                    dtype=np.float32,
                )
                land_profiles: tuple[Float32Array,...]=land_station_similarity_theory(
                    raw_u[selected],raw_v[selected],roughness[selected],anemometer[selected],
                    centers[0],np.asarray(
                        physics_step.surface_density,dtype=np.float32
                    )[station_iy[selected],station_ix[selected]],
                    station_temperature[selected],station_cloud[selected],station_heat[selected],
                    station_mixing_temperature[
                        :,station_iy[selected],station_ix[selected]
                    ],
                    station_mixing_height[
                        :,station_iy[selected],station_ix[selected]
                    ],
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
                    water_profiles: tuple[Float32Array,...]=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],anemometer[selected],
                        physical_context.previous_mixing_height[station_iy[selected],station_ix[selected]],
                        physical_context.coriolis[station_iy[selected],station_ix[selected]],centers,
                        config.z_faces_m,latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                else:
                    if physics_step.overwater_data is None:
                        raise ValueError('ITWPROG=2 requires prognostic offshore data')
                    iy: int | IntpArray=station_iy[selected]
                    ix: int | IntpArray=station_ix[selected]
                    offshore: GriddedSeaData=physics_step.overwater_data
                    air: Float32Array=np.asarray(
                        offshore.air_temperature_k,dtype=np.float32
                    )[iy,ix]
                    sea_temperature: Float32Array=np.asarray(
                        offshore.sea_temperature_k,dtype=np.float32
                    )[iy,ix]
                    delta: Float32Array=air-sea_temperature
                    humidity: Float32Array=np.asarray(
                        offshore.relative_humidity,dtype=np.float32
                    )[iy,ix]
                    thermal_height: Float32Array | None=np.asarray(
                        offshore.air_sensor_height_m,dtype=np.float32
                    )[iy,ix]
                    preliminary: tuple[Float32Array,...]=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],anemometer[selected],
                        physical_context.previous_mixing_height[iy,ix],
                        physical_context.coriolis[iy,ix],centers,config.z_faces_m,
                        latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                    potential_lapse: Float32Array
                    _,_,potential_lapse=profile_temperature_layer(
                        np.asarray(
                            physics_step.profile_temperature,dtype=np.float32
                        )[:,iy,ix][:,None,:],
                        np.asarray(
                            physics_step.profile_height,dtype=np.float32
                        )[:,iy,ix][:,None,:],
                        preliminary[5][None,:],config.mixing_lapse_depth_m,
                        minimum_lapse=config.minimum_lapse_rate)
                    water_common: dict[str,Any]=dict(observed_mixing_height_m=np.full(selected.size,9999.,dtype=np.float32),
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
                        function: Callable[...,tuple[Any,...]]=delta_temperature_water_station_profiles
                        arguments: tuple[object,...]=(raw_u[selected],raw_v[selected],anemometer[selected],
                                   air,delta,humidity)
                        keywords: dict[str,Any]=dict(water_common,
                            temperature_measurement_height_m=thermal_height,
                            sea_temperature_k=sea_temperature)
                    else:
                        cloud: Float32Array=np.asarray(
                            physics_step.cloud_fraction,dtype=np.float32
                        )[iy,ix]
                        shortwave: Float32Array
                        _,shortwave=heat_flux(
                            physical_context.solar_elevation_sine[iy,ix],cloud,
                            np.asarray(
                                physics_step.surface_temperature,dtype=np.float32
                            )[iy,ix],landuse[iy,ix],
                            np.asarray(geo['albedo'][:],dtype=np.float32)[iy,ix],
                            np.asarray(geo['bowen_ratio'][:],dtype=np.float32)[iy,ix],
                            np.asarray(geo['soil_heat_flux_parameter'][:],dtype=np.float32)[iy,ix],
                            np.asarray(geo['anthropogenic_heat_flux'][:],dtype=np.float32)[iy,ix],
                            water_range=common['water_range'],**radiation_options)
                        longwave: Float32Array=downward_longwave(
                            np.asarray(
                                physics_step.surface_temperature,dtype=np.float32
                            )[iy,ix],
                            np.asarray(
                                physics_step.surface_relative_humidity
                            )[iy,ix],cloud,
                            np.asarray(
                                physics_step.ceiling_height,dtype=np.float32
                            )[iy,ix])
                        coast: Float32Array=distance_to_coast(landuse,common['water_range'],
                                                config.grid.dx_km,config.grid.dy_km)
                        function=coare_water_station_profiles
                        arguments=(raw_u[selected],raw_v[selected],anemometer[selected],
                                   thermal_height,air,delta,humidity)
                        keywords=dict(water_common,sea_temperature_k=sea_temperature,
                            shortwave_down_w_m2=shortwave,longwave_down_w_m2=longwave,
                            rainfall_mm_h=np.asarray(
                                physics_step.precipitation_rate,dtype=np.float32
                            )[iy,ix],
                            coare_method=config.overwater_flux_method,
                            coastal_shelf_length_km=config.coastal_shelf_length_km,
                            distance_to_coast_km=coast[iy,ix],cool_skin=bool(config.cool_skin),
                            solar_time_s=np.mod(np.asarray(offshore.station_longitude)[iy,ix]/15.
                                                +(physical_context.gmt_hour or 0)+24.,24.)*3600.,
                            water_sensor_depth_m=.05)
                    water_profiles=ordered_water_profiles(
                        function,arguments,keywords,iy,ix
                    )
                observed_u[:,selected]=water_profiles[0]; observed_v[:,selected]=water_profiles[1]
            if sea_records:
                if config.water_temperature_mode not in {0,1,2}:
                    raise ValueError('ITWPROG must be 0, 1, or 2')
                origin: Float64Array=np.array(
                    [config.grid.x_origin_km,config.grid.y_origin_km]
                )
                sea_relative: Float64Array=sea_points-origin
                sea_ix: IntpArray=np.clip(
                    np.floor(sea_relative[:,0]/config.grid.dx_km).astype(int),
                    0,config.grid.nx-1,
                )
                sea_iy: IntpArray=np.clip(
                    np.floor(sea_relative[:,1]/config.grid.dy_km).astype(int),
                    0,config.grid.ny-1,
                )
                selected=np.arange(
                    station_count,station_count+len(sea_records)
                )
                thermal_height=None
                if config.water_temperature_mode in {0,1}:
                    sea_air: Float32Array=np.asarray(
                        [item.air_temperature_k for item in sea_records],
                        dtype=np.float32,
                    )
                    sea_delta: Float32Array=np.asarray(
                        [item.air_water_delta_k for item in sea_records],
                        dtype=np.float32,
                    )
                    sea_rh: Float32Array=np.asarray(
                        [item.relative_humidity for item in sea_records],
                        dtype=np.float32,
                    )
                    sea_observed: Float32Array=np.asarray(
                        [item.mixing_height_m for item in sea_records],
                        dtype=np.float32,
                    )
                    gradient: Float32Array=np.asarray(
                        [item.gradient_above_k_m for item in sea_records],
                        dtype=np.float32,
                    )
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
                            np.asarray(
                                physics_step.profile_temperature,dtype=np.float32
                            )[:,sea_iy,sea_ix][:,None,:],
                            np.asarray(
                                physics_step.profile_height,dtype=np.float32
                            )[:,sea_iy,sea_ix][:,None,:],
                            preliminary[5][None,:],config.mixing_lapse_depth_m,
                            minimum_lapse=config.minimum_lapse_rate)
                        potential_lapse=potential_lapse[0]
                else:
                    if physics_step.overwater_data is None:
                        raise ValueError('ITWPROG=2 requires prognostic offshore data')
                    offshore=physics_step.overwater_data
                    sea_air=np.asarray(offshore.air_temperature_k,dtype=np.float32)[sea_iy,sea_ix]
                    sea_temperature=np.asarray(offshore.sea_temperature_k,dtype=np.float32)[sea_iy,sea_ix]
                    sea_delta=sea_air-sea_temperature
                    sea_rh=np.asarray(offshore.relative_humidity,dtype=np.float32)[sea_iy,sea_ix]
                    sea_observed=np.full(len(sea_records),9999.,dtype=np.float32)
                    thermal_height=np.asarray(offshore.air_sensor_height_m,dtype=np.float32)[sea_iy,sea_ix]
                    preliminary=neutral_water_station_profiles(
                        raw_u[selected],raw_v[selected],sea_anemometer,
                        physical_context.previous_mixing_height[sea_iy,sea_ix],
                        physical_context.coriolis[sea_iy,sea_ix],centers,
                        config.z_faces_m,wind_latitude[selected],constw=config.constw,
                        minimum_height_m=config.minimum_water_mixing_height_m,
                        maximum_height_m=config.maximum_water_mixing_height_m,ipsifcn=ipsifcn)
                    _,_,potential_lapse=profile_temperature_layer(
                        np.asarray(
                            physics_step.profile_temperature,dtype=np.float32
                        )[:,sea_iy,sea_ix][:,None,:],
                        np.asarray(
                            physics_step.profile_height,dtype=np.float32
                        )[:,sea_iy,sea_ix][:,None,:],
                        preliminary[5][None,:],config.mixing_lapse_depth_m,
                        minimum_lapse=config.minimum_lapse_rate)
                    potential_lapse=potential_lapse[0]
                common_water: dict[str,Any]=dict(
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
                    profile_function: Callable[...,tuple[Any,...]]=delta_temperature_water_station_profiles
                    profile_arguments: tuple[object,...]=(raw_u[selected],raw_v[selected],sea_anemometer,
                                       sea_air,sea_delta,sea_rh)
                    profile_keywords: dict[str,Any]=(common_water if thermal_height is None else
                        dict(common_water,temperature_measurement_height_m=thermal_height,
                             sea_temperature_k=sea_temperature))
                else:
                    cloud=np.asarray(
                        physics_step.cloud_fraction,dtype=np.float32
                    )[sea_iy,sea_ix]
                    _,shortwave=heat_flux(
                        physical_context.solar_elevation_sine[sea_iy,sea_ix],cloud,
                        np.asarray(
                            physics_step.surface_temperature,dtype=np.float32
                        )[sea_iy,sea_ix],
                        landuse[sea_iy,sea_ix],
                        np.asarray(geo['albedo'][:],dtype=np.float32)[sea_iy,sea_ix],
                        np.asarray(geo['bowen_ratio'][:],dtype=np.float32)[sea_iy,sea_ix],
                        np.asarray(geo['soil_heat_flux_parameter'][:],dtype=np.float32)[sea_iy,sea_ix],
                        np.asarray(geo['anthropogenic_heat_flux'][:],dtype=np.float32)[sea_iy,sea_ix],
                        water_range=common['water_range'],**radiation_options)
                    longwave=downward_longwave(
                        np.asarray(
                            physics_step.surface_temperature,dtype=np.float32
                        )[sea_iy,sea_ix],
                        np.asarray(
                            physics_step.surface_relative_humidity
                        )[sea_iy,sea_ix],cloud,
                        np.asarray(
                            physics_step.ceiling_height,dtype=np.float32
                        )[sea_iy,sea_ix])
                    coast=distance_to_coast(landuse,common['water_range'],
                                            config.grid.dx_km,config.grid.dy_km)
                    if thermal_height is None:
                        air_height: Float32Array=np.asarray(
                            [item.air_sensor_height_m for item in sea_records],
                            dtype=np.float32,
                        )
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
                        rainfall_mm_h=np.asarray(
                            physics_step.precipitation_rate,dtype=np.float32
                        )[sea_iy,sea_ix],
                        coare_method=config.overwater_flux_method,
                        wave_period_s=np.asarray(
                            [item.wave_period_s for item in sea_records],dtype=np.float32),
                        wave_height_m=np.asarray(
                            [item.wave_height_m for item in sea_records],dtype=np.float32),
                        coastal_shelf_length_km=config.coastal_shelf_length_km,
                        distance_to_coast_km=coast[sea_iy,sea_ix],cool_skin=bool(config.cool_skin),
                        solar_time_s=np.mod((np.asarray(offshore.station_longitude)[sea_iy,sea_ix]
                            if thermal_height is not None else
                            np.asarray([item[0] for item in sea_locations]))/15.
                            +(physical_context.gmt_hour or 0)+24.,24.)*3600.,
                        water_sensor_depth_m=(np.float32(.05)
                            if thermal_height is not None else np.where(
                                np.asarray([item.water_sensor_depth_m
                                            for item in sea_records],dtype=np.float32)<9998.,
                                np.asarray([item.water_sensor_depth_m
                                            for item in sea_records],dtype=np.float32),
                                np.float32(.6))),
                    )
                sea_profiles: tuple[Float32Array,Float32Array]=ordered_water_profiles(
                    profile_function,profile_arguments,profile_keywords,sea_iy,sea_ix)
                observed_u[:,selected]=sea_profiles[0]; observed_v[:,selected]=sea_profiles[1]
            skip: npt.NDArray[np.bool_]=_near_upper_air(len(wind_speed),surface_relative,
                profile_options['upper_xy'],upper_valid,config.upper_air_separation_cells,
                config.grid.dx_km,config.grid.dy_km)
            neutral_u: Float32Array; neutral_v: Float32Array
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
                tinf: float | None=float(
                    record.temperature_k[config.surface_temperature_option-1]
                )
            elif config.surface_temperature_option==-2:
                tinf=float(_domain_mean(step.temperature[0]))
            else:
                tinf=None
            if (diagnostic_record is not None and
                    diagnostic_record.initial_u_m_s is not None):
                domain_u: float=float(diagnostic_record.initial_u_m_s)
                domain_v: float=float(cast(float,diagnostic_record.initial_v_m_s))
            elif config.noobs==0 and config.initial_guess_upper_station>0:
                selected=config.initial_guess_upper_station-1
                if upper_air:
                    domain_profile: tuple[Float32Array,Float32Array]=upper_air[
                        selected
                    ].layer_wind_at(
                        end,config.initial_guess_height_range_m,
                        upper_convergence[selected])
                    domain_u=float(domain_profile[0][0])
                    domain_v=float(domain_profile[1][0])
                else:
                    valid: npt.NDArray[np.bool_]=(
                        (cast(Float32Array,upper_u)[:,selected]<900.)&
                        (cast(Float32Array,upper_v)[:,selected]<900.)
                    )
                    domain_u=float(_domain_mean(
                        cast(Float32Array,upper_u)[valid,selected]))
                    domain_v=float(_domain_mean(
                        cast(Float32Array,upper_v)[valid,selected]))
            else:
                domain_u=domain_v=0.
            debug_wind: dict[str,Any]=dict(
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
            changes_water_mixing: Float32Array=grid_sea_mixing_height(
                [item.mixing_height_m for item in sea_records],sea_points,config.grid)
        if upper_stations:
            surface_names: set[str]={
                str(station.name).strip().upper()
                for station in surface_stations
            }
            omit: npt.NDArray[np.bool_]=np.array([config.wind_extrapolation<=0 or
                           station.name.strip().upper() in surface_names
                           for station in upper_stations])
            cast(Float32Array,upper_u)[0,omit]=9999.
            cast(Float32Array,upper_v)[0,omit]=9999.
            observed_u=np.hstack((observed_u,cast(Float32Array,upper_u)))
            observed_v=np.hstack((observed_v,cast(Float32Array,upper_v)))
        analysis_common: dict[str,Any]=common
        if config.iprog in {5,15}:
            if step.prognostic_observation_strength is None:
                raise ValueError(f'IPROG={config.iprog} requires prognostic observation strength')
            prognostic_strength: Float32Array=np.broadcast_to(
                step.prognostic_observation_strength,step.u.shape)
            analysis_common={**common,'observation_weight':weights,
                'background_weight':(
                    1.-cast(Float32Array,weights))*prognostic_strength}
        background_u: Float32Array
        background_v: Float32Array
        background_u,background_v=step.u,step.v
        if initial_guess is not None:
            background_u,background_v=_initial_guess_at(
                config,initial_guess,timestamp,centers)
        elif (config.iprog==0 and not config.diagnostic_initial_wind and
              config.initial_guess_upper_station>0):
            station_index: int=config.initial_guess_upper_station-1
            if upper_air:
                end=timestamp+timedelta(seconds=config.timestep_seconds,
                                        hours=config.base_timezone)
                initial_u: float | Float32Array
                initial_v: float | Float32Array
                initial_u,initial_v=upper_air[station_index].layer_wind_at(
                    end,config.initial_guess_height_range_m,
                    upper_convergence[station_index])
                initial_u=float(initial_u[0]); initial_v=float(initial_v[0])
            else:
                selected_u: Float32Array=np.asarray(
                    cast(Float32Array,upper_u)[:,station_index],dtype=np.float32
                )
                selected_v: Float32Array=np.asarray(
                    cast(Float32Array,upper_v)[:,station_index],dtype=np.float32
                )
                bottom: float; top: float
                bottom,top=config.initial_guess_height_range_m
                selected=(
                    (centers>=bottom)&(centers<=top)&
                    (selected_u<900.)&(selected_v<900.)
                )
                if not selected.any():
                    selected=(selected_u<900.)&(selected_v<900.)
                if not selected.any():
                    raise ValueError('DIAG.DAT has no valid upper-air wind for IUPWND')
                initial_u=float(_domain_mean(selected_u[selected]))
                initial_v=float(_domain_mean(selected_v[selected]))
            background_u=np.full_like(step.u,initial_u,dtype=np.float32)
            background_v=np.full_like(step.v,initial_v,dtype=np.float32)
        elif config.iprog==0 and not config.diagnostic_initial_wind:
            wind1_roughness: Float32Array=wind_roughness.copy()
            land_count: int=len(record.wind_speed)
            station_landuse=landuse[station_iy,station_ix]
            land_on_water: npt.NDArray[np.bool_]=((station_landuse>=common['water_range'][0])&
                           (station_landuse<=common['water_range'][1]))
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                land_u: Float32Array=np.asarray(
                    diagnostic_record.surface_u_m_s[:land_count],dtype=np.float32
                )
                land_v: Float32Array=np.asarray(
                    cast(Float64Array,diagnostic_record.surface_v_m_s)[:land_count],
                    dtype=np.float32
                )
                land_speed: Float32Array=np.hypot(land_u,land_v)
                valid_land: npt.NDArray[np.bool_]=(
                    (land_u<=900.)&(land_v<=900.)&(land_speed>0.)
                )
            else:
                land_u,land_v,valid_land,_=_wind_components(
                    record.wind_speed,record.wind_direction,convergence)
                land_speed=np.hypot(land_u,land_v)
                valid_land&=land_speed>0.
            water_wind: npt.NDArray[np.bool_]=land_on_water&valid_land
            dynamic_z0: Float32Array=hosker_roughness(land_speed[water_wind])
            land_roughness: Float32Array=wind1_roughness[:land_count]
            land_roughness[water_wind]=dynamic_z0
            roughness_override: Float32Array=np.asarray(
                geo['roughness_length'][:],dtype=np.float32
            ).copy()
            for x_index,y_index,value in zip(station_ix[water_wind],station_iy[water_wind],dynamic_z0):
                roughness_override[y_index,x_index]=value
            if diagnostic_record is not None and diagnostic_record.surface_u_m_s is not None:
                initial_u,initial_v=component_wind_profiles(
                    diagnostic_record.surface_u_m_s,
                    cast(npt.ArrayLike,diagnostic_record.surface_v_m_s),
                    wind_anemometer,centers,wind1_roughness,config.wind_extrapolation,
                    config.wind_extrapolation_factors,wind_latitude,
                    power_exponent=power_exponent)
            else:
                initial_u,initial_v=wind_profiles(wind_speed,wind_direction,wind_anemometer,
                    centers,wind1_roughness,config.wind_extrapolation,
                    config.wind_extrapolation_factors,wind_latitude,wind_convergence,
                    power_exponent=power_exponent)
            if upper_stations:
                initial_u=np.hstack((initial_u,cast(Float32Array,upper_u)))
                initial_v=np.hstack((initial_v,cast(Float32Array,upper_v)))
            bias: Float32Array=np.asarray(
                cast(Sequence[float],config.wind_bias)[:centers.size],
                dtype=np.float32
            ).copy()
            if config.wind_extrapolation<0.: bias[0]=-1.
            if abs(config.wind_extrapolation)==1: bias[1:]=1.
            background_u,background_v=wind1_initial_guess(initial_u,initial_v,all_relative,
                len(wind_speed),config.grid.nx,config.grid.ny,config.grid.dx_km,
                config.grid.dy_km,bias=bias,
                extrapolate_calm=bool(config.extrapolate_calm),barriers=config.barriers,
                barrier_top_level=cast(int,config.barrier_top_level))
        elif config.iprog==0:
            background_u=np.full_like(
                step.u,cast(DiagnosticRecord,diagnostic_record).initial_u_m_s,
                dtype=np.float32)
            background_v=np.full_like(
                step.v,cast(DiagnosticRecord,diagnostic_record).initial_v_m_s,
                dtype=np.float32)
        u: Float32Array=objective_analysis_component(observed_u,all_relative,background_u,landuse,
            config.grid.dx_km,config.grid.dy_km,**analysis_common)
        v: Float32Array=objective_analysis_component(observed_v,all_relative,background_v,landuse,
            config.grid.dx_km,config.grid.dy_km,**analysis_common)
        if config.lake_breeze_regions:
            u,v=lake_breeze_adjust(
                u,v,observed_u[0],observed_v[0],all_relative*1000.,
                cast(npt.ArrayLike,all_station_ids),
                config.lake_breeze_regions,config.grid.dx_km*1000.,config.grid.dy_km*1000.)
        changes: dict[str,Any]=dict(observation_changes,u=u,v=v)
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
                    step.surface_temperature,diagnostic_record.surface_temperature_k,dtype=np.float32)
            if diagnostic_record.lapse_rate_k_m is not None:
                changes['diagnostic_lapse_rate']=np.full_like(
                    step.surface_temperature,diagnostic_record.lapse_rate_k_m,dtype=np.float32)
        if config.surface_temperature_option>0 and not config.diagnostic_surface_temperature:
            index: int=config.surface_temperature_option-1
            if index>=len(surface_stations): raise ValueError('ISURFT exceeds the SURF.DAT station count')
            changes['diagnostic_surface_temperature']=np.full_like(
                step.surface_temperature,record.temperature_k[index],dtype=np.float32)
        if (config.upper_temperature_option>0 and not config.diagnostic_lapse_rate and
                upper_air):
            index=config.upper_temperature_option-1
            if index>=len(upper_air): raise ValueError('IUPT exceeds the UPDAT station count')
            end=timestamp+timedelta(seconds=config.timestep_seconds,hours=config.base_timezone)
            temperatures: Float32Array=upper_air[index].temperature_faces_at(
                end,(0.,config.lapse_height_m)
            )
            gamma: np.float32=(
                (temperatures[1]-temperatures[0])/config.lapse_height_m
            )
            point: Float64Array=np.asarray(upper_points[index])-np.array(
                [config.grid.x_origin_km,config.grid.y_origin_km])
            ix=int(np.clip(
                np.floor(point[0]/config.grid.dx_km),0,config.grid.nx-1
            ))
            iy=int(np.clip(
                np.floor(point[1]/config.grid.dy_km),0,config.grid.ny-1
            ))
            changes['diagnostic_lapse_rate']=np.full_like(
                step.surface_temperature,gamma,dtype=np.float32,
            )
            changes['diagnostic_lapse_station_index']=(iy,ix)
            changes['diagnostic_lapse_data']=upper_air[index]
            changes['diagnostic_lapse_timestamp']=end
        if sea_records:
            changes['observed_water_mixing_height']=changes_water_mixing
            observed_water: GriddedSeaData | None=nearest_sea_data(
                sea_records,sea_points,[item[0] for item in sea_locations],config.grid)
            if config.water_temperature_mode==2:
                if step.overwater_data is None:
                    raise ValueError('ITWPROG=2 requires prognostic offshore data')
                wave_period: Float32Array; wave_height: Float32Array
                wave_period,wave_height=nearest_sea_waves(
                    sea_records,sea_points,config.grid)
                changes['overwater_data']=replace(step.overwater_data,
                    wave_period_s=wave_period,wave_height_m=wave_height)
            else:
                changes['overwater_data']=observed_water
        if sea and config.water_temperature_mode==0:
            below: Float32Array=np.asarray(
                [getattr(item,'gradient_below_k_m',9999.) for item in sea_records],
                dtype=np.float32,
            )
            above: Float32Array=np.asarray(
                [getattr(item,'gradient_above_k_m',9999.) for item in sea_records],
                dtype=np.float32,
            )
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
        return replace(step,**changes)
    cast(Any,transform).uses_physical_context=True
    return transform,nearest

def run(config: CalmetConfig) -> Path:
    """Run the configured CALMET diagnostic and objective-analysis workflow.

    Args:
        config: Fully normalized ``CalmetConfig`` containing input paths,
            target grid, inclusive/exclusive run interval, physics controls,
            observation choices, and output selections.

    Returns:
        Primary output :class:`Path`. This is the configured NetCDF,
        traditional CALMET, or MESOPAC path; with ``save_output=False`` it is
        still returned after the model and requested diagnostics complete.

    Raises:
        FileNotFoundError: If a configured meteorology or required geophysical
            input path does not exist.
        ValueError: If controls or input metadata are inconsistent, records do
            not bracket every requested boundary, required observations are
            missing, or a selected model path cannot be constructed.
        OSError: If an input cannot be read or an output, diagnostic, listing,
            auxiliary, or temporary file cannot be created.

    Notes:
        The routine opens all readers and writers, advances stateful hourly or
        subhourly fields, and closes every optional output on normal and
        exceptional exits. Model arrays are normally ``float32`` with
        horizontal ``(ny, nx)`` and vertical ``(nz, ny, nx)`` layouts.
    """
    faces: Float32Array=_validate(config)
    meteorology: tuple[Path,...]=(
        () if config.meteorology is None else
        (Path(config.meteorology),) if isinstance(config.meteorology,(str,Path)) else
        tuple(Path(path) for path in config.meteorology)
    )
    if not meteorology and config.iprog!=0:
        raise ValueError(f'IPROG={config.iprog} requires meteorology input')
    if not meteorology and config.cloud_method in {3,4}:
        raise ValueError(
            f'MCLOUD={config.cloud_method} requires meteorology input; '
            'pure-observation IPROG=0 uses MCLOUD=1 or CLOUD.DAT'
        )
    for path in meteorology:
        if not path.is_file(): raise FileNotFoundError(path)
    geophysical: Path=Path(config.geophysical)
    if not geophysical.is_file(): raise FileNotFoundError(geophysical)
    start: datetime; end: datetime
    start,end=config.start,config.end
    if end<=start: raise ValueError('CALMET end must be later than start')
    output: Path=_output_path(config); output.parent.mkdir(parents=True,exist_ok=True)
    with (_meteorology_input(config,meteorology,output.parent,start,end) as meteorology_input,
          _geophysical_dataset(geophysical,output.parent) as geo):
        met: MeteorologyInput
        step_provider: MeteorologyProvider | None
        met,step_provider=meteorology_input
        solver_met: DatasetLike=cast(DatasetLike,met)
        target_shape: tuple[int,int]=(config.grid.ny,config.grid.nx)
        if geo['terrain_height'].shape!=target_shape:
            raise ValueError('geophysical grid does not match the configured CALMET grid')
        _validate_geophysical_grid(geo,config.grid)
        times: list[datetime]=[(dataset_timestamp(solver_met,index) if step_provider is not None else
                _prognostic_local_timestamp(solver_met,index,config.base_timezone))
               for index in range(met['time_text'].shape[0])]
        source_step: int=config.prognostic_timestep_seconds
        timeline_step: int=(
            config.timestep_seconds if step_provider is not None else source_step
        )
        if any((right-left).total_seconds()!=timeline_step for left,right in zip(times[:-1],times[1:])):
            raise ValueError('CALWRF record spacing does not match configured ISTEPPGS')
        duration: float=(end-start).total_seconds()
        if duration%config.timestep_seconds:
            raise ValueError('CALMET run duration must be an integer number of NSECDT timesteps')
        if not times or start<times[0] or end>times[-1]:
            raise ValueError('CALWRF data do not bracket the requested CALMET period')
        indices: list[int] | None=None
        time_samples: list[tuple[int,int,float,datetime]] | None=None
        if config.timestep_seconds==timeline_step:
            indices=[index for index,time in enumerate(times) if start<time<=end]
            if not indices or indices[0]==0 or times[indices[0]-1]!=start or times[indices[-1]]!=end:
                raise ValueError('CALWRF data must contain every hourly CALMET boundary')
        else:
            time_samples=[]; right: int=1
            boundary: datetime=start+timedelta(seconds=config.timestep_seconds)
            while boundary<=end:
                while right<len(times) and times[right]<boundary: right+=1
                if right>=len(times): raise ValueError('CALWRF data end before the requested CALMET period')
                left: int=right-1
                fraction: float=(
                    (boundary-times[left]).total_seconds()/timeline_step
                )
                time_samples.append((left,right,fraction,boundary))
                boundary+=timedelta(seconds=config.timestep_seconds)
        geo_fields: dict[str,npt.NDArray[np.generic]]={
            name:np.asarray(
                geo[name][:],
                dtype=(np.int16 if name=='dominant_landuse' else np.float32),
            )
            for name in (
                'terrain_height','dominant_landuse','roughness_length','albedo',
                'bowen_ratio','soil_heat_flux_parameter','anthropogenic_heat_flux',
            )
        }
        diagnostic: DiagnosticData | None=None
        initial_guess: InitialGuessCalmetSeries | None=(InitialGuessCalmetSeries(config.initial_guess_calmet_files)
                       if config.use_initial_guess_calmet else None)
        if initial_guess is not None and not initial_guess.includes_w_temperature:
            raise ValueError('IGFMET requires IGFDAT written with LCALGRD=T')
        water_range: tuple[int,int]=(
            int(getattr(geo,'water_category_min',50)),
            int(getattr(geo,'water_category_max',55)),
        )
        if water_range[0]>water_range[1]: raise ValueError('geophysical water-category range is invalid')
        if config.distance_to_coast_output is not None:
            _write_distance_to_coast_grid(config.distance_to_coast_output,
                geo_fields['dominant_landuse'],config.grid,water_range)
        category_text: str=str(getattr(geo,'landuse_categories',''))
        landuse_category_count: int=(
            len([item for item in category_text.split(',') if item]) or 14
        )
        latitude: Float64Array; longitude: Float64Array
        latitude,longitude=target_latitude_longitude(config.grid)
        options: dict[str,Any]=dict(base_timezone=config.base_timezone,terrain_radius_km=config.terrain_radius_km,
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
                     cloud_condensate_threshold_g_kg=config.cloud_condensate_threshold_g_kg,
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
        surface_coordinates: Float64Array | tuple[()]=()
        upper_coordinates: Float64Array | tuple[()]=()
        precipitation_coordinates: Float64Array | tuple[()]=()
        surface: SurfaceData | None=None
        surface_points: Float64Array | None=None
        nearest_surface: Int32Array | None=None
        sea: tuple[SeaData,...]=tuple(SeaData(path) for path in config.sea_files)
        _validate_sea_metadata(config,sea)
        precipitation: PrecipitationData | None=(load_precipitation(
                           cast(str | Path,config.precipitation_data),
                           config.precipitation_data_format,
                           station_metadata=_precipitation_control_metadata(config),
                           datum=config.grid.datum)
                       if config.precipitation_station_count>0 else None)
        precipitation_points: Float64Array | None=None
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
            surface_metadata: tuple[SurfaceStation,...] | None=(
                _surface_control_metadata(config)
            )
            omit_surface_file: bool=bool(config.diagnostic_surface_wind and
                                   config.run_type==0)
            if omit_surface_file:
                surface_stations: Sequence[SurfaceStation]=cast(
                    Sequence[SurfaceStation],surface_metadata)
                surface_datum: str=config.grid.datum
            else:
                surface=load_surface(cast(str | Path,config.surface_data),
                                     config.surface_data_format,
                                     station_metadata=surface_metadata,
                                     datum=config.grid.datum)
                _validate_input_timezone(surface.timezone,config.base_timezone,'SURF.DAT')
                _validate_input_span(surface,start,end,'SURF.DAT')
                surface_stations=surface.stations
                surface_datum=surface.datum
            upper_metadata: tuple[UpperAirStation,...] | None=(
                _upper_control_metadata(config)
            )
            omit_upper_files: bool=bool(config.noobs==0 and
                                  config.diagnostic_upper_wind and
                                  config.run_type==0)
            if (config.noobs==0 and not omit_upper_files and
                    upper_metadata is not None and
                    len(upper_metadata)!=len(config.upper_air_files)):
                raise ValueError('upper-air control-station count does not match UPDAT files')
            if omit_upper_files:
                upper: tuple[UpperAirData,...]=()
                upper_stations: Sequence[UpperAirStation]=cast(
                    Sequence[UpperAirStation],upper_metadata)
            else:
                upper=(tuple(UpperAirData(path,
                                         station_metadata=(None if upper_metadata is None else upper_metadata[index]),
                                         datum=config.grid.datum)
                             for index,path in enumerate(config.upper_air_files))
                       if config.noobs==0 else ())
                upper_stations=tuple(data.station for data in upper)
            for data in upper:
                _validate_upper_span(data,start,end,config.base_timezone)
            diagnostic_options: dict[str,Any]=_diagnostic_input_options(config)
            active_diagnostic: bool=any(diagnostic_options.values())
            if active_diagnostic:
                diagnostic=DiagnosticData(
                    cast(str | Path,config.diagnostic_data),start,
                    int(duration/config.timestep_seconds),config.timestep_seconds,
                    **diagnostic_options,
                    surface_wind_names=([item.name for item in surface_stations]+
                        [item.name for item in sea] if diagnostic_options['surface_wind'] else ()),
                    upper_wind_names=([item.name for item in upper_stations]
                        if diagnostic_options['upper_wind'] else ()),
                    levels=faces.size-1,iextrp=config.wind_extrapolation)
            weights: Float32Array | None=(read_weight_data(
                cast(str | Path,config.weight_data),config.grid.nx,config.grid.ny,
                .5*(faces[:-1]+faces[1:])) if config.iprog in {3,5,13,15} and config.noobs!=2 else None)
            options['step_transform'],nearest_surface= _surface_step_transform(
                config,surface,geo,upper,precipitation,sea,weights,diagnostic,initial_guess,
                surface_stations=surface_stations,upper_stations=upper_stations)
            points: Float64Array
            points,_=station_grid_mapping(
                surface_stations,config.grid,surface_datum
            )
            surface_points=points
            surface_coordinates=points-np.array([config.grid.x_origin_km,config.grid.y_origin_km])
            if upper_stations:
                upper_coordinates=np.vstack([
                    station_grid_mapping(
                        cast(Sequence[SurfaceStation],(station,)),config.grid,
                        upper[index].datum if upper else config.grid.datum)[0][0]
                    for index,station in enumerate(upper_stations)
                ])-np.array([config.grid.x_origin_km,config.grid.y_origin_km])
        else:
            diagnostic_options=cast(
                dict[str,Any],_diagnostic_input_options(config))
            active_diagnostic=any(diagnostic_options.values())
            if active_diagnostic:
                diagnostic=DiagnosticData(
                    cast(str | Path,config.diagnostic_data),start,
                    int(duration/config.timestep_seconds),config.timestep_seconds,
                    **diagnostic_options,levels=faces.size-1,
                    iextrp=config.wind_extrapolation)
            if initial_guess is not None or diagnostic is not None or precipitation is not None:
                options['step_transform']=_noobs_step_transform(
                    config,initial_guess,diagnostic,precipitation,
                    precipitation_points,sea)
        if config.cloud_method==2:
            cloud: CloudData=CloudData(
                cast(str | Path,config.cloud_data),config.grid.nx,config.grid.ny,
                config.cloud_data_format,
            )
            options['cloud_provider']=cloud.at_end
        if step_provider is None:
            options['mapper']=PrognosticMapper.build(
                np.asarray(met['latitude'][:],dtype=np.float64),
                np.asarray(met['longitude'][:],dtype=np.float64),config.grid,
                validate_domain=True)
            options['precipitation_radius_km']=_prognostic_precipitation_radius(
                options['mapper'],config.precipitation_radius_km,
                config.precipitation_station_count)
        else:
            options['step_provider']=step_provider
        options['terrain_context']=terrain_extrema(geo_fields['terrain_height'],config.terrain_radius_km,
                                                   config.grid.dx_km*1000.,config.grid.dy_km*1000.)
        options['target_coordinates']=(latitude,longitude)
        hours: Iterator[HourFields]=iter_hours(
            solver_met,config.grid,faces,geo_fields,time_indices=indices,
            time_samples=time_samples,**options
        )
        cloud_writer: CloudDataWriter | None=(CloudDataWriter(
                                      cast(str | Path,config.cloud_data),
                                      config.grid.nx,config.grid.ny,
                                      config.cloud_data_format)
                      if config.cloud_output else None)
        def append_cloud(hour: HourFields) -> None:
            """Append one interval cloud field when cloud output is enabled.

            Args:
                hour: ``HourFields`` whose ``step1.cloud_fraction`` is a
                    dimensionless ``float32`` ``(ny, nx)`` grid.

            Notes:
                Writes the interval ``[timestamp, timestamp + timestep)`` to
                the already-open ``CloudDataWriter``; otherwise has no effect.
            """
            if cloud_writer is not None:
                cloud_writer.append(hour.timestamp,
                    hour.timestamp+timedelta(seconds=config.timestep_seconds),
                    hour.step1.cloud_fraction)
        def open_listing() -> CalmetListingWriter | None:
            """Open the optional CALMET text listing.

            Returns:
                ``CalmetListingWriter`` for the configured or derived listing
                path, or ``None`` when neither field printing nor
                micrometeorology debugging is enabled.

            Raises:
                OSError: If the listing destination cannot be opened.

            Notes:
                Opening the writer creates or replaces the listing file.
            """
            if not config.print_fields and not config.micrometeorology_debug:
                return None
            target: Path=(Path(config.listing_output) if config.listing_output is not None
                    else output.with_suffix('.lst'))
            return CalmetListingWriter(target,config,
                geo_fields['dominant_landuse'],water_range)
        def open_wind_debug(
        ) -> list[tuple[str | None,FinalWindDebugWriter]]:
            """Open requested stage-specific binary wind-debug writers.

            Returns:
                List of ``(stage_attribute, FinalWindDebugWriter)`` pairs.
                ``stage_attribute`` is ``None`` for final winds; the list is
                empty when master wind-debug output is disabled.

            Raises:
                OSError: If a selected debug destination cannot be opened.

            Notes:
                Each enabled writer creates or replaces its destination file.
            """
            if not config.wind_debug_output:
                return []
            specifications: tuple[
                tuple[int,str | None,str | Path | None,str],...
            ]=(
                (5,'kinematic_stage',config.kinematic_wind_debug_output,'test.kin'),
                (6,'froude_stage',config.froude_wind_debug_output,'test.frd'),
                (7,'slope_stage',config.slope_wind_debug_output,'test.slp'),
                (8,None,config.final_wind_debug_output,'test.out'),
            )
            writers: list[tuple[str | None,FinalWindDebugWriter]]=[]
            for index,stage,path,default in specifications:
                if config.wind_debug_fields[index]:
                    target: Path=(
                        Path(path) if path is not None else output.parent/default
                    )
                    writer: FinalWindDebugWriter=FinalWindDebugWriter(target,config.grid.nx,config.grid.ny,
                        faces.size-1,config.grid.dx_km,config.grid.dy_km)
                    writers.append((stage,writer))
            return writers
        def append_wind_debug(
            writers: Sequence[tuple[str | None,FinalWindDebugWriter]],
            hour: HourFields,
        ) -> None:
            """Append one hour to every selected wind-debug output.

            Args:
                writers: Iterable of ``(stage_attribute, writer)`` pairs from
                    ``open_wind_debug``.
                hour: ``HourFields`` containing final and optionally retained
                    stage winds, each in metres per second with shape
                    ``(nz, ny, nx)``.

            Raises:
                RuntimeError: If a requested intermediate wind stage was not
                    retained by the solver.

            Notes:
                Writes records to each supplied ``FinalWindDebugWriter``.
            """
            for stage,writer in writers:
                if stage is None:
                    writer.append(hour)
                else:
                    fields: tuple[Float32Array,...] | None=getattr(
                        hour.wind,stage
                    )
                    if fields is None:
                        raise RuntimeError(f'missing retained wind debug stage {stage}')
                    writer.append_fields(hour.timestamp,*fields)
        def open_wind_print() -> WindPrintWriter | None:
            """Open the optional formatted wind diagnostic writer.

            Returns:
                ``WindPrintWriter`` for the configured or derived path, or
                ``None`` when no wind-debug fields are selected.

            Raises:
                OSError: If the diagnostic destination cannot be opened.

            Notes:
                Opening the writer creates or replaces the output file.
            """
            if not any(config.wind_debug_fields):
                return None
            target: Path=(Path(config.wind_print_output) if config.wind_print_output is not None
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
                            sun: Float32Array=solar_at_step_midpoint(hour.timestamp,
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
        mesopac: bool=config.legacy_output_format==2
        legacy: bool=output.suffix.lower() not in {'.nc','.nc4','.cdf'}
        if mesopac:
            writer: MesopacWriter | LegacyCalmetWriter | CalmetNetCDFWriter=(
                MesopacWriter(output,config.grid,faces,geo,start,end,
                timezone_offset=config.base_timezone,
                surface_coordinates_m=np.asarray(surface_coordinates)*1000.,
                upper_coordinates_m=np.asarray(upper_coordinates)*1000.,
                nearest_surface_station=(None if nearest_surface is None else nearest_surface+1),
                water_range=water_range)
            )
        else:
            writer=(LegacyCalmetWriter(output,config.grid,faces,geo,start,end,
                    control_lines=config.control_lines,
                    base_timezone=(config.base_timezone_label or
                                   format_timezone_label(config.base_timezone)),
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
        auxiliary: AuxiliaryCalmetWriter | None=(AuxiliaryCalmetWriter(str(output)+'.aux',config.grid,faces,start,end,
                    control_lines=config.control_lines,
                    base_timezone=(config.base_timezone_label or
                                   format_timezone_label(config.base_timezone)),
                    available=_has_meteorology_variable(met,'qcloud'),
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
                    sun=solar_at_step_midpoint(
                        hour.timestamp,config.timestep_seconds,
                        config.base_timezone,latitude,longitude,
                    )
                    if listing is not None: listing.append(hour,sun)
                    append_wind_debug(wind_debug,hour)
                    if wind_print is not None: wind_print.append(hour)
                    if mesopac:
                        if surface is None:
                            station_temperature: Float32Array=np.empty(
                                0,dtype=np.float32
                            )
                            station_radiation: Float32Array=np.empty(
                                0,dtype=np.float32
                            )
                            station_humidity: Int32Array | Float32Array=np.empty(
                                0,dtype=np.int32
                            )
                            station_precipitation: Int32Array=np.empty(
                                0,dtype=np.int32
                            )
                        else:
                            record: SurfaceRecord=fill_missing_surface(
                                surface.at_begin(hour.timestamp),
                                cast(npt.ArrayLike,surface_points),
                                cloud_method=config.cloud_method,itprog=config.itprog)
                            station_temperature=record.temperature_k
                            station_humidity=record.relative_humidity
                            station_precipitation=record.precipitation_code
                            cell: npt.NDArray[np.int64]=np.trunc(np.asarray(surface_coordinates)/np.array(
                                [config.grid.dx_km,config.grid.dy_km])).astype(int)
                            ix: npt.NDArray[np.int64]=np.clip(
                                cell[:,0],0,config.grid.nx-1
                            )
                            iy: npt.NDArray[np.int64]=np.clip(
                                cell[:,1],0,config.grid.ny-1
                            )
                            station_radiation=hour.surface.shortwave_flux[iy,ix]
                        cast(MesopacWriter,writer).append(
                            hour,sun,station_temperature_k=station_temperature,
                            station_shortwave_flux=station_radiation,
                            station_relative_humidity=station_humidity,
                            station_precipitation_code=station_precipitation)
                    elif legacy:
                        cast(LegacyCalmetWriter,writer).append(hour,sun)
                    else:
                        cast(CalmetNetCDFWriter,writer).append(
                            hour,sun,np.asarray(geo['dominant_landuse'][:]))
                    if auxiliary is not None: auxiliary.append(hour)
    return output
