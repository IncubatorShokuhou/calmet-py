"""Composable state and grid mapping for the Python CALMET solver."""
from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from itertools import chain
import numpy as np
from scipy.spatial import cKDTree
from .datetime_utils import parse_timestamp
from .config import GridConfig
from .coare import (advance_warm_layer,distance_to_coast,downward_longwave,
                    initial_warm_layer_state,surface_flux,update_warm_layer_fluxes)
from .projections import (grid_convergence,project_geographic,
                          unproject_geographic)
from .calmet_numerics import (cloud_cover_mm5,cloud_fraction,cloud_liquid_water_profile,coriolis_parameter,density_from_pressure_temperature,heat_flux,mixing_height_batchvarova_gryning,mixing_height_maul_carson,
    convective_height_batchvarova_gryning,convective_height_maul_carson,
    convective_velocity_scale,delta_temperature_flux,divergence_and_vertical_velocity,froude_adjustment,lapse_rate,minimize_divergence,
    neutral_water_parameters,slope_flow,smooth_wind,solar_sine,surface_similarity,water_mechanical_height,
    overwater_delta_temperature_state,prognostic_temperature_3d,saturation_vapor_pressure_lowe,terrain_extrema,topographic_vertical_velocity,upwind_mixing_average,
    barrier_blocked_pairs,terrain_adjust_surface_wind)


# Keep model fields in single precision.  Geographic coordinates and the
# explicitly documented reduction/convergence paths remain double precision.
MODEL_DTYPE = np.float32
ACCUMULATION_DTYPE = np.float64
COORDINATE_DTYPE = np.float64


def _model_array(value):
    """Convert a physical field or model state to solver precision.

    Args:
        value: Scalar or array-like numerical data.

    Returns:
        A NumPy array with the input shape and ``float32`` model dtype.
    """
    return np.asarray(value,dtype=MODEL_DTYPE)

def solar_at_step_midpoint(begin,timestep_seconds,base_timezone,latitude,longitude):
    """Interpolate half-hour solar samples at a model-step midpoint.

    Args:
        begin: Naive local datetime at the beginning of the model step.
        timestep_seconds: Step duration in seconds.
        base_timezone: CALMET hours west of UTC.
        latitude: Cell-centre latitude array in degrees north.
        longitude: Cell-centre longitude array in degrees east, matching
            ``latitude``.

    Returns:
        ``float32`` array matching the coordinate shape and containing the
        sine of solar elevation at the midpoint, clipped by the solar model
        to nonnegative daylight values.
    """
    midpoint=begin+timedelta(seconds=timestep_seconds/2)
    samples=_model_array(solar_sine(
        midpoint.timetuple().tm_yday,base_timezone,latitude,longitude))
    midnight=midpoint.replace(hour=0,minute=0,second=0,microsecond=0)
    sample_position=(midpoint-midnight).total_seconds()/3600.+.5
    lower=int(sample_position)
    fraction=sample_position-lower
    if fraction==0.:
        return samples[lower]
    return _model_array(samples[lower]*(1.-fraction)+
                        samples[lower+1]*fraction)

def _prognostic_record_solar(record_time,base_timezone,latitude,longitude):
    """Return the half-hour solar sample for an ending-time prognostic record.

    ``solar_sine`` index zero represents 23:30 on the preceding day, so an
    ending record maps directly to ``record_time.hour`` in the table for that
    record's actual local date.

    Args:
        record_time: Naive local datetime of the interval-ending record.
        base_timezone: CALMET hours west of UTC.
        latitude: Source-grid latitude array in degrees north.
        longitude: Source-grid longitude array in degrees east.

    Returns:
        Array matching the source grid with the sine of solar elevation at
        the record's representative half-hour.
    """
    samples=solar_sine(record_time.timetuple().tm_yday,base_timezone,
                       latitude,longitude)
    return samples[record_time.hour]

def _coare_ending_gmt_hour(ending,base_timezone):
    """Convert a local interval ending to a UTC decimal hour.

    Args:
        ending: Naive datetime in the CALMET base timezone.
        base_timezone: CALMET hours west of UTC.

    Returns:
        UTC hour in the continuous range 0 through just under 24, including
        minute, second, and microsecond fractions.
    """
    utc=ending+timedelta(hours=base_timezone)
    return (utc.hour+utc.minute/60.+utc.second/3600.+
            utc.microsecond/3_600_000_000.)

def _nearest_indices(source,target,count):
    """Find nearest source points for each target coordinate.

    Args:
        source: ``(source_point, 2)`` x/y coordinates.
        target: ``(target_point, 2)`` x/y coordinates in the same units.
        count: Number of nearest source points required per target.

    Returns:
        Integer array with shape ``(target_point, count)``. A singleton
        neighbour dimension is retained when ``count`` is one.

    Raises:
        ValueError: If coordinates do not have two columns, ``count`` is
            below one, or fewer than ``count`` source points are available.
    """
    source=np.asarray(source,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1:]!=(2,) or target.shape[1:]!=(2,):
        raise ValueError('nearest-point coordinates must have shape (n, 2)')
    if count<1 or source.shape[0]<count:
        raise ValueError('not enough source points for nearest-point mapping')
    nearest=np.asarray(cKDTree(source).query(target,k=count)[1],dtype=np.intp)
    if count==1:
        nearest=nearest[:,None]
    return nearest

@dataclass(frozen=True,slots=True)
class PrognosticMapper:
    """CALMET RDHD4/R2INTERP horizontal mapper.

    Attributes:
        source_shape: ``(source_y, source_x)`` shape of prognostic fields.
        indices: ``(target_point, neighbour)`` flattened source indices.
        weights: Matching inverse-distance weights retained for mapper
            metadata; interpolation recomputes distances in coordinate space.
        target_shape: ``(target_y, target_x)`` CALMET grid shape.
        source_coordinates: ``(source_point, 2)`` projected x/y positions in
            kilometres, stored as ``float64``.
        target_coordinates: ``(target_point, 2)`` CALMET cell centres in
            kilometres, stored as ``float64``.
        grid_origin: Projected CALMET southwest origin in kilometres, used to
            create local metric coordinates for precipitation mapping.
    """
    source_shape: tuple[int,int]
    indices: np.ndarray
    weights: np.ndarray
    target_shape: tuple[int,int]
    source_coordinates: np.ndarray
    target_coordinates: np.ndarray
    grid_origin: tuple[float,float]=(0.,0.)

    def __post_init__(self):
        """Normalize coordinate arrays to double precision.

        The frozen dataclass is updated with ``object.__setattr__``; field
        values and interpolation indices otherwise remain unchanged.
        """
        object.__setattr__(self,'source_coordinates',np.asarray(
            self.source_coordinates,dtype=COORDINATE_DTYPE))
        object.__setattr__(self,'target_coordinates',np.asarray(
            self.target_coordinates,dtype=COORDINATE_DTYPE))

    @classmethod
    def build(cls,latitude,longitude,grid: GridConfig,*,validate_domain=False):
        """Construct four-neighbour mapping geometry from a source grid.

        Args:
            latitude: ``(source_y, source_x)`` latitude in degrees north.
            longitude: Matching longitude in degrees east.
            grid: Target CALMET grid definition.
            validate_domain: Check that all four target-grid corners lie
                inside the full projected source boundary.

        Returns:
            Mapper containing projected ``float64`` coordinates and the four
            nearest source indices and weights for each target cell.

        Raises:
            ValueError: If latitude and longitude shapes differ, fewer than
                four source points exist, or validation finds a target corner
                outside the prognostic domain.
        """
        lat=np.asarray(latitude); lon=np.asarray(longitude)
        if lat.shape!=lon.shape: raise ValueError('latitude/longitude shape mismatch')
        sx,sy=project_geographic(lon,lat,grid,'WGS-84',units='km')
        source=np.asarray(np.column_stack((np.ravel(sx,order='F'),
                                           np.ravel(sy,order='F'))),
                          dtype=COORDINATE_DTYPE)
        tx=(float(grid.x_origin_km)+
            (np.arange(grid.nx,dtype=COORDINATE_DTYPE)+.5)*float(grid.dx_km))
        ty=(float(grid.y_origin_km)+
            (np.arange(grid.ny,dtype=COORDINATE_DTYPE)+.5)*float(grid.dy_km))
        xx,yy=np.meshgrid(tx,ty)
        target=np.asarray(np.column_stack((xx.ravel(),yy.ravel())),
                          dtype=COORDINATE_DTYPE)
        if validate_domain:
            # RDHD4/RDHD5 trace every edge point (not merely the four source
            # corners) because a projected curvilinear grid can bow inward.
            boundary=np.concatenate((
                np.column_stack((sx[:,0],sy[:,0])),
                np.column_stack((sx[-1,1:],sy[-1,1:])),
                np.column_stack((sx[-2::-1,-1],sy[-2::-1,-1])),
                np.column_stack((sx[0,-2::-1],sy[0,-2::-1]))))
            corners=np.array(((tx[0],ty[0]),(tx[0],ty[-1]),
                              (tx[-1],ty[0]),(tx[-1],ty[-1])),dtype=float)
            start=boundary[:-1]; finish=boundary[1:]
            segment=finish-start
            length2=np.sum(segment*segment,axis=1)
            relative=corners[:,None,:]-start[None,:,:]
            fraction=np.divide(np.sum(relative*segment[None,:,:],axis=2),length2[None,:],
                               out=np.zeros((4,segment.shape[0])),where=length2[None,:]>0.)
            nearest=start[None,:,:]+np.clip(fraction,0.,1.)[:,:,None]*segment[None,:,:]
            on_edge=np.any(np.sum((corners[:,None,:]-nearest)**2,axis=2)<=1.e-6,axis=1)
            x,y=corners[:,0,None],corners[:,1,None]
            x1,y1=start[:,0][None,:],start[:,1][None,:]
            x2,y2=finish[:,0][None,:],finish[:,1][None,:]
            intersection=np.divide((x2-x1)*(y-y1),y2-y1,
                out=np.full((4,segment.shape[0]),np.inf),where=(y2-y1)!=0.)+x1
            crosses=((y1>y)!=(y2>y)) & (x<intersection)
            inside=np.logical_xor.reduce(crosses,axis=1)|on_edge
            if not np.all(inside):
                outside=np.flatnonzero(~inside)[0]+1
                raise ValueError(f'CALMET grid corner {outside} is outside prognostic domain')
        indices=_nearest_indices(source,target,4)
        delta=source[indices]-target[:,None,:]
        distances=np.sqrt(delta[:,:,0]**2+delta[:,:,1]**2)
        squared=np.maximum(distances*distances,.001); inverse=1./squared; weights=inverse/inverse.sum(axis=1,keepdims=True)
        return cls(lat.shape,indices,weights,(grid.ny,grid.nx),source,target,
                   (grid.x_origin_km,grid.y_origin_km))

    def interpolate(self,field,*,ignore_zero=False):
        """Map a source-grid field with four-neighbour inverse-distance weights.

        Args:
            field: Array shaped ``(..., source_y, source_x)``. Leading axes
                and physical units are preserved.
            ignore_zero: Exclude source values below ``1e-6`` from weighting;
                used for nonzero cloud-ceiling interpolation.

        Returns:
            ``float32`` array shaped ``(..., target_y, target_x)`` in the
            input field's physical units. Targets with no retained weight are
            zero.

        Raises:
            ValueError: If the trailing source-grid dimensions do not match
                ``source_shape``.
        """
        data=np.asarray(field)
        if data.shape[-2:]!=self.source_shape: raise ValueError('source field shape mismatch')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        selected=np.asarray(np.take(flat,self.indices,axis=-1),dtype=MODEL_DTYPE)
        source=self.source_coordinates[self.indices]
        target=self.target_coordinates
        delta=source-target[:,None,:]
        distance2=delta[:,:,0]**2+delta[:,:,1]**2
        weights=1./np.maximum(distance2,.001)
        if ignore_zero:
            weights=np.where(selected>=1.e-6,weights,0.)
        total=(selected*weights).sum(axis=-1,dtype=ACCUMULATION_DTYPE)
        denominator=weights.sum(axis=-1,dtype=ACCUMULATION_DTYPE)
        mapped=np.divide(total,denominator,out=np.zeros_like(total),where=denominator>0.)
        return _model_array(mapped.reshape(leading+self.target_shape))

    def interpolate_observations(self,field,rmin_km,*,barriers=(),barrier_top_level=0):
        """Map prognostic pseudo-observations and retain their weight strength.

        Args:
            field: ``(..., source_y, source_x)`` values, normally level-first
                U or V wind in m/s.
            rmin_km: Positive minimum distance in kilometres used in ``1/R²``
                weights.
            barriers: ``(barrier, 4)`` local-kilometre line segments that
                block source-to-target influence.
            barrier_top_level: Number of lowest vertical levels to which
                barriers apply.

        Returns:
            Pair ``(mapped, strength)``. ``mapped`` is ``float32`` with shape
            ``(..., target_y, target_x)`` and input units; ``strength`` is the
            unnormalised ``1/km²`` weight sum, with a level axis when barriers
            make availability level dependent.

        Raises:
            ValueError: If source dimensions mismatch, ``rmin_km`` is not
                positive, or barriers are requested for a non-3-D field.
        """
        data=np.asarray(field,dtype=MODEL_DTYPE)
        if data.shape[-2:]!=self.source_shape: raise ValueError('prognostic observation field shape mismatch')
        if rmin_km<=0.: raise ValueError('prognostic observation RMIN must be positive')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        selected=np.asarray(np.take(flat,self.indices,axis=-1),dtype=MODEL_DTYPE)
        source=np.asarray(self.source_coordinates[self.indices],dtype=COORDINATE_DTYPE)
        target=np.asarray(self.target_coordinates,dtype=COORDINATE_DTYPE)
        delta=source-target[:,None,:]
        distance2=delta[:,:,0]**2+delta[:,:,1]**2
        floor=np.float64(rmin_km)**2
        raw=1.0/np.maximum(distance2,floor)
        barrier_array=np.asarray(barriers,dtype=COORDINATE_DTYPE)
        if barrier_array.size:
            if data.ndim!=3:
                raise ValueError('level-dependent prognostic barriers require a 3-D field')
            origin=np.asarray(self.grid_origin,dtype=COORDINATE_DTYPE)
            blocked=barrier_blocked_pairs(target-origin,
                                           source-origin,barrier_array)
            available=np.ones((leading[0],)+raw.shape,dtype=bool)
            available[:barrier_top_level] &= ~blocked[None,:,:]
            factors=raw[None,:,:]*available
            strength=factors.sum(axis=-1,dtype=ACCUMULATION_DTYPE)
            numerator=(selected*factors).sum(axis=-1,dtype=ACCUMULATION_DTYPE)
            mapped=np.divide(numerator,strength,out=np.zeros_like(numerator),where=strength>0.)
            mapped=mapped.reshape(leading+self.target_shape)
            return (_model_array(mapped),
                    _model_array(strength.reshape(leading+self.target_shape)))
        strength=raw.sum(axis=-1,dtype=ACCUMULATION_DTYPE)
        numerator=(selected*raw).sum(axis=-1,dtype=ACCUMULATION_DTYPE)
        mapped=np.divide(numerator,strength,out=np.zeros_like(numerator),
                         where=strength>0.).reshape(leading+self.target_shape)
        return _model_array(mapped),_model_array(strength.reshape(self.target_shape))

    def nearest(self,field):
        """Map a field with RDMM5's closest prognostic cell.

        Args:
            field: Array shaped ``(..., source_y, source_x)``.

        Returns:
            ``float32`` array shaped ``(..., target_y, target_x)`` with the
            input physical units.

        Raises:
            ValueError: If the source-grid dimensions do not match.
        """
        data=np.asarray(field)
        if data.shape[-2:]!=self.source_shape: raise ValueError('source field shape mismatch')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        return _model_array(np.take(flat,self.indices[:,0],axis=-1).reshape(
            leading+self.target_shape))

    def nearest_masked(self,field,source_mask,*,dtype=MODEL_DTYPE):
        """Map to the nearest masked source.

        Physical fields use the model dtype by default.  Callers mapping a
        coordinate can request ``COORDINATE_DTYPE`` explicitly.

        Args:
            field: Array shaped ``(..., source_y, source_x)``.
            source_mask: Boolean ``(source_y, source_x)`` availability mask.
            dtype: Output NumPy dtype; defaults to model ``float32``.

        Returns:
            Array shaped ``(..., target_y, target_x)`` filled from the nearest
            available source cell and cast to ``dtype``.

        Raises:
            ValueError: If shapes mismatch or the mask selects no source cell.
        """
        data=np.asarray(field)
        mask=np.asarray(source_mask,dtype=bool)
        if data.shape[-2:]!=self.source_shape or mask.shape!=self.source_shape:
            raise ValueError('masked nearest field/source mask shape mismatch')
        flat_mask=np.ravel(mask,order='F')
        if not np.any(flat_mask):
            raise ValueError('prognostic SST mode requires at least one ocean source cell')
        available=np.flatnonzero(flat_mask)
        source=np.asarray(self.source_coordinates[available],dtype=np.float64)
        target=np.asarray(self.target_coordinates,dtype=np.float64)
        local=_nearest_indices(source,target,1)[:,0]
        selected=available[local]
        leading=data.shape[:-2]
        flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        mapped=np.take(flat,selected,axis=-1).reshape(leading+self.target_shape)
        return np.asarray(mapped,dtype=dtype)

    def radius_interpolate(self,field,radius_km,*,cutoff=0.,method=2):
        """Radius-limited precipitation objective mapping.

        Coordinate distances and weighted reductions use double precision;
        the returned model field is converted to :data:`MODEL_DTYPE` once.
        For exponential mapping with an automatic scale, the Gaussian sigma
        is half the closest wet-to-dry source distance.  The legacy API keeps
        the historical INTERPQR assignment bug that used the unhalved value.

        Args:
            field: One ``(source_y, source_x)`` precipitation field in mm/h.
            radius_km: Search radius in kilometres for methods 1 and 2, or
                Gaussian sigma for method 3; zero requests automatic sigma
                only for method 3.
            cutoff: Values below this precipitation rate in mm/h are set to
                zero when interpolation supplied a result.
            method: ``1`` inverse distance, ``2`` inverse squared distance,
                or ``3`` Gaussian-decayed inverse squared distance.

        Returns:
            ``float32 (target_y, target_x)`` precipitation rate in mm/h.

        Raises:
            ValueError: If the field shape, method, or radius is invalid.
        """
        data=np.asarray(field,dtype=np.float64)
        if data.shape!=self.source_shape: raise ValueError('radius interpolation expects one source 2-D field')
        if method not in {1,2,3}: raise ValueError('INTERPQR method must be 1, 2, or 3')
        if radius_km<0. or (method in {1,2} and radius_km==0.):
            raise ValueError('INTERPQR radius must be non-negative and nonzero for methods 1 and 2')
        source_values=np.ravel(data,order='F')
        origin=np.asarray(self.grid_origin,dtype=COORDINATE_DTYPE)
        source_coordinates=(self.source_coordinates-origin)*1000.
        target_coordinates=(self.target_coordinates-origin)*1000.
        source_tree=cKDTree(source_coordinates); nearest=np.take(source_values,self.indices[:,0])
        automatic_sigma=method==3 and radius_km==0.
        if automatic_sigma:
            wet=source_values>0.; dry=source_values==0.
            if wet.any() and dry.any():
                dry_coordinates=source_coordinates[dry]
                nearest_dry=cKDTree(dry_coordinates).query(source_coordinates[wet],k=1)[1]
                delta=source_coordinates[wet]-dry_coordinates[nearest_dry]
                distance2=delta[:,0]**2+delta[:,1]**2
                minimum_wet_dry_distance2=np.min(distance2)
            else:
                minimum_wet_dry_distance2=1.e37
            radius_m=.5*np.sqrt(minimum_wet_dry_distance2)
            sigma2=radius_m**2
        else:
            radius_m=float(radius_km)*1000.
            sigma2=radius_m**2
        if method in {1,2}:
            limit2=sigma2
        else:
            cutoff_ratio=60.
            maximum=np.finfo(np.float64).max/cutoff_ratio
            limit2=np.inf if sigma2>maximum else cutoff_ratio*sigma2
        search_radius=np.sqrt(float(limit2))
        result=np.empty(self.target_coordinates.shape[0],dtype=np.float64)
        weighted=np.zeros(result.size,dtype=bool)
        # A whole-domain sparse distance matrix can contain tens of millions
        # of pairs on the 610x499 Zhejiang grid.  Chunk targets while retaining
        # INTERPQR's exact all-points-inside-radius weighted average.
        chunk_size=8192 if np.isfinite(search_radius) else 128
        for begin in range(0,result.size,chunk_size):
            end=min(begin+chunk_size,result.size); targets=target_coordinates[begin:end]
            neighbours=(source_tree.query_ball_point(targets,search_radius) if np.isfinite(search_radius)
                        else [np.arange(source_values.size,dtype=np.intp)]*len(targets))
            lengths=np.fromiter((len(item) for item in neighbours),dtype=np.intp,count=end-begin)
            source_indices=np.concatenate(neighbours) if lengths.sum() else np.empty(0,dtype=np.intp)
            local_targets=np.repeat(np.arange(end-begin),lengths)
            difference=source_coordinates[source_indices]-targets[local_targets]
            distance2=difference[:,0]**2+difference[:,1]**2
            if method==1:
                eligible=distance2<=sigma2
                safe_distance=np.maximum(np.sqrt(distance2),.01)
                weights=np.where(eligible,1./safe_distance,0.)
                weighted_values=np.where(
                    eligible,source_values[source_indices]/safe_distance,0.)
            elif method==2:
                eligible=distance2<=sigma2
                safe_distance=np.maximum(distance2,.01)
                weights=np.where(eligible,1./safe_distance,0.)
                weighted_values=np.where(
                    eligible,source_values[source_indices]/safe_distance,0.)
            else:
                exponent=distance2/sigma2
                eligible=exponent<=60.
                decay=np.where(eligible,np.exp(-exponent),0.)
                safe_distance=np.maximum(distance2,.01)
                weights=decay/safe_distance
                weighted_values=source_values[source_indices]*decay/safe_distance
            numerator=np.bincount(local_targets,weights=weighted_values,
                                  minlength=end-begin).astype(ACCUMULATION_DTYPE)
            denominator=np.bincount(local_targets,weights=weights,
                                    minlength=end-begin).astype(ACCUMULATION_DTYPE)
            if method==3:
                # INTERPQR does not use the nearest-cell fallback in the
                # exponential branch.  If every EXP(-R²/SIGMA²) term was
                # cut off (or the weighted precipitation TOP is tiny), the
                # literal test TOP < 1E-30 returns zero.
                result[begin:end]=np.divide(numerator,denominator,
                    out=np.zeros(end-begin,dtype=np.float64),
                    where=(numerator>=1.e-30)&(denominator>0.))
            else:
                weighted[begin:end]=denominator>0.
                result[begin:end]=np.divide(numerator,denominator,
                    out=nearest[begin:end].copy(),where=denominator>0.)
        apply_cutoff=np.ones(result.size,dtype=bool) if method==3 else weighted
        result=np.where(apply_cutoff&(result<float(cutoff)),0.,result)
        return _model_array(result.reshape(self.target_shape))

@dataclass(frozen=True,slots=True)
class SourceThermalState:
    """RDMM5 potential temperatures retained from the last daylight hour.

    Attributes:
        pt20: ``float32 (source_y, source_x)`` lower reference-level potential
            temperature in kelvins.
        pt30: ``float32 (source_y, source_x)`` upper reference-level potential
            temperature in kelvins.
    """
    pt20: np.ndarray
    pt30: np.ndarray

@dataclass(frozen=True,slots=True)
class Step1Fields:
    """Meteorological fields entering one CALMET diagnostic step.

    Unless stated otherwise, arrays use ``float32``; horizontal fields have
    shape ``(y, x)`` and three-dimensional fields ``(level, y, x)``.

    Attributes:
        u: Grid-east wind by CALMET layer, in m/s.
        v: Grid-north wind by CALMET layer, in m/s.
        temperature: Air temperature by CALMET layer, in K.
        surface_temperature: Two-metre or diagnosed surface air temperature,
            in K.
        surface_pressure_mb: Surface pressure in millibars.
        surface_density: Near-surface dry-air density in kg/m3.
        relative_humidity: Relative humidity by CALMET layer, in percent.
        pbl_height: Prognostic planetary-boundary-layer height in metres AGL.
        sensible_heat_flux: Upward sensible heat flux in W/m2.
        friction_velocity: Surface friction velocity in m/s.
        cloud_fraction: Total cloud fraction from zero to one.
        ceiling_height: Cloud ceiling in metres AGL; zero denotes no ceiling.
        profile_temperature: Temperature profile including a surface row, in
            K, shaped ``(profile_level, y, x)``.
        profile_height: Matching profile heights in metres AGL.
        surface_relative_humidity: Surface relative humidity in percent.
        precipitation_rate: Precipitation rate in mm/h.
        precipitation_code: ``int32`` weather code grid: 0 none, 10 rain, or
            20 snow.
        thermal_state: Optional source-grid daylight reference state carried
            between RDMM5 records.
        cloud_relative_humidity_850: Optional mapped 850-mb relative humidity
            in percent for cloud method 3.
        integrated_cloud_water: Optional per-column maximum ``QCLOUD+QICE``
            mixing ratio in g/kg; the historical field name is retained.
        observed_temperature_faces: Optional observed temperatures on CALMET
            layer faces, in K, shaped ``(face, y, x)``.
        water_gradient_below: Optional overwater temperature gradient below
            mixing height, in K/m.
        water_gradient_above: Optional overwater temperature gradient above
            mixing height, in K/m.
        water_mixing_gradient_above: Optional gradient used by the overwater
            mixing-height calculation, in K/m.
        observed_water_mixing_height: Optional observed water mixing height in
            metres AGL.
        diagnostic_surface_temperature: Optional observation-derived surface
            temperature in K.
        diagnostic_lapse_rate: Optional observation-derived lapse rate in K/m.
        diagnostic_lapse_station_index: Optional ``(y, x)`` grid index of the
            selected lapse-rate station.
        diagnostic_lapse_data: Optional data provider used to refresh an
            observation-derived lapse rate.
        diagnostic_lapse_timestamp: Timestamp of the selected lapse-rate
            observation.
        prognostic_observation_u: Optional mapped pseudo-observation U wind in
            m/s, shaped ``(level, y, x)``.
        prognostic_observation_v: Optional mapped pseudo-observation V wind in
            m/s.
        prognostic_observation_strength: Optional unnormalised pseudo-
            observation weight sum in 1/km2.
        roughness_override: Optional dynamic surface roughness in metres.
        overwater_data: Optional gridded SEA-like observations and SST state.
        prognostic_temperature: Optional unmodified prognostic temperature in
            K used by selected diagnostic paths.
        cloud_liquid_water_3d: Optional layer-mean cloud liquid-water mixing
            ratio in g/kg, shaped ``(level, y, x)``.
        cloud_liquid_water_above: Optional mean cloud liquid water above the
            CALMET top, in g/kg.
        cloud_liquid_water_above_bottom: Bottom of that above-model cloud
            layer in metres AGL.
        cloud_liquid_water_above_top: Top of that above-model cloud layer in
            metres AGL.
        debug_surface_record: Optional raw surface record retained for listing
            diagnostics.
        debug_surface_stations: Surface stations retained for diagnostics.
        debug_station_names: Station names corresponding to debug wind arrays.
        debug_station_u: Optional station U wind in m/s.
        debug_station_v: Optional station V wind in m/s.
        debug_domain_temperature: Optional domain-mean temperature in K.
        debug_domain_u: Optional domain-mean U wind in m/s.
        debug_domain_v: Optional domain-mean V wind in m/s.
        debug_upper_data: Upper-air diagnostic records retained for listings.
    """
    u: np.ndarray
    v: np.ndarray
    temperature: np.ndarray
    surface_temperature: np.ndarray
    surface_pressure_mb: np.ndarray
    surface_density: np.ndarray
    relative_humidity: np.ndarray
    pbl_height: np.ndarray
    sensible_heat_flux: np.ndarray
    friction_velocity: np.ndarray
    cloud_fraction: np.ndarray
    ceiling_height: np.ndarray
    profile_temperature: np.ndarray
    profile_height: np.ndarray
    surface_relative_humidity: np.ndarray
    precipitation_rate: np.ndarray
    precipitation_code: np.ndarray
    thermal_state: SourceThermalState | None = None
    cloud_relative_humidity_850: np.ndarray | None = None
    integrated_cloud_water: np.ndarray | None = None
    observed_temperature_faces: np.ndarray | None = None
    water_gradient_below: np.ndarray | None = None
    water_gradient_above: np.ndarray | None = None
    water_mixing_gradient_above: np.ndarray | None = None
    observed_water_mixing_height: np.ndarray | None = None
    diagnostic_surface_temperature: np.ndarray | None = None
    diagnostic_lapse_rate: np.ndarray | None = None
    diagnostic_lapse_station_index: tuple[int,int] | None = None
    diagnostic_lapse_data: object | None = None
    diagnostic_lapse_timestamp: datetime | None = None
    prognostic_observation_u: np.ndarray | None = None
    prognostic_observation_v: np.ndarray | None = None
    prognostic_observation_strength: np.ndarray | None = None
    roughness_override: np.ndarray | None = None
    overwater_data: object | None = None
    prognostic_temperature: np.ndarray | None = None
    cloud_liquid_water_3d: np.ndarray | None = None
    cloud_liquid_water_above: np.ndarray | None = None
    cloud_liquid_water_above_bottom: np.ndarray | None = None
    cloud_liquid_water_above_top: np.ndarray | None = None
    debug_surface_record: object | None = None
    debug_surface_stations: tuple = ()
    debug_station_names: tuple[str,...] = ()
    debug_station_u: np.ndarray | None = None
    debug_station_v: np.ndarray | None = None
    debug_domain_temperature: float | None = None
    debug_domain_u: float | None = None
    debug_domain_v: float | None = None
    debug_upper_data: tuple = ()


@dataclass(slots=True)
class StepTransformContext:
    """Mutable runtime state supplied to physical Step-1 transformations.

    Attributes:
        solar_elevation_sine: ``float32 (y, x)`` sine of midpoint solar
            elevation.
        previous_convective_height: Previous convective height in metres AGL.
        previous_mixing_height: Previous total mixing height in metres AGL.
        previous_temperature_jump: Previous inversion jump in K.
        coriolis: Coriolis parameter in 1/s.
        warm_layer_state: Optional mutable COARE warm-layer state; a transform
            may replace it in place on this context.
        gmt_hour: Optional interval-ending UTC decimal hour.
    """
    solar_elevation_sine: np.ndarray
    previous_convective_height: np.ndarray
    previous_mixing_height: np.ndarray
    previous_temperature_jump: np.ndarray
    coriolis: np.ndarray
    warm_layer_state: object | None = None
    gmt_hour: float | None = None


def _domain_mean(field):
    """Calculate a stable mean for a domain-scale diagnostic.

    Args:
        field: Numerical array in any physical units.

    Returns:
        ``float64`` scalar mean in the same units as ``field``.
    """
    return np.mean(np.asarray(field,dtype=ACCUMULATION_DTYPE),
                   dtype=ACCUMULATION_DTYPE)

def _interpolate_ceiling(left,right,fraction):
    """Interpolate cloud ceiling while retaining a lone nonzero endpoint.

    Args:
        left: Earlier ceiling-height array in metres AGL; zero means absent.
        right: Later ceiling-height array with the same shape and units.
        fraction: Temporal fraction from ``left`` to ``right``.

    Returns:
        ``float32`` ceiling array. Cells with two nonzero endpoints are
        linearly interpolated; cells with only one use that nonzero height.
    """
    f=MODEL_DTYPE(fraction)
    left=np.asarray(left,dtype=MODEL_DTYPE); right=np.asarray(right,dtype=MODEL_DTYPE)
    interpolated=left+f*(right-left)
    return _model_array(np.where(
        (left!=0.)&(right!=0.),interpolated,np.maximum(left,right)))

def interpolate_step1_fields(left: Step1Fields,right: Step1Fields,fraction,*,cloud_method=4,
                             cloud_condensate_threshold_g_kg=.005):
    """Interpolate model fields between two prognostic endpoint records.

    Args:
        left: Earlier Step-1 endpoint.
        right: Later Step-1 endpoint. Discrete carried state is taken from
            this endpoint.
        fraction: Fraction from zero at ``left`` to one at ``right``.
        cloud_method: Prognostic cloud method 3 (850-mb RH) or 4 (all-level
            diagnosed cloud).
        cloud_condensate_threshold_g_kg: Minimum interpolated column-maximum
            ``QCLOUD+QICE`` in g/kg required to retain cloud and ceiling.

    Returns:
        New :class:`Step1Fields` whose continuous arrays are linearly
        interpolated in ``float32``; precipitation code is rediagnosed from
        rate and lowest-layer temperature.

    Raises:
        ValueError: If ``fraction`` is outside [0, 1], ``cloud_method`` is
            unsupported, or method 3 lacks endpoint 850-mb humidity.
    """
    f=MODEL_DTYPE(fraction)
    if not 0.<=f<=1.: raise ValueError('prognostic interpolation fraction must be between zero and one')
    def lerp(a,b):
        """Linearly interpolate one same-shaped model field.

        Args:
            a: Earlier scalar or array.
            b: Later scalar or array with compatible shape.

        Returns:
            ``float32`` value with the broadcast shape and input units.
        """
        a=np.asarray(a,dtype=MODEL_DTYPE); b=np.asarray(b,dtype=MODEL_DTYPE)
        return _model_array(a+f*(b-a))
    if cloud_method==4:
        cloud=lerp(left.cloud_fraction,right.cloud_fraction)
    elif cloud_method==3:
        if left.cloud_relative_humidity_850 is None or right.cloud_relative_humidity_850 is None:
            raise ValueError('MCLOUD=3 interpolation requires endpoint 850 mb relative humidity')
        cloud=cloud_fraction(lerp(left.cloud_relative_humidity_850,right.cloud_relative_humidity_850))
    else:
        raise ValueError('cloud_method must be MCLOUD 3 or 4')
    ceiling=_interpolate_ceiling(left.ceiling_height,right.ceiling_height,f)
    cloud_water=None
    if left.integrated_cloud_water is not None and right.integrated_cloud_water is not None:
        cloud_water=lerp(left.integrated_cloud_water,right.integrated_cloud_water)
        cloudy=cloud_water>=cloud_condensate_threshold_g_kg
        cloud=np.where(cloudy,cloud,0.); ceiling=np.where(cloudy,ceiling,0.)
    temperature=lerp(left.temperature,right.temperature)
    precipitation=lerp(left.precipitation_rate,right.precipitation_rate)
    precipitation_code=np.where(precipitation>0.,np.where(temperature[0]<273.15,20,10),0).astype(np.int32)
    rh850=(None if left.cloud_relative_humidity_850 is None or right.cloud_relative_humidity_850 is None
           else lerp(left.cloud_relative_humidity_850,right.cloud_relative_humidity_850))
    prognostic_temperature=(
        None if left.prognostic_temperature is None or right.prognostic_temperature is None
        else lerp(left.prognostic_temperature,right.prognostic_temperature))
    auxiliary_cloud=[]
    for name in ('cloud_liquid_water_3d','cloud_liquid_water_above',
                 'cloud_liquid_water_above_bottom','cloud_liquid_water_above_top'):
        left_value=getattr(left,name); right_value=getattr(right,name)
        auxiliary_cloud.append(
            None if left_value is None or right_value is None
            else lerp(left_value,right_value))
    return Step1Fields(
        lerp(left.u,right.u),lerp(left.v,right.v),temperature,
        lerp(left.surface_temperature,right.surface_temperature),
        lerp(left.surface_pressure_mb,right.surface_pressure_mb),
        lerp(left.surface_density,right.surface_density),lerp(left.relative_humidity,right.relative_humidity),
        lerp(left.pbl_height,right.pbl_height),lerp(left.sensible_heat_flux,right.sensible_heat_flux),
        lerp(left.friction_velocity,right.friction_velocity),cloud,ceiling,
        lerp(left.profile_temperature,right.profile_temperature),
        lerp(left.profile_height,right.profile_height),
        lerp(left.surface_relative_humidity,right.surface_relative_humidity),
        precipitation,precipitation_code,right.thermal_state,rh850,cloud_water,
        prognostic_observation_u=(None if left.prognostic_observation_u is None else
                                  lerp(left.prognostic_observation_u,right.prognostic_observation_u)),
        prognostic_observation_v=(None if left.prognostic_observation_v is None else
                                  lerp(left.prognostic_observation_v,right.prognostic_observation_v)),
        prognostic_observation_strength=(None if right.prognostic_observation_strength is None else
                                          np.asarray(right.prognostic_observation_strength,
                                                     dtype=MODEL_DTYPE)),
        prognostic_temperature=prognostic_temperature,
        cloud_liquid_water_3d=auxiliary_cloud[0],
        cloud_liquid_water_above=auxiliary_cloud[1],
        cloud_liquid_water_above_bottom=auxiliary_cloud[2],
        cloud_liquid_water_above_top=auxiliary_cloud[3])


def observed_temperature_3d(surface_temperature,face_temperature,convective_height,z_faces_m,
                            *,landuse=None,water_range=(55,55),mixing_height=None,
                            water_gradient_below=None,water_gradient_above=None):
    """Construct observed-mode layer-mean temperatures with continuous overlap.

    Args:
        surface_temperature: ``(y, x)`` surface temperature in K.
        face_temperature: ``(face, y, x)`` observed temperature at CALMET
            layer faces in K.
        convective_height: Land convective mixing height in metres AGL.
        z_faces_m: ``(face,)`` CALMET layer boundaries in metres AGL.
        landuse: Optional ``(y, x)`` dominant land-use categories.
        water_range: Inclusive category range treated as water.
        mixing_height: Optional overwater mixing height in metres AGL.
        water_gradient_below: Optional overwater temperature gradient below
            mixing height in K/m.
        water_gradient_above: Optional overwater gradient above mixing height
            in K/m.

    Returns:
        ``float32 (level, y, x)`` layer-mean temperature in K. Layers crossed
        by a mixing-height surface are weighted continuously by overlapped
        thickness.

    Raises:
        ValueError: If face-temperature dimensions do not match the vertical
            faces and horizontal surface grid.
    """
    surface=np.asarray(surface_temperature,dtype=MODEL_DTYPE)
    faces=np.asarray(face_temperature,dtype=MODEL_DTYPE)
    z=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    if faces.shape!=(z.size,)+surface.shape:
        raise ValueError('observed face temperature dimensions do not match the CALMET grid')
    result=np.empty((z.size-1,)+surface.shape,dtype=MODEL_DTYPE); result[0]=surface
    zi=np.asarray(convective_height,dtype=MODEL_DTYPE)
    for k in range(1,z.size-1):
        low,high=z[k],z[k+1]; top=faces[k+1]
        above=.5*(faces[k]+top)
        below=surface-.0098*.5*(low+high)
        middle=surface-.0098*zi
        lower_sum=surface-.0098*low+middle
        upper_sum=middle+top
        crossing=.5*(lower_sum*(zi-low)+upper_sum*(high-zi))/(high-low)
        result[k]=np.where(low>zi,above,np.where(high<=zi,below,crossing))
    if landuse is not None and water_gradient_below is not None and water_gradient_above is not None:
        water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
        water_zi=np.asarray(mixing_height,dtype=MODEL_DTYPE)
        below_gradient=np.asarray(water_gradient_below,dtype=MODEL_DTYPE)
        above_gradient=np.asarray(water_gradient_above,dtype=MODEL_DTYPE)
        for k in range(1,z.size-1):
            low,high=z[k],z[k+1]; center=.5*(low+high)
            wholly_below=surface+below_gradient*center
            wholly_above=(surface+below_gradient*water_zi+
                           above_gradient*(center-water_zi))
            low_temp=surface+below_gradient*low
            middle_temp=surface+below_gradient*water_zi
            high_temp=middle_temp+above_gradient*(high-water_zi)
            crossing=.5*((low_temp+middle_temp)*(water_zi-low)+
                         (middle_temp+high_temp)*(high-water_zi))/(high-low)
            water_value=np.where(high<=water_zi,wholly_below,
                                 np.where(low>water_zi,wholly_above,crossing))
            result[k]=np.where(water,water_value,result[k])
    return _model_array(result)

def map_prognostic_cloud(source_cloud,source_rh850,mapper,cloud_method):
    """Apply RDMM5's method-specific cloud interpolation order.

    Args:
        source_cloud: ``(source_y, source_x)`` cloud fraction for method 4.
        source_rh850: Source 850-mb relative humidity in percent for method 3.
        mapper: Prognostic-to-CALMET horizontal mapper.
        cloud_method: ``3`` to map RH before diagnosing cloud fraction or
            ``4`` to map source-diagnosed cloud directly.

    Returns:
        Pair ``(cloud_fraction, mapped_rh850)`` on ``(target_y, target_x)``.
        The second item is ``None`` for method 4; arrays use ``float32``.

    Raises:
        ValueError: If ``cloud_method`` is neither 3 nor 4.
    """
    if cloud_method==4: return mapper.interpolate(source_cloud),None
    if cloud_method==3:
        mapped_rh850=mapper.interpolate(source_rh850)
        return cloud_fraction(mapped_rh850),mapped_rh850
    raise ValueError('cloud_method must be MCLOUD 3 or 4')

@dataclass(frozen=True,slots=True)
class SurfaceFields:
    """Surface-layer and mixing-height results for one model step.

    Horizontal arrays have shape ``(y, x)`` and normally use ``float32``.

    Attributes:
        roughness: Effective aerodynamic roughness length in metres.
        density: Surface air density in kg/m3.
        shortwave_flux: Downward shortwave flux in W/m2.
        sensible_heat_flux: Upward sensible heat flux in W/m2.
        friction_velocity: Friction velocity in m/s.
        monin_obukhov_length: Monin-Obukhov length in metres.
        mixing_height: Final total mixing height in metres AGL.
        convective_mixing_height: Convective mixing-height state in metres.
        temperature_jump: Potential-temperature jump at the boundary-layer
            top in K.
        warm_layer_state: Optional updated COARE warm-layer state.
        water_convective_velocity_scale: Optional overwater convective
            velocity scale in m/s, zero over land.
        water_convective_height_state: Optional unaveraged overwater
            convective-height state in metres.
        unaveraged_mixing_height: Optional mixing height before upwind spatial
            averaging, retained for diagnostics.
        unaveraged_convective_mixing_height: Optional convective height before
            upwind spatial averaging.
    """
    roughness: np.ndarray
    density: np.ndarray
    shortwave_flux: np.ndarray
    sensible_heat_flux: np.ndarray
    friction_velocity: np.ndarray
    monin_obukhov_length: np.ndarray
    mixing_height: np.ndarray
    convective_mixing_height: np.ndarray
    temperature_jump: np.ndarray
    warm_layer_state: object | None = None
    water_convective_velocity_scale: np.ndarray | None = None
    water_convective_height_state: np.ndarray | None = None
    unaveraged_mixing_height: np.ndarray | None = None
    unaveraged_convective_mixing_height: np.ndarray | None = None

@dataclass(frozen=True,slots=True)
class DiagnosticWindFields:
    """Final diagnostic wind and optional intermediate stages.

    Attributes:
        u: ``float32 (level, y, x)`` grid-east wind in m/s.
        v: ``float32 (level, y, x)`` grid-north wind in m/s.
        w: ``float32 (face, y, x)`` vertical velocity in m/s.
        divergence: ``float32 (level, y, x)`` horizontal divergence in 1/s.
        lapse_rate: ``float32 (y, x)`` environmental lapse rate in K/m.
        preliminary_heat_flux: ``float32 (y, x)`` preliminary sensible heat
            flux in W/m2 used by slope flow.
        kinematic_stage: Optional ``(u, v, w)`` snapshots after kinematic
            adjustment.
        froude_stage: Optional ``(u, v, w)`` snapshots after Froude adjustment.
        slope_stage: Optional ``(u, v, w)`` snapshots after slope flow.
        interpolated_stage: Optional ``(u, v)`` Step-1 wind snapshot.
        terrain_adjusted_stage: Optional ``(u, v)`` surface-terrain adjustment
            snapshot.
        initial_stage: Optional ``(u, v, w, divergence)`` snapshot before
            divergence minimisation.
    """
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    divergence: np.ndarray
    lapse_rate: np.ndarray
    preliminary_heat_flux: np.ndarray
    kinematic_stage: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None
    froude_stage: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None
    slope_stage: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None
    interpolated_stage: tuple[np.ndarray,np.ndarray] | None = None
    terrain_adjusted_stage: tuple[np.ndarray,np.ndarray] | None = None
    initial_stage: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] | None = None

@dataclass(frozen=True,slots=True)
class HourFields:
    """Complete CALMET output and carried state for one interval.

    Attributes:
        timestamp: Naive local datetime at the beginning of the output
            interval.
        step1: Final Step-1 fields, with diagnostic U/V winds substituted.
        wind: Diagnostic wind, vertical velocity, divergence, and debug stages.
        surface: Surface-layer and mixing-height results.
        temperature: ``float32 (level, y, x)`` final 3-D temperature in K.
        selected_lapse_convective_time: Optional timestamp carried by the
            observed lapse-rate selection logic.
        source_step: Optional Step-1 state before final diagnostic wind
            substitution, retained for diagnostics.
    """
    timestamp: datetime
    step1: Step1Fields
    wind: DiagnosticWindFields
    surface: SurfaceFields
    temperature: np.ndarray
    selected_lapse_convective_time: datetime | None = None
    source_step: Step1Fields | None = None

def target_latitude_longitude(grid: GridConfig):
    """Calculate geographic coordinates at CALMET cell centres.

    Args:
        grid: Target projection, origin, spacing, and dimensions.

    Returns:
        Pair ``(latitude, longitude)`` of ``float64 (y, x)`` arrays in decimal
        degrees using WGS-84.
    """
    ix=np.arange(grid.nx,dtype=COORDINATE_DTYPE)+.5
    iy=np.arange(grid.ny,dtype=COORDINATE_DTYPE)+.5
    x=float(grid.x_origin_km)+ix*float(grid.dx_km)
    y=float(grid.y_origin_km)+iy*float(grid.dy_km)
    xx,yy=np.meshgrid(x,y)
    longitude,latitude=unproject_geographic(xx,yy,grid,'WGS-84',units='km')
    return np.asarray(latitude),np.asarray(longitude)

def _vertical_centers(field,height,centers,*,wind=False):
    """Interpolate source columns to CALMET layer-centre heights.

    Args:
        field: ``float-like (source_level, y, x)`` physical field.
        height: Matching source-level height in metres AGL.
        centers: One-dimensional target heights in metres AGL.
        wind: Extrapolate targets below the first source level with RDMM5's
            logarithmic wind profile; other fields hold the first value.

    Returns:
        ``float32 (target_level, y, x)`` array in the field's input units.

    Raises:
        ValueError: If field and height shapes differ or are not three-
            dimensional.
    """
    data=np.asarray(field,dtype=MODEL_DTYPE)
    z=np.asarray(height,dtype=MODEL_DTYPE)
    if data.shape!=z.shape or data.ndim!=3: raise ValueError('vertical fields must have shape (level,y,x)')
    levels,y,x=data.shape; flat=data.reshape(levels,-1); zflat=z.reshape(levels,-1); columns=np.arange(flat.shape[1]); out=[]
    for center in np.asarray(centers,dtype=MODEL_DTYPE):
        above=np.sum(zflat<=center,axis=0); lower=np.clip(above-1,0,levels-1); upper=np.clip(above,0,levels-1)
        zl,zu=zflat[lower,columns],zflat[upper,columns]; vl,vu=flat[lower,columns],flat[upper,columns]
        ratio=np.divide(center-zl,zu-zl,out=np.zeros_like(zl),where=zu!=zl)
        value=vl+(vu-vl)*ratio
        below=center<zflat[0]; value=np.where(below,flat[0],value); value=np.where(center>=zflat[-1],flat[-1],value)
        if wind:
            log_half=MODEL_DTYPE(np.log(.5))
            numerator=np.log(center)-log_half
            denominator=np.log(np.maximum(zflat[0],.500001))-log_half
            logratio=np.divide(numerator,denominator,out=np.ones_like(zl),
                               where=zflat[0]>.500001)
            value=np.where(below,logratio*flat[0],value)
        out.append(value.reshape(y,x))
    return _model_array(np.stack(out))

def _prognostic_wind_components(speed,direction,convergence):
    """Rotate meteorological wind from true north into grid components.

    Args:
        speed: Wind speed array in m/s.
        direction: Meteorological direction in degrees clockwise from true
            north, matching ``speed``.
        convergence: Grid convergence in degrees, broadcast-compatible with
            the wind fields.

    Returns:
        Pair of ``float32`` arrays ``(u, v)`` in m/s, positive toward grid east
        and grid north.
    """
    velocity=np.asarray(speed,dtype=MODEL_DTYPE)
    grid_direction=(np.asarray(direction,dtype=MODEL_DTYPE)-
                    np.asarray(convergence,dtype=MODEL_DTYPE))%360.
    radians=np.deg2rad(grid_direction)
    return (_model_array(-velocity*np.sin(radians)),
            _model_array(-velocity*np.cos(radians)))

def _relative_humidity(qvapor,pressure_hpa,temperature_k):
    """Diagnose RH from WRF water-vapour mixing ratio.

    ``QVAPOR`` is the mass ratio of water vapour to dry air.  Convert it to
    vapour pressure with ``e = p r / (epsilon + r)`` before comparing it with
    saturation vapour pressure.  The legacy API retains CALWRF's approximation.

    Args:
        qvapor: Water-vapour mixing ratio relative to dry air, in kg/kg.
        pressure_hpa: Atmospheric pressure in hPa.
        temperature_k: Air temperature in K; arrays must be broadcast-
            compatible.

    Returns:
        ``float32`` relative humidity in percent, clipped to [0, 100], with
        the broadcast input shape.
    """
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_hpa,dtype=MODEL_DTYPE)
    vapor=np.asarray(qvapor,dtype=MODEL_DTYPE)
    tc=temperature-273.15
    es=6.112*np.exp(17.67*tc/(tc+243.5))
    vapor_pressure=pressure*vapor/(.622+vapor)
    return _model_array(np.clip(100.*vapor_pressure/es,0.,100.))

def _cloud_water_g_kg(values):
    """Convert canonical cloud condensate to nonnegative g/kg.

    Args:
        values: Cloud mixing ratio in kg/kg, scalar or array-like.

    Returns:
        ``float32`` array with the input shape in g/kg; negative values become
        zero.
    """
    values=np.asarray(values,dtype=MODEL_DTYPE)
    return _model_array(np.maximum(values,0.)*1000.)


def _source_relative_humidity(dataset,time_index,qvapor,pressure,temperature):
    """Read source relative humidity or diagnose it from thermodynamics.

    Args:
        dataset: NetCDF-like canonical meteorology dataset.
        time_index: Record index along the time axis.
        qvapor: Source ``(level, y, x)`` water-vapour mixing ratio in kg/kg.
        pressure: Matching pressure in hPa.
        temperature: Matching temperature in K.

    Returns:
        ``float32 (level, y, x)`` relative humidity in percent, clipped to
        [0, 100].
    """
    if 'relative_humidity' in dataset.variables:
        values=np.asarray(dataset['relative_humidity'][time_index],dtype=MODEL_DTYPE)
    else:
        values=_relative_humidity(qvapor,pressure,temperature)
    return _model_array(np.clip(values,0.,100.))


def _rdmm5_surface_density(pressure_mb,temperature_k,height_agl_m):
    """Calculate RDMM5 surface density from the first above-ground level.

    Args:
        pressure_mb: ``(source_level, y, x)`` pressure in millibars.
        temperature_k: Matching air temperature in K.
        height_agl_m: Matching height in metres AGL.

    Returns:
        ``float32 (y, x)`` dry-air density in kg/m3.

    Raises:
        ValueError: If input shapes differ, are not three-dimensional, or any
            column has no level at or above ground.
    """
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    temperature=np.asarray(temperature_k,dtype=MODEL_DTYPE)
    height=np.asarray(height_agl_m,dtype=MODEL_DTYPE)
    if pressure.shape!=temperature.shape or pressure.shape!=height.shape or pressure.ndim!=3:
        raise ValueError('RDMM5 density profiles must share shape (level,y,x)')
    above=height>=0.
    if not np.all(above.any(axis=0)):
        raise ValueError('RDMM5 profile has no level at or above ground')
    level=np.argmax(above,axis=0); yy,xx=np.indices(level.shape)
    return density_from_pressure_temperature(pressure[level,yy,xx],
                                             temperature[level,yy,xx])


def _rdmm5_above_ground_profiles(height_agl_m,*fields):
    """Pack each RDMM5 column from its first above-ground level upward.

    Args:
        height_agl_m: ``(source_level, y, x)`` heights in metres AGL.
        *fields: Fields with the identical source shape and arbitrary physical
            units.

    Returns:
        Tuple containing packed ``float32`` height followed by each packed
        field. Shapes remain ``(source_level, y, x)``; short columns repeat
        their highest available level.

    Raises:
        ValueError: If shapes differ, height is not three-dimensional, or a
            column has no level at or above ground.
    """
    height=np.asarray(height_agl_m,dtype=MODEL_DTYPE)
    values=(height,*(np.asarray(field,dtype=MODEL_DTYPE) for field in fields))
    if height.ndim!=3 or any(value.shape!=height.shape for value in values):
        raise ValueError('RDMM5 vertical profiles must share shape (level,y,x)')
    levels=height.shape[0]; above=height>=0.; count=above.sum(axis=0)
    if not np.all(count):
        raise ValueError('RDMM5 profile has no level at or above ground')
    level_index=np.arange(levels,dtype=np.int32)[:,None,None]
    packed_index=np.sort(np.where(above,level_index,levels),axis=0)
    rank=np.minimum(level_index,count[None]-1)
    packed_index=np.take_along_axis(packed_index,rank,axis=0)
    return tuple(np.take_along_axis(value,packed_index,axis=0) for value in values)


def _rdmm5_sst_profile(centers_m,lowest_height_m,lowest_temperature_k,sst_k,ocean):
    """Interpolate ocean temperature from SST to the lowest source level.

    Args:
        centers_m: One-dimensional CALMET layer-centre heights in metres AGL.
        lowest_height_m: ``(y, x)`` lowest above-ground source height in m.
        lowest_temperature_k: Temperature at that source level in K.
        sst_k: Sea-surface temperature in K.
        ocean: Boolean source-grid ocean mask.

    Returns:
        Pair ``(profile, use)``. ``profile`` is ``float32 (level, y, x)`` in K;
        Boolean ``use`` marks ocean layers below the lowest source level.
    """
    centers=np.asarray(centers_m,dtype=MODEL_DTYPE)
    height=np.asarray(lowest_height_m,dtype=MODEL_DTYPE)
    temperature=np.asarray(lowest_temperature_k,dtype=MODEL_DTYPE)
    sst=np.asarray(sst_k,dtype=MODEL_DTYPE)
    ratio=np.divide(centers[:,None,None],height[None,:,:],
                    out=np.zeros((centers.size,*height.shape),dtype=MODEL_DTYPE),
                    where=height[None,:,:]!=0.)
    profile=sst[None,:,:]*(1.-ratio)+temperature[None,:,:]*ratio
    use=(centers[:,None,None]<height[None,:,:])&np.asarray(ocean,dtype=bool)[None,:,:]
    return _model_array(profile),use

def _stull_temperature(theta2,z2,theta3,z3,theta20,theta30,z1):
    """Apply STULL nighttime potential-temperature extrapolation.

    Args:
        theta2: Lower current potential temperature in K.
        z2: Lower reference height in metres AGL.
        theta3: Upper current potential temperature in K.
        z3: Upper reference height in metres AGL.
        theta20: Lower daylight reference potential temperature in K.
        theta30: Upper daylight reference potential temperature in K.
        z1: Target height in metres AGL, scalar or broadcast-compatible.

    Returns:
        ``float32`` potential temperature in K with the broadcast horizontal
        shape.
    """
    t2=np.asarray(theta2,dtype=MODEL_DTYPE); t3=np.asarray(theta3,dtype=MODEL_DTYPE)
    t20=np.asarray(theta20,dtype=MODEL_DTYPE); t30=np.asarray(theta30,dtype=MODEL_DTYPE)
    height2=np.asarray(z2,dtype=MODEL_DTYPE); height3=np.asarray(z3,dtype=MODEL_DTYPE)
    dt2=t2-t20; dt3=t3-t30
    well=(t3>t2)&(dt3<0.)&(dt2<0.)&(dt2<dt3)
    ratio=np.divide(dt2,dt3,out=np.ones_like(dt2),where=dt3!=0.)
    logarithm=np.log(np.maximum(ratio,1.e-30))
    fallback=height3/5.
    calculated=np.divide(.77*(height3-height2),logarithm,
                         out=fallback.copy(),where=well&(ratio>0.)&
                         (np.abs(logarithm)>1.e-30))
    scale=np.maximum(np.where(well,calculated,fallback),
                     height2/10.)
    surface_drop=np.clip(dt2*np.exp(.77*height2/scale),-15.,0.)
    drop=surface_drop*np.exp(-.77*np.asarray(z1,dtype=MODEL_DTYPE)/scale)
    drop=np.where(dt2>0.,dt2,drop)
    return _model_array(t20+drop)

def build_step1_fields(dataset,time_index,grid: GridConfig,z_faces_m,*,timestamp=None,
                       base_timezone=0,thermal_state: SourceThermalState | None=None,
                       mapper: PrognosticMapper | None=None,prognostic_ocean_category=16,
                       water_temperature_mode=0,
                       cloud_method=4,cloud_condensate_threshold_g_kg=.005,
                       precipitation_interpolation=2,precipitation_radius_km=50.,
                       precipitation_cutoff=.01,prognostic_observation_rmin_km=None,
                       prognostic_observation_barriers=(),prognostic_observation_barrier_top_level=0):
    """Read one canonical CALWRF record and construct Step-1 fields.

    Args:
        dataset: Open NetCDF-like CALWRF dataset with canonical variables.
        time_index: Source time-record index.
        grid: Target CALMET grid.
        z_faces_m: One-dimensional CALMET layer faces in metres AGL.
        timestamp: Optional local ending timestamp used for daytime/nighttime
            low-level temperature treatment.
        base_timezone: CALMET hours west of UTC used by the solar calculation.
        thermal_state: Optional daylight potential-temperature state carried
            from the preceding source record.
        mapper: Reusable source-to-target mapper; ``None`` builds and validates
            one from dataset coordinates.
        prognostic_ocean_category: Source land-use category treated as ocean.
        water_temperature_mode: ``2`` enables prognostic SST interpolation
            below the first source level.
        cloud_method: ``3`` diagnoses cloud from mapped 850-mb RH; ``4`` maps
            all-level prognostic cloud.
        cloud_condensate_threshold_g_kg: Minimum layer ``QCLOUD+QICE`` in g/kg
            used to retain cloud and diagnose condensate ceiling.
        precipitation_interpolation: Radius mapper method 1, 2, or 3.
        precipitation_radius_km: Precipitation interpolation radius or method-3
            Gaussian sigma in kilometres.
        precipitation_cutoff: Mapped rates below this mm/h threshold become
            zero.
        prognostic_observation_rmin_km: Optional minimum pseudo-observation
            wind distance in kilometres; ``None`` skips those fields.
        prognostic_observation_barriers: Wind-interpolation barrier segments in
            local kilometres.
        prognostic_observation_barrier_top_level: Number of lowest levels to
            which barriers apply.

    Returns:
        :class:`Step1Fields` on the target grid. Physical arrays are
        ``float32``; winds and temperatures have shape ``(level, y, x)`` and
        surface diagnostics ``(y, x)``.

    Raises:
        ValueError: If source geometry, vertical profiles, cloud method, SST,
            precipitation controls, or target-domain coverage is invalid.
    """
    lat=np.asarray(dataset['latitude'][:]); lon=np.asarray(dataset['longitude'][:])
    if mapper is None:
        mapper=PrognosticMapper.build(lat,lon,grid,validate_domain=True)
    # CALWRF stores geopotential height MSL.  RDMM5 subtracts the source-cell
    # terrain before all vertical interpolation and CLOUD4 calculations.
    terrain=np.asarray(dataset['terrain'][:],dtype=MODEL_DTYPE)
    height=(np.asarray(dataset['height'][time_index],dtype=MODEL_DTYPE)-
            terrain[None,:,:])
    speed=np.asarray(dataset['wind_speed'][time_index],dtype=MODEL_DTYPE)
    direction=np.asarray(dataset['wind_direction'][time_index],dtype=MODEL_DTYPE)
    source_sun=None
    if timestamp is not None:
        source_sun=_prognostic_record_solar(timestamp,base_timezone,lat,lon)
    convergence=np.asarray(grid_convergence(grid,lon,lat),dtype=MODEL_DTYPE)
    source_u,source_v=_prognostic_wind_components(speed,direction,convergence[None,:,:])
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    centers=.5*(faces[:-1]+faces[1:])
    wind_height,source_u_above,source_v_above=_rdmm5_above_ground_profiles(
        height,source_u,source_v)
    source_u_centers=_vertical_centers(source_u_above,wind_height,centers,wind=True)
    source_v_centers=_vertical_centers(source_v_above,wind_height,centers,wind=True)
    u=mapper.interpolate(source_u_centers); v=mapper.interpolate(source_v_centers)
    prognostic_u=prognostic_v=prognostic_strength=None
    if prognostic_observation_rmin_km is not None:
        prognostic_u,prognostic_strength=mapper.interpolate_observations(
            source_u_centers,prognostic_observation_rmin_km,
            barriers=prognostic_observation_barriers,
            barrier_top_level=prognostic_observation_barrier_top_level)
        prognostic_v,_=mapper.interpolate_observations(
            source_v_centers,prognostic_observation_rmin_km,
            barriers=prognostic_observation_barriers,
            barrier_top_level=prognostic_observation_barrier_top_level)
    source_temperature=np.asarray(dataset['temperature'][time_index],dtype=MODEL_DTYPE)
    source_q=np.asarray(dataset['qvapor'][time_index],dtype=MODEL_DTYPE)
    source_pressure=np.asarray(dataset['pressure'][time_index],dtype=MODEL_DTYPE)
    theta=_model_array(source_temperature*np.power(1000./source_pressure,.286))
    if np.any(np.sum(height>=0.,axis=0)<2):
        raise ValueError('RDMM5 temperature profile requires two levels at or above ground')
    (thermo_height,source_temperature_above,source_q_above,
     source_pressure_above,theta_above)=_rdmm5_above_ground_profiles(
        height,source_temperature,source_q,source_pressure,theta)
    theta2,theta3=theta_above[0],theta_above[1]
    z2,z3=thermo_height[0],thermo_height[1]
    if thermal_state is None:
        scale=z3/5.
        upper_exp=np.exp(-.77*z3/scale)
        lower_exp=np.exp(-.77*z2/scale)
        denominator=upper_exp-lower_exp
        surface_drop=np.divide(theta3-theta2,denominator,
                               out=np.zeros_like(theta2),where=denominator!=0.)
        surface_drop=np.clip(surface_drop,-15.,0.)
        theta30=_model_array(theta3-surface_drop*upper_exp)
        theta20=theta30.copy()
    else:
        theta20=np.asarray(thermal_state.pt20,dtype=MODEL_DTYPE)
        theta30=np.asarray(thermal_state.pt30,dtype=MODEL_DTYPE)
    if source_sun is not None:
        daylight=source_sun>0.; theta20=np.where(daylight,theta2,theta20); theta30=np.where(daylight,theta3,theta30)
    next_thermal_state=SourceThermalState(theta20.copy(),theta30.copy())
    source_temperature_centers=_vertical_centers(
        source_temperature_above,thermo_height,centers)
    source_sst=None
    if water_temperature_mode==2:
        if 'sst' not in dataset.variables:
            raise ValueError('ITWPROG=2 requires SST in prognostic data')
        source_sst=np.asarray(dataset['sst'][time_index],dtype=MODEL_DTYPE)
        if np.any((source_sst<150.)|(source_sst>350.)):
            raise ValueError('ITWPROG=2 requires prognostic SST between 150 and 350 K')
    if source_sun is not None:
        below=centers[:,None,None]<z2[None,:,:]
        daytime_below=below&(source_sun[None,:,:]>0.)
        extrapolated=(source_temperature_above[0][None,:,:]+.0098*
                      (z2[None,:,:]-centers[:,None,None]))
        source_temperature_centers=np.where(daytime_below,extrapolated,source_temperature_centers)
        night_below=below&~(source_sun[None,:,:]>0.)
        ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        fraction=(centers[:,None,None]-z2[None,:,:])/(z3-z2)[None,:,:]
        water_theta=theta2[None,:,:]+fraction*(theta3-theta2)[None,:,:]
        land_theta=np.stack([_stull_temperature(theta2,z2,theta3,z3,theta20,theta30,center) for center in centers])
        below_theta=np.where(ocean[None,:,:],water_theta,land_theta)
        beta=np.log(source_pressure[0]/source_pressure[1])/(height[1]-height[0])
        center_pressure=source_pressure[0][None,:,:]*np.exp(
            -beta[None,:,:]*(centers[:,None,None]-height[0][None,:,:]))
        night_temperature=below_theta*np.power(center_pressure/1000.,.286)
        source_temperature_centers=np.where(night_below,night_temperature,source_temperature_centers)
    if source_sst is not None:
        ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        sst_profile,use_sst=_rdmm5_sst_profile(
            centers,z2,source_temperature_above[0],source_sst,ocean)
        source_temperature_centers=np.where(use_sst,sst_profile,
                                            source_temperature_centers)
    temperature=mapper.interpolate(source_temperature_centers)
    q=mapper.interpolate(_vertical_centers(source_q_above,thermo_height,centers))
    pressure=mapper.interpolate(_vertical_centers(source_pressure_above,thermo_height,centers))
    # Prefer the canonical relative-humidity field when it is available.
    source_rh=_source_relative_humidity(
        dataset,time_index,source_q,source_pressure,source_temperature)
    _,source_rh_above=_rdmm5_above_ground_profiles(height,source_rh)
    rh850=None
    if cloud_method==4:
        source_cloud,source_ceiling=cloud_cover_mm5(source_rh,source_pressure,height)
    elif cloud_method==3:
        below=source_pressure<850.; first=np.argmax(below,axis=0); level=np.maximum(first-1,0)
        yy,xx=np.indices(level.shape); source_rh850=np.where(below.any(axis=0),source_rh[level,yy,xx],0.)
        source_cloud=cloud_fraction(source_rh850)
        source_ceiling=np.full_like(source_cloud,80.*30.48)
    else:
        raise ValueError('cloud_method must be MCLOUD 3 or 4')
    has_qcloud='qcloud' in dataset.variables
    has_qice='qice' in dataset.variables
    if has_qcloud or has_qice:
        source_qcloud=(_cloud_water_g_kg(dataset['qcloud'][time_index])
                       if has_qcloud else np.zeros_like(height,dtype=MODEL_DTYPE))
        source_qice=(_cloud_water_g_kg(dataset['qice'][time_index])
                     if has_qice else np.zeros_like(height,dtype=MODEL_DTYPE))
        source_condensate=_model_array(source_qcloud+source_qice)
        mapped_condensate=mapper.interpolate(source_condensate)
        # Retain the historical internal field name, but store a per-column
        # maximum in g/kg rather than summing mixing ratios over model levels.
        cloud_water=_model_array(np.max(mapped_condensate,axis=0))
        cloudy=cloud_water>=cloud_condensate_threshold_g_kg
        present=source_condensate>=cloud_condensate_threshold_g_kg
        first=np.argmax(present,axis=0)
        source_qc_ceiling=np.take_along_axis(height,first[None,:,:],axis=0)[0]
        source_qc_ceiling=np.where(present.any(axis=0),source_qc_ceiling,0.)
        mapped_ceiling=mapper.interpolate(source_qc_ceiling,ignore_zero=True)
    else:
        cloudy=np.ones(mapper.target_shape,dtype=bool)
        cloud_water=None
        mapped_ceiling=mapper.interpolate(source_ceiling,ignore_zero=True)
    if has_qcloud:
        source_aux=cloud_liquid_water_profile(source_qcloud,height,z_faces_m)
        auxiliary_cloud=(mapper.nearest(source_aux[0]),mapper.nearest(source_aux[1]),
                         mapper.nearest(source_aux[2]),mapper.nearest(source_aux[3]))
    else:
        auxiliary_cloud=(None,None,None,None)
    mapped_source_cloud,rh850=map_prognostic_cloud(
        source_cloud,source_rh850 if cloud_method==3 else None,mapper,cloud_method)
    mapped_cloud=np.where(cloudy,mapped_source_cloud,0.)
    mapped_ceiling=np.where(cloudy,mapped_ceiling,0.)
    surface_profile=source_temperature_above[0]+.0098*z2
    if source_sun is not None:
        water_surface=theta2-z2*(theta3-theta2)/(z3-z2)
        land_surface=_stull_temperature(theta2,z2,theta3,z3,theta20,theta30,0.)
        surface_theta=np.where(
            np.asarray(dataset['landuse'][:])==prognostic_ocean_category,
            water_surface,land_surface)
        beta=np.log(source_pressure[0]/source_pressure[1])/(height[1]-height[0])
        surface_p=source_pressure[0]*np.exp(beta*height[0])
        night_surface=surface_theta*np.power(surface_p/1000.,.286)
        surface_profile=np.where(source_sun>0.,surface_profile,night_surface)
    if source_sst is not None:
        ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        surface_profile=_model_array(np.where(ocean,source_sst,surface_profile))
    profile_temperature=np.concatenate((mapper.nearest(surface_profile)[None,:,:],
                                        mapper.nearest(source_temperature_above)),axis=0)
    profile_height=np.concatenate((np.zeros((1,grid.ny,grid.nx),dtype=MODEL_DTYPE),
                                   mapper.nearest(thermo_height)),axis=0)
    surface_temperature=mapper.interpolate(np.asarray(
        dataset['temperature_2m'][time_index],dtype=MODEL_DTYPE))
    raw_surface_pressure=np.asarray(
        dataset['surface_pressure'][time_index],dtype=MODEL_DTYPE)
    surface_pressure=mapper.interpolate(raw_surface_pressure*.01)
    surface_density=mapper.interpolate(_rdmm5_surface_density(
        source_pressure,source_temperature,height))
    # Surface RH follows the first three-dimensional prognostic layer.
    surface_rh=mapper.interpolate(source_rh[0])
    precipitation=mapper.radius_interpolate(np.asarray(dataset['precipitation'][time_index]),
        precipitation_radius_km,cutoff=precipitation_cutoff,method=precipitation_interpolation)
    overwater_data=None
    if source_sst is not None:
        from .sea_data import GriddedSeaData
        source_ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        nearest_ocean=lambda field: mapper.nearest_masked(field,source_ocean)
        ocean_height=nearest_ocean(z2)
        ocean_longitude=mapper.nearest_masked(
            lon,source_ocean,dtype=COORDINATE_DTYPE)
        shape=mapper.target_shape
        overwater_data=GriddedSeaData(
            nearest_ocean(source_temperature_above[0]),nearest_ocean(source_sst),
            nearest_ocean(source_rh_above[0]),ocean_height,ocean_height,
            np.full(shape,.05,dtype=MODEL_DTYPE),
            np.full(shape,-999.,dtype=MODEL_DTYPE),
            np.full(shape,-999.,dtype=MODEL_DTYPE),ocean_longitude)
    return Step1Fields(
        u,v,temperature,
        surface_temperature,
        surface_pressure,
        surface_density,
        mapper.interpolate(_vertical_centers(source_rh_above,thermo_height,centers)),
        mapper.interpolate(np.asarray(dataset['pbl_height'][time_index])),
        mapper.interpolate(np.asarray(dataset['sensible_heat_flux'][time_index])),
        mapper.interpolate(np.asarray(dataset['friction_velocity'][time_index])),
        mapped_cloud,
        mapped_ceiling,
        profile_temperature,
        profile_height,
        surface_rh,
        precipitation,
        np.where(precipitation>0.,np.where(temperature[0]<273.15,20,10),0).astype(np.int32),
        next_thermal_state,
        rh850,
        cloud_water,
        prognostic_observation_u=prognostic_u,
        prognostic_observation_v=prognostic_v,
        prognostic_observation_strength=prognostic_strength,
        overwater_data=overwater_data,
        prognostic_temperature=temperature,
        cloud_liquid_water_3d=auxiliary_cloud[0],
        cloud_liquid_water_above=auxiliary_cloud[1],
        cloud_liquid_water_above_bottom=auxiliary_cloud[2],
        cloud_liquid_water_above_top=auxiliary_cloud[3],
    )


def build_prog_step_fields(prog_data,ending_hour,grid: GridConfig,z_faces_m,*,record=None):
    """Create the wind-only Step-1 scaffold used by legacy ``PROG.DAT``.

    Args:
        prog_data: Parsed prognostic wind data with interpolation methods.
        ending_hour: Decimal ``HH.MM`` ending-time label used to select a PROG
            record.
        grid: Target CALMET grid.
        z_faces_m: CALMET layer faces in metres AGL.
        record: Optional already-selected PROG record; ``None`` lets the data
            object select by ``ending_hour``.

    Returns:
        :class:`Step1Fields` with interpolated ``float32 (level, y, x)`` U/V
        winds in m/s and neutral placeholder thermodynamic fields.
    """
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    centers=.5*(faces[:-1]+faces[1:])
    u,v=(prog_data.interpolate(ending_hour,grid,centers) if record is None else
         prog_data.interpolate_record(record,grid,centers))
    shape=(grid.ny,grid.nx)
    level_shape=(centers.size,*shape)
    surface_temperature=np.full(shape,288.7,dtype=MODEL_DTYPE)
    profile_height=np.broadcast_to(np.r_[MODEL_DTYPE(0.),centers][:,None,None],
                                   (centers.size+1,*shape)).copy()
    profile_temperature=np.full_like(profile_height,288.7)
    return Step1Fields(
        u,v,np.full(level_shape,288.7,dtype=MODEL_DTYPE),surface_temperature,
        np.full(shape,1000.,dtype=MODEL_DTYPE),
        np.full(shape,.3484321*1000./288.7,dtype=MODEL_DTYPE),
        np.full(level_shape,80.,dtype=MODEL_DTYPE),
        np.full(shape,50.,dtype=MODEL_DTYPE),np.zeros(shape,dtype=MODEL_DTYPE),
        np.full(shape,.1,dtype=MODEL_DTYPE),np.zeros(shape,dtype=MODEL_DTYPE),
        np.zeros(shape,dtype=MODEL_DTYPE),profile_temperature,profile_height,
        np.full(shape,80.,dtype=MODEL_DTYPE),np.zeros(shape,dtype=MODEL_DTYPE),
        np.zeros(shape,dtype=np.int32),prognostic_temperature=np.full(
            level_shape,288.7,dtype=MODEL_DTYPE))


def build_mm4_step_fields(mm4_data,record,grid: GridConfig,z_faces_m,*,cloud_method=4,
                          precipitation_interpolation=2,precipitation_radius_km=50.,
                          precipitation_cutoff=.01,base_timezone=0,thermal_state=None,
                          prognostic_observation_rmin_km=None,
                          prognostic_observation_barriers=(),
                          prognostic_observation_barrier_top_level=0,
                          prognostic_ocean_category=16):
    """Create Step-1 fields from a native MM4 meteorological record.

    Args:
        mm4_data: Parsed MM4 data object supporting record interpolation.
        record: Selected MM4 record.
        grid: Target CALMET grid.
        z_faces_m: CALMET layer faces in metres AGL.
        cloud_method: Prognostic cloud method 3 or 4.
        precipitation_interpolation: Radius mapper method 1, 2, or 3.
        precipitation_radius_km: Precipitation interpolation radius or
            Gaussian sigma in kilometres.
        precipitation_cutoff: Mapped precipitation below this mm/h value is
            zeroed.
        base_timezone: CALMET hours west of UTC.
        thermal_state: Optional RDMM4 daylight thermal state from the previous
            record.
        prognostic_observation_rmin_km: Optional minimum pseudo-observation
            wind distance in kilometres.
        prognostic_observation_barriers: Wind barrier segments in local km.
        prognostic_observation_barrier_top_level: Number of lowest levels to
            which barriers apply.
        prognostic_ocean_category: MM4 land-use category treated as ocean.

    Returns:
        Complete :class:`Step1Fields` on ``(grid.ny, grid.nx)`` with
        ``float32`` level-first meteorology and ``int32`` precipitation codes.

    Raises:
        ValueError: If MM4 interpolation controls, source data, or target
            geometry are invalid.
    """
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    centers=.5*(faces[:-1]+faces[1:])
    fields=mm4_data.interpolate_record(record,grid,centers,cloud_method=cloud_method,
        precipitation_method=precipitation_interpolation,
        precipitation_radius_km=precipitation_radius_km,
        precipitation_cutoff=precipitation_cutoff,base_timezone=base_timezone,
        thermal_state=thermal_state,
        prognostic_observation_rmin_km=prognostic_observation_rmin_km,
        prognostic_observation_barriers=prognostic_observation_barriers,
        prognostic_observation_barrier_top_level=prognostic_observation_barrier_top_level,
        prognostic_ocean_category=prognostic_ocean_category)
    shape=(grid.ny,grid.nx)
    return Step1Fields(fields.u,fields.v,fields.temperature_k,
        fields.surface_temperature_k,fields.surface_pressure_mb,fields.surface_density,
        fields.relative_humidity,np.zeros(shape,dtype=MODEL_DTYPE),
        np.zeros(shape,dtype=MODEL_DTYPE),np.full(shape,.1,dtype=MODEL_DTYPE),
        fields.cloud_fraction,fields.ceiling_height_m,fields.profile_temperature_k,
        fields.profile_height_m,_model_array(fields.surface_relative_humidity),
        fields.precipitation_mm_h,fields.precipitation_code,fields.thermal_state,
        prognostic_observation_u=fields.prognostic_observation_u,
        prognostic_observation_v=fields.prognostic_observation_v,
        prognostic_observation_strength=fields.prognostic_observation_strength,
        prognostic_temperature=fields.temperature_k)

def build_surface_fields(step: Step1Fields, landuse, roughness, albedo,
                         bowen_ratio, soil_heat_flux_parameter,
                         anthropogenic_heat_flux, solar_elevation_sine,
                         coriolis, previous_convective_height,
                         previous_temperature_jump, *, first_level_height_m=10.,
                         minimum_height_m=50., maximum_height_m=2500.,
                         minimum_water_height_m=50., maximum_water_height_m=2500.,
                         water_range=(50,55),
                         turbidity_coefficient_1=990.,turbidity_coefficient_2=-30.,
                         cloudiness_coefficient_1=-.75,cloudiness_coefficient_2=3.4,
                         net_radiation_coefficient_1=5.31e-13,
                         net_radiation_coefficient_2=60.,net_radiation_coefficient_3=.12,
                         mixing_method=1, minimum_lapse=.001, lapse_depth_m=200.,
                         threshold_flux_per_m=0., constb=1.41, conste=.15,
                         constn=2400., constw=.16, relaxation=True,
                         relaxation_time_s=800.,
                         timestep_s=3600.,
                         surface_temperature=None,
                         overwater_flux_method=0,coastal_shelf_length_km=0.,
                         cool_skin=False,dx_km=1.,dy_km=1.,
                         water_flux_threshold=0.,water_temperature_mode=0,
                         default_water_gradient_above=-.0045,
                         warm_layer=False,warm_layer_state=None,
                         longitude=None,gmt_hour=0.,
                         observed_water_mixing_height=None,
                         previous_water_convective_height=None,
                         previous_water_mixing_height=None):
    """Calculate COMP surface similarity and mixing-height fields.

    Args:
        step: Step-1 meteorology for the current interval.
        landuse: ``(y, x)`` dominant land-use category.
        roughness: Aerodynamic roughness length in metres.
        albedo: Surface albedo fraction.
        bowen_ratio: Bowen ratio used in land heat-flux partitioning.
        soil_heat_flux_parameter: Fraction controlling soil heat storage.
        anthropogenic_heat_flux: Anthropogenic heat flux in W/m2.
        solar_elevation_sine: Sine of midpoint solar elevation.
        coriolis: Coriolis parameter in 1/s.
        previous_convective_height: Previous land convective height in metres.
        previous_temperature_jump: Previous inversion jump in K.
        first_level_height_m: Height of the lowest CALMET layer centre in m.
        minimum_height_m: Minimum overland mixing height in metres.
        maximum_height_m: Maximum overland mixing height in metres.
        minimum_water_height_m: Minimum overwater mixing height in metres.
        maximum_water_height_m: Maximum overwater mixing height in metres.
        water_range: Inclusive land-use category range treated as water.
        turbidity_coefficient_1: First empirical clear-sky turbidity
            coefficient.
        turbidity_coefficient_2: Second clear-sky turbidity coefficient.
        cloudiness_coefficient_1: First cloud attenuation coefficient.
        cloudiness_coefficient_2: Second cloud attenuation coefficient.
        net_radiation_coefficient_1: First net longwave-radiation coefficient.
        net_radiation_coefficient_2: Second net-radiation coefficient.
        net_radiation_coefficient_3: Third net-radiation coefficient.
        mixing_method: ``1`` or ``-1`` for Maul-Carson, ``2`` or ``-2`` for
            Batchvarova-Gryning; negative values select the alternate water
            pathway.
        minimum_lapse: Minimum potential-temperature lapse rate in K/m.
        lapse_depth_m: Near-surface depth used to diagnose lapse rate, in m.
        threshold_flux_per_m: Land heat-flux-per-height threshold in W/m3.
        constb: Maul-Carson mechanical-height coefficient.
        conste: Batchvarova-Gryning entrainment coefficient.
        constn: Batchvarova-Gryning neutral boundary-layer constant in m.
        constw: Overwater mechanical-height coefficient.
        relaxation: Relax convective height during weak positive heat flux.
        relaxation_time_s: Convective-height relaxation e-folding time in s.
        timestep_s: Model timestep in seconds.
        surface_temperature: Optional final ``(y, x)`` temperature in K;
            ``None`` uses the lowest Step-1 layer.
        overwater_flux_method: ``0`` for OCD or an absolute COARE method code
            of 10 through 12.
        coastal_shelf_length_km: COARE nearshore roughness scale in km.
        cool_skin: Enable the COARE cool-skin correction.
        dx_km: Horizontal x spacing in kilometres.
        dy_km: Horizontal y spacing in kilometres.
        water_flux_threshold: Water heat-flux-per-height threshold in W/m3.
        water_temperature_mode: Select observed or prognostic overwater
            temperature gradients.
        default_water_gradient_above: Fallback gradient above the water
            mixing height in K/m.
        warm_layer: Enable the COARE warm-layer correction.
        warm_layer_state: Optional COARE state carried from the prior step.
        longitude: Optional ``(y, x)`` longitude in degrees east for COARE.
        gmt_hour: Interval-ending UTC decimal hour for COARE state updates.
        observed_water_mixing_height: Optional observed water mixing height in
            metres.
        previous_water_convective_height: Optional prior water convective-
            height state in metres.
        previous_water_mixing_height: Optional prior total water mixing height
            in metres.

    Returns:
        :class:`SurfaceFields` containing ``float32 (y, x)`` fluxes,
        similarity parameters, and mixing-height states in their documented
        physical units. A supplied warm-layer state may be advanced.

    Raises:
        ValueError: If ``mixing_method`` is unsupported or an observed or
            carried overwater state does not match the horizontal grid.
    """
    landuse=np.asarray(landuse); water=(landuse>=water_range[0])&(landuse<=water_range[1]); land=~water
    temp=np.asarray(step.temperature[0] if surface_temperature is None else
                    surface_temperature,dtype=MODEL_DTYPE)
    density=np.asarray(step.surface_density,dtype=MODEL_DTYPE)
    qh,qsw=heat_flux(solar_elevation_sine,step.cloud_fraction,temp,
        landuse,albedo,bowen_ratio,soil_heat_flux_parameter,anthropogenic_heat_flux,
        water_range=water_range,turbidity_coefficient_1=turbidity_coefficient_1,
        turbidity_coefficient_2=turbidity_coefficient_2,
        cloudiness_coefficient_1=cloudiness_coefficient_1,
        cloudiness_coefficient_2=cloudiness_coefficient_2,
        net_radiation_coefficient_1=net_radiation_coefficient_1,
        net_radiation_coefficient_2=net_radiation_coefficient_2,
        net_radiation_coefficient_3=net_radiation_coefficient_3)
    ustar,length,qh=surface_similarity(step.u[0],step.v[0],roughness,first_level_height_m,
        density,temp,step.cloud_fraction,qh,landuse,water_range=water_range)
    common=dict(depth_m=lapse_depth_m,minimum_lapse=minimum_lapse,
        minimum_height_m=minimum_height_m,maximum_height_m=maximum_height_m,
        threshold_flux_per_m=threshold_flux_per_m,constb=constb,constn=constn,
        relaxation=relaxation,relaxation_time_s=relaxation_time_s,land_mask=land,
        timestep_s=timestep_s)
    if abs(mixing_method)==1:
        zi,zic,jump=mixing_height_maul_carson(step.profile_temperature,step.profile_height,
            previous_convective_height,previous_temperature_jump,ustar,length,qh,density,coriolis,
            conste=conste,**common)
    elif abs(mixing_method)==2:
        zi,zic,jump=mixing_height_batchvarova_gryning(step.profile_temperature,step.profile_height,
            previous_convective_height,previous_temperature_jump,ustar,length,qh,density,coriolis,
            surface_temperature_k=temp,**common)
    else:
        raise ValueError('mixing_method must be one of -2, -1, 1, or 2')
    water_z0,water_ustar,water_length,water_qh,water_zi,water_zic=neutral_water_parameters(
        step.u[0],step.v[0],coriolis,constw=constw,minimum_height_m=minimum_water_height_m,
        maximum_height_m=maximum_water_height_m)
    initial_water_mechanical=_model_array(
        constw*water_ustar/np.asarray(coriolis,dtype=MODEL_DTYPE))
    # WATERP (ITWPROG=2) always computes offshore mixing height from the
    # prognostic profile.  SEA.DAT may still be open for wave observations,
    # but its ZIOW records are not consulted on this path.
    if water_temperature_mode==2:
        observed_water_mixing_height=None
    if observed_water_mixing_height is None:
        observed_water=np.zeros_like(water,dtype=bool)
        observed_zi=np.zeros_like(water_zi,dtype=MODEL_DTYPE)
    else:
        observed_zi=np.asarray(observed_water_mixing_height,dtype=MODEL_DTYPE)
        if observed_zi.shape!=water.shape:
            raise ValueError('observed water mixing-height shape mismatch')
        observed_water=water&np.isfinite(observed_zi)
        water_zi=_model_array(np.where(observed_water,observed_zi,water_zi))
    coare_water_zi=_model_array(np.where(
        observed_water,observed_zi,initial_water_mechanical))
    # Preserve each water cell's state until its own surface calculation
    # supplies an update; land cells use the newly calculated jump.
    water_jump=_model_array(np.where(
        water,np.asarray(previous_temperature_jump,dtype=MODEL_DTYPE),
        np.asarray(jump,dtype=MODEL_DTYPE)))
    water_theta=temp
    water_state=(np.zeros_like(water_zi,dtype=MODEL_DTYPE)
                 if previous_water_convective_height is None else
                 np.asarray(previous_water_convective_height,dtype=MODEL_DTYPE).copy())
    if water_state.shape!=water.shape:
        raise ValueError('previous water convective-height state shape mismatch')
    if step.overwater_data is not None:
        sea=step.overwater_data
        air_temperature=np.asarray(sea.air_temperature_k,dtype=MODEL_DTYPE)
        humidity=_model_array(np.where(
            sea.relative_humidity<9998.99,sea.relative_humidity,100.))
        saturation=saturation_vapor_pressure_lowe(air_temperature-273.15)
        vapor_pressure=.01*humidity*saturation
        mixing_ratio=.622*vapor_pressure/(1000.-vapor_pressure)
        specific_humidity=_model_array(mixing_ratio/(1.+mixing_ratio)*1000.)
        water_theta=_model_array(
            air_temperature*(1.+.61*mixing_ratio)+
            .01*np.asarray(sea.anemometer_height_m,dtype=MODEL_DTYPE))
        if overwater_flux_method==0:
            water_theta,difference,difference_10m,theta_10m=(
                overwater_delta_temperature_state(
                    sea.air_temperature_k,sea.sea_temperature_k,humidity,
                    sea.anemometer_height_m))
            water_length,water_ustar,water_qh=delta_temperature_flux(
                np.hypot(step.u[0],step.v[0]),sea.anemometer_height_m,difference,
                difference_10m,theta_10m,water_theta,water_z0)
            water_length=np.asarray(water_length,dtype=MODEL_DTYPE)
            water_ustar=np.asarray(water_ustar,dtype=MODEL_DTYPE)
            water_qh=np.asarray(water_qh,dtype=MODEL_DTYPE)
            water_wt=_model_array(water_qh/1195.2)
        else:
            longwave=downward_longwave(temp,step.surface_relative_humidity,
                                       step.cloud_fraction,step.ceiling_height)
            coast=distance_to_coast(landuse,water_range,dx_km,dy_km)
            # WATER/WATERP clamp SST only on entry to COARE.  DELTAT above
            # deliberately receives the original, potentially colder value.
            sea_temperature=np.maximum(
                np.asarray(sea.sea_temperature_k,dtype=MODEL_DTYPE),270.15)
            next_warm_state=warm_layer_state
            if warm_layer:
                if next_warm_state is None:
                    next_warm_state=initial_warm_layer_state(np.shape(sea_temperature))
                # WATER uses XOWLON at the selected SEA station; WATERP uses
                # XLONG4 at IGRABW/JGRABW.  Both are stored on the gridded sea
                # object, not at the CALMET target-cell longitude.
                sea_longitude=np.asarray(sea.station_longitude)
                solar_time=np.mod(sea_longitude/15.+gmt_hour+24.,24.)*3600.
                sea_temperature,next_warm_state,_=advance_warm_layer(
                    next_warm_state,solar_time,sea_temperature,sea.water_sensor_depth_m,
                    qsw,longwave)
            flux=surface_flux(np.hypot(step.u[0],step.v[0]),sea.air_temperature_k,
                sea_temperature,specific_humidity,
                wind_height_m=first_level_height_m,
                temperature_height_m=sea.air_sensor_height_m,pressure_mb=1008.,
                boundary_layer_height_m=coare_water_zi,shortwave_down_w_m2=qsw,
                longwave_down_w_m2=longwave,rainfall_mm_h=step.precipitation_rate,
                wave_method=abs(overwater_flux_method)-10,
                wave_period_s=(sea.wave_period_s if overwater_flux_method<0 else -999.),
                wave_height_m=(sea.wave_height_m if overwater_flux_method<0 else -999.),
                coastal_shelf_length_km=coastal_shelf_length_km,
                distance_to_coast_km=coast,cool_skin=cool_skin)
            water_z0=np.asarray(flux.roughness_m,dtype=MODEL_DTYPE)
            water_ustar=np.asarray(flux.friction_velocity_m_s,dtype=MODEL_DTYPE)
            water_length=np.asarray(flux.monin_obukhov_m,dtype=MODEL_DTYPE)
            water_length=_model_array(np.where(
                np.abs(water_length)<5.,np.where(water_length<0.,-5.,5.),water_length))
            water_wt=np.asarray(flux.buoyancy_flux_k_m_s,dtype=MODEL_DTYPE)
            water_qh=_model_array(water_wt*1195.2)
            if warm_layer:
                next_warm_state=update_warm_layer_fluxes(next_warm_state,flux)
        mechanical=water_mechanical_height(water_ustar,coriolis,constw=constw,
            minimum_height_m=minimum_water_height_m,
            maximum_height_m=maximum_water_height_m)
        if mixing_method>0:
            prior_water_zi=(np.full_like(water_state,minimum_water_height_m)
                            if previous_water_mixing_height is None else
                            np.asarray(previous_water_mixing_height,dtype=MODEL_DTYPE))
            if prior_water_zi.shape!=water.shape:
                raise ValueError('previous water mixing-height state shape mismatch')
            # WATER first bounds last hour's ZI by ZIMINW, then constrains its
            # saved ZICOLD not to exceed that total height.
            prior_water_zi=np.maximum(prior_water_zi,minimum_water_height_m)
            old_water=np.maximum(np.minimum(water_state,prior_water_zi),
                                 minimum_water_height_m)
            if water_temperature_mode==0:
                gradient=(default_water_gradient_above
                          if step.water_mixing_gradient_above is None
                          else step.water_mixing_gradient_above)
                potential_lapse=np.asarray(gradient,dtype=MODEL_DTYPE)+.0098
            else:
                _,_,potential_lapse=profile_temperature_layer(
                    step.profile_temperature,step.profile_height,old_water,lapse_depth_m,
                    minimum_lapse=minimum_lapse)
            if abs(mixing_method)==1:
                active_water=water&~observed_water
                calculated_zic,calculated_jump=convective_height_maul_carson(
                    old_water,np.asarray(previous_temperature_jump,dtype=MODEL_DTYPE),
                    water_wt,np.broadcast_to(
                        np.asarray(potential_lapse,dtype=MODEL_DTYPE),water.shape),
                    timestep_s=timestep_s,
                    threshold_flux_per_m=water_flux_threshold,
                    maximum_height_m=maximum_water_height_m,
                    minimum_lapse=minimum_lapse,conste=conste,
                    relaxation=relaxation,
                    relaxation_time_s=relaxation_time_s)
                water_zic=_model_array(np.where(active_water,calculated_zic,0.))
                water_jump=_model_array(np.where(
                    water,np.where(active_water,calculated_jump,
                                   np.asarray(previous_temperature_jump,
                                              dtype=MODEL_DTYPE)),
                    np.asarray(jump,dtype=MODEL_DTYPE)))
            else:
                water_zic=convective_height_batchvarova_gryning(
                    old_water,water_wt,potential_lapse,water_ustar,
                    water_length,sea.air_temperature_k,timestep_s=timestep_s,
                    minimum_height_m=minimum_water_height_m,
                    maximum_height_m=maximum_water_height_m,
                    threshold_flux_per_m=water_flux_threshold,minimum_lapse=minimum_lapse,
                    relaxation=relaxation,relaxation_time_s=relaxation_time_s)
                water_jump=np.asarray(previous_temperature_jump,dtype=MODEL_DTYPE)
            water_zi=_model_array(np.clip(
                np.maximum(mechanical,np.asarray(water_zic,dtype=MODEL_DTYPE)),
                minimum_water_height_m,maximum_water_height_m))
            water_state=_model_array(np.where(
                water&~observed_water,water_zic,water_state))
        else:
            water_zi=mechanical; water_zic=np.zeros_like(mechanical)
            water_jump=np.zeros_like(mechanical)
            water_state=_model_array(np.where(water&~observed_water,0.,water_state))
    water_zi=_model_array(np.where(observed_water,observed_zi,water_zi))
    water_wstar=convective_velocity_scale(
        water_zi if mixing_method<0 else water_zic,water_qh,water_theta,density,
        landuse,water_range=water_range)
    # WATER's scalar ZICONV is local to that subroutine and is used for its
    # pre-AVEMIX WSTAR only.  The COMP/MIXHT ZICONV grid skips water cells,
    # retaining their previous value until optional AVEMIX changes the grid.
    grid_zic=np.where(water,np.asarray(previous_convective_height,dtype=MODEL_DTYPE),zic)
    return SurfaceFields(
        _model_array(np.where(water,water_z0,roughness)),_model_array(density),
        _model_array(qsw),_model_array(np.where(water,water_qh,qh)),
        _model_array(np.where(water,water_ustar,ustar)),
        _model_array(np.where(water,water_length,length)),
        _model_array(np.where(water,water_zi,zi)),_model_array(grid_zic),
        _model_array(water_jump if abs(mixing_method)==1 and mixing_method>0 else
                     np.where(water,water_jump,jump)),
        next_warm_state if step.overwater_data is not None and
        abs(overwater_flux_method)>=10 and warm_layer else warm_layer_state,
        _model_array(np.where(water,water_wstar,0.)),
        _model_array(water_state),
    )

def build_diagnostic_wind(step: Step1Fields, terrain_m, landuse, roughness,
                          albedo, bowen_ratio, soil_heat_flux_parameter,
                          anthropogenic_heat_flux, solar_elevation_sine,
                          previous_convective_height, z_faces_m,
                          *, dx_m=1000., dy_m=1000., terrain_radius_km=15.,
                          terrain_context=None,
                          lapse_height_m=200.,spatial_lapse=True,spatial_surface_temperature=True,
                          temperature_mode=0,
                          lapse_rate_override=None,
                          minimum_average_radius=10,half_average_angle=30.,
                          water_range=(50,55),
                          turbidity_coefficient_1=990.,turbidity_coefficient_2=-30.,
                          cloudiness_coefficient_1=-.75,cloudiness_coefficient_2=3.4,
                          net_radiation_coefficient_1=5.31e-13,
                          net_radiation_coefficient_2=60.,net_radiation_coefficient_3=.12,
                          critical_froude=1.,use_froude=True,use_kinematic=False,use_slope=True,use_obrien=False,
                          kinematic_alpha=.1,
                          timestep_s=3600.,
                          divergence_limit=5.e-6,divergence_iterations=50,
                          smoothing_passes=(2,4,4,4,4,4,4,4,4,4),
                          retain_debug_stages=False):
    """Apply CALMET diagnostic terrain effects to a Step-1 wind field.

    Args:
        step: Step-1 wind and thermodynamic fields.
        terrain_m: ``(y, x)`` terrain elevation in metres MSL.
        landuse: Dominant land-use category grid.
        roughness: Surface roughness length in metres.
        albedo: Surface albedo fraction.
        bowen_ratio: Surface Bowen ratio.
        soil_heat_flux_parameter: Soil heat-storage parameter.
        anthropogenic_heat_flux: Anthropogenic heat flux in W/m2.
        solar_elevation_sine: Sine of midpoint solar elevation.
        previous_convective_height: Prior convective height in metres AGL.
        z_faces_m: CALMET layer faces in metres AGL.
        dx_m: Horizontal x spacing in metres.
        dy_m: Horizontal y spacing in metres.
        terrain_radius_km: Radius in kilometres for terrain extrema.
        terrain_context: Optional precomputed ``(highest, lowest,
            distance_high, distance_low)`` terrain arrays in metres.
        lapse_height_m: Depth in metres used for lapse-rate diagnosis.
        spatial_lapse: Diagnose lapse rate per cell instead of from a domain
            mean profile.
        spatial_surface_temperature: Use spatial lowest-layer temperature in
            preliminary heat-flux calculations.
        temperature_mode: Temperature-source mode; value 2 retains
            prognostic surface density in the slope pathway.
        lapse_rate_override: Optional ``(y, x)`` lapse rate in K/m.
        minimum_average_radius: Minimum upwind averaging radius in cells.
        half_average_angle: Upwind cone half-angle in degrees.
        water_range: Inclusive land-use category range treated as water.
        turbidity_coefficient_1: First clear-sky turbidity coefficient.
        turbidity_coefficient_2: Second clear-sky turbidity coefficient.
        cloudiness_coefficient_1: First cloud attenuation coefficient.
        cloudiness_coefficient_2: Second cloud attenuation coefficient.
        net_radiation_coefficient_1: First net-radiation coefficient.
        net_radiation_coefficient_2: Second net-radiation coefficient.
        net_radiation_coefficient_3: Third net-radiation coefficient.
        critical_froude: Critical Froude number for flow blocking.
        use_froude: Apply thermodynamic terrain-flow blocking.
        use_kinematic: Apply kinematic terrain vertical velocity and mass
            consistency.
        use_slope: Add parameterized slope flow.
        use_obrien: Apply O'Brien top-boundary correction before divergence
            minimisation.
        kinematic_alpha: Kinematic terrain-effect multiplier.
        timestep_s: Model timestep in seconds for upwind averaging.
        divergence_limit: Convergence limit for divergence minimisation.
        divergence_iterations: Maximum divergence-minimisation iterations.
        smoothing_passes: Per-level horizontal wind smoothing counts.
        retain_debug_stages: Copy intermediate wind stages into the result.

    Returns:
        :class:`DiagnosticWindFields` with ``float32 (level, y, x)`` U/V in
        m/s, ``(face, y, x)`` W in m/s, divergence in 1/s, lapse rate in K/m,
        preliminary heat flux in W/m2, and optional stage snapshots.
    """
    terrain=np.asarray(terrain_m,dtype=MODEL_DTYPE)
    faces=np.asarray(z_faces_m,dtype=MODEL_DTYPE)
    centers=.5*(faces[:-1]+faces[1:])
    gamma=(lapse_rate(step.temperature,previous_convective_height,centers,lapse_height_m,spatial=spatial_lapse)
           if lapse_rate_override is None else np.asarray(lapse_rate_override,dtype=MODEL_DTYPE))
    if terrain_context is None:
        terrain_context=terrain_extrema(terrain,terrain_radius_km,dx_m,dy_m)
    highest,lowest,distance_high,distance_low=terrain_context
    u=np.asarray(step.u,dtype=MODEL_DTYPE)
    v=np.asarray(step.v,dtype=MODEL_DTYPE)
    if step.diagnostic_surface_temperature is not None:
        diagnostic_temperature=np.asarray(step.diagnostic_surface_temperature,
                                          dtype=MODEL_DTYPE)
    elif spatial_surface_temperature:
        diagnostic_temperature=np.asarray(step.temperature[0],dtype=MODEL_DTYPE)
    else:
        prognostic=(step.temperature if step.prognostic_temperature is None else
                    step.prognostic_temperature)
        diagnostic_temperature=np.full_like(
            step.temperature[0],_domain_mean(np.asarray(prognostic)[0]))
    stage_w=np.zeros((faces.size,*terrain.shape),dtype=MODEL_DTYPE)
    if use_kinematic:
        boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
        topo_w=topographic_vertical_velocity(u,v,terrain,z_faces_m,gamma,step.temperature[0],
                                             kinematic_alpha,dx_m,dy_m)
        stage_w=topo_w
        u,v,_,_=minimize_divergence(u,v,topo_w,z_faces_m,dx_m,dy_m,boundaries=boundaries,
            maximum_iterations=divergence_iterations,limit=divergence_limit)
    kinematic_stage=((u.copy(),v.copy(),stage_w.copy()) if retain_debug_stages else None)
    if use_froude:
        preliminary_temp,_=upwind_mixing_average(u[0],v[0],diagnostic_temperature,diagnostic_temperature,dx_m,dy_m,
            minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
        u,v=froude_adjustment(u,v,terrain,highest,gamma,preliminary_temp,centers,critical_froude,dx_m,dy_m)
    froude_stage=((u.copy(),v.copy(),stage_w.copy()) if retain_debug_stages else None)
    preliminary_qh=np.zeros_like(terrain)
    if use_slope:
        preliminary_temp,_=upwind_mixing_average(u[0],v[0],diagnostic_temperature,diagnostic_temperature,dx_m,dy_m,
            minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
        preliminary_qh,_=heat_flux(solar_elevation_sine,step.cloud_fraction,preliminary_temp,
            landuse,albedo,bowen_ratio,soil_heat_flux_parameter,anthropogenic_heat_flux,
            water_range=water_range,turbidity_coefficient_1=turbidity_coefficient_1,
            turbidity_coefficient_2=turbidity_coefficient_2,
            cloudiness_coefficient_1=cloudiness_coefficient_1,
            cloudiness_coefficient_2=cloudiness_coefficient_2,
            net_radiation_coefficient_1=net_radiation_coefficient_1,
            net_radiation_coefficient_2=net_radiation_coefficient_2,
            net_radiation_coefficient_3=net_radiation_coefficient_3)
        # DIAGNO calls AIRDEN here only while ITPROG < 2.  The complete
        # prognostic-temperature path retains RDMM5's density derived from
        # the first source-model level at or above ground.
        density=(np.asarray(step.surface_density,dtype=MODEL_DTYPE)
                 if temperature_mode==2 else
                 density_from_pressure_temperature(
                     step.surface_pressure_mb,preliminary_temp))
        _,_,preliminary_qh=surface_similarity(u[0],v[0],roughness,centers[0],density,
            preliminary_temp,step.cloud_fraction,preliminary_qh,landuse,water_range=water_range)
        slope_u,slope_v=slope_flow(terrain,highest,lowest,distance_high,distance_low,
            preliminary_qh,preliminary_temp,landuse,z_faces_m,dx_m,dy_m,water_range=water_range)
        u,v=u+slope_u,v+slope_v
    slope_stage=((u.copy(),v.copy(),stage_w.copy()) if retain_debug_stages else None)
    interpolated_stage=((u.copy(),v.copy()) if retain_debug_stages else None)
    terrain_adjusted_stage=((u.copy(),v.copy()) if retain_debug_stages else None)
    boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    u,v=smooth_wind(u,v,smoothing_passes)
    divergence,w=divergence_and_vertical_velocity(u,v,z_faces_m,dx_m,dy_m,obrien=use_obrien,boundaries=boundaries)
    initial_stage=((u.copy(),v.copy(),w.copy(),divergence.copy())
                   if retain_debug_stages else None)
    if use_obrien:
        u,v,divergence,_=minimize_divergence(u,v,w,z_faces_m,dx_m,dy_m,boundaries=boundaries,
            maximum_iterations=divergence_iterations,limit=divergence_limit)
    return DiagnosticWindFields(u,v,w,divergence,gamma,preliminary_qh,
        kinematic_stage,froude_stage,slope_stage,interpolated_stage,
        terrain_adjusted_stage,initial_stage)

def build_objective_wind(step: Step1Fields, terrain_m, previous_convective_height,
                         z_faces_m, *, dx_m, dy_m, lapse_height_m=200.,
                         spatial_lapse=True, lapse_rate_override=None,
                         use_obrien=False, divergence_limit=5.e-6,
                         divergence_iterations=50,
                         smoothing_passes=(2,4,4,4,4,4,4,4,4,4),
                         retain_debug_stages=False):
    """Finalize an IWFCOD=0 objective-analysis wind field.

    Args:
        step: Objectively analysed Step-1 wind and temperature fields.
        terrain_m: ``(y, x)`` terrain elevation in metres MSL.
        previous_convective_height: Prior convective height in metres AGL.
        z_faces_m: CALMET layer faces in metres AGL.
        dx_m: Horizontal x spacing in metres.
        dy_m: Horizontal y spacing in metres.
        lapse_height_m: Depth used for lapse-rate diagnosis in metres.
        spatial_lapse: Diagnose lapse rate per cell when true.
        lapse_rate_override: Optional ``(y, x)`` lapse rate in K/m.
        use_obrien: Apply the O'Brien top-boundary vertical-wind correction.
        divergence_limit: Convergence limit for divergence minimisation.
        divergence_iterations: Maximum divergence-minimisation iterations.
        smoothing_passes: Per-level horizontal smoothing counts.
        retain_debug_stages: Retain copies before terrain adjustment and
            divergence minimisation.

    Returns:
        :class:`DiagnosticWindFields` with ``float32`` U/V/W, divergence,
        lapse rate, zero preliminary heat flux, and optional debug snapshots.
    """
    gamma=(lapse_rate(step.temperature,previous_convective_height,
                      .5*(np.asarray(z_faces_m[:-1],dtype=MODEL_DTYPE)+
                          np.asarray(z_faces_m[1:],dtype=MODEL_DTYPE)),
                      lapse_height_m,spatial=spatial_lapse)
           if lapse_rate_override is None else np.asarray(lapse_rate_override,dtype=MODEL_DTYPE))
    interpolated=((np.asarray(step.u,dtype=MODEL_DTYPE).copy(),
                   np.asarray(step.v,dtype=MODEL_DTYPE).copy())
                  if retain_debug_stages else None)
    u,v=terrain_adjust_surface_wind(step.u,step.v,terrain_m,z_faces_m[-1],dx_m,dy_m)
    adjusted=((u.copy(),v.copy()) if retain_debug_stages else None)
    boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    u,v=smooth_wind(u,v,smoothing_passes)
    divergence,w=divergence_and_vertical_velocity(
        u,v,z_faces_m,dx_m,dy_m,obrien=use_obrien,boundaries=boundaries)
    initial=((u.copy(),v.copy(),w.copy(),divergence.copy())
             if retain_debug_stages else None)
    if use_obrien:
        u,v,divergence,_=minimize_divergence(
            u,v,w,z_faces_m,dx_m,dy_m,boundaries=boundaries,
            maximum_iterations=divergence_iterations,limit=divergence_limit)
    return DiagnosticWindFields(u,v,w,divergence,gamma,
        np.zeros_like(terrain_m,dtype=MODEL_DTYPE),interpolated_stage=interpolated,
        terrain_adjusted_stage=adjusted,initial_stage=initial)

def solve_hour(dataset, time_index, grid: GridConfig, z_faces_m, geophysical,
               timestamp: datetime, previous_convective_height,
               previous_temperature_jump, *, base_timezone=0,source_thermal_state=None,
               mapper=None,terrain_context=None,target_coordinates=None,
               selected_lapse_convective_time=None,
               terrain_radius_km=15., minimum_average_radius=10,
               half_average_angle=30.,critical_froude=1.,
               use_froude=True,use_kinematic=False,use_slope=True,use_obrien=False,
               kinematic_alpha=.1,
               average_temperature=True,average_mixing_height=True,mixing_wind_level=1,
               mixing_method=1,minimum_lapse=.001,lapse_depth_m=200.,
               minimum_mixing_height_m=50.,maximum_mixing_height_m=2500.,
               minimum_water_mixing_height_m=50.,maximum_water_mixing_height_m=2500.,
               land_flux_threshold=0.,constb=1.41,conste=.15,constn=2400.,constw=.16,
               coriolis_value=1.e-4,convective_relaxation=True,convective_relaxation_time_s=800.,
               precipitation_interpolation=2,precipitation_radius_km=50.,precipitation_cutoff=.01,
               turbidity_coefficient_1=990.,turbidity_coefficient_2=-30.,
               cloudiness_coefficient_1=-.75,cloudiness_coefficient_2=3.4,
               net_radiation_coefficient_1=5.31e-13,
               net_radiation_coefficient_2=60.,net_radiation_coefficient_3=.12,
               surface_temperature_option=-1,upper_temperature_option=-1,lapse_height_m=200.,
               temperature_mode=2,
               prognostic_observation_mode=False,prognostic_observation_rmin_km=.1,
               prognostic_observation_barriers=(),prognostic_observation_barrier_top_level=0,
               water_range=(50,55),cloud_method=4,
               cloud_condensate_threshold_g_kg=.005,
               overwater_flux_method=0,coastal_shelf_length_km=0.,cool_skin=False,
               water_flux_threshold=0.,water_temperature_mode=0,
               prognostic_ocean_category=16,
               default_water_gradient_above=-.0045,
               warm_layer=False,warm_layer_state=None,
               previous_water_convective_height=None,
               previous_water_mixing_height=None,
               diagnostic_wind=True,
               wind_only=False,
               cloud_fraction_override=None,cloud_ceiling_override=None,
               step_transform=None,
               step_override=None,
               step_provider=None,
               timestep_s=3600.,
               divergence_limit=5.e-6,divergence_iterations=50,
               smoothing_passes=(2,4,4,4,4,4,4,4,4,4),
               retain_debug_stages=False,
               retain_micrometeorology_debug=False):
    """Solve one current-config CALMET timestep and return its new state.

    ``timestamp`` is the beginning of the timestep in the CALMET base time
    zone. SOLAR is linearly interpolated to the timestep midpoint as in COMP.

    Args:
        dataset: Canonical prognostic dataset or adapter used for Step 1.
        time_index: Interval-ending prognostic record index.
        grid: Target CALMET grid.
        z_faces_m: CALMET layer faces in metres AGL.
        geophysical: Mapping containing terrain, land use, roughness, albedo,
            Bowen ratio, soil heat parameter, and anthropogenic heat flux.
        timestamp: Naive local datetime at the beginning of this output step.
        previous_convective_height: Prior ``(y, x)`` convective height in m.
        previous_temperature_jump: Prior ``(y, x)`` inversion jump in K.
        base_timezone: CALMET hours west of UTC.
        source_thermal_state: Optional RDMM5 daylight state from the previous
            source record.
        mapper: Optional reusable prognostic horizontal mapper.
        terrain_context: Optional precomputed terrain extrema and distances.
        target_coordinates: Optional ``(latitude, longitude)`` arrays in
            degrees; ``None`` derives target cell-centre coordinates.
        selected_lapse_convective_time: Optional carried timestamp used by
            diagnostic lapse-rate selection.
        terrain_radius_km: Terrain-extrema search radius in kilometres.
        minimum_average_radius: Minimum upwind averaging radius in cells.
        half_average_angle: Upwind averaging cone half-angle in degrees.
        critical_froude: Critical Froude number for terrain blocking.
        use_froude: Enable the Froude terrain-flow adjustment.
        use_kinematic: Enable kinematic terrain adjustment.
        use_slope: Enable slope-flow parameterization.
        use_obrien: Enable O'Brien top-boundary vertical-wind correction.
        kinematic_alpha: Kinematic terrain multiplier.
        average_temperature: Apply upwind averaging to surface temperature.
        average_mixing_height: Apply upwind averaging to mixing heights.
        mixing_wind_level: One-based wind layer defining the averaging cone.
        mixing_method: Land/water mixing-height method code ``-2``, ``-1``,
            ``1``, or ``2``.
        minimum_lapse: Minimum potential-temperature lapse rate in K/m.
        lapse_depth_m: Near-surface lapse-rate depth in metres.
        minimum_mixing_height_m: Minimum land mixing height in metres.
        maximum_mixing_height_m: Maximum land mixing height in metres.
        minimum_water_mixing_height_m: Minimum water mixing height in metres.
        maximum_water_mixing_height_m: Maximum water mixing height in metres.
        land_flux_threshold: Land heat-flux-per-height threshold in W/m3.
        constb: Maul-Carson mechanical-height coefficient.
        conste: Batchvarova-Gryning entrainment coefficient.
        constn: Batchvarova-Gryning neutral-height constant in metres.
        constw: Overwater mechanical-height coefficient.
        coriolis_value: Absolute Coriolis parameter in 1/s; ``999`` derives it
            from target latitude.
        convective_relaxation: Enable weak-flux convective-height relaxation.
        convective_relaxation_time_s: Relaxation e-folding time in seconds.
        precipitation_interpolation: Precipitation mapper method 1, 2, or 3.
        precipitation_radius_km: Precipitation radius or Gaussian sigma in km.
        precipitation_cutoff: Minimum retained precipitation rate in mm/h.
        turbidity_coefficient_1: First clear-sky turbidity coefficient.
        turbidity_coefficient_2: Second clear-sky turbidity coefficient.
        cloudiness_coefficient_1: First cloud attenuation coefficient.
        cloudiness_coefficient_2: Second cloud attenuation coefficient.
        net_radiation_coefficient_1: First net-radiation coefficient.
        net_radiation_coefficient_2: Second net-radiation coefficient.
        net_radiation_coefficient_3: Third net-radiation coefficient.
        surface_temperature_option: ``-1`` retains spatial surface
            temperature during diagnostic wind calculations.
        upper_temperature_option: ``-1`` diagnoses spatial lapse rate.
        lapse_height_m: Lapse-rate diagnosis depth in metres.
        temperature_mode: Temperature pathway: observed profile or selected
            prognostic surface/three-dimensional modes.
        prognostic_observation_mode: Replace Step-1 U/V with mapped
            prognostic pseudo-observations.
        prognostic_observation_rmin_km: Minimum pseudo-observation distance in
            kilometres.
        prognostic_observation_barriers: Barrier segments in local km.
        prognostic_observation_barrier_top_level: Number of lowest levels to
            which barriers apply.
        water_range: Inclusive land-use category range treated as water.
        cloud_method: Cloud source method 1 through 4; methods 1/2 supply an
            external override after prognostic Step-1 construction.
        cloud_condensate_threshold_g_kg: Minimum ``QCLOUD+QICE`` in g/kg used
            to retain prognostic cloud.
        overwater_flux_method: OCD code 0 or COARE method code 10 through 12.
        coastal_shelf_length_km: COARE nearshore roughness scale in km.
        cool_skin: Enable COARE cool-skin correction.
        water_flux_threshold: Water heat-flux-per-height threshold in W/m3.
        water_temperature_mode: Select observed or prognostic water
            temperature gradients and SST.
        prognostic_ocean_category: Source land-use category treated as ocean.
        default_water_gradient_above: Fallback water gradient in K/m.
        warm_layer: Enable COARE warm-layer state updates.
        warm_layer_state: Optional warm-layer state from the previous step.
        previous_water_convective_height: Optional prior water convective
            height in metres.
        previous_water_mixing_height: Optional prior water mixing height in m.
        diagnostic_wind: Use DIAGNO terrain effects; false finalizes an
            objective-analysis wind.
        wind_only: Skip surface micrometeorology and return placeholder state.
        cloud_fraction_override: Optional ``(y, x)`` cloud fraction.
        cloud_ceiling_override: Optional ``(y, x)`` ceiling in hundreds of
            feet, converted internally to metres.
        step_transform: Optional callable applied to Step-1 fields before
            diagnostic calculations; physical transforms may accept a mutable
            :class:`StepTransformContext`.
        step_override: Optional fully prepared :class:`Step1Fields` replacing
            dataset extraction.
        step_provider: Optional callable receiving the interval-ending local
            datetime and returning :class:`Step1Fields`.
        timestep_s: Model timestep in seconds.
        divergence_limit: Divergence-minimisation convergence limit.
        divergence_iterations: Maximum divergence-minimisation iterations.
        smoothing_passes: Per-level horizontal wind smoothing counts.
        retain_debug_stages: Retain intermediate diagnostic wind copies.
        retain_micrometeorology_debug: Retain mixing heights before spatial
            averaging.

    Returns:
        :class:`HourFields` containing ``float32`` Step-1, diagnostic wind,
        surface, and 3-D temperature fields plus state for the next interval.
        A mutable transform context or warm-layer object may be updated.

    Raises:
        ValueError: If requested pseudo-observation fields are absent, an
            override or dynamic roughness has the wrong grid shape, or a
            delegated physical calculation receives invalid controls.
    """
    geo={name:(np.asarray(geophysical[name]) if name=='dominant_landuse' else
               np.asarray(geophysical[name],dtype=MODEL_DTYPE))
         for name in ('terrain_height','dominant_landuse','roughness_length',
                      'albedo','bowen_ratio','soil_heat_flux_parameter',
                      'anthropogenic_heat_flux')}
    if step_override is not None:
        step=step_override
    elif step_provider is not None:
        step=step_provider(timestamp+timedelta(seconds=timestep_s))
    else:
        record_time=_prognostic_local_timestamp(
            dataset,time_index,base_timezone)
        step=build_step1_fields(dataset,time_index,grid,z_faces_m,
              timestamp=record_time,base_timezone=base_timezone,
              thermal_state=source_thermal_state,mapper=mapper,cloud_method=4 if cloud_method in {1,2} else cloud_method,
              cloud_condensate_threshold_g_kg=cloud_condensate_threshold_g_kg,
              precipitation_interpolation=precipitation_interpolation,
              precipitation_radius_km=precipitation_radius_km,
              precipitation_cutoff=precipitation_cutoff,
              prognostic_observation_rmin_km=(prognostic_observation_rmin_km
                                               if prognostic_observation_mode else None),
              prognostic_observation_barriers=prognostic_observation_barriers,
              prognostic_observation_barrier_top_level=prognostic_observation_barrier_top_level,
              prognostic_ocean_category=prognostic_ocean_category)
    if prognostic_observation_mode:
        if step.prognostic_observation_u is None:
            raise ValueError('IPROG=15 requires prognostic observation mapping fields')
        step=replace(step,u=step.prognostic_observation_u,v=step.prognostic_observation_v)
    if cloud_fraction_override is not None:
        cloud=np.asarray(cloud_fraction_override,dtype=MODEL_DTYPE)
        if cloud.shape!=(grid.ny,grid.nx): raise ValueError('CLOUD.DAT field does not match CALMET grid')
        step=replace(step,cloud_fraction=cloud)
    if cloud_ceiling_override is not None:
        ceiling=np.asarray(cloud_ceiling_override,dtype=MODEL_DTYPE)
        if ceiling.shape!=(grid.ny,grid.nx): raise ValueError('surface cloud ceiling does not match CALMET grid')
        # ICEIL is stored in hundreds of feet; Step1Fields uses metres.
        step=replace(step,ceiling_height=_model_array(ceiling*30.48))
    latitude,longitude=(target_latitude_longitude(grid)
                        if target_coordinates is None else target_coordinates)
    sun=solar_at_step_midpoint(timestamp,timestep_s,base_timezone,latitude,longitude)
    coriolis=coriolis_parameter(latitude,coriolis_value)
    end_time=timestamp+timedelta(seconds=timestep_s)
    # Preserve the complete interval-ending instant for sub-hourly COARE runs.
    gmt_hour=_coare_ending_gmt_hour(end_time,base_timezone)
    if step_transform is not None:
        if getattr(step_transform,'uses_physical_context',False):
            previous_mixing=(previous_convective_height
                if previous_water_mixing_height is None else previous_water_mixing_height)
            context=StepTransformContext(
                sun,np.asarray(previous_convective_height,dtype=MODEL_DTYPE),
                np.asarray(previous_mixing,dtype=MODEL_DTYPE),
                np.asarray(previous_temperature_jump,dtype=MODEL_DTYPE),coriolis,
                warm_layer_state,gmt_hour)
            step=step_transform(step,timestamp,context)
            warm_layer_state=context.warm_layer_state
        else:
            step=step_transform(step,timestamp)
    roughness=(geo['roughness_length'] if step.roughness_override is None else
               np.asarray(step.roughness_override,dtype=MODEL_DTYPE))
    if roughness.shape!=(grid.ny,grid.nx):
        raise ValueError('dynamic roughness field does not match the CALMET grid')
    lapse_override=step.diagnostic_lapse_rate
    if lapse_override is not None and step.diagnostic_lapse_station_index is not None:
        iy,ix=step.diagnostic_lapse_station_index
        if previous_convective_height[iy,ix]>=lapse_height_m:
            lapse_override=np.full_like(lapse_override,-.0098)
            selected_lapse_convective_time=step.diagnostic_lapse_timestamp
        elif step.diagnostic_lapse_data is not None:
            value=step.diagnostic_lapse_data.lapse_rate_at(
                step.diagnostic_lapse_timestamp,lapse_height_m,selected_lapse_convective_time)
            lapse_override=np.full_like(lapse_override,value)
    surface_temperature_source=(step.surface_temperature if temperature_mode==1 else
                                step.temperature[0])
    if temperature_mode==1:
        step=replace(step,temperature=np.concatenate((
            np.asarray(surface_temperature_source)[None,...],
            np.asarray(step.temperature)[1:]),axis=0))
    wind=(build_diagnostic_wind(step,geo['terrain_height'],geo['dominant_landuse'],roughness,geo['albedo'],geo['bowen_ratio'],geo['soil_heat_flux_parameter'],geo['anthropogenic_heat_flux'],sun,previous_convective_height,z_faces_m,dx_m=grid.dx_km*1000.,dy_m=grid.dy_km*1000.,terrain_radius_km=terrain_radius_km,terrain_context=terrain_context,lapse_height_m=lapse_height_m,spatial_lapse=upper_temperature_option==-1,spatial_surface_temperature=surface_temperature_option==-1,temperature_mode=temperature_mode,lapse_rate_override=lapse_override,minimum_average_radius=minimum_average_radius,half_average_angle=half_average_angle,water_range=water_range,turbidity_coefficient_1=turbidity_coefficient_1,turbidity_coefficient_2=turbidity_coefficient_2,cloudiness_coefficient_1=cloudiness_coefficient_1,cloudiness_coefficient_2=cloudiness_coefficient_2,net_radiation_coefficient_1=net_radiation_coefficient_1,net_radiation_coefficient_2=net_radiation_coefficient_2,net_radiation_coefficient_3=net_radiation_coefficient_3,critical_froude=critical_froude,use_froude=use_froude,use_kinematic=use_kinematic,use_slope=use_slope,use_obrien=use_obrien,kinematic_alpha=kinematic_alpha,timestep_s=timestep_s,divergence_limit=divergence_limit,divergence_iterations=divergence_iterations,smoothing_passes=smoothing_passes,retain_debug_stages=retain_debug_stages)
          if diagnostic_wind else
          build_objective_wind(step,geo['terrain_height'],previous_convective_height,
              z_faces_m,dx_m=grid.dx_km*1000.,dy_m=grid.dy_km*1000.,
              lapse_height_m=lapse_height_m,spatial_lapse=upper_temperature_option==-1,
              lapse_rate_override=lapse_override,use_obrien=use_obrien,
              divergence_limit=divergence_limit,divergence_iterations=divergence_iterations,
              smoothing_passes=smoothing_passes,
              retain_debug_stages=retain_debug_stages))
    final_step=replace(step,u=wind.u,v=wind.v)
    if wind_only:
        zeros=np.zeros_like(previous_convective_height,dtype=MODEL_DTYPE)
        surface=SurfaceFields(
            np.asarray(roughness,dtype=MODEL_DTYPE),
            np.asarray(step.surface_density,dtype=MODEL_DTYPE),zeros,zeros,zeros,zeros,
            np.asarray(previous_convective_height,dtype=MODEL_DTYPE),
            np.asarray(previous_convective_height,dtype=MODEL_DTYPE),
            np.asarray(previous_temperature_jump,dtype=MODEL_DTYPE),warm_layer_state)
        return HourFields(timestamp,final_step,wind,surface,
                          np.asarray(step.temperature,dtype=MODEL_DTYPE),
                          selected_lapse_convective_time,step)
    if average_temperature:
        final_temp,_=upwind_mixing_average(wind.u[0],wind.v[0],surface_temperature_source,surface_temperature_source,grid.dx_km*1000.,grid.dy_km*1000.,minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
    else:
        final_temp=np.asarray(surface_temperature_source,dtype=MODEL_DTYPE).copy()
    surface=build_surface_fields(final_step,geo['dominant_landuse'],roughness,geo['albedo'],geo['bowen_ratio'],geo['soil_heat_flux_parameter'],geo['anthropogenic_heat_flux'],sun,coriolis,previous_convective_height,previous_temperature_jump,first_level_height_m=.5*(z_faces_m[0]+z_faces_m[1]),minimum_height_m=minimum_mixing_height_m,maximum_height_m=maximum_mixing_height_m,minimum_water_height_m=minimum_water_mixing_height_m,maximum_water_height_m=maximum_water_mixing_height_m,water_range=water_range,turbidity_coefficient_1=turbidity_coefficient_1,turbidity_coefficient_2=turbidity_coefficient_2,cloudiness_coefficient_1=cloudiness_coefficient_1,cloudiness_coefficient_2=cloudiness_coefficient_2,net_radiation_coefficient_1=net_radiation_coefficient_1,net_radiation_coefficient_2=net_radiation_coefficient_2,net_radiation_coefficient_3=net_radiation_coefficient_3,mixing_method=mixing_method,minimum_lapse=minimum_lapse,lapse_depth_m=lapse_depth_m,threshold_flux_per_m=land_flux_threshold,constb=constb,conste=conste,constn=constn,constw=constw,relaxation=convective_relaxation,relaxation_time_s=convective_relaxation_time_s,timestep_s=timestep_s,surface_temperature=final_temp,overwater_flux_method=overwater_flux_method,coastal_shelf_length_km=coastal_shelf_length_km,cool_skin=cool_skin,dx_km=grid.dx_km,dy_km=grid.dy_km,water_flux_threshold=water_flux_threshold,water_temperature_mode=water_temperature_mode,default_water_gradient_above=default_water_gradient_above,warm_layer=warm_layer,warm_layer_state=warm_layer_state,longitude=longitude,gmt_hour=gmt_hour,observed_water_mixing_height=step.observed_water_mixing_height,previous_water_convective_height=previous_water_convective_height,previous_water_mixing_height=previous_water_mixing_height)
    if retain_micrometeorology_debug:
        surface=replace(surface,
            unaveraged_mixing_height=surface.mixing_height.copy(),
            unaveraged_convective_mixing_height=surface.convective_mixing_height.copy())
    if average_mixing_height:
        level=mixing_wind_level-1
        zi,zic=upwind_mixing_average(wind.u[level],wind.v[level],surface.mixing_height,surface.convective_mixing_height,grid.dx_km*1000.,grid.dy_km*1000.,minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
        surface=replace(surface,mixing_height=zi,convective_mixing_height=zic)
    else:
        zic=surface.convective_mixing_height
    temperature=(observed_temperature_3d(final_temp,step.observed_temperature_faces,zic,z_faces_m,
                    landuse=geo['dominant_landuse'],water_range=water_range,
                    mixing_height=surface.mixing_height,water_gradient_below=step.water_gradient_below,
                    water_gradient_above=step.water_gradient_above)
                 if temperature_mode==0 else
                 prognostic_temperature_3d(
                     step.temperature,final_temp,zic,z_faces_m))
    return HourFields(timestamp,final_step,wind,surface,temperature,
                      selected_lapse_convective_time,step)

def dataset_timestamp(dataset, time_index):
    """Decode a canonical CALWRF ``time_text`` record.

    Args:
        dataset: NetCDF-like dataset containing ``time_text``.
        time_index: Record index along its time axis.

    Returns:
        Naive, whole-second UTC :class:`datetime` represented by the byte,
        Unicode, or integer character array.

    Raises:
        ValueError: If the assembled timestamp text is invalid.
    """
    raw=np.asarray(dataset['time_text'][time_index])
    if raw.dtype.kind=='S': text=b''.join(raw.ravel().tolist()).decode('ascii')
    elif raw.dtype.kind=='U': text=''.join(raw.ravel().tolist())
    else: text=''.join(chr(int(value)) for value in raw.ravel())
    return parse_timestamp(text)

def _prognostic_local_timestamp(dataset,time_index,base_timezone):
    """Convert a canonical CALWRF UTC record to CALMET local time.

    Args:
        dataset: NetCDF-like dataset containing ``time_text``.
        time_index: Source record index.
        base_timezone: CALMET hours west of UTC.

    Returns:
        Naive local :class:`datetime`, calculated as UTC minus hours west.
    """
    return dataset_timestamp(dataset,time_index)-timedelta(hours=base_timezone)

def iter_hours(dataset, grid: GridConfig, z_faces_m, geophysical, *,
               time_indices=None,time_samples=None,initial_convective_height=None,
               initial_temperature_jump=None, **solver_options):
    """Stream CALMET intervals while carrying prognostic state.

    CALMET labels the first output interval ``00:00--01:00``, while RDMM5
    evaluates its instantaneous prognostic fields at the interval ending time
    (``01:00``).  Consequently ``time_indices`` denotes ending-time CALWRF
    indices.  By default index zero supplies the left boundary and output is
    produced from indices ``1..N-1``.

    Args:
        dataset: Canonical prognostic dataset or adapter.
        grid: Target CALMET grid.
        z_faces_m: CALMET layer faces in metres AGL.
        geophysical: Mapping of target-grid terrain and surface properties.
        time_indices: Optional ordered interval-ending source indices.
        time_samples: Optional iterable of ``(left_index, right_index,
            fraction, ending_datetime)`` samples for sub-record interpolation.
        initial_convective_height: Optional ``float-like (y, x)`` convective
            height in metres; ``None`` starts at zero.
        initial_temperature_jump: Optional ``float-like (y, x)`` inversion
            jump in K; ``None`` starts at zero.
        **solver_options: Keyword controls forwarded to :func:`solve_hour`,
            plus optional ``cloud_provider``, ``step_provider``, and initial
            ``warm_layer_state``.

    Yields:
        :class:`HourFields` in chronological order. Each result contains
        ``float32`` level-first wind/temperature arrays and horizontal surface
        state; convective, water, thermal, warm-layer, and lapse-selection
        state is carried into the next yielded interval.

    Raises:
        ValueError: If initial states mismatch ``(grid.ny, grid.nx)``, endpoint
            requests are nonchronological, or delegated interpolation/solver
            validation fails.
    """
    shape=(grid.ny,grid.nx)
    convective=(np.zeros(shape,dtype=MODEL_DTYPE) if initial_convective_height is None else
                np.asarray(initial_convective_height,dtype=MODEL_DTYPE))
    jump=(np.zeros(shape,dtype=MODEL_DTYPE) if initial_temperature_jump is None else
          np.asarray(initial_temperature_jump,dtype=MODEL_DTYPE))
    source_thermal_state=None
    warm_layer_state=solver_options.pop('warm_layer_state',None)
    water_convective_state=np.zeros(shape,dtype=MODEL_DTYPE)
    water_mixing_state=np.zeros(shape,dtype=MODEL_DTYPE)
    selected_lapse_convective_time=None
    if convective.shape!=shape or jump.shape!=shape: raise ValueError('initial CALMET state does not match output grid')
    step_provider=solver_options.get('step_provider')
    use_endpoints=(time_samples is not None or step_provider is None)
    base_timezone=solver_options.get('base_timezone',0)
    def timeline_timestamp(index):
        """Return one endpoint timestamp on the active local timeline.

        Args:
            index: Source record index.

        Returns:
            Naive local :class:`datetime`. Native MM4/PROG adapters already
            synthesize local timestamps; canonical CALWRF records are shifted
            from UTC by ``base_timezone``.
        """
        # Native MM4/PROG adapters synthesize an already-local timeline.
        return (dataset_timestamp(dataset,index) if step_provider is not None else
                _prognostic_local_timestamp(dataset,index,base_timezone))
    if use_endpoints:
        endpoint_cache={}; endpoint_state=None; last_endpoint=-1
        def endpoint(index):
            """Build and cache one chronological prognostic endpoint.

            Args:
                index: Source record index to materialize.

            Returns:
                :class:`Step1Fields` for ``index`` with thermal state advanced
                through every preceding uncached record.

            Raises:
                ValueError: If an uncached endpoint is requested earlier than
                    the most recently built endpoint.
            """
            nonlocal endpoint_state,last_endpoint
            if index in endpoint_cache: return endpoint_cache[index]
            if index<=last_endpoint: raise ValueError('prognostic endpoint requests must be chronological')
            for current in range(last_endpoint+1,index+1):
                record_time=timeline_timestamp(current)
                method=solver_options.get('cloud_method',4)
                value=build_step1_fields(dataset,current,grid,z_faces_m,
                    timestamp=record_time,
                    base_timezone=solver_options.get('base_timezone',0),thermal_state=endpoint_state,
                    mapper=solver_options.get('mapper'),cloud_method=4 if method in {1,2} else method,
                    cloud_condensate_threshold_g_kg=solver_options.get(
                        'cloud_condensate_threshold_g_kg',.005),
                    precipitation_interpolation=solver_options.get('precipitation_interpolation',2),
                    precipitation_radius_km=solver_options.get('precipitation_radius_km',50.),
                    precipitation_cutoff=solver_options.get('precipitation_cutoff',.01),
                    prognostic_observation_rmin_km=(solver_options.get('prognostic_observation_rmin_km',.1)
                                                   if solver_options.get('prognostic_observation_mode',False) else None),
                    prognostic_observation_barriers=solver_options.get('prognostic_observation_barriers',()),
                    prognostic_observation_barrier_top_level=solver_options.get(
                        'prognostic_observation_barrier_top_level',0),
                    prognostic_ocean_category=solver_options.get(
                        'prognostic_ocean_category',16),
                    water_temperature_mode=solver_options.get('water_temperature_mode',0))
                endpoint_cache[current]=value; endpoint_state=value.thermal_state; last_endpoint=current
            return endpoint_cache[index]
        if time_samples is not None:
            samples=list(time_samples)
            iterator=((*sample,index==len(samples)-1)
                      for index,sample in enumerate(samples))
        else:
            indices=list(range(1,dataset['time_text'].shape[0]) if time_indices is None else time_indices)
            indices=iter(indices)
            try: first_index=next(indices)
            except StopIteration: iterator=iter(())
            else:
                remaining=list(indices)
                first=(None,first_index,1.,timeline_timestamp(first_index),
                       not remaining)
                rest=((time_index-1,time_index,1.,timeline_timestamp(time_index),
                       index==len(remaining)-1)
                      for index,time_index in enumerate(remaining))
                iterator=chain((first,),rest)
    else:
        indices=range(1,dataset['time_text'].shape[0]) if time_indices is None else time_indices
        indices=list(range(1,dataset['time_text'].shape[0]) if time_indices is None else time_indices)
        iterator=((None,time_index,1.,timeline_timestamp(time_index),
                  index==len(indices)-1) for index,time_index in enumerate(indices))
    for left_index,time_index,fraction,end,final_input in iterator:
        timestep_s=float(solver_options.get('timestep_s',3600.))
        begin=end-timedelta(seconds=timestep_s)
        step_override=(endpoint(time_index)
                       if use_endpoints and left_index is None else None)
        if left_index is not None:
            method=solver_options.get('cloud_method',4)
            step_override=interpolate_step1_fields(
                endpoint(left_index),endpoint(time_index),fraction,
                cloud_method=4 if method in {1,2} else method,
                cloud_condensate_threshold_g_kg=solver_options.get(
                    'cloud_condensate_threshold_g_kg',.005))
        cloud_provider=solver_options.get('cloud_provider')
        supplied=None if cloud_provider is None else cloud_provider(end)
        if isinstance(supplied,tuple): cloud_override,ceiling_override=supplied
        else: cloud_override,ceiling_override=supplied,None
        solve_options={key:value for key,value in solver_options.items()
                       if key not in {'prognostic_timestep_s','cloud_provider'}}
        solve_options['cloud_fraction_override']=cloud_override
        solve_options['cloud_ceiling_override']=ceiling_override
        result=solve_hour(dataset,time_index,grid,z_faces_m,geophysical,begin,convective,jump,source_thermal_state=source_thermal_state,warm_layer_state=warm_layer_state,previous_water_convective_height=water_convective_state,previous_water_mixing_height=water_mixing_state,step_override=step_override,selected_lapse_convective_time=selected_lapse_convective_time,**solve_options)
        convective=result.surface.convective_mixing_height.copy(); jump=result.surface.temperature_jump.copy()
        next_water_state=getattr(result.surface,'water_convective_height_state',None)
        if next_water_state is not None:
            water_convective_state=next_water_state.copy()
        next_water_mixing=getattr(result.surface,'mixing_height',None)
        if next_water_mixing is not None:
            water_mixing_state=next_water_mixing.copy()
        source_thermal_state=result.step1.thermal_state
        warm_layer_state=getattr(result.surface,'warm_layer_state',warm_layer_state)
        selected_lapse_convective_time=getattr(result,'selected_lapse_convective_time',
                                               selected_lapse_convective_time)
        yield result
