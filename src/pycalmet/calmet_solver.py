"""Composable state and grid mapping for the Python CALMET solver."""
from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from itertools import chain
import math
import numpy as np
from scipy.spatial import cKDTree
from .config import GridConfig
from .coare import (advance_warm_layer,distance_to_coast,downward_longwave,
                    initial_warm_layer_state,surface_flux,update_warm_layer_fluxes)
from .projections import (grid_convergence,grid_crs,project_geographic,
                          unproject_geographic)
from .calmet_numerics import (cloud_cover_mm5,cloud_fraction,cloud_liquid_water_profile,coriolis_parameter,density_from_pressure_temperature,heat_flux,mixing_height_batchvarova_gryning,mixing_height_maul_carson,
    convective_height_batchvarova_gryning,convective_height_maul_carson,
    convective_velocity_scale,delta_temperature_flux,divergence_and_vertical_velocity,froude_adjustment,lapse_rate,minimize_divergence,
    neutral_water_parameters,slope_flow,smooth_wind,solar_sine,surface_similarity,water_mechanical_height,
    overwater_delta_temperature_state,prognostic_temperature_3d,saturation_vapor_pressure_lowe,terrain_extrema,topographic_vertical_velocity,upwind_mixing_average,
    barrier_blocked_pairs,terrain_adjust_surface_wind,_real4_libm,_real4_math,_real4_power)

def solar_at_step_midpoint(begin,timestep_seconds,base_timezone,latitude,longitude):
    """Interpolate SOLAR's half-hour samples exactly as COMP does."""
    samples=solar_sine(begin.timetuple().tm_yday,base_timezone,latitude,longitude)
    middle=int(begin.minute*60+begin.second)+int(timestep_seconds)//2
    distance=np.float32(abs(np.float32(middle-1800)/np.float32(3600.)))
    center=begin.hour+1
    if middle==1800.: return samples[center]
    adjacent=center+1 if middle>1800. else center-1
    if middle>1800:
        return np.asarray(samples[center]*(np.float32(1.)-distance)+
                          samples[adjacent]*distance,dtype=np.float32)
    return np.asarray(samples[adjacent]*distance+
                      samples[center]*(np.float32(1.)-distance),dtype=np.float32)

def _prognostic_record_solar(record_time,base_timezone,latitude,longitude,*,
                             solar_day=None):
    """Return RDMM5's half-hour solar sample for an ending-time M3D record.

    ``solar_sine`` index zero represents 23:30 on the preceding day.  RDMM5's
    one-based ``MHR + 1`` subscript therefore maps to zero-based
    ``record_time.hour`` regardless of the prognostic-record interval.  MHR is
    the record hour after RDMM5 converts it from GMT to CALMET base time.

    COMP computes one SOLAR table per CALMET day.  A future prognostic record
    can already be on the next date, so ``solar_day`` keeps the table tied to
    the current CALMET day just as the legacy common block does.
    """
    table_time=record_time if solar_day is None else solar_day
    samples=solar_sine(table_time.timetuple().tm_yday,base_timezone,
                       latitude,longitude)
    return samples[record_time.hour]

def _coare_ending_gmt_hour(ending,base_timezone):
    """Return COARE's integer NHRZ, with CALMET's west-positive IBTZ."""
    return int((ending.hour+base_timezone)%24)

def _output_crs(g: GridConfig):
    return grid_crs(g,units='km')

def _fortran_nearest_indices(source,target,count):
    """Return nearest points ordered like a strict Fortran scan."""
    source=np.asarray(source,dtype=np.float32)
    target=np.asarray(target,dtype=np.float32)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1:]!=(2,) or target.shape[1:]!=(2,):
        raise ValueError('nearest-point coordinates must have shape (n, 2)')
    if count<1 or source.shape[0]<count:
        raise ValueError('not enough source points for nearest-point mapping')
    tree=cKDTree(source)
    provisional=np.asarray(tree.query(target,k=count)[1],dtype=np.intp)
    if count==1:
        provisional=provisional[:,None]
    edge_delta=np.asarray(source[provisional[:,-1]]-target,dtype=np.float32)
    edge_d2=np.asarray(edge_delta[:,0]*edge_delta[:,0]+
                       edge_delta[:,1]*edge_delta[:,1],dtype=np.float32)
    edge=np.asarray(_real4_libm('sqrtf',edge_d2),dtype=np.float32)
    # cKDTree measures the float32 coordinates with float64 multiply/add,
    # whereas CMPD2 and SQRT round every operation to default REAL.  A point
    # on the REAL-distance boundary can therefore lie beyond a one-ULP tree
    # radius.  Eight float32 ULPs cover the two differences, two products,
    # their sum, and SQRT while remaining tiny relative to a grid spacing.
    radius=edge.copy()
    for _ in range(8):
        radius=np.nextafter(radius,np.float32(np.inf),dtype=np.float32)
    candidates=tree.query_ball_point(target,np.asarray(radius,dtype=float))
    lengths=np.fromiter((len(items) for items in candidates),dtype=np.intp,
                        count=target.shape[0])
    offsets=np.concatenate(([0],np.cumsum(lengths)))
    candidate=np.concatenate(candidates).astype(np.intp,copy=False)
    repeated=np.repeat(np.arange(target.shape[0]),lengths)
    delta=np.asarray(source[candidate]-target[repeated],dtype=np.float32)
    distance2=np.asarray(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1],
                         dtype=np.float32)
    distance=np.asarray(_real4_libm('sqrtf',distance2),dtype=np.float32)
    # lexsort's last key is primary: target, REAL distance, source scan index.
    order=np.lexsort((candidate,distance,repeated))
    candidate=candidate[order]
    ranks=np.arange(candidate.size)-np.repeat(offsets[:-1],lengths)
    return candidate[ranks<count].reshape(target.shape[0],count)

@dataclass(frozen=True,slots=True)
class PrognosticMapper:
    """CALMET RDHD4/R2INTERP four-neighbour horizontal mapper."""
    source_shape: tuple[int,int]
    indices: np.ndarray
    weights: np.ndarray
    target_shape: tuple[int,int]
    source_coordinates: np.ndarray
    target_coordinates: np.ndarray
    grid_origin: tuple[float,float]=(0.,0.)

    @classmethod
    def build(cls,latitude,longitude,grid: GridConfig,*,validate_domain=False):
        lat=np.asarray(latitude); lon=np.asarray(longitude)
        if lat.shape!=lon.shape: raise ValueError('latitude/longitude shape mismatch')
        sx,sy=project_geographic(lon,lat,grid,'WGS-84',units='km')
        source=np.asarray(np.column_stack((np.ravel(sx,order='F'),np.ravel(sy,order='F'))),dtype=np.float32)
        tx=np.asarray(np.float32(grid.x_origin_km)+(np.arange(grid.nx,dtype=np.float32)+np.float32(.5))*np.float32(grid.dx_km),dtype=np.float32)
        ty=np.asarray(np.float32(grid.y_origin_km)+(np.arange(grid.ny,dtype=np.float32)+np.float32(.5))*np.float32(grid.dy_km),dtype=np.float32)
        xx,yy=np.meshgrid(tx,ty); target=np.asarray(np.column_stack((xx.ravel(),yy.ravel())),dtype=np.float32)
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
        indices=_fortran_nearest_indices(source,target,4)
        delta=np.asarray(source[indices]-target[:,None,:],dtype=np.float32)
        distances=np.asarray(_real4_libm('sqrtf',np.asarray(
            delta[:,:,0]*delta[:,:,0]+delta[:,:,1]*delta[:,:,1],dtype=np.float32)),
            dtype=np.float32)
        squared=np.maximum(distances*distances,.001); inverse=1./squared; weights=inverse/inverse.sum(axis=1,keepdims=True)
        return cls(lat.shape,indices,weights,(grid.ny,grid.nx),source,target,
                   (grid.x_origin_km,grid.y_origin_km))

    def interpolate(self,field,*,ignore_zero=False):
        """Map ``(..., source_y, source_x)`` to ``(..., target_y, target_x)``."""
        data=np.asarray(field)
        if data.shape[-2:]!=self.source_shape: raise ValueError('source field shape mismatch')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        selected=np.asarray(np.take(flat,self.indices,axis=-1),dtype=np.float32)
        # R2INTERP2 retains REAL*4 coordinates, inverse distances, SUM and
        # SUM1.  Recreate the unnormalised weights and the four ordered adds;
        # a double-precision dot product can land just below an integer and
        # changes downstream INT fields such as IRHPG.
        source=np.asarray(self.source_coordinates[self.indices],dtype=np.float32)
        target=np.asarray(self.target_coordinates,dtype=np.float32)
        delta=np.asarray(source-target[:,None,:],dtype=np.float32)
        distance2=np.asarray(delta[:,:,0]*delta[:,:,0]+delta[:,:,1]*delta[:,:,1],
                             dtype=np.float32)
        weights=np.asarray(np.float32(1.)/
                           np.maximum(distance2,np.float32(.001)),dtype=np.float32)
        if ignore_zero:
            weights=np.where(selected>=np.float32(1.e-6),weights,np.float32(0.))
        total=np.zeros(selected.shape[:-1],dtype=np.float32)
        denominator=np.zeros(weights.shape[:-1],dtype=np.float32)
        for neighbour in range(4):
            total=np.asarray(total+selected[...,neighbour]*weights[:,neighbour],
                             dtype=np.float32)
            denominator=np.asarray(denominator+weights[:,neighbour],dtype=np.float32)
        mapped=np.divide(total,denominator,out=np.zeros_like(total),where=denominator>0.)
        return mapped.reshape(leading+self.target_shape)

    def interpolate_observations(self,field,rmin_km,*,barriers=(),barrier_top_level=0):
        """Map four MM5 neighbours and retain INTERP's unnormalised 1/R² sum."""
        f32=np.float32
        data=np.asarray(field,dtype=f32)
        if data.shape[-2:]!=self.source_shape: raise ValueError('prognostic observation field shape mismatch')
        if rmin_km<=0.: raise ValueError('prognostic observation RMIN must be positive')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        selected=np.asarray(np.take(flat,self.indices,axis=-1),dtype=f32)
        source=np.asarray(self.source_coordinates[self.indices],dtype=f32)
        target=np.asarray(self.target_coordinates,dtype=f32)
        delta=np.asarray(source-target[:,None,:],dtype=f32)
        distance2=np.asarray(delta[:,:,0]**2+delta[:,:,1]**2,dtype=f32)
        floor=f32(f32(rmin_km)*f32(rmin_km))
        raw=np.asarray(f32(1.)/np.maximum(distance2,floor),dtype=f32)
        barrier_array=np.asarray(barriers,dtype=f32)
        if barrier_array.size:
            if data.ndim!=3:
                raise ValueError('level-dependent prognostic barriers require a 3-D field')
            origin=np.asarray(self.grid_origin,dtype=f32)
            blocked=barrier_blocked_pairs(target-origin,
                                           source-origin,barrier_array)
            available=np.ones((leading[0],)+raw.shape,dtype=bool)
            available[:barrier_top_level] &= ~blocked[None,:,:]
            factors=np.asarray(raw[None,:,:]*available,dtype=f32)
            strength=np.zeros(factors.shape[:-1],dtype=f32)
            numerator=np.zeros_like(strength)
            for neighbour in range(4):
                strength=np.asarray(strength+factors[...,neighbour],dtype=f32)
                numerator=np.asarray(numerator+selected[...,neighbour]*factors[...,neighbour],dtype=f32)
            mapped=np.divide(numerator,strength,out=np.zeros_like(numerator),where=strength>f32(0.))
            mapped=mapped.reshape(leading+self.target_shape)
            return mapped,strength.reshape(leading+self.target_shape)
        strength=np.zeros(raw.shape[0],dtype=f32)
        numerator=np.zeros(selected.shape[:-1],dtype=f32)
        for neighbour in range(4):
            strength=np.asarray(strength+raw[:,neighbour],dtype=f32)
            numerator=np.asarray(numerator+selected[...,neighbour]*raw[:,neighbour],dtype=f32)
        mapped=np.divide(numerator,strength,out=np.zeros_like(numerator),where=strength>f32(0.)).reshape(leading+self.target_shape)
        return mapped,strength.reshape(self.target_shape)

    def nearest(self,field):
        """Map with RDMM5's closest prognostic cell (mixing-height profiles)."""
        data=np.asarray(field)
        if data.shape[-2:]!=self.source_shape: raise ValueError('source field shape mismatch')
        leading=data.shape[:-2]; flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        return np.take(flat,self.indices[:,0],axis=-1).reshape(leading+self.target_shape)

    def nearest_masked(self,field,source_mask):
        """Map to RDHD5's nearest masked source, preserving scan-order ties."""
        data=np.asarray(field)
        mask=np.asarray(source_mask,dtype=bool)
        if data.shape[-2:]!=self.source_shape or mask.shape!=self.source_shape:
            raise ValueError('masked nearest field/source mask shape mismatch')
        flat_mask=np.ravel(mask,order='F')
        if not np.any(flat_mask):
            raise ValueError('prognostic SST mode requires at least one ocean source cell')
        available=np.flatnonzero(flat_mask)
        source=np.asarray(self.source_coordinates[available],dtype=np.float32)
        target=np.asarray(self.target_coordinates,dtype=np.float32)
        # RDHD5 scans I outside J and replaces IGRABW only for a strictly
        # smaller default-REAL SQRT distance.  The F-order flat indices above
        # have that same scan order, whereas cKDTree does not define which
        # point wins a tie.  The shared selector resolves it with REAL*4
        # distance and the local indices retain the global scan order.
        local=_fortran_nearest_indices(source,target,1)[:,0]
        selected=available[local]
        leading=data.shape[:-2]
        flat=np.reshape(np.swapaxes(data,-1,-2),leading+(-1,))
        return np.take(flat,selected,axis=-1).reshape(leading+self.target_shape)

    def radius_interpolate(self,field,radius_km,*,cutoff=0.,method=2):
        """INTERPQR radius-limited precipitation objective mapping."""
        data=np.asarray(field,dtype=np.float32)
        if data.shape!=self.source_shape: raise ValueError('radius interpolation expects one source 2-D field')
        if method not in {1,2,3}: raise ValueError('INTERPQR method must be 1, 2, or 3')
        if radius_km<0. or (method in {1,2} and radius_km==0.):
            raise ValueError('INTERPQR radius must be non-negative and nonzero for methods 1 and 2')
        source_values=np.ravel(data,order='F')
        origin=np.asarray(self.grid_origin,dtype=np.float32)
        source_coordinates=np.asarray(
            (np.asarray(self.source_coordinates,dtype=np.float32)-origin)*np.float32(1000.),
            dtype=np.float32)
        target_coordinates=np.asarray(np.asarray(
            np.asarray(self.target_coordinates,dtype=np.float32)-origin,dtype=np.float32)*
            np.float32(1000.),dtype=np.float32)
        source_tree=cKDTree(source_coordinates); nearest=np.take(source_values,self.indices[:,0])
        automatic_sigma=method==3 and radius_km==0.
        if automatic_sigma:
            wet=source_values>0.; dry=source_values==0.
            if wet.any() and dry.any():
                dry_coordinates=source_coordinates[dry]
                nearest_dry=cKDTree(dry_coordinates).query(source_coordinates[wet],k=1)[1]
                delta=np.asarray(source_coordinates[wet]-dry_coordinates[nearest_dry],
                                 dtype=np.float32)
                distance2=np.asarray(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1],
                                     dtype=np.float32)
                sigma2=np.min(distance2)
            else:
                sigma2=np.float32(1.e37)
            # INTERPQR assigns SIGMA1=.5*SQRT(SIGMA2), but never writes the
            # halved value back to SIGMA2.  The exponential branch below
            # therefore uses the original minimum-distance squared.
            radius_m=np.asarray(_real4_libm('sqrtf',sigma2),dtype=np.float32)
        else:
            radius_m=np.float32(radius_km)*np.float32(1000.)
            sigma2=np.asarray(radius_m*radius_m,dtype=np.float32)
        if method in {1,2}:
            limit2=sigma2
        else:
            cutoff_ratio=np.nextafter(np.float32(60.),np.float32(np.inf),dtype=np.float32)
            maximum=np.float32(np.finfo(np.float32).max/float(cutoff_ratio))
            limit2=(np.float32(np.inf) if sigma2>maximum else
                    np.asarray(cutoff_ratio*sigma2,dtype=np.float32))
        # KDTree measures in double precision, while CMPD2 and the inclusion
        # tests use REAL*4.  Search through the next representable squared
        # bound, then apply the exact float32 condition below.
        candidate_limit2=np.nextafter(limit2,np.float32(np.inf),dtype=np.float32)
        search_radius=np.sqrt(float(candidate_limit2))
        result=np.empty(self.target_coordinates.shape[0],dtype=np.float32)
        weighted=np.zeros(result.size,dtype=bool)
        # A whole-domain sparse distance matrix can contain tens of millions
        # of pairs on the 610x499 Zhejiang grid.  Chunk targets while retaining
        # INTERPQR's exact all-points-inside-radius weighted average.
        chunk_size=8192 if np.isfinite(search_radius) else 128
        for begin in range(0,result.size,chunk_size):
            end=min(begin+chunk_size,result.size); targets=target_coordinates[begin:end]
            neighbours=(source_tree.query_ball_point(targets,search_radius) if np.isfinite(search_radius)
                        else [np.arange(source_values.size,dtype=np.intp)]*len(targets))
            neighbours=[np.sort(item) for item in neighbours]
            lengths=np.fromiter((len(item) for item in neighbours),dtype=np.intp,count=end-begin)
            source_indices=np.concatenate(neighbours) if lengths.sum() else np.empty(0,dtype=np.intp)
            local_targets=np.repeat(np.arange(end-begin),lengths)
            difference=np.asarray(source_coordinates[source_indices]-targets[local_targets],
                                  dtype=np.float32)
            distance2=np.asarray(difference[:,0]*difference[:,0]+difference[:,1]*difference[:,1],
                                 dtype=np.float32)
            if method==1:
                eligible=distance2<=sigma2
                safe_distance=np.maximum(_real4_libm('sqrtf',distance2),np.float32(.01))
                weights=np.where(eligible,np.asarray(
                    np.float32(1.)/safe_distance,dtype=np.float32),np.float32(0.))
                weighted_values=np.where(eligible,np.asarray(
                    source_values[source_indices]/safe_distance,dtype=np.float32),np.float32(0.))
            elif method==2:
                eligible=distance2<=sigma2
                safe_distance=np.maximum(distance2,np.float32(.01))
                weights=np.where(eligible,np.asarray(
                    np.float32(1.)/safe_distance,dtype=np.float32),np.float32(0.))
                weighted_values=np.where(eligible,np.asarray(
                    source_values[source_indices]/safe_distance,dtype=np.float32),np.float32(0.))
            else:
                exponent=np.asarray(distance2/sigma2,dtype=np.float32)
                eligible=exponent<=np.float32(60.)
                decay=np.where(eligible,_real4_libm('expf',np.asarray(-exponent,dtype=np.float32)),
                               np.float32(0.))
                safe_distance=np.maximum(distance2,np.float32(.01))
                weights=np.asarray(decay/safe_distance,dtype=np.float32)
                weighted_values=np.asarray(
                    source_values[source_indices]*decay/safe_distance,dtype=np.float32)
            width=int(lengths.max(initial=0))
            numerator_terms=np.zeros((end-begin,width),dtype=np.float32)
            denominator_terms=np.zeros_like(numerator_terms)
            if source_indices.size:
                offsets=np.repeat(np.cumsum(lengths)-lengths,lengths)
                columns=np.arange(source_indices.size)-offsets
                denominator_terms[local_targets,columns]=weights
                numerator_terms[local_targets,columns]=weighted_values
            # CUMSUM along the source-point axis preserves INTERPQR's ordered
            # REAL additions while evaluating all target cells together.
            numerator=(np.cumsum(numerator_terms,axis=1,dtype=np.float32)[:,-1]
                       if width else np.zeros(end-begin,dtype=np.float32))
            denominator=(np.cumsum(denominator_terms,axis=1,dtype=np.float32)[:,-1]
                         if width else np.zeros(end-begin,dtype=np.float32))
            if method==3:
                # INTERPQR does not use the nearest-cell fallback in the
                # exponential branch.  If every EXP(-R²/SIGMA²) term was
                # cut off (or the weighted precipitation TOP is tiny), the
                # literal test TOP < 1E-30 returns zero.
                result[begin:end]=np.divide(numerator,denominator,
                    out=np.zeros(end-begin,dtype=np.float32),
                    where=(numerator>=np.float32(1.e-30))&(denominator>np.float32(0.)))
            else:
                weighted[begin:end]=denominator>np.float32(0.)
                result[begin:end]=np.divide(numerator,denominator,
                    out=nearest[begin:end].copy(),where=denominator>0.)
        apply_cutoff=np.ones(result.size,dtype=bool) if method==3 else weighted
        result=np.where(apply_cutoff&(result<np.float32(cutoff)),np.float32(0.),result)
        return result.reshape(self.target_shape)

@dataclass(frozen=True,slots=True)
class SourceThermalState:
    """RDMM5 potential temperatures retained from the last daylight hour."""
    pt20: np.ndarray
    pt30: np.ndarray

@dataclass(frozen=True,slots=True)
class Step1Fields:
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
    """Runtime state needed by DIAGNO station physics before interpolation."""
    solar_elevation_sine: np.ndarray
    previous_convective_height: np.ndarray
    previous_mixing_height: np.ndarray
    previous_temperature_jump: np.ndarray
    coriolis: np.ndarray
    warm_layer_state: object | None = None
    gmt_hour: int | None = None


def _domain_mean_real4(field):
    """CGAMMA2 domain mean in its I-outer/J-inner REAL accumulation order."""
    values=np.asarray(field,dtype=np.float32)
    total=np.float32(0.)
    for value in values.ravel(order='F'):
        total=np.float32(total+value)
    return np.float32(total/np.float32(values.size))

def _interpolate_ceiling(left,right,fraction,*,literal_endpoint=False):
    """RDMM5's special nonzero-aware ICEILG time interpolation."""
    f=np.float32(fraction)
    left=np.asarray(left,dtype=np.float32); right=np.asarray(right,dtype=np.float32)
    if f==np.float32(1.) and not literal_endpoint:
        interpolated=right.copy()
    else:
        interpolated=np.asarray(
            left+np.asarray(f*np.asarray(right-left,dtype=np.float32),dtype=np.float32),
            dtype=np.float32)
    return np.where((left!=np.float32(0.))&(right!=np.float32(0.)),
                    interpolated,np.maximum(left,right)).astype(np.float32)

def interpolate_step1_fields(left: Step1Fields,right: Step1Fields,fraction,*,cloud_method=4,
                             literal_endpoint=False):
    """RDMM5 sub-hour interpolation between two actual prognostic records."""
    f=np.float32(fraction)
    if not 0.<=f<=1.: raise ValueError('prognostic interpolation fraction must be between zero and one')
    def lerp(a,b):
        a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
        # At ordinary source-record boundaries the production GNU -O2 build
        # folds OLD+1*(NEW-OLD) to NEW.  The final input record follows a
        # distinct RDMM5 path and retains the literal arithmetic.
        if f==np.float32(1.) and not literal_endpoint:
            return b.copy()
        delta=np.asarray(b-a,dtype=np.float32)
        return np.asarray(a+np.asarray(f*delta,dtype=np.float32),dtype=np.float32)
    if cloud_method==4:
        cloud=lerp(left.cloud_fraction,right.cloud_fraction)
    elif cloud_method==3:
        if left.cloud_relative_humidity_850 is None or right.cloud_relative_humidity_850 is None:
            raise ValueError('MCLOUD=3 interpolation requires endpoint 850 mb relative humidity')
        cloud=cloud_fraction(lerp(left.cloud_relative_humidity_850,right.cloud_relative_humidity_850))
    else:
        raise ValueError('cloud_method must be MCLOUD 3 or 4')
    ceiling=_interpolate_ceiling(left.ceiling_height,right.ceiling_height,f,
                                 literal_endpoint=literal_endpoint)
    cloud_water=None
    if left.integrated_cloud_water is not None and right.integrated_cloud_water is not None:
        cloud_water=lerp(left.integrated_cloud_water,right.integrated_cloud_water)
        cloudy=cloud_water>=1.e-6; cloud=np.where(cloudy,cloud,0.); ceiling=np.where(cloudy,ceiling,0.)
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
        # RDMM5 deliberately does not time-interpolate the sounding used by
        # MIXDT2.  TZ0/ZL0 are copied from the current (left) M3D record at
        # the start of the step, while the ordinary prognostic fields are
        # interpolated through the ending time.
        np.asarray(left.profile_temperature,dtype=np.float32),
        np.asarray(left.profile_height,dtype=np.float32),
        np.trunc(lerp(left.surface_relative_humidity,right.surface_relative_humidity)).astype(np.int32),
        precipitation,precipitation_code,right.thermal_state,rh850,cloud_water,
        prognostic_observation_u=(None if left.prognostic_observation_u is None else
                                  lerp(left.prognostic_observation_u,right.prognostic_observation_u)),
        prognostic_observation_v=(None if left.prognostic_observation_v is None else
                                  lerp(left.prognostic_observation_v,right.prognostic_observation_v)),
        prognostic_observation_strength=(None if right.prognostic_observation_strength is None else
                                          np.asarray(right.prognostic_observation_strength,
                                                     dtype=np.float32)),
        prognostic_temperature=prognostic_temperature,
        cloud_liquid_water_3d=auxiliary_cloud[0],
        cloud_liquid_water_above=auxiliary_cloud[1],
        cloud_liquid_water_above_bottom=auxiliary_cloud[2],
        cloud_liquid_water_above_top=auxiliary_cloud[3])


def observed_temperature_3d(surface_temperature,face_temperature,convective_height,z_faces_m,
                            *,landuse=None,water_range=(55,55),mixing_height=None,
                            water_gradient_below=None,water_gradient_above=None):
    """Vectorized land branch of TEMP3D for ITPROG=0."""
    f32=np.float32
    surface=np.asarray(surface_temperature,dtype=f32)
    faces=np.asarray(face_temperature,dtype=f32)
    z=np.asarray(z_faces_m,dtype=f32)
    if faces.shape!=(z.size,)+surface.shape:
        raise ValueError('observed face temperature dimensions do not match the CALMET grid')
    result=np.empty((z.size-1,)+surface.shape,dtype=f32); result[0]=surface
    zi=np.asarray(convective_height,dtype=f32)
    for k in range(1,z.size-1):
        low,high=z[k],z[k+1]; top=faces[k+1]
        above=np.asarray(f32(.5)*np.asarray(faces[k]+top,dtype=f32),dtype=f32)
        below=np.asarray(surface-f32(.0098)*f32(.5)*f32(low+high),dtype=f32)
        middle=np.asarray(surface-f32(.0098)*zi,dtype=f32)
        lower_sum=np.asarray(surface-f32(.0098)*low+middle,dtype=f32)
        upper_sum=np.asarray(middle+top,dtype=f32)
        crossing=np.asarray(f32(.5)*np.asarray(
            lower_sum*np.asarray(zi-low,dtype=f32)+
            upper_sum*np.asarray(high-zi,dtype=f32),dtype=f32)/f32(high-low),dtype=f32)
        result[k]=np.where(low>zi,above,np.where(high<=zi,below,crossing))
    if landuse is not None and water_gradient_below is not None and water_gradient_above is not None:
        water=(np.asarray(landuse)>=water_range[0])&(np.asarray(landuse)<=water_range[1])
        water_zi=np.asarray(mixing_height,dtype=f32)
        below_gradient=np.asarray(water_gradient_below,dtype=f32)
        above_gradient=np.asarray(water_gradient_above,dtype=f32)
        for k in range(1,z.size-1):
            low,high=z[k],z[k+1]; center=f32(f32(.5)*f32(low+high))
            wholly_below=np.asarray(surface+below_gradient*center,dtype=f32)
            wholly_above=np.asarray(surface+below_gradient*water_zi+
                                     above_gradient*np.asarray(center-water_zi,dtype=f32),dtype=f32)
            low_temp=np.asarray(surface+below_gradient*low,dtype=f32)
            middle_temp=np.asarray(surface+below_gradient*water_zi,dtype=f32)
            high_temp=np.asarray(middle_temp+above_gradient*np.asarray(high-water_zi,dtype=f32),dtype=f32)
            crossing=np.asarray(f32(.5)*np.asarray(
                np.asarray(low_temp+middle_temp,dtype=f32)*np.asarray(water_zi-low,dtype=f32)+
                np.asarray(middle_temp+high_temp,dtype=f32)*np.asarray(high-water_zi,dtype=f32),
                dtype=f32)/f32(high-low),dtype=f32)
            water_value=np.where(high<=water_zi,wholly_below,
                                 np.where(low>water_zi,wholly_above,crossing))
            result[k]=np.where(water,water_value,result[k])
    return result

def map_prognostic_cloud(source_cloud,source_rh850,mapper,cloud_method):
    """Apply RDMM5's method-specific horizontal interpolation order."""
    if cloud_method==4: return mapper.interpolate(source_cloud),None
    if cloud_method==3:
        mapped_rh850=mapper.interpolate(source_rh850)
        return cloud_fraction(mapped_rh850),mapped_rh850
    raise ValueError('cloud_method must be MCLOUD 3 or 4')

@dataclass(frozen=True,slots=True)
class SurfaceFields:
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
    timestamp: datetime
    step1: Step1Fields
    wind: DiagnosticWindFields
    surface: SurfaceFields
    temperature: np.ndarray
    selected_lapse_convective_time: datetime | None = None
    source_step: Step1Fields | None = None

def target_latitude_longitude(grid: GridConfig):
    """Latitude/longitude at CALMET cell centres (MICROI grid geometry)."""
    # MICROI stores DGRID in metres, then forms kilometres through a second
    # REAL*4 multiplication by the inexact literal 0.001.  Computing directly
    # with DX_KM loses that rounding (e.g. 29.5 becomes 29.500002 km).
    f32=np.float32
    ix=np.arange(grid.nx,dtype=f32)+f32(.5)
    iy=np.arange(grid.ny,dtype=f32)+f32(.5)
    dx_m=f32(f32(grid.dx_km)*f32(1000.))
    dy_m=f32(f32(grid.dy_km)*f32(1000.))
    x=np.asarray(f32(grid.x_origin_km)+np.asarray(ix*dx_m,dtype=f32)*f32(.001),dtype=f32)
    y=np.asarray(f32(grid.y_origin_km)+np.asarray(iy*dy_m,dtype=f32)*f32(.001),dtype=f32)
    xx,yy=np.meshgrid(x,y)
    longitude,latitude=unproject_geographic(xx,yy,grid,'WGS-84',units='km')
    return np.asarray(latitude),np.asarray(longitude)

def _vertical_centers(field,height,centers,*,wind=False):
    """RDMM5 interpolation/persistence, including low-level wind log law."""
    f32=np.float32
    data=np.asarray(field,dtype=f32); z=np.asarray(height,dtype=f32)
    if data.shape!=z.shape or data.ndim!=3: raise ValueError('vertical fields must have shape (level,y,x)')
    levels,y,x=data.shape; flat=data.reshape(levels,-1); zflat=z.reshape(levels,-1); columns=np.arange(flat.shape[1]); out=[]
    for center in np.asarray(centers,dtype=f32):
        above=np.sum(zflat<=center,axis=0); lower=np.clip(above-1,0,levels-1); upper=np.clip(above,0,levels-1)
        zl,zu=zflat[lower,columns],zflat[upper,columns]; vl,vu=flat[lower,columns],flat[upper,columns]
        ratio=np.divide(np.asarray(center-zl,dtype=f32),np.asarray(zu-zl,dtype=f32),
                        out=np.zeros_like(zl,dtype=f32),where=zu!=zl)
        value=np.asarray(vl+np.asarray((vu-vl)*ratio,dtype=f32),dtype=f32)
        below=center<zflat[0]; value=np.where(below,flat[0],value); value=np.where(center>=zflat[-1],flat[-1],value)
        if wind:
            # RDMM5 assigns each ALOG result to default REAL before forming
            # the logarithmic-profile ratio.  NumPy's float32 log ufunc may
            # differ from GNU logf by one ulp for the time-varying first WRF
            # half-sigma height, which appears only from later records in the
            # Zhejiang sequence.
            log_half=f32(_real4_libm('logf',f32(.5)))
            numerator=f32(_real4_libm('logf',center)-log_half)
            denominator=np.asarray(_real4_libm(
                'logf',np.maximum(zflat[0],f32(.500001)))-log_half,dtype=f32)
            logratio=np.divide(numerator,denominator,out=np.ones_like(zl,dtype=f32),
                               where=zflat[0]>f32(.500001))
            value=np.where(below,np.asarray(logratio*flat[0],dtype=f32),value)
        out.append(value.reshape(y,x))
    return np.stack(out).astype(f32)

def _prognostic_wind_components(speed,direction,convergence):
    """RDMM5 true-north direction rotation and REAL*4 component split."""
    f32=np.float32
    velocity=np.asarray(speed,dtype=f32)
    grid_direction=np.asarray(np.asarray(direction,dtype=f32)-
                              np.asarray(convergence,dtype=f32),dtype=f32)
    # RDMM5 wraps map-relative direction before applying its rounded
    # DEG2RAD constant. Although a negative angle is mathematically
    # equivalent, 360*0.0174533 is not exactly 2*pi in REAL*4.
    grid_direction=np.where(grid_direction<f32(0.),grid_direction+f32(360.),grid_direction)
    grid_direction=np.where(grid_direction>f32(360.),grid_direction-f32(360.),
                            grid_direction).astype(f32)
    radians=np.asarray(f32(.0174533)*grid_direction,dtype=f32)
    return (np.asarray(-velocity*_real4_libm('sinf',radians),dtype=f32),
            np.asarray(-velocity*_real4_libm('cosf',radians),dtype=f32))

def _relative_humidity(qvapor,pressure_hpa,temperature_k):
    f32=np.float32
    temperature=np.asarray(temperature_k,dtype=f32)
    pressure=np.asarray(pressure_hpa,dtype=f32)
    vapor=np.asarray(qvapor,dtype=f32)
    tc=np.asarray(temperature-f32(273.15),dtype=f32)
    # CALWRF GETRH1 deliberately uses qsat=0.622*es/p (rather than the
    # more usual epsilon*es/(p-es)); preserve that legacy definition.
    exponent=np.asarray(f32(17.67)*tc/np.asarray(temperature-f32(29.65),dtype=f32),dtype=f32)
    es=np.asarray(f32(6.112)*np.exp(exponent),dtype=f32)
    saturation=np.asarray(f32(.622)*es/pressure,dtype=f32)
    return np.clip(np.asarray(f32(100.)*vapor/saturation,dtype=f32),f32(0.),f32(100.))

def _fortran_nint_positive(values):
    """Fortran NINT for the non-negative CALWRF fields used here."""
    data=np.asarray(values,dtype=np.float32)
    return np.floor(np.asarray(data+np.float32(.5),dtype=np.float32)).astype(np.float32)


def _quantize_real4(values,scale):
    """Round a non-negative REAL*4 field after applying a decimal scale."""
    f32=np.float32; factor=f32(scale)
    data=np.asarray(values,dtype=f32)
    return np.asarray(_fortran_nint_positive(np.asarray(data*factor,dtype=f32))/factor,dtype=f32)


def _m3d_cloud_water_g_kg(values):
    """Recover the F6.3 M3D cloud-water value from canonical kg/kg."""
    data=np.asarray(values,dtype=np.float32)
    thousandths=_fortran_nint_positive(
        np.asarray(data*np.float32(1.e6),dtype=np.float32))
    return np.asarray(thousandths/np.float32(1000.),dtype=np.float32)


def _m3d_relative_humidity(dataset,time_index,qvapor,pressure,temperature):
    """Return RDMM5's stored integer IRH field, with a schema fallback."""
    if 'relative_humidity' in dataset.variables:
        values=np.asarray(dataset['relative_humidity'][time_index])
    else:
        values=_relative_humidity(qvapor,pressure,temperature)
    return np.clip(_fortran_nint_positive(values),1.,100.)


def _rdmm5_surface_density(pressure_mb,temperature_k,height_agl_m):
    """RDMM5 RHOP from the first prognostic level at/above ground."""
    pressure=np.asarray(pressure_mb,dtype=np.float32)
    temperature=np.asarray(temperature_k,dtype=np.float32)
    height=np.asarray(height_agl_m,dtype=np.float32)
    if pressure.shape!=temperature.shape or pressure.shape!=height.shape or pressure.ndim!=3:
        raise ValueError('RDMM5 density profiles must share shape (level,y,x)')
    above=height>=np.float32(0.)
    if not np.all(above.any(axis=0)):
        raise ValueError('RDMM5 profile has no level at or above ground')
    level=np.argmax(above,axis=0); yy,xx=np.indices(level.shape)
    return density_from_pressure_temperature(pressure[level,yy,xx],
                                             temperature[level,yy,xx])


def _rdmm5_above_ground_profiles(height_agl_m,*fields):
    """Pack RDMM5 levels from per-column INDPX through the model top."""
    height=np.asarray(height_agl_m,dtype=np.float32)
    values=(height,*(np.asarray(field,dtype=np.float32) for field in fields))
    if height.ndim!=3 or any(value.shape!=height.shape for value in values):
        raise ValueError('RDMM5 vertical profiles must share shape (level,y,x)')
    levels=height.shape[0]; above=height>=np.float32(0.); count=above.sum(axis=0)
    if not np.all(count):
        raise ValueError('RDMM5 profile has no level at or above ground')
    level_index=np.arange(levels,dtype=np.int32)[:,None,None]
    packed_index=np.sort(np.where(above,level_index,levels),axis=0)
    rank=np.minimum(level_index,count[None]-1)
    packed_index=np.take_along_axis(packed_index,rank,axis=0)
    return tuple(np.take_along_axis(value,packed_index,axis=0) for value in values)


def _rdmm5_sst_profile(centers_m,lowest_height_m,lowest_temperature_k,sst_k,ocean):
    """RDMM5 ITWPROG=2 actual-temperature interpolation below INDPX."""
    centers=np.asarray(centers_m,dtype=np.float32)
    height=np.asarray(lowest_height_m,dtype=np.float32)
    temperature=np.asarray(lowest_temperature_k,dtype=np.float32)
    sst=np.asarray(sst_k,dtype=np.float32)
    ratio=np.divide(centers[:,None,None],height[None,:,:],
                    out=np.zeros((centers.size,*height.shape),dtype=np.float32),
                    where=height[None,:,:]!=np.float32(0.))
    profile=np.asarray(sst[None,:,:]*np.asarray(np.float32(1.)-ratio,dtype=np.float32)+
                       temperature[None,:,:]*ratio,dtype=np.float32)
    use=(centers[:,None,None]<height[None,:,:])&np.asarray(ocean,dtype=bool)[None,:,:]
    return profile,use

def _stull_temperature(theta2,z2,theta3,z3,theta20,theta30,z1):
    """STULL nighttime potential-temperature extrapolation."""
    f32=np.float32
    t2=np.asarray(theta2,dtype=f32); t3=np.asarray(theta3,dtype=f32)
    t20=np.asarray(theta20,dtype=f32); t30=np.asarray(theta30,dtype=f32)
    height2=np.asarray(z2,dtype=f32); height3=np.asarray(z3,dtype=f32)
    dt2=np.asarray(t2-t20,dtype=f32); dt3=np.asarray(t3-t30,dtype=f32)
    well=(t3>t2)&(dt3<f32(0.))&(dt2<f32(0.))&(dt2<dt3)
    ratio=np.divide(dt2,dt3,out=np.ones_like(dt2,dtype=f32),where=dt3!=f32(0.))
    logarithm=_real4_libm('logf',np.maximum(ratio,f32(1.e-30)))
    fallback=np.asarray(height3/f32(5.),dtype=f32)
    calculated=np.divide(np.asarray(f32(.77)*(height3-height2),dtype=f32),logarithm,
                         out=fallback.copy(),where=well&(ratio>f32(0.))&
                         (np.abs(logarithm)>f32(1.e-30)))
    scale=np.maximum(np.where(well,calculated,fallback),
                     np.asarray(height2/f32(10.),dtype=f32))
    surface_drop=np.maximum(np.minimum(f32(0.),np.asarray(
        dt2*_real4_libm('expf',np.asarray(f32(.77)*height2/scale,dtype=f32)),
        dtype=f32)),f32(-15.))
    drop=np.asarray(surface_drop*_real4_libm('expf',np.asarray(
        -f32(.77)*np.asarray(z1,dtype=f32)/scale,dtype=f32)),dtype=f32)
    drop=np.where(dt2>f32(0.),dt2,drop).astype(f32)
    return np.asarray(t20+drop,dtype=f32)

def build_step1_fields(dataset,time_index,grid: GridConfig,z_faces_m,*,timestamp=None,
                       solar_day=None,
                       base_timezone=0,thermal_state: SourceThermalState | None=None,
                       mapper: PrognosticMapper | None=None,prognostic_ocean_category=16,
                       water_temperature_mode=0,
                       cloud_method=4,
                       precipitation_interpolation=2,precipitation_radius_km=50.,
                       precipitation_cutoff=.01,prognostic_observation_rmin_km=None,
                       prognostic_observation_barriers=(),prognostic_observation_barrier_top_level=0):
    """Read one canonical CALWRF time and reproduce RDMM5 Step-1 fields."""
    lat=np.asarray(dataset['latitude'][:]); lon=np.asarray(dataset['longitude'][:])
    if mapper is None:
        mapper=PrognosticMapper.build(lat,lon,grid,validate_domain=True)
    # CALWRF stores geopotential height MSL.  RDMM5 subtracts the source-cell
    # terrain before all vertical interpolation and CLOUD4 calculations.
    terrain=_fortran_nint_positive(np.asarray(dataset['terrain'][:]))
    height=_fortran_nint_positive(np.asarray(dataset['height'][time_index]))-terrain[None,:,:]
    speed=_quantize_real4(dataset['wind_speed'][time_index],10.)
    direction=np.asarray(
        _fortran_nint_positive(np.asarray(dataset['wind_direction'][time_index])),
        dtype=np.float32)
    source_sun=None
    if timestamp is not None:
        source_sun=_prognostic_record_solar(
            timestamp,base_timezone,lat,lon,solar_day=solar_day)
    convergence=np.asarray(grid_convergence(grid,lon,lat),dtype=np.float32)
    source_u,source_v=_prognostic_wind_components(speed,direction,convergence[None,:,:])
    centers=np.asarray(np.float32(.5)*(np.asarray(z_faces_m[:-1],dtype=np.float32)+
                                       np.asarray(z_faces_m[1:],dtype=np.float32)),dtype=np.float32)
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
    source_temperature=_quantize_real4(dataset['temperature'][time_index],10.)
    source_q=_quantize_real4(dataset['qvapor'][time_index],1.e5)
    source_pressure=_fortran_nint_positive(np.asarray(dataset['pressure'][time_index]))
    theta=np.asarray(source_temperature*_real4_power(
        np.asarray(np.float32(1000.)/source_pressure,dtype=np.float32),np.float32(.286)),
        dtype=np.float32)
    if np.any(np.sum(height>=np.float32(0.),axis=0)<2):
        raise ValueError('RDMM5 temperature profile requires two levels at or above ground')
    (thermo_height,source_temperature_above,source_q_above,
     source_pressure_above,theta_above)=_rdmm5_above_ground_profiles(
        height,source_temperature,source_q,source_pressure,theta)
    theta2,theta3=theta_above[0],theta_above[1]
    z2,z3=thermo_height[0],thermo_height[1]
    if thermal_state is None:
        scale=np.asarray(z3/np.float32(5.),dtype=np.float32)
        upper_exp=_real4_libm('expf',np.asarray(-np.float32(.77)*z3/scale,dtype=np.float32))
        lower_exp=_real4_libm('expf',np.asarray(-np.float32(.77)*z2/scale,dtype=np.float32))
        denominator=np.asarray(upper_exp-lower_exp,dtype=np.float32)
        surface_drop=np.divide(np.asarray(theta3-theta2,dtype=np.float32),denominator,
                               out=np.zeros_like(theta2,dtype=np.float32),where=denominator!=np.float32(0.))
        surface_drop=np.maximum(np.minimum(surface_drop,np.float32(0.)),np.float32(-15.))
        theta30=np.asarray(theta3-np.asarray(surface_drop*upper_exp,dtype=np.float32),
                           dtype=np.float32)
        theta20=theta30.copy()
    else:
        theta20=np.asarray(thermal_state.pt20,dtype=np.float32)
        theta30=np.asarray(thermal_state.pt30,dtype=np.float32)
    if source_sun is not None:
        daylight=source_sun>0.; theta20=np.where(daylight,theta2,theta20); theta30=np.where(daylight,theta3,theta30)
    next_thermal_state=SourceThermalState(theta20.copy(),theta30.copy())
    source_temperature_centers=_vertical_centers(
        source_temperature_above,thermo_height,centers)
    source_sst=None
    if water_temperature_mode==2:
        if 'sst' not in dataset.variables:
            raise ValueError('ITWPROG=2 requires SST in prognostic data')
        source_sst=_quantize_real4(dataset['sst'][time_index],10.)
        if np.any((source_sst<np.float32(150.))|(source_sst>np.float32(350.))):
            raise ValueError('ITWPROG=2 requires prognostic SST between 150 and 350 K')
    if source_sun is not None:
        below=centers[:,None,None]<z2[None,:,:]
        daytime_below=below&(source_sun[None,:,:]>0.)
        extrapolated=np.asarray(source_temperature_above[0][None,:,:]+np.float32(.0098)*
            np.asarray(z2[None,:,:]-centers[:,None,None],dtype=np.float32),dtype=np.float32)
        source_temperature_centers=np.where(daytime_below,extrapolated,source_temperature_centers)
        night_below=below&~(source_sun[None,:,:]>0.)
        ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        fraction=np.asarray((centers[:,None,None]-z2[None,:,:])/
                            np.asarray(z3-z2,dtype=np.float32)[None,:,:],dtype=np.float32)
        water_theta=np.asarray(theta2[None,:,:]+fraction*np.asarray(
            theta3-theta2,dtype=np.float32)[None,:,:],dtype=np.float32)
        land_theta=np.stack([_stull_temperature(theta2,z2,theta3,z3,theta20,theta30,center) for center in centers])
        below_theta=np.where(ocean[None,:,:],water_theta,land_theta)
        beta=np.asarray(_real4_libm('logf',np.asarray(
            source_pressure[0]/source_pressure[1],dtype=np.float32))/
            np.asarray(height[1]-height[0],dtype=np.float32),dtype=np.float32)
        center_pressure=np.asarray(source_pressure[0][None,:,:]*_real4_libm('expf',np.asarray(
            -beta[None,:,:]*np.asarray(centers[:,None,None]-height[0][None,:,:],dtype=np.float32),
            dtype=np.float32)),dtype=np.float32)
        night_temperature=np.asarray(below_theta*_real4_power(np.asarray(
            center_pressure/np.float32(1000.),dtype=np.float32),np.float32(.286)),dtype=np.float32)
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
    # RDMM5 reads the integer IRH field written in every M3D level record; it
    # does not reconstruct RH from the separately rounded Q/T/P fields.  That
    # reconstruction can differ by one percent (and much more in very dry
    # upper layers), which changes CLOUD4 at its 99/100-percent boundary.
    source_rh=_m3d_relative_humidity(
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
    if 'qcloud' in dataset.variables:
        # M3D v2.1 writes cloud water as F6.3 g/kg.  RDMM5 initially
        # derives CLOUD4 from RH, but subsequently forces cloud/ceiling to
        # zero wherever the horizontally mapped integrated cloud water is
        # below 1e-6 g/kg.
        # M3D stores F6.3 g/kg, while the canonical dataset uses kg/kg.
        # Recover the integer thousandths once, then divide directly by
        # 1000.  Quantizing in kg/kg and multiplying back introduces a second
        # REAL rounding and changes PRFVAR output by one ULP.
        source_qcloud=_m3d_cloud_water_g_kg(dataset['qcloud'][time_index])
        source_aux=cloud_liquid_water_profile(source_qcloud,height,z_faces_m)
        auxiliary_cloud=(mapper.nearest(source_aux[0]),mapper.nearest(source_aux[1]),
                         mapper.nearest(source_aux[2]),mapper.nearest(source_aux[3]))
        cloud_water=mapper.interpolate(np.sum(source_qcloud,axis=0))
        cloudy=cloud_water>=1.e-6
        present=source_qcloud>1.e-9
        first=np.argmax(present,axis=0)
        source_qc_ceiling=np.take_along_axis(height,first[None,:,:],axis=0)[0]
        source_qc_ceiling=np.where(present.any(axis=0),source_qc_ceiling,0.)
        mapped_ceiling=mapper.interpolate(source_qc_ceiling,ignore_zero=True)
        # ICEILG is integer hundreds of feet; expose metres while retaining
        # the old INT(ceil/30.28) quantisation used by PGTSTB.
        mapped_ceiling=np.trunc(mapped_ceiling/30.28)*30.48
    else:
        cloudy=np.ones(mapper.target_shape,dtype=bool)
        cloud_water=None
        auxiliary_cloud=(None,None,None,None)
        mapped_ceiling=mapper.interpolate(source_ceiling,ignore_zero=True)
    mapped_source_cloud,rh850=map_prognostic_cloud(
        source_cloud,source_rh850 if cloud_method==3 else None,mapper,cloud_method)
    mapped_cloud=np.where(cloudy,mapped_source_cloud,0.)
    mapped_ceiling=np.where(cloudy,mapped_ceiling,0.)
    surface_profile=np.asarray(source_temperature_above[0]+np.float32(.0098)*z2,
                               dtype=np.float32)
    if source_sun is not None:
        water_surface=theta2-z2*(theta3-theta2)/(z3-z2)
        land_surface=_stull_temperature(theta2,z2,theta3,z3,theta20,theta30,0.)
        surface_theta=np.where(
            np.asarray(dataset['landuse'][:])==prognostic_ocean_category,
            water_surface,land_surface)
        beta=np.asarray(_real4_libm('logf',np.asarray(
            source_pressure[0]/source_pressure[1],dtype=np.float32))/
            np.asarray(height[1]-height[0],dtype=np.float32),dtype=np.float32)
        surface_p=np.asarray(source_pressure[0]*_real4_libm(
            'expf',np.asarray(beta*height[0],dtype=np.float32)),dtype=np.float32)
        night_surface=np.asarray(surface_theta*_real4_power(np.asarray(
            surface_p/np.float32(1000.),dtype=np.float32),np.float32(.286)),dtype=np.float32)
        surface_profile=np.where(source_sun>0.,surface_profile,night_surface)
    if source_sst is not None:
        ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        surface_profile=np.where(ocean,source_sst,surface_profile).astype(np.float32)
    profile_temperature=np.concatenate((mapper.nearest(surface_profile)[None,:,:],
                                        mapper.nearest(source_temperature_above)),axis=0)
    profile_height=np.concatenate((np.zeros((1,grid.ny,grid.nx),dtype=np.float32),
                                   mapper.nearest(thermo_height)),axis=0)
    surface_temperature=mapper.interpolate(_quantize_real4(dataset['temperature_2m'][time_index],10.))
    raw_surface_pressure=np.asarray(dataset['surface_pressure'][time_index],dtype=np.float32)
    surface_pressure=mapper.interpolate(np.asarray(_fortran_nint_positive(
        np.asarray(raw_surface_pressure*np.float32(.1),dtype=np.float32))/np.float32(10.),
        dtype=np.float32))
    surface_density=mapper.interpolate(_rdmm5_surface_density(
        source_pressure,source_temperature,height))
    # Despite its comment, RDMM5's R2INTERP2 call receives RH(:,:,1): the
    # integer RH of the first 3-D prognostic layer, not Q2/T2 surface RH.
    mapped_surface_rh=mapper.interpolate(source_rh[0])
    # R2INTERP2 accumulates REAL*4 values.  A double-precision normalized dot
    # product of four identical integer RH values can instead land one ULP
    # below that integer (for example 90.99999999999999), after which INT
    # would spuriously lose one percent.
    nearest_integer=np.rint(mapped_surface_rh)
    mapped_surface_rh=np.where(
        np.abs(mapped_surface_rh-nearest_integer)<1.e-6,
        nearest_integer,mapped_surface_rh)
    surface_rh=np.trunc(mapped_surface_rh).astype(np.int32)
    precipitation=mapper.radius_interpolate(np.asarray(dataset['precipitation'][time_index]),
        precipitation_radius_km,cutoff=precipitation_cutoff,method=precipitation_interpolation)
    overwater_data=None
    if source_sst is not None:
        from .sea_data import GriddedSeaData
        source_ocean=np.asarray(dataset['landuse'][:])==prognostic_ocean_category
        nearest_ocean=lambda field: mapper.nearest_masked(field,source_ocean)
        ocean_height=nearest_ocean(z2)
        ocean_longitude=nearest_ocean(lon)
        shape=mapper.target_shape
        overwater_data=GriddedSeaData(
            nearest_ocean(source_temperature_above[0]),nearest_ocean(source_sst),
            nearest_ocean(source_rh_above[0]),ocean_height,ocean_height,
            np.full(shape,np.float32(.05),dtype=np.float32),
            np.full(shape,np.float32(-999.),dtype=np.float32),
            np.full(shape,np.float32(-999.),dtype=np.float32),ocean_longitude)
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
    """Create the wind-only Step-1 scaffold used by legacy ``PROG.DAT``."""
    faces=np.asarray(z_faces_m,dtype=np.float32)
    centers=.5*(faces[:-1]+faces[1:])
    u,v=(prog_data.interpolate(ending_hour,grid,centers) if record is None else
         prog_data.interpolate_record(record,grid,centers))
    shape=(grid.ny,grid.nx)
    level_shape=(centers.size,*shape)
    surface_temperature=np.full(shape,np.float32(288.7),dtype=np.float32)
    profile_height=np.broadcast_to(np.r_[np.float32(0.),centers][:,None,None],
                                   (centers.size+1,*shape)).copy()
    profile_temperature=np.full_like(profile_height,np.float32(288.7))
    return Step1Fields(
        u,v,np.full(level_shape,np.float32(288.7),dtype=np.float32),surface_temperature,
        np.full(shape,np.float32(1000.),dtype=np.float32),
        np.full(shape,np.float32(.3484321)*np.float32(1000.)/np.float32(288.7),dtype=np.float32),
        np.full(level_shape,np.float32(80.),dtype=np.float32),
        np.full(shape,np.float32(50.),dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.full(shape,np.float32(.1),dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.zeros(shape,dtype=np.float32),profile_temperature,profile_height,
        np.full(shape,80,dtype=np.int32),np.zeros(shape,dtype=np.float32),
        np.zeros(shape,dtype=np.int32),prognostic_temperature=np.full(
            level_shape,np.float32(288.7),dtype=np.float32))


def build_mm4_step_fields(mm4_data,record,grid: GridConfig,z_faces_m,*,cloud_method=4,
                          precipitation_interpolation=2,precipitation_radius_km=50.,
                          precipitation_cutoff=.01,base_timezone=0,thermal_state=None,
                          prognostic_observation_rmin_km=None,
                          prognostic_observation_barriers=(),
                          prognostic_observation_barrier_top_level=0,
                          prognostic_ocean_category=16):
    """Create the complete Step-1 scaffold produced by native ``RDMM4``."""
    faces=np.asarray(z_faces_m,dtype=np.float32)
    centers=np.float32(.5)*(faces[:-1]+faces[1:])
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
        fields.relative_humidity,np.zeros(shape,dtype=np.float32),
        np.zeros(shape,dtype=np.float32),np.full(shape,np.float32(.1),dtype=np.float32),
        fields.cloud_fraction,fields.ceiling_height_m,fields.profile_temperature_k,
        fields.profile_height_m,fields.surface_relative_humidity,
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
    """Compose COMP's current NOOBS land and no-SEA.DAT water pathways."""
    landuse=np.asarray(landuse); water=(landuse>=water_range[0])&(landuse<=water_range[1]); land=~water
    temp=np.asarray(step.temperature[0] if surface_temperature is None else surface_temperature)
    density=np.asarray(step.surface_density)
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
    initial_water_mechanical=np.asarray(
        np.float32(constw)*water_ustar/np.asarray(coriolis,dtype=np.float32),
        dtype=np.float32)
    # WATERP (ITWPROG=2) always computes offshore mixing height from the
    # prognostic profile.  SEA.DAT may still be open for wave observations,
    # but its ZIOW records are not consulted on this path.
    if water_temperature_mode==2:
        observed_water_mixing_height=None
    if observed_water_mixing_height is None:
        observed_water=np.zeros_like(water,dtype=bool)
        observed_zi=np.zeros_like(water_zi,dtype=np.float32)
    else:
        observed_zi=np.asarray(observed_water_mixing_height,dtype=np.float32)
        if observed_zi.shape!=water.shape:
            raise ValueError('observed water mixing-height shape mismatch')
        observed_water=water&np.isfinite(observed_zi)
        # WATER assigns observed ZIOW before COARE maps ZI to its REAL*8 RZI
        # input.  Restore it again below because only unobserved cells execute
        # the subsequent mechanical/convective height update in the source.
        water_zi=np.where(observed_water,observed_zi,water_zi).astype(np.float32)
    coare_water_zi=np.where(observed_water,observed_zi,
                            initial_water_mechanical).astype(np.float32)
    # MIXHT/MIXHT2 has already updated land DPTT before WATER is called.
    # Water cells retain their previous values until WATER's scalar
    # sequence-association path updates DPTT(1,1).  Initialising the complete
    # array to zero discarded every new land-cell jump when no SEA data were
    # configured, which first became visible in the following hour.
    water_jump=np.where(
        water,np.asarray(previous_temperature_jump,dtype=np.float32),
        np.asarray(jump,dtype=np.float32)).astype(np.float32)
    water_theta=temp
    water_state=(np.zeros_like(water_zi,dtype=np.float32)
                 if previous_water_convective_height is None else
                 np.asarray(previous_water_convective_height,dtype=np.float32).copy())
    if water_state.shape!=water.shape:
        raise ValueError('previous water convective-height state shape mismatch')
    if step.overwater_data is not None:
        sea=step.overwater_data
        f32=np.float32
        humidity=np.asarray(np.where(sea.relative_humidity<f32(9998.99),
                                     sea.relative_humidity,f32(100.)),dtype=f32)
        saturation=saturation_vapor_pressure_lowe(
            np.asarray(sea.air_temperature_k,dtype=f32)-f32(273.15))
        vapor_pressure=np.asarray(f32(.01)*humidity*saturation,dtype=f32)
        mixing_ratio=np.asarray(f32(.622)*vapor_pressure/
                                (f32(1000.)-vapor_pressure),dtype=f32)
        specific_humidity=np.asarray(mixing_ratio/(f32(1.)+mixing_ratio)*f32(1000.),
                                     dtype=f32)
        water_theta=np.asarray(np.asarray(sea.air_temperature_k,dtype=f32)*
            (f32(1.)+f32(.61)*mixing_ratio)+
            f32(.01)*np.asarray(sea.anemometer_height_m,dtype=f32),dtype=f32)
        if overwater_flux_method==0:
            water_theta,difference,difference_10m,theta_10m=(
                overwater_delta_temperature_state(
                    sea.air_temperature_k,sea.sea_temperature_k,humidity,
                    sea.anemometer_height_m))
            water_length,water_ustar,water_qh=delta_temperature_flux(
                np.hypot(step.u[0],step.v[0]),sea.anemometer_height_m,difference,
                difference_10m,theta_10m,water_theta,water_z0)
            water_length=np.asarray(water_length,dtype=np.float32)
            water_ustar=np.asarray(water_ustar,dtype=np.float32)
            water_qh=np.asarray(water_qh,dtype=np.float32)
            water_wt=np.asarray(water_qh/np.float32(1195.2),dtype=np.float32)
        else:
            longwave=downward_longwave(temp,step.surface_relative_humidity,
                                       step.cloud_fraction,step.ceiling_height)
            coast=distance_to_coast(landuse,water_range,dx_km,dy_km)
            # WATER/WATERP clamp SST only on entry to COARE.  DELTAT above
            # deliberately receives the original, potentially colder value.
            sea_temperature=np.maximum(np.asarray(sea.sea_temperature_k,dtype=np.float32),
                                       np.float32(270.15))
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
            water_z0=np.asarray(flux.roughness_m,dtype=np.float32)
            water_ustar=np.asarray(flux.friction_velocity_m_s,dtype=np.float32)
            water_length=np.asarray(flux.monin_obukhov_m,dtype=np.float32)
            # WATER converts COARE's REAL*8 EL back to default REAL before
            # applying its near-zero guard.  EL=0 follows the positive arm.
            water_length=np.where(np.abs(water_length)<np.float32(5.),
                np.where(water_length<np.float32(0.),np.float32(-5.),np.float32(5.)),
                water_length).astype(np.float32)
            water_wt=np.asarray(flux.buoyancy_flux_k_m_s,dtype=np.float32)
            water_qh=np.asarray(water_wt*np.float32(1195.2),dtype=np.float32)
            if warm_layer:
                next_warm_state=update_warm_layer_fluxes(next_warm_state,flux)
        mechanical=water_mechanical_height(water_ustar,coriolis,constw=constw,
            minimum_height_m=minimum_water_height_m,
            maximum_height_m=maximum_water_height_m)
        if mixing_method>0:
            prior_water_zi=(np.full_like(water_state,np.float32(minimum_water_height_m))
                            if previous_water_mixing_height is None else
                            np.asarray(previous_water_mixing_height,dtype=np.float32))
            if prior_water_zi.shape!=water.shape:
                raise ValueError('previous water mixing-height state shape mismatch')
            # WATER first bounds last hour's ZI by ZIMINW, then constrains its
            # saved ZICOLD not to exceed that total height.
            prior_water_zi=np.maximum(prior_water_zi,np.float32(minimum_water_height_m))
            old_water=np.maximum(np.minimum(water_state,prior_water_zi),
                                 np.float32(minimum_water_height_m))
            if water_temperature_mode==0:
                gradient=(default_water_gradient_above
                          if step.water_mixing_gradient_above is None
                          else step.water_mixing_gradient_above)
                potential_lapse=np.asarray(gradient)+.0098
            else:
                _,_,potential_lapse=profile_temperature_layer(
                    step.profile_temperature,step.profile_height,old_water,lapse_depth_m,
                    minimum_lapse=minimum_lapse)
            if abs(mixing_method)==1:
                # WATER passes bare ``DPTT`` (rather than DPTT(I,J)) to
                # MIXHMC.  Fortran sequence association therefore makes all
                # water calls share DPTT(1,1).  Preserve its I-outer/J-inner
                # order because each call feeds the next one.
                water_zic=np.zeros_like(old_water,dtype=np.float32)
                shared_jump=np.float32(
                    np.asarray(previous_temperature_jump,dtype=np.float32)[0,0])
                active_water=water&~observed_water
                cell_lapse=np.broadcast_to(np.asarray(potential_lapse,dtype=np.float32),
                                           water.shape)
                for ix in range(water.shape[1]):
                    for iy in range(water.shape[0]):
                        if not active_water[iy,ix]:
                            continue
                        cell_zic,cell_jump=convective_height_maul_carson(
                            old_water[iy,ix],shared_jump,water_wt[iy,ix],
                            cell_lapse[iy,ix],timestep_s=timestep_s,
                            threshold_flux_per_m=water_flux_threshold,
                            maximum_height_m=maximum_water_height_m,
                            minimum_lapse=minimum_lapse,conste=conste,
                            relaxation=relaxation,
                            relaxation_time_s=relaxation_time_s)
                        water_zic[iy,ix]=cell_zic
                        shared_jump=np.float32(cell_jump)
                water_jump=np.where(
                    water,np.asarray(previous_temperature_jump,dtype=np.float32),
                    np.asarray(jump,dtype=np.float32)).astype(np.float32)
                water_jump[0,0]=shared_jump
            else:
                water_zic=convective_height_batchvarova_gryning(
                    old_water,water_wt,potential_lapse,water_ustar,
                    water_length,sea.air_temperature_k,timestep_s=timestep_s,
                    minimum_height_m=minimum_water_height_m,
                    maximum_height_m=maximum_water_height_m,
                    threshold_flux_per_m=water_flux_threshold,minimum_lapse=minimum_lapse,
                    relaxation=relaxation,relaxation_time_s=relaxation_time_s)
                water_jump=np.asarray(previous_temperature_jump,dtype=np.float32)
            water_zi=np.clip(np.maximum(mechanical,np.asarray(water_zic,dtype=np.float32)),
                np.float32(minimum_water_height_m),np.float32(maximum_water_height_m)).astype(np.float32)
            water_state=np.where(water&~observed_water,water_zic,water_state).astype(np.float32)
        else:
            water_zi=mechanical; water_zic=np.zeros_like(mechanical)
            water_jump=np.zeros_like(mechanical)
            water_state=np.where(water&~observed_water,np.float32(0.),water_state).astype(
                np.float32)
    water_zi=np.where(observed_water,observed_zi,water_zi).astype(np.float32)
    water_wstar=convective_velocity_scale(
        water_zi if mixing_method<0 else water_zic,water_qh,water_theta,density,
        landuse,water_range=water_range)
    # WATER's scalar ZICONV is local to that subroutine and is used for its
    # pre-AVEMIX WSTAR only.  The COMP/MIXHT ZICONV grid skips water cells,
    # retaining their previous value until optional AVEMIX changes the grid.
    grid_zic=np.where(water,np.asarray(previous_convective_height,dtype=np.float32),zic)
    return SurfaceFields(
        np.where(water,water_z0,roughness),density,qsw,np.where(water,water_qh,qh),
        np.where(water,water_ustar,ustar),np.where(water,water_length,length),
        np.where(water,water_zi,zi),grid_zic,
        (water_jump if abs(mixing_method)==1 and mixing_method>0 else
         np.where(water,water_jump,jump)),
        next_warm_state if step.overwater_data is not None and
        abs(overwater_flux_method)>=10 and warm_layer else warm_layer_state,
        np.where(water,water_wstar,0.).astype(np.float32),
        water_state,
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
    """Current Zhejiang DIAGNO path (IKINE=0, IFRADJ=1, ISLOPE=1)."""
    terrain=np.asarray(terrain_m,dtype=np.float32)
    faces=np.asarray(z_faces_m,dtype=np.float32)
    centers=np.float32(.5)*(faces[:-1]+faces[1:])
    gamma=(lapse_rate(step.temperature,previous_convective_height,centers,lapse_height_m,spatial=spatial_lapse)
           if lapse_rate_override is None else np.asarray(lapse_rate_override,dtype=np.float32))
    if terrain_context is None:
        terrain_context=terrain_extrema(terrain,terrain_radius_km,dx_m,dy_m)
    highest,lowest,distance_high,distance_low=terrain_context
    u=np.asarray(step.u); v=np.asarray(step.v)
    if step.diagnostic_surface_temperature is not None:
        diagnostic_temperature=np.asarray(step.diagnostic_surface_temperature)
    elif spatial_surface_temperature:
        diagnostic_temperature=np.asarray(step.temperature[0])
    else:
        prognostic=(step.temperature if step.prognostic_temperature is None else
                    step.prognostic_temperature)
        diagnostic_temperature=np.full_like(
            step.temperature[0],_domain_mean_real4(np.asarray(prognostic)[0]))
    stage_w=np.zeros((faces.size,*terrain.shape),dtype=np.float32)
    if use_kinematic:
        boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
        topo_w=topographic_vertical_velocity(u,v,terrain,z_faces_m,gamma,step.temperature[0],
                                             kinematic_alpha,dx_m,dy_m)
        stage_w=topo_w
        u,v,_,_=minimize_divergence(u,v,topo_w,z_faces_m,dx_m,dy_m,boundaries=boundaries,
            maximum_iterations=divergence_iterations,limit=divergence_limit)
    kinematic_stage=((u.copy(),v.copy(),stage_w.copy()) if retain_debug_stages else None)
    if use_froude:
        preliminary_temp,_=upwind_mixing_average(u[0],v[0],diagnostic_temperature,diagnostic_temperature,dx_m,
            minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
        u,v=froude_adjustment(u,v,terrain,highest,gamma,preliminary_temp,centers,critical_froude,dx_m,dy_m)
    froude_stage=((u.copy(),v.copy(),stage_w.copy()) if retain_debug_stages else None)
    preliminary_qh=np.zeros_like(terrain)
    if use_slope:
        preliminary_temp,_=upwind_mixing_average(u[0],v[0],diagnostic_temperature,diagnostic_temperature,dx_m,
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
        density=(np.asarray(step.surface_density,dtype=np.float32)
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
    """IWFCOD=0 wind finalisation after INTERP objective analysis."""
    gamma=(lapse_rate(step.temperature,previous_convective_height,
                      np.float32(.5)*(np.asarray(z_faces_m[:-1],dtype=np.float32)+
                                      np.asarray(z_faces_m[1:],dtype=np.float32)),
                      lapse_height_m,spatial=spatial_lapse)
           if lapse_rate_override is None else np.asarray(lapse_rate_override,dtype=np.float32))
    interpolated=((np.asarray(step.u).copy(),np.asarray(step.v).copy())
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
        np.zeros_like(terrain_m,dtype=np.float32),interpolated_stage=interpolated,
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
    """
    geo={name:np.asarray(geophysical[name]) for name in ('terrain_height','dominant_landuse','roughness_length','albedo','bowen_ratio','soil_heat_flux_parameter','anthropogenic_heat_flux')}
    if step_override is not None:
        step=step_override
    elif step_provider is not None:
        step=step_provider(timestamp+timedelta(seconds=timestep_s))
    else:
        record_time=_prognostic_local_timestamp(
            dataset,time_index,base_timezone)
        step=build_step1_fields(dataset,time_index,grid,z_faces_m,
              timestamp=record_time,solar_day=timestamp,base_timezone=base_timezone,
              thermal_state=source_thermal_state,mapper=mapper,cloud_method=4 if cloud_method in {1,2} else cloud_method,
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
        cloud=np.asarray(cloud_fraction_override,dtype=np.float32)
        if cloud.shape!=(grid.ny,grid.nx): raise ValueError('CLOUD.DAT field does not match CALMET grid')
        step=replace(step,cloud_fraction=cloud)
    if cloud_ceiling_override is not None:
        ceiling=np.asarray(cloud_ceiling_override,dtype=np.float32)
        if ceiling.shape!=(grid.ny,grid.nx): raise ValueError('surface cloud ceiling does not match CALMET grid')
        # ICEIL is stored in hundreds of feet; Step1Fields uses metres.
        step=replace(step,ceiling_height=ceiling*np.float32(30.48))
    latitude,longitude=(target_latitude_longitude(grid)
                        if target_coordinates is None else target_coordinates)
    sun=solar_at_step_midpoint(timestamp,timestep_s,base_timezone,latitude,longitude)
    coriolis=coriolis_parameter(latitude,coriolis_value)
    end_time=timestamp+timedelta(seconds=timestep_s)
    # COMP passes only integer NHRZE to COARE; NSECE is deliberately absent.
    gmt_hour=_coare_ending_gmt_hour(end_time,base_timezone)
    if step_transform is not None:
        if getattr(step_transform,'uses_physical_context',False):
            previous_mixing=(previous_convective_height
                if previous_water_mixing_height is None else previous_water_mixing_height)
            context=StepTransformContext(
                sun,np.asarray(previous_convective_height,dtype=np.float32),
                np.asarray(previous_mixing,dtype=np.float32),
                np.asarray(previous_temperature_jump,dtype=np.float32),coriolis,
                warm_layer_state,gmt_hour)
            step=step_transform(step,timestamp,context)
            warm_layer_state=context.warm_layer_state
        else:
            step=step_transform(step,timestamp)
    roughness=(geo['roughness_length'] if step.roughness_override is None else
               np.asarray(step.roughness_override,dtype=np.float32))
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
        zeros=np.zeros_like(previous_convective_height,dtype=np.float32)
        surface=SurfaceFields(
            np.asarray(roughness,dtype=np.float32),
            np.asarray(step.surface_density,dtype=np.float32),zeros,zeros,zeros,zeros,
            np.asarray(previous_convective_height,dtype=np.float32),
            np.asarray(previous_convective_height,dtype=np.float32),
            np.asarray(previous_temperature_jump,dtype=np.float32),warm_layer_state)
        return HourFields(timestamp,final_step,wind,surface,
                          np.asarray(step.temperature,dtype=np.float32),
                          selected_lapse_convective_time,step)
    if average_temperature:
        final_temp,_=upwind_mixing_average(wind.u[0],wind.v[0],surface_temperature_source,surface_temperature_source,grid.dx_km*1000.,minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
    else:
        final_temp=np.asarray(surface_temperature_source).copy()
    surface=build_surface_fields(final_step,geo['dominant_landuse'],roughness,geo['albedo'],geo['bowen_ratio'],geo['soil_heat_flux_parameter'],geo['anthropogenic_heat_flux'],sun,coriolis,previous_convective_height,previous_temperature_jump,first_level_height_m=.5*(z_faces_m[0]+z_faces_m[1]),minimum_height_m=minimum_mixing_height_m,maximum_height_m=maximum_mixing_height_m,minimum_water_height_m=minimum_water_mixing_height_m,maximum_water_height_m=maximum_water_mixing_height_m,water_range=water_range,turbidity_coefficient_1=turbidity_coefficient_1,turbidity_coefficient_2=turbidity_coefficient_2,cloudiness_coefficient_1=cloudiness_coefficient_1,cloudiness_coefficient_2=cloudiness_coefficient_2,net_radiation_coefficient_1=net_radiation_coefficient_1,net_radiation_coefficient_2=net_radiation_coefficient_2,net_radiation_coefficient_3=net_radiation_coefficient_3,mixing_method=mixing_method,minimum_lapse=minimum_lapse,lapse_depth_m=lapse_depth_m,threshold_flux_per_m=land_flux_threshold,constb=constb,conste=conste,constn=constn,constw=constw,relaxation=convective_relaxation,relaxation_time_s=convective_relaxation_time_s,timestep_s=timestep_s,surface_temperature=final_temp,overwater_flux_method=overwater_flux_method,coastal_shelf_length_km=coastal_shelf_length_km,cool_skin=cool_skin,dx_km=grid.dx_km,dy_km=grid.dy_km,water_flux_threshold=water_flux_threshold,water_temperature_mode=water_temperature_mode,default_water_gradient_above=default_water_gradient_above,warm_layer=warm_layer,warm_layer_state=warm_layer_state,longitude=longitude,gmt_hour=gmt_hour,observed_water_mixing_height=step.observed_water_mixing_height,previous_water_convective_height=previous_water_convective_height,previous_water_mixing_height=previous_water_mixing_height)
    if retain_micrometeorology_debug:
        surface=replace(surface,
            unaveraged_mixing_height=surface.mixing_height.copy(),
            unaveraged_convective_mixing_height=surface.convective_mixing_height.copy())
    if average_mixing_height:
        level=mixing_wind_level-1
        zi,zic=upwind_mixing_average(wind.u[level],wind.v[level],surface.mixing_height,surface.convective_mixing_height,grid.dx_km*1000.,minimum_radius_cells=minimum_average_radius,half_angle_degrees=half_average_angle,timestep_s=timestep_s)
        surface=replace(surface,mixing_height=zi,convective_mixing_height=zic)
    else:
        zic=surface.convective_mixing_height
    temperature=(observed_temperature_3d(final_temp,step.observed_temperature_faces,zic,z_faces_m,
                    landuse=geo['dominant_landuse'],water_range=water_range,
                    mixing_height=surface.mixing_height,water_gradient_below=step.water_gradient_below,
                    water_gradient_above=step.water_gradient_above)
                 if temperature_mode==0 else
                 prognostic_temperature_3d(step.temperature,final_temp,zic,.5*(np.asarray(z_faces_m[:-1])+np.asarray(z_faces_m[1:]))))
    return HourFields(timestamp,final_step,wind,surface,temperature,
                      selected_lapse_convective_time,step)

def dataset_timestamp(dataset, time_index):
    """Decode the canonical CALWRF ``time_text`` value."""
    raw=np.asarray(dataset['time_text'][time_index])
    if raw.dtype.kind=='S': text=b''.join(raw.ravel().tolist()).decode('ascii')
    elif raw.dtype.kind=='U': text=''.join(raw.ravel().tolist())
    else: text=''.join(chr(int(value)) for value in raw.ravel())
    return datetime.strptime(text.rstrip('\x00 '),'%Y-%m-%d_%H:%M:%S')

def _prognostic_local_timestamp(dataset,time_index,base_timezone):
    """Convert a canonical CALWRF/M3D GMT record time to CALMET base time."""
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
    """
    shape=(grid.ny,grid.nx)
    convective=(np.zeros(shape,dtype=np.float32) if initial_convective_height is None else
                np.asarray(initial_convective_height,dtype=np.float32))
    jump=(np.zeros(shape,dtype=np.float32) if initial_temperature_jump is None else
          np.asarray(initial_temperature_jump,dtype=np.float32))
    source_thermal_state=None
    warm_layer_state=solver_options.pop('warm_layer_state',None)
    water_convective_state=np.zeros(shape,dtype=np.float32)
    water_mixing_state=np.zeros(shape,dtype=np.float32)
    selected_lapse_convective_time=None
    if convective.shape!=shape or jump.shape!=shape: raise ValueError('initial CALMET state does not match output grid')
    step_provider=solver_options.get('step_provider')
    use_endpoints=(time_samples is not None or step_provider is None)
    base_timezone=solver_options.get('base_timezone',0)
    def timeline_timestamp(index):
        # Native MM4/PROG adapters synthesize an already-local timeline.
        return (dataset_timestamp(dataset,index) if step_provider is not None else
                _prognostic_local_timestamp(dataset,index,base_timezone))
    if use_endpoints:
        endpoint_cache={}; endpoint_state=None; last_endpoint=-1
        def endpoint(index,solar_day):
            nonlocal endpoint_state,last_endpoint
            if index in endpoint_cache: return endpoint_cache[index]
            if index<=last_endpoint: raise ValueError('prognostic endpoint requests must be chronological')
            for current in range(last_endpoint+1,index+1):
                record_time=timeline_timestamp(current)
                method=solver_options.get('cloud_method',4)
                value=build_step1_fields(dataset,current,grid,z_faces_m,
                    timestamp=record_time,solar_day=solar_day,
                    base_timezone=solver_options.get('base_timezone',0),thermal_state=endpoint_state,
                    mapper=solver_options.get('mapper'),cloud_method=4 if method in {1,2} else method,
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
                # RDMM5 initialises its first in-memory endpoint directly.
                # From the next output onward even an exact boundary executes
                # OLD + 1*(NEW-OLD), with REAL*4 rounding at every operation.
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
    first_output=True
    for left_index,time_index,fraction,end,final_input in iterator:
        timestep_s=float(solver_options.get('timestep_s',3600.))
        begin=end-timedelta(seconds=timestep_s)
        step_override=(endpoint(time_index,begin)
                       if use_endpoints and left_index is None else None)
        if left_index is not None:
            method=solver_options.get('cloud_method',4)
            step_override=interpolate_step1_fields(
                endpoint(left_index,begin),endpoint(time_index,begin),fraction,
                cloud_method=4 if method in {1,2} else method,
                literal_endpoint=final_input)
        # At an exact M3D boundary, RDMM5 immediately reads the following
        # record and resets DTINC to zero.  The sole exception is the final
        # step after its initial call; initialization always reads ahead even
        # for a one-step run.  Ordinary fields are unchanged whether written
        # as OLD+1*(NEW-OLD) or NEW+0*(FUTURE-NEW), but ICEILG takes MAX when
        # either endpoint is clear.  Override just that special field.  Native
        # MM4/PROG providers already implement their own record progression.
        eager_ceiling=(step_provider is None and fraction==1. and
                       (first_output or not final_input))
        if eager_ceiling:
            if time_index+1>=dataset['time_text'].shape[0]:
                raise ValueError(
                    'eager CALMET M3D boundary requires a following prognostic record')
            current=endpoint(time_index,begin)
            following=endpoint(time_index+1,begin)
            ceiling=_interpolate_ceiling(
                current.ceiling_height,following.ceiling_height,0.)
            if step_override.integrated_cloud_water is not None:
                ceiling=np.where(
                    step_override.integrated_cloud_water>=np.float32(1.e-6),
                    ceiling,np.float32(0.)).astype(np.float32)
            step_override=replace(step_override,ceiling_height=ceiling)
        first_output=False
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
