import numpy as np
import netCDF4 as nc
import pytest
import shutil
import struct
import subprocess
import textwrap
from types import SimpleNamespace
import pycalmet.calmet_solver as calmet_solver
from dataclasses import replace
from datetime import datetime,timedelta
from pycalmet.calmet_numerics import (air_density,cloud_cover_mm5,cloud_cover_teixeira_850,cloud_fraction,cloud_liquid_water_profile,coare_water_station_profiles,convective_height_batchvarova_gryning,convective_height_maul_carson,convective_velocity_scale,coriolis_parameter,delta_temperature_flux,delta_temperature_water_station_profiles,froude_adjustment,hosker_roughness,inverse_distance_squared,lake_breeze_adjust,lapse_rate,mesopac_layer_average,mixing_height_batchvarova_gryning,mixing_height_maul_carson,neutral_water_parameters,objective_analysis_component,overwater_delta_temperature_state,pgt_stability,profile_temperature_layer,prognostic_temperature_3d,saturation_vapor_pressure_lowe,upwind_mixing_average,
    divergence_and_vertical_velocity,heat_flux,land_station_mixing_height,land_station_similarity_theory,land_station_surface_similarity,minimize_divergence,neutral_water_station_profiles,psi_momentum_dyer,similarity_wind_profiles,slope_flow,smooth_wind,solar_sine,surface_similarity,terrain_extrema,terrain_gradients,
    smooth_wind,terrain_adjust_surface_wind,topographic_vertical_velocity,vertical_interpolate,vertical_layer_average,water_mechanical_height,wind1_initial_guess,wind_speed_direction)
from pycalmet.calmet_output import (AuxiliaryCalmetWriter,CalmetNetCDFWriter,
                                    LegacyCalmetWriter,MesopacWriter)
from pycalmet.legacy._compat.calmet_output import (
    MesopacWriter as LegacyMesopacWriter,
)
from pycalmet.aux_data import AuxiliaryData
from pycalmet.legacy_io import FortranSequentialReader,FortranSequentialWriter,fixed_string
from pycalmet.calmet_solver import DiagnosticWindFields,HourFields,PrognosticMapper,Step1Fields,StepTransformContext,SurfaceFields,_coare_ending_gmt_hour,_nearest_indices,_source_relative_humidity,_prognostic_local_timestamp,_prognostic_record_solar,_prognostic_wind_components,_rdmm5_above_ground_profiles,_rdmm5_sst_profile,_rdmm5_surface_density,_relative_humidity,_stull_temperature,_vertical_centers,build_diagnostic_wind,build_objective_wind,build_surface_fields,dataset_timestamp,interpolate_step1_fields,iter_hours,map_prognostic_cloud,observed_temperature_3d,solar_at_step_midpoint,solve_hour,target_latitude_longitude
from pycalmet.config import GridConfig
from pycalmet.coare import initial_warm_layer_state,surface_flux
from pycalmet.sea_data import GriddedSeaData


@pytest.mark.parametrize('method',[1,2])
def test_mixht2st_vectorizes_station_columns_without_changing_grid_kernel(method):
    temperature=np.broadcast_to(
        np.array([291.,290.,287.,279.],dtype=np.float32)[:,None],(4,3))
    height=np.broadcast_to(
        np.array([0.,100.,400.,1200.],dtype=np.float32)[:,None],(4,3))
    previous=np.array([50.,100.,150.],dtype=np.float32)
    jump=np.array([0.,.2,.4],dtype=np.float32)
    ustar=np.array([.2,.25,.3],dtype=np.float32)
    length=np.array([-100.,80.,-200.],dtype=np.float32)
    heat=np.array([100.,-10.,150.],dtype=np.float32)
    density=np.full(3,1.2,dtype=np.float32)
    coriolis=np.full(3,1.e-4,dtype=np.float32)
    options={'surface_temperature_k':np.full(3,290.,dtype=np.float32)} if method==2 else {}
    station=land_station_mixing_height(temperature,height,previous,jump,ustar,
        length,heat,density,coriolis,mixing_method=method,**options)
    shaped=(temperature[:,None,:],height[:,None,:],previous[None,:],jump[None,:],
            ustar[None,:],length[None,:],heat[None,:],density[None,:],coriolis[None,:])
    if method==1:
        grid=mixing_height_maul_carson(*shaped)
    else:
        grid=mixing_height_batchvarova_gryning(
            *shaped,surface_temperature_k=np.full((1,3),290.,dtype=np.float32))
    for actual,expected in zip(station,grid,strict=True):
        assert actual.dtype==np.float32
        np.testing.assert_allclose(actual,expected[0],rtol=1.e-6,atol=1.e-6)




def test_cloud_liquid_water_profile_matches_z2face_prfvar_layers():
    cloud=np.array([.002,.01,.02,.004],dtype=np.float32)[:,None,None]
    height=np.array([100.,300.,700.,1300.],dtype=np.float32)[:,None,None]
    mapped,average,bottom,top=cloud_liquid_water_profile(
        cloud,height,[0.,400.,800.])
    np.testing.assert_allclose(mapped[:,0,0],[.006,.0175],rtol=0.,atol=1.e-8)
    np.testing.assert_array_equal(average,np.array([[.02]],dtype=np.float32))
    np.testing.assert_array_equal(bottom,[[500.]])
    np.testing.assert_array_equal(top,[[1000.]])




def test_cloud_liquid_water_profile_clips_below_ground_and_threshold():
    cloud=np.array([.1,.004],dtype=np.float32)[:,None,None]
    height=np.array([-50.,100.],dtype=np.float32)[:,None,None]
    mapped,average,bottom,top=cloud_liquid_water_profile(cloud,height,[0.,100.])
    np.testing.assert_allclose(mapped[:,0,0],[.028],rtol=0.,atol=1.e-7)
    np.testing.assert_array_equal(average,[[0.]])
    np.testing.assert_array_equal(bottom,[[0.]])
    np.testing.assert_array_equal(top,[[0.]])


def test_auxiliary_calmet_writer_header_and_zero_run_compression(tmp_path):
    path=tmp_path/'calmet.dat.aux'; grid=GridConfig(2,2,1.,1.)
    begin=datetime(2018,3,18)
    field=np.array([[0.,1.],[0.,0.]],dtype=np.float32)
    step=SimpleNamespace(cloud_liquid_water_above=field,
        cloud_liquid_water_above_bottom=field*2.,
        cloud_liquid_water_above_top=field*3.,
        cloud_liquid_water_3d=np.stack((field,field*4.)))
    hour=SimpleNamespace(timestamp=begin,step1=step)
    with AuxiliaryCalmetWriter(path,grid,[0.,20.,100.],begin,
            begin+timedelta(hours=1),control_lines=()) as writer:
        writer.append(hour)
    with FortranSequentialReader(path) as reader:
        assert reader.read_record()==(
            fixed_string('CALMET.AUX',16)+fixed_string('1.1',16)+
            fixed_string('Generic file structure with compression',64))
        assert np.frombuffer(reader.read_record(),dtype='<i4').item()==1
        assert reader.read_record().startswith(b'Produced by CALMET Version: 6.5.0')
        assert len(reader.read_record())==144
        assert reader.read_record()[:8]==b'ZFACE   '
        assert reader.read_record()[:8]==b'2D_VARS '
        assert reader.read_record()[:8]==b'3D_VARS '
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),
                                      [201807700,0,201807701,0])
        assert np.frombuffer(reader.read_record(),dtype='<i4').item()==3
        record=reader.read_record(); assert record[:12]==b'CLDMRUP     '
        expected=np.array([-(np.float32(1.)+np.float32(.0001)),1.,
                           -(np.float32(2.)+np.float32(.0001))],dtype=np.float32)
        np.testing.assert_array_equal(np.frombuffer(record[12:],dtype='<f4'),expected)
    with AuxiliaryData(path) as auxiliary:
        assert (auxiliary.header.nx,auxiliary.header.ny,auxiliary.header.nz)==(2,2,2)
        assert auxiliary.header.start==begin
        assert auxiliary.header.end==begin+timedelta(hours=1)
        assert auxiliary.header.wire_start==(2018,3,18,0,0)
        assert auxiliary.header.wire_end==(2018,3,18,1,0)
        result=auxiliary.read_hour()
        assert (result.begin,result.end)==(begin,begin+timedelta(hours=1))
        assert result.wire_time_bounds==(201807700,0,201807701,0)
        np.testing.assert_array_equal(result.cloud_water_above_g_kg,field)
        np.testing.assert_array_equal(result.cloud_bottom_above_m,field*2.)
        np.testing.assert_array_equal(result.cloud_top_above_m,field*3.)
        np.testing.assert_array_equal(result.cloud_water_3d_g_kg,np.stack((field,field*4.)))
        assert auxiliary.read_hour() is None


def test_auxiliary_calmet_writer_is_empty_without_source_cloud_water(tmp_path):
    path=tmp_path/'calmet.dat.aux'
    with AuxiliaryCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.],
            datetime(2018,1,1),datetime(2018,1,1,1),available=False):
        pass
    assert path.read_bytes()==b''


@pytest.mark.parametrize(('begin','expected'),(
    (datetime(2018,3,18,23,30),(201807723,1800,201807800,1800)),
    (datetime(2018,12,31,23,30),(201836523,1800,201900100,1800)),
))
def test_auxiliary_calmet_writer_normalizes_end_timestamp_across_dates(
        tmp_path,begin,expected):
    path=tmp_path/f'{begin:%Y%m%d}.aux'
    field=np.zeros((1,1),dtype=np.float32)
    step=SimpleNamespace(cloud_liquid_water_above=field,
        cloud_liquid_water_above_bottom=field,
        cloud_liquid_water_above_top=field,
        cloud_liquid_water_3d=field[None])
    hour=SimpleNamespace(timestamp=begin,step1=step)
    end=begin+timedelta(hours=1)
    with AuxiliaryCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.],begin,end) as writer:
        writer.append(hour)
    with AuxiliaryData(path) as auxiliary:
        result=auxiliary.read_hour()
        assert (result.begin,result.end)==(begin,end)
        assert result.wire_time_bounds==expected


@pytest.mark.parametrize('fields',(
    (201936600,0,202000100,0),
    (201900100,-1,201900101,0),
    (201900100,0,201900100,0),
    (201900101,0,201900100,0),
))
def test_auxiliary_reader_rejects_invalid_or_nonadvancing_timestamps(fields):
    with pytest.raises(ValueError,match='CALMET.AUX'):
        AuxiliaryData._decode_timestamp(struct.pack('<4i',*fields))



def test_coare_ending_hour_uses_west_positive_timezone_and_full_time():
    assert _coare_ending_gmt_hour(
        datetime(2018,1,1,0,15,30,500000),8)==pytest.approx(
            8.+15./60.+30.5/3600.)
    assert _coare_ending_gmt_hour(
        datetime(2018,1,1,23,59,59),-8)==pytest.approx(15.+59./60.+59./3600.)


@pytest.mark.parametrize(('begin','timestep_seconds','expected_day','expected_position'),(
    (datetime(2018,3,18,0,29,59),1.,77,
     (29.*60.+59.5)/3600.+.5),
    (datetime(2018,3,18,23,59,59,750000),.5,78,.5),
    (datetime(2018,12,31,23,59,59,500000),1.,1,.5),
))
def test_solar_midpoint_uses_full_datetime_across_date_boundaries(
        monkeypatch,begin,timestep_seconds,expected_day,expected_position):
    requested_days=[]
    def fake_solar(day,base_timezone,latitude,longitude):
        requested_days.append(day)
        return np.arange(26,dtype=np.float32)[:,None,None]
    monkeypatch.setattr('pycalmet.calmet_solver.solar_sine',fake_solar)

    result=solar_at_step_midpoint(
        begin,timestep_seconds,8,np.array([[30.]]),np.array([[120.]]))

    assert requested_days==[expected_day]
    assert result.dtype==np.float32
    np.testing.assert_allclose(result,expected_position,rtol=0.,atol=2.e-6)


def test_irtype0_diagnostic_lapse_stops_after_wind(monkeypatch):
    shape=(1,1); level=np.ones((1,1,1),dtype=np.float32)
    step=Step1Fields(level,level,level*280.,np.full(shape,280.,dtype=np.float32),
        np.full(shape,1000.,dtype=np.float32),np.ones(shape,dtype=np.float32),
        level*50.,np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.full(shape,.1,dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.zeros(shape,dtype=np.float32),np.full((2,1,1),280.,dtype=np.float32),
        np.zeros((2,1,1),dtype=np.float32),np.full(shape,50,dtype=np.int32),
        np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.int32),
        diagnostic_lapse_rate=np.full(shape,-.006,dtype=np.float32))
    geo={name:np.zeros(shape,dtype=np.float32) for name in
         ('terrain_height','albedo','bowen_ratio','soil_heat_flux_parameter',
          'anthropogenic_heat_flux')}
    geo['dominant_landuse']=np.full(shape,20,dtype=np.int16)
    geo['roughness_length']=np.full(shape,.1,dtype=np.float32)
    monkeypatch.setattr('pycalmet.calmet_solver.build_surface_fields',
                        lambda *args,**kwargs: pytest.fail('wind-only run entered surface physics'))
    previous_height=np.full(shape,123.,dtype=np.float32)
    previous_jump=np.full(shape,4.,dtype=np.float32)
    captured={}
    def transform(value,stamp,context):
        captured.update(stamp=stamp,context=context)
        return value
    transform.uses_physical_context=True
    result=solve_hour(None,0,GridConfig(1,1,1.,1.),(0.,20.),geo,
        datetime(2018,1,1),previous_height,previous_jump,step_override=step,
        target_coordinates=(np.zeros(shape),np.zeros(shape)),wind_only=True,
        use_froude=False,use_slope=False,smoothing_passes=(0,),step_transform=transform)
    assert captured['stamp']==datetime(2018,1,1)
    assert isinstance(captured['context'],StepTransformContext)
    np.testing.assert_array_equal(captured['context'].previous_convective_height,previous_height)
    np.testing.assert_array_equal(captured['context'].previous_mixing_height,previous_height)
    np.testing.assert_array_equal(captured['context'].previous_temperature_jump,previous_jump)
    assert captured['context'].solar_elevation_sine.shape==shape
    assert captured['context'].coriolis.shape==shape
    np.testing.assert_array_equal(result.surface.convective_mixing_height,
                                  previous_height)
    np.testing.assert_array_equal(result.surface.temperature_jump,previous_jump)


def test_lake_breeze_interpolates_by_relative_coast_distance():
    u=np.zeros((2,3,1)); v=np.zeros_like(u)
    region=(1.,1.,1.,3.,0.,1500.,1000.,1500.,(101,202))
    adjusted_u,adjusted_v=lake_breeze_adjust(
        u,v,[2.,10.],[4.,12.],[[500.,500.],[500.,2500.]],[101,202],
        (region,),1000.,1000.)
    np.testing.assert_allclose(adjusted_u[0,:,0],[2.,6.,10.],atol=1.e-12)
    np.testing.assert_allclose(adjusted_v[0,:,0],[4.,8.,12.],atol=1.e-12)
    np.testing.assert_array_equal(adjusted_u[1],0.)
    np.testing.assert_array_equal(adjusted_v[1],0.)





def test_observed_temperature_3d_uses_sea_gradients_and_mixing_height():
    surface=np.array([[290.]])
    faces=np.array([[[290.]],[[289.]],[[287.]]])
    got=observed_temperature_3d(surface,faces,np.array([[50.]]),[0.,100.,300.],
        landuse=np.array([[55]]),water_range=(55,55),mixing_height=np.array([[150.]]),
        water_gradient_below=np.array([[-.01]]),water_gradient_above=np.array([[-.005]]))
    middle=290.-.01*150.
    expected=.5*((290.-1.+middle)*50.+(middle+(middle-.005*150.))*150.)/200.
    np.testing.assert_allclose(got[:,0,0],[290.,expected])


def test_inverse_distance_minimum_and_cloud_zero_filter():
    x=np.array([[0.,1.,2.,3.]]); y=np.zeros_like(x); values=np.array([[10.,0.,30.,40.]])
    assert inverse_distance_squared(values,0.,0.,x,y)>9.9
    assert inverse_distance_squared(values,0.,0.,x,y,ignore_zero=True)>9.9




def test_cloud3_uses_level_immediately_below_850_mb_crossing():
    pressure=np.array([1000.,900.,840.,700.])[:,None,None]
    rh=np.array([20.,50.,90.,100.])[:,None,None]
    np.testing.assert_allclose(cloud_cover_teixeira_850(rh,pressure),cloud_fraction([[50.]]))
    no_crossing=np.full_like(pressure,900.)
    np.testing.assert_allclose(cloud_cover_teixeira_850(rh,no_crossing),cloud_fraction([[0.]]))

def test_cloud3_maps_rh_before_applying_nonlinear_teixeira_formula():
    class AverageMapper:
        @staticmethod
        def interpolate(values): return np.asarray(values).mean(keepdims=True).reshape(1,1)
    source_rh=np.array([[0.,100.]])
    source_cloud=cloud_fraction(source_rh)
    got,mapped_rh=map_prognostic_cloud(source_cloud,source_rh,AverageMapper(),3)
    np.testing.assert_allclose(mapped_rh,50.)
    np.testing.assert_allclose(got,cloud_fraction([[50.]]))
    assert not np.allclose(got,source_cloud.mean())



def test_source_uses_stored_relative_humidity_instead_of_reconstructing_it():
    class Dataset:
        variables={'relative_humidity':np.array([[[[100.]]]])}
        def __getitem__(self,name): return self.variables[name]
    got=_source_relative_humidity(
        Dataset(),0,np.array([[[.0099]]]),np.array([[[900.]]]),
        np.array([[[280.]]]))
    np.testing.assert_array_equal(got,100.)


def test_relative_humidity_uses_exact_wrf_vapor_pressure_conversion():
    qvapor=np.array([.01],dtype=np.float32)
    pressure=np.array([1000.],dtype=np.float32)
    temperature=np.array([300.],dtype=np.float32)
    tc=temperature-273.15
    saturation_vapor_pressure=6.112*np.exp(17.67*tc/(tc+243.5))
    vapor_pressure=pressure*qvapor/(.622+qvapor)

    result=_relative_humidity(qvapor,pressure,temperature)

    assert result.dtype==np.float32
    np.testing.assert_allclose(
        result,100.*vapor_pressure/saturation_vapor_pressure,rtol=2.e-6)


def test_step_interpolation_preserves_fractional_surface_humidity():
    shape=(1,1)
    level=np.ones((1,*shape),dtype=np.float32)
    profile=np.ones((2,*shape),dtype=np.float32)
    def step(humidity):
        return Step1Fields(
            level,level,level*280.,np.full(shape,280.,dtype=np.float32),
            np.full(shape,1000.,dtype=np.float32),np.ones(shape,dtype=np.float32),
            level*50.,np.zeros(shape,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.full(shape,.1,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32),
            profile*280.,profile*100.,np.full(shape,humidity,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.int32))

    result=interpolate_step1_fields(step(50.25),step(50.75),.25)

    assert result.surface_relative_humidity.dtype==np.float32
    np.testing.assert_allclose(result.surface_relative_humidity,50.375)


def test_step_interpolation_includes_profile_temperature_and_height():
    shape=(1,1)
    level=np.ones((1,*shape),dtype=np.float32)
    def step(profile_temperature,profile_height):
        profile_temperature=np.asarray(profile_temperature,dtype=np.float32)[:,None,None]
        profile_height=np.asarray(profile_height,dtype=np.float32)[:,None,None]
        return Step1Fields(
            level,level,level*280.,np.full(shape,280.,dtype=np.float32),
            np.full(shape,1000.,dtype=np.float32),np.ones(shape,dtype=np.float32),
            level*50.,np.zeros(shape,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.full(shape,.1,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32),
            profile_temperature,profile_height,np.full(shape,50.,dtype=np.float32),
            np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.int32))

    left=step([280.,275.],[0.,100.])
    right=step([284.,279.],[0.,120.])
    quarter=interpolate_step1_fields(left,right,.25)
    end=interpolate_step1_fields(left,right,1.)

    np.testing.assert_allclose(quarter.profile_temperature[:,0,0],[281.,276.])
    np.testing.assert_allclose(quarter.profile_height[:,0,0],[0.,105.])
    np.testing.assert_array_equal(end.profile_temperature,right.profile_temperature)
    np.testing.assert_array_equal(end.profile_height,right.profile_height)
    assert quarter.profile_temperature.dtype==np.float32
    assert quarter.profile_height.dtype==np.float32


def test_build_step1_preserves_fractional_canonical_surface_humidity():
    class Dataset:
        variables={
            'latitude':np.array([[30.]],dtype=np.float64),
            'longitude':np.array([[120.]],dtype=np.float64),
            'terrain':np.array([[0.]],dtype=np.float32),
            'height':np.array([[[[10.]],[[100.]],[[500.]]]],dtype=np.float32),
            'wind_speed':np.ones((1,3,1,1),dtype=np.float32),
            'wind_direction':np.zeros((1,3,1,1),dtype=np.float32),
            'temperature':np.array([[[[290.]],[[288.]],[[280.]]]],dtype=np.float32),
            'qvapor':np.full((1,3,1,1),.005,dtype=np.float32),
            'pressure':np.array([[[[1000.]],[[900.]],[[800.]]]],dtype=np.float32),
            'relative_humidity':np.array(
                [[[[50.25]],[[60.5]],[[70.75]]]],dtype=np.float32),
            'landuse':np.array([[1]],dtype=np.int16),
            'temperature_2m':np.array([[[290.]]],dtype=np.float32),
            'surface_pressure':np.array([[[100000.]]],dtype=np.float32),
            'precipitation':np.zeros((1,1,1),dtype=np.float32),
            'pbl_height':np.full((1,1,1),100.,dtype=np.float32),
            'sensible_heat_flux':np.zeros((1,1,1),dtype=np.float32),
            'friction_velocity':np.full((1,1,1),.1,dtype=np.float32),
        }
        def __getitem__(self,name):
            return self.variables[name]

    class IdentityMapper:
        target_shape=(1,1)
        def interpolate(self,field,*,ignore_zero=False):
            return np.asarray(field,dtype=np.float32)
        def nearest(self,field):
            return np.asarray(field,dtype=np.float32)
        def radius_interpolate(self,field,*args,**kwargs):
            return np.asarray(field,dtype=np.float32)

    result=calmet_solver.build_step1_fields(
        Dataset(),0,GridConfig(1,1,1.,1.),[0.,50.,200.],
        mapper=IdentityMapper(),cloud_method=3)

    assert result.surface_relative_humidity.dtype==np.float32
    np.testing.assert_array_equal(result.surface_relative_humidity,[[50.25]])




def test_rdmm5_packs_wind_from_each_columns_first_nonnegative_level():
    height=np.array([[[-20.,10.]],[[5.,40.]],[[100.,100.]]],dtype=np.float32)
    wind=np.array([[[1.,2.]],[[3.,4.]],[[5.,6.]]],dtype=np.float32)
    packed_height,packed_wind=_rdmm5_above_ground_profiles(height,wind)
    np.testing.assert_array_equal(packed_height,
        np.array([[[5.,10.]],[[100.,40.]],[[100.,100.]]],dtype=np.float32))
    np.testing.assert_array_equal(packed_wind,
        np.array([[[3.,2.]],[[5.,4.]],[[5.,6.]]],dtype=np.float32))


def test_rdmm5_itwprog2_interpolates_actual_temperature_from_sst():
    profile,use=_rdmm5_sst_profile(
        np.array([25.,75.,125.]),np.array([[100.,100.]]),
        np.array([[290.,290.]]),np.array([[280.,280.]]),
        np.array([[True,False]]))
    np.testing.assert_array_equal(profile[:,0,0],[282.5,287.5,292.5])
    np.testing.assert_array_equal(use[:,0,:],
        [[True,False],[True,False],[False,False]])









def test_mixdt2_profile_lapse_and_maul_carson_day_night():
    z=np.array([0.,100.,300.,1000.])[:,None,None]
    t=np.array([290.,289.,287.,280.])[:,None,None]
    tht,thtp,gamma=profile_temperature_layer(t,z,np.array([[50.]]),200.)
    np.testing.assert_allclose(tht,289.5); np.testing.assert_allclose(thtp,287.5); np.testing.assert_allclose(gamma,.001)
    profiles=np.broadcast_to(t,(4,1,2)); heights=np.broadcast_to(z,(4,1,2))
    total,convective,jump=mixing_height_maul_carson(profiles,heights,np.zeros((1,2)),np.zeros((1,2)),np.full((1,2),.2),np.array([[-100.,100.]]),np.array([[100.,-10.]]),np.full((1,2),1.2),np.full((1,2),1.e-4))
    assert convective[0,0]>0. and jump[0,0]>0. and total[0,0]>=convective[0,0]
    assert convective[0,1]==0. and jump[0,1]==0. and 50.<=total[0,1]<=2500.







def test_water2_coare_station_carries_warm_layer_between_hours():
    common=dict(
        anemometer_height_m=np.array([10.],dtype=np.float32),
        air_sensor_height_m=np.array([10.],dtype=np.float32),
        air_temperature_k=np.array([288.15],dtype=np.float32),
        air_minus_sea_temperature_k=np.array([-2.],dtype=np.float32),
        relative_humidity_percent=np.array([80.],dtype=np.float32),
        observed_mixing_height_m=np.array([9999.],dtype=np.float32),
        current_mixing_height_m=np.array([600.],dtype=np.float32),
        coriolis=np.array([1.e-4],dtype=np.float32),
        potential_lapse_k_m=np.array([.004],dtype=np.float32),
        previous_temperature_jump_k=np.array([.2],dtype=np.float32),
        z_centers_m=np.array([10.,40.,100.],dtype=np.float32),
        z_faces_m=np.array([0.,20.,60.,140.],dtype=np.float32),
        station_latitude=np.array([30.],dtype=np.float32),
        shortwave_down_w_m2=np.array([700.],dtype=np.float32),
        longwave_down_w_m2=np.array([350.],dtype=np.float32),
        rainfall_mm_h=np.array([0.],dtype=np.float32),mixing_method=-1)
    first=coare_water_station_profiles(
        np.array([3.],dtype=np.float32),np.array([4.],dtype=np.float32),**common,
        warm_layer_state=initial_warm_layer_state((1,)),solar_time_s=8.*3600.)
    second=coare_water_station_profiles(
        np.array([3.],dtype=np.float32),np.array([4.],dtype=np.float32),**common,
        warm_layer_state=first[-1],solar_time_s=9.*3600.)
    assert first[-1].initialized.all() and second[-1].initialized.all()
    np.testing.assert_array_equal(first[-1].time_old_s,[8.*3600.])
    np.testing.assert_array_equal(second[-1].time_old_s,[9.*3600.])
    assert second[-1].heat_column_j_m2.item()!=first[-1].heat_column_j_m2.item()
    mixed={name:(np.repeat(value,2) if np.shape(value)==(1,) else value)
           for name,value in common.items()}
    mixed['air_minus_sea_temperature_k']=np.array([-2.,9999.],dtype=np.float32)
    mixed_result=coare_water_station_profiles(
        np.array([3.,3.],dtype=np.float32),np.array([4.,4.],dtype=np.float32),
        **mixed,warm_layer_state=initial_warm_layer_state((2,)),
        solar_time_s=np.array([8.,8.])*3600.)
    np.testing.assert_array_equal(mixed_result[-1].initialized,[True,False])


def test_water_station_profiles_preserve_return_contract_when_all_inactive():
    f32=lambda value: np.array([value],dtype=np.float32)
    centers=np.array([10.,40.],dtype=np.float32)
    faces=np.array([0.,20.,60.],dtype=np.float32)
    neutral=neutral_water_station_profiles(
        f32(0.),f32(0.),f32(10.),f32(600.),f32(1.e-4),centers,faces,f32(30.))
    assert len(neutral)==6
    assert all(np.isnan(field).all() for field in neutral[2:])

    warm_state=initial_warm_layer_state((1,))
    coare=coare_water_station_profiles(
        f32(0.),f32(0.),f32(10.),f32(10.),f32(288.15),f32(-2.),f32(80.),
        f32(9999.),f32(600.),f32(1.e-4),f32(.004),f32(.2),centers,faces,
        f32(30.),f32(700.),f32(350.),f32(0.),mixing_method=-1,
        warm_layer_state=warm_state)
    assert len(coare)==10
    assert all(np.isnan(field).all() for field in coare[2:-1])
    assert coare[-1] is warm_state


def test_maul_carson_threshold_relaxes_without_resetting_jump():
    heights=np.array([0.,100.,400.])[:,None,None]
    temperatures=np.array([291.,290.,287.])[:,None,None]
    total,convective,new_jump=mixing_height_maul_carson(
        temperatures,heights,np.array([[100.]]),np.array([[.7]]),np.array([[.2]]),
        np.array([[-100.]]),np.array([[1.]]),np.array([[1.2]]),np.array([[1.e-4]]),
        threshold_flux_per_m=20.,relaxation=True)
    assert 0.<convective.item()<100.
    assert np.isclose(new_jump.item(),.7) and 50.<=total.item()<=2500.




def test_surface_fields_combine_land_and_no_station_water():
    shape=(1,2); profile_z=np.broadcast_to(np.array([0.,100.,300.])[:,None,None],(3,1,2)); profile_t=np.broadcast_to(np.array([290.,289.,287.])[:,None,None],(3,1,2))
    step=Step1Fields(np.ones((1,1,2)),np.zeros((1,1,2)),np.full((1,1,2),289.),np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),np.full((1,1,2),50.),np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),profile_t,profile_z,np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    result=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),np.array([[.8,.8]]),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape))
    assert result.sensible_heat_flux[0,0]>0. and result.sensible_heat_flux[0,1]==0.
    assert result.convective_mixing_height[0,0]>0. and result.convective_mixing_height[0,1]==0.
    assert result.temperature_jump[0,0]>0.
    assert result.temperature_jump[0,1]==0.
    assert np.isfinite(result.mixing_height).all()
    custom=build_surface_fields(step,np.array([[7,42]]),np.full(shape,.25),np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),np.array([[.8,.8]]),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),water_range=(42,42))
    assert custom.sensible_heat_flux[0,0]>0. and custom.sensible_heat_flux[0,1]==0.



def test_psiud_uses_exact_cube_root_and_pi_constants():
    zl=np.array([-1.])
    x=(1.-16.*zl)**.25; y=np.cbrt(1.-10.15*zl)
    psik=2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-2.*np.arctan(x)+np.pi/2.
    psic=1.5*np.log((1.+y+y*y)/3.)-np.sqrt(3.)*np.arctan((1.+2.*y)/np.sqrt(3.))+np.pi/np.sqrt(3.)
    expected=.5*(psik+psic)
    np.testing.assert_allclose(psi_momentum_dyer(zl),expected,rtol=2.e-7)


def test_avemix_zero_wind_uses_displaced_inverse_square_local_box():
    zi=np.arange(9.,dtype=float).reshape(3,3); zic=zi*2.
    got,got_convective=upwind_mixing_average(np.zeros_like(zi),np.zeros_like(zi),zi,zic,1000.,minimum_radius_cells=1)
    weights=np.array([[1/3,.5,1/3],[.5,1.,.5],[1/3,.5,1/3]])
    expected=np.sum(weights*zi)/weights.sum()
    np.testing.assert_allclose(got[1,1],expected); np.testing.assert_allclose(got_convective,got*2.)

def test_avemix_wind_extends_search_upwind_beyond_minimum_box():
    zi=np.zeros((1,5)); zi[0,0]=100.; zic=zi.copy(); u=np.full_like(zi,1.); v=np.zeros_like(zi)
    calm,_=upwind_mixing_average(np.zeros_like(u),v,zi,zic,1000.,minimum_radius_cells=1)
    windy,_=upwind_mixing_average(u,v,zi,zic,1000.,minimum_radius_cells=1)
    assert calm[0,3]==0. and windy[0,3]>0.





def test_avemix_uses_configured_trajectory_duration_for_subhour_runs():
    zi=np.arange(7.,dtype=np.float32)[None,:]
    u=np.full_like(zi,1.)
    zero=np.zeros_like(zi)
    hourly=upwind_mixing_average(u,zero,zi,zi,1000.,minimum_radius_cells=0,
                                 timestep_s=3600.)[0]
    subhour=upwind_mixing_average(u,zero,zi,zi,1000.,minimum_radius_cells=0,
                                  timestep_s=900.)[0]
    np.testing.assert_array_equal(subhour,zi)
    assert not np.array_equal(subhour,hourly)


def test_avemix_rectangular_grid_is_invariant_under_ninety_degree_rotation():
    rng=np.random.default_rng(184)
    shape=(5,7)
    zi=rng.uniform(100.,1000.,shape)
    zic=rng.uniform(0.,500.,shape)
    u=rng.uniform(-2.,2.,shape)
    v=rng.uniform(-2.,2.,shape)
    averaged,averaged_convective=upwind_mixing_average(
        u,v,zi,zic,2000.,1000.,minimum_radius_cells=1,
        half_angle_degrees=35.,timestep_s=1800.)

    # np.rot90 maps the new +x direction to old +y and new +y to old -x.
    rotated,rotated_convective=upwind_mixing_average(
        np.rot90(v),-np.rot90(u),np.rot90(zi),np.rot90(zic),1000.,2000.,
        minimum_radius_cells=1,half_angle_degrees=35.,timestep_s=1800.)
    np.testing.assert_allclose(rotated,np.rot90(averaged),rtol=0.,atol=0.)
    np.testing.assert_allclose(
        rotated_convective,np.rot90(averaged_convective),rtol=0.,atol=0.)


def test_avemix_pure_north_wind_uses_southern_upwind_cells():
    zi=np.zeros((5,1),dtype=np.float32)
    zi[0,0]=100.
    northward=np.ones_like(zi)
    averaged,_=upwind_mixing_average(
        np.zeros_like(zi),northward,zi,zi,2000.,1000.,
        minimum_radius_cells=0,timestep_s=3600.)
    calm,_=upwind_mixing_average(
        np.zeros_like(zi),np.zeros_like(zi),zi,zi,2000.,1000.,
        minimum_radius_cells=0,timestep_s=3600.)
    assert averaged[3,0]>0.
    assert calm[3,0]==0.


def test_mesopac_layer_average_matches_pacave_partial_layers():
    faces=np.array([0.,100.,300.,1000.,3000.],dtype=np.float32)
    u=np.array([1.,2.,4.,8.],dtype=np.float32)[:,None,None]
    v=-u
    lower_u,lower_v,upper_u,upper_v=mesopac_layer_average(
        u,v,np.array([[200.]],dtype=np.float32),faces)
    # Lower: 100 m of layer 1 + 100 m of layer 2.
    assert lower_u.item()==np.float32(1.5)
    # Upper: remaining 100 m of layer 2, 700 m of layer 3, 2000 m of layer 4.
    expected=np.float32((100.*2.+700.*4.+2000.*8.)/2800.)
    assert upper_u.item()==expected
    np.testing.assert_array_equal(lower_v,-lower_u)
    np.testing.assert_array_equal(upper_v,-upper_u)


def test_mesopac_layer_average_rejects_upper_layer_above_model_top():
    wind=np.ones((2,1,1),dtype=np.float32)
    with pytest.raises(ValueError,match='upper averaging height'):
        mesopac_layer_average(wind,wind,np.array([[250.]]),[0.,100.,500.])


def test_mesopac_layer_average_uses_layer_value_when_both_tops_share_layer():
    u=np.array([1.234567,9.],dtype=np.float32)[:,None,None]
    v=-u
    result=mesopac_layer_average(u,v,np.array([[200.]],dtype=np.float32),
                                 [0.,5000.,6000.])
    assert result[2].item()==u[0,0,0]
    assert result[3].item()==v[0,0,0]


def test_mesopac_layer_average_upper_top_uses_raw_mixing_height():
    wind=np.array([1.,3.],dtype=np.float32)[:,None,None]
    result=mesopac_layer_average(wind,wind,np.array([[-100.]],dtype=np.float32),
                                 [0.,100.,200.],upper_top_m=0.,
                                 minimum_upper_depth_m=150.)
    # ZZTOP=max(0,20,-100+150)=50, while lower ZI is independently 10 m.
    assert result[2].item()==wind[0,0,0]






def test_target_grid_coordinates_and_hourly_solar_index():
    lat,lon=target_latitude_longitude(GridConfig(2,2,1.,1.))
    assert lat.shape==(2,2) and lon.shape==(2,2)
    assert np.all(np.diff(lon,axis=1)>0.) and np.all(np.diff(lat,axis=0)>0.)

def test_canonical_dataset_timestamp_decoder():
    class Variable:
        shape=(1,19)
        def __getitem__(self,index): return np.frombuffer(b'2018-03-18_00:00:00',dtype='S1')
    dataset={'time_text':Variable()}
    assert dataset_timestamp(dataset,0).isoformat()=='2018-03-18T00:00:00'
    assert _prognostic_local_timestamp(dataset,0,5)==datetime(2018,3,17,19)


def test_exact_boundaries_use_their_adjacent_endpoint_ceilings(monkeypatch):
    text=np.array([list(value) for value in (
        '2018-03-18_00:00:00','2018-03-18_01:00:00',
        '2018-03-18_02:00:00','2018-03-18_03:00:00',
        '2018-03-18_04:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    ceilings=(0.,0.,100.,0.,20.); received=[]; builds=[]
    def make_step(index):
        return Step1Fields(level*index,level*index,level*280.,np.ones(shape)*280.,
            np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
            np.zeros(shape),np.ones(shape),np.zeros(shape),
            np.full(shape,ceilings[index],dtype=np.float32),profile,profile,
            np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_build(dataset,index,*args,**kwargs):
        builds.append(index)
        return make_step(index)
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        step=options['step_override']; received.append(step.ceiling_height.item())
        surface=type('Surface',(),{
            'convective_mixing_height':convective,'temperature_jump':jump})()
        return type('Hour',(),{'surface':surface,'step1':step})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{}))
    assert received==[0.,100.,100.,20.]
    assert builds==[0,1,2,3,4]

def test_single_hour_boundary_does_not_read_an_unused_future_record(monkeypatch):
    text=np.array([list(value) for value in (
        '2018-03-18_00:00:00','2018-03-18_01:00:00',
        '2018-03-18_02:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1)); received=[]
    def fake_build(dataset,index,*args,**kwargs):
        ceiling=(0.,0.,40.)[index]
        return Step1Fields(level,level,level*280.,np.ones(shape)*280.,
            np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
            np.zeros(shape),np.ones(shape),np.zeros(shape),
            np.full(shape,ceiling,dtype=np.float32),profile,profile,
            np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        step=options['step_override']; received.append(step.ceiling_height.item())
        surface=type('Surface',(),{
            'convective_mixing_height':convective,'temperature_jump':jump})()
        return type('Hour',(),{'surface':surface,'step1':step})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{},
                    time_indices=[1]))
    assert received==[0.]
    received.clear()
    list(iter_hours({'time_text':text[:2]},GridConfig(1,1,1.,1.),[0.,20.],{},
                    time_indices=[1]))
    assert received==[0.]

def test_three_hour_endpoints_pass_record_time_to_rdmm5(monkeypatch):
    text=np.array([list(value) for value in
        ('2018-03-18_00:00:00','2018-03-18_03:00:00',
         '2018-03-18_06:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1)); builds=[]
    step=Step1Fields(level,level,level*280.,np.ones(shape)*280.,
        np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
        np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),
        profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_build(dataset,index,*args,**kwargs):
        builds.append((index,kwargs['timestamp']))
        return step
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        surface=type('Surface',(),{
            'convective_mixing_height':convective,'temperature_jump':jump})()
        return type('Hour',(),{'surface':surface,'step1':options['step_override']})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{},
                    time_indices=[1],prognostic_timestep_s=10800.))
    assert builds==[(0,datetime(2018,3,18)),(1,datetime(2018,3,18,3))]

def test_gmt_endpoints_convert_each_record_to_base_time(monkeypatch):
    # 04/05 UTC are 23/00 in CALMET's positive-west IBTZ=5 base time.
    text=np.array([list(value) for value in
        ('2018-06-22_04:00:00','2018-06-22_05:00:00',
         '2018-06-22_06:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1)); builds=[]; calls=[]
    step=Step1Fields(level,level,level*280.,np.ones(shape)*280.,
        np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
        np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),
        profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_build(dataset,index,*args,**kwargs):
        assert 'solar_day' not in kwargs
        builds.append((index,kwargs['timestamp']))
        return step
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        calls.append(timestamp)
        surface=type('Surface',(),{
            'convective_mixing_height':convective,'temperature_jump':jump})()
        return type('Hour',(),{'surface':surface,'step1':options['step_override']})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{},
                    time_indices=[1],base_timezone=5))
    beginning=datetime(2018,6,21,23)
    assert builds==[(0,beginning),(1,datetime(2018,6,22))]
    assert calls==[beginning]

def test_hour_iterator_injects_external_cloud_at_interval_end(monkeypatch):
    text=np.array([list(value) for value in (
        '2018-03-18_00:00:00','2018-03-18_01:00:00',
        '2018-03-18_02:00:00')],dtype='S1')
    requested=[]; received=[]; shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    def fake_build(dataset,index,*args,**kwargs):
        return Step1Fields(level,level,level*280.,np.ones(shape)*280.,
            np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
            np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),
            profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def provider(end):
        requested.append(end); return np.full((1,1),.75),np.full((1,1),70)
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        received.append((options['cloud_fraction_override'].item(),options['cloud_ceiling_override'].item()))
        surface=type('Surface',(),{'convective_mixing_height':convective,'temperature_jump':jump})()
        step=(fake_build(dataset,index) if options['step_override'] is None
              else options['step_override'])
        return type('Hour',(),{'surface':surface,'step1':step})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{},
                    time_indices=[1],cloud_method=2,cloud_provider=provider))
    assert requested==[datetime(2018,3,18,1)] and received==[(.75,70)]

def test_subhour_iterator_caches_endpoints_and_interpolates_step1(monkeypatch):
    text=np.array([list(value) for value in ('2018-03-18_00:00:00','2018-03-18_01:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1)); builds=[]; calls=[]
    def fake_build(dataset,index,*args,**kwargs):
        builds.append((index,kwargs['prognostic_ocean_category'])); value=float(index*4)
        return Step1Fields(level*value,level*value,level*280.,np.ones(shape)*280.,np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        calls.append((timestamp,float(options['step_override'].u.item())))
        surface=type('Surface',(),{'convective_mixing_height':convective,'temperature_jump':jump})()
        return type('Hour',(),{'surface':surface,'step1':options['step_override']})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    ends=[datetime(2018,3,18)+timedelta(minutes=15*i) for i in range(1,5)]
    samples=[(0,1,i/4.,end) for i,end in enumerate(ends,1)]
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{},time_samples=samples,
                    timestep_s=900.,prognostic_timestep_s=3600.,
                    prognostic_ocean_category=7))
    assert builds==[(0,7),(1,7)]
    assert calls==[(end-timedelta(minutes=15),float(i)) for i,end in enumerate(ends,1)]


def test_streaming_calmet_netcdf_writer(tmp_path):
    shape=(1,1); three=np.ones((3,1,1)); step=Step1Fields(three,three,three*290.,np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),three*50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),three*290.,np.array([0.,100.,300.])[:,None,None],np.full(shape,50.25,dtype=np.float32),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(three,three,np.ones((4,1,1)),three,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),np.full(shape,500.),np.full(shape,300.),np.zeros(shape))
    hour=HourFields(datetime(2018,3,18),step,wind,surface,three*280.); geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20)}
    path=tmp_path/'calmet.nc'
    with CalmetNetCDFWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo) as writer: writer.append(hour,np.full(shape,.5),geo['dominant_landuse'])
    with nc.Dataset(path) as ds:
        assert ds.dimensions['time'].size==1; np.testing.assert_allclose(ds['wind_u'][0],three)
        np.testing.assert_array_equal(ds['surface_relative_humidity'][0],50.25)
        np.testing.assert_array_equal(ds['precipitation_rate'][0],0.)
        assert ds['pgt_stability_class'][0,0,0]>=1
    wind_only=tmp_path/'wind_only.nc'
    with CalmetNetCDFWriter(wind_only,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo,run_type=0) as writer:
        writer.append(hour,np.full(shape,.5),geo['dominant_landuse'])
    with nc.Dataset(wind_only) as ds:
        assert 'wind_u' in ds.variables and 'wind_v' in ds.variables
        assert 'temperature' not in ds.variables and 'mixing_height' not in ds.variables

def test_streaming_writer_preserves_pre_average_water_wstar(tmp_path):
    shape=(1,1); three=np.ones((3,1,1))
    step=Step1Fields(three,three,three*290.,np.full(shape,290.),np.full(shape,1000.),
        np.full(shape,1.2),three*50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),
        np.zeros(shape),np.zeros(shape),three*290.,np.array([0.,100.,300.])[:,None,None],
        np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(three,three,np.ones((4,1,1)),three,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),
        np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),np.full(shape,500.),
        np.full(shape,999.),np.zeros(shape),None,np.full(shape,7.,dtype=np.float32))
    hour=HourFields(datetime(2018,3,18),step,wind,surface,three*280.)
    geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,55)}
    path=tmp_path/'water.nc'
    with CalmetNetCDFWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo) as writer:
        writer.append(hour,np.full(shape,.5),geo['dominant_landuse'])
    with nc.Dataset(path) as ds:
        assert ds['convective_velocity_scale'][0,0,0]==7.

def test_surface_fields_can_retain_pre_average_mixing_grids():
    shape=(2,2)
    original=np.arange(4,dtype=np.float32).reshape(shape)+100.
    convective=original+50.
    surface=SurfaceFields(
        np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),
        np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),
        original+10.,convective+10.,np.zeros(shape),
        unaveraged_mixing_height=original.copy(),
        unaveraged_convective_mixing_height=convective.copy())
    np.testing.assert_array_equal(surface.unaveraged_mixing_height,original)
    np.testing.assert_array_equal(
        surface.unaveraged_convective_mixing_height,convective)
    assert not np.shares_memory(surface.mixing_height,
                                surface.unaveraged_mixing_height)

def test_temp3d_prognostic_branch_applies_surface_and_convective_adiabat():
    source=np.array([[[290.]],[[285.]],[[275.]]]); surface=np.array([[292.]])
    got=prognostic_temperature_3d(
        source,surface,np.array([[180.]]),[0.,20.,180.,420.])
    np.testing.assert_allclose(got[:,0,0],[292.,292.-.0098*90.,275.])

def test_temp3d_prognostic_branch_blends_by_layer_overlap_continuously():
    source=np.broadcast_to(
        np.array([290.,280.,270.],dtype=np.float32)[:,None,None],(3,1,3))
    surface=np.full((1,3),300.,dtype=np.float32)
    lid=np.array([[100.,200.,300.]],dtype=np.float32)
    got=prognostic_temperature_3d(source,surface,lid,[0.,100.,300.,700.])
    mixed=300.-.0098*(200.-50.)
    np.testing.assert_allclose(got[1,0],[280.,.5*(280.+mixed),mixed],rtol=2e-6)
    np.testing.assert_allclose(got[2,0],270.)

    epsilon=np.float32(.001)
    adjacent_lids=np.array([[200.-epsilon,200.+epsilon]],dtype=np.float32)
    adjacent_source=np.broadcast_to(source[:,:,:1],(3,1,2))
    adjacent_surface=np.full((1,2),300.,dtype=np.float32)
    adjacent=prognostic_temperature_3d(
        adjacent_source,adjacent_surface,adjacent_lids,[0.,100.,300.,700.])
    expected_delta=(mixed-280.)*(2.*epsilon/200.)
    np.testing.assert_allclose(
        adjacent[1,0,1]-adjacent[1,0,0],expected_delta,rtol=.01,atol=1e-6)

@pytest.mark.parametrize('faces',([0.,100.,300.],[0.,100.,100.,700.]))
def test_temp3d_prognostic_branch_rejects_invalid_faces(faces):
    source=np.zeros((3,1,1),dtype=np.float32)
    with pytest.raises(ValueError,match='one increasing face'):
        prognostic_temperature_3d(source,np.zeros((1,1)),np.zeros((1,1)),faces)

def test_gfortran_sequential_record_roundtrip(tmp_path):
    path=tmp_path/'records.dat'; payload=fixed_string('CALMET.DAT',16)+fixed_string('2.1',16)+fixed_string('No-Obs file structure with embedded control file',64)
    with FortranSequentialWriter(path) as writer: writer.write_record(payload); writer.write_record(np.array([7],dtype='<i4'))
    with FortranSequentialReader(path) as reader:
        assert reader.read_record()==payload
        assert np.frombuffer(reader.read_record(),dtype='<i4').item()==7
        assert reader.read_record() is None


def test_gfortran_sequential_subrecords_roundtrip_and_validate_markers(tmp_path):
    path=tmp_path/'subrecords.dat'
    with FortranSequentialWriter(path,max_subrecord_bytes=3) as writer:
        writer.write_record(b'abcdefgh')
        writer.write_record(b'')
    expected=(b'\xfd\xff\xff\xffabc\x03\x00\x00\x00'
              b'\xfd\xff\xff\xffdef\xfd\xff\xff\xff'
              b'\x02\x00\x00\x00gh\xfe\xff\xff\xff'
              b'\x00\x00\x00\x00\x00\x00\x00\x00')
    assert path.read_bytes()==expected
    with FortranSequentialReader(path) as reader:
        assert reader.read_record()==b'abcdefgh'
        assert reader.read_record()==b''
        assert reader.read_record() is None

    malformed=tmp_path/'malformed.dat'
    malformed.write_bytes(expected[:7]+b'\xfd\xff\xff\xff')
    with FortranSequentialReader(malformed) as reader:
        with pytest.raises(ValueError,match='subrecord markers'):
            reader.read_record()

def test_legacy_calmet_header_and_hour_record_layout(tmp_path):
    shape=(1,1); three=np.ones((3,1,1)); step=Step1Fields(three,three,three*290.,np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),three*50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),three*290.,np.array([0.,100.,300.])[:,None,None],np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(three,three,np.ones((4,1,1)),three,np.zeros(shape),np.zeros(shape)); surface=SurfaceFields(np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),np.full(shape,500.),np.full(shape,300.),np.zeros(shape)); hour=HourFields(datetime(2018,3,18),step,wind,surface,three*280.)
    geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20),'roughness_length':np.full(shape,.1),'leaf_area_index':np.ones(shape)}; path=tmp_path/'calmet.dat'
    with LegacyCalmetWriter(path,GridConfig(1,1,1.,1.,projection='LCC'),[0.,20.,100.,300.],geo,datetime(2018,3,18),datetime(2018,3,18,1),control_lines=['TEST'],base_timezone='UTC-0830',water_range=(42,42),landuse_category_count=2,run_type=0) as writer:
        writer.append(hour,np.full(shape,.5),surface_relative_humidity=np.full(shape,50),precipitation_rate=np.zeros(shape),precipitation_code=np.zeros(shape))
    with FortranSequentialReader(path) as reader:
        assert len(reader.read_record())==96; n=np.frombuffer(reader.read_record(),dtype='<i4').item(); assert n==3
        for _ in range(n): assert len(reader.read_record())==132
        run=reader.read_record(); assert len(run)==176
        assert run[40:48]==b'UTC-0830'
        np.testing.assert_array_equal(np.frombuffer(run[48:68],dtype='<i4'),[1,0,1,1,3])
        np.testing.assert_array_equal(np.frombuffer(run[80:116],dtype='<i4'),[1,0,0,-1,0,2,42,42,0])
        assert run[116:124]==b'LCC     ' and run[124:132]==b'WGS-84  '
        np.testing.assert_allclose(np.frombuffer(run[144:152],dtype='<f4'),[0.,0.])
        assert run[152:156]==b'N   ' and np.frombuffer(run[156:160],dtype='<i4').item()==50
        assert reader.read_record()[:8]==b'ZFACE   '
        for _ in range(4): reader.read_record()
        first_hour=reader.read_record(); assert first_hour[:8]==b'U-LEV  1'; assert len(first_hour)==28
        np.testing.assert_array_equal(np.frombuffer(first_hour[8:24],dtype='<i4'),
                                      [201807700,0,201807701,0])

    no_precip=tmp_path/'calmet-no-precip.dat'
    with LegacyCalmetWriter(no_precip,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo,
            datetime(2018,3,18),datetime(2018,3,18,1),run_type=1,
            precipitation_station_count=0) as writer:
        writer.append(hour,np.full(shape,.5))
    records=[]
    with FortranSequentialReader(no_precip) as reader:
        while (record:=reader.read_record()) is not None: records.append(record)
    run=records[4]
    assert np.frombuffer(run[92:96],dtype='<i4').item()==0
    labels={record[:8].decode('ascii',errors='ignore').strip() for record in records if len(record)>=8}
    assert 'RMM' not in labels and 'IPCODE' not in labels

    stations=tmp_path/'calmet-stations.dat'
    with LegacyCalmetWriter(stations,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo,
            datetime(2018,3,18),datetime(2018,3,18,1),run_type=0,
            precipitation_station_count=1,surface_coordinates_km=[(1.,2.)],
                upper_coordinates_km=[(3.,4.)],precipitation_coordinates_km=[(5.,6.)],
                iwfcod=0,sea_station_count=2) as writer:
        writer.append(hour,np.full(shape,.5))
    with FortranSequentialReader(stations) as reader:
        reader.read_record(); reader.read_record(); reader.read_record(); reader.read_record()
        run=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(run[80:100],dtype='<i4'),[0,1,1,1,2])
        assert reader.read_record()[:8]==b'ZFACE   '
        values=[]
        for label in (b'XSSTA   ',b'YSSTA   ',b'XUSTA   ',b'YUSTA   '):
            record=reader.read_record(); assert record[:8]==label
            values.append(np.frombuffer(record[24:],dtype='<f4').item())
        np.testing.assert_array_equal(values,[1000.,2000.,3000.,4000.])
        for label,value in ((b'XPSTA   ',5000.),(b'YPSTA   ',6000.)):
            record=reader.read_record(); assert record[:8]==label
            assert np.frombuffer(record[24:],dtype='<f4').item()==value
        for label in (b'Z0      ',b'ILANDU  ',b'ELEV    ',b'XLAI    '):
            assert reader.read_record()[:8]==label
        nearest=reader.read_record()
        assert nearest[:8]==b'NEARS   '
        assert np.frombuffer(nearest[24:],dtype='<i4').item()==1


def test_legacy_calmet_subhour_record_uses_fortran_seconds_encoding(tmp_path):
    shape=(1,1); three=np.ones((3,1,1))
    step=Step1Fields(three,three,three*290.,np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),three*50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),three*290.,np.array([0.,100.,300.])[:,None,None],np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(three,three,np.ones((4,1,1)),three,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),np.full(shape,500.),np.full(shape,300.),np.zeros(shape))
    run_begin=datetime(2018,3,18); begin=run_begin+timedelta(minutes=15)
    hour=HourFields(begin,step,wind,surface,three*280.)
    geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20),'roughness_length':np.full(shape,.1),'leaf_area_index':np.ones(shape)}
    path=tmp_path/'quarter_hour.dat'
    with LegacyCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],
            geo,run_begin,run_begin+timedelta(hours=1),run_type=0,
            timestep_seconds=900) as writer:
        writer.append(hour,np.full(shape,.5))
    with FortranSequentialReader(path) as reader:
        reader.read_record(); count=np.frombuffer(reader.read_record(),dtype='<i4').item()
        for _ in range(count): reader.read_record()
        reader.read_record()
        for _ in range(5): reader.read_record()
        record=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(record[8:24],dtype='<i4'),
                                      [201807700,900,201807700,1800])


def test_legacy_calmet_field_record_honors_explicit_end_across_year(tmp_path):
    shape=(1,1)
    geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20),
         'roughness_length':np.full(shape,.1),'leaf_area_index':np.ones(shape)}
    begin=datetime(2018,12,31,23,59,30)
    end=datetime(2019,1,1,0,0,15)
    path=tmp_path/'cross-year.dat'
    run_begin=datetime(2018,12,31,23)
    with LegacyCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.],geo,
            run_begin,run_begin+timedelta(hours=1),
            run_type=0,timestep_seconds=900) as writer:
        writer._field_record('CHECK',np.ones(shape),begin,end)
    with FortranSequentialReader(path) as reader:
        while (record:=reader.read_record()) is not None:
            if record[:8]==b'CHECK   ':
                np.testing.assert_array_equal(np.frombuffer(record[8:24],dtype='<i4'),
                    [201836523,3570,201900100,15])
                break
        else:
            pytest.fail('CHECK record was not written')


def test_integer_hour_legacy_headers_reject_fractional_run_duration(tmp_path):
    grid=GridConfig(1,1,1.,1.)
    begin=datetime(2018,3,18)
    end=begin+timedelta(minutes=15)
    geo={'terrain_height':np.zeros((1,1)),
         'dominant_landuse':np.full((1,1),20),
         'roughness_length':np.full((1,1),.1),
         'leaf_area_index':np.ones((1,1))}
    with pytest.raises(ValueError,match='whole-hour'):
        AuxiliaryCalmetWriter(tmp_path/'calmet.aux',grid,[0.,20.],begin,end)
    with pytest.raises(ValueError,match='whole-hour'):
        LegacyCalmetWriter(
            tmp_path/'calmet.dat',grid,[0.,20.],geo,begin,end,run_type=0)
    with pytest.raises(ValueError,match='whole-hour'):
        MesopacWriter(tmp_path/'pacout.dat',grid,[0.,20.],geo,begin,end)


def test_mesopac_writer_matches_outpc_record_layout(tmp_path):
    shape=(2,2); one=np.arange(4,dtype=np.float32).reshape(1,2,2)+1.
    step=Step1Fields(one,one,one*0.+290.,np.full(shape,290.),np.full(shape,1000.),
        np.full(shape,1.2),one*0.+50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),
        np.full(shape,2.),np.full(shape,7),one*0.+290.,np.zeros((1,2,2)),
        np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(one,-one,np.ones((2,2,2)),one,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.1),np.array([[1.,2.],[3.,4.]],dtype=np.float32),
        np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),
        np.full(shape,500.),np.full(shape,300.),np.zeros(shape))
    hour=HourFields(datetime(2018,3,18,4),step,wind,surface,one*0.+280.)
    geo={'dominant_landuse':np.full(shape,20),'roughness_length':np.full(shape,.125)}
    path=tmp_path/'pacout.dat'; grid=GridConfig(2,2,1.,1.)
    with MesopacWriter(path,grid,[0.,5000.],geo,datetime(2018,3,18),
            datetime(2018,3,18,1),timezone_offset=8,
            surface_coordinates_m=[(500.,500.),(1500.,1500.)],
            upper_coordinates_m=[(500.,1500.)],nearest_surface_station=[[1,1],[2,2]]) as writer:
        writer.append(hour,np.full(shape,.5),station_temperature_k=[280.,281.],
            station_shortwave_flux=[10.,20.],station_relative_humidity=[50,60],
            station_precipitation_code=[0,1])
    with FortranSequentialReader(path) as reader:
        header=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(header[:40],dtype='<i4'),
                                      [2018,77,1,2,1,2,2,8,2,4])
        np.testing.assert_array_equal(np.frombuffer(header[40:],dtype='<f4'),
                                      np.array([1000.,.4],dtype=np.float32))
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<f4'),[1.,2.,1.,2.])
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<f4'),[1.,2.])
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<f4'),np.full(4,.125))
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),[1,1,2,2])
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),np.full(4,20))
        np.testing.assert_array_equal(np.frombuffer(reader.read_record(),dtype='<i4'),[2018,77,4])
        grids=[reader.read_record() for _ in range(10)]
        np.testing.assert_array_equal(np.frombuffer(grids[0],dtype='<f4'),one.ravel())
        np.testing.assert_array_equal(np.frombuffer(grids[1],dtype='<f4'),-one.ravel())
        np.testing.assert_array_equal(np.frombuffer(grids[8],dtype='<i4').shape,(4,))
        tail=reader.read_record()
        assert np.frombuffer(tail[:4],dtype='<f4').item()==np.float32(2.5)
        np.testing.assert_array_equal(np.frombuffer(tail[4:20],dtype='<f4'),[280.,281.,10.,20.])
        np.testing.assert_array_equal(np.frombuffer(tail[20:],dtype='<i4'),[50,60,0,1])
        assert reader.read_record() is None


def test_legacy_mesopac_writer_is_byte_identical_to_gfortran_outpc_records(tmp_path):
    compiler=shutil.which('gfortran')
    if compiler is None: pytest.skip('gfortran is required for the binary oracle')
    source=tmp_path/'oracle.f90'
    source.write_text(textwrap.dedent('''
        program oracle
          implicit none
          integer, parameter :: nx=2,ny=2,ns=2,nu=1
          integer :: io
          real :: xs(ns),ys(ns),xu(nu),yu(nu),z0(nx,ny)
          real :: ul(nx,ny),vl(nx,ny),uup(nx,ny),vup(nx,ny)
          real :: htmix(nx,ny),ustar(nx,ny),wstar(nx,ny)
          real :: xmonin(nx,ny),rmm(nx,ny),tempk(ns),srad(ns),avrho
          integer :: nears(nx,ny),ilandu(nx,ny),ipgt(nx,ny),irh(ns),ipcode(ns)
          xs=(/1.,2./); ys=(/1.,2./); xu=(/1./); yu=(/2./)
          z0=.125; nears=reshape((/1,1,2,2/),(/nx,ny/)); ilandu=55
          ul=reshape((/1.,2.,3.,4./),(/nx,ny/)); vl=-ul; uup=ul; vup=vl
          htmix=500.; ustar=.2
          wstar=(9.81*(20./1195.2)*300./280.)**.3333333
          xmonin=-100.; ipgt=4; rmm=2.
          avrho=2.5; tempk=(/280.,281./); srad=(/10.,20./)
          irh=(/50,60/); ipcode=(/0,1/)
          open(newunit=io,file='oracle.dat',form='unformatted',status='replace')
          write(io) 2018,77,1,ns,nu,nx,ny,8,2,4,1000.,.4
          write(io) xs,ys
          write(io) xu,yu
          write(io) z0
          write(io) nears
          write(io) ilandu
          write(io) 2018,77,4
          write(io) ul
          write(io) vl
          write(io) uup
          write(io) vup
          write(io) htmix
          write(io) ustar
          write(io) wstar
          write(io) xmonin
          write(io) ipgt
          write(io) rmm
          write(io) avrho,tempk,srad,irh,ipcode
          close(io)
        end program oracle
    '''),encoding='ascii')
    subprocess.run([compiler,source.name,'-o','oracle'],cwd=tmp_path,check=True,
                   capture_output=True,text=True)
    subprocess.run([str(tmp_path/'oracle')],cwd=tmp_path,check=True,
                   capture_output=True,text=True)

    shape=(2,2); one=np.arange(4,dtype=np.float32).reshape(1,2,2)+1.
    step=Step1Fields(one,one,one*0.+290.,np.full(shape,290.),np.full(shape,1000.),
        np.array([[1.,2.],[3.,4.]],dtype=np.float32),one*0.+50.,np.zeros(shape),
        np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),one*0.+290.,
        np.zeros((1,2,2)),np.full(shape,50),np.full(shape,2.),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(one,-one,np.ones((2,2,2)),one,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.125),np.array([[1.,2.],[3.,4.]],dtype=np.float32),
        np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),
        np.full(shape,500.),np.full(shape,300.),np.zeros(shape))
    hour=HourFields(datetime(2018,3,18,4),step,wind,surface,one*0.+280.)
    geo={'dominant_landuse':np.full(shape,55),'roughness_length':np.full(shape,.125)}
    python_output=tmp_path/'python.dat'
    with LegacyMesopacWriter(python_output,GridConfig(2,2,1.,1.),[0.,5000.],geo,
            datetime(2018,3,18),datetime(2018,3,18,1),timezone_offset=8,
            surface_coordinates_m=[(500.,500.),(1500.,1500.)],
            upper_coordinates_m=[(500.,1500.)],nearest_surface_station=[[1,1],[2,2]]) as writer:
        writer.append(hour,np.zeros(shape),station_temperature_k=[280.,281.],
            station_shortwave_flux=[10.,20.],station_relative_humidity=[50,60],
            station_precipitation_code=[0,1])
    assert python_output.read_bytes()==(tmp_path/'oracle.dat').read_bytes()



def test_rdmm5_ending_record_maps_to_preceding_half_hour_solar_sample():
    latitude=np.array([[30.]],dtype=np.float32)
    longitude=np.array([[120.]],dtype=np.float32)
    samples=solar_sine(77,0,latitude,longitude)
    got=_prognostic_record_solar(
        datetime(2018,3,18,10),0,latitude,longitude)
    np.testing.assert_array_equal(got,samples[10])
    assert not np.array_equal(got,samples[11])

def test_prognostic_midnight_record_uses_its_actual_date_solar_table():
    latitude=np.array([[70.]],dtype=np.float32)
    longitude=np.array([[0.]],dtype=np.float32)
    record_time=datetime(2018,6,22)
    got=_prognostic_record_solar(record_time,0,latitude,longitude)
    expected=solar_sine(173,0,latitude,longitude)[0]
    previous_day=solar_sine(172,0,latitude,longitude)[0]
    np.testing.assert_array_equal(got,expected)
    assert not np.array_equal(got,previous_day)


@pytest.mark.parametrize('record_time,previous_julian',(
    (datetime(2019,1,1),365),
    (datetime(2021,1,1),366),
))
def test_prognostic_midnight_record_uses_new_year_date(record_time,previous_julian):
    latitude=np.array([[70.]],dtype=np.float32)
    longitude=np.array([[0.]],dtype=np.float32)
    got=_prognostic_record_solar(record_time,0,latitude,longitude)
    np.testing.assert_array_equal(got,solar_sine(1,0,latitude,longitude)[0])
    assert not np.array_equal(
        got,solar_sine(previous_julian,0,latitude,longitude)[0])




def test_topographic_velocity_is_zero_for_flat_terrain():
    u=np.ones((2,2,3)); v=np.zeros_like(u)
    got=topographic_vertical_velocity(u,v,np.zeros((2,3)),[0.,20.,40.],np.zeros((2,3)),np.full((2,3),300.),.1,1000.,1000.)
    np.testing.assert_array_equal(got,0.)

@pytest.mark.parametrize(('terrain','value_index','distance_index','expected'),[
    (np.array([[10.,5.,5.,5.,10.,5.,5.]]),0,2,10.),
    (np.array([[0.,5.,5.,5.,0.,5.,5.]]),1,3,0.),
])
def test_terrain_extrema_equal_values_use_nearest_occurrence_independent_of_scan_direction(
        terrain,value_index,distance_index,expected):
    result=terrain_extrema(terrain,4.,1000.,1000.)
    reversed_result=terrain_extrema(terrain[:,::-1],4.,1000.,1000.)

    for field,reversed_field in zip(result,reversed_result,strict=True):
        assert field.dtype==np.float32
        np.testing.assert_array_equal(reversed_field,field[:,::-1])
    assert result[value_index][0,3]==expected
    assert result[distance_index][0,3]==1000.


def test_terrain_extrema_respects_physical_distance_on_rectangular_grid():
    terrain=np.zeros((5,5))
    terrain[2,3]=10.  # one 2 km x-cell from the centre
    terrain[3,2]=10.  # one 1 km y-cell from the centre

    result=terrain_extrema(terrain,3.,2000.,1000.)
    transposed=terrain_extrema(terrain.T,3.,1000.,2000.)

    assert result[0][2,2]==10.
    assert result[2][2,2]==1000.
    for field,transposed_field in zip(result,transposed,strict=True):
        np.testing.assert_array_equal(transposed_field,field.T)


def test_slope_flow_respects_water_and_layer_depth():
    terrain=np.array([[0.,10.,20.]])
    high,low,dhi,dlo=terrain_extrema(terrain,3.,1000.,1000.)
    us,vs=slope_flow(terrain,high,low,dhi,dlo,np.full_like(terrain,-50.),np.full_like(terrain,290.),np.array([[20,55,20]]),[0.,.25,1.],1000.,1000.)
    assert us.shape==(2,1,3); assert np.all(us[:,0,1]==0.); assert np.allclose(vs,0.)

def test_slope_flow_uses_standard_power_for_gradient_magnitude():
    terrain=np.array([[0.,10.,20.]],dtype=np.float32)
    high,low,dhi,dlo=terrain_extrema(terrain,3.,1000.,1000.)
    u,v=slope_flow(terrain,high,low,dhi,dlo,np.full_like(terrain,-50.),
        np.full_like(terrain,290.),np.full(terrain.shape,20),
        [0.,.25,1.],1000.,1000.)
    assert np.isfinite(u).all() and np.isfinite(v).all()

def test_prognostic_mapper_hits_exact_source_point():
    lat=np.array([[27.,27.],[28.,28.]])
    lon=np.array([[117.5,118.5],[117.5,118.5]])
    mapper=PrognosticMapper.build(
        lat,lon,GridConfig(1,1,.001,.001,projection='LCC'))
    got=mapper.interpolate(np.array([[7.,2.],[3.,4.]]))
    assert got.shape==(1,1); assert abs(got.item()-7.)<.02

def test_precipitation_radius_mapping_uses_all_nearby_sources_and_fallback():
    lat=np.array([[27.,27.],[28.,28.]])
    lon=np.array([[117.5,118.5],[117.5,118.5]])
    mapper=PrognosticMapper.build(
        lat,lon,GridConfig(1,1,.001,.001,projection='LCC'))
    field=np.array([[7.,2.],[3.,4.]])
    nearest=mapper.radius_interpolate(field,.00001)
    np.testing.assert_allclose(nearest,mapper.nearest(field))
    assert mapper.radius_interpolate(field,500.).item()>2.
    np.testing.assert_array_equal(mapper.radius_interpolate(np.zeros((2,2)),50.,cutoff=.01),0.)

def test_interpqr_supports_inverse_distance_and_exponential_methods():
    mapper=PrognosticMapper((1,3),np.array([[0,1,2,2]]),np.ones((1,4))/4.,(1,1),
        np.array([[1.,0.],[2.,0.],[3.,0.]]),np.array([[0.,0.]]))
    field=np.array([[0.,10.,20.]])
    expected1=(10./2.+20./3.)/(1.+1./2.+1./3.)
    expected2=(10./4.+20./9.)/(1.+1./4.+1./9.)
    np.testing.assert_allclose(mapper.radius_interpolate(field,4.,method=1),expected1)
    np.testing.assert_allclose(mapper.radius_interpolate(field,4.,method=2),expected2)
    weights=np.exp(-np.array([1.,4.,9.])/4.)/np.array([1.,4.,9.])
    np.testing.assert_allclose(mapper.radius_interpolate(field,2.,method=3),np.sum(field.ravel()*weights)/weights.sum())
    automatic_weights=np.exp(-np.array([1.,4.,9.])/.25)/np.array([1.,4.,9.])
    expected_automatic=np.sum(field.ravel()*automatic_weights)/automatic_weights.sum()
    np.testing.assert_allclose(
        mapper.radius_interpolate(field,0.,method=3),expected_automatic,rtol=2.e-6)


def test_interpqr_exponential_has_no_nearest_fallback_beyond_cutoff():
    mapper=PrognosticMapper((1,1),np.array([[0,0,0,0]]),np.ones((1,4)),(1,1),
        np.array([[0.,0.]],dtype=np.float32),
        np.array([[100.,100.]],dtype=np.float32))
    result=mapper.radius_interpolate(np.array([[7.]],dtype=np.float32),
                                     .001,method=3,cutoff=0.)
    np.testing.assert_array_equal(result,np.zeros((1,1),dtype=np.float32))
    tiny=mapper.radius_interpolate(np.array([[1.e-35]],dtype=np.float32),
                                   1000.,method=3,cutoff=0.)
    np.testing.assert_array_equal(tiny,np.zeros((1,1),dtype=np.float32))


def test_interpqr_nearest_fallback_does_not_apply_cutoff():
    mapper=PrognosticMapper((1,1),np.array([[0,0,0,0]]),np.ones((1,4)),(1,1),
        np.array([[0.,0.]],dtype=np.float32),
        np.array([[100.,100.]],dtype=np.float32))
    field=np.array([[.005]],dtype=np.float32)
    for method in (1,2):
        np.testing.assert_array_equal(
            mapper.radius_interpolate(field,.001,method=method,cutoff=.01),field)




def test_prognostic_observation_mapping_retains_raw_weight_strength():
    mapper=PrognosticMapper((1,3),np.array([[0,1,2,2]]),np.ones((1,4))/4.,(1,1),
        np.array([[1.,0.],[2.,0.],[3.,0.]]),np.array([[0.,0.]]))
    field=np.array([[0.,10.,20.]])
    mapped,strength=mapper.interpolate_observations(field,.1)
    raw=np.array([1.,1./4.,1./9.,1./9.])
    assert strength.item()==pytest.approx(raw.sum())
    assert mapped.item()==pytest.approx(np.dot([0.,10.,20.,20.],raw)/raw.sum())


def test_prognostic_observation_mapping_applies_relative_barriers_by_level():
    mapper=PrognosticMapper((1,4),np.array([[0,1,2,3]]),np.ones((1,4))/4.,(1,1),
        np.array([[101.,100.],[100.,101.],[100.,99.],[99.,100.]]),
        np.array([[100.,100.]]),(100.,100.))
    field=np.array([[[10.,20.,30.,40.]],[[10.,20.,30.,40.]]])
    mapped,strength=mapper.interpolate_observations(
        field,.1,barriers=((.5,-1.,.5,1.),),barrier_top_level=1)
    np.testing.assert_allclose(mapped[:,0,0],[30.,25.])
    np.testing.assert_allclose(strength[:,0,0],[3.,4.])


def test_prognostic_observation_mapping_removes_fully_blocked_target():
    mapper=PrognosticMapper((1,4),np.array([[0,1,2,3]]),np.ones((1,4))/4.,(1,1),
        np.array([[1.,0.],[2.,0.],[3.,0.],[4.,0.]]),np.array([[0.,0.]]))
    mapped,strength=mapper.interpolate_observations(np.ones((1,1,4)),.1,
        barriers=((.5,-1.,.5,1.),),barrier_top_level=1)
    np.testing.assert_array_equal(mapped,0.)
    np.testing.assert_array_equal(strength,0.)


def test_prognostic_sst_uses_nearest_ocean_source_not_nearest_any_source():
    mapper=PrognosticMapper((1,3),np.array([[0,1,2,2]]),np.ones((1,4))/4.,(1,1),
        np.array([[0.,0.],[1.,0.],[2.,0.]],dtype=np.float32),
        np.array([[.1,0.]],dtype=np.float32))
    field=np.array([[100.,200.,300.]],dtype=np.float32)
    np.testing.assert_array_equal(
        mapper.nearest_masked(field,np.array([[False,True,True]])),[[200.]])
    with pytest.raises(ValueError,match='at least one ocean source'):
        mapper.nearest_masked(field,np.zeros((1,3),dtype=bool))


def test_prognostic_sst_nearest_ocean_accepts_any_equidistant_source():
    mapper=PrognosticMapper((2,2),np.array([[0,1,2,3]]),np.ones((1,4))/4.,(1,1),
        np.array([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.]],dtype=np.float32),
        np.array([[0.,0.]],dtype=np.float32))
    field=np.array([[10.,30.],[20.,40.]],dtype=np.float32)
    selected=mapper.nearest_masked(field,np.ones((2,2),dtype=bool)).item()
    assert selected in field.ravel()


def test_prognostic_four_neighbours_select_the_nearest_tied_points():
    source=np.array([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.],
                     [-2.,0.],[0.,-2.],[2.,0.],[0.,2.]],dtype=np.float32)
    got=_nearest_indices(source,np.array([[0.,0.]],dtype=np.float32),4)
    np.testing.assert_array_equal(np.sort(got[0]),[0,1,2,3])
    np.testing.assert_allclose(np.linalg.norm(source[got[0]],axis=1),1.)




def test_wind1_initial_guess_removes_calm_and_barrier_by_level():
    points=np.array([[.5,.5],[.5,2.5]])
    u=np.array([[0.,8.],[0.,10.]])
    v=np.zeros_like(u)
    got_u,_=wind1_initial_guess(u,v,points,1,1,1,1.,1.,extrapolate_calm=False,
        barriers=((-1.,1.,2.,1.),),barrier_top_level=1)
    # The upper station is blocked in layer 1, while the calm surface station
    # is removed aloft and the barrier no longer applies in layer 2.
    np.testing.assert_allclose(got_u[:,0,0],[0.,10.])


def test_wind1_rectangular_cells_give_equal_weight_to_equal_cell_offsets():
    # The target is at (1.0, 0.5) km.  Each station is exactly one cell away:
    # one 2 km column east and one 1 km row north.
    points=np.array([[3.,.5],[1.,1.5]])
    u=np.array([[2.,6.]])
    v=np.zeros_like(u)
    got_u,got_v=wind1_initial_guess(u,v,points,2,1,1,2.,1.)
    np.testing.assert_allclose(got_u,np.array([[[4.]]],dtype=np.float32))
    np.testing.assert_array_equal(got_v,0.)


@pytest.mark.parametrize(('dx_km','dy_km'),[
    (0.,1.),(1.,0.),(-1.,1.),(1.,-1.),(np.inf,1.),(1.,np.nan),
])
def test_wind1_rejects_invalid_rectangular_grid_spacing(dx_km,dy_km):
    with pytest.raises(ValueError,match='finite and positive'):
        wind1_initial_guess([[1.]],[[0.]],[[.5,.5]],1,1,1,dx_km,dy_km)


def test_wind1_uses_internal_editl_missing_threshold():
    points=np.array([[.5,.5],[1.5,.5]])
    u=np.array([[901.,2.],[999.,3.]])
    v=np.zeros_like(u)
    got_u,got_v=wind1_initial_guess(u,v,points,2,1,1,1.,1.)
    np.testing.assert_array_equal(got_u[:,0,0],np.array([2.,3.],dtype=np.float32))
    np.testing.assert_array_equal(got_v,0.)


def test_inter2_uses_strict_internal_editl_threshold():
    values=np.array([[900.,901.,2.]])
    points=np.array([[.5,.5],[.5,.5],[.5,.5]])
    guess=np.array([[[10.]]]); landuse=np.array([[10]])
    result=objective_analysis_component(values,points,guess,landuse,1.,1.,
        stations_per_level=3,background_weight=np.zeros((1,1,1)))
    np.testing.assert_array_equal(result,np.array([[[2.]]],dtype=np.float32))

def test_vertical_layer_average_integrates_linear_profile_exactly():
    values=np.array([[0.,20.,40.],[10.,30.,50.]])
    heights=np.array([0.,100.,200.])
    got=vertical_layer_average(values,heights,[0.,50.,150.,200.])
    np.testing.assert_allclose(got,[[5.,20.,35.],[15.,30.,45.]])
    np.testing.assert_allclose(vertical_interpolate(values,heights,[50.,150.]),[[10.,30.],[20.,40.]])


def test_smoothing_constant_wind_and_divergence():
    u=np.full((2,3,4),3.); v=np.full_like(u,-2.)
    us,vs=smooth_wind(u,v,[2,1]); np.testing.assert_allclose(us,u); np.testing.assert_allclose(vs,v)
    div,w=divergence_and_vertical_velocity(us,vs,[0.,20.,40.],1000.,1000.)
    np.testing.assert_allclose(div,0.); np.testing.assert_allclose(w,0.)

def test_divergence_linear_flow_and_obrien_top_zero():
    x=np.arange(5.); u=np.broadcast_to(x,(1,3,5)); v=np.zeros_like(u)
    div,w=divergence_and_vertical_velocity(u,v,[0.,20.],1.,1.,obrien=True)
    np.testing.assert_allclose(div[0,:,1:-1],1.); np.testing.assert_allclose(w[-1],0.)
    zeros=np.zeros((1,1,3)); boundaries=(np.array([[-2.]]),np.array([[2.]]),np.zeros((1,3)),np.zeros((1,3)))
    bounded,_=divergence_and_vertical_velocity(zeros,zeros,[0.,1.],1.,1.,boundaries=boundaries)
    np.testing.assert_allclose(bounded[0,0],[1.,0.,1.])

def test_minim_vector_sweeps_reduce_divergence():
    rng=np.random.default_rng(7); u=rng.normal(size=(2,8,9)); v=rng.normal(size=u.shape)
    boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    initial,w=divergence_and_vertical_velocity(u,v,[0.,20.,50.],1000.,1000.,obrien=True,boundaries=boundaries)
    adjusted_u,adjusted_v,final,iterations=minimize_divergence(
        u,v,w,[0.,20.,50.],1000.,1000.,boundaries=boundaries,maximum_iterations=50,limit=5.e-6)
    assert np.max(np.abs(final))<=5.e-6 and np.all(iterations>0)
    assert np.max(np.abs(final))<np.max(np.abs(initial))
    assert np.isfinite(adjusted_u).all() and np.isfinite(adjusted_v).all()



def test_heat_flux_and_similarity_day_night_water_branches():
    land=np.array([[20,20,55]]); common=np.ones((1,3))
    qh,qsw=heat_flux(np.array([[.8,-.2,.8]]),np.zeros((1,3)),np.full((1,3),290.),land,.15,1.,.15,0.)
    assert qsw[0,0]>0 and qsw[0,1]==0.
    assert qh[0,0]>0 and np.isclose(qh[0,1],-.1) and qh[0,2]==0.
    rho=np.full((1,3),1.2)
    ustar,length,qh2=surface_similarity(common,np.zeros_like(common),np.full((1,3),.25),10.,rho,np.full((1,3),290.),np.zeros((1,3)),qh,land)
    assert ustar[0,0]>=.05 and length[0,0]<0.; assert qh2[0,1]<0 and length[0,1]>0.; assert ustar[0,2]==0.
    assert psi_momentum_dyer(np.array([-1.])).item()>0

def test_similarity_stops_each_daytime_cell_at_fortran_convergence():
    shape=(1,2); land=np.full(shape,20); qh=np.array([[20.,60.]])
    ustar,length,_=surface_similarity(np.array([[2.,8.]]),np.zeros(shape),np.array([[.05,1.]]),10.,np.full(shape,1.2),np.full(shape,290.),np.zeros(shape),qh,land)
    assert np.all((ustar>=.05)&(ustar<5.)); assert np.all(length<0.)









def test_interp_uses_spatial_observation_and_prognostic_weights():
    weight=np.array([[[.25]]]); background=(1.-weight)/2.**2
    result=objective_analysis_component(np.array([[10.]]),np.array([[1.5,.5]]),
        np.full((1,1,1),2.),np.array([[20]]),1.,1.,rmax_surface_km=5.,
        stations_per_level=1,observation_weight=weight,background_weight=background)
    assert result.item()==pytest.approx((10.*.25+2.*.1875)/(.25+.1875))
    no_observation=objective_analysis_component(np.array([[9999.]]),np.array([[1.5,.5]]),
        np.full((1,1,1),2.),np.array([[20]]),1.,1.,rmax_surface_km=5.,
        stations_per_level=1,observation_weight=weight,background_weight=background)
    assert no_observation.item()==2.

def test_inter2_nintrp_and_lvary_nearest_fallback():
    values=np.array([[10.,20.]])
    points=np.array([[1.5,.5],[2.5,.5]])
    nearest=objective_analysis_component(values,points,np.zeros((1,1,1)),np.array([[20]]),1.,1.,
        rmax_surface_km=.5,surface_scale_km=10.,stations_per_level=1,vary_radius=True)
    assert nearest.item()==pytest.approx(10./(1.+.01))
    background=objective_analysis_component(values,points,np.full((1,1,1),7.),np.array([[20]]),1.,1.,
        rmax_surface_km=.5,surface_scale_km=10.,stations_per_level=1,vary_radius=False)
    assert background.item()==pytest.approx(7.)

def test_inter2_barrier_blocks_only_configured_lower_levels():
    values=np.array([[10.],[10.]])
    points=np.array([[2.5,.5]]); guess=np.full((2,1,1),2.)
    result=objective_analysis_component(values,points,guess,np.array([[20]]),1.,1.,
        rmax_surface_km=5.,rmax_upper_km=5.,surface_scale_km=1.,upper_scale_km=1.,
        stations_per_level=(1,1),vary_radius=False,barriers=((1.5,0.,1.5,1.),),barrier_top_level=1)
    assert result[0].item()==pytest.approx(2.)
    assert result[1].item()>2.

def test_inter2_finite_barrier_does_not_block_beyond_endpoint():
    result=objective_analysis_component(np.array([[10.]]),np.array([[2.5,.5]]),
        np.full((1,1,1),2.),np.array([[20]]),1.,1.,rmax_surface_km=5.,surface_scale_km=1.,
        stations_per_level=1,vary_radius=False,barriers=((1.5,2.,1.5,3.),),barrier_top_level=1)
    assert result.item()>2.

def test_inter2_lvary_does_not_fall_back_through_barrier():
    with pytest.raises(ValueError,match='unblocked'):
        objective_analysis_component(np.array([[10.]]),np.array([[2.5,.5]]),np.zeros((1,1,1)),
            np.array([[20]]),1.,1.,rmax_surface_km=.1,stations_per_level=1,vary_radius=True,
            barriers=((1.5,0.,1.5,1.),),barrier_top_level=1)
