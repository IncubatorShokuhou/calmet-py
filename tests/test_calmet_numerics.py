import numpy as np
import netCDF4 as nc
import pytest
import shutil
import subprocess
import textwrap
from types import SimpleNamespace
import pycalmet.calmet_solver as calmet_solver
from dataclasses import replace
from datetime import datetime,timedelta
from pycalmet.calmet_numerics import (_real4_libm,_real4_power,air_density,cloud_cover_mm5,cloud_cover_teixeira_850,cloud_fraction,cloud_liquid_water_profile,coare_water_station_profiles,convective_height_batchvarova_gryning,convective_height_maul_carson,convective_velocity_scale,coriolis_parameter,delta_temperature_flux,delta_temperature_water_station_profiles,froude_adjustment,hosker_roughness,inverse_distance_squared,lake_breeze_adjust,lapse_rate,mesopac_layer_average,mixing_height_batchvarova_gryning,mixing_height_maul_carson,neutral_water_parameters,objective_analysis_component,overwater_delta_temperature_state,pgt_stability,profile_temperature_layer,prognostic_temperature_3d,saturation_vapor_pressure_lowe,upwind_mixing_average,
    divergence_and_vertical_velocity,heat_flux,land_station_mixing_height,land_station_similarity_theory,land_station_surface_similarity,minimize_divergence,neutral_water_station_profiles,psi_momentum_dyer,similarity_wind_profiles,slope_flow,smooth_wind,solar_sine,surface_similarity,terrain_extrema,terrain_gradients,
    smooth_wind,terrain_adjust_surface_wind,topographic_vertical_velocity,vertical_interpolate,vertical_layer_average,water_mechanical_height,wind1_initial_guess,wind_speed_direction)
from pycalmet.calmet_output import (AuxiliaryCalmetWriter,CalmetNetCDFWriter,
                                    LegacyCalmetWriter,MesopacWriter)
from pycalmet.aux_data import AuxiliaryData
from pycalmet.legacy_io import FortranSequentialReader,FortranSequentialWriter,fixed_string
from pycalmet.calmet_solver import DiagnosticWindFields,HourFields,PrognosticMapper,Step1Fields,StepTransformContext,SurfaceFields,_coare_ending_gmt_hour,_domain_mean_real4,_fortran_nearest_indices,_fortran_nint_positive,_m3d_cloud_water_g_kg,_m3d_relative_humidity,_prognostic_local_timestamp,_prognostic_record_solar,_prognostic_wind_components,_rdmm5_above_ground_profiles,_rdmm5_sst_profile,_rdmm5_surface_density,_relative_humidity,_stull_temperature,_vertical_centers,build_diagnostic_wind,build_objective_wind,build_surface_fields,dataset_timestamp,interpolate_step1_fields,iter_hours,map_prognostic_cloud,observed_temperature_3d,solar_at_step_midpoint,solve_hour,target_latitude_longitude
from pycalmet.config import GridConfig
from pycalmet.coare import initial_warm_layer_state,surface_flux
from pycalmet.sea_data import GriddedSeaData


def test_similt_profiles_match_gnu_real4_bits():
    u,v=similarity_wind_profiles(
        [-3.,4.,-5.,2.5],[4.,-3.,2.,-6.],[10.,12.,8.,15.],
        [10.,40.,100.],[0.,20.,60.,140.],[.1,.25,.05,.4],
        [10000.,-45.,75.,-12.],[500.,90.,45.,130.],[30.,30.,-25.,-25.],
        ipsifcn=0)
    # The fourth station uses the legacy PSIUC branch, so compute it separately.
    u1,v1=similarity_wind_profiles([2.5],[-6.],[15.],[10.,40.,100.],
        [0.,20.,60.,140.],[.4],[-12.],[130.],[-25.],ipsifcn=1)
    u[:,3]=u1[:,0]; v[:,3]=v1[:,0]
    expected=np.array([
        [0xC03FFFF2,0x40800006,0xC06AF566,0x40ABD8B0,0xC0726BF5,0x40CF246E],
        [0x40770EAF,0xC0382EE2,0x40963BAD,0xC073773C,0x461C3C00,0x461C3C00],
        [0xC0AA9B3C,0x40051DB4,0x461C3C00,0x461C3C00,0x461C3C00,0x461C3C00],
        [0x400FF639,0xC0B09BF4,0x404B6200,0xC0DD3AB9,0x461C3C00,0x461C3C00],
    ],dtype=np.uint32).reshape(4,3,2)
    actual=np.stack((u.T,v.T),axis=-1).view(np.uint32)
    np.testing.assert_array_equal(actual,expected)


def test_elustr2_station_height_and_psi_branches_match_gnu_bits():
    inputs=dict(roughness_m=np.array([.1,.25,.05,.4],dtype=np.float32),
        height_m=10.,density=np.array([1.2,1.1,1.25,1.],dtype=np.float32),
        temperature_k=np.array([295.,300.,280.,305.],dtype=np.float32),
        cloud_fraction_field=np.array([.2,.7,.9,.1],dtype=np.float32),
        sensible_heat_flux=np.array([120.,80.,-.1,-.1],dtype=np.float32),
        landuse=np.zeros(4,dtype=np.int16),
        wind_measurement_height_m=np.array([10.,12.,8.,15.],dtype=np.float32))
    u=np.array([-3.,4.,-.2,7.],dtype=np.float32)
    v=np.array([4.,-3.,.1,1.],dtype=np.float32)
    actual=[]
    for index,ipsifcn in enumerate((0,1,0,1)):
        state=land_station_surface_similarity(u[index:index+1],v[index:index+1],
            inputs['roughness_m'][index:index+1],
            inputs['wind_measurement_height_m'][index:index+1],10.,
            inputs['density'][index:index+1],inputs['temperature_k'][index:index+1],
            inputs['cloud_fraction_field'][index:index+1],
            inputs['sensible_heat_flux'][index:index+1],ipsifcn=ipsifcn)
        actual.append([state[0][0],state[1][0],state[2][0]])
    expected=np.array([
        [0x3EEFF3A6,0xC29A1D99,0x42F00000],
        [0x3F11EC36,0xC341FA9A,0x42A00000],
        [0x3D4CCCCD,0x439BA358,0xBD121FD4],
        [0x3F5DB734,0x447A711F,0xC248D02D],
    ],dtype=np.uint32)
    np.testing.assert_array_equal(np.asarray(actual,dtype=np.float32).view(np.uint32),expected)


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
        np.testing.assert_array_equal(actual.view(np.uint32),expected[0].view(np.uint32))


def test_land_stheor_composes_valid_stations_and_preserves_early_returns():
    u=np.array([-3.,9999.,0.],dtype=np.float32)
    v=np.array([4.,9999.,0.],dtype=np.float32)
    profile=np.broadcast_to(
        np.array([295.,293.,289.,281.],dtype=np.float32)[:,None],(4,3))
    heights=np.broadcast_to(
        np.array([0.,100.,400.,1200.],dtype=np.float32)[:,None],(4,3))
    common=dict(roughness_m=np.array([.1,.2,.3],dtype=np.float32),
        anemometer_height_m=np.array([10.,10.,10.],dtype=np.float32),
        first_level_height_m=10.,density=np.full(3,1.2,dtype=np.float32),
        temperature_k=np.full(3,295.,dtype=np.float32),
        cloud_fraction=np.full(3,.2,dtype=np.float32),
        sensible_heat_flux=np.full(3,120.,dtype=np.float32),
        temperature_profile_k=profile,profile_height_m=heights,
        previous_convective_height_m=np.array([50.,60.,70.],dtype=np.float32),
        temperature_jump_k=np.array([0.,.1,.2],dtype=np.float32),
        coriolis=np.full(3,1.e-4,dtype=np.float32),
        z_centers_m=np.array([10.,40.,100.],dtype=np.float32),
        z_faces_m=np.array([0.,20.,60.,140.],dtype=np.float32),
        station_latitude=np.array([30.,30.,30.],dtype=np.float32))
    result=land_station_similarity_theory(u,v,**common)
    ustar,length,heat=land_station_surface_similarity(
        u[:1],v[:1],common['roughness_m'][:1],common['anemometer_height_m'][:1],10.,
        common['density'][:1],common['temperature_k'][:1],common['cloud_fraction'][:1],
        common['sensible_heat_flux'][:1])
    mixing=land_station_mixing_height(profile[:,:1],heights[:,:1],
        common['previous_convective_height_m'][:1],common['temperature_jump_k'][:1],
        ustar,length,heat,common['density'][:1],common['coriolis'][:1])
    expected_uv=similarity_wind_profiles(u[:1],v[:1],common['anemometer_height_m'][:1],
        common['z_centers_m'],common['z_faces_m'],common['roughness_m'][:1],length,
        mixing[0],common['station_latitude'][:1])
    for actual,expected in zip(result[:2],expected_uv,strict=True):
        np.testing.assert_array_equal(actual[:,0].view(np.uint32),expected[:,0].view(np.uint32))
        np.testing.assert_array_equal(actual[:,1:],np.array([[9999.,0.],[9999.,9999.],
                                                              [9999.,9999.]],dtype=np.float32))
    for actual,expected in zip(result[2:],(*[ustar,length,heat],*mixing),strict=True):
        np.testing.assert_array_equal(actual[:1].view(np.uint32),expected.view(np.uint32))
        assert np.isnan(actual[1:]).all()


def test_cloud_liquid_water_profile_matches_z2face_prfvar_layers():
    cloud=np.array([.002,.01,.02,.004],dtype=np.float32)[:,None,None]
    height=np.array([100.,300.,700.,1300.],dtype=np.float32)[:,None,None]
    mapped,average,bottom,top=cloud_liquid_water_profile(
        cloud,height,[0.,400.,800.])
    np.testing.assert_allclose(mapped[:,0,0],[.006,.0175],rtol=0.,atol=1.e-8)
    np.testing.assert_array_equal(average,np.array([[.02]],dtype=np.float32))
    np.testing.assert_array_equal(bottom,[[500.]])
    np.testing.assert_array_equal(top,[[1000.]])


def test_m3d_cloud_water_recovers_single_rounded_g_per_kg_value():
    canonical=np.float32(.009)*np.float32(.001)
    result=_m3d_cloud_water_g_kg(canonical)
    assert result.view(np.uint32)==np.float32(.009).view(np.uint32)


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
                                      [201807700,0,201807700,3600])
        assert np.frombuffer(reader.read_record(),dtype='<i4').item()==3
        record=reader.read_record(); assert record[:12]==b'CLDMRUP     '
        expected=np.array([-(np.float32(1.)+np.float32(.0001)),1.,
                           -(np.float32(2.)+np.float32(.0001))],dtype=np.float32)
        np.testing.assert_array_equal(np.frombuffer(record[12:],dtype='<f4'),expected)
    with AuxiliaryData(path) as auxiliary:
        assert (auxiliary.header.nx,auxiliary.header.ny,auxiliary.header.nz)==(2,2,2)
        result=auxiliary.read_hour()
        assert (result.begin_code,result.begin_seconds,result.end_code,
                result.end_seconds)==(201807700,0,201807700,3600)
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

def test_air_density_uses_nearest_station():
    got=air_density([1000.,900.],np.array([[300.,300.]]),np.array([[0,1]]))
    expected=np.asarray(np.float32(.3484321)*np.array([[1000.,900.]],dtype=np.float32)
                        /np.float32(300.),dtype=np.float32)
    np.testing.assert_array_equal(got,expected)


def test_coare_ending_hour_uses_west_positive_timezone_and_ignores_seconds():
    assert _coare_ending_gmt_hour(datetime(2018,1,1,0,15,30),8)==8
    assert _coare_ending_gmt_hour(datetime(2018,1,1,23,59,59),-8)==15


def test_domain_temperature_mean_uses_cgamma2_real4_ij_order():
    field=np.array([[1.e8,-1.e8],[1.,1.]],dtype=np.float32)
    assert _domain_mean_real4(field)==np.float32(.25)


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

def test_lake_breeze_preserves_interb_mixed_precision_accumulation():
    u=np.zeros((1,4,1)); v=np.zeros_like(u)
    region=(1.,1.,1.,4.,0.,1500.,1000.,1500.,(11,22,33))
    adjusted_u,_=lake_breeze_adjust(u,v,[1.234567,20.125,-3.75],[1.,2.,3.],
        [[500.,100.],[500.,1300.],[500.,2600.]],[11,22,33],(region,),1000.,1000.)
    # GNU INTERB oracle: DOUBLE PRECISION DPR/WT/BOT, implicit REAL TOP/GDAT.
    expected=np.array([4.765492,19.009848,-3.5770488,-.23877653],dtype=np.float32)
    np.testing.assert_array_equal(adjusted_u[0,:,0],expected)


def test_observed_temperature_3d_applies_temp3d_convective_layer_rules():
    surface=np.array([[300.]])
    faces=np.array([[[300.]],[[299.]],[[297.]]])
    got=observed_temperature_3d(surface,faces,np.array([[150.]]),[0.,100.,300.])
    assert got.dtype==np.float32
    middle=300.-.0098*150.
    expected=.5*((300.-.0098*100.+middle)*50.+(middle+297.)*150.)/200.
    np.testing.assert_allclose(got[:,0,0],[300.,expected])


def test_observed_temperature_3d_uses_sea_gradients_and_mixing_height():
    surface=np.array([[290.]])
    faces=np.array([[[290.]],[[289.]],[[287.]]])
    got=observed_temperature_3d(surface,faces,np.array([[50.]]),[0.,100.,300.],
        landuse=np.array([[55]]),water_range=(55,55),mixing_height=np.array([[150.]]),
        water_gradient_below=np.array([[-.01]]),water_gradient_above=np.array([[-.005]]))
    middle=290.-.01*150.
    expected=.5*((290.-1.+middle)*50.+(middle+(middle-.005*150.))*150.)/200.
    np.testing.assert_allclose(got[:,0,0],[290.,expected])

def test_microi_coriolis_constant_latitude_and_equatorial_floor():
    latitude=np.array([[0.,30.,-60.,89.9]])
    np.testing.assert_array_equal(coriolis_parameter(latitude,1.e-4),
                                  np.full(latitude.shape,np.float32(1.e-4)))
    expected=np.array([[2.5e-5,7.2722025e-5,1.2595825e-4,1.4544386e-4]],dtype=np.float32)
    np.testing.assert_array_equal(coriolis_parameter(latitude,999.),expected)

def test_inverse_distance_minimum_and_cloud_zero_filter():
    x=np.array([[0.,1.,2.,3.]]); y=np.zeros_like(x); values=np.array([[10.,0.,30.,40.]])
    assert inverse_distance_squared(values,0.,0.,x,y)>9.9
    assert inverse_distance_squared(values,0.,0.,x,y,ignore_zero=True)>9.9

def test_wind_direction_is_meteorological_from_direction():
    speed,direction=wind_speed_direction(np.array([0.,1.,0.,-1.]),np.array([0.,0.,1.,0.]))
    assert speed.dtype==np.float32 and direction.dtype==np.float32
    np.testing.assert_array_equal(speed,[0.,1.,1.,1.]); np.testing.assert_array_equal(direction,[0.,270.,180.,90.])


def test_rtheta_real4_quadrants_and_calm_threshold():
    u=np.array([1.234567,-2.5,-.75,3.125,1.e-6],dtype=np.float32)
    v=np.array([-4.25,1.75,-3.5,2.625,-2.e-6],dtype=np.float32)
    speed,direction=wind_speed_direction(u,v)
    expected_speed=np.array([0x408d9f2f,0x40434e0d,0x406515cc,0x4082993f,0x36160f5b],dtype=np.uint32)
    expected_direction=np.array([0x43abe6ac,0x42f9fbea,0x41418420,0x4365f841,0x00000000],dtype=np.uint32)
    np.testing.assert_array_equal(speed.view(np.uint32),expected_speed)
    np.testing.assert_array_equal(direction.view(np.uint32),expected_direction)

def test_cloud_limits_and_solar_shape():
    fraction=cloud_fraction([99.,100.])
    assert fraction.dtype==np.float32
    np.testing.assert_array_equal(fraction,[1.,1.])
    result=solar_sine(172,8,np.array([[30.]]),np.array([[120.]]))
    assert result.shape==(26,1,1); assert result.max()>.8; assert result.min()<0

def test_cloud4_pressure_bands_and_ceiling():
    pressure=np.array([1000.,950.,750.,400.])[:,None,None]
    rh=np.array([30.,90.,80.,70.])[:,None,None]
    height=np.array([10.,100.,2000.,7000.])[:,None,None]
    cover,ceiling=cloud_cover_mm5(rh,pressure,height)
    assert cover.dtype==np.float32 and ceiling.dtype==np.float32
    expected=np.float32(np.float32(4.)*np.float32(90.)/np.float32(100.)-np.float32(3.))
    np.testing.assert_array_equal(cover,[[expected]])
    np.testing.assert_array_equal(ceiling,[[np.float32(100.)]])

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

def test_rdmm5_subhour_interpolation_preserves_special_field_rules():
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    left=Step1Fields(level,level,level*272.,np.ones(shape)*280.,np.ones(shape)*1000.,np.ones(shape),level*40.,np.ones(shape),np.ones(shape),np.ones(shape),np.ones(shape)*.2,np.zeros(shape),profile,np.zeros_like(profile),np.ones(shape)*40.,np.zeros(shape),np.zeros(shape,dtype=int),cloud_relative_humidity_850=np.ones(shape)*40.,integrated_cloud_water=np.zeros(shape),cloud_liquid_water_3d=np.zeros_like(level),cloud_liquid_water_above=np.zeros(shape),cloud_liquid_water_above_bottom=np.zeros(shape),cloud_liquid_water_above_top=np.zeros(shape))
    right=replace(left,u=level*5.,temperature=level*276.,cloud_fraction=np.ones(shape)*.9,
        ceiling_height=np.ones(shape)*2000.,profile_temperature=profile*9.,surface_relative_humidity=np.ones(shape)*43.,
        precipitation_rate=np.ones(shape)*2.,cloud_relative_humidity_850=np.ones(shape)*80.,
        integrated_cloud_water=np.ones(shape)*4.e-6,
        cloud_liquid_water_3d=np.ones_like(level)*.04,
        cloud_liquid_water_above=np.ones(shape)*.08,
        cloud_liquid_water_above_bottom=np.ones(shape)*1000.,
        cloud_liquid_water_above_top=np.ones(shape)*2000.)
    got=interpolate_step1_fields(left,right,.25,cloud_method=3)
    np.testing.assert_allclose(got.u,2.); np.testing.assert_allclose(got.profile_temperature,1.)
    # MIXDT2 uses the latest actual sounding at the beginning of the
    # interval; unlike ordinary prognostic fields it is not interpolated.
    assert not np.shares_memory(got.profile_temperature,left.profile_temperature)
    np.testing.assert_allclose(got.cloud_fraction,cloud_fraction([[50.]]))
    np.testing.assert_allclose(got.ceiling_height,2000.)
    np.testing.assert_array_equal(got.surface_relative_humidity,40)
    np.testing.assert_array_equal(got.precipitation_code,20)
    np.testing.assert_allclose(got.cloud_liquid_water_3d,.01)
    np.testing.assert_allclose(got.cloud_liquid_water_above,.02)
    np.testing.assert_allclose(got.cloud_liquid_water_above_bottom,250.)
    np.testing.assert_allclose(got.cloud_liquid_water_above_top,500.)
    assert got.u.dtype==np.float32 and got.profile_height.dtype==np.float32

    # RDMM5 stores both endpoints and DTINC as REAL*4.  Adding one to 2**24
    # distinguishes its per-operation rounding from one final float32 cast.
    edge_left=replace(left,u=np.full_like(level,16777216.,dtype=np.float64))
    edge_right=replace(right,u=np.full_like(level,16777218.,dtype=np.float64))
    edge=interpolate_step1_fields(edge_left,edge_right,.5,cloud_method=4)
    np.testing.assert_array_equal(
        edge.u,np.full_like(level,np.float32(16777216.),dtype=np.float32))
    endpoint_left=replace(left,u=np.full_like(level,np.float32(-7.116808)))
    endpoint_right=replace(right,u=np.full_like(level,np.float32(8.972989)))
    endpoint=interpolate_step1_fields(endpoint_left,endpoint_right,1.,cloud_method=4)
    # Ordinary GNU -O2 endpoints fold to NEW.  The final source record uses
    # a distinct literal-arithmetic RDMM5 path selected by the iterator.
    assert endpoint.u[0,0,0].view(np.uint32)==np.uint32(0x410f915d)
    np.testing.assert_array_equal(endpoint.profile_temperature,
                                  endpoint_left.profile_temperature)
    final_endpoint=interpolate_step1_fields(
        endpoint_left,endpoint_right,1.,cloud_method=4,literal_endpoint=True)
    assert final_endpoint.u[0,0,0].view(np.uint32)==np.uint32(0x410f915e)

def test_calwrf_getrh1_legacy_saturation_definition():
    temperature=np.array([293.15]); pressure=np.array([1000.]); q=np.array([.01])
    expected=100.*q/(.622*6.112/pressure*np.exp(17.67*20./263.5))
    got=_relative_humidity(q,pressure,temperature)
    assert got.dtype==np.float32
    np.testing.assert_allclose(got,expected,rtol=2.e-6)

def test_rdmm5_uses_stored_integer_rh_instead_of_reconstructing_it():
    class Dataset:
        variables={'relative_humidity':np.array([[[[100.]]]])}
        def __getitem__(self,name): return self.variables[name]
    got=_m3d_relative_humidity(
        Dataset(),0,np.array([[[.0099]]]),np.array([[[900.]]]),
        np.array([[[280.]]]))
    np.testing.assert_array_equal(got,100.)


def test_rdmm5_density_uses_first_nonnegative_level_per_source_column():
    height=np.array([[[-20.,10.]],[[5.,40.]],[[100.,100.]]],dtype=np.float32)
    pressure=np.array([[[1000.,900.]],[[800.,700.]],[[600.,500.]]],dtype=np.float32)
    temperature=np.array([[[300.,290.]],[[280.,270.]],[[260.,250.]]],dtype=np.float32)
    got=_rdmm5_surface_density(pressure,temperature,height)
    expected=(np.float32(.3484321)*np.array([[800.,900.]],dtype=np.float32)/
              np.array([[280.,290.]],dtype=np.float32))
    np.testing.assert_array_equal(got,np.asarray(expected,dtype=np.float32))
    with pytest.raises(ValueError,match='no level at or above ground'):
        _rdmm5_surface_density(pressure,temperature,np.full_like(height,-1.))


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


def test_rdmm5_nint_rounds_real4_input_not_python_double():
    got=_fortran_nint_positive(np.array([.49999999,1.49999994],dtype=np.float64))
    assert got.dtype==np.float32
    np.testing.assert_array_equal(got,[1.,1.])

def test_stull_nighttime_cooling_and_warm_front_paths():
    cooled=_stull_temperature(np.array([289.]),np.array([40.]),np.array([290.]),np.array([100.]),np.array([291.]),np.array([291.]),0.)
    assert cooled.dtype==np.float32
    assert 276.<=cooled.item()<289.
    warmed=_stull_temperature(np.array([292.]),np.array([40.]),np.array([293.]),np.array([100.]),np.array([291.]),np.array([291.]),0.)
    np.testing.assert_allclose(warmed,292.)

def test_vectorized_smooth_matches_fortran_ordered_stencil():
    rng=np.random.default_rng(42); u=rng.normal(size=(2,4,5)); v=rng.normal(size=(2,4,5))
    expected_u=np.asarray(u,dtype=np.float32); expected_v=np.asarray(v,dtype=np.float32)
    ub_left=expected_u[:,:,0].copy(); ub_right=expected_u[:,:,-1].copy()
    vb_bottom=expected_v[:,0,:].copy(); vb_top=expected_v[:,-1,:].copy()
    for k,count in enumerate((2,3)):
        for _ in range(count):
            for j in range(4):
                for i in range(5):
                    uim1=ub_left[k,j] if i==0 else expected_u[k,j,i-1]
                    uip1=ub_right[k,j] if i==4 else expected_u[k,j,i+1]
                    ujm1=expected_u[k,j,i] if j==0 else expected_u[k,j-1,i]
                    ujp1=expected_u[k,j,i] if j==3 else expected_u[k,j+1,i]
                    vim1=expected_v[k,j,i] if i==0 else expected_v[k,j,i-1]
                    vip1=expected_v[k,j,i] if i==4 else expected_v[k,j,i+1]
                    vjm1=vb_bottom[k,i] if j==0 else expected_v[k,j-1,i]
                    vjp1=vb_top[k,i] if j==3 else expected_v[k,j+1,i]
                    expected_u[k,j,i]=np.float32(.5)*expected_u[k,j,i]+np.float32(.125)*(uip1+uim1+ujp1+ujm1)
                    expected_v[k,j,i]=np.float32(.5)*expected_v[k,j,i]+np.float32(.125)*(vip1+vim1+vjp1+vjm1)
    actual_u,actual_v=smooth_wind(u,v,(2,3))
    assert actual_u.dtype==np.float32 and actual_v.dtype==np.float32
    np.testing.assert_array_equal(actual_u,expected_u)
    np.testing.assert_array_equal(actual_v,expected_v)

def test_cgamma2_interpolation_convective_override_and_average():
    temp=np.array([[[290.,290.]],[[289.,288.]],[[287.,284.]]])
    zi=np.array([[0.,250.]])
    got=lapse_rate(temp,zi,[10.,120.,230.],200.)
    assert got.dtype==np.float32
    ratio=np.float32(np.float32(230.-200.)/np.float32(230.-120.))
    upper=np.float32(np.float32(287.)-np.float32(np.float32(287.-289.)*ratio))
    expected_first=np.float32(np.float32(upper-np.float32(290.))/np.float32(190.))
    np.testing.assert_array_equal(got,np.array([[expected_first,-.0098]],dtype=np.float32))
    averaged=lapse_rate(temp,zi,[10.,120.,230.],200.,spatial=False)
    np.testing.assert_array_equal(averaged,np.full_like(got,
        np.float32(np.float32(got[0,0]+got[0,1])/np.float32(2.))))


def test_cgamma2_domain_average_uses_fortran_i_j_accumulation():
    temp=np.array([[[290.125,291.25,289.75],[288.5,292.125,290.875]],
                   [[289.5,290.75,288.625],[287.75,291.5,290.]],
                   [[287.,288.25,286.5],[285.125,289.,287.75]]],dtype=np.float32)
    zi=np.array([[50.,250.,75.],[300.,125.,20.]],dtype=np.float32)
    local=lapse_rate(temp,zi,[12.5,117.25,241.75],200.)
    expected=np.float32(0.)
    for i in range(3):
        for j in range(2): expected=np.float32(expected+local[j,i])
    expected=np.float32(expected/np.float32(6.))
    averaged=lapse_rate(temp,zi,[12.5,117.25,241.75],200.,spatial=False)
    np.testing.assert_array_equal(averaged,np.full((2,3),expected,dtype=np.float32))

def test_froude_adjustment_turns_upslope_wind_tangent_only_when_blocked():
    terrain=np.array([[0.,100.,200.]])
    maximum=np.full_like(terrain,1000.)
    u=np.ones((1,1,3)); v=np.zeros_like(u)
    adjusted_u,adjusted_v=froude_adjustment(u,v,terrain,maximum,np.zeros_like(terrain),np.full_like(terrain,290.),[10.],1.,1000.,1000.)
    assert adjusted_u.dtype==np.float32 and adjusted_v.dtype==np.float32
    # FRADJ's rounded DPR/RPD pair leaves a small non-zero tangent component.
    np.testing.assert_allclose(adjusted_u,5.2603082e-5,rtol=0.,atol=1.e-11)
    np.testing.assert_allclose(np.abs(adjusted_v),1.)


def test_fradj_real4_multilevel_regression():
    terrain=np.array([[0.,30.,100.],[15.,65.,160.]],dtype=np.float32)
    maximum=np.array([[250.,260.,300.],[240.,280.,350.]],dtype=np.float32)
    u=np.array([[[2.,-1.,3.],[.5,4.,-2.]],[[6.,2.,-3.],[1.,-4.,5.]]],dtype=np.float32)
    v=np.array([[[-.5,2.,1.],[3.,-.25,2.5]],[[1.,-2.,4.],[-3.,2.,.5]]],dtype=np.float32)
    got_u,got_v=froude_adjustment(u,v,terrain,maximum,
        np.array([[-.005,.001,.004],[.008,-.002,.006]],dtype=np.float32),
        np.array([[288.,291.,294.],[287.,290.,293.]],dtype=np.float32),
        [10.,70.],5.,713.,887.)
    # Bit patterns from the source FRADJ compiled with the repository's GNU
    # flags; in particular scalar ATAN/COS/SIN do not match NumPy's SIMD ufuncs.
    expected_u=np.array([0x3f44c8e3,0xbf800000,0x3fe5a24b,0xbf367bc6,
                         0x3f435f68,0xc0000000,0x4011282d,0x3f440f91,
                         0xc0400000,0x3f3dbcbf,0xc0800000,0x401190bc],dtype=np.uint32)
    expected_v=np.array([0xbff4d94b,0x40000000,0xc026aa1e,0x403d3a08,
                         0xc07bce61,0x40200000,0xc0b49c71,0xc02e41cc,
                         0x40800000,0xc044bfa0,0x40000000,0xc08f620b],dtype=np.uint32)
    np.testing.assert_array_equal(got_u.view(np.uint32).ravel(),expected_u)
    np.testing.assert_array_equal(got_v.view(np.uint32).ravel(),expected_v)


def test_fradj_half_power_uses_powf_instead_of_sqrtf():
    squared=np.float32(18.142693)
    powered=np.float32(_real4_power(squared,np.float32(.5)))
    rooted=np.float32(_real4_libm('sqrtf',squared))
    # GNU FRADJ at (K=2,I=29,J=197), before tangent projection.
    assert powered.view(np.uint32)==np.uint32(0x40884d34)
    assert rooted.view(np.uint32)==np.uint32(0x40884d33)

def test_real4_atan_uses_runtime_float_intrinsic():
    value=np.array(0x40fb9741,dtype=np.uint32).view(np.float32)
    result=np.asarray(_real4_libm('atanf',value),dtype=np.float32)
    assert result.view(np.uint32)==0x3fb8de54

def test_real4_power_uses_runtime_powf():
    result=np.asarray(_real4_power(np.float32(82.03068),
                                   np.float32(1.)/np.float32(3.)),dtype=np.float32)
    assert result.view(np.uint32)==0x408b0a6f

def test_rdmm5_wraps_negative_grid_direction_before_component_split():
    u,v=_prognostic_wind_components(
        np.array([4.5],dtype=np.float32),np.array([1.],dtype=np.float32),
        np.array([1.036069393157959],dtype=np.float32))
    # GNU RDMM5 evaluates SIN at 359.963928 degrees, not the mathematically
    # equivalent negative angle; rounded DEG2RAD makes the distinction visible.
    np.testing.assert_array_equal(u.view(np.uint32),[0x3b38decb])
    np.testing.assert_array_equal(v.view(np.uint32),[0xc08ffffe])

def test_rdmm5_component_split_uses_runtime_real_trigonometry():
    u,v=_prognostic_wind_components(
        np.array([10.5],dtype=np.float32),
        np.array([1.980198],dtype=np.float32),
        np.array([.12345],dtype=np.float32))
    assert u[0].view(np.uint32)==np.uint32(0xbeae2fab)
    assert v[0].view(np.uint32)==np.uint32(0xc127e96b)

def test_mixdt2_profile_lapse_and_maul_carson_day_night():
    z=np.array([0.,100.,300.,1000.])[:,None,None]
    t=np.array([290.,289.,287.,280.])[:,None,None]
    tht,thtp,gamma=profile_temperature_layer(t,z,np.array([[50.]]),200.)
    np.testing.assert_allclose(tht,289.5); np.testing.assert_allclose(thtp,287.5); np.testing.assert_allclose(gamma,.001)
    profiles=np.broadcast_to(t,(4,1,2)); heights=np.broadcast_to(z,(4,1,2))
    total,convective,jump=mixing_height_maul_carson(profiles,heights,np.zeros((1,2)),np.zeros((1,2)),np.full((1,2),.2),np.array([[-100.,100.]]),np.array([[100.,-10.]]),np.full((1,2),1.2),np.full((1,2),1.e-4))
    assert convective[0,0]>0. and jump[0,0]>0. and total[0,0]>=convective[0,0]
    assert convective[0,1]==0. and jump[0,1]==0. and 50.<=total[0,1]<=2500.


def test_mixht_real_power_and_square_root_intrinsics_match_gnu():
    f=np.float32
    gamma=f(.00137); tave=f(287.35); coriolis=f(8.7e-5)
    ustar=f(.17776217); length=f(81.25)
    bvf=f(_real4_power(f(f(9.81)*gamma/tave),f(.25)))
    cmech=f(f(1.41)/_real4_libm('sqrtf',coriolis))
    mechanical=f(f(cmech*ustar)/bvf)
    stable1=f(f(2400.)*_real4_power(ustar,f(1.5)))
    stable2=f(f(.4)*_real4_libm('sqrtf',f(f(ustar*length)/coriolis)))
    convective=f(_real4_libm('sqrtf',f(456789.125))+
                 f(_real4_libm('sqrtf',f(.0234567))/gamma))
    # diagnostics/code/mixht_real_intrinsics_reference.f90
    assert [value.view(np.uint32) for value in
            (bvf,cmech,mechanical,stable1,stable2,convective)]==[
                np.uint32(0x3da95d87),np.uint32(0x43172af9),
                np.uint32(0x43a2786c),np.uint32(0x4333dff1),
                np.uint32(0x4322faa2),np.uint32(0x4444e9e0)]

def test_batchvarova_gryning_is_vectorized_and_preserves_temperature_jump():
    shape=(2,3)
    heights=np.broadcast_to(np.array([0.,100.,400.,1200.])[:,None,None],(4,*shape))
    temperatures=np.broadcast_to(np.array([291.,290.,287.,279.])[:,None,None],(4,*shape))
    old=np.array([[0.,80.,150.],[20.,100.,200.]])
    jump=np.arange(6.,dtype=float).reshape(shape)/10.
    length=np.array([[-100.,-200.,100.],[-600.,-50.,80.]])
    heat=np.array([[120.,180.,90.],[200.,0.,-20.]])
    total,convective,new_jump=mixing_height_batchvarova_gryning(
        temperatures,heights,old,jump,np.full(shape,.25),length,heat,
        np.full(shape,1.2),np.full(shape,1.e-4),surface_temperature_k=np.full(shape,290.))
    assert total.shape==shape and np.isfinite(total).all()
    assert np.all((total>=50.)&(total<=2500.))
    assert np.all(convective[length<0.]>=0.)
    assert np.all(convective[length>0.]==0.)
    np.testing.assert_array_equal(new_jump,jump.astype(np.float32))
    old_for_lapse=np.maximum(old,50.)
    _,_,gamma=profile_temperature_layer(temperatures,heights,old_for_lapse,200.)
    direct=convective_height_batchvarova_gryning(
        old_for_lapse,heat/(1.2*996.),gamma,np.full(shape,.25),length,
        np.full(shape,290.))
    np.testing.assert_allclose(direct,convective)

def test_mixhbg_matches_real4_false_position_and_maximum_overwrite():
    result=convective_height_batchvarova_gryning(
        np.array([100.]),np.array([.12]),np.array([.002]),np.array([.25]),
        np.array([-100.]),np.array([290.]),maximum_height_m=100.,
        threshold_flux_per_m=0.)
    # GNU MIXHBG oracle. Its final MAX(H,ZIMN) overwrites MIN(H,ZIMX).
    np.testing.assert_array_equal(result,np.array([864.1491],dtype=np.float32))

def test_neutral_water_parameters_match_water_real4_oracle():
    result=neutral_water_parameters(np.array([3.,0.,12.3]),np.array([4.,0.,-4.5]),
        np.array([1.e-4,2.5e-5,8.7e-5]))
    np.testing.assert_array_equal(result[0],np.array(
        [1.118034e-4,2.e-6,1.2416145e-3],dtype=np.float32))
    np.testing.assert_array_equal(result[1],np.array(
        [.17541777,1.e-9,.58249646],dtype=np.float32))
    np.testing.assert_array_equal(result[2],np.full(3,np.float32(9.9e9)))
    np.testing.assert_array_equal(result[3],np.zeros(3,dtype=np.float32))
    np.testing.assert_array_equal(result[4],np.array(
        [280.66843,50.,1071.2578],dtype=np.float32))
    np.testing.assert_array_equal(result[5],np.zeros(3,dtype=np.float32))


def test_neutral_water_ustar_uses_runtime_logf():
    result=neutral_water_parameters(
        np.array([.040097892],dtype=np.float32),
        np.array([5.543456],dtype=np.float32),
        np.array([1.e-4],dtype=np.float32))
    # GNU WATER cell (I=506,J=3); NumPy log gives 0x3e4bc4ae.
    assert result[1][0].view(np.uint32)==np.uint32(0x3e4bc4b0)


def test_water2_neutral_station_rescales_to_ten_metres_and_caps_height():
    result=neutral_water_station_profiles(
        np.array([3.,9999.,0.],dtype=np.float32),
        np.array([4.,9999.,0.],dtype=np.float32),
        np.array([20.,10.,10.],dtype=np.float32),
        np.array([180.,200.,200.],dtype=np.float32),
        np.full(3,1.e-4,dtype=np.float32),
        np.array([10.,40.,100.],dtype=np.float32),
        np.array([0.,20.,60.,140.],dtype=np.float32),
        np.full(3,30.,dtype=np.float32))
    speed=np.float32(5.)
    first_z0=hosker_roughness(np.array([speed],dtype=np.float32))[0]
    speed10=np.float32(speed*_real4_libm('logf',np.float32(10.)/first_z0)/
                       _real4_libm('logf',np.float32(20.)/first_z0))
    z0=np.float32(np.float32(2.e-6)*_real4_power(speed10,np.float32(2.5)))
    ustar=np.float32(np.float32(.4)*speed10/
                     _real4_libm('logf',np.float32(10.)/z0))
    expected=similarity_wind_profiles(
        np.array([3.],dtype=np.float32),np.array([4.],dtype=np.float32),
        np.array([20.],dtype=np.float32),np.array([10.,40.,100.],dtype=np.float32),
        np.array([0.,20.,60.,140.],dtype=np.float32),np.array([z0]),
        np.array([np.float32(9.9e9)]),np.array([180.],dtype=np.float32),
        np.array([30.],dtype=np.float32))
    inactive=np.array([[9999.,0.],[9999.,9999.],[9999.,9999.]],dtype=np.float32)
    for actual,wanted in zip(result[:2],expected,strict=True):
        np.testing.assert_array_equal(actual[:,0].view(np.uint32),wanted[:,0].view(np.uint32))
        np.testing.assert_array_equal(actual[:,1:],inactive)
    np.testing.assert_array_equal(result[2][0].view(np.uint32),z0.view(np.uint32))
    np.testing.assert_array_equal(result[3][0].view(np.uint32),ustar.view(np.uint32))
    assert result[4][0]==np.float32(9.9e9) and result[5][0]==np.float32(180.)
    assert all(np.isnan(value[1:]).all() for value in result[2:])


def test_water2_deltat_station_composes_flux_and_neutral_fallback():
    u=np.array([3.,3.],dtype=np.float32); v=np.array([4.,4.],dtype=np.float32)
    common=dict(anemometer_height_m=np.array([10.,10.],dtype=np.float32),
        air_temperature_k=np.array([290.,290.],dtype=np.float32),
        air_minus_sea_temperature_k=np.array([-2.,9999.],dtype=np.float32),
        relative_humidity_percent=np.array([70.,70.],dtype=np.float32),
        observed_mixing_height_m=np.array([9999.,9999.],dtype=np.float32),
        current_mixing_height_m=np.array([300.,300.],dtype=np.float32),
        coriolis=np.full(2,1.e-4,dtype=np.float32),
        potential_lapse_k_m=np.full(2,.004,dtype=np.float32),
        previous_temperature_jump_k=np.array([.2,.2],dtype=np.float32),
        z_centers_m=np.array([10.,40.,100.],dtype=np.float32),
        z_faces_m=np.array([0.,20.,60.,140.],dtype=np.float32),
        station_latitude=np.full(2,30.,dtype=np.float32))
    result=delta_temperature_water_station_profiles(u,v,**common,mixing_method=-1)
    neutral=neutral_water_station_profiles(u[1:],v[1:],
        common['anemometer_height_m'][1:],common['current_mixing_height_m'][1:],
        common['coriolis'][1:],common['z_centers_m'],common['z_faces_m'],
        common['station_latitude'][1:])
    for actual,expected in zip(result[:2],neutral[:2],strict=True):
        np.testing.assert_array_equal(actual[:,1].view(np.uint32),expected[:,0].view(np.uint32))
    theta,difference,difference10,theta10=overwater_delta_temperature_state(
        np.array([290.],dtype=np.float32),np.array([292.],dtype=np.float32),
        np.array([70.],dtype=np.float32),np.array([10.],dtype=np.float32))
    z0=hosker_roughness(np.array([5.],dtype=np.float32))
    length,ustar,flux=delta_temperature_flux(np.array([5.],dtype=np.float32),
        np.array([10.],dtype=np.float32),difference,difference10,theta10,theta,z0)
    mixing=water_mechanical_height(ustar,np.array([1.e-4],dtype=np.float32))
    profiles=similarity_wind_profiles(u[:1],v[:1],np.array([10.],dtype=np.float32),
        common['z_centers_m'],common['z_faces_m'],z0,length,mixing,
        common['station_latitude'][:1])
    for actual,expected in zip(result[:2],profiles,strict=True):
        np.testing.assert_array_equal(actual[:,0].view(np.uint32),expected[:,0].view(np.uint32))
    expected_state=(z0,ustar,length,mixing,np.array([0.],dtype=np.float32),
                    np.array([.2],dtype=np.float32),flux)
    for actual,expected in zip(result[2:],expected_state,strict=True):
        np.testing.assert_array_equal(actual[:1].view(np.uint32),expected.view(np.uint32))


def test_water2_coare_station_maps_double_outputs_back_to_real4():
    u=np.array([3.,3.],dtype=np.float32); v=np.array([4.,4.],dtype=np.float32)
    common=dict(anemometer_height_m=np.full(2,10.,dtype=np.float32),
        air_sensor_height_m=np.full(2,10.,dtype=np.float32),
        air_temperature_k=np.full(2,288.15,dtype=np.float32),
        air_minus_sea_temperature_k=np.array([-2.,9999.],dtype=np.float32),
        relative_humidity_percent=np.full(2,80.,dtype=np.float32),
        observed_mixing_height_m=np.full(2,9999.,dtype=np.float32),
        current_mixing_height_m=np.full(2,600.,dtype=np.float32),
        coriolis=np.full(2,1.e-4,dtype=np.float32),
        potential_lapse_k_m=np.full(2,.004,dtype=np.float32),
        previous_temperature_jump_k=np.full(2,.2,dtype=np.float32),
        z_centers_m=np.array([10.,40.,100.],dtype=np.float32),
        z_faces_m=np.array([0.,20.,60.,140.],dtype=np.float32),
        station_latitude=np.full(2,30.,dtype=np.float32),
        shortwave_down_w_m2=np.full(2,300.,dtype=np.float32),
        longwave_down_w_m2=np.full(2,350.,dtype=np.float32),
        rainfall_mm_h=np.zeros(2,dtype=np.float32))
    result=coare_water_station_profiles(u,v,**common,mixing_method=-1)
    neutral=neutral_water_station_profiles(u[1:],v[1:],
        common['anemometer_height_m'][1:],common['current_mixing_height_m'][1:],
        common['coriolis'][1:],common['z_centers_m'],common['z_faces_m'],
        common['station_latitude'][1:])
    for actual,expected in zip(result[:2],neutral[:2],strict=True):
        np.testing.assert_array_equal(actual[:,1].view(np.uint32),expected[:,0].view(np.uint32))
    saturation=saturation_vapor_pressure_lowe(np.array([15.],dtype=np.float32))
    vapor=np.asarray(np.float32(.01)*np.float32(80.)*saturation,dtype=np.float32)
    ratio=np.asarray(np.float32(.622)*vapor/(np.float32(1000.)-vapor),dtype=np.float32)
    humidity=np.asarray(ratio/(np.float32(1.)+ratio)*np.float32(1000.),dtype=np.float32)
    initial=neutral_water_parameters(np.array([3.],dtype=np.float32),
        np.array([4.],dtype=np.float32),np.array([1.e-4],dtype=np.float32))[4]
    flux=surface_flux(np.array([5.],dtype=np.float32),np.array([288.15],dtype=np.float32),
        np.array([290.15],dtype=np.float32),humidity,boundary_layer_height_m=initial,
        shortwave_down_w_m2=300.,longwave_down_w_m2=350.)
    z0=np.asarray(flux.roughness_m,dtype=np.float32)
    ustar=np.asarray(flux.friction_velocity_m_s,dtype=np.float32)
    length=np.asarray(flux.monin_obukhov_m,dtype=np.float32)
    mixing=water_mechanical_height(ustar,np.array([1.e-4],dtype=np.float32))
    profiles=similarity_wind_profiles(u[:1],v[:1],np.array([10.],dtype=np.float32),
        common['z_centers_m'],common['z_faces_m'],z0,length,mixing,
        common['station_latitude'][:1])
    for actual,expected in zip(result[:2],profiles,strict=True):
        np.testing.assert_array_equal(actual[:,0].view(np.uint32),expected[:,0].view(np.uint32))
    expected_state=(z0,ustar,length,mixing,np.array([0.],dtype=np.float32),
                    np.array([.2],dtype=np.float32),
                    np.asarray(np.asarray(flux.buoyancy_flux_k_m_s,dtype=np.float32)*
                               np.float32(1195.2),dtype=np.float32))
    for actual,expected in zip(result[2:-1],expected_state,strict=True):
        np.testing.assert_array_equal(actual[:1].view(np.uint32),expected.view(np.uint32))
    assert result[-1] is None


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


def test_maul_carson_threshold_relaxes_without_resetting_jump():
    heights=np.array([0.,100.,400.])[:,None,None]
    temperatures=np.array([291.,290.,287.])[:,None,None]
    total,convective,new_jump=mixing_height_maul_carson(
        temperatures,heights,np.array([[100.]]),np.array([[.7]]),np.array([[.2]]),
        np.array([[-100.]]),np.array([[1.]]),np.array([[1.2]]),np.array([[1.e-4]]),
        threshold_flux_per_m=20.,relaxation=True)
    assert 0.<convective.item()<100.
    assert np.isclose(new_jump.item(),.7) and 50.<=total.item()<=2500.

def test_water_maul_carson_uses_direct_buoyancy_flux():
    old=np.array([[100.,100.]])
    jump=np.array([[.2,.7]])
    flux=np.array([[.03,.001]])
    gamma=np.array([[.002,.002]])
    convective,new_jump=convective_height_maul_carson(
        old,jump,flux,gamma,threshold_flux_per_m=.05,relaxation=True)
    assert convective.dtype==np.float32 and new_jump.dtype==np.float32
    threshold=.05*old[0,0]/(1.2*996.)
    expected_jump=np.sqrt(.002*2.*3600.*.15*(.03-threshold))
    expected_height=np.sqrt(100.**2+2.*((.03-threshold)*3600.*1.15-.2*100.)/.002)
    expected_height+=expected_jump/.002
    np.testing.assert_allclose(convective[0,0],min(expected_height,2500.))
    np.testing.assert_allclose(new_jump[0,0],expected_jump)
    equilibrium=1.2*996.*.001/.05
    expected_relaxed=equilibrium+(100.-equilibrium)*np.exp(-3600./800.)
    np.testing.assert_allclose(convective[0,1],expected_relaxed)
    assert new_jump[0,1]==np.float32(jump[0,1])

def test_no_station_water_path_is_neutral_and_mechanical():
    z0,ustar,length,qh,zi,zic=neutral_water_parameters(np.array([0.,5.]),np.zeros(2),np.full(2,1.e-4))
    np.testing.assert_allclose(z0,[2.e-6,2.e-6*5.**2.5])
    assert ustar[0]==np.float32(1.e-9) and ustar[1]>0.
    np.testing.assert_array_equal(length,np.full(2,np.float32(9.9e9)))
    np.testing.assert_array_equal(qh,0.); np.testing.assert_array_equal(zic,0.)
    assert zi[0]==50. and 50.<zi[1]<2500.


def test_hosker_roughness_matches_gnu_real4_power_bits():
    speed=np.array([0x40b654a9,0,0x412665a9,0x3851b717],dtype=np.uint32).view(np.float32)
    expected=np.array([0x392284e6,0x360637bd,0x3a36ddda,0x360637bd],dtype=np.uint32)
    np.testing.assert_array_equal(hosker_roughness(speed).view(np.uint32),expected)

def test_delta_temperature_flux_matches_fortran_reference():
    cases=np.array([
        [5.,10.,-2.,-2.,290.,290.,1.11803399e-4],
        [5.,4.,2.,2.3153384,292.,292.,1.11803399e-4],
        [0.,10.,0.,0.,290.,290.,2.e-6],
    ])
    expected=np.array([
        [-24.946964263916016,.18638177216053009,22.92462158203125],
        [16.32159423828125,.14045938849449158,-15.100250244140625],
        [9.900000256e9,9.9999997171806854e-10,-8.922248976554835e-33],
    ],dtype=np.float32)
    got=np.column_stack(delta_temperature_flux(*cases.T))
    # NumPy's vector libm and GNU Fortran's scalar libm differ by one ULP for
    # two unstable-branch transcendental results on x86_64.
    np.testing.assert_array_max_ulp(got,expected,maxulp=1)
    assert saturation_vapor_pressure_lowe(0.)==pytest.approx(6.107799961)


def test_esat_and_water_delta_inputs_match_real4_oracle():
    air=np.array([288.15,273.15,303.15])
    sea=np.array([290.15,275.15,298.15])
    humidity=np.array([80.,100.,35.])
    height=np.array([10.,4.,17.5])
    theta,difference,difference_10m,theta_10m=overwater_delta_temperature_state(
        air,sea,humidity,height)
    saturation=saturation_vapor_pressure_lowe(
        sea.astype(np.float32)-np.float32(273.15))
    got=np.column_stack((saturation,theta,difference,difference_10m,theta_10m))
    # Bit patterns printed by diagnostics/code/esat_water_reference.f90.
    expected_bits=np.array([
        [0x419AEC34,0x4390E16E,0xC0240500,0xC02404FF,0x4390E16E],
        [0x40E1BDE6,0x4388E9D7,0xC0042580,0xC00F92BC,0x4388D2FD],
        [0x41FD5AD2,0x43988788,0x404D6200,0x4043DCC6,0x4398747E],
    ],dtype=np.uint32)
    np.testing.assert_array_equal(got.view(np.uint32),expected_bits)

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


def test_surface_fields_apply_coare_to_water_with_sea_data(monkeypatch):
    shape=(1,2)
    profile_z=np.broadcast_to(np.array([0.,100.,300.])[:,None,None],(3,1,2))
    profile_t=np.broadcast_to(np.array([290.,289.,287.])[:,None,None],(3,1,2))
    sea=GriddedSeaData(*[np.full(shape,value) for value in
        (288.15,290.15,80.,10.,4.,.6,-999.,-999.,120.)])
    step=Step1Fields(np.full((1,1,2),5.),np.zeros((1,1,2)),np.full((1,1,2),289.),
        np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),
        np.full((1,1,2),50.),np.zeros(shape),np.zeros(shape),np.full(shape,.1),
        np.full(shape,.5),np.full(shape,1000.),profile_t,profile_z,np.full(shape,50),
        np.zeros(shape),np.zeros(shape,dtype=int),overwater_data=sea)
    result=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.array([[0.,75.]]),np.zeros(shape),
        overwater_flux_method=10)
    assert result.sensible_heat_flux[0,1]>0.
    assert result.roughness[0,1]!=pytest.approx(2.e-6*5.**2.5)
    assert result.friction_velocity[0,1]>0.
    assert result.monin_obukhov_length[0,1]<0.
    for field in (result.roughness,result.friction_velocity,
                  result.monin_obukhov_length,result.sensible_heat_flux):
        assert field[0,1]==np.float32(field[0,1])
    # WATER's local ZICONV drives WSTAR but does not overwrite COMP's global
    # ZICONV array; MIXHT skips water before optional AVEMIX.
    assert result.convective_mixing_height[0,1]==np.float32(75.)
    assert result.water_convective_velocity_scale[0,1]>0.
    assert result.water_convective_height_state[0,1]>0.
    assert result.mixing_height[0,1]>=result.convective_mixing_height[0,1]
    batch=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.full(shape,.4),
        overwater_flux_method=10,mixing_method=2)
    assert batch.convective_mixing_height[0,1]==0.
    assert batch.temperature_jump[0,1]==np.float32(.4)
    delta=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=0)
    assert delta.sensible_heat_flux[0,1]>0.
    assert delta.monin_obukhov_length[0,1]<0.
    mechanical=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=0,mixing_method=-1,
        previous_water_convective_height=np.full(shape,75.,dtype=np.float32))
    assert mechanical.water_convective_height_state[0,1]==np.float32(0.)
    observed_mechanical=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=0,mixing_method=-1,
        previous_water_convective_height=np.full(shape,75.,dtype=np.float32),
        observed_water_mixing_height=np.array([[np.nan,432.]],dtype=np.float32))
    assert observed_mechanical.water_convective_height_state[0,1]==np.float32(75.)
    assert observed_mechanical.mixing_height[0,1]==np.float32(432.)
    prognostic_mechanical=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=0,mixing_method=-1,water_temperature_mode=2,
        previous_water_convective_height=np.full(shape,75.,dtype=np.float32),
        observed_water_mixing_height=np.array([[np.nan,432.]],dtype=np.float32))
    assert prognostic_mechanical.water_convective_height_state[0,1]==np.float32(0.)
    assert prognostic_mechanical.mixing_height[0,1]!=np.float32(432.)
    warm_first=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,warm_layer=True,gmt_hour=0.)
    warm_second=build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,warm_layer=True,gmt_hour=1.,
        warm_layer_state=warm_first.warm_layer_state)
    assert warm_second.warm_layer_state.initialized.all()
    assert np.all(warm_second.warm_layer_state.time_old_s>
                  warm_first.warm_layer_state.time_old_s)
    passed_sst=[]
    passed_humidity=[]
    original_surface_flux=calmet_solver.surface_flux
    def capture_sst(wind,air,sea_temperature,*args,**kwargs):
        passed_sst.append(np.asarray(sea_temperature).copy())
        passed_sst.append(np.asarray(kwargs['boundary_layer_height_m']).copy())
        passed_humidity.append(np.asarray(args[0]).copy())
        flux=original_surface_flux(wind,air,sea_temperature,*args,**kwargs)
        return replace(flux,monin_obukhov_m=np.array([[-.25,.25]]))
    monkeypatch.setattr(calmet_solver,'surface_flux',capture_sst)
    cold=replace(step,overwater_data=replace(
        sea,sea_temperature_k=np.full(shape,260.,dtype=np.float32)))
    cold_result=build_surface_fields(cold,np.array([[55,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,
        observed_water_mixing_height=np.array([[np.nan,432.]],dtype=np.float32))
    np.testing.assert_array_equal(passed_sst[-2],np.full(shape,270.15,dtype=np.float32))
    assert passed_sst[-1][0,1]==np.float32(432.)
    # WATER converts RH to g/kg at its fixed 1000 mb before calling COARE;
    # values below 2 would instead be interpreted as an RH fraction there.
    # Bit pattern from diagnostics/code/water_humidity_reference.f90.
    np.testing.assert_array_equal(passed_humidity[-1].astype(np.float32).view(np.uint32),
                                  np.full(shape,0x41086237,dtype=np.uint32))
    np.testing.assert_array_equal(cold_result.monin_obukhov_length,[[-5.,5.]])
    assert cold_result.mixing_height[0,1]==np.float32(432.)
    captured_old=[]
    def capture_old(old,jump,*args,**kwargs):
        captured_old.append(np.asarray(old).copy())
        return np.asarray(old,dtype=np.float32),np.asarray(jump,dtype=np.float32)
    monkeypatch.setattr(calmet_solver,'convective_height_maul_carson',capture_old)
    build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,
        previous_water_convective_height=np.full(shape,200.,dtype=np.float32),
        previous_water_mixing_height=np.full(shape,60.,dtype=np.float32))
    assert captured_old[-1]==np.float32(60.)

    shared=[]
    def advance_shared(old,jump,*args,**kwargs):
        shared.append(np.float32(jump))
        return np.float32(old),np.float32(jump)+np.float32(.25)
    monkeypatch.setattr(calmet_solver,'convective_height_maul_carson',advance_shared)
    shared_result=build_surface_fields(step,np.array([[55,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),
        np.array([[.25,.75]],dtype=np.float32),overwater_flux_method=10)
    # WATER's legacy bare-DPTT call aliases DPTT(1,1), so both water cells
    # consume one scalar chain and only that common element is updated.
    np.testing.assert_array_equal(shared,np.array([.25,.5],dtype=np.float32))
    np.testing.assert_array_equal(shared_result.temperature_jump,
                                  np.array([[.75,.75]],dtype=np.float32))
    calm=replace(step,u=np.zeros_like(step.u),v=np.zeros_like(step.v))
    calm_result=build_surface_fields(calm,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,mixing_method=-1)
    assert passed_sst[-1][0,1]<np.float32(50.)
    assert calm_result.mixing_height[0,1]>=np.float32(50.)
    captured_solar=[]
    original_advance=calmet_solver.advance_warm_layer
    def capture_solar_time(state,solar_time,*args,**kwargs):
        captured_solar.append(np.asarray(solar_time).copy())
        return original_advance(state,solar_time,*args,**kwargs)
    monkeypatch.setattr(calmet_solver,'advance_warm_layer',capture_solar_time)
    build_surface_fields(step,np.array([[20,55]]),np.full(shape,.25),
        np.full(shape,.15),np.ones(shape),np.full(shape,.15),np.zeros(shape),
        np.full(shape,.8),np.full(shape,1.e-4),np.zeros(shape),np.zeros(shape),
        overwater_flux_method=10,warm_layer=True,gmt_hour=0.,
        longitude=np.zeros(shape))
    np.testing.assert_array_equal(captured_solar[-1],np.full(shape,8.*3600.))

def test_psiud_convective_constant_matches_four_atan_one():
    zl=np.array([-1.])
    x=(1.-16.*zl)**.25; y=(1.-10.15*zl)**.3333
    psik=2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-2.*np.arctan(x)+np.pi/2.
    psic=1.5*np.log((1.+y+y*y)/3.)-np.sqrt(3.)*np.arctan((1.+2.*y)/np.sqrt(3.))+np.pi/np.sqrt(3.)
    expected=.5*(psik+psic)
    np.testing.assert_allclose(psi_momentum_dyer(zl),expected)

def test_psiud_matches_gnu_mixed_kind_runtime_bits():
    # GNU CALMET ELUSTR iteration: REAL zm/xl promoted into REAL*8 PSIUD.
    zl=np.float64(np.float32(10.)/np.float32(-1226.91028))
    got=np.float32(psi_momentum_dyer(np.array([zl]))[0])
    assert got.view(np.uint32)==np.uint32(0x3d006cad)

def test_heat_flux_matches_gnu_optimized_integer_powers():
    temperature=np.array([[np.array(0x438fe818,dtype=np.uint32).view(np.float32)]])
    expected_qsw=np.array(0x42d9d146,dtype=np.uint32).view(np.float32)
    sun=np.asarray((expected_qsw/np.float32(.25)+np.float32(30.))/
                   np.float32(990.),dtype=np.float32)
    qh,qsw=heat_flux(np.array([[sun]],dtype=np.float32),
        np.ones((1,1),dtype=np.float32),temperature,np.array([[40]]),
        np.array([[.12]],dtype=np.float32),np.ones((1,1),dtype=np.float32),
        np.array([[.15]],dtype=np.float32),np.zeros((1,1),dtype=np.float32))
    assert qsw[0,0].view(np.uint32)==np.uint32(0x42d9d146)
    assert qh[0,0].view(np.uint32)==np.uint32(0x41d03f27)

def test_heat_flux_nondefault_radiation_coefficients_match_gnu_bits():
    # GNU -O2 scalar HEATFX expressions using all seven Group-6 controls.
    shape=(1,1)
    qh,qsw=heat_flux(
        np.full(shape,.63,dtype=np.float32),
        np.full(shape,.42,dtype=np.float32),
        np.full(shape,301.25,dtype=np.float32),np.full(shape,20),
        np.full(shape,.17,dtype=np.float32),np.full(shape,.8,dtype=np.float32),
        np.full(shape,.12,dtype=np.float32),np.full(shape,5.,dtype=np.float32),
        turbidity_coefficient_1=850.5,turbidity_coefficient_2=-12.75,
        cloudiness_coefficient_1=-.55,cloudiness_coefficient_2=2.2,
        net_radiation_coefficient_1=6.e-13,
        net_radiation_coefficient_2=45.25,net_radiation_coefficient_3=.2)
    assert qsw[0,0].view(np.uint32)==np.uint32(0x43f03344)
    assert qh[0,0].view(np.uint32)==np.uint32(0x43045629)

def test_convective_velocity_scale_uses_runtime_real_power():
    zi=np.array([[600.]],dtype=np.float32)
    qh=np.array([[np.array(0x41d03f27,dtype=np.uint32).view(np.float32)]])
    temperature=np.array([[np.array(0x438fe818,dtype=np.uint32).view(np.float32)]])
    density=np.array([[np.array(0x3f8b3fd2,dtype=np.uint32).view(np.float32)]])
    got=convective_velocity_scale(zi,qh,temperature,density,np.array([[40]]))
    argument=np.asarray(np.float32(9.81)*qh*zi/
        (temperature*density*np.float32(996.)),dtype=np.float32)
    expected=np.float32(_real4_power(argument[0,0],np.float32(.3333333)))
    assert got[0,0].view(np.uint32)==expected.view(np.uint32)

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

def test_avetmp_converts_integer_offsets_to_default_real_before_products():
    shape=(31,23)
    temperature=np.asarray(np.float32(287.)+np.arange(
        np.prod(shape),dtype=np.float32).reshape(shape)*np.float32(.001),
        dtype=np.float32)
    u=np.zeros(shape,dtype=np.float32); v=np.zeros(shape,dtype=np.float32)
    u[20,12]=np.float32(2.4461055); v[20,12]=np.float32(4.387089)
    averaged,_=upwind_mixing_average(
        u,v,temperature,temperature,1000.,minimum_radius_cells=10,
        half_angle_degrees=30.)
    # AVETMP evaluates FLOAT(IREL)*ETAX in REAL*4.  Letting NumPy promote the
    # integer offset and product to float64 changes this GNU reference bit.
    assert averaged[20,12].view(np.uint32)==np.uint32(0x438fb496)


def test_avemix_and_avetmp_integer_powers_stay_default_real_on_numpy_2():
    temperature=np.array([
        [276.65152,284.49252,294.08954],
        [277.3709,276.89392,290.89072],
        [281.31958,272.98096,274.90564],
    ],dtype=np.float32)
    u=np.zeros((3,3),dtype=np.float32); v=np.zeros_like(u)
    u[1,2]=np.float32(7.3554974); v[1,2]=np.float32(-3.969456)
    averaged,_=upwind_mixing_average(
        u,v,temperature,temperature,1000.,minimum_radius_cells=2)
    # GNU AVETMP and NumPy 1.26 evaluate both **2 terms in REAL*4.
    # NumPy 2 promotes a float32 base raised to np.int32(2) to float64;
    # that old expression produced 0x438d6950 here.
    assert averaged[1,2].view(np.uint32)==np.uint32(0x438d6951)


def test_avemix_uses_legacy_fixed_one_hour_trajectory_for_subhour_runs():
    zi=np.arange(7.,dtype=np.float32)[None,:]
    u=np.full_like(zi,1.)
    zero=np.zeros_like(zi)
    hourly=upwind_mixing_average(u,zero,zi,zi,1000.,minimum_radius_cells=0,
                                 timestep_s=3600.)[0]
    subhour=upwind_mixing_average(u,zero,zi,zi,1000.,minimum_radius_cells=0,
                                  timestep_s=900.)[0]
    np.testing.assert_array_equal(subhour,hourly)


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

def test_current_diagnostic_wind_path_composes_all_kernels():
    shape=(3,3); zfaces=np.array([0.,20.,300.,600.]); profile_z=np.broadcast_to(np.array([0.,100.,300.])[:,None,None],(3,3,3)); profile_t=np.broadcast_to(np.array([290.,289.,287.])[:,None,None],(3,3,3))
    step=Step1Fields(np.ones((3,3,3)),np.zeros((3,3,3)),np.broadcast_to(np.array([290.,289.,287.])[:,None,None],(3,3,3)),np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),np.full((3,3,3),50.),np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),profile_t,profile_z,np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    terrain=np.tile(np.arange(3.)*10.,(3,1)); land=np.full(shape,20); constant=np.ones(shape)
    result=build_diagnostic_wind(step,terrain,land,np.full(shape,.25),np.full(shape,.15),constant,np.full(shape,.15),np.zeros(shape),np.full(shape,.5),np.zeros(shape),zfaces,terrain_radius_km=2.,smoothing_passes=(1,1,1),retain_debug_stages=True)
    assert result.u.shape==(3,3,3) and result.w.shape==(4,3,3)
    assert np.isfinite(result.u).all() and np.isfinite(result.w).all() and np.isfinite(result.lapse_rate).all()
    assert result.kinematic_stage[2].shape==(4,3,3)
    assert result.froude_stage[0].shape==(3,3,3)
    assert result.slope_stage[1].shape==(3,3,3)
    assert result.interpolated_stage[0].shape==(3,3,3)
    assert result.terrain_adjusted_stage[1].shape==(3,3,3)
    assert result.initial_stage[2].shape==(4,3,3)
    assert result.initial_stage[3].shape==(3,3,3)
    disabled=build_diagnostic_wind(step,terrain,land,np.full(shape,.25),np.full(shape,.15),constant,
        np.full(shape,.15),np.zeros(shape),np.full(shape,.5),np.zeros(shape),zfaces,
        terrain_radius_km=2.,use_froude=False,use_slope=False,smoothing_passes=(0,0,0))
    np.testing.assert_array_equal(disabled.u,step.u)
    np.testing.assert_array_equal(disabled.v,step.v)
    kinematic=build_diagnostic_wind(step,terrain,land,np.full(shape,.25),np.full(shape,.15),constant,
        np.full(shape,.15),np.zeros(shape),np.full(shape,.5),np.zeros(shape),zfaces,
        terrain_radius_km=2.,use_froude=False,use_kinematic=True,use_slope=False,
        smoothing_passes=(0,0,0),divergence_iterations=3)
    assert np.isfinite(kinematic.u).all() and np.isfinite(kinematic.v).all()
    assert not np.array_equal(kinematic.u,step.u)
    domain=build_diagnostic_wind(step,terrain,land,np.full(shape,.25),np.full(shape,.15),constant,
        np.full(shape,.15),np.zeros(shape),np.full(shape,.5),np.zeros(shape),zfaces,
        terrain_radius_km=2.,spatial_lapse=False,spatial_surface_temperature=False,
        minimum_average_radius=1,half_average_angle=20.,use_froude=False,use_slope=False,
        smoothing_passes=(0,0,0))
    np.testing.assert_allclose(domain.lapse_rate,domain.lapse_rate.mean())
    selected=np.full(shape,-.007)
    fixed=build_diagnostic_wind(step,terrain,land,np.full(shape,.25),np.full(shape,.15),constant,
        np.full(shape,.15),np.zeros(shape),np.full(shape,.5),np.zeros(shape),zfaces,
        terrain_radius_km=2.,lapse_rate_override=selected,use_froude=False,use_slope=False,
        smoothing_passes=(0,0,0))
    np.testing.assert_array_equal(fixed.lapse_rate,selected.astype(np.float32))


def test_diagnostic_wind_reuses_rdmm5_density_for_itprog_2(monkeypatch):
    shape=(2,2); zfaces=np.array([0.,20.,300.],dtype=np.float32)
    temperatures=np.broadcast_to(
        np.array([290.,287.],dtype=np.float32)[:,None,None],(2,*shape))
    profiles=np.broadcast_to(
        np.array([290.,289.,287.],dtype=np.float32)[:,None,None],(3,*shape))
    heights=np.broadcast_to(
        np.array([0.,100.,300.],dtype=np.float32)[:,None,None],(3,*shape))
    source_density=np.full(shape,np.float32(1.111111),dtype=np.float32)
    step=Step1Fields(
        np.ones((2,*shape),dtype=np.float32),
        np.zeros((2,*shape),dtype=np.float32),temperatures,
        np.full(shape,np.float32(290.),dtype=np.float32),
        np.full(shape,np.float32(1000.),dtype=np.float32),source_density,
        np.full((2,*shape),np.float32(50.),dtype=np.float32),
        np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.full(shape,np.float32(.1),dtype=np.float32),
        np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32),
        profiles,heights,np.full(shape,50,dtype=np.int32),
        np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.int32))
    captured=[]
    def capture_similarity(*args,**kwargs):
        captured.append(np.asarray(args[4],dtype=np.float32).copy())
        zeros=np.zeros(shape,dtype=np.float32)
        return zeros,zeros,np.asarray(args[7],dtype=np.float32)
    monkeypatch.setattr(calmet_solver,'surface_similarity',capture_similarity)
    common=(step,np.zeros(shape,dtype=np.float32),np.full(shape,20),
        np.full(shape,np.float32(.25),dtype=np.float32),
        np.full(shape,np.float32(.15),dtype=np.float32),np.ones(shape,dtype=np.float32),
        np.full(shape,np.float32(.15),dtype=np.float32),np.zeros(shape,dtype=np.float32),
        np.full(shape,np.float32(.5),dtype=np.float32),np.zeros(shape,dtype=np.float32),
        zfaces)
    options=dict(use_froude=False,use_slope=True,smoothing_passes=(0,0),
                 minimum_average_radius=1,lapse_height_m=100.)
    build_diagnostic_wind(*common,temperature_mode=2,**options)
    build_diagnostic_wind(*common,temperature_mode=1,**options)
    np.testing.assert_array_equal(captured[0],source_density)
    averaged_temperature,_=upwind_mixing_average(
        step.u[0],step.v[0],step.temperature[0],step.temperature[0],1000.,
        minimum_radius_cells=1)
    recomputed=calmet_solver.density_from_pressure_temperature(
        step.surface_pressure_mb,averaged_temperature)
    np.testing.assert_array_equal(captured[1],recomputed)
    assert not np.array_equal(captured[0],captured[1])


def test_adjust_and_objective_wind_follow_iwfcod_zero_path():
    shape=(3,4); zfaces=np.array([0.,20.,300.])
    u=np.broadcast_to(np.array([2.,3.])[:,None,None],(2,*shape)).copy()
    v=np.broadcast_to(np.array([1.,-.5])[:,None,None],(2,*shape)).copy()
    flat_u,flat_v=terrain_adjust_surface_wind(u,v,np.zeros(shape),zfaces[-1],1000.,1000.)
    np.testing.assert_array_equal(flat_u,u); np.testing.assert_array_equal(flat_v,v)
    terrain=np.arange(12.,dtype=float).reshape(shape)**2
    adjusted_u,adjusted_v=terrain_adjust_surface_wind(u,v,terrain,zfaces[-1],1000.,1000.)
    assert adjusted_u.dtype==np.float32 and adjusted_v.dtype==np.float32
    assert not np.array_equal(adjusted_u[0],u[0])
    np.testing.assert_array_equal(adjusted_u[1:],u[1:])
    np.testing.assert_array_equal(adjusted_v[1:],v[1:])
    profile_z=np.broadcast_to(np.array([0.,200.,500.])[:,None,None],(3,*shape))
    profile_t=np.broadcast_to(np.array([290.,288.,285.])[:,None,None],(3,*shape))
    step=Step1Fields(u,v,np.broadcast_to(np.array([290.,288.])[:,None,None],(2,*shape)),
        np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),
        np.full((2,*shape),50.),np.zeros(shape),np.zeros(shape),np.full(shape,.1),
        np.zeros(shape),np.zeros(shape),profile_t,profile_z,np.full(shape,50.),
        np.zeros(shape),np.zeros(shape,dtype=int))
    result=build_objective_wind(step,terrain,np.zeros(shape),zfaces,dx_m=1000.,dy_m=1000.,
        lapse_height_m=100.,smoothing_passes=(0,0))
    assert result.w.shape==(3,*shape) and np.isfinite(result.w).all()
    np.testing.assert_array_equal(result.preliminary_heat_flux,np.zeros(shape))


def test_adjust_real4_four_sweep_regression():
    terrain=(np.arange(12,dtype=np.float32).reshape(3,4)**2)+np.array(
        [[.25,-.5,.75,-.125],[.5,.25,-.75,.875],[-.25,.625,.125,-.5]],dtype=np.float32)
    u=np.stack((np.array([[2.,-1.,3.,.5],[1.5,2.25,-.75,4.],
                          [-2.,.25,1.75,-1.25]],dtype=np.float32),
                np.full((3,4),7.,dtype=np.float32)))
    v=np.stack((np.array([[-.5,1.25,2.,-1.],[3.,-.25,1.5,.75],
                          [.5,-2.,2.5,1.]],dtype=np.float32),
                np.full((3,4),-3.,dtype=np.float32)))
    got_u,got_v=terrain_adjust_surface_wind(u,v,terrain,300.,713.,887.)
    expected_u=np.array([0x3ffece5d,0xbf844c18,0x4040b95a,0x3f091a53,
                         0x3fc27b0a,0x400ea3e2,0xbf49e4f0,0x40801034,
                         0xc000deb0,0x3e0f65b3,0x3fd6a175,0xbf9d08a1],dtype=np.uint32)
    expected_v=np.array([0xbf09a04e,0x3f9e23ff,0x3ffc747c,0xbf870005,
                         0x4041d651,0xbe661d83,0x3fbbd07d,0x3f35149a,
                         0x3f10f994,0xbffae7af,0x401fae01,0x3f818a51],dtype=np.uint32)
    np.testing.assert_array_equal(got_u[0].view(np.uint32).ravel(),expected_u)
    np.testing.assert_array_equal(got_v[0].view(np.uint32).ravel(),expected_v)
    np.testing.assert_array_equal(got_u[1],u[1]); np.testing.assert_array_equal(got_v[1],v[1])

def test_target_grid_coordinates_and_hourly_solar_index():
    lat,lon=target_latitude_longitude(GridConfig(2,2,1.,1.))
    assert lat.shape==(2,2) and lon.shape==(2,2)
    assert np.all(np.diff(lon,axis=1)>0.) and np.all(np.diff(lat,axis=0)>0.)


def test_microi_cell_coordinates_retain_metres_to_kilometres_rounding():
    grid=GridConfig(610,499,1.,1.,projection='LCC',lat0=27.,lon0=117.5,
                    lat1=30.,lat2=60.,datum='WGS-84')
    latitude,longitude=target_latitude_longitude(grid)
    for y,x,latitude_bits,longitude_bits in (
        (21,241,0x41d945d6,0x42efcf31),
        (49,388,0x41dadb1f,0x42f2c294),
        (153,470,0x41e1f61c,0x42f48491),
        (165,497,0x41e2b376,0x42f513f0),
        (170,230,0x41e3f0c7,0x42efad54),
    ):
        assert np.float32(latitude[y,x]).view(np.uint32)==np.uint32(latitude_bits)
        assert np.float32(longitude[y,x]).view(np.uint32)==np.uint32(longitude_bits)
    y,x=458,29
    assert np.float32(latitude[y,x]).view(np.uint32)==np.uint32(0x41f8f1e5)
    assert np.float32(longitude[y,x]).view(np.uint32)==np.uint32(0x42eb9f16)
    sun=solar_sine(77,0,latitude[y:y+1,x:x+1],longitude[y:y+1,x:x+1])
    assert sun[1,0,0].view(np.uint32)==np.uint32(0x3ee9a3dd)
    # This second point distinguishes NumPy's SIMD cosine (…5f) from the
    # scalar REAL*4 libm result produced by GNU Fortran (…60).
    y,x=154,43
    sun=solar_sine(77,0,latitude[y:y+1,x:x+1],longitude[y:y+1,x:x+1])
    assert sun[1,0,0].view(np.uint32)==np.uint32(0x3ef1a060)
    # This production point exercises SOLAR's runtime sinf/cosf/asinf path;
    # evaluating ideal double intrinsics and narrowing later is four QSW ulps
    # above the GNU result.
    y,x=44,313
    midpoint=solar_at_step_midpoint(datetime(2018,3,18),3600,0,
        latitude[y:y+1,x:x+1],longitude[y:y+1,x:x+1])
    assert midpoint[0,0].view(np.uint32)==np.uint32(0x3f02dfae)

def test_canonical_dataset_timestamp_decoder():
    class Variable:
        shape=(1,19)
        def __getitem__(self,index): return np.frombuffer(b'2018-03-18_00:00:00',dtype='S1')
    dataset={'time_text':Variable()}
    assert dataset_timestamp(dataset,0).isoformat()=='2018-03-18T00:00:00'
    assert _prognostic_local_timestamp(dataset,0,5)==datetime(2018,3,17,19)

def test_hour_iterator_initializes_endpoints_before_first_ending_record(monkeypatch):
    text=np.array([list(value) for value in ('2018-03-18_00:00:00','2018-03-18_01:00:00','2018-03-18_02:00:00')],dtype='S1')
    calls=[]; builds=[]; build_times=[]; shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1))
    def make_step(index):
        value=np.float32(index)
        return Step1Fields(level*value,level*value,level*280.,np.ones(shape)*280.,
            np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
            np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),
            profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_build(dataset,index,*args,**kwargs):
        builds.append(index)
        build_times.append(kwargs['timestamp'])
        return make_step(index)
    def fake_solve(dataset,index,grid,z_faces,geo,timestamp,convective,jump,**options):
        assert convective.dtype==np.float32 and jump.dtype==np.float32
        water_state=options['previous_water_convective_height']
        np.testing.assert_array_equal(water_state,np.full(shape,len(calls),dtype=np.float32))
        np.testing.assert_array_equal(options['previous_water_mixing_height'],
                                      np.full(shape,len(calls),dtype=np.float32))
        calls.append((index,timestamp)); surface=type('Surface',(),{
            'convective_mixing_height':convective,'mixing_height':water_state+np.float32(1.),
            'temperature_jump':jump,
            'water_convective_height_state':water_state+np.float32(1.)})()
        step=(make_step(index) if options['step_override'] is None else options['step_override'])
        return type('Hour',(),{'surface':surface,'step1':step})()
    monkeypatch.setattr('pycalmet.calmet_solver.build_step1_fields',fake_build)
    monkeypatch.setattr('pycalmet.calmet_solver.solve_hour',fake_solve)
    list(iter_hours({'time_text':text},GridConfig(1,1,1.,1.),[0.,20.],{}))
    assert calls==[(1,datetime(2018,3,18)),(2,datetime(2018,3,18,1))]
    # RDMM5 reads the initial boundary before the first ending record so that
    # STULL's carried nighttime temperature state is initialized in order.
    assert builds==[0,1,2]
    assert build_times==[datetime(2018,3,18,hour) for hour in range(3)]

def test_nonfinal_exact_boundary_uses_current_and_future_ceiling(monkeypatch):
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
    # Startup reads endpoint 1/2 even though endpoint 1 is the first ending.
    # At ending hour three RDMM5 has shifted to endpoint 3/4 at DTINC=0, so
    # ICEILG takes MAX(0,20), not MAX(100,0) from the stale endpoint 2/3 pair.
    # The final hour retains endpoint 3/4 and evaluates it at DTINC=1.
    assert received==[100.,100.,20.,20.]
    assert builds==[0,1,2,3,4]

def test_single_hour_first_boundary_still_reads_future_ceiling(monkeypatch):
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
    assert received==[40.]
    with pytest.raises(ValueError,match='requires a following prognostic record'):
        list(iter_hours({'time_text':text[:2]},GridConfig(1,1,1.,1.),[0.,20.],{},
                        time_indices=[1]))

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
    assert builds==[(0,datetime(2018,3,18)),(1,datetime(2018,3,18,3)),
                    (2,datetime(2018,3,18,6))]

def test_gmt_endpoints_convert_to_base_time_and_keep_comp_solar_day(monkeypatch):
    # 04/05 UTC are 23/00 in CALMET's positive-west IBTZ=5 base time.  COMP
    # retains the June 21 SOLAR table while RDMM5 reads the future 00 record.
    text=np.array([list(value) for value in
        ('2018-06-22_04:00:00','2018-06-22_05:00:00',
         '2018-06-22_06:00:00')],dtype='S1')
    shape=(1,1); level=np.ones((1,1,1)); profile=np.ones((2,1,1)); builds=[]; calls=[]
    step=Step1Fields(level,level,level*280.,np.ones(shape)*280.,
        np.ones(shape)*1000.,np.ones(shape),level*50.,np.zeros(shape),
        np.zeros(shape),np.ones(shape),np.zeros(shape),np.zeros(shape),
        profile,profile,np.ones(shape)*50.,np.zeros(shape),np.zeros(shape,dtype=int))
    def fake_build(dataset,index,*args,**kwargs):
        builds.append((index,kwargs['timestamp'],kwargs['solar_day']))
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
    assert builds==[(0,beginning,beginning),
                    (1,datetime(2018,6,22),beginning),
                    (2,datetime(2018,6,22,1),beginning)]
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

def test_pgt_land_and_water_and_convective_velocity():
    u=np.array([[1.,5.,2.]]); zero=np.zeros_like(u); land=np.array([[20,20,55]])
    classes=pgt_stability(u,zero,np.array([[.9,-.1,.2]]),np.array([[0.,0.,0.]]),zero,land,np.array([[-10.,10.,9.9e9]]))
    np.testing.assert_array_equal(classes,[[1,5,4]])
    wstar=convective_velocity_scale(np.array([[500.,0.]]),np.array([[100.,100.]]),np.full((1,2),290.),np.full((1,2),1.2),np.full((1,2),20))
    assert wstar[0,0]>0. and wstar[0,1]==0.
    water=convective_velocity_scale(np.array([[500.]]),np.array([[100.]]),
        np.array([[290.]]),np.array([[1.2]]),np.array([[55]]))
    argument=np.array([[np.float32(9.81)*(np.float32(100.)/np.float32(1195.2))*
                        np.float32(500.)/np.float32(290.)]],dtype=np.float32)
    expected=np.asarray(_real4_power(argument,np.float32(.3333333)),dtype=np.float32)
    np.testing.assert_array_equal(water,expected)

def test_streaming_calmet_netcdf_writer(tmp_path):
    shape=(1,1); three=np.ones((3,1,1)); step=Step1Fields(three,three,three*290.,np.full(shape,290.),np.full(shape,1000.),np.full(shape,1.2),three*50.,np.zeros(shape),np.zeros(shape),np.full(shape,.1),np.zeros(shape),np.zeros(shape),three*290.,np.array([0.,100.,300.])[:,None,None],np.full(shape,50),np.zeros(shape),np.zeros(shape,dtype=int))
    wind=DiagnosticWindFields(three,three,np.ones((4,1,1)),three,np.zeros(shape),np.zeros(shape))
    surface=SurfaceFields(np.full(shape,.1),np.full(shape,1.2),np.full(shape,100.),np.full(shape,20.),np.full(shape,.2),np.full(shape,-100.),np.full(shape,500.),np.full(shape,300.),np.zeros(shape))
    hour=HourFields(datetime(2018,3,18),step,wind,surface,three*280.); geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20)}
    path=tmp_path/'calmet.nc'
    with CalmetNetCDFWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo) as writer: writer.append(hour,np.full(shape,.5),geo['dominant_landuse'])
    with nc.Dataset(path) as ds:
        assert ds.dimensions['time'].size==1; np.testing.assert_allclose(ds['wind_u'][0],three)
        np.testing.assert_array_equal(ds['surface_relative_humidity'][0],50)
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
    got=prognostic_temperature_3d(source,surface,np.array([[200.]]),[10.,100.,300.])
    np.testing.assert_allclose(got[:,0,0],[292.,292.-.0098*90.,275.])

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
    with LegacyCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo,datetime(2018,3,18),datetime(2018,3,18,1),control_lines=['TEST'],base_timezone='UTC-0830',water_range=(42,42),landuse_category_count=2,run_type=0) as writer:
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
                                      [201807700,0,201807700,3600])

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
    begin=datetime(2018,3,18,0,15); hour=HourFields(begin,step,wind,surface,three*280.)
    geo={'terrain_height':np.zeros(shape),'dominant_landuse':np.full(shape,20),'roughness_length':np.full(shape,.1),'leaf_area_index':np.ones(shape)}
    path=tmp_path/'quarter_hour.dat'
    with LegacyCalmetWriter(path,GridConfig(1,1,1.,1.),[0.,20.,100.,300.],geo,begin,begin+timedelta(minutes=15),run_type=0,timestep_seconds=900) as writer:
        writer.append(hour,np.full(shape,.5))
    with FortranSequentialReader(path) as reader:
        reader.read_record(); count=np.frombuffer(reader.read_record(),dtype='<i4').item()
        for _ in range(count): reader.read_record()
        reader.read_record()
        for _ in range(5): reader.read_record()
        record=reader.read_record()
        np.testing.assert_array_equal(np.frombuffer(record[8:24],dtype='<i4'),
                                      [201807700,900,201807700,1800])


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


def test_mesopac_writer_is_byte_identical_to_gfortran_outpc_records(tmp_path):
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
    with MesopacWriter(python_output,GridConfig(2,2,1.,1.),[0.,5000.],geo,
            datetime(2018,3,18),datetime(2018,3,18,1),timezone_offset=8,
            surface_coordinates_m=[(500.,500.),(1500.,1500.)],
            upper_coordinates_m=[(500.,1500.)],nearest_surface_station=[[1,1],[2,2]]) as writer:
        writer.append(hour,np.zeros(shape),station_temperature_k=[280.,281.],
            station_shortwave_flux=[10.,20.],station_relative_humidity=[50,60],
            station_precipitation_code=[0,1])
    assert python_output.read_bytes()==(tmp_path/'oracle.dat').read_bytes()


def test_subhour_solar_angle_matches_comp_midpoint_interpolation():
    latitude=np.array([[30.]]); longitude=np.array([[120.]])
    samples=solar_sine(77,0,latitude,longitude)
    assert samples.dtype==np.float32
    assert samples[1,0,0].view(np.uint32)==1056611676
    got=solar_at_step_midpoint(datetime(2018,3,18,10),900,0,latitude,longitude)
    # 10:07:30 lies 5/8 of the way from SOLAR's 09:30 to 10:30 samples.
    np.testing.assert_allclose(got,samples[10]*.375+samples[11]*.625)

def test_rdmm5_ending_record_maps_to_preceding_half_hour_solar_sample():
    latitude=np.array([[30.]],dtype=np.float32)
    longitude=np.array([[120.]],dtype=np.float32)
    samples=solar_sine(77,0,latitude,longitude)
    got=_prognostic_record_solar(
        datetime(2018,3,18,10),0,latitude,longitude)
    np.testing.assert_array_equal(got,samples[10])
    assert not np.array_equal(got,samples[11])

def test_rdmm5_midnight_record_uses_current_comp_day_solar_table():
    latitude=np.array([[70.]],dtype=np.float32)
    longitude=np.array([[0.]],dtype=np.float32)
    current_day=datetime(2018,6,21,23)
    record_time=datetime(2018,6,22)
    got=_prognostic_record_solar(
        record_time,0,latitude,longitude,solar_day=current_day)
    expected=solar_sine(172,0,latitude,longitude)[0]
    next_day=solar_sine(173,0,latitude,longitude)[0]
    np.testing.assert_array_equal(got,expected)
    assert not np.array_equal(got,next_day)

def test_terrain_extrema_and_gradients():
    terrain=np.array([[0.,1.,2.],[0.,1.,9.],[0.,1.,2.]])
    gx,gy=terrain_gradients(terrain,1000.,1000.)
    np.testing.assert_allclose(gx[0],[.0005,.001,.0005])
    highest,lowest,dhi,dlo=terrain_extrema(terrain,1.01,1000.,1000.)
    assert all(item.dtype==np.float32 for item in (highest,lowest,dhi,dlo))
    assert highest[1,1]==9.; assert dhi[1,1]==1000.
    assert lowest[1,1]==0.; assert dlo[1,1]==1000.


def test_terset_keeps_first_scan_distance_for_equal_extrema():
    terrain=np.array([[9.,0.,9.],[0.,1.,0.],[9.,0.,9.]],dtype=np.float32)
    high,low,dhigh,dlow=terrain_extrema(terrain,2.,713.,887.)
    # L is outer and K inner in TERSET. Equal later extrema do not replace
    # the distance recorded when the strict GT/LT comparison first succeeds.
    assert high[1,1]==np.float32(9.) and low[1,1]==np.float32(0.)
    dx=np.float32(np.float32(713.)*np.float32(.001))
    dy=np.float32(np.float32(887.)*np.float32(.001))
    expected_high=np.float32(np.sqrt(np.float32(np.float32(dx*dx)+
                                                 np.float32(dy*dy)))*np.float32(1000.))
    assert dhigh[1,1]==expected_high
    assert dlow[1,1]==np.float32(887.)

def test_topographic_velocity_is_zero_for_flat_terrain():
    u=np.ones((2,2,3)); v=np.zeros_like(u)
    got=topographic_vertical_velocity(u,v,np.zeros((2,3)),[0.,20.,40.],np.zeros((2,3)),np.full((2,3),300.),.1,1000.,1000.)
    np.testing.assert_array_equal(got,0.)


def test_topof2_preserves_real4_vertical_recurrence():
    terrain=np.array([[0.,20.,75.],[10.,45.,120.]],dtype=np.float32)
    u=np.array([[[2.,-1.,3.],[.5,4.,-2.]],[[6.,2.,-3.],[1.,-4.,5.]]],dtype=np.float32)
    v=np.array([[[-.5,2.,1.],[3.,-.25,2.5]],[[1.,-2.,4.],[-3.,2.,.5]]],dtype=np.float32)
    got=topographic_vertical_velocity(u,v,terrain,[0.,23.,91.],
        np.array([[-.02,-.005,.001],[.004,-.015,.008]],dtype=np.float32),
        np.array([[288.,291.,294.],[287.,290.,293.]],dtype=np.float32),.13,713.,887.)
    expected=np.array([
        0xaf800000,0x00000000,0x32000000,0x00000000,0x00000000,0x00000000,
        0xb9972848,0x39a69c00,0xbacaadc0,0xba123800,0xbb646b80,0x3a173b00,
        0xbd8642e1,0xbdcd7a70,0x3e1984d6,0x3ca205da,0x3f12c40c,0xbea186cf],dtype=np.uint32)
    assert got.dtype==np.float32 and got.shape==(3,2,3)
    np.testing.assert_array_equal(got.view(np.uint32).ravel(),expected)

def test_slope_flow_respects_water_and_layer_depth():
    terrain=np.array([[0.,10.,20.]])
    high,low,dhi,dlo=terrain_extrema(terrain,3.,1000.,1000.)
    us,vs=slope_flow(terrain,high,low,dhi,dlo,np.full_like(terrain,-50.),np.full_like(terrain,290.),np.array([[20,55,20]]),[0.,.25,1.],1000.,1000.)
    assert us.shape==(2,1,3); assert np.all(us[:,0,1]==0.); assert np.allclose(vs,0.)

def test_slope_flow_uses_gnu_powf_for_gradient_magnitude(monkeypatch):
    exponents=[]
    original=_real4_power
    def recording_power(base,exponent):
        exponents.append(np.asarray(exponent,dtype=np.float32).copy())
        return original(base,exponent)
    monkeypatch.setattr('pycalmet.calmet_numerics._real4_power',recording_power)
    terrain=np.array([[0.,10.,20.]],dtype=np.float32)
    high,low,dhi,dlo=terrain_extrema(terrain,3.,1000.,1000.)
    slope_flow(terrain,high,low,dhi,dlo,np.full_like(terrain,-50.),
        np.full_like(terrain,290.),np.full(terrain.shape,20),
        [0.,.25,1.],1000.,1000.)
    assert any(value==np.float32(.5) for value in exponents)

def test_prognostic_mapper_hits_exact_source_point():
    lat=np.array([[27.,27.],[28.,28.]])
    lon=np.array([[117.5,118.5],[117.5,118.5]])
    mapper=PrognosticMapper.build(lat,lon,GridConfig(1,1,.001,.001))
    got=mapper.interpolate(np.array([[7.,2.],[3.,4.]]))
    assert got.shape==(1,1); assert abs(got.item()-7.)<.02

def test_precipitation_radius_mapping_uses_all_nearby_sources_and_fallback():
    lat=np.array([[27.,27.],[28.,28.]])
    lon=np.array([[117.5,118.5],[117.5,118.5]])
    mapper=PrognosticMapper.build(lat,lon,GridConfig(1,1,.001,.001))
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
    assert np.isfinite(mapper.radius_interpolate(field,0.,method=3)).all()


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


def test_interpqr_target_relative_coordinates_are_real4_before_scaling():
    mapper=PrognosticMapper((1,1),np.array([[0,0,0,0]]),np.ones((1,4)),(1,1),
        np.array([[100000.1,0.]],dtype=np.float32),
        np.array([[100000.6,0.]],dtype=np.float32),grid_origin=(100000.1,0.))
    # REAL*4 subtraction gives exactly 0.5 km, so the source is included at
    # the inclusive radius boundary and CUTP applies.  Subtracting the Python
    # float origin first gives 0.5015625 km and incorrectly takes fallback.
    np.testing.assert_array_equal(mapper.radius_interpolate(
        np.array([[.005]],dtype=np.float32),.5,method=2,cutoff=.01),
        np.zeros((1,1),dtype=np.float32))


def test_interpqr_candidate_search_defers_boundary_to_real4_distance_squared():
    mapper=PrognosticMapper((1,1),np.array([[0,0,0,0]]),np.ones((1,4)),(1,1),
        np.array([[.00000500005,.5]],dtype=np.float32),
        np.array([[0.,0.]],dtype=np.float32))
    # Double distance is just above 500 m, but CMPD2 rounds x²+y² to exactly
    # 250000 in REAL*4 and includes it at SIGMAP=0.5 km.  Weighted CUTP then
    # clears the value; premature KDTree exclusion would return .005 fallback.
    np.testing.assert_array_equal(mapper.radius_interpolate(
        np.array([[.005]],dtype=np.float32),.5,method=2,cutoff=.01),
        np.zeros((1,1),dtype=np.float32))


def test_interpqr_automatic_sigma_retains_unhalved_distance_squared():
    mapper=PrognosticMapper((1,2),np.array([[1,0,0,0]]),np.ones((1,4)),(1,1),
        np.array([[0.,0.],[1.,0.]],dtype=np.float32),
        np.array([[.75,0.]],dtype=np.float32))
    result=mapper.radius_interpolate(np.array([[0.,10.]],dtype=np.float32),
                                     0.,method=3,cutoff=0.)
    distance2=np.array([750.**2,250.**2],dtype=np.float32)
    decay=np.exp(np.asarray(-distance2/np.float32(1000.**2),dtype=np.float32)).astype(np.float32)
    numerator=np.float32(np.float32(0.)+
                         np.float32(np.float32(10.)*decay[1]/distance2[1]))
    denominator=np.float32(np.float32(decay[0]/distance2[0])+
                           np.float32(decay[1]/distance2[1]))
    np.testing.assert_array_equal(result,np.array([[np.float32(numerator/denominator)]]))

def test_interpqr_uses_ordered_real4_divisions_like_fortran():
    coordinates=np.asarray([[1.001,0.],[2.003,0.],[3.007,0.]],dtype=np.float32)
    mapper=PrognosticMapper((1,3),np.array([[0,1,2,2]]),np.ones((1,4))/4.,(1,1),
        coordinates,np.array([[0.,0.]]))
    values=np.asarray([[.1,0.,.1]],dtype=np.float32)
    top=np.float32(0.); bottom=np.float32(0.)
    for value,coordinate in zip(values.ravel(),coordinates[:,0]):
        distance=np.float32(coordinate*np.float32(1000.))
        distance2=np.float32(distance*distance)
        top=np.float32(top+np.float32(value/distance2))
        bottom=np.float32(bottom+np.float32(np.float32(1.)/distance2))
    expected=np.float32(top/bottom)
    np.testing.assert_array_equal(mapper.radius_interpolate(values,4.,method=2),expected)


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


def test_prognostic_sst_nearest_ocean_tie_keeps_fortran_scan_order():
    mapper=PrognosticMapper((2,2),np.array([[0,1,2,3]]),np.ones((1,4))/4.,(1,1),
        np.array([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.]],dtype=np.float32),
        np.array([[0.,0.]],dtype=np.float32))
    field=np.array([[10.,30.],[20.,40.]],dtype=np.float32)
    # F-order is Fortran's I-outer/J-inner RDHD5 scan: (0,0), (1,0), ... .
    np.testing.assert_array_equal(
        mapper.nearest_masked(field,np.ones((2,2),dtype=bool)),[[10.]])


def test_prognostic_four_neighbours_keep_fortran_scan_order_on_ties():
    source=np.array([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.],
                     [-2.,0.],[0.,-2.],[2.,0.],[0.,2.]],dtype=np.float32)
    got=_fortran_nearest_indices(source,np.array([[0.,0.]],dtype=np.float32),4)
    np.testing.assert_array_equal(got,[[0,1,2,3]])


def test_wind1_initial_guess_applies_group_bias_to_all_station_weights():
    points=np.array([[.5,.5],[2.5,.5],[.5,2.5]])
    u=np.array([[1.,3.,9.],[2.,4.,10.]])
    v=np.zeros_like(u)
    got_u,got_v=wind1_initial_guess(u,v,points,2,1,1,1.,1.,bias=[0.,.5])
    assert got_u.dtype==np.float32 and got_v.dtype==np.float32
    np.testing.assert_allclose(got_u[:,0,0],[8./3.,41./6.])
    np.testing.assert_allclose(got_v,0.)


def test_wind1_initial_guess_removes_calm_and_barrier_by_level():
    points=np.array([[.5,.5],[.5,2.5]])
    u=np.array([[0.,8.],[0.,10.]])
    v=np.zeros_like(u)
    got_u,_=wind1_initial_guess(u,v,points,1,1,1,1.,1.,extrapolate_calm=False,
        barriers=((-1.,1.,2.,1.),),barrier_top_level=1)
    # The upper station is blocked in layer 1, while the calm surface station
    # is removed aloft and the barrier no longer applies in layer 2.
    np.testing.assert_allclose(got_u[:,0,0],[0.,10.])


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


def test_wind1_weights_match_gnu_real4_grid_normalization():
    points=np.array([[.41325,.7445],[8.91275,-3.811125],[-1.7255,6.200875]])
    # Fortran RESHAPE fills the first dimension first.
    u=np.array([[1.2345,8.125,-3.1415],[-2.7182,2.3456,9.875]])
    v=np.array([[-4.25,.75,6.25],[5.5,-5.125,1.5]])
    got_u,got_v=wind1_initial_guess(u,v,points,2,1,1,3.7,3.7,bias=[.23,-.37])
    np.testing.assert_array_equal(got_u[:,0,0].view(np.uint32),np.array(
        [0xbe024a40,0x3cd17ec0],dtype=np.uint32))
    np.testing.assert_array_equal(got_v[:,0,0].view(np.uint32),np.array(
        [0x3f4015f6,0x4063a8b7],dtype=np.uint32))


def test_wind1_sequential_calm_and_missing_removal_matches_gnu_bits():
    points=np.array([[.41325,.7445],[8.91275,-3.811125],[-1.7255,6.200875],
                     [4.55025,1.51075],[12.0205,-7.20025]])
    u=np.array([[0.,2.,3.,4.,5.],[0.,2.25,-3.5,999.,7.75]])
    v=np.array([[0.,1.,2.,3.,4.],[0.,-1.5,4.25,999.,-2.75]])
    got_u,got_v=wind1_initial_guess(u,v,points,2,1,1,3.7,3.7,
                                     bias=[0.,.17],extrapolate_calm=False)
    assert got_u[1,0,0].view(np.uint32)==np.uint32(0xbf870e8c)
    assert got_v[1,0,0].view(np.uint32)==np.uint32(0x40138421)

def test_vertical_layer_average_integrates_linear_profile_exactly():
    values=np.array([[0.,20.,40.],[10.,30.,50.]])
    heights=np.array([0.,100.,200.])
    got=vertical_layer_average(values,heights,[0.,50.,150.,200.])
    np.testing.assert_allclose(got,[[5.,20.,35.],[15.,30.,45.]])
    np.testing.assert_allclose(vertical_interpolate(values,heights,[50.,150.]),[[10.,30.],[20.,40.]])

def test_rdmm5_vertical_center_persistence_and_log_wind():
    height=np.array([[[20.]],[[100.]],[[200.]]]); value=np.array([[[2.]],[[10.]],[[20.]]])
    scalar=_vertical_centers(value,height,[10.,60.,300.])[:,0,0]
    wind=_vertical_centers(value,height,[10.,60.,300.],wind=True)[:,0,0]
    assert scalar.dtype==np.float32 and wind.dtype==np.float32
    np.testing.assert_allclose(scalar,[2.,6.,20.]); assert 0.<wind[0]<2.; np.testing.assert_allclose(wind[1:],[6.,20.])

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


def test_divcel_and_vertical_recurrence_preserve_real4_operation_order():
    u=np.array([[[1.125,-2.75,4.5],[.375,3.25,-1.5]],
                [[-3.5,.625,2.125],[5.75,-.875,1.25]]],dtype=np.float32)
    v=np.array([[[2.5,-1.125,.75],[-4.25,3.5,1.625]],
                [[.5,2.875,-3.25],[1.125,-2.5,4.75]]],dtype=np.float32)
    faces=np.array([0.,17.5,61.25],dtype=np.float32)
    west=u[:,:,0].copy(); east=u[:,:,-1].copy()
    south=v[:,0,:].copy(); north=v[:,-1,:].copy()
    got_div,got_w=divergence_and_vertical_velocity(
        u,v,faces,np.float32(713.),np.float32(887.),obrien=True,
        boundaries=(west,east,south,north))
    expected_div=np.empty_like(u); expected_w=np.zeros((3,2,3),dtype=np.float32)
    dxi=np.float32(1.)/(np.float32(2.)*np.float32(713.))
    dyi=np.float32(1.)/(np.float32(2.)*np.float32(887.))
    for k in range(2):
        for j in range(2):
            for i in range(3):
                up=east[k,j] if i==2 else u[k,j,i+1]
                um=west[k,j] if i==0 else u[k,j,i-1]
                vp=north[k,i] if j==1 else v[k,j+1,i]
                vm=south[k,i] if j==0 else v[k,j-1,i]
                dudx=np.float32((up-um)*dxi); dvdy=np.float32((vp-vm)*dyi)
                expected_div[k,j,i]=np.float32(dudx+dvdy)
        expected_w[k+1]=np.asarray(
            -((faces[k+1]-faces[k])*expected_div[k])+expected_w[k],dtype=np.float32)
    top=expected_w[-1].copy()
    for k in range(2):
        expected_w[k+1]=np.asarray(
            expected_w[k+1]-np.float32(faces[k+1]/faces[-1])*top,dtype=np.float32)
    np.testing.assert_array_equal(got_div,expected_div)
    np.testing.assert_array_equal(got_w,expected_w)

def test_minim_vector_sweeps_reduce_divergence():
    rng=np.random.default_rng(7); u=rng.normal(size=(2,8,9)); v=rng.normal(size=u.shape)
    boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    initial,w=divergence_and_vertical_velocity(u,v,[0.,20.,50.],1000.,1000.,obrien=True,boundaries=boundaries)
    adjusted_u,adjusted_v,final,iterations=minimize_divergence(
        u,v,w,[0.,20.,50.],1000.,1000.,boundaries=boundaries,maximum_iterations=50,limit=5.e-6)
    assert np.max(np.abs(final))<=5.e-6 and np.all(iterations>0)
    assert np.max(np.abs(final))<np.max(np.abs(initial))
    assert np.isfinite(adjusted_u).all() and np.isfinite(adjusted_v).all()


def test_minim_vector_sweeps_match_fortran_cell_loops():
    rng=np.random.default_rng(17); u=rng.normal(size=(1,3,4)).astype(np.float32); v=rng.normal(size=u.shape).astype(np.float32)
    faces=np.array([0.,30.],dtype=np.float32); dx=np.float32(700.); dy=np.float32(900.)
    boundaries=(u[:,:,0].copy(),u[:,:,-1].copy(),v[:,0,:].copy(),v[:,-1,:].copy())
    _,w=divergence_and_vertical_velocity(u,v,faces,dx,dy,obrien=True,boundaries=boundaries)
    expected_u=u.copy(); expected_v=v.copy()
    west,east,south,north=(item.copy() for item in boundaries)

    def divergence():
        up=np.concatenate((west[0,:,None],expected_u[0],east[0,:,None]),axis=1)
        vp=np.concatenate((south[0,None,:],expected_v[0],north[0,None,:]),axis=0)
        dxi=np.float32(1.)/(np.float32(2.)*dx)
        dyi=np.float32(1.)/(np.float32(2.)*dy)
        horizontal=(up[:,2:]-up[:,:-2])*dxi+(vp[2:]-vp[:-2])*dyi
        return np.asarray(horizontal+(w[1].astype(np.float32)-w[0].astype(np.float32))
                          /(faces[1]-faces[0]),dtype=np.float32)

    div=divergence()
    for idir in range(4):
        irange=range(4) if idir in {0,2} else range(3,-1,-1)
        jrange=range(3) if idir in {0,1} else range(2,-1,-1)
        for j in jrange:
            for i in irange:
                ut=np.float32(-np.float32(2.)*(div[j,i]*dx)/np.float32(2.))
                vt=np.float32(-np.float32(2.)*(div[j,i]*dy)/np.float32(2.))
                if i>0: expected_u[0,j,i-1]=np.float32(expected_u[0,j,i-1]-np.float32(.5)*ut)
                if i<3: expected_u[0,j,i+1]=np.float32(expected_u[0,j,i+1]+np.float32(.5)*ut)
                if j>0: expected_v[0,j-1,i]=np.float32(expected_v[0,j-1,i]-np.float32(.5)*vt)
                if j<2: expected_v[0,j+1,i]=np.float32(expected_v[0,j+1,i]+np.float32(.5)*vt)
        west[0]=expected_u[0,:,0]; east[0]=expected_u[0,:,-1]
        south[0]=expected_v[0,0]; north[0]=expected_v[0,-1]
        div=divergence()

    got_u,got_v,got_div,iterations=minimize_divergence(
        u,v,w,faces,dx,dy,boundaries=boundaries,maximum_iterations=1,limit=0.)
    np.testing.assert_array_equal(got_u,expected_u)
    np.testing.assert_array_equal(got_v,expected_v)
    np.testing.assert_array_equal(got_div[0],div)
    np.testing.assert_array_equal(iterations,1)

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


def test_elustr_wind_speed_uses_fortran_sum_then_sqrt_rounding():
    shape=(1,1)
    ustar,_,_=surface_similarity(
        np.array([[8.861122]],dtype=np.float32),
        np.array([[-9.195294]],dtype=np.float32),np.full(shape,.25),10.,
        np.full(shape,1.2),np.full(shape,290.),np.zeros(shape),
        np.zeros(shape),np.full(shape,20))
    # diagnostics/code/elustr_speed_reference.f90 prints WS then USTAR.
    assert ustar.view(np.uint32).item()==np.uint32(0x3fb13dee)


def test_elustr_stable_refinement_uses_fortran_105_percent_exit():
    shape=(1,1); one=np.ones(shape,dtype=np.float32)
    ustar,length,qh=surface_similarity(
        one*np.float32(20.),np.zeros(shape,dtype=np.float32),
        one*np.float32(.25),10.,one*np.float32(1.2),one*np.float32(290.),
        np.zeros(shape,dtype=np.float32),one*np.float32(-.1),
        np.full(shape,20))
    # diagnostics/code/elustr_stable_reference.f90 prints USTAR, QH and EL.
    assert ustar.view(np.uint32).item()==np.uint32(0x400aad81)
    assert qh.view(np.uint32).item()==np.uint32(0xc26fa15d)
    assert length.view(np.uint32).item()==np.uint32(0x466a6256)


def test_elustr_cloud_index_square_matches_gnu_default_real_bits():
    shape=(1,1)
    _,_,qh=surface_similarity(
        np.full(shape,np.float32(-2.7476854),dtype=np.float32),
        np.full(shape,np.float32(6.6769905),dtype=np.float32),
        np.full(shape,np.float32(.25),dtype=np.float32),10.,
        np.full(shape,np.float32(1.2205907),dtype=np.float32),
        np.full(shape,np.float32(288.83032),dtype=np.float32),
        np.full(shape,np.float32(.94914865),dtype=np.float32),
        np.full(shape,np.float32(-.1),dtype=np.float32),
        np.full(shape,20,dtype=np.int32))
    # GNU CALMET grid cell (I=364,J=73) has JCC=9.  Evaluating JCC**2 as
    # default REAL gives THETA1=0x3d5b5740 and this final QH bit pattern;
    # NumPy's old float32/int32 promotion path produced 0xc248bebc.
    assert qh.view(np.uint32).item()==np.uint32(0xc248bebd)


def test_elustr_daytime_cube_and_convergence_match_fortran():
    shape=(1,1)
    u=np.full(shape,-1.012058973312378,dtype=np.float32)
    v=np.full(shape,-2.737722635269165,dtype=np.float32)
    ustar,length,_=surface_similarity(
        u,v,np.full(shape,.25,dtype=np.float32),10.,
        np.full(shape,1.235607385635376,dtype=np.float32),
        np.full(shape,285.5619201660156,dtype=np.float32),
        np.ones(shape,dtype=np.float32),
        np.full(shape,25.949554443359375,dtype=np.float32),
        np.full(shape,20))
    # Iteration 4 is just above the 0.5% threshold when REAL**3 is lowered
    # as two REAL*4 multiplies, so ELUSTR performs iteration 5.
    # See diagnostics/code/elustr_daytime_reference.f90.
    assert ustar.view(np.uint32).item()==np.uint32(0x3eac6a5b)
    assert length.view(np.uint32).item()==np.uint32(0xc303c7d2)


def test_elustr_daytime_logf_matches_low_wind_fortran_cell():
    shape=(1,1)
    ustar,length,_=surface_similarity(
        np.full(shape,np.float32(-.18413368),dtype=np.float32),
        np.full(shape,np.float32(.15045358),dtype=np.float32),
        np.full(shape,np.float32(.8815),dtype=np.float32),10.,
        np.full(shape,np.float32(1.22742593),dtype=np.float32),
        np.full(shape,np.float32(287.876495),dtype=np.float32),
        np.zeros(shape,dtype=np.float32),
        np.full(shape,np.float32(129.466507),dtype=np.float32),
        np.full(shape,40))
    # GNU CALMET grid cell (I=323,J=77): its ALOG result is one ulp above
    # NumPy's log ufunc, which changes every subsequent ELUSTR iteration.
    assert ustar.view(np.uint32).item()==np.uint32(0x3e5ee029)
    assert length.view(np.uint32).item()==np.uint32(0xbe2b490b)

def test_inter2_observation_and_step1_inverse_square_weights():
    values=np.array([[10.,20.]])
    # The single target is at (0.5, 0.5), one and two km from stations.
    points=np.array([[1.5,.5],[2.5,.5]])
    result=objective_analysis_component(values,points,np.zeros((1,1,1)),np.array([[20]]),1.,1.,
        rmin_km=.1,rmax_surface_km=5.,surface_scale_km=10.,stations_per_level=2)
    assert result.dtype==np.float32
    expected=(10./1.**2+20./2.**2)/(1./1.**2+1./2.**2+1./10.**2)
    assert result.item()==pytest.approx(expected)


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
