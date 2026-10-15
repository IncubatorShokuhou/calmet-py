import numpy as np

from pycalmet.calmet_numerics import (
    cloud_fraction,
    cloud_liquid_water_profile,
    convective_height_batchvarova_gryning,
    coriolis_parameter,
    density_from_pressure_temperature,
    heat_flux,
    lapse_rate,
    mixing_height_maul_carson,
    objective_analysis_component,
    psi_momentum_dyer,
    saturation_vapor_pressure_lowe,
    surface_similarity,
    upwind_mixing_average,
    wind_speed_direction,
)
from pycalmet.calmet_solver import (
    PrognosticMapper,
    _cloud_water_g_kg,
    _source_relative_humidity,
)


def _assert_fp32(*values):
    assert all(np.asarray(value).dtype == np.float32 for value in values)


def test_ordinary_calmet_kernels_return_fp32_from_float64_inputs():
    density=density_from_pressure_temperature([1000.0],[290.0])
    coriolis=coriolis_parameter([30.0])
    speed,direction=wind_speed_direction([3.0],[4.0])
    cloud=cloud_fraction([80.0])
    vapor=saturation_vapor_pressure_lowe([20.0])
    mapped,average,bottom,top=cloud_liquid_water_profile(
        np.array([.002,.01,.02,.004])[:,None,None],
        np.array([100.,300.,700.,1300.])[:,None,None],
        [0.,400.,800.])
    heat,shortwave=heat_flux(
        [[.8]],[[.25]],[[290.]],[[20]],[[.15]],[[1.]],[[.15]],[[0.]])

    _assert_fp32(density,coriolis,speed,direction,cloud,vapor,
                 mapped,average,bottom,top,heat,shortwave)


def test_double_precision_reductions_return_fp32_model_fields():
    shape=(1,1)
    profile_temperature=np.array([290.,289.,287.])[:,None,None]
    profile_height=np.array([0.,100.,300.])[:,None,None]
    total,convective,jump=mixing_height_maul_carson(
        profile_temperature,profile_height,np.full(shape,50.),np.zeros(shape),
        np.full(shape,.2),np.full(shape,-100.),np.full(shape,100.),
        np.full(shape,1.2),np.full(shape,1.e-4))
    analyzed=objective_analysis_component(
        np.array([[10.]]),np.array([[.5,.5]]),np.zeros((1,1,1)),
        np.array([[20]]),1.,1.,stations_per_level=1)
    ustar,length,sensible=surface_similarity(
        [[3.]],[[4.]],[[.1]],10.,[[1.2]],[[290.]],[[.2]],[[100.]],[[20]])
    averaged,averaged_convective=upwind_mixing_average(
        np.ones(shape),np.zeros(shape),np.full(shape,500.),
        np.full(shape,300.),1000.,minimum_radius_cells=1)

    _assert_fp32(total,convective,jump,analyzed,ustar,length,sensible,
                 averaged,averaged_convective)


def test_mapper_keeps_geometric_coordinates_double_and_model_results_single():
    mapper=PrognosticMapper(
        (1,2),np.array([[0,1,1,1]]),np.full((1,4),.25),
        (1,1),np.array([[0.,0.],[1.,0.]],dtype=np.float64),
        np.array([[.5,0.]],dtype=np.float64))
    field=np.array([[1.25,2.75]],dtype=np.float64)

    assert mapper.source_coordinates.dtype == np.float64
    assert mapper.target_coordinates.dtype == np.float64
    _assert_fp32(mapper.interpolate(field),mapper.radius_interpolate(field,2.))


def test_mapper_normalizes_coordinates_to_fp64_and_masked_values_to_fp32():
    mapper=PrognosticMapper(
        (1,2),np.array([[0,1,1,1]]),np.full((1,4),.25),
        (1,1),np.array([[0.,0.],[1.,0.]],dtype=np.float32),
        np.array([[.5,0.]],dtype=np.float32))
    field=np.array([[1.25,2.75]],dtype=np.float64)

    assert mapper.source_coordinates.dtype == np.float64
    assert mapper.target_coordinates.dtype == np.float64
    _assert_fp32(mapper.nearest_masked(field,np.ones((1,2),dtype=bool)))


def test_upwind_average_uses_the_requested_timestep():
    mixing=np.arange(7,dtype=np.float32)[None,:]
    u=np.ones_like(mixing)
    v=np.zeros_like(mixing)

    subhour=upwind_mixing_average(
        u,v,mixing,mixing,1000.,minimum_radius_cells=0,
        timestep_s=900.)[0]
    hourly=upwind_mixing_average(
        u,v,mixing,mixing,1000.,minimum_radius_cells=0,
        timestep_s=3600.)[0]

    np.testing.assert_array_equal(subhour,mixing)
    assert not np.array_equal(subhour,hourly)


def test_batchvarova_gryning_honors_configured_maximum_height():
    height=convective_height_batchvarova_gryning(
        [50.],[.01],[.001],[.5],[-100.],[290.],maximum_height_m=100.)

    assert height.dtype == np.float32
    np.testing.assert_array_equal(height,np.array([100.],dtype=np.float32))


def test_nonspatial_lapse_mean_accumulates_in_fp64_and_returns_fp32():
    desired=np.array([[.01,1.e-8,-.01]],dtype=np.float32)
    temperature=np.stack((np.zeros_like(desired),
                          np.float32(200.)*desired))
    convective_height=np.zeros_like(desired)
    centers=np.array([0.,200.],dtype=np.float32)
    spatial=lapse_rate(
        temperature,convective_height,centers,100.,spatial=True)
    expected=np.float32(np.mean(spatial,dtype=np.float64))

    averaged=lapse_rate(
        temperature,convective_height,centers,100.,spatial=False)

    assert averaged.dtype == np.float32
    np.testing.assert_array_equal(averaged,np.full_like(spatial,expected))


def test_stability_function_defaults_to_fp32_but_supports_fp64_convergence():
    z_over_l=np.array([-.5,.5],dtype=np.float64)

    ordinary=psi_momentum_dyer(z_over_l)
    convergence=psi_momentum_dyer(z_over_l,dtype=np.float64)

    assert ordinary.dtype == np.float32
    assert convergence.dtype == np.float64
    np.testing.assert_allclose(ordinary,convergence,rtol=1.e-6,atol=1.e-7)


def test_canonical_fields_are_converted_without_m3d_decimal_quantization():
    cloud=_cloud_water_g_kg(np.array([1.23456e-3],dtype=np.float64))

    class Dataset:
        variables={'relative_humidity':np.array([[50.25]],dtype=np.float64)}

        def __getitem__(self,name):
            return self.variables[name]

    humidity=_source_relative_humidity(
        Dataset(),0,np.zeros(1),np.ones(1),np.ones(1))

    _assert_fp32(cloud,humidity)
    np.testing.assert_allclose(cloud,[1.23456],rtol=1.e-6)
    np.testing.assert_array_equal(humidity,[50.25])
