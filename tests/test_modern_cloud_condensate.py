import numpy as np

from pycalmet import GridConfig
from pycalmet.calmet_solver import build_step1_fields


class _Dataset:
    def __init__(self):
        horizontal=(1,3)
        profile=(1,3,*horizontal)
        self.variables={
            'latitude':np.full(horizontal,30.,dtype=np.float64),
            'longitude':np.array([[120.,120.01,120.02]],dtype=np.float64),
            'terrain':np.zeros(horizontal,dtype=np.float32),
            'height':np.broadcast_to(
                np.array([10.,100.,500.],dtype=np.float32)[None,:,None,None],
                profile).copy(),
            'wind_speed':np.ones(profile,dtype=np.float32),
            'wind_direction':np.zeros(profile,dtype=np.float32),
            'temperature':np.broadcast_to(
                np.array([290.,285.,275.],dtype=np.float32)[None,:,None,None],
                profile).copy(),
            'qvapor':np.full(profile,.005,dtype=np.float32),
            'pressure':np.broadcast_to(
                np.array([1000.,900.,800.],dtype=np.float32)[None,:,None,None],
                profile).copy(),
            'relative_humidity':np.full(profile,100.,dtype=np.float32),
            'landuse':np.ones(horizontal,dtype=np.int16),
            'temperature_2m':np.full((1,*horizontal),290.,dtype=np.float32),
            'surface_pressure':np.full((1,*horizontal),100000.,dtype=np.float32),
            'precipitation':np.zeros((1,*horizontal),dtype=np.float32),
            'pbl_height':np.full((1,*horizontal),100.,dtype=np.float32),
            'sensible_heat_flux':np.zeros((1,*horizontal),dtype=np.float32),
            'friction_velocity':np.full((1,*horizontal),.1,dtype=np.float32),
            'qcloud':np.array(
                [[[[4.e-6,4.e-6,-4.e-6]],[[0.,0.,0.]],[[0.,0.,0.]]]],
                dtype=np.float32),
            'qice':np.array(
                [[[[2.e-6,0.,0.]],[[0.,0.,6.e-6]],[[0.,0.,0.]]]],
                dtype=np.float32),
        }

    def __getitem__(self,name):
        return self.variables[name]


class _IdentityMapper:
    target_shape=(1,3)

    @staticmethod
    def interpolate(field,*,ignore_zero=False):
        return np.asarray(field,dtype=np.float32)

    @staticmethod
    def nearest(field):
        return np.asarray(field,dtype=np.float32)

    @staticmethod
    def radius_interpolate(field,*args,**kwargs):
        return np.asarray(field,dtype=np.float32)


def _build(dataset,threshold=.005):
    return build_step1_fields(
        dataset,0,GridConfig(3,1,1.,1.),[0.,50.,200.],
        mapper=_IdentityMapper(),cloud_method=4,
        cloud_condensate_threshold_g_kg=threshold)


def test_cloud_mask_uses_each_layers_liquid_plus_ice_condensate():
    result=_build(_Dataset())

    np.testing.assert_allclose(
        result.integrated_cloud_water,[[.006,.004,.006]],rtol=2.e-6)
    assert result.cloud_fraction[0,0]>0.
    assert result.cloud_fraction[0,1]==0.
    assert result.cloud_fraction[0,2]>0.
    np.testing.assert_array_equal(result.ceiling_height,[[10.,0.,100.]])


def test_cloud_mask_accepts_ice_when_liquid_variable_is_absent():
    dataset=_Dataset()
    dataset.variables.pop('qcloud')

    result=_build(dataset)

    np.testing.assert_allclose(
        result.integrated_cloud_water,[[.002,0.,.006]],rtol=2.e-6)
    np.testing.assert_array_equal(result.cloud_fraction[0,:2],0.)
    assert result.cloud_fraction[0,2]>0.
    assert result.cloud_liquid_water_3d is None


def test_cloud_mask_uses_configured_physical_threshold():
    result=_build(_Dataset(),threshold=.007)

    np.testing.assert_array_equal(result.cloud_fraction,0.)
    np.testing.assert_array_equal(result.ceiling_height,0.)


def test_cloud_mask_keeps_relative_humidity_fallback_without_condensate_fields():
    dataset=_Dataset()
    dataset.variables.pop('qcloud')
    dataset.variables.pop('qice')

    result=_build(dataset)

    assert np.all(result.cloud_fraction>0.)
    assert result.integrated_cloud_water is None
