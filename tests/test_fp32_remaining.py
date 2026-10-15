from dataclasses import fields
from datetime import datetime

import numpy as np

from pycalmet._makegeo import _fields
from pycalmet.coare import (advance_warm_layer,initial_warm_layer_state,
                            surface_flux)
from pycalmet.config import GridConfig,MakegeoConfig
from pycalmet.igf_data import (_r2interpi_four,_rdcalmet_rotate,
                               _rdcalmet_vertical)
from pycalmet.precip_data import grid_precipitation,unpack_precipitation
from pycalmet.prog_data import ProgData,ProgRecord
from pycalmet.sea_data import grid_sea_mixing_height
from pycalmet.surface_data import _wind_components
from pycalmet.upper_air import Sounding
from pycalmet.weight_data import read_weight_data


def test_makegeo_physical_fields_default_to_float32():
    fractions=np.zeros((38,1,1),dtype=np.float64)
    fractions[0]=1.
    config=MakegeoConfig('terrain','landuse','geo',GridConfig(1,1,1.,1.))
    dominant,*physical=_fields(fractions,config)
    assert np.issubdtype(dominant.dtype,np.integer)
    assert all(field.dtype==np.float32 for field in physical)


def test_legacy_meteorology_adapters_return_float32(tmp_path):
    grid=GridConfig(1,1,1.,1.,x_origin_km=.5,y_origin_km=.5)
    record=ProgRecord(1.,0.,0.,1.,np.array([10.,110.]),
        np.array([[[100.]],[[200.]]]),np.array([[[300.]],[[500.]]]))
    u,v=ProgData.interpolate_record(record,grid,[60.])
    assert u.dtype==v.dtype==np.float32

    rain=grid_precipitation([1.,2.],[(0.,0.),(1.,1.)],grid)
    assert rain.dtype==np.float32
    assert unpack_precipitation([1.,-2.],3).dtype==np.float32

    path=tmp_path/'wt.dat'
    path.write_text(
        'header one\nheader two\nheader three\n'
        '               0.0 0.0 1 1 1.0\n'
        '               0.0 0.0 1 1 1.0\n\n\n'
        '            10.0000\n      1\n      0.25\n      end\n',
        encoding='ascii')
    assert read_weight_data(path,1,1,[10.]).dtype==np.float32


def test_initial_guess_interpolation_uses_float32_fields():
    points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    interpolated=_r2interpi_four(points,[[.5,.5]],[[1.,2.,3.,4.]])
    vertical=_rdcalmet_vertical(
        np.array([[[1.]],[[3.]]]),[10.,30.],[5.,20.,40.],[[.1]])
    rotated=_rdcalmet_rotate([1.],[0.],[10.])
    assert interpolated.dtype==np.float32
    assert vertical.dtype==np.float32
    assert all(component.dtype==np.float32 for component in rotated)


def test_station_observation_fields_default_to_float32():
    grid=GridConfig(1,1,1.,1.)
    sea=grid_sea_mixing_height([100.],[(.5,.5)],grid)
    wind=_wind_components([5.],[270.])
    sounding=Sounding(
        datetime(2018,1,1),np.array([1000.,900.]),np.array([100.,200.]),
        np.array([293.15,288.15]),np.array([270.,270.]),np.array([5.,6.]))
    converted=sounding.converted()
    assert sea.dtype==np.float32
    assert wind[0].dtype==wind[1].dtype==np.float32
    assert all(value.dtype==np.float32 for value in converted)


def test_coare_fields_are_float32_but_warm_accumulators_are_float64():
    flux=surface_flux([5.],[288.15],[290.15],[.8])
    assert all(getattr(flux,item.name).dtype==np.float32 for item in fields(flux))

    state=initial_warm_layer_state((1,))
    adjusted,state,warming=advance_warm_layer(
        state,[8.*3600.],[300.],[.6],[700.],[350.])
    assert adjusted.dtype==warming.dtype==np.float32
    assert state.heat_column_j_m2.dtype==np.float64

