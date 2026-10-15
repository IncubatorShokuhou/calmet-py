from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from pycalmet import CalmetConfig, GridConfig
from pycalmet.calmet_listing import (CalmetListingWriter, format_grid,
                                     format_full_prepdi_debug,
                                     format_no_upper_prepdi_debug,
                                     format_surface_input_debug,
                                     format_upper_air_input_debug)
from pycalmet.upper_air import Sounding
from pycalmet.legacy._compat.calmet_listing import (
    format_upper_air_input_debug as format_legacy_upper_air_input_debug,
)
from pycalmet.legacy._compat.upper_air import Sounding as LegacySounding


def test_line_printer_scales_signs_and_reverses_y_rows():
    text=format_grid(np.array([[.01234,-.05678],[.09,0.]],dtype=np.float32),
                     'Diagnostic field',significant_digits=3,
                     begin=datetime(2018,3,18),timestep_seconds=3600)
    assert 'Multiply all values by 10 **  -3' in text
    rows=[line for line in text.splitlines() if line.startswith(('   1 I','   2 I'))]
    assert rows==['   2 I 90   0 ','   1 I 12  57 ']
    signs=[line for line in text.splitlines() if line.startswith('     I')]
    assert signs==['     I  +   + ','     I  +   - ']
    assert 'FROM: year: 2018' in text and 'hour:  1' in text


def test_line_printer_zero_field_uses_fortran_short_form():
    text=format_grid(np.zeros((2,3)), 'All zero')
    assert 'GRID NOT PRINTED -- all values zero' in text


def test_surface_input_debug_reproduces_station_table():
    record=SimpleNamespace(begin=datetime(2018,3,18),
        end=datetime(2018,3,18,1),wind_speed=np.array([1.25]),
        wind_direction=np.array([270.]),ceiling_hundreds_ft=np.array([12]),
        cloud_tenths=np.array([7]),temperature_k=np.array([288.15]),
        relative_humidity=np.array([65]),pressure_mb=np.array([1001.2]),
        precipitation_code=np.array([0]))
    text=format_surface_input_debug(record,
        [SimpleNamespace(identifier=58457)])
    assert 'YYYYJJJHH:  201807700' in text
    assert 'to -- YYYYJJJHH:  201807701' in text
    assert 'Station ID' in text
    assert '    58457       1.25   270.0' in text
    with pytest.raises(ValueError,match='station count'):
        format_surface_input_debug(record,[])


def test_no_upper_prepdi_debug_reproduces_domain_and_station_winds():
    step=SimpleNamespace(debug_domain_temperature=289.125,
        debug_station_names=('STA1','SEA2'),
        debug_station_u=np.array([1.25,-2.5]),
        debug_station_v=np.array([3.5,4.75]))
    text=format_no_upper_prepdi_debug(step,datetime(2018,3,18,1,15))
    assert 'Year: 2018   Julian day:  77   Hour:  1   Second:  900' in text
    assert 'TINF) =     289.12' in text
    assert 'STA1        1        1.25      3.50' in text
    assert 'SEA2        2       -2.50      4.75' in text
    assert 'upper air data not yet defined' in text
    step.debug_domain_temperature=None
    text=format_no_upper_prepdi_debug(step,datetime(2018,3,18,1))
    assert 'TINF) = N/A (not available)' in text


def test_upper_air_input_debug_reproduces_raw_and_layer_profiles():
    sounding=Sounding(datetime(2018,3,18),
        np.array([1000.,900.]),np.array([100.,1100.]),
        np.array([288.15,281.15]),np.array([270.,180.]),np.array([5.,10.]))
    text=format_upper_air_input_debug(2,sounding,0.,
        np.array([4.5,6.]),np.array([-.5,1.]),
        np.array([288.15,285.,281.]))
    assert 'Julian Day: 77  Hour: 0 (GMT)' in text
    assert '   2  1000.0      0.0  288.15' in text
    assert 'U -- cell averaged' in text and '     4.500' in text
    assert 'V -- cell averaged' in text and '    -0.500' in text
    assert 'T -- cell face' in text and '   288.150' in text


def test_legacy_upper_air_listing_does_not_convert_kelvin_twice():
    sounding=LegacySounding(
        datetime(2018,3,18),np.array([1000.,900.]),np.array([100.,1100.]),
        np.array([288.15,281.15]),np.array([270.,180.]),np.array([5.,10.]),
    )

    text=format_legacy_upper_air_input_debug(
        2,sounding,0.,np.array([4.5,6.]),np.array([-.5,1.]),
        np.array([288.15,285.,281.]),
    )

    assert '   2  1000.0      0.0  288.15' in text
    assert '561.30' not in text


def test_full_prepdi_debug_reproduces_domain_and_upper_station_fields():
    station=SimpleNamespace(name='UP01')
    step=SimpleNamespace(debug_domain_temperature=289.25,
        debug_domain_u=2.5,debug_domain_v=-1.25,
        debug_station_names=('S001',),debug_station_u=np.array([1.]),
        debug_station_v=np.array([2.]),
        debug_upper_data=((station,(),0.,np.array([3.,4.]),
                           np.array([-2.,-1.])),))
    text=format_full_prepdi_debug(step,datetime(2018,3,18,1),
                                  np.array([[-.0065]]))
    assert 'GAMMA) =   -0.00650' in text
    assert 'TINF) =     289.25' in text
    assert '(UM) =      2.500' in text and '(VM) =     -1.250' in text
    assert 'Upper air station: UP01' in text
    assert '          1       3.00    -2.00' in text
    assert 'Multiply all values' not in text
    step.debug_domain_temperature=None
    text=format_full_prepdi_debug(step,datetime(2018,3,18,1),
                                  np.array([[-.0065]]))
    assert 'TINF) = N/A (not available)' in text


def test_line_printer_validates_shape_and_digits():
    with pytest.raises(ValueError,match='two-dimensional'):
        format_grid(np.zeros(2),'bad')
    with pytest.raises(ValueError,match='1 through 5'):
        format_grid(np.zeros((1,1)),'bad',significant_digits=0)


def test_listing_writer_honors_interval_and_selected_fields(tmp_path):
    config=CalmetConfig('met','geo','out',datetime(2018,3,18).isoformat(),
        datetime(2018,3,18,2).isoformat(),GridConfig(2,1,1.,1.),
        z_faces_m=(0.,20.),print_fields=True,print_interval_hours=2,
        print_uv_levels=(1,),print_w_levels=(0,),print_temperature_levels=(0,),
        print_surface_fields=(0,0,0,0,0,1,0,0))
    array=np.ones((1,2),dtype=np.float32)
    step=SimpleNamespace(cloud_fraction=array,ceiling_height=array,
                         precipitation_rate=array*2)
    surface=SimpleNamespace(friction_velocity=array,monin_obukhov_length=array,
        mixing_height=array,convective_mixing_height=array,
        sensible_heat_flux=array,density=array,
        water_convective_velocity_scale=None)
    wind=SimpleNamespace(u=array[None],v=array[None]*3,
                         w=np.zeros((2,1,2),dtype=np.float32))
    first=SimpleNamespace(timestamp=datetime(2018,3,18),wind=wind,step1=step,
                          surface=surface,temperature=array[None]*280)
    second=SimpleNamespace(timestamp=datetime(2018,3,18,1),wind=wind,step1=step,
                           surface=surface,temperature=array[None]*280)
    path=tmp_path/'calmet.lst'
    with CalmetListingWriter(path,config,np.full((1,2),10),(50,55)) as writer:
        writer.append(first,array)
        writer.append(second,array)
    text=path.read_text(encoding='ascii')
    assert 'U-component (m/s) -- Level:   1' in text
    assert 'V-component (m/s) -- Level:   1' in text
    assert 'Precipitation rate (mm/hr)' in text
    assert text.count('FROM: year: 2018')==3


def test_listing_interval_counts_hours_not_substeps(tmp_path):
    config=CalmetConfig('met','geo','out','2018-03-18_00:00:00',
        '2018-03-18_02:00:00',GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),
        timestep_seconds=900,print_fields=True,print_interval_hours=2,
        print_uv_levels=(1,),print_w_levels=(0,),print_temperature_levels=(0,))
    array=np.ones((1,1),dtype=np.float32)
    step=SimpleNamespace(cloud_fraction=array,ceiling_height=array,
                         precipitation_rate=array)
    surface=SimpleNamespace(friction_velocity=array,monin_obukhov_length=array,
        mixing_height=array,convective_mixing_height=array,
        sensible_heat_flux=array,density=array,
        water_convective_velocity_scale=None)
    wind=SimpleNamespace(u=array[None],v=array[None],
                         w=np.zeros((2,1,1),dtype=np.float32))
    path=tmp_path/'subhour.lst'
    with CalmetListingWriter(path,config,np.full((1,1),10),(50,55)) as writer:
        for minutes in range(0,121,15):
            hour=SimpleNamespace(timestamp=datetime(2018,3,18)+
                timedelta(minutes=minutes),wind=wind,step1=step,
                surface=surface,temperature=array[None])
            writer.append(hour,array)
    text=path.read_text(encoding='ascii')
    assert text.count('U-component')==1
    assert 'hour:  1  second:    0' in text


def test_noobs_micrometeorology_debug_uses_requested_steps_and_preaverage_fields(tmp_path):
    config=CalmetConfig('met','geo','out','2018-03-18_00:00:00',
        '2018-03-18_02:00:00',GridConfig(1,1,1.,1.),z_faces_m=(0.,20.),
        noobs=2,
        micrometeorology_debug=True,debug_first_step=2,debug_last_step=2,
        print_surface_fields=(0,0,0,1,0,0,0,1))
    array=np.ones((1,1),dtype=np.float32)
    surface=SimpleNamespace(density=array*1.2,shortwave_flux=array*100.,
        temperature_jump=array*3.,mixing_height=array*999.,
        convective_mixing_height=array*888.,friction_velocity=array,
        monin_obukhov_length=array,sensible_heat_flux=array,
        water_convective_velocity_scale=None,
        unaveraged_mixing_height=array*123.,
        unaveraged_convective_mixing_height=array*45.)
    step=SimpleNamespace(cloud_fraction=array,ceiling_height=array,
                         precipitation_rate=array)
    wind=SimpleNamespace(u=array[None],v=array[None],
                         w=np.zeros((2,1,1),dtype=np.float32))
    path=tmp_path/'debug.lst'
    with CalmetListingWriter(path,config,np.full((1,1),10),(50,55)) as writer:
        for hour in range(2):
            record=SimpleNamespace(timestamp=datetime(2018,3,18,hour),
                wind=wind,step1=step,surface=surface,temperature=array[None])
            writer.append(record,array*.5)
    text=path.read_text(encoding='ascii')
    assert text.count('Computed solar elevation angle data')==1
    assert 'Hour:  1' in text
    assert 'No observation mode' in text
    assert 'Mixing height (m) -- Before averaging' in text
    assert 'Convective mixing height (m) -- Before averaging' in text
    assert 'Temperature jump at top of mixed layer' in text
