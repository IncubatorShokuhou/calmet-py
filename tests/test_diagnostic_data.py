from datetime import datetime

import pytest

from pycalmet.diagnostic_data import DiagnosticData


def test_diagnostic_data_reads_prepdi_scalar_record_order(tmp_path):
    path=tmp_path/'diag.dat'
    path.write_text(
        'TINF      280.25\n'
        'GAMMA     -6.5 \n'
        'U WIND     2.0 \n'
        'V WIND    -3.0 \n'
        'TINF      281.25\n'
        'GAMMA     -7.5 \n'
        'U WIND     4.0 \n'
        'V WIND    -5.0 \n',encoding='ascii')
    data=DiagnosticData(path,datetime(2018,1,1),2,3600,
        surface_temperature=True,lapse_rate=True,initial_wind=True)
    first=data.at_begin(datetime(2018,1,1))
    second=data.at_begin(datetime(2018,1,1,1))
    assert first.surface_temperature_k==280.25
    assert first.lapse_rate_k_m==pytest.approx(-.0065)
    assert (first.initial_u_m_s,first.initial_v_m_s)==(2.,-3.)
    assert second.surface_temperature_k==281.25
    assert second.lapse_rate_k_m==pytest.approx(-.0075)
    assert (second.initial_u_m_s,second.initial_v_m_s)==(4.,-5.)


def test_diagnostic_data_rejects_short_and_missing_records(tmp_path):
    path=tmp_path/'diag.dat'; path.write_text('short\n',encoding='ascii')
    with pytest.raises(ValueError,match='shorter'): DiagnosticData(
        path,datetime(2018,1,1),1,3600,surface_temperature=True)
    path.write_text('U WIND     2.0 \n',encoding='ascii')
    with pytest.raises(ValueError,match='initial V'): DiagnosticData(
        path,datetime(2018,1,1),1,3600,initial_wind=True)


def _surface_line(name,*values):
    return ' '*15+f'{name:<4}'+' '+''.join(f'{value:5.1f}' for value in values)+'\n'


def _upper_line(name,*values):
    return ' '*13+f'{name:<4}'+' '+''.join(f'{value:5.1f}' for value in values)+'\n'


def test_diagnostic_data_reads_preprocessed_station_wind_blocks(tmp_path):
    path=tmp_path/'diag.dat'
    path.write_text(
        _surface_line('SFC1',1.,2.,-3.)+
        _surface_line('SEA1',.5,999.,999.)+
        ' '*15+'LAST\n'+
        _upper_line('SFC1',1.,10.,11.,12.,13.)+
        _upper_line('UPR2',1.,20.,21.,22.,23.)+
        _upper_line('LAST',0.,0.,0.,0.,0.),encoding='ascii')
    data=DiagnosticData(path,datetime(2018,1,1),1,3600,
        surface_wind_names=('SFC1','SEA1'),upper_wind_names=('SFC1','UPR2'),
        levels=2,iextrp=2)
    record=data.at_begin(datetime(2018,1,1))
    assert record.surface_u_m_s.tolist()==[2.,9999.]
    assert record.surface_v_m_s.tolist()==[-3.,9999.]
    # Same-name upper data are omitted only in layer one.
    assert record.upper_u_m_s.tolist()==[[9999.,20.],[12.,22.]]
    assert record.upper_v_m_s.tolist()==[[9999.,21.],[13.,23.]]


def test_diagnostic_data_consumes_enabled_empty_station_blocks(tmp_path):
    path=tmp_path/'diag.dat'
    path.write_text(' '*15+'LAST\n'+_upper_line('LAST',0.,0.,0.,0.,0.),encoding='ascii')
    data=DiagnosticData(path,datetime(2018,1,1),1,3600,
        surface_wind=True,upper_wind=True,levels=2)
    record=data.at_begin(datetime(2018,1,1))
    assert record.surface_u_m_s.shape==(0,)
    assert record.surface_v_m_s.shape==(0,)
    assert record.upper_u_m_s.shape==(2,0)
    assert record.upper_v_m_s.shape==(2,0)
