from pathlib import Path
import re

import pycalmet
import pytest
from pycalmet.control import (_angle,_date,assignments,calmet_config,calwrf_config,ctgproc_config,
                              grid_from_assignments,makegeo_config,terrel_config)
from pycalmet.config import CtgprocSource

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=Path(__file__).with_name('fixtures')
CALWRF_CONTROL=FIXTURES/'calwrf.inp'
CALMET_CONTROL=FIXTURES/'calmet.inp'


def test_assignment_reader_ignores_inactive_asterisk_fields(tmp_path):
    control=tmp_path/'sample.inp'
    control.write_text('! NX = 4 !\n* NY = 99 *\n! NX = 5 ! !END!\n',encoding='ascii')
    assert assignments(control)=={'NX':['4','5']}


def test_directional_angles_accept_legacy_prefix_and_wrap_longitude():
    assert _angle('N 27')==27.
    assert _angle('243E')==-117.


def test_common_grid_control_preserves_ttm_scale_factor():
    grid=grid_from_assignments({'NX':['2'],'NY':['3'],'DGRIDKM':['1'],
                                'PMAP':['TTM'],'TMSCALE':['0.9996']})
    assert grid.ttm_scale==pytest.approx(.9996)


def test_calmet_python_defaults_match_legacy_model_not_bundled_case():
    config=pycalmet.CalmetConfig('met.nc','geo.nc','out.nc','2018010100','2018010101',
                                 pycalmet.GridConfig(1,1,1.,1.))
    assert config.mnmdav==1
    assert config.maximum_mixing_height_m==3000.
    assert config.maximum_water_mixing_height_m==3000.
    assert config.precipitation_radius_km==100.
    radiation=(990.,-30.,-.75,3.4,5.31e-13,60.,.12)
    assert (config.turbidity_coefficient_1,config.turbidity_coefficient_2,
            config.cloudiness_coefficient_1,config.cloudiness_coefficient_2,
            config.net_radiation_coefficient_1,config.net_radiation_coefficient_2,
            config.net_radiation_coefficient_3)==radiation
    bundled=calmet_config(CALMET_CONTROL)
    assert (bundled.turbidity_coefficient_1,bundled.turbidity_coefficient_2,
            bundled.cloudiness_coefficient_1,bundled.cloudiness_coefficient_2,
            bundled.net_radiation_coefficient_1,bundled.net_radiation_coefficient_2,
            bundled.net_radiation_coefficient_3)==radiation


def test_calmet_control_missing_values_use_model_defaults(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    for name in ('MNMDAV','ZIMAX','ZIMAXW','SIGMAP'):
        text,count=re.subn(rf'!\s*{name}\s*=\s*[^!]*!',
                           lambda match: match.group().replace('!','*'),text,count=1)
        assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    config=calmet_config(path)
    assert (config.mnmdav,config.maximum_mixing_height_m,
            config.maximum_water_mixing_height_m,
            config.precipitation_radius_km)==(1,3000.,3000.,100.)


def test_calmet_control_uses_pacdat_for_mesopac_output(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text,count=re.subn(r'\*\s*PACDAT\s*=\s*[^*]*\*',
                       '! PACDAT = products/custom.pac !',text,count=1)
    assert count==1
    text,count=re.subn(r'!\s*IFORMO\s*=\s*[^!]*!',
                       '! IFORMO = 2 !',text,count=1)
    assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    config=calmet_config(path)
    assert config.output==tmp_path/'calmet.dat'
    assert config.mesopac_output==tmp_path/'products/custom.pac'


def test_calmet_control_honors_uppercase_file_policy(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text,count=re.subn(r'!\s*LCFILES\s*=\s*[^!]*!',
                       '! LCFILES = F !',text,count=1)
    assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    config=calmet_config(path)
    assert config.geophysical==tmp_path/'GEO.DAT'
    assert config.meteorology==tmp_path/'CALWRFOUT_2018031800_2018031812.M3D'


def test_calmet_control_restores_readfn_default_file_names(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text=text.replace('! NM3D =  1  !','! NM3D =  2  !',1)
    text=text.replace('! NIGF =  0  !','! NIGF =  2  !',1)
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    config=calmet_config(path)
    assert config.meteorology==(
        tmp_path/'calwrfout_2018031800_2018031812.m3d',tmp_path/'mm52.dat')
    assert config.initial_guess_calmet_files==(tmp_path/'igf1.dat',tmp_path/'igf2.dat')
    assert config.geophysical==tmp_path/'geo.dat'
    assert config.surface_data==tmp_path/'surf.dat'
    assert config.precipitation_data==tmp_path/'precip.dat'
    assert config.cloud_data==tmp_path/'cloud.dat'
    assert config.diagnostic_data==tmp_path/'diag.dat'
    assert config.weight_data==tmp_path/'wt.dat'


def test_calmet_control_enables_default_coast_distance_grid(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text=text.replace('! LDBCST = F !','! LDBCST = T !',1)
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    assert calmet_config(path).distance_to_coast_output==tmp_path/'dcst.grd'


@pytest.mark.parametrize('name',('XORIGKM','YORIGKM','RLAT0','RLON0','XLAT1','XLAT2'))
def test_calmet_control_rejects_missing_mandatory_lcc_grid_field(tmp_path,name):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text,count=re.subn(rf'!\s*{name}\s*=\s*[^!]*!',
                       lambda match: match.group().replace('!','*'),text,count=1)
    assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match=rf'mandatory grid fields:.*{name}'):
        calmet_config(path)


def test_calmet_missing_projection_uses_fortran_utm_default_and_requires_zone(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    for name in ('PMAP','IUTMZN'):
        text,count=re.subn(rf'!\s*{name}\s*=\s*[^!]*!',
                           lambda match: match.group().replace('!','*'),text,count=1)
        assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='mandatory grid fields:.*IUTMZN'):
        calmet_config(path)


def test_calmet_modern_control_requires_explicit_timezone(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text,count=re.subn(r'!\s*ABTZ\s*=\s*[^!]*!',
                       lambda match: match.group().replace('!','*'),text,count=1)
    assert count==1
    path=tmp_path/'calmet.inp'; path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='2.2 control requires explicit ABTZ'):
        calmet_config(path)


@pytest.mark.parametrize(('relative','reader','program'),(
    ('examples/zhejiang/config/terrel.inp',terrel_config,'TERREL'),
    ('examples/zhejiang/config/ctgproc.inp',ctgproc_config,'CTGPROC'),
    ('examples/zhejiang/config/makegeo.inp',makegeo_config,'MAKEGEO'),
))
def test_preprocessor_controls_require_explicit_reference_origin(
        tmp_path,relative,reader,program):
    text=(ROOT/relative).read_text(encoding='ascii')
    text,count=re.subn(r'!\s*XREFKM\s*=\s*[^!]*!',
                       lambda match: match.group().replace('!','*'),text,count=1)
    assert count==1
    path=tmp_path/Path(relative).name; path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match=rf'{program} control lacks.*XREFKM'):
        reader(path)


@pytest.mark.parametrize(('relative','reader','count_name'),(
    ('examples/zhejiang/config/terrel.inp',terrel_config,'NTDF'),
    ('examples/zhejiang/config/ctgproc.inp',ctgproc_config,'NDBF'),
))
def test_preprocessor_controls_validate_declared_source_count(
        tmp_path,relative,reader,count_name):
    text=(ROOT/relative).read_text(encoding='ascii')
    text,count=re.subn(rf'!\s*{count_name}\s*=\s*[^!]*!',
                       f'! {count_name} = 999 !',text,count=1)
    assert count==1
    path=tmp_path/Path(relative).name; path.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match=rf'{count_name} declares 999 files'):
        reader(path)


def test_ctgproc_legacy_zero_ndbf_uses_luindat_and_lulc(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! NDBF = 1 !','! NDBF = 0 !')
    text+='\n! LUINDAT = legacy-landuse.dat !\n! LULC = 2 !\n! IGLAZR = 5 !\n'
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.landuse_sources==(CtgprocSource(
        tmp_path/'LEGACY-LANDUSE.DAT','GLAZAF','ESR-S',12),)
    assert config.input_type=='GLAZAF'


def test_ctgproc_legacy_glaz_requires_valid_region(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! NDBF = 1 !','! NDBF = 0 !')
    text+='\n! LULC = 2 !\n! IGLAZR = 0 !\n'
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='IGLAZR from 1 through 6'):
        ctgproc_config(control)


def test_calmet_control_hour_24_rolls_to_next_day():
    values={'IBYR':['2018'],'IBMO':['12'],'IBDY':['31'],'IBHR':['24'],
            'IBSEC':['3600']}
    assert _date(values,'IB')=='2019-01-01_01:00:00'
    values['IBHR']=['25']
    with pytest.raises(ValueError,match='between 0 and 24'): _date(values,'IB')


def test_calmet_control_preserves_fractional_utc_label(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text=text.replace('! ABTZ= UTC-0000 !','! ABTZ= UTC-0830 !')
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    config=calmet_config(control)
    assert config.base_timezone==8
    assert config.base_timezone_label=='UTC-0830'


def test_pre21_calmet_control_converts_hour_ending_period_and_ibtz(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text='CALMET.INP      2.0             legacy hour-ending controls\n'+text.split('\n',1)[1]
    for name in ('IBSEC','IEYR','IEMO','IEDY','IEHR','IESEC'):
        text=re.sub(rf'!\s*{name}\s*=\s*[^!]*!',lambda match: match.group().replace('!','*'),text)
    for name in ('MCLOUD','ICLDOUT'):
        text=re.sub(rf'!\s*{name}\s*=\s*[^!]*!',lambda match: match.group().replace('!','*'),text)
    text+='\n! IRLG = 3 !\n! IBTZ = 8 !\n! ICLOUD = 4 !\n!END!\n'
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    config=calmet_config(control)
    assert config.start=='2018-03-17_23:00:00'
    assert config.end=='2018-03-18_02:00:00'
    assert config.base_timezone==8


def test_pre21_calmet_control_rejects_explicit_time_fields(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text='CALMET.INP      2.0             inconsistent controls\n'+text.split('\n',1)[1]
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='pre-2.1.*explicit'):
        calmet_config(control)


def test_calwrf_control_preserves_inclusive_subdomain_ranges(tmp_path):
    lines=(CALWRF_CONTROL).read_text(encoding='ascii').splitlines()
    lines[3]='2,7,3,8,4,9'
    control=tmp_path/'calwrf.inp'; control.write_text('\n'.join(lines)+'\n',encoding='ascii')
    config=calwrf_config(control)
    assert config.i_range==(2,7) and config.j_range==(3,8) and config.k_range==(4,9)


def test_calwrf_control_preserves_fixed_title_and_listing_records(tmp_path):
    lines=(CALWRF_CONTROL).read_text(encoding='ascii').splitlines()
    lines[0]='CALWRF title with embedded spaces'
    lines[1]='reports/custom-calwrf.lst ! Log file name'
    control=tmp_path/'calwrf.inp'; control.write_text('\n'.join(lines)+'\n',encoding='ascii')
    config=calwrf_config(control)
    assert config.title=='CALWRF title with embedded spaces'
    assert config.listing_output==tmp_path/'reports/custom-calwrf.lst'


def test_calwrf_control_honors_optional_2d_output_switch(tmp_path):
    text=(CALWRF_CONTROL).read_text(encoding='ascii')+'0\n'
    control=tmp_path/'calwrf.inp'; control.write_text(text,encoding='ascii')
    assert not calwrf_config(control).create_surface


def test_calwrf_control_accepts_inline_comments_and_inactive_paths(tmp_path):
    lines=CALWRF_CONTROL.read_text(encoding='ascii').splitlines()
    lines[4]='-2018051700 ! use all available records'
    lines[5]='-2018051712 ! use all available records'
    lines[7:]=['! ignored/wrf.nc', 'active/wrf.nc ! selected input']
    control=tmp_path/'calwrf.inp'
    control.write_text('\n'.join(lines)+'\n',encoding='ascii')
    config=calwrf_config(control)
    assert config.start=='-2018051700'
    assert config.wrf_files==[tmp_path/'active/wrf.nc']


def test_calwrf_output_override_disables_hidden_control_side_outputs(tmp_path):
    lines=(CALWRF_CONTROL).read_text(encoding='ascii').splitlines()
    lines[7]='inputs/wrf.nc'
    control=tmp_path/'calwrf.inp'
    control.write_text('\n'.join(lines)+'\n',encoding='ascii')
    redirected=calwrf_config(control,output=tmp_path/'candidate.nc')
    assert redirected.output==tmp_path/'candidate.nc'
    assert redirected.legacy_output is None
    assert redirected.surface_output is None
    assert redirected.listing_output is None
    explicit=calwrf_config(control,output=tmp_path/'candidate.nc',
                           legacy_output=tmp_path/'candidate.m3d',
                           listing_output=tmp_path/'candidate.lst')
    assert explicit.legacy_output==tmp_path/'candidate.m3d'
    assert explicit.listing_output==tmp_path/'candidate.lst'


def test_terrel_output_override_disables_hidden_control_side_outputs(tmp_path):
    redirected=terrel_config(ROOT/'examples/zhejiang/config/terrel.inp',output=tmp_path/'candidate.dat')
    assert redirected.output==tmp_path/'candidate.dat'
    assert redirected.save_output is None
    assert redirected.plot_output is None
    assert redirected.raw_echo_output is None
    explicit=terrel_config(ROOT/'examples/zhejiang/config/terrel.inp',output=tmp_path/'candidate.dat',
                           save_output=tmp_path/'candidate.sav')
    assert explicit.save_output==tmp_path/'candidate.sav'


def test_makegeo_output_override_disables_hidden_control_side_outputs(tmp_path):
    redirected=makegeo_config(ROOT/'examples/zhejiang/config/makegeo.inp',output=tmp_path/'candidate.dat')
    assert redirected.output==tmp_path/'candidate.dat'
    assert redirected.landuse_plot_output is None
    assert redirected.terrain_plot_output is None
    assert redirected.color_map_output is None
    explicit=makegeo_config(ROOT/'examples/zhejiang/config/makegeo.inp',output=tmp_path/'candidate.dat',
                            color_map_output=tmp_path/'candidate.clr')
    assert explicit.color_map_output==tmp_path/'candidate.clr'


def test_calmet_control_rejects_nz_zface_count_mismatch(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text=text.replace('! NZ =  10  !','! NZ =   9  !',1)
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='NZ.*ZFACE'):
        calmet_config(control)

def test_calmet_control_maps_legacy_icloud_to_new_cloud_controls(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text='CALMET.INP      2.1             legacy cloud control\n'+text.split('\n',1)[1]
    text=text.replace('! ICLDOUT = 0  !','* ICLDOUT = 0  *')
    text=text.replace('! MCLOUD =  4  !','* MCLOUD =  4  *')
    text+='\n! ICLOUD = 1 ! !END!\n'
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    config=calmet_config(control)
    assert config.cloud_method==1 and config.cloud_output==1


def test_calmet_22_rejects_missing_new_cloud_controls(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text=text.replace('! ICLDOUT = 0  !','* ICLDOUT = 0  *')
    text=text.replace('! MCLOUD =  4  !','* MCLOUD =  4  *')
    text+='\n! ICLOUD = 4 ! !END!\n'
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    with pytest.raises(ValueError,match='2.2.*MCLOUD.*ICLDOUT'):
        calmet_config(control)


def test_calmet_21_uses_icloud_even_if_new_fields_are_present(tmp_path):
    text=(CALMET_CONTROL).read_text(encoding='ascii')
    text='CALMET.INP      2.1             legacy cloud control\n'+text.split('\n',1)[1]
    text+='\n! ICLOUD = 2 ! !END!\n'
    control=tmp_path/'calmet.inp'; control.write_text(text,encoding='ascii')
    config=calmet_config(control)
    assert config.cloud_method==2 and config.cloud_output==0


def test_existing_component_controls_map_to_typed_configs():
    calwrf=calwrf_config(CALWRF_CONTROL)
    assert calwrf.start=='2018031800' and len(calwrf.wrf_files)==1
    assert Path(calwrf.output).suffix=='.nc'
    assert Path(calwrf.legacy_output).suffix=='.m3d'
    assert Path(calwrf.surface_output).suffix=='.m2d'
    assert calwrf.title=='Create 3D.DAT file for WRF output'
    assert calwrf.listing_output==FIXTURES/'calwrf.lst'
    assert calwrf.i_range is None and calwrf.j_range is None and calwrf.k_range is None

    terrel=terrel_config(ROOT/'examples/zhejiang/config/terrel.inp')
    assert (terrel.grid.nx,terrel.grid.ny,terrel.datum)==(610,499,'WGS-84')
    assert terrel.model==1 and terrel.processing_method==2
    assert terrel.plot_output==ROOT/'examples/zhejiang/config/QATERR.GRD'
    assert terrel.raw_echo_output is None and terrel.qa_hit_threshold_percent==75
    assert len(terrel.terrain_files)==42 and Path(terrel.terrain_files[0]).name=='N26E117.HGT'
    assert len(terrel.terrain_sources)==42
    assert terrel.terrain_sources[0].input_type=='SRTM3'
    assert terrel.terrain_sources[0].input_datum=='WGS-96'

    ctg=ctgproc_config(ROOT/'examples/zhejiang/config/ctgproc.inp')
    assert ctg.input_type=='GLAZAS' and ctg.mesh_density==12 and ctg.input_datum=='ESR-S'

    geo=makegeo_config(ROOT/'examples/zhejiang/config/makegeo.inp')
    assert geo.title=='Demo Application'
    assert geo.grid.lon0==117.5 and Path(geo.terrain).name.casefold()=='terr.dat'
    assert (geo.water_threshold,geo.missing_category,geo.minimum_coverage)==(.5,55,.96)
    assert geo.secondary_landuse is None and geo.use_terrain
    assert len(geo.input_properties)==38 and geo.input_properties[11]==(.5,.18,1.,.2,0.,1.,10)
    assert tuple(geo.water_categories)==(51,52,53,54,55)
    assert len(geo.output_categories)==14 and not geo.category_splits
    assert geo.qa_cell==(20,15) and not geo.snow_processing
    assert geo.landuse_plot_output==ROOT/'examples/zhejiang/config/qaluse.grd'
    assert geo.terrain_plot_output==ROOT/'examples/zhejiang/config/qaterr.grd'
    assert geo.color_map_output==ROOT/'examples/zhejiang/config/luse.clr'
    assert geo.image_mode==0

    calmet=calmet_config(CALMET_CONTROL)
    assert calmet.start=='2018-03-18_00:00:00'
    assert calmet.end=='2018-03-18_12:00:00'
    assert calmet.base_timezone==0 and calmet.iprog==14
    assert calmet.regulatory_mode==0
    assert calmet.run_type==1 and not calmet.include_w_temperature
    assert calmet.save_output and calmet.legacy_output_format==1 and calmet.test_mode==2
    assert calmet.timestep_seconds==3600
    assert calmet.prognostic_timestep_seconds==3600
    assert calmet.cloud_method==4 and calmet.cloud_output==0
    assert calmet.precipitation_station_count==-1
    assert calmet.critical_froude==1. and tuple(calmet.smoothing_passes)==(2,4,4,4,4,4,4,4,4,4)
    assert calmet.divergence_limit==5.e-6 and calmet.divergence_iterations==50
    assert calmet.wind_extrapolation==-1 and calmet.varying_radius
    assert calmet.upper_air_separation_cells==-1. and calmet.extrapolate_calm==0
    assert tuple(calmet.wind_bias)==(0.,)*10
    assert calmet.diagnostic_data==FIXTURES/'diag.dat'
    assert (calmet.diagnostic_surface_temperature,calmet.diagnostic_lapse_rate,
            calmet.diagnostic_initial_wind,calmet.diagnostic_surface_wind,
            calmet.diagnostic_upper_wind)==(0,0,0,0,0)
    assert calmet.use_initial_guess_calmet==0
    assert not calmet.initial_guess_calmet_files
    assert calmet.initial_guess_upper_station==-1
    assert tuple(calmet.initial_guess_height_range_m)==(1.,1000.)
    assert (calmet.minimum_station_radius_km,calmet.maximum_surface_radius_km,
            calmet.maximum_upper_radius_km,calmet.maximum_water_radius_km)==(.1,30.,30.,50.)
    assert tuple(calmet.stations_per_level)==(5,)*10 and tuple(calmet.wind_extrapolation_factors)==(0.,)*10
    assert calmet.kinematic_alpha==.1
    assert calmet.mixing_wind_level==1 and calmet.half_average_angle==30.
    assert (calmet.constb,calmet.conste,calmet.constn,calmet.constw)==(1.41,.15,2400.,.16)
    assert calmet.coriolis==1.e-4
    assert (calmet.minimum_lapse_rate,calmet.mixing_lapse_depth_m)==(.001,200.)
    assert (calmet.minimum_mixing_height_m,calmet.maximum_mixing_height_m)==(50.,2500.)
    assert (calmet.minimum_water_mixing_height_m,calmet.maximum_water_mixing_height_m)==(50.,2500.)
    assert (calmet.land_flux_threshold,calmet.convective_relaxation,calmet.convective_relaxation_time_s)==(0.,1,800.)
    assert (calmet.water_flux_threshold,calmet.water_temperature_mode,
            calmet.prognostic_ocean_category)==(.05,0,16)
    assert (calmet.overwater_flux_method,calmet.coastal_shelf_length_km,
            calmet.warm_layer,calmet.cool_skin)==(10,0.,0,0)
    assert (calmet.precipitation_interpolation,calmet.precipitation_radius_km,calmet.precipitation_cutoff)==(2,50.,.01)
    assert (calmet.surface_temperature_option,calmet.upper_temperature_option,calmet.lapse_height_m)==(-1,-1,200.)
    assert (calmet.temperature_interpolation_power,calmet.temperature_radius_km,
            calmet.temperature_station_limit)==(1,20.,5)
    assert calmet.sea_files==()
    assert calmet.surface_control_stations[0]==('BGR ',14606,514.5,4960.533,5.,10.)
    assert len(calmet.surface_control_stations)==5
    assert calmet.upper_control_stations==()
    assert calmet.precipitation_control_stations==()
    assert calmet.weight_data==FIXTURES/'wt.dat'
    assert calmet.prognostic_radius_km==0.
    assert (calmet.water_temperature_gradient_below,calmet.water_temperature_gradient_above)==(-.0098,-.0045)
    assert (calmet.water_temperature_category_min,calmet.water_temperature_category_max)==(55,55)
    assert (calmet.iwfcod,calmet.ifradj,calmet.ikine,calmet.iobr,calmet.islope)==(1,1,0,0,1)
    assert any('GEODAT=GEO.DAT' in line for line in calmet.control_lines)
    assert list(calmet.z_faces_m)==[0.,20.,40.,80.,160.,300.,600.,1000.,1500.,2200.,3000.]


def test_calmet_iprog1_control_selects_progdat_instead_of_m3ddat(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! IPROG =  14  !','! IPROG =  1  !')
    source=source.replace('PROG.DAT      input      * PRGDAT=                  *',
                          'PROG.DAT      input      ! PRGDAT= PROG.DAT !')
    control=tmp_path/'calmet.inp'
    control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert config.iprog==1
    assert Path(config.meteorology)==tmp_path/'prog.dat'


def test_calmet_control_parses_lake_breeze_regions(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    replacements={
        '! LLBREZE = F !':'! LLBREZE = T !','! NBOX =  0  !':'! NBOX = 1 !',
        '! XG1 = 0. !':'! XG1 = 1. !','! XG2 = 0. !':'! XG2 = 3. !',
        '! YG1 = 0. !':'! YG1 = 2. !','! YG2 = 0. !':'! YG2 = 4. !',
        '! XBCST = 0. !':'! XBCST = 1.5 !','! YBCST = 0. !':'! YBCST = 2. !',
        '! XECST = 0. !':'! XECST = 1.5 !','! YECST = 0. !':'! YECST = 4. !',
        '! NLB =  0 !':'! NLB = 1 !','! METBXID =  0 !':'! METBXID = 14606 !',
    }
    for old,new in replacements.items():
        assert old in source
        source=source.replace(old,new,1)
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert config.lake_breeze_regions==(
        (1.,3.,2.,4.,1500.,2000.,1500.,4000.,(14606,)),)


@pytest.mark.parametrize(('source','reader','attribute','expected'),(
    (CALWRF_CONTROL,calwrf_config,'start','2018031800'),
    (ROOT/'examples/zhejiang/config/terrel.inp',terrel_config,'grid.nx',610),
    (ROOT/'examples/zhejiang/config/ctgproc.inp',ctgproc_config,'mesh_density',12),
    (ROOT/'examples/zhejiang/config/makegeo.inp',makegeo_config,'grid.ny',499),
    (CALMET_CONTROL,calmet_config,'iprog',14),
))
def test_list_summary_follows_recorded_control(tmp_path,source,reader,attribute,expected):
    control=tmp_path/Path(source).name
    control.write_text(Path(source).read_text(encoding='ascii'),encoding='ascii')
    listing=tmp_path/f'{reader.__name__}.lst'
    listing.write_text(f'Run summary\nControl File Used\n{control.name}\n',encoding='ascii')
    value=reader(listing)
    for name in attribute.split('.'):
        value=getattr(value,name)
    assert value==expected


def test_empty_list_file_is_rejected_explicitly(tmp_path):
    listing=tmp_path/'empty.lst'
    listing.write_text('',encoding='ascii')
    with pytest.raises(ValueError,match='list file is empty'):
        calmet_config(listing)


def test_failed_partial_calmet_listing_is_not_used_as_control(tmp_path):
    listing=tmp_path/'failed.lst'
    listing.write_text('ERROR IN SUBR. READIN\n',encoding='ascii')
    with pytest.raises(ValueError,match='truncated failed run'):
        calmet_config(listing)


def test_ctgproc_control_reads_preprocessed_coastline_fill(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LCOAST = F !','! LCOAST = T !').replace(
        '! LBLNREAD = F  !','! LBLNREAD = T  !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.coastline_file==tmp_path/'COAST.BLN'
    assert config.fill_empty_ocean and not config.coastline_swap
    assert config.ocean_category==55


def test_ctgproc_control_selects_direct_gshhs_extraction(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LCOAST = F !','! LCOAST = T !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.gshhs_file==tmp_path/'GSHHS_F.B'
    assert config.coastline_file==tmp_path/'COAST.BLN'
    assert config.coastline_source_datum=='WGS-84'


def test_ctgproc_control_reads_continuation_options(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LPREV = F  !','! LPREV = T  !').replace(
        '! LFINAL = T !','! LFINAL = F !').replace(
        '! PREVDAT =        !','! PREVDAT = PREV.DAT !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.previous_landuse==tmp_path/'PREV.DAT'
    assert not config.final_output


def test_ctgproc_control_uses_default_continuation_file(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LPREV = F  !','! LPREV = T  !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    assert ctgproc_config(control).previous_landuse==tmp_path/'PREV.DAT'


def test_ctgproc_control_honors_lowercase_file_policy(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LCFILES = F !','! LCFILES = T !').replace(
        '! LPREV = F  !','! LPREV = T  !').replace(
        '! PREVDAT =        !','! PREVDAT = Previous.DAT !').replace(
        '! LCOAST = F !','! LCOAST = T !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.landuse_sources[0].path==tmp_path/'eausgs1.img'
    assert config.output==tmp_path/'lulc.dat'
    assert config.previous_landuse==tmp_path/'previous.dat'
    assert config.gshhs_file==tmp_path/'gshhs_f.b'
    assert config.coastline_file==tmp_path/'coast.bln'


def test_ctgproc_control_reads_snow_and_qa_options(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! LSNOW = F  !','! LSNOW = T  !').replace(
        '! LQACELL =  F  !','! LQACELL =  T  !').replace(
        '! ITHRESH = 75 !','! ITHRESH = 60 !')
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert config.snow_processing
    assert config.qa_cell_output
    assert config.qa_hit_threshold_percent==60


def test_makegeo_control_reads_daily_snow_configuration(tmp_path):
    base=(ROOT/'examples/zhejiang/config/makegeo.inp').read_text(encoding='ascii')
    properties=assignments(ROOT/'examples/zhejiang/config/makegeo.inp')['X']
    extra=['! LSNOW = T !','! SNOWDAT = daily-snow.dat !',
           '! IDATEBEG = 20180101 !','! IDATEEND = 20180103 !',
           '! MSRL = 1 !','! HSCL = 0.4 !','! MSAL = 2 !','! SDPMIN = 0.02 !']
    extra.extend(f'! XS = {record} !' for record in properties)
    extra.extend(f'! XF = {record.split(",",1)[0]}, 5, 0.1 !' for record in properties)
    control=tmp_path/'makegeo.inp'
    control.write_text(base+'\n'+'\n'.join(extra)+'\n',encoding='ascii')
    config=makegeo_config(control)
    assert config.snow_processing
    assert config.snow_data==tmp_path/'daily-snow.dat'
    assert (config.snow_start_date,config.snow_end_date)==(20180101,20180103)
    assert config.snow_properties[11]==config.input_properties[11]
    assert config.snow_age_parameters[11]==(5,.1)
    assert (config.snow_roughness_method,config.snow_height_scale_m)==(1,.4)
    assert (config.snow_albedo_method,config.snow_depth_minimum_m)==(2,.02)


def test_makegeo_control_honors_uppercase_file_policy(tmp_path):
    text=(ROOT/'examples/zhejiang/config/makegeo.inp').read_text(encoding='ascii')
    text=text.replace('! LCFILES = T     !','! LCFILES = F     !').replace(
        '! LLU2 = F    !','! LLU2 = T    !')
    control=tmp_path/'makegeo.inp'; control.write_text(text,encoding='ascii')
    config=makegeo_config(control)
    assert config.terrain==tmp_path/'TERR.DAT'
    assert config.landuse==tmp_path/'LULC.DAT'
    assert config.secondary_landuse==tmp_path/'LUGLOBE.DAT'
    assert config.output==tmp_path/'GEO.DAT'
    assert config.landuse_plot_output==tmp_path/'QALUSE.GRD'
    assert config.terrain_plot_output==tmp_path/'QATERR.GRD'


def test_makegeo_control_uses_default_secondary_landuse_file(tmp_path):
    text=(ROOT/'examples/zhejiang/config/makegeo.inp').read_text(encoding='ascii')
    text=text.replace('! LU2DAT = Luglobe.dat      !',
                      '* LU2DAT = Luglobe.dat      *').replace(
        '! LLU2 = F    !','! LLU2 = T    !')
    control=tmp_path/'makegeo.inp'; control.write_text(text,encoding='ascii')
    assert makegeo_config(control).secondary_landuse==tmp_path/'lu2.dat'


def test_makegeo_control_preserves_a80_title_columns(tmp_path):
    lines=(ROOT/'examples/zhejiang/config/makegeo.inp').read_text(encoding='ascii').splitlines()
    lines[0]='  Regional application '+('x'*80)
    control=tmp_path/'makegeo.inp'
    control.write_text('\n'.join(lines)+'\n',encoding='ascii')
    assert makegeo_config(control).title==lines[0][:80]


def test_ctgproc_control_builds_mixed_sources_in_processing_order(tmp_path):
    text=(ROOT/'examples/zhejiang/config/ctgproc.inp').read_text(encoding='ascii')
    text=text.replace('! NDBF = 1 !','! NDBF = 3 !')
    text+='\n! CTG = extra.ctg !\n! CORINE = europe.tif !\n! DCORINE = WGS-84 !\n! MESHCORN = 3 !\n'
    control=tmp_path/'ctgproc.inp'; control.write_text(text,encoding='ascii')
    config=ctgproc_config(control)
    assert [item.input_type for item in config.landuse_sources]==['CTG','GLAZAS','CORINE']
    assert config.landuse_sources[0].path==tmp_path/'EXTRA.CTG'
    assert config.landuse_sources[0].input_datum=='NAS-C'
    assert config.landuse_sources[0].mesh_density==12
    assert config.landuse_sources[-1]==CtgprocSource(
        tmp_path/'EUROPE.TIF','CORINE','WGS-84',3)


def test_terrel_control_builds_mixed_sources_with_source_datums(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! NTDF = 42 !", "! NTDF = 43 !")
    text += "\n! GEN = terrain.gen !\n! DGEN = NAS-C !\n"
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.terrain_sources[-1].path == tmp_path / "TERRAIN.GEN"
    assert config.terrain_sources[-1].input_type == "GEN"
    assert config.terrain_sources[-1].input_datum == "NAS-C"


def test_terrel_control_reads_void_fill_options(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! LVOIDFIL = F !", "! LVOIDFIL = T !").replace(
        "! CELLRADKM = 5. !", "! CELLRADKM = 7. !"
    )
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.void_fill and config.cell_radius_km == 7.0


def test_terrel_control_builds_polar_ring_and_ray_grid(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! IGRID = 1 !", "! IGRID = 3 !")
    text = text.replace("! IMODEL = 1 !", "! IMODEL = 3 !")
    text = text.replace("! NRING =  0   !", "! NRING = 2 !")
    text = text.replace("! NRAYS =  0  !", "! NRAYS = 2 !")
    text += ("\n! DISKM = 1.0 !\n! DISKM = 2.0 !\n"
             "! ANGDEG = 10.0 !\n! ANGDEG = 190.0 !\n")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.grid.grid_type == 3
    assert (config.grid.nx, config.grid.ny) == (2, 2)
    assert config.grid.ring_distances_km == (1.0, 2.0)
    assert config.grid.ray_angles_deg == (10.0, 190.0)


def test_terrel_control_uses_source_defaults_for_receptors_and_cded(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! NTDF = 42 !", "! NTDF = 43 !")
    text = text.replace("! NXYCOL = 2!", "* NXYCOL = 2*")
    text = text.replace("! DCDED    = WGS-84    !", "* DCDED    = WGS-84    *")
    text += "\n! CDED = sample.dem !\n"
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.receptor_columns == 4
    cded=next(source for source in config.terrain_sources if source.input_type=='CDED')
    assert cded.input_datum == "NAR-C"


def test_terrel_control_uses_legacy_grid_and_save_defaults(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! IGRID = 1 !", "* IGRID = 1 *")
    text = text.replace("! SAVEFIL = TERR.SAV     !", "* SAVEFIL = TERR.SAV     *")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.grid.grid_type == 1
    assert config.save_output == tmp_path / "TERREL.SAV"


def test_makegeo_control_rejects_unsupported_ifmtgeo(tmp_path):
    text = (ROOT / "examples/zhejiang/config/makegeo.inp").read_text(encoding="ascii")
    control = tmp_path / "makegeo.inp"
    control.write_text(text + "\n! IFMTGEO = 2 !\n", encoding="ascii")
    with pytest.raises(ValueError, match="IFMTGEO=1"):
        makegeo_config(control)


def test_makegeo_control_uses_legacy_snow_parameter_defaults():
    config=makegeo_config(ROOT / "examples/zhejiang/config/makegeo.inp")
    assert config.snow_height_scale_m==10.
    assert config.snow_depth_minimum_m==.01


def test_makegeo_control_skips_optional_version_record_for_title(tmp_path):
    lines = (ROOT / "examples/zhejiang/config/makegeo.inp").read_text(encoding="ascii").splitlines()
    lines[:1] = ["MAKEGEO.INP 3.2", "Versioned regional title"]
    control = tmp_path / "makegeo.inp"
    control.write_text("\n".join(lines) + "\n", encoding="ascii")
    assert makegeo_config(control).title == "Versioned regional title"


def test_makegeo_control_ignores_zero_outcat_padding(tmp_path):
    text = (ROOT / "examples/zhejiang/config/makegeo.inp").read_text(encoding="ascii")
    text = text.replace("60, 61!", "60, 61, 0!")
    control = tmp_path / "makegeo.inp"
    control.write_text(text, encoding="ascii")
    config = makegeo_config(control)
    assert len(config.output_categories) == 14
    assert 0 not in config.output_categories


def test_terrel_file_override_rebuilds_control_sources(tmp_path):
    files=(tmp_path / "N30E120.HGT",tmp_path / "N30E121.HGT")
    config=terrel_config(ROOT / "examples/zhejiang/config/terrel.inp",terrain_files=files)
    assert tuple(config.terrain_files)==files
    assert tuple(item.path for item in config.terrain_sources)==files
    assert {(item.input_type,item.input_datum) for item in config.terrain_sources}=={
        ("SRTM3","WGS-96")}


def test_ctgproc_file_override_rebuilds_control_sources(tmp_path):
    files=(tmp_path / "replacement.img",)
    config=ctgproc_config(ROOT / "examples/zhejiang/config/ctgproc.inp",landuse_files=files)
    assert tuple(config.landuse_files)==files
    assert config.landuse_sources==(CtgprocSource(
        files[0],"GLAZAS","ESR-S",12),)


def test_ctgproc_source_setting_overrides_rebuild_control_sources():
    config=ctgproc_config(ROOT / "examples/zhejiang/config/ctgproc.inp",mesh_density=3,
                          input_datum="WGS-84")
    assert config.landuse_sources==(CtgprocSource(
        ROOT / "examples/zhejiang/config/EAUSGS1.IMG","GLAZAS","WGS-84",3),)


def test_terrel_center_reference_is_normalized_to_cell_corner(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! IGRID = 1 !", "! IGRID = 2 !")
    text = text.replace("! IMODEL = 1 !", "! IMODEL = 2 !")
    text = text.replace("! XREFKM = 0  !", "! XREFKM = 10 !")
    text = text.replace("! YREFKM = 0 !", "! YREFKM = 20 !")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    grid = terrel_config(control).grid
    assert grid.grid_type == 2
    assert (grid.x_origin_km, grid.y_origin_km) == (9.5, 19.5)


def test_terrel_calmet_rejects_cell_centre_reference(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! IGRID = 1 !", "! IGRID = 2 !")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match="CALMET output requires IGRID=1"):
        terrel_config(control)


def test_terrel_control_reads_discrete_receptor_options(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! LXY = F !", "! LXY = T !")
    text += "\n! XYINP = receptors.inp !\n! XYOUT = receptors.out !\n"
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.receptor_input == tmp_path / "RECEPTORS.INP"
    assert config.receptor_output == tmp_path / "RECEPTORS.OUT"
    assert config.receptor_columns == 2
    assert config.interpolate_receptors
    assert config.receptor_radius_km == 0.15


def test_terrel_control_uses_default_receptor_and_continuation_files(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! LXY = F !", "! LXY = T !")
    text = text.replace("! LPREV = F !", "! LPREV = T !")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.receptor_input == tmp_path / "XYINP.DAT"
    assert config.receptor_output == tmp_path / "XYOUT.DAT"
    assert config.previous_save == tmp_path / "PREV.SAV"


def test_terrel_control_reads_continuation_save_files(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! LPREV = F !", "! LPREV = T !")
    text += "\n! PREVFIL = previous.sav !\n"
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.previous_save == tmp_path / "PREVIOUS.SAV"
    assert config.save_output == tmp_path / "TERR.SAV"


def test_terrel_control_selects_direct_sample_projection(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! MSHEET = 0 !", "! MSHEET = 1 !")
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    assert terrel_config(control).sampling == "direct"


def test_terrel_control_reads_replacement_and_coastline_options(tmp_path):
    text = (ROOT / "examples/zhejiang/config/terrel.inp").read_text(encoding="ascii")
    text = text.replace("! LCOAST = F !", "! LCOAST = T !")
    text = text.replace("! LBLNREAD = F  !", "! LBLNREAD = T !")
    text = text.replace("! INOISEREP =   0,0   !", "! INOISEREP = 1,2 !")
    text = text.replace("!    ZNOISE = 0.,1.   !", "! ZNOISE = -2.,3. !")
    text = text.replace("! ITERREP =  3, 0 !", "! ITERREP = 2,1 !")
    text = text.replace("!  TERDEF =  0.,0. !", "! TERDEF = -1.,4. !")
    text += "\n! COASTBLN = coast.bln !\n"
    control = tmp_path / "terrel.inp"
    control.write_text(text, encoding="ascii")
    config = terrel_config(control)
    assert config.coastline_file == tmp_path / "COAST.BLN"
    assert config.gshhs_file is None
    assert config.noise_replacement == (1, 2, 0, 0, 0)
    assert config.noise_minimum_m == (-2.0, 3.0, 1.0, 1.0, 1.0)
    assert config.missing_replacement == (2, 1, 0, 0, 0)
    assert config.default_elevation_m == (-1.0, 4.0, 0.0, 0.0, 0.0)


def test_terrel_control_reads_raw_echo_output(tmp_path):
    text=(ROOT/'examples/zhejiang/config/terrel.inp').read_text(encoding='ascii')
    text=text.replace('! LRAWECHO = F !','! LRAWECHO = T !')
    text+='\n! RAWECHO = raw.dat !\n'
    control=tmp_path/'terrel.inp'; control.write_text(text,encoding='ascii')
    assert terrel_config(control).raw_echo_output==tmp_path/'RAW.DAT'


def test_terrel_control_honors_lowercase_file_policy(tmp_path):
    text=(ROOT/'examples/zhejiang/config/terrel.inp').read_text(encoding='ascii')
    text=text.replace('! LCFILES = F !','! LCFILES = T !').replace(
        '! LXY = F !','! LXY = T !').replace(
        '! LPREV = F !','! LPREV = T !').replace(
        '! LCOAST = F !','! LCOAST = T !').replace(
        '! LRAWECHO = F !','! LRAWECHO = T !')
    text+='\n! XYINP = Receptors.INP !\n! XYOUT = Receptors.OUT !\n'
    text+='! PREVFIL = Previous.SAV !\n! GSHHSIN = Shore.B !\n'
    text+='! COASTBLN = Coast.BLN !\n! RAWECHO = Raw.DAT !\n'
    control=tmp_path/'terrel.inp'; control.write_text(text,encoding='ascii')
    config=terrel_config(control)
    assert config.terrain_sources[0].path==tmp_path/'n26e117.hgt'
    assert config.output==tmp_path/'terr.dat'
    assert config.receptor_input==tmp_path/'receptors.inp'
    assert config.receptor_output==tmp_path/'receptors.out'
    assert config.previous_save==tmp_path/'previous.sav'
    assert config.save_output==tmp_path/'terr.sav'
    assert config.gshhs_file==tmp_path/'shore.b'
    assert config.coastline_file==tmp_path/'coast.bln'
    assert config.plot_output==tmp_path/'qaterr.grd'
    assert config.raw_echo_output==tmp_path/'raw.dat'


def test_calmet_subhour_seconds_and_timestep_are_parsed(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! IBSEC =  0  !','! IBSEC =  900  !').replace(
        '! IESEC =  0  !','! IESEC =  1800  !').replace('! NSECDT =  3600  !','! NSECDT =  900  !')
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert config.start=='2018-03-18_00:15:00'
    assert config.end=='2018-03-18_12:30:00'
    assert config.timestep_seconds==900


def test_calmet_control_parses_print_and_debug_options(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! LPRINT = F !','! LPRINT = T !').replace(
        '! IPRINF =  1  !','! IPRINF =  3  !').replace(
        '! IUVOUT = 0,0,0,0,0,0,0,0,0,0 !',
        '! IUVOUT = 1,0,0,0,0,0,0,0,0,1 !').replace(
        '!  MIXHT      =           0           !','! MIXHT = 1 !').replace(
        '! LDB = F !','! LDB = T !').replace('! NN2 =  1  !','! NN2 =  4  !').replace(
        '! IOUTD =  0  !','! IOUTD =  1  !').replace(
        '!  IPR8 =  0  !','! IPR8 =  1  !')
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert config.listing_output==tmp_path/'calmet.lst'
    assert config.print_fields and config.print_interval_hours==3
    assert tuple(config.print_uv_levels)==(1,0,0,0,0,0,0,0,0,1)
    assert tuple(config.print_surface_fields)==(0,0,0,1,0,0,0,0)
    assert config.micrometeorology_debug and config.debug_last_step==4
    assert config.wind_debug_output==1
    assert tuple(config.wind_debug_fields)==(0,0,0,0,0,0,0,0,1)
    assert config.final_wind_debug_output==tmp_path/'test.out'
    assert config.kinematic_wind_debug_output==tmp_path/'test.kin'
    assert config.froude_wind_debug_output==tmp_path/'test.frd'
    assert config.slope_wind_debug_output==tmp_path/'test.slp'
    assert config.wind_print_output==tmp_path/'test.prt'


def test_calmet_control_parses_prognostic_ocean_category(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    assert '! ILUOC3D =  16  !' in source
    source=source.replace('! ILUOC3D =  16  !','! ILUOC3D =  7  !',1)
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    assert calmet_config(control).prognostic_ocean_category==7


def test_calmet_control_preserves_multiple_meteorology_files(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! NM3D =  1  !','! NM3D =  2  !')
    source+='\nMM52.DAT input 1 ! M3DDAT = second.m3d ! !END!\n'
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert config.meteorology==(tmp_path/'calwrfout_2018031800_2018031812.m3d',
                                tmp_path/'second.m3d')


def test_calmet_control_reads_nondefault_radiation_coefficients(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source+='\n! HA1 = 850.5 ! ! HA2 = -12.75 ! ! HB1 = -.55 ! ! HB2 = 2.2 !\n'
    source+='! HC1 = 6.0E-13 ! ! HC2 = 45.25 ! ! HC3 = .2 !\n'
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    config=calmet_config(control)
    assert (config.turbidity_coefficient_1,config.turbidity_coefficient_2)==(850.5,-12.75)
    assert (config.cloudiness_coefficient_1,config.cloudiness_coefficient_2)==(-.55,2.2)
    assert (config.net_radiation_coefficient_1,config.net_radiation_coefficient_2,
            config.net_radiation_coefficient_3)==(6.e-13,45.25,.2)


def test_calmet_control_supports_legacy_prognostic_hours_fallback(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! ISTEPPGS =  3600   !',
                          '! ISTEPPG = 3 !\n       ! ISTEPPGS = 0 !',1)
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    assert calmet_config(control).prognostic_timestep_seconds==10800


def test_calmet_control_supports_pre_multiple_file_mm4dat(tmp_path):
    source=(CALMET_CONTROL).read_text(encoding='ascii')
    source=source.replace('! NM3D =  1  !','! NM3D =  0  !',1)
    source+='\n! MM4DAT = legacy.mm4 !\n'
    control=tmp_path/'calmet.inp'; control.write_text(source,encoding='ascii')
    assert calmet_config(control).meteorology==tmp_path/'legacy.mm4'


def test_public_api_accepts_control_path_and_path_overrides(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr('pycalmet._calmet.run',lambda config: captured.append(config) or Path(config.output))
    output=tmp_path/'result.nc'
    result=pycalmet.calmet(CALMET_CONTROL,meteorology=tmp_path/'met.nc',
                          geophysical=tmp_path/'geo.nc',output=output)
    assert result==output
    assert captured[0].meteorology==tmp_path/'met.nc'
    assert captured[0].grid.nx==610
