from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import pytest

import pycalmet
from pycalmet.config import (CalmetConfig, CalwrfConfig, CtgprocConfig,
                             MakegeoConfig, TerrelConfig)
from pycalmet.control import assignments
from pycalmet.legacy import GridConfig as LegacyGridConfig
from pycalmet.legacy import MakegeoConfig as LegacyMakegeoConfig
from pycalmet.legacy._compat.control import calwrf_config as legacy_calwrf_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")
CALWRF_CONTROL = FIXTURES / "calwrf.inp"
CALMET_CONTROL = FIXTURES / "calmet.inp"


def test_legacy_preprocessor_defaults_match_fortran():
    grid = LegacyGridConfig(1, 1, 1.0, 1.0)
    makegeo = LegacyMakegeoConfig("terrain", "landuse", "geo", grid)
    terrel = pycalmet.legacy.TerrelConfig("terrain", "out", grid)
    ctgproc = pycalmet.legacy.CtgprocConfig(("landuse",), "out", grid)
    assert grid.projection == "UTM"
    assert grid.polar_processing == 2
    assert terrel.receptor_columns == 4
    assert (ctgproc.input_type, ctgproc.mesh_density,
            ctgproc.input_datum) == ("CTG", 1, "NAS-C")
    assert ctgproc.fill_empty_ocean is False
    assert pycalmet.legacy.CtgprocConfig(
        ("landuse",), "out", grid, coastline_file="coast.bln",
    ).fill_empty_ocean is True
    assert makegeo.snow_height_scale_m == 10.0
    assert makegeo.snow_depth_minimum_m == 0.01


def test_legacy_calwrf_netcdf_control_does_not_overwrite_it_with_m3d():
    config = legacy_calwrf_config(
        ROOT / "examples/zhejiang/config/calwrf.inp")
    assert config.output.suffix.casefold() == ".nc"
    assert config.legacy_output is None


def test_calwrf_accepts_fortran_parameter_names(monkeypatch, tmp_path):
    monkeypatch.setattr("pycalmet._calwrf.run", lambda config: config)
    config = pycalmet.calwrf({
        "flnames": [tmp_path / "wrf.nc"], "NFILES": "1",
        "IdAtEb": "2018010100", "idatee": "2018010101",
        "FIO": tmp_path / "calwrf.m3d", "IOSRF": "F",
        "title": "A custom CALWRF conversion", "FlG": tmp_path / "calwrf.lst",
        "IB": 2, "IE": 4, "JB": 3, "JE": 5, "KB": 1, "KE": 6,
    })
    assert isinstance(config, CalwrfConfig)
    assert config.output == tmp_path / "calwrf.nc"
    assert config.legacy_output == tmp_path / "calwrf.m3d"
    assert config.title == "A custom CALWRF conversion"
    assert config.listing_output == tmp_path / "calwrf.lst"
    assert not config.create_surface
    assert (config.i_range, config.j_range, config.k_range) == ((2, 4), (3, 5), (1, 6))


def test_legacy_calwrf_file_rain_reset_parses_logical(monkeypatch, tmp_path):
    monkeypatch.setattr("pycalmet.legacy._compat._calwrf.run", lambda config: config)
    config=pycalmet.legacy.calwrf(
        WRF_FILES=[tmp_path/"wrf.nc"],START="2018010100",END="2018010100",
        OUTPUT=tmp_path/"calwrf.nc",LEGACY_FILE_RAIN_RESET="T",
    )
    assert config.legacy_file_rain_reset is True


def test_calwrf_fio_derives_legacy_default_surface_name(monkeypatch, tmp_path):
    monkeypatch.setattr("pycalmet._calwrf.run", lambda config: config)
    config = pycalmet.calwrf(
        FLNAMES=[tmp_path / "wrf.nc"], NFILES=1,
        IDATEB="2018010100", IDATEE="2018010101",
        FIO=tmp_path / "calwrf.m3d",
    )
    assert config.create_surface
    assert config.surface_output == tmp_path / "calwrf.m2d"


def test_calwrf_accepts_explicit_legacy_surface_name(monkeypatch, tmp_path):
    monkeypatch.setattr("pycalmet._calwrf.run", lambda config: config)
    config = pycalmet.calwrf(
        FLNAMES=[tmp_path / "wrf.nc"], NFILES=1,
        IDATEB="2018010100", IDATEE="2018010101",
        FIO=tmp_path / "calwrf.m3d", IOSRF=1,
        FSRF=tmp_path / "surface-output.m2d",
    )
    assert config.create_surface
    assert config.surface_output == tmp_path / "surface-output.m2d"


def test_calwrf_fio_control_override_does_not_reuse_original_listing(monkeypatch, tmp_path):
    monkeypatch.setattr("pycalmet._calwrf.run", lambda config: config)
    target=tmp_path/"redirected.m3d"
    config=pycalmet.calwrf(CALWRF_CONTROL,FIO=target)
    assert config.output==target.with_suffix(".nc")
    assert config.legacy_output==target
    assert config.surface_output==target.with_suffix(".m2d")
    assert config.listing_output is None


def test_terrel_accepts_legacy_sources_and_nested_grid(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._terrel.run", lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    output = tmp_path / "terr.dat"
    assert pycalmet.terrel(
        srtm3=[tmp_path / "n00e000.hgt"], ntdf=1, outfil=output,
        nx=2, ny=3, dgridkm="1.5", pmap="utm", iutmzn=50,
        xrefkm=10, yrefkm=20, datum="WGS-84",
    ) == output
    config = captured[0]
    assert isinstance(config, TerrelConfig)
    assert (config.grid.nx, config.grid.ny) == (2, 3)
    assert (config.grid.dx_km, config.grid.dy_km) == (1.5, 1.5)
    assert config.terrain_sources[0].input_type == "SRTM3"
    assert config.terrain_sources[0].input_datum == "WGS-96"


def test_terrel_accepts_actual_polar_and_coast_datum_names(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._terrel.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.terrel(
        GEN=tmp_path/'terrain.gen',NTDF=1,OUTFIL=tmp_path/'terrain.dat',
        NX=1,NY=1,DGRIDKM=1,PMAP='UTM',IUTMZN=50,XREFKM=0,YREFKM=0,
        DATUM='WGS-84',IMODEL=3,IGRID=3,NRING=2,NRAYS=2,
        DISKM=(1.,2.),ANGDEG=(10.,190.),DWVS='WGS-84',DWDBII='WGS-72',
    )
    config=captured[0]
    assert config.grid.ring_distances_km==(1.,2.)
    assert config.grid.ray_angles_deg==(10.,190.)
    assert config.coastline_wdbii_datum=='WGS-72'


def test_public_legacy_grid_angles_accept_prefixes_and_wrap_longitude(
        monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._terrel.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.terrel(
        GEN=tmp_path/'terrain.gen',NTDF=1,OUTFIL=tmp_path/'terrain.dat',
        NX=1,NY=1,DGRIDKM=1,PMAP='LCC',XREFKM=0,YREFKM=0,
        RLAT0='N 27',RLON0='243E',RLAT1='N 30',RLAT2='N 60',
        DATUM='WGS-84',
    )
    grid=captured[0].grid
    assert (grid.lat0,grid.lon0,grid.lat1,grid.lat2)==(27.,-117.,30.,60.)


def test_terrel_legacy_flags_activate_default_files(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._terrel.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.terrel(
        SRTM3=tmp_path/'n00e000.hgt',NTDF=1,OUTFIL=tmp_path/'terrain.dat',
        NX=1,NY=1,DGRIDKM=1,PMAP='UTM',IUTMZN=50,XREFKM=0,YREFKM=0,
        DATUM='WGS-84',LXY=True,LPREV=True,LCOAST=True,LBLNREAD=True,
        LRAWECHO=True,
    )
    config=captured[0]
    assert config.receptor_input=='xyinp.dat'
    assert config.receptor_output=='xyout.dat'
    assert config.previous_save=='prev.sav'
    assert config.coastline_file=='coast.bln'
    assert config.gshhs_file is None
    assert config.raw_echo_output=='rawecho.dat'


def test_ctrproc_accepts_legacy_source_and_grid_names(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._ctgproc.run", lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    output = tmp_path / "lu.dat"
    assert pycalmet.ctrproc(
        glazas=tmp_path / "eausgs1.img", ndbf=1, meshglaz=12, ludat=output,
        nx=2, ny=2, dgridkm=1, pmap="UTM", iutmzn=50,
        xrefkm=0, yrefkm=0, datum="WGS-84",
    ) == output
    config = captured[0]
    assert isinstance(config, CtgprocConfig)
    assert config.input_type == "GLAZAS"
    assert config.mesh_density == 12
    assert config.landuse_files == (tmp_path / "eausgs1.img",)


def test_ctgproc_legacy_zero_file_count_and_flags_use_old_defaults(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._ctgproc.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.ctrproc(
        LUINDAT=tmp_path/'land.img',NDBF=0,LULC=2,IGLAZR=4,LUDAT=tmp_path/'lu.dat',
        NX=1,NY=1,DGRIDKM=1,PMAP='UTM',IUTMZN=50,XREFKM=0,YREFKM=0,
        DATUM='WGS-84',LPREV=True,LCOAST=True,LBLNREAD=True,
    )
    config=captured[0]
    assert config.previous_landuse=='prev.dat'
    assert config.coastline_file=='coast.bln'
    assert config.gshhs_file is None
    assert config.landuse_sources[0].input_type=='GLAZAS'


def test_source_specific_legacy_kwargs_override_control_metadata(monkeypatch,tmp_path):
    terrain=[]; landuse=[]
    monkeypatch.setattr("pycalmet._terrel.run",lambda config:
                        terrain.append(config) or SimpleNamespace(output=config.output))
    monkeypatch.setattr("pycalmet._ctgproc.run",lambda config:
                        landuse.append(config) or SimpleNamespace(output=config.output))
    root=Path(__file__).resolve().parents[1]
    pycalmet.terrel(root/'examples/zhejiang/config/terrel.inp',DSRTM3='WGS-84',
                    output=tmp_path/'terrain.nc')
    pycalmet.ctrproc(root/'examples/zhejiang/config/ctgproc.inp',DUSGSLA='WGS-84',MESHGLAZ=3,
                     output=tmp_path/'landuse.nc')
    assert {source.input_datum for source in terrain[0].terrain_sources}=={'WGS-84'}
    assert {(source.input_datum,source.mesh_density)
            for source in landuse[0].landuse_sources}=={('WGS-84',3)}


def test_makegeo_accepts_legacy_names_and_record_values(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._makegeo.run", lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    output = tmp_path / "geo.dat"
    assert pycalmet.makegeo(
        terrdat=tmp_path / "terr.dat", ludat=tmp_path / "lu.dat", geodat=output,
        nx=2, ny=3, dgridkm=1, pmap="UTM", iutmzn=50,
        xrefkm=0, yrefkm=0, datum="WGS-84", cfract="0.6", imiss="55",
        x=["11, 1, .2, 3, .4, .5, .6, 20"], nincat=1,
        iwat="51,55", numwat=2, ixqa=1, iyqa=2,
        outcat="10,0,20", noutcat=2,
    ) == output
    config = captured[0]
    assert isinstance(config, MakegeoConfig)
    assert config.water_threshold == .6
    assert config.input_properties[11] == (1., .2, 3., .4, .5, .6, 20)
    assert config.water_categories == (51, 55)
    assert config.output_categories == (10, 20)
    assert config.qa_cell == (1, 2)


def test_makegeo_legacy_flags_activate_default_files(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._makegeo.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.makegeo(
        TERRDAT=tmp_path/'terr.dat',LUDAT=tmp_path/'lu.dat',GEODAT=tmp_path/'geo.dat',
        NX=1,NY=1,DGRIDKM=1,PMAP='UTM',IUTMZN=50,XREFKM=0,YREFKM=0,
        DATUM='WGS-84',LLU2=True,LSNOW=True,
    )
    config=captured[0]
    assert config.secondary_landuse=='lu2.dat'
    assert config.snow_data=='snow.dat'
    assert config.snow_height_scale_m==10.
    assert config.snow_depth_minimum_m==.01


def test_makegeo_legacy_ifmtgeo_rejects_nonportable_format():
    with pytest.raises(ValueError,match='IFMTGEO=1'):
        pycalmet.makegeo(IFMTGEO=2)


def test_legacy_listing_and_lcfiles_kwargs_are_accepted_without_rewriting_paths(
        monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._makegeo.run",lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    output=tmp_path/'Geo.Dat'
    pycalmet.makegeo(
        TERRDAT=tmp_path/'Terrain.Dat',LUDAT=tmp_path/'Landuse.Dat',GEODAT=output,
        NX=1,NY=1,DGRIDKM=1,PMAP='UTM',IUTMZN=50,XREFKM=0,YREFKM=0,
        DATUM='WGS-84',LCFILES=True,RUNLST=tmp_path/'Makegeo.Lst',
    )
    config=captured[0]
    assert config.terrain==tmp_path/'Terrain.Dat'
    assert config.landuse==tmp_path/'Landuse.Dat'
    assert config.output==output


def test_calmet_accepts_control_names_dates_and_timezone(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._calmet.run", lambda config:
                        captured.append(config) or config.output)
    output = pycalmet.calmet({
        "m3ddat": tmp_path / "met.m3d", "Nm3d": 1,
        "geodat": tmp_path / "geo.dat", "metdat": tmp_path / "calmet.dat",
        "ibyr": 2018, "ibmo": 3, "ibdy": 18, "ibhr": 0,
        "ieyr": 2018, "iemo": 3, "iedy": 18, "iehr": 1,
        "abtz": "UTC-0800", "nx": 2, "ny": 3, "dgridkm": 1,
        "pmap": "utm", "iutmzn": 50, "xorigkm": 0, "yorigkm": 0,
        "datum": "WGS-84", "zface": "0, 20", "nz": 1,
        "iprog": "14", "lcalgrd": "F",
    })
    config = captured[0]
    assert output == tmp_path / "calmet.dat"
    assert isinstance(config, CalmetConfig)
    assert config.start == datetime(2018,3,18)
    assert config.end == datetime(2018,3,18,1)
    assert config.base_timezone == 8
    assert config.base_timezone_label == "UTC-0800"
    assert config.iprog == 14
    assert config.z_faces_m == (0., 20.)


def _minimal_calmet_aliases(tmp_path):
    return {
        'M3DDAT':tmp_path/'met.m3d','NM3D':1,
        'GEODAT':tmp_path/'geo.dat','METDAT':tmp_path/'calmet.dat',
        'IBYR':2018,'IBMO':3,'IBDY':18,'IBHR':0,'IBSEC':0,
        'IEYR':2018,'IEMO':3,'IEDY':18,'IEHR':1,'IESEC':0,
        'ABTZ':'UTC-0000','NX':1,'NY':1,'DGRIDKM':1,
        'PMAP':'UTM','IUTMZN':50,'XORIGKM':0,'YORIGKM':0,
        'DATUM':'WGS-84','ZFACE':'0,20','NZ':1,
    }


def test_calmet_public_aliases_expand_two_digit_years(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr('pycalmet._calmet.run',lambda config:
                        captured.append(config) or config.output)
    values=_minimal_calmet_aliases(tmp_path)
    values.update(IBYR=18,IEYR=18)
    pycalmet.calmet(values)
    assert captured[0].start==datetime(2018,3,18)
    assert captured[0].end==datetime(2018,3,18,1)


@pytest.mark.parametrize(('field','value'),(
    ('IBYR',2018.5),('IBMO',3.5),('IBDY',18.5),('IBHR',.5),
    ('IBSEC',.5),('IBSEC',3601),('NSECDT',900.5),
    ('ISTEPPGS',900.5),('ISTEPPG',1.5),
))
def test_calmet_public_time_aliases_never_truncate_fractional_values(
        monkeypatch,tmp_path,field,value):
    monkeypatch.setattr('pycalmet._calmet.run',lambda config: config.output)
    values=_minimal_calmet_aliases(tmp_path)
    values[field]=value
    if field=='ISTEPPG':
        values['ISTEPPGS']=0
    with pytest.raises(ValueError,match=field):
        pycalmet.calmet(values)


def test_calmet_accepts_legacy_radiation_and_mm4_timestep_names(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("pycalmet._calmet.run", lambda config:
                        captured.append(config) or config.output)
    pycalmet.calmet({
        "MM4DAT": tmp_path / "legacy.mm4", "NM3D": 0,
        "GEODAT": tmp_path / "geo.dat", "METDAT": tmp_path / "calmet.dat",
        "IBYR": 2018, "IBMO": 3, "IBDY": 18, "IBHR": 0,
        "IEYR": 2018, "IEMO": 3, "IEDY": 18, "IEHR": 1,
        "ABTZ": "UTC-0000", "NX": 2, "NY": 2, "DGRIDKM": 1,
        "PMAP": "UTM", "IUTMZN": 50, "XORIGKM": 0, "YORIGKM": 0,
        "DATUM": "WGS-84", "ZFACE": "0,20", "NZ": 1,
        "ISTEPPG": 3, "ISTEPPGS": 0,
        "HA1": 850.5, "HA2": -12.75, "HB1": -.55, "HB2": 2.2,
        "HC1": 6.e-13, "HC2": 45.25, "HC3": .2,
    })
    config = captured[0]
    assert config.meteorology == tmp_path / "legacy.mm4"
    assert config.prognostic_timestep_seconds == 10800
    assert (config.turbidity_coefficient_1, config.turbidity_coefficient_2) == (850.5, -12.75)
    assert (config.cloudiness_coefficient_1, config.cloudiness_coefficient_2) == (-.55, 2.2)
    assert (config.net_radiation_coefficient_1, config.net_radiation_coefficient_2,
            config.net_radiation_coefficient_3) == (6.e-13, 45.25, .2)


def test_calmet_raw_isteppgs_zero_uses_default_legacy_hour(monkeypatch,tmp_path):
    captured=[]
    monkeypatch.setattr("pycalmet._calmet.run",lambda config:
                        captured.append(config) or config.output)
    pycalmet.calmet(
        M3DDAT=tmp_path/'met.m3d',NM3D=1,
        GEODAT=tmp_path/'geo.dat',METDAT=tmp_path/'calmet.dat',
        IBYR=2018,IBMO=3,IBDY=18,IBHR=0,
        IEYR=2018,IEMO=3,IEDY=18,IEHR=1,
        ABTZ='UTC-0000',NX=1,NY=1,DGRIDKM=1,
        PMAP='UTM',IUTMZN=50,XORIGKM=0,YORIGKM=0,
        DATUM='WGS-84',ZFACE='0,20',NZ=1,ISTEPPGS=0,
    )
    assert captured[0].prognostic_timestep_seconds==3600


def test_legacy_grid_override_is_applied_to_control_config(monkeypatch):
    monkeypatch.setattr("pycalmet._makegeo.run", lambda config:
                        SimpleNamespace(output=config.grid.nx))
    # Facades require a path return, so capture the config through the engine.
    captured = []
    monkeypatch.setattr("pycalmet._makegeo.run", lambda config:
                        captured.append(config) or SimpleNamespace(output=config.output))
    pycalmet.makegeo(ROOT / "examples/zhejiang/config/makegeo.inp", NX=7)
    assert captured[0].grid.nx == 7
    pycalmet.makegeo(
        ROOT / "examples/zhejiang/config/makegeo.inp", grid={"nx": 8})
    assert captured[1].grid.nx == 8
    assert isinstance(captured[1].grid, pycalmet.GridConfig)


def test_descriptive_alias_conflicts_are_rejected_and_grid_fields_merge(
        monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="duplicate CalwrfConfig parameter 'output'"):
        pycalmet.calwrf({
            "WRF_FILES": [tmp_path / "wrf.nc"], "START": "2018010100",
            "END": "2018010101", "OUTPUT": tmp_path / "a.nc",
            "FIO": tmp_path / "b.m3d",
        })
    captured = []
    monkeypatch.setattr("pycalmet._calmet.run",
                        lambda config: captured.append(config) or Path(config.output))
    pycalmet.calmet({
        "meteorology": "m", "geophysical": "g", "output": "o",
        "start": "2018010100", "end": "2018010101",
        "grid": pycalmet.GridConfig(1, 1, 1., 1.), "NX": 2,
    })
    assert captured[0].grid.nx == 2


def test_legacy_declared_file_counts_are_validated(tmp_path):
    with pytest.raises(ValueError, match="NFILES does not match"):
        pycalmet.calwrf(
            FLNAMES=[tmp_path / "one.nc"], NFILES=2,
            IDATEB="2018010100", IDATEE="2018010101", FIO=tmp_path / "out.m3d",
        )


@pytest.mark.parametrize(("stage", "source", "engine", "config_type"), (
    ("calwrf", CALWRF_CONTROL, "pycalmet._calwrf.run", CalwrfConfig),
    ("terrel", ROOT / "examples/zhejiang/config/terrel.inp", "pycalmet._terrel.run", TerrelConfig),
    ("ctrproc", ROOT / "examples/zhejiang/config/ctgproc.inp", "pycalmet._ctgproc.run", CtgprocConfig),
    ("makegeo", ROOT / "examples/zhejiang/config/makegeo.inp", "pycalmet._makegeo.run", MakegeoConfig),
    ("calmet", CALMET_CONTROL, "pycalmet._calmet.run", CalmetConfig),
))
def test_readlst_is_kwargs_ready_for_each_public_stage(
        monkeypatch, tmp_path, stage, source, engine, config_type):
    captured = []
    if stage in {"terrel", "ctrproc", "makegeo"}:
        monkeypatch.setattr(engine, lambda config:
                            captured.append(config) or SimpleNamespace(output=config.output))
    else:
        monkeypatch.setattr(engine, lambda config:
                            captured.append(config) or Path(config.output))
    control=tmp_path/Path(source).name
    control.write_text(Path(source).read_text(encoding="ascii"),encoding="ascii")
    listing=tmp_path/f"{stage}.lst"
    listing.write_text(f"Run summary\nControl File Used\n{control.name}\n",encoding="ascii")
    result = getattr(pycalmet, stage)(**pycalmet.readlst(listing))
    assert result == Path(captured[0].output)
    assert isinstance(captured[0], config_type)
    if stage != "calwrf":
        assert isinstance(captured[0].grid, pycalmet.GridConfig)


def test_readlst_component_override_accepts_ctrproc_spelling(tmp_path):
    control = tmp_path / "generic.inp"
    control.write_text((ROOT / "examples/zhejiang/config/ctgproc.inp").read_text(encoding="ascii"),
                       encoding="ascii")
    params = pycalmet.readlst(control, component="ctrproc")
    assert isinstance(params["grid"], pycalmet.GridConfig)
    assert params["input_type"] == "GEN"


@pytest.mark.parametrize(("stage", "control", "engine", "config_type"), (
    ("terrel", ROOT / "examples/zhejiang/config/terrel.inp", "pycalmet._terrel.run", TerrelConfig),
    ("ctrproc", ROOT / "examples/zhejiang/config/ctgproc.inp", "pycalmet._ctgproc.run", CtgprocConfig),
    ("makegeo", ROOT / "examples/zhejiang/config/makegeo.inp", "pycalmet._makegeo.run", MakegeoConfig),
    ("calmet", CALMET_CONTROL, "pycalmet._calmet.run", CalmetConfig),
))
def test_every_active_assignment_name_is_accepted_as_raw_kwargs(
        monkeypatch, stage, control, engine, config_type):
    captured = []
    if stage in {"terrel", "ctrproc", "makegeo"}:
        monkeypatch.setattr(engine, lambda config:
                            captured.append(config) or SimpleNamespace(output=config.output))
    else:
        monkeypatch.setattr(engine, lambda config:
                            captured.append(config) or Path(config.output))
    getattr(pycalmet, stage)(**assignments(control))
    assert isinstance(captured[0], config_type)
    assert isinstance(captured[0].grid, pycalmet.GridConfig)
