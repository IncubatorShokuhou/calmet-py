from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'run_zhejiang_python.sh'


def _fake_python(tmp_path: Path) -> Path:
    executable=tmp_path/'python-stub'
    executable.write_text('#!/bin/sh\nexit 0\n',encoding='ascii')
    executable.chmod(0o755)
    return executable


def test_skip_static_needs_only_wrf_time_and_existing_geo(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    work=tmp_path/'work'; work.mkdir(); (work/'geo.nc').touch()
    environment={**os.environ,'PYTHON_BIN':str(_fake_python(tmp_path))}
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--start','2018031800','--end','2018031812',
         '--work-dir',work,'--output',tmp_path/'calmet.nc','--skip-static'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==0,result.stderr


def test_skip_static_accepts_explicit_geophysical_path(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    geo=tmp_path/'bundled-geo.dat'; geo.touch()
    environment={**os.environ,'PYTHON_BIN':str(_fake_python(tmp_path))}
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--start','2018031800','--end','2018031800',
         '--geo',geo,'--work-dir',tmp_path/'work',
         '--output',tmp_path/'calmet.nc','--skip-static'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==0,result.stderr


def test_runner_converts_following_hour_for_calmet_boundary(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    geo=tmp_path/'geo.nc'; geo.touch()
    capture=tmp_path/'calwrf-end.txt'
    patch_dir=tmp_path/'python-patch'; patch_dir.mkdir()
    (patch_dir/'sitecustomize.py').write_text(dedent('''
        import os
        from pathlib import Path
        import pycalmet

        def _touch(path):
            path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.touch()
            return path

        def _calwrf(*_args,**kwargs):
            Path(os.environ['PYCALMET_TEST_CAPTURE']).write_text(kwargs['end'])
            _touch(kwargs['legacy_output'])
            return _touch(kwargs['output'])

        def _calmet(*args,**kwargs):
            Path(os.environ['PYCALMET_TEST_MET_CAPTURE']).write_text(
                str(args[0])+'\\n'+str(kwargs['meteorology'])
            )
            return _touch(kwargs['output'])

        pycalmet.calwrf=_calwrf
        pycalmet.calmet=_calmet
    '''),encoding='ascii')
    environment={
        **os.environ,'PYTHON_BIN':sys.executable,'PYTHONPATH':str(patch_dir),
        'PYCALMET_TEST_CAPTURE':str(capture),
        'PYCALMET_TEST_MET_CAPTURE':str(tmp_path/'calmet-inputs.txt'),
    }
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--start','2018031800','--end','2018031801',
         '--geo',geo,'--work-dir',tmp_path/'work',
         '--output',tmp_path/'calmet.nc','--skip-static'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==0,result.stderr
    assert capture.read_text(encoding='ascii')=='2018031802'
    control,meteorology=(tmp_path/'calmet-inputs.txt').read_text().splitlines()
    assert control.endswith('/calmet_mpi/calmet_zhejiang.inp')
    assert meteorology.endswith('/work/calwrf.m3d')


def test_static_run_requires_explicit_landuse_input(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    srtm=tmp_path/'srtm'; srtm.mkdir()
    environment={**os.environ,'PYTHON_BIN':str(_fake_python(tmp_path))}
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--srtm-dir',srtm,
         '--start','2018031800','--end','2018031812'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==2
    assert 'Land-use input not found' in result.stderr


def test_static_run_accepts_explicit_landuse_input(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    srtm=tmp_path/'srtm'; srtm.mkdir()
    landuse=tmp_path/'landuse.img'; landuse.touch()
    environment={**os.environ,'PYTHON_BIN':str(_fake_python(tmp_path))}
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--srtm-dir',srtm,'--landuse',landuse,
         '--start','2018031800','--end','2018031812',
         '--work-dir',tmp_path/'work','--output',tmp_path/'calmet.nc'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==0,result.stderr


def test_static_run_preserves_mixed_terrain_source_paths_and_metadata(tmp_path):
    wrf=tmp_path/'wrf.nc'; wrf.touch()
    srtm=tmp_path/'srtm'; srtm.mkdir()
    primary=srtm/'N00E000.HGT'; primary.touch()
    fallback_dir=tmp_path/'fallback'; fallback_dir.mkdir()
    fallback=fallback_dir/'N00E001.HGT'; fallback.touch()
    landuse=tmp_path/'landuse.img'; landuse.touch()
    configured_dir=tmp_path/'configured'; configured_dir.mkdir()
    capture=tmp_path/'terrain-sources.json'

    patch_dir=tmp_path/'python-patch'; patch_dir.mkdir()
    (patch_dir/'sitecustomize.py').write_text(dedent('''
        import json
        import os
        from pathlib import Path

        import pycalmet
        import pycalmet.control as control
        from pycalmet.config import GridConfig, TerrelConfig, TerrelSource

        def _terrel_config(_path):
            configured=Path(os.environ['PYCALMET_TEST_CONFIGURED'])
            fallback=Path(os.environ['PYCALMET_TEST_FALLBACK'])
            sources=(
                TerrelSource(configured/'N00E000.HGT','SRTM3','WGS-96'),
                TerrelSource(fallback,'GTOPO30','WGS-84'),
            )
            return TerrelConfig(
                configured,'unused.nc',GridConfig(2,2,1.,1.),
                terrain_files=tuple(item.path for item in sources),
                terrain_sources=sources,
            )

        def _touch(path):
            path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.touch()
            return path

        def _calwrf(*_args,**kwargs):
            return _touch(kwargs['output'])

        def _terrel(config):
            payload={
                'files':[str(item) for item in config.terrain_files],
                'sources':[{
                    'path':str(item.path),'input_type':item.input_type,
                    'input_datum':item.input_datum,
                } for item in config.terrain_sources],
                'side_outputs':[
                    config.save_output,config.plot_output,config.raw_echo_output,
                ],
            }
            Path(os.environ['PYCALMET_TEST_CAPTURE']).write_text(json.dumps(payload))
            return _touch(config.output)

        def _ctrproc(config):
            return _touch(config.output)

        def _makegeo(*_args,**kwargs):
            return _touch(kwargs['output'])

        def _calmet(*_args,**kwargs):
            return _touch(kwargs['output'])

        control.terrel_config=_terrel_config
        pycalmet.calwrf=_calwrf
        pycalmet.terrel=_terrel
        pycalmet.ctrproc=_ctrproc
        pycalmet.makegeo=_makegeo
        pycalmet.calmet=_calmet
    '''),encoding='ascii')
    environment={
        **os.environ,'PYTHON_BIN':sys.executable,'PYTHONPATH':str(patch_dir),
        'PYCALMET_TEST_CONFIGURED':str(configured_dir),
        'PYCALMET_TEST_FALLBACK':str(fallback),
        'PYCALMET_TEST_CAPTURE':str(capture),
    }
    result=subprocess.run(
        (SCRIPT,'--wrf',wrf,'--srtm-dir',srtm,'--landuse',landuse,
         '--start','2018031800','--end','2018031812',
         '--work-dir',tmp_path/'work','--output',tmp_path/'calmet.nc'),
        env=environment,text=True,capture_output=True,check=False,
    )
    assert result.returncode==0,result.stderr
    payload=json.loads(capture.read_text(encoding='ascii'))
    assert payload['files']==[str(primary),str(fallback)]
    assert payload['sources']==[
        {'path':str(primary),'input_type':'SRTM3','input_datum':'WGS-96'},
        {'path':str(fallback),'input_type':'GTOPO30','input_datum':'WGS-84'},
    ]
    assert payload['side_outputs']==[None,None,None]
