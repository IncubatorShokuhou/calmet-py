from datetime import datetime
from pathlib import Path
from typing import cast

from netCDF4 import Dataset
import numpy as np
import numpy.typing as npt
import pytest

from pycalmet._calmet import _observation_only_meteorology,run
from pycalmet.calmet_numerics import density_from_pressure_temperature
from pycalmet.calmet_output import CalmetNetCDFWriter
from pycalmet.calmet_solver import (DatasetLike,HourFields,Step1Fields,
                                    dataset_timestamp,target_latitude_longitude)
from pycalmet.config import CalmetConfig,GridConfig


def _write_geophysical(path: Path,shape: tuple[int,int]) -> None:
    """Write the minimal target-grid GEO fields consumed by CALMET."""
    with Dataset(path,'w') as dataset:
        dataset.createDimension('y',shape[0])
        dataset.createDimension('x',shape[1])
        values={
            'terrain_height':0.,'dominant_landuse':20,
            'roughness_length':.1,'albedo':.2,'bowen_ratio':1.,
            'soil_heat_flux_parameter':.15,'anthropogenic_heat_flux':0.,
        }
        for name,value in values.items():
            dtype='i4' if name=='dominant_landuse' else 'f4'
            dataset.createVariable(name,dtype,('y','x'))[:]=np.full(
                shape,value
            )


def test_observation_only_scaffold_has_local_boundaries_and_fp32_shapes() -> None:
    grid=GridConfig(3,2,1.,1.)
    config=CalmetConfig(
        None,'geo.nc','calmet.nc',datetime(2018,1,2,3),
        datetime(2018,1,2,4),grid,timestep_seconds=1800,
        iprog=0,noobs=0,itprog=0,irhprog=0,cloud_method=1,
        surface_data='surf.dat',upper_air_files=('up.dat',),
        z_faces_m=(0.,20.,60.),
    )

    dataset,provider=_observation_only_meteorology(
        config,config.start,config.end
    )

    assert dataset['time_text'].shape==(3,19)
    assert [dataset_timestamp(cast(DatasetLike,dataset),index)
            for index in range(3)]==[
        datetime(2018,1,2,3),datetime(2018,1,2,3,30),
        datetime(2018,1,2,4),
    ]
    step=provider(datetime(2018,1,2,3,30))
    assert step.u.shape==(2,2,3)
    assert step.surface_temperature.shape==(2,3)
    assert step.profile_temperature.shape==(3,2,3)
    assert step.u.dtype==np.dtype('float32')
    assert step.temperature.dtype==np.dtype('float32')
    assert step.surface_density.dtype==np.dtype('float32')
    assert step.precipitation_code.dtype==np.dtype('int32')
    with pytest.raises(ValueError,match='no boundary'):
        provider(datetime(2018,1,2,3))


def test_calmet_runs_from_surface_and_upper_air_without_prognostic_file(
        tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    grid=GridConfig(
        1,1,1.,1.,projection='LCC',lat0=30.,lon0=-71.,
        lat1=30.,lat2=60.,datum='WGS-84',
    )
    latitude,longitude=target_latitude_longitude(grid)
    geo=tmp_path/'geo.nc'
    _write_geophysical(geo,(1,1))
    surface=tmp_path/'SURF.DAT'
    surface.write_text(f"""SURF.DAT        2.1             observation-only test
0
LL
WGS-84  20200101
DEG
UTC-0000
2018 2 3 0 2018 2 4 0 1
1 SFC {latitude.item():.8f} {longitude.item():.8f} 10
2018 2 3 0 2018 2 4 0 5 270 70 6 280 65 1000 0
""",encoding='ascii')
    upper=tmp_path/'UP.DAT'
    upper.write_text(f"""UP.DAT          2.1             observation-only test
0
LL
WGS-84  20200101
DEG
UTC+0000
2018 2 3 0 2018 2 5 0 100 3 2
T T T T
2 UPR {latitude.item():.8f} {longitude.item():.8f} 0
2 2018 1 2 3 2
1000 0 280 270 5 900 100 270 270 6
2 2018 1 2 4 2
1000 0 280 270 5 900 100 270 270 6
2 2018 1 2 5 2
1000 0 280 270 5 900 100 270 270 6
""",encoding='ascii')
    output=tmp_path/'calmet.nc'
    config=CalmetConfig(
        None,geo,output,datetime(2018,1,2,3),datetime(2018,1,2,4),grid,
        iprog=0,noobs=0,itprog=0,irhprog=0,cloud_method=1,
        surface_data=surface,upper_air_files=(upper,),
        z_faces_m=(0.,20.,40.),smoothing_passes=(0,0),
        stations_per_level=(2,2),wind_extrapolation=-1,
        wind_extrapolation_factors=(1.,1.),overwater_flux_method=0,
        ifradj=0,islope=0,ikine=0,iobr=0,iavet=0,iavezi=0,
        lapse_height_m=20.,mixing_lapse_depth_m=20.,
    )
    captured_steps: list[Step1Fields]=[]
    original_append=CalmetNetCDFWriter.append

    def capture_append(
        writer: CalmetNetCDFWriter,
        hour: HourFields,
        solar_elevation_sine: npt.ArrayLike,
        landuse: npt.ArrayLike,
    ) -> None:
        """Capture the observation-adjusted state before writing it."""
        captured_steps.append(hour.step1)
        original_append(writer,hour,solar_elevation_sine,landuse)

    monkeypatch.setattr(CalmetNetCDFWriter,'append',capture_append)

    assert run(config)==output
    assert len(captured_steps)==1
    step=captured_steps[0]
    np.testing.assert_allclose(step.surface_temperature,280.)
    np.testing.assert_allclose(step.temperature[:,0,0],[280.,277.])
    np.testing.assert_allclose(step.surface_pressure_mb,1000.)
    np.testing.assert_allclose(
        step.surface_density,
        density_from_pressure_temperature(
            np.full((1,1),1000.,dtype=np.float32),
            np.full((1,1),280.,dtype=np.float32),
        ),
    )
    np.testing.assert_allclose(step.relative_humidity[0],65.)
    assert np.isnan(step.relative_humidity[1:]).all()

    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].size==1
        np.testing.assert_allclose(dataset['surface_temperature'][:],280.)
        np.testing.assert_allclose(dataset['surface_relative_humidity'][:],65.)
        np.testing.assert_allclose(dataset['relative_humidity'][:,0],65.)
        assert np.isnan(dataset['relative_humidity'][:,1:]).all()
        np.testing.assert_allclose(
            dataset['surface_density'][0],step.surface_density,
        )
        np.testing.assert_allclose(dataset['cloud_fraction'][:],.6)
        assert np.isfinite(dataset['wind_u'][:]).all()
        assert np.isfinite(dataset['wind_v'][:]).all()
        assert np.isfinite(dataset['temperature'][:]).all()
        assert dataset['wind_u'].dtype==np.dtype('float32')


def test_calmet_rejects_missing_meteorology_for_positive_iprog(
        tmp_path: Path) -> None:
    config=CalmetConfig(
        None,tmp_path/'geo.nc',tmp_path/'calmet.nc',
        datetime(2018,1,2,3),datetime(2018,1,2,4),
        GridConfig(1,1,1.,1.),noobs=2,iprog=14,itprog=2,irhprog=1,
        z_faces_m=(0.,20.),smoothing_passes=(0,),barrier_top_level=0,
        wind_extrapolation=-1,varying_radius=True,
    )

    with pytest.raises(ValueError,match='IPROG=14 requires meteorology input'):
        run(config)


@pytest.mark.parametrize('cloud_method',[3,4])
def test_observation_only_run_rejects_prognostic_cloud_method(
        tmp_path: Path,cloud_method: int) -> None:
    config=CalmetConfig(
        None,tmp_path/'geo.nc',tmp_path/'calmet.nc',
        datetime(2018,1,2,3),datetime(2018,1,2,4),
        GridConfig(1,1,1.,1.),noobs=0,iprog=0,itprog=0,irhprog=0,
        cloud_method=cloud_method,surface_data=tmp_path/'surf.dat',
        upper_air_files=(tmp_path/'up.dat',),z_faces_m=(0.,20.),
        smoothing_passes=(0,),stations_per_level=(1,),
        wind_extrapolation_factors=(1.,),wind_extrapolation=-1,
    )

    with pytest.raises(ValueError,match=rf'MCLOUD={cloud_method} requires meteorology'):
        run(config)
