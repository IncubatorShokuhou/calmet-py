# PyCALMET

[中文文档](README.zh-CN.md)

PyCALMET is a NumPy implementation of the CALMET meteorological preprocessing
and diagnostic wind-field workflow. It provides Python APIs for CALWRF,
TERREL, CTGPROC, MAKEGEO, and CALMET, with NetCDF as the preferred interchange
format and compatibility readers and writers for legacy CALPUFF files.

The package is intended for reproducible research, automated data processing,
and modernization of existing CALMET workflows. A complete Zhejiang example,
the required SRTM3 terrain tiles, and the original Fortran sources used as
numerical references are kept in separate, clearly scoped directories.

## Highlights

- Pure Python and NumPy/SciPy execution; no Fortran compiler is required.
- Five public processing stages exposed through one package.
- WRF NetCDF input and CF-compatible NetCDF output.
- Compatibility with legacy CALWRF, TERREL, CTGPROC, MAKEGEO, and CALMET files.
- Streaming WRF processing for datasets larger than memory.
- A real-data Zhejiang example with losslessly compressed WRF inputs.
- A separate opt-in legacy API for reproducing historical Fortran-oriented runs.

## Installation

PyCALMET requires Python 3.10 or newer.

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Quick start

Each stage accepts either explicit keyword arguments or its legacy
CALMET-family control file. These `.inp` files are program-specific control
files rather than Fortran `NAMELIST` files. The Zhejiang example defines all
five:

| Stage | Example control file |
| --- | --- |
| CALWRF | `examples/zhejiang/config/calwrf.inp` |
| TERREL | `examples/zhejiang/config/terrel.inp` |
| CTGPROC | `examples/zhejiang/config/ctgproc.inp` |
| MAKEGEO | `examples/zhejiang/config/makegeo.inp` |
| CALMET | `examples/zhejiang/config/calmet.inp` |

Paths inside a control file are resolved relative to that file. The bundled
example is fully defined by these files, so no duplicate Python arguments are
needed. Keyword overrides remain available for applications that deliberately
reuse a control file with different data. Stage functions return the path of
their primary output.

```python
import pycalmet

base = "examples/zhejiang/config"
pycalmet.calwrf(f"{base}/calwrf.inp")
pycalmet.terrel(f"{base}/terrel.inp")
pycalmet.ctrproc(f"{base}/ctgproc.inp")
pycalmet.makegeo(f"{base}/makegeo.inp")
result = pycalmet.calmet(f"{base}/calmet.inp")
```

Every stage also has a complete typed keyword API. Leave `inp` at its default
`None` and pass the control values directly; grid controls such as `nx`,
`dx_km`, and `projection` are ordinary parameters. Dates accept `datetime`,
paths accept `str` or `Path`, and repeated controls accept Python sequences.
Parameters omitted by the caller use the defaults defined by the original
program. For example:

```python
from datetime import datetime
from pathlib import Path

result = pycalmet.calmet(
    inp=None,
    meteorology=Path("output/calwrf.nc"),
    geophysical=Path("output/geo.nc"),
    output=Path("output/calmet.nc"),
    start=datetime(2018, 3, 18, 0),
    end=datetime(2018, 3, 18, 1),
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    lat0=27.0,
    lon0=117.5,
)
```

The signatures list every supported control, including controls with Fortran
defaults that may be absent from a particular `.inp` file. They are visible to
IDE completion, type checkers, `help()`, and `inspect.signature()`.

<!-- parameter-reference:start -->
## Complete parameter reference

The tables below are generated from the public function signatures and typed
configuration models. “Default” is the value used in `inp=None` mode when the
caller omits that parameter; “Zhejiang value” is the effective value after
parsing the bundled control file. `inp` itself and `**legacy_parameters` are
entry/compatibility mechanisms rather than model controls, so they are not
repeated in each table. Paths are relative to the repository root. Long
sequences are abbreviated with their total item count. Regenerate the tables
with `python tools/generate_parameter_reference.py`; use `--check` in CI.

<details>
<summary><strong>CALWRF</strong> — 14 parameters; Zhejiang control: `examples/zhejiang/config/calwrf.inp`.</summary>

| Parameter | Python type | Default | Zhejiang value |
| --- | --- | --- | --- |
| `wrf_files` | `Sequence[str \| Path] \| None` | `required` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | `str \| int \| datetime \| None` | `required` | `'2018031800'` |
| `end` | `str \| int \| datetime \| None` | `required` | `'2018031802'` |
| `output` | `str \| Path \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `create_surface` | `bool` | `True` | `False` |
| `compression` | `int` | `1` | `1` |
| `i_range` | `tuple[int, int] \| None` | `None` | `None` |
| `j_range` | `tuple[int, int] \| None` | `None` | `None` |
| `k_range` | `tuple[int, int] \| None` | `None` | `None` |
| `legacy_output` | `str \| Path \| None` | `None` | `None` |
| `surface_output` | `str \| Path \| None` | `None` | `None` |
| `title` | `str` | `'Create 3D.DAT file for WRF output'` | `'Create CALWRF data for the Zhejiang example'` |
| `listing_output` | `str \| Path \| None` | `None` | `output/zhejiang/calwrf.lst` |
| `legacy_file_rain_reset` | `bool` | `False` | `False` |

</details>

<details>
<summary><strong>TERREL</strong> — 49 parameters; Zhejiang control: `examples/zhejiang/config/terrel.inp`.</summary>

| Parameter | Python type | Default | Zhejiang value |
| --- | --- | --- | --- |
| `srtm_dir` | `str \| Path \| None` | `required` | `examples/zhejiang/config` |
| `output` | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | `Sequence[str \| Path]` | `()` | `[terrain/srtm3/N26E117.hgt, terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=terrain/…datum='WGS-96'), … (42 items))` |
| `void_fill` | `bool` | `False` | `False` |
| `cell_radius_km` | `float` | `0.0` | `5.0` |
| `model` | `int` | `1` | `1` |
| `processing_method` | `int \| None` | `None` | `2` |
| `receptor_input` | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | `int` | `2` | `2` |
| `interpolate_receptors` | `bool` | `False` | `True` |
| `receptor_radius_km` | `float` | `0.0` | `0.15` |
| `previous_save` | `str \| Path \| None` | `None` | `None` |
| `save_output` | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | `int` | `75` | `75` |
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | `int \| None` | `required` | `610` |
| `ny` | `int \| None` | `required` | `499` |
| `dx_km` | `float \| None` | `required` | `1.0` |
| `dy_km` | `float \| None` | `required` | `1.0` |
| `projection` | `str` | `'LCC'` | `'LCC'` |
| `lat0` | `float` | `27.0` | `27.0` |
| `lon0` | `float` | `117.5` | `117.5` |
| `lat1` | `float` | `30.0` | `30.0` |
| `lat2` | `float` | `60.0` | `60.0` |
| `x_origin_km` | `float` | `0.0` | `0.0` |
| `y_origin_km` | `float` | `0.0` | `0.0` |
| `false_easting_km` | `float` | `0.0` | `0.0` |
| `false_northing_km` | `float` | `0.0` | `0.0` |
| `utm_zone` | `int` | `50` | `50` |
| `hemisphere` | `str` | `'N'` | `'N'` |
| `grid_type` | `int` | `1` | `1` |
| `ring_distances_km` | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | `Sequence[float]` | `()` | `()` |
| `polar_processing` | `int` | `1` | `2` |
| `ttm_scale` | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 43 parameters; Zhejiang control: `examples/zhejiang/config/ctgproc.inp`.</summary>

| Parameter | Python type | Default | Zhejiang value |
| --- | --- | --- | --- |
| `landuse_files` | `Sequence[str \| Path] \| None` | `required` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `category_map` | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | `str` | `'GLAZAS'` | `'GEN'` |
| `mesh_density` | `int` | `12` | `12` |
| `input_datum` | `str` | `'ESR-S'` | `'WGS-84'` |
| `output_format` | `str` | `'auto'` | `'auto'` |
| `coastline_file` | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | `bool` | `False` | `False` |
| `fill_empty_ocean` | `bool` | `False` | `False` |
| `ocean_category` | `int` | `55` | `55` |
| `previous_landuse` | `str \| Path \| None` | `None` | `None` |
| `final_output` | `bool` | `True` | `True` |
| `landuse_sources` | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=examples/zhejiang/landuse/zhejiang_generic.landuse, input_type='GEN', input_datum='WGS-84', mesh_density=12))` |
| `snow_processing` | `bool` | `False` | `False` |
| `qa_cell_output` | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | `int` | `75` | `75` |
| `qa_output_directory` | `str \| Path \| None` | `None` | `None` |
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | `int \| None` | `required` | `610` |
| `ny` | `int \| None` | `required` | `499` |
| `dx_km` | `float \| None` | `required` | `1.0` |
| `dy_km` | `float \| None` | `required` | `1.0` |
| `projection` | `str` | `'LCC'` | `'LCC'` |
| `lat0` | `float` | `27.0` | `27.0` |
| `lon0` | `float` | `117.5` | `117.5` |
| `lat1` | `float` | `30.0` | `30.0` |
| `lat2` | `float` | `60.0` | `60.0` |
| `x_origin_km` | `float` | `0.0` | `0.0` |
| `y_origin_km` | `float` | `0.0` | `0.0` |
| `false_easting_km` | `float` | `0.0` | `0.0` |
| `false_northing_km` | `float` | `0.0` | `0.0` |
| `utm_zone` | `int` | `50` | `50` |
| `hemisphere` | `str` | `'N'` | `'N'` |
| `datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | `int` | `1` | `1` |
| `ring_distances_km` | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | `Sequence[float]` | `()` | `()` |
| `polar_processing` | `int` | `1` | `1` |
| `ttm_scale` | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 52 parameters; Zhejiang control: `examples/zhejiang/config/makegeo.inp`.</summary>

| Parameter | Python type | Default | Zhejiang value |
| --- | --- | --- | --- |
| `terrain` | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `landuse` | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `output` | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `water_threshold` | `float` | `0.5` | `0.5` |
| `missing_category` | `int` | `55` | `55` |
| `minimum_coverage` | `float` | `0.96` | `0.96` |
| `secondary_landuse` | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | `bool` | `True` | `True` |
| `input_properties` | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | `int` | `50` | `50` |
| `output_water_max` | `int` | `55` | `55` |
| `category_splits` | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | `bool` | `False` | `False` |
| `snow_data` | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | `int \| None` | `None` | `None` |
| `snow_end_date` | `int \| None` | `None` | `None` |
| `snow_properties` | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | `int` | `1` | `1` |
| `snow_height_scale_m` | `float` | `1.0` | `10.0` |
| `snow_albedo_method` | `int` | `1` | `1` |
| `snow_depth_minimum_m` | `float` | `0.0` | `0.01` |
| `landuse_plot_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | `int` | `0` | `0` |
| `title` | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | `int \| None` | `required` | `610` |
| `ny` | `int \| None` | `required` | `499` |
| `dx_km` | `float \| None` | `required` | `1.0` |
| `dy_km` | `float \| None` | `required` | `1.0` |
| `projection` | `str` | `'LCC'` | `'LCC'` |
| `lat0` | `float` | `27.0` | `27.0` |
| `lon0` | `float` | `117.5` | `117.5` |
| `lat1` | `float` | `30.0` | `30.0` |
| `lat2` | `float` | `60.0` | `60.0` |
| `x_origin_km` | `float` | `0.0` | `0.0` |
| `y_origin_km` | `float` | `0.0` | `0.0` |
| `false_easting_km` | `float` | `0.0` | `0.0` |
| `false_northing_km` | `float` | `0.0` | `0.0` |
| `utm_zone` | `int` | `50` | `50` |
| `hemisphere` | `str` | `'N'` | `'N'` |
| `datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | `int` | `1` | `1` |
| `ring_distances_km` | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | `Sequence[float]` | `()` | `()` |
| `polar_processing` | `int` | `1` | `1` |
| `ttm_scale` | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CALMET</strong> — 165 parameters; Zhejiang control: `examples/zhejiang/config/calmet.inp`.</summary>

| Parameter | Python type | Default | Zhejiang value |
| --- | --- | --- | --- |
| `meteorology` | `str \| Path \| Sequence[str \| Path] \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `geophysical` | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `output` | `str \| Path \| None` | `required` | `output/zhejiang/calmet.nc` |
| `start` | `str \| datetime \| None` | `required` | `'2018-03-18_00:00:00'` |
| `end` | `str \| datetime \| None` | `required` | `'2018-03-18_01:00:00'` |
| `noobs` | `int` | `2` | `2` |
| `run_type` | `int` | `1` | `1` |
| `save_output` | `bool` | `True` | `True` |
| `legacy_output_format` | `int` | `1` | `1` |
| `test_mode` | `int` | `2` | `2` |
| `regulatory_mode` | `int` | `0` | `0` |
| `timestep_seconds` | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | `int` | `3600` | `3600` |
| `cloud_method` | `int` | `4` | `4` |
| `cloud_data` | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | `int` | `2` | `2` |
| `cloud_output` | `int` | `0` | `0` |
| `surface_data` | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | `int` | `2` | `2` |
| `surface_control_stations` | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | `int` | `0` | `0` |
| `diagnostic_initial_wind` | `int` | `0` | `0` |
| `diagnostic_surface_wind` | `int` | `0` | `0` |
| `diagnostic_upper_wind` | `int` | `0` | `0` |
| `initial_guess_upper_station` | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | `int` | `0` | `0` |
| `initial_guess_calmet_files` | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | `int` | `-1` | `-1` |
| `precipitation_data` | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | `int` | `2` | `2` |
| `precipitation_control_stations` | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | `int` | `14` | `14` |
| `irhprog` | `int` | `1` | `1` |
| `itprog` | `int` | `2` | `2` |
| `imixh` | `int` | `1` | `1` |
| `constb` | `float` | `1.41` | `1.41` |
| `conste` | `float` | `0.15` | `0.15` |
| `constn` | `float` | `2400.0` | `2400.0` |
| `constw` | `float` | `0.16` | `0.16` |
| `coriolis` | `float` | `0.0001` | `0.0001` |
| `minimum_lapse_rate` | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | `int` | `0` | `0` |
| `prognostic_ocean_category` | `int` | `16` | `16` |
| `overwater_flux_method` | `int` | `10` | `10` |
| `coastal_shelf_length_km` | `float` | `0.0` | `0.0` |
| `warm_layer` | `int` | `0` | `0` |
| `cool_skin` | `int` | `0` | `0` |
| `convective_relaxation` | `int` | `1` | `1` |
| `convective_relaxation_time_s` | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | `int` | `2` | `2` |
| `precipitation_radius_km` | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | `int` | `-1` | `-1` |
| `upper_temperature_option` | `int` | `-1` | `-1` |
| `lapse_height_m` | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | `int` | `1` | `1` |
| `temperature_radius_km` | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | `int` | `5` | `5` |
| `water_temperature_gradient_below` | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | `int` | `55` | `55` |
| `water_temperature_category_max` | `int` | `55` | `55` |
| `iavet` | `int` | `1` | `1` |
| `iavezi` | `int` | `1` | `1` |
| `mixing_wind_level` | `int` | `1` | `1` |
| `mnmdav` | `int` | `1` | `10` |
| `iwfcod` | `int` | `1` | `1` |
| `wind_extrapolation` | `int` | `-1` | `-1` |
| `upper_air_separation_cells` | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | `int` | `0` | `0` |
| `wind_bias` | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | `bool` | `True` | `True` |
| `minimum_station_radius_km` | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | `float` | `0.0` | `0.0` |
| `stations_per_level` | `Sequence[int]` | `(5, 5, … (10 items))` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | `int` | `0` | `0` |
| `barrier_top_level` | `int` | `0` | `10` |
| `barriers` | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | `int` | `1` | `1` |
| `ikine` | `int` | `0` | `0` |
| `iobr` | `int` | `0` | `0` |
| `islope` | `int` | `1` | `1` |
| `critical_froude` | `float` | `1.0` | `1.0` |
| `smoothing_passes` | `Sequence[int]` | `(2, 4, … (10 items))` | `[2, 4, … (10 items)]` |
| `divergence_limit` | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | `int` | `50` | `50` |
| `kinematic_alpha` | `float` | `0.1` | `0.1` |
| `z_faces_m` | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | `int` | `0` | `0` |
| `base_timezone_label` | `str \| None` | `None` | `'UTC-0000'` |
| `terrain_radius_km` | `float` | `15.0` | `15.0` |
| `half_average_angle` | `float` | `30.0` | `30.0` |
| `include_w_temperature` | `bool` | `False` | `False` |
| `compression` | `int` | `1` | `1` |
| `control_lines` | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | `str \| Path \| None` | `None` | `None` |
| `listing_output` | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | `bool` | `False` | `False` |
| `print_interval_hours` | `int` | `1` | `1` |
| `print_uv_levels` | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | `bool` | `False` | `False` |
| `debug_first_step` | `int` | `1` | `1` |
| `debug_last_step` | `int` | `1` | `1` |
| `wind_debug_output` | `int` | `0` | `0` |
| `wind_debug_levels` | `int` | `1` | `0` |
| `wind_debug_fields` | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | `int \| None` | `required` | `610` |
| `ny` | `int \| None` | `required` | `499` |
| `dx_km` | `float \| None` | `required` | `1.0` |
| `dy_km` | `float \| None` | `required` | `1.0` |
| `projection` | `str` | `'LCC'` | `'LCC'` |
| `lat0` | `float` | `27.0` | `27.0` |
| `lon0` | `float` | `117.5` | `117.5` |
| `lat1` | `float` | `30.0` | `30.0` |
| `lat2` | `float` | `60.0` | `60.0` |
| `x_origin_km` | `float` | `0.0` | `0.0` |
| `y_origin_km` | `float` | `0.0` | `0.0` |
| `false_easting_km` | `float` | `0.0` | `0.0` |
| `false_northing_km` | `float` | `0.0` | `0.0` |
| `utm_zone` | `int` | `50` | `50` |
| `hemisphere` | `str` | `'N'` | `'N'` |
| `datum` | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | `int` | `1` | `1` |
| `ring_distances_km` | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | `Sequence[float]` | `()` | `()` |
| `polar_processing` | `int` | `1` | `1` |
| `ttm_scale` | `float` | `1.0` | `1.0` |

</details>
<!-- parameter-reference:end -->

The public processing API is:

```python
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc is a compatibility alias
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

The default API uses regular NumPy calculations, `pyproj`, `datetime`, and a
direct NetCDF flow from CALWRF to CALMET. Workflows that specifically require
the former single-precision operation order or M3D staging can opt in to the
frozen compatibility implementation:

```python
pycalmet.legacy.calwrf(...)
pycalmet.legacy.calmet(...)
```

## Zhejiang example

The bundled example contains three consecutive WRF records, 42 original SRTM3
`.hgt` tiles, and a categorical land-use point dataset. It rebuilds terrain,
land use, and the geophysical grid instead of using a prepared `geo.nc`.
Run all five stages from the repository root:

```bash
sha256sum -c examples/zhejiang/SHA256SUMS
cd terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../..

./examples/zhejiang/run.sh
```

See [examples/zhejiang/README.md](examples/zhejiang/README.md) for provenance,
input boundaries, source data, and generated products.

## Repository layout

```text
src/pycalmet/          Python package and opt-in legacy namespace
tests/                 Unit and integration tests
examples/zhejiang/     Runnable real-data example and control files
terrain/srtm3/         SRTM3 tiles used by the Zhejiang domain
fortran/               Reference Fortran sources; not used at runtime
tools/                 Maintainer utilities
```

The `fortran/` tree contains source material only. Legacy NCL programs,
regional C-shell runners, build wrappers, generated executables, and runtime
products are intentionally excluded from the maintained library.

## Data and output conventions

Canonical outputs are NetCDF files readable by `xarray.open_dataset`. Legacy
formatted and unformatted CALMET-family files remain available when an output
path or compatibility option explicitly requests them. Large inputs should be
kept outside the source tree; generated files belong under `output/`, which is
ignored by Git.

The Zhejiang example uses an LCC grid with a 1 km spacing and a `610 × 499`
domain. Its WRF records cover 2018-03-18 00:00–02:00 UTC. The final record is
the following-hour boundary required to produce the 00:00–01:00 CALMET period.

## Validation

Run the complete Python suite with:

```bash
python -m pytest -q
```

The default suite checks numerical behavior and complete data flow without
requiring a particular compiler or C math library. A focused legacy test keeps
the compatibility namespace separate and verifies its historical
single-precision path. Fortran sources under `fortran/` remain available for
audit and are not an installation dependency.

## Contributing

Keep runtime code in `src/pycalmet`, add focused tests for behavioral changes,
and avoid committing generated model products. Preserve fixed-form layout when
editing reference Fortran. Pull requests should identify the affected stage,
the input configuration, and the exact validation command used.

## Acknowledgements

PyCALMET follows the processing model and file conventions of the CALPUFF
system. The included WRF subset and terrain data are provided to make numerical
validation reproducible; their provenance and checksums are documented beside
the data.
