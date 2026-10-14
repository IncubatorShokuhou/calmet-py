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

| Parameter | Meaning | Python type | Default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `wrf_files` | WRF NetCDF files, in chronological order. | `Sequence[str \| Path] \| None` | `required` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | First timestamp included in the run. | `str \| int \| datetime \| None` | `required` | `'2018031800'` |
| `end` | Last timestamp included in the run. | `str \| int \| datetime \| None` | `required` | `'2018031802'` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `create_surface` | Whether to create the separate legacy surface file. | `bool` | `True` | `False` |
| `compression` | Compression. | `int` | `1` | `1` |
| `i_range` | Inclusive one-based WRF west-east index range. | `tuple[int, int] \| None` | `None` | `None` |
| `j_range` | Inclusive one-based WRF south-north index range. | `tuple[int, int] \| None` | `None` | `None` |
| `k_range` | Inclusive one-based WRF vertical-level range. | `tuple[int, int] \| None` | `None` | `None` |
| `legacy_output` | Path for legacy output. | `str \| Path \| None` | `None` | `None` |
| `surface_output` | Path for surface output. | `str \| Path \| None` | `None` | `None` |
| `title` | Title. | `str` | `'Create 3D.DAT file for WRF output'` | `'Create CALWRF data for the Zhejiang example'` |
| `listing_output` | Path for listing output. | `str \| Path \| None` | `None` | `output/zhejiang/calwrf.lst` |
| `legacy_file_rain_reset` | Reset accumulated rain at each legacy input-file boundary. | `bool` | `False` | `False` |

</details>

<details>
<summary><strong>TERREL</strong> — 49 parameters; Zhejiang control: `examples/zhejiang/config/terrel.inp`.</summary>

| Parameter | Meaning | Python type | Default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `srtm_dir` | Directory containing raw SRTM elevation tiles. | `str \| Path \| None` | `required` | `examples/zhejiang/config` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `datum` | Datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | Terrain sampling method: direct or bilinear/legacy. | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | Path for terrain files. | `Sequence[str \| Path]` | `()` | `[terrain/srtm3/N26E117.hgt, terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | Terrain files with their format and source datum. | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=terrain/…datum='WGS-96'), … (42 items))` |
| `void_fill` | Whether to fill void cells in elevation inputs. | `bool` | `False` | `False` |
| `cell_radius_km` | Cell radius (km). | `float` | `0.0` | `5.0` |
| `model` | Model. | `int` | `1` | `1` |
| `processing_method` | Processing method. | `int \| None` | `None` | `2` |
| `receptor_input` | Path for receptor input. | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | Path for receptor output. | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | Number of columns in the discrete-receptor input. | `int` | `2` | `2` |
| `interpolate_receptors` | Whether to calculate terrain at discrete receptors. | `bool` | `False` | `True` |
| `receptor_radius_km` | Receptor radius (km). | `float` | `0.0` | `0.15` |
| `previous_save` | Path for previous save. | `str \| Path \| None` | `None` | `None` |
| `save_output` | Path for save output. | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | Path for coastline file. | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | Path for gshhs file. | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | Coastline source datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | Coastline wdbii datum. | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | Noise replacement. | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | Noise minimum (m). | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | Missing replacement. | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | Default elevation (m). | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | Path for plot output. | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | Path for raw echo output. | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | Qa hit threshold (%). | `int` | `75` | `75` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Nx. | `int \| None` | `required` | `610` |
| `ny` | Ny. | `int \| None` | `required` | `499` |
| `dx_km` | Dx (km). | `float \| None` | `required` | `1.0` |
| `dy_km` | Dy (km). | `float \| None` | `required` | `1.0` |
| `projection` | Projection. | `str` | `'LCC'` | `'LCC'` |
| `lat0` | Lat0. | `float` | `27.0` | `27.0` |
| `lon0` | Lon0. | `float` | `117.5` | `117.5` |
| `lat1` | Lat1. | `float` | `30.0` | `30.0` |
| `lat2` | Lat2. | `float` | `60.0` | `60.0` |
| `x_origin_km` | X origin (km). | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y origin (km). | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting (km). | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing (km). | `float` | `0.0` | `0.0` |
| `utm_zone` | Utm zone. | `int` | `50` | `50` |
| `hemisphere` | Hemisphere. | `str` | `'N'` | `'N'` |
| `grid_type` | Grid type. | `int` | `1` | `1` |
| `ring_distances_km` | Radial-grid ring distances in kilometres. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | Radial-grid ray angles in degrees. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | Polar processing. | `int` | `1` | `2` |
| `ttm_scale` | Ttm scale. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 43 parameters; Zhejiang control: `examples/zhejiang/config/ctgproc.inp`.</summary>

| Parameter | Meaning | Python type | Default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `landuse_files` | Path for landuse files. | `Sequence[str \| Path] \| None` | `required` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `category_map` | Mapping from input land-use categories to output categories. | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | Input type. | `str` | `'GLAZAS'` | `'GEN'` |
| `mesh_density` | Mesh density. | `int` | `12` | `12` |
| `input_datum` | Input datum. | `str` | `'ESR-S'` | `'WGS-84'` |
| `output_format` | Output format. | `str` | `'auto'` | `'auto'` |
| `coastline_file` | Path for coastline file. | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | Path for gshhs file. | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | Coastline source datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | Coastline wdbii datum. | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | Whether to coastline swap. | `bool` | `False` | `False` |
| `fill_empty_ocean` | Whether to fill empty ocean. | `bool` | `False` | `False` |
| `ocean_category` | Ocean category. | `int` | `55` | `55` |
| `previous_landuse` | Path for previous landuse. | `str \| Path \| None` | `None` | `None` |
| `final_output` | Whether to final output. | `bool` | `True` | `True` |
| `landuse_sources` | Land-use files with format, datum, and mesh density. | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=examples/zhejiang/landuse/zhejiang_generic.landuse, input_type='GEN', input_datum='WGS-84', mesh_density=12))` |
| `snow_processing` | Whether to snow processing. | `bool` | `False` | `False` |
| `qa_cell_output` | Whether to qa cell output. | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | Qa hit threshold (%). | `int` | `75` | `75` |
| `qa_output_directory` | Path for qa output directory. | `str \| Path \| None` | `None` | `None` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Nx. | `int \| None` | `required` | `610` |
| `ny` | Ny. | `int \| None` | `required` | `499` |
| `dx_km` | Dx (km). | `float \| None` | `required` | `1.0` |
| `dy_km` | Dy (km). | `float \| None` | `required` | `1.0` |
| `projection` | Projection. | `str` | `'LCC'` | `'LCC'` |
| `lat0` | Lat0. | `float` | `27.0` | `27.0` |
| `lon0` | Lon0. | `float` | `117.5` | `117.5` |
| `lat1` | Lat1. | `float` | `30.0` | `30.0` |
| `lat2` | Lat2. | `float` | `60.0` | `60.0` |
| `x_origin_km` | X origin (km). | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y origin (km). | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting (km). | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing (km). | `float` | `0.0` | `0.0` |
| `utm_zone` | Utm zone. | `int` | `50` | `50` |
| `hemisphere` | Hemisphere. | `str` | `'N'` | `'N'` |
| `datum` | Datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | Grid type. | `int` | `1` | `1` |
| `ring_distances_km` | Radial-grid ring distances in kilometres. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | Radial-grid ray angles in degrees. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | Polar processing. | `int` | `1` | `1` |
| `ttm_scale` | Ttm scale. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 52 parameters; Zhejiang control: `examples/zhejiang/config/makegeo.inp`.</summary>

| Parameter | Meaning | Python type | Default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `terrain` | Path for terrain. | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `landuse` | Path for landuse. | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `water_threshold` | Water threshold. | `float` | `0.5` | `0.5` |
| `missing_category` | Missing category. | `int` | `55` | `55` |
| `minimum_coverage` | Minimum coverage. | `float` | `0.96` | `0.96` |
| `secondary_landuse` | Path for secondary landuse. | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | Whether to use terrain. | `bool` | `True` | `True` |
| `input_properties` | Surface-property tuple for each input land-use category. | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | Water categories. | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | Output categories. | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | Output water min. | `int` | `50` | `50` |
| `output_water_max` | Output water max. | `int` | `55` | `55` |
| `category_splits` | Input category, output category, and fractional split records. | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | Qa cell. | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | Whether to snow processing. | `bool` | `False` | `False` |
| `snow_data` | Path for snow data. | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | Snow start date. | `int \| None` | `None` | `None` |
| `snow_end_date` | Snow end date. | `int \| None` | `None` | `None` |
| `snow_properties` | Snow-covered surface-property tuple for each category. | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | Snow-age class and decay factor by category. | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | Snow roughness method. | `int` | `1` | `1` |
| `snow_height_scale_m` | Snow height scale (m). | `float` | `1.0` | `10.0` |
| `snow_albedo_method` | Snow albedo method. | `int` | `1` | `1` |
| `snow_depth_minimum_m` | Snow depth minimum (m). | `float` | `0.0` | `0.01` |
| `landuse_plot_output` | Path for landuse plot output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | Path for terrain plot output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | Path for color map output. | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | Image mode. | `int` | `0` | `0` |
| `title` | Title. | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Nx. | `int \| None` | `required` | `610` |
| `ny` | Ny. | `int \| None` | `required` | `499` |
| `dx_km` | Dx (km). | `float \| None` | `required` | `1.0` |
| `dy_km` | Dy (km). | `float \| None` | `required` | `1.0` |
| `projection` | Projection. | `str` | `'LCC'` | `'LCC'` |
| `lat0` | Lat0. | `float` | `27.0` | `27.0` |
| `lon0` | Lon0. | `float` | `117.5` | `117.5` |
| `lat1` | Lat1. | `float` | `30.0` | `30.0` |
| `lat2` | Lat2. | `float` | `60.0` | `60.0` |
| `x_origin_km` | X origin (km). | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y origin (km). | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting (km). | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing (km). | `float` | `0.0` | `0.0` |
| `utm_zone` | Utm zone. | `int` | `50` | `50` |
| `hemisphere` | Hemisphere. | `str` | `'N'` | `'N'` |
| `datum` | Datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | Grid type. | `int` | `1` | `1` |
| `ring_distances_km` | Radial-grid ring distances in kilometres. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | Radial-grid ray angles in degrees. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | Polar processing. | `int` | `1` | `1` |
| `ttm_scale` | Ttm scale. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CALMET</strong> — 165 parameters; Zhejiang control: `examples/zhejiang/config/calmet.inp`.</summary>

| Parameter | Meaning | Python type | Default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `meteorology` | Path for meteorology. | `str \| Path \| Sequence[str \| Path] \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `geophysical` | Path for geophysical. | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/calmet.nc` |
| `start` | First timestamp included in the run. | `str \| datetime \| None` | `required` | `'2018-03-18_00:00:00'` |
| `end` | Last timestamp included in the run. | `str \| datetime \| None` | `required` | `'2018-03-18_01:00:00'` |
| `noobs` | Observation mode: prognostic-only or diagnostic use of observations. | `int` | `2` | `2` |
| `run_type` | CALMET run type: wind-only or wind plus micrometeorology. | `int` | `1` | `1` |
| `save_output` | Whether to save output. | `bool` | `True` | `True` |
| `legacy_output_format` | Traditional CALMET output format selector. | `int` | `1` | `1` |
| `test_mode` | Test mode. | `int` | `2` | `2` |
| `regulatory_mode` | Regulatory mode. | `int` | `0` | `0` |
| `timestep_seconds` | Timestep (s). | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | Prognostic timestep (s). | `int` | `3600` | `3600` |
| `cloud_method` | Cloud method. | `int` | `4` | `4` |
| `cloud_data` | Path for cloud data. | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | Cloud data format. | `int` | `2` | `2` |
| `cloud_output` | Cloud-field output option. | `int` | `0` | `0` |
| `surface_data` | Path for surface data. | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | Surface data format. | `int` | `2` | `2` |
| `surface_control_stations` | Surface-station ID, number, coordinates, time zone, and anemometer height. | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | Path for upper air files. | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | Upper-air station ID, number, coordinates, and time zone. | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | Path for sea files. | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | Path for diagnostic data. | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | Diagnostic surface temperature. | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | Use of observed lapse rates in diagnostic fields. | `int` | `0` | `0` |
| `diagnostic_initial_wind` | Diagnostic initial wind. | `int` | `0` | `0` |
| `diagnostic_surface_wind` | Diagnostic surface wind. | `int` | `0` | `0` |
| `diagnostic_upper_wind` | Diagnostic upper wind. | `int` | `0` | `0` |
| `initial_guess_upper_station` | Initial guess upper station. | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | Initial guess height range (m). | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | Use a previous CALMET field as the initial wind guess. | `int` | `0` | `0` |
| `initial_guess_calmet_files` | Path for initial guess calmet files. | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | Path for weight data. | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | Precipitation station count. | `int` | `-1` | `-1` |
| `precipitation_data` | Path for precipitation data. | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | Precipitation data format. | `int` | `2` | `2` |
| `precipitation_control_stations` | Precipitation station ID, number, and coordinates. | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | Prognostic wind-field input and interpolation option. | `int` | `14` | `14` |
| `irhprog` | Relative-humidity source option for prognostic data. | `int` | `1` | `1` |
| `itprog` | Temperature source option for prognostic data. | `int` | `2` | `2` |
| `imixh` | Mixing-height calculation method selector. | `int` | `1` | `1` |
| `constb` | Neutral mechanical mixing-height coefficient B. | `float` | `1.41` | `1.41` |
| `conste` | Convective mixing-height entrainment coefficient. | `float` | `0.15` | `0.15` |
| `constn` | Stable mechanical mixing-height coefficient N. | `float` | `2400.0` | `2400.0` |
| `constw` | Convective velocity-scale coefficient W. | `float` | `0.16` | `0.16` |
| `coriolis` | Coriolis. | `float` | `0.0001` | `0.0001` |
| `minimum_lapse_rate` | Minimum potential-temperature lapse rate. | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | Mixing lapse depth (m). | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | Minimum mixing height (m). | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | Maximum mixing height (m). | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | Minimum water mixing height (m). | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | Maximum water mixing height (m). | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | Land flux threshold. | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | Water flux threshold. | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | Water temperature mode. | `int` | `0` | `0` |
| `prognostic_ocean_category` | Prognostic ocean category. | `int` | `16` | `16` |
| `overwater_flux_method` | Overwater flux method. | `int` | `10` | `10` |
| `coastal_shelf_length_km` | Coastal shelf length (km). | `float` | `0.0` | `0.0` |
| `warm_layer` | COARE warm-layer correction option. | `int` | `0` | `0` |
| `cool_skin` | COARE cool-skin correction option. | `int` | `0` | `0` |
| `convective_relaxation` | Convective relaxation. | `int` | `1` | `1` |
| `convective_relaxation_time_s` | Convective relaxation time (s). | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | Precipitation interpolation. | `int` | `2` | `2` |
| `precipitation_radius_km` | Precipitation radius (km). | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | Precipitation cutoff. | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | Turbidity coefficient 1. | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | Turbidity coefficient 2. | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | Cloudiness coefficient 1. | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | Cloudiness coefficient 2. | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | Net radiation coefficient 1. | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | Net radiation coefficient 2. | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | Net radiation coefficient 3. | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | Surface temperature option. | `int` | `-1` | `-1` |
| `upper_temperature_option` | Upper temperature option. | `int` | `-1` | `-1` |
| `lapse_height_m` | Lapse height (m). | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | Temperature interpolation power. | `int` | `1` | `1` |
| `temperature_radius_km` | Temperature radius (km). | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | Temperature station limit. | `int` | `5` | `5` |
| `water_temperature_gradient_below` | Water temperature gradient below. | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | Water temperature gradient above. | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | Water temperature category min. | `int` | `55` | `55` |
| `water_temperature_category_max` | Water temperature category max. | `int` | `55` | `55` |
| `iavet` | Temperature-field vertical averaging option. | `int` | `1` | `1` |
| `iavezi` | Mixing-height wind averaging option. | `int` | `1` | `1` |
| `mixing_wind_level` | Mixing wind level. | `int` | `1` | `1` |
| `mnmdav` | Minimum number of valid values required for averaging. | `int` | `1` | `10` |
| `iwfcod` | Wind-field model formulation code. | `int` | `1` | `1` |
| `wind_extrapolation` | Wind extrapolation. | `int` | `-1` | `-1` |
| `upper_air_separation_cells` | Upper air separation cells. | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | Extrapolate calm. | `int` | `0` | `0` |
| `wind_bias` | Per-level additive wind-direction or component bias controls. | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | Whether to varying radius. | `bool` | `True` | `True` |
| `minimum_station_radius_km` | Minimum station radius (km). | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | Maximum surface radius (km). | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | Maximum upper radius (km). | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | Maximum water radius (km). | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | Surface background radius (km). | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | Upper background radius (km). | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | Prognostic radius (km). | `float` | `0.0` | `0.0` |
| `stations_per_level` | Maximum observation stations used at each vertical level. | `Sequence[int]` | `(5, 5, … (10 items))` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | Per-level factors used to extrapolate winds vertically. | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | Barrier count. | `int` | `0` | `0` |
| `barrier_top_level` | Barrier top level. | `int` | `0` | `10` |
| `barriers` | Barrier start and end coordinates for flow blocking. | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | Lake-breeze boxes, coastline segments, and assigned stations. | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | Froude-number terrain-flow adjustment switch. | `int` | `1` | `1` |
| `ikine` | Kinematic terrain-effect adjustment switch. | `int` | `0` | `0` |
| `iobr` | Blocking-effect adjustment switch. | `int` | `0` | `0` |
| `islope` | Slope-flow adjustment switch. | `int` | `1` | `1` |
| `critical_froude` | Critical froude. | `float` | `1.0` | `1.0` |
| `smoothing_passes` | Number of wind-field smoothing passes at each level. | `Sequence[int]` | `(2, 4, … (10 items))` | `[2, 4, … (10 items)]` |
| `divergence_limit` | Divergence limit. | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | Divergence iterations. | `int` | `50` | `50` |
| `kinematic_alpha` | Relaxation coefficient for the kinematic wind adjustment. | `float` | `0.1` | `0.1` |
| `z_faces_m` | Heights of vertical layer faces; one more value than model layers. | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | Base timezone. | `int` | `0` | `0` |
| `base_timezone_label` | Base timezone label. | `str \| None` | `None` | `'UTC-0000'` |
| `terrain_radius_km` | Terrain radius (km). | `float` | `15.0` | `15.0` |
| `half_average_angle` | Half average angle. | `float` | `30.0` | `30.0` |
| `include_w_temperature` | Whether to include w temperature. | `bool` | `False` | `False` |
| `compression` | Compression. | `int` | `1` | `1` |
| `control_lines` | Original CALMET control-file lines copied into legacy output metadata. | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | Path for mesopac output. | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | Path for distance to coast output. | `str \| Path \| None` | `None` | `None` |
| `listing_output` | Path for listing output. | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | Whether to print fields. | `bool` | `False` | `False` |
| `print_interval_hours` | Print interval (hours). | `int` | `1` | `1` |
| `print_uv_levels` | Print uv levels. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | Print w levels. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | Print temperature levels. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | Switches selecting surface fields for printed output. | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | Whether to micrometeorology debug. | `bool` | `False` | `False` |
| `debug_first_step` | Debug first step. | `int` | `1` | `1` |
| `debug_last_step` | Debug last step. | `int` | `1` | `1` |
| `wind_debug_output` | Wind diagnostic-output option. | `int` | `0` | `0` |
| `wind_debug_levels` | Wind debug levels. | `int` | `1` | `0` |
| `wind_debug_fields` | Switches selecting wind-model diagnostic fields. | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | Path for final wind debug output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | Path for kinematic wind debug output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | Path for froude wind debug output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | Path for slope wind debug output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | Path for wind print output. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Nx. | `int \| None` | `required` | `610` |
| `ny` | Ny. | `int \| None` | `required` | `499` |
| `dx_km` | Dx (km). | `float \| None` | `required` | `1.0` |
| `dy_km` | Dy (km). | `float \| None` | `required` | `1.0` |
| `projection` | Projection. | `str` | `'LCC'` | `'LCC'` |
| `lat0` | Lat0. | `float` | `27.0` | `27.0` |
| `lon0` | Lon0. | `float` | `117.5` | `117.5` |
| `lat1` | Lat1. | `float` | `30.0` | `30.0` |
| `lat2` | Lat2. | `float` | `60.0` | `60.0` |
| `x_origin_km` | X origin (km). | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y origin (km). | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting (km). | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing (km). | `float` | `0.0` | `0.0` |
| `utm_zone` | Utm zone. | `int` | `50` | `50` |
| `hemisphere` | Hemisphere. | `str` | `'N'` | `'N'` |
| `datum` | Datum. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | Grid type. | `int` | `1` | `1` |
| `ring_distances_km` | Radial-grid ring distances in kilometres. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | Radial-grid ray angles in degrees. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | Polar processing. | `int` | `1` | `1` |
| `ttm_scale` | Ttm scale. | `float` | `1.0` | `1.0` |

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
