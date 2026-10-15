# PyCALMET

[中文文档](README.zh-CN.md)

PyCALMET is a NumPy implementation of the CALMET meteorological preprocessing
and diagnostic wind-field workflow. It provides Python APIs for CALWRF,
TERREL, CTGPROC, MAKEGEO, and CALMET, with NetCDF as the preferred interchange
format and compatibility readers and writers for legacy CALPUFF files.

The package is intended for reproducible research, automated data processing,
and modernization of existing CALMET workflows. A complete Zhejiang example,
its original terrain and land-use sources, and the Fortran sources used as
numerical references are kept in separate, clearly scoped directories.

## Highlights

- Pure Python and NumPy/SciPy execution; no Fortran compiler is required.
- Five public processing stages plus an end-to-end pipeline exposed through one package.
- WRF NetCDF input and CF-compatible NetCDF output through `netCDF4` only.
- FP32 model fields with explicit high-precision boundaries for time, geodesy,
  spatial searches, and sensitive reductions.
- Rasterio-backed standard rasters, Shapely geometry, and SciPy spatial
  indexing instead of bespoke general-purpose implementations.
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

`run_pipeline` exposes the complete five-stage workflow as one typed keyword
API. Stage prefixes keep same-named controls unambiguous. Among output paths,
only the final CALMET destination is needed because the pipeline manages every
interstage dataset:

```python
from datetime import datetime
from pathlib import Path

import pycalmet

example = Path("examples/zhejiang")
grid = pycalmet.GridConfig(
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
)

result = pycalmet.run_pipeline(
    inp=None,
    calwrf_wrf_files=[
        example / "wrfout_d01_2018-03-18_00_00_00.nc",
        example / "wrfout_d01_2018-03-18_01_00_00.nc",
        example / "wrfout_d01_2018-03-18_02_00_00.nc",
    ],
    calwrf_start=datetime(2018, 3, 18, 0),
    calwrf_end=datetime(2018, 3, 18, 2),
    terrel_srtm_dir=Path("data/terrain/srtm3"),
    terrel_grid=grid,
    terrel_sampling="bilinear",
    ctgproc_landuse_files=[Path("data/landuse/usgs-glcc/EAUSGS1.IMG")],
    ctgproc_grid=grid,
    ctgproc_input_type="GLAZAS",
    ctgproc_input_datum="ESR-S",
    ctgproc_mesh_density=12,
    makegeo_grid=grid,
    calmet_start=datetime(2018, 3, 18, 0),
    calmet_end=datetime(2018, 3, 18, 1),
    calmet_grid=grid,
    calmet_noobs=2,
    calmet_iprog=14,
    calmet_irhprog=1,
    calmet_itprog=2,
    calmet_coriolis=1.0e-4,
    calmet_maximum_mixing_height_m=2500.0,
    calmet_maximum_water_mixing_height_m=2500.0,
    calmet_precipitation_radius_km=50.0,
    calmet_mnmdav=10,
    calmet_wind_extrapolation=-1,
    calmet_precipitation_station_count=-1,
    calmet_include_w_temperature=False,
    output=Path("output/zhejiang/calmet.nc"),
)
```

The Zhejiang land-use source is the original `EAUSGS1.IMG`: a `13000 × 12000`
headerless unsigned 8-bit raster for the Asia-optimized USGS Global/GLCC-style
Lambert azimuthal equal-area grid. `GLAZAS` selects that fixed raster geometry,
`ESR-S` identifies its source datum, and `mesh_density=12` is the Python form of
the traditional `MESHGLAZ=12` resampling control. The source datum is distinct
from the target `GridConfig.datum="WGS-84"` and therefore remains explicit.

Every field in the stage parameter reference below is available under its
`calwrf_`, `terrel_`, `ctgproc_`, `makegeo_`, or `calmet_` prefix. For example,
CALMET `z_faces_m` becomes `calmet_z_faces_m`. A stage grid can be a
`GridConfig`, as above, or prefixed flat controls such as `terrel_nx`,
`terrel_ny`, and `terrel_dx_km`. Python defaults match the individual stage
APIs. The pipeline rewires interstage paths and disables intermediate and
diagnostic outputs to honor its single-final-product contract, so those paths
do not need to be supplied in parameter-only mode.

That contract requires `makegeo_snow_processing=False`,
`calmet_save_output=True`, and `calmet_test_mode=2`: snow processing creates
multiple dated GEO files, while the other two values are required to perform
and save the final CALMET calculation.

The stages exchange NetCDF datasets inside an automatically cleaned temporary
directory. The pipeline disables traditional M3D/M2D, listing, QA, plot, save,
and debug products, so the selected CALMET file is the only persistent output
when using NetCDF as shown above. The legacy CALMET final format also writes
its required `.aux` companion.

As a convenience for an existing control-file workflow, the first argument
may instead be a **directory** containing all five control files:

```python
result = pycalmet.run_pipeline(
    Path("examples/zhejiang/config"),  # folder, not a single .inp file
    output=Path("output/zhejiang/calmet.nc"),
)
```

This directory form discovers `calwrf.inp`, `terrel.inp`, `ctgproc.inp`,
`makegeo.inp`, and `calmet.inp`. If `output` is omitted, it uses the effective
output path from `calmet.inp`. The grouped `calwrf=`, `terrel=`, `ctgproc=`,
`makegeo=`, and `calmet=` inputs also accept one control-file path, parameter
mapping, or typed configuration object per stage. `PipelineConfig` groups the
five typed stage configurations and can be passed as the first argument to
`run_pipeline`. The runnable `examples/zhejiang/run.py` instead demonstrates
the parameter-only pipeline form shown in Quick start; execute it with
`./examples/zhejiang/run.sh`. To run and inspect individual stages using the
five traditional control files, use `examples/zhejiang/run_stages.py`.

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
Where the original program defines a usable scalar or model-control default,
the keyword API uses it. File paths stay explicit instead of inheriting legacy
working-directory filenames; controls with a mandatory-value sentinel or no
Fortran default use the documented Python default in the table below. For
example, this is the prognostic-only setup used by the Zhejiang WRF workflow:

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
    noobs=2,
    iprog=14,
    irhprog=1,
    itprog=2,
    precipitation_station_count=-1,
    wind_extrapolation=-1,
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
)
```

The signatures list every supported control, including controls with Fortran
defaults that may be absent from a particular `.inp` file. They are visible to
IDE completion, type checkers, `help()`, and `inspect.signature()`.

<!-- parameter-reference:start -->
## Complete parameter reference

The tables below are generated from the public function signatures and typed
configuration models. “Python default” is the value used in `inp=None` mode when the
caller omits that parameter; “Zhejiang value” is the effective value after
parsing the bundled control file. `inp` itself and `**legacy_parameters` are
entry/compatibility mechanisms rather than model controls, so they are not
repeated in each table. Paths are relative to the repository root. Long
sequences are abbreviated with their total item count. Meanings and option
effects are based primarily on Scire, Robe, Fernau, and Yamartino, *A User's
Guide for the CALMET Meteorological Model (Version 5)* (January 2000), especially
§2 and §4.2–4.3. That guide predates controls such as NOOBS, the newer
prognostic-data switches, COARE, and snow processing; those controls are checked
against the bundled CALMET 6.5 templates, reference Fortran sources, and the
Python code that consumes each value. Regenerate the tables
with `python tools/generate_parameter_reference.py`; use `--check` in CI.

<details>
<summary><strong>CALWRF</strong> — 13 parameters; Zhejiang control: `examples/zhejiang/config/calwrf.inp`.</summary>

| Parameter | Meaning | Python type | Python default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `wrf_files` | WRF NetCDF files, in chronological order. | `Sequence[str \| Path] \| None` | `required` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | First naive, second-precision timestamp included in the run; accepts datetime or YYYYMMDDHH/ISO-like text. | `datetime \| str \| None` | `required` | `2018-03-18 00:00:00` |
| `end` | Last naive, second-precision timestamp included in the run; accepts datetime or YYYYMMDDHH/ISO-like text. | `datetime \| str \| None` | `required` | `2018-03-18 02:00:00` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `create_surface` | Create the separate legacy M2D surface file when legacy_output is also requested. | `bool` | `True` | `False` |
| `compression` | NetCDF deflate compression level for CALWRF output (0 disables compression). | `int` | `1` | `1` |
| `i_range` | Inclusive one-based WRF west-east index range. | `tuple[int, int] \| None` | `None` | `None` |
| `j_range` | Inclusive one-based WRF south-north index range. | `tuple[int, int] \| None` | `None` | `None` |
| `k_range` | Inclusive one-based WRF vertical-level range. | `tuple[int, int] \| None` | `None` | `None` |
| `legacy_output` | Optional traditional CALWRF 3D.DAT/M3D output generated alongside the primary NetCDF file. | `str \| Path \| None` | `None` | `None` |
| `surface_output` | Optional traditional CALWRF two-dimensional surface/M2D path, written when create_surface is true and legacy_output is requested. | `str \| Path \| None` | `None` | `None` |
| `title` | Descriptive title stored in CALWRF output metadata. | `str` | `'Create 3D.DAT file for WRF output'` | `'Create CALWRF data for the Zhejiang example'` |
| `listing_output` | CALWRF processing log/listing destination. | `str \| Path \| None` | `None` | `output/zhejiang/calwrf.lst` |

</details>

<details>
<summary><strong>TERREL</strong> — 49 parameters; Zhejiang control: `examples/zhejiang/config/terrel.inp`.</summary>

| Parameter | Meaning | Python type | Python default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `srtm_dir` | Directory scanned for raw SRTM `[NS]dd[EW]ddd.hgt` tiles when terrain_sources is empty. | `str \| Path \| None` | `required` | `examples/zhejiang/config` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `datum` | Output-grid datum/region code; individual input sources specify their own input datum separately. | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | Terrain coordinate conversion: `direct` transforms every DEM sample (more accurate); `bilinear`/`legacy` transforms sheet corners and interpolates interior coordinates. | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | Selected terrain-file paths; relative SRTM tile names are resolved inside srtm_dir. | `Sequence[str \| Path]` | `()` | `[data/terrain/srtm3/N26E117.hgt, data/terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | Terrain-source records `(path, input_type, input_datum)`; input_type selects the DEM reader (for example SRTM1/3, GTOPO30, CDED, GEN, or GEOTIFF). | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=data/terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=dat…datum='WGS-96'), … (42 items))` |
| `void_fill` | Fill empty output terrain cells from nearby valid cells, searching no farther than cell_radius_km. | `bool` | `False` | `False` |
| `cell_radius_km` | Maximum search radius, in kilometres, used to find valid neighbours when void_fill is enabled; use at least one grid spacing. | `float` | `0.0` | `5.0` |
| `model` | TERREL output model: CALMET (1), MESOPAC (2), ISC polar (3), ISC Cartesian (4), NUATMOS (5), or generic (6); this also selects cell-average versus peak elevations. | `int` | `1` | `1` |
| `processing_method` | TERREL processing method; for polar grids, 1 selects NORMAL sectors and 2 SCREEN sectors, overriding grid.polar_processing. | `int \| None` | `None` | `2` |
| `receptor_input` | Optional discrete-receptor XY input file; supplying it enables receptor processing. | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | Output file containing the receptor rows with TERREL-calculated terrain elevation. | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | Discrete-receptor input width: 2 for (X, Y), or 4 for (X, Y, terrain elevation, receptor height); TERREL replaces column 3. | `int` | `4` | `2` |
| `interpolate_receptors` | Discrete-receptor method: interpolate elevation when true; search for the peak elevation when false. | `bool` | `False` | `True` |
| `receptor_radius_km` | Search/interpolation radius for discrete-receptor elevation, in kilometres; peak-search mode requires a positive value. | `float` | `0.0` | `0.15` |
| `previous_save` | TERREL binary accumulator from an earlier pass, used to resume sample sums and hit counts. | `str \| Path \| None` | `None` | `None` |
| `save_output` | Write a TERREL binary accumulator containing sample sums and hit counts for a later continuation pass. | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | Processed BLN coastline input; when gshhs_file is supplied, this path receives the generated BLN data. | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | Raw GSHHS coastline database used to generate the processed BLN coastline. | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | Datum/region code assigned to coordinates read from the raw GSHHS/coastline source. | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | Datum/region code assigned to WDBII coastline coordinates. | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | Per coastline class noise rule: 0 no test; 1 values below noise_minimum_m become missing; 2 clamp them to the minimum; 3 replace them with default_elevation_m. | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | Minimum valid terrain elevation used by the noise tests, by database class. | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | Per coastline class void rule: 0 retain; 1 replace only final empty cells; 2 replace missing samples and final cells; 3 force the class to default elevation. | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | Fallback elevation used when terrain samples remain missing. | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | Optional Surfer ASCII grid containing TERREL terrain elevations for QA/plotting. | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | Optional ASCII dump of raw unfiltered terrain samples as `(x_km, y_km, elevation_m)`. | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | Flag cells whose sample count is below this percentage (0–100) of the domain-average hit count. | `int` | `75` | `75` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings. | `int \| None` | `required` | `610` |
| `ny` | Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of rays. | `int \| None` | `required` | `499` |
| `dx_km` | Cartesian X grid spacing in kilometres; independent of dy_km and unused for polar ring positions. | `float \| None` | `required` | `1.0` |
| `dy_km` | Cartesian Y grid spacing in kilometres; independent of dx_km and unused for polar ray positions. | `float \| None` | `required` | `1.0` |
| `projection` | Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert azimuthal equal-area (LAZA). | `str` | `'UTM'` | `'LCC'` |
| `lat0` | Signed latitude of the projection origin or tangent point, in degrees (north positive); EM fixes it at zero. | `float` | `27.0` | `27.0` |
| `lon0` | Signed central-meridian or tangent-point longitude, in degrees (east positive). | `float` | `117.5` | `117.5` |
| `lat1` | First standard latitude for LCC, or true-scale latitude for PS. | `float` | `30.0` | `30.0` |
| `lat2` | Second standard latitude for LCC; unused by PS. | `float` | `60.0` | `60.0` |
| `x_origin_km` | Easting of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `y_origin_km` | Northing of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM zone number from 1 through 60; used only for UTM. | `int` | `50` | `50` |
| `hemisphere` | UTM hemisphere, N or S; used only for UTM. | `str` | `'N'` | `'N'` |
| `grid_type` | TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2 Cartesian origin at its centre; 3 polar origin at ring centre. | `int` | `1` | `1` |
| `ring_distances_km` | TERREL polar-grid ring radii, in kilometres; supply one value per ring when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL polar-grid ray azimuths, in degrees clockwise from north; used when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL-only polar method: 1 NORMAL sectors lie halfway between adjacent rings/rays; 2 SCREEN sectors span the current to next outer ring and halfway between rays. | `int` | `2` | `2` |
| `ttm_scale` | Scale factor on the central meridian for TTM projection. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 43 parameters; Zhejiang control: `examples/zhejiang/config/ctgproc.inp`.</summary>

| Parameter | Meaning | Python type | Python default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `landuse_files` | Input land-use database file paths consumed by CTGPROC. | `Sequence[str \| Path] \| None` | `required` | `[data/landuse/usgs-glcc/EAUSGS1.IMG]` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `category_map` | Post-binning map from source to target land-use categories; hit counts mapped to the same target are summed. | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | Land-use database reader name/code; GLAZAS selects the fixed 13000 × 12000 Asia USGS Global/GLCC-style LAEA raster, while other readers include CTG, GEN, NLCD, CORINE, GLC2K, UMDGLC, and MODIS. | `str` | `'CTG'` | `'GLAZAS'` |
| `mesh_density` | Per-side subdivision count for each input land-use cell, producing mesh_density² sample points; this is the MESHGLAZ control for GLAZ inputs and must be at least 1. | `int` | `1` | `12` |
| `input_datum` | Source datum for land-use formats without usable datum/CRS metadata; headerless GLAZAS data therefore requires the explicit source datum. | `str` | `'NAS-C'` | `'ESR-S'` |
| `output_format` | Output encoding: choose from the filename (auto), traditional CTGPROC format (legacy), or NetCDF. | `str` | `'auto'` | `'auto'` |
| `coastline_file` | Processed BLN coastline input; when gshhs_file is supplied, this path receives the generated BLN data. | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | Raw GSHHS coastline database used to generate the processed BLN coastline. | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | Datum/region code assigned to coordinates read from the raw GSHHS/coastline source. | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | Datum/region code assigned to WDBII coastline coordinates. | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | While reading samples, replace each point lying offshore of the supplied coastline with ocean_category. | `bool` | `False` | `False` |
| `fill_empty_ocean` | On the final pass, fill zero-hit cells whose centre lies offshore with ocean_category; None follows the Fortran default and enables this only when coastline_file is supplied. | `bool \| None` | `None → coastline_file is not None` | `False` |
| `ocean_category` | Land-use category assigned to ocean cells. | `int` | `55` | `55` |
| `previous_landuse` | Prior CTGPROC integer hit-count output to continue a multi-pass land-use accumulation. | `str \| Path \| None` | `None` | `None` |
| `final_output` | CTGPROC pass type: true writes fractional land use for MAKEGEO; false writes integer hit counts for continuation by previous_landuse. | `bool` | `True` | `True` |
| `landuse_sources` | Land-use-source records `(path, input_type, input_datum, mesh_density)`. | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=data/landuse/usgs-glcc/EAUSGS1.IMG, input_type='GLAZAS', input_datum='ESR-S', mesh_density=12))` |
| `snow_processing` | Enable the stage's snow workflow: CTGPROC writes SNODAS reference records; MAKEGEO creates daily snow-adjusted GEO files. | `bool` | `False` | `False` |
| `qa_cell_output` | Write CTGPROC QA coordinate files for original and meshed input points; output may be large. | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | Flag cells whose sample count is below this percentage (0–100) of the domain-average hit count. | `int` | `75` | `75` |
| `qa_output_directory` | Directory receiving CTGPROC QACTG/QAGLAZ/QAMESH coordinate QA files. | `str \| Path \| None` | `None` | `None` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings. | `int \| None` | `required` | `610` |
| `ny` | Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of rays. | `int \| None` | `required` | `499` |
| `dx_km` | Cartesian X grid spacing in kilometres; independent of dy_km and unused for polar ring positions. | `float \| None` | `required` | `1.0` |
| `dy_km` | Cartesian Y grid spacing in kilometres; independent of dx_km and unused for polar ray positions. | `float \| None` | `required` | `1.0` |
| `projection` | Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert azimuthal equal-area (LAZA). | `str` | `'UTM'` | `'LCC'` |
| `lat0` | Signed latitude of the projection origin or tangent point, in degrees (north positive); EM fixes it at zero. | `float` | `27.0` | `27.0` |
| `lon0` | Signed central-meridian or tangent-point longitude, in degrees (east positive). | `float` | `117.5` | `117.5` |
| `lat1` | First standard latitude for LCC, or true-scale latitude for PS. | `float` | `30.0` | `30.0` |
| `lat2` | Second standard latitude for LCC; unused by PS. | `float` | `60.0` | `60.0` |
| `x_origin_km` | Easting of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `y_origin_km` | Northing of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM zone number from 1 through 60; used only for UTM. | `int` | `50` | `50` |
| `hemisphere` | UTM hemisphere, N or S; used only for UTM. | `str` | `'N'` | `'N'` |
| `datum` | Output-grid datum/region code; individual input sources specify their own input datum separately. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2 Cartesian origin at its centre; 3 polar origin at ring centre. | `int` | `1` | `1` |
| `ring_distances_km` | TERREL polar-grid ring radii, in kilometres; supply one value per ring when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL polar-grid ray azimuths, in degrees clockwise from north; used when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL-only polar method: 1 NORMAL sectors lie halfway between adjacent rings/rays; 2 SCREEN sectors span the current to next outer ring and halfway between rays. | `int` | `2` | `2` |
| `ttm_scale` | Scale factor on the central meridian for TTM projection. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 52 parameters; Zhejiang control: `examples/zhejiang/config/makegeo.inp`.</summary>

| Parameter | Meaning | Python type | Python default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `terrain` | Gridded elevation input, normally TERREL output; read only when use_terrain is true. | `str \| Path \| None` | `required` | `output/zhejiang/terrain.nc` |
| `landuse` | Primary final fractional land-use data set produced by CTGPROC. | `str \| Path \| None` | `required` | `output/zhejiang/landuse.nc` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `water_threshold` | Water-category fraction required before dominant land use is selected only among water categories; otherwise selection is among non-water categories. | `float` | `0.5` | `0.5` |
| `missing_category` | Category inserted when total usable land-use coverage is below 0.001; it must exist in input_properties. | `int` | `55` | `55` |
| `minimum_coverage` | Minimum acceptable total land-use fraction: values at or above it are renormalized, values below 0.001 are missing, and intermediate values are errors. | `float` | `0.96` | `0.96` |
| `secondary_landuse` | Optional secondary land-use data used only where the primary data set has missing cells. | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | Read terrain elevations into GEO output when true; write zero terrain when false. | `bool` | `True` | `True` |
| `input_properties` | Per-category `(roughness_m, albedo, Bowen_ratio, soil_heat_flux_parameter, anthropogenic_heat_W_m-2, leaf_area_index, output_category)` records; roughness uses logarithmic area weighting and the other properties arithmetic weighting. | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | Input land-use categories treated as water. | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | Ordered land-use categories retained in the geophysical output. | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | Lowest output category regarded as water by CALMET. | `int` | `50` | `50` |
| `output_water_max` | Highest output category regarded as water by CALMET. | `int` | `55` | `55` |
| `category_splits` | MAKEGEO `(source_category, target_category, percent)` records applied before property weighting; percentages for a source should total 100. | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | Optional one-based `(i, j)` cell for reporting intermediate terrain, land-use, and surface-property calculations. | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | Create one snow-adjusted GEO file per day from SNODAS depth/age data over the inclusive snow_start_date..snow_end_date range. | `bool` | `False` | `False` |
| `snow_data` | SNODAS snow-depth/age input used to build daily snow-adjusted geophysical files. | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | First calendar date covered by snow processing; accepts date, midnight datetime, or YYYY-MM-DD/YYYYMMDD text. | `date \| datetime \| str \| None` | `None` | `None` |
| `snow_end_date` | Last calendar date covered by snow processing; accepts date, midnight datetime, or YYYY-MM-DD/YYYYMMDD text. | `date \| datetime \| str \| None` | `None` | `None` |
| `snow_properties` | Snow-covered records with the same seven fields as input_properties; the mapped output category must match the normal record. | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | Per snow class `(age_time_scale_days, albedo_reduction_factor)` used by the ageing-albedo method. | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | Snow roughness method: 1 reduces inferred obstacle height by snow depth; 2 uses the user snow roughness wherever snow is present. | `int` | `1` | `1` |
| `snow_height_scale_m` | HSCL obstacle-height-to-roughness scale used by snow roughness method 1. | `float` | `10.0` | `10.0` |
| `snow_albedo_method` | Snow albedo method: 1 ages table albedo toward a reduced value; 2 uses user snow albedo wherever snow is present. | `int` | `1` | `1` |
| `snow_depth_minimum_m` | Minimum snow depth required to apply snow properties. | `float` | `0.01` | `0.01` |
| `landuse_plot_output` | Optional Surfer ASCII grid of the dominant MAKEGEO land-use category. | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | Optional Surfer ASCII grid of MAKEGEO terrain elevation. | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | Optional Surfer colour-map companion for the land-use plot grid. | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | QA Surfer-grid coordinate convention: cell-centre bounds (0) or cell-edge bounds (1). | `int` | `0` | `0` |
| `title` | GEO metadata title, limited by the traditional format to 80 characters. | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings. | `int \| None` | `required` | `610` |
| `ny` | Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of rays. | `int \| None` | `required` | `499` |
| `dx_km` | Cartesian X grid spacing in kilometres; independent of dy_km and unused for polar ring positions. | `float \| None` | `required` | `1.0` |
| `dy_km` | Cartesian Y grid spacing in kilometres; independent of dx_km and unused for polar ray positions. | `float \| None` | `required` | `1.0` |
| `projection` | Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert azimuthal equal-area (LAZA). | `str` | `'UTM'` | `'LCC'` |
| `lat0` | Signed latitude of the projection origin or tangent point, in degrees (north positive); EM fixes it at zero. | `float` | `27.0` | `27.0` |
| `lon0` | Signed central-meridian or tangent-point longitude, in degrees (east positive). | `float` | `117.5` | `117.5` |
| `lat1` | First standard latitude for LCC, or true-scale latitude for PS. | `float` | `30.0` | `30.0` |
| `lat2` | Second standard latitude for LCC; unused by PS. | `float` | `60.0` | `60.0` |
| `x_origin_km` | Easting of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `y_origin_km` | Northing of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM zone number from 1 through 60; used only for UTM. | `int` | `50` | `50` |
| `hemisphere` | UTM hemisphere, N or S; used only for UTM. | `str` | `'N'` | `'N'` |
| `datum` | Output-grid datum/region code; individual input sources specify their own input datum separately. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2 Cartesian origin at its centre; 3 polar origin at ring centre. | `int` | `1` | `1` |
| `ring_distances_km` | TERREL polar-grid ring radii, in kilometres; supply one value per ring when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL polar-grid ray azimuths, in degrees clockwise from north; used when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL-only polar method: 1 NORMAL sectors lie halfway between adjacent rings/rays; 2 SCREEN sectors span the current to next outer ring and halfway between rays. | `int` | `2` | `2` |
| `ttm_scale` | Scale factor on the central meridian for TTM projection. | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CALMET</strong> — 166 parameters; Zhejiang control: `examples/zhejiang/config/calmet.inp`.</summary>

| Parameter | Meaning | Python type | Python default | Zhejiang value |
| --- | --- | --- | --- | --- |
| `meteorology` | Gridded prognostic meteorology input; iprog selects whether it supplies a Step-1 field, initial guess, or pseudo-observations. | `str \| Path \| Sequence[str \| Path] \| None` | `required` | `output/zhejiang/calwrf.nc` |
| `geophysical` | GEO geophysical grid containing terrain, land use, roughness, albedo, Bowen ratio, soil/anthropogenic heat parameters, and leaf-area index. | `str \| Path \| None` | `required` | `output/zhejiang/geo.nc` |
| `output` | Primary output file path. | `str \| Path \| None` | `required` | `output/zhejiang/calmet.nc` |
| `start` | Inclusive lower boundary of the CALMET output interval `[start, end)`; meteorological boundary times are selected with `start < t <= end`, and each output timestamp is `t - timestep_seconds`. | `datetime \| str \| None` | `required` | `2018-03-18 00:00:00` |
| `end` | Exclusive upper boundary of the CALMET output interval `[start, end)`; meteorological boundary times are selected with `start < t <= end`, and each output timestamp is `t - timestep_seconds`. | `datetime \| str \| None` | `required` | `2018-03-18 01:00:00` |
| `noobs` | Observation mode: 0 uses surface/overwater/upper-air observations; 1 replaces upper-air observations with prognostic data; 2 uses prognostic data at all levels. | `int` | `0` | `2` |
| `run_type` | Run type: 0 computes wind fields only; 1 also computes micrometeorology and is required by CALPUFF/CALGRID. | `int` | `1` | `1` |
| `save_output` | Write the gridded meteorological disk output selected by legacy_output_format. | `bool` | `True` | `True` |
| `legacy_output_format` | Traditional disk-output type: 1 writes CALPUFF/CALGRID CALMET.DAT; 2 writes MESOPUFF-II PACOUT.DAT. | `int` | `1` | `1` |
| `test_mode` | 1 stops after input setup checks; 2 continues into model computation. | `int` | `2` | `2` |
| `regulatory_mode` | Regulatory check mode: 0 disables it; 1 enforces the USEPA-required mixing-height, overwater-flux, relaxation, and representative-temperature choices. | `int` | `0` | `0` |
| `timestep_seconds` | CALMET computational time step; it must divide evenly into one hour. | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | Seconds between prognostic records: a positive multiple of timestep_seconds and no more than 12 hours. Traditional PROG.DAT and native MM4.DAT inputs remain hourly by file format. | `int` | `3600` | `3600` |
| `cloud_method` | Cloud source: 1 surface observations (NOOBS 0/1), 2 existing CLOUD.DAT, 3 prognostic 850-hPa RH, or 4 prognostic RH at all levels. | `int` | `4` | `4` |
| `cloud_condensate_threshold_g_kg` | Minimum layer-total non-precipitating cloud condensate (`QCLOUD + QICE`), in g kg⁻¹, used to retain prognostic cloud cover and diagnose its condensate-based ceiling. | `float` | `0.005` | `0.005` |
| `cloud_data` | CLOUD.DAT path: input when cloud_method=2, or output destination when cloud_output=1. | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | CLOUD.DAT format: unformatted binary (1) or formatted text (2). | `int` | `2` | `2` |
| `cloud_output` | Write the internally generated gridded cloud field as CLOUD.DAT (0=no, 1=yes); output is unavailable when cloud_method=2 reads an existing file. | `int` | `0` | `0` |
| `surface_data` | Hourly SURF.DAT observations for NOOBS 0/1. DIAG.DAT may replace their winds; full meteorological runs still read SURF.DAT thermodynamic fields. | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | SURF.DAT format: unformatted binary (1) or formatted text (2). | `int` | `2` | `2` |
| `surface_control_stations` | Records `(4-char name, station ID, x_km, y_km, timezone, anemometer_height_m)` in model coordinates; list land stations only when land/water interpolation is separate. | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | One sounding file per upper-air station for NOOBS=0. DIAG.DAT may replace sounding winds; full meteorological runs still read sounding temperatures. | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | Records `(4-char name, station ID, x_km, y_km, timezone)`, one per upper-air file. | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | One file per overwater station; may contain water ΔT, lapse rate, observed mixing height, and observed waves for negative COARE wave codes. | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | Hourly preprocessed DIAG.DAT temperatures, lapse rates, or U/V winds. A file is required only when the selected observation mode actually consumes an enabled diagnostic source. | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | Surface-temperature source for the diagnostic wind module: compute internally (0) or read hourly preprocessed values from DIAG.DAT (1). | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | Domain lapse-rate source for terrain circulations: compute internally from upper-air/prognostic data (0) or read hourly DIAG.DAT values (1). | `int` | `0` | `0` |
| `diagnostic_initial_wind` | Diagnostic-module initial-guess wind source: compute from upper/prognostic data (0) or read hourly preprocessed U/V from DIAG.DAT (1). | `int` | `0` | `0` |
| `diagnostic_surface_wind` | Surface-wind source: SURF.DAT speed/direction (0) or preprocessed DIAG.DAT U/V (1). In full runs, option 1 replaces only wind and retains SURF.DAT thermodynamic fields. | `int` | `0` | `0` |
| `diagnostic_upper_wind` | Upper-wind source: sounding speed/direction (0) or preprocessed DIAG.DAT U/V (1). In full runs, option 1 replaces only wind and retains sounding temperatures. | `int` | `0` | `0` |
| `initial_guess_upper_station` | With internally computed initial winds, 1..NUSTA selects one sounding for a uniform guess; -1 uses all soundings with 1/R² weighting for a spatially varying guess. | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | Bottom and top heights, in metres, used to form the diagnostic module's domain-uniform initial wind. | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | Use coarse CALMET fields as the initial wind guess (0=no, 1=yes); when enabled, it replaces the background wind for every IPROG mode. | `int` | `0` | `0` |
| `initial_guess_calmet_files` | Coarse-grid CALMET files used as the initial wind guess when use_initial_guess_calmet=1. | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | WT.DAT three-dimensional observation weight Wo; the prognostic field receives complementary weight 1−Wo. | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | Precipitation source: positive is the number of PRECIP.DAT stations, 0 disables precipitation input, and -1 uses prognostic precipitation. | `int` | `0` | `-1` |
| `precipitation_data` | PRECIP.DAT hourly station precipitation input, used when precipitation_station_count is positive. | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | PRECIP.DAT format: unformatted binary (1) or formatted text (2). | `int` | `2` | `2` |
| `precipitation_control_stations` | Records `(4-char name, six-digit station code, x_km, y_km)`; the traditional code combines a two-digit state and four-digit station ID. | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | Prognostic-wind use: 0 off; 1/3/13 Step-1 field (IWFCOD=0); 2/4/14 initial guess (IWFCOD=1); 5/15 pseudo-observations. Code families select CSUMM, MM4, or MM5/3D. | `int` | `0` | `14` |
| `irhprog` | Relative-humidity source for NOOBS 0/1: 0 observed SURF/SEA data, 1 prognostic RH. NOOBS=2 always uses prognostic RH without modifying this setting. | `int` | `0` | `1` |
| `itprog` | Temperature source: 0 surface and upper observations (NOOBS=0); 1 surface observations plus prognostic aloft; 2 prognostic surface and aloft. | `int` | `0` | `2` |
| `imixh` | Mixing-height method: 1 Maul–Carson land/water; -1 Maul–Carson land plus OCD water; 2 Batchvarova–Gryning land/water; -2 BG land plus OCD water. | `int` | `1` | `1` |
| `constb` | Dimensionless B in the Venkatram neutral mechanical mixing-height relation, scaling friction velocity by Coriolis and free-atmosphere stability. | `float` | `1.41` | `1.41` |
| `conste` | Dimensionless entrainment constant E in the modified Maul–Carson convective-growth equation; used when `abs(imixh)==1`. | `float` | `0.15` | `0.15` |
| `constn` | Coefficient in the stable mechanical mixing-height relation `h = CONSTN × u*^(3/2)`, evaluated with a second stability limit. | `float` | `2400.0` | `2400.0` |
| `constw` | Dimensionless overwater mechanical constant in `h = CONSTW × u* / abs(f)`; an observed SEA.DAT mixing height overrides it. | `float` | `0.16` | `0.16` |
| `coriolis` | Absolute Coriolis parameter `abs(f)` in s⁻¹ for mechanical mixing-height equations; 999 derives it from latitude. | `float` | `999.0` | `0.0001` |
| `minimum_lapse_rate` | Floor on the stable-layer potential-temperature gradient above the previous convective mixing height, in K m⁻¹. | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | Depth in metres above the previous-time-step convective mixing height used to calculate the stable potential-temperature gradient. | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | Minimum overland mixing height, in metres. | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | Maximum overland mixing height, in metres. | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | Minimum overwater mixing height, in metres; unused where SEA.DAT supplies an observed height. | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | Maximum overwater mixing height, in metres; unused where SEA.DAT supplies an observed height. | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | Land heat-flux-per-boundary-layer-depth threshold, in W m⁻³; convective growth requires QH greater than threshold × height. | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | Overwater heat-flux-per-boundary-layer-depth threshold, in W m⁻³; convective growth requires QH greater than threshold × height. | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | Overwater inputs: 0 SEA.DAT lapse rate and ΔT; 1 prognostic lapse rate plus SEA.DAT ΔT (IPROG>2); 2 prognostic lapse rate and ΔT (IPROG>12, 3D.DAT v2+). | `int` | `0` | `0` |
| `prognostic_ocean_category` | Land-use category representing ocean in the prognostic meteorological input. | `int` | `16` | `16` |
| `overwater_flux_method` | Overwater flux: 0 legacy OCD; 10 COARE Charnock/no waves; ±11 Oost; ±12 Taylor-Yelland. Negative codes read observed waves; positive 11/12 use default waves. | `int` | `10` | `10` |
| `coastal_shelf_length_km` | COARE nearshore/shallow-water roughness length scale in kilometres; zero disables this modification. | `float` | `0.0` | `0.0` |
| `warm_layer` | COARE warm-layer correction (0=off, 1=on); keep it off for infrared-radiometer SST. | `int` | `0` | `0` |
| `cool_skin` | COARE cool-skin correction (0=off, 1=on); keep it off for infrared-radiometer SST. | `int` | `0` | `0` |
| `convective_relaxation` | Relax convective mixing height toward equilibrium in the weak-positive-flux regime `0 < QH <= threshold × height` (0=no, 1=yes). | `int` | `1` | `1` |
| `convective_relaxation_time_s` | E-folding relaxation time in seconds when convective_relaxation is enabled; must be at least 1 s. | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | Precipitation weighting: 1 uses 1/R, 2 uses 1/R², and 3 uses a Thiessen-like exp(-R²/σ²)/R² treatment for discontinuous fields. | `int` | `2` | `2` |
| `precipitation_radius_km` | For precipitation methods 1/2, station radius of influence (km); for method 3, exponential σ (km). Method 3 with zero derives half the minimum wet-to-dry-station distance. | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | Rates below this threshold are set to zero, in mm h⁻¹. | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | HA1 in `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`, multiplying solar elevation; units W m⁻². | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | HA2 clear-sky intercept in `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`; units W m⁻². | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | Dimensionless HB1 cloud-attenuation amplitude in the shortwave factor `(1+HB1·N^HB2)`. | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | Dimensionless HB2 exponent applied to cloud fraction N in the shortwave parameterization. | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | HC1 coefficient multiplying air-temperature T⁶ in the net-radiation equation; units W m⁻² K⁻⁶. | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | HC2 coefficient multiplying cloud fraction N in the net-radiation equation; units W m⁻². | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | Dimensionless HC3 denominator correction in the net-radiation equation `(…)/(1+HC3)`. | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | When diagnostic_surface_temperature=0: positive selects a surface station, -1 a spatially varying field, and -2 the domain-mean prognostic temperature. | `int` | `-1` | `-1` |
| `upper_temperature_option` | When diagnostic_lapse_rate=0: positive selects an upper-air station, -1 a spatially varying lapse rate, and -2 the domain-mean prognostic lapse rate. | `int` | `-1` | `-1` |
| `lapse_height_m` | Depth in metres used to calculate the domain-scale lapse rate for diagnostic terrain-induced circulations. | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | Observed surface temperature/RH weighting: 1 uses 1/R and 2 uses 1/R²; upper-air interpolation remains 1/R². | `int` | `1` | `1` |
| `temperature_radius_km` | Radius of influence, in kilometres, for observed surface temperature and relative-humidity interpolation. | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | Maximum number of surface stations used in observed temperature and relative-humidity interpolation. | `int` | `5` | `5` |
| `water_temperature_gradient_below` | Fallback water temperature gradient below mixing height, in K m⁻¹, when SEA.DAT does not provide one. | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | Fallback water temperature gradient above mixing height, in K m⁻¹, when SEA.DAT does not provide one. | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | First category in the inclusive range receiving separate overwater temperature/RH interpolation; set above all categories with max to disable. | `int` | `999` | `55` |
| `water_temperature_category_max` | Last category in the inclusive range receiving separate overwater temperature/RH interpolation; set above all categories with min to disable. | `int` | `999` | `55` |
| `iavet` | Enable spatial averaging of interpolated temperatures using the mixing-height averaging geometry. | `int` | `1` | `1` |
| `iavezi` | Enable upwind spatial averaging of effective and convective mixing heights (0=no, 1=yes). | `int` | `1` | `1` |
| `mixing_wind_level` | One-based wind layer (1..NZ) whose direction defines the upwind averaging cone. | `int` | `1` | `1` |
| `mnmdav` | Half-width, in grid cells, of the square supplementary search box used with the upwind cone for mixing-height and temperature averaging. | `int` | `1` | `10` |
| `iwfcod` | Wind model: 0 objective analysis only; 1 full diagnostic module with terrain adjustments before objective analysis. | `int` | `1` | `1` |
| `wind_extrapolation` | Vertical surface-wind extrapolation: `abs(code)` 1 none, 2 power law, 3 user factors, 4 similarity theory; negative codes also ignore layer-1 upper-air data. | `int` | `-4` | `-1` |
| `upper_air_separation_cells` | Surface-to-upper-air proximity threshold in grid-cell spacings; a surface wind inside it is not extrapolated vertically, and -1 disables the exclusion. | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | Extrapolate a surface observation even when it reports calm wind. | `int` | `0` | `0` |
| `wind_bias` | Per-layer station-weight bias from -1 to 1: negative reduces upper-air weight, positive reduces surface-station weight, and zero leaves 1/R² weights unchanged; None derives zero for every layer. | `Sequence[float] \| None` | `None → (0.0,) * NZ` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | When no valid unblocked station lies inside the relevant maximum radius, use the closest valid station outside it. | `bool` | `False` | `True` |
| `minimum_station_radius_km` | Small lower bound, in kilometres, on wind-interpolation distance; it prevents division by zero in 1/R² weighting. | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | Maximum land surface-layer station radius, in kilometres; choose it to reflect terrain-limited influence. | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | Maximum station radius aloft over land, in kilometres; it is generally larger than the surface radius as terrain influence decreases with height. | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | Maximum overwater station radius, in kilometres, used at every vertical layer over water. | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | Distance where surface observations and the first-guess field receive equal weight. | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | Distance where upper-air observations and the first-guess field receive equal weight. | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | Prognostic Step-1 wind weighting distance, in kilometres; it must be nonzero for IPROG 1/3/13. | `float` | `0.0` | `0.0` |
| `stations_per_level` | Per-layer maximum number of closest stations used at a grid point; None derives the Fortran default of 99 for every layer. | `Sequence[int] \| None` | `None → (99,) * NZ` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | One multiplicative factor per layer, used only when `abs(wind_extrapolation)==3`; None derives zero for every layer, and the required layer-1 value is unused. | `Sequence[float] \| None` | `None → (0.0,) * NZ` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | Number of line barriers that block wind interpolation. | `int` | `0` | `0` |
| `barrier_top_level` | One-based highest layer (1..NZ) to which station-interpolation barriers apply; None selects NZ, and it is used only when barrier_count>0. | `int \| None` | `None → NZ` | `10` |
| `barriers` | Station-wind interpolation barriers as `(x_start, y_start, x_end, y_end)` line segments in kilometres. | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | Region records containing grid-line bounds, coastline endpoints in kilometres, and assigned station IDs; winds are recomputed from only those stations using coast-relative 1/R² weighting. | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | Compute Froude-number terrain-flow adjustment (0=no, 1=yes); active only with iwfcod=1. | `int` | `1` | `1` |
| `ikine` | Compute kinematic terrain effects (0=no, 1=yes); active only with iwfcod=1. | `int` | `0` | `0` |
| `iobr` | Apply the O'Brien procedure that forces vertical velocity to zero at the model top, then divergence-minimizes U/V for mass consistency (0=no, 1=yes). | `int` | `0` | `0` |
| `islope` | Compute slope-flow parameterization (0=no, 1=yes). | `int` | `1` | `1` |
| `critical_froude` | With Froude adjustment enabled, uphill flow below this critical value is rotated tangent to terrain without changing speed; flow at or above it passes over. | `float` | `1.0` | `1.0` |
| `smoothing_passes` | Wind-field smoothing passes by level; None derives the Fortran defaults 2 for layer 1 and 4 for every higher layer. | `Sequence[int] \| None` | `None → (2,) + (4,) * (NZ - 1)` | `[2, 4, … (10 items)]` |
| `divergence_limit` | Convergence target for wind-field divergence minimization. | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | Maximum iterations allowed for divergence minimization. | `int` | `50` | `50` |
| `kinematic_alpha` | Empirical multiplier applied to the vertical gradient of terrain-induced vertical velocity in the kinematic adjustment. | `float` | `0.1` | `0.1` |
| `z_faces_m` | Vertical cell-face heights in metres; supply NZ+1 values, and layer k is centred at `(face[k] + face[k+1]) / 2`. | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | Base time-zone offset in CALMET's hours-west-of-UTC convention (for example 5=EST; negative values are east). | `int` | `0` | `0` |
| `base_timezone_label` | Modern `UTC±HHMM` label for the same offset represented by base_timezone. | `str \| None` | `None` | `'UTC+0000'` |
| `terrain_radius_km` | Terrain-search radius, in kilometres, used to determine nearby extrema and effective obstacle height for the Froude adjustment. | `float` | `15.0` | `15.0` |
| `half_average_angle` | Half-opening angle of the upwind spatial-averaging cone, in degrees; the full opening is twice this value. | `float` | `30.0` | `30.0` |
| `include_w_temperature` | Store 3-D vertical-wind and temperature fields required by CALGRID and CALPUFF's subgrid complex-terrain option. | `bool` | `True` | `False` |
| `compression` | NetCDF deflate compression level for Python CALMET output (0 disables compression). | `int` | `1` | `1` |
| `control_lines` | Original CALMET control-file lines copied into legacy output metadata. | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | PACOUT.DAT destination used when save_output is true and legacy_output_format=2. | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | DCST.GRD destination for the distance-to-land/coast diagnostic grid. | `str \| Path \| None` | `None` | `None` |
| `listing_output` | CALMET listing/text report destination, including selected printed fields and debug information. | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | Write selected gridded meteorological fields to listing_output every print_interval_hours. | `bool` | `False` | `False` |
| `print_interval_hours` | Interval between printed meteorological fields, in hours. | `int` | `1` | `1` |
| `print_uv_levels` | Per-layer switches selecting U/V wind output when print_fields is enabled. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | NZ 0/1 switches for W at top faces 2..NZ+1 (ground-face W is zero); requires print_fields and include_w_temperature. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | NZ 0/1 switches for three-dimensional temperature layers; requires print_fields and include_w_temperature. | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | Eight 0/1 print switches in this order: PGT stability, friction velocity, Monin-Obukhov length, mixing height, convective velocity scale, precipitation, sensible heat flux, convective mixing height. | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | Print micrometeorological inputs and internal variables; produces large output. | `bool` | `False` | `False` |
| `debug_first_step` | First time step included in micrometeorology debug output. | `int` | `1` | `1` |
| `debug_last_step` | Last time step included in micrometeorology debug output. | `int` | `1` | `1` |
| `wind_debug_output` | Master switch for writing TEST.PRT/OUT/KIN/FRD/SLP wind diagnostic files (0=off, 1=on). | `int` | `0` | `0` |
| `wind_debug_levels` | Number of wind-field levels written to diagnostic files, starting at the surface. | `int` | `1` | `0` |
| `wind_debug_fields` | Nine 0/1 switches (IPR0–IPR8): interpolated, terrain-adjusted, smoothed/initial-divergence, final speed/direction, final divergence, post-kinematic, post-Froude, post-slope, and final-component fields. | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | TEST.OUT destination for final wind diagnostic fields. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | TEST.KIN destination for wind fields after kinematic adjustment. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | TEST.FRD destination for wind fields after Froude-number adjustment. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | TEST.SLP destination for wind fields after slope-flow adjustment. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | TEST.PRT destination for general wind-module diagnostic variables. | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | Complete grid object; an alternative to the flattened grid parameters. | `GridConfig \| Mapping[str, Any] \| None` | `None` | `see flattened grid fields` |
| `nx` | Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings. | `int \| None` | `required` | `610` |
| `ny` | Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of rays. | `int \| None` | `required` | `499` |
| `dx_km` | Cartesian X grid spacing in kilometres; independent of dy_km and unused for polar ring positions. | `float \| None` | `required` | `1.0` |
| `dy_km` | Cartesian Y grid spacing in kilometres; independent of dx_km and unused for polar ray positions. | `float \| None` | `required` | `1.0` |
| `projection` | Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert azimuthal equal-area (LAZA). | `str` | `'UTM'` | `'LCC'` |
| `lat0` | Signed latitude of the projection origin or tangent point, in degrees (north positive); EM fixes it at zero. | `float` | `27.0` | `27.0` |
| `lon0` | Signed central-meridian or tangent-point longitude, in degrees (east positive). | `float` | `117.5` | `117.5` |
| `lat1` | First standard latitude for LCC, or true-scale latitude for PS. | `float` | `30.0` | `30.0` |
| `lat2` | Second standard latitude for LCC; unused by PS. | `float` | `60.0` | `60.0` |
| `x_origin_km` | Easting of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `y_origin_km` | Northing of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres. | `float` | `0.0` | `0.0` |
| `false_easting_km` | False easting at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `false_northing_km` | False northing at the projection origin; used by TTM, LCC, and LAZA. | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM zone number from 1 through 60; used only for UTM. | `int` | `50` | `50` |
| `hemisphere` | UTM hemisphere, N or S; used only for UTM. | `str` | `'N'` | `'N'` |
| `datum` | Output-grid datum/region code; individual input sources specify their own input datum separately. | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2 Cartesian origin at its centre; 3 polar origin at ring centre. | `int` | `1` | `1` |
| `ring_distances_km` | TERREL polar-grid ring radii, in kilometres; supply one value per ring when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL polar-grid ray azimuths, in degrees clockwise from north; used when grid_type=3. | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL-only polar method: 1 NORMAL sectors lie halfway between adjacent rings/rays; 2 SCREEN sectors span the current to next outer ring and halfway between rays. | `int` | `2` | `2` |
| `ttm_scale` | Scale factor on the central meridian for TTM projection. | `float` | `1.0` | `1.0` |

</details>
<!-- parameter-reference:end -->

The public processing API is:

```python
pycalmet.run_pipeline(...)
pycalmet.PipelineConfig(...)
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc is a compatibility alias
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

The default API uses regular NumPy calculations, `pyproj`, `datetime`, and
`netCDF4` for every NetCDF read and write. CALWRF passes its NetCDF output
directly to CALMET. Workflows that specifically require
the former single-precision operation order or M3D staging can opt in to the
frozen compatibility implementation:

```python
pycalmet.legacy.calwrf(...)
pycalmet.legacy.calmet(...)
```

## Modern thermodynamics and cloud diagnostics

The default API uses the following treatments for WRF-driven CALMET
calculations:

- **Relative humidity.** When relative humidity is diagnosed from WRF
  `QVAPOR`, the dry-air mixing ratio `r` is first converted to water-vapour
  partial pressure with `e = p r / (ε + r)`, where `ε = 0.622`. Relative
  humidity is then `RH = 100 e / e_s(T)`.
- **Cloud condensate.** When `QCLOUD` or `QICE` is present, prognostic cloud
  cover derived from relative humidity is retained only where the layer-total
  non-precipitating condensate `QCLOUD + QICE` reaches
  `cloud_condensate_threshold_g_kg`. The default is `0.005 g kg⁻¹`
  (`5 × 10⁻⁶ kg kg⁻¹`); the lowest qualifying source layer supplies the
  condensate-based cloud ceiling. Use
  `calmet_cloud_condensate_threshold_g_kg` for the same control in
  `run_pipeline`. If neither condensate field exists, the RH-derived cloud
  field remains available without this mask.
- **TEMP3D layer overlap.** With `ITPROG=2`, a CALMET layer crossed by the
  convective mixing height is blended by its geometric thickness. For layer
  `k`, `f_k = clip((z_i - z_{k-1/2}) / (z_{k+1/2} - z_{k-1/2}), 0, 1)` and
  `T_k = f_k T_mix,k + (1 - f_k) T_prog,k`. Layers wholly below or above the
  mixing height retain the mixed or prognostic profile, respectively, while
  the intersected layer changes continuously as the mixing height moves.

These treatments belong to the default API. `pycalmet.legacy` retains the
frozen historical formulas and level-centre TEMP3D switch for reproducibility.

## Zhejiang example

The bundled example contains three consecutive WRF records, 42 original SRTM3
`.hgt` tiles, and the original `EAUSGS1.IMG` land-use raster. It rebuilds
terrain, land use, and the geophysical grid instead of using a prepared
`geo.nc`.
Run the complete five-stage pipeline from the repository root:

```bash
git lfs pull
sha256sum -c examples/zhejiang/SHA256SUMS
cd data/terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../../..
cd data/landuse/usgs-glcc && sha256sum -c SHA256SUMS && cd ../../..

./examples/zhejiang/run.sh
```

See [examples/zhejiang/README.md](examples/zhejiang/README.md) for provenance,
input boundaries, source data, and generated products.

## Repository layout

```text
src/pycalmet/          Python package and opt-in legacy namespace
tests/                 Unit and integration tests
examples/zhejiang/     Runnable real-data example and control files
data/terrain/srtm3/    Original SRTM3 tiles used by the Zhejiang domain
data/landuse/usgs-glcc/ Original USGS Global/GLCC-style land-use raster
fortran/               Reference Fortran sources; not used at runtime
tools/                 Maintainer utilities
```

The `fortran/` tree contains source material only. Legacy NCL programs,
regional C-shell runners, build wrappers, generated executables, and runtime
products are intentionally excluded from the maintained library.

## Data and output conventions

Canonical outputs are NetCDF files created and read with `netCDF4.Dataset`.
The package does not depend on xarray. Legacy formatted and unformatted
CALMET-family files remain available when an output path or compatibility
option explicitly requests them. The bundled raw data is grouped under
`data/terrain/` and `data/landuse/`; large binary sources such as
`EAUSGS1.IMG` are tracked through Git LFS. `EAUSGS2.IMG` is not used by the
Zhejiang configuration and is not included. User-supplied large inputs should
normally stay outside the source tree, while generated files belong under
`output/`, which is ignored by Git.

Physical fields and model-state arrays use 32-bit floating point by default.
Time coordinates, integer categories and indices, projection coordinates, and
selected convergence or large-reduction scalars retain the precision required
for correct temporal, spatial, and numerical behavior.

Standard SRTM HGT and GeoTIFF inputs are decoded by Rasterio, datum operations
are delegated to PROJ through `pyproj`, coastline clipping uses Shapely, and
nearest-neighbour searches use SciPy `cKDTree`. CALMET-specific parameterizations
and uncommon historical input formats remain implemented in the package.

The Zhejiang example uses an LCC grid with a 1 km spacing and a `610 × 499`
domain. Its WRF records cover 2018-03-18 00:00–02:00 UTC. The configured
00:00–01:00 CALMET period uses the first two records; the 02:00 record is
available when extending the run through the next interval.

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
