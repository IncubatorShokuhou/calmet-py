# PyCALMET

[English](README.md)

PyCALMET 是 CALMET 气象预处理与诊断风场流程的 NumPy 实现。它以统一的 Python
API 提供 CALWRF、TERREL、CTGPROC、MAKEGEO 和 CALMET 五个阶段，推荐使用
NetCDF 作为中间格式，同时保留对 CALPUFF 传统文件的读写兼容能力。

本项目面向可复现实验、自动化气象数据处理和既有 CALMET 工作流现代化。仓库将
Python 库、Fortran 参考源码、浙江真实数据示例和 SRTM3 地形数据分别管理，运行
Python 主流程无需编译 Fortran。

## 主要特性

- 使用 Python、NumPy 和 SciPy 执行，无需 Fortran 编译器。
- 一个软件包覆盖五个公开处理阶段。
- 支持 WRF NetCDF 输入和 CF 兼容的 NetCDF 输出。
- 兼容传统 CALWRF、TERREL、CTGPROC、MAKEGEO 和 CALMET 文件。
- WRF 数据按时次流式处理，可处理超过内存容量的数据集。
- 提供经过无损压缩的浙江真实 WRF 示例。
- 提供独立的 legacy API，供确实需要复现旧 Fortran 计算路径的工作流显式调用。

## 安装

PyCALMET 需要 Python 3.10 或更高版本。

```bash
python -m pip install .
```

开发和测试环境：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## 快速开始

每个阶段既可以直接传入关键字参数，也可以读取对应的 CALMET 系列传统控制文件。
这里的 `.inp` 是各程序自己的控制文件，并不是标准 Fortran `NAMELIST`。浙江示例
已经为五个阶段分别定义了控制文件：

| 阶段 | 示例控制文件 |
| --- | --- |
| CALWRF | `examples/zhejiang/config/calwrf.inp` |
| TERREL | `examples/zhejiang/config/terrel.inp` |
| CTGPROC | `examples/zhejiang/config/ctgproc.inp` |
| MAKEGEO | `examples/zhejiang/config/makegeo.inp` |
| CALMET | `examples/zhejiang/config/calmet.inp` |

控制文件内的相对路径以该控制文件所在目录为基准。仓库中的示例已由这五份文件
完整定义，因此 Python 调用不再重复传参。只有明确要让同一控制文件使用其他数据
时，才需要关键字参数覆盖。阶段函数会返回主输出文件路径。

```python
import pycalmet

base = "examples/zhejiang/config"
pycalmet.calwrf(f"{base}/calwrf.inp")
pycalmet.terrel(f"{base}/terrel.inp")
pycalmet.ctrproc(f"{base}/ctgproc.inp")
pycalmet.makegeo(f"{base}/makegeo.inp")
result = pycalmet.calmet(f"{base}/calmet.inp")
```

每个阶段同时提供完整的类型化关键字 API。让 `inp` 保持默认值 `None`，即可直接
传入控制参数；`nx`、`dx_km`、`projection` 等网格项也是普通函数参数。日期接受
`datetime`，路径接受 `str` 或 `Path`，重复项接受 Python 序列。调用方没有传入的
参数会使用原程序定义的默认值。例如：

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

函数签名会列出全部支持项，包括某份 `.inp` 没有写出、但 Fortran 程序定义了默认
值的参数；IDE 补全、类型检查器、`help()` 和 `inspect.signature()` 都能直接看到。

<!-- parameter-reference:start -->
## 完整参数参考

以下表格由公开函数签名和类型化配置模型生成。“默认值”表示 `inp=None` 模式下调用方
省略参数时采用的值；“浙江示例值”表示解析仓库内置控制文件后得到的实际值。`inp`
本身和 `**legacy_parameters` 是入口及兼容机制，不属于模型控制项，因此不在每张表中
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。API 变化后运行
`python tools/generate_parameter_reference.py` 重新生成；CI 可使用 `--check` 检查。

<details>
<summary><strong>CALWRF</strong> — 共 14 个参数；浙江控制文件：`examples/zhejiang/config/calwrf.inp`。</summary>

| 参数 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- |
| `wrf_files` | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | `str \| int \| datetime \| None` | `必填` | `'2018031800'` |
| `end` | `str \| int \| datetime \| None` | `必填` | `'2018031802'` |
| `output` | `str \| Path \| None` | `必填` | `output/zhejiang/calwrf.nc` |
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
<summary><strong>TERREL</strong> — 共 49 个参数；浙江控制文件：`examples/zhejiang/config/terrel.inp`。</summary>

| 参数 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- |
| `srtm_dir` | `str \| Path \| None` | `必填` | `examples/zhejiang/config` |
| `output` | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
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
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | `int \| None` | `必填` | `610` |
| `ny` | `int \| None` | `必填` | `499` |
| `dx_km` | `float \| None` | `必填` | `1.0` |
| `dy_km` | `float \| None` | `必填` | `1.0` |
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
<summary><strong>CTRPROC / CTGPROC</strong> — 共 43 个参数；浙江控制文件：`examples/zhejiang/config/ctgproc.inp`。</summary>

| 参数 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- |
| `landuse_files` | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
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
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | `int \| None` | `必填` | `610` |
| `ny` | `int \| None` | `必填` | `499` |
| `dx_km` | `float \| None` | `必填` | `1.0` |
| `dy_km` | `float \| None` | `必填` | `1.0` |
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
<summary><strong>MAKEGEO</strong> — 共 52 个参数；浙江控制文件：`examples/zhejiang/config/makegeo.inp`。</summary>

| 参数 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- |
| `terrain` | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `landuse` | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `output` | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
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
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | `int \| None` | `必填` | `610` |
| `ny` | `int \| None` | `必填` | `499` |
| `dx_km` | `float \| None` | `必填` | `1.0` |
| `dy_km` | `float \| None` | `必填` | `1.0` |
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
<summary><strong>CALMET</strong> — 共 165 个参数；浙江控制文件：`examples/zhejiang/config/calmet.inp`。</summary>

| 参数 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- |
| `meteorology` | `str \| Path \| Sequence[str \| Path] \| None` | `必填` | `output/zhejiang/calwrf.nc` |
| `geophysical` | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `output` | `str \| Path \| None` | `必填` | `output/zhejiang/calmet.nc` |
| `start` | `str \| datetime \| None` | `必填` | `'2018-03-18_00:00:00'` |
| `end` | `str \| datetime \| None` | `必填` | `'2018-03-18_01:00:00'` |
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
| `grid` | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | `int \| None` | `必填` | `610` |
| `ny` | `int \| None` | `必填` | `499` |
| `dx_km` | `float \| None` | `必填` | `1.0` |
| `dy_km` | `float \| None` | `必填` | `1.0` |
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

公开 API：

```python
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc 是兼容别名
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

默认 API 使用常规 NumPy 计算、`pyproj`、`datetime`，并让 CALWRF 通过 NetCDF
直接向 CALMET 传递数据。只有确实需要旧单精度运算顺序或 M3D 中转时，才使用冻结
的兼容实现：

```python
pycalmet.legacy.calwrf(...)
pycalmet.legacy.calmet(...)
```

## 浙江示例

仓库内置三个连续 WRF 时次、42 个原始 SRTM3 `.hgt` 瓦片和分类土地利用点数据。
示例会重新生成地形、土地利用和地理场，不依赖预先做好的 `geo.nc`。从仓库根目录
运行全部五个阶段：

```bash
sha256sum -c examples/zhejiang/SHA256SUMS
cd terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../..

./examples/zhejiang/run.sh
```

数据来源、时间边界和生成结果见
[examples/zhejiang/README.md](examples/zhejiang/README.md)。

## 仓库结构

```text
src/pycalmet/          Python 软件包和可选 legacy 命名空间
tests/                 单元测试和集成测试
examples/zhejiang/     可运行的真实数据示例及控制文件
terrain/srtm3/         浙江区域使用的 SRTM3 瓦片
fortran/               仅供参考的 Fortran 源码，运行时不使用
tools/                 维护工具
```

`fortran/` 仅保留源码和必要的包含文件。旧 NCL 程序、区域 C-shell 运行脚本、构建
包装脚本、编译产物和运行产物不再属于维护范围。

## 数据和输出约定

规范输出为可由 `xarray.open_dataset` 打开的 NetCDF 文件。只有在显式指定传统
输出路径或兼容选项时，才会写出 CALMET 系列旧格式文件。大型用户输入应放在源码
树之外；生成结果统一写入 Git 忽略的 `output/`。

浙江示例采用 LCC 投影、1 km 网格距和 `610 × 499` 网格。示例 WRF 数据覆盖
2018-03-18 00:00–02:00 UTC；最后一个时次用于提供生成 00:00–01:00 CALMET
结果所需的下一小时边界。

## 验证

运行完整 Python 测试：

```bash
python -m pytest -q
```

默认测试检查数值行为和完整数据流，不绑定特定编译器或 C 数学库。单独的 legacy
测试用于确认兼容命名空间与默认实现隔离，并验证其旧单精度路径。`fortran/` 中的
源码仅用于审计，不是安装依赖。

## 参与贡献

运行时代码放在 `src/pycalmet`，行为修改应增加针对性测试，不要提交生成的模型
产物。修改参考 Fortran 时应保留固定格式布局。拉取请求应说明受影响阶段、输入
配置以及实际执行的验证命令。

## 致谢

PyCALMET 遵循 CALPUFF 系统的处理模型和文件约定。仓库附带的 WRF 子集与地形
数据用于保证数值验证可复现，其来源和校验值记录在对应数据目录中。
