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

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `wrf_files` | 按时间顺序排列的 WRF NetCDF 文件。 | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | 计算开始时刻。 | `str \| int \| datetime \| None` | `必填` | `'2018031800'` |
| `end` | 计算结束时刻。 | `str \| int \| datetime \| None` | `必填` | `'2018031802'` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/calwrf.nc` |
| `create_surface` | 是否另外生成传统地面气象文件。 | `bool` | `True` | `False` |
| `compression` | 压缩级别。 | `int` | `1` | `1` |
| `i_range` | WRF 东西向索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `j_range` | WRF 南北向索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `k_range` | WRF 垂直层索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `legacy_output` | 传统格式输出路径。 | `str \| Path \| None` | `None` | `None` |
| `surface_output` | 地面输出路径。 | `str \| Path \| None` | `None` | `None` |
| `title` | 标题。 | `str` | `'Create 3D.DAT file for WRF output'` | `'Create CALWRF data for the Zhejiang example'` |
| `listing_output` | 列表文件输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/calwrf.lst` |
| `legacy_file_rain_reset` | 是否在每个传统输入文件边界重置累计降水。 | `bool` | `False` | `False` |

</details>

<details>
<summary><strong>TERREL</strong> — 共 49 个参数；浙江控制文件：`examples/zhejiang/config/terrel.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `srtm_dir` | 原始 SRTM 高程瓦片所在目录。 | `str \| Path \| None` | `必填` | `examples/zhejiang/config` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `datum` | 大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | 地形采样方法：直接采样或双线性/传统采样。 | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | 地形文件路径。 | `Sequence[str \| Path]` | `()` | `[terrain/srtm3/N26E117.hgt, terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | 带有格式和源基准信息的地形文件列表。 | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=terrain/…datum='WGS-96'), … (42 items))` |
| `void_fill` | 是否填补高程输入中的空洞格点。 | `bool` | `False` | `False` |
| `cell_radius_km` | 格点半径（单位：km）。 | `float` | `0.0` | `5.0` |
| `model` | 模型。 | `int` | `1` | `1` |
| `processing_method` | 处理方法。 | `int \| None` | `None` | `2` |
| `receptor_input` | 接收点输入路径。 | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | 接收点输出路径。 | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | 离散接收点输入文件的列数。 | `int` | `2` | `2` |
| `interpolate_receptors` | 是否计算离散接收点处的地形高度。 | `bool` | `False` | `True` |
| `receptor_radius_km` | 接收点半径（单位：km）。 | `float` | `0.0` | `0.15` |
| `previous_save` | 先前保存路径。 | `str \| Path \| None` | `None` | `None` |
| `save_output` | 保存输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | 岸线文件路径。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | GSHHS文件路径。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 岸线来源大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | 岸线WDBII大地基准。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | 噪声替换值。 | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | 噪声最小（单位：m）。 | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | 缺失值替换值。 | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | 默认高程（单位：m）。 | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | PLOT输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | 原始回显输出路径。 | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | QAHIT阈值（单位：%）。 | `int` | `75` | `75` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | NX。 | `int \| None` | `必填` | `610` |
| `ny` | NY。 | `int \| None` | `必填` | `499` |
| `dx_km` | DX（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | DY（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | LAT0。 | `float` | `27.0` | `27.0` |
| `lon0` | LON0。 | `float` | `117.5` | `117.5` |
| `lat1` | LAT1。 | `float` | `30.0` | `30.0` |
| `lat2` | LAT2。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | X原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 假东距（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 假北距（单位：km）。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM分区。 | `int` | `50` | `50` |
| `hemisphere` | 半球。 | `str` | `'N'` | `'N'` |
| `grid_type` | 网格类型。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 极坐标处理。 | `int` | `1` | `2` |
| `ttm_scale` | TTM比例。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 共 43 个参数；浙江控制文件：`examples/zhejiang/config/ctgproc.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `landuse_files` | 土地利用文件路径。 | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `category_map` | 输入土地利用类别到输出类别的映射。 | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | 输入类型。 | `str` | `'GLAZAS'` | `'GEN'` |
| `mesh_density` | MESH密度。 | `int` | `12` | `12` |
| `input_datum` | 输入大地基准。 | `str` | `'ESR-S'` | `'WGS-84'` |
| `output_format` | 输出格式。 | `str` | `'auto'` | `'auto'` |
| `coastline_file` | 岸线文件路径。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | GSHHS文件路径。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 岸线来源大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | 岸线WDBII大地基准。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | 是否启用岸线交换。 | `bool` | `False` | `False` |
| `fill_empty_ocean` | 是否启用填补空海洋。 | `bool` | `False` | `False` |
| `ocean_category` | 海洋类别。 | `int` | `55` | `55` |
| `previous_landuse` | 先前土地利用路径。 | `str \| Path \| None` | `None` | `None` |
| `final_output` | 是否启用最终输出。 | `bool` | `True` | `True` |
| `landuse_sources` | 带有格式、基准和网格采样密度的土地利用文件列表。 | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=examples/zhejiang/landuse/zhejiang_generic.landuse, input_type='GEN', input_datum='WGS-84', mesh_density=12))` |
| `snow_processing` | 是否启用积雪处理。 | `bool` | `False` | `False` |
| `qa_cell_output` | 是否启用QA格点输出。 | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | QAHIT阈值（单位：%）。 | `int` | `75` | `75` |
| `qa_output_directory` | QA输出目录路径。 | `str \| Path \| None` | `None` | `None` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | NX。 | `int \| None` | `必填` | `610` |
| `ny` | NY。 | `int \| None` | `必填` | `499` |
| `dx_km` | DX（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | DY（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | LAT0。 | `float` | `27.0` | `27.0` |
| `lon0` | LON0。 | `float` | `117.5` | `117.5` |
| `lat1` | LAT1。 | `float` | `30.0` | `30.0` |
| `lat2` | LAT2。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | X原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 假东距（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 假北距（单位：km）。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM分区。 | `int` | `50` | `50` |
| `hemisphere` | 半球。 | `str` | `'N'` | `'N'` |
| `datum` | 大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格类型。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 极坐标处理。 | `int` | `1` | `1` |
| `ttm_scale` | TTM比例。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 共 52 个参数；浙江控制文件：`examples/zhejiang/config/makegeo.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `terrain` | 地形路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `landuse` | 土地利用路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `water_threshold` | 水体阈值。 | `float` | `0.5` | `0.5` |
| `missing_category` | 缺失值类别。 | `int` | `55` | `55` |
| `minimum_coverage` | 最小覆盖率。 | `float` | `0.96` | `0.96` |
| `secondary_landuse` | 第二套土地利用路径。 | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | 是否启用使用地形。 | `bool` | `True` | `True` |
| `input_properties` | 每个输入土地利用类别对应的地表属性元组。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | 水体类别。 | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | 输出类别。 | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | 输出水体MIN。 | `int` | `50` | `50` |
| `output_water_max` | 输出水体MAX。 | `int` | `55` | `55` |
| `category_splits` | 输入类别、输出类别及拆分比例记录。 | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | QA格点。 | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | 是否启用积雪处理。 | `bool` | `False` | `False` |
| `snow_data` | 积雪数据路径。 | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | 积雪开始日期。 | `int \| None` | `None` | `None` |
| `snow_end_date` | 积雪END日期。 | `int \| None` | `None` | `None` |
| `snow_properties` | 每个类别在积雪条件下对应的地表属性元组。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | 各类别的雪龄分级和衰减因子。 | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | 积雪粗糙度方法。 | `int` | `1` | `1` |
| `snow_height_scale_m` | 积雪高度比例（单位：m）。 | `float` | `1.0` | `10.0` |
| `snow_albedo_method` | 积雪反照率方法。 | `int` | `1` | `1` |
| `snow_depth_minimum_m` | 积雪深度最小（单位：m）。 | `float` | `0.0` | `0.01` |
| `landuse_plot_output` | 土地利用PLOT输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | 地形PLOT输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | 颜色映射输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | 图像模式。 | `int` | `0` | `0` |
| `title` | 标题。 | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | NX。 | `int \| None` | `必填` | `610` |
| `ny` | NY。 | `int \| None` | `必填` | `499` |
| `dx_km` | DX（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | DY（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | LAT0。 | `float` | `27.0` | `27.0` |
| `lon0` | LON0。 | `float` | `117.5` | `117.5` |
| `lat1` | LAT1。 | `float` | `30.0` | `30.0` |
| `lat2` | LAT2。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | X原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 假东距（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 假北距（单位：km）。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM分区。 | `int` | `50` | `50` |
| `hemisphere` | 半球。 | `str` | `'N'` | `'N'` |
| `datum` | 大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格类型。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 极坐标处理。 | `int` | `1` | `1` |
| `ttm_scale` | TTM比例。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CALMET</strong> — 共 165 个参数；浙江控制文件：`examples/zhejiang/config/calmet.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `meteorology` | 气象路径。 | `str \| Path \| Sequence[str \| Path] \| None` | `必填` | `output/zhejiang/calwrf.nc` |
| `geophysical` | 地理物理路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/calmet.nc` |
| `start` | 计算开始时刻。 | `str \| datetime \| None` | `必填` | `'2018-03-18_00:00:00'` |
| `end` | 计算结束时刻。 | `str \| datetime \| None` | `必填` | `'2018-03-18_01:00:00'` |
| `noobs` | 观测资料模式：仅预报场，或结合观测进行诊断。 | `int` | `2` | `2` |
| `run_type` | CALMET 运行类型：仅风场，或风场加微气象场。 | `int` | `1` | `1` |
| `save_output` | 是否启用保存输出。 | `bool` | `True` | `True` |
| `legacy_output_format` | 传统 CALMET 输出格式选项。 | `int` | `1` | `1` |
| `test_mode` | 测试模式。 | `int` | `2` | `2` |
| `regulatory_mode` | 法规模式。 | `int` | `0` | `0` |
| `timestep_seconds` | 时间步长（单位：s）。 | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | 预报时间步长（单位：s）。 | `int` | `3600` | `3600` |
| `cloud_method` | 云量方法。 | `int` | `4` | `4` |
| `cloud_data` | 云量数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | 云量数据格式。 | `int` | `2` | `2` |
| `cloud_output` | 云场输出选项。 | `int` | `0` | `0` |
| `surface_data` | 地面数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | 地面数据格式。 | `int` | `2` | `2` |
| `surface_control_stations` | 地面站名称、编号、坐标、时区和风速仪高度记录。 | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | 高空探空文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | 高空站名称、编号、坐标和时区记录。 | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | 海面文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | 诊断数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | 诊断地面温度。 | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | 诊断场是否使用观测温度递减率。 | `int` | `0` | `0` |
| `diagnostic_initial_wind` | 诊断初始风场。 | `int` | `0` | `0` |
| `diagnostic_surface_wind` | 诊断地面风场。 | `int` | `0` | `0` |
| `diagnostic_upper_wind` | 诊断高空风场。 | `int` | `0` | `0` |
| `initial_guess_upper_station` | 初始猜测场高空站点。 | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | 初始猜测场高度范围（单位：m）。 | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | 是否使用已有 CALMET 风场作为初始猜测场。 | `int` | `0` | `0` |
| `initial_guess_calmet_files` | 初始猜测场CALMET文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | 权重数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | 降水站点数量。 | `int` | `-1` | `-1` |
| `precipitation_data` | 降水数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | 降水数据格式。 | `int` | `2` | `2` |
| `precipitation_control_stations` | 降水站名称、编号和坐标记录。 | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | 预报风场输入及插值方案控制。 | `int` | `14` | `14` |
| `irhprog` | 预报资料相对湿度来源选项。 | `int` | `1` | `1` |
| `itprog` | 预报资料温度来源选项。 | `int` | `2` | `2` |
| `imixh` | 混合层高度计算方法选项。 | `int` | `1` | `1` |
| `constb` | 中性机械混合层高度系数 B。 | `float` | `1.41` | `1.41` |
| `conste` | 对流混合层夹卷系数。 | `float` | `0.15` | `0.15` |
| `constn` | 稳定机械混合层高度系数 N。 | `float` | `2400.0` | `2400.0` |
| `constw` | 对流速度尺度系数 W。 | `float` | `0.16` | `0.16` |
| `coriolis` | 科里奥利参数。 | `float` | `0.0001` | `0.0001` |
| `minimum_lapse_rate` | 最小位温递减率。 | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | 混合层递减率深度（单位：m）。 | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | 最小混合层高度（单位：m）。 | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | 最大混合层高度（单位：m）。 | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | 最小水体混合层高度（单位：m）。 | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | 最大水体混合层高度（单位：m）。 | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | 陆地通量阈值。 | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | 水体通量阈值。 | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | 水体温度模式。 | `int` | `0` | `0` |
| `prognostic_ocean_category` | 预报海洋类别。 | `int` | `16` | `16` |
| `overwater_flux_method` | 水面通量方法。 | `int` | `10` | `10` |
| `coastal_shelf_length_km` | 沿海陆架长度（单位：km）。 | `float` | `0.0` | `0.0` |
| `warm_layer` | COARE 暖层修正选项。 | `int` | `0` | `0` |
| `cool_skin` | COARE 冷皮层修正选项。 | `int` | `0` | `0` |
| `convective_relaxation` | 对流松弛。 | `int` | `1` | `1` |
| `convective_relaxation_time_s` | 对流松弛时间（单位：s）。 | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | 降水插值。 | `int` | `2` | `2` |
| `precipitation_radius_km` | 降水半径（单位：km）。 | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | 降水截断阈值。 | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | 浑浊度系数1。 | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | 浑浊度系数2。 | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | 云量系数1。 | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | 云量系数2。 | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | 净辐射系数1。 | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | 净辐射系数2。 | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | 净辐射系数3。 | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | 地面温度选项。 | `int` | `-1` | `-1` |
| `upper_temperature_option` | 高空温度选项。 | `int` | `-1` | `-1` |
| `lapse_height_m` | 递减率高度（单位：m）。 | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | 温度插值幂指数。 | `int` | `1` | `1` |
| `temperature_radius_km` | 温度半径（单位：km）。 | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | 温度站点上限。 | `int` | `5` | `5` |
| `water_temperature_gradient_below` | 水体温度梯度下方。 | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | 水体温度梯度上方。 | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | 水体温度类别MIN。 | `int` | `55` | `55` |
| `water_temperature_category_max` | 水体温度类别MAX。 | `int` | `55` | `55` |
| `iavet` | 温度场垂直平均选项。 | `int` | `1` | `1` |
| `iavezi` | 混合层高度所用风场平均选项。 | `int` | `1` | `1` |
| `mixing_wind_level` | 混合层风场层。 | `int` | `1` | `1` |
| `mnmdav` | 执行平均所需的最少有效值数量。 | `int` | `1` | `10` |
| `iwfcod` | 风场模型计算方案代码。 | `int` | `1` | `1` |
| `wind_extrapolation` | 风场外推。 | `int` | `-1` | `-1` |
| `upper_air_separation_cells` | 高空探空间距格点。 | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | 外推静风。 | `int` | `0` | `0` |
| `wind_bias` | 逐层风向或风分量加性偏差控制值。 | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | 是否启用可变半径。 | `bool` | `True` | `True` |
| `minimum_station_radius_km` | 最小站点半径（单位：km）。 | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | 最大地面半径（单位：km）。 | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | 最大高空半径（单位：km）。 | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | 最大水体半径（单位：km）。 | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | 地面背景场半径（单位：km）。 | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | 高空背景场半径（单位：km）。 | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | 预报半径（单位：km）。 | `float` | `0.0` | `0.0` |
| `stations_per_level` | 每个垂直层最多使用的观测站数量。 | `Sequence[int]` | `(5, 5, … (10 items))` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | 风场垂直外推使用的逐层系数。 | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | 障碍物数量。 | `int` | `0` | `0` |
| `barrier_top_level` | 障碍物顶部层。 | `int` | `0` | `10` |
| `barriers` | 用于气流阻塞计算的障碍物起止坐标。 | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | 湖陆风区域框、岸线线段及其关联站点。 | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | 基于弗劳德数的地形绕流调整开关。 | `int` | `1` | `1` |
| `ikine` | 运动学地形效应调整开关。 | `int` | `0` | `0` |
| `iobr` | 地形阻塞效应调整开关。 | `int` | `0` | `0` |
| `islope` | 坡面流调整开关。 | `int` | `1` | `1` |
| `critical_froude` | 临界弗劳德数。 | `float` | `1.0` | `1.0` |
| `smoothing_passes` | 每个垂直层的风场平滑次数。 | `Sequence[int]` | `(2, 4, … (10 items))` | `[2, 4, … (10 items)]` |
| `divergence_limit` | 散度上限。 | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | 散度迭代次数。 | `int` | `50` | `50` |
| `kinematic_alpha` | 运动学风场调整的松弛系数 Alpha。 | `float` | `0.1` | `0.1` |
| `z_faces_m` | 垂直层界面高度，数量比模型层数多 1。 | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | 基准时区。 | `int` | `0` | `0` |
| `base_timezone_label` | 基准时区标签。 | `str \| None` | `None` | `'UTC-0000'` |
| `terrain_radius_km` | 地形半径（单位：km）。 | `float` | `15.0` | `15.0` |
| `half_average_angle` | 半平均角度。 | `float` | `30.0` | `30.0` |
| `include_w_temperature` | 是否启用包含W温度。 | `bool` | `False` | `False` |
| `compression` | 压缩级别。 | `int` | `1` | `1` |
| `control_lines` | 写入传统输出元数据的原始 CALMET 控制文件文本。 | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | MESOPAC输出路径。 | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | 距离到海岸输出路径。 | `str \| Path \| None` | `None` | `None` |
| `listing_output` | 列表文件输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | 是否启用打印字段。 | `bool` | `False` | `False` |
| `print_interval_hours` | 打印间隔（单位：小时）。 | `int` | `1` | `1` |
| `print_uv_levels` | 打印UV层。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | 打印W层。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | 打印温度层。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | 选择写入文本输出的地面场开关数组。 | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | 是否启用微气象调试。 | `bool` | `False` | `False` |
| `debug_first_step` | 调试首个时步。 | `int` | `1` | `1` |
| `debug_last_step` | 调试末个时步。 | `int` | `1` | `1` |
| `wind_debug_output` | 风场诊断输出选项。 | `int` | `0` | `0` |
| `wind_debug_levels` | 风场调试层。 | `int` | `1` | `0` |
| `wind_debug_fields` | 选择风场模型诊断量的开关数组。 | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | 最终风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | 运动学风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | 弗劳德数风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | 坡面流风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | 风场打印输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | NX。 | `int \| None` | `必填` | `610` |
| `ny` | NY。 | `int \| None` | `必填` | `499` |
| `dx_km` | DX（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | DY（单位：km）。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | LAT0。 | `float` | `27.0` | `27.0` |
| `lon0` | LON0。 | `float` | `117.5` | `117.5` |
| `lat1` | LAT1。 | `float` | `30.0` | `30.0` |
| `lat2` | LAT2。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | X原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | Y原点（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 假东距（单位：km）。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 假北距（单位：km）。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM分区。 | `int` | `50` | `50` |
| `hemisphere` | 半球。 | `str` | `'N'` | `'N'` |
| `datum` | 大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格类型。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 极坐标处理。 | `int` | `1` | `1` |
| `ttm_scale` | TTM比例。 | `float` | `1.0` | `1.0` |

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
