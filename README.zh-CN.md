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
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。参数含义和选项
效果依据仓库内 CALMET 系列控制文件的详细注释、参考 Fortran 源码，以及实际读取
这些参数的 Python 代码交叉核对。API 变化后运行
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
| `datum` | 用于解释和写出坐标的大地基准区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | 地形采样方法：直接采样或双线性/传统采样。 | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | 地形文件路径。 | `Sequence[str \| Path]` | `()` | `[terrain/srtm3/N26E117.hgt, terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | 带有格式和源基准信息的地形文件列表。 | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=terrain/…datum='WGS-96'), … (42 items))` |
| `void_fill` | 是否填补高程输入中的空洞格点。 | `bool` | `False` | `False` |
| `cell_radius_km` | 每个输出格点收集地形样本时采用的搜索半径。 | `float` | `0.0` | `5.0` |
| `model` | TERREL 输出模型：CALMET (1)、MESOPAC (2)、ISC 极坐标 (3)、ISC 笛卡尔 (4)、NUATMOS (5) 或通用格式 (6)；同时决定输出格点平均高程还是峰值高程。 | `int` | `1` | `1` |
| `processing_method` | TERREL 矩形或极坐标处理算法选项。 | `int \| None` | `None` | `2` |
| `receptor_input` | 接收点输入路径。 | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | 接收点输出路径。 | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | 离散接收点输入文件的列数。 | `int` | `2` | `2` |
| `interpolate_receptors` | 是否计算离散接收点处的地形高度。 | `bool` | `False` | `True` |
| `receptor_radius_km` | 计算离散接收点高程时采用的采样半径。 | `float` | `0.0` | `0.15` |
| `previous_save` | 先前保存路径。 | `str \| Path \| None` | `None` | `None` |
| `save_output` | 保存输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | 岸线文件路径。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | GSHHS文件路径。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 岸线来源大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | 岸线WDBII大地基准。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | 按输入数据库类别指定噪声地形值的替换方法代码。 | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | 各数据库类别用于噪声检查的最小有效高程。 | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | 按数据库类别指定缺失地形值的替换方法代码。 | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | 地形样本仍缺失时采用的后备高程。 | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | PLOT输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | 原始回显输出路径。 | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | 质量检查通过所需的最小有效采样比例。 | `int` | `75` | `75` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 东西方向的网格单元数。 | `int \| None` | `必填` | `610` |
| `ny` | 南北方向的网格单元数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 东西方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 南北方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `projection` | 地图投影代码：UTM、TTM、LCC、PS、EM 或 LAZA。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | 投影原点或切点纬度，单位为十进制度。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点经度，单位为十进制度。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 网格单元 (1,1) 西南角的东向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 网格单元 (1,1) 西南角的北向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `grid_type` | 网格几何类型：矩形、偏移矩形或极坐标网格。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL 极坐标网格处理方法。 | `int` | `1` | `2` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 共 43 个参数；浙江控制文件：`examples/zhejiang/config/ctgproc.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `landuse_files` | 土地利用文件路径。 | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `category_map` | 输入土地利用类别到输出类别的映射。 | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | 土地利用数据库格式或数据族代码。 | `str` | `'GLAZAS'` | `'GEN'` |
| `mesh_density` | 土地利用数据库在每个输出格内采用的子网格采样密度。 | `int` | `12` | `12` |
| `input_datum` | 输入土地利用坐标采用的大地基准。 | `str` | `'ESR-S'` | `'WGS-84'` |
| `output_format` | 输出编码：按文件名判断 (auto)、传统 CTGPROC 格式 (legacy)，或 NetCDF。 | `str` | `'auto'` | `'auto'` |
| `coastline_file` | 岸线文件路径。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | GSHHS文件路径。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 岸线来源大地基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | 岸线WDBII大地基准。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | 依据给定岸线交换陆地与水体分类。 | `bool` | `False` | `False` |
| `fill_empty_ocean` | 使用 ocean_category 填充没有样本的海洋格点。 | `bool` | `False` | `False` |
| `ocean_category` | 赋给海洋格点的土地利用类别。 | `int` | `55` | `55` |
| `previous_landuse` | 先前土地利用路径。 | `str \| Path \| None` | `None` | `None` |
| `final_output` | 是否写出最终 CTGPROC 土地利用数据集。 | `bool` | `True` | `True` |
| `landuse_sources` | 带有格式、基准和网格采样密度的土地利用文件列表。 | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=examples/zhejiang/landuse/zhejiang_generic.landuse, input_type='GEN', input_datum='WGS-84', mesh_density=12))` |
| `snow_processing` | 是否启用随季节变化的积雪地表属性。 | `bool` | `False` | `False` |
| `qa_cell_output` | 是否启用QA格点输出。 | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | 质量检查通过所需的最小有效采样比例。 | `int` | `75` | `75` |
| `qa_output_directory` | QA输出目录路径。 | `str \| Path \| None` | `None` | `None` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 东西方向的网格单元数。 | `int \| None` | `必填` | `610` |
| `ny` | 南北方向的网格单元数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 东西方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 南北方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `projection` | 地图投影代码：UTM、TTM、LCC、PS、EM 或 LAZA。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | 投影原点或切点纬度，单位为十进制度。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点经度，单位为十进制度。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 网格单元 (1,1) 西南角的东向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 网格单元 (1,1) 西南角的北向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 用于解释和写出坐标的大地基准区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格几何类型：矩形、偏移矩形或极坐标网格。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL 极坐标网格处理方法。 | `int` | `1` | `1` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 共 52 个参数；浙江控制文件：`examples/zhejiang/config/makegeo.inp`。</summary>

| 参数 | 含义 | Python 类型 | 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `terrain` | 地形路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `landuse` | 土地利用路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `water_threshold` | 将格点判定为水体所需的最小水体比例。 | `float` | `0.5` | `0.5` |
| `missing_category` | 有效覆盖不足时赋给格点的土地利用类别。 | `int` | `55` | `55` |
| `minimum_coverage` | 格点可接受的最小土地利用覆盖比例总和。 | `float` | `0.96` | `0.96` |
| `secondary_landuse` | 第二套土地利用路径。 | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | 是否读取地形高程并写入地理物理输出。 | `bool` | `True` | `True` |
| `input_properties` | 每个输入土地利用类别对应的地表属性元组。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | 被视为水体的输入土地利用类别。 | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | 地理物理输出中保留的有序土地利用类别。 | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | CALMET 视为水体的最小输出类别。 | `int` | `50` | `50` |
| `output_water_max` | CALMET 视为水体的最大输出类别。 | `int` | `55` | `55` |
| `category_splits` | 输入类别、输出类别及拆分比例记录。 | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | 输出 MAKEGEO 中间计算信息的网格单元，索引从 1 开始。 | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | 是否启用随季节变化的积雪地表属性。 | `bool` | `False` | `False` |
| `snow_data` | 积雪数据路径。 | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | 积雪处理覆盖的首个日期，格式 YYYYMMDD。 | `int \| None` | `None` | `None` |
| `snow_end_date` | 积雪处理覆盖的末个日期，格式 YYYYMMDD。 | `int \| None` | `None` | `None` |
| `snow_properties` | 每个类别在积雪条件下对应的地表属性元组。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | 各类别的雪龄分级和衰减因子。 | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | 积雪条件下空气动力粗糙度的修正方法。 | `int` | `1` | `1` |
| `snow_height_scale_m` | 控制粗糙度修正的积雪深度尺度。 | `float` | `1.0` | `10.0` |
| `snow_albedo_method` | 根据雪龄计算积雪反照率的方法。 | `int` | `1` | `1` |
| `snow_depth_minimum_m` | 应用积雪地表属性所需的最小雪深。 | `float` | `0.0` | `0.01` |
| `landuse_plot_output` | 土地利用PLOT输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | 地形PLOT输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | 颜色映射输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | 质量检查 Surfer 网格的坐标约定：格点中心边界 (0) 或格点外缘边界 (1)。 | `int` | `0` | `0` |
| `title` | 标题。 | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 东西方向的网格单元数。 | `int \| None` | `必填` | `610` |
| `ny` | 南北方向的网格单元数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 东西方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 南北方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `projection` | 地图投影代码：UTM、TTM、LCC、PS、EM 或 LAZA。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | 投影原点或切点纬度，单位为十进制度。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点经度，单位为十进制度。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 网格单元 (1,1) 西南角的东向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 网格单元 (1,1) 西南角的北向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 用于解释和写出坐标的大地基准区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格几何类型：矩形、偏移矩形或极坐标网格。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL 极坐标网格处理方法。 | `int` | `1` | `1` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

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
| `noobs` | 观测模式：0 使用地面、水面和高空观测；1 用预报资料替代高空观测；2 所有高度均使用预报资料。 | `int` | `2` | `2` |
| `run_type` | CALMET 运行类型：仅风场，或风场加微气象场。 | `int` | `1` | `1` |
| `save_output` | 是否启用保存输出。 | `bool` | `True` | `True` |
| `legacy_output_format` | 传统 CALMET 输出格式选项。 | `int` | `1` | `1` |
| `test_mode` | 1 在输入设置检查后停止；2 继续执行模型计算。 | `int` | `2` | `2` |
| `regulatory_mode` | 是否检查所选技术选项符合监管模式要求。 | `int` | `0` | `0` |
| `timestep_seconds` | CALMET 计算时间步长，必须能整除 1 小时。 | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | 相邻预报场输入记录之间的时间间隔。 | `int` | `3600` | `3600` |
| `cloud_method` | 云场方法：观测 (1)、CLOUD.DAT (2)、850 hPa 相对湿度 (3) 或全层相对湿度 (4)。 | `int` | `4` | `4` |
| `cloud_data` | 云量数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | CLOUD.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `cloud_output` | 云场输出选项。 | `int` | `0` | `0` |
| `surface_data` | 地面数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | SURF.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `surface_control_stations` | 地面站名称、编号、坐标、时区和风速仪高度记录。 | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | 高空探空文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | 高空站名称、编号、坐标和时区记录。 | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | 海面文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | 诊断数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | 地表温度来源：内部计算 (0) 或读取 DIAG.DAT (1)。 | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | 诊断场是否使用观测温度递减率。 | `int` | `0` | `0` |
| `diagnostic_initial_wind` | 初始猜测风来源：内部计算 (0) 或读取 DIAG.DAT (1)。 | `int` | `0` | `0` |
| `diagnostic_surface_wind` | 地面风来源：SURF.DAT 风速风向 (0) 或 DIAG.DAT U/V (1)。 | `int` | `0` | `0` |
| `diagnostic_upper_wind` | 高空风来源：UP.DAT 风速风向 (0) 或 DIAG.DAT U/V (1)。 | `int` | `0` | `0` |
| `initial_guess_upper_station` | 初始风猜测场使用的高空站；-1 保留空间变化的三维猜测场。 | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | 生成区域平均初始风时使用的底部和顶部高度。 | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | 是否使用已有 CALMET 风场作为初始猜测场。 | `int` | `0` | `0` |
| `initial_guess_calmet_files` | 初始猜测场CALMET文件路径。 | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | 权重数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | 降水站数量；-1 表示使用预报资料中的降水。 | `int` | `-1` | `-1` |
| `precipitation_data` | 降水数据路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | PRECIP.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `precipitation_control_stations` | 降水站名称、编号和坐标记录。 | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | 预报风进入风场模型的方式：禁用 (0)、作为第一步风场 (1/3/13)、作为初始猜测场 (2/4/14)，或作为伪观测 (5/15)；不同代码对应 CSUMM、MM4 和 MM5/3D 输入。 | `int` | `14` | `14` |
| `irhprog` | 相对湿度来源：SURF.DAT 观测 (0) 或预报场 (1)。 | `int` | `1` | `1` |
| `itprog` | 温度来源：观测 (0)、地面观测加高层预报 (1)，或所有高度均使用预报场 (2)。 | `int` | `2` | `2` |
| `imixh` | 对流混合层方法：±1 为 Maul–Carson，±2 为 Batchvarova–Gryning；负值在水面使用 OCD。 | `int` | `1` | `1` |
| `constb` | 中性机械混合层高度系数 B。 | `float` | `1.41` | `1.41` |
| `conste` | 对流混合层夹卷系数。 | `float` | `0.15` | `0.15` |
| `constn` | 稳定机械混合层高度系数 N。 | `float` | `2400.0` | `2400.0` |
| `constw` | 对流速度尺度系数 W。 | `float` | `0.16` | `0.16` |
| `coriolis` | 混合层高度方程使用的科里奥利参数绝对值，单位 s⁻¹。 | `float` | `0.0001` | `0.0001` |
| `minimum_lapse_rate` | 最小位温递减率。 | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | 在当前对流混合层顶以上计算稳定层温度递减率时采用的层厚。 | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | 最小混合层高度（单位：m）。 | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | 最大混合层高度（单位：m）。 | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | 最小水体混合层高度（单位：m）。 | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | 最大水体混合层高度（单位：m）。 | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | 维持陆地对流混合层增长所需的最小正浮力通量。 | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | 维持水面对流混合层增长所需的最小正浮力通量。 | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | 水面递减率来源：SEA.DAT (0)、预报递减率 (1)，或预报递减率和 ΔT (2)。 | `int` | `0` | `0` |
| `prognostic_ocean_category` | 预报气象输入中代表海洋的土地利用类别。 | `int` | `16` | `16` |
| `overwater_flux_method` | 水面通量方法：传统 OCD (0) 或相应 COARE 波浪方案 (10、±11、±12)。 | `int` | `10` | `10` |
| `coastal_shelf_length_km` | COARE 粗糙度计算使用的沿岸或浅水长度尺度。 | `float` | `0.0` | `0.0` |
| `warm_layer` | COARE 暖层修正选项。 | `int` | `0` | `0` |
| `cool_skin` | COARE 冷皮层修正选项。 | `int` | `0` | `0` |
| `convective_relaxation` | 浮力通量低于阈值时，是否让对流混合层高度向平衡值松弛。 | `int` | `1` | `1` |
| `convective_relaxation_time_s` | 启用对流混合层松弛时使用的时间尺度。 | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | 降水插值权重：1/R (1)、1/R² (2) 或指数/R² (3)。 | `int` | `2` | `2` |
| `precipitation_radius_km` | 降水站的影响半径。 | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | 低于该阈值的降水率置零，单位 mm/h。 | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | Holtslag–van Ulden 能量收支太阳辐射/浑浊度参数化中的 HA1 系数，单位 W/m²。 | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | Holtslag–van Ulden 能量收支太阳辐射/浑浊度参数化中的 HA2 系数，单位 W/m²。 | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | Holtslag–van Ulden 能量收支中的无量纲云量系数 HB1。 | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | Holtslag–van Ulden 能量收支中的无量纲云量系数 HB2。 | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | Holtslag–van Ulden 能量收支中的净辐射系数 HC1，单位 W/(m²·K⁶)。 | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | Holtslag–van Ulden 能量收支中的净辐射系数 HC2，单位 W/m²。 | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | Holtslag–van Ulden 能量收支中的无量纲净辐射系数 HC3。 | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | 地表温度代表站：站号；-1 为空间变化场；-2 为预报温度区域平均。 | `int` | `-1` | `-1` |
| `upper_temperature_option` | 递减率代表高空站：站号；-1 为空间变化场；-2 为预报递减率区域平均。 | `int` | `-1` | `-1` |
| `lapse_height_m` | 计算区域尺度温度递减率所用的层厚。 | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | 温度插值权重：距离反比 (1) 或距离平方反比 (2)。 | `int` | `1` | `1` |
| `temperature_radius_km` | 温度插值的影响半径。 | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | 温度插值最多使用的站点数。 | `int` | `5` | `5` |
| `water_temperature_gradient_below` | 水体格点混合层以下的默认温度梯度，单位 K/m。 | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | 水体格点混合层以上的默认温度梯度，单位 K/m。 | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | 温度插值中视为水体的起始土地利用类别。 | `int` | `55` | `55` |
| `water_temperature_category_max` | 温度插值中视为水体的结束土地利用类别。 | `int` | `55` | `55` |
| `iavet` | 是否使用混合层平均的搜索几何对插值温度进行空间平均。 | `int` | `1` | `1` |
| `iavezi` | 是否对计算得到的混合层高度进行空间平均。 | `int` | `1` | `1` |
| `mixing_wind_level` | 确定上风向平均方向时使用的风场层。 | `int` | `1` | `1` |
| `mnmdav` | 混合层高度和温度空间平均的最大搜索半径，单位为网格单元。 | `int` | `1` | `10` |
| `iwfcod` | 风场模型：仅客观分析 (0)，或完整诊断风场模块 (1)。 | `int` | `1` | `1` |
| `wind_extrapolation` | 地面风向高层外推方法；负值还会忽略高空站第一层资料。 | `int` | `-1` | `-1` |
| `upper_air_separation_cells` | 允许地面风外推的地面站与高空站最小间距；-1 取消该限制。 | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | 地面站报告静风时是否仍进行垂直外推。 | `int` | `0` | `0` |
| `wind_bias` | 逐层站点权重偏差，范围 -1 到 1：负值降低高空站权重，正值降低地面站权重，0 保持 1/R² 权重不变。 | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | 最大影响半径内没有站点时，是否扩大半径直至找到最近站点。 | `bool` | `True` | `True` |
| `minimum_station_radius_km` | 风场插值使用的最小影响半径。 | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | 陆地下垫面近地层站点的最大影响半径。 | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | 陆地高层站点的最大影响半径。 | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | 水面站点的最大影响半径。 | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | 近地层观测与初始猜测场权重相等时的距离尺度。 | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | 高层观测与初始猜测场权重相等时的距离尺度。 | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | IPROG 将预报风作为第一步风场时采用的权重距离。 | `float` | `0.0` | `0.0` |
| `stations_per_level` | 每个垂直层最多使用的观测站数量。 | `Sequence[int]` | `(5, 5, … (10 items))` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | 风场垂直外推使用的逐层系数。 | `Sequence[float]` | `(0.0, 0.0, … (10 items))` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | 阻挡风场插值的线状障碍物数量。 | `int` | `0` | `0` |
| `barrier_top_level` | 插值障碍物影响到的最高模型层。 | `int` | `0` | `10` |
| `barriers` | 用于气流阻塞计算的障碍物起止坐标。 | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | 湖陆风区域框、岸线线段及其关联站点。 | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | 基于弗劳德数的地形绕流调整开关。 | `int` | `1` | `1` |
| `ikine` | 运动学地形效应调整开关。 | `int` | `0` | `0` |
| `iobr` | 地形阻塞效应调整开关。 | `int` | `0` | `0` |
| `islope` | 坡面流调整开关。 | `int` | `1` | `1` |
| `critical_froude` | 区分地形阻塞与越岭气流的临界弗劳德数。 | `float` | `1.0` | `1.0` |
| `smoothing_passes` | 每个垂直层的风场平滑次数。 | `Sequence[int]` | `(2, 4, … (10 items))` | `[2, 4, … (10 items)]` |
| `divergence_limit` | 风场散度最小化过程的收敛目标。 | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | 散度最小化允许的最大迭代次数。 | `int` | `50` | `50` |
| `kinematic_alpha` | 运动学风场调整的松弛系数 Alpha。 | `float` | `0.1` | `0.1` |
| `z_faces_m` | 垂直层界面高度，数量比模型层数多 1。 | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | 基准时区。 | `int` | `0` | `0` |
| `base_timezone_label` | 基准时区标签。 | `str \| None` | `None` | `'UTC-0000'` |
| `terrain_radius_km` | 地形影响诊断风场调整的水平半径。 | `float` | `15.0` | `15.0` |
| `half_average_angle` | 空间平均所用上风向搜索锥的半角。 | `float` | `30.0` | `30.0` |
| `include_w_temperature` | 是否写出 CALGRID 所需的三维垂直风速和温度场。 | `bool` | `False` | `False` |
| `compression` | 压缩级别。 | `int` | `1` | `1` |
| `control_lines` | 写入传统输出元数据的原始 CALMET 控制文件文本。 | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | MESOPAC输出路径。 | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | 距离到海岸输出路径。 | `str \| Path \| None` | `None` | `None` |
| `listing_output` | 列表文件输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | 是否启用传统文本形式的气象场输出。 | `bool` | `False` | `False` |
| `print_interval_hours` | 气象场文本输出的时间间隔，单位小时。 | `int` | `1` | `1` |
| `print_uv_levels` | 启用 print_fields 时选择输出 U/V 风场层的逐层开关。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | 选择垂直风速输出层的逐层开关；还需启用 include_w_temperature。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | 选择三维温度输出层的逐层开关；还需启用 include_w_temperature。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | 选择写入文本输出的地面场开关数组。 | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | 是否打印微气象输入和内部变量；会产生大量输出。 | `bool` | `False` | `False` |
| `debug_first_step` | 微气象调试输出包含的首个时步。 | `int` | `1` | `1` |
| `debug_last_step` | 微气象调试输出包含的末个时步。 | `int` | `1` | `1` |
| `wind_debug_output` | 风场诊断输出选项。 | `int` | `0` | `0` |
| `wind_debug_levels` | 从地面开始写入风场诊断文件的层数。 | `int` | `1` | `0` |
| `wind_debug_fields` | 选择风场模型诊断量的开关数组。 | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | 最终风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | 运动学风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | 弗劳德数风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | 坡面流风场调试输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | 风场打印输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 东西方向的网格单元数。 | `int \| None` | `必填` | `610` |
| `ny` | 南北方向的网格单元数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 东西方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 南北方向网格分辨率，单位 km。 | `float \| None` | `必填` | `1.0` |
| `projection` | 地图投影代码：UTM、TTM、LCC、PS、EM 或 LAZA。 | `str` | `'LCC'` | `'LCC'` |
| `lat0` | 投影原点或切点纬度，单位为十进制度。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点经度，单位为十进制度。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 网格单元 (1,1) 西南角的东向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 网格单元 (1,1) 西南角的北向坐标，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 用于解释和写出坐标的大地基准区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | 网格几何类型：矩形、偏移矩形或极坐标网格。 | `int` | `1` | `1` |
| `ring_distances_km` | 极坐标网格各环的半径，单位 km。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | 极坐标网格各射线的角度，单位度。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | TERREL 极坐标网格处理方法。 | `int` | `1` | `1` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

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
