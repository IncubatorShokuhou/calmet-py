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
- 一个软件包覆盖五个公开处理阶段和完整流水线。
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

`run_pipeline` 把五个阶段作为一个完整的类型化关键字 API 暴露出来。参数名前的阶段
前缀用于区分同名控制项；阶段间数据由流水线管理，在各类输出路径中只需指定最终的
CALMET 输出位置：

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
    terrel_srtm_dir=Path("terrain/srtm3"),
    terrel_grid=grid,
    terrel_sampling="bilinear",
    ctgproc_landuse_files=[example / "landuse/zhejiang_generic.landuse"],
    ctgproc_grid=grid,
    ctgproc_input_type="GEN",
    ctgproc_input_datum="WGS-84",
    ctgproc_mesh_density=12,
    makegeo_grid=grid,
    calmet_start=datetime(2018, 3, 18, 0),
    calmet_end=datetime(2018, 3, 18, 1),
    calmet_grid=grid,
    calmet_noobs=2,
    calmet_iprog=14,
    calmet_irhprog=1,
    calmet_itprog=2,
    calmet_wind_extrapolation=-1,
    calmet_precipitation_station_count=-1,
    output=Path("output/zhejiang/calmet.nc"),
)
```

下方各阶段参数表中的每个字段，都可分别加上 `calwrf_`、`terrel_`、`ctgproc_`、
`makegeo_` 或 `calmet_` 前缀后直接传入。例如 CALMET 的 `z_faces_m` 对应
`calmet_z_faces_m`。网格既可以像上例一样传入 `GridConfig`，也可以写成
`terrel_nx`、`terrel_ny`、`terrel_dx_km` 等带阶段前缀的独立参数。未传项沿用对应
单阶段 API 的 Python 默认值。为了保证只保留最终产品，流水线会重新连接阶段间路径，
并关闭中间输出和诊断输出；纯参数模式不需要填写这些路径。

这一单一最终产品约定要求 `makegeo_snow_processing=False`、
`calmet_save_output=True` 和 `calmet_test_mode=2`：积雪流程会生成多份按日期区分的 GEO
文件，后两项则分别用于执行计算和保存最终 CALMET 结果。

各阶段通过自动清理的临时目录交换 NetCDF 数据；流水线会关闭传统 M3D/M2D、列表、
质检、绘图、保存和调试产物，因此持久写入的只有指定的最终 CALMET 文件（使用上例
的 NetCDF 格式时）。传统 CALMET 最终格式还会按格式要求写出 `.aux` 配套文件。

已有控制文件时，也可以把包含全部五份控制文件的**文件夹**作为第一个参数：

```python
result = pycalmet.run_pipeline(
    Path("examples/zhejiang/config"),  # 文件夹，不是单个 .inp 文件
    output=Path("output/zhejiang/calmet.nc"),
)
```

目录模式会自动查找 `calwrf.inp`、`terrel.inp`、`ctgproc.inp`、
`makegeo.inp` 和 `calmet.inp`。省略 `output` 时，使用 `calmet.inp` 中实际生效的
输出路径。分组形式的 `calwrf=`、`terrel=`、`ctgproc=`、`makegeo=` 和 `calmet=`
也可以分别接收一份控制文件路径、参数映射或对应的类型化配置对象。`PipelineConfig`
可将五个阶段的类型化配置集中为一个对象，并作为 `run_pipeline` 的第一个参数传入。

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
`datetime`，路径接受 `str` 或 `Path`，重复项接受 Python 序列。对于调用方未传入的参数，
如果原程序为相应标量或模型控制项提供了可直接使用的默认值，关键字 API 会沿用该值。文件
路径仍要求显式提供，不继承传统程序基于工作目录的默认文件名；源码默认值只是“必须
填写”哨兵或 Fortran 没有规定默认值的控制项，则采用下方表格列出的 Python 默认值。
下面是浙江 WRF 流程使用的纯预报资料配置：

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

函数签名会列出全部支持项，包括某份 `.inp` 没有写出、但 Fortran 程序定义了默认
值的参数；IDE 补全、类型检查器、`help()` 和 `inspect.signature()` 都能直接看到。

<!-- parameter-reference:start -->
## 完整参数参考

以下表格由公开函数签名和类型化配置模型生成。“Python 默认值”表示 `inp=None` 模式下调用方
省略参数时采用的值；“浙江示例值”表示解析仓库内置控制文件后得到的实际值。`inp`
本身和 `**legacy_parameters` 是入口及兼容机制，不属于模型控制项，因此不在每张表中
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。参数含义和选项
效果主要依据 Scire、Robe、Fernau 和 Yamartino 的 *A User's Guide for the CALMET
Meteorological Model (Version 5)*（2000 年 1 月），尤其是第 2 章及 4.2–4.3 节。
该指南早于 NOOBS、新版预报资料开关、COARE 和积雪处理等控制项；这些较新控制项
再与仓库内 CALMET 6.5 控制模板、参考 Fortran 源码，以及实际读取参数的 Python
代码交叉核对。API 变化后运行
`python tools/generate_parameter_reference.py` 重新生成；CI 可使用 `--check` 检查。

<details>
<summary><strong>CALWRF</strong> — 共 14 个参数；浙江控制文件：`examples/zhejiang/config/calwrf.inp`。</summary>

| 参数 | 含义 | Python 类型 | Python 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `wrf_files` | 按时间顺序排列的 WRF NetCDF 文件。 | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc, examples/zhejiang/wrfout_d01_2018-03-18_01_00_00.nc, … (3 items)]` |
| `start` | 计算开始时刻。 | `str \| int \| datetime \| None` | `必填` | `'2018031800'` |
| `end` | 计算结束时刻。 | `str \| int \| datetime \| None` | `必填` | `'2018031802'` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/calwrf.nc` |
| `create_surface` | 同时请求 legacy_output 时，是否另外生成传统 M2D 地面气象文件。 | `bool` | `True` | `False` |
| `compression` | CALWRF 输出的 NetCDF deflate 压缩级别（0 表示不压缩）。 | `int` | `1` | `1` |
| `i_range` | WRF 东西向索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `j_range` | WRF 南北向索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `k_range` | WRF 垂直层索引范围，采用从 1 开始的闭区间。 | `tuple[int, int] \| None` | `None` | `None` |
| `legacy_output` | 可选的传统 CALWRF 3D.DAT/M3D 输出，与主 NetCDF 文件同时生成。 | `str \| Path \| None` | `None` | `None` |
| `surface_output` | 可选的传统 CALWRF 二维地面/M2D 路径；create_surface 为真且同时请求 legacy_output 时写出。 | `str \| Path \| None` | `None` | `None` |
| `title` | 写入 CALWRF 输出元数据的说明标题。 | `str` | `'Create 3D.DAT file for WRF output'` | `'Create CALWRF data for the Zhejiang example'` |
| `listing_output` | CALWRF 处理日志/列表文件输出路径。 | `str \| Path \| None` | `None` | `output/zhejiang/calwrf.lst` |
| `legacy_file_rain_reset` | 是否在每个传统输入文件边界重置累计降水。 | `bool` | `False` | `False` |

</details>

<details>
<summary><strong>TERREL</strong> — 共 49 个参数；浙江控制文件：`examples/zhejiang/config/terrel.inp`。</summary>

| 参数 | 含义 | Python 类型 | Python 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `srtm_dir` | terrain_sources 为空时扫描原始 SRTM `[NS]dd[EW]ddd.hgt` 瓦片的目录。 | `str \| Path \| None` | `必填` | `examples/zhejiang/config` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `datum` | 输出网格的大地基准/区域代码；各输入源另行指定其输入基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `sampling` | 地形坐标转换方法：`direct` 逐个转换 DEM 样本，精度较高；`bilinear`/`legacy` 只转换图幅角点，再插值内部坐标。 | `str` | `'bilinear'` | `'legacy'` |
| `terrain_files` | 选定的地形文件路径；相对 SRTM 瓦片名会在 srtm_dir 内解析。 | `Sequence[str \| Path]` | `()` | `[terrain/srtm3/N26E117.hgt, terrain/srtm3/N26E118.hgt, … (42 items)]` |
| `terrain_sources` | 地形源记录 `(路径, 输入类型, 输入基准)`；输入类型选择 DEM 读取器，例如 SRTM1/3、GTOPO30、CDED、GEN 或 GEOTIFF。 | `Sequence[TerrelSource]` | `()` | `(TerrelSource(path=terrain/srtm3/N26E117.hgt, input_type='SRTM3', input_datum='WGS-96'), TerrelSource(path=terrain/…datum='WGS-96'), … (42 items))` |
| `void_fill` | 是否用邻近有效格点填补空白地形输出格点，最大搜索距离为 cell_radius_km。 | `bool` | `False` | `False` |
| `cell_radius_km` | 启用 void_fill 时查找有效邻格的最大半径，单位 km；应至少为一个网格间距。 | `float` | `0.0` | `5.0` |
| `model` | TERREL 输出模型：CALMET (1)、MESOPAC (2)、ISC 极坐标 (3)、ISC 笛卡尔 (4)、NUATMOS (5) 或通用格式 (6)；同时决定输出格点平均高程还是峰值高程。 | `int` | `1` | `1` |
| `processing_method` | TERREL 处理方法；极坐标网格中 1 选择 NORMAL 扇区，2 选择 SCREEN 扇区，并覆盖 grid.polar_processing。 | `int \| None` | `None` | `2` |
| `receptor_input` | 可选的离散接收点 XY 输入文件；提供该文件即启用接收点处理。 | `str \| Path \| None` | `None` | `None` |
| `receptor_output` | 写出接收点记录及 TERREL 计算地形高程的文件。 | `str \| Path \| None` | `None` | `None` |
| `receptor_columns` | 离散接收点输入列数：2 表示 (X, Y)，4 表示 (X, Y, 地形高程, 接收点高度)；TERREL 会替换第 3 列。 | `int` | `4` | `2` |
| `interpolate_receptors` | 离散接收点算法：为真时插值高程，为假时搜索最高高程。 | `bool` | `False` | `True` |
| `receptor_radius_km` | 离散接收点高程的搜索/插值半径，单位 km；最高值搜索模式要求为正数。 | `float` | `0.0` | `0.15` |
| `previous_save` | 先前一轮 TERREL 的二进制累积文件，用于继续累计样本总和与命中数。 | `str \| Path \| None` | `None` | `None` |
| `save_output` | 是否写出包含样本总和及命中数的 TERREL 二进制累积文件，供后续轮次继续。 | `str \| Path \| None` | `None` | `output/zhejiang/terrel.sav` |
| `coastline_file` | 处理后的 BLN 岸线输入；提供 gshhs_file 时，该路径接收生成的 BLN 数据。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | 用于生成处理后 BLN 岸线的原始 GSHHS 岸线数据库。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 原始 GSHHS/岸线源坐标采用的大地基准/区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | WDBII 岸线坐标采用的大地基准/区域代码。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `noise_replacement` | 各岸线类别的噪声处理：0 不检查；1 把低于 noise_minimum_m 的值置为缺失；2 截到最小值；3 替换为 default_elevation_m。 | `Sequence[int]` | `(0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0)` |
| `noise_minimum_m` | 各数据库类别用于噪声检查的最小有效高程。 | `Sequence[float]` | `(0.0, 1.0, 1.0, 1.0, 1.0)` | `(0.0, 1.0, 1.0, 1.0, 1.0)` |
| `missing_replacement` | 各岸线类别的空值处理：0 保留；1 仅替换最终空格点；2 替换缺失样本及最终空格点；3 强制该类别使用默认高程。 | `Sequence[int]` | `(3, 0, 0, 0, 0)` | `(3, 0, 0, 0, 0)` |
| `default_elevation_m` | 地形样本仍缺失时采用的后备高程。 | `Sequence[float]` | `(0.0, 0.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 0.0, 0.0)` |
| `plot_output` | 可选的 Surfer ASCII 网格，写出 TERREL 地形高程供质检或绘图。 | `str \| Path \| None` | `None` | `output/zhejiang/terrain.grd` |
| `raw_echo_output` | 可选的原始未过滤地形样本 ASCII 文件，记录 `(x_km, y_km, elevation_m)`。 | `str \| Path \| None` | `None` | `None` |
| `qa_hit_threshold_percent` | 标记采样数低于区域平均命中数这一百分比（0–100）的格点。 | `int` | `75` | `75` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 笛卡尔 X/东西方向网格数；TERREL 极坐标网格中表示环数。 | `int \| None` | `必填` | `610` |
| `ny` | 笛卡尔 Y/南北方向网格数；TERREL 极坐标网格中表示射线数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 笛卡尔 X 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 笛卡尔 Y 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影代码：UTM、横轴墨卡托 (TTM)、兰伯特正形圆锥 (LCC)、极射赤面 (PS)、赤道墨卡托 (EM) 或兰伯特方位等积 (LAZA)。 | `str` | `'UTM'` | `'LCC'` |
| `lat0` | 投影原点或切点的带符号纬度，单位度（北纬为正）；EM 固定为 0。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点的带符号经度，单位度（东经为正）。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 笛卡尔网格单元 (1,1) 西南角的东向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 笛卡尔网格单元 (1,1) 西南角的北向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `grid_type` | TERREL 网格类型：1 原点为单元 (1,1) 西南角；2 原点为该单元中心；3 原点为极坐标环中心。 | `int` | `1` | `1` |
| `ring_distances_km` | TERREL 极坐标网格各环半径，单位 km；grid_type=3 时每个环提供一个值。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL 极坐标网格射线方位角，从正北顺时针计，单位度；grid_type=3 时使用。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 仅供 TERREL 使用的极坐标方法：1 NORMAL，扇区边界位于相邻环/射线中间；2 SCREEN，径向从当前环延至下一外环，角向边界位于相邻射线中间。 | `int` | `2` | `2` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CTRPROC / CTGPROC</strong> — 共 43 个参数；浙江控制文件：`examples/zhejiang/config/ctgproc.inp`。</summary>

| 参数 | 含义 | Python 类型 | Python 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `landuse_files` | CTGPROC 读取的土地利用数据库文件路径。 | `Sequence[str \| Path] \| None` | `必填` | `[examples/zhejiang/landuse/zhejiang_generic.landuse]` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `category_map` | 分箱后应用的源土地利用类别到目标类别映射；映射到同一目标的命中数会相加。 | `dict[int, int] \| None` | `{}` | `{}` |
| `input_type` | 土地利用数据库读取器名称/代码，例如 CTG、GEN、NLCD、CORINE、GLC2K、UMDGLC 或 MODIS。 | `str` | `'CTG'` | `'GEN'` |
| `mesh_density` | 每个输入土地利用单元在每条边上的细分数，共生成 mesh_density² 个采样点；不得小于 1。 | `int` | `1` | `12` |
| `input_datum` | 输入土地利用坐标采用的大地基准。 | `str` | `'NAS-C'` | `'WGS-84'` |
| `output_format` | 输出编码：按文件名判断 (auto)、传统 CTGPROC 格式 (legacy)，或 NetCDF。 | `str` | `'auto'` | `'auto'` |
| `coastline_file` | 处理后的 BLN 岸线输入；提供 gshhs_file 时，该路径接收生成的 BLN 数据。 | `str \| Path \| None` | `None` | `None` |
| `gshhs_file` | 用于生成处理后 BLN 岸线的原始 GSHHS 岸线数据库。 | `str \| Path \| None` | `None` | `None` |
| `coastline_source_datum` | 原始 GSHHS/岸线源坐标采用的大地基准/区域代码。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `coastline_wdbii_datum` | WDBII 岸线坐标采用的大地基准/区域代码。 | `str` | `'WGS-72'` | `'WGS-72'` |
| `coastline_swap` | 读取样本时，把位于所给岸线外海侧的点替换为 ocean_category。 | `bool` | `False` | `False` |
| `fill_empty_ocean` | 最终处理时，把中心位于海上且命中数为零的格点填为 ocean_category；None 沿用 Fortran 默认逻辑，仅在提供 coastline_file 时启用。 | `bool \| None` | `None → coastline_file is not None` | `False` |
| `ocean_category` | 赋给海洋格点的土地利用类别。 | `int` | `55` | `55` |
| `previous_landuse` | 用于继续多轮土地利用累计的先前 CTGPROC 整数命中数输出。 | `str \| Path \| None` | `None` | `None` |
| `final_output` | CTGPROC 处理阶段：为真时写出供 MAKEGEO 使用的土地利用比例；为假时写出整数命中数，供 previous_landuse 继续处理。 | `bool` | `True` | `True` |
| `landuse_sources` | 土地利用源记录 `(路径, 输入类型, 输入基准, 网格细分密度)`。 | `Sequence[CtgprocSource]` | `()` | `(CtgprocSource(path=examples/zhejiang/landuse/zhejiang_generic.landuse, input_type='GEN', input_datum='WGS-84', mesh_density=12))` |
| `snow_processing` | 启用相应阶段的积雪流程：CTGPROC 写出 SNODAS 参考记录；MAKEGEO 生成逐日积雪修正 GEO 文件。 | `bool` | `False` | `False` |
| `qa_cell_output` | 是否写出 CTGPROC 原始及细分输入点的质检坐标文件；输出可能很大。 | `bool` | `False` | `False` |
| `qa_hit_threshold_percent` | 标记采样数低于区域平均命中数这一百分比（0–100）的格点。 | `int` | `75` | `75` |
| `qa_output_directory` | 接收 CTGPROC QACTG/QAGLAZ/QAMESH 坐标质检文件的目录。 | `str \| Path \| None` | `None` | `None` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 笛卡尔 X/东西方向网格数；TERREL 极坐标网格中表示环数。 | `int \| None` | `必填` | `610` |
| `ny` | 笛卡尔 Y/南北方向网格数；TERREL 极坐标网格中表示射线数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 笛卡尔 X 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 笛卡尔 Y 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影代码：UTM、横轴墨卡托 (TTM)、兰伯特正形圆锥 (LCC)、极射赤面 (PS)、赤道墨卡托 (EM) 或兰伯特方位等积 (LAZA)。 | `str` | `'UTM'` | `'LCC'` |
| `lat0` | 投影原点或切点的带符号纬度，单位度（北纬为正）；EM 固定为 0。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点的带符号经度，单位度（东经为正）。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 笛卡尔网格单元 (1,1) 西南角的东向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 笛卡尔网格单元 (1,1) 西南角的北向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 输出网格的大地基准/区域代码；各输入源另行指定其输入基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL 网格类型：1 原点为单元 (1,1) 西南角；2 原点为该单元中心；3 原点为极坐标环中心。 | `int` | `1` | `1` |
| `ring_distances_km` | TERREL 极坐标网格各环半径，单位 km；grid_type=3 时每个环提供一个值。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL 极坐标网格射线方位角，从正北顺时针计，单位度；grid_type=3 时使用。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 仅供 TERREL 使用的极坐标方法：1 NORMAL，扇区边界位于相邻环/射线中间；2 SCREEN，径向从当前环延至下一外环，角向边界位于相邻射线中间。 | `int` | `2` | `2` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>MAKEGEO</strong> — 共 52 个参数；浙江控制文件：`examples/zhejiang/config/makegeo.inp`。</summary>

| 参数 | 含义 | Python 类型 | Python 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `terrain` | 格点高程输入，通常为 TERREL 输出；仅在 use_terrain 为真时读取。 | `str \| Path \| None` | `必填` | `output/zhejiang/terrain.nc` |
| `landuse` | CTGPROC 生成的主要最终土地利用比例数据集。 | `str \| Path \| None` | `必填` | `output/zhejiang/landuse.nc` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `water_threshold` | 水体类别比例达到此阈值后，优势类别只从水体类别中选择；否则只从非水体类别中选择。 | `float` | `0.5` | `0.5` |
| `missing_category` | 土地利用有效总覆盖率低于 0.001 时插入的类别；该类别必须存在于 input_properties。 | `int` | `55` | `55` |
| `minimum_coverage` | 可接受的土地利用总比例下限：达到该值则重新归一化，低于 0.001 视为缺失，介于两者之间则报错。 | `float` | `0.96` | `0.96` |
| `secondary_landuse` | 可选的第二土地利用数据，仅用于补充主数据集中的缺失格点。 | `str \| Path \| None` | `None` | `None` |
| `use_terrain` | 为真时读取地形高程写入 GEO；为假时地形高程写为零。 | `bool` | `True` | `True` |
| `input_properties` | 各类别的 `(粗糙度_m, 反照率, Bowen 比, 土壤热通量参数, 人为热_W/m², 叶面积指数, 输出类别)`；粗糙度按面积作对数加权，其余属性作算术加权。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{11: (0.5, 0.18, 1.0, 0.2, 0.0, 1.0, 10), 12: (1.0, 0.18, 1.5, 0.25, 0.0, 0.2, 10), … (38 items)}` |
| `water_categories` | 被视为水体的输入土地利用类别。 | `Sequence[int]` | `(51, 52, 53, 54, 55)` | `[51, 52, 53, 54, 55]` |
| `output_categories` | 地理物理输出中保留的有序土地利用类别。 | `Sequence[int]` | `(10, 20, … (14 items))` | `[10, 20, … (14 items)]` |
| `output_water_min` | CALMET 视为水体的最小输出类别。 | `int` | `50` | `50` |
| `output_water_max` | CALMET 视为水体的最大输出类别。 | `int` | `55` | `55` |
| `category_splits` | MAKEGEO 在属性加权前应用的 `(源类别, 目标类别, 百分数)` 记录；同一源类别的百分数应合计为 100。 | `Sequence[tuple[int, int, float]]` | `()` | `()` |
| `qa_cell` | 可选的从 1 开始 `(i, j)` 格点，用于报告地形、土地利用和地表属性的中间计算。 | `tuple[int, int] \| None` | `None` | `(20, 15)` |
| `snow_processing` | 根据 SNODAS 雪深/雪龄，在包含首尾日期的 snow_start_date..snow_end_date 范围内逐日生成积雪修正 GEO 文件。 | `bool` | `False` | `False` |
| `snow_data` | 用于生成逐日积雪修正地理物理文件的 SNODAS 雪深/雪龄输入。 | `str \| Path \| None` | `None` | `None` |
| `snow_start_date` | 积雪处理覆盖的首个日期，格式 YYYYMMDD。 | `int \| None` | `None` | `None` |
| `snow_end_date` | 积雪处理覆盖的末个日期，格式 YYYYMMDD。 | `int \| None` | `None` | `None` |
| `snow_properties` | 积雪条件属性，七个字段与 input_properties 相同；映射后的输出类别必须与正常属性记录一致。 | `dict[int, tuple[float, float, float, float, float, float, int]] \| None` | `{}` | `{}` |
| `snow_age_parameters` | 各积雪类别的 `(老化时间尺度_天, 反照率衰减系数)`，供雪龄反照率算法使用。 | `dict[int, tuple[int, float]] \| None` | `{}` | `{}` |
| `snow_roughness_method` | 积雪粗糙度方法：1 从推算的障碍物高度中扣除雪深；2 有积雪时直接采用用户给定的积雪粗糙度。 | `int` | `1` | `1` |
| `snow_height_scale_m` | 积雪粗糙度方法 1 使用的障碍物高度与粗糙度比例尺度 HSCL。 | `float` | `10.0` | `10.0` |
| `snow_albedo_method` | 积雪反照率方法：1 随雪龄把表中反照率衰减到较低值；2 有积雪时直接采用用户给定反照率。 | `int` | `1` | `1` |
| `snow_depth_minimum_m` | 应用积雪地表属性所需的最小雪深。 | `float` | `0.01` | `0.01` |
| `landuse_plot_output` | 可选的 MAKEGEO 优势土地利用类别 Surfer ASCII 网格。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaluse.grd` |
| `terrain_plot_output` | 可选的 MAKEGEO 地形高程 Surfer ASCII 网格。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/qaterr.grd` |
| `color_map_output` | 可选的土地利用绘图网格 Surfer 配色文件。 | `str \| Path \| None` | `None` | `output/zhejiang/luse.clr` |
| `image_mode` | 质量检查 Surfer 网格的坐标约定：格点中心边界 (0) 或格点外缘边界 (1)。 | `int` | `0` | `0` |
| `title` | GEO 元数据标题；传统格式最多 80 个字符。 | `str` | `'PyCALMET Application'` | `'Demo Application'` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 笛卡尔 X/东西方向网格数；TERREL 极坐标网格中表示环数。 | `int \| None` | `必填` | `610` |
| `ny` | 笛卡尔 Y/南北方向网格数；TERREL 极坐标网格中表示射线数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 笛卡尔 X 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 笛卡尔 Y 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影代码：UTM、横轴墨卡托 (TTM)、兰伯特正形圆锥 (LCC)、极射赤面 (PS)、赤道墨卡托 (EM) 或兰伯特方位等积 (LAZA)。 | `str` | `'UTM'` | `'LCC'` |
| `lat0` | 投影原点或切点的带符号纬度，单位度（北纬为正）；EM 固定为 0。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点的带符号经度，单位度（东经为正）。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 笛卡尔网格单元 (1,1) 西南角的东向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 笛卡尔网格单元 (1,1) 西南角的北向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 输出网格的大地基准/区域代码；各输入源另行指定其输入基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL 网格类型：1 原点为单元 (1,1) 西南角；2 原点为该单元中心；3 原点为极坐标环中心。 | `int` | `1` | `1` |
| `ring_distances_km` | TERREL 极坐标网格各环半径，单位 km；grid_type=3 时每个环提供一个值。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL 极坐标网格射线方位角，从正北顺时针计，单位度；grid_type=3 时使用。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 仅供 TERREL 使用的极坐标方法：1 NORMAL，扇区边界位于相邻环/射线中间；2 SCREEN，径向从当前环延至下一外环，角向边界位于相邻射线中间。 | `int` | `2` | `2` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>

<details>
<summary><strong>CALMET</strong> — 共 165 个参数；浙江控制文件：`examples/zhejiang/config/calmet.inp`。</summary>

| 参数 | 含义 | Python 类型 | Python 默认值 | 浙江示例值 |
| --- | --- | --- | --- | --- |
| `meteorology` | 格点预报气象输入；iprog 决定其作为第一步风场、初始猜测场还是伪观测。 | `str \| Path \| Sequence[str \| Path] \| None` | `必填` | `output/zhejiang/calwrf.nc` |
| `geophysical` | GEO 地理物理网格，包含地形、土地利用、粗糙度、反照率、Bowen 比、土壤/人为热参数和叶面积指数。 | `str \| Path \| None` | `必填` | `output/zhejiang/geo.nc` |
| `output` | 主输出文件路径。 | `str \| Path \| None` | `必填` | `output/zhejiang/calmet.nc` |
| `start` | CALMET 输出区间 `[start, end)` 的闭合下界；实现选择满足 `start < t <= end` 的气象边界时刻，并将输出时间戳记为 `t - timestep_seconds`。 | `str \| datetime \| None` | `必填` | `'2018-03-18_00:00:00'` |
| `end` | CALMET 输出区间 `[start, end)` 的开放上界；实现选择满足 `start < t <= end` 的气象边界时刻，并将输出时间戳记为 `t - timestep_seconds`。 | `str \| datetime \| None` | `必填` | `'2018-03-18_01:00:00'` |
| `noobs` | 观测模式：0 使用地面、水面和高空观测；1 用预报资料替代高空观测；2 所有高度均使用预报资料。 | `int` | `0` | `2` |
| `run_type` | 运行类型：0 仅计算风场；1 还计算微气象量，CALPUFF/CALGRID 要求使用 1。 | `int` | `1` | `1` |
| `save_output` | 是否写出由 legacy_output_format 选择的格点气象磁盘输出。 | `bool` | `True` | `True` |
| `legacy_output_format` | 传统磁盘输出类型：1 写出供 CALPUFF/CALGRID 使用的 CALMET.DAT；2 写出 MESOPUFF-II 的 PACOUT.DAT。 | `int` | `1` | `1` |
| `test_mode` | 1 在输入设置检查后停止；2 继续执行模型计算。 | `int` | `2` | `2` |
| `regulatory_mode` | 法规检查模式：0 关闭；1 强制采用 USEPA 要求的混合层、水面通量、松弛和代表温度选项。 | `int` | `0` | `0` |
| `timestep_seconds` | CALMET 计算时间步长，必须能整除 1 小时。 | `int` | `3600` | `3600` |
| `prognostic_timestep_seconds` | 相邻预报记录的秒数：必须为 timestep_seconds 的正整数倍且不超过 12 小时；传统 IPROG<13 要求 3600 s。 | `int` | `3600` | `3600` |
| `cloud_method` | 云场来源：1 地面观测（NOOBS 0/1），2 已有 CLOUD.DAT，3 预报场 850 hPa 相对湿度，4 预报场所有高度的相对湿度。 | `int` | `4` | `4` |
| `cloud_data` | CLOUD.DAT 路径：cloud_method=2 时作为输入，cloud_output=1 时作为输出目标。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/cloud.dat` |
| `cloud_data_format` | CLOUD.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `cloud_output` | 是否把内部生成的格点云场写为 CLOUD.DAT（0=否，1=是）；cloud_method=2 读取已有文件时不能同时输出。 | `int` | `0` | `0` |
| `surface_data` | 逐小时 SURF.DAT 观测；NOOBS 为 0/1 时使用，除非相应变量改由 DIAG.DAT 提供。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/surf.dat` |
| `surface_data_format` | SURF.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `surface_control_stations` | 模型坐标下的 `(4字符站名, 站号, x_km, y_km, 时区, 风速仪高度_m)`；陆水分开插值时这里只列陆地站。 | `Sequence[tuple[str, int, float, float, float, float]]` | `()` | `()` |
| `upper_air_files` | 每个高空站一个探空文件；NOOBS=0 时使用，除非高空风改由 DIAG.DAT 提供。 | `Sequence[str \| Path]` | `()` | `()` |
| `upper_control_stations` | 高空站记录 `(4字符站名, 站号, x_km, y_km, 时区)`，每个高空文件对应一条。 | `Sequence[tuple[str, int, float, float, float]]` | `()` | `()` |
| `sea_files` | 每个水面站一个文件；可包含水面 ΔT、递减率、观测混合层高度，以及负 COARE 波浪代码所需的观测波浪。 | `Sequence[str \| Path]` | `()` | `()` |
| `diagnostic_data` | 逐小时预处理 DIAG.DAT 温度、递减率或 U/V 风；相应 diagnostic_* 来源选项设为 1 时需要。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/diag.dat` |
| `diagnostic_surface_temperature` | 诊断风模块的地表温度来源：内部计算 (0)，或从 DIAG.DAT 读取逐小时预处理值 (1)。 | `int` | `0` | `0` |
| `diagnostic_lapse_rate` | 地形环流所用区域温度递减率来源：由高空观测/预报资料内部计算 (0)，或读取 DIAG.DAT 逐小时值 (1)。 | `int` | `0` | `0` |
| `diagnostic_initial_wind` | 诊断模块初始猜测风来源：由高空/预报资料计算 (0)，或从 DIAG.DAT 读取逐小时预处理 U/V (1)。 | `int` | `0` | `0` |
| `diagnostic_surface_wind` | 地面风来源：SURF.DAT 风速风向 (0)，或预处理 DIAG.DAT U/V (1)；传统校验仅在只计算风场时允许选项 1。 | `int` | `0` | `0` |
| `diagnostic_upper_wind` | 高空风来源：探空风速风向 (0)，或预处理 DIAG.DAT U/V (1)；传统校验仅在只计算风场时允许选项 1。 | `int` | `0` | `0` |
| `initial_guess_upper_station` | 内部计算初始风时，1..NUSTA 选择一个探空站生成均匀猜测场；-1 对全部探空站作 1/R² 加权，生成空间变化猜测场。 | `int` | `-1` | `-1` |
| `initial_guess_height_range_m` | 生成诊断模块区域均匀初始风时采用的底部与顶部高度，单位 m。 | `Sequence[float]` | `(1.0, 1000.0)` | `(1.0, 1000.0)` |
| `use_initial_guess_calmet` | 是否使用粗网格 CALMET 风场作为初始猜测（0=否，1=是），并取代预报初始猜测；与第一步 IPROG 1/3/13 不兼容。 | `int` | `0` | `0` |
| `initial_guess_calmet_files` | use_initial_guess_calmet=1 时作为初始风猜测场的粗网格 CALMET 文件。 | `Sequence[str \| Path]` | `()` | `()` |
| `weight_data` | WT.DAT 三维观测权重 Wo；预报场使用互补权重 1−Wo。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/wt.dat` |
| `precipitation_station_count` | 降水来源：正数为 PRECIP.DAT 站点数，0 不使用降水输入，-1 使用预报降水。 | `int` | `0` | `-1` |
| `precipitation_data` | PRECIP.DAT 逐小时站点降水输入；precipitation_station_count 为正数时使用。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/precip.dat` |
| `precipitation_data_format` | PRECIP.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。 | `int` | `2` | `2` |
| `precipitation_control_stations` | 降水站记录 `(4字符站名, 6位站码, x_km, y_km)`；传统站码由 2 位州码和 4 位站号组成。 | `Sequence[tuple[str, int, float, float]]` | `()` | `()` |
| `iprog` | 预报风用法：0 关闭；1/3/13 作第一步风场（IWFCOD=0）；2/4/14 作初始猜测（IWFCOD=1）；5/15 作伪观测。各代码族对应 CSUMM、MM4 或 MM5/3D。 | `int` | `0` | `14` |
| `irhprog` | 相对湿度来源：0 使用 SURF/SEA 观测（NOOBS 0/1）；1 使用预报相对湿度（NOOBS=2 时必需）。 | `int` | `0` | `1` |
| `itprog` | 温度来源：0 地面及高空观测（NOOBS=0）；1 地面观测加高层预报；2 地面及高层均用预报资料。 | `int` | `0` | `2` |
| `imixh` | 混合层方法：1 陆水均用 Maul–Carson；-1 陆地 Maul–Carson、水面 OCD；2 陆水均用 Batchvarova–Gryning；-2 陆地 BG、水面 OCD。 | `int` | `1` | `1` |
| `constb` | Venkatram 中性机械混合层高度关系中的无量纲 B，控制摩擦速度相对于科里奥利参数和自由大气稳定度的尺度。 | `float` | `1.41` | `1.41` |
| `conste` | 修正 Maul–Carson 对流增长方程中的无量纲夹卷常数 E；用于 `abs(imixh)==1`。 | `float` | `0.15` | `0.15` |
| `constn` | 稳定机械混合层高度关系 `h = CONSTN × u*^(3/2)` 中的系数，并与另一稳定度上限共同判断。 | `float` | `2400.0` | `2400.0` |
| `constw` | 水面机械混合层公式 `h = CONSTW × u* / abs(f)` 中的无量纲常数；SEA.DAT 提供观测混合层高度时不使用。 | `float` | `0.16` | `0.16` |
| `coriolis` | 机械混合层高度方程使用的科里奥利参数绝对值 `abs(f)`，单位 s⁻¹；999 表示由纬度计算。 | `float` | `999.0` | `0.0001` |
| `minimum_lapse_rate` | 上一时步对流混合层顶以上稳定层位温梯度下限，单位 K/m。 | `float` | `0.001` | `0.001` |
| `mixing_lapse_depth_m` | 在上一时步对流混合层顶以上计算稳定位温梯度所用的层厚，单位 m。 | `float` | `200.0` | `200.0` |
| `minimum_mixing_height_m` | 陆地混合层高度下限，单位 m。 | `float` | `50.0` | `50.0` |
| `maximum_mixing_height_m` | 陆地混合层高度上限，单位 m。 | `float` | `3000.0` | `2500.0` |
| `minimum_water_mixing_height_m` | 水面混合层高度下限，单位 m；SEA.DAT 提供观测高度时不使用。 | `float` | `50.0` | `50.0` |
| `maximum_water_mixing_height_m` | 水面混合层高度上限，单位 m；SEA.DAT 提供观测高度时不使用。 | `float` | `3000.0` | `2500.0` |
| `land_flux_threshold` | 陆地单位边界层深度热通量阈值，单位 W/m³；QH 大于阈值×高度时才维持对流增长。 | `float` | `0.0` | `0.0` |
| `water_flux_threshold` | 水面单位边界层深度热通量阈值，单位 W/m³；QH 大于阈值×高度时才维持对流增长。 | `float` | `0.05` | `0.05` |
| `water_temperature_mode` | 水面输入：0 使用 SEA.DAT 递减率和 ΔT；1 使用预报递减率加 SEA.DAT ΔT（IPROG>2）；2 均用预报递减率和 ΔT（IPROG>12，3D.DAT v2+）。 | `int` | `0` | `0` |
| `prognostic_ocean_category` | 预报气象输入中代表海洋的土地利用类别。 | `int` | `16` | `16` |
| `overwater_flux_method` | 水面通量：0 传统 OCD；10 COARE Charnock/无波浪；±11 Oost；±12 Taylor–Yelland。负值读取观测波浪，正的 11/12 使用默认波浪。 | `int` | `10` | `10` |
| `coastal_shelf_length_km` | COARE 近岸/浅水粗糙度修正尺度，单位 km；0 表示禁用该修正。 | `float` | `0.0` | `0.0` |
| `warm_layer` | COARE 暖层修正（0=关，1=开）；红外辐射计测得的 SST 应关闭此项。 | `int` | `0` | `0` |
| `cool_skin` | COARE 冷皮层修正（0=关，1=开）；红外辐射计测得的 SST 应关闭此项。 | `int` | `0` | `0` |
| `convective_relaxation` | 在弱正通量区间 `0 < QH <= 阈值 × 高度` 内，使对流混合层高度向平衡值松弛（0=否，1=是）。 | `int` | `1` | `1` |
| `convective_relaxation_time_s` | 启用对流混合层松弛时的 e 折时间，单位秒；不得小于 1 s。 | `float` | `800.0` | `800.0` |
| `precipitation_interpolation` | 降水插值权重：1 为 1/R，2 为 1/R²，3 对不连续降水场使用类似 Thiessen 的 exp(-R²/σ²)/R²。 | `int` | `2` | `2` |
| `precipitation_radius_km` | 降水方法 1/2 中为站点影响半径（km）；方法 3 中为指数权重 σ（km）。方法 3 取 0 时，内部计算为湿站与干站最小间距的一半。 | `float` | `100.0` | `50.0` |
| `precipitation_cutoff` | 低于该阈值的降水率置零，单位 mm/h。 | `float` | `0.01` | `0.01` |
| `turbidity_coefficient_1` | 短波公式 `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)` 中乘太阳高度的 HA1，单位 W/m²。 | `float` | `990.0` | `990.0` |
| `turbidity_coefficient_2` | 短波公式 `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)` 中的晴空截距 HA2，单位 W/m²。 | `float` | `-30.0` | `-30.0` |
| `cloudiness_coefficient_1` | 短波云量因子 `(1+HB1·N^HB2)` 中的无量纲衰减幅度 HB1。 | `float` | `-0.75` | `-0.75` |
| `cloudiness_coefficient_2` | 短波参数化中施加于云量比例 N 的无量纲指数 HB2。 | `float` | `3.4` | `3.4` |
| `net_radiation_coefficient_1` | 净辐射方程中乘空气温度 T⁶ 的 HC1 系数，单位 W/(m²·K⁶)。 | `float` | `5.31e-13` | `5.31e-13` |
| `net_radiation_coefficient_2` | 净辐射方程中乘云量比例 N 的 HC2 系数，单位 W/m²。 | `float` | `60.0` | `60.0` |
| `net_radiation_coefficient_3` | 净辐射方程 `(…)/(1+HC3)` 中的无量纲分母修正 HC3。 | `float` | `0.12` | `0.12` |
| `surface_temperature_option` | diagnostic_surface_temperature=0 时：正数选择地面站，-1 使用空间变化场，-2 使用预报温度区域平均。 | `int` | `-1` | `-1` |
| `upper_temperature_option` | diagnostic_lapse_rate=0 时：正数选择高空站，-1 使用空间变化递减率，-2 使用预报递减率区域平均。 | `int` | `-1` | `-1` |
| `lapse_height_m` | 诊断地形环流所用区域温度递减率的计算层厚，单位 m。 | `float` | `200.0` | `200.0` |
| `temperature_interpolation_power` | 观测地面温度/相对湿度权重：1 使用 1/R，2 使用 1/R²；高空插值固定使用 1/R²。 | `int` | `1` | `1` |
| `temperature_radius_km` | 观测地面温度及相对湿度插值的影响半径，单位 km。 | `float` | `500.0` | `20.0` |
| `temperature_station_limit` | 观测温度及相对湿度插值最多使用的地面站数。 | `int` | `5` | `5` |
| `water_temperature_gradient_below` | SEA.DAT 未提供时，水面混合层以下采用的后备温度梯度，单位 K/m。 | `float` | `-0.0098` | `-0.0098` |
| `water_temperature_gradient_above` | SEA.DAT 未提供时，水面混合层以上采用的后备温度梯度，单位 K/m。 | `float` | `-0.0045` | `-0.0045` |
| `water_temperature_category_min` | 单独进行水面温度/相对湿度插值的闭区间起始类别；与 max 一起设为高于所有类别可禁用。 | `int` | `999` | `55` |
| `water_temperature_category_max` | 单独进行水面温度/相对湿度插值的闭区间结束类别；与 min 一起设为高于所有类别可禁用。 | `int` | `999` | `55` |
| `iavet` | 是否使用混合层平均的搜索几何对插值温度进行空间平均。 | `int` | `1` | `1` |
| `iavezi` | 是否对有效及对流混合层高度作上风向空间平均（0=否，1=是）。 | `int` | `1` | `1` |
| `mixing_wind_level` | 确定上风向平均锥方向的风场层，索引从 1 开始（1..NZ）。 | `int` | `1` | `1` |
| `mnmdav` | 混合层高度和温度平均时与上风向搜索锥配合使用的补充正方形搜索框半宽，单位为网格单元。 | `int` | `1` | `10` |
| `iwfcod` | 风场模型：0 仅作客观分析；1 先作地形诊断修正，再作客观分析。 | `int` | `1` | `1` |
| `wind_extrapolation` | 地面风垂直外推：代码绝对值 1 不外推、2 幂律、3 用户逐层系数、4 相似理论；负值还会忽略高空站第一层资料。 | `int` | `-4` | `-1` |
| `upper_air_separation_cells` | 地面站到高空站的邻近阈值，单位为网格间距；阈值内的地面风不作垂直外推，-1 取消该限制。 | `float` | `4.0` | `-1.0` |
| `extrapolate_calm` | 地面站报告静风时是否仍进行垂直外推。 | `int` | `0` | `0` |
| `wind_bias` | 逐层站点权重偏差，范围 -1 到 1：负值降低高空站权重，正值降低地面站权重，0 保持 1/R² 权重不变；None 为每层生成 0。 | `Sequence[float] \| None` | `None → (0.0,) * NZ` | `(0.0, 0.0, … (10 items))` |
| `varying_radius` | 相应最大半径内没有有效且未被障碍线隔开的站点时，使用半径外最近的有效站点。 | `bool` | `False` | `True` |
| `minimum_station_radius_km` | 风场插值距离的最小下限，单位 km；用于避免 1/R² 权重出现除零。 | `float` | `0.1` | `0.1` |
| `maximum_surface_radius_km` | 陆地近地层站点最大影响半径，单位 km；应反映地形对站点代表范围的限制。 | `float` | `30.0` | `30.0` |
| `maximum_upper_radius_km` | 陆地高层站点最大影响半径，单位 km；地形影响随高度减弱，通常大于近地层半径。 | `float` | `30.0` | `30.0` |
| `maximum_water_radius_km` | 水面站点最大影响半径，单位 km；水面上所有垂直层都使用该值。 | `float` | `50.0` | `50.0` |
| `surface_background_radius_km` | 近地层观测与初始猜测场权重相等时的距离尺度。 | `float` | `1.0` | `1.0` |
| `upper_background_radius_km` | 高层观测与初始猜测场权重相等时的距离尺度。 | `float` | `1.0` | `1.0` |
| `prognostic_radius_km` | 预报第一步风场的权重距离，单位 km；IPROG 为 1/3/13 时不得为零。 | `float` | `0.0` | `0.0` |
| `stations_per_level` | 每层在一个格点最多使用的最近站点数；None 按 Fortran 默认值为每层生成 99。 | `Sequence[int] \| None` | `None → (99,) * NZ` | `(5, 5, … (10 items))` |
| `wind_extrapolation_factors` | 每层一个乘法外推系数，仅在 `abs(wind_extrapolation)==3` 时使用；None 为每层生成 0，且必须提供的第 1 层值实际不使用。 | `Sequence[float] \| None` | `None → (0.0,) * NZ` | `(0.0, 0.0, … (10 items))` |
| `barrier_count` | 阻挡风场插值的线状障碍物数量。 | `int` | `0` | `0` |
| `barrier_top_level` | 站点插值障碍线作用的最高层（从 1 开始，1..NZ）；None 表示 NZ，仅 barrier_count>0 时使用。 | `int \| None` | `None → NZ` | `10` |
| `barriers` | 站点风插值障碍线，记录为 `(起点x, 起点y, 终点x, 终点y)`，单位 km。 | `Sequence[tuple[float, float, float, float]]` | `()` | `()` |
| `lake_breeze_regions` | 区域记录包含网格线边界、岸线端点（km）和指定站号；仅用这些站按相对岸线的 1/R² 权重重新计算风场。 | `Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]]` | `()` | `()` |
| `ifradj` | 是否计算基于弗劳德数的地形气流修正（0=否，1=是）；仅 iwfcod=1 时有效。 | `int` | `1` | `1` |
| `ikine` | 是否计算运动学地形效应（0=否，1=是）；仅 iwfcod=1 时有效。 | `int` | `0` | `0` |
| `iobr` | 是否应用 O’Brien 方法，把模型顶层垂直速度强制为零，再最小化 U/V 散度以恢复质量一致性（0=否，1=是）。 | `int` | `0` | `0` |
| `islope` | 是否计算坡面流参数化（0=否，1=是）。 | `int` | `1` | `1` |
| `critical_froude` | 启用 Froude 修正时，低于该临界值的上坡风会在速度不变的情况下转为沿地形切向；达到或超过时越过地形。 | `float` | `1.0` | `1.0` |
| `smoothing_passes` | 逐层风场平滑次数；None 按 Fortran 默认值生成第 1 层 2 次、其余各层 4 次。 | `Sequence[int] \| None` | `None → (2,) + (4,) * (NZ - 1)` | `[2, 4, … (10 items)]` |
| `divergence_limit` | 风场散度最小化过程的收敛目标。 | `float` | `5e-06` | `5e-06` |
| `divergence_iterations` | 散度最小化允许的最大迭代次数。 | `int` | `50` | `50` |
| `kinematic_alpha` | 运动学修正中施加于地形诱导垂直速度垂直梯度的经验乘数。 | `float` | `0.1` | `0.1` |
| `z_faces_m` | 垂直网格界面高度，单位 m；需提供 NZ+1 个值，第 k 层中心为 `(face[k] + face[k+1]) / 2`。 | `Sequence[float]` | `(0.0, 20.0, … (11 items))` | `[0.0, 20.0, … (11 items)]` |
| `base_timezone` | CALMET 采用的基准时区偏移，按 UTC 以西小时数计（例如 5=EST；东半球为负）。 | `int` | `0` | `0` |
| `base_timezone_label` | 与 base_timezone 表示同一偏移量的现代 `UTC±HHMM` 标签。 | `str \| None` | `None` | `'UTC-0000'` |
| `terrain_radius_km` | 搜索附近地形极值并确定 Froude 修正有效障碍高度的半径，单位 km。 | `float` | `15.0` | `15.0` |
| `half_average_angle` | 上风向空间平均锥的半张角，单位度；完整张角为该值的两倍。 | `float` | `30.0` | `30.0` |
| `include_w_temperature` | 是否存储 CALGRID 以及 CALPUFF 亚网格复杂地形选项所需的三维垂直风速和温度场。 | `bool` | `True` | `False` |
| `compression` | Python CALMET 输出的 NetCDF deflate 压缩级别（0 表示不压缩）。 | `int` | `1` | `1` |
| `control_lines` | 写入传统输出元数据的原始 CALMET 控制文件文本。 | `Sequence[str]` | `<0 control lines>` | `<1106 control lines>` |
| `mesopac_output` | save_output 为真且 legacy_output_format=2 时使用的 PACOUT.DAT 输出路径。 | `str \| Path \| None` | `None` | `None` |
| `distance_to_coast_output` | 距陆地/海岸诊断网格 DCST.GRD 的输出路径。 | `str \| Path \| None` | `None` | `None` |
| `listing_output` | CALMET 列表/文本报告输出路径，包含所选打印场和调试信息。 | `str \| Path \| None` | `None` | `output/zhejiang/calmet.lst` |
| `print_fields` | 每隔 print_interval_hours 把选定格点气象场写入 listing_output。 | `bool` | `False` | `False` |
| `print_interval_hours` | 气象场文本输出的时间间隔，单位小时。 | `int` | `1` | `1` |
| `print_uv_levels` | 启用 print_fields 时选择输出 U/V 风场层的逐层开关。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_w_levels` | W 在顶部界面 2..NZ+1 的 NZ 个 0/1 开关（地面界面 W=0）；要求启用 print_fields 和 include_w_temperature。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_temperature_levels` | 三维温度层的 NZ 个 0/1 开关；要求启用 print_fields 和 include_w_temperature。 | `Sequence[int]` | `()` | `(0, 0, … (10 items))` |
| `print_surface_fields` | 八个 0/1 文本输出开关，依次为 PGT 稳定度、摩擦速度、Monin–Obukhov 长度、混合层高度、对流速度尺度、降水、感热通量、对流混合层高度。 | `Sequence[int]` | `(0, 0, 0, 0, 0, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0, 0, 0)` |
| `micrometeorology_debug` | 是否打印微气象输入和内部变量；会产生大量输出。 | `bool` | `False` | `False` |
| `debug_first_step` | 微气象调试输出包含的首个时步。 | `int` | `1` | `1` |
| `debug_last_step` | 微气象调试输出包含的末个时步。 | `int` | `1` | `1` |
| `wind_debug_output` | 写出 TEST.PRT/OUT/KIN/FRD/SLP 风场诊断文件的总开关（0=关，1=开）。 | `int` | `0` | `0` |
| `wind_debug_levels` | 从地面开始写入风场诊断文件的层数。 | `int` | `1` | `0` |
| `wind_debug_fields` | 九个 0/1 开关（IPR0–IPR8），依次控制插值风、地形修正风、平滑风/初始散度、最终风速风向、最终散度、运动学修正后、Froude 修正后、坡面流修正后和最终风分量场。 | `Sequence[int]` | `(0, 0, … (9 items))` | `(0, 0, … (9 items))` |
| `final_wind_debug_output` | 最终风场诊断文件 TEST.OUT 的输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.out` |
| `kinematic_wind_debug_output` | 运动学修正后风场诊断文件 TEST.KIN 的输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.kin` |
| `froude_wind_debug_output` | Froude 数修正后风场诊断文件 TEST.FRD 的输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.frd` |
| `slope_wind_debug_output` | 坡面流修正后风场诊断文件 TEST.SLP 的输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.slp` |
| `wind_print_output` | 风场模块通用诊断变量文件 TEST.PRT 的输出路径。 | `str \| Path \| None` | `None` | `examples/zhejiang/config/test.prt` |
| `grid` | 完整网格对象，可替代下方展开的网格参数。 | `GridConfig \| Mapping[str, Any] \| None` | `None` | `见下方展开的网格参数` |
| `nx` | 笛卡尔 X/东西方向网格数；TERREL 极坐标网格中表示环数。 | `int \| None` | `必填` | `610` |
| `ny` | 笛卡尔 Y/南北方向网格数；TERREL 极坐标网格中表示射线数。 | `int \| None` | `必填` | `499` |
| `dx_km` | 笛卡尔 X 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `dy_km` | 笛卡尔 Y 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。 | `float \| None` | `必填` | `1.0` |
| `projection` | 投影代码：UTM、横轴墨卡托 (TTM)、兰伯特正形圆锥 (LCC)、极射赤面 (PS)、赤道墨卡托 (EM) 或兰伯特方位等积 (LAZA)。 | `str` | `'UTM'` | `'LCC'` |
| `lat0` | 投影原点或切点的带符号纬度，单位度（北纬为正）；EM 固定为 0。 | `float` | `27.0` | `27.0` |
| `lon0` | 中央经线或切点的带符号经度，单位度（东经为正）。 | `float` | `117.5` | `117.5` |
| `lat1` | LCC 第一标准纬线，或 PS 真比例纬线。 | `float` | `30.0` | `30.0` |
| `lat2` | LCC 第二标准纬线；PS 不使用。 | `float` | `60.0` | `60.0` |
| `x_origin_km` | 笛卡尔网格单元 (1,1) 西南角的东向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `y_origin_km` | 笛卡尔网格单元 (1,1) 西南角的北向坐标；TERREL 极坐标网格中为环中心，单位 km。 | `float` | `0.0` | `0.0` |
| `false_easting_km` | 投影原点假东距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `false_northing_km` | 投影原点假北距；用于 TTM、LCC 和 LAZA。 | `float` | `0.0` | `0.0` |
| `utm_zone` | UTM 分区号 1–60；仅 UTM 投影使用。 | `int` | `50` | `50` |
| `hemisphere` | UTM 半球，N 或 S；仅 UTM 投影使用。 | `str` | `'N'` | `'N'` |
| `datum` | 输出网格的大地基准/区域代码；各输入源另行指定其输入基准。 | `str` | `'WGS-84'` | `'WGS-84'` |
| `grid_type` | TERREL 网格类型：1 原点为单元 (1,1) 西南角；2 原点为该单元中心；3 原点为极坐标环中心。 | `int` | `1` | `1` |
| `ring_distances_km` | TERREL 极坐标网格各环半径，单位 km；grid_type=3 时每个环提供一个值。 | `Sequence[float]` | `()` | `()` |
| `ray_angles_deg` | TERREL 极坐标网格射线方位角，从正北顺时针计，单位度；grid_type=3 时使用。 | `Sequence[float]` | `()` | `()` |
| `polar_processing` | 仅供 TERREL 使用的极坐标方法：1 NORMAL，扇区边界位于相邻环/射线中间；2 SCREEN，径向从当前环延至下一外环，角向边界位于相邻射线中间。 | `int` | `2` | `2` |
| `ttm_scale` | TTM 投影中央经线比例因子。 | `float` | `1.0` | `1.0` |

</details>
<!-- parameter-reference:end -->

公开 API：

```python
pycalmet.run_pipeline(...)
pycalmet.PipelineConfig(...)
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
