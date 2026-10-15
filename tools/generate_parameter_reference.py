#!/usr/bin/env python3
"""Regenerate the complete public-parameter tables in both READMEs."""
from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
import argparse
import inspect
import os
from pathlib import Path
from typing import Any

import pycalmet
from pycalmet.config import GridConfig
from pycalmet.control import (
    calmet_config,
    calwrf_config,
    ctgproc_config,
    makegeo_config,
    terrel_config,
)

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- parameter-reference:start -->"
END = "<!-- parameter-reference:end -->"
GRID_FIELDS = {item.name: item for item in fields(GridConfig)}
STAGES = (
    ("CALWRF", "calwrf", pycalmet.CalwrfConfig, calwrf_config, "calwrf.inp"),
    ("TERREL", "terrel", pycalmet.TerrelConfig, terrel_config, "terrel.inp"),
    ("CTRPROC / CTGPROC", "ctrproc", pycalmet.CtgprocConfig, ctgproc_config, "ctgproc.inp"),
    ("MAKEGEO", "makegeo", pycalmet.MakegeoConfig, makegeo_config, "makegeo.inp"),
    ("CALMET", "calmet", pycalmet.CalmetConfig, calmet_config, "calmet.inp"),
)

# Names that need more context than a readable expansion of snake_case.  The
# remaining descriptive API names are rendered by _meaning() below.
MEANING_OVERRIDES = {
    "wrf_files": ("WRF NetCDF files, in chronological order.", "按时间顺序排列的 WRF NetCDF 文件。"),
    "start": ("First timestamp included in the run.", "计算开始时刻。"),
    "end": ("Last timestamp included in the run.", "计算结束时刻。"),
    "output": ("Primary output file path.", "主输出文件路径。"),
    "srtm_dir": ("Directory scanned for raw SRTM `[NS]dd[EW]ddd.hgt` tiles when terrain_sources is empty.", "terrain_sources 为空时扫描原始 SRTM `[NS]dd[EW]ddd.hgt` 瓦片的目录。"),
    "create_surface": ("Create the separate legacy M2D surface file when legacy_output is also requested.", "同时请求 legacy_output 时，是否另外生成传统 M2D 地面气象文件。"),
    "legacy_file_rain_reset": ("Reset accumulated rain at each legacy input-file boundary.", "是否在每个传统输入文件边界重置累计降水。"),
    "legacy_output": ("Optional traditional CALWRF 3D.DAT/M3D output generated alongside the primary NetCDF file.", "可选的传统 CALWRF 3D.DAT/M3D 输出，与主 NetCDF 文件同时生成。"),
    "surface_output": ("Optional traditional CALWRF two-dimensional surface/M2D path, written when create_surface is true and legacy_output is requested.", "可选的传统 CALWRF 二维地面/M2D 路径；create_surface 为真且同时请求 legacy_output 时写出。"),
    "grid": ("Complete grid object; an alternative to the flattened grid parameters.", "完整网格对象，可替代下方展开的网格参数。"),
    "i_range": ("Inclusive one-based WRF west-east index range.", "WRF 东西向索引范围，采用从 1 开始的闭区间。"),
    "j_range": ("Inclusive one-based WRF south-north index range.", "WRF 南北向索引范围，采用从 1 开始的闭区间。"),
    "k_range": ("Inclusive one-based WRF vertical-level range.", "WRF 垂直层索引范围，采用从 1 开始的闭区间。"),
    "sampling": ("Terrain coordinate conversion: `direct` transforms every DEM sample (more accurate); `bilinear`/`legacy` transforms sheet corners and interpolates interior coordinates.", "地形坐标转换方法：`direct` 逐个转换 DEM 样本，精度较高；`bilinear`/`legacy` 只转换图幅角点，再插值内部坐标。"),
    "void_fill": ("Fill empty output terrain cells from nearby valid cells, searching no farther than cell_radius_km.", "是否用邻近有效格点填补空白地形输出格点，最大搜索距离为 cell_radius_km。"),
    "receptor_columns": ("Discrete-receptor input width: 2 for (X, Y), or 4 for (X, Y, terrain elevation, receptor height); TERREL replaces column 3.", "离散接收点输入列数：2 表示 (X, Y)，4 表示 (X, Y, 地形高程, 接收点高度)；TERREL 会替换第 3 列。"),
    "interpolate_receptors": ("Discrete-receptor method: interpolate elevation when true; search for the peak elevation when false.", "离散接收点算法：为真时插值高程，为假时搜索最高高程。"),
    "terrain_sources": ("Terrain-source records `(path, input_type, input_datum)`; input_type selects the DEM reader (for example SRTM1/3, GTOPO30, CDED, GEN, or GEOTIFF).", "地形源记录 `(路径, 输入类型, 输入基准)`；输入类型选择 DEM 读取器，例如 SRTM1/3、GTOPO30、CDED、GEN 或 GEOTIFF。"),
    "terrain_files": ("Selected terrain-file paths; relative SRTM tile names are resolved inside srtm_dir.", "选定的地形文件路径；相对 SRTM 瓦片名会在 srtm_dir 内解析。"),
    "receptor_input": ("Optional discrete-receptor XY input file; supplying it enables receptor processing.", "可选的离散接收点 XY 输入文件；提供该文件即启用接收点处理。"),
    "receptor_output": ("Output file containing the receptor rows with TERREL-calculated terrain elevation.", "写出接收点记录及 TERREL 计算地形高程的文件。"),
    "previous_save": ("TERREL binary accumulator from an earlier pass, used to resume sample sums and hit counts.", "先前一轮 TERREL 的二进制累积文件，用于继续累计样本总和与命中数。"),
    "coastline_file": ("Processed BLN coastline input; when gshhs_file is supplied, this path receives the generated BLN data.", "处理后的 BLN 岸线输入；提供 gshhs_file 时，该路径接收生成的 BLN 数据。"),
    "gshhs_file": ("Raw GSHHS coastline database used to generate the processed BLN coastline.", "用于生成处理后 BLN 岸线的原始 GSHHS 岸线数据库。"),
    "coastline_source_datum": ("Datum/region code assigned to coordinates read from the raw GSHHS/coastline source.", "原始 GSHHS/岸线源坐标采用的大地基准/区域代码。"),
    "coastline_wdbii_datum": ("Datum/region code assigned to WDBII coastline coordinates.", "WDBII 岸线坐标采用的大地基准/区域代码。"),
    "plot_output": ("Optional Surfer ASCII grid containing TERREL terrain elevations for QA/plotting.", "可选的 Surfer ASCII 网格，写出 TERREL 地形高程供质检或绘图。"),
    "raw_echo_output": ("Optional ASCII dump of raw unfiltered terrain samples as `(x_km, y_km, elevation_m)`.", "可选的原始未过滤地形样本 ASCII 文件，记录 `(x_km, y_km, elevation_m)`。"),
    "landuse_sources": ("Land-use-source records `(path, input_type, input_datum, mesh_density)`.", "土地利用源记录 `(路径, 输入类型, 输入基准, 网格细分密度)`。"),
    "landuse_files": ("Input land-use database file paths consumed by CTGPROC.", "CTGPROC 读取的土地利用数据库文件路径。"),
    "previous_landuse": ("Prior CTGPROC integer hit-count output to continue a multi-pass land-use accumulation.", "用于继续多轮土地利用累计的先前 CTGPROC 整数命中数输出。"),
    "qa_cell_output": ("Write CTGPROC QA coordinate files for original and meshed input points; output may be large.", "是否写出 CTGPROC 原始及细分输入点的质检坐标文件；输出可能很大。"),
    "qa_output_directory": ("Directory receiving CTGPROC QACTG/QAGLAZ/QAMESH coordinate QA files.", "接收 CTGPROC QACTG/QAGLAZ/QAMESH 坐标质检文件的目录。"),
    "category_map": ("Post-binning map from source to target land-use categories; hit counts mapped to the same target are summed.", "分箱后应用的源土地利用类别到目标类别映射；映射到同一目标的命中数会相加。"),
    "input_properties": ("Per-category `(roughness_m, albedo, Bowen_ratio, soil_heat_flux_parameter, anthropogenic_heat_W_m-2, leaf_area_index, output_category)` records; roughness uses logarithmic area weighting and the other properties arithmetic weighting.", "各类别的 `(粗糙度_m, 反照率, Bowen 比, 土壤热通量参数, 人为热_W/m², 叶面积指数, 输出类别)`；粗糙度按面积作对数加权，其余属性作算术加权。"),
    "snow_properties": ("Snow-covered records with the same seven fields as input_properties; the mapped output category must match the normal record.", "积雪条件属性，七个字段与 input_properties 相同；映射后的输出类别必须与正常属性记录一致。"),
    "snow_age_parameters": ("Per snow class `(age_time_scale_days, albedo_reduction_factor)` used by the ageing-albedo method.", "各积雪类别的 `(老化时间尺度_天, 反照率衰减系数)`，供雪龄反照率算法使用。"),
    "category_splits": ("MAKEGEO `(source_category, target_category, percent)` records applied before property weighting; percentages for a source should total 100.", "MAKEGEO 在属性加权前应用的 `(源类别, 目标类别, 百分数)` 记录；同一源类别的百分数应合计为 100。"),
    "terrain": ("Gridded elevation input, normally TERREL output; read only when use_terrain is true.", "格点高程输入，通常为 TERREL 输出；仅在 use_terrain 为真时读取。"),
    "landuse": ("Primary final fractional land-use data set produced by CTGPROC.", "CTGPROC 生成的主要最终土地利用比例数据集。"),
    "secondary_landuse": ("Optional secondary land-use data used only where the primary data set has missing cells.", "可选的第二土地利用数据，仅用于补充主数据集中的缺失格点。"),
    "snow_data": ("SNODAS snow-depth/age input used to build daily snow-adjusted geophysical files.", "用于生成逐日积雪修正地理物理文件的 SNODAS 雪深/雪龄输入。"),
    "landuse_plot_output": ("Optional Surfer ASCII grid of the dominant MAKEGEO land-use category.", "可选的 MAKEGEO 优势土地利用类别 Surfer ASCII 网格。"),
    "terrain_plot_output": ("Optional Surfer ASCII grid of MAKEGEO terrain elevation.", "可选的 MAKEGEO 地形高程 Surfer ASCII 网格。"),
    "color_map_output": ("Optional Surfer colour-map companion for the land-use plot grid.", "可选的土地利用绘图网格 Surfer 配色文件。"),
    "surface_control_stations": ("Records `(4-char name, station ID, x_km, y_km, timezone, anemometer_height_m)` in model coordinates; list land stations only when land/water interpolation is separate.", "模型坐标下的 `(4字符站名, 站号, x_km, y_km, 时区, 风速仪高度_m)`；陆水分开插值时这里只列陆地站。"),
    "upper_control_stations": ("Records `(4-char name, station ID, x_km, y_km, timezone)`, one per upper-air file.", "高空站记录 `(4字符站名, 站号, x_km, y_km, 时区)`，每个高空文件对应一条。"),
    "precipitation_control_stations": ("Records `(4-char name, six-digit station code, x_km, y_km)`; the traditional code combines a two-digit state and four-digit station ID.", "降水站记录 `(4字符站名, 6位站码, x_km, y_km)`；传统站码由 2 位州码和 4 位站号组成。"),
    "noobs": ("Observation mode: prognostic-only or diagnostic use of observations.", "观测资料模式：仅预报场，或结合观测进行诊断。"),
    "run_type": ("Run type: 0 computes wind fields only; 1 also computes micrometeorology and is required by CALPUFF/CALGRID.", "运行类型：0 仅计算风场；1 还计算微气象量，CALPUFF/CALGRID 要求使用 1。"),
    "legacy_output_format": ("Traditional disk-output type: 1 writes CALPUFF/CALGRID CALMET.DAT; 2 writes MESOPUFF-II PACOUT.DAT.", "传统磁盘输出类型：1 写出供 CALPUFF/CALGRID 使用的 CALMET.DAT；2 写出 MESOPUFF-II 的 PACOUT.DAT。"),
    "cloud_output": ("Write the internally generated gridded cloud field as CLOUD.DAT (0=no, 1=yes); output is unavailable when cloud_method=2 reads an existing file.", "是否把内部生成的格点云场写为 CLOUD.DAT（0=否，1=是）；cloud_method=2 读取已有文件时不能同时输出。"),
    "wind_debug_output": ("Master switch for writing TEST.PRT/OUT/KIN/FRD/SLP wind diagnostic files (0=off, 1=on).", "写出 TEST.PRT/OUT/KIN/FRD/SLP 风场诊断文件的总开关（0=关，1=开）。"),
    "diagnostic_lapse_rate": ("Domain lapse-rate source for terrain circulations: compute internally from upper-air/prognostic data (0) or read hourly DIAG.DAT values (1).", "地形环流所用区域温度递减率来源：由高空观测/预报资料内部计算 (0)，或读取 DIAG.DAT 逐小时值 (1)。"),
    "minimum_lapse_rate": ("Floor on the stable-layer potential-temperature gradient above the previous convective mixing height, in K m⁻¹.", "上一时步对流混合层顶以上稳定层位温梯度下限，单位 K/m。"),
    "warm_layer": ("COARE warm-layer correction (0=off, 1=on); keep it off for infrared-radiometer SST.", "COARE 暖层修正（0=关，1=开）；红外辐射计测得的 SST 应关闭此项。"),
    "cool_skin": ("COARE cool-skin correction (0=off, 1=on); keep it off for infrared-radiometer SST.", "COARE 冷皮层修正（0=关，1=开）；红外辐射计测得的 SST 应关闭此项。"),
    "kinematic_alpha": ("Empirical multiplier applied to the vertical gradient of terrain-induced vertical velocity in the kinematic adjustment.", "运动学修正中施加于地形诱导垂直速度垂直梯度的经验乘数。"),
    "use_initial_guess_calmet": ("Use coarse CALMET fields as the initial guess (0=no, 1=yes), replacing any prognostic initial guess; incompatible with Step-1 IPROG 1/3/13.", "是否使用粗网格 CALMET 风场作为初始猜测（0=否，1=是），并取代预报初始猜测；与第一步 IPROG 1/3/13 不兼容。"),
    "iprog": ("Prognostic wind-field input and interpolation option.", "预报风场输入及插值方案控制。"),
    "irhprog": ("Relative-humidity source option for prognostic data.", "预报资料相对湿度来源选项。"),
    "itprog": ("Temperature source option for prognostic data.", "预报资料温度来源选项。"),
    "imixh": ("Mixing-height calculation method selector.", "混合层高度计算方法选项。"),
    "constb": ("Dimensionless B in the Venkatram neutral mechanical mixing-height relation, scaling friction velocity by Coriolis and free-atmosphere stability.", "Venkatram 中性机械混合层高度关系中的无量纲 B，控制摩擦速度相对于科里奥利参数和自由大气稳定度的尺度。"),
    "conste": ("Dimensionless entrainment constant E in the modified Maul–Carson convective-growth equation; used when `abs(imixh)==1`.", "修正 Maul–Carson 对流增长方程中的无量纲夹卷常数 E；用于 `abs(imixh)==1`。"),
    "constn": ("Coefficient in the stable mechanical mixing-height relation `h = CONSTN × u*^(3/2)`, evaluated with a second stability limit.", "稳定机械混合层高度关系 `h = CONSTN × u*^(3/2)` 中的系数，并与另一稳定度上限共同判断。"),
    "constw": ("Dimensionless overwater mechanical constant in `h = CONSTW × u* / abs(f)`; an observed SEA.DAT mixing height overrides it.", "水面机械混合层公式 `h = CONSTW × u* / abs(f)` 中的无量纲常数；SEA.DAT 提供观测混合层高度时不使用。"),
    "iavet": ("Temperature-field vertical averaging option.", "温度场垂直平均选项。"),
    "iavezi": ("Mixing-height wind averaging option.", "混合层高度所用风场平均选项。"),
    "mnmdav": ("Minimum number of valid values required for averaging.", "执行平均所需的最少有效值数量。"),
    "iwfcod": ("Wind-field model formulation code.", "风场模型计算方案代码。"),
    "ifradj": ("Compute Froude-number terrain-flow adjustment (0=no, 1=yes); active only with iwfcod=1.", "是否计算基于弗劳德数的地形气流修正（0=否，1=是）；仅 iwfcod=1 时有效。"),
    "ikine": ("Compute kinematic terrain effects (0=no, 1=yes); active only with iwfcod=1.", "是否计算运动学地形效应（0=否，1=是）；仅 iwfcod=1 时有效。"),
    "iobr": ("Apply the O'Brien procedure that forces vertical velocity to zero at the model top, then divergence-minimizes U/V for mass consistency (0=no, 1=yes).", "是否应用 O’Brien 方法，把模型顶层垂直速度强制为零，再最小化 U/V 散度以恢复质量一致性（0=否，1=是）。"),
    "islope": ("Compute slope-flow parameterization (0=no, 1=yes).", "是否计算坡面流参数化（0=否，1=是）。"),
    "z_faces_m": ("Vertical cell-face heights in metres; supply NZ+1 values, and layer k is centred at `(face[k] + face[k+1]) / 2`.", "垂直网格界面高度，单位 m；需提供 NZ+1 个值，第 k 层中心为 `(face[k] + face[k+1]) / 2`。"),
    "wind_bias": ("Per-level additive wind-direction or component bias controls.", "逐层风向或风分量加性偏差控制值。"),
    "stations_per_level": ("Per-layer maximum number of closest stations used at a grid point; None derives the Fortran default of 99 for every layer.", "每层在一个格点最多使用的最近站点数；None 按 Fortran 默认值为每层生成 99。"),
    "wind_extrapolation_factors": ("One multiplicative factor per layer, used only when `abs(wind_extrapolation)==3`; None derives zero for every layer, and the required layer-1 value is unused.", "每层一个乘法外推系数，仅在 `abs(wind_extrapolation)==3` 时使用；None 为每层生成 0，且必须提供的第 1 层值实际不使用。"),
    "smoothing_passes": ("Wind-field smoothing passes by level; None derives the Fortran defaults 2 for layer 1 and 4 for every higher layer.", "逐层风场平滑次数；None 按 Fortran 默认值生成第 1 层 2 次、其余各层 4 次。"),
    "barriers": ("Station-wind interpolation barriers as `(x_start, y_start, x_end, y_end)` line segments in kilometres.", "站点风插值障碍线，记录为 `(起点x, 起点y, 终点x, 终点y)`，单位 km。"),
    "lake_breeze_regions": ("Region records containing grid-line bounds, coastline endpoints in kilometres, and assigned station IDs; winds are recomputed from only those stations using coast-relative 1/R² weighting.", "区域记录包含网格线边界、岸线端点（km）和指定站号；仅用这些站按相对岸线的 1/R² 权重重新计算风场。"),
    "control_lines": ("Original CALMET control-file lines copied into legacy output metadata.", "写入传统输出元数据的原始 CALMET 控制文件文本。"),
    "print_surface_fields": ("Eight 0/1 print switches in this order: PGT stability, friction velocity, Monin-Obukhov length, mixing height, convective velocity scale, precipitation, sensible heat flux, convective mixing height.", "八个 0/1 文本输出开关，依次为 PGT 稳定度、摩擦速度、Monin–Obukhov 长度、混合层高度、对流速度尺度、降水、感热通量、对流混合层高度。"),
    "wind_debug_fields": ("Nine 0/1 switches (IPR0–IPR8): interpolated, terrain-adjusted, smoothed/initial-divergence, final speed/direction, final divergence, post-kinematic, post-Froude, post-slope, and final-component fields.", "九个 0/1 开关（IPR0–IPR8），依次控制插值风、地形修正风、平滑风/初始散度、最终风速风向、最终散度、运动学修正后、Froude 修正后、坡面流修正后和最终风分量场。"),
    "ring_distances_km": ("TERREL polar-grid ring radii, in kilometres; supply one value per ring when grid_type=3.", "TERREL 极坐标网格各环半径，单位 km；grid_type=3 时每个环提供一个值。"),
    "ray_angles_deg": ("TERREL polar-grid ray azimuths, in degrees clockwise from north; used when grid_type=3.", "TERREL 极坐标网格射线方位角，从正北顺时针计，单位度；grid_type=3 时使用。"),
}

MEANING_OVERRIDES.update({
    # Projection and grid definitions shared by TERREL, CTGPROC, MAKEGEO, and CALMET.
    "nx": ("Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings.", "笛卡尔 X/东西方向网格数；TERREL 极坐标网格中表示环数。"),
    "ny": ("Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of rays.", "笛卡尔 Y/南北方向网格数；TERREL 极坐标网格中表示射线数。"),
    "dx_km": ("Cartesian X grid spacing in kilometres; TERREL requires equal X/Y spacing and ignores it for polar ring positions.", "笛卡尔 X 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。"),
    "dy_km": ("Cartesian Y grid spacing in kilometres; TERREL requires equal X/Y spacing and ignores it for polar ring positions.", "笛卡尔 Y 方向网格间距，单位 km；TERREL 要求 X/Y 间距相等，极坐标环位置不使用该值。"),
    "projection": ("Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert azimuthal equal-area (LAZA).", "投影代码：UTM、横轴墨卡托 (TTM)、兰伯特正形圆锥 (LCC)、极射赤面 (PS)、赤道墨卡托 (EM) 或兰伯特方位等积 (LAZA)。"),
    "lat0": ("Signed latitude of the projection origin or tangent point, in degrees (north positive); EM fixes it at zero.", "投影原点或切点的带符号纬度，单位度（北纬为正）；EM 固定为 0。"),
    "lon0": ("Signed central-meridian or tangent-point longitude, in degrees (east positive).", "中央经线或切点的带符号经度，单位度（东经为正）。"),
    "lat1": ("First standard latitude for LCC, or true-scale latitude for PS.", "LCC 第一标准纬线，或 PS 真比例纬线。"),
    "lat2": ("Second standard latitude for LCC; unused by PS.", "LCC 第二标准纬线；PS 不使用。"),
    "x_origin_km": ("Easting of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres.", "笛卡尔网格单元 (1,1) 西南角的东向坐标；TERREL 极坐标网格中为环中心，单位 km。"),
    "y_origin_km": ("Northing of the south-west corner of Cartesian cell (1,1), or the TERREL polar-ring centre, in kilometres.", "笛卡尔网格单元 (1,1) 西南角的北向坐标；TERREL 极坐标网格中为环中心，单位 km。"),
    "false_easting_km": ("False easting at the projection origin; used by TTM, LCC, and LAZA.", "投影原点假东距；用于 TTM、LCC 和 LAZA。"),
    "false_northing_km": ("False northing at the projection origin; used by TTM, LCC, and LAZA.", "投影原点假北距；用于 TTM、LCC 和 LAZA。"),
    "utm_zone": ("UTM zone number from 1 through 60; used only for UTM.", "UTM 分区号 1–60；仅 UTM 投影使用。"),
    "hemisphere": ("UTM hemisphere, N or S; used only for UTM.", "UTM 半球，N 或 S；仅 UTM 投影使用。"),
    "datum": ("Output-grid datum/region code; individual input sources specify their own input datum separately.", "输出网格的大地基准/区域代码；各输入源另行指定其输入基准。"),
    "grid_type": ("TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2 Cartesian origin at its centre; 3 polar origin at ring centre.", "TERREL 网格类型：1 原点为单元 (1,1) 西南角；2 原点为该单元中心；3 原点为极坐标环中心。"),
    "polar_processing": ("TERREL-only polar method: 1 NORMAL sectors lie halfway between adjacent rings/rays; 2 SCREEN sectors span the current to next outer ring and halfway between rays.", "仅供 TERREL 使用的极坐标方法：1 NORMAL，扇区边界位于相邻环/射线中间；2 SCREEN，径向从当前环延至下一外环，角向边界位于相邻射线中间。"),
    "ttm_scale": ("Scale factor on the central meridian for TTM projection.", "TTM 投影中央经线比例因子。"),
    # TERREL and CTGPROC controls.
    "cell_radius_km": ("Maximum search radius, in kilometres, used to find valid neighbours when void_fill is enabled; use at least one grid spacing.", "启用 void_fill 时查找有效邻格的最大半径，单位 km；应至少为一个网格间距。"),
    "processing_method": ("TERREL processing method; for polar grids, 1 selects NORMAL sectors and 2 SCREEN sectors, overriding grid.polar_processing.", "TERREL 处理方法；极坐标网格中 1 选择 NORMAL 扇区，2 选择 SCREEN 扇区，并覆盖 grid.polar_processing。"),
    "model": ("TERREL output model: CALMET (1), MESOPAC (2), ISC polar (3), ISC Cartesian (4), NUATMOS (5), or generic (6); this also selects cell-average versus peak elevations.", "TERREL 输出模型：CALMET (1)、MESOPAC (2)、ISC 极坐标 (3)、ISC 笛卡尔 (4)、NUATMOS (5) 或通用格式 (6)；同时决定输出格点平均高程还是峰值高程。"),
    "receptor_radius_km": ("Search/interpolation radius for discrete-receptor elevation, in kilometres; peak-search mode requires a positive value.", "离散接收点高程的搜索/插值半径，单位 km；最高值搜索模式要求为正数。"),
    "noise_replacement": ("Per coastline class noise rule: 0 no test; 1 values below noise_minimum_m become missing; 2 clamp them to the minimum; 3 replace them with default_elevation_m.", "各岸线类别的噪声处理：0 不检查；1 把低于 noise_minimum_m 的值置为缺失；2 截到最小值；3 替换为 default_elevation_m。"),
    "noise_minimum_m": ("Minimum valid terrain elevation used by the noise tests, by database class.", "各数据库类别用于噪声检查的最小有效高程。"),
    "missing_replacement": ("Per coastline class void rule: 0 retain; 1 replace only final empty cells; 2 replace missing samples and final cells; 3 force the class to default elevation.", "各岸线类别的空值处理：0 保留；1 仅替换最终空格点；2 替换缺失样本及最终空格点；3 强制该类别使用默认高程。"),
    "default_elevation_m": ("Fallback elevation used when terrain samples remain missing.", "地形样本仍缺失时采用的后备高程。"),
    "qa_hit_threshold_percent": ("Flag cells whose sample count is below this percentage (0–100) of the domain-average hit count.", "标记采样数低于区域平均命中数这一百分比（0–100）的格点。"),
    "mesh_density": ("Per-side subdivision count for each input land-use cell, producing mesh_density² sample points; must be at least 1.", "每个输入土地利用单元在每条边上的细分数，共生成 mesh_density² 个采样点；不得小于 1。"),
    "input_type": ("Land-use database reader name/code, such as CTG, GEN, NLCD, CORINE, GLC2K, UMDGLC, or MODIS.", "土地利用数据库读取器名称/代码，例如 CTG、GEN、NLCD、CORINE、GLC2K、UMDGLC 或 MODIS。"),
    "input_datum": ("Datum of the input land-use coordinates.", "输入土地利用坐标采用的大地基准。"),
    "output_format": ("Output encoding: choose from the filename (auto), traditional CTGPROC format (legacy), or NetCDF.", "输出编码：按文件名判断 (auto)、传统 CTGPROC 格式 (legacy)，或 NetCDF。"),
    "coastline_swap": ("While reading samples, replace each point lying offshore of the supplied coastline with ocean_category.", "读取样本时，把位于所给岸线外海侧的点替换为 ocean_category。"),
    "fill_empty_ocean": ("On the final pass, fill zero-hit cells whose centre lies offshore with ocean_category; None follows the Fortran default and enables this only when coastline_file is supplied.", "最终处理时，把中心位于海上且命中数为零的格点填为 ocean_category；None 沿用 Fortran 默认逻辑，仅在提供 coastline_file 时启用。"),
    "ocean_category": ("Land-use category assigned to ocean cells.", "赋给海洋格点的土地利用类别。"),
    "final_output": ("CTGPROC pass type: true writes fractional land use for MAKEGEO; false writes integer hit counts for continuation by previous_landuse.", "CTGPROC 处理阶段：为真时写出供 MAKEGEO 使用的土地利用比例；为假时写出整数命中数，供 previous_landuse 继续处理。"),
    # MAKEGEO controls.
    "water_threshold": ("Water-category fraction required before dominant land use is selected only among water categories; otherwise selection is among non-water categories.", "水体类别比例达到此阈值后，优势类别只从水体类别中选择；否则只从非水体类别中选择。"),
    "missing_category": ("Category inserted when total usable land-use coverage is below 0.001; it must exist in input_properties.", "土地利用有效总覆盖率低于 0.001 时插入的类别；该类别必须存在于 input_properties。"),
    "minimum_coverage": ("Minimum acceptable total land-use fraction: values at or above it are renormalized, values below 0.001 are missing, and intermediate values are errors.", "可接受的土地利用总比例下限：达到该值则重新归一化，低于 0.001 视为缺失，介于两者之间则报错。"),
    "use_terrain": ("Read terrain elevations into GEO output when true; write zero terrain when false.", "为真时读取地形高程写入 GEO；为假时地形高程写为零。"),
    "water_categories": ("Input land-use categories treated as water.", "被视为水体的输入土地利用类别。"),
    "output_categories": ("Ordered land-use categories retained in the geophysical output.", "地理物理输出中保留的有序土地利用类别。"),
    "output_water_min": ("Lowest output category regarded as water by CALMET.", "CALMET 视为水体的最小输出类别。"),
    "output_water_max": ("Highest output category regarded as water by CALMET.", "CALMET 视为水体的最大输出类别。"),
    "qa_cell": ("Optional one-based `(i, j)` cell for reporting intermediate terrain, land-use, and surface-property calculations.", "可选的从 1 开始 `(i, j)` 格点，用于报告地形、土地利用和地表属性的中间计算。"),
    "snow_processing": ("Enable the stage's snow workflow: CTGPROC writes SNODAS reference records; MAKEGEO creates daily snow-adjusted GEO files.", "启用相应阶段的积雪流程：CTGPROC 写出 SNODAS 参考记录；MAKEGEO 生成逐日积雪修正 GEO 文件。"),
    "snow_start_date": ("First YYYYMMDD date covered by snow processing.", "积雪处理覆盖的首个日期，格式 YYYYMMDD。"),
    "snow_end_date": ("Last YYYYMMDD date covered by snow processing.", "积雪处理覆盖的末个日期，格式 YYYYMMDD。"),
    "snow_roughness_method": ("Snow roughness method: 1 reduces inferred obstacle height by snow depth; 2 uses the user snow roughness wherever snow is present.", "积雪粗糙度方法：1 从推算的障碍物高度中扣除雪深；2 有积雪时直接采用用户给定的积雪粗糙度。"),
    "snow_height_scale_m": ("HSCL obstacle-height-to-roughness scale used by snow roughness method 1.", "积雪粗糙度方法 1 使用的障碍物高度与粗糙度比例尺度 HSCL。"),
    "snow_albedo_method": ("Snow albedo method: 1 ages table albedo toward a reduced value; 2 uses user snow albedo wherever snow is present.", "积雪反照率方法：1 随雪龄把表中反照率衰减到较低值；2 有积雪时直接采用用户给定反照率。"),
    "snow_depth_minimum_m": ("Minimum snow depth required to apply snow properties.", "应用积雪地表属性所需的最小雪深。"),
    "image_mode": ("QA Surfer-grid coordinate convention: cell-centre bounds (0) or cell-edge bounds (1).", "质量检查 Surfer 网格的坐标约定：格点中心边界 (0) 或格点外缘边界 (1)。"),
    # General CALMET and meteorological-data controls.
    "meteorology": ("Gridded prognostic meteorology input; iprog selects whether it supplies a Step-1 field, initial guess, or pseudo-observations.", "格点预报气象输入；iprog 决定其作为第一步风场、初始猜测场还是伪观测。"),
    "geophysical": ("GEO geophysical grid containing terrain, land use, roughness, albedo, Bowen ratio, soil/anthropogenic heat parameters, and leaf-area index.", "GEO 地理物理网格，包含地形、土地利用、粗糙度、反照率、Bowen 比、土壤/人为热参数和叶面积指数。"),
    "cloud_data": ("CLOUD.DAT path: input when cloud_method=2, or output destination when cloud_output=1.", "CLOUD.DAT 路径：cloud_method=2 时作为输入，cloud_output=1 时作为输出目标。"),
    "surface_data": ("Hourly SURF.DAT observations, used in NOOBS modes 0/1 unless corresponding fields come from DIAG.DAT.", "逐小时 SURF.DAT 观测；NOOBS 为 0/1 时使用，除非相应变量改由 DIAG.DAT 提供。"),
    "upper_air_files": ("One sounding file per upper-air station, used in NOOBS=0 unless upper winds come from DIAG.DAT.", "每个高空站一个探空文件；NOOBS=0 时使用，除非高空风改由 DIAG.DAT 提供。"),
    "sea_files": ("One file per overwater station; may contain water ΔT, lapse rate, observed mixing height, and observed waves for negative COARE wave codes.", "每个水面站一个文件；可包含水面 ΔT、递减率、观测混合层高度，以及负 COARE 波浪代码所需的观测波浪。"),
    "diagnostic_data": ("Hourly preprocessed DIAG.DAT temperatures, lapse rates, or U/V winds, required for each diagnostic_* source option set to 1.", "逐小时预处理 DIAG.DAT 温度、递减率或 U/V 风；相应 diagnostic_* 来源选项设为 1 时需要。"),
    "initial_guess_calmet_files": ("Coarse-grid CALMET files used as the initial wind guess when use_initial_guess_calmet=1.", "use_initial_guess_calmet=1 时作为初始风猜测场的粗网格 CALMET 文件。"),
    "weight_data": ("WT.DAT three-dimensional observation weight Wo; the prognostic field receives complementary weight 1−Wo.", "WT.DAT 三维观测权重 Wo；预报场使用互补权重 1−Wo。"),
    "precipitation_data": ("PRECIP.DAT hourly station precipitation input, used when precipitation_station_count is positive.", "PRECIP.DAT 逐小时站点降水输入；precipitation_station_count 为正数时使用。"),
    "timestep_seconds": ("CALMET computational time step; it must divide evenly into one hour.", "CALMET 计算时间步长，必须能整除 1 小时。"),
    "prognostic_timestep_seconds": ("Seconds between prognostic records: a positive multiple of timestep_seconds, no more than 12 hours; legacy IPROG<13 requires 3600 s.", "相邻预报记录的秒数：必须为 timestep_seconds 的正整数倍且不超过 12 小时；传统 IPROG<13 要求 3600 s。"),
    "test_mode": ("1 stops after input setup checks; 2 continues into model computation.", "1 在输入设置检查后停止；2 继续执行模型计算。"),
    "regulatory_mode": ("Regulatory check mode: 0 disables it; 1 enforces the USEPA-required mixing-height, overwater-flux, relaxation, and representative-temperature choices.", "法规检查模式：0 关闭；1 强制采用 USEPA 要求的混合层、水面通量、松弛和代表温度选项。"),
    "include_w_temperature": ("Store 3-D vertical-wind and temperature fields required by CALGRID and CALPUFF's subgrid complex-terrain option.", "是否存储 CALGRID 以及 CALPUFF 亚网格复杂地形选项所需的三维垂直风速和温度场。"),
    "cloud_method": ("Cloud source: 1 surface observations (NOOBS 0/1), 2 existing CLOUD.DAT, 3 prognostic 850-hPa RH, or 4 prognostic RH at all levels.", "云场来源：1 地面观测（NOOBS 0/1），2 已有 CLOUD.DAT，3 预报场 850 hPa 相对湿度，4 预报场所有高度的相对湿度。"),
    "cloud_data_format": ("CLOUD.DAT format: unformatted binary (1) or formatted text (2).", "CLOUD.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "surface_data_format": ("SURF.DAT format: unformatted binary (1) or formatted text (2).", "SURF.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "precipitation_data_format": ("PRECIP.DAT format: unformatted binary (1) or formatted text (2).", "PRECIP.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "precipitation_station_count": ("Precipitation source: positive is the number of PRECIP.DAT stations, 0 disables precipitation input, and -1 uses prognostic precipitation.", "降水来源：正数为 PRECIP.DAT 站点数，0 不使用降水输入，-1 使用预报降水。"),
    # Wind-field options from CALMET input groups 4 and 5.
    "wind_extrapolation": ("Vertical surface-wind extrapolation: `abs(code)` 1 none, 2 power law, 3 user factors, 4 similarity theory; negative codes also ignore layer-1 upper-air data.", "地面风垂直外推：代码绝对值 1 不外推、2 幂律、3 用户逐层系数、4 相似理论；负值还会忽略高空站第一层资料。"),
    "extrapolate_calm": ("Extrapolate a surface observation even when it reports calm wind.", "地面站报告静风时是否仍进行垂直外推。"),
    "upper_air_separation_cells": ("Surface-to-upper-air proximity threshold in grid-cell spacings; a surface wind inside it is not extrapolated vertically, and -1 disables the exclusion.", "地面站到高空站的邻近阈值，单位为网格间距；阈值内的地面风不作垂直外推，-1 取消该限制。"),
    "varying_radius": ("When no valid unblocked station lies inside the relevant maximum radius, use the closest valid station outside it.", "相应最大半径内没有有效且未被障碍线隔开的站点时，使用半径外最近的有效站点。"),
    "minimum_station_radius_km": ("Small lower bound, in kilometres, on wind-interpolation distance; it prevents division by zero in 1/R² weighting.", "风场插值距离的最小下限，单位 km；用于避免 1/R² 权重出现除零。"),
    "maximum_surface_radius_km": ("Maximum land surface-layer station radius, in kilometres; choose it to reflect terrain-limited influence.", "陆地近地层站点最大影响半径，单位 km；应反映地形对站点代表范围的限制。"),
    "maximum_upper_radius_km": ("Maximum station radius aloft over land, in kilometres; it is generally larger than the surface radius as terrain influence decreases with height.", "陆地高层站点最大影响半径，单位 km；地形影响随高度减弱，通常大于近地层半径。"),
    "maximum_water_radius_km": ("Maximum overwater station radius, in kilometres, used at every vertical layer over water.", "水面站点最大影响半径，单位 km；水面上所有垂直层都使用该值。"),
    "surface_background_radius_km": ("Distance where surface observations and the first-guess field receive equal weight.", "近地层观测与初始猜测场权重相等时的距离尺度。"),
    "upper_background_radius_km": ("Distance where upper-air observations and the first-guess field receive equal weight.", "高层观测与初始猜测场权重相等时的距离尺度。"),
    "prognostic_radius_km": ("Prognostic Step-1 wind weighting distance, in kilometres; it must be nonzero for IPROG 1/3/13.", "预报第一步风场的权重距离，单位 km；IPROG 为 1/3/13 时不得为零。"),
    "barrier_count": ("Number of line barriers that block wind interpolation.", "阻挡风场插值的线状障碍物数量。"),
    "barrier_top_level": ("One-based highest layer (1..NZ) to which station-interpolation barriers apply; None selects NZ, and it is used only when barrier_count>0.", "站点插值障碍线作用的最高层（从 1 开始，1..NZ）；None 表示 NZ，仅 barrier_count>0 时使用。"),
    "critical_froude": ("With Froude adjustment enabled, uphill flow below this critical value is rotated tangent to terrain without changing speed; flow at or above it passes over.", "启用 Froude 修正时，低于该临界值的上坡风会在速度不变的情况下转为沿地形切向；达到或超过时越过地形。"),
    "divergence_limit": ("Convergence target for wind-field divergence minimization.", "风场散度最小化过程的收敛目标。"),
    "divergence_iterations": ("Maximum iterations allowed for divergence minimization.", "散度最小化允许的最大迭代次数。"),
    "terrain_radius_km": ("Terrain-search radius, in kilometres, used to determine nearby extrema and effective obstacle height for the Froude adjustment.", "搜索附近地形极值并确定 Froude 修正有效障碍高度的半径，单位 km。"),
    # Mixing height, temperature, humidity, precipitation, and overwater fluxes.
    "coriolis": ("Absolute Coriolis parameter `abs(f)` in s⁻¹ for mechanical mixing-height equations; 999 derives it from latitude.", "机械混合层高度方程使用的科里奥利参数绝对值 `abs(f)`，单位 s⁻¹；999 表示由纬度计算。"),
    "iavezi": ("Enable upwind spatial averaging of effective and convective mixing heights (0=no, 1=yes).", "是否对有效及对流混合层高度作上风向空间平均（0=否，1=是）。"),
    "mixing_wind_level": ("One-based wind layer (1..NZ) whose direction defines the upwind averaging cone.", "确定上风向平均锥方向的风场层，索引从 1 开始（1..NZ）。"),
    "half_average_angle": ("Half-opening angle of the upwind spatial-averaging cone, in degrees; the full opening is twice this value.", "上风向空间平均锥的半张角，单位度；完整张角为该值的两倍。"),
    "imixh": ("Mixing-height method: 1 Maul–Carson land/water; -1 Maul–Carson land plus OCD water; 2 Batchvarova–Gryning land/water; -2 BG land plus OCD water.", "混合层方法：1 陆水均用 Maul–Carson；-1 陆地 Maul–Carson、水面 OCD；2 陆水均用 Batchvarova–Gryning；-2 陆地 BG、水面 OCD。"),
    "minimum_mixing_height_m": ("Minimum overland mixing height, in metres.", "陆地混合层高度下限，单位 m。"),
    "maximum_mixing_height_m": ("Maximum overland mixing height, in metres.", "陆地混合层高度上限，单位 m。"),
    "minimum_water_mixing_height_m": ("Minimum overwater mixing height, in metres; unused where SEA.DAT supplies an observed height.", "水面混合层高度下限，单位 m；SEA.DAT 提供观测高度时不使用。"),
    "maximum_water_mixing_height_m": ("Maximum overwater mixing height, in metres; unused where SEA.DAT supplies an observed height.", "水面混合层高度上限，单位 m；SEA.DAT 提供观测高度时不使用。"),
    "land_flux_threshold": ("Land heat-flux-per-boundary-layer-depth threshold, in W m⁻³; convective growth requires QH greater than threshold × height.", "陆地单位边界层深度热通量阈值，单位 W/m³；QH 大于阈值×高度时才维持对流增长。"),
    "water_flux_threshold": ("Overwater heat-flux-per-boundary-layer-depth threshold, in W m⁻³; convective growth requires QH greater than threshold × height.", "水面单位边界层深度热通量阈值，单位 W/m³；QH 大于阈值×高度时才维持对流增长。"),
    "convective_relaxation": ("Relax convective mixing height toward equilibrium in the weak-positive-flux regime `0 < QH <= threshold × height` (0=no, 1=yes).", "在弱正通量区间 `0 < QH <= 阈值 × 高度` 内，使对流混合层高度向平衡值松弛（0=否，1=是）。"),
    "convective_relaxation_time_s": ("E-folding relaxation time in seconds when convective_relaxation is enabled; must be at least 1 s.", "启用对流混合层松弛时的 e 折时间，单位秒；不得小于 1 s。"),
    "water_temperature_mode": ("Overwater inputs: 0 SEA.DAT lapse rate and ΔT; 1 prognostic lapse rate plus SEA.DAT ΔT (IPROG>2); 2 prognostic lapse rate and ΔT (IPROG>12, 3D.DAT v2+).", "水面输入：0 使用 SEA.DAT 递减率和 ΔT；1 使用预报递减率加 SEA.DAT ΔT（IPROG>2）；2 均用预报递减率和 ΔT（IPROG>12，3D.DAT v2+）。"),
    "prognostic_ocean_category": ("Land-use category representing ocean in the prognostic meteorological input.", "预报气象输入中代表海洋的土地利用类别。"),
    "overwater_flux_method": ("Overwater flux: 0 legacy OCD; 10 COARE Charnock/no waves; ±11 Oost; ±12 Taylor-Yelland. Negative codes read observed waves; positive 11/12 use default waves.", "水面通量：0 传统 OCD；10 COARE Charnock/无波浪；±11 Oost；±12 Taylor–Yelland。负值读取观测波浪，正的 11/12 使用默认波浪。"),
    "coastal_shelf_length_km": ("COARE nearshore/shallow-water roughness length scale in kilometres; zero disables this modification.", "COARE 近岸/浅水粗糙度修正尺度，单位 km；0 表示禁用该修正。"),
    "irhprog": ("Relative-humidity source: 0 observed SURF/SEA data (NOOBS 0/1); 1 prognostic RH (required for NOOBS=2).", "相对湿度来源：0 使用 SURF/SEA 观测（NOOBS 0/1）；1 使用预报相对湿度（NOOBS=2 时必需）。"),
    "itprog": ("Temperature source: 0 surface and upper observations (NOOBS=0); 1 surface observations plus prognostic aloft; 2 prognostic surface and aloft.", "温度来源：0 地面及高空观测（NOOBS=0）；1 地面观测加高层预报；2 地面及高层均用预报资料。"),
    "temperature_interpolation_power": ("Observed surface temperature/RH weighting: 1 uses 1/R and 2 uses 1/R²; upper-air interpolation remains 1/R².", "观测地面温度/相对湿度权重：1 使用 1/R，2 使用 1/R²；高空插值固定使用 1/R²。"),
    "temperature_radius_km": ("Radius of influence, in kilometres, for observed surface temperature and relative-humidity interpolation.", "观测地面温度及相对湿度插值的影响半径，单位 km。"),
    "temperature_station_limit": ("Maximum number of surface stations used in observed temperature and relative-humidity interpolation.", "观测温度及相对湿度插值最多使用的地面站数。"),
    "iavet": ("Enable spatial averaging of interpolated temperatures using the mixing-height averaging geometry.", "是否使用混合层平均的搜索几何对插值温度进行空间平均。"),
    "water_temperature_gradient_below": ("Fallback water temperature gradient below mixing height, in K m⁻¹, when SEA.DAT does not provide one.", "SEA.DAT 未提供时，水面混合层以下采用的后备温度梯度，单位 K/m。"),
    "water_temperature_gradient_above": ("Fallback water temperature gradient above mixing height, in K m⁻¹, when SEA.DAT does not provide one.", "SEA.DAT 未提供时，水面混合层以上采用的后备温度梯度，单位 K/m。"),
    "water_temperature_category_min": ("First category in the inclusive range receiving separate overwater temperature/RH interpolation; set above all categories with max to disable.", "单独进行水面温度/相对湿度插值的闭区间起始类别；与 max 一起设为高于所有类别可禁用。"),
    "water_temperature_category_max": ("Last category in the inclusive range receiving separate overwater temperature/RH interpolation; set above all categories with min to disable.", "单独进行水面温度/相对湿度插值的闭区间结束类别；与 min 一起设为高于所有类别可禁用。"),
    "precipitation_interpolation": ("Precipitation weighting: 1 uses 1/R, 2 uses 1/R², and 3 uses a Thiessen-like exp(-R²/σ²)/R² treatment for discontinuous fields.", "降水插值权重：1 为 1/R，2 为 1/R²，3 对不连续降水场使用类似 Thiessen 的 exp(-R²/σ²)/R²。"),
    "precipitation_radius_km": ("For precipitation methods 1/2, station radius of influence (km); for method 3, exponential σ (km). Method 3 with zero derives half the minimum wet-to-dry-station distance.", "降水方法 1/2 中为站点影响半径（km）；方法 3 中为指数权重 σ（km）。方法 3 取 0 时，内部计算为湿站与干站最小间距的一半。"),
    "precipitation_cutoff": ("Rates below this threshold are set to zero, in mm h⁻¹.", "低于该阈值的降水率置零，单位 mm/h。"),
    # Diagnostic inputs and output/debug selection.
    "diagnostic_surface_temperature": ("Surface-temperature source for the diagnostic wind module: compute internally (0) or read hourly preprocessed values from DIAG.DAT (1).", "诊断风模块的地表温度来源：内部计算 (0)，或从 DIAG.DAT 读取逐小时预处理值 (1)。"),
    "diagnostic_initial_wind": ("Diagnostic-module initial-guess wind source: compute from upper/prognostic data (0) or read hourly preprocessed U/V from DIAG.DAT (1).", "诊断模块初始猜测风来源：由高空/预报资料计算 (0)，或从 DIAG.DAT 读取逐小时预处理 U/V (1)。"),
    "diagnostic_surface_wind": ("Surface-wind source: SURF.DAT speed/direction (0) or preprocessed DIAG.DAT U/V (1); legacy validation permits option 1 only for wind-only runs.", "地面风来源：SURF.DAT 风速风向 (0)，或预处理 DIAG.DAT U/V (1)；传统校验仅在只计算风场时允许选项 1。"),
    "diagnostic_upper_wind": ("Upper-wind source: sounding speed/direction (0) or preprocessed DIAG.DAT U/V (1); legacy validation permits option 1 only for wind-only runs.", "高空风来源：探空风速风向 (0)，或预处理 DIAG.DAT U/V (1)；传统校验仅在只计算风场时允许选项 1。"),
    "surface_temperature_option": ("When diagnostic_surface_temperature=0: positive selects a surface station, -1 a spatially varying field, and -2 the domain-mean prognostic temperature.", "diagnostic_surface_temperature=0 时：正数选择地面站，-1 使用空间变化场，-2 使用预报温度区域平均。"),
    "upper_temperature_option": ("When diagnostic_lapse_rate=0: positive selects an upper-air station, -1 a spatially varying lapse rate, and -2 the domain-mean prognostic lapse rate.", "diagnostic_lapse_rate=0 时：正数选择高空站，-1 使用空间变化递减率，-2 使用预报递减率区域平均。"),
    "lapse_height_m": ("Depth in metres used to calculate the domain-scale lapse rate for diagnostic terrain-induced circulations.", "诊断地形环流所用区域温度递减率的计算层厚，单位 m。"),
    "initial_guess_upper_station": ("With internally computed initial winds, 1..NUSTA selects one sounding for a uniform guess; -1 uses all soundings with 1/R² weighting for a spatially varying guess.", "内部计算初始风时，1..NUSTA 选择一个探空站生成均匀猜测场；-1 对全部探空站作 1/R² 加权，生成空间变化猜测场。"),
    "initial_guess_height_range_m": ("Bottom and top heights, in metres, used to form the diagnostic module's domain-uniform initial wind.", "生成诊断模块区域均匀初始风时采用的底部与顶部高度，单位 m。"),
    "base_timezone": ("Base time-zone offset in CALMET's hours-west-of-UTC convention (for example 5=EST; negative values are east).", "CALMET 采用的基准时区偏移，按 UTC 以西小时数计（例如 5=EST；东半球为负）。"),
    "base_timezone_label": ("Modern `UTC±HHMM` label for the same offset represented by base_timezone.", "与 base_timezone 表示同一偏移量的现代 `UTC±HHMM` 标签。"),
    "mesopac_output": ("PACOUT.DAT destination used when save_output is true and legacy_output_format=2.", "save_output 为真且 legacy_output_format=2 时使用的 PACOUT.DAT 输出路径。"),
    "distance_to_coast_output": ("DCST.GRD destination for the distance-to-land/coast diagnostic grid.", "距陆地/海岸诊断网格 DCST.GRD 的输出路径。"),
    "listing_output": ("CALMET listing/text report destination, including selected printed fields and debug information.", "CALMET 列表/文本报告输出路径，包含所选打印场和调试信息。"),
    "final_wind_debug_output": ("TEST.OUT destination for final wind diagnostic fields.", "最终风场诊断文件 TEST.OUT 的输出路径。"),
    "kinematic_wind_debug_output": ("TEST.KIN destination for wind fields after kinematic adjustment.", "运动学修正后风场诊断文件 TEST.KIN 的输出路径。"),
    "froude_wind_debug_output": ("TEST.FRD destination for wind fields after Froude-number adjustment.", "Froude 数修正后风场诊断文件 TEST.FRD 的输出路径。"),
    "slope_wind_debug_output": ("TEST.SLP destination for wind fields after slope-flow adjustment.", "坡面流修正后风场诊断文件 TEST.SLP 的输出路径。"),
    "wind_print_output": ("TEST.PRT destination for general wind-module diagnostic variables.", "风场模块通用诊断变量文件 TEST.PRT 的输出路径。"),
    "print_fields": ("Write selected gridded meteorological fields to listing_output every print_interval_hours.", "每隔 print_interval_hours 把选定格点气象场写入 listing_output。"),
    "print_interval_hours": ("Interval between printed meteorological fields, in hours.", "气象场文本输出的时间间隔，单位小时。"),
    "print_uv_levels": ("Per-layer switches selecting U/V wind output when print_fields is enabled.", "启用 print_fields 时选择输出 U/V 风场层的逐层开关。"),
    "print_w_levels": ("NZ 0/1 switches for W at top faces 2..NZ+1 (ground-face W is zero); requires print_fields and include_w_temperature.", "W 在顶部界面 2..NZ+1 的 NZ 个 0/1 开关（地面界面 W=0）；要求启用 print_fields 和 include_w_temperature。"),
    "print_temperature_levels": ("NZ 0/1 switches for three-dimensional temperature layers; requires print_fields and include_w_temperature.", "三维温度层的 NZ 个 0/1 开关；要求启用 print_fields 和 include_w_temperature。"),
    "micrometeorology_debug": ("Print micrometeorological inputs and internal variables; produces large output.", "是否打印微气象输入和内部变量；会产生大量输出。"),
    "debug_first_step": ("First time step included in micrometeorology debug output.", "微气象调试输出包含的首个时步。"),
    "debug_last_step": ("Last time step included in micrometeorology debug output.", "微气象调试输出包含的末个时步。"),
    "wind_debug_levels": ("Number of wind-field levels written to diagnostic files, starting at the surface.", "从地面开始写入风场诊断文件的层数。"),
    "noobs": ("Observation mode: 0 uses surface/overwater/upper-air observations; 1 replaces upper-air observations with prognostic data; 2 uses prognostic data at all levels.", "观测模式：0 使用地面、水面和高空观测；1 用预报资料替代高空观测；2 所有高度均使用预报资料。"),
    "iwfcod": ("Wind model: 0 objective analysis only; 1 full diagnostic module with terrain adjustments before objective analysis.", "风场模型：0 仅作客观分析；1 先作地形诊断修正，再作客观分析。"),
    "iprog": ("Prognostic-wind use: 0 off; 1/3/13 Step-1 field (IWFCOD=0); 2/4/14 initial guess (IWFCOD=1); 5/15 pseudo-observations. Code families select CSUMM, MM4, or MM5/3D.", "预报风用法：0 关闭；1/3/13 作第一步风场（IWFCOD=0）；2/4/14 作初始猜测（IWFCOD=1）；5/15 作伪观测。各代码族对应 CSUMM、MM4 或 MM5/3D。"),
    "mnmdav": ("Half-width, in grid cells, of the square supplementary search box used with the upwind cone for mixing-height and temperature averaging.", "混合层高度和温度平均时与上风向搜索锥配合使用的补充正方形搜索框半宽，单位为网格单元。"),
    "wind_bias": ("Per-layer station-weight bias from -1 to 1: negative reduces upper-air weight, positive reduces surface-station weight, and zero leaves 1/R² weights unchanged; None derives zero for every layer.", "逐层站点权重偏差，范围 -1 到 1：负值降低高空站权重，正值降低地面站权重，0 保持 1/R² 权重不变；None 为每层生成 0。"),
    "mixing_lapse_depth_m": ("Depth in metres above the previous-time-step convective mixing height used to calculate the stable potential-temperature gradient.", "在上一时步对流混合层顶以上计算稳定位温梯度所用的层厚，单位 m。"),
    "turbidity_coefficient_1": ("HA1 in `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`, multiplying solar elevation; units W m⁻².", "短波公式 `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)` 中乘太阳高度的 HA1，单位 W/m²。"),
    "turbidity_coefficient_2": ("HA2 clear-sky intercept in `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`; units W m⁻².", "短波公式 `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)` 中的晴空截距 HA2，单位 W/m²。"),
    "cloudiness_coefficient_1": ("Dimensionless HB1 cloud-attenuation amplitude in the shortwave factor `(1+HB1·N^HB2)`.", "短波云量因子 `(1+HB1·N^HB2)` 中的无量纲衰减幅度 HB1。"),
    "cloudiness_coefficient_2": ("Dimensionless HB2 exponent applied to cloud fraction N in the shortwave parameterization.", "短波参数化中施加于云量比例 N 的无量纲指数 HB2。"),
    "net_radiation_coefficient_1": ("HC1 coefficient multiplying air-temperature T⁶ in the net-radiation equation; units W m⁻² K⁻⁶.", "净辐射方程中乘空气温度 T⁶ 的 HC1 系数，单位 W/(m²·K⁶)。"),
    "net_radiation_coefficient_2": ("HC2 coefficient multiplying cloud fraction N in the net-radiation equation; units W m⁻².", "净辐射方程中乘云量比例 N 的 HC2 系数，单位 W/m²。"),
    "net_radiation_coefficient_3": ("Dimensionless HC3 denominator correction in the net-radiation equation `(…)/(1+HC3)`.", "净辐射方程 `(…)/(1+HC3)` 中的无量纲分母修正 HC3。"),
})

STAGE_MEANING_OVERRIDES = {
    ("calmet", "start"): ("Inclusive lower boundary of the CALMET output interval `[start, end)`; meteorological boundary times are selected with `start < t <= end`, and each output timestamp is `t - timestep_seconds`.", "CALMET 输出区间 `[start, end)` 的闭合下界；实现选择满足 `start < t <= end` 的气象边界时刻，并将输出时间戳记为 `t - timestep_seconds`。"),
    ("calmet", "end"): ("Exclusive upper boundary of the CALMET output interval `[start, end)`; meteorological boundary times are selected with `start < t <= end`, and each output timestamp is `t - timestep_seconds`.", "CALMET 输出区间 `[start, end)` 的开放上界；实现选择满足 `start < t <= end` 的气象边界时刻，并将输出时间戳记为 `t - timestep_seconds`。"),
    ("terrel", "save_output"): ("Write a TERREL binary accumulator containing sample sums and hit counts for a later continuation pass.", "是否写出包含样本总和及命中数的 TERREL 二进制累积文件，供后续轮次继续。"),
    ("calmet", "save_output"): ("Write the gridded meteorological disk output selected by legacy_output_format.", "是否写出由 legacy_output_format 选择的格点气象磁盘输出。"),
    ("ctgproc", "snow_processing"): ("Attach SNODAS-to-land-use-grid reference records needed by MAKEGEO snow processing; valid only on the final CTGPROC pass.", "附加 MAKEGEO 积雪处理所需的 SNODAS 到土地利用网格参考记录；仅能用于 CTGPROC 最终轮次。"),
    ("makegeo", "snow_processing"): ("Create one snow-adjusted GEO file per day from SNODAS depth/age data over the inclusive snow_start_date..snow_end_date range.", "根据 SNODAS 雪深/雪龄，在包含首尾日期的 snow_start_date..snow_end_date 范围内逐日生成积雪修正 GEO 文件。"),
    ("calwrf", "compression"): ("NetCDF deflate compression level for CALWRF output (0 disables compression).", "CALWRF 输出的 NetCDF deflate 压缩级别（0 表示不压缩）。"),
    ("calmet", "compression"): ("NetCDF deflate compression level for Python CALMET output (0 disables compression).", "Python CALMET 输出的 NetCDF deflate 压缩级别（0 表示不压缩）。"),
    ("calwrf", "title"): ("Descriptive title stored in CALWRF output metadata.", "写入 CALWRF 输出元数据的说明标题。"),
    ("calwrf", "listing_output"): ("CALWRF processing log/listing destination.", "CALWRF 处理日志/列表文件输出路径。"),
    ("makegeo", "title"): ("GEO metadata title, limited by the traditional format to 80 characters.", "GEO 元数据标题；传统格式最多 80 个字符。"),
}

CALMET_DYNAMIC_DEFAULTS = {
    "wind_bias": "None → (0.0,) * NZ",
    "stations_per_level": "None → (99,) * NZ",
    "wind_extrapolation_factors": "None → (0.0,) * NZ",
    "smoothing_passes": "None → (2,) + (4,) * (NZ - 1)",
    "barrier_top_level": "None → NZ",
}

STAGE_DYNAMIC_DEFAULTS = {
    ("ctrproc", "fill_empty_ocean"): "None → coastline_file is not None",
    ("ctgproc", "fill_empty_ocean"): "None → coastline_file is not None",
}

ZH_WORDS = {
    "minimum": "最小", "maximum": "最大", "surface": "地面", "upper": "高空",
    "water": "水体", "temperature": "温度", "wind": "风场", "cloud": "云量",
    "precipitation": "降水", "terrain": "地形", "landuse": "土地利用", "snow": "积雪",
    "output": "输出", "input": "输入", "data": "数据", "file": "文件", "files": "文件",
    "radius": "半径", "height": "高度", "depth": "深度", "distance": "距离",
    "coefficient": "系数", "option": "选项", "method": "方法", "format": "格式",
    "count": "数量", "level": "层", "levels": "层", "station": "站点", "stations": "站点",
    "category": "类别", "categories": "类别", "grid": "网格", "projection": "投影",
    "datum": "大地基准", "compression": "压缩级别", "interval": "间隔", "timestep": "时间步长",
    "debug": "调试", "print": "打印", "initial": "初始", "guess": "猜测场",
    "diagnostic": "诊断", "prognostic": "预报", "mixing": "混合层", "lapse": "递减率",
    "flux": "通量", "threshold": "阈值", "interpolation": "插值", "factor": "系数",
    "angle": "角度", "origin": "原点", "false": "假", "easting": "东距",
    "northing": "北距", "zone": "分区", "hemisphere": "半球", "title": "标题",
    "model": "模型", "processing": "处理", "source": "来源", "default": "默认",
    "replacement": "替换值", "noise": "噪声", "missing": "缺失值", "coverage": "覆盖率",
    "properties": "属性", "roughness": "粗糙度", "albedo": "反照率", "image": "图像",
    "mode": "模式", "create": "生成", "include": "包含", "varying": "可变",
    "save": "保存", "legacy": "传统格式", "listing": "列表文件", "control": "控制",
    "first": "首个", "last": "末个", "base": "基准", "timezone": "时区",
    "divergence": "散度", "iterations": "迭代次数", "critical": "临界", "froude": "弗劳德数",
    "extrapolation": "外推", "bias": "偏差", "warm": "暖层", "cool": "冷皮层",
    "regulatory": "法规", "test": "测试", "type": "类型", "density": "密度",
    "receptor": "接收点", "receptors": "接收点", "void": "空洞", "fill": "填补",
    "columns": "列数", "range": "范围", "air": "探空", "sea": "海面", "weight": "权重",
    "rate": "率", "land": "陆地", "ocean": "海洋", "overwater": "水面",
    "coastal": "沿海", "shelf": "陆架", "length": "长度", "layer": "层", "skin": "表皮层",
    "convective": "对流", "relaxation": "松弛", "time": "时间", "cutoff": "截断阈值",
    "turbidity": "浑浊度", "cloudiness": "云量", "net": "净", "radiation": "辐射",
    "coriolis": "科里奥利参数", "use": "使用", "interpolate": "插值", "reset": "重置",
    "rain": "降水", "raw": "原始", "echo": "回显", "previous": "先前", "final": "最终",
    "cell": "格点", "fraction": "比例", "power": "幂指数", "below": "下方", "above": "上方",
    "separation": "间距", "calm": "静风", "background": "背景场", "per": "每",
    "barrier": "障碍物", "top": "顶部", "kinematic": "运动学", "slope": "坡面流",
    "alpha": "Alpha", "faces": "界面", "half": "半", "average": "平均",
    "fields": "字段", "field": "字段", "meteorology": "气象", "step": "时步",
    "coastline": "岸线", "wdbii": "WDBII", "elevation": "高程", "polar": "极坐标",
    "scale": "比例", "swap": "交换", "empty": "空", "directory": "目录",
    "secondary": "第二套", "start": "开始", "date": "日期", "color": "颜色",
    "map": "映射", "geophysical": "地理物理", "calmet": "CALMET", "limit": "上限",
    "gradient": "梯度", "cells": "格点", "extrapolate": "外推", "label": "标签",
    "mesopac": "MESOPAC", "to": "到", "coast": "海岸", "hours": "小时",
    "micrometeorology": "微气象", "gshhs": "GSHHS",
}


def _meaning(name: str, language: str, parameter: inspect.Parameter,
             function_name: str) -> str:
    stage_meaning = STAGE_MEANING_OVERRIDES.get((function_name, name))
    if stage_meaning is not None:
        return stage_meaning[0 if language == "en" else 1]
    if name in MEANING_OVERRIDES:
        return MEANING_OVERRIDES[name][0 if language == "en" else 1]
    units = {"_km": "km", "_m": "m", "_s": "s", "_deg": "degrees",
             "_percent": "%", "_seconds": "s", "_hours": "hours"}
    unit = next((label for suffix, label in units.items() if name.endswith(suffix)), None)
    words = name
    for suffix in units:
        if words.endswith(suffix):
            words = words[:-len(suffix)]
            break
    annotation = _type(parameter)
    is_path = "Path" in annotation
    is_boolean = annotation == "bool"
    if language == "en":
        label = words.replace("_", " ")
        if is_path:
            text = f"Path for {label}."
        elif is_boolean:
            text = f"Whether to {label.replace('_', ' ')}."
        else:
            text = label.capitalize() + "."
        return text[:-1] + (f" ({unit})." if unit else ".")
    tokens = words.split("_")
    label = "".join(ZH_WORDS.get(token, token.upper() if len(token) <= 4 else token) for token in tokens)
    if is_path:
        text = label + "路径"
    elif is_boolean:
        text = "是否启用" + label
    else:
        text = label
    unit = {"degrees": "度", "hours": "小时"}.get(unit, unit)
    return text + (f"（单位：{unit}）。" if unit else "。")


def _field_defaults(config_type: type[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(config_type):
        if item.default is not MISSING:
            result[item.name] = item.default
        elif item.default_factory is not MISSING:
            result[item.name] = item.default_factory()
        else:
            result[item.name] = MISSING
    return result


def _relative_path(value: Path) -> str:
    normalized = Path(os.path.normpath(value))
    try:
        return normalized.relative_to(ROOT).as_posix()
    except ValueError:
        return normalized.as_posix()


def _value(value: Any, *, name: str = "") -> str:
    if value is MISSING:
        return "required"
    if value is None:
        return "None"
    if isinstance(value, Path):
        return _relative_path(value)
    if name == "control_lines":
        return f"<{len(value)} control lines>"
    if is_dataclass(value):
        parts = [f"{item.name}={_value(getattr(value, item.name), name=item.name)}"
                 for item in fields(value)]
        text = f"{type(value).__name__}({', '.join(parts)})"
    elif isinstance(value, dict):
        items = [f"{_value(key)}: {_value(item)}" for key, item in value.items()]
        text = "{" + ", ".join(items) + "}"
        if len(text) > 150:
            text = "{" + ", ".join(items[:2]) + f", … ({len(items)} items)" + "}"
    elif isinstance(value, (list, tuple)):
        opening, closing = ("[", "]") if isinstance(value, list) else ("(", ")")
        items = [_value(item) for item in value]
        full = opening + ", ".join(items) + closing
        if len(items) > 8 or len(full) > 150:
            text = opening + ", ".join(items[:2]) + f", … ({len(items)} items)" + closing
        else:
            text = full
    else:
        text = repr(value) if isinstance(value, str) else str(value)
    return text if len(text) <= 150 else text[:115] + "…" + text[-30:]


def _type(parameter: inspect.Parameter) -> str:
    annotation = parameter.annotation
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return text.replace("typing.", "").replace("|", "\\|")


def _table(label: str, function_name: str, config_type: type[Any], loader: Any,
           control_name: str, language: str) -> str:
    control = ROOT / "examples" / "zhejiang" / "config" / control_name
    example = loader(control)
    signature = inspect.signature(getattr(pycalmet, function_name))
    defaults = _field_defaults(config_type)
    if hasattr(example, "grid"):
        defaults.update({name: (item.default if item.default is not MISSING else MISSING)
                         for name, item in GRID_FIELDS.items()})
    headings = ("Parameter", "Meaning", "Python type", "Python default", "Zhejiang value") if language == "en" else (
        "参数", "含义", "Python 类型", "Python 默认值", "浙江示例值")
    rows = [f"| {' | '.join(headings)} |", "| --- | --- | --- | --- | --- |"]
    for name, parameter in signature.parameters.items():
        if name in {"inp", "legacy_parameters"}:
            continue
        if name == "grid":
            default = None
            actual = "see flattened grid fields" if language == "en" else "见下方展开的网格参数"
        else:
            default = defaults.get(name, parameter.default)
            if name in GRID_FIELDS and name not in {item.name for item in fields(config_type)}:
                actual = _value(getattr(example.grid, name), name=name)
            else:
                actual = _value(getattr(example, name), name=name)
        if (function_name, name) in STAGE_DYNAMIC_DEFAULTS:
            default_text = STAGE_DYNAMIC_DEFAULTS[function_name, name]
        elif function_name == "calmet" and name in CALMET_DYNAMIC_DEFAULTS:
            default_text = CALMET_DYNAMIC_DEFAULTS[name]
        else:
            default_text = _value(default, name=name)
        if default_text == "required":
            default_text = "required" if language == "en" else "必填"
        values = (f"`{name}`", _meaning(name, language, parameter, function_name), f"`{_type(parameter)}`",
                  f"`{default_text}`", f"`{actual}`")
        rows.append("| " + " | ".join(item.replace("\n", " ") for item in values) + " |")
    summary = (f"{len(rows) - 2} parameters; Zhejiang control: `examples/zhejiang/config/{control_name}`."
               if language == "en" else
               f"共 {len(rows) - 2} 个参数；浙江控制文件：`examples/zhejiang/config/{control_name}`。")
    return (f"<details>\n<summary><strong>{label}</strong> — {summary}</summary>\n\n"
            + "\n".join(rows) + "\n\n</details>")


def _section(language: str) -> str:
    if language == "en":
        intro = """## Complete parameter reference

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
with `python tools/generate_parameter_reference.py`; use `--check` in CI."""
    else:
        intro = """## 完整参数参考

以下表格由公开函数签名和类型化配置模型生成。“Python 默认值”表示 `inp=None` 模式下调用方
省略参数时采用的值；“浙江示例值”表示解析仓库内置控制文件后得到的实际值。`inp`
本身和 `**legacy_parameters` 是入口及兼容机制，不属于模型控制项，因此不在每张表中
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。参数含义和选项
效果主要依据 Scire、Robe、Fernau 和 Yamartino 的 *A User's Guide for the CALMET
Meteorological Model (Version 5)*（2000 年 1 月），尤其是第 2 章及 4.2–4.3 节。
该指南早于 NOOBS、新版预报资料开关、COARE 和积雪处理等控制项；这些较新控制项
再与仓库内 CALMET 6.5 控制模板、参考 Fortran 源码，以及实际读取参数的 Python
代码交叉核对。API 变化后运行
`python tools/generate_parameter_reference.py` 重新生成；CI 可使用 `--check` 检查。"""
    tables = [_table(*stage, language) for stage in STAGES]
    return START + "\n" + intro + "\n\n" + "\n\n".join(tables) + "\n" + END


def _updated(path: Path, section: str, anchor: str) -> str:
    text = path.read_text(encoding="utf-8")
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + section + after
    else:
        text = text.replace(anchor, section + "\n\n" + anchor, 1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail instead of updating stale README tables")
    arguments = parser.parse_args()
    documents = (
        (ROOT / "README.md", _section("en"), "The public processing API is:"),
        (ROOT / "README.zh-CN.md", _section("zh"), "公开 API："),
    )
    stale = []
    for path, section, anchor in documents:
        expected = _updated(path, section, anchor)
        if expected == path.read_text(encoding="utf-8"):
            continue
        if arguments.check:
            stale.append(path.name)
        else:
            path.write_text(expected, encoding="utf-8")
    if stale:
        parser.error("stale generated parameter tables: " + ", ".join(stale))


if __name__ == "__main__":
    main()
