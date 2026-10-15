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
    "srtm_dir": ("Directory containing raw SRTM elevation tiles.", "原始 SRTM 高程瓦片所在目录。"),
    "create_surface": ("Whether to create the separate legacy surface file.", "是否另外生成传统地面气象文件。"),
    "legacy_file_rain_reset": ("Reset accumulated rain at each legacy input-file boundary.", "是否在每个传统输入文件边界重置累计降水。"),
    "grid": ("Complete grid object; an alternative to the flattened grid parameters.", "完整网格对象，可替代下方展开的网格参数。"),
    "i_range": ("Inclusive one-based WRF west-east index range.", "WRF 东西向索引范围，采用从 1 开始的闭区间。"),
    "j_range": ("Inclusive one-based WRF south-north index range.", "WRF 南北向索引范围，采用从 1 开始的闭区间。"),
    "k_range": ("Inclusive one-based WRF vertical-level range.", "WRF 垂直层索引范围，采用从 1 开始的闭区间。"),
    "sampling": ("Terrain sampling method: direct or bilinear/legacy.", "地形采样方法：直接采样或双线性/传统采样。"),
    "void_fill": ("Whether to fill void cells in elevation inputs.", "是否填补高程输入中的空洞格点。"),
    "receptor_columns": ("Number of columns in the discrete-receptor input.", "离散接收点输入文件的列数。"),
    "interpolate_receptors": ("Whether to calculate terrain at discrete receptors.", "是否计算离散接收点处的地形高度。"),
    "terrain_sources": ("Terrain files with their format and source datum.", "带有格式和源基准信息的地形文件列表。"),
    "landuse_sources": ("Land-use files with format, datum, and mesh density.", "带有格式、基准和网格采样密度的土地利用文件列表。"),
    "category_map": ("Mapping from input land-use categories to output categories.", "输入土地利用类别到输出类别的映射。"),
    "input_properties": ("Surface-property tuple for each input land-use category.", "每个输入土地利用类别对应的地表属性元组。"),
    "snow_properties": ("Snow-covered surface-property tuple for each category.", "每个类别在积雪条件下对应的地表属性元组。"),
    "snow_age_parameters": ("Snow-age class and decay factor by category.", "各类别的雪龄分级和衰减因子。"),
    "category_splits": ("Input category, output category, and fractional split records.", "输入类别、输出类别及拆分比例记录。"),
    "surface_control_stations": ("Surface-station ID, number, coordinates, time zone, and anemometer height.", "地面站名称、编号、坐标、时区和风速仪高度记录。"),
    "upper_control_stations": ("Upper-air station ID, number, coordinates, and time zone.", "高空站名称、编号、坐标和时区记录。"),
    "precipitation_control_stations": ("Precipitation station ID, number, and coordinates.", "降水站名称、编号和坐标记录。"),
    "noobs": ("Observation mode: prognostic-only or diagnostic use of observations.", "观测资料模式：仅预报场，或结合观测进行诊断。"),
    "run_type": ("CALMET run type: wind-only or wind plus micrometeorology.", "CALMET 运行类型：仅风场，或风场加微气象场。"),
    "legacy_output_format": ("Traditional CALMET output format selector.", "传统 CALMET 输出格式选项。"),
    "cloud_output": ("Cloud-field output option.", "云场输出选项。"),
    "wind_debug_output": ("Wind diagnostic-output option.", "风场诊断输出选项。"),
    "diagnostic_lapse_rate": ("Use of observed lapse rates in diagnostic fields.", "诊断场是否使用观测温度递减率。"),
    "minimum_lapse_rate": ("Minimum potential-temperature lapse rate.", "最小位温递减率。"),
    "warm_layer": ("COARE warm-layer correction option.", "COARE 暖层修正选项。"),
    "cool_skin": ("COARE cool-skin correction option.", "COARE 冷皮层修正选项。"),
    "kinematic_alpha": ("Relaxation coefficient for the kinematic wind adjustment.", "运动学风场调整的松弛系数 Alpha。"),
    "use_initial_guess_calmet": ("Use a previous CALMET field as the initial wind guess.", "是否使用已有 CALMET 风场作为初始猜测场。"),
    "iprog": ("Prognostic wind-field input and interpolation option.", "预报风场输入及插值方案控制。"),
    "irhprog": ("Relative-humidity source option for prognostic data.", "预报资料相对湿度来源选项。"),
    "itprog": ("Temperature source option for prognostic data.", "预报资料温度来源选项。"),
    "imixh": ("Mixing-height calculation method selector.", "混合层高度计算方法选项。"),
    "constb": ("Neutral mechanical mixing-height coefficient B.", "中性机械混合层高度系数 B。"),
    "conste": ("Convective mixing-height entrainment coefficient.", "对流混合层夹卷系数。"),
    "constn": ("Stable mechanical mixing-height coefficient N.", "稳定机械混合层高度系数 N。"),
    "constw": ("Convective velocity-scale coefficient W.", "对流速度尺度系数 W。"),
    "iavet": ("Temperature-field vertical averaging option.", "温度场垂直平均选项。"),
    "iavezi": ("Mixing-height wind averaging option.", "混合层高度所用风场平均选项。"),
    "mnmdav": ("Minimum number of valid values required for averaging.", "执行平均所需的最少有效值数量。"),
    "iwfcod": ("Wind-field model formulation code.", "风场模型计算方案代码。"),
    "ifradj": ("Froude-number terrain-flow adjustment switch.", "基于弗劳德数的地形绕流调整开关。"),
    "ikine": ("Kinematic terrain-effect adjustment switch.", "运动学地形效应调整开关。"),
    "iobr": ("Blocking-effect adjustment switch.", "地形阻塞效应调整开关。"),
    "islope": ("Slope-flow adjustment switch.", "坡面流调整开关。"),
    "z_faces_m": ("Heights of vertical layer faces; one more value than model layers.", "垂直层界面高度，数量比模型层数多 1。"),
    "wind_bias": ("Per-level additive wind-direction or component bias controls.", "逐层风向或风分量加性偏差控制值。"),
    "stations_per_level": ("Maximum observation stations used at each vertical level.", "每个垂直层最多使用的观测站数量。"),
    "wind_extrapolation_factors": ("Per-level factors used to extrapolate winds vertically.", "风场垂直外推使用的逐层系数。"),
    "smoothing_passes": ("Number of wind-field smoothing passes at each level.", "每个垂直层的风场平滑次数。"),
    "barriers": ("Barrier start and end coordinates for flow blocking.", "用于气流阻塞计算的障碍物起止坐标。"),
    "lake_breeze_regions": ("Lake-breeze boxes, coastline segments, and assigned stations.", "湖陆风区域框、岸线线段及其关联站点。"),
    "control_lines": ("Original CALMET control-file lines copied into legacy output metadata.", "写入传统输出元数据的原始 CALMET 控制文件文本。"),
    "print_surface_fields": ("Switches selecting surface fields for printed output.", "选择写入文本输出的地面场开关数组。"),
    "wind_debug_fields": ("Switches selecting wind-model diagnostic fields.", "选择风场模型诊断量的开关数组。"),
    "ring_distances_km": ("Radial-grid ring distances in kilometres.", "极坐标网格各环的半径，单位 km。"),
    "ray_angles_deg": ("Radial-grid ray angles in degrees.", "极坐标网格各射线的角度，单位度。"),
}

MEANING_OVERRIDES.update({
    # Projection and grid definitions shared by TERREL, CTGPROC, MAKEGEO, and CALMET.
    "nx": ("Number of grid cells in the east-west direction.", "东西方向的网格单元数。"),
    "ny": ("Number of grid cells in the north-south direction.", "南北方向的网格单元数。"),
    "dx_km": ("East-west grid-cell size in kilometres.", "东西方向网格分辨率，单位 km。"),
    "dy_km": ("North-south grid-cell size in kilometres.", "南北方向网格分辨率，单位 km。"),
    "projection": ("Map projection code: UTM, TTM, LCC, PS, EM, or LAZA.", "地图投影代码：UTM、TTM、LCC、PS、EM 或 LAZA。"),
    "lat0": ("Latitude of the projection origin or tangent point, in decimal degrees.", "投影原点或切点纬度，单位为十进制度。"),
    "lon0": ("Central-meridian or tangent-point longitude, in decimal degrees.", "中央经线或切点经度，单位为十进制度。"),
    "lat1": ("First standard latitude for LCC, or true-scale latitude for PS.", "LCC 第一标准纬线，或 PS 真比例纬线。"),
    "lat2": ("Second standard latitude for LCC; unused by PS.", "LCC 第二标准纬线；PS 不使用。"),
    "x_origin_km": ("Easting of the south-west corner of grid cell (1, 1), in kilometres.", "网格单元 (1,1) 西南角的东向坐标，单位 km。"),
    "y_origin_km": ("Northing of the south-west corner of grid cell (1, 1), in kilometres.", "网格单元 (1,1) 西南角的北向坐标，单位 km。"),
    "false_easting_km": ("False easting at the projection origin; used by TTM, LCC, and LAZA.", "投影原点假东距；用于 TTM、LCC 和 LAZA。"),
    "false_northing_km": ("False northing at the projection origin; used by TTM, LCC, and LAZA.", "投影原点假北距；用于 TTM、LCC 和 LAZA。"),
    "utm_zone": ("UTM zone number from 1 through 60; used only for UTM.", "UTM 分区号 1–60；仅 UTM 投影使用。"),
    "hemisphere": ("UTM hemisphere, N or S; used only for UTM.", "UTM 半球，N 或 S；仅 UTM 投影使用。"),
    "datum": ("Datum-region code used to interpret and write coordinates.", "用于解释和写出坐标的大地基准区域代码。"),
    "grid_type": ("Grid geometry code: rectangular, shifted rectangular, or polar.", "网格几何类型：矩形、偏移矩形或极坐标网格。"),
    "polar_processing": ("Polar-grid processing method used by TERREL.", "TERREL 极坐标网格处理方法。"),
    "ttm_scale": ("Scale factor on the central meridian for TTM projection.", "TTM 投影中央经线比例因子。"),
    # TERREL and CTGPROC controls.
    "cell_radius_km": ("Radius around each output cell used to collect terrain samples.", "每个输出格点收集地形样本时采用的搜索半径。"),
    "processing_method": ("TERREL rectangular/polar processing algorithm selector.", "TERREL 矩形或极坐标处理算法选项。"),
    "model": ("TERREL output model: CALMET (1), MESOPAC (2), ISC polar (3), ISC Cartesian (4), NUATMOS (5), or generic (6); this also selects cell-average versus peak elevations.", "TERREL 输出模型：CALMET (1)、MESOPAC (2)、ISC 极坐标 (3)、ISC 笛卡尔 (4)、NUATMOS (5) 或通用格式 (6)；同时决定输出格点平均高程还是峰值高程。"),
    "receptor_radius_km": ("Sampling radius used to calculate discrete-receptor elevations.", "计算离散接收点高程时采用的采样半径。"),
    "noise_replacement": ("Replacement-method code for noisy terrain values, by input database class.", "按输入数据库类别指定噪声地形值的替换方法代码。"),
    "noise_minimum_m": ("Minimum valid terrain elevation used by the noise tests, by database class.", "各数据库类别用于噪声检查的最小有效高程。"),
    "missing_replacement": ("Replacement-method code for missing terrain values, by database class.", "按数据库类别指定缺失地形值的替换方法代码。"),
    "default_elevation_m": ("Fallback elevation used when terrain samples remain missing.", "地形样本仍缺失时采用的后备高程。"),
    "qa_hit_threshold_percent": ("Minimum percentage of expected samples required for a QA pass.", "质量检查通过所需的最小有效采样比例。"),
    "mesh_density": ("Number of sub-cell sampling intervals used for land-use databases.", "土地利用数据库在每个输出格内采用的子网格采样密度。"),
    "input_type": ("Land-use database format or family code.", "土地利用数据库格式或数据族代码。"),
    "input_datum": ("Datum of the input land-use coordinates.", "输入土地利用坐标采用的大地基准。"),
    "output_format": ("Output encoding: choose from the filename (auto), traditional CTGPROC format (legacy), or NetCDF.", "输出编码：按文件名判断 (auto)、传统 CTGPROC 格式 (legacy)，或 NetCDF。"),
    "coastline_swap": ("Swap land and water classifications across the supplied coastline.", "依据给定岸线交换陆地与水体分类。"),
    "fill_empty_ocean": ("Fill unsampled ocean cells with ocean_category.", "使用 ocean_category 填充没有样本的海洋格点。"),
    "ocean_category": ("Land-use category assigned to ocean cells.", "赋给海洋格点的土地利用类别。"),
    "final_output": ("Write the final CTGPROC land-use data set.", "是否写出最终 CTGPROC 土地利用数据集。"),
    # MAKEGEO controls.
    "water_threshold": ("Minimum water fraction for treating a grid cell as water.", "将格点判定为水体所需的最小水体比例。"),
    "missing_category": ("Land-use category assigned when usable coverage is insufficient.", "有效覆盖不足时赋给格点的土地利用类别。"),
    "minimum_coverage": ("Minimum summed land-use fraction accepted for a grid cell.", "格点可接受的最小土地利用覆盖比例总和。"),
    "use_terrain": ("Read and include terrain elevations in the geophysical output.", "是否读取地形高程并写入地理物理输出。"),
    "water_categories": ("Input land-use categories treated as water.", "被视为水体的输入土地利用类别。"),
    "output_categories": ("Ordered land-use categories retained in the geophysical output.", "地理物理输出中保留的有序土地利用类别。"),
    "output_water_min": ("Lowest output category regarded as water by CALMET.", "CALMET 视为水体的最小输出类别。"),
    "output_water_max": ("Highest output category regarded as water by CALMET.", "CALMET 视为水体的最大输出类别。"),
    "qa_cell": ("One-based grid cell whose intermediate MAKEGEO calculations are reported.", "输出 MAKEGEO 中间计算信息的网格单元，索引从 1 开始。"),
    "snow_processing": ("Enable seasonally varying snow-covered surface properties.", "是否启用随季节变化的积雪地表属性。"),
    "snow_start_date": ("First YYYYMMDD date covered by snow processing.", "积雪处理覆盖的首个日期，格式 YYYYMMDD。"),
    "snow_end_date": ("Last YYYYMMDD date covered by snow processing.", "积雪处理覆盖的末个日期，格式 YYYYMMDD。"),
    "snow_roughness_method": ("Method used to modify aerodynamic roughness for snow.", "积雪条件下空气动力粗糙度的修正方法。"),
    "snow_height_scale_m": ("Snow-depth scale controlling the roughness adjustment.", "控制粗糙度修正的积雪深度尺度。"),
    "snow_albedo_method": ("Method used to calculate snow albedo as snow ages.", "根据雪龄计算积雪反照率的方法。"),
    "snow_depth_minimum_m": ("Minimum snow depth required to apply snow properties.", "应用积雪地表属性所需的最小雪深。"),
    "image_mode": ("QA Surfer-grid coordinate convention: cell-centre bounds (0) or cell-edge bounds (1).", "质量检查 Surfer 网格的坐标约定：格点中心边界 (0) 或格点外缘边界 (1)。"),
    # General CALMET and meteorological-data controls.
    "timestep_seconds": ("CALMET computational time step; it must divide evenly into one hour.", "CALMET 计算时间步长，必须能整除 1 小时。"),
    "prognostic_timestep_seconds": ("Time interval between consecutive prognostic input records.", "相邻预报场输入记录之间的时间间隔。"),
    "test_mode": ("1 stops after input setup checks; 2 continues into model computation.", "1 在输入设置检查后停止；2 继续执行模型计算。"),
    "regulatory_mode": ("Enable checks that selected technical options follow regulatory guidance.", "是否检查所选技术选项符合监管模式要求。"),
    "include_w_temperature": ("Write 3-D vertical wind and temperature fields required by CALGRID.", "是否写出 CALGRID 所需的三维垂直风速和温度场。"),
    "cloud_method": ("Cloud method: observations (1), CLOUD.DAT (2), 850-hPa RH (3), or all-level RH (4).", "云场方法：观测 (1)、CLOUD.DAT (2)、850 hPa 相对湿度 (3) 或全层相对湿度 (4)。"),
    "cloud_data_format": ("CLOUD.DAT format: unformatted binary (1) or formatted text (2).", "CLOUD.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "surface_data_format": ("SURF.DAT format: unformatted binary (1) or formatted text (2).", "SURF.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "precipitation_data_format": ("PRECIP.DAT format: unformatted binary (1) or formatted text (2).", "PRECIP.DAT 格式：非格式化二进制 (1) 或格式化文本 (2)。"),
    "precipitation_station_count": ("Number of precipitation stations; -1 selects precipitation from prognostic data.", "降水站数量；-1 表示使用预报资料中的降水。"),
    # Wind-field options from CALMET input groups 4 and 5.
    "wind_extrapolation": ("Surface-wind extrapolation method; negative codes also ignore layer-1 upper-air data.", "地面风向高层外推方法；负值还会忽略高空站第一层资料。"),
    "extrapolate_calm": ("Extrapolate a surface observation even when it reports calm wind.", "地面站报告静风时是否仍进行垂直外推。"),
    "upper_air_separation_cells": ("Minimum surface-to-upper-air station separation that permits surface-wind extrapolation; -1 disables the restriction.", "允许地面风外推的地面站与高空站最小间距；-1 取消该限制。"),
    "varying_radius": ("Expand the influence radius until a station is found when none lies within the configured maximum.", "最大影响半径内没有站点时，是否扩大半径直至找到最近站点。"),
    "minimum_station_radius_km": ("Minimum influence radius used for wind interpolation.", "风场插值使用的最小影响半径。"),
    "maximum_surface_radius_km": ("Maximum station influence radius over land in the surface layer.", "陆地下垫面近地层站点的最大影响半径。"),
    "maximum_upper_radius_km": ("Maximum station influence radius over land aloft.", "陆地高层站点的最大影响半径。"),
    "maximum_water_radius_km": ("Maximum station influence radius over water.", "水面站点的最大影响半径。"),
    "surface_background_radius_km": ("Distance where surface observations and the first-guess field receive equal weight.", "近地层观测与初始猜测场权重相等时的距离尺度。"),
    "upper_background_radius_km": ("Distance where upper-air observations and the first-guess field receive equal weight.", "高层观测与初始猜测场权重相等时的距离尺度。"),
    "prognostic_radius_km": ("Weighting distance for prognostic winds when IPROG uses them as a step-1 field.", "IPROG 将预报风作为第一步风场时采用的权重距离。"),
    "barrier_count": ("Number of line barriers that block wind interpolation.", "阻挡风场插值的线状障碍物数量。"),
    "barrier_top_level": ("Highest model layer to which interpolation barriers apply.", "插值障碍物影响到的最高模型层。"),
    "critical_froude": ("Critical Froude number separating terrain blocking from flow-over behavior.", "区分地形阻塞与越岭气流的临界弗劳德数。"),
    "divergence_limit": ("Convergence target for wind-field divergence minimization.", "风场散度最小化过程的收敛目标。"),
    "divergence_iterations": ("Maximum iterations allowed for divergence minimization.", "散度最小化允许的最大迭代次数。"),
    "terrain_radius_km": ("Horizontal radius over which terrain affects the diagnostic wind adjustment.", "地形影响诊断风场调整的水平半径。"),
    # Mixing height, temperature, humidity, precipitation, and overwater fluxes.
    "coriolis": ("Absolute Coriolis parameter used by mixing-height equations, in s⁻¹.", "混合层高度方程使用的科里奥利参数绝对值，单位 s⁻¹。"),
    "iavezi": ("Enable spatial averaging of calculated mixing heights.", "是否对计算得到的混合层高度进行空间平均。"),
    "mixing_wind_level": ("Wind-field layer used to define the upwind averaging direction.", "确定上风向平均方向时使用的风场层。"),
    "half_average_angle": ("Half-angle of the upwind search cone used for spatial averaging.", "空间平均所用上风向搜索锥的半角。"),
    "imixh": ("Convective mixing-height method: ±1 Maul-Carson or ±2 Batchvarova-Gryning; negative uses OCD over water.", "对流混合层方法：±1 为 Maul–Carson，±2 为 Batchvarova–Gryning；负值在水面使用 OCD。"),
    "land_flux_threshold": ("Minimum positive land buoyancy flux that sustains convective mixing-height growth.", "维持陆地对流混合层增长所需的最小正浮力通量。"),
    "water_flux_threshold": ("Minimum positive water buoyancy flux that sustains convective mixing-height growth.", "维持水面对流混合层增长所需的最小正浮力通量。"),
    "convective_relaxation": ("Relax convective mixing height toward equilibrium when buoyancy flux is below its threshold.", "浮力通量低于阈值时，是否让对流混合层高度向平衡值松弛。"),
    "convective_relaxation_time_s": ("Relaxation time scale used when convective_relaxation is enabled.", "启用对流混合层松弛时使用的时间尺度。"),
    "water_temperature_mode": ("Overwater lapse-rate source: SEA.DAT (0), prognostic lapse rate (1), or prognostic lapse rate and ΔT (2).", "水面递减率来源：SEA.DAT (0)、预报递减率 (1)，或预报递减率和 ΔT (2)。"),
    "prognostic_ocean_category": ("Land-use category representing ocean in the prognostic meteorological input.", "预报气象输入中代表海洋的土地利用类别。"),
    "overwater_flux_method": ("Overwater flux method: legacy OCD (0) or a selected COARE wave option (10, ±11, ±12).", "水面通量方法：传统 OCD (0) 或相应 COARE 波浪方案 (10、±11、±12)。"),
    "coastal_shelf_length_km": ("Coastal/shallow-water length scale used by COARE roughness calculations.", "COARE 粗糙度计算使用的沿岸或浅水长度尺度。"),
    "irhprog": ("Relative-humidity source: SURF.DAT observations (0) or prognostic fields (1).", "相对湿度来源：SURF.DAT 观测 (0) 或预报场 (1)。"),
    "itprog": ("Temperature source: observations (0), surface observations plus prognostic aloft (1), or prognostic at all levels (2).", "温度来源：观测 (0)、地面观测加高层预报 (1)，或所有高度均使用预报场 (2)。"),
    "temperature_interpolation_power": ("Temperature interpolation weighting: inverse distance (1) or inverse-distance squared (2).", "温度插值权重：距离反比 (1) 或距离平方反比 (2)。"),
    "temperature_radius_km": ("Radius of influence for temperature interpolation.", "温度插值的影响半径。"),
    "temperature_station_limit": ("Maximum number of stations included in temperature interpolation.", "温度插值最多使用的站点数。"),
    "iavet": ("Enable spatial averaging of interpolated temperatures using the mixing-height averaging geometry.", "是否使用混合层平均的搜索几何对插值温度进行空间平均。"),
    "water_temperature_gradient_below": ("Default water-cell temperature gradient below mixing height, in K m⁻¹.", "水体格点混合层以下的默认温度梯度，单位 K/m。"),
    "water_temperature_gradient_above": ("Default water-cell temperature gradient above mixing height, in K m⁻¹.", "水体格点混合层以上的默认温度梯度，单位 K/m。"),
    "water_temperature_category_min": ("First land-use category treated as water during temperature interpolation.", "温度插值中视为水体的起始土地利用类别。"),
    "water_temperature_category_max": ("Last land-use category treated as water during temperature interpolation.", "温度插值中视为水体的结束土地利用类别。"),
    "precipitation_interpolation": ("Precipitation weighting: 1/R (1), 1/R² (2), or exponential/R² (3).", "降水插值权重：1/R (1)、1/R² (2) 或指数/R² (3)。"),
    "precipitation_radius_km": ("Precipitation-station radius of influence.", "降水站的影响半径。"),
    "precipitation_cutoff": ("Rates below this threshold are set to zero, in mm h⁻¹.", "低于该阈值的降水率置零，单位 mm/h。"),
    # Diagnostic inputs and output/debug selection.
    "diagnostic_surface_temperature": ("Surface temperature source: compute internally (0) or read DIAG.DAT (1).", "地表温度来源：内部计算 (0) 或读取 DIAG.DAT (1)。"),
    "diagnostic_initial_wind": ("Initial-guess wind source: compute internally (0) or read DIAG.DAT (1).", "初始猜测风来源：内部计算 (0) 或读取 DIAG.DAT (1)。"),
    "diagnostic_surface_wind": ("Surface wind source: SURF.DAT speed/direction (0) or DIAG.DAT U/V (1).", "地面风来源：SURF.DAT 风速风向 (0) 或 DIAG.DAT U/V (1)。"),
    "diagnostic_upper_wind": ("Upper-air wind source: UP.DAT speed/direction (0) or DIAG.DAT U/V (1).", "高空风来源：UP.DAT 风速风向 (0) 或 DIAG.DAT U/V (1)。"),
    "surface_temperature_option": ("Surface station used for temperature: station number, -1 for spatially varying, or -2 for domain-average prognostic temperature.", "地表温度代表站：站号；-1 为空间变化场；-2 为预报温度区域平均。"),
    "upper_temperature_option": ("Upper-air station used for lapse rate: station number, -1 for spatially varying, or -2 for domain-average prognostic lapse rate.", "递减率代表高空站：站号；-1 为空间变化场；-2 为预报递减率区域平均。"),
    "lapse_height_m": ("Layer depth used to calculate the domain-scale temperature lapse rate.", "计算区域尺度温度递减率所用的层厚。"),
    "initial_guess_upper_station": ("Upper-air station for initial wind guess; -1 retains a spatially varying 3-D guess.", "初始风猜测场使用的高空站；-1 保留空间变化的三维猜测场。"),
    "initial_guess_height_range_m": ("Bottom and top heights used to form a domain-average initial wind.", "生成区域平均初始风时使用的底部和顶部高度。"),
    "print_fields": ("Enable line-printer-style meteorological field output.", "是否启用传统文本形式的气象场输出。"),
    "print_interval_hours": ("Interval between printed meteorological fields, in hours.", "气象场文本输出的时间间隔，单位小时。"),
    "print_uv_levels": ("Per-layer switches selecting U/V wind output when print_fields is enabled.", "启用 print_fields 时选择输出 U/V 风场层的逐层开关。"),
    "print_w_levels": ("Per-layer switches selecting vertical-wind output; also requires include_w_temperature.", "选择垂直风速输出层的逐层开关；还需启用 include_w_temperature。"),
    "print_temperature_levels": ("Per-layer switches selecting 3-D temperature output; also requires include_w_temperature.", "选择三维温度输出层的逐层开关；还需启用 include_w_temperature。"),
    "micrometeorology_debug": ("Print micrometeorological inputs and internal variables; produces large output.", "是否打印微气象输入和内部变量；会产生大量输出。"),
    "debug_first_step": ("First time step included in micrometeorology debug output.", "微气象调试输出包含的首个时步。"),
    "debug_last_step": ("Last time step included in micrometeorology debug output.", "微气象调试输出包含的末个时步。"),
    "wind_debug_levels": ("Number of wind-field levels written to diagnostic files, starting at the surface.", "从地面开始写入风场诊断文件的层数。"),
    "noobs": ("Observation mode: 0 uses surface/overwater/upper-air observations; 1 replaces upper-air observations with prognostic data; 2 uses prognostic data at all levels.", "观测模式：0 使用地面、水面和高空观测；1 用预报资料替代高空观测；2 所有高度均使用预报资料。"),
    "iwfcod": ("Wind model: objective analysis only (0) or the full diagnostic wind module (1).", "风场模型：仅客观分析 (0)，或完整诊断风场模块 (1)。"),
    "iprog": ("How prognostic winds enter the wind model: disabled (0), step-1 field (1/3/13), initial guess (2/4/14), or pseudo-observations (5/15); codes distinguish CSUMM, MM4, and MM5/3D inputs.", "预报风进入风场模型的方式：禁用 (0)、作为第一步风场 (1/3/13)、作为初始猜测场 (2/4/14)，或作为伪观测 (5/15)；不同代码对应 CSUMM、MM4 和 MM5/3D 输入。"),
    "mnmdav": ("Maximum search radius, in grid cells, for mixing-height and temperature spatial averaging.", "混合层高度和温度空间平均的最大搜索半径，单位为网格单元。"),
    "wind_bias": ("Per-layer station-weight bias from -1 to 1: negative reduces upper-air weight, positive reduces surface-station weight, and zero leaves 1/R² weights unchanged.", "逐层站点权重偏差，范围 -1 到 1：负值降低高空站权重，正值降低地面站权重，0 保持 1/R² 权重不变。"),
    "mixing_lapse_depth_m": ("Depth above the current convective mixing height over which the stable-layer lapse rate is calculated.", "在当前对流混合层顶以上计算稳定层温度递减率时采用的层厚。"),
    "turbidity_coefficient_1": ("HA1 coefficient, in W m⁻², in the Holtslag–van Ulden energy-budget solar/turbidity parameterization.", "Holtslag–van Ulden 能量收支太阳辐射/浑浊度参数化中的 HA1 系数，单位 W/m²。"),
    "turbidity_coefficient_2": ("HA2 coefficient, in W m⁻², in the Holtslag–van Ulden energy-budget solar/turbidity parameterization.", "Holtslag–van Ulden 能量收支太阳辐射/浑浊度参数化中的 HA2 系数，单位 W/m²。"),
    "cloudiness_coefficient_1": ("Dimensionless HB1 cloud-cover coefficient in the Holtslag–van Ulden energy budget.", "Holtslag–van Ulden 能量收支中的无量纲云量系数 HB1。"),
    "cloudiness_coefficient_2": ("Dimensionless HB2 cloud-cover coefficient in the Holtslag–van Ulden energy budget.", "Holtslag–van Ulden 能量收支中的无量纲云量系数 HB2。"),
    "net_radiation_coefficient_1": ("HC1 net-radiation coefficient in the Holtslag–van Ulden energy budget, in W m⁻² K⁻⁶.", "Holtslag–van Ulden 能量收支中的净辐射系数 HC1，单位 W/(m²·K⁶)。"),
    "net_radiation_coefficient_2": ("HC2 net-radiation coefficient in the Holtslag–van Ulden energy budget, in W m⁻².", "Holtslag–van Ulden 能量收支中的净辐射系数 HC2，单位 W/m²。"),
    "net_radiation_coefficient_3": ("Dimensionless HC3 net-radiation coefficient in the Holtslag–van Ulden energy budget.", "Holtslag–van Ulden 能量收支中的无量纲净辐射系数 HC3。"),
})

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


def _meaning(name: str, language: str, parameter: inspect.Parameter) -> str:
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
    headings = ("Parameter", "Meaning", "Python type", "Default", "Zhejiang value") if language == "en" else (
        "参数", "含义", "Python 类型", "默认值", "浙江示例值")
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
        default_text = _value(default, name=name)
        if default_text == "required":
            default_text = "required" if language == "en" else "必填"
        values = (f"`{name}`", _meaning(name, language, parameter), f"`{_type(parameter)}`",
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
configuration models. “Default” is the value used in `inp=None` mode when the
caller omits that parameter; “Zhejiang value” is the effective value after
parsing the bundled control file. `inp` itself and `**legacy_parameters` are
entry/compatibility mechanisms rather than model controls, so they are not
repeated in each table. Paths are relative to the repository root. Long
sequences are abbreviated with their total item count. Meanings and option
effects are checked against the detailed comments in the bundled CALMET-family
control templates, the reference Fortran sources, and the Python code that
consumes each value. Regenerate the tables
with `python tools/generate_parameter_reference.py`; use `--check` in CI."""
    else:
        intro = """## 完整参数参考

以下表格由公开函数签名和类型化配置模型生成。“默认值”表示 `inp=None` 模式下调用方
省略参数时采用的值；“浙江示例值”表示解析仓库内置控制文件后得到的实际值。`inp`
本身和 `**legacy_parameters` 是入口及兼容机制，不属于模型控制项，因此不在每张表中
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。参数含义和选项
效果依据仓库内 CALMET 系列控制文件的详细注释、参考 Fortran 源码，以及实际读取
这些参数的 Python 代码交叉核对。API 变化后运行
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
