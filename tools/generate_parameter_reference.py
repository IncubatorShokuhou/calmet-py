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
sequences are abbreviated with their total item count. Regenerate the tables
with `python tools/generate_parameter_reference.py`; use `--check` in CI."""
    else:
        intro = """## 完整参数参考

以下表格由公开函数签名和类型化配置模型生成。“默认值”表示 `inp=None` 模式下调用方
省略参数时采用的值；“浙江示例值”表示解析仓库内置控制文件后得到的实际值。`inp`
本身和 `**legacy_parameters` 是入口及兼容机制，不属于模型控制项，因此不在每张表中
重复列出。路径均相对于仓库根目录，较长序列会缩写并标出元素总数。API 变化后运行
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
