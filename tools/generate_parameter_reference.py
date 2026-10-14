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
    headings = ("Parameter", "Python type", "Default", "Zhejiang value") if language == "en" else (
        "参数", "Python 类型", "默认值", "浙江示例值")
    rows = [f"| {' | '.join(headings)} |", "| --- | --- | --- | --- |"]
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
        values = (f"`{name}`", f"`{_type(parameter)}`", f"`{default_text}`", f"`{actual}`")
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
