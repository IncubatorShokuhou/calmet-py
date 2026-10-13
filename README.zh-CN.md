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
- 保留 Fortran 参考源码，并用数值回归测试验证 Python 实现。

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

每个阶段既可以直接传入关键字参数，也可以读取 CALMET 系列传统控制文件。阶段
函数会返回主输出文件路径。

```python
from pathlib import Path

import pycalmet

wrf = pycalmet.calwrf(
    wrf_files=[Path("wrfout_d01_2018-03-18_00_00_00")],
    start="2018031800",
    end="2018031801",
    output="output/calwrf.nc",
)

calmet = pycalmet.calmet(
    "examples/zhejiang/config/calmet.inp",
    meteorology=wrf,
    geophysical="examples/zhejiang/geo.nc",
    output="output/calmet.nc",
    start="2018031800",
    end="2018031801",
)
```

公开 API：

```python
pycalmet.calwrf(...)
pycalmet.terrel(...)
pycalmet.ctrproc(...)   # pycalmet.ctgproc 是兼容别名
pycalmet.makegeo(...)
pycalmet.calmet(...)
pycalmet.readlst(...)
```

## 浙江示例

仓库内置三个连续 WRF 时次和已生成的地理场。从仓库根目录运行：

```bash
sha256sum -c examples/zhejiang/SHA256SUMS

./examples/zhejiang/run.sh \
  --wrf examples/zhejiang \
  --geo examples/zhejiang/geo.nc \
  --start 2018031800 \
  --end 2018031801 \
  --skip-static \
  --output output/zhejiang.nc
```

数据来源、时间边界和静态场重建方法见
[examples/zhejiang/README.md](examples/zhejiang/README.md)。

## 仓库结构

```text
src/pycalmet/          Python 软件包
tests/                 单元、集成和数值回归测试
examples/zhejiang/     可运行的真实数据示例及控制文件
terrain/srtm3/         浙江区域使用的 SRTM3 瓦片
fortran/               仅供参考的 Fortran 源码，运行时不使用
tools/                 Python 诊断与回归工具
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

绝大多数回归结果与平台无关。少量测试会检查 GNU `REAL*4` 超越函数的精确位模式；
不同 glibc 版本可能产生 1–2 ULP 差异，但物理结果等价。

`tools/` 中的脚本可逐字段比较 NetCDF 或传统输出。`fortran/` 中的源码用于审计和
数值参考，不是安装依赖。

## 参与贡献

运行时代码放在 `src/pycalmet`，行为修改应增加针对性测试，不要提交生成的模型
产物。修改参考 Fortran 时应保留固定格式布局。拉取请求应说明受影响阶段、输入
配置以及实际执行的验证命令。

## 致谢

PyCALMET 遵循 CALPUFF 系统的处理模型和文件约定。仓库附带的 WRF 子集与地形
数据用于保证数值验证可复现，其来源和校验值记录在对应数据目录中。
