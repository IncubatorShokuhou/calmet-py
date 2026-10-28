# MAKEGEO

[English](../makegeo.md)

MAKEGEO 把 TERREL 地形网格和 CTGPROC 土地利用计数合成 CALMET 读取的地理物理场：优势土地利用、粗糙度长度、反照率、Bowen 比、土壤热通量参数、人为热和叶面积指数。

```python
from pathlib import Path
import pycalmet

geo = pycalmet.makegeo(
    inp=None,
    grid=pycalmet.GridConfig(nx=80, ny=60, dx_km=1.0, dy_km=1.0),
    terrain=Path("output/terrain.nc"),
    landuse=Path("output/landuse.nc"),
    output=Path("output/geo.nc"),
)
```

未提供属性表时，MAKEGEO 使用标准 USGS-38 记录。粗糙度是各类别取值按土地利用比例加权后的几何平均。其余物理场为算术平均。优势土地利用取比例最大的类别；仅当水体比例超过 `water_threshold`（默认 0.5）时，候选范围限于水体类别。

## 加权场与后端

比例求和、水体比例求和，以及算术与对数加权平均，都在当前后端上以 float64 计算，返回时写成 float32 场。优势类别的索引在主机上用 `argmax` 得到，结果写入整数图。

覆盖率低于 `0.001` 的格点用 `missing_category` 填上（默认 55，水体）。覆盖率介于该下限与 `minimum_coverage`（默认 0.96）之间时视为错误，两个后端都报错。对应测试检查的是数值结果：JAX 运行必须对 NumPy 运行拒绝的同一网格报错。

逐日积雪修正的 GEO 文件走单独的代码路径。流水线不接受这条路径，因为它会写出多个文件。研究需要 `snow_processing=True` 时直接调用 `makegeo`；其中的雪龄反照率衰减仍在 NumPy 上计算。

等价的控制文件调用是 `pycalmet.makegeo("makegeo.inp")`。此时网格基准、土地利用基准和地形基准可以不同：CTGPROC 和 TERREL 已经把两套数据重采样到 `grid` 上。
