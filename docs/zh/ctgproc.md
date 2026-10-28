# CTGPROC

[English](../ctgproc.md)

CTGPROC 把分类土地利用栅格重采样到 CALMET 网格上，并按类别保存命中数。MAKEGEO 再把这些计数变成比例。浙江的土地利用源是 `EAUSGS1.IMG`：一张无文件头的 8 位栅格，位于亚洲兰伯特方位网格上。`input_type="GLAZAS"` 选择这套几何，`input_datum="ESR-S"` 标明它的基准。目标网格的 WGS-84 基准并不包含这两项，需要单独给出。

```python
from pathlib import Path
import pycalmet

grid = pycalmet.GridConfig(nx=80, ny=60, dx_km=1.0, dy_km=1.0, projection="LCC")
landuse = pycalmet.ctgproc(
    inp=None,
    grid=grid,
    landuse_files=[Path("data/landuse/usgs-glcc/EAUSGS1.IMG")],
    input_type="GLAZAS",
    input_datum="ESR-S",
    mesh_density=12,
    output=Path("output/landuse.nc"),
)
```

`mesh_density=12` 是 `MESHGLAZ=12` 的 Python 写法：每个源像元在分箱前被分成 12×12 的细分网格。`ctgproc` 与 `ctrproc` 是同一阶段；`ctrproc` 是历史入口名。

## 分箱与后端

对每个被接受的样本，计算核做三件事：

1. 把样本投影到目标网格的公里坐标。
2. 用 `floor((x - origin) / dx)` 把公里坐标换成从 0 开始的格点索引。
3. 按类别对样本做直方图。

第 2 步和第 3 步使用当前后端。直方图累加到主机上的 `int64` 计数数组。后续轮次、岸线编辑和 NetCDF 写出器都会原地修改这块缓冲区。岸线多边形、GeoTIFF 窗口和基准变换分别使用 Shapely、Rasterio 和 PROJ。

索引保持原值，不会裁进网格。网格外的样本要由调用方在分箱之前剔除；浙江读取器用细分窗口完成这一步。类别不在输出映射中时，直方图一旦寻址该类别就会报错；NumPy 与 JAX 后端都是如此。

积雪记账（供后续 MAKEGEO 使用的 `snow` 网格）是主机上的稀疏字典，JAX 后端不改动它。流水线关闭积雪处理，从而保证只产生一个输出文件。
