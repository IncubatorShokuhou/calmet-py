# CALWRF

[English](../calwrf.md)

CALWRF 读取 WRF-ARW NetCDF，写出 CALMET 所需的三维气象场：质量层上的气压、温度、水汽、水成物和相对地球的风，以及地面风和降水增量。

```python
from pathlib import Path
import pycalmet

output = pycalmet.calwrf(
    FLNAMES=[Path("wrfout_d01_2018-03-18_00_00_00.nc")],
    NFILES=1,
    IDATEB="2018031800",
    IDATEE="2018031800",
    FIO=Path("output/calwrf.nc"),
)
```

`calwrf` 接受上述 legacy 控制项名称，或一个 `CalwrfConfig`。函数返回 NetCDF 路径。只有设置 `legacy_output` 时才写出传统 M3D 文件。水平与垂直子集使用从 1 开始的闭区间（`i_range`、`j_range`、`k_range`）；默认保留完整 WRF 网格。

## 计算核做什么

对每个选中的小时，读取器先用 `netCDF4` 取出 WRF 场，然后：

1. 对相邻点取平均，把 U、V、W 和位势从交错格点还原到质量点。
2. 把扰动气压和位温换算为百帕和开尔文。
3. 按 WRF 地图投影，把相对格点的风转到向东、向北分量，再给出气象风速和风向。
4. 把累积量 `RAINC` 和 `RAINNC` 换成逐小时增量；累积量下降时按重置处理。

这四步是[数组后端](backends.md)上的 CALWRF 计算核。用邻点替换少量无效格点的范围检查留在 NumPy：相对三维场它们很小，并且使用带掩码的下标更新。

文件中的 WRF 时间是无时区时间戳。CALWRF 的时间窗两端都包含在内，因此一小时的 CALMET 时段需要起点和终点两条记录。浙江配置把这对边界写在 `examples/zhejiang/config/calwrf.inp`。
