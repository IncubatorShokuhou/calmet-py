# CALMET

[English](../calmet.md)

CALMET 诊断质量守恒风场，以及扩散模式要读取的混合层高度、温度和湍流尺度。公开入口是 `pycalmet.calmet`。它可以接受控制文件，也可以接受关键字。

下面是纯预报的一个小时，也就是浙江 WRF 运行的形态：

```python
from datetime import datetime
from pathlib import Path
import pycalmet

grid = pycalmet.GridConfig(
    nx=80,
    ny=60,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
)

result = pycalmet.calmet(
    inp=None,
    start=datetime(2018, 3, 18, 0),
    end=datetime(2018, 3, 18, 1),
    grid=grid,
    geophysical=Path("output/geo.nc"),
    meteorology=[Path("output/calwrf.nc")],
    noobs=2,
    iprog=14,
    irhprog=1,
    itprog=2,
    output=Path("output/calmet.nc"),
)
```

`noobs=2` 与 `iprog=14` 一起使用时，CALMET 从 CALWRF 数据集读取初始风和标量场，地面站和高空站不参与。按这种形式使用关键字 API 时，`end` 是模拟区间的开上界。每个控制项采用闭区间还是开区间，以 `README.zh-CN.md` 的参数表为准。基于观测的运行改为给出地面、降水和探空路径，配置上也不使用 `noobs=2`。两种做法都写在浙江指南 `examples/zhejiang/README.md` 里。

## 哪些数值计算使用 JAX

`pycalmet.calmet_numerics` 中的诊断构件是普通函数。其中铺满数组、且不做原地扫描的函数遵循[后端](backends.md)：

- 干空气密度和科里奥利参数
- 站点值的反距离插值
- 矢量风转为风速和气象风向
- 由相对湿度得到的 Teixeira 云量
- 热通量方案使用的半小时太阳高度正弦
- 感热和短波辐射
- 水平散度，以及由它向上积分的垂直速度，包括可选的 O'Brien 订正

`minimize_divergence` 和 `smooth_wind` 按固定顺序访问每个格点，并更新刚刚写过的风。无论选择哪个后端，它们都在 NumPy 上执行。客观分析和障碍线例程同样如此。选择 JAX 时这些扫描照常执行；被放到设备上的是周围的数组计算核。

求解器内部的模型状态是 float32。散度积分用 float64 累加，再由计算核把存盘的场转回 float32。NumPy 路径使用同一拆分，因此 JAX 后端会打开 64 位模式。若关闭 64 位模式，收敛检验接受的格点会改变。

legacy API（`pycalmet.legacy.calmet`）复现历史上每条语句都使用 float32 的路径。这条路径不读取后端设置。对照 Fortran 二进制结果时使用它；对照本包的 NumPy 求解器时使用 `pycalmet.calmet`。

## 读取结果

`calmet` 写出的 NetCDF 就是 `run_pipeline` 保留的产品。变量名遵循 `README.zh-CN.md` 中已有的 CALMET 输出约定（风分量、混合层高度、温度，以及二维地面场）。只有输出路径或兼容开关明确要求时，才写出 legacy 的 `CALMET.DAT` 及其 `.aux` 配套文件。

一小时的 WRF 窗口用于回归检查。生产运行需要前一夜的气象场，混合层才有可延续的初值。浙江 README 在复现命令旁说明了这一限制。
