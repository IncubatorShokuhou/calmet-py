# 数组后端

[English](../backends.md)

PyCALMET 只有一个分发器 :mod:`pycalmet.backend`，各阶段的矢量化计算核都调用它。文件读取器、NetCDF 写出器和原地风场求解器都只有这一份实现。

```python
import pycalmet

pycalmet.set_backend("numpy")          # 默认
pycalmet.set_backend("jax")             # 需要 gpu extra
pycalmet.set_backend("jax", device="cpu")
pycalmet.set_backend("auto")            # 能导入 JAX 则用 JAX，否则用 NumPy

with pycalmet.use_backend("jax", device="gpu"):
    path = pycalmet.run_pipeline(...)
```

环境变量在后端第一次被用到时读取；:func:`pycalmet.reset_backend` 之后会再读一次：

| 变量 | 取值 | 作用 |
| --- | --- | --- |
| `PYCALMET_BACKEND` | `numpy`（默认）、`jax`、`auto` | 尚未调用 `set_backend` 时选择后端。 |
| `PYCALMET_DEVICE` | `auto`（默认）、`cpu`、`gpu`、`cuda`、`cuda:0`、`gpu:1` | 仅 JAX 后端使用。 |

设备取 `auto` 时，优先使用 JAX 可见的第一块 GPU，否则使用 CPU。`device="gpu"` 或 `PYCALMET_DEVICE=gpu` 在 JAX 没有 GPU 时抛出 `RuntimeError`，批处理脚本不会在缺少 GPU 时静默改到主机上跑完。

后端选择对整个进程生效。:func:`pycalmet.use_backend` 退出时恢复进入之前的后端；若本包改动过 JAX 的 x64 开关和默认设备，这两项也一并恢复。同一时刻只应从一个线程切换后端。

## 哪些计算在所选设备上运行

下面这些计算核走当前后端。在 JAX 设备上它们即时执行，而不是把整个阶段融成一次 `jax.jit`。调用方写出 NetCDF 或改写主机缓冲区之前，结果会拷回 NumPy 数组：

| 阶段 | 计算核 |
| --- | --- |
| CALWRF | 交错格点还原、位势高度、格点风转到地理坐标、相对地球的风、风速与风向、降水增量、气压与温度 |
| TERREL | 高程累加和的格点平均 |
| CTGPROC | 投影坐标到格点索引，以及土地利用直方图 |
| MAKEGEO | 类别求和、水体比例求和，以及加权地表场 |
| CALMET | 密度、科里奥利参数、反距离插值、风速与风向、Teixeira 云量、太阳高度、感热通量、水平散度与积分垂直速度 |

依赖 GDAL、PROJ、Shapely 或 SciPy `cKDTree` 的几何运算在主机上执行。按顺序扫描的 NumPy 循环也在主机上：散度最小化、Gauss–Seidel 平滑和客观分析。这些更新按固定顺序原地写入；要在 JAX 数组上做同样的事，需要单独的求解器。`pycalmet.legacy` 下的兼容命名空间保持原样，始终使用 NumPy。

因此，每个 CALMET 小时会把上述计算核用到的大型诊断数组放到设备上，再回到 NumPy 做迭代风场调整。未选择 JAX 时，这条分界使浙江示例已有的 NumPy 结果保持不变。

## 浮点类型与设备放置

物理模型场是 `float32`。时间、地图坐标，以及文档写明的归约保持 `float64`。JAX 自身默认关闭 64 位模式，那会把这些归约悄悄收窄。选择 JAX 后端时，本进程的 `jax_enable_x64` 设为 `True`，`jax_default_device` 设为解析到的设备。通过 `set_backend("numpy")`、`reset_backend()` 离开 JAX，或在 `use_backend` 结束时，都会恢复先前的 JAX 设置。

由此：

- PyCALMET 以外的主机 NumPy 代码不读取 JAX 开关，因此不受这些设置影响。
- 同一进程中的其他 JAX 代码会看到 x64 已打开，以及本包选定的默认设备，直到后端被恢复。
- 超越函数（`sin`、`exp`、`power`）与 NumPy 的差别可以出现在最后几位。JAX 输出与 NumPy 或 Fortran 参考数值等价，最后几位可以不同。
- 每个加速计算核都返回 NumPy `ndarray`。随后按下标写 NetCDF 变量的代码不必知道数组来自哪个后端。
- 第一次 JAX 调用可能用于初始化 GPU 运行时。随后的调用复用该进程。

```python
import numpy as np
import pycalmet
from pycalmet.calmet_numerics import density_from_pressure_temperature

with pycalmet.use_backend("jax", device="cpu"):
    density = density_from_pressure_temperature(
        np.array([1000.0], dtype=np.float32),
        np.array([300.0], dtype=np.float32),
    )

assert isinstance(density, np.ndarray)
assert density.dtype == np.float32
print(pycalmet.backend_info())
```

JAX 处于活动状态时，`backend_info()` 报告 `model_dtype="float32"`、`x64_enabled=True`，以及 JAX 设备标签。该标签在主机上包含 `cpu`，在 NVIDIA 设备上包含 `cuda` 或 `gpu`；具体拼写跟随已安装的 JAX 版本。
