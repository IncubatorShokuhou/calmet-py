# 端到端流水线

[English](../pipeline.md)

`pycalmet.run_pipeline` 用一次调用依次运行 CALWRF、TERREL、CTGPROC、MAKEGEO 和 CALMET。中间 NetCDF 文件放在临时目录里，流水线结束时删除。作为 `output` 传入的路径是保留下来的 CALMET 数据集。

各阶段共用同一套水平网格。先构建一次，再传给每个需要几何的阶段：

```python
from datetime import datetime
from pathlib import Path

import pycalmet

grid = pycalmet.GridConfig(
    nx=610,
    ny=499,
    dx_km=1.0,
    dy_km=1.0,
    projection="LCC",
    lat0=27.0,
    lon0=117.5,
    lat1=30.0,
    lat2=60.0,
    datum="WGS-84",
)

example = Path("examples/zhejiang")
result = pycalmet.run_pipeline(
    inp=None,
    calwrf_wrf_files=[
        example / "wrfout_d01_2018-03-18_00_00_00.nc",
        example / "wrfout_d01_2018-03-18_01_00_00.nc",
    ],
    calwrf_start=datetime(2018, 3, 18, 0),
    calwrf_end=datetime(2018, 3, 18, 1),
    terrel_srtm_dir=Path("data/terrain/srtm3"),
    terrel_grid=grid,
    terrel_sampling="bilinear",
    ctgproc_landuse_files=[Path("data/landuse/usgs-glcc/EAUSGS1.IMG")],
    ctgproc_grid=grid,
    ctgproc_input_type="GLAZAS",
    ctgproc_input_datum="ESR-S",
    ctgproc_mesh_density=12,
    makegeo_grid=grid,
    calmet_start=datetime(2018, 3, 18, 0),
    calmet_end=datetime(2018, 3, 18, 1),
    calmet_grid=grid,
    calmet_noobs=2,
    calmet_iprog=14,
    output=Path("output/zhejiang/calmet.nc"),
)
```

关键字是阶段字段加上前缀（`calwrf_`、`terrel_`、`ctgproc_`、`makegeo_`、`calmet_`）。网格也可以拆成带前缀的标量，例如 `terrel_nx` 和 `terrel_dx_km`。完整表格见 `README.zh-CN.md` 的参数参考。

第一次运行之前，有三条只属于流水线的约束：

- `makegeo_snow_processing` 保持为假。积雪处理每天写一份 GEO 文件，与只保留一个输出的约定冲突。
- `calmet_save_output` 保持为真，`calmet_test_mode` 保持为 `2`，最后一个小时会完成计算并写出。
- CALWRF 的 `start` 与 `end` 都包含在内，对应的是 WRF 记录。它们夹住 CALMET 时段；该时段的结束时刻为开边界。上例传入 00:00 和 01:00，CALMET 据此填满 `[00:00, 01:00)`。

另一个入口是已经放好五份控制文件的目录。每份文件内部的路径都相对该文件自身解析：

```python
from pathlib import Path
import pycalmet

result = pycalmet.run_pipeline(
    Path("examples/zhejiang/config"),
    output=Path("output/zhejiang/calmet.nc"),
)
```

调用方已经备好五个类型化阶段对象时，用 `PipelineConfig` 把它们放在一起。`run_pipeline` 把该对象当作第一个参数。

## 后端在何处生效

调用 `run_pipeline` 之前先调用 :func:`pycalmet.set_backend`，或进入 :func:`pycalmet.use_backend`。流水线不会自行选择后端。默认 NumPy 后端下，浙江示例脚本的行为与引入 JAX 之前相同。选中 JAX 后，[数组后端](backends.md)列出的计算核在设备上运行；文件读写和迭代风场求解器在主机上运行。

可运行示例是 `examples/zhejiang/run.py`，由 `./examples/zhejiang/run.sh` 启动。要检查中间文件时，用 `examples/zhejiang/run_stages.py` 按五份控制文件逐个阶段运行。
