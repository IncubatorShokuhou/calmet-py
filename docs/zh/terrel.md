# TERREL

[English](../terrel.md)

TERREL 把数字高程模型采样到 CALMET 网格上，写出 MAKEGEO 将要读取的地形高度。Rasterio 解码 SRTM HGT 和 GeoTIFF。较旧的 CALMET 格式（USGS DEM、3CD、ARM3 等）由包内读取器处理；GDAL 对这些格式的识别不稳定。

```python
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
    x_origin_km=0.0,
    y_origin_km=0.0,
)

terrain = pycalmet.terrel(
    inp=None,
    grid=grid,
    srtm_dir=Path("data/terrain/srtm3"),
    sampling="bilinear",
    output=Path("output/terrain.nc"),
)
```

控制文件形式是 `pycalmet.terrel("terrel.inp")`。源基准与目标基准彼此独立：网格的 `datum` 是输出所在的基准，每个地形文件各自标明自己坐标的基准。

## 采样与后端

地形样本按 float64 累加和与整数命中数累计。这样，一个覆盖许多 SRTM 高程点的 1 km 格点不会因 float32 舍入丢掉平均值。收尾时在当前后端上用这些和做除法，并写出 float32 高度。空格点保留 TERREL 缺测值 `-999`。

最近邻搜索和岸线裁剪使用 SciPy 和 Shapely。极坐标空缺填补在主机上使用 `cKDTree`。选择 JAX 时加快的是格点平均这一步宽数组运算，空间索引仍在 CPU 上。

`sampling` 决定高程点如何落入格点。浙江案例对 SRTM3 使用 `bilinear`。TERREL 配置里另有保留峰值的开关：用样本最大值代替平均值，后端约定不变。

输出格式默认跟随文件名：`.nc` 为 NetCDF。只有设置了对应路径时，才写出 legacy SURFER 文件和 TERREL 保存文件。
