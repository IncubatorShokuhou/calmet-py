# 原始地理空间数据

[English](README.md)

本目录将纳入版本管理的原始地理空间数据与 Python 包及可运行示例分开存放：

- [`landuse/usgs-glcc/`](landuse/usgs-glcc/) 保存供 CTGPROC 使用的亚洲和欧洲
  USGS GLCC 区域栅格。浙江示例只读取亚洲栅格 `EAUSGS1.IMG`。
- [`terrain/srtm3/`](terrain/srtm3/) 保存 TERREL 为浙江区域读取的原始 SRTM3
  高程瓦片。

`LULC.DAT`、`TERR.DAT` 和 `geo.dat` 等派生数据不应存放在这里。分阶段运行的
示例可能会将它们对应的 NetCDF 文件写入 `output/`，而
`examples/zhejiang/run.py` 会在内存中传递这些数据，只保留最终的 CALMET
文件。

每个源数据目录均包含格式、来源和 SHA-256 校验信息。
