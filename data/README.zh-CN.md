# 原始地理空间数据

[English](README.md)

本目录将科学输入与 Python 包及可运行示例分开存放。允许再分发的地理空间
原始数据纳入版本管理；受限制的气象观测则按照可复现流程在本地获取：

- [`landuse/usgs-glcc/`](landuse/usgs-glcc/) 保存供 CTGPROC 使用的亚洲和欧洲
  USGS GLCC 区域栅格。浙江示例只读取亚洲栅格 `EAUSGS1.IMG`。
- [`terrain/srtm3/`](terrain/srtm3/) 保存 TERREL 为浙江区域读取的原始 SRTM3
  高程瓦片。
- [`observations/zhejiang-20180318/`](observations/zhejiang-20180318/) 定义浙江
  案例使用的 NCEI 官方地面与高空观测，包括获取 URL 和固定校验值。由于美国
  以外的 ISD 数据禁止再分发，观测字节只保留在本地。

`LULC.DAT`、`TERR.DAT` 和 `geo.dat` 等派生数据不应存放在这里。分阶段运行的
示例可能会将它们对应的 NetCDF 文件写入 `output/`。三个全参数浙江运行脚本会
通过各自私有临时目录中的 NetCDF 文件传递这些数据，只保留最终的 CALMET 文件。

每个源数据目录均包含格式、来源和 SHA-256 校验信息。
