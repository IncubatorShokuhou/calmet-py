# 浙江区域 SRTM3 数据子集

[English](README.md) · [数据总览](../../README.zh-CN.md)

本目录仅包含 `examples/zhejiang/config/terrel.inp` 引用的 42 个 SRTM3
瓦片：纬度范围为北纬 26° 至 31°，经度范围为东经 117° 至 123°。每个文件均为
采用标准 SRTM `.hgt` 布局的 1201 × 1201 有符号 16 位高程网格。

运行 `cd data/terrain/srtm3 && sha256sum -c SHA256SUMS` 可校验这些文件。
完整的本地 SRTM 数据库有意排除在 Git 之外。
