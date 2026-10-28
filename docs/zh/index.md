# PyCALMET 用户指南

[English](../index.md)

PyCALMET 把 CALMET 预处理流程做成带类型的 Python 包。本指南说明如何安装、
NumPy 与 JAX 后端有何不同，以及每个处理阶段期望什么输入。体例与 scikit-learn
用户指南相同：短叙述、最小示例，并指向公开 API。参数的完整列表不在这里。

参数参考仍在仓库的 `README.zh-CN.md`。配置新模拟区域或选择数组后端时，从本指南开始。

## 目录

1. [安装](install.md) — 默认安装、`gpu` extra 与开发环境
2. [数组后端](backends.md) — 默认 NumPy、可选 JAX、设备与浮点类型
3. [端到端流水线](pipeline.md) — `run_pipeline`，以及各阶段交换的文件
4. [CALWRF](calwrf.md) — 从 WRF NetCDF 到 CALMET 气象场
5. [TERREL](terrel.md) — CALMET 网格上的地形高程
6. [CTGPROC](ctgproc.md) — 分类土地利用到各类命中数
7. [MAKEGEO](makegeo.md) — 由地形和土地利用得到地理物理场
8. [CALMET](calmet.md) — 诊断风、混合层高度与最终数据集

```{toctree}
:maxdepth: 2
:caption: 用户指南

install
backends
pipeline
calwrf
terrel
ctgproc
makegeo
calmet
```

上面的 `toctree` 供 Sphinx 读取。在 GitHub 上请使用编号列表。

## 生成 HTML

在仓库根目录安装文档依赖后，与英文指南同一次构建。中文页面在 `zh/` 下：

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

打开 `docs/_build/html/zh/index.html`。同一次构建的英文页面在
`docs/_build/html/index.html`。把警告当作错误时使用
`sphinx-build -W -b html docs docs/_build/html`，源目录仍是 `docs/`。
HTML 构建目录已被 Git 忽略。
