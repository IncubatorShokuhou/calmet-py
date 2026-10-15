# 2018 年 3 月 18 日浙江观测数据

[English](README.md) · [数据总览](../../README.zh-CN.md)

本目录定义浙江纯观测案例与 WRF+观测融合案例使用的真实地面和高空观测。
观测字节由用户直接从 NOAA NCEI 获取，并只保存在本地；仓库仅提交本文档和
机器可读的来源、校验预期文件 [`acquisition.json`](acquisition.json)。整个
流程不会把 WRF 输出伪装成观测值。

## 再分发限制与本地获取

NCEI 的 [ISD 获取点说明](https://www.ncei.noaa.gov/pub/data/noaa/readme.txt)
明确写道：“The non-U.S. data in ISD are subject to WMO Resolution 40
restrictions, and cannot be redistributed to other users or customers.”
因此，本项目不能提交中国站点的 ISD 记录，也不能提交由这些记录生成的
CALMET 观测文件。IGRA 的说明将 `access/` 中的文件称为面向公共访问的数据，
但没有在说明中给出明确的再分发授权，所以探空数据也采用相同的官方直取流程。

从 NCEI 获取数据并校验已固定的历史内容：

```bash
python tools/fetch_zhejiang_observations.py
python tools/fetch_zhejiang_observations.py --check
```

第一条命令就是完整准备流程：它把数据下载到 Git 忽略的 `raw/`，校验全部
来源，再在同样被 Git 忽略的 `calmet/` 中生成 `SURF.DAT`、4 个 `UP_*.DAT`
和清单。第二条命令不联网，对两层文件同时执行 SHA-256 检查。如果校验失败，
必须先审查 NCEI 上游修订，再更新 [`acquisition.json`](acquisition.json)，
不能静默接受变化。

排错时可用 `--fetch-only` 在获取 `raw/` 后停止，或用 `--prepare-only` 从已
校验的本地来源重建 `calmet/`。获取器会自动把 10 个 ISD 来源、4 个 IGRA
切片、站名、有来源的站高覆盖值和固定站点目录传给
`tools/build_zhejiang_observations.py`，用户不需要手工拼接参数。

## 时段与目录结构

CALMET 案例时段是 `[2018-03-18 00:00, 01:00) UTC`。地面预处理从 2018 年
年度 ISD 归档选择该逐小时时段的记录；高空插值由 3 月 18 日 00 UTC 和
12 UTC 两次探空夹定。

```text
zhejiang-20180318/
├── acquisition.json             # 纳入版本管理的官方 URL 与校验预期
├── README.md
├── README.zh-CN.md
├── raw/                         # 仅本地；获取脚本生成
│   ├── manifest.json            # 实际 URL、HTTP 元数据、大小与 SHA-256
│   ├── surface/                 # 10 个完整的 2018 年 ISD .gz 归档
│   └── upper_air/
│       ├── archives/            # 4 个完整 IGRA 全时段 ZIP
│       ├── *-20180318-00-12Z.txt# 预处理实际读取的原字节切片
│       ├── igra2-station-list.txt
│       └── igra2-station-list-frozen-f9b7fd27.txt # 站高来源
└── calmet/                      # 仅本地；预处理脚本生成
    ├── SURF.DAT
    ├── UP_CHM00058362.DAT
    ├── UP_CHM00058457.DAT
    ├── UP_CHM00058633.DAT
    ├── UP_CHM00058665.DAT
    └── manifest.json
```

`raw/` 保存来源字节；`calmet/` 保存
`tools/build_zhejiang_observations.py` 确定性生成的 CALMET 2.1 文本输入。
后者是派生的模型输入，不是新增观测。两个目录均被 Git 明确忽略。

## 地面观测

标准来源是年度 gzip 压缩固定宽度的
[Integrated Surface Database](https://www.ncei.noaa.gov/pub/data/noaa/2018/)。
固定宽度文件保留每条报告自身的站点元数据，这对 584570 尤其关键：同一归档中
既有位于 30.233° N、120.167° E、海拔 43 m 的杭州 `FM-12` 报告，也有位于
30.228° N、120.432° E、海拔 7 m 的萧山 `FM-15` 报告。当时间距离和数据完整度
相同时，预处理优先选择天气学同步观测 `FM-12`；NCEI 日期服务 CSV 会把两条
记录都归一成萧山元数据，从而丢失这一差别。

| USAF-WBAN | 站点 | 归档字节数 | SHA-256 |
| --- | --- | ---: | --- |
| `582390-99999` | 栎社 | 213,626 | `780dd554b1204df3b0dfebd59c6c16bc0406c2c0d6c73906ebca5aa9c12784fd` |
| `583670-99999` | 上海虹桥国际机场 | 363,795 | `721df51fe64c12bdaede856fab342c003a8912c0e3dc8ea0f17a4b8254f974e8` |
| `584570-99999` | 杭州 / 萧山 | 544,427 | `5973f915b6644bca1f1e0ac67288d826913a33a23dece51096c9243544cbdeea` |
| `584770-99999` | 定海 | 119,031 | `f1735adb425034f82156f24940b971eadb0da7b20514c2a09335165bce6e7fe3` |
| `585560-99999` | 嵊县 | 103,612 | `f818b19dcbfdb972f859d2078b7e6ce2e981452aa63d4f9aa0a44a7fb1a48195` |
| `585690-99999` | 石浦 | 106,193 | `e027723db4dc0431ce21b1929c1019c4eae96ab5f2cc61d2f55084c154ba3fc2` |
| `586330-99999` | 衢县 | 74,493 | `57048dac01aaea971e4c7b96ea2aca3f01ad1fdc6243d2cf542530c9f7debf74` |
| `586460-99999` | 丽水 | 99,265 | `4d58c6cbf803841f6f686644bf8ce142d2d44cdd1daafa2513dbd275e57866bd` |
| `586660-99999` | 大陈岛 | 120,802 | `749d200ac7658cd8cc5fdd84dbf9b8c92c020415dc30af09c500d995611383b7` |
| `587520-99999` | 瑞安 | 117,648 | `5ac9db168e547fa98dd149eb41441c354b904aca79cd4910c56f4b60a0958248` |

每个文件的 URL 都是
`https://www.ncei.noaa.gov/pub/data/noaa/2018/{USAF}-99999-2018.gz`。
获取脚本先校验整个压缩归档，再检查其 ISD 站号以及是否含 3 月 18 日记录。

## 高空观测

来源是 NCEI 的
[Integrated Global Radiosonde Archive v2.2](https://www.ncei.noaa.gov/pub/data/igra/)。
脚本下载每站全时段 ZIP，然后逐字节复制 2018-03-18 00 UTC 与 12 UTC 的完整
固定宽度记录块。随着 NCEI 追加新探空，全时段归档可能变化，因此其摘要用于
标识参考版本；真正强制复现的是所选历史记录切片的摘要。

原始切片保留 IGRA 中全部有效层。转换为 CALMET `UP.DAT` 时只写出海拔低于
9,998 m MSL 的层，因为官方 CALMET 6.5 的 `RDUP` 读取器把所有不低于
9,998 m 的高度保留为缺测码。输出廓线仍达到 9.45–9.53 km MSL，明显高于本
案例 3 km 的模式顶。转换清单逐次记录省略层数、输出风场顶高和温度顶高。

| IGRA ID | 站点 | 用途 | 参考 ZIP SHA-256 | 所选切片 SHA-256 |
| --- | --- | --- | --- | --- |
| `CHM00058362` | 上海宝山 | 位于网格北界外约 1.6 km，用于约束北部边缘 | `33a650675ce719d0a8b4d6b6ebf68fe1d266ed3c4a0d2bc64fee432cd7bffdef` | `554fff21adabac78cdf380d89b7e9c84ea0d8bbb5d18e88e651fc5476447eb12` |
| `CHM00058457` | 杭州 | 域内 | `2c719551602eb0dcb0cf291bdf220e3682199c03133f51fab39f71d10c9c11b2` | `06bfe5a02a691370c8f8a5f28efd0b2d36cdf7dc20762410ae2babce5d35e5ad` |
| `CHM00058633` | 衢县 | 域内 | `a0e4abf998cac2a48ad75c9958b940891068c0800be344185a37e8e13550d955` | `f77a15a5b96fb73984c442f60c558135813471dae49e681acff846c70042b761` |
| `CHM00058665` | 洪家 | 域内 | `5a571097b9e6e3384d8754479a2fd559ae23b9dbc10d0f4f1368cf3e6ddcc83b` | `6400722e91d6b074f0c745214ee6ef97bb30d021438174711d7f15e185c9255b` |

每站全时段归档 URL 是
`https://www.ncei.noaa.gov/pub/data/igra/data/data-por/{IGRA_ID}-data.txt.zip`。
脚本还会下载 IGRA 实时站点表。预处理来源清单明确记录宝山、杭州、衢县和洪家
海拔分别为 9.0、43.0、82.9 和 4.6 m MSL；当前 NCEI 实时站点表缺失前三个
海拔。这些值只补充站点元数据，不会改动任何实测廓线值。

这些海拔可由 `Matthieu-Lacroix/igra2_interactivemap` 镜像在提交
[`f9b7fd27adc6c9cee0ba1aa653cf4c99caf91ce6`](https://github.com/Matthieu-Lacroix/igra2_interactivemap/blob/f9b7fd27adc6c9cee0ba1aa653cf4c99caf91ce6/igra2-station-list.txt)
保存的旧版 NCEI 站点目录审计。该目录为 260,147 字节，SHA-256 为
`d1deb3396f6b8b5d8d49c8338d2ac62a842fed1c7d06fd7723890a6f5ce669c5`。
获取器在本地下载并校验这一固定版本，预处理清单也将它记录为站高来源。
在生成任何 `UP_*.DAT` 之前，包装脚本还会逐站核对固定列目录：坐标容差为
`0.00005°`，海拔容差为 `0.05 m`。

## 来源边界

仓库内校验值标识本案例已审查的来源版本。`raw/manifest.json` 记录实际获取的
字节，包括重定向目标、ETag 和 NCEI 提供的 `Last-Modified`；
`calmet/manifest.json` 记录所有预处理选择和输出摘要。科学结果应在本地保留
这两份清单，但不能再分发含观测值的目录。
