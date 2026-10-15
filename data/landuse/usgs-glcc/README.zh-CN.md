# USGS GLCC 土地利用栅格

[English](README.md) · [数据总览](../../README.zh-CN.md)

本目录保存两份 USGS Global Land Cover Characterization 2.0 原始栅格。文件没有
头信息，每个格点以一个字节表示；CTGPROC 根据读取器名称赋予固定的 1 km
兰伯特方位等积投影几何，而不是从文件头读取投影。

| 本地文件 | 公开来源图像 | CTGPROC 读取器及固定几何 | 大小 | SHA-256 |
| --- | --- | --- | ---: | --- |
| `EAUSGS1.IMG` | `eausgs2_0la.img` | `GLAZAS`；13,000 × 12,000；北纬 45°、东经 100° | 156,000,000 字节 | `4f2a1977d141e2040da4d36ec6a4826c5c960c84686eaf10f226f388cb3ab90a` |
| `EAUSGS2.IMG` | `eausgs2_0le.img` | `GLAZEU`；13,000 × 13,000；北纬 55°、东经 20° | 169,000,000 字节 | `0fafa581ece7b23b595aa895832e94034c0e93e4efaf6aef60fbcc6be57860da` |

两者都采用 `ESR-S` 球面基准。这两份图像分别解压自公开第三方镜像
`ydopku/parallel-CALPUFF` 中固定版本的
[`eausgs2_0la.7z`](https://raw.githubusercontent.com/ydopku/parallel-CALPUFF/aa04af6399cc5080a7776d92dec7316acb3488c6/CalpuffWorkerG/eausgs2_0la.7z)
和
[`eausgs2_0le.7z`](https://raw.githubusercontent.com/ydopku/parallel-CALPUFF/aa04af6399cc5080a7776d92dec7316acb3488c6/CalpuffWorkerG/eausgs2_0le.7z)。
本项目计算得到的两个压缩包 SHA-256 依次为
`2ae72d2824bb0c5ba9ad9690563fc7d556d8b29446713861a5988e1dda6e2e16` 和
`6a56fb920f3c77ca8a1054f7e8fa287859a9b47e76b5a7caefc831e1dc57333a`。

`EAUSGS1.IMG` 和 `EAUSGS2.IMG` 是旧工程使用的本地名称，并非 CTGPROC 的标准
标识。旧文件曾被 Git 忽略，也没有留下校验值。亚洲文件的映射具有很强的交叉
验证：历史 CTGPROC 实现使用该图像得到 304,390 个有效浙江格点和 43,548,891
个样本，与旧记录中的两项统计完全一致。欧洲文件与亚洲文件成对公开，是目前对
所需 `EAUSGS2.IMG` 最可靠的恢复来源。由于缺少旧文件哈希，无法严格证明恢复文件
与两个旧本地文件逐字节相同。

浙江控制文件仍设置 `NDBF=1`，只将 `EAUSGS1.IMG` 作为 `GLAZAS` 读取。
`EAUSGS2.IMG` 作为 `GLAZEU` 配套原始数据保留，不参与浙江示例计算。

两份 `.IMG` 文件均使用 Git LFS 存储。克隆后，如果任一文件仍是 LFS 指针，请
运行 `git lfs pull`。在本目录执行以下命令可校验两份实际下载的文件：

```bash
sha256sum -c SHA256SUMS
```
