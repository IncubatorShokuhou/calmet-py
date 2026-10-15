# Zhejiang end-to-end examples

[Project documentation in Chinese](../../README.zh-CN.md) ·
[Observation data](../../data/observations/zhejiang-20180318/README.md) ·
[观测数据中文说明](../../data/observations/zhejiang-20180318/README.zh-CN.md)

This directory provides three runnable cases on the same Zhejiang grid and for
the same CALMET interval, 2018-03-18 00:00–01:00 UTC:

| Runner | Meteorological inputs | Main CALMET controls | Final output |
| --- | --- | --- | --- |
| `run.py` | WRF only | `NOOBS=2`, `IPROG=14` | `output/zhejiang/calmet.nc` |
| `run_observations.py` | Zhejiang and nearby surface (`SURF`) + upper-air (`UP`) observations | `NOOBS=0`, `IPROG=0`, `MCLOUD=1` | `output/zhejiang/calmet-observations.nc` |
| `run_wrf_observations.py` | WRF + the same `SURF` + `UP` observations | `NOOBS=0`, `IPROG=14`, `MCLOUD=4` | `output/zhejiang/calmet-wrf-observations.nc` |

All three use typed Python parameters and rebuild the same `610 × 499`, 1 km
LCC geophysical grid from original terrain and land-use inputs. The
observation-only case does not manufacture a gridded meteorological input; it
passes `meteorology=None` and analyzes the real station reports directly.

## Time range and intended use

The modern WRF runners read the 00:00 and 01:00 UTC files. CALWRF includes
both boundary times, which bracket CALMET's half-open `[00:00, 01:00)` output
interval. The repository also retains the 02:00 WRF record only for the
official Fortran comparison: the reference CALWRF/CALMET path eagerly reads
one additional boundary record. The modern runners do not read 02:00.

This one-hour setup is a data-path and regression case. It begins at 08:00
China Standard Time, after sunrise, and contains no preceding nocturnal-to-
morning boundary-layer history. A production planetary-boundary-layer study
should start with sufficient WRF and observation history from the preceding
night or before dawn so that the diagnostic boundary-layer calculation has an
appropriate spin-up.

## Shared terrain and land use

The runners build their GEO input instead of reading a prepared `geo.nc`:

- 42 original SRTM3 tiles under `data/terrain/srtm3/`, covering N26–N31 and
  E117–E123. Each `.hgt` file is a `1201 × 1201` signed 16-bit elevation grid.
- `data/landuse/usgs-glcc/EAUSGS1.IMG`, the original `13000 × 12000`
  headerless `uint8` Asia raster. CTGPROC reads it as `GLAZAS`, with the
  `ESR-S` source datum and `mesh_density=12`; the target CALMET grid uses
  `WGS-84`.

The companion `EAUSGS2.IMG` is the Europe-optimized `GLAZEU` archive. It is
retained as source data, but this Zhejiang grid uses one Asia database
(`NDBF=1`) and does not read it.

Pull and verify the repository data from the repository root:

```bash
git lfs pull
sha256sum -c examples/zhejiang/SHA256SUMS
cd data/terrain/srtm3 && sha256sum -c SHA256SUMS && cd ../../..
cd data/landuse/usgs-glcc && sha256sum -c SHA256SUMS && cd ../../..
```

## Prepare the local observations

NCEI states that non-U.S. ISD data are subject to WMO Resolution 40 and cannot
be redistributed to other users or customers. Consequently, this repository
does not contain the Chinese observation bytes or the CALMET `SURF.DAT` and
`UP_*.DAT` files derived from them. It contains official NCEI URLs, pinned
checksums, station selections, and deterministic acquisition and conversion
tools. Raw and converted observation files remain local and are ignored by
Git.

Fetch, verify, and convert the official records first:

```bash
python tools/fetch_zhejiang_observations.py
python tools/fetch_zhejiang_observations.py --check
```

The first command performs the complete preparation workflow. It downloads
and validates `raw/`, then creates
`data/observations/zhejiang-20180318/calmet/SURF.DAT`, four `UP_*.DAT` files,
and a conversion manifest. The second command checks both the acquired and
prepared files without downloading again. Troubleshooting modes and full
provenance are documented in
[`data/observations/zhejiang-20180318/README.md`](../../data/observations/zhejiang-20180318/README.md).
Neither the acquisition tool nor the converter uses WRF values as
observations.

The surface analysis uses these ten ISD stations:

| USAF-WBAN | Station | Location represented at 00 UTC |
| --- | --- | --- |
| `582390-99999` | Ningbo Lishe | Zhejiang |
| `583670-99999` | Shanghai Hongqiao | Shanghai |
| `584570-99999` | Hangzhou | Zhejiang; the selected `FM-12` report retains the Hangzhou site metadata |
| `584770-99999` | Dinghai | Zhejiang |
| `585560-99999` | Shengxian | Zhejiang |
| `585690-99999` | Shipu | Zhejiang |
| `586330-99999` | Qu Xian | Zhejiang |
| `586460-99999` | Lishui | Zhejiang |
| `586660-99999` | Dachen Dao | Zhejiang |
| `587520-99999` | Ruian | Zhejiang |

The upper-air analysis uses four IGRA soundings. The 00 and 12 UTC profiles
bracket the CALMET hour:

| IGRA ID | Station | Domain role |
| --- | --- | --- |
| `CHM00058362` | Shanghai (Baoshan) | About 1.6 km north of the grid; constrains the northern edge |
| `CHM00058457` | Hangzhou | In domain |
| `CHM00058633` | Qu Xian | In domain |
| `CHM00058665` | Hongjia | In domain |

The acquisition manifest records the official source URLs and hashes, the
four upper-air station coordinates, selection metadata, and redistribution
terms. `SURF.DAT` stores the ten surface-station coordinates and anemometer
heights. The local, Git-ignored `calmet/manifest.json` separately records each
selected report's station elevation and conversion metadata.

## Run the WRF-only case

Install the package and run the parameter-only five-stage pipeline:

```bash
python -m pip install -e .
./examples/zhejiang/run.sh
```

`run.py` declares both WRF boundary files, the terrain and land-use sources,
datetimes, shared `GridConfig`, and active CALMET controls as typed Python
values. It calls `pycalmet.run_pipeline(inp=None, ...)`; no `.inp` file is
read.

## Run the observation-only case

After preparing the local observations, run:

```bash
PYTHONPATH=src python examples/zhejiang/run_observations.py
```

This case uses observed surface and upper-air winds and temperatures, plus
observed surface relative humidity. `MCLOUD=1` reads surface cloud cover and
ceiling from `SURF.DAT`; preprocessing requires cross-station coverage for
both fields. `LVARY=T` allows the closest valid station to cover cells outside
the normal station radius. Because this case has no offshore station,
`JWAT1=JWAT2=999` lets land stations provide temperature and humidity over
water cells while retaining the GEO water categories.

## Run the WRF + SURF + UP fusion case

After preparing the same local observations, run:

```bash
PYTHONPATH=src python examples/zhejiang/run_wrf_observations.py
```

`IPROG=14` supplies the CALWRF three-dimensional wind as the initial guess,
then CALMET incorporates surface and sounding winds through objective
analysis. `ITPROG=0` uses the observed surface and sounding temperatures.
`IRHPROG=0` replaces near-surface relative humidity with `SURF.DAT`; because
these `UP` profiles do not contain humidity, layer humidity aloft remains the
WRF prognostic field. `MCLOUD=4` uses the WRF prognostic cloud path, with
cloud water and cloud ice forming the condensate mask; WRF also supplies
precipitation.

Each runner creates terrain, land-use, GEO, and, where needed, CALWRF NetCDF
datasets in a private temporary directory. It removes those intermediates at
completion and persists only the final NetCDF file listed in the first table.
The pipeline does not create traditional M3D or M2D files.

## Official CALMET 6.5 comparison

The observation-only and WRF-fusion cases were run to completion with the
reference executable identifying itself as **CALMET 6.5.0, Level 150223**.
Every comparison uses the same generated `SURF.DAT`, the same four `UP_*.DAT`
files, and the same active CALMET controls. Each Python audit parses the exact
control file used by its paired Fortran run and overrides only the input,
output, and listing paths. Python NetCDF records are aligned to the
**beginning** of the corresponding Fortran interval, and every difference is
`Python - Fortran`.

The current geophysical source and its legacy representation are pinned as:

- source `geo.nc` SHA-256:
  `cd45b10738cc6850037803c74db18906962f0aa402c7f96676108bc8d59a53c9`
- converted `GEO.DAT` SHA-256:
  `704cc04c78bf64eea216f18251460cc3bf9acea034443b5686e9362f9893e0fa`

For the static fields reported by the comparator, conversion preserves every
land-use category exactly. Terrain differs only through the decimal precision
of the legacy text format: RMSE is `0.002276321 m` and the maximum absolute
difference is `0.005004883 m`.

### Public end-to-end paths

This comparison exercises the data flow of the normal public runners. Python
reads the NetCDF
GEO produced by the current pipeline; the official run reads the converted
`GEO.DAT`. In the fusion case, Python also uses its direct CALWRF NetCDF data
flow while the official chain reads traditional `3D.DAT`. These results
therefore include both preprocessing and CALMET-core differences.

| Quantity | SURF + UP | WRF + SURF + UP |
| --- | ---: | ---: |
| U / V / top-face W RMSE (m s⁻¹) | 0.100953 / 0.106632 / 0.0419055 | 0.164373 / 0.170302 / 0.0532680 |
| Three-dimensional temperature RMSE (K) | 0.00746089 | 0.110104 |
| Surface relative-humidity RMSE (percentage points) | 0.0351462 | 0.0351462 |
| Mixing-height (`ZI`) RMSE (m) | 1.12635 | 19.5066 |
| Physical Monin–Obukhov length (`EL`) RMSE (m) | 2.57000 | 28.2135 |
| `EL` missing-mask disagreement | 0% (125,021 missing on each side) | 0% (125,021 missing on each side) |
| Pasquill–Gifford–Turner class mismatch | 0.0896876% | 4.00769% |
| Shortwave-flux RMSE (W m⁻²) | 0.000104270 | 61.9985 |

### Exact-shared CALMET core

A second audit reruns Python from the exact legacy `GEO.DAT` used by Fortran.
The fusion audit also supplies Python with the exact official `3D.DAT`. This
removes the GEO text conversion and CALWRF data-flow differences and more
closely isolates the CALMET implementations; terrain and land use are both
identical in this comparison.

| Quantity | SURF + UP | WRF + SURF + UP |
| --- | ---: | ---: |
| U / V / top-face W RMSE (m s⁻¹) | 0.100881 / 0.106618 / 0.0419025 | 0.162781 / 0.169146 / 0.0530793 |
| Three-dimensional temperature RMSE (K) | 0.00618395 | 0.107286 |
| Surface relative-humidity RMSE (percentage points) | 0.0351462 | 0.0351462 |
| Mixing-height (`ZI`) RMSE (m) | 1.11964 | 19.1837 |
| Physical Monin–Obukhov length (`EL`) RMSE (m) | 2.55763 | 26.2057 |
| `EL` missing-mask disagreement | 0% (125,021 missing on each side) | 0% (125,021 missing on each side) |
| Pasquill–Gifford–Turner class mismatch | 0.0841026% | 3.88679% |
| Shortwave-flux RMSE (W m⁻²) | 0.000104270 | 61.9323 |

RMSE uses every paired grid cell and every model layer for three-dimensional
fields. CALMET's approximately `9.9e9` undefined-`EL` sentinel is excluded
from the physical `EL` RMSE; its mask is compared separately. All reported
dynamic fields have zero missing-mask disagreement.

Reproduce the public-path statistics after retaining the corresponding
reference `CALMET.DAT` files:

```bash
python tools/compare_calmet_observations.py --netcdf-time begin \
  output/zhejiang/calmet-observations.nc \
  /path/to/official/obs-only/CALMET.DAT

python tools/compare_calmet_observations.py --netcdf-time begin \
  output/zhejiang/calmet-wrf-observations.nc \
  /path/to/official/wrf-observations/CALMET.DAT
```

The fusion shortwave difference has been traced to the prognostic-cloud
criterion. Sharing the official `3D.DAT` changes QSW RMSE only from
`61.9985` to `61.9323 W m⁻²`, so the main residual remains inside the CALMET
cloud treatment. The modern path horizontally maps each layer's
`QCLOUD + QICE`, takes the column maximum, and requires it to reach
`0.005 g kg⁻¹`; it finds the lowest qualifying source layer before mapping the
cloud ceiling. The historical path writes fixed-precision `3D.DAT` values,
sums liquid `QCLOUD` layer values, horizontally maps that column sum, and
treats a mapped total below `1e-6 g kg⁻¹` as clear. The cloudy/clear masks
differ over `3.31745%` of this grid, and those cells account for `99.722%` of
the QSW squared error. No cell became cloudy solely because of `QICE` during
this hour. These cloud-mask changes feed shortwave radiation and then the
stability and boundary-layer diagnostics.

Both cases set `ITPROG=0` and analyze observed surface and sounding
temperatures. The separate modern `ITPROG=2` continuous TEMP3D layer-overlap
algorithm is therefore not exercised here and does not explain the measured
temperature residuals. The remaining fusion temperature and boundary-layer
differences reflect the coupled CALMET calculation under the inputs described
above; this audit does not assign each residual to one routine.

The reference executable SHA-256 is
`21d847c6efba5e2564317c9ccbfd62907349febabb604d2c0bc590bb9475c756`.
Both runs returned status 0 with empty standard error, entered the termination
phase, and reported `LAST PERIOD PROCESSED` ending at 2018-03-18 01:00 UTC.
The uncommitted reference output hashes are:

- SURF + UP `CALMET.DAT`:
  `ea916fb5686011d5c9f6c4ad105c330ef10cd474f2ecd7d5c5b2ddd52c290dd1`
- WRF + SURF + UP `CALMET.DAT`:
  `dfae6ed206d1f3f2c3d72843cdb11d70c6994a61238bcb2e2936085409dd3577`

These measurements show that both compared workflows finish and make the
remaining field-level differences explicit. They are numerical regression
evidence and
do not establish bitwise or scientific equivalence between Python and the
Fortran implementation.

### 中文说明

对比分为两个层次。公开端到端对比使用正常示例链路：Python 读取当前
NetCDF GEO，官方程序读取由它转换出的 `GEO.DAT`；融合案例还同时包含
Python 直接 CALWRF NetCDF 链路与官方 `3D.DAT` 链路的差异。严格共享输入
对比则让 Python 与 Fortran 读取同一个 `GEO.DAT`，融合案例再共享同一个
官方 `3D.DAT`，因此更接近单独比较 CALMET 核心。

当前源 `geo.nc` 的 SHA-256 为
`cd45b10738cc6850037803c74db18906962f0aa402c7f96676108bc8d59a53c9`，
转换后 `GEO.DAT` 的 SHA-256 为
`704cc04c78bf64eea216f18251460cc3bf9acea034443b5686e9362f9893e0fa`。
土地利用类别完全一致；公开链路中的地形差异只来自传统文本格式精度，
RMSE 为 `0.002276321 m`，最大绝对差为 `0.005004883 m`。严格共享输入时，
地形和土地利用差异都为零。

Python 输出按时段起点与 Fortran 时段对齐，差值均为 `Python - Fortran`。
`EL` 的约 `9.9e9` 缺测哨兵不参与物理 RMSE；双方各有 125,021 个缺测网格，
掩膜差异为零。纯观测严格共享输入结果中，三维温度 RMSE 为
`0.00618395 K`、`ZI` RMSE 为 `1.11964 m`；融合结果分别为 `0.107286 K`
和 `19.1837 m`。融合案例的 PGT 类别差异为 `3.88679%`，短波辐射 RMSE
为 `61.9323 W m⁻²`，仍属于需要明确保留的实质差异。

融合短波差异主要来自云凝结物判据。即使共享官方 `3D.DAT`，QSW RMSE 也
只从 `61.9985` 变为 `61.9323 W m⁻²`，主体仍来自 CALMET 内部云场处理。
现代链路先水平映射各层 `QCLOUD + QICE`，再取列最大值并要求其达到
`0.005 g/kg`；云底则由源网格最低达标层确定后再映射。历史链路累加各层
液态 `QCLOUD`，再水平映射列和，映射值低于 `1e-6 g/kg` 时视为晴空。
两种云掩膜在 `3.31745%` 的网格上不同，这些网格贡献了 `99.722%` 的 QSW
平方误差；本时段没有网格仅因加入 `QICE` 而变为有云。云掩膜变化会继续
影响短波、稳定度和边界层诊断。本例的温度设置是 `ITPROG=0`，因此没有
启用单独的现代 `ITPROG=2` TEMP3D 连续层重叠算法，不能把表中的温度差异
归因于该算法。

官方两次运行均以状态 0 正常进入终止阶段，日志处理到
2018-03-18 01:00 UTC；官方输出 SHA-256 已列在上方。这些结果证明两条
对比链路都能完成，并量化了当前差异；它们不能证明 Python 与 Fortran 在
位级或大气科学结果上等价。

## Step-by-step WRF debugging

To retain and inspect every WRF-only stage, run the separate control-file
example:

```bash
PYTHONPATH=src python examples/zhejiang/run_stages.py
```

It reads `config/calwrf.inp`, `terrel.inp`, `ctgproc.inp`, `makegeo.inp`, and
`calmet.inp` in sequence, and retains their configured products under
`output/zhejiang/`. The three primary runners above use typed parameters; this
step-by-step runner exists to demonstrate and debug the traditional `.inp`
interface.
