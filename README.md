# PyCALMET：浙江 WRF → CALMET 动力降尺度

本仓库的主要交付物是无需编译 Fortran 的 `pycalmet` 包。旧 Fortran 源码和
GCC/GFortran 脚本保留为参考实现及数值回归基准。流程是：WRF NetCDF → CALWRF →
TERREL 地形 + CTGPROC 土地利用 → MAKEGEO 地理场 → CALMET 1 km 风场。
浙江默认网格沿用原脚本：LCC 投影，左下角 `27N,117.5E`，`610 × 499`，
间距 1 km，时间步 1 小时。

安装后可直接调用五个公开阶段；关键字可使用对应 `.lst` 变量名，大小写不敏感：

```bash
python -m pip install .
```

```python
import pycalmet

calwrf_path = pycalmet.calwrf(**lst_params_calwrf)
terrain_path = pycalmet.terrel(**lst_params_terrel)
landuse_path = pycalmet.ctrproc(**lst_params_ctrproc)
geophysical_path = pycalmet.makegeo(**lst_params_makegeo)
calmet_path = pycalmet.calmet(**lst_params_calmet)
```

`pycalmet.readlst(path)` 可把任一阶段的 `.inp` 或 `.lst` 解码成可直接展开的参数
字典；原始 `.lst` 变量字典也可直接展开。推荐使用可由
`xarray.open_dataset(path)` 打开的 NetCDF 主输出；显式传统格式的行为见下文。
也可把控制文件路径作为第一个位置参数；旧名 `pycalmet.ctgproc(...)` 保留为兼容
别名。

## 1. Fortran 参考环境

重新生成 Fortran 参考结果时需要 `gcc`、`gfortran`、`csh`、NetCDF-C/Fortran
（`nc-config`、`nf-config`）和 `unzip`。无需 PGI；旧 PGI 目录和旧二进制保存在
`deprecated/pgi-legacy/`。

## 2. 静态输入数据

仓库随附浙江计算域实际使用的 42 个 SRTM3 瓦片，位于 `TERREL/SRTM3/`：纬度
N26–N31、经度 E117–E123。`run_zhejiang.sh` 默认使用该目录；也可用
`--srtm-dir` 指向另一份同名瓦片。`TERREL/SRTM3/SHA256SUMS` 可校验数据完整性。

用于从头生成土地利用场的原始 `EAUSGS1.IMG` 不随仓库分发。Python 入口用
`--landuse` 指向 CTGPROC 可读的土地利用文件；Fortran 入口从
`CTGPROC/EAUSGS1.IMG` 读取。浙江控制参数默认按 `GLAZAS`、`ESR-S` 和网格密度
12 处理。仓库同时提供已生成的 `MAKEGEO/geo.dat` 和 `example/zhejiang/geo.nc`，
用于 `--skip-static` 示例。

## 3. GNU 编译

```bash
cd calmet
./build_gnu.sh
```

会生成 `CALWRF/calwrf.exe`、`TERREL/terrel.exe`、
`CTGPROC/ctgproc.exe`、`MAKEGEO/makegeo.exe` 和
`calmet_mpi/calmet.exe`。老代码有跨类型实参，因此使用
`-std=legacy -fallow-argument-mismatch`；CALMET 另外使用 `-fno-automatic`
复现原 PGI `-Msave` 的局部变量持久化行为，并启用 OpenMP。运行脚本还会提高
栈限制，因为该旧程序含有较大的局部数组。

## 4. 准备浙江 WRF 输入

CALWRF 要求标准 WRF NetCDF。仓库中的
`example/zhejiang/` 含真实浙江 WRF 输出的 2018-03-18 00:00、01:00 和 02:00 UTC
三条记录，保留 CALWRF 读取的全部变量，并使用无损压缩。它们的 CALWRF 结果与
3.0 GiB 原文件的对应时次逐字节一致；第三条记录为 00:00–01:00 CALMET 运行提供
结束时刻后的预报边界。可用 `tools/make_zhejiang_example.py` 从完整 WRF 文件重新
生成这些样例。

历史 ZJWARMS/NCL 局部替换流程仅供问题追溯，不参与当前流程。其他输入也应先用
`ncdump -h 文件` 确认可读，并按文件中的 `Times` 选择 UTC 起止时刻。

## 5. 一键运行

```bash
./run_zhejiang.sh \
  --wrf example/zhejiang \
  --start 2018031800 --end 2018031801 \
  --jobs 8 \
  --skip-static \
  --output ./output/zhejiang_example_fortran.dat
```

`--wrf` 可指向单个文件或包含多个 `wrfout*` 的目录。起止时间均为 UTC，格式
`YYYYMMDDHH`，并且必须落在 WRF `Times` 范围内。`--skip-static` 复用随附的
`MAKEGEO/geo.dat`；省略该选项会从 SRTM 瓦片和外部土地利用文件重建静态场。
两个一键入口都会为末端时间平均自动多读取 `--end` 之后一个小时的 WRF 记录。

若另行准备了未经 NCL 修改的 13 时次原始 WRF 文件，可这样验证完整时段：

```bash
./run_zhejiang.sh \
  --wrf /path/to/wrfout_d01_2018-03-18_00_00_00.nc \
  --start 2018031800 \
  --end 2018031812 \
  --jobs 8 \
  --skip-static \
  --output output/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat
```

历史 5 月混合文件和相应 NCL 脚本保存在 `diagnostics/legacy/zj2wrfout_ncl/`，
只供问题追溯，不应作为 NumPy 重写的基准输入。

## 6. 检查结果

每阶段成功时退出码为 0。失败先查看对应日志：

- `CALWRF/calwrf.log` 与 `CALWRF/calwrf.lst`
- `TERREL/terrel_check.log`、`TERREL/terrel.log` 与 `TERREL/TERR.LST`
- `CTGPROC/ctgproc.log` 与 `CTGPROC/CTGPROC.LST`
- `MAKEGEO/makegeo.log`、`MAKEGEO/makegeo.lst` 与 `MAKEGEO/qa.geo`
- `calmet_mpi/calmet.log`、`calmet_mpi/calmet.lst` 与 `calmet_mpi/qa.met`

上述 Fortran 入口的最终文件是 `--output` 指定的 CALMET 二进制数据。旧 NetCDF
后处理示例已移至 `diagnostics/legacy/output_pro/`；其中网格和时次数写死，不参与
当前运行链。

## 7. 对比可视化

`diagnostics/code/visualize_downscaling.py` 可直接读取 WRF NetCDF 和 CALMET
Fortran 二进制文件，
并使用相同范围、相同色标比较地形和 10 m 风场。例如绘制随附样例：

```bash
python3 diagnostics/code/visualize_downscaling.py \
  --wrf example/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc \
  --calmet output/zhejiang_example_fortran.dat \
  --hour 0 \
  --output output/zhejiang_example_wind.png
```

`--hour` 从 0 开始；图中 CALMET 第一层位于 0–20 m 层中心，即 10 m，可与
WRF 的 `U10/V10` 直接进行同高度的定性比较。

温度、相对湿度、混合层高度和逐小时降水对比：

```bash
python3 diagnostics/code/visualize_other_fields.py \
  --wrf example/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc \
  --calmet output/zhejiang_example_fortran.dat \
  --hour 0 \
  --output output/zhejiang_example_fields.png
```

其中 WRF `T2` 与 CALMET `TEMPK`、WRF `PBLH` 与 CALMET `ZI` 的物理定义并不
完全相同，适合检查空间结构，不应直接视为同一变量的误差验证。所有诊断代码、
图表、参考输出和旧后处理工具均集中在 `diagnostics/`，不会参与主运行流程。

## 8. 纯 Python 路径

无需编译 Fortran 即可安装当前 NumPy/SciPy 实现：

```bash
python -m pip install .
```

浙江现有 `NOOBS=2, IPROG=14, IRHPROG=1, ITPROG=2, IMIXH=1,
IAVET=1, IAVEZI=1` 分支已经可以从 Python CALWRF NetCDF 和 MAKEGEO
NetCDF 直接生成 CALMET NetCDF 或传统 `.dat` 文件：

```python
import pycalmet

grid = pycalmet.GridConfig(
    nx=610, ny=499, dx_km=1.0, dy_km=1.0,
    lat0=27.0, lon0=117.5, lat1=30.0, lat2=60.0,
)

pycalmet.calmet(
    meteorology="output/calwrf.nc",
    geophysical="output/geo.nc",
    output="output/calmet.nc",  # 改成 .dat 可写传统 CALMET 文件
    start="2018-03-18_00:00:00",
    end="2018-03-18_12:00:00",
    grid=grid,
)
```

也可以直接把原组件的 `.inp` 或对应 `.lst` 交给公开函数。列表文件若只含运行
摘要，解析器会跟随其中的 `Control File Used` 找回原控制文件；CALMET 这类完整
回显则直接解析。Python 阶段使用 NetCDF 中间文件时，只需覆盖旧控制文件中的
M3D/GEO 路径：

```python
params = pycalmet.readlst("controls/calmet.lst")
params.update(
    meteorology="output/calwrf.nc",
    geophysical="output/geo.nc",
    output="output/calmet.dat",
)
pycalmet.calmet(**params)
```

同样支持直接以关键字调用 `pycalmet.calwrf(...)`，以及把仓库内受版本控制的
`TERREL/terrel.inp`、`CTGPROC/ctgproc.inp` 和 `MAKEGEO/makegeo.inp` 交给相应
阶段。列表文件可来自用户自己的历史运行。例如
`pycalmet.ctrproc("controls/ctgproc.lst", ...)`（旧拼写
`pycalmet.ctgproc(...)` 仍可用）和
`pycalmet.makegeo("controls/makegeo.lst", ...)`。相对文件名始终以实际控制文件
所在目录解析，避免调用进程工作目录改变输入含义。
CALWRF 覆盖 `output` 时只写该 NetCDF；控制文件原有的 M3D/M2D 路径会自动禁用，
避免短时审计覆盖完整传统产物。需要同时写传统格式时应显式传入
`legacy_output="output/calwrf.m3d"`（以及可选的 `surface_output`）。

无需编译 Fortran 的浙江一键入口为：

```bash
./run_zhejiang_python.sh \
  --wrf example/zhejiang \
  --geo example/zhejiang/geo.nc \
  --start 2018031800 --end 2018031801 \
  --skip-static \
  --output output/zhejiang_example_python.nc
```

省略 `--skip-static` 时，脚本依次调用五个 Python 组件，并把 NetCDF 中间场放入
`--work-dir`；此时必须传入 `--srtm-dir` 和 `--landuse`。使用 `--skip-static` 时，
`--geo` 可指向已有的传统 `GEO.DAT` 或 canonical `geo.nc`；未指定时仍默认复用
`work-dir/geo.nc`。脚本从 `calmet_mpi/calmet_zhejiang.inp` 读取与 GNU Fortran
入口相同的 CALMET 控制参数。它同时保留 canonical `calwrf.nc`，并用 Python 写出的
传统 `calwrf.m3d` 驱动 CALMET，以复现 GNU 路径的接口量化和逐位结果；整个流程只
调用 `pycalmet` 的 NumPy/SciPy 实现。

随附样例的 00:00–01:00 UTC 验证中，Python 与 GNU CALMET 的 31 个输出字段、
9,436,090 个数值全部逐位一致，最大绝对差为 0；详细计数见
`example/zhejiang/README.md`。

该入口会严格检查首尾边界、配置的预报时次以及旧程序规定的参数组合；无法可靠
重建坐标信息的早期文件格式和未知输入类型会明确停止，不会静默换用近似算法。
各组件已覆盖的输入格式、物理分支和逐字段数值回归见 `PYTHON_PORT_STATUS.md`。

## 9. Known deviations

- Python CALWRF 会汇总所有输入文件后按时间排序、去重，并要求请求区间首尾完整且
  每小时连续。旧版 Fortran 按文件和记录顺序处理，文件内部的缺时次检查不完整。
- 多个 WRF 文件之间，Python 默认连续传递总降水、对流降水和格点降水的累计基线。
  仓库随附的 Fortran CALWRF 也保留总降水 `PPOLD`，因此拆分文件与相同时次存放在
  单个文件中的结果一致。原始 CALWRF 2.0.2 会在新文件开始时只清零 `PPOLD`、却
  保留两个分量基线；需要与这种未修正的历史输出逐值比较时，可设置
  `legacy_file_rain_reset=True`，只在文件边界重置总降水基线；对于按时间排列且无
  重叠时次的常规文件序列，它会精确复现该行为。canonical NetCDF 会用同名全局
  属性记录所用模式。
- canonical NetCDF 支持 WRF 新增的 `MAP_PROJ>=4` 非旋转网格；CALWRF 2.0.2 的
  传统 M3D/M2D 协议只定义 1（LCC）、2（PS）和 3（EM），因此请求传统输出时仍会
  明确拒绝其他投影编号。
- `IOSRF=0` 时 Python 不创建 M2D。旧源码仍会向未打开的 Fortran unit 11 写数据，
  某些运行时会意外生成一个没有合法头部的 `fort.11` 文件。
- 为避免一次装入十几 GB 的 WRF 文件，CALWRF 核心通过 netCDF4 分时次流式读写，
  而不是由 xarray 执行实际 I/O；写出的 canonical NetCDF 可直接用
  `xarray.open_dataset()` 打开并保留命名维度、单位和阶段元数据。
- TERREL、CTGPROC（Python API 为 `ctrproc`）和 MAKEGEO 在临时文件完整写出后
  原子替换主输出；旧 Fortran 程序直接写目标文件，因此运行失败时可能留下部分
  文件。TERREL 还会在读取大数据前拒绝控制文件中缺失或重复的地形文件。
- 五个阶段 API 返回主输出的 `pathlib.Path`。CALWRF 的主输出和返回值始终是 canonical
  NetCDF；显式配置的 M3D/M2D 作为副产物写出。TERREL、CTGPROC、MAKEGEO 和
  CALMET 显式请求传统格式时返回对应传统文件路径；这些路径不能交给
  `xarray.open_dataset()`，可用包内相应 legacy reader 转换为 NetCDF。
- Python 传统 CALMET.DAT/AUX 头保留控制文件内容，但实现说明及注释记录组织与现有
  GNU 基准不同。浙江基准中两个 Python 文件分别少 6 条
  132 字节注释记录（含记录标记时整文件各少 840 字节），因此逐位回归会在解析
  各自头部后对齐逐小时数据记录，而不直接比较整文件偏移。跳过注释块后，DAT 的
  176 字节运行控制记录和 `ZFACE/Z0/ILANDU/ELEV/XLAI` 五条静态记录，以及 AUX 的
  144 字节运行控制记录和 `ZFACE/2D_VARS/3D_VARS` 声明均逐字节一致。
- 浙江 2018-03-18 00--12 的 fixed4 门禁曾在第 11 小时留下 28 个 `IPGT`
  等级差异。它们已定位到 RDMM5 在非末时次精确边界上的云底端点推进：此时程序
  预读下一条 M3D 记录，以“当前/未来”云底和 `DTINC=0` 执行 `ICEILG` 的非零
  感知插值；只有最后时次才保留“前一/当前”端点并以 `DTINC=1` 计算。普通线性场
  在这两种写法下相同，但当一侧云底为零时，`ICEILG` 会取两端最大值并改变稳定度
  分类。Python 现已复现该推进顺序。第 11 小时离线逐格重放改变了 21,054 个云底
  值；重新计算的 304,390 个 `IPGT` 与 GNU 全部相同，且相对 fixed4 恰好只改变
  原来的 28 个等级值。修正后的 fixed5 完整门禁中，DAT 全部 12 小时均为 31/31
  个字段、100% 元素逐位一致且 31/31 条 payload 逐字节一致；AUX 全部 12 小时均为
  4/4 个字段、100% 元素逐位一致且 30/30 条 payload 逐字节一致。strict 和
  practical 策略均通过；当前没有未解决的数值或 payload 偏差，整文件差异仅为
  上一条所述的 6 条头部注释记录。
- 投影和 GeoTIFF 输入分别使用 `pyproj` 与 `tifffile`。两者是安装依赖，因为
  NumPy、SciPy 和 xarray 本身不提供 CALMET 所需的 CRS 变换及 TIFF 栅格解码。
- GeoTIFF 地形和土地利用需要文件提供可解析的 EPSG 地理或投影坐标系键。旧
  `TIFFINFO` 还可从一组非 EPSG 的 GeoTIFF 投影参数键拼装坐标系；仓库没有这类
  输入或可核对参考，因此 Python 会明确拒绝它们，不猜测 datum 或投影定义。
- CALWRF 的 `FLG`、TERREL 的 `LSTFIL` 以及 CTGPROC/MAKEGEO 的 `RUNLST` 只控制
  旧程序的人类可读运行摘要。Python 接受这些记录，但不另造 Fortran 风格
  列表文件；主结果、QA 网格和显式数据副产物仍按配置写出。`LCFILES` 只有作为
  `.inp/.lst` 记录读取时才应用大小写转换；作为 Python 关键字时仅为兼容而接受，
  所有直接以关键字传入的路径都保持调用者给出的拼写。
- `DWDBII` 仍可从旧控制文件和公开参数读取。发布版 COASTEXT 将 `LWDBII` 固定为
  false，因此跳过 GSHHS 中 source=0 的 WDBII 多边形；Python 保留这一可观察行为，
  source=1 的 WVS/GSHHS 多边形继续使用 `DWVS` 转换。
- CTGPROC 字典仍列出 `IDEFTYP`，但 7.0 发布源码已经注释其验证与分类逻辑，改用
  `LMARSWAP` 和 `IOCEAN`。Python 读取旧控制文件时容许该无效记录，不为它添加行为。
- 水平网格运算均按数组批量执行。时间递推、垂直层递推、输入记录扫描及传统定宽/
  sequential-unformatted 写出保留短 Python 循环，因为这些步骤有可观察的顺序状态
  或必须限制峰值内存；改成一次性广播会改变结果或使浙江真实数据无法流式处理。
- 五个公共组件模块、配置 dataclass 和公开调用边界有类型注解。数值移植内核仍有
  一部分参数依靠 NumPy 数组约定和测试限定，尚未逐函数补齐静态类型注解。
- Python 会拒绝旧 CALMET 中已验证会崩溃或读取未定义内存的控制组合，例如
  `IPR3=1` 的错误 FORMAT 语句，以及 `LDB=T, NOOBS=0/1, ISURFT=-1` 的未初始化
  `TINF`。这些组合没有可移植的数值参考；其他合法打印和调试分支继续执行。
- 与 GNU CALMET 做等价门禁时使用已逐字节验证的 M3D。传统 M3D 的 `F6.1`、`I3`
  字段量化属于接口语义；把 CALWRF canonical NetCDF 的未格式化高精度值直接送入
  CALMET 会在阈值敏感格点产生差异，因此两条输入路径不承诺逐位相同。
- 旧 CALMET 的 `LDB` 太阳高度诊断在设置本时步开关前读取未初始化或滞后一时步的
  LOGICAL。Python 按当前 `NN1..NN2` 时步确定性输出，物理求解和正式气象场不受影响。
- MAKEGEO 积雪模式可按日期写出多个产品；公开函数返回详细结果中的首个日期产品
  路径，其余日期路径按同一输出命名规则生成。
