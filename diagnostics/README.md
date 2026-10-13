# 诊断与参考结果

本目录只保存验证、画图、历史后处理和参考产物，不参与 CALMET 主运行链。下一步
将 Fortran 改写为 NumPy 时，可将 `results/reference_outputs/` 作为回归测试基准。

## 目录

- `code/`：CALMET 二进制读取、单时次对比和逐时统计脚本。
- `results/figures/`：PNG 图与逐时统计 CSV。
- `results/reference_outputs/`：已验证运行生成的 CALMET 二进制基准结果。
- `legacy/output_pro/`：旧的输出转换程序及历史产物，不属于当前运行链。
- `archived_examples/`：已停止使用的 Easter100 WRF 示例。

## 原始 WRF 基准

仓库随附的单时次基准为
`example/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc`。完整的 13 时次基准不纳入
Git；本地若有完整原文件，其对应 CALMET 参考输出为：

`results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat`

单时次地形和风场：

```bash
python3 diagnostics/code/visualize_downscaling.py \
  --wrf example/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc \
  --calmet diagnostics/results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat \
  --hour 0
```

温湿度、混合层和降水：

```bash
python3 diagnostics/code/visualize_other_fields.py \
  --wrf example/zhejiang/wrfout_d01_2018-03-18_00_00_00.nc \
  --calmet diagnostics/results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat \
  --hour 0
```

逐时统计：

```bash
python3 diagnostics/code/analyze_downscaling_timeseries.py \
  --wrf /path/to/full/wrfout_d01_2018-03-18_00_00_00.nc \
  --calmet diagnostics/results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat
```

比较 Python 与 GNU 传统输出中某个完整时段的全部字段：

```bash
python3 tools/regress_calmet.py \
  --reference diagnostics/results/reference_outputs/calmetout_2018031800_2018031812_zhejiang_original_wrf.dat \
  --candidate /path/to/python.dat \
  --ending 2018-03-18T01:00:00
```

加 `--require-bitwise` 可在任意浮点位模式不同时返回失败；整数场始终要求完全一致。

比较两个 CALWRF M2D 文件的全部二维网格记录：

```bash
python3 tools/regress_calwrf_m2d.py \
  --reference CALWRF/calwrf_out/calwrfout_2018031800_2018031812.m2d \
  --candidate /path/to/python.m2d --hours 1
```

流式比较两个 CALWRF M3D 文件的静态网格和全部逐格三维记录：

```bash
python3 tools/regress_calwrf_m3d.py \
  --reference CALWRF/calwrf_out/calwrfout_2018031800_2018031812.m3d \
  --candidate /path/to/python.m3d --hours 1
```

比较两个传统 TERREL 文件的全部地形格点（控制文件提供网格及投影定义）：

```bash
python3 tools/regress_terrel.py --control TERREL/terrel.inp \
  --reference TERREL/TERR.DAT --candidate /path/to/python/TERR.DAT
```

`tools/regress_ctgproc.py` 的 `--python` 参数同时接受规范 NetCDF 和传统 `LULC.DAT`；
参考文件必须是使用 `build_gnu.sh` 中 `-fno-automatic` 构建的有效 GNU 结果：

```bash
python3 tools/regress_ctgproc.py --fortran /path/to/gnu/LULC.DAT \
  --python /path/to/python/LULC.DAT --nx 610 --ny 499
```

MAKEGEO 比较工具也同时接受规范 NetCDF 或传统候选文件：

```bash
python3 tools/regress_makegeo.py --fortran MAKEGEO/geo.dat \
  --python /path/to/python/geo.dat
```

仓库内完整的浙江首小时位级回归可一键执行：

```bash
tools/regress_zhejiang_first_hour.sh
```

该命令通过公开 `pycalmet.calmet(...)` 读取真实传统 M3D 和 GEO.DAT，在临时目录
生成传统 CALMET.DAT 和 CALMET.AUX，然后要求 31 个主文件字段及 4 个 AUX 云水
字段全部逐位一致。输入约 1.6 GB，运行时会使用约 1.3 GB 内存；临时文件在成功或
失败后都会清理。可用 `PYTHON_BIN=/path/to/python` 覆盖默认 `.venv/bin/python`。
