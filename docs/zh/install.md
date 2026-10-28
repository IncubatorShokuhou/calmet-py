# 安装

[English](../install.md)

PyCALMET 需要 Python 3.10 或更高版本。发布包名是 `pycalmet`。

## 默认安装

默认安装包含 NumPy、SciPy，以及读取器已经在用的地理空间库，不含 JAX：

```bash
python -m pip install pycalmet
```

从源码目录做同样的选择：

```bash
python -m pip install .
```

导入本包和运行浙江示例都停留在 NumPy，直到调用方改选后端。见[数组后端](backends.md)。

## 可选的 JAX / GPU extra

```bash
python -m pip install "pycalmet[gpu]"
```

从源码目录：

```bash
python -m pip install ".[gpu]"
```

该 extra 装上哪些轮子，取决于平台：

| 平台 | `[gpu]` 拉取的轮子 |
| --- | --- |
| Linux x86_64 和 aarch64 | `jax[cuda12]`，即 JAX、CUDA 12 插件和 NVIDIA 的 pip 轮子。驱动须为 525 或更新。 |
| macOS、Windows 以及其他 Linux 架构 | 可移植的 `jax` 轮子。未另行安装平台插件时，JAX 使用 CPU 后端。 |

安装 `[gpu]` 之后运行时不变。在选择 JAX 之前，`pycalmet.get_backend()` 仍返回 `"numpy"`。

CUDA 13 轮子（`jax[cuda13]`）需要更新的驱动。机器提供 CUDA 13 时可以自行安装，此时不必再装 CUDA 12 插件。已经装有 CUDA 工具包和 cuDNN 的机器可以使用 `jax[cuda12-local]`，直接用本机库。这两项都是上游 JAX 的安装方式。本 extra 固定在 CUDA 12 的 pip 栈上，与集群里常见的驱动一致。

pip 只会安装当前解释器能够使用的 JAX 版本。Python 3.10 解析到较旧的 JAX；更新的 Python 解析到当前的 JAX。PyCALMET 用到的导入面（`jax`、`jax.numpy`，以及 x64 和默认设备配置）在这个范围内保持稳定。

## 检查当前选择

```python
import pycalmet

print(pycalmet.available_backends())
print(pycalmet.backend_info())
```

JAX 尚不能导入时，`available_backends()` 为 `("numpy",)`；装上 extra 之后为 `("numpy", "jax")`。`backend_info()` 报告当前后端、JAX 是否已导入，以及选中 JAX 之后的设备标签。

## 开发安装

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

默认测试既不安装也不导入 JAX。与 NumPy 对照的测试在缺少 `jax` 时跳过，因此只有 NumPy 的环境仍能通过。

在没有 GPU 的机器上运行 JAX 测试时，安装可移植的 CPU 构建：

```bash
python -m pip install "jax>=0.4.31"
python -m pytest -q tests/test_backend.py
```

CPU 版 JAX 运行同一批计算核，也不会下载 NVIDIA 轮子。
