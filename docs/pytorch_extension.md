# PyTorch CUDA Extension 接入报告

## 目标与边界

Standalone kernel、launcher 和统一 CMake 基准稳定后，才开始 PyTorch 接入。本阶段不改写 kernel，也不改变 standalone benchmark 口径；它只验证 Tensor 到裸指针的桥接、当前 CUDA stream、dtype/shape 检查与 `TORCH_LIBRARY` 注册。

接入分两步：

1. 先以 Reduce/Softmax 建立最小闭环并单独提交。
2. 验证闭环稳定后，再注册 Transpose、RMSNorm、GEMM 和 Attention。

## 环境

| 项目 | 实测值 |
|---|---|
| 设备 | Jetson Orin Nano Super（SM 8.7） |
| Host JetPack | R36.4.7 / JetPack 6.2 |
| 容器 | `nvcr.io/nvidia/pytorch:25.06-py3-igpu` |
| PyTorch | `2.8.0a0+5228986c39.nv25.06` |
| 容器 CUDA Toolkit | 12.9 |
| Standalone CUDA Toolkit | 12.6.68 |

PyTorch 接入正确性与 standalone 性能数据属于两条独立证据链。简历性能数字仍使用已冻结的 standalone CUDA 12.6 测量结果，不把容器 JIT 编译环境混入原有对比。

## Phase 1：Reduce / Softmax 最小注册

### 实现

- `python/core_binding.cpp`：只承担输入校验、设备保护、输出分配、当前 stream 获取和 launcher 调用。
- `python/load_core_extension.py`：用 `torch.utils.cpp_extension.load` JIT 编译，目标架构固定为 Orin 的 SM 8.7。
- `python/test_core_extension.py`：与 PyTorch reference 对齐，并覆盖非默认 stream 与错误路径。
- 通过 `TORCH_LIBRARY(operatorlib, ...)` 声明 schema，通过 `TORCH_LIBRARY_IMPL(operatorlib, CUDA, ...)` 注册 CUDA 实现。
- 不使用 pybind11 暴露算子；共享库加载后通过 `torch.ops.operatorlib.*` 调用。

公开接口：

```python
torch.ops.operatorlib.reduce_sum(input) -> Tensor
torch.ops.operatorlib.softmax(input) -> Tensor
```

当前两个接口只接受 contiguous CUDA FP32 Tensor。Softmax 沿最后一维计算；空 Tensor、空最后一维、非 contiguous 或错误 dtype 会给出明确异常。

### Stream 语义

binding 在输入设备上建立 `CUDAGuard`，随后读取该设备的 current CUDA stream，并原样传给 standalone launcher。测试在显式创建的非默认 `torch.cuda.Stream` 中生成输入、调用 custom op 和 reference，之后只同步该 stream。

### 验证矩阵

| 算子 | 用例 | 目的 | 结果 |
|---|---|---|---|
| Reduce | `N=1,000,003` | 非对齐长度、非默认 stream | PASS |
| Softmax | `[19,128]` | 常用对齐宽度 | PASS |
| Softmax | `[17,1024]` | block-per-row 主路径 | PASS |
| Softmax | `[17,1003]`, 输入放大 1000 倍 | 非对齐边界与数值稳定性 | PASS |
| 校验 | Reduce FP16 | dtype 拒绝 | PASS |
| 校验 | Softmax 非 contiguous | layout 拒绝 | PASS |
| 校验 | Softmax `[3,0]` | 空最后一维拒绝 | PASS |

Reduce 使用 `rtol=5e-4, atol=5e-4`，Softmax 使用 `rtol=1e-4, atol=1e-5` 与 PyTorch 输出对齐，并额外检查每行输出和接近 1。

### 复现

```bash
sudo docker run --rm --runtime nvidia --ipc=host \
  -v /path/to/orin-cuda-operator-library:/workspace/operatorLib \
  -w /workspace/operatorLib \
  nvcr.io/nvidia/pytorch:25.06-py3-igpu \
  python3 python/test_core_extension.py
```

首次运行会在 `python/build/core/` 编译共享库，之后在源码未变化时复用 Ninja 构建结果。该目录被 Git 忽略。

## 后续阶段

第二阶段继续复用同一注册方式，加入 Transpose、RMSNorm、FP32/FP16 GEMM 和 FP32/FP16 Attention。Attention 以 PyTorch 数学参考实现验证即可，cuDNN 不作为完成条件；当前项目只实现 forward，不承诺 autograd/backward。
