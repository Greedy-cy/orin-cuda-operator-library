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

PyTorch 接入正确性与 standalone 性能数据属于两条独立证据链。对外发布的性能数字仍使用已冻结的 standalone CUDA 12.6 测量结果，不把容器 JIT 编译环境混入原有对比。

## Phase 1：Reduce / Softmax 最小注册

### 实现（历史最小阶段）

- `python/core_binding.cpp`：只承担输入校验、设备保护、输出分配、当前 stream 获取和 launcher 调用。
- `python/load_core_extension.py`：用 `torch.utils.cpp_extension.load` JIT 编译。
- `python/test_core_extension.py`：与 PyTorch reference 对齐，并覆盖非默认 stream 与错误路径。
- 通过 `TORCH_LIBRARY(operatorlib, ...)` 声明 schema，通过 `TORCH_LIBRARY_IMPL(operatorlib, CUDA, ...)` 注册 CUDA 实现。
- 不使用 pybind11 暴露算子；共享库加载后通过 `torch.ops.operatorlib.*` 调用。

这些最小文件保留在 Phase 1 Git 提交中；Phase 2 将它们扩展并重命名为最终入口。

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
  -v "$PWD":/workspace/operatorLib \
  -w /workspace/operatorLib \
  nvcr.io/nvidia/pytorch:25.06-py3-igpu \
  python3 python/test_core_extension.py
```

Phase 1 首次运行会在 `python/build/core/` 编译共享库，之后在源码未变化时复用 Ninja 构建结果。该目录被 Git 忽略。

## Phase 2：六类算子完整注册

### 从最小闭环到完整接口

Phase 1 证明注册方式、stream 和错误处理可行后，最终入口演变为：

- `python/operatorlib_binding.cpp`
- `python/load_extension.py`
- `python/test_extension.py`

加载器编译六个稳定的 launcher 源文件，不复制 kernel。容器预设的 `TORCH_CUDA_ARCH_LIST` 最初同时生成 SM 10.1 与 SM 8.7，虽然运行正确，但对固定目标 Orin 属于无意义编译；最终改为默认强制 SM 8.7，同时保留 `OPERATORLIB_CUDA_ARCH_LIST` 作为显式覆盖入口。

### 最终 Python 接口

```python
torch.ops.operatorlib.reduce_sum(input)
torch.ops.operatorlib.softmax(input)
torch.ops.operatorlib.transpose(input)
torch.ops.operatorlib.rmsnorm(input, weight, epsilon=1e-5)
torch.ops.operatorlib.gemm(a, b)
torch.ops.operatorlib.attention(q, k, v, causal=False)
```

| 算子 | 输入 dtype | 输出 dtype | 形状约定 |
|---|---|---|---|
| Reduce | FP32 | FP32 scalar | 任意非空 contiguous Tensor |
| Softmax | FP32 | FP32 | 沿最后一维 |
| Transpose | FP32 | FP32 | 2D `[rows,cols] -> [cols,rows]` |
| RMSNorm | FP32 | FP32 | 最后一维为 hidden，weight 为 `[hidden]` |
| GEMM | FP32 / FP16 | FP32 | `[M,K] @ [K,N]` |
| Attention | FP32 / FP16 | FP32 | Q/K/V 均为 `[B,H,S,D]`，`D<=128` |

FP16 GEMM 和 Attention 延续 standalone kernel 的语义：输入保留 FP16，累加/online-softmax 状态及输出使用 FP32。

### 完整正确性矩阵

所有用例均放在显式创建的非默认 `torch.cuda.Stream` 上执行，完成后只同步该 stream。

| 算子 | dtype 与尺寸 | 覆盖路径 | 结果 |
|---|---|---|---|
| Reduce | FP32 `N=1,000,003` | 非对齐长度 | PASS |
| Softmax | FP32 `[19,128]`, `[17,1024]`, `[17,1003]` | warp/block、边界、极端输入 | PASS |
| Transpose | FP32 `[64,96]`, `[65,37]` | tiled 与边界 | PASS |
| RMSNorm | FP32 `[3,7,1024]`, `[5,1003]` | float4 与 scalar | PASS |
| GEMM | FP32 `(128,64,32)`, `(7,37,23)` | regular 与 scalar fallback | PASS |
| GEMM | FP16 `(128,128,32)`, `(16,128,32)`, `(32,32,32)`, `(17,19,23)` | tiled、small-M、WMMA baseline、scalar | PASS |
| Attention | FP32/FP16，`S=32,D=64/128` | 优化路径、causal/non-causal | PASS |
| Attention | FP32/FP16，`S=37,D=71,causal` | 非对齐 scalar fallback | PASS |

容差：FP32 GEMM `5e-4`，FP32 Attention `8e-4`，FP16 GEMM/Attention `2e-2`；Transpose 要求逐元素完全一致。另验证 dtype、contiguous、维度、矩阵乘形状、RMSNorm weight 长度和 Attention `D<=128` 的错误路径。

Attention reference 由 PyTorch 的 FP32 `matmul -> scale/mask -> softmax -> matmul` 组成。它足以验证当前 forward 语义和 online softmax 数值结果；cuDNN 不作为本阶段的依赖或完成标准。

### 最终复现命令

```bash
sudo docker run --rm --runtime nvidia --ipc=host \
  -v "$PWD":/workspace/operatorLib \
  -w /workspace/operatorLib \
  nvcr.io/nvidia/pytorch:25.06-py3-igpu \
  python3 python/test_extension.py
```

最终共享库缓存在 `python/build/operatorlib/`，由 Ninja 增量构建，且不会进入 Git。

## 当前限制

- 当前仅注册 CUDA backend，不提供 CPU kernel。
- 当前所有输入必须 contiguous，不在 binding 中隐式复制。
- Reduce、Softmax、Transpose、RMSNorm 当前只开放 FP32。
- GEMM 与 Attention 接受 FP32/FP16 输入并返回 FP32。
- 当前项目定位为推理算子实验库，只实现 forward，不承诺 autograd/backward。
