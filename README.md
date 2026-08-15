# Orin CUDA Operator Library

面向 Jetson Orin Nano Super（SM 8.7）的 CUDA 推理算子实验库，覆盖 Reduce、Softmax、Transpose、RMSNorm、GEMM 与 FlashAttention-style Attention，并通过 PyTorch `TORCH_LIBRARY` 提供统一调用接口。

项目从可验证的 standalone CUDA kernel 出发，使用 CUDA Event、Nsight Systems、Nsight Compute 和 compute-sanitizer 驱动优化；算子稳定后再提取 stream-aware launcher、最小 CMake 静态库和 PyTorch 接口。代码保留 `v0 -> final` 的演进与逐版本实验报告，便于复现实验结论与理解优化取舍。

## 已完成内容

| 算子 | 版本演进 | Standalone final | 关键优化 |
|---|---|---|---|
| [Reduce](reduce/README.md) | v0-v4 | v3 | block/warp reduction、四路 ILP；float4 负优化保留 |
| [Softmax](softmax/README.md) | v0-v4 | v4 | warp/block 分派、float4、寄存器缓存、稳定 softmax |
| [Transpose](transpose/README.md) | v0-v4 | v2 | shared tile、`32x33` padding 消除 bank conflict |
| [RMSNorm](rmsnorm/README.md) | v0-v4 | v3 | 四 kernel 融合、warp reduction、float4 与边界 fallback |
| [GEMM](gemm/README.md) | FP32 v0-v6；FP16 v7-v8.3 | FP32 v6 / FP16 v8 | tiling、寄存器分块、cp.async、WMMA、shared skew、尺寸分派 |
| [Attention](attention/README.md) | FP32 v0-v5；FP16 v6-v7.1 | FP32 v5 / FP16 v6 | online softmax、K/V tiling、cp.async、WMMA、causal tile skipping |

每个版本均有独立 `.cu` 和 `reports/v*.md`。报告记录假设、正确性、主/边界尺寸、median、Nsight 指标、资源占用、结论和下一步；未达到采用门槛的实验也会保留。

## Jetson Orin 实测结果

以下是 2026-08-12 在 MAXN_SUPER、GPU 1020 MHz、EMC 3199 MHz 下重新编译复验的 CUDA Event median。详细口径见 [standalone benchmark 冻结记录](docs/standalone_benchmark.md)。

| 算子 | 主形状 | Final 结果 | 相对 baseline / 参考 |
|---|---|---:|---:|
| Reduce FP32 | N=16,777,216 | 0.716 ms，93.67 GB/s | v0 的 2.09x |
| Softmax FP32 | 4096x1024 | 0.353 ms，95.05 GB/s | v0 的 1.89x |
| Transpose FP32 | 4096x4096 | 1.423 ms，94.34 GB/s | v0 的 6.26x；理论峰值约 92% |
| RMSNorm FP32 | 2048x4096 | 0.708 ms，94.83 GB/s | v0 的 4.20x |
| GEMM FP32 | 4096^3 | 115.395 ms，1.191 TFLOP/s | cuBLAS 的 87.88% |
| GEMM FP16 | 4096^3 | 26.447 ms，5.197 TFLOP/s | 同口径 FP16 cuBLAS 的 64.54% |
| Attention FP32 | B1,H12,S1024,D128 | 37.340 ms，172.54 GFLOP/s | FP32 v0 的 6.36x |
| Attention FP16 | B1,H12,S1024,D128 | 20.817 ms，309.48 GFLOP/s | 消除 48 MiB S^2 score；全路线 11.4x* |

`*` Attention 的 11.4x 从 FP32 naive 演进到 FP16 Tensor Core final，包含 dtype 变化，不表示同精度 kernel 加速。项目没有把不可比的旧 cuDNN MHA 数字包装成外部库比例；Attention 以 custom baseline、显存复杂度、正确性和 Nsight 证据收口。

## 目录

```text
orin-cuda-operator-library/
├── reduce|softmax|transpose|rmsnorm|gemm|attention/
│   ├── <operator>_v*.cu       # 可独立编译的版本演进
│   ├── reports/v*.md          # 每版本实验报告
│   └── README.md              # 版本总表与最终选择
├── include/operatorlib/operators.h
├── src/                       # 从 standalone final 提取的 launcher
├── tests/                     # launcher 的 CUDA/stream/sanitizer smoke tests
├── python/                    # TORCH_LIBRARY binding、JIT loader、PyTorch 对齐测试
├── tools/                     # 锁频与统一复验脚本
├── docs/                      # 性能口径、launcher、构建与接入记录
└── CMakeLists.txt             # final launcher 稳定后加入的最小构建
```

## 快速复现

### 1. Standalone final

```bash
git clone https://github.com/Greedy-cy/orin-cuda-operator-library.git
cd orin-cuda-operator-library
./tools/run_final_standalone.sh smoke

# 只有记录正式性能时才需要锁频
sudo ./tools/lock_orin_clocks.sh
./tools/run_final_standalone.sh benchmark
```

### 2. 静态库与 launcher 测试

```bash
cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build -j2
ctest --test-dir cmake-build --output-on-failure
```

公开 C++ API 位于 `include/operatorlib/operators.h`。所有 launcher 接收调用方的 `cudaStream_t`，只 enqueue、不做设备同步。

### 3. PyTorch 自定义算子

```bash
sudo docker run --rm --runtime nvidia --ipc=host \
  -v "$PWD":/workspace/operatorLib \
  -w /workspace/operatorLib \
  nvcr.io/nvidia/pytorch:25.06-py3-igpu \
  python3 python/test_extension.py
```

加载后可通过 `torch.ops.operatorlib.reduce_sum/softmax/transpose/rmsnorm/gemm/attention` 调用。测试覆盖非默认 stream、常用与非对齐尺寸、FP32/FP16 dispatcher、causal Attention 和错误路径；详情见 [PyTorch 接入报告](docs/pytorch_extension.md)。当前定位为推理 forward 实验库，不提供 backward，也不会在 binding 中为非 contiguous 输入隐式复制。

## 证据导航

- [Standalone final 统一复验](docs/standalone_benchmark.md)
- [Final launcher 提取与 sanitizer](docs/launcher_extraction.md)
- [最小 CMake 构建](docs/library_build.md)
- [PyTorch TORCH_LIBRARY 接入](docs/pytorch_extension.md)

## 适用范围

- 目标架构为 Jetson Orin Nano Super 的 SM 8.7，当前只提供推理 forward。
- Reduce、Softmax、Transpose 与 RMSNorm 的公开接口使用 FP32；GEMM 与 Attention 支持 FP32/FP16 输入并使用 FP32 输出或累加状态。
- 输入必须位于 CUDA 设备、布局连续且 dtype 满足接口约束；binding 不会隐式复制输入。
- Attention 结果以 custom baseline、显存复杂度、正确性和 Nsight 证据为准，不宣称与官方 FlashAttention 或 cuDNN Attention 等价。

## License

本项目采用 [MIT License](LICENSE)。
