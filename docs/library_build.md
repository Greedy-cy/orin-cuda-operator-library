# 最小库构建与验证

## 1. 为什么现在引入 CMake

在 Reduce、Softmax、Transpose、RMSNorm、GEMM、Attention 六类 standalone final
和 launcher 全部稳定后，项目第一次需要同时编译多个 translation unit 并生成可复用
库，因此增加根目录最小 CMake。各算子 v0→final 仍使用目录 README 中的独立
`nvcc` 命令；CMake 不接管历史实验，也不改变已经冻结的性能口径。

当前构建只包含：

- `liboperatorlib.a`：六类 final launcher；
- `operatorlib_core_smoke`：Reduce/Softmax/Transpose/RMSNorm；
- `operatorlib_gemm_smoke`：FP32/FP16 GEMM 全 dispatcher；
- `operatorlib_attention_smoke`：FP32/FP16 Attention 对齐与 fallback。

没有安装规则、自动调优器、Python 包、PyTorch 或第三方测试框架。

## 2. 构建

```bash
git clone https://github.com/Greedy-cy/orin-cuda-operator-library.git
cd orin-cuda-operator-library
cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build -j2
ctest --test-dir cmake-build --output-on-failure
```

目标设备固定 `CUDA_ARCHITECTURES=87`，库 CUDA 源码使用 Release 优化和 `-lineinfo`。
静态库启用 PIC，为后续被共享对象/PyTorch Extension 链接保留条件，但当前不生成
Python `.so`。

## 3. API 使用约定

```cpp
#include <operatorlib/operators.h>

cudaStream_t stream = /* caller-owned stream */;
cudaError_t error = operatorlib::softmax_f32(
    device_input, device_output, rows, cols, stream);
```

launcher 只负责参数检查、尺寸分派和 kernel enqueue。调用方拥有内存和 stream；除
Reduce 为保证求和语义在同一 stream 内异步清零单元素输出外，launcher 不初始化
输出，也不做同步。函数返回成功只表示参数与 launch 成功，异步执行错误由调用方在
后续 stream/device 同步处观察。

详细的版本来源、dispatcher 和 sanitizer 结果见
[`launcher_extraction.md`](launcher_extraction.md)。standalone final 性能冻结见
[`standalone_benchmark.md`](standalone_benchmark.md)。
