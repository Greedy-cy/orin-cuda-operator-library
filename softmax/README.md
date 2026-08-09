# Softmax：从 shared-memory baseline 开始

本目录实现最后一维的稳定 Softmax。当前阶段使用独立 CUDA/C++ 程序，不依赖
PyTorch、CMake 或公共 benchmark 框架。

## v0：一行一个 block

v0 使用 256 个线程完成三步：

1. shared-memory tree reduction 求行最大值；
2. 计算 `exp(x-max)`，写入全局输出，再进行 shared-memory sum reduction；
3. 从全局输出读回指数值并归一化。

```bash
mkdir -p build profile
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 softmax_v0.cu -o build/softmax_v0

./build/softmax_v0 --rows=4096 --cols=1024
./build/softmax_v0 --rows=128 --cols=8192
./build/softmax_v0 --rows=4 --cols=1003 --extreme
```

## 实测记录

| 版本 | 优化点 | 主尺寸耗时 | 相对 v0 | 有效 I/O 带宽 | 关键瓶颈 |
|---|---|---:|---:|---:|---|
| v0 | 两次 shared tree reduction，中间指数值写全局内存 | 0.668 ms | 1.00× | 50.21 GB/s | Compute/Memory 约 60% 均衡；16 次 block 同步和中间全局读写都有优化空间 |

主尺寸为 `rows=4096, cols=1024`，锁定 GPU 1020 MHz、EMC 3199 MHz。
完整正确性、性能和 Nsight 记录见 [`reports/v0.md`](reports/v0.md)。
