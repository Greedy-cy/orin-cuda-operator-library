# GEMM：从朴素 FP32 实现开始

本目录实现行主序矩阵乘法 `C[M,N] = A[M,K] × B[K,N]`。当前阶段只保留
standalone CUDA/C++ benchmark；每版同时运行自定义 kernel 和 cuBLAS FP32
参考实现，分别用 CUDA Event 多次计时并取 median。

```bash
mkdir -p build profiles
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 gemm_v0.cu -o build/gemm_v0 -lcublas
./build/gemm_v0 --m=1024 --n=1024 --k=1024
./build/gemm_v0 --m=37 --n=65 --k=1003
```

## 实测记录

| 版本 | 优化点 | `4096³` | 性能 | 相对 v0 | cuBLAS 占比 | 关键瓶颈 |
|---|---|---:|---:|---:|---:|---|
| v0 | 每线程计算一个输出，直接从 global memory 读取 A/B | 805.372 ms | 170.653 GFLOP/s | 1.00× | 12.60% | K 循环中没有跨线程数据复用；L1/TEX throughput 96%，long scoreboard 占指令间隔约 59.7% |

完整的形状矩阵、正确性、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)。

## 当前选择

v0 是刻意保留低数据复用的教学 baseline，不作为最终实现。下一版只引入
shared-memory tiling，让同一 tile 中的 A/B 元素由 block 内线程复用；寄存器分块、
向量化和异步流水留到后续版本，便于分别量化收益。
