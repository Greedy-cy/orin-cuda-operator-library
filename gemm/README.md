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
| v1 | `16×16` shared-memory tiling，仍为一线程一输出 | 563.534 ms | 243.888 GFLOP/s | 1.43× | 18.00% | long scoreboard 降至 41.7%，但每线程只累加一个输出，同步和加载指令相对 FMA 仍多 |
| v2 | `64×64×16` block tile，每线程寄存器累加 `4×4` 输出 | 185.263 ms | 741.857 GFLOP/s | 4.35× | 54.74% | 寄存器使 occupancy 降到 64.5%；标量写回仅利用 8/32 bytes/sector，额外 sector 占 8% |

完整的形状矩阵、正确性、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)；shared-memory 复用的收益和剩余瓶颈见
[`reports/v1.md`](reports/v1.md)；寄存器分块的计算强度、occupancy 代价与写回问题见
[`reports/v2.md`](reports/v2.md)。

## 当前选择

v2 已把计算重用提升到新的层级，但仍不是最终实现。下一版针对 NCU 指出的标量
写回 sector 利用率，给对齐形状增加 `float4` 加载/写回路径，并保留非对齐标量
回退；双缓冲和异步加载仍留到后续版本。
