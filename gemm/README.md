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

完整的形状矩阵、正确性、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)；shared-memory 复用的收益和剩余瓶颈见
[`reports/v1.md`](reports/v1.md)。

## 当前选择

v1 证明了 shared-memory 数据复用有效，但仍不是最终实现。下一版扩大输出 tile，
让每个线程在寄存器中计算多个输出，提高每次 A/B 加载和同步对应的 FMA 数量；
向量化和异步流水仍留到后续版本，便于分别量化收益。
