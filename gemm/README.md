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
| v3 | 对齐主路径以 `float4` 加载 A/B tile 并写回 C；非对齐回退 v2 | 137.840 ms | 997.089 GFLOP/s | 5.84× | 73.56% | 执行指令较 v2 降 33.5%，uncoalesced store 警告消失；64-register 限制 occupancy，内存吞吐回升至 86.3% |
| v4 | 两份 shared tile；下一 tile 预取至寄存器，与当前 tile 计算交错 | 135.186 ms | 1,016.669 GFLOP/s | 5.96× | 75.04% | 较 v3 仅快 1.96%；指令反增 5.4%，但 eligible warp 改善；收益低于 3% 阈值 |

完整的形状矩阵、正确性、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)；shared-memory 复用的收益和剩余瓶颈见
[`reports/v1.md`](reports/v1.md)；寄存器分块的计算强度、occupancy 代价与写回问题见
[`reports/v2.md`](reports/v2.md)；向量化主路径、标量回退与 sector 改善见
[`reports/v3.md`](reports/v3.md)；同步双缓冲的小收益、资源变化和长重复复验见
[`reports/v4.md`](reports/v4.md)。

## 当前选择

v4 是当前 FP32 最快版本，但相对 v3 只有约 2% 的稳定小收益，低于显著收益阈值。
下一版用 Ampere `cp.async` 替换“global→register→shared”搬运，保持其余配置，
检验硬件异步拷贝能否产生超过噪声线的收益；最终配置扫描留给 v6。
