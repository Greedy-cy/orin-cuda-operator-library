# GEMM：从朴素 FP32 到 FP16 Tensor Core

本目录实现行主序矩阵乘法 `C[M,N] = A[M,K] × B[K,N]`。当前阶段只保留
standalone CUDA/C++ benchmark；每版同时运行自定义 kernel 和相同数据类型的
cuBLAS 参考，分别用 CUDA Event 多次计时并取 median。

```bash
mkdir -p build profiles
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 gemm_v0.cu -o build/gemm_v0 -lcublas
./build/gemm_v0 --m=1024 --n=1024 --k=1024
./build/gemm_v0 --m=37 --n=65 --k=1003
```

## FP32 实测记录

| 版本 | 优化点 | `4096³` | 性能 | 相对 v0 | cuBLAS 占比 | 关键瓶颈 |
|---|---|---:|---:|---:|---:|---|
| v0 | 每线程计算一个输出，直接从 global memory 读取 A/B | 805.372 ms | 170.653 GFLOP/s | 1.00× | 12.60% | K 循环中没有跨线程数据复用；L1/TEX throughput 96%，long scoreboard 占指令间隔约 59.7% |
| v1 | `16×16` shared-memory tiling，仍为一线程一输出 | 563.534 ms | 243.888 GFLOP/s | 1.43× | 18.00% | long scoreboard 降至 41.7%，但每线程只累加一个输出，同步和加载指令相对 FMA 仍多 |
| v2 | `64×64×16` block tile，每线程寄存器累加 `4×4` 输出 | 185.263 ms | 741.857 GFLOP/s | 4.35× | 54.74% | 寄存器使 occupancy 降到 64.5%；标量写回仅利用 8/32 bytes/sector，额外 sector 占 8% |
| v3 | 对齐主路径以 `float4` 加载 A/B tile 并写回 C；非对齐回退 v2 | 137.840 ms | 997.089 GFLOP/s | 5.84× | 73.56% | 执行指令较 v2 降 33.5%，uncoalesced store 警告消失；64-register 限制 occupancy，内存吞吐回升至 86.3% |
| v4 | 两份 shared tile；下一 tile 预取至寄存器，与当前 tile 计算交错 | 135.186 ms | 1,016.669 GFLOP/s | 5.96× | 75.04% | 较 v3 仅快 1.96%；指令反增 5.4%，但 eligible warp 改善；收益低于 3% 阈值 |
| v5 | Ampere `cp.async` 两级流水，global→shared 不经通用寄存器 | 127.726 ms | 1,076.048 GFLOP/s | 6.31× | 79.39% | 较 v4 快 5.84%；60 registers、无 spill；NCU 转而指出 shared access wavefront 冗余 |
| v6 | 有限配置扫描；regular-M 选 `128×64×16/8×4`，另加 M=1/16 专用配置 | **115.418 ms** | **1,190.795 GFLOP/s** | **6.98×** | **87.86%** | 107 registers、32.6% occupancy，但 issue slots 71.6%；五次跨进程结果跨度仅 0.036% |

完整的形状矩阵、正确性、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)；shared-memory 复用的收益和剩余瓶颈见
[`reports/v1.md`](reports/v1.md)；寄存器分块的计算强度、occupancy 代价与写回问题见
[`reports/v2.md`](reports/v2.md)；向量化主路径、标量回退与 sector 改善见
[`reports/v3.md`](reports/v3.md)；同步双缓冲的小收益、资源变化和长重复复验见
[`reports/v4.md`](reports/v4.md)；`cp.async` 的同步审计和流水收益见
[`reports/v5.md`](reports/v5.md)；配置资源表、跨形状筛选、dispatcher 和稳定性见
[`reports/v6.md`](reports/v6.md)。

## FP16 Tensor Core 实测记录

输入为 FP16，Tensor Core/参考实现均使用 FP32 accumulation，输出为 FP32；该表
与 FP32 表使用不同 cuBLAS 口径，二者比例不能混用。

| 版本 | 优化点 | `4096³` | 性能 | 相对 v7 | FP16 cuBLAS 占比 | 关键瓶颈 |
|---|---|---:|---:|---:|---:|---|
| v7 | 一 warp 计算一个 `16×16×16` WMMA tile，直接从 global memory 加载 | 140.987 ms | 974.836 GFLOP/s | 1.00× | 12.29% | L1/TEX throughput 99.76%；long scoreboard 占 92.8%；跨 warp 不复用 A/B |
| v8 | `128×128×16` block tile、每 warp 2×4 WMMA tiles、shared skew 与 cp.async 双缓冲 | **26.505 ms** | **5,185.455 GFLOP/s** | **5.32×** | **64.29%** | 115 registers 将 occupancy 限到 33.1%；仍有 global/shared excessive access，L2 throughput 88.7% |

v7 的 FP16 精度口径、完整形状与 Nsight 证据见
[`reports/v7.md`](reports/v7.md)。v8.0→v8.3 的逐步实验、否决项、最终全形状结果和
Nsight 对照见 [`reports/v8.md`](reports/v8.md)。

## 当前选择

v6 是 FP32 最终 standalone 版本。自动 dispatcher 选择：M=1 使用
`1×256×16/1×1`，M≤16 使用 `16×128×16/1×8`，其余对齐形状使用
`128×64×16/8×4`；任意非对齐尺寸回退到标量边界 kernel。FP16 最终选择 v8：
regular-M 使用 `128×128×16` WMMA tile，`16≤M<128` 使用 16-row 专用路径，
M=1 和任意非对齐尺寸回退标量。两条数据类型路线始终分别报告；GEMM 阶段在此
收口，等待审阅后再继续下一算子。
