# RMSNorm：从多 kernel baseline 开始

本目录实现 FP32 RMSNorm：

```text
y[token, col] = x[token, col] * rsqrt(mean(x[token, :]²) + epsilon) * weight[col]
```

当前仍为独立 CUDA/C++ benchmark，不依赖 PyTorch 或公共框架。

```bash
mkdir -p build profiles
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 rmsnorm_v0.cu -o build/rmsnorm_v0
./build/rmsnorm_v0 --tokens=2048 --hidden=4096
./build/rmsnorm_v0 --tokens=17 --hidden=1003
```

## 实测记录

| 版本 | 优化点 | `2048×4096` | 相对 v0 | 理想口径带宽 | 关键瓶颈 |
|---|---|---:|---:|---:|---|
| v0 | square/reduce/normalize/weight 四 kernel，两块中间缓冲 | 2.990 ms | 1.00× | 22.45 GB/s | 4 次 launch；约 9C 元素流量，相对融合理想 3C；四个 kernel 均受内存等待影响 |
| v1 | 每行一 block，tree reduction 后直接归一化并乘 weight | 0.907 ms | 3.30× | 73.99 GB/s | launch 4→1、逻辑流量约 9C→4C；仍有 9 次全 block reduction barrier |
| v2 | warp shuffle + warp 结果二级规约 | 0.897 ms | 3.33× | 74.85 GB/s | 指令较 v1 降 11.9%，主形状仅快 1.2%；长行已由内存流量主导 |
| v3 | 对齐 hidden 使用 `float4`，非对齐回退 v2 标量 | **0.712 ms** | **4.20×** | **94.27 GB/s** | 指令较 v2 降 58.6%，Memory throughput 升到 81.5%；仍为 256-thread 固定配置 |

完整正确性、逐形状性能与 Nsight 分解见 [`reports/v0.md`](reports/v0.md)。
v1 的 fusion 收益与流量分析见 [`reports/v1.md`](reports/v1.md)。
v2 的规约同步对照见 [`reports/v2.md`](reports/v2.md)。
v3 的 float4 路径与标量回退分析见 [`reports/v3.md`](reports/v3.md)。
