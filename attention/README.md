# Attention / FlashAttention

本目录从显式保存 score 矩阵的 scaled dot-product attention 开始，逐步引入
kernel fusion、block tiling、online softmax、异步加载和 Tensor Core。输入和输出
统一采用 `[B,H,S,D]` 行主序布局；首期只实现 forward，并同时覆盖 causal 与
non-causal。

当前仍是 standalone CUDA/C++ 实验，不依赖 PyTorch Extension、根 CMake 或统一
算子框架。每个版本直接使用 `nvcc` 编译：

```bash
mkdir -p build
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 \
    attention_v0.cu -o build/attention_v0
./build/attention_v0 --b=1 --h=8 --s=128 --d=64 \
    --causal=0 --warmup=5 --repeat=15
```

## 实测记录

以下数据来自锁频后的 Jetson Orin Nano Super，时间为 warmup 后 15 次 CUDA Event
测量的 median。性能统一按 dense attention 的 `4*B*H*S*S*D` FLOPs 计算，使
causal/non-causal 结果可按同一口径观察；这不代表 causal 实际执行了同样数量的
有效乘加。

| 版本 | 优化点 | 主形状 | Median | Dense 性能 | 相对 v0 | cuDNN 占比 | 关键瓶颈 |
|---|---|---|---:|---:|---:|---:|---|
| v0 | 三 kernel：QK、串行稳定 Softmax、PV；显式 `[B,H,S,S]` FP32 缓冲 | B=1,H=12,S=1024,D=128 | 237.441 ms | 27.133 GFLOP/s | 1.00x | 延后到稳定公共评测代码 | QK 占 GPU kernel 时间 81.6%；L1/TEX 98.41%，85% global sectors 冗余 |
| v1 | warp 协作 dot-product、Q shared staging、QK+block Softmax 融合；仍显式保存概率 | B=1,H=12,S=1024,D=128 | **53.031 ms** | **121.484 GFLOP/s** | **4.48x** | 延后到 v2 common | L1/TEX 降到 44.77%，issue slots 升至 65.21%；global probability store 和 S² 缓冲仍存在 |
| v2 | `BM=8,BN=32` K/V shared tiling + FP32 online softmax；不保存 S² | B=1,H=12,S=1024,D=128 | 56.124 ms | 114.790 GFLOP/s | 4.23x | 待统一 cuDNN 接口 | non-causal 比 v1 慢 5.8%，但 causal 快 1.32x；compute 86.45%，active lanes 仅 23.9/warp |
| v3 | 扫描 BM/BN；D128 选 `32x32`，长序列 D64 选 `16x64` | B=1,H=12,S=1024,D=128 | **43.216 ms** | **149.076 GFLOP/s** | **5.50x** | 待统一 cuDNN 接口 | 比 v2 快 1.30x；40 registers、无 spill，66.7% occupancy 仍优于高 occupancy 小 BM |

v0 的正确性、全形状、显存、sanitizer 与 Nsight 证据见
[`reports/v0.md`](reports/v0.md)。v1 的合并访问、融合收益及 v0/v1 NCU 对照见
[`reports/v1.md`](reports/v1.md)。v2 的 online-softmax 推导、显存下降、负优化与
完整资源证据见 [`reports/v2.md`](reports/v2.md)。v3 的完整配置扫描、dispatcher
和 Nsight 对照见 [`reports/v3.md`](reports/v3.md)。

## 当前下一步

v4 将每次 online-softmax 更新从一个 key 改为一对 key：同时维护两个独立 dot
accumulator，再合并更新 `(m,l,O)`，减少串行 rescale/exp 次数并增加指令级并行。
