# Standalone final benchmark 冻结记录

## 1. 冻结范围

本记录只冻结六类算子的 standalone final，不改变各目录原有的 v0→final 版本、
独立 `main()`、CPU/NVIDIA 参考和详细 Nsight 报告。冻结使用的 kernel 源码来自
commit `576b2c9`；此后的文档与复现脚本提交不修改这些 kernel。

最终选择如下：

| 算子 | FP32 final | FP16 final/补充路线 |
|---|---|---|
| Reduce Sum | v3，四路独立累加 + warp shuffle | 暂无 |
| Softmax | v4，按尺寸分派 warp/block/register cache | FP16 延后到库接口阶段 |
| Transpose | v2，`32x33` padded shared tile | 暂无 |
| RMSNorm | v3，融合规约/归一化/weight + float4 | FP16 延后到库接口阶段 |
| GEMM | v6，配置扫描后的 cp.async tiled kernel | v8，WMMA + shared skew + cp.async |
| Attention | v5，FP32 cp.async online softmax | v6，FP16 WMMA + FP32 online state |

## 2. 环境

| 项目 | 冻结值 |
|---|---|
| 设备 | Jetson Orin Nano Engineering Reference Developer Kit Super，8 SM |
| Jetson Linux | R36.4.7 |
| CUDA / nvcc | CUDA 12.6，nvcc 12.6.68 |
| cuDNN | 9.3.0.75 |
| 架构与编译 | `-O3 -lineinfo -std=c++17 -arch=sm_87` |
| 功耗模式 | MAXN_SUPER |
| GPU | min=max=1,020,000,000 Hz |
| EMC | cap=rate=3,199,000,000 Hz |
| 计时 | CUDA Event，warmup 后取 median |

该 JetPack 镜像重启后即使处于 MAXN_SUPER，`jetson_clocks` 仍可能只把 EMC 锁到
2.133 GHz。实测此时四个内存型主形状只有约 63–65 GB/s，而历史正式数据约
93–95 GB/s。运行 `sudo tools/lock_orin_clocks.sh` 会显式恢复并校验 3.199 GHz；
若校验失败，不应记录正式数据。

Transpose 报告中的 102.4 GB/s 峰值使用 128-bit LPDDR5 与 3.199 GHz 数据时钟
口径；不能使用 CUDA `memoryClockRate` 或未校验的 sysfs 数值重新推导。

## 3. 冻结数据与复验

“报告值”来自各版本详细报告；“复验值”是 2026-08-12 在相同主形状、重新锁频和
重新编译后得到的独立 median。两列差异均在正常波动范围内。

| 算子/version | 主形状 | 报告值 | 复验值 | 最终结论 |
|---|---|---:|---:|---|
| Reduce v3 | N=16,777,216 | 0.718 ms / 93.43 GB/s | **0.716464 ms / 93.67 GB/s** | 相对 v0 2.09x；v4 向量化未采用 |
| Softmax v4 | 4096x1024 | 0.354 ms / 94.84 GB/s | **0.353024 ms / 95.05 GB/s** | 相对 v0 1.89x |
| Transpose v2 | 4096x4096 | 1.417 ms / 94.70 GB/s | **1.422720 ms / 94.34 GB/s** | 相对 v0 6.26x，约峰值 92% |
| RMSNorm v3 | 2048x4096 | 0.712 ms / 94.27 GB/s | **0.707856 ms / 94.83 GB/s** | 相对 v0 4.20x |
| GEMM FP32 v6 | 4096^3 | 115.418 ms / 1.191 TFLOP/s | **115.394783 ms / 1.191 TFLOP/s** | cuBLAS 的 **87.88%** |
| GEMM FP16 v8 | 4096^3 | 26.505 ms / 5.185 TFLOP/s | **26.447104 ms / 5.197 TFLOP/s** | FP16 cuBLAS 的 **64.54%** |
| Attention FP32 v5 | B1,H12,S1024,D128 | 37.331 ms / 172.58 GFLOP/s | **37.339615 ms / 172.54 GFLOP/s** | 相对 FP32 v0 6.36x |
| Attention FP16 v6 | B1,H12,S1024,D128 | 20.733 ms / 310.74 GFLOP/s | **20.817280 ms / 309.48 GFLOP/s** | 无 S^2 score；FP32→FP16 全路线 11.4x |

GEMM 的 cuBLAS 比例来自相同 dtype、相同输入与同一可执行文件中的独立 CUDA Event
median。Attention 不把 cuDNN 作为完成条件：主要成果是相对 custom FP32 naive 的
演进、online-softmax 数值正确性、48 MiB S^2 中间矩阵消除、S=4096 压力测试和
Nsight 证据。FP16 v6 相对 FP32 v0 的 11.4x 包含 dtype 变化，不能表述为同精度
kernel 加速。

## 4. 边界路径复验

同一次审计还重新运行了以下非对齐/极端路径，全部正确：

- Reduce `N=1000`；
- Softmax `17x1003, extreme`；
- Transpose `1000x1023`；
- RMSNorm `17x1003` scalar fallback；
- GEMM FP32/FP16 `M37,N65,K1003` scalar fallback，与各自 cuBLAS 对齐；
- Attention FP32/FP16 `B1,H2,S37,D71,causal,extreme` scalar fallback，与量化口径
  double CPU reference 对齐。

复杂同步路径此前已经在对应版本报告中完成 memcheck、racecheck 和 synccheck；本次
没有重复无变化的全量 sanitizer。

## 5. 复现命令

```bash
git clone https://github.com/Greedy-cy/orin-cuda-operator-library.git
cd orin-cuda-operator-library
sudo ./tools/lock_orin_clocks.sh
./tools/run_final_standalone.sh smoke
./tools/run_final_standalone.sh benchmark
```

`smoke` 会重新编译 final 并验证非对齐路径；`benchmark` 会先校验锁频，再运行上表
全部主形状。脚本不会删除历史二进制、profile 或报告。
