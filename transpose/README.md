# Transpose：从直接转置到合并访存

本目录实现二维 FP32 out-of-place transpose，输入布局为 `[rows, cols]`，输出
布局为 `[cols, rows]`。每个版本仍是可独立编译运行的 CUDA/C++ 程序。

```bash
mkdir -p build profiles
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 transpose_v0.cu -o build/transpose_v0
./build/transpose_v0 --rows=4096 --cols=4096
./build/transpose_v0 --rows=1000 --cols=1023
```

## 实测记录

| 版本 | 优化点 | `4096×4096` | 相对 v0 | 有效带宽 | 峰值占比 | 关键瓶颈 |
|---|---|---:|---:|---:|---:|---|
| v0 | 每线程复制一个元素，读取合并、写入跨行 | 8.879 ms | 1.00× | 15.12 GB/s | 14.76% | 全局写每请求 32 sectors，78% L2 sectors 多余 |
| v1 | `32×32` shared tile，全局读写均合并 | 2.418 ms | 3.67× | 55.51 GB/s | 54.21% | 全局访问已为 4 sectors/request；shared load 为 32.5-way bank conflict |
| v2 | shared tile padding 为 `32×33` | **1.417 ms** | **6.26×** | **94.70 GB/s** | **92.48%** | shared load 冲突从 32.5-way 降至 1.1-way；接近内存带宽上限 |
| v3 | 扫描 `blockRows=4/8/16/32`，保留 8 | 1.421 ms | 6.25× | 94.44 GB/s | 92.22% | 4 与 8 差异不足 1%；16/32 为负优化，32 的 occupancy 仅 63.95% |
| v4 | 对齐矩阵 `float4` load/store，非对齐标量回退 | 1.422 ms | 6.24× | 94.40 GB/s | 92.19% | 指令下降 66.7% 但耗时持平，内存瓶颈下的负优化 |

峰值带宽按锁定 EMC 3199 MHz、128-bit LPDDR5、DDR 传输计算为约
102.4 GB/s。CUDA 在该 Jetson 上报告的 `memoryClockRate` 与 GPU 1020 MHz
锁频相同，不能用于 LPDDR5 峰值计算。

完整数据与 Nsight 证据见 [`reports/v0.md`](reports/v0.md)。
v1 的全局事务与 shared bank conflict 对比见 [`reports/v1.md`](reports/v1.md)。
v2 的 padding 对照实验见 [`reports/v2.md`](reports/v2.md)。
v3 的线程块形状扫描与负优化分析见 [`reports/v3.md`](reports/v3.md)。
v4 的向量化负优化、稳定性与最终选择见 [`reports/v4.md`](reports/v4.md)。

## 当前选择

最终 standalone kernel 选择 v2 的 `32×33` padded tile、`32×8` block。v3
没有找到更优线程形状，v4 虽显著减少指令但没有降低耗时；连续两个方向提升均
不足 3%，Transpose 在此收口。主尺寸相对 v0 为 6.26×，有效带宽约为锁频峰值
的 92.5%。
