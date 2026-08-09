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

峰值带宽按锁定 EMC 3199 MHz、128-bit LPDDR5、DDR 传输计算为约
102.4 GB/s。CUDA 在该 Jetson 上报告的 `memoryClockRate` 与 GPU 1020 MHz
锁频相同，不能用于 LPDDR5 峰值计算。

完整数据与 Nsight 证据见 [`reports/v0.md`](reports/v0.md)。
