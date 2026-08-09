# Softmax：从 shared-memory baseline 开始

本目录实现最后一维的稳定 Softmax。当前阶段使用独立 CUDA/C++ 程序，不依赖
PyTorch、CMake 或公共 benchmark 框架。

## v0：一行一个 block

v0 使用 256 个线程完成三步：

1. shared-memory tree reduction 求行最大值；
2. 计算 `exp(x-max)`，写入全局输出，再进行 shared-memory sum reduction；
3. 从全局输出读回指数值并归一化。

```bash
mkdir -p build profile
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 softmax_v0.cu -o build/softmax_v0

./build/softmax_v0 --rows=4096 --cols=1024
./build/softmax_v0 --rows=128 --cols=8192
./build/softmax_v0 --rows=4 --cols=1003 --extreme
```

## 实测记录

| 版本 | 优化点 | 主尺寸耗时 | 相对 v0 | 有效 I/O 带宽 | 关键瓶颈 |
|---|---|---:|---:|---:|---|
| v0 | 两次 shared tree reduction，中间指数值写全局内存 | 0.668 ms | 1.00× | 50.24 GB/s | Compute/Memory 约 60% 均衡；19 次 block 同步和中间全局读写都有优化空间 |
| v1 | warp shuffle + warp 结果二级规约 | 0.596 ms | 1.12× | 56.26 GB/s | 指令下降 27%，转为 Memory 67.9% 主导；中间指数值仍全局往返 |
| v2 | 对齐行使用 `float4`，非对齐回退标量 | 0.426 ms | 1.57× | 78.67 GB/s | L2 吞吐 93.1%；`cols=128` 因仅一个 warp 有数据工作而略退化 |
| v3 | `cols<=128` warp-per-row，其余保留 block path | 0.427 ms | 1.57× | 78.66 GB/s | 主形状持平；`4096×128` 达 0.0627 ms，相对 v0 5.24× |
| v4 | 资源允许的短/中等行在寄存器缓存输入和指数值 | **0.354 ms** | **1.89×** | **94.84 GB/s** | 主形状只做最小输入/输出流量；大行和非对齐行保留通用路径 |

主尺寸为 `rows=4096, cols=1024`，锁定 GPU 1020 MHz、EMC 3199 MHz。
完整正确性、性能和 Nsight 记录见 [`reports/v0.md`](reports/v0.md)。
v1 的完整对比见 [`reports/v1.md`](reports/v1.md)。
v2 的尺寸分化和 Nsight 证据见 [`reports/v2.md`](reports/v2.md)。
v3 的短行专用 kernel 分析见 [`reports/v3.md`](reports/v3.md)。
v4 的寄存器缓存、完整收口数据和停止条件见 [`reports/v4.md`](reports/v4.md)。

## 正确性修正记录

v4 的首次 `compute-sanitizer --tool racecheck` 发现 max 规约结果仍被其他
warp 读取时，sum 规约可能覆盖相同 shared-memory 槽位。修复是在两阶段之间
增加 block 同步，并将相同修正回填 v0-v3。修复后五个版本在
`17×1024` 上均为 `0 hazards`；上表和各报告性能均来自修复后的二进制。
