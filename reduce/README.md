# Reduce Sum：从 atomic baseline 开始

本目录只关注 Reduce Sum 算子本身。当前阶段每个版本都是可以直接用
`nvcc` 编译运行的独立程序，不依赖公共框架、CMake 或 PyTorch。

## v0：每线程一次全局 atomicAdd

每个线程通过 grid-stride loop 累加自己负责的输入，最后对同一个全局地址
执行一次 `atomicAdd`。这个实现能正确处理任意正长度，但所有线程竞争同一
地址，预期瓶颈是全局 atomic 串行化。

```bash
nvcc -O3 -lineinfo -std=c++17 -arch=sm_87 reduce_v0.cu -o reduce_v0

./reduce_v0 --n=32
./reduce_v0 --n=1000
./reduce_v0 --n=1024
./reduce_v0 --n=4096
./reduce_v0 --n=1048576
./reduce_v0 --n=16777216
```

单次 profiling 使用较少的 warmup/repeat，避免采集无关 kernel：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output=reduce_v0_nsys --force-overwrite=true \
  ./reduce_v0 --n=16777216 --warmup=1 --repeat=1

/usr/local/cuda-12.6/bin/ncu --set full \
  --kernel-name regex:reduce_atomic_kernel --launch-skip 1 --launch-count 1 \
  --export reduce_v0_ncu --force-overwrite \
  ./reduce_v0 --n=16777216 --warmup=1 --repeat=1
```

## 实测记录

| 版本 | 优化点 | 主尺寸耗时 | 相对 v0 | 有效带宽 | 关键瓶颈 |
|---|---|---:|---:|---:|---|
| v0 | grid-stride + 每线程一次 atomic | 1.504 ms | 1.00× | 44.61 GB/s | 91.9% 的发射间隔在等待 L1TEX scoreboard，atomic 竞争使 warp 无法就绪 |
| v1 | shared-memory tree reduction，每 block 一次 atomic | 1.473 ms | 1.02× | 45.55 GB/s | atomic 数量下降但大尺寸仍有 91.7% L1TEX scoreboard 等待 |
| v2 | shared reduction 尾部改用 warp shuffle | 1.473 ms | 1.02× | 45.55 GB/s | occupancy 提高到 88.7%，但 scoreboard 等待仍为 91.8% |
| v3 | 四个独立累加器提供 load/add ILP | 0.718 ms | 2.09× | 93.43 GB/s | 内存吞吐升至 75.7%，转为接近硬件带宽上限的 memory-bound kernel |

测试条件：Jetson Orin Nano Super、MAXN_SUPER、GPU 1020 MHz、EMC 3199 MHz、
`N=16,777,216`、warmup 10 次、计时 30 次取 median。完整过程和 Nsight
指标见 [`reports/v0.md`](reports/v0.md)。

v0 的数据支持使用 block 内 shared-memory reduction，把全局 atomic 数量从
“每个参与线程一次”减少为“每个 block 一次”。v1 在短输入上最高达到约 2.13×，
但主尺寸只提升 2.1%，说明原假设只部分成立。完整分析见
[`reports/v1.md`](reports/v1.md)。v2 将只替换 block reduction 的尾部，使用
warp shuffle 减少 shared-memory 访问和同步。实测主尺寸与 v1 持平，详见
[`reports/v2.md`](reports/v2.md)。v3 将保留 v2 的规约尾部，引入四个独立
累加器，为标量全局加载提供 ILP。该假设得到验证，主尺寸提升约 2.05×，详见
[`reports/v3.md`](reports/v3.md)。v4 将把四次标量加载改为一次 `float4`
加载，验证在访存已基本合并时，减少 load 指令能否继续提高带宽。
