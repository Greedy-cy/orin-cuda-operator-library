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

测试条件：Jetson Orin Nano Super、MAXN_SUPER、GPU 1020 MHz、EMC 3199 MHz、
`N=16,777,216`、warmup 10 次、计时 30 次取 median。完整过程和 Nsight
指标见 [`reports/v0.md`](reports/v0.md)。

v0 的数据支持下一步使用 block 内 shared-memory reduction，把全局 atomic
数量从“每个参与线程一次”减少为“每个 block 一次”。
