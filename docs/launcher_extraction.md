# Final launcher 提取记录

## 1. 目的与边界

六类 standalone 算子稳定后，才从已验证版本中提取最小库接口。提取不修改
`<operator>_v0.cu` 到 final 的实验代码；`src/` 只保留 kernel、尺寸 dispatcher 和
launcher，移除数据生成、CPU reference、计时与 `main()`。

公开接口位于 `include/operatorlib/operators.h`。统一约定：

- 参数是 device pointer；
- 调用方显式传入 `cudaStream_t`；
- launcher 只将工作入队，不做 `cudaDeviceSynchronize`；
- 返回 `cudaError_t`，非法尺寸/空指针返回 `cudaErrorInvalidValue`；
- 仍以 Jetson Orin Nano Super / sm_87 为首要目标，不提前设计通用模板框架。

## 2. 第一批 FP32 core launcher

| API | 来源 | 保留的 final 策略 | 边界处理 |
|---|---|---|---|
| `reduce_sum_f32` | Reduce v3 | 四路 ILP、shared/warp reduction、最多 64 blocks | launcher 在同一 stream 先清零输出；任意 N>0 |
| `softmax_f32` | Softmax v4 | `cols<=128` warp cache；对齐 `cols<=1024` register/float4 cache；其余 general | general 同时覆盖大列和非 4 对齐 |
| `transpose_f32` | Transpose v2 | `32x33` padded shared tile、`32x8` block | 每次 load/store 都检查矩形边界 |
| `rmsnorm_f32` | RMSNorm v3 | 单 kernel 融合；hidden 对齐时 float4 | 非 4 对齐 hidden 走 scalar fallback |

Reduce 仍使用 v3 的 block-level atomic final，因此 launcher 必须负责输出清零。清零
使用 `cudaMemsetAsync` 并与 kernel 放在调用方 stream 中，不引入全局同步。64-block
上限对应目标设备 8 SM x 8 blocks/SM，与 standalone v3 的 Orin dispatcher 一致。

## 3. 验证

`tests/core_ops_smoke.cu` 创建 non-blocking CUDA stream，在该 stream 上执行 H2D、
launcher 和 D2H，确保接口没有偷偷依赖 default stream 或 device synchronize。

覆盖矩阵：

- Reduce：`N=1000`；
- Softmax：`19x128` warp path、`17x1024` cached float4 path、
  `17x1003` extreme general path；
- Transpose：`37x65` 非对齐矩形；
- RMSNorm：`32x4096` float4 path、`17x1003` scalar path。

全部与 CPU reference 对齐并 PASS。对同一个测试二进制执行：

```text
compute-sanitizer --tool memcheck  : 0 errors
compute-sanitizer --tool racecheck : 0 hazards
compute-sanitizer --tool synccheck : 0 errors
```

本阶段仍直接使用 `nvcc` 链接四个 `.cu`，尚未引入 CMake；等 GEMM/Attention
launcher 均完成后再建立一次最小构建入口。
