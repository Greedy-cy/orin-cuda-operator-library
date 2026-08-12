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

## 4. GEMM launcher

`gemm_f32` 从 v6 只提取最终 auto-dispatcher 实际会选择的三项配置，不把用于扫描
的八项候选全部带入库：

- `M=1`：`1x256x16 / 1x1`；
- `M<=16`：`16x128x16 / 1x8`；
- regular M：`128x64x16 / 8x4`；
- 任一 tile 条件不满足时：`64x64x16 / 4x4` scalar boundary fallback。

`gemm_f16` 保留 v8 的 `128x128x16` WMMA tiled、16-row small-M、WMMA baseline
fallback 和任意非对齐 scalar fallback。输入为 FP16，Tensor Core/标量路径都使用
FP32 accumulation，输出 FP32。

`tests/gemm_smoke.cu` 在 non-blocking stream 上逐一触发上述八条路径，并与相同
stream、相同 dtype 的 cuBLAS 对齐，全部 PASS。memcheck 0 errors、racecheck 0
hazards、synccheck 0 errors。ptxas 资源与 standalone final 保持一致：FP32 regular
107 registers、FP16 tiled 115 registers，所有路径 0 spill。

## 5. Attention launcher

库接口不依赖 cuDNN，保留两条来自 standalone final 的实现：

- `attention_f32`：v5 cp.async online-softmax。`D=64/128 && S%32=0` 使用异步
  tiled kernel，其余走 FP32 scalar online fallback；
- `attention_f16`：v6 FP16 WMMA QK/PV、FP32 online state/output。
  `D=64/128 && S%16=0` 使用 WMMA，其余走 FP16-input/FP32-compute fallback。

两者输入/输出布局均为连续 `[B,H,S,D]`，支持 causal/non-causal，只实现 inference
forward，并始终不分配 S^2 score。FP32 D128 路径需要 64 KiB dynamic shared，
launcher 在 launch 前设置对应 kernel attribute；该操作不引入 device synchronize。

`tests/attention_smoke.cu` 对 FP32/FP16 分别验证 D64、D128 causal 和
`S37,D71,causal,extreme` fallback，输入在 non-blocking stream 上传输，并与 double
CPU reference 对齐。六条路径全部 PASS；memcheck 0 errors、racecheck 0 hazards、
synccheck 0 errors。ptxas 与原版本一致：v5 44 registers、v6 47 registers，0 spill。

至此六类 final launcher 均已提取。下一步才增加根目录最小 CMake 和库级测试入口；
各算子目录的独立 `nvcc` 基准仍作为性能口径，不被新构建系统替代。
