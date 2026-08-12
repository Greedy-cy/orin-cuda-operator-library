#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace operatorlib {

// All pointers refer to device memory. Launchers enqueue work on stream and
// never synchronize the device. They return argument or CUDA launch errors.

cudaError_t reduce_sum_f32(const float* input, float* output, std::size_t n,
                           cudaStream_t stream = nullptr);

cudaError_t softmax_f32(const float* input, float* output, int rows, int cols,
                        cudaStream_t stream = nullptr);

cudaError_t transpose_f32(const float* input, float* output, int rows, int cols,
                          cudaStream_t stream = nullptr);

cudaError_t rmsnorm_f32(const float* input, const float* weight, float* output,
                        int tokens, int hidden, float epsilon,
                        cudaStream_t stream = nullptr);

// Row-major C[M,N] = A[M,K] * B[K,N].
cudaError_t gemm_f32(const float* a, const float* b, float* c, int m, int n,
                     int k, cudaStream_t stream = nullptr);

// FP16 inputs, FP32 Tensor Core accumulation and FP32 output.
cudaError_t gemm_f16(const __half* a, const __half* b, float* c, int m, int n,
                     int k, cudaStream_t stream = nullptr);

// Q/K/V and output use contiguous [B,H,S,D] row-major storage.
cudaError_t attention_f32(const float* q, const float* k, const float* v,
                          float* output, int batch, int heads, int sequence,
                          int head_dim, bool causal,
                          cudaStream_t stream = nullptr);

// FP16 Q/K/V, FP32 online-softmax state and FP32 output.
cudaError_t attention_f16(const __half* q, const __half* k, const __half* v,
                          float* output, int batch, int heads, int sequence,
                          int head_dim, bool causal,
                          cudaStream_t stream = nullptr);

}  // namespace operatorlib
