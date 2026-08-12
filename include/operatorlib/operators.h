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

}  // namespace operatorlib
