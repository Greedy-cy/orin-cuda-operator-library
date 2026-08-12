#include "operatorlib/operators.h"

#include <algorithm>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__global__ void reduce_ilp4_kernel(const float* input, float* output,
                                   std::size_t n) {
  extern __shared__ float shared[];
  const unsigned int lane = threadIdx.x;
  const std::size_t index = blockIdx.x * blockDim.x + lane;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t loop_stride = stride * 4;

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  std::size_t i = index;
  for (; i + 3 * stride < n; i += loop_stride) {
    sum0 += input[i];
    sum1 += input[i + stride];
    sum2 += input[i + 2 * stride];
    sum3 += input[i + 3 * stride];
  }
  for (; i < n; i += stride) sum0 += input[i];

  shared[lane] = (sum0 + sum1) + (sum2 + sum3);
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > warpSize;
       offset >>= 1) {
    if (lane < offset) shared[lane] += shared[lane + offset];
    __syncthreads();
  }
  if (lane < warpSize) {
    float value = shared[lane] + shared[lane + warpSize];
    value = warp_reduce_sum(value);
    if (lane == 0) atomicAdd(output, value);
  }
}

}  // namespace

namespace operatorlib {

cudaError_t reduce_sum_f32(const float* input, float* output, std::size_t n,
                           cudaStream_t stream) {
  if (input == nullptr || output == nullptr || n == 0)
    return cudaErrorInvalidValue;

  constexpr int threads = 256;
  constexpr int max_blocks = 64;  // 8 blocks/SM on the target 8-SM Orin.
  const std::size_t required_blocks = (n + threads - 1) / threads;
  const int blocks =
      static_cast<int>(std::min<std::size_t>(required_blocks, max_blocks));
  cudaError_t error = cudaMemsetAsync(output, 0, sizeof(float), stream);
  if (error != cudaSuccess) return error;
  reduce_ilp4_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
      input, output, n);
  return cudaGetLastError();
}

}  // namespace operatorlib
