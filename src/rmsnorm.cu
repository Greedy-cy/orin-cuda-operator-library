#include "operatorlib/operators.h"

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ float block_reduce_sum(float value, float* warp_results) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) warp_results[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x >> 5) ? warp_results[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) warp_results[0] = block_sum;
  }
  __syncthreads();
  return warp_results[0];
}

__global__ void rmsnorm_float4_kernel(const float4* input,
                                      const float4* weight, float4* output,
                                      int vectors_per_row, int hidden,
                                      float epsilon) {
  extern __shared__ float warp_results[];
  const int token = blockIdx.x;
  const std::size_t row_offset =
      static_cast<std::size_t>(token) * vectors_per_row;
  float square_sum = 0.0f;
  for (int index = threadIdx.x; index < vectors_per_row; index += blockDim.x) {
    const float4 value = input[row_offset + index];
    square_sum += (value.x * value.x + value.y * value.y) +
                  (value.z * value.z + value.w * value.w);
  }
  const float inverse_rms =
      rsqrtf(block_reduce_sum(square_sum, warp_results) / hidden + epsilon);
  for (int index = threadIdx.x; index < vectors_per_row; index += blockDim.x) {
    const float4 value = input[row_offset + index];
    const float4 scale = weight[index];
    output[row_offset + index] =
        make_float4(value.x * inverse_rms * scale.x,
                    value.y * inverse_rms * scale.y,
                    value.z * inverse_rms * scale.z,
                    value.w * inverse_rms * scale.w);
  }
}

__global__ void rmsnorm_scalar_kernel(const float* input, const float* weight,
                                      float* output, int hidden,
                                      float epsilon) {
  extern __shared__ float warp_results[];
  const int token = blockIdx.x;
  const std::size_t row_offset = static_cast<std::size_t>(token) * hidden;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = input[row_offset + col];
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_reduce_sum(square_sum, warp_results) / hidden + epsilon);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    output[row_offset + col] =
        input[row_offset + col] * inverse_rms * weight[col];
  }
}

}  // namespace

namespace operatorlib {

cudaError_t rmsnorm_f32(const float* input, const float* weight, float* output,
                        int tokens, int hidden, float epsilon,
                        cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      tokens <= 0 || hidden <= 0 || epsilon <= 0.0f)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  constexpr int shared_bytes = (threads / 32) * sizeof(float);
  if ((hidden & 3) == 0) {
    rmsnorm_float4_kernel<<<tokens, threads, shared_bytes, stream>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(weight),
        reinterpret_cast<float4*>(output), hidden / 4, hidden, epsilon);
  } else {
    rmsnorm_scalar_kernel<<<tokens, threads, shared_bytes, stream>>>(
        input, weight, output, hidden, epsilon);
  }
  return cudaGetLastError();
}

}  // namespace operatorlib
