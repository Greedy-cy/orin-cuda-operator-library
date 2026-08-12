#include "operatorlib/operators.h"

#include <cfloat>

namespace {

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_value = lane < (blockDim.x >> 5) ? shared[lane] : -FLT_MAX;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) shared[0] = block_value;
  }
  __syncthreads();
  return shared[0];
}

__device__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __syncthreads();
  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_value = lane < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) shared[0] = block_value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void softmax_warp_cached_kernel(const float* input, float* output,
                                           int rows, int cols) {
  constexpr int warps_per_block = 8;
  constexpr int items_per_lane = 4;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= rows) return;
  const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
  float values[items_per_lane];
  float maximum = -FLT_MAX;
#pragma unroll
  for (int item = 0; item < items_per_lane; ++item) {
    const int col = lane + item * warpSize;
    values[item] = col < cols ? input[row_offset + col] : -FLT_MAX;
    maximum = fmaxf(maximum, values[item]);
  }
  maximum = __shfl_sync(0xffffffff, warp_reduce_max(maximum), 0);
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < items_per_lane; ++item) {
    const int col = lane + item * warpSize;
    values[item] = col < cols ? expf(values[item] - maximum) : 0.0f;
    sum += values[item];
  }
  const float inverse_sum =
      1.0f / __shfl_sync(0xffffffff, warp_reduce_sum(sum), 0);
#pragma unroll
  for (int item = 0; item < items_per_lane; ++item) {
    const int col = lane + item * warpSize;
    if (col < cols) output[row_offset + col] = values[item] * inverse_sum;
  }
}

__global__ void softmax_block_cached_float4_kernel(const float* input,
                                                    float* output, int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int vector_count = cols / 4;
  const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
  const float4* row_input = reinterpret_cast<const float4*>(input + row_offset);
  float4* row_output = reinterpret_cast<float4*>(output + row_offset);
  const bool active = lane < vector_count;
  float4 value = active ? row_input[lane]
                        : make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  float maximum = fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w));
  maximum = block_reduce_max(maximum, shared);
  if (active) {
    value.x = expf(value.x - maximum);
    value.y = expf(value.y - maximum);
    value.z = expf(value.z - maximum);
    value.w = expf(value.w - maximum);
  } else {
    value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const float local_sum = (value.x + value.y) + (value.z + value.w);
  const float inverse_sum = 1.0f / block_reduce_sum(local_sum, shared);
  if (active) {
    value.x *= inverse_sum;
    value.y *= inverse_sum;
    value.z *= inverse_sum;
    value.w *= inverse_sum;
    row_output[lane] = value;
  }
}

__global__ void softmax_block_general_kernel(const float* input, float* output,
                                             int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const std::size_t row_offset = static_cast<std::size_t>(row) * cols;
  const bool aligned = (cols & 3) == 0;
  float maximum = -FLT_MAX;
  if (aligned) {
    const float4* row_input =
        reinterpret_cast<const float4*>(input + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      const float4 value = row_input[i];
      maximum = fmaxf(maximum, value.x);
      maximum = fmaxf(maximum, value.y);
      maximum = fmaxf(maximum, value.z);
      maximum = fmaxf(maximum, value.w);
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x)
      maximum = fmaxf(maximum, input[row_offset + col]);
  }
  maximum = block_reduce_max(maximum, shared);
  float sum = 0.0f;
  if (aligned) {
    const float4* row_input =
        reinterpret_cast<const float4*>(input + row_offset);
    float4* row_output = reinterpret_cast<float4*>(output + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      const float4 source = row_input[i];
      const float4 value = {expf(source.x - maximum), expf(source.y - maximum),
                            expf(source.z - maximum), expf(source.w - maximum)};
      row_output[i] = value;
      sum += (value.x + value.y) + (value.z + value.w);
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x) {
      const float value = expf(input[row_offset + col] - maximum);
      output[row_offset + col] = value;
      sum += value;
    }
  }
  const float inverse_sum = 1.0f / block_reduce_sum(sum, shared);
  if (aligned) {
    float4* row_output = reinterpret_cast<float4*>(output + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      float4 value = row_output[i];
      value.x *= inverse_sum;
      value.y *= inverse_sum;
      value.z *= inverse_sum;
      value.w *= inverse_sum;
      row_output[i] = value;
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x)
      output[row_offset + col] *= inverse_sum;
  }
}

}  // namespace

namespace operatorlib {

cudaError_t softmax_f32(const float* input, float* output, int rows, int cols,
                        cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  if (cols <= 128) {
    constexpr int warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    softmax_warp_cached_kernel<<<blocks, threads, 0, stream>>>(input, output,
                                                               rows, cols);
  } else if (cols <= 1024 && (cols & 3) == 0) {
    softmax_block_cached_float4_kernel<<<
        rows, threads, (threads / 32) * sizeof(float), stream>>>(input, output,
                                                                 cols);
  } else {
    softmax_block_general_kernel<<<
        rows, threads, (threads / 32) * sizeof(float), stream>>>(input, output,
                                                                 cols);
  }
  return cudaGetLastError();
}

}  // namespace operatorlib
