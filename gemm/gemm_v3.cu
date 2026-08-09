#include "gemm_common.h"

namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;
constexpr int kThreads = 256;

template <bool Vectorized>
__global__ __launch_bounds__(kThreads) void sgemm_vectorized_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k) {
  __shared__ __align__(16) float tile_a[kBlockM][kBlockK];
  __shared__ __align__(16) float tile_b[kBlockK][kBlockN];

  const int tid = threadIdx.x;
  const int thread_tile_row = tid / (kBlockN / kThreadN);
  const int thread_tile_col = tid % (kBlockN / kThreadN);
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  float accumulators[kThreadM][kThreadN] = {};

  for (int tile_begin = 0; tile_begin < k; tile_begin += kBlockK) {
    if constexpr (Vectorized) {
      const int a_row = tid / (kBlockK / 4);
      const int a_col = (tid % (kBlockK / 4)) * 4;
      const int b_row = tid / (kBlockN / 4);
      const int b_col = (tid % (kBlockN / 4)) * 4;
      *reinterpret_cast<float4*>(&tile_a[a_row][a_col]) =
          *reinterpret_cast<const float4*>(
              &a[static_cast<size_t>(block_row + a_row) * k + tile_begin +
                 a_col]);
      *reinterpret_cast<float4*>(&tile_b[b_row][b_col]) =
          *reinterpret_cast<const float4*>(
              &b[static_cast<size_t>(tile_begin + b_row) * n + block_col +
                 b_col]);
    } else {
#pragma unroll
      for (int load = tid; load < kBlockM * kBlockK; load += kThreads) {
        const int tile_row = load / kBlockK;
        const int tile_col = load % kBlockK;
        const int global_row = block_row + tile_row;
        const int global_col = tile_begin + tile_col;
        tile_a[tile_row][tile_col] =
            (global_row < m && global_col < k)
                ? a[static_cast<size_t>(global_row) * k + global_col]
                : 0.0f;
      }
#pragma unroll
      for (int load = tid; load < kBlockK * kBlockN; load += kThreads) {
        const int tile_row = load / kBlockN;
        const int tile_col = load % kBlockN;
        const int global_row = tile_begin + tile_row;
        const int global_col = block_col + tile_col;
        tile_b[tile_row][tile_col] =
            (global_row < k && global_col < n)
                ? b[static_cast<size_t>(global_row) * n + global_col]
                : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlockK; ++inner) {
      float fragment_a[kThreadM];
      float fragment_b[kThreadN];
#pragma unroll
      for (int i = 0; i < kThreadM; ++i) {
        fragment_a[i] = tile_a[thread_tile_row * kThreadM + i][inner];
      }
#pragma unroll
      for (int j = 0; j < kThreadN; ++j) {
        fragment_b[j] = tile_b[inner][thread_tile_col * kThreadN + j];
      }
#pragma unroll
      for (int i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
          accumulators[i][j] =
              fmaf(fragment_a[i], fragment_b[j], accumulators[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int output_col = block_col + thread_tile_col * kThreadN;
#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int output_row = block_row + thread_tile_row * kThreadM + i;
    if constexpr (Vectorized) {
      const float4 result = make_float4(
          accumulators[i][0], accumulators[i][1], accumulators[i][2],
          accumulators[i][3]);
      *reinterpret_cast<float4*>(
          &c[static_cast<size_t>(output_row) * n + output_col]) = result;
    } else {
#pragma unroll
      for (int j = 0; j < kThreadN; ++j) {
        const int col = output_col + j;
        if (output_row < m && col < n) {
          c[static_cast<size_t>(output_row) * n + col] = accumulators[i][j];
        }
      }
    }
  }
}

void launch_custom(const float* a, const float* b, float* c, int m, int n,
                   int k) {
  const dim3 block(kThreads);
  const dim3 grid((n + kBlockN - 1) / kBlockN,
                  (m + kBlockM - 1) / kBlockM);
  if (m % kBlockM == 0 && n % kBlockN == 0 && k % kBlockK == 0) {
    sgemm_vectorized_kernel<true><<<grid, block>>>(a, b, c, m, n, k);
  } else {
    sgemm_vectorized_kernel<false><<<grid, block>>>(a, b, c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  return gemm_common::run_benchmark(
      argc, argv, "gemm_v3", "v3_float4", "tile=64x64x16 thread_tile=4x4",
      launch_custom);
}
