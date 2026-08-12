#include "operatorlib/operators.h"

#include <mma.h>

namespace {

__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                            const void* global_source) {
  const unsigned shared_address =
      static_cast<unsigned>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(shared_address), "l"(global_source));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
}

// ------------------------------ FP32 v6 ------------------------------

constexpr int kFp32Threads = 256;

template <int BM, int BN, int BK>
__device__ __forceinline__ void issue_float_tile(
    const float* __restrict__ a, const float* __restrict__ b, float* shared_a,
    float* shared_b, int block_row, int block_col, int tile_begin, int n,
    int k) {
  constexpr int a_vectors = BM * BK / 4;
  constexpr int b_vectors = BK * BN / 4;
  const int tid = threadIdx.x;
#pragma unroll
  for (int vector_index = tid; vector_index < a_vectors;
       vector_index += kFp32Threads) {
    const int element = vector_index * 4;
    const int row = element / BK;
    const int col = element % BK;
    cp_async_16(&shared_a[element],
                &a[static_cast<std::size_t>(block_row + row) * k +
                   tile_begin + col]);
  }
#pragma unroll
  for (int vector_index = tid; vector_index < b_vectors;
       vector_index += kFp32Threads) {
    const int element = vector_index * 4;
    const int row = element / BN;
    const int col = element % BN;
    cp_async_16(&shared_b[element],
                &b[static_cast<std::size_t>(tile_begin + row) * n +
                   block_col + col]);
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(kFp32Threads) void sgemm_configured_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int n, int k) {
  static_assert((BM / TM) * (BN / TN) == kFp32Threads);
  __shared__ __align__(16) float tile_a[2][BM][BK];
  __shared__ __align__(16) float tile_b[2][BK][BN];
  const int tid = threadIdx.x;
  const int thread_tile_row = tid / (BN / TN);
  const int thread_tile_col = tid % (BN / TN);
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  float accumulators[TM][TN] = {};

  issue_float_tile<BM, BN, BK>(a, b, &tile_a[0][0][0], &tile_b[0][0][0],
                               block_row, block_col, 0, n, k);
  cp_async_commit();
  const int tile_count = k / BK;
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int read_buffer = tile_index & 1;
    const int write_buffer = read_buffer ^ 1;
    if (tile_index + 1 < tile_count) {
      issue_float_tile<BM, BN, BK>(
          a, b, &tile_a[write_buffer][0][0], &tile_b[write_buffer][0][0],
          block_row, block_col, (tile_index + 1) * BK, n, k);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
      float fragment_a[TM];
      float fragment_b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        fragment_a[i] = tile_a[read_buffer][thread_tile_row * TM + i][inner];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        fragment_b[j] = tile_b[read_buffer][inner][thread_tile_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j)
          accumulators[i][j] =
              fmaf(fragment_a[i], fragment_b[j], accumulators[i][j]);
      }
    }
    __syncthreads();
  }

  const int output_col = block_col + thread_tile_col * TN;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int output_row = block_row + thread_tile_row * TM + i;
    if constexpr (TN % 4 == 0) {
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 result =
            make_float4(accumulators[i][j], accumulators[i][j + 1],
                        accumulators[i][j + 2], accumulators[i][j + 3]);
        *reinterpret_cast<float4*>(
            &c[static_cast<std::size_t>(output_row) * n + output_col + j]) =
            result;
      }
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j)
        c[static_cast<std::size_t>(output_row) * n + output_col + j] =
            accumulators[i][j];
    }
  }
}

__global__ __launch_bounds__(kFp32Threads) void sgemm_scalar_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 4;
  constexpr int TN = 4;
  __shared__ float tile_a[BM][BK];
  __shared__ float tile_b[BK][BN];
  const int tid = threadIdx.x;
  const int thread_tile_row = tid / (BN / TN);
  const int thread_tile_col = tid % (BN / TN);
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  float accumulators[TM][TN] = {};
  for (int tile_begin = 0; tile_begin < k; tile_begin += BK) {
    for (int load = tid; load < BM * BK; load += kFp32Threads) {
      const int row = load / BK;
      const int col = load % BK;
      const int global_row = block_row + row;
      const int global_col = tile_begin + col;
      tile_a[row][col] =
          global_row < m && global_col < k
              ? a[static_cast<std::size_t>(global_row) * k + global_col]
              : 0.0f;
    }
    for (int load = tid; load < BK * BN; load += kFp32Threads) {
      const int row = load / BN;
      const int col = load % BN;
      const int global_row = tile_begin + row;
      const int global_col = block_col + col;
      tile_b[row][col] =
          global_row < k && global_col < n
              ? b[static_cast<std::size_t>(global_row) * n + global_col]
              : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
      float fragment_a[TM];
      float fragment_b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        fragment_a[i] = tile_a[thread_tile_row * TM + i][inner];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        fragment_b[j] = tile_b[inner][thread_tile_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j)
          accumulators[i][j] =
              fmaf(fragment_a[i], fragment_b[j], accumulators[i][j]);
      }
    }
    __syncthreads();
  }
  const int output_col = block_col + thread_tile_col * TN;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int output_row = block_row + thread_tile_row * TM + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = output_col + j;
      if (output_row < m && col < n)
        c[static_cast<std::size_t>(output_row) * n + col] = accumulators[i][j];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_sgemm_config(const float* a, const float* b, float* c, int m,
                         int n, int k, cudaStream_t stream) {
  if (m % BM != 0 || n % BN != 0 || k % BK != 0) {
    const dim3 grid((n + 63) / 64, (m + 63) / 64);
    sgemm_scalar_kernel<<<grid, kFp32Threads, 0, stream>>>(a, b, c, m, n, k);
    return;
  }
  const dim3 grid(n / BN, m / BM);
  sgemm_configured_kernel<BM, BN, BK, TM, TN>
      <<<grid, kFp32Threads, 0, stream>>>(a, b, c, n, k);
}

// ------------------------------ FP16 v8 ------------------------------

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kHalfBlockM = 128;
constexpr int kHalfBlockN = 128;
constexpr int kHalfBlockK = 16;
constexpr int kSkew = 8;
constexpr int kWarpsPerBlock = 8;
constexpr int kHalfThreads = kWarpsPerBlock * 32;

template <int BM, int BN>
__device__ __forceinline__ void issue_half_tile(
    const __half* __restrict__ a, const __half* __restrict__ b,
    __half* shared_a, __half* shared_b, int block_row, int block_col,
    int tile_begin, int n, int k) {
  constexpr int a_vectors = BM * kHalfBlockK / 8;
  constexpr int b_vectors = kHalfBlockK * BN / 8;
  const int tid = threadIdx.x;
  for (int vector_index = tid; vector_index < a_vectors;
       vector_index += kHalfThreads) {
    const int element = vector_index * 8;
    const int row = element / kHalfBlockK;
    const int col = element % kHalfBlockK;
    cp_async_16(&shared_a[row * (kHalfBlockK + kSkew) + col],
                &a[static_cast<std::size_t>(block_row + row) * k +
                   tile_begin + col]);
  }
  for (int vector_index = tid; vector_index < b_vectors;
       vector_index += kHalfThreads) {
    const int element = vector_index * 8;
    const int row = element / BN;
    const int col = element % BN;
    cp_async_16(&shared_b[row * (BN + kSkew) + col],
                &b[static_cast<std::size_t>(tile_begin + row) * n +
                   block_col + col]);
  }
}

__global__ __launch_bounds__(kHalfThreads) void wmma_tiled_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b,
    float* __restrict__ c, int n, int k) {
  __shared__ __align__(16)
      __half tile_a[2][kHalfBlockM][kHalfBlockK + kSkew];
  __shared__ __align__(16)
      __half tile_b[2][kHalfBlockK][kHalfBlockN + kSkew];
  const int warp = threadIdx.x / warpSize;
  const int warp_group_row = warp / 2;
  const int warp_group_col = warp % 2;
  const int block_row = blockIdx.y * kHalfBlockM;
  const int block_col = blockIdx.x * kHalfBlockN;
  using Accumulator = nvcuda::wmma::fragment<
      nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
  Accumulator accumulators[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      nvcuda::wmma::fill_fragment(accumulators[i][j], 0.0f);
  }
  issue_half_tile<kHalfBlockM, kHalfBlockN>(
      a, b, &tile_a[0][0][0], &tile_b[0][0][0], block_row, block_col, 0, n, k);
  cp_async_commit();
  const int tile_count = k / kHalfBlockK;
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int read_buffer = tile_index & 1;
    const int write_buffer = read_buffer ^ 1;
    if (tile_index + 1 < tile_count) {
      issue_half_tile<kHalfBlockM, kHalfBlockN>(
          a, b, &tile_a[write_buffer][0][0], &tile_b[write_buffer][0][0],
          block_row, block_col, (tile_index + 1) * kHalfBlockK, n, k);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                           __half, nvcuda::wmma::row_major>
        fragment_a[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                           __half, nvcuda::wmma::row_major>
        fragment_b[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = warp_group_row * 32 + i * kWmmaM;
      nvcuda::wmma::load_matrix_sync(fragment_a[i],
                                     &tile_a[read_buffer][row][0],
                                     kHalfBlockK + kSkew);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = warp_group_col * 64 + j * kWmmaN;
      nvcuda::wmma::load_matrix_sync(fragment_b[j],
                                     &tile_b[read_buffer][0][col],
                                     kHalfBlockN + kSkew);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        nvcuda::wmma::mma_sync(accumulators[i][j], fragment_a[i],
                               fragment_b[j], accumulators[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = block_row + warp_group_row * 32 + i * kWmmaM;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = block_col + warp_group_col * 64 + j * kWmmaN;
      nvcuda::wmma::store_matrix_sync(
          &c[static_cast<std::size_t>(row) * n + col], accumulators[i][j], n,
          nvcuda::wmma::mem_row_major);
    }
  }
}

__global__ __launch_bounds__(kHalfThreads) void wmma_small_m_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b,
    float* __restrict__ c, int n, int k) {
  constexpr int BM = 16;
  constexpr int BN = 128;
  __shared__ __align__(16) __half tile_a[2][BM][kHalfBlockK + kSkew];
  __shared__ __align__(16) __half tile_b[2][kHalfBlockK][BN + kSkew];
  const int warp = threadIdx.x / warpSize;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);
  issue_half_tile<BM, BN>(a, b, &tile_a[0][0][0], &tile_b[0][0][0],
                          block_row, block_col, 0, n, k);
  cp_async_commit();
  const int tile_count = k / kHalfBlockK;
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int read_buffer = tile_index & 1;
    const int write_buffer = read_buffer ^ 1;
    if (tile_index + 1 < tile_count) {
      issue_half_tile<BM, BN>(
          a, b, &tile_a[write_buffer][0][0], &tile_b[write_buffer][0][0],
          block_row, block_col, (tile_index + 1) * kHalfBlockK, n, k);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                           __half, nvcuda::wmma::row_major>
        fragment_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                           __half, nvcuda::wmma::row_major>
        fragment_b;
    nvcuda::wmma::load_matrix_sync(fragment_a, &tile_a[read_buffer][0][0],
                                   kHalfBlockK + kSkew);
    nvcuda::wmma::load_matrix_sync(
        fragment_b, &tile_b[read_buffer][0][warp * kWmmaN], BN + kSkew);
    nvcuda::wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    __syncthreads();
  }
  nvcuda::wmma::store_matrix_sync(
      &c[static_cast<std::size_t>(block_row) * n + block_col +
         warp * kWmmaN],
      accumulator, n, nvcuda::wmma::mem_row_major);
}

__global__ void wmma_baseline_kernel(const __half* __restrict__ a,
                                     const __half* __restrict__ b,
                                     float* __restrict__ c, int m, int n,
                                     int k) {
  const int warp_in_block = threadIdx.x / warpSize;
  const int warp_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int tile_columns = n / kWmmaN;
  const int tile_count = (m / kWmmaM) * tile_columns;
  if (warp_index >= tile_count) return;
  const int row = (warp_index / tile_columns) * kWmmaM;
  const int col = (warp_index % tile_columns) * kWmmaN;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      fragment_a;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      fragment_b;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);
  for (int tile_begin = 0; tile_begin < k; tile_begin += kWmmaK) {
    nvcuda::wmma::load_matrix_sync(
        fragment_a, &a[static_cast<std::size_t>(row) * k + tile_begin], k);
    nvcuda::wmma::load_matrix_sync(
        fragment_b, &b[static_cast<std::size_t>(tile_begin) * n + col], n);
    nvcuda::wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
  }
  nvcuda::wmma::store_matrix_sync(&c[static_cast<std::size_t>(row) * n + col],
                                   accumulator, n,
                                   nvcuda::wmma::mem_row_major);
}

__global__ void hgemm_scalar_kernel(const __half* __restrict__ a,
                                    const __half* __restrict__ b,
                                    float* __restrict__ c, int m, int n,
                                    int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner)
    sum = fmaf(__half2float(a[static_cast<std::size_t>(row) * k + inner]),
               __half2float(b[static_cast<std::size_t>(inner) * n + col]),
               sum);
  c[static_cast<std::size_t>(row) * n + col] = sum;
}

}  // namespace

namespace operatorlib {

cudaError_t gemm_f32(const float* a, const float* b, float* c, int m, int n,
                     int k, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
      k <= 0)
    return cudaErrorInvalidValue;
  if (m == 1) {
    launch_sgemm_config<1, 256, 16, 1, 1>(a, b, c, m, n, k, stream);
  } else if (m <= 16) {
    launch_sgemm_config<16, 128, 16, 1, 8>(a, b, c, m, n, k, stream);
  } else {
    launch_sgemm_config<128, 64, 16, 8, 4>(a, b, c, m, n, k, stream);
  }
  return cudaGetLastError();
}

cudaError_t gemm_f16(const __half* a, const __half* b, float* c, int m, int n,
                     int k, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
      k <= 0)
    return cudaErrorInvalidValue;
  if (m >= kHalfBlockM && m % kHalfBlockM == 0 && n % kHalfBlockN == 0 &&
      k % kHalfBlockK == 0) {
    const dim3 grid(n / kHalfBlockN, m / kHalfBlockM);
    wmma_tiled_kernel<<<grid, kHalfThreads, 0, stream>>>(a, b, c, n, k);
  } else if (m < kHalfBlockM && m % 16 == 0 && n % 128 == 0 &&
             k % kHalfBlockK == 0) {
    const dim3 grid(n / 128, m / 16);
    wmma_small_m_kernel<<<grid, kHalfThreads, 0, stream>>>(a, b, c, n, k);
  } else if (m % 16 == 0 && n % 16 == 0 && k % 16 == 0) {
    const int tile_count = (m / 16) * (n / 16);
    const int blocks = (tile_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    wmma_baseline_kernel<<<blocks, kHalfThreads, 0, stream>>>(a, b, c, m, n,
                                                              k);
  } else {
    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    hgemm_scalar_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
  }
  return cudaGetLastError();
}

}  // namespace operatorlib
