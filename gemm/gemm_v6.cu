#include "gemm_common.h"

#include <cstdio>

namespace {

constexpr int kThreads = 256;
int g_forced_config = -1;
char g_configuration[160] = "dispatcher=pending";

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

template <int BM, int BN, int BK, int Threads>
__device__ __forceinline__ void issue_tile(
    const float* __restrict__ a, const float* __restrict__ b,
    float* shared_a, float* shared_b, int block_row, int block_col,
    int tile_begin, int n, int k) {
  constexpr int a_vectors = BM * BK / 4;
  constexpr int b_vectors = BK * BN / 4;
  const int tid = threadIdx.x;
#pragma unroll
  for (int vector_index = tid; vector_index < a_vectors;
       vector_index += Threads) {
    const int element = vector_index * 4;
    const int row = element / BK;
    const int col = element % BK;
    cp_async_16(&shared_a[element],
                &a[static_cast<size_t>(block_row + row) * k + tile_begin + col]);
  }
#pragma unroll
  for (int vector_index = tid; vector_index < b_vectors;
       vector_index += Threads) {
    const int element = vector_index * 4;
    const int row = element / BN;
    const int col = element % BN;
    cp_async_16(&shared_b[element],
                &b[static_cast<size_t>(tile_begin + row) * n + block_col + col]);
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(kThreads) void sgemm_configured_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k) {
  static_assert((BM / TM) * (BN / TN) == kThreads);
  static_assert(BK % 4 == 0 && BN % 4 == 0);
  __shared__ __align__(16) float tile_a[2][BM][BK];
  __shared__ __align__(16) float tile_b[2][BK][BN];

  const int tid = threadIdx.x;
  const int thread_tile_row = tid / (BN / TN);
  const int thread_tile_col = tid % (BN / TN);
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  float accumulators[TM][TN] = {};

  issue_tile<BM, BN, BK, kThreads>(
      a, b, &tile_a[0][0][0], &tile_b[0][0][0], block_row, block_col, 0, n, k);
  cp_async_commit();

  const int tile_count = k / BK;
  for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
    const int read_buffer = tile_index & 1;
    const int write_buffer = read_buffer ^ 1;
    const bool has_next = tile_index + 1 < tile_count;
    if (has_next) {
      issue_tile<BM, BN, BK, kThreads>(
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
            &c[static_cast<size_t>(output_row) * n + output_col + j]) = result;
      }
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j)
        c[static_cast<size_t>(output_row) * n + output_col + j] =
            accumulators[i][j];
    }
  }
}

__global__ __launch_bounds__(kThreads) void sgemm_scalar_fallback_kernel(
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
#pragma unroll
    for (int load = tid; load < BM * BK; load += kThreads) {
      const int row = load / BK;
      const int col = load % BK;
      const int global_row = block_row + row;
      const int global_col = tile_begin + col;
      tile_a[row][col] =
          (global_row < m && global_col < k)
              ? a[static_cast<size_t>(global_row) * k + global_col]
              : 0.0f;
    }
#pragma unroll
    for (int load = tid; load < BK * BN; load += kThreads) {
      const int row = load / BN;
      const int col = load % BN;
      const int global_row = tile_begin + row;
      const int global_col = block_col + col;
      tile_b[row][col] =
          (global_row < k && global_col < n)
              ? b[static_cast<size_t>(global_row) * n + global_col]
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
        c[static_cast<size_t>(output_row) * n + col] = accumulators[i][j];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_configuration(const float* a, const float* b, float* c, int m,
                          int n, int k, int config, const char* name) {
  if (m % BM != 0 || n % BN != 0 || k % BK != 0) {
    const dim3 fallback_grid((n + 63) / 64, (m + 63) / 64);
    sgemm_scalar_fallback_kernel<<<fallback_grid, kThreads>>>(a, b, c, m, n, k);
    std::snprintf(g_configuration, sizeof(g_configuration),
                  "config=%d(%s) path=scalar_fallback", config, name);
    return;
  }
  const dim3 grid(n / BN, m / BM);
  sgemm_configured_kernel<BM, BN, BK, TM, TN><<<grid, kThreads>>>(a, b, c, m, n,
                                                                 k);
  std::snprintf(g_configuration, sizeof(g_configuration),
                "config=%d(%s) tile=%dx%dx%d thread_tile=%dx%d", config, name,
                BM, BN, BK, TM, TN);
}

int choose_auto_config(int m) {
  if (m == 1) return 7;
  if (m <= 16) return 6;
  return 2;
}

void launch_custom(const float* a, const float* b, float* c, int m, int n,
                   int k) {
  const int config = g_forced_config >= 0 ? g_forced_config
                                           : choose_auto_config(m);
  switch (config) {
    case 0:
      launch_configuration<64, 64, 16, 4, 4>(a, b, c, m, n, k, 0, "R64x64");
      break;
    case 1:
      launch_configuration<64, 128, 16, 4, 8>(a, b, c, m, n, k, 1,
                                               "R64x128");
      break;
    case 2:
      launch_configuration<128, 64, 16, 8, 4>(a, b, c, m, n, k, 2,
                                               "R128x64");
      break;
    case 3:
      launch_configuration<128, 128, 16, 8, 8>(a, b, c, m, n, k, 3,
                                                "R128x128");
      break;
    case 4:
      launch_configuration<64, 128, 32, 4, 8>(a, b, c, m, n, k, 4,
                                               "K32_64x128");
      break;
    case 5:
      launch_configuration<32, 128, 16, 2, 8>(a, b, c, m, n, k, 5,
                                               "R32x128");
      break;
    case 6:
      launch_configuration<16, 128, 16, 1, 8>(a, b, c, m, n, k, 6,
                                               "smallM16");
      break;
    case 7:
      launch_configuration<1, 256, 16, 1, 1>(a, b, c, m, n, k, 7,
                                              "gemvM1");
      break;
    default:
      std::cerr << "config must be in [0,7]\n";
      std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cudaGetLastError());
}

int extract_config_argument(int argc, char** argv, std::vector<char*>* filtered) {
  filtered->reserve(argc);
  filtered->push_back(argv[0]);
  int forced = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument.rfind("--config=", 0) == 0) {
      forced = std::stoi(argument.substr(9));
    } else {
      filtered->push_back(argv[i]);
    }
  }
  return forced;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<char*> filtered_arguments;
  g_forced_config =
      extract_config_argument(argc, argv, &filtered_arguments);
  return gemm_common::run_benchmark(
      static_cast<int>(filtered_arguments.size()), filtered_arguments.data(),
      "gemm_v6", "v6_config_scan", g_configuration, launch_custom);
}
