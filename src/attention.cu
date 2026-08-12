#include "operatorlib/operators.h"

#include <mma.h>

namespace {

constexpr int kMaxFragments = 4;

__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                            const void* global_source) {
  const unsigned address =
      static_cast<unsigned>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(address), "l"(global_source));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// ------------------------------ FP32 v5 ------------------------------

template <int Threads, int KeyTile>
__device__ __forceinline__ void issue_kv_tile(
    float* shared_k, float* shared_v, int buffer, const float* k,
    const float* v, long long head_base, int key_begin, int d) {
  const int chunks = KeyTile * d / 4;
  for (int chunk = threadIdx.x; chunk < chunks; chunk += Threads) {
    const int element = chunk * 4;
    cp_async_16(shared_k + buffer * KeyTile * d + element,
                k + head_base + static_cast<long long>(key_begin) * d +
                    element);
    cp_async_16(shared_v + buffer * KeyTile * d + element,
                v + head_base + static_cast<long long>(key_begin) * d +
                    element);
  }
}

template <int BlockRows, int KeyTile>
__global__ void attention_cp_async_kernel(const float* q, const float* k,
                                          const float* v, float* output,
                                          int heads, int s, int d,
                                          bool causal) {
  constexpr int Threads = BlockRows * 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bh = blockIdx.x;
  const int row = blockIdx.y * BlockRows + warp;
  const bool valid_row = bh < heads && row < s;
  const long long head_base = static_cast<long long>(bh) * s * d;
  extern __shared__ __align__(16) float shared[];
  float* shared_k = shared;
  float* shared_v = shared + 2 * KeyTile * d;
  float q_values[kMaxFragments];
  float out_values[kMaxFragments];
#pragma unroll
  for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
    const int x = lane + fragment * 32;
    q_values[fragment] =
        valid_row && x < d
            ? q[head_base + static_cast<long long>(row) * d + x]
            : 0.0f;
    out_values[fragment] = 0.0f;
  }
  const int total_tiles = s / KeyTile;
  const int needed_keys = min(s, (blockIdx.y + 1) * BlockRows);
  const int tile_count =
      causal ? (needed_keys + KeyTile - 1) / KeyTile : total_tiles;
  issue_kv_tile<Threads, KeyTile>(shared_k, shared_v, 0, k, v, head_base, 0,
                                  d);
  cp_async_commit();
  float running_maximum = -INFINITY;
  float running_sum = 0.0f;
  const float scale = rsqrtf(static_cast<float>(d));
  for (int tile = 0; tile < tile_count; ++tile) {
    const int read_buffer = tile & 1;
    const int write_buffer = read_buffer ^ 1;
    if (tile + 1 < tile_count) {
      issue_kv_tile<Threads, KeyTile>(
          shared_k, shared_v, write_buffer, k, v, head_base,
          (tile + 1) * KeyTile, d);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const int key_begin = tile * KeyTile;
    const float* tile_k = shared_k + read_buffer * KeyTile * d;
    const float* tile_v = shared_v + read_buffer * KeyTile * d;
    for (int key_in_tile = 0; key_in_tile < KeyTile; ++key_in_tile) {
      const int key = key_begin + key_in_tile;
      if (!valid_row || (causal && key > row)) continue;
      float dot = 0.0f;
#pragma unroll
      for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
        const int x = lane + fragment * 32;
        if (x < d)
          dot = fmaf(q_values[fragment], tile_k[key_in_tile * d + x], dot);
      }
      dot = warp_sum(dot);
      const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        const float next = fmaxf(running_maximum, score);
        alpha = expf(running_maximum - next);
        beta = expf(score - next);
        running_sum = running_sum * alpha + beta;
        running_maximum = next;
      }
      alpha = __shfl_sync(0xffffffffu, alpha, 0);
      beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
      for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
        const int x = lane + fragment * 32;
        if (x < d) {
          out_values[fragment] = out_values[fragment] * alpha +
                                 beta * tile_v[key_in_tile * d + x];
        }
      }
    }
    __syncthreads();
  }
  float inverse = lane == 0 ? 1.0f / running_sum : 0.0f;
  inverse = __shfl_sync(0xffffffffu, inverse, 0);
  if (valid_row) {
#pragma unroll
    for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
      const int x = lane + fragment * 32;
      if (x < d) {
        output[head_base + static_cast<long long>(row) * d + x] =
            out_values[fragment] * inverse;
      }
    }
  }
}

__global__ void attention_f32_scalar_kernel(const float* q, const float* k,
                                            const float* v, float* output,
                                            int heads, int s, int d,
                                            bool causal) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int bh = blockIdx.x;
  const int row = blockIdx.y * 8 + warp;
  if (bh >= heads || row >= s) return;
  const long long base = static_cast<long long>(bh) * s * d;
  float q_values[kMaxFragments] = {};
  float out_values[kMaxFragments] = {};
#pragma unroll
  for (int f = 0; f < kMaxFragments; ++f) {
    const int x = lane + f * 32;
    if (x < d) q_values[f] = q[base + static_cast<long long>(row) * d + x];
  }
  float maximum = -INFINITY;
  float sum = 0.0f;
  const int visible = causal ? row + 1 : s;
  const float scale = rsqrtf(static_cast<float>(d));
  for (int key = 0; key < visible; ++key) {
    float dot = 0.0f;
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
      const int x = lane + f * 32;
      if (x < d)
        dot = fmaf(q_values[f],
                   k[base + static_cast<long long>(key) * d + x], dot);
    }
    dot = warp_sum(dot);
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
    float alpha = 1.0f;
    float beta = 0.0f;
    if (lane == 0) {
      const float next = fmaxf(maximum, score);
      alpha = expf(maximum - next);
      beta = expf(score - next);
      sum = sum * alpha + beta;
      maximum = next;
    }
    alpha = __shfl_sync(0xffffffffu, alpha, 0);
    beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
      const int x = lane + f * 32;
      if (x < d) {
        out_values[f] = out_values[f] * alpha +
                        beta * v[base + static_cast<long long>(key) * d + x];
      }
    }
  }
  float inverse = lane == 0 ? 1.0f / sum : 0.0f;
  inverse = __shfl_sync(0xffffffffu, inverse, 0);
#pragma unroll
  for (int f = 0; f < kMaxFragments; ++f) {
    const int x = lane + f * 32;
    if (x < d) {
      output[base + static_cast<long long>(row) * d + x] =
          out_values[f] * inverse;
    }
  }
}

// ------------------------------ FP16 v6 ------------------------------

constexpr int kRows = 16;
constexpr int kKeys = 16;
constexpr int kThreads = 256;

__global__ void attention_wmma_kernel(const __half* q, const __half* k,
                                      const __half* v, float* output,
                                      int heads, int s, int d, bool causal) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int bh = blockIdx.x;
  const int row_begin = blockIdx.y * kRows;
  const long long head_base = static_cast<long long>(bh) * s * d;
  extern __shared__ __align__(16) unsigned char storage[];
  __half* shared_q = reinterpret_cast<__half*>(storage);
  __half* shared_k = shared_q + kRows * d;
  __half* shared_v = shared_k + kKeys * d;
  __half* shared_p = shared_v + kKeys * d;
  float* scores = reinterpret_cast<float*>(shared_p + kRows * kKeys);
  float* output_state = scores + kRows * kKeys;
  float* pv_tile = output_state + kRows * d;
  float* row_maximum = pv_tile + kRows * d;
  float* row_sum = row_maximum + kRows;
  float* row_alpha = row_sum + kRows;

  for (int index = tid; index < kRows * d; index += kThreads) {
    const int row = index / d;
    const int x = index - row * d;
    const int global_row = row_begin + row;
    shared_q[index] =
        global_row < s
            ? q[head_base + static_cast<long long>(global_row) * d + x]
            : __float2half(0.0f);
    output_state[index] = 0.0f;
  }
  if (tid < kRows) {
    row_maximum[tid] = -INFINITY;
    row_sum[tid] = 0.0f;
  }
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(d));
  const int needed_keys = causal ? min(s, row_begin + kRows) : s;
  const int tile_count = (needed_keys + kKeys - 1) / kKeys;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int key_begin = tile * kKeys;
    for (int index = tid; index < kKeys * d; index += kThreads) {
      const int key = index / d;
      const int x = index - key * d;
      const int global_key = key_begin + key;
      shared_k[index] =
          global_key < s
              ? k[head_base + static_cast<long long>(global_key) * d + x]
              : __float2half(0.0f);
      shared_v[index] =
          global_key < s
              ? v[head_base + static_cast<long long>(global_key) * d + x]
              : __float2half(0.0f);
    }
    __syncthreads();
    if (warp == 0) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          accumulator;
      nvcuda::wmma::fill_fragment(accumulator, 0.0f);
      for (int x = 0; x < d; x += 16) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                               nvcuda::wmma::row_major>
            q_fragment;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                               nvcuda::wmma::col_major>
            k_fragment;
        nvcuda::wmma::load_matrix_sync(q_fragment, shared_q + x, d);
        nvcuda::wmma::load_matrix_sync(k_fragment, shared_k + x, d);
        nvcuda::wmma::mma_sync(accumulator, q_fragment, k_fragment,
                               accumulator);
      }
      nvcuda::wmma::store_matrix_sync(scores, accumulator, kKeys,
                                       nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < kRows) {
      const int row = tid;
      const int global_row = row_begin + row;
      float tile_maximum = -INFINITY;
      for (int col = 0; col < kKeys; ++col) {
        const int global_key = key_begin + col;
        if (global_row < s && global_key < s &&
            (!causal || global_key <= global_row)) {
          const float score = scores[row * kKeys + col] * scale;
          scores[row * kKeys + col] = score;
          tile_maximum = fmaxf(tile_maximum, score);
        } else {
          scores[row * kKeys + col] = -INFINITY;
        }
      }
      const float next = fmaxf(row_maximum[row], tile_maximum);
      const float alpha = expf(row_maximum[row] - next);
      float tile_sum = 0.0f;
      for (int col = 0; col < kKeys; ++col) {
        const float probability = expf(scores[row * kKeys + col] - next);
        shared_p[row * kKeys + col] = __float2half_rn(probability);
        tile_sum += probability;
      }
      row_sum[row] = row_sum[row] * alpha + tile_sum;
      row_maximum[row] = next;
      row_alpha[row] = alpha;
    }
    __syncthreads();
    for (int index = tid; index < kRows * d; index += kThreads) {
      const int row = index / d;
      output_state[index] *= row_alpha[row];
      pv_tile[index] = 0.0f;
    }
    __syncthreads();
    if (warp < d / 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                             nvcuda::wmma::row_major>
          p_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                             nvcuda::wmma::row_major>
          v_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          accumulator;
      nvcuda::wmma::load_matrix_sync(p_fragment, shared_p, kKeys);
      nvcuda::wmma::load_matrix_sync(v_fragment, shared_v + warp * 16, d);
      nvcuda::wmma::fill_fragment(accumulator, 0.0f);
      nvcuda::wmma::mma_sync(accumulator, p_fragment, v_fragment, accumulator);
      nvcuda::wmma::store_matrix_sync(pv_tile + warp * 16, accumulator, d,
                                       nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = tid; index < kRows * d; index += kThreads)
      output_state[index] += pv_tile[index];
    __syncthreads();
  }
  for (int index = tid; index < kRows * d; index += kThreads) {
    const int row = index / d;
    const int x = index - row * d;
    const int global_row = row_begin + row;
    if (global_row < s) {
      output[head_base + static_cast<long long>(global_row) * d + x] =
          output_state[index] / row_sum[row];
    }
  }
}

__global__ void attention_f16_scalar_kernel(
    const __half* q, const __half* k, const __half* v, float* output,
    int heads, int s, int d, bool causal) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int bh = blockIdx.x;
  const int row = blockIdx.y * 8 + warp;
  if (bh >= heads || row >= s) return;
  const long long base = static_cast<long long>(bh) * s * d;
  float q_values[kMaxFragments] = {};
  float out_values[kMaxFragments] = {};
#pragma unroll
  for (int f = 0; f < kMaxFragments; ++f) {
    const int x = lane + f * 32;
    if (x < d) {
      q_values[f] =
          __half2float(q[base + static_cast<long long>(row) * d + x]);
    }
  }
  float maximum = -INFINITY;
  float sum = 0.0f;
  const int visible = causal ? row + 1 : s;
  const float scale = rsqrtf(static_cast<float>(d));
  for (int key = 0; key < visible; ++key) {
    float dot = 0.0f;
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
      const int x = lane + f * 32;
      if (x < d) {
        dot = fmaf(q_values[f],
                   __half2float(
                       k[base + static_cast<long long>(key) * d + x]),
                   dot);
      }
    }
    dot = warp_sum(dot);
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
    float alpha = 1.0f;
    float beta = 0.0f;
    if (lane == 0) {
      const float next = fmaxf(maximum, score);
      alpha = expf(maximum - next);
      beta = expf(score - next);
      sum = sum * alpha + beta;
      maximum = next;
    }
    alpha = __shfl_sync(0xffffffffu, alpha, 0);
    beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
      const int x = lane + f * 32;
      if (x < d) {
        out_values[f] =
            out_values[f] * alpha +
            beta * __half2float(
                       v[base + static_cast<long long>(key) * d + x]);
      }
    }
  }
  float inverse = lane == 0 ? 1.0f / sum : 0.0f;
  inverse = __shfl_sync(0xffffffffu, inverse, 0);
#pragma unroll
  for (int f = 0; f < kMaxFragments; ++f) {
    const int x = lane + f * 32;
    if (x < d) {
      output[base + static_cast<long long>(row) * d + x] =
          out_values[f] * inverse;
    }
  }
}

std::size_t wmma_shared_bytes(int d) {
  const std::size_t half_elements =
      kRows * d + 2 * kKeys * d + kRows * kKeys;
  const std::size_t float_elements =
      kRows * kKeys + 2 * kRows * d + 3 * kRows;
  return half_elements * sizeof(__half) + float_elements * sizeof(float);
}

}  // namespace

namespace operatorlib {

cudaError_t attention_f32(const float* q, const float* k, const float* v,
                          float* output, int batch, int heads, int sequence,
                          int head_dim, bool causal, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch <= 0 || heads <= 0 || sequence <= 0 || head_dim <= 0 ||
      head_dim > 128)
    return cudaErrorInvalidValue;
  const int total_heads = batch * heads;
  const bool aligned = (head_dim == 64 || head_dim == 128) &&
                       sequence % 32 == 0;
  if (!aligned) {
    const dim3 grid(total_heads, (sequence + 7) / 8);
    attention_f32_scalar_kernel<<<grid, 256, 0, stream>>>(
        q, k, v, output, total_heads, sequence, head_dim, causal);
  } else if (head_dim == 64) {
    constexpr int rows = 16;
    constexpr int keys = 32;
    const std::size_t shared_bytes =
        4ULL * keys * head_dim * sizeof(float);
    const dim3 grid(total_heads, (sequence + rows - 1) / rows);
    attention_cp_async_kernel<rows, keys>
        <<<grid, rows * 32, shared_bytes, stream>>>(
            q, k, v, output, total_heads, sequence, head_dim, causal);
  } else {
    constexpr int rows = 32;
    constexpr int keys = 32;
    const std::size_t shared_bytes =
        4ULL * keys * head_dim * sizeof(float);
    cudaError_t error = cudaFuncSetAttribute(
        attention_cp_async_kernel<rows, keys>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (error != cudaSuccess) return error;
    const dim3 grid(total_heads, (sequence + rows - 1) / rows);
    attention_cp_async_kernel<rows, keys>
        <<<grid, rows * 32, shared_bytes, stream>>>(
            q, k, v, output, total_heads, sequence, head_dim, causal);
  }
  return cudaGetLastError();
}

cudaError_t attention_f16(const __half* q, const __half* k, const __half* v,
                          float* output, int batch, int heads, int sequence,
                          int head_dim, bool causal, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch <= 0 || heads <= 0 || sequence <= 0 || head_dim <= 0 ||
      head_dim > 128)
    return cudaErrorInvalidValue;
  const int total_heads = batch * heads;
  const bool aligned = (head_dim == 64 || head_dim == 128) &&
                       sequence % 16 == 0;
  if (aligned) {
    const std::size_t shared_bytes = wmma_shared_bytes(head_dim);
    const dim3 grid(total_heads, (sequence + kRows - 1) / kRows);
    attention_wmma_kernel<<<grid, kThreads, shared_bytes, stream>>>(
        q, k, v, output, total_heads, sequence, head_dim, causal);
  } else {
    const dim3 grid(total_heads, (sequence + 7) / 8);
    attention_f16_scalar_kernel<<<grid, 256, 0, stream>>>(
        q, k, v, output, total_heads, sequence, head_dim, causal);
  }
  return cudaGetLastError();
}

}  // namespace operatorlib
