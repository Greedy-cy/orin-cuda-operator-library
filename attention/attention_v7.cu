// attention_v7.cu - padded WMMA shared layouts and parallel partial QK.

#include "attention_fp16_common.h"

#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kRows = 16;
constexpr int kKeys = 16;
constexpr int kThreads = 256;
constexpr int kHalfPad = 8;
constexpr int kProbabilityLd = 24;
constexpr int kScoreLd = 20;

template <bool ParallelQk>
__global__ void attention_wmma_padded_kernel(
    const __half* q, const __half* k, const __half* v, float* output,
    int heads, int s, int d, bool causal)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int bh = blockIdx.x;
    const int row_begin = blockIdx.y * kRows;
    const int half_ld = d + kHalfPad;
    const int output_ld = d + 4;
    const int d_tiles = d / 16;
    const long long head_base = static_cast<long long>(bh) * s * d;

    extern __shared__ __align__(16) unsigned char storage[];
    __half* shared_q = reinterpret_cast<__half*>(storage);
    __half* shared_k = shared_q + kRows * half_ld;
    __half* shared_v = shared_k + kKeys * half_ld;
    __half* shared_p = shared_v + kKeys * half_ld;
    float* partial_scores = reinterpret_cast<float*>(
        shared_p + kRows * kProbabilityLd);
    constexpr int direct_score_tiles = 1;
    const int score_tiles = ParallelQk ? d_tiles : direct_score_tiles;
    float* output_state = partial_scores + score_tiles * kRows * kScoreLd;
    float* pv_tile = output_state + kRows * output_ld;
    float* row_maximum = pv_tile + kRows * output_ld;
    float* row_sum = row_maximum + kRows;
    float* row_alpha = row_sum + kRows;

    for (int index = tid; index < kRows * d; index += kThreads) {
        const int row = index / d;
        const int x = index - row * d;
        const int global_row = row_begin + row;
        shared_q[row * half_ld + x] = global_row < s
            ? q[head_base + static_cast<long long>(global_row) * d + x]
            : __float2half(0.0f);
        output_state[row * output_ld + x] = 0.0f;
    }
    if (tid < kRows) {
        row_maximum[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    const int needed_keys = causal ? min(s, row_begin + kRows) : s;
    const int tile_count = (needed_keys + kKeys - 1) / kKeys;
    const float scale = rsqrtf(static_cast<float>(d));
    for (int tile = 0; tile < tile_count; ++tile) {
        const int key_begin = tile * kKeys;
        for (int index = tid; index < kKeys * d; index += kThreads) {
            const int key = index / d;
            const int x = index - key * d;
            const int global_key = key_begin + key;
            shared_k[key * half_ld + x] = global_key < s
                ? k[head_base + static_cast<long long>(global_key) * d + x]
                : __float2half(0.0f);
            shared_v[key * half_ld + x] = global_key < s
                ? v[head_base + static_cast<long long>(global_key) * d + x]
                : __float2half(0.0f);
        }
        __syncthreads();

        if constexpr (ParallelQk) {
            if (warp < d_tiles) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::col_major> k_fragment;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
                wmma::load_matrix_sync(q_fragment, shared_q + warp * 16, half_ld);
                wmma::load_matrix_sync(k_fragment, shared_k + warp * 16, half_ld);
                wmma::fill_fragment(accumulator, 0.0f);
                wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
                wmma::store_matrix_sync(
                    partial_scores + warp * kRows * kScoreLd, accumulator,
                    kScoreLd, wmma::mem_row_major);
            }
            __syncthreads();
            if (tid < kRows * kKeys) {
                const int row = tid / kKeys;
                const int col = tid - row * kKeys;
                float score = 0.0f;
                for (int dt = 0; dt < d_tiles; ++dt)
                    score += partial_scores[dt * kRows * kScoreLd +
                                            row * kScoreLd + col];
                partial_scores[row * kScoreLd + col] = score * scale;
            }
        } else {
            if (warp == 0) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
                wmma::fill_fragment(accumulator, 0.0f);
                for (int dt = 0; dt < d_tiles; ++dt) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                   wmma::row_major> q_fragment;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                   wmma::col_major> k_fragment;
                    wmma::load_matrix_sync(q_fragment, shared_q + dt * 16,
                                           half_ld);
                    wmma::load_matrix_sync(k_fragment, shared_k + dt * 16,
                                           half_ld);
                    wmma::mma_sync(accumulator, q_fragment, k_fragment,
                                   accumulator);
                }
                wmma::store_matrix_sync(partial_scores, accumulator,
                                        kScoreLd, wmma::mem_row_major);
            }
            __syncthreads();
            if (tid < kRows * kKeys)
                partial_scores[(tid / kKeys) * kScoreLd + tid % kKeys] *= scale;
        }
        __syncthreads();

        if (tid < kRows) {
            const int row = tid;
            const int global_row = row_begin + row;
            float tile_maximum = -INFINITY;
            for (int col = 0; col < kKeys; ++col) {
                const int global_key = key_begin + col;
                float& score = partial_scores[row * kScoreLd + col];
                if (global_row < s && global_key < s &&
                    (!causal || global_key <= global_row))
                    tile_maximum = fmaxf(tile_maximum, score);
                else
                    score = -INFINITY;
            }
            const float new_maximum = fmaxf(row_maximum[row], tile_maximum);
            const float alpha = expf(row_maximum[row] - new_maximum);
            float tile_sum = 0.0f;
            for (int col = 0; col < kKeys; ++col) {
                const float probability = expf(
                    partial_scores[row * kScoreLd + col] - new_maximum);
                shared_p[row * kProbabilityLd + col] =
                    __float2half_rn(probability);
                tile_sum += probability;
            }
            row_sum[row] = row_sum[row] * alpha + tile_sum;
            row_maximum[row] = new_maximum;
            row_alpha[row] = alpha;
        }
        __syncthreads();

        for (int index = tid; index < kRows * d; index += kThreads) {
            const int row = index / d;
            const int x = index - row * d;
            output_state[row * output_ld + x] *= row_alpha[row];
            pv_tile[row * output_ld + x] = 0.0f;
        }
        __syncthreads();

        if (warp < d_tiles) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::row_major> v_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::load_matrix_sync(p_fragment, shared_p, kProbabilityLd);
            wmma::load_matrix_sync(v_fragment, shared_v + warp * 16, half_ld);
            wmma::fill_fragment(accumulator, 0.0f);
            wmma::mma_sync(accumulator, p_fragment, v_fragment, accumulator);
            wmma::store_matrix_sync(pv_tile + warp * 16, accumulator,
                                    output_ld, wmma::mem_row_major);
        }
        __syncthreads();
        for (int index = tid; index < kRows * d; index += kThreads) {
            const int row = index / d;
            const int x = index - row * d;
            output_state[row * output_ld + x] +=
                pv_tile[row * output_ld + x];
        }
        __syncthreads();
    }

    for (int index = tid; index < kRows * d; index += kThreads) {
        const int row = index / d;
        const int x = index - row * d;
        const int global_row = row_begin + row;
        if (global_row < s)
            output[head_base + static_cast<long long>(global_row) * d + x] =
                output_state[row * output_ld + x] / row_sum[row];
    }
}

size_t padded_shared_bytes(int d, bool parallel_qk)
{
    const int d_tiles = d / 16;
    const int half_ld = d + kHalfPad;
    const int output_ld = d + 4;
    const size_t half_elements = kRows * half_ld + 2 * kKeys * half_ld +
                                 kRows * kProbabilityLd;
    const int score_tiles = parallel_qk ? d_tiles : 1;
    const size_t float_elements = score_tiles * kRows * kScoreLd +
                                  2 * kRows * output_ld + 3 * kRows;
    return half_elements * sizeof(__half) + float_elements * sizeof(float);
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    AttentionHostData host(options);
    HalfAttentionDeviceData device(options, host);
    const bool aligned = (options.d == 64 || options.d == 128) &&
                         options.s % 16 == 0;
    const bool parallel_qk = options.config == 1;
    const size_t shared_bytes = aligned
        ? padded_shared_bytes(options.d, parallel_qk) : 0;
    if (aligned && shared_bytes > 48 * 1024) {
        if (parallel_qk)
            CUDA_CHECK(cudaFuncSetAttribute(attention_wmma_padded_kernel<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes)));
        else
            CUDA_CHECK(cudaFuncSetAttribute(attention_wmma_padded_kernel<false>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes)));
    }
    auto launch = [&]() {
        if (aligned) {
            dim3 grid(options.heads(), (options.s + kRows - 1) / kRows);
            if (parallel_qk)
                attention_wmma_padded_kernel<true><<<grid, kThreads, shared_bytes>>>(
                    device.q, device.k, device.v, device.output, options.heads(),
                    options.s, options.d, options.causal);
            else
                attention_wmma_padded_kernel<false><<<grid, kThreads, shared_bytes>>>(
                    device.q, device.k, device.v, device.output, options.heads(),
                    options.s, options.d, options.causal);
        } else {
            launch_half_attention_fallback(options, device);
        }
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check_output(
        options, host, device.output, 2.0e-2, 2.0e-2);
    const double ms = attention_median_ms(launch, options.warmup, options.repeat);
    const char* path = !aligned ? "fp16_scalar_fallback"
        : (parallel_qk ? "padded_parallel_qk_wmma"
                       : "padded_direct_qk_wmma");
    attention_report("attention_v7", path,
                     options, check, ms, 0);
    std::printf("input=fp16 output=fp32 state=fp32 d_tiles=%d "
                "parallel_qk=%d shared_bytes=%zu\n",
                aligned ? options.d / 16 : 0, parallel_qk ? 1 : 0,
                shared_bytes);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
