// attention_v6.cu - FP16 Q/K/V, WMMA QK and PV, FP32 online-softmax state.

#include "attention_common.h"

#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kRows = 16;
constexpr int kKeys = 16;
constexpr int kThreads = 256;
constexpr int kMaxFragments = 4;

struct HalfDeviceData {
    __half *q = nullptr, *k = nullptr, *v = nullptr;
    float* output = nullptr;

    HalfDeviceData(const AttentionOptions& o, AttentionHostData& host)
    {
        std::vector<__half> hq(o.tensor_elements()), hk(o.tensor_elements()),
            hv(o.tensor_elements());
        for (size_t i = 0; i < o.tensor_elements(); ++i) {
            hq[i] = __float2half_rn(host.q[i]);
            hk[i] = __float2half_rn(host.k[i]);
            hv[i] = __float2half_rn(host.v[i]);
            // CPU reference must see the same quantized inputs as the kernel.
            host.q[i] = __half2float(hq[i]);
            host.k[i] = __half2float(hk[i]);
            host.v[i] = __half2float(hv[i]);
        }
        const size_t half_bytes = o.tensor_elements() * sizeof(__half);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&q), half_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&k), half_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&v), half_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), o.tensor_bytes()));
        CUDA_CHECK(cudaMemcpy(q, hq.data(), half_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k, hk.data(), half_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v, hv.data(), half_bytes, cudaMemcpyHostToDevice));
    }

    ~HalfDeviceData()
    {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(output);
    }
};

__global__ void attention_wmma_kernel(const __half* q, const __half* k,
                                      const __half* v, float* output,
                                      int heads, int s, int d, bool causal)
{
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
        shared_q[index] = global_row < s
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
            shared_k[index] = global_key < s
                ? k[head_base + static_cast<long long>(global_key) * d + x]
                : __float2half(0.0f);
            shared_v[index] = global_key < s
                ? v[head_base + static_cast<long long>(global_key) * d + x]
                : __float2half(0.0f);
        }
        __syncthreads();

        if (warp == 0) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
            for (int x = 0; x < d; x += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::col_major> k_fragment;
                wmma::load_matrix_sync(q_fragment, shared_q + x, d);
                // K[row,key-dim] is the column-major storage of logical K^T.
                wmma::load_matrix_sync(k_fragment, shared_k + x, d);
                wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
            }
            wmma::store_matrix_sync(scores, accumulator, kKeys,
                                    wmma::mem_row_major);
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
            const float new_maximum = fmaxf(row_maximum[row], tile_maximum);
            const float alpha = expf(row_maximum[row] - new_maximum);
            float tile_sum = 0.0f;
            for (int col = 0; col < kKeys; ++col) {
                const float probability =
                    expf(scores[row * kKeys + col] - new_maximum);
                shared_p[row * kKeys + col] = __float2half_rn(probability);
                tile_sum += probability;
            }
            row_sum[row] = row_sum[row] * alpha + tile_sum;
            row_maximum[row] = new_maximum;
            row_alpha[row] = alpha;
        }
        __syncthreads();

        for (int index = tid; index < kRows * d; index += kThreads) {
            const int row = index / d;
            output_state[index] *= row_alpha[row];
            pv_tile[index] = 0.0f;
        }
        __syncthreads();

        const int output_tiles = d / 16;
        if (warp < output_tiles) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                           wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                           wmma::row_major> v_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::load_matrix_sync(p_fragment, shared_p, kKeys);
            wmma::load_matrix_sync(v_fragment, shared_v + warp * 16, d);
            wmma::fill_fragment(accumulator, 0.0f);
            wmma::mma_sync(accumulator, p_fragment, v_fragment, accumulator);
            wmma::store_matrix_sync(pv_tile + warp * 16, accumulator, d,
                                    wmma::mem_row_major);
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
        if (global_row < s)
            output[head_base + static_cast<long long>(global_row) * d + x] =
                output_state[index] / row_sum[row];
    }
}

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void attention_half_fallback(const __half* q, const __half* k,
                                        const __half* v, float* output,
                                        int heads, int s, int d, bool causal)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int bh = blockIdx.x;
    const int row = blockIdx.y * 8 + warp;
    if (bh >= heads || row >= s) return;
    const long long base = static_cast<long long>(bh) * s * d;
    float q_values[kMaxFragments] = {}, out_values[kMaxFragments] = {};
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
        const int x = lane + f * 32;
        if (x < d)
            q_values[f] = __half2float(q[base + static_cast<long long>(row) * d + x]);
    }
    float maximum = -INFINITY, sum = 0.0f;
    const int visible = causal ? row + 1 : s;
    const float scale = rsqrtf(static_cast<float>(d));
    for (int key = 0; key < visible; ++key) {
        float dot = 0.0f;
#pragma unroll
        for (int f = 0; f < kMaxFragments; ++f) {
            const int x = lane + f * 32;
            if (x < d)
                dot = fmaf(q_values[f], __half2float(
                    k[base + static_cast<long long>(key) * d + x]), dot);
        }
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float alpha = 1.0f, beta = 0.0f;
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
            if (x < d)
                out_values[f] = out_values[f] * alpha + beta * __half2float(
                    v[base + static_cast<long long>(key) * d + x]);
        }
    }
    float inverse = lane == 0 ? 1.0f / sum : 0.0f;
    inverse = __shfl_sync(0xffffffffu, inverse, 0);
#pragma unroll
    for (int f = 0; f < kMaxFragments; ++f) {
        const int x = lane + f * 32;
        if (x < d)
            output[base + static_cast<long long>(row) * d + x] =
                out_values[f] * inverse;
    }
}

size_t wmma_shared_bytes(int d)
{
    const size_t half_elements = kRows * d + 2 * kKeys * d + kRows * kKeys;
    const size_t float_elements = kRows * kKeys + 2 * kRows * d + 3 * kRows;
    return half_elements * sizeof(__half) + float_elements * sizeof(float);
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    AttentionHostData host(options);
    HalfDeviceData device(options, host);
    const bool aligned = (options.d == 64 || options.d == 128) &&
                         options.s % 16 == 0;
    const size_t shared_bytes = aligned ? wmma_shared_bytes(options.d) : 0;
    if (aligned && shared_bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(attention_wmma_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)));
    auto launch = [&]() {
        if (aligned) {
            dim3 grid(options.heads(), (options.s + kRows - 1) / kRows);
            attention_wmma_kernel<<<grid, kThreads, shared_bytes>>>(
                device.q, device.k, device.v, device.output, options.heads(),
                options.s, options.d, options.causal);
        } else {
            dim3 grid(options.heads(), (options.s + 7) / 8);
            attention_half_fallback<<<grid, 256>>>(
                device.q, device.k, device.v, device.output, options.heads(),
                options.s, options.d, options.causal);
        }
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check_output(
        options, host, device.output, 2.0e-2, 2.0e-2);
    const double ms = attention_median_ms(launch, options.warmup, options.repeat);
    attention_report("attention_v6",
                     aligned ? "fp16_wmma_online" : "fp16_scalar_fallback",
                     options, check, ms, 0);
    std::printf("input=fp16 output=fp32 state=fp32 tile_rows=16 tile_keys=16 "
                "shared_bytes=%zu\n", shared_bytes);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
