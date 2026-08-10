// attention_v2.cu - FP32 tiled online-softmax attention.
// K/V stream through shared memory; no [B,H,S,S] allocation or writeback.

#include "attention_common.h"

constexpr int kBlockRows = 8;
constexpr int kKeyTile = 32;
constexpr int kThreads = 256;
constexpr int kMaxFragments = 4;  // D <= 128, one value per lane per fragment.

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void attention_online_kernel(const float* q, const float* k,
                                        const float* v, float* output,
                                        int heads, int s, int d, bool causal)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bh = blockIdx.x;
    const int row = blockIdx.y * kBlockRows + warp;
    const bool valid_row = bh < heads && row < s;
    const long long head_base = static_cast<long long>(bh) * s * d;

    extern __shared__ float shared[];
    float* shared_k = shared;
    float* shared_v = shared + kKeyTile * d;

    float q_values[kMaxFragments];
    float out_values[kMaxFragments];
#pragma unroll
    for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
        const int x = lane + fragment * 32;
        q_values[fragment] = valid_row && x < d
                                 ? q[head_base + static_cast<long long>(row) * d + x]
                                 : 0.0f;
        out_values[fragment] = 0.0f;
    }

    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    const float scale = rsqrtf(static_cast<float>(d));
    for (int key_begin = 0; key_begin < s; key_begin += kKeyTile) {
        for (int index = tid; index < kKeyTile * d; index += kThreads) {
            const int key_in_tile = index / d;
            const int x = index - key_in_tile * d;
            const int key = key_begin + key_in_tile;
            const float kval = key < s ? k[head_base +
                static_cast<long long>(key) * d + x] : 0.0f;
            const float vval = key < s ? v[head_base +
                static_cast<long long>(key) * d + x] : 0.0f;
            shared_k[index] = kval;
            shared_v[index] = vval;
        }
        __syncthreads();

        const int valid_keys = min(kKeyTile, s - key_begin);
        for (int key_in_tile = 0; key_in_tile < valid_keys; ++key_in_tile) {
            const int key = key_begin + key_in_tile;
            if (!valid_row || (causal && key > row)) continue;
            float dot = 0.0f;
#pragma unroll
            for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
                const int x = lane + fragment * 32;
                if (x < d)
                    dot = fmaf(q_values[fragment],
                               shared_k[key_in_tile * d + x], dot);
            }
            dot = warp_sum(dot);
            const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                const float new_maximum = fmaxf(running_maximum, score);
                alpha = expf(running_maximum - new_maximum);
                beta = expf(score - new_maximum);
                running_sum = running_sum * alpha + beta;
                running_maximum = new_maximum;
            }
            alpha = __shfl_sync(0xffffffffu, alpha, 0);
            beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
            for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
                const int x = lane + fragment * 32;
                if (x < d)
                    out_values[fragment] =
                        out_values[fragment] * alpha +
                        beta * shared_v[key_in_tile * d + x];
            }
        }
        __syncthreads();
        if (causal && key_begin + kKeyTile >
                          (blockIdx.y + 1) * kBlockRows - 1)
            break;
    }

    float inverse_sum = lane == 0 ? 1.0f / running_sum : 0.0f;
    inverse_sum = __shfl_sync(0xffffffffu, inverse_sum, 0);
    if (valid_row) {
#pragma unroll
        for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
            const int x = lane + fragment * 32;
            if (x < d)
                output[head_base + static_cast<long long>(row) * d + x] =
                    out_values[fragment] * inverse_sum;
        }
    }
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    AttentionHostData host(options);
    AttentionDeviceData device(options, host);
    const size_t shared_bytes =
        2ULL * kKeyTile * options.d * sizeof(float);
    dim3 grid(options.heads(),
              (options.s + kBlockRows - 1) / kBlockRows);
    auto launch = [&]() {
        attention_online_kernel<<<grid, kThreads, shared_bytes>>>(
            device.q, device.k, device.v, device.output, options.heads(),
            options.s, options.d, options.causal);
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check(options, host, device);
    const double ms =
        attention_median_ms(launch, options.warmup, options.repeat);
    attention_report("attention_v2", "tiled_online_softmax", options, check,
                     ms, 0);
    std::printf("tile_rows=%d tile_keys=%d shared_bytes=%zu\n", kBlockRows,
                kKeyTile, shared_bytes);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
