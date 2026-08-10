// attention_v4.cu - paired-key dot accumulators and online-softmax update.

#include "attention_common.h"

constexpr int kMaxFragments = 4;

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template <int BlockRows, int KeyTile>
__global__ void attention_paired_kernel(const float* q, const float* k,
                                        const float* v, float* output,
                                        int heads, int s, int d, bool causal)
{
    constexpr int Threads = BlockRows * 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bh = blockIdx.x;
    const int row = blockIdx.y * BlockRows + warp;
    const bool valid_row = bh < heads && row < s;
    const long long head_base = static_cast<long long>(bh) * s * d;

    extern __shared__ float shared[];
    float* shared_k = shared;
    float* shared_v = shared + KeyTile * d;
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
    for (int key_begin = 0; key_begin < s; key_begin += KeyTile) {
        for (int index = tid; index < KeyTile * d; index += Threads) {
            const int key_in_tile = index / d;
            const int x = index - key_in_tile * d;
            const int key = key_begin + key_in_tile;
            shared_k[index] = key < s
                ? k[head_base + static_cast<long long>(key) * d + x] : 0.0f;
            shared_v[index] = key < s
                ? v[head_base + static_cast<long long>(key) * d + x] : 0.0f;
        }
        __syncthreads();

        const int valid_keys = min(KeyTile, s - key_begin);
        for (int first = 0; first < valid_keys; first += 2) {
            const int key0 = key_begin + first;
            const int key1 = key0 + 1;
            const bool use0 = valid_row && (!causal || key0 <= row);
            const bool use1 = use0 && first + 1 < valid_keys &&
                              (!causal || key1 <= row);
            if (!use0) continue;

            float dot0 = 0.0f;
            float dot1 = 0.0f;
#pragma unroll
            for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
                const int x = lane + fragment * 32;
                if (x < d) {
                    dot0 = fmaf(q_values[fragment],
                                shared_k[first * d + x], dot0);
                    if (use1)
                        dot1 = fmaf(q_values[fragment],
                                    shared_k[(first + 1) * d + x], dot1);
                }
            }
            dot0 = warp_sum(dot0);
            dot1 = warp_sum(dot1);
            const float score0 = __shfl_sync(0xffffffffu, dot0, 0) * scale;
            const float score1 = __shfl_sync(0xffffffffu, dot1, 0) * scale;

            float alpha = 1.0f, beta0 = 0.0f, beta1 = 0.0f;
            if (lane == 0) {
                float new_maximum = fmaxf(running_maximum, score0);
                if (use1) new_maximum = fmaxf(new_maximum, score1);
                alpha = expf(running_maximum - new_maximum);
                beta0 = expf(score0 - new_maximum);
                beta1 = use1 ? expf(score1 - new_maximum) : 0.0f;
                running_sum = running_sum * alpha + beta0 + beta1;
                running_maximum = new_maximum;
            }
            alpha = __shfl_sync(0xffffffffu, alpha, 0);
            beta0 = __shfl_sync(0xffffffffu, beta0, 0);
            beta1 = __shfl_sync(0xffffffffu, beta1, 0);
#pragma unroll
            for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
                const int x = lane + fragment * 32;
                if (x < d) {
                    float value = out_values[fragment] * alpha;
                    value = fmaf(beta0, shared_v[first * d + x], value);
                    if (use1)
                        value = fmaf(beta1,
                                     shared_v[(first + 1) * d + x], value);
                    out_values[fragment] = value;
                }
            }
        }
        __syncthreads();
        if (causal && key_begin + KeyTile >
                          (blockIdx.y + 1) * BlockRows - 1)
            break;
    }

    float inverse = lane == 0 ? 1.0f / running_sum : 0.0f;
    inverse = __shfl_sync(0xffffffffu, inverse, 0);
    if (valid_row) {
#pragma unroll
        for (int fragment = 0; fragment < kMaxFragments; ++fragment) {
            const int x = lane + fragment * 32;
            if (x < d)
                output[head_base + static_cast<long long>(row) * d + x] =
                    out_values[fragment] * inverse;
        }
    }
}

template <int Rows, int Keys>
void launch_tile(const AttentionOptions& o, const AttentionDeviceData& data,
                 size_t shared_bytes)
{
    dim3 grid(o.heads(), (o.s + Rows - 1) / Rows);
    attention_paired_kernel<Rows, Keys><<<grid, Rows * 32, shared_bytes>>>(
        data.q, data.k, data.v, data.output, o.heads(), o.s, o.d, o.causal);
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    AttentionHostData host(options);
    AttentionDeviceData device(options, host);
    const bool use_16x64 = options.s > 128 && options.d <= 64;
    const int rows = use_16x64 ? 16 : 32;
    const int keys = use_16x64 ? 64 : 32;
    const size_t shared_bytes = 2ULL * keys * options.d * sizeof(float);
    auto launch = [&]() {
        if (use_16x64) launch_tile<16,64>(options, device, shared_bytes);
        else launch_tile<32,32>(options, device, shared_bytes);
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check(options, host, device);
    const double ms = attention_median_ms(launch, options.warmup, options.repeat);
    attention_report("attention_v4", "paired_online_softmax", options, check,
                     ms, 0);
    std::printf("pair=2 tile_rows=%d tile_keys=%d threads=%d shared_bytes=%zu\n",
                rows, keys, rows * 32, shared_bytes);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
