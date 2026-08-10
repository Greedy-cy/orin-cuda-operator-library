// attention_v3.cu - finite BM/BN scan for the FP32 online-softmax kernel.

#include "attention_common.h"

constexpr int kMaxFragments = 4;

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template <int BlockRows, int KeyTile>
__global__ void attention_tuned_kernel(const float* q, const float* k,
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
            float alpha = 1.0f, beta = 0.0f;
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

struct TileConfig { int rows, keys; };

constexpr TileConfig kConfigs[] = {
    {4, 16}, {4, 32}, {8, 16}, {8, 32}, {8, 64},
    {16, 16}, {16, 32}, {16, 64}, {32, 32}
};

template <int Rows, int Keys>
void set_dynamic_shared(size_t bytes)
{
    if (bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(attention_tuned_kernel<Rows, Keys>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bytes)));
}

template <int Rows, int Keys>
void launch_tile(const AttentionOptions& o, const AttentionDeviceData& data,
                 size_t shared_bytes)
{
    dim3 grid(o.heads(), (o.s + Rows - 1) / Rows);
    attention_tuned_kernel<Rows, Keys><<<grid, Rows * 32, shared_bytes>>>(
        data.q, data.k, data.v, data.output, o.heads(), o.s, o.d, o.causal);
}

void configure(int config, size_t bytes)
{
    switch (config) {
        case 0: set_dynamic_shared<4,16>(bytes); break;
        case 1: set_dynamic_shared<4,32>(bytes); break;
        case 2: set_dynamic_shared<8,16>(bytes); break;
        case 3: set_dynamic_shared<8,32>(bytes); break;
        case 4: set_dynamic_shared<8,64>(bytes); break;
        case 5: set_dynamic_shared<16,16>(bytes); break;
        case 6: set_dynamic_shared<16,32>(bytes); break;
        case 7: set_dynamic_shared<16,64>(bytes); break;
        case 8: set_dynamic_shared<32,32>(bytes); break;
    }
}

void launch_config(int config, const AttentionOptions& o,
                   const AttentionDeviceData& data, size_t bytes)
{
    switch (config) {
        case 0: launch_tile<4,16>(o, data, bytes); break;
        case 1: launch_tile<4,32>(o, data, bytes); break;
        case 2: launch_tile<8,16>(o, data, bytes); break;
        case 3: launch_tile<8,32>(o, data, bytes); break;
        case 4: launch_tile<8,64>(o, data, bytes); break;
        case 5: launch_tile<16,16>(o, data, bytes); break;
        case 6: launch_tile<16,32>(o, data, bytes); break;
        case 7: launch_tile<16,64>(o, data, bytes); break;
        case 8: launch_tile<32,32>(o, data, bytes); break;
        default:
            std::fprintf(stderr, "config must be in [0,8]\n");
            std::exit(EXIT_FAILURE);
    }
}

int auto_config(const AttentionOptions& o)
{
    // Filled from the finite scan documented in reports/v3.md.
    if (o.s <= 128) return 8;  // BM=32, BN=32: more row parallelism.
    if (o.d <= 64) return 7;   // BM=16, BN=64: shared remains 32 KiB.
    return 8;                  // BM=32, BN=32 for D=128.
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    const int config = options.config >= 0 ? options.config : auto_config(options);
    if (config < 0 || config >= static_cast<int>(sizeof(kConfigs) /
                                                 sizeof(kConfigs[0]))) {
        std::fprintf(stderr, "config must be in [0,8]\n");
        return EXIT_FAILURE;
    }
    const TileConfig tile = kConfigs[config];
    const size_t shared_bytes =
        2ULL * tile.keys * options.d * sizeof(float);
    configure(config, shared_bytes);

    AttentionHostData host(options);
    AttentionDeviceData device(options, host);
    auto launch = [&]() {
        launch_config(config, options, device, shared_bytes);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check(options, host, device);
    const double ms = attention_median_ms(launch, options.warmup, options.repeat);
    attention_report("attention_v3", "tuned_online_softmax", options, check,
                     ms, 0);
    std::printf("config=%d tile_rows=%d tile_keys=%d threads=%d shared_bytes=%zu\n",
                config, tile.rows, tile.keys, tile.rows * 32, shared_bytes);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
