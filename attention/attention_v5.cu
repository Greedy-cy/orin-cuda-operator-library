// attention_v5.cu - vectorized cp.async double-buffered K/V tiles.

#include "attention_common.h"

constexpr int kMaxFragments = 4;

__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                            const void* global_source)
{
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 : : "r"(address), "l"(global_source));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
}

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

template <int Threads, int KeyTile>
__device__ __forceinline__ void issue_kv_tile(
    float* shared_k, float* shared_v, int buffer, const float* k,
    const float* v, long long head_base, int key_begin, int d)
{
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

    extern __shared__ __align__(16) float shared[];
    float* shared_k = shared;
    float* shared_v = shared + 2 * KeyTile * d;
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

    const int total_tiles = s / KeyTile;
    const int needed_keys = min(s, (blockIdx.y + 1) * BlockRows);
    const int tile_count = causal
        ? (needed_keys + KeyTile - 1) / KeyTile : total_tiles;
    issue_kv_tile<Threads, KeyTile>(shared_k, shared_v, 0, k, v, head_base,
                                    0, d);
    cp_async_commit();

    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    const float scale = rsqrtf(static_cast<float>(d));
    for (int tile = 0; tile < tile_count; ++tile) {
        const int read_buffer = tile & 1;
        const int write_buffer = read_buffer ^ 1;
        const bool has_next = tile + 1 < tile_count;
        if (has_next) {
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
                    dot = fmaf(q_values[fragment],
                               tile_k[key_in_tile * d + x], dot);
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
                    out_values[fragment] = out_values[fragment] * alpha +
                        beta * tile_v[key_in_tile * d + x];
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
            if (x < d)
                output[head_base + static_cast<long long>(row) * d + x] =
                    out_values[fragment] * inverse;
        }
    }
}

// Boundary-only fallback: one warp per row streams K/V directly from global.
__global__ void attention_scalar_fallback(const float* q, const float* k,
                                          const float* v, float* output,
                                          int heads, int s, int d, bool causal)
{
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
    float maximum = -INFINITY, sum = 0.0f;
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
                out_values[f] = out_values[f] * alpha + beta *
                    v[base + static_cast<long long>(key) * d + x];
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

template <int Rows, int Keys>
void launch_async(const AttentionOptions& o, const AttentionDeviceData& data,
                  size_t shared_bytes)
{
    dim3 grid(o.heads(), (o.s + Rows - 1) / Rows);
    attention_cp_async_kernel<Rows, Keys><<<grid, Rows * 32, shared_bytes>>>(
        data.q, data.k, data.v, data.output, o.heads(), o.s, o.d, o.causal);
}

int main(int argc, char** argv)
{
    const AttentionOptions options = AttentionOptions::parse(argc, argv);
    AttentionHostData host(options);
    AttentionDeviceData device(options, host);
    const bool aligned = (options.d == 64 || options.d == 128) &&
                         options.s % 32 == 0;
    const int rows = options.d == 64 ? 16 : 32;
    const int keys = 32;
    const size_t shared_bytes = 4ULL * keys * options.d * sizeof(float);
    if (aligned && options.d == 128)
        CUDA_CHECK(cudaFuncSetAttribute(attention_cp_async_kernel<32,32>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)));
    auto launch = [&]() {
        if (!aligned) {
            dim3 grid(options.heads(), (options.s + 7) / 8);
            attention_scalar_fallback<<<grid, 256>>>(
                device.q, device.k, device.v, device.output, options.heads(),
                options.s, options.d, options.causal);
        } else if (options.d == 64) {
            launch_async<16,32>(options, device, shared_bytes);
        } else {
            launch_async<32,32>(options, device, shared_bytes);
        }
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const AttentionCheckResult check = attention_check(options, host, device);
    const double ms = attention_median_ms(launch, options.warmup, options.repeat);
    attention_report("attention_v5",
                     aligned ? "cp_async_double_buffer" : "scalar_fallback",
                     options, check, ms, 0);
    std::printf("aligned=%d tile_rows=%d tile_keys=%d shared_bytes=%zu\n",
                aligned ? 1 : 0, aligned ? rows : 8,
                aligned ? keys : 0, aligned ? shared_bytes : 0);
    return check.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
