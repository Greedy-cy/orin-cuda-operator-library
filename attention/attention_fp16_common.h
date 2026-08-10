#pragma once

#include "attention_common.h"

#include <cuda_fp16.h>

struct HalfAttentionDeviceData {
    __half *q = nullptr, *k = nullptr, *v = nullptr;
    float* output = nullptr;

    HalfAttentionDeviceData(const AttentionOptions& o, AttentionHostData& host)
    {
        std::vector<__half> hq(o.tensor_elements()), hk(o.tensor_elements()),
            hv(o.tensor_elements());
        for (size_t i = 0; i < o.tensor_elements(); ++i) {
            hq[i] = __float2half_rn(host.q[i]);
            hk[i] = __float2half_rn(host.k[i]);
            hv[i] = __float2half_rn(host.v[i]);
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

    ~HalfAttentionDeviceData()
    {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(output);
    }
};

__device__ __forceinline__ float fp16_warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void half_attention_scalar_fallback(
    const __half* q, const __half* k, const __half* v, float* output,
    int heads, int s, int d, bool causal)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int bh = blockIdx.x;
    const int row = blockIdx.y * 8 + warp;
    if (bh >= heads || row >= s) return;
    const long long base = static_cast<long long>(bh) * s * d;
    float q_values[4] = {}, out_values[4] = {};
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        const int x = lane + f * 32;
        if (x < d)
            q_values[f] = __half2float(
                q[base + static_cast<long long>(row) * d + x]);
    }
    float maximum = -INFINITY, sum = 0.0f;
    const int visible = causal ? row + 1 : s;
    const float scale = rsqrtf(static_cast<float>(d));
    for (int key = 0; key < visible; ++key) {
        float dot = 0.0f;
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const int x = lane + f * 32;
            if (x < d)
                dot = fmaf(q_values[f], __half2float(
                    k[base + static_cast<long long>(key) * d + x]), dot);
        }
        dot = fp16_warp_sum(dot);
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
        for (int f = 0; f < 4; ++f) {
            const int x = lane + f * 32;
            if (x < d)
                out_values[f] = out_values[f] * alpha + beta * __half2float(
                    v[base + static_cast<long long>(key) * d + x]);
        }
    }
    float inverse = lane == 0 ? 1.0f / sum : 0.0f;
    inverse = __shfl_sync(0xffffffffu, inverse, 0);
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        const int x = lane + f * 32;
        if (x < d)
            output[base + static_cast<long long>(row) * d + x] =
                out_values[f] * inverse;
    }
}

inline void launch_half_attention_fallback(
    const AttentionOptions& o, const HalfAttentionDeviceData& data)
{
    dim3 grid(o.heads(), (o.s + 7) / 8);
    half_attention_scalar_fallback<<<grid, 256>>>(
        data.q, data.k, data.v, data.output, o.heads(), o.s, o.d, o.causal);
}

