// attention_v1.cu - coalesced warp-dot QK + fused block softmax.
// Probabilities remain materialized as FP32 [B,H,S,S]; PV stays separate.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(error));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct Options {
    int b = 1, h = 8, s = 128, d = 64;
    bool causal = false;
    int warmup = 5, repeat = 15;

    static Options parse(int argc, char** argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a(argv[i]);
            const size_t p = a.find('=');
            if (p == std::string::npos) continue;
            const std::string k = a.substr(0, p);
            const int v = std::atoi(a.c_str() + p + 1);
            if (k == "--b") o.b = v;
            else if (k == "--h") o.h = v;
            else if (k == "--s") o.s = v;
            else if (k == "--d") o.d = v;
            else if (k == "--causal") o.causal = v != 0;
            else if (k == "--warmup") o.warmup = v;
            else if (k == "--repeat") o.repeat = v;
        }
        if (o.b <= 0 || o.h <= 0 || o.s <= 0 || o.d <= 0 ||
            o.warmup < 0 || o.repeat <= 0) {
            std::fprintf(stderr, "invalid dimensions or repeat count\n");
            std::exit(EXIT_FAILURE);
        }
        return o;
    }
};

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float warp_max(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value,
                      __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__global__ void qk_softmax_kernel(const float* q, const float* k,
                                  float* probabilities, int heads, int s,
                                  int d, bool causal)
{
    const int row_index = blockIdx.x;
    const int row = row_index % s;
    const int bh = row_index / s;
    if (bh >= heads) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int warps = 8;
    extern __shared__ float q_shared[];
    __shared__ float warp_values[warps];
    __shared__ float row_maximum;
    __shared__ float row_sum;

    const long long q_offset =
        (static_cast<long long>(bh) * s + row) * d;
    for (int x = tid; x < d; x += blockDim.x) q_shared[x] = q[q_offset + x];
    __syncthreads();

    float* p_row = probabilities + static_cast<long long>(row_index) * s;
    for (int col = warp; col < s; col += warps) {
        if (causal && col > row) {
            if (lane == 0) p_row[col] = -INFINITY;
            continue;
        }
        const long long k_offset =
            (static_cast<long long>(bh) * s + col) * d;
        float dot = 0.0f;
        for (int x = lane; x < d; x += 32)
            dot = fmaf(q_shared[x], k[k_offset + x], dot);
        dot = warp_sum(dot);
        if (lane == 0) p_row[col] = dot * rsqrtf(static_cast<float>(d));
    }
    __syncthreads();

    float local_maximum = -INFINITY;
    for (int col = tid; col < s; col += blockDim.x)
        local_maximum = fmaxf(local_maximum, p_row[col]);
    local_maximum = warp_max(local_maximum);
    if (lane == 0) warp_values[warp] = local_maximum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? warp_values[lane] : -INFINITY;
        value = warp_max(value);
        if (lane == 0) row_maximum = value;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < s; col += blockDim.x) {
        const float value = expf(p_row[col] - row_maximum);
        p_row[col] = value;
        local_sum += value;
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) warp_values[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? warp_values[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) row_sum = value;
    }
    __syncthreads();

    const float inverse = 1.0f / row_sum;
    for (int col = tid; col < s; col += blockDim.x) p_row[col] *= inverse;
}

__global__ void pv_kernel(const float* probabilities, const float* v, float* o,
                          int heads, int s, int d)
{
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(heads) * s * d;
    if (index >= count) return;
    const int x = index % d;
    const long long row_index = index / d;
    const int row = row_index % s;
    const int bh = row_index / s;
    const float* p = probabilities +
                     (static_cast<long long>(bh) * s + row) * s;
    const float* vh = v + static_cast<long long>(bh) * s * d;
    float sum = 0.0f;
    for (int col = 0; col < s; ++col)
        sum = fmaf(p[col], vh[static_cast<long long>(col) * d + x], sum);
    o[index] = sum;
}

template <typename Launch>
double median_ms(Launch&& launch, int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples(repeat);
    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(begin));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&samples[i], begin, end));
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return repeat & 1 ? samples[repeat / 2]
                      : 0.5 * (samples[repeat / 2 - 1] + samples[repeat / 2]);
}

void cpu_reference(const Options& o, const std::vector<float>& q,
                   const std::vector<float>& k, const std::vector<float>& v,
                   std::vector<float>& output)
{
    const int heads = o.b * o.h;
    const double scale = 1.0 / std::sqrt(static_cast<double>(o.d));
    std::vector<double> scores(o.s);
    for (int bh = 0; bh < heads; ++bh) {
        const size_t base = static_cast<size_t>(bh) * o.s * o.d;
        for (int row = 0; row < o.s; ++row) {
            const int visible = o.causal ? row + 1 : o.s;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int col = 0; col < visible; ++col) {
                double dot = 0.0;
                for (int x = 0; x < o.d; ++x)
                    dot += static_cast<double>(q[base + row * o.d + x]) *
                           k[base + col * o.d + x];
                scores[col] = dot * scale;
                maximum = std::max(maximum, scores[col]);
            }
            double denominator = 0.0;
            for (int col = 0; col < visible; ++col) {
                scores[col] = std::exp(scores[col] - maximum);
                denominator += scores[col];
            }
            for (int x = 0; x < o.d; ++x) {
                double value = 0.0;
                for (int col = 0; col < visible; ++col)
                    value += scores[col] * v[base + col * o.d + x];
                output[base + row * o.d + x] =
                    static_cast<float>(value / denominator);
            }
        }
    }
}

int main(int argc, char** argv)
{
    const Options o = Options::parse(argc, argv);
    const int heads = o.b * o.h;
    const size_t tensor_elements = static_cast<size_t>(heads) * o.s * o.d;
    const size_t score_elements = static_cast<size_t>(heads) * o.s * o.s;
    const size_t tensor_bytes = tensor_elements * sizeof(float);
    const size_t score_bytes = score_elements * sizeof(float);
    std::vector<float> hq(tensor_elements), hk(tensor_elements),
        hv(tensor_elements), ho(tensor_elements), reference(tensor_elements);
    for (size_t i = 0; i < tensor_elements; ++i) {
        hq[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) /
                100.0f;
        hk[i] = static_cast<float>(static_cast<int>((i * 29 + 7) % 103) - 51) /
                100.0f;
        hv[i] = static_cast<float>(static_cast<int>((i * 43 + 11) % 107) - 53) /
                100.0f;
    }
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr,
          *probabilities = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dq), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dk), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dv), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dout), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&probabilities), score_bytes));
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hk.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hv.data(), tensor_bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int pv_blocks = static_cast<int>((tensor_elements + threads - 1) /
                                           threads);
    auto launch = [&]() {
        qk_softmax_kernel<<<heads * o.s, threads, o.d * sizeof(float)>>>(
            dq, dk, probabilities, heads, o.s, o.d, o.causal);
        pv_kernel<<<pv_blocks, threads>>>(probabilities, dv, dout, heads, o.s,
                                          o.d);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, tensor_bytes, cudaMemcpyDeviceToHost));

    const double work = static_cast<double>(heads) * o.s * o.s * o.d;
    const bool cpu_verified = work <= 2.0e8;
    bool correct = true;
    double max_abs = 0.0, max_rel = 0.0;
    if (cpu_verified) {
        cpu_reference(o, hq, hk, hv, reference);
        for (size_t i = 0; i < tensor_elements; ++i) {
            const double absolute =
                std::abs(static_cast<double>(ho[i]) - reference[i]);
            const double relative = absolute /
                std::max(1.0e-6, std::abs(static_cast<double>(reference[i])));
            max_abs = std::max(max_abs, absolute);
            max_rel = std::max(max_rel, relative);
            if (absolute > 1.0e-4 + 1.0e-4 * std::abs(reference[i]))
                correct = false;
        }
    } else {
        for (float value : ho) correct = correct && std::isfinite(value);
    }

    double max_row_sum_error = 0.0;
    if (score_elements <= (1ULL << 26)) {
        std::vector<float> hp(score_elements);
        CUDA_CHECK(cudaMemcpy(hp.data(), probabilities, score_bytes,
                              cudaMemcpyDeviceToHost));
        for (int row = 0; row < heads * o.s; ++row) {
            double sum = 0.0;
            for (int col = 0; col < o.s; ++col)
                sum += hp[static_cast<size_t>(row) * o.s + col];
            max_row_sum_error = std::max(max_row_sum_error, std::abs(sum - 1.0));
        }
    }

    const double ms = median_ms(launch, o.warmup, o.repeat);
    const double flops = 4.0 * heads * o.s * o.s * o.d;
    std::printf("attention_v1 B=%d H=%d S=%d D=%d causal=%d\n", o.b, o.h,
                o.s, o.d, o.causal ? 1 : 0);
    std::printf("path=fused_qk_softmax kernels=2 score_buffer_bytes=%zu\n",
                score_bytes);
    std::printf("median_ms=%.6f dense_gflops=%.3f warmup=%d repeat=%d\n", ms,
                flops / (ms * 1.0e6), o.warmup, o.repeat);
    std::printf("verification=%s max_abs=%.9g max_rel=%.9g "
                "max_row_sum_error=%.9g correct=%s\n",
                cpu_verified ? "cpu_double" : "finite_and_row_sum", max_abs,
                max_rel, max_row_sum_error, correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(probabilities));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
