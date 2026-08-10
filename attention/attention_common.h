#pragma once

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

struct AttentionOptions {
    int b = 1, h = 8, s = 128, d = 64;
    bool causal = false;
    bool extreme = false;
    int config = -1;
    int warmup = 5, repeat = 15;

    static AttentionOptions parse(int argc, char** argv)
    {
        AttentionOptions o;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const size_t pos = arg.find('=');
            if (pos == std::string::npos) continue;
            const std::string key = arg.substr(0, pos);
            const int value = std::atoi(arg.c_str() + pos + 1);
            if (key == "--b") o.b = value;
            else if (key == "--h") o.h = value;
            else if (key == "--s") o.s = value;
            else if (key == "--d") o.d = value;
            else if (key == "--causal") o.causal = value != 0;
            else if (key == "--extreme") o.extreme = value != 0;
            else if (key == "--config") o.config = value;
            else if (key == "--warmup") o.warmup = value;
            else if (key == "--repeat") o.repeat = value;
        }
        if (o.b <= 0 || o.h <= 0 || o.s <= 0 || o.d <= 0 || o.d > 128 ||
            o.warmup < 0 || o.repeat <= 0) {
            std::fprintf(stderr,
                         "require positive B/H/S, 1<=D<=128 and repeat>0\n");
            std::exit(EXIT_FAILURE);
        }
        return o;
    }

    int heads() const { return b * h; }
    size_t tensor_elements() const
    {
        return static_cast<size_t>(heads()) * s * d;
    }
    size_t tensor_bytes() const { return tensor_elements() * sizeof(float); }
    double dense_flops() const
    {
        return 4.0 * heads() * static_cast<double>(s) * s * d;
    }
    bool cpu_verifiable() const
    {
        return static_cast<double>(heads()) * s * s * d <= 2.0e8;
    }
};

struct AttentionHostData {
    std::vector<float> q, k, v, output, reference;

    explicit AttentionHostData(const AttentionOptions& o)
        : q(o.tensor_elements()), k(o.tensor_elements()),
          v(o.tensor_elements()), output(o.tensor_elements()),
          reference(o.tensor_elements())
    {
        const float qk_scale = o.extreme ? 20.0f : 1.0f;
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = qk_scale *
                   static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) /
                   100.0f;
            k[i] = qk_scale *
                   static_cast<float>(static_cast<int>((i * 29 + 7) % 103) - 51) /
                   100.0f;
            v[i] = static_cast<float>(static_cast<int>((i * 43 + 11) % 107) - 53) /
                   100.0f;
        }
    }
};

struct AttentionDeviceData {
    float *q = nullptr, *k = nullptr, *v = nullptr, *output = nullptr;

    AttentionDeviceData(const AttentionOptions& o, const AttentionHostData& host)
    {
        const size_t bytes = o.tensor_bytes();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&q), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&k), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&v), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
        CUDA_CHECK(cudaMemcpy(q, host.q.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k, host.k.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v, host.v.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~AttentionDeviceData()
    {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(output);
    }

    AttentionDeviceData(const AttentionDeviceData&) = delete;
    AttentionDeviceData& operator=(const AttentionDeviceData&) = delete;
};

inline void attention_cpu_reference(const AttentionOptions& o,
                                    const AttentionHostData& data,
                                    std::vector<float>& output)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(o.d));
    std::vector<double> scores(o.s);
    for (int bh = 0; bh < o.heads(); ++bh) {
        const size_t base = static_cast<size_t>(bh) * o.s * o.d;
        for (int row = 0; row < o.s; ++row) {
            const int visible = o.causal ? row + 1 : o.s;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int col = 0; col < visible; ++col) {
                double dot = 0.0;
                for (int x = 0; x < o.d; ++x)
                    dot += static_cast<double>(data.q[base + row * o.d + x]) *
                           data.k[base + col * o.d + x];
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
                    value += scores[col] * data.v[base + col * o.d + x];
                output[base + row * o.d + x] =
                    static_cast<float>(value / denominator);
            }
        }
    }
}
struct AttentionCheckResult {
    bool correct = true;
    bool cpu_verified = false;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

inline AttentionCheckResult attention_check_output(const AttentionOptions& o,
                                                    AttentionHostData& host,
                                                    const float* device_output,
                                                    double atol = 1.0e-4,
                                                    double rtol = 1.0e-4)
{
    CUDA_CHECK(cudaMemcpy(host.output.data(), device_output, o.tensor_bytes(),
                          cudaMemcpyDeviceToHost));
    AttentionCheckResult result;
    result.cpu_verified = o.cpu_verifiable();
    if (!result.cpu_verified) {
        for (float value : host.output)
            result.correct = result.correct && std::isfinite(value);
        return result;
    }

    attention_cpu_reference(o, host, host.reference);
    for (size_t i = 0; i < host.output.size(); ++i) {
        const double absolute = std::abs(static_cast<double>(host.output[i]) -
                                         host.reference[i]);
        const double relative = absolute /
            std::max(1.0e-6, std::abs(static_cast<double>(host.reference[i])));
        result.max_abs = std::max(result.max_abs, absolute);
        result.max_rel = std::max(result.max_rel, relative);
        if (absolute > atol + rtol * std::abs(host.reference[i]))
            result.correct = false;
    }
    return result;
}

inline AttentionCheckResult attention_check(const AttentionOptions& o,
                                            AttentionHostData& host,
                                            const AttentionDeviceData& device)
{
    return attention_check_output(o, host, device.output);
}

template <typename Launch>
double attention_median_ms(Launch&& launch, int warmup, int repeat)
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

inline void attention_report(const char* version, const char* path,
                             const AttentionOptions& o,
                             const AttentionCheckResult& check, double ms,
                             size_t intermediate_bytes)
{
    std::printf("%s B=%d H=%d S=%d D=%d causal=%d extreme=%d\n", version, o.b,
                o.h, o.s, o.d, o.causal ? 1 : 0, o.extreme ? 1 : 0);
    std::printf("path=%s intermediate_bytes=%zu\n", path, intermediate_bytes);
    std::printf("median_ms=%.6f dense_gflops=%.3f warmup=%d repeat=%d\n", ms,
                o.dense_flops() / (ms * 1.0e6), o.warmup, o.repeat);
    std::printf("verification=%s max_abs=%.9g max_rel=%.9g correct=%s\n",
                check.cpu_verified ? "cpu_double" : "finite_output",
                check.max_abs, check.max_rel, check.correct ? "PASS" : "FAIL");
}
