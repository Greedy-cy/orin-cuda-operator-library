// attention_v0.cu - explicit-score scaled dot-product attention baseline.
//
// Layout: Q/K/V/O [B,H,S,D], scores/probabilities [B,H,S,S].
// Pipeline: QK^T -> stable row softmax -> PV. All storage is FP32.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    int b = 1;
    int h = 8;
    int s = 128;
    int d = 64;
    bool causal = false;
    int warmup = 5;
    int repeat = 15;

    static Options parse(int argc, char** argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const size_t pos = arg.find('=');
            if (pos == std::string::npos) {
                continue;
            }
            const std::string key = arg.substr(0, pos);
            const int value = std::atoi(arg.c_str() + pos + 1);
            if (key == "--b") o.b = value;
            else if (key == "--h") o.h = value;
            else if (key == "--s") o.s = value;
            else if (key == "--d") o.d = value;
            else if (key == "--causal") o.causal = value != 0;
            else if (key == "--warmup") o.warmup = value;
            else if (key == "--repeat") o.repeat = value;
        }
        if (o.b <= 0 || o.h <= 0 || o.s <= 0 || o.d <= 0 ||
            o.warmup < 0 || o.repeat <= 0) {
            std::fprintf(stderr, "all dimensions/repeat must be positive\n");
            std::exit(EXIT_FAILURE);
        }
        return o;
    }
};

__global__ void qk_kernel(const float* q, const float* k, float* scores,
                          int heads, int s, int d, bool causal)
{
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long count = static_cast<long long>(heads) * s * s;
    if (index >= count) return;

    const int col = index % s;
    const long long row_index = index / s;
    const int row = row_index % s;
    const int bh = row_index / s;
    if (causal && col > row) {
        scores[index] = -INFINITY;
        return;
    }

    const long long q_offset = (static_cast<long long>(bh) * s + row) * d;
    const long long k_offset = (static_cast<long long>(bh) * s + col) * d;
    float sum = 0.0f;
    for (int x = 0; x < d; ++x) {
        sum = fmaf(q[q_offset + x], k[k_offset + x], sum);
    }
    scores[index] = sum * rsqrtf(static_cast<float>(d));
}

// Deliberately serial row baseline. Later versions replace this bottleneck with
// block/warp reductions, but v0 remains the simplest stable implementation.
__global__ void softmax_serial_kernel(float* scores, int rows, int s)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float* values = scores + static_cast<long long>(row) * s;

    float maximum = -INFINITY;
    for (int col = 0; col < s; ++col) maximum = fmaxf(maximum, values[col]);

    float sum = 0.0f;
    for (int col = 0; col < s; ++col) {
        values[col] = expf(values[col] - maximum);
        sum += values[col];
    }
    const float inverse = 1.0f / sum;
    for (int col = 0; col < s; ++col) values[col] *= inverse;
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
    const float* p_row = probabilities +
                         (static_cast<long long>(bh) * s + row) * s;
    const float* v_head = v + static_cast<long long>(bh) * s * d;
    float sum = 0.0f;
    for (int col = 0; col < s; ++col) {
        sum = fmaf(p_row[col], v_head[static_cast<long long>(col) * d + x], sum);
    }
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
    if (repeat & 1) return samples[repeat / 2];
    return 0.5 * (samples[repeat / 2 - 1] + samples[repeat / 2]);
}

void cpu_reference(const Options& o, const std::vector<float>& q,
                   const std::vector<float>& k, const std::vector<float>& v,
                   std::vector<float>& output)
{
    const int heads = o.b * o.h;
    const double scale = 1.0 / std::sqrt(static_cast<double>(o.d));
    std::vector<double> scores(o.s);
    for (int bh = 0; bh < heads; ++bh) {
        const size_t head_offset = static_cast<size_t>(bh) * o.s * o.d;
        for (int row = 0; row < o.s; ++row) {
            double maximum = -std::numeric_limits<double>::infinity();
            const int visible = o.causal ? row + 1 : o.s;
            for (int col = 0; col < visible; ++col) {
                double dot = 0.0;
                for (int x = 0; x < o.d; ++x) {
                    dot += static_cast<double>(q[head_offset + row * o.d + x]) *
                           k[head_offset + col * o.d + x];
                }
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
                for (int col = 0; col < visible; ++col) {
                    value += scores[col] *
                             v[head_offset + col * o.d + x];
                }
                output[head_offset + row * o.d + x] =
                    static_cast<float>(value / denominator);
            }
        }
    }
}

int main(int argc, char** argv)
{
    const Options options = Options::parse(argc, argv);
    const int heads = options.b * options.h;
    const size_t tensor_elements =
        static_cast<size_t>(heads) * options.s * options.d;
    const size_t score_elements =
        static_cast<size_t>(heads) * options.s * options.s;
    const size_t tensor_bytes = tensor_elements * sizeof(float);
    const size_t score_bytes = score_elements * sizeof(float);

    std::vector<float> h_q(tensor_elements), h_k(tensor_elements),
        h_v(tensor_elements), h_o(tensor_elements), h_reference(tensor_elements);
    for (size_t i = 0; i < tensor_elements; ++i) {
        h_q[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) /
                 100.0f;
        h_k[i] = static_cast<float>(static_cast<int>((i * 29 + 7) % 103) - 51) /
                 100.0f;
        h_v[i] = static_cast<float>(static_cast<int>((i * 43 + 11) % 107) - 53) /
                 100.0f;
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr,
          *d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_q), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_k), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_v), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o), tensor_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scores), score_bytes));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), tensor_bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int qk_blocks = static_cast<int>((score_elements + threads - 1) / threads);
    const int pv_blocks =
        static_cast<int>((tensor_elements + threads - 1) / threads);
    auto launch = [&]() {
        qk_kernel<<<qk_blocks, threads>>>(d_q, d_k, d_scores, heads, options.s,
                                          options.d, options.causal);
        softmax_serial_kernel<<<heads * options.s, 1>>>(
            d_scores, heads * options.s, options.s);
        pv_kernel<<<pv_blocks, threads>>>(d_scores, d_v, d_o, heads, options.s,
                                          options.d);
    };

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, tensor_bytes, cudaMemcpyDeviceToHost));

    const double verification_work = static_cast<double>(heads) * options.s *
                                     options.s * options.d;
    bool correct = true;
    double max_abs = 0.0, max_rel = 0.0;
    if (verification_work <= 2.0e8) {
        cpu_reference(options, h_q, h_k, h_v, h_reference);
        for (size_t i = 0; i < tensor_elements; ++i) {
            const double abs_error = std::abs(static_cast<double>(h_o[i]) -
                                              h_reference[i]);
            const double rel_error = abs_error /
                std::max(1.0e-6, std::abs(static_cast<double>(h_reference[i])));
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, rel_error);
            if (abs_error > 1.0e-4 + 1.0e-4 * std::abs(h_reference[i])) {
                correct = false;
            }
        }
    } else {
        for (float value : h_o) correct = correct && std::isfinite(value);
    }

    std::vector<float> h_probabilities;
    double max_row_sum_error = 0.0;
    if (score_elements <= (1ULL << 26)) {
        h_probabilities.resize(score_elements);
        CUDA_CHECK(cudaMemcpy(h_probabilities.data(), d_scores, score_bytes,
                              cudaMemcpyDeviceToHost));
        for (int row = 0; row < heads * options.s; ++row) {
            double sum = 0.0;
            for (int col = 0; col < options.s; ++col) {
                sum += h_probabilities[static_cast<size_t>(row) * options.s + col];
            }
            max_row_sum_error = std::max(max_row_sum_error, std::abs(sum - 1.0));
        }
    }

    const double milliseconds =
        median_ms(launch, options.warmup, options.repeat);
    const double dense_flops = 4.0 * heads * options.s * options.s * options.d;
    const double gflops = dense_flops / (milliseconds * 1.0e6);
    std::printf("attention_v0 B=%d H=%d S=%d D=%d causal=%d\n", options.b,
                options.h, options.s, options.d, options.causal ? 1 : 0);
    std::printf("path=explicit_scores kernels=3 score_buffer_bytes=%zu\n",
                score_bytes);
    std::printf("median_ms=%.6f dense_gflops=%.3f warmup=%d repeat=%d\n",
                milliseconds, gflops, options.warmup, options.repeat);
    std::printf("max_abs=%.9g max_rel=%.9g max_row_sum_error=%.9g correct=%s\n",
                max_abs, max_rel, max_row_sum_error, correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_scores));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
