#include "operatorlib/operators.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                   cudaGetErrorString(error));                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

template <typename T>
T* copy_to_device(const std::vector<T>& host, cudaStream_t stream) {
  T* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  return device;
}

std::vector<float> cpu_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v, int total_heads,
                                 int s, int d, bool causal) {
  std::vector<float> output(q.size());
  std::vector<double> scores(s);
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  for (int head = 0; head < total_heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head) * s * d;
    for (int row = 0; row < s; ++row) {
      const int visible = causal ? row + 1 : s;
      double maximum = -INFINITY;
      for (int key = 0; key < visible; ++key) {
        double dot = 0.0;
        for (int x = 0; x < d; ++x) {
          dot += static_cast<double>(q[base + row * d + x]) *
                 k[base + key * d + x];
        }
        scores[key] = dot * scale;
        maximum = std::max(maximum, scores[key]);
      }
      double sum = 0.0;
      for (int key = 0; key < visible; ++key) {
        scores[key] = std::exp(scores[key] - maximum);
        sum += scores[key];
      }
      for (int x = 0; x < d; ++x) {
        double value = 0.0;
        for (int key = 0; key < visible; ++key)
          value += scores[key] * v[base + key * d + x];
        output[base + row * d + x] = static_cast<float>(value / sum);
      }
    }
  }
  return output;
}

void check_output(const std::vector<float>& output,
                  const std::vector<float>& reference, float atol,
                  float rtol) {
  std::size_t violations = 0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float error = std::abs(output[i] - reference[i]);
    if (!std::isfinite(output[i]) ||
        error > atol + rtol * std::abs(reference[i]))
      ++violations;
  }
  if (violations != 0) std::exit(EXIT_FAILURE);
}

void test_f32(cudaStream_t stream, int s, int d, bool causal, bool extreme) {
  constexpr int batch = 1;
  constexpr int heads = 2;
  const std::size_t count = static_cast<std::size_t>(batch) * heads * s * d;
  std::vector<float> q(count), k(count), v(count);
  std::mt19937 generator(31 + s + d);
  const float qk_range = extreme ? 20.0f : 1.0f;
  std::uniform_real_distribution<float> qk_distribution(-qk_range, qk_range);
  std::uniform_real_distribution<float> v_distribution(-1.0f, 1.0f);
  for (float& value : q) value = qk_distribution(generator);
  for (float& value : k) value = qk_distribution(generator);
  for (float& value : v) value = v_distribution(generator);
  const std::vector<float> reference =
      cpu_attention(q, k, v, batch * heads, s, d, causal);
  float* device_q = copy_to_device(q, stream);
  float* device_k = copy_to_device(k, stream);
  float* device_v = copy_to_device(v, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  CUDA_CHECK(operatorlib::attention_f32(device_q, device_k, device_v,
                                        device_output, batch, heads, s, d,
                                        causal, stream));
  std::vector<float> output(count);
  CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output,
                             count * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  check_output(output, reference, 1.0e-4f, 1.0e-4f);
  CUDA_CHECK(cudaFree(device_q));
  CUDA_CHECK(cudaFree(device_k));
  CUDA_CHECK(cudaFree(device_v));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("attention_f32[S%d,D%d,causal%d,extreme%d]=PASS\n", s, d,
              causal ? 1 : 0, extreme ? 1 : 0);
}

void test_f16(cudaStream_t stream, int s, int d, bool causal, bool extreme) {
  constexpr int batch = 1;
  constexpr int heads = 2;
  const std::size_t count = static_cast<std::size_t>(batch) * heads * s * d;
  std::vector<__half> q(count), k(count), v(count);
  std::vector<float> q_float(count), k_float(count), v_float(count);
  std::mt19937 generator(43 + s + d);
  const float qk_range = extreme ? 20.0f : 1.0f;
  std::uniform_real_distribution<float> qk_distribution(-qk_range, qk_range);
  std::uniform_real_distribution<float> v_distribution(-1.0f, 1.0f);
  for (std::size_t i = 0; i < count; ++i) {
    q[i] = __float2half_rn(qk_distribution(generator));
    q_float[i] = __half2float(q[i]);
  }
  for (std::size_t i = 0; i < count; ++i) {
    k[i] = __float2half_rn(qk_distribution(generator));
    k_float[i] = __half2float(k[i]);
  }
  for (std::size_t i = 0; i < count; ++i) {
    v[i] = __float2half_rn(v_distribution(generator));
    v_float[i] = __half2float(v[i]);
  }
  const std::vector<float> reference = cpu_attention(
      q_float, k_float, v_float, batch * heads, s, d, causal);
  __half* device_q = copy_to_device(q, stream);
  __half* device_k = copy_to_device(k, stream);
  __half* device_v = copy_to_device(v, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  CUDA_CHECK(operatorlib::attention_f16(device_q, device_k, device_v,
                                        device_output, batch, heads, s, d,
                                        causal, stream));
  std::vector<float> output(count);
  CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output,
                             count * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  check_output(output, reference, 2.0e-2f, 2.0e-2f);
  CUDA_CHECK(cudaFree(device_q));
  CUDA_CHECK(cudaFree(device_k));
  CUDA_CHECK(cudaFree(device_v));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("attention_f16[S%d,D%d,causal%d,extreme%d]=PASS\n", s, d,
              causal ? 1 : 0, extreme ? 1 : 0);
}

}  // namespace

int main() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  test_f32(stream, 32, 64, false, false);
  test_f32(stream, 32, 128, true, false);
  test_f32(stream, 37, 71, true, true);
  test_f16(stream, 32, 64, false, false);
  test_f16(stream, 32, 128, true, false);
  test_f16(stream, 37, 71, true, true);
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("attention_smoke=PASS\n");
  return EXIT_SUCCESS;
}
