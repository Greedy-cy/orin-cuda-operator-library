#include "operatorlib/operators.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

void test_reduce(cudaStream_t stream) {
  std::vector<float> input(1000);
  std::iota(input.begin(), input.end(), -500.0f);
  const float reference =
      static_cast<float>(std::accumulate(input.begin(), input.end(), 0.0));
  float* device_input = copy_to_device(input, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
  CUDA_CHECK(operatorlib::reduce_sum_f32(device_input, device_output,
                                         input.size(), stream));
  float output = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(&output, device_output, sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (std::abs(output - reference) > 1.0e-3f) std::exit(EXIT_FAILURE);
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("reduce_sum_f32=PASS\n");
}

void test_softmax_case(cudaStream_t stream, int rows, int cols, float range) {
  std::vector<float> input(rows * cols);
  std::mt19937 generator(7);
  std::uniform_real_distribution<float> distribution(-range, range);
  for (float& value : input) value = distribution(generator);
  float* device_input = copy_to_device(input, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(operatorlib::softmax_f32(device_input, device_output, rows, cols,
                                      stream));
  std::vector<float> output(input.size());
  CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output,
                             output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float max_error = 0.0f;
  float max_sum_error = 0.0f;
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * cols;
    const float maximum =
        *std::max_element(input.begin() + offset, input.begin() + offset + cols);
    double sum = 0.0;
    for (int col = 0; col < cols; ++col)
      sum += std::exp(static_cast<double>(input[offset + col] - maximum));
    double output_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      const float reference = static_cast<float>(
          std::exp(static_cast<double>(input[offset + col] - maximum)) / sum);
      max_error = std::max(max_error,
                           std::abs(output[offset + col] - reference));
      output_sum += output[offset + col];
    }
    max_sum_error = std::max(
        max_sum_error, static_cast<float>(std::abs(output_sum - 1.0)));
  }
  if (max_error > 2.0e-5f || max_sum_error > 2.0e-5f)
    std::exit(EXIT_FAILURE);
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("softmax_f32[%dx%d]=PASS\n", rows, cols);
}

void test_transpose(cudaStream_t stream) {
  constexpr int rows = 37;
  constexpr int cols = 65;
  std::vector<float> input(rows * cols);
  std::iota(input.begin(), input.end(), -100.0f);
  float* device_input = copy_to_device(input, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(operatorlib::transpose_f32(device_input, device_output, rows, cols,
                                        stream));
  std::vector<float> output(input.size());
  CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output,
                             output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      if (output[static_cast<std::size_t>(col) * rows + row] !=
          input[static_cast<std::size_t>(row) * cols + col])
        std::exit(EXIT_FAILURE);
    }
  }
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("transpose_f32=PASS\n");
}

void test_rmsnorm_case(cudaStream_t stream, int tokens, int hidden) {
  constexpr float epsilon = 1.0e-5f;
  std::vector<float> input(tokens * hidden);
  std::vector<float> weight(hidden);
  std::mt19937 generator(11);
  std::uniform_real_distribution<float> input_distribution(-3.0f, 3.0f);
  std::uniform_real_distribution<float> weight_distribution(0.5f, 1.5f);
  for (float& value : input) value = input_distribution(generator);
  for (float& value : weight) value = weight_distribution(generator);
  float* device_input = copy_to_device(input, stream);
  float* device_weight = copy_to_device(weight, stream);
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(operatorlib::rmsnorm_f32(device_input, device_weight,
                                      device_output, tokens, hidden, epsilon,
                                      stream));
  std::vector<float> output(input.size());
  CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output,
                             output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float max_abs = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    const std::size_t offset = static_cast<std::size_t>(token) * hidden;
    double square_sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      const double value = input[offset + col];
      square_sum += value * value;
    }
    const double inverse_rms =
        1.0 / std::sqrt(square_sum / hidden + epsilon);
    for (int col = 0; col < hidden; ++col) {
      const float reference = static_cast<float>(
          input[offset + col] * inverse_rms * weight[col]);
      max_abs = std::max(max_abs,
                         std::abs(output[offset + col] - reference));
    }
  }
  if (max_abs > 2.0e-4f) std::exit(EXIT_FAILURE);
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_weight));
  CUDA_CHECK(cudaFree(device_output));
  std::printf("rmsnorm_f32[%dx%d]=PASS\n", tokens, hidden);
}

}  // namespace

int main() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  test_reduce(stream);
  test_softmax_case(stream, 19, 128, 5.0f);
  test_softmax_case(stream, 17, 1024, 5.0f);
  test_softmax_case(stream, 17, 1003, 1000.0f);
  test_transpose(stream);
  test_rmsnorm_case(stream, 32, 4096);
  test_rmsnorm_case(stream, 17, 1003);
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("core_ops_smoke=PASS\n");
  return EXIT_SUCCESS;
}
