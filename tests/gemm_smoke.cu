#include "operatorlib/operators.h"

#include <cublas_v2.h>

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
      std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(error));                                  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (call);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "%s:%d cuBLAS status=%d\n", __FILE__, __LINE__,   \
                   static_cast<int>(status));                                   \
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

void test_f32(cublasHandle_t handle, cudaStream_t stream, int m, int n, int k) {
  std::mt19937 generator(17 + m);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<float> a(static_cast<std::size_t>(m) * k);
  std::vector<float> b(static_cast<std::size_t>(k) * n);
  for (float& value : a) value = distribution(generator);
  for (float& value : b) value = distribution(generator);
  float* device_a = copy_to_device(a, stream);
  float* device_b = copy_to_device(b, stream);
  float* custom = nullptr;
  float* reference = nullptr;
  CUDA_CHECK(cudaMalloc(&custom, static_cast<std::size_t>(m) * n *
                                     sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference, static_cast<std::size_t>(m) * n *
                                        sizeof(float)));
  CUDA_CHECK(operatorlib::gemm_f32(device_a, device_b, custom, m, n, k,
                                   stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                           device_b, n, device_a, k, &beta, reference, n));
  std::vector<float> host_custom(static_cast<std::size_t>(m) * n);
  std::vector<float> host_reference(host_custom.size());
  CUDA_CHECK(cudaMemcpyAsync(host_custom.data(), custom,
                             host_custom.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(host_reference.data(), reference,
                             host_reference.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::size_t violations = 0;
  for (std::size_t i = 0; i < host_custom.size(); ++i) {
    const float error = std::abs(host_custom[i] - host_reference[i]);
    if (error > 1.0e-3f + 1.0e-3f * std::abs(host_reference[i])) ++violations;
  }
  if (violations != 0) std::exit(EXIT_FAILURE);
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(custom));
  CUDA_CHECK(cudaFree(reference));
  std::printf("gemm_f32[%d,%d,%d]=PASS\n", m, n, k);
}

void test_f16(cublasHandle_t handle, cudaStream_t stream, int m, int n, int k) {
  std::mt19937 generator(23 + m);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<__half> a(static_cast<std::size_t>(m) * k);
  std::vector<__half> b(static_cast<std::size_t>(k) * n);
  for (__half& value : a) value = __float2half(distribution(generator));
  for (__half& value : b) value = __float2half(distribution(generator));
  __half* device_a = copy_to_device(a, stream);
  __half* device_b = copy_to_device(b, stream);
  float* custom = nullptr;
  float* reference = nullptr;
  CUDA_CHECK(cudaMalloc(&custom, static_cast<std::size_t>(m) * n *
                                     sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference, static_cast<std::size_t>(m) * n *
                                        sizeof(float)));
  CUDA_CHECK(operatorlib::gemm_f16(device_a, device_b, custom, m, n, k,
                                   stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, CUDA_R_16F,
      n, device_a, CUDA_R_16F, k, &beta, reference, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  std::vector<float> host_custom(static_cast<std::size_t>(m) * n);
  std::vector<float> host_reference(host_custom.size());
  CUDA_CHECK(cudaMemcpyAsync(host_custom.data(), custom,
                             host_custom.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(host_reference.data(), reference,
                             host_reference.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::size_t violations = 0;
  for (std::size_t i = 0; i < host_custom.size(); ++i) {
    const float error = std::abs(host_custom[i] - host_reference[i]);
    if (error > 2.0e-2f + 2.0e-2f * std::abs(host_reference[i])) ++violations;
  }
  if (violations != 0) std::exit(EXIT_FAILURE);
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(custom));
  CUDA_CHECK(cudaFree(reference));
  std::printf("gemm_f16[%d,%d,%d]=PASS\n", m, n, k);
}

}  // namespace

int main() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  test_f32(handle, stream, 1, 256, 64);
  test_f32(handle, stream, 16, 128, 64);
  test_f32(handle, stream, 128, 64, 32);
  test_f32(handle, stream, 37, 65, 71);
  test_f16(handle, stream, 128, 128, 32);
  test_f16(handle, stream, 16, 128, 32);
  test_f16(handle, stream, 32, 32, 32);
  test_f16(handle, stream, 37, 65, 71);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("gemm_smoke=PASS\n");
  return EXIT_SUCCESS;
}
