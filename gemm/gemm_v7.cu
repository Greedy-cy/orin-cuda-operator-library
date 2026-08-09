#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "     \
                << cudaGetErrorString(error) << '\n';                            \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    const cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error at " << __FILE__ << ':' << __LINE__            \
                << ": status=" << static_cast<int>(status) << '\n';              \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                            \
  } while (0)

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 5;
  int repeat = 15;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    const auto parse_int = [&](const char* prefix, int* value) {
      const std::string key(prefix);
      if (argument.rfind(key, 0) == 0) {
        *value = std::stoi(argument.substr(key.size()));
        return true;
      }
      return false;
    };
    if (parse_int("--m=", &options.m) || parse_int("--n=", &options.n) ||
        parse_int("--k=", &options.k) ||
        parse_int("--warmup=", &options.warmup) ||
        parse_int("--repeat=", &options.repeat))
      continue;
    if (argument == "--help") {
      std::cout << "Usage: ./gemm_v7 [--m=N] [--n=N] [--k=N] [--warmup=N] "
                   "[--repeat=N]\n";
      std::exit(EXIT_SUCCESS);
    }
    std::cerr << "Unknown argument: " << argument << '\n';
    std::exit(EXIT_FAILURE);
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.warmup < 0 || options.repeat <= 0) {
    std::cerr << "m, n, k and repeat must be positive; warmup must be "
                 "non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__global__ void wmma_baseline_kernel(const __half* __restrict__ a,
                                     const __half* __restrict__ b,
                                     float* __restrict__ c, int m, int n,
                                     int k) {
  const int warp_in_block = threadIdx.x / warpSize;
  const int warp_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int tile_columns = n / kWmmaN;
  const int tile_count = (m / kWmmaM) * tile_columns;
  if (warp_index >= tile_count) return;

  const int output_row = (warp_index / tile_columns) * kWmmaM;
  const int output_col = (warp_index % tile_columns) * kWmmaN;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      fragment_a;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                         __half, nvcuda::wmma::row_major>
      fragment_b;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  for (int tile_begin = 0; tile_begin < k; tile_begin += kWmmaK) {
    nvcuda::wmma::load_matrix_sync(
        fragment_a, &a[static_cast<size_t>(output_row) * k + tile_begin], k);
    nvcuda::wmma::load_matrix_sync(
        fragment_b, &b[static_cast<size_t>(tile_begin) * n + output_col], n);
    nvcuda::wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
  }
  nvcuda::wmma::store_matrix_sync(
      &c[static_cast<size_t>(output_row) * n + output_col], accumulator, n,
      nvcuda::wmma::mem_row_major);
}

__global__ void half_scalar_fallback_kernel(const __half* __restrict__ a,
                                            const __half* __restrict__ b,
                                            float* __restrict__ c, int m, int n,
                                            int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum = fmaf(__half2float(a[static_cast<size_t>(row) * k + inner]),
               __half2float(b[static_cast<size_t>(inner) * n + col]), sum);
  }
  c[static_cast<size_t>(row) * n + col] = sum;
}

void launch_custom(const __half* a, const __half* b, float* c, int m, int n,
                   int k) {
  if (m % kWmmaM == 0 && n % kWmmaN == 0 && k % kWmmaK == 0) {
    const int tile_count = (m / kWmmaM) * (n / kWmmaN);
    const int blocks = (tile_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    wmma_baseline_kernel<<<blocks, kThreads>>>(a, b, c, m, n, k);
  } else {
    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    half_scalar_fallback_kernel<<<grid, block>>>(a, b, c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
}

void launch_cublas(cublasHandle_t handle, const __half* a, const __half* b,
                   float* c, int m, int n, int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, CUDA_R_16F, n, a,
      CUDA_R_16F, k, &beta, c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

float median(std::vector<float> values) {
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2 == 1) return values[middle];
  const float upper = values[middle];
  std::nth_element(values.begin(), values.begin() + middle - 1,
                   values.begin() + middle);
  return 0.5f * (values[middle - 1] + upper);
}

template <typename Launch>
float measure_median(Launch launch, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> timings;
  timings.reserve(repeat);
  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    timings.push_back(milliseconds);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return median(timings);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  const size_t a_count = static_cast<size_t>(options.m) * options.k;
  const size_t b_count = static_cast<size_t>(options.k) * options.n;
  const size_t c_count = static_cast<size_t>(options.m) * options.n;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::mt19937 generator(20260809);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<__half> host_a(a_count);
  std::vector<__half> host_b(b_count);
  for (__half& value : host_a) value = __float2half(distribution(generator));
  for (__half& value : host_b) value = __float2half(distribution(generator));

  __half* device_a = nullptr;
  __half* device_b = nullptr;
  float* device_custom = nullptr;
  float* device_reference = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, a_count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&device_b, b_count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&device_custom, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_reference, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_count * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_count * sizeof(__half),
                        cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  launch_custom(device_a, device_b, device_custom, options.m, options.n,
                options.k);
  launch_cublas(handle, device_a, device_b, device_reference, options.m,
                options.n, options.k);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> custom(c_count);
  std::vector<float> reference(c_count);
  CUDA_CHECK(cudaMemcpy(custom.data(), device_custom, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference.data(), device_reference,
                        c_count * sizeof(float), cudaMemcpyDeviceToHost));
  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;
  size_t violations = 0;
  for (size_t index = 0; index < c_count; ++index) {
    const float absolute_error = std::abs(custom[index] - reference[index]);
    const float relative_error =
        absolute_error / std::max(std::abs(reference[index]), 1.0e-5f);
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, relative_error);
    if (absolute_error > 2.0e-2f + 2.0e-2f * std::abs(reference[index]))
      ++violations;
  }
  const bool correct = violations == 0;

  const float custom_ms = measure_median(
      [&] {
        launch_custom(device_a, device_b, device_custom, options.m, options.n,
                      options.k);
      },
      options.warmup, options.repeat);
  const float cublas_ms = measure_median(
      [&] {
        launch_cublas(handle, device_a, device_b, device_reference, options.m,
                      options.n, options.k);
      },
      options.warmup, options.repeat);
  const double operations =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  const double custom_gflops = operations / (custom_ms * 1.0e6);
  const double cublas_gflops = operations / (cublas_ms * 1.0e6);
  const bool aligned = options.m % 16 == 0 && options.n % 16 == 0 &&
                       options.k % 16 == 0;

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v7_wmma_baseline m=" << options.m
            << " n=" << options.n << " k=" << options.k
            << " input=fp16 accumulation=fp32 output=fp32 path="
            << (aligned ? "wmma" : "scalar_fallback") << "\n"
            << "max_abs_error=" << max_absolute_error
            << " max_rel_error=" << max_relative_error
            << " violations=" << violations << "\n"
            << "custom_median_ms=" << custom_ms
            << " custom_GFLOPs=" << custom_gflops << "\n"
            << "cublas_median_ms=" << cublas_ms
            << " cublas_GFLOPs=" << cublas_gflops << "\n"
            << "cublas_ratio_percent=" << (100.0 * custom_gflops / cublas_gflops)
            << "\ncorrect=" << (correct ? "PASS" : "FAIL") << '\n';

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_custom));
  CUDA_CHECK(cudaFree(device_reference));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
