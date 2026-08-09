#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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

namespace {

struct Options {
  int rows = 4096;
  int cols = 1024;
  int warmup = 10;
  int repeat = 30;
  bool extreme = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto parse_int = [&](const char* prefix, int* value) {
      const std::string key(prefix);
      if (arg.rfind(key, 0) == 0) {
        *value = std::stoi(arg.substr(key.size()));
        return true;
      }
      return false;
    };
    if (parse_int("--rows=", &options.rows) ||
        parse_int("--cols=", &options.cols) ||
        parse_int("--warmup=", &options.warmup) ||
        parse_int("--repeat=", &options.repeat)) {
      continue;
    }
    if (arg == "--extreme") {
      options.extreme = true;
      continue;
    }
    if (arg == "--help") {
      std::cout << "Usage: ./softmax_v3 [--rows=N] [--cols=N] [--warmup=N] "
                   "[--repeat=N] [--extreme]\n";
      std::exit(EXIT_SUCCESS);
    }
    std::cerr << "Unknown argument: " << arg << '\n';
    std::exit(EXIT_FAILURE);
  }
  if (options.rows <= 0 || options.cols <= 0 || options.warmup < 0 ||
      options.repeat <= 0) {
    std::cerr << "rows, cols and repeat must be positive; warmup must be "
                 "non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_value = lane < (blockDim.x >> 5) ? shared[lane] : -FLT_MAX;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) shared[0] = block_value;
  }
  __syncthreads();
  return shared[0];
}

__device__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // Separate consumption of the max result from reuse of shared memory for sum.
  __syncthreads();
  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_value = lane < (blockDim.x >> 5) ? shared[lane] : 0.0f;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) shared[0] = block_value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void softmax_warp_kernel(const float* input, float* output, int rows,
                                    int cols) {
  constexpr int warps_per_block = 8;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= rows) return;

  const size_t row_offset = static_cast<size_t>(row) * cols;
  float maximum = -FLT_MAX;
  for (int col = lane; col < cols; col += warpSize)
    maximum = fmaxf(maximum, input[row_offset + col]);
  maximum = warp_reduce_max(maximum);
  maximum = __shfl_sync(0xffffffff, maximum, 0);

  float sum = 0.0f;
  for (int col = lane; col < cols; col += warpSize) {
    const float value = expf(input[row_offset + col] - maximum);
    output[row_offset + col] = value;
    sum += value;
  }
  sum = warp_reduce_sum(sum);
  const float inverse_sum = 1.0f / __shfl_sync(0xffffffff, sum, 0);
  for (int col = lane; col < cols; col += warpSize)
    output[row_offset + col] *= inverse_sum;
}

__global__ void softmax_block_kernel(const float* input, float* output,
                                     int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const size_t row_offset = static_cast<size_t>(row) * cols;
  const bool aligned = (cols & 3) == 0;

  float maximum = -FLT_MAX;
  if (aligned) {
    const float4* row_input =
        reinterpret_cast<const float4*>(input + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      const float4 value = row_input[i];
      maximum = fmaxf(maximum, value.x);
      maximum = fmaxf(maximum, value.y);
      maximum = fmaxf(maximum, value.z);
      maximum = fmaxf(maximum, value.w);
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x)
      maximum = fmaxf(maximum, input[row_offset + col]);
  }
  maximum = block_reduce_max(maximum, shared);

  float sum = 0.0f;
  if (aligned) {
    const float4* row_input =
        reinterpret_cast<const float4*>(input + row_offset);
    float4* row_output = reinterpret_cast<float4*>(output + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      const float4 source = row_input[i];
      const float4 value = {expf(source.x - maximum), expf(source.y - maximum),
                            expf(source.z - maximum), expf(source.w - maximum)};
      row_output[i] = value;
      sum += (value.x + value.y) + (value.z + value.w);
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x) {
      const float value = expf(input[row_offset + col] - maximum);
      output[row_offset + col] = value;
      sum += value;
    }
  }
  const float inverse_sum = 1.0f / block_reduce_sum(sum, shared);

  if (aligned) {
    float4* row_output = reinterpret_cast<float4*>(output + row_offset);
    for (int i = lane; i < cols / 4; i += blockDim.x) {
      float4 value = row_output[i];
      value.x *= inverse_sum;
      value.y *= inverse_sum;
      value.z *= inverse_sum;
      value.w *= inverse_sum;
      row_output[i] = value;
    }
  } else {
    for (int col = lane; col < cols; col += blockDim.x)
      output[row_offset + col] *= inverse_sum;
  }
}

void cpu_softmax(const std::vector<float>& input, std::vector<float>* output,
                 int rows, int cols) {
  output->resize(input.size());
  for (int row = 0; row < rows; ++row) {
    const size_t offset = static_cast<size_t>(row) * cols;
    float maximum = -FLT_MAX;
    for (int col = 0; col < cols; ++col)
      maximum = std::max(maximum, input[offset + col]);
    double sum = 0.0;
    for (int col = 0; col < cols; ++col)
      sum += std::exp(static_cast<double>(input[offset + col] - maximum));
    for (int col = 0; col < cols; ++col)
      (*output)[offset + col] = static_cast<float>(
          std::exp(static_cast<double>(input[offset + col] - maximum)) / sum);
  }
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

void launch_softmax(const float* input, float* output, int rows, int cols) {
  constexpr int threads = 256;
  if (cols <= 128) {
    constexpr int warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    softmax_warp_kernel<<<blocks, threads>>>(input, output, rows, cols);
  } else {
    softmax_block_kernel<<<rows, threads, (threads / 32) * sizeof(float)>>>(
        input, output, cols);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace


int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  const size_t count = static_cast<size_t>(options.rows) * options.cols;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<float> input(count);
  std::mt19937 generator(20260809);
  const float range = options.extreme ? 1000.0f : 5.0f;
  std::uniform_real_distribution<float> distribution(-range, range);
  for (float& value : input) value = distribution(generator);
  std::vector<float> reference;
  cpu_softmax(input, &reference, options.rows, options.cols);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < options.warmup; ++i)
    launch_softmax(device_input, device_output, options.rows, options.cols);
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_softmax(device_input, device_output, options.rows, options.cols);

  std::vector<float> result(count);
  CUDA_CHECK(cudaMemcpy(result.data(), device_output, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  float max_sum_error = 0.0f;
  for (int row = 0; row < options.rows; ++row) {
    const size_t offset = static_cast<size_t>(row) * options.cols;
    double sum = 0.0;
    for (int col = 0; col < options.cols; ++col) {
      const size_t index = offset + col;
      const float error = std::abs(result[index] - reference[index]);
      max_abs = std::max(max_abs, error);
      max_rel = std::max(
          max_rel, error / std::max(std::abs(reference[index]), 1.0e-7f));
      sum += result[index];
    }
    max_sum_error =
        std::max(max_sum_error, static_cast<float>(std::abs(sum - 1.0)));
  }
  const bool correct = max_abs <= 2.0e-5f && max_sum_error <= 2.0e-5f;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> timings;
  timings.reserve(options.repeat);
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_softmax(device_input, device_output, options.rows, options.cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    timings.push_back(ms);
  }
  const float median_ms = median(timings);
  const double io_gbps = static_cast<double>(count * sizeof(float) * 2) /
                         (median_ms * 1.0e6);

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v3_dispatch rows=" << options.rows
            << " cols=" << options.cols
            << " path=" << (options.cols <= 128 ? "warp" : "block")
            << " extreme=" << (options.extreme ? 1 : 0) << "\n"
            << "max_abs_error=" << max_abs << " max_rel_error=" << max_rel
            << " max_row_sum_error=" << max_sum_error << "\n"
            << "median_ms=" << median_ms << " effective_io_GBps=" << io_gbps
            << "\ncorrect=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
