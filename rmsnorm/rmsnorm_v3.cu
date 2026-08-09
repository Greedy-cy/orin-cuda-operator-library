#include <cuda_runtime.h>

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

namespace {

struct Options {
  int tokens = 512;
  int hidden = 4096;
  int warmup = 10;
  int repeat = 30;
  float epsilon = 1.0e-5f;
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
    if (parse_int("--tokens=", &options.tokens) ||
        parse_int("--hidden=", &options.hidden) ||
        parse_int("--warmup=", &options.warmup) ||
        parse_int("--repeat=", &options.repeat)) {
      continue;
    }
    if (arg.rfind("--epsilon=", 0) == 0) {
      options.epsilon = std::stof(arg.substr(std::string("--epsilon=").size()));
      continue;
    }
    if (arg == "--help") {
      std::cout << "Usage: ./rmsnorm_v3 [--tokens=N] [--hidden=N] "
                   "[--epsilon=F] [--warmup=N] [--repeat=N]\n";
      std::exit(EXIT_SUCCESS);
    }
    std::cerr << "Unknown argument: " << arg << '\n';
    std::exit(EXIT_FAILURE);
  }
  if (options.tokens <= 0 || options.hidden <= 0 || options.warmup < 0 ||
      options.repeat <= 0 || options.epsilon <= 0.0f) {
    std::cerr << "tokens, hidden, repeat and epsilon must be positive; warmup "
                 "must be non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ float block_reduce_sum(float value, float* warp_results) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) warp_results[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x >> 5) ? warp_results[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) warp_results[0] = block_sum;
  }
  __syncthreads();
  return warp_results[0];
}

__global__ void rmsnorm_float4_kernel(const float4* input, const float4* weight,
                                      float4* output, int vectors_per_row,
                                      int hidden, float epsilon) {
  extern __shared__ float warp_results[];
  const int token = blockIdx.x;
  const size_t row_offset = static_cast<size_t>(token) * vectors_per_row;

  float square_sum = 0.0f;
  for (int index = threadIdx.x; index < vectors_per_row; index += blockDim.x) {
    const float4 value = input[row_offset + index];
    square_sum += (value.x * value.x + value.y * value.y) +
                  (value.z * value.z + value.w * value.w);
  }
  const float inverse_rms =
      rsqrtf(block_reduce_sum(square_sum, warp_results) / hidden + epsilon);

  for (int index = threadIdx.x; index < vectors_per_row; index += blockDim.x) {
    const float4 value = input[row_offset + index];
    const float4 scale = weight[index];
    output[row_offset + index] =
        make_float4(value.x * inverse_rms * scale.x,
                    value.y * inverse_rms * scale.y,
                    value.z * inverse_rms * scale.z,
                    value.w * inverse_rms * scale.w);
  }
}

__global__ void rmsnorm_scalar_fallback_kernel(
    const float* input, const float* weight, float* output, int hidden,
    float epsilon) {
  extern __shared__ float warp_results[];
  const int token = blockIdx.x;
  const size_t row_offset = static_cast<size_t>(token) * hidden;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = input[row_offset + col];
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_reduce_sum(square_sum, warp_results) / hidden + epsilon);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    output[row_offset + col] = input[row_offset + col] * inverse_rms * weight[col];
  }
}

void launch_rmsnorm(const float* input, const float* weight, float* output,
                    int tokens, int hidden, float epsilon) {
  constexpr int threads = 256;
  constexpr int shared_bytes = (threads / 32) * sizeof(float);
  if ((hidden & 3) == 0) {
    rmsnorm_float4_kernel<<<tokens, threads, shared_bytes>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(weight),
        reinterpret_cast<float4*>(output), hidden / 4, hidden, epsilon);
  } else {
    rmsnorm_scalar_fallback_kernel<<<tokens, threads, shared_bytes>>>(
        input, weight, output, hidden, epsilon);
  }
  CUDA_CHECK(cudaGetLastError());
}

void cpu_rmsnorm(const std::vector<float>& input,
                 const std::vector<float>& weight, std::vector<float>* output,
                 int tokens, int hidden, float epsilon) {
  output->resize(input.size());
  for (int token = 0; token < tokens; ++token) {
    const size_t offset = static_cast<size_t>(token) * hidden;
    double square_sum = 0.0;
    for (int col = 0; col < hidden; ++col) {
      const double value = input[offset + col];
      square_sum += value * value;
    }
    const double inverse_rms =
        1.0 / std::sqrt(square_sum / hidden + static_cast<double>(epsilon));
    for (int col = 0; col < hidden; ++col) {
      (*output)[offset + col] = static_cast<float>(
          input[offset + col] * inverse_rms * weight[col]);
    }
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

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  const size_t count = static_cast<size_t>(options.tokens) * options.hidden;
  const size_t bytes = count * sizeof(float);
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::mt19937 generator(20260809);
  std::uniform_real_distribution<float> input_distribution(-3.0f, 3.0f);
  std::uniform_real_distribution<float> weight_distribution(0.5f, 1.5f);
  std::vector<float> input(count);
  std::vector<float> weight(options.hidden);
  for (float& value : input) value = input_distribution(generator);
  for (float& value : weight) value = weight_distribution(generator);
  std::vector<float> reference;
  cpu_rmsnorm(input, weight, &reference, options.tokens, options.hidden,
              options.epsilon);

  float* device_input = nullptr;
  float* device_weight = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_weight, options.hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_weight, weight.data(),
                        options.hidden * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < options.warmup; ++i)
    launch_rmsnorm(device_input, device_weight, device_output, options.tokens,
                   options.hidden, options.epsilon);
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_rmsnorm(device_input, device_weight, device_output, options.tokens,
                 options.hidden, options.epsilon);

  std::vector<float> result(count);
  CUDA_CHECK(cudaMemcpy(result.data(), device_output, bytes,
                        cudaMemcpyDeviceToHost));
  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float absolute_error = std::abs(result[i] - reference[i]);
    const float relative_error =
        absolute_error / std::max(std::abs(reference[i]), 1.0e-7f);
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, relative_error);
  }
  const bool correct = max_absolute_error <= 2.0e-4f &&
                       max_relative_error <= 2.0e-4f;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> timings;
  timings.reserve(options.repeat);
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_rmsnorm(device_input, device_weight, device_output, options.tokens,
                   options.hidden, options.epsilon);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    timings.push_back(milliseconds);
  }
  const float median_ms = median(timings);
  const double ideal_bytes =
      static_cast<double>((2 * count + options.hidden) * sizeof(float));
  const double effective_gbps = ideal_bytes / (median_ms * 1.0e6);

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v3_float4 tokens=" << options.tokens
            << " hidden=" << options.hidden
            << " path=" << (((options.hidden & 3) == 0) ? "float4" : "scalar")
            << " epsilon=" << options.epsilon << " launches=1\n"
            << "max_abs_error=" << max_absolute_error
            << " max_rel_error=" << max_relative_error << "\n"
            << "median_ms=" << median_ms
            << " ideal_effective_bandwidth_GBps=" << effective_gbps << "\n"
            << "correct=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_weight));
  CUDA_CHECK(cudaFree(device_output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
