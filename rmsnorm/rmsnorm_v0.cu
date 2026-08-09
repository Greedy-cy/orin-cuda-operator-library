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
      std::cout << "Usage: ./rmsnorm_v0 [--tokens=N] [--hidden=N] "
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

__global__ void square_kernel(const float* input, float* squared, size_t count) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const float value = input[index];
    squared[index] = value * value;
  }
}

__global__ void row_inverse_rms_kernel(const float* squared, float* inverse_rms,
                                       int hidden, float epsilon) {
  extern __shared__ float shared[];
  const int token = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += squared[static_cast<size_t>(token) * hidden + col];
  }
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    inverse_rms[token] = rsqrtf(shared[0] / hidden + epsilon);
  }
}

__global__ void normalize_kernel(const float* input, const float* inverse_rms,
                                 float* normalized, int hidden, size_t count) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int token = static_cast<int>(index / hidden);
    normalized[index] = input[index] * inverse_rms[token];
  }
}

__global__ void weight_kernel(const float* normalized, const float* weight,
                              float* output, int hidden, size_t count) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    output[index] = normalized[index] * weight[index % hidden];
  }
}

struct DeviceBuffers {
  float* input = nullptr;
  float* weight = nullptr;
  float* squared = nullptr;
  float* inverse_rms = nullptr;
  float* normalized = nullptr;
  float* output = nullptr;
};

void launch_rmsnorm(const DeviceBuffers& buffers, int tokens, int hidden,
                    float epsilon) {
  constexpr int threads = 256;
  const size_t count = static_cast<size_t>(tokens) * hidden;
  const int element_blocks =
      std::min<int>((count + threads - 1) / threads, 4096);
  square_kernel<<<element_blocks, threads>>>(buffers.input, buffers.squared,
                                              count);
  row_inverse_rms_kernel<<<tokens, threads, threads * sizeof(float)>>>(
      buffers.squared, buffers.inverse_rms, hidden, epsilon);
  normalize_kernel<<<element_blocks, threads>>>(
      buffers.input, buffers.inverse_rms, buffers.normalized, hidden, count);
  weight_kernel<<<element_blocks, threads>>>(buffers.normalized, buffers.weight,
                                              buffers.output, hidden, count);
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

  DeviceBuffers buffers;
  CUDA_CHECK(cudaMalloc(&buffers.input, bytes));
  CUDA_CHECK(cudaMalloc(&buffers.weight, options.hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.squared, bytes));
  CUDA_CHECK(cudaMalloc(&buffers.inverse_rms,
                        options.tokens * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.normalized, bytes));
  CUDA_CHECK(cudaMalloc(&buffers.output, bytes));
  CUDA_CHECK(cudaMemcpy(buffers.input, input.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.weight, weight.data(),
                        options.hidden * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < options.warmup; ++i) {
    launch_rmsnorm(buffers, options.tokens, options.hidden, options.epsilon);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_rmsnorm(buffers, options.tokens, options.hidden, options.epsilon);

  std::vector<float> result(count);
  CUDA_CHECK(cudaMemcpy(result.data(), buffers.output, bytes,
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
    launch_rmsnorm(buffers, options.tokens, options.hidden, options.epsilon);
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
            << "version=v0_four_kernels tokens=" << options.tokens
            << " hidden=" << options.hidden << " epsilon=" << options.epsilon
            << " launches=4\n"
            << "max_abs_error=" << max_absolute_error
            << " max_rel_error=" << max_relative_error << "\n"
            << "median_ms=" << median_ms
            << " ideal_effective_bandwidth_GBps=" << effective_gbps << "\n"
            << "correct=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(buffers.input));
  CUDA_CHECK(cudaFree(buffers.weight));
  CUDA_CHECK(cudaFree(buffers.squared));
  CUDA_CHECK(cudaFree(buffers.inverse_rms));
  CUDA_CHECK(cudaFree(buffers.normalized));
  CUDA_CHECK(cudaFree(buffers.output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
