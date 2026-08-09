#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
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
  size_t n = 1ULL << 24;
  int warmup = 10;
  int repeat = 30;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto parse_size = [&](const char* prefix, size_t* value) {
      const std::string key(prefix);
      if (arg.rfind(key, 0) == 0) {
        *value = std::stoull(arg.substr(key.size()));
        return true;
      }
      return false;
    };
    const auto parse_int = [&](const char* prefix, int* value) {
      const std::string key(prefix);
      if (arg.rfind(key, 0) == 0) {
        *value = std::stoi(arg.substr(key.size()));
        return true;
      }
      return false;
    };

    if (parse_size("--n=", &options.n) ||
        parse_int("--warmup=", &options.warmup) ||
        parse_int("--repeat=", &options.repeat)) {
      continue;
    }
    if (arg == "--help") {
      std::cout << "Usage: ./reduce_v1 [--n=N] [--warmup=N] [--repeat=N]\n";
      std::exit(EXIT_SUCCESS);
    }
    std::cerr << "Unknown argument: " << arg << '\n';
    std::exit(EXIT_FAILURE);
  }

  if (options.n == 0 || options.warmup < 0 || options.repeat <= 0) {
    std::cerr << "n and repeat must be positive; warmup must be non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__global__ void reduce_shared_kernel(const float* input, float* output,
                                     size_t n) {
  extern __shared__ float shared[];
  const unsigned int lane = threadIdx.x;
  const size_t index = blockIdx.x * blockDim.x + lane;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  float thread_sum = 0.0f;
  for (size_t i = index; i < n; i += stride) {
    thread_sum += input[i];
  }
  shared[lane] = thread_sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      shared[lane] += shared[lane + offset];
    }
    __syncthreads();
  }

  // v1 performs one global atomic operation per block instead of one per
  // participating thread.
  if (lane == 0) {
    atomicAdd(output, shared[0]);
  }
}

float cpu_reference(const std::vector<float>& input) {
  const double sum = std::accumulate(input.begin(), input.end(), 0.0);
  return static_cast<float>(sum);
}

float median(std::vector<float> values) {
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  const float upper = values[middle];
  std::nth_element(values.begin(), values.begin() + middle - 1,
                   values.begin() + middle);
  return 0.5f * (values[middle - 1] + upper);
}

void launch_reduce(const float* input, float* output, size_t n, int blocks,
                   int threads) {
  reduce_shared_kernel<<<blocks, threads, threads * sizeof(float)>>>(input,
                                                                     output, n);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<float> host_input(options.n);
  std::mt19937 generator(20260809);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (float& value : host_input) {
    value = distribution(generator);
  }
  const float reference = cpu_reference(host_input);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, options.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        options.n * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const size_t required_blocks = (options.n + threads - 1) / threads;
  const int blocks = static_cast<int>(
      std::min<size_t>(required_blocks, properties.multiProcessorCount * 8));

  for (int i = 0; i < options.warmup; ++i) {
    CUDA_CHECK(cudaMemset(device_output, 0, sizeof(float)));
    launch_reduce(device_input, device_output, options.n, blocks, threads);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemset(device_output, 0, sizeof(float)));
  launch_reduce(device_input, device_output, options.n, blocks, threads);
  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  const float absolute_error = std::abs(result - reference);
  const float relative_error = absolute_error / std::max(std::abs(reference), 1.0f);
  const float tolerance = 5.0e-4f * std::max(std::abs(reference), 1.0f);
  const bool correct = absolute_error <= tolerance;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> timings;
  timings.reserve(options.repeat);
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaMemsetAsync(device_output, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    launch_reduce(device_input, device_output, options.n, blocks, threads);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    timings.push_back(elapsed_ms);
  }

  const float median_ms = median(timings);
  const double effective_gbps =
      static_cast<double>(options.n * sizeof(float)) / (median_ms * 1.0e6);

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v1_shared n=" << options.n << " blocks=" << blocks
            << " threads=" << threads << "\n"
            << "reference=" << reference << " result=" << result
            << " abs_error=" << absolute_error
            << " rel_error=" << relative_error << "\n"
            << "median_ms=" << median_ms
            << " effective_bandwidth_GBps=" << effective_gbps << "\n"
            << "correct=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
