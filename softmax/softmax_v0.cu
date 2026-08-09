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
      std::cout << "Usage: ./softmax_v0 [--rows=N] [--cols=N] [--warmup=N] "
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

__global__ void softmax_shared_kernel(const float* input, float* output,
                                      int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const size_t row_offset = static_cast<size_t>(row) * cols;

  float thread_max = -FLT_MAX;
  for (int col = lane; col < cols; col += blockDim.x) {
    thread_max = fmaxf(thread_max, input[row_offset + col]);
  }
  shared[lane] = thread_max;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      shared[lane] = fmaxf(shared[lane], shared[lane + offset]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float thread_sum = 0.0f;
  for (int col = lane; col < cols; col += blockDim.x) {
    const float value = expf(input[row_offset + col] - row_max);
    output[row_offset + col] = value;
    thread_sum += value;
  }
  // All warps must consume shared[0] (row_max) before shared memory is reused
  // for the sum reduction. Without this barrier racecheck reports a WAR race.
  __syncthreads();
  shared[lane] = thread_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      shared[lane] += shared[lane + offset];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0f / shared[0];

  // v0 stores exp(x-max) to global memory, then reads it back for normalize.
  for (int col = lane; col < cols; col += blockDim.x) {
    output[row_offset + col] *= inverse_sum;
  }
}

void cpu_softmax(const std::vector<float>& input, std::vector<float>* output,
                 int rows, int cols) {
  output->resize(input.size());
  for (int row = 0; row < rows; ++row) {
    const size_t offset = static_cast<size_t>(row) * cols;
    float maximum = -FLT_MAX;
    for (int col = 0; col < cols; ++col) {
      maximum = std::max(maximum, input[offset + col]);
    }
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      sum += std::exp(static_cast<double>(input[offset + col] - maximum));
    }
    for (int col = 0; col < cols; ++col) {
      (*output)[offset + col] = static_cast<float>(
          std::exp(static_cast<double>(input[offset + col] - maximum)) / sum);
    }
  }
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

void launch_softmax(const float* input, float* output, int rows, int cols,
                    int threads) {
  softmax_shared_kernel<<<rows, threads, threads * sizeof(float)>>>(input, output,
                                                                    cols);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  const size_t element_count =
      static_cast<size_t>(options.rows) * options.cols;

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<float> host_input(element_count);
  std::mt19937 generator(20260809);
  const float range = options.extreme ? 1000.0f : 5.0f;
  std::uniform_real_distribution<float> distribution(-range, range);
  for (float& value : host_input) {
    value = distribution(generator);
  }

  std::vector<float> reference;
  cpu_softmax(host_input, &reference, options.rows, options.cols);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, element_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, element_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        element_count * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    launch_softmax(device_input, device_output, options.rows, options.cols,
                   threads);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_softmax(device_input, device_output, options.rows, options.cols, threads);
  std::vector<float> result(element_count);
  CUDA_CHECK(cudaMemcpy(result.data(), device_output,
                        element_count * sizeof(float), cudaMemcpyDeviceToHost));

  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;
  float max_row_sum_error = 0.0f;
  for (int row = 0; row < options.rows; ++row) {
    const size_t offset = static_cast<size_t>(row) * options.cols;
    double row_sum = 0.0;
    for (int col = 0; col < options.cols; ++col) {
      const size_t index = offset + col;
      const float absolute_error = std::abs(result[index] - reference[index]);
      const float relative_error =
          absolute_error / std::max(std::abs(reference[index]), 1.0e-7f);
      max_absolute_error = std::max(max_absolute_error, absolute_error);
      max_relative_error = std::max(max_relative_error, relative_error);
      row_sum += result[index];
    }
    max_row_sum_error =
        std::max(max_row_sum_error, static_cast<float>(std::abs(row_sum - 1.0)));
  }
  const bool correct = max_absolute_error <= 2.0e-5f &&
                       max_row_sum_error <= 2.0e-5f;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> timings;
  timings.reserve(options.repeat);
  for (int iteration = 0; iteration < options.repeat; ++iteration) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_softmax(device_input, device_output, options.rows, options.cols,
                   threads);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    timings.push_back(elapsed_ms);
  }

  const float median_ms = median(timings);
  const double effective_io_gbps =
      static_cast<double>(element_count * sizeof(float) * 2) /
      (median_ms * 1.0e6);

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v0_shared rows=" << options.rows
            << " cols=" << options.cols << " threads=" << threads
            << " extreme=" << (options.extreme ? 1 : 0) << "\n"
            << "max_abs_error=" << max_absolute_error
            << " max_rel_error=" << max_relative_error
            << " max_row_sum_error=" << max_row_sum_error << "\n"
            << "median_ms=" << median_ms
            << " effective_io_GBps=" << effective_io_gbps << "\n"
            << "correct=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
