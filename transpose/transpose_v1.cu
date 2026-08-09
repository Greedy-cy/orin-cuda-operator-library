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

constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr double kPeakMemoryGBps = 102.4;

struct Options {
  int rows = 4096;
  int cols = 4096;
  int warmup = 10;
  int repeat = 30;
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
    if (arg == "--help") {
      std::cout << "Usage: ./transpose_v1 [--rows=N] [--cols=N] [--warmup=N] "
                   "[--repeat=N]\n";
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

__global__ void transpose_tiled_kernel(const float* input, float* output,
                                       int rows, int cols) {
  __shared__ float tile[kTile][kTile];

  const int input_col = blockIdx.x * kTile + threadIdx.x;
  const int input_row = blockIdx.y * kTile + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    if (input_col < cols && input_row + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<size_t>(input_row + offset) * cols + input_col];
    }
  }
  __syncthreads();

  const int output_col = blockIdx.y * kTile + threadIdx.x;
  const int output_row = blockIdx.x * kTile + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    if (output_col < rows && output_row + offset < cols) {
      output[static_cast<size_t>(output_row + offset) * rows + output_col] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

void launch_transpose(const float* input, float* output, int rows, int cols) {
  const dim3 block(kTile, kBlockRows);
  const dim3 grid((cols + kTile - 1) / kTile,
                  (rows + kTile - 1) / kTile);
  transpose_tiled_kernel<<<grid, block>>>(input, output, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void cpu_transpose(const std::vector<float>& input, std::vector<float>* output,
                   int rows, int cols) {
  output->resize(input.size());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      (*output)[static_cast<size_t>(col) * rows + row] =
          input[static_cast<size_t>(row) * cols + col];
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
  const size_t count = static_cast<size_t>(options.rows) * options.cols;
  const size_t bytes = count * sizeof(float);
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::vector<float> input(count);
  std::mt19937 generator(20260809);
  std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
  for (float& value : input) value = distribution(generator);
  std::vector<float> reference;
  cpu_transpose(input, &reference, options.rows, options.cols);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < options.warmup; ++i)
    launch_transpose(device_input, device_output, options.rows, options.cols);
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_transpose(device_input, device_output, options.rows, options.cols);

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
  const bool correct = max_absolute_error == 0.0f;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> timings;
  timings.reserve(options.repeat);
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_transpose(device_input, device_output, options.rows, options.cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    timings.push_back(milliseconds);
  }
  const float median_ms = median(timings);
  const double effective_gbps =
      static_cast<double>(2 * bytes) / (median_ms * 1.0e6);
  const double peak_percent = 100.0 * effective_gbps / kPeakMemoryGBps;

  std::cout << std::fixed << std::setprecision(6)
            << "device=" << properties.name << " sm_count="
            << properties.multiProcessorCount << "\n"
            << "version=v1_tiled rows=" << options.rows
            << " cols=" << options.cols << " tile=32x32 block=32x8 padding=0\n"
            << "max_abs_error=" << max_absolute_error
            << " max_rel_error=" << max_relative_error << "\n"
            << "median_ms=" << median_ms
            << " effective_bandwidth_GBps=" << effective_gbps
            << " peak_bandwidth_GBps=" << kPeakMemoryGBps
            << " peak_percent=" << peak_percent << "\n"
            << "correct=" << (correct ? "PASS" : "FAIL") << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
