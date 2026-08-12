#include "operatorlib/operators.h"

namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

__global__ void transpose_padded_kernel(const float* input, float* output,
                                        int rows, int cols) {
  __shared__ float tile[kTile][kTile + 1];
  const int input_col = blockIdx.x * kTile + threadIdx.x;
  const int input_row = blockIdx.y * kTile + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    if (input_col < cols && input_row + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<std::size_t>(input_row + offset) * cols +
                input_col];
    }
  }
  __syncthreads();

  const int output_col = blockIdx.y * kTile + threadIdx.x;
  const int output_row = blockIdx.x * kTile + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    if (output_col < rows && output_row + offset < cols) {
      output[static_cast<std::size_t>(output_row + offset) * rows +
             output_col] = tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

}  // namespace

namespace operatorlib {

cudaError_t transpose_f32(const float* input, float* output, int rows, int cols,
                          cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const dim3 block(kTile, kBlockRows);
  const dim3 grid((cols + kTile - 1) / kTile,
                  (rows + kTile - 1) / kTile);
  transpose_padded_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  return cudaGetLastError();
}

}  // namespace operatorlib
