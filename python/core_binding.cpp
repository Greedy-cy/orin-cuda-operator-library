#include "operatorlib/operators.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <limits>

namespace {

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_launch(cudaError_t error, const char* operation) {
  TORCH_CHECK(error == cudaSuccess, operation, " launch failed: ",
              cudaGetErrorString(error));
}

at::Tensor reduce_sum_cuda(const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "reduce_sum supports float32 input only");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty({}, input.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  check_launch(operatorlib::reduce_sum_f32(
                   input.data_ptr<float>(), output.data_ptr<float>(),
                   static_cast<std::size_t>(input.numel()), stream),
               "reduce_sum");
  return output;
}

at::Tensor softmax_cuda(const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "softmax supports float32 input only");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  const int64_t cols64 = input.size(-1);
  TORCH_CHECK(cols64 > 0, "last dimension must be non-empty");
  const int64_t rows64 = input.numel() / cols64;
  TORCH_CHECK(cols64 <= std::numeric_limits<int>::max(),
              "last dimension is outside the supported int range");
  TORCH_CHECK(rows64 > 0 && rows64 <= std::numeric_limits<int>::max(),
              "row count is outside the supported int range");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty_like(input);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  check_launch(operatorlib::softmax_f32(
                   input.data_ptr<float>(), output.data_ptr<float>(),
                   static_cast<int>(rows64), static_cast<int>(cols64), stream),
               "softmax");
  return output;
}

}  // namespace

TORCH_LIBRARY(operatorlib, m) {
  m.def("reduce_sum(Tensor input) -> Tensor");
  m.def("softmax(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(operatorlib, CUDA, m) {
  m.impl("reduce_sum", &reduce_sum_cuda);
  m.impl("softmax", &softmax_cuda);
}
