#include "operatorlib/operators.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

namespace {

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device_dtype(const at::Tensor& first,
                             const at::Tensor& second,
                             const char* first_name,
                             const char* second_name) {
  TORCH_CHECK(first.device() == second.device(), first_name, " and ",
              second_name, " must be on the same CUDA device");
  TORCH_CHECK(first.scalar_type() == second.scalar_type(), first_name, " and ",
              second_name, " must have the same dtype");
}

int checked_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name,
              " must be in the positive int32 range");
  return static_cast<int>(value);
}

void check_launch(cudaError_t error, const char* operation) {
  TORCH_CHECK(error == cudaSuccess, operation, " launch failed: ",
              cudaGetErrorString(error));
}

cudaStream_t current_stream(const at::Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device());
}

at::Tensor reduce_sum_cuda(const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "reduce_sum supports float32 input only");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty({}, input.options());
  check_launch(operatorlib::reduce_sum_f32(
                   input.data_ptr<float>(), output.data_ptr<float>(),
                   static_cast<std::size_t>(input.numel()),
                   current_stream(input)),
               "reduce_sum");
  return output;
}

at::Tensor softmax_cuda(const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "softmax supports float32 input only");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  const int cols = checked_positive_int(input.size(-1), "last dimension");
  const int rows =
      checked_positive_int(input.numel() / input.size(-1), "row count");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty_like(input);
  check_launch(operatorlib::softmax_f32(
                   input.data_ptr<float>(), output.data_ptr<float>(), rows,
                   cols, current_stream(input)),
               "softmax");
  return output;
}

at::Tensor transpose_cuda(const at::Tensor& input) {
  check_cuda_contiguous(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "transpose supports float32 input only");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D matrix");
  const int rows = checked_positive_int(input.size(0), "rows");
  const int cols = checked_positive_int(input.size(1), "cols");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty({cols, rows}, input.options());
  check_launch(operatorlib::transpose_f32(
                   input.data_ptr<float>(), output.data_ptr<float>(), rows,
                   cols, current_stream(input)),
               "transpose");
  return output;
}

at::Tensor rmsnorm_cuda(const at::Tensor& input, const at::Tensor& weight,
                        double epsilon) {
  check_cuda_contiguous(input, "input");
  check_cuda_contiguous(weight, "weight");
  check_same_device_dtype(input, weight, "input", "weight");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "rmsnorm supports float32 input only");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "epsilon must be finite and positive");
  const int hidden = checked_positive_int(input.size(-1), "hidden size");
  TORCH_CHECK(weight.numel() == hidden,
              "weight length must equal the input hidden size");
  const int tokens =
      checked_positive_int(input.numel() / input.size(-1), "token count");
  const c10::cuda::CUDAGuard device_guard(input.device());
  at::Tensor output = at::empty_like(input);
  check_launch(operatorlib::rmsnorm_f32(
                   input.data_ptr<float>(), weight.data_ptr<float>(),
                   output.data_ptr<float>(), tokens, hidden,
                   static_cast<float>(epsilon), current_stream(input)),
               "rmsnorm");
  return output;
}

at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b) {
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_same_device_dtype(a, b, "a", "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D matrices");
  TORCH_CHECK(a.size(1) == b.size(0),
              "a.size(1) must equal b.size(0)");
  TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf,
              "gemm supports float32 or float16 inputs only");
  const int m = checked_positive_int(a.size(0), "M");
  const int k = checked_positive_int(a.size(1), "K");
  const int n = checked_positive_int(b.size(1), "N");
  const c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor output =
      at::empty({m, n}, a.options().dtype(at::ScalarType::Float));
  cudaError_t error;
  if (a.scalar_type() == at::kFloat) {
    error = operatorlib::gemm_f32(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), m,
        n, k, current_stream(a));
  } else {
    error = operatorlib::gemm_f16(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        output.data_ptr<float>(), m, n, k, current_stream(a));
  }
  check_launch(error, "gemm");
  return output;
}

at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, bool causal) {
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_same_device_dtype(q, k, "q", "k");
  check_same_device_dtype(q, v, "q", "v");
  TORCH_CHECK(q.dim() == 4, "q, k and v must use [B,H,S,D] layout");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "q, k and v must have identical shapes");
  TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf,
              "attention supports float32 or float16 inputs only");
  const int batch = checked_positive_int(q.size(0), "batch");
  const int heads = checked_positive_int(q.size(1), "heads");
  const int sequence = checked_positive_int(q.size(2), "sequence");
  const int head_dim = checked_positive_int(q.size(3), "head dimension");
  TORCH_CHECK(head_dim <= 128, "head dimension must be at most 128");
  TORCH_CHECK(static_cast<int64_t>(batch) * heads <=
                  std::numeric_limits<int>::max(),
              "batch * heads exceeds the supported int32 range");
  const c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor output = at::empty(q.sizes(),
                                q.options().dtype(at::ScalarType::Float));
  cudaError_t error;
  if (q.scalar_type() == at::kFloat) {
    error = operatorlib::attention_f32(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch, heads, sequence, head_dim, causal,
        current_stream(q));
  } else {
    error = operatorlib::attention_f16(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        output.data_ptr<float>(), batch, heads, sequence, head_dim, causal,
        current_stream(q));
  }
  check_launch(error, "attention");
  return output;
}

}  // namespace

TORCH_LIBRARY(operatorlib, m) {
  m.def("reduce_sum(Tensor input) -> Tensor");
  m.def("softmax(Tensor input) -> Tensor");
  m.def("transpose(Tensor input) -> Tensor");
  m.def("rmsnorm(Tensor input, Tensor weight, float epsilon=1e-5) -> Tensor");
  m.def("gemm(Tensor a, Tensor b) -> Tensor");
  m.def("attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(operatorlib, CUDA, m) {
  m.impl("reduce_sum", &reduce_sum_cuda);
  m.impl("softmax", &softmax_cuda);
  m.impl("transpose", &transpose_cuda);
  m.impl("rmsnorm", &rmsnorm_cuda);
  m.impl("gemm", &gemm_cuda);
  m.impl("attention", &attention_cuda);
}
