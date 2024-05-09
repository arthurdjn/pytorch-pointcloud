#include <torch/extension.h>

at::Tensor k_interpolate_cpu(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor k_interpolate_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths,
    const int64_t M);

at::Tensor k_interpolate_cuda(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor k_interpolate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths,
    const int64_t M);

at::Tensor k_interpolate(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return k_interpolate_cuda(points, idxs, weights, K, lengths, out_lengths);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return k_interpolate_cpu(points, idxs, weights, K, lengths, out_lengths);
  }
}

at::Tensor k_interpolate_backward(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& K,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths,
    const int64_t M) {
  if (grad_out.is_cuda()) {
#ifdef WITH_CUDA
    return k_interpolate_backward_cuda(
        grad_out, idxs, weights, K, lengths, out_lengths, M);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return k_interpolate_backward_cpu(
        grad_out, idxs, weights, K, lengths, out_lengths, M);
  }
}
