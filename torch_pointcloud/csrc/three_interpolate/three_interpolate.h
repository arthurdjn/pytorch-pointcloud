#include <torch/extension.h>

at::Tensor three_interpolate_cpu(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor three_interpolate_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor three_interpolate_cuda(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor three_interpolate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths);

at::Tensor three_interpolate(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return three_interpolate_cuda(points, idxs, weights, lengths, out_lengths);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return three_interpolate_cpu(points, idxs, weights, lengths, out_lengths);
  }
}

at::Tensor three_interpolate_backward(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  if (grad_out.is_cuda()) {
#ifdef WITH_CUDA
    return three_interpolate_backward_cuda(grad_out, idxs, weights, lengths, out_lengths);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return three_interpolate_backward_cpu(grad_out, idxs, weights, lengths, out_lengths);
  }
}
