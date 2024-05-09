#include <torch/extension.h>
#include <tuple>

at::Tensor fps_cpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& num_samples,
    const at::Tensor& start_idxs);

at::Tensor fps_cuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& num_samples,
    const at::Tensor& start_idxs);

at::Tensor fps(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& num_samples,
    const at::Tensor& start_idxs) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return fps_cuda(points, lengths, num_samples, start_idxs);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return fps_cpu(points, lengths, num_samples, start_idxs);
  }
}
