#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_cpu(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution);

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_cuda(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution);

at::Tensor trilinear_devoxelize_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const int resolution);

at::Tensor trilinear_devoxelize_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const int resolution);

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  if (features.is_cuda()) {
#ifdef WITH_CUDA
    return trilinear_devoxelize_cuda(coords, features, resolution);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return trilinear_devoxelize_cpu(coords, features, resolution);
  }
}

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const int resolution) {
  if (grad_out.is_cuda()) {
#ifdef WITH_CUDA
    return trilinear_devoxelize_backward_cuda(grad_out, idxs, weights, resolution);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return trilinear_devoxelize_backward_cpu(grad_out, idxs, weights, resolution);
  }
}
