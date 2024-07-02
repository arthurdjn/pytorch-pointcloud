#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor> avg_voxelize_cpu(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution);

std::tuple<at::Tensor, at::Tensor, at::Tensor> avg_voxelize_cuda(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution);

at::Tensor avg_voxelize_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& indices,
    const at::Tensor& counts);

at::Tensor avg_voxelize_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& indices,
    const at::Tensor& counts);

std::tuple<at::Tensor, at::Tensor, at::Tensor> avg_voxelize(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  if (features.is_cuda()) {
#ifdef WITH_CUDA
    return avg_voxelize_cuda(coords, features, resolution);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return avg_voxelize_cpu(coords, features, resolution);
  }
}

at::Tensor avg_voxelize_backward(
    const at::Tensor& grad_out,
    const at::Tensor& indices,
    const at::Tensor& counts) {
  if (grad_out.is_cuda()) {
#ifdef WITH_CUDA
    return avg_voxelize_backward_cuda(grad_out, indices, counts);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return avg_voxelize_backward_cpu(grad_out, indices, counts);
  }
}
