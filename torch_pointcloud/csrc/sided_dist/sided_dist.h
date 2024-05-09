#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> sided_distance_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> sided_distance_backward_cpu(
    const at::Tensor& grad_dists,
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& idxs,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> sided_distance_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> sided_distance_backward_cuda(
    const at::Tensor& grad_dists,
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& idxs,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> sided_distance(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  if (pc1.is_cuda() || pc2.is_cuda()) {
#ifdef WITH_CUDA
    return sided_distance_cuda(pc1, pc2, lengths1, lengths2);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return sided_distance_cpu(pc1, pc2, lengths1, lengths2);
  }
}

std::tuple<at::Tensor, at::Tensor> sided_distance_backward(
    const at::Tensor& grad_dists,
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& idxs,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  if (pc1.is_cuda() || pc2.is_cuda()) {
#ifdef WITH_CUDA
    return sided_distance_backward_cuda(
        grad_dists, pc1, pc2, idxs, lengths1, lengths2);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return sided_distance_backward_cpu(
        grad_dists, pc1, pc2, idxs, lengths1, lengths2);
  }
}
