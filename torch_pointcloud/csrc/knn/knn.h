#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> knn_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& K,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> knn_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& K,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2);

std::tuple<at::Tensor, at::Tensor> knn(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& K,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  if (pc1.is_cuda() && pc2.is_cuda()) {
#ifdef WITH_CUDA
    return knn_cuda(pc1, pc2, K, lengths1, lengths2);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return knn_cpu(pc1, pc2, K, lengths1, lengths2);
  }
}
