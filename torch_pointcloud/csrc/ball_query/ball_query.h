#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> ball_query_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& max_neighbors,
    const at::Tensor& radiuses);

std::tuple<at::Tensor, at::Tensor> ball_query_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& max_neighbors,
    const at::Tensor& radiuses);

std::tuple<at::Tensor, at::Tensor> ball_query(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& max_neighbors,
    const at::Tensor& radiuses) {
  if (pc1.is_cuda() && pc2.is_cuda()) {
#ifdef WITH_CUDA
    return ball_query_cuda(
        pc1, pc2, lengths1, lengths2, max_neighbors, radiuses);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return ball_query_cpu(
        pc1, pc2, lengths1, lengths2, max_neighbors, radiuses);
  }
}
