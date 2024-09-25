#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_cpu(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_cuda(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_cuda(reduce, points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_cpu(reduce, points, cluster_ids, lengths, padding_value);
  }
}

at::Tensor scatter_backward_cpu(
    const std::string& reduce,
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices);

at::Tensor scatter_backward_cuda(
    const std::string& reduce,
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& cluster_sizes,
    const at::Tensor& indices);

at::Tensor scatter_backward(
    const std::string& reduce,
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_backward_cuda(
        reduce,
        grad_output,
        points,
        cluster_ids,
        lengths,
        num_clusters,
        counts,
        indices);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_backward_cpu(
        reduce,
        grad_output,
        points,
        cluster_ids,
        lengths,
        num_clusters,
        counts,
        indices);
  }
}
