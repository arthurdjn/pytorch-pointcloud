#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> scatter_sum_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor> scatter_sum_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor> scatter_sum(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_sum_cuda(points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_sum_cpu(points, cluster_ids, lengths, padding_value);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_mean_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_mean_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_mean(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_mean_cpu(points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_mean_cuda(points, cluster_ids, lengths, padding_value);
  }
}

std::tuple<at::Tensor, at::Tensor> scatter_prod_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor> scatter_prod_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor> scatter_prod(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_prod_cpu(points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_prod_cuda(points, cluster_ids, lengths, padding_value);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_min_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_min_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_min(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_min_cpu(points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_min_cuda(points, cluster_ids, lengths, padding_value);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_max_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_max_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0);

std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_max(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_max_cpu(points, cluster_ids, lengths, padding_value);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_max_cuda(points, cluster_ids, lengths, padding_value);
  }
}

at::Tensor scatter_sum_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters);

at::Tensor scatter_sum_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters);

at::Tensor scatter_sum_backward(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_sum_backward_cuda(
        grad_output, points, cluster_ids, lengths, num_clusters);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_sum_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters);
  }
}

at::Tensor scatter_mean_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts);

at::Tensor scatter_mean_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts);

at::Tensor scatter_mean_backward(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_mean_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, counts);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_mean_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, counts);
  }
}

at::Tensor scatter_prod_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& output);

at::Tensor scatter_prod_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& output);

at::Tensor scatter_prod_backward(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& output) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_prod_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, output);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_prod_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, output);
  }
}

at::Tensor scatter_min_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices);

at::Tensor scatter_min_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices);

at::Tensor scatter_min_backward(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_min_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, indices);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_min_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, indices);
  }
}

at::Tensor scatter_max_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices);

at::Tensor scatter_max_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices);

at::Tensor scatter_max_backward(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return scatter_max_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, indices);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return scatter_max_backward_cpu(
        grad_output, points, cluster_ids, lengths, num_clusters, indices);
  }
}
