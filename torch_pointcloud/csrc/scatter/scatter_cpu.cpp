#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

/**
 * @brief Count the number of clusters in each batch.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B,) tensor. The lengths of each batch in case of padding.
 * @return (B,) tensor. The number of clusters in each batch.
 */
at::Tensor count_num_clusters_cpu(
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths) {
  auto B = cluster_ids.size(0);
  auto N = cluster_ids.size(1);

  auto num_clusters = at::zeros({B}, cluster_ids.options());

  auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
  auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
  auto lengths_a = lengths.accessor<int64_t, 1>();

  for (int64_t b = 0; b < B; ++b) {
    int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
    std::vector<bool> cluster_seen(N, false);

    for (int64_t n = 0; n < length_b; ++n) {
      int64_t cluster_id = cluster_ids_a[b][n];
      if (!cluster_seen[cluster_id] && cluster_id >= 0 && cluster_id < N) {
        cluster_seen[cluster_id] = true;
        num_clusters_a[b]++;
      }
    }
  }

  return num_clusters;
}

/**
 * @brief Sum the input points together based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 *   This tensor is used during the backward pass, to avoid computing
 *   the number of clusters again.
 */
std::tuple<at::Tensor, at::Tensor> scatter_sum_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto output = at::zeros({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_sum_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < max_num_clusters) {
          for (int64_t d = 0; d < C; ++d) {
            output_a[b][cluster_id][d] += points_a[b][n][d];
          }
        }
      }

      // Padding for unused clusters
      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters);
}

/**
 * @brief Compute the mean of the input points based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - counts (B, max_num_clusters) tensor. The number of points in each cluster.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_mean_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
  auto output = at::zeros({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_mean_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      // Sum the points per cluster
      for (int64_t n = 0; n < length_b; ++n) {
        const int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < max_num_clusters) {
          for (int64_t d = 0; d < C; ++d) {
            output_a[b][cluster_id][d] += points_a[b][n][d];
          }
          counts_a[b][cluster_id] += 1;
        }
      }

      // Divide by the number of points per cluster
      for (int64_t n = 0; n < length_b; ++n) {
        const int64_t cluster_id = cluster_ids_a[b][n];
        if (counts_a[b][cluster_id] > 0 && cluster_id >= 0 &&
            cluster_id < max_num_clusters) {
          for (int64_t d = 0; d < C; ++d) {
            output_a[b][cluster_id][d] /= counts_a[b][cluster_id];
          }
        }
      }

      // Padding for unused clusters
      for (int64_t n = num_clusters_a[b]; n < max_num_clusters; ++n) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][n][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts);
}

/**
 * @brief Compute the product of the input points based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 */
std::tuple<at::Tensor, at::Tensor> scatter_prod_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto output = at::ones({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_prod_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t n = 0; n < length_b; ++n) {
        const int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < max_num_clusters) {
          for (int64_t c = 0; c < C; ++c) {
            output_a[b][cluster_id][c] *= points_a[b][n][c];
          }
        }
      }

      // Padding for unused clusters
      for (int64_t n = num_clusters_a[b]; n < max_num_clusters; ++n) {
        for (int64_t c = 0; c < C; ++c) {
          output_a[b][n][c] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters);
}

/**
 * @brief Compute the minimum channel-wise of the input points based on the cluster
 * ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - indices (B, max_num_clusters, C) tensor. The indices of the minimum points.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_min_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto indices = at::full({B, max_num_clusters, C}, -1, at::kLong);
  auto output = at::full(
      {B, max_num_clusters, C}, std::numeric_limits<float>::max(), points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_min_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto indices_a = indices.accessor<int64_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t n = 0; n < length_b; ++n) {
        const int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < max_num_clusters) {
          for (int64_t c = 0; c < C; ++c) {
            if (points_a[b][n][c] < output_a[b][cluster_id][c]) {
              output_a[b][cluster_id][c] = points_a[b][n][c];
              indices_a[b][cluster_id][c] = n;
            }
          }
        }
      }

      // Padding for unused clusters
      for (int64_t n = num_clusters_a[b]; n < max_num_clusters; ++n) {
        for (int64_t c = 0; c < C; ++c) {
          output_a[b][n][c] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, indices);
}

/**
 * @brief Compute the maximum channel-wise of the input points based on the cluster
 * ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - indices (B, max_num_clusters, C) tensor. The indices of the maximum points.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_max_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto indices = at::full({B, max_num_clusters, C}, -1, at::kLong);
  auto output = at::full(
      {B, max_num_clusters, C},
      std::numeric_limits<float>::lowest(),
      points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_max_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto indices_a = indices.accessor<int64_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t n = 0; n < length_b; ++n) {
        const int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < max_num_clusters) {
          for (int64_t c = 0; c < C; ++c) {
            if (points_a[b][n][c] > output_a[b][cluster_id][c]) {
              output_a[b][cluster_id][c] = points_a[b][n][c];
              indices_a[b][cluster_id][c] = n;
            }
          }
        }
      }

      // Padding for unused clusters
      for (int64_t n = num_clusters_a[b]; n < max_num_clusters; ++n) {
        for (int64_t c = 0; c < C; ++c) {
          output_a[b][n][c] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, indices);
}

/**
 * @brief Compute the backward pass for the scatter sum operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_sum_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_sum_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      const int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster_id = cluster_ids_a[b][n];
        if (cluster_id >= 0 && cluster_id < num_clusters_a[b]) {
          for (int64_t c = 0; c < C; ++c) {
            grad_points_a[b][n][c] += grad_output_a[b][cluster_id][c];
          }
        }
      }
    }
  });

  return grad_points;
}

/**
 * @brief Compute the backward pass for the scatter mean operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param counts (B, max_num_clusters) tensor. The number of points in each cluster.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_mean_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_mean_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto counts_a = counts.accessor<int64_t, 2>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        if (cluster >= 0 && cluster < num_clusters_a[b]) {
          int64_t count = counts_a[b][cluster];
          if (count > 0) {
            for (int64_t d = 0; d < C; ++d) {
              grad_points_a[b][n][d] +=
                  grad_output_a[b][cluster][d] / static_cast<scalar_t>(count);
            }
          }
        }
      }
    }
  });

  return grad_points;
}

/**
 * @brief Compute the backward pass for the scatter prod operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_prod_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& output) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_prod_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto points_a = points.accessor<scalar_t, 3>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();
    auto output_a = output.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        if (cluster >= 0 && cluster < num_clusters_a[b]) {
          for (int64_t d = 0; d < C; ++d) {
            if (points_a[b][n][d] != 0) {
              grad_points_a[b][n][d] += grad_output_a[b][cluster][d] *
                  output_a[b][cluster][d] / points_a[b][n][d];
            }
          }
        }
      }
    }
  });

  return grad_points;
}

/**
 * @brief Compute the backward pass for the scatter min operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param indices (B, max_num_clusters, C) tensor. The indices of the minimum points.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_min_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_min_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto indices_a = indices.accessor<int64_t, 3>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);
      for (int64_t cluster_id = 0; cluster_id < num_clusters_a[b]; ++cluster_id) {
        for (int64_t c = 0; c < C; ++c) {
          int64_t min_idx = indices_a[b][cluster_id][c];
          if (min_idx >= 0 && min_idx < length_b) {
            grad_points_a[b][min_idx][c] += grad_output_a[b][cluster_id][c];
          }
        }
      }
    }
  });

  return grad_points;
}

/**
 * @brief Compute the backward pass for the scatter max operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param indices (B, max_num_clusters,  C) tensor. The indices of the maximum
 * points.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_max_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_max_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto indices_a = indices.accessor<int64_t, 3>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = std::clamp(lengths_a[b], int64_t(0), N);

      for (int64_t cluster_id = 0; cluster_id < num_clusters_a[b]; ++cluster_id) {
        for (int64_t c = 0; c < C; ++c) {
          int64_t max_idx = indices_a[b][cluster_id][c];
          if (max_idx >= 0 && max_idx < length_b) {
            grad_points_a[b][max_idx][c] += grad_output_a[b][cluster_id][c];
          }
        }
      }
    }
  });

  return grad_points;
}
