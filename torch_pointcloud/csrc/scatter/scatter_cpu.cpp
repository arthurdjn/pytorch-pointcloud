#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

at::Tensor count_num_clusters_cpu(
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths) {
  auto B = cluster_ids.size(0);
  auto N = cluster_ids.size(1);

  auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
  auto lengths_accessor = lengths.accessor<int64_t, 1>();

  auto options = cluster_ids.options();
  auto num_clusters = at::zeros({B}, options);
  auto num_clusters_a = num_clusters.accessor<int64_t, 1>();

  for (int64_t b = 0; b < B; ++b) {
    int64_t length_b = lengths_accessor[b];
    std::vector<bool> cluster_seen(N, false);

    for (int64_t n = 0; n < length_b; ++n) {
      int64_t cluster_id = cluster_ids_a[b][n];
      if (!cluster_seen[cluster_id]) {
        cluster_seen[cluster_id] = true;
        num_clusters_a[b]++;
      }
    }
  }

  return num_clusters;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_sum_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
  auto indices = at::full({B, max_num_clusters, C}, -1, at::kLong);
  auto output = at::zeros({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_sum_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t n = 0; n < lengths_a[b]; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][cluster][d] += points_a[b][n][d];
        }

        counts_a[b][cluster] += 1;
      }

      // Padding for unused clusters
      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_mean_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
  auto indices = at::full({B, max_num_clusters, C}, -1, at::kLong);
  auto output = at::zeros({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_mean_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t n = 0; n < lengths_a[b]; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][cluster][d] += points_a[b][n][d];
        }

        counts_a[b][cluster] += 1;
      }

      for (int64_t cluster = 0; cluster < num_clusters_a[b]; ++cluster) {
        if (counts_a[b][cluster] > 0) {
          for (int64_t d = 0; d < C; ++d) {
            output_a[b][cluster][d] /= counts_a[b][cluster];
          }
        }
      }

      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_prod_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
  auto indices = at::full({B, max_num_clusters, C}, -1, at::kLong);
  auto output = at::ones({B, max_num_clusters, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_prod_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto output_a = output.accessor<scalar_t, 3>();
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t n = 0; n < lengths_a[b]; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][cluster][d] *= points_a[b][n][d];
        }

        counts_a[b][cluster] += 1;
      }

      // Padding for unused clusters
      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_min_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
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
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        for (int64_t d = 0; d < C; ++d) {
          if (points_a[b][n][d] < output_a[b][cluster][d]) {
            output_a[b][cluster][d] = points_a[b][n][d];
            indices_a[b][cluster][d] = n;
          }
        }

        counts_a[b][cluster] += 1;
      }

      // Padding for unused clusters
      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_max_cpu(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto num_clusters = count_num_clusters_cpu(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros({B, max_num_clusters}, at::kLong);
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
    auto counts_a = counts.accessor<int64_t, 2>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        for (int64_t d = 0; d < C; ++d) {
          if (points_a[b][n][d] > output_a[b][cluster][d]) {
            output_a[b][cluster][d] = points_a[b][n][d];
            indices_a[b][cluster][d] = n;
          }
        }

        counts_a[b][cluster] += 1;
      }

      // Padding for unused clusters
      for (int64_t c = num_clusters_a[b]; c < max_num_clusters; ++c) {
        for (int64_t d = 0; d < C; ++d) {
          output_a[b][c][d] = static_cast<scalar_t>(padding_value);
        }
      }
    }
  });

  return std::make_tuple(output, num_clusters, counts, indices);
}

#define AT_DISPATCH_REDUCTION_TYPES(                                        \
    reduce, points, cluster_ids, lengths, padding_value)                    \
  [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {     \
    if (reduce == "sum")                                                    \
      return scatter_sum_cpu(points, cluster_ids, lengths, padding_value);  \
    else if (reduce == "mean")                                              \
      return scatter_mean_cpu(points, cluster_ids, lengths, padding_value); \
    else if (reduce == "prod")                                              \
      return scatter_prod_cpu(points, cluster_ids, lengths, padding_value); \
    else if (reduce == "min")                                               \
      return scatter_min_cpu(points, cluster_ids, lengths, padding_value);  \
    else if (reduce == "max")                                               \
      return scatter_max_cpu(points, cluster_ids, lengths, padding_value);  \
    else                                                                    \
      AT_ERROR("Unknown reduction type: ", reduce);                         \
  }()

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_cpu(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  return AT_DISPATCH_REDUCTION_TYPES(
      reduce, points, cluster_ids, lengths, padding_value);
}

at::Tensor scatter_sum_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_sum_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        if (cluster >= 0 && cluster < num_clusters_a[b]) {
          for (int64_t d = 0; d < C; ++d) {
            grad_points_a[b][n][d] += grad_output_a[b][cluster][d];
          }
        }
      }
    }
  });

  return grad_points;
}

at::Tensor scatter_mean_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
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
          int64_t cluster_size = counts_a[b][cluster];
          if (cluster_size > 0) {
            for (int64_t d = 0; d < C; ++d) {
              grad_points_a[b][n][d] +=
                  grad_output_a[b][cluster][d] / static_cast<scalar_t>(cluster_size);
            }
          }
        }
      }
    }
  });

  return grad_points;
}

at::Tensor scatter_prod_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);
  auto prod_all_except_n = at::ones_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_prod_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 2>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();
    auto points_a = points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t n = 0; n < length_b; ++n) {
        int64_t cluster = cluster_ids_a[b][n];
        if (cluster >= 0 && cluster < num_clusters_a[b]) {
          for (int64_t d = 0; d < C; ++d) {
            scalar_t product = 1;
            for (int64_t m = 0; m < length_b; ++m) {
              if (m != n && cluster_ids_a[b][m] == cluster) {
                product *= points_a[b][m][d];
              }
            }

            grad_points_a[b][n][d] = grad_output_a[b][cluster][d] * product;
          }
        }
      }
    }
  });

  return grad_points;
}

at::Tensor scatter_min_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_min_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto indices_a = indices.accessor<int64_t, 3>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];
      for (int64_t cluster = 0; cluster < num_clusters_a[b]; ++cluster) {
        for (int64_t d = 0; d < C; ++d) {
          int64_t min_idx = indices_a[b][cluster][d];
          if (min_idx >= 0 && min_idx < length_b) {
            grad_points_a[b][min_idx][d] += grad_output_a[b][cluster][d];
          }
        }
      }
    }
  });

  return grad_points;
}

at::Tensor scatter_max_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "scatter_max_backward_cpu", [&] {
    auto grad_output_a = grad_output.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto num_clusters_a = num_clusters.accessor<int64_t, 1>();
    auto indices_a = indices.accessor<int64_t, 3>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();

    for (int64_t b = 0; b < B; ++b) {
      int64_t length_b = lengths_a[b];

      for (int64_t cluster = 0; cluster < num_clusters_a[b]; ++cluster) {
        for (int64_t d = 0; d < C; ++d) {
          int64_t max_idx = indices_a[b][cluster][d];
          if (max_idx >= 0 && max_idx < length_b) {
            grad_points_a[b][max_idx][d] += grad_output_a[b][cluster][d];
          }
        }
      }
    }
  });

  return grad_points;
}

#define AT_DISPATCH_REDUCTION_BACKWARD_TYPES(       \
    reduce,                                         \
    grad_output,                                    \
    points,                                         \
    cluster_ids,                                    \
    lengths,                                        \
    num_clusters,                                   \
    counts,                                         \
    indices)                                        \
  [&]() -> at::Tensor {                             \
    if (reduce == "sum")                            \
      return scatter_sum_backward_cpu(              \
          grad_output,                              \
          points,                                   \
          cluster_ids,                              \
          lengths,                                  \
          num_clusters,                             \
          counts,                                   \
          indices);                                 \
    else if (reduce == "mean")                      \
      return scatter_mean_backward_cpu(             \
          grad_output,                              \
          points,                                   \
          cluster_ids,                              \
          lengths,                                  \
          num_clusters,                             \
          counts,                                   \
          indices);                                 \
    else if (reduce == "prod")                      \
      return scatter_prod_backward_cpu(             \
          grad_output,                              \
          points,                                   \
          cluster_ids,                              \
          lengths,                                  \
          num_clusters,                             \
          counts,                                   \
          indices);                                 \
    else if (reduce == "min")                       \
      return scatter_min_backward_cpu(              \
          grad_output,                              \
          points,                                   \
          cluster_ids,                              \
          lengths,                                  \
          num_clusters,                             \
          counts,                                   \
          indices);                                 \
    else if (reduce == "max")                       \
      return scatter_max_backward_cpu(              \
          grad_output,                              \
          points,                                   \
          cluster_ids,                              \
          lengths,                                  \
          num_clusters,                             \
          counts,                                   \
          indices);                                 \
    else                                            \
      AT_ERROR("Unknown reduction type: ", reduce); \
  }()

at::Tensor scatter_backward_cpu(
    const std::string& reduce,
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts,
    const at::Tensor& indices) {
  return AT_DISPATCH_REDUCTION_BACKWARD_TYPES(
      reduce,
      grad_output,
      points,
      cluster_ids,
      lengths,
      num_clusters,
      counts,
      indices);
}