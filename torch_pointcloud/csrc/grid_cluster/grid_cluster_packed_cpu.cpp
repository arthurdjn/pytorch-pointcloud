#include <torch/extension.h>
#include <cmath>
#include <unordered_map>
#include <vector>

at::Tensor grid_cluster_packed_cpu(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size) {
  const int64_t N = points.size(0);
  const int64_t C = points.size(1);
  const int64_t B = batch_idxs.max().item<int64_t>() + 1;

  TORCH_CHECK(C == 3, "Expected point cloud to have 3 coordinates per point");

  auto opts = points.options().dtype(at::kLong);
  at::Tensor cluster_ids = at::full({N}, -1, opts);

  return AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "grid_cluster_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 2>();
    auto batch_idxs_a = batch_idxs.accessor<int64_t, 1>();
    auto cluster_ids_a = cluster_ids.accessor<int64_t, 1>();

    std::vector<scalar_t> min_x(B, std::numeric_limits<scalar_t>::max());
    std::vector<scalar_t> max_x(B, std::numeric_limits<scalar_t>::lowest());
    std::vector<scalar_t> min_y(B, std::numeric_limits<scalar_t>::max());
    std::vector<scalar_t> max_y(B, std::numeric_limits<scalar_t>::lowest());
    std::vector<scalar_t> min_z(B, std::numeric_limits<scalar_t>::max());
    std::vector<scalar_t> max_z(B, std::numeric_limits<scalar_t>::lowest());

    // First pass: compute bounding boxes for each batch
    for (int64_t i = 0; i < N; ++i) {
      int64_t b = batch_idxs_a[i];
      min_x[b] = std::min(min_x[b], points_a[i][0]);
      max_x[b] = std::max(max_x[b], points_a[i][0]);
      min_y[b] = std::min(min_y[b], points_a[i][1]);
      max_y[b] = std::max(max_y[b], points_a[i][1]);
      min_z[b] = std::min(min_z[b], points_a[i][2]);
      max_z[b] = std::max(max_z[b], points_a[i][2]);
    }

    std::vector<scalar_t> origin_x(B), origin_y(B), origin_z(B);
    std::vector<int64_t> dim_x(B), dim_y(B);

    // Compute grid origins and dimensions for each batch
    for (int64_t b = 0; b < B; ++b) {
      origin_x[b] = std::floor(min_x[b] / voxel_size) * voxel_size;
      origin_y[b] = std::floor(min_y[b] / voxel_size) * voxel_size;
      origin_z[b] = std::floor(min_z[b] / voxel_size) * voxel_size;

      dim_x[b] =
          static_cast<int64_t>(std::floor((max_x[b] - origin_x[b]) / voxel_size)) +
          1;
      dim_y[b] =
          static_cast<int64_t>(std::floor((max_y[b] - origin_y[b]) / voxel_size)) +
          1;
    }

    // Second pass: assign cluster IDs
    for (int64_t i = 0; i < N; ++i) {
      int64_t b = batch_idxs_a[i];
      scalar_t x = points_a[i][0];
      scalar_t y = points_a[i][1];
      scalar_t z = points_a[i][2];

      int64_t gx = static_cast<int64_t>(std::floor((x - origin_x[b]) / voxel_size));
      int64_t gy = static_cast<int64_t>(std::floor((y - origin_y[b]) / voxel_size));
      int64_t gz = static_cast<int64_t>(std::floor((z - origin_z[b]) / voxel_size));

      int64_t voxel_idx = gx + dim_x[b] * gy + dim_x[b] * dim_y[b] * gz;
      cluster_ids_a[i] = voxel_idx;
    }

    return cluster_ids;
  });
}