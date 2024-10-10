#include <torch/extension.h>
#include <cmath>
#include <limits>

at::Tensor grid_cluster_packed_cpu(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size) {
  const int64_t N = points.size(0);
  const int64_t C = points.size(1);
  const int64_t B = batch_idxs.max().item<int64_t>() + 1;

  const auto opts = points.options();
  const auto long_opts = opts.dtype(at::kLong);
  auto cluster_ids = at::full({N}, -1, long_opts);
  auto min_coords = at::full({B, 3}, std::numeric_limits<float>::max(), opts);
  auto max_coords = at::full({B, 3}, std::numeric_limits<float>::min(), opts);

  return AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "grid_cluster_packed_cpu", [&] {
        auto points_a = points.accessor<scalar_t, 2>();
        auto batch_idxs_a = batch_idxs.accessor<int64_t, 1>();
        auto cluster_ids_a = cluster_ids.accessor<int64_t, 1>();
        auto min_coords_a = min_coords.accessor<scalar_t, 2>();
        auto max_coords_a = max_coords.accessor<scalar_t, 2>();

        for (int64_t i = 0; i < N; ++i) {
          const int64_t b = batch_idxs_a[i];
          min_coords_a[b][0] = std::min(min_coords_a[b][0], points_a[i][0]); // Min x
          max_coords_a[b][0] = std::max(max_coords_a[b][0], points_a[i][0]); // Max x
          min_coords_a[b][1] = std::min(min_coords_a[b][1], points_a[i][1]); // Min y
          max_coords_a[b][1] = std::max(max_coords_a[b][1], points_a[i][1]); // Max y
          min_coords_a[b][2] = std::min(min_coords_a[b][2], points_a[i][2]); // Min z
          max_coords_a[b][2] = std::max(max_coords_a[b][2], points_a[i][2]); // Max z
        }

        for (int64_t i = 0; i < N; ++i) {
          const int64_t b = batch_idxs_a[i];
          const scalar_t x = points_a[i][0];
          const scalar_t y = points_a[i][1];
          const scalar_t z = points_a[i][2];

          const scalar_t min_x = min_coords_a[b][0];
          const scalar_t min_y = min_coords_a[b][1];
          const scalar_t min_z = min_coords_a[b][2];

          const scalar_t max_x = max_coords_a[b][0];
          const scalar_t max_y = max_coords_a[b][1];
          const scalar_t max_z = max_coords_a[b][2];

          // Define the origin of the grid at the center of the bounding box
          scalar_t origin_x = std::floor(min_x / voxel_size) * voxel_size;
          scalar_t origin_y = std::floor(min_y / voxel_size) * voxel_size;
          scalar_t origin_z = std::floor(min_z / voxel_size) * voxel_size;

          // Calculate grid dimensions (the number of voxels along each axis)
          int64_t dim_x =
              static_cast<int64_t>(std::floor((max_x - origin_x) / voxel_size)) + 1;
          int64_t dim_y =
              static_cast<int64_t>(std::floor((max_y - origin_y) / voxel_size)) + 1;

          int64_t gx = static_cast<int64_t>(std::floor((x - origin_x) / voxel_size));
          int64_t gy = static_cast<int64_t>(std::floor((y - origin_y) / voxel_size));
          int64_t gz = static_cast<int64_t>(std::floor((z - origin_z) / voxel_size));

          int64_t voxel_idx = gx + dim_x * gy + dim_x * dim_y * gz;
          cluster_ids_a[i] = voxel_idx;
        }

        return cluster_ids;
      });
}
