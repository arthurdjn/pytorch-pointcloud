#include <torch/extension.h>
#include <cmath>
#include <unordered_map>
#include <vector>

at::Tensor grid_cluster_cpu(
    const at::Tensor& points, // shape (B, N, 3)
    const at::Tensor& lengths, // shape (B), actual number of points per batch
    const float voxel_size) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  TORCH_CHECK(C == 3, "Expected point cloud to have 3 coordinates per point");

  auto opts = points.options().dtype(at::kLong);
  at::Tensor cluster_ids = at::full({B, N}, -1, opts);

  return AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "grid_cluster_cpu", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto cluster_ids_a =
        cluster_ids.accessor<int64_t, 2>(); // Accessor for cluster IDs (B, N)

    // Loop over each batch
    for (int64_t b = 0; b < B; ++b) {
      int64_t cur_length = lengths_a[b]; // Actual number of points in current batch

      // Determine the grid origin based on the point cloud's bounding box
      scalar_t min_x = points_a[b][0][0], max_x = min_x;
      scalar_t min_y = points_a[b][0][1], max_y = min_y;
      scalar_t min_z = points_a[b][0][2], max_z = min_z;

      // Compute the bounding box for the current batch
      for (int64_t i = 0; i < cur_length; ++i) {
        min_x = std::min(min_x, points_a[b][i][0]);
        max_x = std::max(max_x, points_a[b][i][0]);
        min_y = std::min(min_y, points_a[b][i][1]);
        max_y = std::max(max_y, points_a[b][i][1]);
        min_z = std::min(min_z, points_a[b][i][2]);
        max_z = std::max(max_z, points_a[b][i][2]);
      }

      // Define the origin of the grid at the center of the bounding box
      scalar_t origin_x = std::floor(min_x / voxel_size) * voxel_size;
      scalar_t origin_y = std::floor(min_y / voxel_size) * voxel_size;
      scalar_t origin_z = std::floor(min_z / voxel_size) * voxel_size;

      // Calculate grid dimensions (the number of voxels along each axis)
      int64_t dim_x =
          static_cast<int64_t>(std::floor((max_x - origin_x) / voxel_size)) + 1;
      int64_t dim_y =
          static_cast<int64_t>(std::floor((max_y - origin_y) / voxel_size)) + 1;

      // Loop through each point and assign a voxel index (cluster ID)
      for (int64_t i = 0; i < cur_length; ++i) {
        scalar_t x = points_a[b][i][0];
        scalar_t y = points_a[b][i][1];
        scalar_t z = points_a[b][i][2];

        // Calculate voxel indices along each axis
        int64_t gx = static_cast<int64_t>(std::floor((x - origin_x) / voxel_size));
        int64_t gy = static_cast<int64_t>(std::floor((y - origin_y) / voxel_size));
        int64_t gz = static_cast<int64_t>(std::floor((z - origin_z) / voxel_size));

        // Compute a unique voxel (cluster) ID using a 3D grid
        int64_t voxel_idx = gx + dim_x * gy + dim_x * dim_y * gz;
        cluster_ids_a[b][i] = voxel_idx;
      }
    }

    return cluster_ids;
  });
}
