#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>

#include "../cuda_utils.h"
#include "../scatter/atomics.cuh"

template <typename scalar_t>
__global__ void grid_cluster_packed_kernel(
    const at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> batch_idxs,
    const int64_t B,
    const scalar_t voxel_size,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> cluster_ids,
    at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> min_coords,
    at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> max_coords) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_points = points.size(0);

  if (idx >= total_points)
    return;

  const int64_t b = batch_idxs[idx];

  scalar_t x = points[idx][0];
  scalar_t y = points[idx][1];
  scalar_t z = points[idx][2];

  atomMin(&min_coords[b][0], x);
  atomMax(&max_coords[b][0], x);
  atomMin(&min_coords[b][1], y);
  atomMax(&max_coords[b][1], y);
  atomMin(&min_coords[b][2], z);
  atomMax(&max_coords[b][2], z);

  scalar_t origin_x = floor(min_coords[b][0] / voxel_size) * voxel_size;
  scalar_t origin_y = floor(min_coords[b][1] / voxel_size) * voxel_size;
  scalar_t origin_z = floor(min_coords[b][2] / voxel_size) * voxel_size;

  int64_t dim_x =
      static_cast<int64_t>(floor((max_coords[b][0] - origin_x) / voxel_size)) + 1;
  int64_t dim_y =
      static_cast<int64_t>(floor((max_coords[b][1] - origin_y) / voxel_size)) + 1;

  int64_t gx = static_cast<int64_t>(floor((x - origin_x) / voxel_size));
  int64_t gy = static_cast<int64_t>(floor((y - origin_y) / voxel_size));
  int64_t gz = static_cast<int64_t>(floor((z - origin_z) / voxel_size));

  int64_t voxel_idx = gx + dim_x * gy + dim_x * dim_y * gz;

  atomicCAS(
      reinterpret_cast<unsigned long long int*>(&cluster_ids[idx]),
      static_cast<unsigned long long int>(-1),
      static_cast<unsigned long long int>(voxel_idx));
}

at::Tensor grid_cluster_packed_cuda(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size) {
  const int64_t N = points.size(0);
  const int64_t C = points.size(1);
  const int64_t B = batch_idxs.max().item<int64_t>() + 1;

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto cluster_ids = at::full({N}, -1, long_opts);
  auto min_coords = at::full({B, 3}, std::numeric_limits<float>::max(), opts);
  auto max_coords = at::full({B, 3}, std::numeric_limits<float>::lowest(), opts);

  dim3 threads_per_block(1024);
  dim3 num_blocks((N + 1024 - 1) / 1024);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "grid_cluster_packed_cuda", ([&] {
        grid_cluster_packed_kernel<scalar_t>
            <<<num_blocks, threads_per_block, 0, stream>>>(
                points.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                batch_idxs.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                B,
                voxel_size,
                cluster_ids.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                min_coords.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                max_coords.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>());
      }));

  return cluster_ids;
}
