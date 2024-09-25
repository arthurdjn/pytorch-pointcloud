#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "../cuda_utils.h"
#include "../scatter/atomics.cuh"

template <typename scalar_t>
__global__ void grid_cluster_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const scalar_t voxel_size,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids) {
  const int b = blockIdx.x; // Batch index
  const int tid = threadIdx.x + blockDim.x * blockIdx.y; // Thread index
  const int total_threads = blockDim.x * gridDim.y; // Total threads in the grid

  const int64_t cur_length = lengths[b]; // Actual number of points in current batch

  if (tid < cur_length) {
    scalar_t min_x = points[b][0][0], max_x = min_x;
    scalar_t min_y = points[b][0][1], max_y = min_y;
    scalar_t min_z = points[b][0][2], max_z = min_z;

    // Compute bounding box for the current batch
    for (int i = 0; i < cur_length; ++i) {
      min_x = fminf(min_x, points[b][i][0]);
      max_x = fmaxf(max_x, points[b][i][0]);
      min_y = fminf(min_y, points[b][i][1]);
      max_y = fmaxf(max_y, points[b][i][1]);
      min_z = fminf(min_z, points[b][i][2]);
      max_z = fmaxf(max_z, points[b][i][2]);
    }

    scalar_t origin_x = floor(min_x / voxel_size) * voxel_size;
    scalar_t origin_y = floor(min_y / voxel_size) * voxel_size;
    scalar_t origin_z = floor(min_z / voxel_size) * voxel_size;

    int64_t dim_x = static_cast<int64_t>(floor((max_x - origin_x) / voxel_size)) + 1;
    int64_t dim_y = static_cast<int64_t>(floor((max_y - origin_y) / voxel_size)) + 1;

    // Parallel iteration over points
    for (int64_t i = tid; i < cur_length; i += total_threads) {
      scalar_t x = points[b][i][0];
      scalar_t y = points[b][i][1];
      scalar_t z = points[b][i][2];

      int64_t gx = static_cast<int64_t>(floor((x - origin_x) / voxel_size));
      int64_t gy = static_cast<int64_t>(floor((y - origin_y) / voxel_size));
      int64_t gz = static_cast<int64_t>(floor((z - origin_z) / voxel_size));

      // Compute a unique voxel index (cluster ID)
      int64_t voxel_idx = gx + dim_x * gy + dim_x * dim_y * gz;

      // Atomically assign cluster ID
      //   atomicCAS(&cluster_ids[b][i], -1, voxel_idx); // Ensure thread safety
      atomicCAS(
          reinterpret_cast<unsigned long long int*>(&cluster_ids[b][i]),
          static_cast<unsigned long long int>(-1),
          static_cast<unsigned long long int>(voxel_idx));
    }
  }
}

at::Tensor grid_cluster_cuda(
    const at::Tensor& points, // shape (B, N, 3)
    const at::Tensor& lengths, // shape (B), actual number of points per batch
    const float voxel_size) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);

  auto opts = points.options().dtype(at::kLong); // Use long for cluster IDs
  at::Tensor cluster_ids = at::full({B, N}, -1, opts);

  dim3 threads_per_block(1024); // Set threads per block (this can be adjusted)
  dim3 num_blocks(B, (N + 1024 - 1) / 1024); // Calculate number of blocks needed

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "grid_cluster_cuda", ([&] {
        grid_cluster_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
            points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            voxel_size,
            cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>());
      }));

  return cluster_ids;
}
