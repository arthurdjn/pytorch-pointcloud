#include <torch/extension.h>
#include <tuple>

at::Tensor grid_cluster_cpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const float voxel_size);

at::Tensor grid_cluster_cuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const float voxel_size);

at::Tensor grid_cluster(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const float voxel_size) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return grid_cluster_cuda(points, lengths, voxel_size);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return grid_cluster_cpu(points, lengths, voxel_size);
  }
}

at::Tensor grid_cluster_packed_cpu(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size);

at::Tensor grid_cluster_packed_cuda(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size);

at::Tensor grid_cluster_packed(
    const at::Tensor& points,
    const at::Tensor& batch_idxs,
    const float voxel_size) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    return grid_cluster_packed_cuda(points, batch_idxs, voxel_size);
#else
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return grid_cluster_packed_cpu(points, batch_idxs, voxel_size);
  }
}
