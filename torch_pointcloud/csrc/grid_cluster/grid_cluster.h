#include <torch/extension.h>
#include <tuple>

at::Tensor grid_cluster_cpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const float voxel_size);

at::Tensor grid_cluster(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const float voxel_size) {
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("CUDA support is not available in this build.");
#endif
  } else {
    return grid_cluster_cpu(points, lengths, voxel_size);
  }
}
