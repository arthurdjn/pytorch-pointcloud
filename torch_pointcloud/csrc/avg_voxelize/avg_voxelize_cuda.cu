#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

#include "../cuda_utils.h"
#include "../utils.h"

/**
 * @brief Count the number of points in each voxel grid
 *
 * @param coords (B, N, 3) tensor, voxelized coordinates of each point
 * @param idxs (B, N, 3) tensor, voxel index of each point
 * @param counts (B, R, R, R) tensor, number of points in each voxel index
 */
__global__ void grid_stats_kernel(
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> coords,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<int, 4, at::RestrictPtrTraits> counts) {
  int b = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  const int64_t N = coords.size(1);

  for (int i = index; i < N; i += stride) {
    int x = coords[b][i][0];
    int y = coords[b][i][1];
    int z = coords[b][i][2];

    idxs[b][i][0] = x;
    idxs[b][i][1] = y;
    idxs[b][i][2] = z;

    atomicAdd(&counts[b][x][y][z], 1);
  }
}

/**
 * @brief Average pool voxelization kernel (forward)
 *
 * @param features (B, C, N) tensor, features of each point
 * @param idxs (B, N, 3) tensor, voxel index of each point, of shape (b, n)
 * @param counts (B, R, R, R) tensor, number of points in each voxel grid
 * @param out (B, N, R, R, R) tensor, features in each voxel grid
 */
__global__ void avg_voxelize_kernel(
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> features,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<int, 4, at::RestrictPtrTraits> counts,
    at::PackedTensorAccessor64<float, 5, at::RestrictPtrTraits> out) {
  int batch_idx = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  int C = features.size(1);
  int N = idxs.size(1);

  for (int i = index; i < N; i += stride) {
    int x = idxs[batch_idx][i][0]; // x-coordinate of voxel cell
    int y = idxs[batch_idx][i][1]; // y-coordinate of voxel cell
    int z = idxs[batch_idx][i][2]; // z-coordinate of voxel cell
    int cnt = counts[batch_idx][x][y][z]; // Access count using 3D index

    if (cnt > 0) {
      float inv_cnt = 1.0f / static_cast<float>(cnt);
      for (int c = 0; c < C; c++) {
        atomicAdd(&out[batch_idx][c][x][y][z], features[batch_idx][c][i] * inv_cnt);
      }
    }
  }
}

// TODO(arthurdjn) support all data types once `atomicAdd` supports doubles
// NOTE: The counts tensor must be of type int, not long, because of the `atomicAdd`
// function types overload (does not support (int64_t *, int) signature)
/**
 * @brief Average pool voxelization (forward)
 *
 * @param coords (B, N, 3) tensor, voxelized coordinates of each point
 * @param features (B, C, N) tensor, features of each point
 * @param resolution voxel resolution
 * @return std::tuple<at::Tensor, at::Tensor, at::Tensor>, features in each voxel
 * grid, voxel index of each point, number of points in each voxel grid
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> avg_voxelize_cuda(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  at::TensorArg coords_t{coords, "coords", 1}, features_t{features, "features", 2};
  at::CheckedFrom c = "avg_voxelize_cuda";
  at::checkAllSameGPU(c, {coords_t, features_t});

  CHECK_IS_CONTIGUOUS(coords);
  CHECK_IS_CONTIGUOUS(features);
  TORCH_CHECK(coords.dim() == 3, "coords must have 3 dimensions");
  TORCH_CHECK(features.dim() == 3, "features must have 3 dimensions");

  const int64_t B = features.size(0);
  const int64_t C = features.size(1);
  const int64_t N = features.size(2);
  int R = resolution;

  auto long_opt = features.options().dtype(at::kLong);
  auto int_opt = features.options().dtype(at::kInt);
  auto idxs = torch::zeros({B, N, 3}, long_opt);
  auto counts = torch::zeros({B, R, R, R}, int_opt);
  auto out = torch::zeros({B, C, R, R, R}, features.options());

  auto coords_a = coords.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>();
  auto features_a = features.packed_accessor64<float, 3, at::RestrictPtrTraits>();
  auto idxs_a = idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>();
  auto counts_a = counts.packed_accessor64<int, 4, at::RestrictPtrTraits>();
  auto out_a = out.packed_accessor64<float, 5, at::RestrictPtrTraits>();

  grid_stats_kernel<<<B, optimal_num_threads(N)>>>(coords_a, idxs_a, counts_a);
  avg_voxelize_kernel<<<B, optimal_num_threads(N)>>>(
      features_a, idxs_a, counts_a, out_a);

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(out, idxs, counts);
}

/**
 * @brief Average pool voxelization kernel (backward)
 *
 * @param idxs (B, N, 3) tensor, voxel index of each point
 * @param counts (B, R, R, R) tensor, number of points in each voxel grid
 * @param grad_out (B, C, R, R, R) tensor, gradients of each voxel grid
 * @param grad_features (B, C, N) tensor, gradients of each point
 */
__global__ void avg_voxelize_backward_kernel(
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<int, 4, at::RestrictPtrTraits> counts,
    const at::PackedTensorAccessor64<float, 5, at::RestrictPtrTraits> grad_out,
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> grad_features) {
  int batch_idx = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  int N = idxs.size(1);
  int C = grad_features.size(1);

  for (int i = index; i < N; i += stride) {
    int x = idxs[batch_idx][i][0];
    int y = idxs[batch_idx][i][1];
    int z = idxs[batch_idx][i][2];
    int cnt = counts[batch_idx][x][y][z];
    if (cnt > 0) {
      float inv_cnt = 1.0 / static_cast<float>(cnt);
      for (int c = 0; c < C; c++) {
        atomicAdd(
            &grad_features[batch_idx][c][i],
            grad_out[batch_idx][c][x][y][z] * inv_cnt);
      }
    }
  }
}

// TODO(arthurdjn) support all data types once `atomicAdd` supports doubles
// NOTE: The counts tensor must be of type int, not long, because of the `atomicAdd`
// function types overload (does not support (int64_t *, int) signature)
/**
 * @brief Average pool voxelization (backward)
 *
 * @param grad_out (B, C, R, R, R) tensor, gradients of each voxel grid
 * @param idxs (B, N, 3) tensor, voxel index of each point
 * @param counts (B, R, R, R) tensor, number of points in each voxel grid
 * @return at::Tensor, gradients of each point
 */
at::Tensor avg_voxelize_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& counts) {
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      counts_t{counts, "counts", 3};
  at::CheckedFrom c = "avg_voxelize_backward_cuda";
  at::checkAllSameGPU(c, {grad_out_t, idxs_t, counts_t});

  CHECK_IS_CONTIGUOUS(grad_out);
  CHECK_IS_CONTIGUOUS(idxs);
  CHECK_IS_CONTIGUOUS(counts);
  TORCH_CHECK(grad_out.dim() == 5, "grad_out must have 5 dimensions");
  TORCH_CHECK(idxs.dim() == 3, "idxs must have 3 dimensions");
  TORCH_CHECK(counts.dim() == 4, "counts must have 4 dimensions");

  int B = grad_out.size(0);
  int C = grad_out.size(1);
  int N = idxs.size(1);

  auto float_options = grad_out.options().dtype(at::kFloat);
  auto grad_features = torch::zeros({B, C, N}, float_options);

  avg_voxelize_backward_kernel<<<B, optimal_num_threads(N)>>>(
      idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
      counts.packed_accessor64<int, 4, at::RestrictPtrTraits>(),
      grad_out.packed_accessor64<float, 5, at::RestrictPtrTraits>(),
      grad_features.packed_accessor64<float, 3, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return grad_features;
}