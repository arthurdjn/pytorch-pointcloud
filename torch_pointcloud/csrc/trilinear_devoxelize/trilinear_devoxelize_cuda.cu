#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

#include "../cuda_utils.h"
#include "../utils.h"

/**
 * @brief Trilinear devoxelization kernel (forward)
 *
 * @param coords (B, N, 3) tensor, coordinates of each point
 * @param features (B, C, R, R, R) tensor, features of each voxel grid
 * @param idxs (B, N, 8, 3) tensor, voxel idxs of each point cube
 * @param weights (B, N, 8) tensor, weights for trilinear interpolation
 * @param out (B, C, N) tensor, features of each point
 */
__global__ void trilinear_devoxelize_kernel(
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> coords,
    at::PackedTensorAccessor64<float, 5, at::RestrictPtrTraits> features,
    at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> weights,
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> out) {
  int batch_idx = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  int N = coords.size(1);
  int C = features.size(1);
  int R = features.size(2);

  for (int i = index; i < N; i += stride) {
    float x = coords[batch_idx][i][0];
    float y = coords[batch_idx][i][1];
    float z = coords[batch_idx][i][2];

    // Compute the voxel cell coordinates of the 8 corners containing the point
    float x_lo_f = floorf(x);
    float y_lo_f = floorf(y);
    float z_lo_f = floorf(z);

    int x_lo = static_cast<int>(x_lo_f);
    int y_lo = static_cast<int>(y_lo_f);
    int z_lo = static_cast<int>(z_lo_f);
    int x_hi = std::min(x_lo + 1, R - 1);
    int y_hi = std::min(y_lo + 1, R - 1);
    int z_hi = std::min(z_lo + 1, R - 1);

    float x_d_0 = x - x_lo_f; // Distance to the higher x boundary
    float y_d_0 = y - y_lo_f; // Distance to the higher y boundary
    float z_d_0 = z - z_lo_f; // Distance to the higher z boundary
    float x_d_1 = 1.0f - x_d_0; // Distance to the lower x boundary
    float y_d_1 = 1.0f - y_d_0; // Distance to the lower y boundary
    float z_d_1 = 1.0f - z_d_0; // Distance to the lower z boundary

    // Calculate indices and weights for the 8 neighboring voxels
    for (int dx = 0; dx < 2; ++dx) {
      for (int dy = 0; dy < 2; ++dy) {
        for (int dz = 0; dz < 2; ++dz) {
          int corner_idx = dx * 4 + dy * 2 + dz; // 0, 1, 2, 3, 4, 5, 6, 7
          idxs[batch_idx][i][corner_idx][0] = (dx == 0) ? x_lo : x_hi;
          idxs[batch_idx][i][corner_idx][1] = (dy == 0) ? y_lo : y_hi;
          idxs[batch_idx][i][corner_idx][2] = (dz == 0) ? z_lo : z_hi;

          float wx = (dx == 0) ? x_d_1 : x_d_0;
          float wy = (dy == 0) ? y_d_1 : y_d_0;
          float wz = (dz == 0) ? z_d_1 : z_d_0;
          weights[batch_idx][i][corner_idx] = wx * wy * wz;
        }
      }
    }

    // Perform trilinear interpolation
    for (int c = 0; c < C; c++) {
      float result = 0.0f;
      for (int j = 0; j < 8; j++) {
        int64_t xi = idxs[batch_idx][i][j][0];
        int64_t yi = idxs[batch_idx][i][j][1];
        int64_t zi = idxs[batch_idx][i][j][2];
        float weight = weights[batch_idx][i][j];

        result += weight * features[batch_idx][c][xi][yi][zi];
      }
      out[batch_idx][c][i] = result;
    }
  }
}

/**
 * @brief Trilinear devoxelization (forward)
 *
 * @param coords (B, N, 3) tensor, coordinates of each point
 * @param features (B, C, R, R, R) tensor, features of each voxel grid
 * @param resolution int, voxel resolution R
 * @return std::tuple<at::Tensor, at::Tensor, at::Tensor>
 *  out (B, C, N) tensor, features of each point
 *  idxs (B, N, 8, 3) tensor, voxel idxs of each point cube corners
 *  weights (B, N, 8) tensor, weights for trilinear interpolation
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_cuda(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  at::TensorArg coords_t{coords, "coords", 1}, features_t{features, "features", 2};
  at::CheckedFrom c = "trilinear_devoxelize_cuda";
  at::checkAllSameGPU(c, {coords_t, features_t});
  at::checkAllSameType(c, {coords_t, features_t});

  CHECK_IS_CONTIGUOUS(coords);
  CHECK_IS_CONTIGUOUS(features);
  CHECK_IS_FLOAT(coords);
  CHECK_IS_FLOAT(features);

  TORCH_CHECK(coords.dim() == 3, "coords must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(
      features.dim() == 5, "features must be a tensor of shape (B, C, R, R, R)");

  int B = features.size(0);
  int C = features.size(1);
  int N = coords.size(1);

  auto long_opts = features.options().dtype(at::kLong);
  auto idxs = torch::zeros({B, N, 8, 3}, long_opts);
  auto weights = torch::zeros({B, N, 8}, features.options());
  auto out = torch::zeros({B, C, N}, features.options());

  trilinear_devoxelize_kernel<<<B, optimal_num_threads(N)>>>(
      coords.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      features.packed_accessor64<float, 5, at::RestrictPtrTraits>(),
      idxs.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>(),
      weights.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      out.packed_accessor64<float, 3, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(out, idxs, weights);
}

/**
 * @brief Trilinear devoxelization kernel (backward)
 *
 * @param idxs (B, N, 8, 3) tensor, voxel idxs of each point cube corners
 * @param weights (B, N, 8) tensor, weights for trilinear interpolation
 * @param grad_out (B, C, N) tensor, gradients of each point
 * @param grad_features (B, C, R, R, R) tensor, gradients of each voxel grid
 */
__global__ void trilinear_devoxelize_backward_kernel(
    at::PackedTensorAccessor64<int64_t, 4, at::RestrictPtrTraits> idxs,
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> weights,
    at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> grad_out,
    at::PackedTensorAccessor64<float, 5, at::RestrictPtrTraits> grad_features) {
  int batch_idx = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  int C = grad_out.size(1);
  int N = grad_out.size(2);

  for (int point_idx = index; point_idx < N; point_idx += stride) {
    for (int c = 0; c < C; ++c) {
      float grad = grad_out[batch_idx][c][point_idx];
      for (int corner_idx = 0; corner_idx < 8; ++corner_idx) {
        int64_t xi = idxs[batch_idx][point_idx][corner_idx][0];
        int64_t yi = idxs[batch_idx][point_idx][corner_idx][1];
        int64_t zi = idxs[batch_idx][point_idx][corner_idx][2];
        float weight = weights[batch_idx][point_idx][corner_idx];

        atomicAdd(&grad_features[batch_idx][c][xi][yi][zi], weight * grad);
      }
    }
  }
}

// TODO(arthurdjn) support all data types once `atomicAdd` supports doubles
/**
 * @brief Trilinear devoxelization (backward)
 *
 * @param grad_out (B, C, N) tensor, gradients of each point
 * @param idxs (B, N, 8, 3) tensor, voxel idxs of each point cube corners
 * @param weights (B, N, 8) tensor, weights for trilinear interpolation
 * @param resolution int, voxel resolution R
 * @return grad_features (B, C, R, R, R) tensor, gradients of each voxel
 * grid
 */
at::Tensor trilinear_devoxelize_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const int resolution) {
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3};
  at::CheckedFrom c = "trilinear_devoxelize_backward_cuda";
  at::checkAllSameGPU(c, {grad_out_t, weights_t});
  at::checkAllSameType(c, {grad_out_t, weights_t});

  CHECK_IS_CONTIGUOUS(grad_out);
  CHECK_IS_CONTIGUOUS(idxs);
  CHECK_IS_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_FLOAT(weights);

  int B = grad_out.size(0);
  int C = grad_out.size(1);
  int N = grad_out.size(2);
  int R = resolution;

  auto grad_features = torch::zeros({B, C, R, R, R}, grad_out.options());

  trilinear_devoxelize_backward_kernel<<<B, optimal_num_threads(N)>>>(
      idxs.packed_accessor64<int64_t, 4, at::RestrictPtrTraits>(),
      weights.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      grad_out.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
      grad_features.packed_accessor64<float, 5, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return grad_features;
}
