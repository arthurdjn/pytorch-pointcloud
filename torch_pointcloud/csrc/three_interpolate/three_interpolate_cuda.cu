#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdlib.h>
#include <cmath>
#include <limits>
#include <vector>

#include "../cuda_utils.h"
#include "../utils.h"

/**
 * @brief Interpolates the features of the input points to the output points
 * using the three nearest neighbors
 *
 * @param points (B, M, 3) tensor, input points
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward
 * pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point
 * clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @param out (B, N, 3) tensor, interpolated points
 * @return (B, N, 3) tensor, interpolated points
 */
template <typename scalar_t>
__global__ void three_interpolate_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> out_lengths,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> out) {
  const int b = blockIdx.x; // batch index
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;

  const int C = points.size(2);
  const int64_t M = points.size(1);
  const int64_t N = idxs.size(1);

  const int64_t M_b = std::min(M, lengths[b]);
  const int64_t N_b = std::min(N, out_lengths[b]);

  for (int i = index; i < N_b * C; i += stride) {
    const int j = i / C; // Output point index
    const int c = i % C; // Channel index

    // Extract indices for the three nearest neighbors
    int i1 = idxs[b][j][0];
    int i2 = idxs[b][j][1];
    int i3 = idxs[b][j][2];

    // Extract weights for the three nearest neighbors
    scalar_t w1 = weights[b][j][0];
    scalar_t w2 = weights[b][j][1];
    scalar_t w3 = weights[b][j][2];

    // Perform the interpolation for the specified point and channel
    // Ensure indices are within fbounds
    if (i1 >= 0 && i1 < M_b && i2 >= 0 && i2 < M_b && i3 >= 0 && i3 < M_b) {
      scalar_t c1 = points[b][i1][c];
      scalar_t c2 = points[b][i2][c];
      scalar_t c3 = points[b][i3][c];
      out[b][j][c] = c1 * w1 + c2 * w2 + c3 * w3;
    }
  }
}

/**
 * @brief Interpolates the features of the input points to the output points
 *
 * @param points (B, M, 3) tensor, input points
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward
 * pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point
 * clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @return (B, N, 3) tensor, interpolated points
 */
at::Tensor three_interpolate_cuda(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  at::TensorArg points_t{points, "points", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, lengths_t{lengths, "lengths", 4},
      out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "three_interpolate_cuda";
  at::checkAllSameGPU(c, {points_t, idxs_t, weights_t, lengths_t, out_lengths_t});
  at::checkAllSameType(c, {points_t, weights_t});
  at::checkAllSameType(c, {lengths_t, out_lengths_t});

  CHECK_IS_CONTIGUOUS_CUDA(points);
  CHECK_IS_CONTIGUOUS_CUDA(idxs);
  CHECK_IS_CONTIGUOUS_CUDA(weights);
  CHECK_IS_CONTIGUOUS_CUDA(lengths);
  CHECK_IS_CONTIGUOUS_CUDA(out_lengths);

  TORCH_CHECK(points.dim() == 3, "points must be a tensor of shape (B, M, 3)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(weights.dim() == 3, "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be a tensor of shape (B,)");

  const int B = points.size(0);
  const int M = points.size(1);
  const int C = points.size(2);
  const int N = idxs.size(1);

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N && idxs.size(2) == 3,
      "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N && weights.size(2) == 3,
      "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(
      out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  // Interpolated points
  auto out = at::zeros({B, N, C}, points.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "three_interpolate_cuda", ([&] {
        three_interpolate_kernel<scalar_t><<<B, opt_block_config(N, C), 0, stream>>>(
            points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            out_lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            out.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  return out;
}

// TODO(arthurdjn) support all data types once `atomicAdd` supports doubles
/**
 * @brief Backward pass for the three_interpolate function
 *
 * @param grad_out (B, N, C) tensor, gradients from the previously interpolated
 * points
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward
 * pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point
 * clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @return (B, M, C) tensor, gradients with respect to the input points
 */
__global__ void three_interpolate_backward_kernel(
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_out,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> out_lengths,
    at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_points) {
  const int64_t C = grad_out.size(2);
  const int64_t N = grad_out.size(1);
  const int64_t M = grad_points.size(1);

  const int b = blockIdx.x; // batch index
  const int64_t M_b = std::min(M, lengths[b]);
  const int64_t N_b = std::min(N, out_lengths[b]);

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;

  for (int i = index; i < N_b * C; i += stride) {
    const int j = i / C;
    const int c = i % C;

    // Access the indices for the three nearest neighbors
    int64_t i1 = idxs[b][j][0];
    int64_t i2 = idxs[b][j][1];
    int64_t i3 = idxs[b][j][2];

    // Access the weights for the three nearest neighbors
    float_t w1 = weights[b][j][0];
    float_t w2 = weights[b][j][1];
    float_t w3 = weights[b][j][2];

    // Update the gradients
    float_t grad = grad_out[b][j][c];
    if (i1 >= 0 && i1 < M_b)
      atomicAdd(&grad_points[b][i1][c], grad * w1);
    if (i2 >= 0 && i2 < M_b)
      atomicAdd(&grad_points[b][i2][c], grad * w2);
    if (i3 >= 0 && i3 < M_b)
      atomicAdd(&grad_points[b][i3][c], grad * w3);
  }
}

// TODO(arthurdjn) use AT_DISPATCH_FLOATING_TYPES once `atomicAdd` supports doubles
/**
 * @brief Backward pass for the three_interpolate function
 *
 * @param grad_out (B, N, C) tensor, previously computed gradients
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward
 * pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point
 * clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @return (B, M, C) tensor, gradients with respect to the input points
 */
at::Tensor three_interpolate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, lengths_t{lengths, "lengths", 4},
      out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "three_interpolate_backward_cuda";
  at::checkAllSameGPU(c, {grad_out_t, idxs_t, weights_t, lengths_t, out_lengths_t});
  at::checkAllSameType(c, {grad_out_t, weights_t});
  at::checkAllSameType(c, {lengths_t, out_lengths_t});

  CHECK_IS_CONTIGUOUS_CUDA(grad_out);
  CHECK_IS_CONTIGUOUS_CUDA(idxs);
  CHECK_IS_CONTIGUOUS_CUDA(weights);
  CHECK_IS_CONTIGUOUS_CUDA(lengths);
  CHECK_IS_CONTIGUOUS_CUDA(out_lengths);

  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be a tensor of shape (B, N, C)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(weights.dim() == 3, "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be a tensor of shape (B,)");

  const int B = grad_out.size(0);
  const int N = grad_out.size(1);
  const int C = grad_out.size(2);
  const int64_t M = lengths.max().item<int64_t>();

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N && idxs.size(2) == 3,
      "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N && weights.size(2) == 3,
      "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(
      out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  // Initialize gradients w.r.t the input points
  auto grad_points = at::zeros({B, M, C}, grad_out.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_interpolate_backward_kernel<<<B, opt_block_config(N, C), 0, stream>>>(
      grad_out.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
      weights.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      out_lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      grad_points.packed_accessor64<float_t, 3, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_points;
}
