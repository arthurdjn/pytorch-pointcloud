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
 * using the k nearest neighbors
 *
 * @param points (B, M, C) tensor, input points
 * @param idxs (B, N, K) tensor, indices of the knn used during the forward pass
 * @param weights (B, N, K) tensor, weights for the knn used during the forward
 * pass
 * @param K (B,) tensor, number of neighbors to interpolate from
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input
 * point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @param out (B, N, C) tensor, interpolated points
 */
template <typename scalar_t>
__global__ void k_interpolate_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> out_lengths,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> out)
{
  const int b = blockIdx.x; // batch index
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;

  const int C = points.size(2);
  const int64_t M = points.size(1);
  const int64_t N = idxs.size(1);

  const int64_t M_b = std::min(M, lengths[b]);
  const int64_t N_b = std::min(N, out_lengths[b]);
  const int64_t K_b = std::min(N_b, K[b]);

  for (int i = index; i < N_b * C; i += stride)
  {
    const int j = i / C;
    const int c = i % C;

    // Interpolate from the K nearest neighbors
    scalar_t interpolated_value = 0;
    for (int k = 0; k < K_b; k++)
    {
      int ik = idxs[b][j][k];         // Index of the k-th nearest neighbor
      scalar_t wk = weights[b][j][k]; // Weight of the k-th nearest neighbor
      if (ik >= 0 && ik < M_b)
      {
        interpolated_value += points[b][ik][c] * wk;
      }
    }
    out[b][j][c] = interpolated_value;
  }
}

/**
 * @brief Interpolates the features of the input points to the output points
 *
 * @param points (B, M, C) tensor, input points
 * @param idxs (B, N, K) tensor, indices of the knn used during the forward pass
 * @param weights (B, N, K) tensor, weights for the knn used during the forward
 * pass
 * @param K (B,) tensor, number of neighbors to interpolate from
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input
 * point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @return (B, N, C) tensor, interpolated points
 */
at::Tensor k_interpolate_cuda(
    const at::Tensor &points,
    const at::Tensor &idxs,
    const at::Tensor &weights,
    const at::Tensor &K,
    const at::Tensor &lengths,
    const at::Tensor &out_lengths)
{
  at::TensorArg points_t{points, "points", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, K_t{K, "K", 4},
      lengths_t{lengths, "lengths", 4}, out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "k_interpolate_cuda";
  at::checkAllSameGPU(c, {points_t, idxs_t, weights_t, lengths_t, out_lengths_t});
  at::checkAllSameType(c, {points_t, weights_t});
  at::checkAllSameType(c, {K_t, lengths_t, out_lengths_t});

  CHECK_CONTIGUOUS_CUDA(points);
  CHECK_CONTIGUOUS_CUDA(idxs);
  CHECK_CONTIGUOUS_CUDA(weights);
  CHECK_CONTIGUOUS_CUDA(lengths);
  CHECK_CONTIGUOUS_CUDA(out_lengths);

  TORCH_CHECK(points.dim() == 3, "points must be a tensor of shape (B, M, C)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be a tensor of shape (B, N, K)");
  TORCH_CHECK(weights.dim() == 3, "weights must be a tensor of shape (B, N, K)");
  TORCH_CHECK(K.dim() == 1, "K must be a tensor of shape (B,)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be a tensor of shape (B,)");

  const int B = points.size(0);
  const int M = points.size(1);
  const int C = points.size(2);
  const int N = idxs.size(1);

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N,
      "idxs must be a tensor of shape (B, N, K)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N,
      "weights must be a tensor of shape (B, N, K)");
  TORCH_CHECK(K.size(0) == B, "K must be a tensor of shape (B,)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(
      out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  // Interpolated points
  auto out = at::zeros({B, N, C}, points.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "k_interpolate_cuda", ([&]
                                                   { k_interpolate_kernel<scalar_t><<<B, opt_block_config(N, C), 0, stream>>>(
                                                         points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                                                         idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
                                                         weights.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                                                         K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                         lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                         out_lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                         out.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>()); }));

  AT_CUDA_CHECK(cudaGetLastError());

  return out;
}

// TODO(arthurdjn) support all data types once AtomicAdd supports doubles
/**
 * @brief Backward pass for the k_interpolate function
 *
 * @param grad_out (B, N, C) tensor, gradients from the previously interpolated
 * points
 * @param idxs (B, N, K) tensor, indices of the knn used during the forward pass
 * @param weights (B, N, K) tensor, weights for the knn used during the forward
 * pass
 * @param K (B,) tensor, number of neighbors to interpolate from
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input
 * point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @param grad_points (B, M, C) tensor, gradients with respect to the input points
 */
__global__ void k_interpolate_backward_kernel(
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_out,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> out_lengths,
    at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_points)
{
  const int64_t C = grad_out.size(2);
  const int64_t N = grad_out.size(1);
  const int64_t M = grad_points.size(1);

  const int b = blockIdx.x; // batch index
  const int64_t M_b = std::min(M, lengths[b]);
  const int64_t N_b = std::min(N, out_lengths[b]);
  const int64_t K_b = std::min(N_b, K[b]);

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < N_b * C; i += stride)
  {
    const int j = i / C; // Output point index
    const int c = i % C; // Channel index

    for (int k = 0; k < K_b; k++)
    {
      int64_t ik = idxs[b][j][k];    // Index of the k-th nearest neighbor
      float_t wk = weights[b][j][k]; // Weight of the k-th nearest neighbor
      if (ik >= 0 && ik < M_b)
      {
        atomicAdd(&grad_points[b][ik][c], grad_out[b][j][c] * wk);
      }
    }
  }
}

// TODO(arthurdjn) use AT_DISPATCH_FLOATING_TYPES once AtomicAdd supports
// doubles
/**
 * @brief Backward pass for the k_interpolate function
 *
 * @param grad_out (B, N, C) tensor, previously computed gradients
 * @param idxs (B, N, K) tensor, indices of the knn used during the forward pass
 * @param weights (B, N, K) tensor, weights for the knn used during the forward
 * pass
 * @param K (B,) tensor, number of neighbors to interpolate from
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input
 * point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output
 * point clouds
 * @param M int, number of input points used during the forward pass
 * @return (B, M, C) tensor, gradients with respect to the input points
 */
at::Tensor k_interpolate_backward_cuda(
    const at::Tensor &grad_out,
    const at::Tensor &idxs,
    const at::Tensor &weights,
    const at::Tensor &K,
    const at::Tensor &lengths,
    const at::Tensor &out_lengths,
    const int64_t M)
{
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, K_t{K, "K", 4},
      lengths_t{lengths, "lengths", 4}, out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "k_interpolate_backward_cuda";
  at::checkAllSameGPU(c, {grad_out_t, idxs_t, weights_t, lengths_t, out_lengths_t});
  at::checkAllSameType(c, {grad_out_t, weights_t});
  at::checkAllSameType(c, {K_t, lengths_t, out_lengths_t});

  CHECK_CONTIGUOUS_CUDA(grad_out);
  CHECK_CONTIGUOUS_CUDA(idxs);
  CHECK_CONTIGUOUS_CUDA(weights);
  CHECK_CONTIGUOUS_CUDA(K);
  CHECK_CONTIGUOUS_CUDA(lengths);
  CHECK_CONTIGUOUS_CUDA(out_lengths);

  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be a tensor of shape (B, N, C)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be a tensor of shape (B, N, K)");
  TORCH_CHECK(weights.dim() == 3, "weights must be a tensor of shape (B, N, K)");
  TORCH_CHECK(K.dim() == 1, "K must be a tensor of shape (B,)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be a tensor of shape (B,)");

  const int B = grad_out.size(0);
  const int N = grad_out.size(1);
  const int C = grad_out.size(2);
  // const int64_t M = lengths.max().item<int64_t>();

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N,
      "idxs must be a tensor of shape (B, N, K)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N,
      "weights must be a tensor of shape (B, N, K)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(
      out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  // Initialize gradients w.r.t the input points
  auto grad_points = at::zeros({B, M, C}, grad_out.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  k_interpolate_backward_kernel<<<B, opt_block_config(N, C), 0, stream>>>(
      grad_out.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
      weights.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      out_lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      grad_points.packed_accessor64<float_t, 3, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());

  return grad_points;
}
