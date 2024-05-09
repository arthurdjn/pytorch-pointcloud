#include <torch/extension.h>
#include <cmath>
#include <vector>

#include "../utils.h"

/**
 * @brief Interpolates the features of the input points to the output points
 *
 * @param points (B, M, 3) tensor, input points
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output point
 * clouds
 * @return (B, N, 3) tensor, interpolated points
 */
at::Tensor three_interpolate_cpu(
    const at::Tensor& points,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  // Ensure inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, lengths_t{lengths, "lengths", 4},
      out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "three_interpolate_cpu";
  at::checkAllSameType(c, {points_t, weights_t});
  at::checkAllSameType(c, {lengths_t, out_lengths_t});

  CHECK_CONTIGUOUS_CPU(points);
  CHECK_CONTIGUOUS_CPU(lengths);
  CHECK_CONTIGUOUS_CPU(idxs);
  CHECK_CONTIGUOUS_CPU(weights);

  TORCH_CHECK(points.dim() == 3, "points must be a tensor of shape (B, M, 3)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(weights.dim() == 3, "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be a tensor of shape (B,)");

  const int B = points.size(0);
  const int C = points.size(2);
  const int64_t M = points.size(1);
  const int64_t N = idxs.size(1);

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N && idxs.size(2) == 3,
      "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N && weights.size(2) == 3,
      "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  auto out = at::zeros({B, N, C}, points.options());

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "three_interpolate_cpu", ([&] {
        auto points_a = points.accessor<scalar_t, 3>();
        auto lengths_a = lengths.accessor<int64_t, 1>();
        auto out_lengths_a = out_lengths.accessor<int64_t, 1>();
        auto idxs_a = idxs.accessor<int64_t, 3>();
        auto weights_a = weights.accessor<scalar_t, 3>();
        auto out_a = out.accessor<scalar_t, 3>();

        for (int b = 0; b < B; ++b) {
          int64_t M_b = std::min(lengths_a[b], M);
          int64_t N_b = std::min(out_lengths_a[b], N);
          for (int j = 0; j < N_b; ++j) {
            for (int64_t c = 0; c < C; ++c) {
              // Get the three nearest neighbors
              int64_t i1 = idxs_a[b][j][0];
              int64_t i2 = idxs_a[b][j][1];
              int64_t i3 = idxs_a[b][j][2];

              // Get the three weights
              scalar_t w1 = weights_a[b][j][0];
              scalar_t w2 = weights_a[b][j][1];
              scalar_t w3 = weights_a[b][j][2];

              // Interpolate the points from the three nearest neighbors
              if (i1 >= 0 && i1 < M_b && i2 >= 0 && i2 < M_b && i3 >= 0 && i3 < M_b) {
                scalar_t c1 = points_a[b][i1][c];
                scalar_t c2 = points_a[b][i2][c];
                scalar_t c3 = points_a[b][i3][c];
                out_a[b][j][c] = c1 * w1 + c2 * w2 + c3 * w3;
              }
            }
          }
        }
      }));

  return out;
}

/**
 * @brief Backward pass for the three_interpolate function
 *
 * @param grad_out (B, N, C) tensor, previously computed gradients
 * @param idxs (B, N, 3) tensor, indices of the three nn used during the forward pass
 * @param weights (B, N, 3) tensor, weights for the three nn used during the forward pass
 * @param lengths (B,) tensor, containing the sizes (size < M) of the input point clouds
 * @param out_lengths (B,) tensor, containing the sizes (size < N) of the output point
 * clouds
 * @return (B, M, C) tensor, gradients with respect to the input points
 */
at::Tensor three_interpolate_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const at::Tensor& lengths,
    const at::Tensor& out_lengths) {
  // Ensure inputs are on the same device
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3}, lengths_t{lengths, "lengths", 4},
      out_lengths_t{out_lengths, "out_lengths", 5};
  at::CheckedFrom c = "three_interpolate_backward_cpu";
  at::checkAllSameType(c, {grad_out_t, weights_t});
  at::checkAllSameType(c, {lengths_t, out_lengths_t});

  CHECK_CONTIGUOUS_CPU(grad_out);
  CHECK_CONTIGUOUS_CPU(idxs);
  CHECK_CONTIGUOUS_CPU(weights);
  CHECK_CONTIGUOUS_CPU(lengths);
  CHECK_CONTIGUOUS_CPU(out_lengths);

  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be of shape (B, N, 3)");
  TORCH_CHECK(idxs.dim() == 3, "idxs must be of shape (B, N, 3)");
  TORCH_CHECK(weights.dim() == 3, "weights must be of shape (B, N, 3)");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be of shape (B,)");
  TORCH_CHECK(out_lengths.dim() == 1, "out_lengths must be of shape (B,)");

  const int B = grad_out.size(0);
  const int64_t N = grad_out.size(1);
  const int C = grad_out.size(2);
  const int64_t M = lengths.max().item<int64_t>();

  TORCH_CHECK(
      idxs.size(0) == B && idxs.size(1) == N && idxs.size(2) == 3,
      "idxs must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(
      weights.size(0) == B && weights.size(1) == N && weights.size(2) == 3,
      "weights must be a tensor of shape (B, N, 3)");
  TORCH_CHECK(lengths.size(0) == B, "lengths must be a tensor of shape (B,)");
  TORCH_CHECK(out_lengths.size(0) == B, "out_lengths must be a tensor of shape (B,)");

  // Initialize gradients w.r.t the input points
  auto grad_points = at::zeros({B, M, C}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "three_interpolate_backward_cpu", ([&] {
        auto grad_out_a = grad_out.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 3>();
        auto weights_a = weights.accessor<scalar_t, 3>();
        auto grad_points_a = grad_points.accessor<scalar_t, 3>();
        auto lengths_a = lengths.accessor<int64_t, 1>();
        auto out_lengths_a = out_lengths.accessor<int64_t, 1>();

        for (int b = 0; b < B; ++b) {
          const int64_t M_b = std::min(lengths_a[b], M);
          const int64_t N_b = std::min(out_lengths_a[b], N);
          for (int j = 0; j < N_b; ++j) {
            for (int c = 0; c < C; ++c) {
              // Access the indices for the three nearest neighbors
              int64_t i1 = idxs_a[b][j][0];
              int64_t i2 = idxs_a[b][j][1];
              int64_t i3 = idxs_a[b][j][2];

              // Access the weights for the three nearest neighbors
              scalar_t w1 = weights_a[b][j][0];
              scalar_t w2 = weights_a[b][j][1];
              scalar_t w3 = weights_a[b][j][2];

              // Update the gradients
              scalar_t grad = grad_out_a[b][j][c];
              if (i1 >= 0 && i1 < M_b)
                grad_points_a[b][i1][c] += grad * w1;
              if (i2 >= 0 && i2 < M_b)
                grad_points_a[b][i2][c] += grad * w2;
              if (i3 >= 0 && i3 < M_b)
                grad_points_a[b][i3][c] += grad * w3;
            }
          }
        }
      }));

  return grad_points;
}
