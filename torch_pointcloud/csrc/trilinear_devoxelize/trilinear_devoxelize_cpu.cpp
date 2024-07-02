#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

#include "../utils.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_cpu(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  at::TensorArg coords_t{coords, "coords", 1}, features_t{features, "features", 2};
  at::CheckedFrom c = "trilinear_devoxelize_cpu";
  at::checkAllSameType(c, {coords_t, features_t});

  int B = coords.size(0);
  int N = coords.size(1);
  int C = features.size(1);
  int R = resolution;

  auto long_opts = coords.options().dtype(at::kLong);
  auto idxs = at::zeros({B, N, 8, 3}, long_opts);
  auto weights = at::zeros({B, N, 8}, features.options());
  auto out = at::zeros({B, C, N}, features.options());

  AT_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "trilinear_devoxelize_cpu", ([&] {
        auto features_a = features.accessor<scalar_t, 5>();
        auto coords_a = coords.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 4>();
        auto weights_a = weights.accessor<scalar_t, 3>();
        auto out_a = out.accessor<scalar_t, 3>();

        for (int batch_idx = 0; batch_idx < B; ++batch_idx) {
          for (int i = 0; i < N; ++i) {
            scalar_t x = coords_a[batch_idx][i][0];
            scalar_t y = coords_a[batch_idx][i][1];
            scalar_t z = coords_a[batch_idx][i][2];

            // Compute the voxel cell coordinates of the 8 corners containing the
            // point
            float x_lo_f = floorf(x);
            float y_lo_f = floorf(y);
            float z_lo_f = floorf(z);

            int x_lo = static_cast<int>(x_lo_f);
            int y_lo = static_cast<int>(y_lo_f);
            int z_lo = static_cast<int>(z_lo_f);
            int x_hi = std::min(x_lo + 1, R - 1);
            int y_hi = std::min(y_lo + 1, R - 1);
            int z_hi = std::min(z_lo + 1, R - 1);

            scalar_t x_d_0 = x - x_lo_f;
            scalar_t y_d_0 = y - y_lo_f;
            scalar_t z_d_0 = z - z_lo_f;
            scalar_t x_d_1 = 1.0 - x_d_0;
            scalar_t y_d_1 = 1.0 - y_d_0;
            scalar_t z_d_1 = 1.0 - z_d_0;

            // Calculate indices and weights for the 8 neighboring voxels
            for (int dx = 0; dx < 2; ++dx) {
              for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                  int corner_idx = dx * 4 + dy * 2 + dz; // 0, 1, 2, 3, 4, 5, 6, 7
                  idxs_a[batch_idx][i][corner_idx][0] = (dx == 0) ? x_lo : x_hi;
                  idxs_a[batch_idx][i][corner_idx][1] = (dy == 0) ? y_lo : y_hi;
                  idxs_a[batch_idx][i][corner_idx][2] = (dz == 0) ? z_lo : z_hi;

                  scalar_t wx = (dx == 0) ? x_d_1 : x_d_0;
                  scalar_t wy = (dy == 0) ? y_d_1 : y_d_0;
                  scalar_t wz = (dz == 0) ? z_d_1 : z_d_0;
                  weights_a[batch_idx][i][corner_idx] = wx * wy * wz;
                }
              }
            }

            // Perform trilinear interpolation
            for (int c = 0; c < C; ++c) {
              scalar_t result = 0.0f;
              for (int j = 0; j < 8; ++j) {
                int xi = idxs_a[batch_idx][i][j][0];
                int yi = idxs_a[batch_idx][i][j][1];
                int zi = idxs_a[batch_idx][i][j][2];
                scalar_t weight = weights_a[batch_idx][i][j];
                result += weight * features_a[batch_idx][c][xi][yi][zi];
              }
              out_a[batch_idx][c][i] = result;
            }
          }
        }
      }));

  return std::make_tuple(out, idxs, weights);
}

at::Tensor trilinear_devoxelize_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& weights,
    const int resolution) {
  at::TensorArg grad_out_t{grad_out, "grad_out", 1}, idxs_t{idxs, "idxs", 2},
      weights_t{weights, "weights", 3};
  at::CheckedFrom c = "trilinear_devoxelize_backward_cpu";
  at::checkAllSameType(c, {grad_out_t, weights_t});

  int B = grad_out.size(0);
  int C = grad_out.size(1);
  int N = grad_out.size(2);
  int R = resolution;

  auto grad_features = torch::zeros({B, C, R, R, R}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "trilinear_devoxelize_backward_cpu", ([&] {
        auto grad_out_a = grad_out.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 4>();
        auto weights_a = weights.accessor<scalar_t, 3>();
        auto grad_features_a = grad_features.accessor<scalar_t, 5>();

        for (int b = 0; b < B; ++b) {
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              scalar_t grad = grad_out_a[b][c][n];

              // Loop over all corners of the voxel cube
              for (int corner_idx = 0; corner_idx < 8; ++corner_idx) {
                int64_t xi = idxs_a[b][n][corner_idx][0];
                int64_t yi = idxs_a[b][n][corner_idx][1];
                int64_t zi = idxs_a[b][n][corner_idx][2];
                scalar_t weight = weights_a[b][n][corner_idx];

                grad_features_a[b][c][xi][yi][zi] += weight * grad;
              }
            }
          }
        }
      }));

  return grad_features;
}
