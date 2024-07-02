#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

#include "../utils.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> avg_voxelize_cpu(
    const at::Tensor& coords,
    const at::Tensor& features,
    const int resolution) {
  CHECK_IS_CONTIGUOUS(features);
  CHECK_IS_CONTIGUOUS(coords);
  TORCH_CHECK(features.dim() == 3, "features must have 3 dimensions");
  TORCH_CHECK(coords.dim() == 3, "coords must have 3 dimensions");

  int B = features.size(0);
  int C = features.size(1);
  int N = features.size(2);
  int R = resolution;

  auto long_opts = features.options().dtype(at::kLong);
  auto idxs = at::zeros({B, N, 3}, long_opts);
  auto counts = at::zeros({B, R, R, R}, long_opts);
  auto out = at::zeros({B, C, R, R, R}, features.options());

  AT_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "avg_voxelize_cpu", ([&] {
        auto coords_a = coords.accessor<int64_t, 3>();
        auto features_a = features.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 3>();
        auto counts_a = counts.accessor<int64_t, 4>();
        auto out_a = out.accessor<scalar_t, 5>();

        for (int b = 0; b < B; b++) {
          for (int i = 0; i < N; i++) {
            int x = coords_a[b][i][0];
            int y = coords_a[b][i][1];
            int z = coords_a[b][i][2];

            idxs_a[b][i][0] = x; // x-coordinate of the voxel cell
            idxs_a[b][i][1] = y; // y-coordinate of the voxel cell
            idxs_a[b][i][2] = z; // z-coordinate of the voxel cell

            counts_a[b][x][y][z] += 1;

            for (int c = 0; c < C; c++) {
              out_a[b][c][x][y][z] += features_a[b][c][i];
            }
          }
        }

        // Normalize the output by the counts
        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            for (int x = 0; x < R; x++) {
              for (int y = 0; y < R; y++) {
                for (int z = 0; z < R; z++) {
                  if (counts_a[b][x][y][z] > 0) {
                    out_a[b][c][x][y][z] /= counts_a[b][x][y][z];
                  }
                }
              }
            }
          }
        }
      }));

  return std::make_tuple(out, idxs, counts);
}

at::Tensor avg_voxelize_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& idxs,
    const at::Tensor& counts) {
  CHECK_IS_CONTIGUOUS(grad_out);
  CHECK_IS_CONTIGUOUS(idxs);
  CHECK_IS_CONTIGUOUS(counts);
  TORCH_CHECK(grad_out.dim() == 5, "grad_out must have 5 dimensions");
  TORCH_CHECK(idxs.dim() == 3, "idxs must have 3 dimensions");
  TORCH_CHECK(counts.dim() == 4, "counts must have 4 dimensions");

  int B = grad_out.size(0);
  int C = grad_out.size(1);
  int N = idxs.size(1);

  auto grad_features = torch::zeros({B, C, N}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "avg_voxelize_backward_cpu", ([&] {
        auto grad_out_a = grad_out.accessor<scalar_t, 5>();
        auto idxs_a = idxs.accessor<int64_t, 3>();
        auto counts_a = counts.accessor<int64_t, 4>();
        auto grad_features_a = grad_features.accessor<scalar_t, 3>();

        for (int b = 0; b < B; b++) {
          for (int i = 0; i < N; i++) {
            int x = idxs_a[b][i][0];
            int y = idxs_a[b][i][1];
            int z = idxs_a[b][i][2];
            int64_t cnt = counts_a[b][x][y][z];

            if (cnt > 0) {
              scalar_t inv_cnt = 1.0 / static_cast<scalar_t>(cnt);
              for (int c = 0; c < C; c++) {
                grad_features_a[b][c][i] += grad_out_a[b][c][x][y][z] * inv_cnt;
              }
            }
          }
        }
      }));

  return grad_features;
}