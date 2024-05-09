#include <torch/extension.h>
#include <limits>
#include <tuple>
#include "../utils.h"

std::tuple<at::Tensor, at::Tensor> sided_distance_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  CHECK_CPU(pc1);
  CHECK_CPU(pc2);
  CHECK_CPU(lengths1);
  CHECK_CPU(lengths2);

  AT_ASSERTM(
      pc1.dim() == 3 && pc2.dim() == 3, "pc1 and pc2 must have the shape [B, N, C]");
  AT_ASSERTM(
      lengths1.dim() == 1 && lengths2.dim() == 1,
      "lengths1 and lengths2 must have the shape [B]");

  const auto B = pc1.size(0);
  const auto N1 = pc1.size(1);
  const auto N2 = pc2.size(1);
  const auto C = pc1.size(2);

  auto long_options = lengths1.options().dtype(at::kLong);
  auto dists = at::full({B, N1}, 0, pc1.options());
  auto idxs = at::full({B, pc1.size(1)}, -1, long_options);

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "sided_distance_cpu", ([&] {
        auto pc1_a = pc1.accessor<scalar_t, 3>();
        auto pc2_a = pc2.accessor<scalar_t, 3>();
        auto lengths1_a = lengths1.accessor<int64_t, 1>();
        auto lengths2_a = lengths2.accessor<int64_t, 1>();
        auto dists_a = dists.accessor<scalar_t, 2>();
        auto idxs_a = idxs.accessor<int64_t, 2>();

        for (int64_t b = 0; b < B; ++b) {
          const auto N1_b = std::min(N1, lengths1_a[b]);
          const auto N2_b = std::min(N2, lengths2_a[b]);
          for (int64_t i1 = 0; i1 < N1_b; ++i1) {
            scalar_t min_dist = std::numeric_limits<scalar_t>::infinity();

            for (int64_t i2 = 0; i2 < N2_b; ++i2) {
              scalar_t dist = 0.0;
              for (int64_t c = 0; c < C; ++c) {
                scalar_t diff = pc1_a[b][i1][c] - pc2_a[b][i2][c];
                dist += diff * diff;
              }

              if (dist < min_dist) {
                min_dist = dist;
                dists_a[b][i1] = dist;
                idxs_a[b][i1] = i2;
              }
            }
          }
        }
      }));

  return std::make_tuple(dists, idxs);
}

std::tuple<at::Tensor, at::Tensor> sided_distance_backward_cpu(
    const at::Tensor& grad_dists,
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& idxs,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  CHECK_CPU(grad_dists);
  CHECK_CPU(pc1);
  CHECK_CPU(pc2);
  CHECK_CPU(idxs);
  CHECK_CPU(lengths1);
  CHECK_CPU(lengths2);

  AT_ASSERTM(
      pc1.dim() == 3 && pc2.dim() == 3, "pc1 and pc2 must have the shape (B, N, C)");
  AT_ASSERTM(idxs.dim() == 2, "idxs must have the shape (B, N)");
  AT_ASSERTM(lengths1.dim() == 1, "lengths must have the shape (B,)");
  AT_ASSERTM(lengths2.dim() == 1, "lengths must have the shape (B,)");

  const auto B = pc1.size(0);
  const auto N1 = pc1.size(1);
  const auto N2 = pc2.size(1);
  const auto C = pc1.size(2);

  auto grad_pc1 = at::zeros_like(pc1);
  auto grad_pc2 = at::zeros_like(pc2);

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "sided_distance_backward_cpu", ([&] {
        auto pc1_a = pc1.accessor<scalar_t, 3>();
        auto pc2_a = pc2.accessor<scalar_t, 3>();
        auto grad_dists_a = grad_dists.accessor<scalar_t, 2>();
        auto idxs_a = idxs.accessor<int64_t, 2>();
        auto lengths1_a = lengths1.accessor<int64_t, 1>();
        auto lengths2_a = lengths2.accessor<int64_t, 1>();
        auto grad_pc1_a = grad_pc1.accessor<scalar_t, 3>();
        auto grad_pc2_a = grad_pc2.accessor<scalar_t, 3>();

        for (int64_t b = 0; b < B; ++b) {
          int64_t N1_b = std::min(N1, lengths1_a[b]);
          int64_t N2_b = std::min(N2, lengths2_a[b]);

          for (int64_t i1 = 0; i1 < N1_b; ++i1) {
            int64_t i2 = idxs_a[b][i1];
            if (i2 >= 0 && i2 < N2_b) {
              for (int64_t c = 0; c < C; ++c) {
                scalar_t diff = pc1_a[b][i1][c] - pc2_a[b][i2][c];
                scalar_t grad = grad_dists_a[b][i1];

                grad_pc1_a[b][i1][c] += 2 * diff * grad;
                grad_pc2_a[b][i2][c] -= 2 * diff * grad;
              }
            }
          }
        }
      }));

  return std::make_tuple(grad_pc1, grad_pc2);
}
