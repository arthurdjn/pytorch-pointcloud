#include <torch/extension.h>
#include <queue>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> ball_query_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& max_neighbors,
    const at::Tensor& radiuses) {
  const int B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int C = pc1.size(2);
  const int64_t K = max_neighbors.max().item<int64_t>();

  auto long_opts = lengths1.options().dtype(torch::kInt64);
  auto idxs = torch::full({B, N1, K}, -1, long_opts);
  auto dists = torch::full({B, N1, K}, 0, pc1.options());

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "ball_query_cpu", ([&] {
        auto pc1_a = pc1.accessor<scalar_t, 3>();
        auto pc2_a = pc2.accessor<scalar_t, 3>();
        auto lengths1_a = lengths1.accessor<int64_t, 1>();
        auto lengths2_a = lengths2.accessor<int64_t, 1>();
        auto max_neighbors_a = max_neighbors.accessor<int64_t, 1>();
        auto radiuses_a = radiuses.accessor<scalar_t, 1>();
        auto idxs_a = idxs.accessor<int64_t, 3>();
        auto dists_a = dists.accessor<scalar_t, 3>();

        for (int b = 0; b < B; ++b) {
          const int64_t n1 = std::min(N1, lengths1_a[b]);
          const int64_t n2 = std::min(N2, lengths2_a[b]);
          const int64_t k = max_neighbors_a[b];
          const scalar_t radius2 = radiuses_a[b] * radiuses_a[b];

          for (int64_t i1 = 0; i1 < n1; ++i1) {
            for (int64_t i2 = 0, count = 0; i2 < n2 && count < k; ++i2) {
              scalar_t dist2 = 0;
              for (int c = 0; c < C; ++c) {
                scalar_t diff = pc1_a[b][i1][c] - pc2_a[b][i2][c];
                dist2 += diff * diff;
              }
              if (dist2 < radius2) {
                dists_a[b][i1][count] = dist2;
                idxs_a[b][i1][count] = i2;
                ++count;
              }
            }
          }
        }
      }));

  return std::make_tuple(dists, idxs);
}
