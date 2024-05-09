#include <torch/extension.h>
#include <limits>
#include <vector>

std::tuple<at::Tensor, at::Tensor> knn_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& K,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  const int B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);
  const int64_t K_max = at::max(K).item<int64_t>();

  auto inf = std::numeric_limits<float>::infinity();
  auto long_opts = lengths1.options().dtype(at::kLong);
  auto dists = at::full({B, N1, K_max}, inf, pc1.options());
  auto idxs = at::full({B, N1, K_max}, -1, long_opts);

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "knn_cpu", ([&] {
        auto pc1_a = pc1.accessor<scalar_t, 3>();
        auto pc2_a = pc2.accessor<scalar_t, 3>();
        auto K_a = K.accessor<int64_t, 1>();
        auto lengths1_a = lengths1.accessor<int64_t, 1>();
        auto lengths2_a = lengths2.accessor<int64_t, 1>();
        auto dists_a = dists.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 3>();

        for (int b = 0; b < B; ++b) {
          const int64_t N1_b = std::min(lengths1_a[b], N1);
          const int64_t N2_b = std::min(lengths2_a[b], N2);
          const int64_t K_b = std::min(K_a[b], N2_b);

          for (int i1 = 0; i1 < N1_b; ++i1) {
            for (int i2 = 0; i2 < N2_b; ++i2) {
              scalar_t dist = 0;
              for (int c = 0; c < C; ++c) {
                scalar_t diff = pc1_a[b][i1][c] - pc2_a[b][i2][c];
                dist += diff * diff;
              }

              for (int k = 0; k < K_b; ++k) {
                if (dist < dists_a[b][i1][k]) {
                  // Keep the neighbors sorted
                  for (int m = K_b - 1; m > k; --m) {
                    dists_a[b][i1][m] = dists_a[b][i1][m - 1]; // Sort distances
                    idxs_a[b][i1][m] = idxs_a[b][i1][m - 1]; // Sort indices
                  }
                  dists_a[b][i1][k] = dist;
                  idxs_a[b][i1][k] = i2;
                  break;
                }
              }
            }
          }
        }
      }));

  return std::make_tuple(dists, idxs);
}
