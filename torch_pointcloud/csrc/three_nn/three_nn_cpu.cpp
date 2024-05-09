#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

std::tuple<at::Tensor, at::Tensor> three_nn_cpu(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  const int B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);

  auto opts = pc1.options();
  auto long_opts = lengths1.options().dtype(at::kLong);
  auto inf = std::numeric_limits<float>::infinity();
  auto dists = at::full({B, N1, 3}, inf, opts);
  auto idxs = at::full({B, N1, 3}, -1, long_opts);

  auto lengths1_a = lengths1.accessor<int64_t, 1>();
  auto lengths2_a = lengths2.accessor<int64_t, 1>();

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "three_nn_cpu", ([&] {
        auto pc1_a = pc1.accessor<scalar_t, 3>();
        auto pc2_a = pc2.accessor<scalar_t, 3>();
        auto dists_a = dists.accessor<scalar_t, 3>();
        auto idxs_a = idxs.accessor<int64_t, 3>();

        for (int b = 0; b < B; ++b) {
          int n1 = std::min(lengths1_a[b], N1);
          int n2 = std::min(lengths2_a[b], N2);

          for (int i1 = 0; i1 < n1; ++i1) {
            scalar_t min_dist1 = std::numeric_limits<scalar_t>::infinity();
            scalar_t min_dist2 = std::numeric_limits<scalar_t>::infinity();
            scalar_t min_dist3 = std::numeric_limits<scalar_t>::infinity();
            int64_t besti1 = -1, besti2 = -1, besti3 = -1;

            for (int i2 = 0; i2 < n2; ++i2) {
              scalar_t dist = 0;
              for (int c = 0; c < C; ++c) {
                scalar_t diff = pc1_a[b][i1][c] - pc2_a[b][i2][c];
                dist += diff * diff;
              }

              if (dist < min_dist1) {
                min_dist3 = min_dist2;
                besti3 = besti2;
                min_dist2 = min_dist1;
                besti2 = besti1;
                min_dist1 = dist;
                besti1 = i2;
              } else if (dist < min_dist2) {
                min_dist3 = min_dist2;
                besti3 = besti2;
                min_dist2 = dist;
                besti2 = i2;
              } else if (dist < min_dist3) {
                min_dist3 = dist;
                besti3 = i2;
              }

              dists_a[b][i1][0] = min_dist1;
              dists_a[b][i1][1] = min_dist2;
              dists_a[b][i1][2] = min_dist3;

              idxs_a[b][i1][0] = besti1;
              idxs_a[b][i1][1] = besti2;
              idxs_a[b][i1][2] = besti3;
            }
          }
        }
      }));

  return std::make_tuple(dists, idxs);
}
