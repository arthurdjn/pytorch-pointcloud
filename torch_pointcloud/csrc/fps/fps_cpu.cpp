#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

#include "../utils.h"

at::Tensor fps_cpu(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& num_samples,
    const at::Tensor& start_idxs) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t K = at::max(num_samples).item<int64_t>();

  // Initialize an output array for the sampled indices
  auto opts = lengths.options();
  torch::Tensor idxs = torch::full({B, K}, -1, opts);

  // Create accessors for all tensors
  auto points_a = points.accessor<float, 3>();
  auto lengths_a = lengths.accessor<int64_t, 1>();
  auto num_samples_a = num_samples.accessor<int64_t, 1>();
  auto idxs_a = idxs.accessor<int64_t, 2>();
  auto start_idxs_a = start_idxs.accessor<int64_t, 1>();

  // Initialize a mask to prevent duplicates
  // If true, the point has already been selected.
  std::vector<unsigned char> selected_points_mask(N, false);

  // Initialize to infinity a vector of
  // distances from each point to any of the previously selected points
  std::vector<float> dists(N, std::numeric_limits<float>::max());

  for (int64_t b = 0; b < B; ++b) {
    // Resize and reset points mask and distances for each batch
    selected_points_mask.resize(lengths_a[b]);
    dists.resize(lengths_a[b]);
    std::fill(selected_points_mask.begin(), selected_points_mask.end(), false);
    std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::max());

    // Initialize the first selected index based on start_idxs
    int64_t current_idx = start_idxs_a[b];
    selected_points_mask[current_idx] = true; // Mark as selected
    idxs_a[b][0] = current_idx; // Add to the sampled list

    // Number of points to sample in this batch
    const int64_t batch_k = std::min(lengths_a[b], num_samples_a[b]);

    // Start the furthest point sampling loop
    for (int64_t k = 1; k < batch_k; ++k) {
      // Update distances from the last added point to all others
      for (int64_t n = 0; n < lengths_a[b]; ++n) {
        if (!selected_points_mask[n]) {
          float dist = 0.0;
          for (int64_t c = 0; c < C; ++c) {
            float diff = points_a[b][current_idx][c] - points_a[b][n][c];
            dist += diff * diff;
          }
          dists[n] = std::min(dists[n], dist);
        }
      }

      // Select the next point - one with the maximum distance to the set of
      // selected points
      float max_dist = -std::numeric_limits<float>::max();
      for (int64_t n = 0; n < lengths_a[b]; ++n) {
        if (dists[n] > max_dist && !selected_points_mask[n]) {
          max_dist = dists[n];
          current_idx = n;
        }
      }

      // Add the selected point to the sampled indices
      idxs_a[b][k] = current_idx;
      selected_points_mask[current_idx] = true;
    }
  }

  return idxs;
}