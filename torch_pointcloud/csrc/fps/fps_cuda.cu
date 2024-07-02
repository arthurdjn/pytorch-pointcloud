#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <tuple>

#include "../cuda_utils.h"

// points: (B, N, C)
// lengths: (B,), with values in [0, N]
// num_samples: (B,), with values in [0, N]
// dists: (B, N)
// idxs: (B, K) with values in [0, N]
template <unsigned int block_size>
__global__ void fps_kernel(
    const at::PackedTensorAccessor64<float, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_samples,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> start_idxs,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> dists,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs) {
  __shared__ int shared_idxs[block_size];
  __shared__ float shared_dists[block_size];

  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int C = points.size(2);

  int tid = threadIdx.x;
  const int b = blockIdx.x;
  const int64_t n = std::min(lengths[b], N);
  const int64_t k = std::min(num_samples[b], n);

  if (k <= 0 || k > n) {
    return;
  }

  int selected_idx = start_idxs[b];
  // Init the farthest point to the thread 0
  if (tid == 0)
    idxs[b][0] = selected_idx;

  __syncthreads();
  for (int i = 1; i < k; i++) {
    int best_idx = 0;
    float best_dist = -1;

    for (int idx = tid; idx < n; idx += block_size) {
      float dist = 0;
      for (int c = 0; c < C; c++) {
        float diff = points[b][idx][c] - points[b][selected_idx][c];
        dist += diff * diff;
      }

      dist = min(dist, dists[b][idx]);
      dists[b][idx] = dist;
      if (dist > best_dist) {
        best_idx = idx;
        best_dist = dist;
      }
    }

    shared_dists[tid] = best_dist;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    // Aggregate results with shared memory
    for (unsigned int block = block_size / 2; block > 0; block >>= 1) {
      if (tid < block) {
        if (shared_dists[tid + block] > shared_dists[tid]) {
          shared_dists[tid] = shared_dists[tid + block];
          shared_idxs[tid] = shared_idxs[tid + block];
        }
      }
      __syncthreads();
    }

    // Add the index of the farthest point to the thread 0
    selected_idx = shared_idxs[0];
    if (tid == 0) {
      idxs[b][i] = selected_idx;
    }
  }
}

at::Tensor fps_cuda(
    const at::Tensor& points,
    const at::Tensor& lengths,
    const at::Tensor& num_samples,
    const at::Tensor& start_idxs) {
  at::TensorArg points_t{points, "points", 1}, lengths_t{lengths, "lengths", 1},
      num_samples_t{num_samples, "num_samples", 1};
  at::CheckedFrom c = "fps_cuda";
  at::checkAllSameGPU(c, {points_t, lengths_t, num_samples_t});
  at::checkAllSameType(c, {lengths_t, num_samples_t});

  const int B = points.size(0);
  const int N = points.size(1);
  const int K = num_samples.max().item<int>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);
  auto inf = std::numeric_limits<float>::infinity();
  auto dists = at::full({B, N}, inf, opts);
  auto idxs = at::full({B, K}, -1, long_opts);

  auto points_a = points.packed_accessor64<float, 3, at::RestrictPtrTraits>();
  auto lengths_a = lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto num_samples_a =
      num_samples.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto start_idxs_a =
      start_idxs.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>();
  auto dists_a = dists.packed_accessor64<float, 2, at::RestrictPtrTraits>();
  auto idxs_a = idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>();

  unsigned int threads = optimal_num_threads(N);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (threads) {
    case 1024:
      fps_kernel<1024><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 512:
      fps_kernel<512><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 256:
      fps_kernel<256><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 128:
      fps_kernel<128><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 64:
      fps_kernel<64><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 32:
      fps_kernel<32><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 16:
      fps_kernel<16><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 8:
      fps_kernel<8><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 4:
      fps_kernel<4><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 2:
      fps_kernel<2><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    case 1:
      fps_kernel<1><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
      break;
    default:
      fps_kernel<512><<<B, threads, 0, stream>>>(
          points_a, lengths_a, num_samples_a, start_idxs_a, dists_a, idxs_a);
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return idxs;
}