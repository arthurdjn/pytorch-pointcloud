#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

template <typename scalar_t>
__global__ void three_nn_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> dists,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs) {
  const int b = blockIdx.x; // batch index
  const int i1 = blockIdx.y * blockDim.x + threadIdx.x;

  const int64_t B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);

  const int64_t n1 = std::min(lengths1[b], N1);
  const int64_t n2 = std::min(lengths2[b], N2);

  if (b >= B || i1 < 0 || i1 >= n1)
    return;

  scalar_t min_dist1 = std::numeric_limits<scalar_t>::infinity();
  scalar_t min_dist2 = std::numeric_limits<scalar_t>::infinity();
  scalar_t min_dist3 = std::numeric_limits<scalar_t>::infinity();
  int64_t besti1 = -1, besti2 = -1, besti3 = -1;

  for (int i2 = 0; i2 < n2; ++i2) {
    scalar_t dist = 0;
    for (int c = 0; c < C; ++c) {
      scalar_t diff = pc1[b][i1][c] - pc2[b][i2][c];
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
  }
  dists[b][i1][0] = min_dist1;
  dists[b][i1][1] = min_dist2;
  dists[b][i1][2] = min_dist3;

  idxs[b][i1][0] = besti1;
  idxs[b][i1][1] = besti2;
  idxs[b][i1][2] = besti3;
}

std::tuple<at::Tensor, at::Tensor> three_nn_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  const int64_t B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t C = pc1.size(2);

  auto opts = pc1.options();
  auto long_opts = opts.dtype(at::kLong);
  auto inf = std::numeric_limits<float>::infinity();
  auto dists = at::full({B, N1, 3}, inf, opts);
  auto idxs = at::full({B, N1, 3}, -1, long_opts);

  const dim3 blocks(B, (N1 + 255) / 256);
  const dim3 threads(256);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "three_nn_cuda", ([&] {
        three_nn_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            pc1.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            pc2.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            dists.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>());
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dists, idxs);
}
