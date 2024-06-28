#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <typename scalar_t>
__global__ void ball_query_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        max_neighbors,
    const at::PackedTensorAccessor64<scalar_t, 1, at::RestrictPtrTraits> radiuses,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> dists,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs) {
  const int64_t B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);

  const int64_t chunks_per_cloud = (1 + (N1 - 1) / blockDim.x);
  const int64_t chunks_to_do = B * chunks_per_cloud;

  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t b = chunk / chunks_per_cloud; // batch index
    const int64_t n1 = std::min(N1, lengths1[b]);
    const int64_t n2 = std::min(N2, lengths2[b]);
    const int64_t k = max_neighbors[b];
    const scalar_t radius2 = radiuses[b] * radiuses[b];

    int64_t i1 = blockDim.x * (chunk % chunks_per_cloud) + threadIdx.x;
    if (i1 < 0 || i1 >= n1) {
      continue;
    }

    for (int64_t i2 = 0, count = 0; i2 < n2 && count < k; ++i2) {
      scalar_t dist2 = 0.0;
      for (int c = 0; c < C; ++c) {
        scalar_t diff = pc1[b][i1][c] - pc2[b][i2][c];
        dist2 += (diff * diff);
      }

      if (dist2 < radius2) {
        idxs[b][i1][count] = i2;
        dists[b][i1][count] = dist2;
        ++count;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> ball_query_cuda(
    const at::Tensor& pc1, // (B, N1, 3)
    const at::Tensor& pc2, // (B, N2, 3)
    const at::Tensor& lengths1, // (B,)
    const at::Tensor& lengths2, // (B,)
    const at::Tensor& max_neighbors, // (B,)
    const at::Tensor& radiuses) { // (B,)
  // Check inputs are on the same device
  at::TensorArg pc1_t{pc1, "pc1", 1}, pc2_t{pc2, "pc2", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4},
      max_neighbors_t{max_neighbors, "max_neighbors", 4},
      radiuses_t{radiuses, "radiuses", 4};
  at::CheckedFrom c = "ball_query_cuda";
  at::checkAllSameGPU(
      c, {pc1_t, pc2_t, lengths1_t, lengths2_t, max_neighbors_t, radiuses_t});
  at::checkAllSameType(c, {pc1_t, pc2_t, radiuses_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t, max_neighbors_t});

  const int B = pc1.size(0);
  const int N1 = pc1.size(1);
  const int64_t K = max_neighbors.max().item<int64_t>();

  // Output tensor with indices of neighbors for each point in pc1
  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({B, N1, K}, -1, long_dtype);
  auto dists = at::zeros({B, N1, K}, pc1.options());

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(dists, idxs);
  }

  const size_t blocks = 256;
  const size_t threads = 256;

  // Set the device for the kernel launch based on the device of pc1
  at::cuda::CUDAGuard device_guard(pc1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "ball_query_cuda", ([&] {
        ball_query_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            pc1.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            pc2.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            max_neighbors.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            radiuses.packed_accessor64<scalar_t, 1, at::RestrictPtrTraits>(),
            dists.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>());
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(dists, idxs);
}