#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <tuple>

#include "../cuda_utils.h"
#include "../utils.h"

template <typename scalar_t>
__global__ void sided_distance_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> dists,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs) {
  const int b = blockIdx.x;
  const int i1 = blockIdx.y * blockDim.x + threadIdx.x;

  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);

  const int64_t n1 = std::min(N1, lengths1[b]);
  const int64_t n2 = std::min(N2, lengths2[b]);

  if (i1 >= n1 || i1 < 0)
    return;

  scalar_t min_dist = std::numeric_limits<scalar_t>::max();

  for (int i2 = 0; i2 < n2; ++i2) {
    scalar_t dist = 0;
    for (int c = 0; c < C; ++c) {
      scalar_t diff = pc1[b][i1][c] - pc2[b][i2][c];
      dist += diff * diff;
    }

    if (dist < min_dist) {
      min_dist = dist;
      dists[b][i1] = dist;
      idxs[b][i1] = i2;
    }
  }
}

std::tuple<at::Tensor, at::Tensor> sided_distance_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  at::TensorArg pc1_t{pc1, "pc1", 1}, pc2_t{pc1, "pc1", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "sided_distance_cuda";
  at::checkAllSameGPU(c, {pc1_t, pc2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {pc1_t, pc2_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});

  const auto B = pc1.size(0);
  const auto N1 = pc1.size(1);
  const auto N2 = pc2.size(1);
  const auto C = pc1.size(2);

  auto opts = pc1.options();
  auto long_opts = opts.dtype(at::kLong);
  auto dists = at::full({B, N1}, 0, opts);
  auto idxs = at::full({B, N1}, -1, long_opts);

  const dim3 blocks(B, (N1 + 255) / 256);
  const dim3 threads(256);

  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "sided_distance_cuda", ([&] {
        sided_distance_kernel<scalar_t><<<blocks, threads>>>(
            pc1.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            pc2.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            dists.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
            idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>());
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(dists, idxs);
}

// TODO: (arthurdjn) support all data types once AtomicAdd supports doubles
__global__ void sided_distance_backward_kernel(
    const at::PackedTensorAccessor64<float_t, 2, at::RestrictPtrTraits> grad_dists,
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> pc1,
    const at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> pc2,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idxs,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_pc1,
    at::PackedTensorAccessor64<float_t, 3, at::RestrictPtrTraits> grad_pc2) {
  const int b = blockIdx.x;
  const int i1 = blockIdx.y * blockDim.x + threadIdx.x;

  const int64_t B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int64_t C = pc1.size(2);

  const int64_t n1 = std::min(N1, lengths1[b]);
  const int64_t n2 = std::min(N2, lengths2[b]);

  if (b >= B || i1 >= n1 || i1 < 0)
    return;

  int64_t i2 = idxs[b][i1];
  if (i2 >= 0 && i2 < n2) {
    for (int c = 0; c < C; ++c) {
      float_t diff = pc1[b][i1][c] - pc2[b][i2][c];
      float_t grad = grad_dists[b][i1];

      atomicAdd(&grad_pc1[b][i1][c], 2 * diff * grad);
      atomicAdd(&grad_pc2[b][i2][c], -2 * diff * grad);
    }
  }
}

// TODO(arthurdjn) use AT_DISPATCH_FLOATING_TYPES once AtomicAdd supports doubles
std::tuple<at::Tensor, at::Tensor> sided_distance_backward_cuda(
    const at::Tensor& grad_dists,
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& idxs,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  at::TensorArg grad_dists_t{grad_dists, "grad_dists", 1}, idxs_t{idxs, "idxs", 1},
      pc1_t{pc1, "pc1", 1}, pc2_t{pc1, "pc1", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "sided_distance_backward_cuda";
  at::checkAllSameGPU(
      c, {grad_dists_t, idxs_t, pc1_t, pc2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {grad_dists_t, pc1_t, pc2_t});
  at::checkAllSameType(c, {idxs_t, lengths1_t, lengths2_t});

  const auto B = pc1.size(0);
  const auto N1 = pc1.size(1);

  auto grad_pc1 = at::zeros_like(pc1);
  auto grad_pc2 = at::zeros_like(pc2);

  const dim3 blocks(B, (N1 + 255) / 256);
  const dim3 threads(256);

  sided_distance_backward_kernel<<<blocks, threads>>>(
      grad_dists.packed_accessor64<float_t, 2, at::RestrictPtrTraits>(),
      pc1.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      pc2.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      idxs.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
      lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      grad_pc1.packed_accessor64<float_t, 3, at::RestrictPtrTraits>(),
      grad_pc2.packed_accessor64<float_t, 3, at::RestrictPtrTraits>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_pc1, grad_pc2);
}