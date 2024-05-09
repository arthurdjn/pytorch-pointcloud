#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <tuple>

template <typename scalar_t>
__global__ void knn_cuda_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> pc2,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> K,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> dists,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> idxs) {
  const int b = blockIdx.x;
  const int i1 = threadIdx.x + blockIdx.y * blockDim.x;

  const int B = pc1.size(0);
  const int64_t N1 = pc1.size(1);
  const int64_t N2 = pc2.size(1);
  const int C = pc1.size(2);

  const int64_t N1_b = std::min(lengths1[b], N1);
  const int64_t N2_b = std::min(lengths2[b], N2);
  const int64_t K_b = std::min(K[b], N2_b);

  if (b >= B || i1 < 0 || i1 >= N1_b)
    return;

  for (int i2 = 0; i2 < N2_b; ++i2) {
    scalar_t dist = 0;
    for (int c = 0; c < C; ++c) {
      scalar_t diff = pc1[b][i1][c] - pc2[b][i2][c];
      dist += diff * diff;
    }

    for (int k = 0; k < K_b; ++k) {
      if (dist < dists[b][i1][k]) {
        // Keep the neighbors sorted
        for (int m = K_b - 1; m > k; --m) {
          dists[b][i1][m] = dists[b][i1][m - 1];
          idxs[b][i1][m] = idxs[b][i1][m - 1];
        }
        dists[b][i1][k] = dist;
        idxs[b][i1][k] = i2;
        break;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> knn_cuda(
    const at::Tensor& pc1,
    const at::Tensor& pc2,
    const at::Tensor& K,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2) {
  const int B = pc1.size(0);
  const int N1 = pc1.size(1);
  const int N2 = pc2.size(1);
  const int C = pc1.size(2);
  const int K_max = at::max(K).item<int>();

  auto inf = std::numeric_limits<float>::infinity();
  auto opts = pc1.options();
  auto long_opts = opts.dtype(at::kLong);
  auto dists = at::full({B, N1, K_max}, inf, pc1.options());
  auto idxs = at::full({B, N1, K_max}, -1, long_opts);

  const dim3 blocks(B, (N1 + 255) / 256);
  const dim3 threads(256);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      pc1.scalar_type(), "knn_cuda", ([&] {
        knn_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            pc1.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            pc2.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            K.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            lengths1.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            lengths2.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            dists.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            idxs.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>());
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dists, idxs);
}
