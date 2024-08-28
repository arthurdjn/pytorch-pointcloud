#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Atomic add for int64_t
__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        static_cast<unsigned long long int>(val + static_cast<int64_t>(assumed)));
  } while (assumed != old);

  return static_cast<int64_t>(old);
}

// CUDA kernel for counting number of clusters
__global__ void count_num_clusters_kernel(
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < cluster_ids.size(0)) {
    int64_t length_b = lengths[b];
    bool cluster_seen[1024] = {false}; // Assuming max possible clusters is 1024
    int64_t count = 0;

    for (int64_t n = 0; n < length_b; ++n) {
      int64_t cluster_id = cluster_ids[b][n];
      if (!cluster_seen[cluster_id]) {
        cluster_seen[cluster_id] = true;
        count++;
      }
    }

    num_clusters[b] = count;
  }
}

// CUDA kernel for scatter sum
template <typename scalar_t>
__global__ void scatter_sum_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    scalar_t padding_value) {
  int b = blockIdx.x;
  int n = threadIdx.x;

  if (b < points.size(0) && n < lengths[b]) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomicAdd(&output[b][cluster][d], points[b][n][d]);
    }

    int64_t count = atomicAdd(&counts[b][cluster], 1);
    indices[b][cluster][count] = n;
  }

  __syncthreads();

  if (n == 0) {
    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

// Wrapper function for count_num_clusters CUDA kernel
at::Tensor count_num_clusters_cuda(
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths) {
  auto num_clusters = at::zeros_like(lengths);

  const int threads = 256;
  const int blocks = (cluster_ids.size(0) + threads - 1) / threads;

  count_num_clusters_kernel<<<blocks, threads>>>(
      cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
      lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
      num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());

  return num_clusters;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_sum_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto counts = at::zeros(
      {points.size(0), max_num_clusters}, points.options().dtype(at::kLong));
  auto indices = at::full(
      {points.size(0), max_num_clusters, points.size(2)},
      -1,
      points.options().dtype(at::kLong));
  auto output = at::zeros(
      {points.size(0), max_num_clusters, points.size(2)}, points.options());

  const int threads = 256;
  const int blocks = points.size(0);

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_sum_cuda", ([&] {
        scatter_sum_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
            indices.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
            counts.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters, counts, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_cuda(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  return scatter_sum_cuda(points, cluster_ids, lengths, padding_value);
}
