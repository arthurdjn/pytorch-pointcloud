#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "../cuda_utils.h"
#include "./atomics.cuh"

// TODO: Update cluster_seen to be size of max cluster id (or N)
__global__ void count_num_clusters_kernel(
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < cluster_ids.size(0)) {
    int64_t length_b = lengths[b];
    bool cluster_seen[1024] = {false};
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
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomicAdd(&output[b][cluster][d], points[b][n][d]);
    }

    atomAdd(&counts[b][cluster], static_cast<int64_t>(1));
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_sum_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  int64_t B = points.size(0);
  int64_t N = points.size(1);
  int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto counts = at::zeros({B, max_num_clusters}, long_opts);
  auto indices = at::full({B, max_num_clusters, C}, -1, long_opts);
  auto output = at::zeros({B, max_num_clusters, C}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_sum_cuda", ([&] {
        scatter_sum_kernel<scalar_t><<<B, optimal_block_config(N, C), 0, stream>>>(
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

template <typename scalar_t>
__global__ void scatter_mean_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomicAdd(&output[b][cluster][d], points[b][n][d]);
    }

    atomAdd(&counts[b][cluster], static_cast<int64_t>(1));
  }

  __syncthreads();

  // Compute the mean for each cluster and pad unused clusters
  if (tid == 0) {
    for (int64_t cluster = 0; cluster < num_clusters[b]; ++cluster) {
      if (counts[b][cluster] > 0) {
        for (int64_t d = 0; d < output.size(2); ++d) {
          output[b][cluster][d] /= static_cast<scalar_t>(counts[b][cluster]);
        }
      }
    }

    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_mean_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  int64_t B = points.size(0);
  int64_t N = points.size(1);
  int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto counts = at::zeros({B, max_num_clusters}, long_opts);
  auto indices = at::full({B, max_num_clusters, C}, -1, long_opts);
  auto output = at::zeros({B, max_num_clusters, C}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_mean_cuda", ([&] {
        scatter_mean_kernel<scalar_t><<<B, optimal_block_config(N, C), 0, stream>>>(
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

template <typename scalar_t>
__global__ void scatter_prod_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomMul(&output[b][cluster][d], points[b][n][d]);
    }

    atomAdd(&counts[b][cluster], static_cast<int64_t>(1));
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_prod_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  int64_t B = points.size(0);
  int64_t N = points.size(1);
  int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto counts = at::zeros({B, max_num_clusters}, long_opts);
  auto indices = at::full({B, max_num_clusters, C}, -1, long_opts);
  auto output = at::ones({B, max_num_clusters, C}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_prod_cuda", ([&] {
        scatter_prod_kernel<scalar_t><<<B, optimal_block_config(N, C), 0, stream>>>(
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

template <typename scalar_t>
__global__ void scatter_min_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      scalar_t point_value = points[b][n][d];
      scalar_t old_value = atomMin(&output[b][cluster][d], point_value);

      if (point_value == output[b][cluster][d]) { // Only update if it's the min
        indices[b][cluster][d] = n;
      }
    }

    atomAdd(&counts[b][cluster], static_cast<int64_t>(1));
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_min_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  int64_t B = points.size(0);
  int64_t N = points.size(1);
  int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto counts = at::zeros({B, max_num_clusters}, long_opts);
  auto indices = at::full({B, max_num_clusters, C}, -1, long_opts);
  auto inf = std::numeric_limits<float>::max();
  auto output = at::full({B, max_num_clusters, C}, inf, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_min_cuda", ([&] {
        scatter_min_kernel<scalar_t><<<B, optimal_block_config(N, C), 0, stream>>>(
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

template <typename scalar_t>
__global__ void scatter_max_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      scalar_t point_value = points[b][n][d];
      scalar_t old_value = atomMax(&output[b][cluster][d], point_value);

      if (point_value == output[b][cluster][d]) { // Only update if it's the max
        indices[b][cluster][d] = n;
      }
    }

    atomAdd(&counts[b][cluster], static_cast<int64_t>(1));
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t c = num_clusters[b]; c < output.size(1); ++c) {
      for (int64_t d = 0; d < output.size(2); ++d) {
        output[b][c][d] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_max_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  int64_t B = points.size(0);
  int64_t N = points.size(1);
  int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  int64_t max_num_clusters = num_clusters.max().item<int64_t>();

  auto opts = points.options();
  auto long_opts = opts.dtype(at::kLong);

  auto counts = at::zeros({B, max_num_clusters}, long_opts);
  auto indices = at::full({B, max_num_clusters, C}, -1, long_opts);
  auto inf = std::numeric_limits<float>::min();
  auto output = at::full({B, max_num_clusters, C}, inf, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_max_cuda", ([&] {
        scatter_max_kernel<scalar_t><<<B, optimal_block_config(N, C), 0, stream>>>(
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

#define AT_DISPATCH_REDUCTION_TYPES(                                         \
    reduce, points, cluster_ids, lengths, padding_value)                     \
  [&]() -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {      \
    if (reduce == "sum")                                                     \
      return scatter_sum_cuda(points, cluster_ids, lengths, padding_value);  \
    else if (reduce == "mean")                                               \
      return scatter_mean_cuda(points, cluster_ids, lengths, padding_value); \
    else if (reduce == "prod")                                               \
      return scatter_prod_cuda(points, cluster_ids, lengths, padding_value); \
    else if (reduce == "min")                                                \
      return scatter_min_cuda(points, cluster_ids, lengths, padding_value);  \
    else if (reduce == "max")                                                \
      return scatter_max_cuda(points, cluster_ids, lengths, padding_value);  \
    else                                                                     \
      AT_ERROR("Unknown reduction type: ", reduce);                          \
  }()

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> scatter_cuda(
    const std::string& reduce,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  return AT_DISPATCH_REDUCTION_TYPES(
      reduce, points, cluster_ids, lengths, padding_value);
}
