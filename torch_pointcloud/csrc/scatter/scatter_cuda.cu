#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "../cuda_utils.h"
#include "./atomics.cuh"

__global__ void count_num_clusters_kernel(
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;
  __shared__ unsigned int cluster_bitset[128];

  // Initialize shared memory
  for (int i = tid; i < 128; i += stride) {
    cluster_bitset[i] = 0;
  }
  __syncthreads();

  int64_t length_b = lengths[b];
  for (int64_t n = tid; n < length_b; n += stride) {
    int64_t cluster_id = cluster_ids[b][n];
    if (cluster_id < 4096) {
      atomicOr(&cluster_bitset[cluster_id / 32], 1U << (cluster_id % 32));
    }
  }
  __syncthreads();

  // Count set bits
  if (tid == 0) {
    unsigned int count = 0;
    for (int i = 0; i < 128; ++i) {
      count += __popc(cluster_bitset[i]);
    }
    num_clusters[b] = count;
  }
}

/**
 * @brief Count the number of clusters in each batch.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B,) tensor. The lengths of each batch in case of padding.
 * @return (B,) tensor. The number of clusters in each batch.
 */
at::Tensor count_num_clusters_cuda(
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths) {
  int64_t B = cluster_ids.size(0);
  auto num_clusters = at::zeros_like(lengths);

  const int blocks = B;
  const int threads = 256;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  count_num_clusters_kernel<<<blocks, threads, 0, stream>>>(
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
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomicAdd(&output[b][cluster][d], points[b][n][d]);
    }
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

/**
 * @brief Sum the input points together based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 *   This tensor is used during the backward pass, to avoid computing
 *   the number of clusters again.
 */
std::tuple<at::Tensor, at::Tensor> scatter_sum_cuda(
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
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters);
}

template <typename scalar_t>
__global__ void scatter_mean_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
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

/**
 * @brief Compute the mean of the input points based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - counts (B, max_num_clusters) tensor. The number of points in each cluster.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_mean_cuda(
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
            counts.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters, counts);
}

template <typename scalar_t>
__global__ void scatter_prod_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    scalar_t padding_value) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    for (int64_t d = 0; d < points.size(2); ++d) {
      atomMul(&output[b][cluster][d], points[b][n][d]);
    }
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

/**
 * @brief Compute the product of the input points based on the cluster ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 */
std::tuple<at::Tensor, at::Tensor> scatter_prod_cuda(
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
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters);
}

template <typename scalar_t>
__global__ void scatter_min_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
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

/**
 * @brief Compute the minimum channel-wise of the input points based on the cluster
 * ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - indices (B, max_num_clusters, C) tensor. The indices of the minimum points.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_min_cuda(
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
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters, indices);
}

template <typename scalar_t>
__global__ void scatter_max_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
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

/**
 * @brief Compute the maximum channel-wise of the input points based on the cluster
 * ids.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param padding_value The value to use for padding.
 * @return A tuple of:
 * - output (B, max_num_clusters, C) tensor. The output tensor.
 * - num_clusters (B) tensor. The number of clusters in each batch.
 * - indices (B, max_num_clusters, C) tensor. The indices of the maximum points.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> scatter_max_cuda(
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
            static_cast<scalar_t>(padding_value));
      }));

  return std::make_tuple(output, num_clusters, indices);
}

template <typename scalar_t>
__global__ void scatter_sum_backward_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_points) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    if (cluster >= 0 && cluster < num_clusters[b]) {
      for (int64_t d = 0; d < grad_points.size(2); ++d) {
        atomicAdd(&grad_points[b][n][d], grad_output[b][cluster][d]);
      }
    }
  }
}

/**
 * @brief Compute the backward pass for the scatter sum operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_sum_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters) {
  auto B = points.size(0);
  auto N = points.size(1);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_sum_backward_cuda", ([&] {
        scatter_sum_backward_kernel<scalar_t>
            <<<B, optimal_block_config(N, C), 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                grad_points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  return grad_points;
}

template <typename scalar_t>
__global__ void scatter_mean_backward_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> counts,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_points) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    if (cluster >= 0 && cluster < num_clusters[b]) {
      int64_t cluster_count = counts[b][cluster];

      if (cluster_count > 0) {
        for (int64_t d = 0; d < grad_points.size(2); ++d) {
          atomicAdd(
              &grad_points[b][n][d],
              grad_output[b][cluster][d] / static_cast<scalar_t>(cluster_count));
        }
      }
    }
  }
}

/**
 * @brief Compute the backward pass for the scatter mean operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param counts (B, max_num_clusters) tensor. The number of points in each cluster.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_mean_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& counts) {
  auto B = points.size(0);
  auto N = points.size(1);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_mean_backward_cuda", ([&] {
        scatter_mean_backward_kernel<scalar_t>
            <<<B, optimal_block_config(N, C), 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                counts.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                grad_points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  return grad_points;
}

template <typename scalar_t>
__global__ void scatter_prod_backward_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_points) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    if (cluster >= 0 && cluster < num_clusters[b]) {
      for (int64_t d = 0; d < grad_points.size(2); ++d) {
        if (points[b][n][d] != 0) {
          scalar_t grad =
              grad_output[b][cluster][d] * output[b][cluster][d] / points[b][n][d];
          atomicAdd(&grad_points[b][n][d], grad);
        }
      }
    }
  }
}

/**
 * @brief Compute the backward pass for the scatter prod operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param output (B, max_num_clusters, C) tensor. The output of the forward pass.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_prod_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& output) {
  auto B = points.size(0);
  auto N = points.size(1);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_prod_backward_cuda", ([&] {
        scatter_prod_backward_kernel<scalar_t>
            <<<B, optimal_block_config(N, C), 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                grad_points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  return grad_points;
}

template <typename scalar_t>
__global__ void scatter_min_backward_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_points) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    if (cluster >= 0 && cluster < num_clusters[b]) {
      for (int64_t d = 0; d < grad_points.size(2); ++d) {
        if (indices[b][cluster][d] == n) {
          atomicAdd(&grad_points[b][n][d], grad_output[b][cluster][d]);
        }
      }
    }
  }
}

/**
 * @brief Compute the backward pass for the scatter min operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param indices (B, max_num_clusters, C) tensor. The indices of the minimum points.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_min_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto N = points.size(1);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_min_backward_cuda", ([&] {
        scatter_min_backward_kernel<scalar_t>
            <<<B, optimal_block_config(N, C), 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
                grad_points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  return grad_points;
}

template <typename scalar_t>
__global__ void scatter_max_backward_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    const at::PackedTensorAccessor64<int64_t, 3, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_points) {
  int b = blockIdx.x; // batch index
  int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  int total_threads = blockDim.x * blockDim.y; // total threads in the block

  for (int n = tid; n < lengths[b]; n += total_threads) {
    int64_t cluster = cluster_ids[b][n];

    if (cluster >= 0 && cluster < num_clusters[b]) {
      for (int64_t d = 0; d < grad_points.size(2); ++d) {
        if (indices[b][cluster][d] == n) {
          atomicAdd(&grad_points[b][n][d], grad_output[b][cluster][d]);
        }
      }
    }
  }
}

/**
 * @brief Compute the backward pass for the scatter max operation.
 * @param grad_output (B, max_num_clusters, C) tensor. The gradients of the output.
 * @param points (B, N, C) tensor. The input points.
 * @param cluster_ids (B, N) tensor. The cluster ids.
 * @param lengths (B) tensor. The lengths of each batch in case of padding.
 * @param num_clusters (B) tensor. The number of clusters in each batch.
 * @param indices (B, max_num_clusters, C) tensor. The indices of the maximum points.
 * @return (B, N, C) tensor. The gradients of the input points.
 */
at::Tensor scatter_max_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const at::Tensor& num_clusters,
    const at::Tensor& indices) {
  auto B = points.size(0);
  auto N = points.size(1);
  auto C = points.size(2);

  auto grad_points = at::zeros_like(points);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "scatter_max_backward_cuda", ([&] {
        scatter_max_backward_kernel<scalar_t>
            <<<B, optimal_block_config(N, C), 0, stream>>>(
                grad_output.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                cluster_ids.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                lengths.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                num_clusters.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 3, at::RestrictPtrTraits>(),
                grad_points.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
      }));

  return grad_points;
}
