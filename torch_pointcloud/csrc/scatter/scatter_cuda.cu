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
  __shared__ unsigned int
      cluster_bitset[128]; // 4096 bits, can handle up to 4096 clusters

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
  const int b = blockIdx.x; // batch index
  const int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  const int total_threads = blockDim.x * blockDim.y; // total threads in the block

  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t max_num_clusters = output.size(1);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    // Sum the points per cluster
    const int64_t cluster_id = cluster_ids[b][n];
    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        atomicAdd(&output[b][cluster_id][c], points[b][n][c]);
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    // Padding for unused clusters
    for (int64_t n = num_clusters[b]; n < max_num_clusters; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        output[b][n][c] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> scatter_sum_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

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

  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t max_num_clusters = output.size(1);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    // Sum the points per cluster
    const int64_t cluster_id = cluster_ids[b][n];
    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        atomicAdd(&output[b][cluster_id][c], points[b][n][c]);
      }
      atomAdd(&counts[b][cluster_id], static_cast<int64_t>(1));
    }
  }

  __syncthreads();

  if (tid == 0) {
    // Divide by the number of points per cluster
    for (int n = 0; n < length_b; ++n) {
      const int64_t cluster_id = cluster_ids[b][n];
      if (counts[b][cluster_id] && cluster_id >= 0 && cluster_id < num_clusters[b]) {
        for (int64_t c = 0; c < C; ++c) {
          output[b][cluster_id][c] /= counts[b][cluster_id];
        }
      }
    }

    // Padding for unused clusters
    for (int64_t n = num_clusters[b]; n < max_num_clusters; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        output[b][n][c] = padding_value;
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
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

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
template <typename scalar_t>
__global__ void scatter_prod_kernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> points,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> cluster_ids,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> num_clusters,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> output,
    scalar_t padding_value) {
  const int b = blockIdx.x; // batch index
  const int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread index
  const int total_threads = blockDim.x * blockDim.y; // total threads in the block

  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t max_num_clusters = output.size(1);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    const int64_t cluster_id = cluster_ids[b][n];
    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        atomMul(&output[b][cluster_id][c], points[b][n][c]);
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t n = num_clusters[b]; n < max_num_clusters; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        output[b][n][c] = padding_value;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> scatter_prod_cuda(
    const at::Tensor& points,
    const at::Tensor& cluster_ids,
    const at::Tensor& lengths,
    const float padding_value = 0.0) {
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

  auto num_clusters = count_num_clusters_cuda(cluster_ids, lengths);
  const int64_t max_num_clusters = num_clusters.max().item<int64_t>();

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

  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t max_num_clusters = output.size(1);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];

    for (int64_t c = 0; c < C; ++c) {
      scalar_t point_value = points[b][n][c];
      scalar_t old_value = atomMin(&output[b][cluster_id][c], point_value);

      // Only update if it's the min
      if (point_value == output[b][cluster_id][c]) {
        indices[b][cluster_id][c] = n;
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t n = num_clusters[b]; n < max_num_clusters; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        output[b][n][c] = padding_value;
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

  const int64_t N = points.size(1);
  const int64_t C = points.size(2);
  const int64_t max_num_clusters = output.size(1);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];

    for (int64_t c = 0; c < C; ++c) {
      scalar_t point_value = points[b][n][c];
      scalar_t old_value = atomMax(&output[b][cluster_id][c], point_value);

      if (point_value == output[b][cluster_id][c]) { // Only update if it's the max
        indices[b][cluster_id][c] = n;
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
    for (int64_t n = num_clusters[b]; n < max_num_clusters; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        output[b][n][c] = padding_value;
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

  const int64_t N = grad_points.size(1);
  const int64_t C = grad_points.size(2);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];

    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        atomicAdd(&grad_points[b][n][c], grad_output[b][cluster_id][c]);
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

  const int64_t N = grad_points.size(1);
  const int64_t C = grad_points.size(2);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];
    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      int64_t cluster_count = counts[b][cluster_id];
      if (cluster_count > 0) {
        for (int64_t c = 0; c < C; ++c) {
          atomicAdd(
              &grad_points[b][n][c],
              grad_output[b][cluster_id][c] / static_cast<scalar_t>(cluster_count));
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

  const int64_t N = grad_points.size(1);
  const int64_t C = grad_points.size(2);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];

    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        if (points[b][n][c] != 0) {
          scalar_t grad = grad_output[b][cluster_id][c] * output[b][cluster_id][c] /
              points[b][n][c];
          atomicAdd(&grad_points[b][n][c], grad);
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

  const int64_t N = grad_points.size(1);
  const int64_t C = grad_points.size(2);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    const int64_t cluster_id = cluster_ids[b][n];

    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < C; ++c) {
        if (indices[b][cluster_id][c] == n) {
          atomicAdd(&grad_points[b][n][c], grad_output[b][cluster_id][c]);
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
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

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

  const int64_t N = grad_points.size(1);
  const int64_t C = grad_points.size(2);
  const int64_t length_b = std::clamp(lengths[b], int64_t(0), N);

  for (int n = tid; n < length_b; n += total_threads) {
    int64_t cluster_id = cluster_ids[b][n];

    if (cluster_id >= 0 && cluster_id < num_clusters[b]) {
      for (int64_t c = 0; c < grad_points.size(2); ++c) {
        if (indices[b][cluster_id][c] == n) {
          atomicAdd(&grad_points[b][n][c], grad_output[b][cluster_id][c]);
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
  const int64_t B = points.size(0);
  const int64_t N = points.size(1);
  const int64_t C = points.size(2);

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