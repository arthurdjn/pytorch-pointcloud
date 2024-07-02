#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define MAXIMUM_THREADS 1024

inline int optimal_num_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAXIMUM_THREADS), 1);
}

inline dim3 optimal_block_config(int x, int y) {
  const int x_threads = optimal_num_threads(x);
  const int y_threads =
      max(min(optimal_num_threads(y), MAXIMUM_THREADS / x_threads), 1);
  dim3 block_config(x_threads, y_threads, 1);

  return block_config;
}

#endif
