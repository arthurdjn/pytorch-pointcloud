#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_IS_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")

#define CHECK_IS_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor.")

#define CHECK_IS_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")

#define CHECK_IS_CONTIGUOUS_CUDA(x) \
  CHECK_IS_CUDA(x);                 \
  CHECK_IS_CONTIGUOUS(x)

#define CHECK_IS_CONTIGUOUS_CPU(x) \
  CHECK_IS_CPU(x);                 \
  CHECK_IS_CONTIGUOUS(x)

#define CHECK_IS_INT(x) \
  TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be an int tensor.")

#define CHECK_IS_LONG(x) \
  TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be a long tensor.")

#define CHECK_IS_FLOAT(x) \
  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be a float tensor.")

#define CHECK_IS_DOUBLE(x) \
  TORCH_CHECK(x.scalar_type() == at::kDouble, #x " must be a double tensor.")
