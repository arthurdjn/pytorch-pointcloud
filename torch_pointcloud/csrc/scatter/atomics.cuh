#pragma once

#define ATOMIC(NAME)                                                              \
  template <typename scalar, size_t size>                                         \
  struct Atomic##NAME##IntegerImpl;                                               \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##IntegerImpl<scalar, 1> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      uint32_t* address_as_ui = (uint32_t*)(address - ((size_t)address & 3));     \
      uint32_t old = *address_as_ui;                                              \
      uint32_t shift = ((size_t)address & 3) * 8;                                 \
      uint32_t sum;                                                               \
      uint32_t assumed;                                                           \
      scalar old_val;                                                             \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old_val = scalar((old >> shift) & 0xff);                                  \
        sum = OP(val, old_val);                                                   \
        old = (old & ~(0x000000ff << shift)) | (sum << shift);                    \
        old = atomicCAS(address_as_ui, assumed, old);                             \
      } while (assumed != old);                                                   \
                                                                                  \
      return old_val;                                                             \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##IntegerImpl<scalar, 2> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      uint32_t* address_as_ui =                                                   \
          (uint32_t*)((char*)address - ((size_t)address & 2));                    \
      uint32_t old = *address_as_ui;                                              \
      uint32_t sum;                                                               \
      uint32_t newval;                                                            \
      uint32_t assumed;                                                           \
      scalar old_val;                                                             \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old_val = (size_t)address & 2 ? scalar(old >> 16) : scalar(old & 0xffff); \
        sum = OP(val, old_val);                                                   \
        newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16)               \
                                     : (old & 0xffff0000) | sum;                  \
        old = atomicCAS(address_as_ui, assumed, newval);                          \
      } while (assumed != old);                                                   \
                                                                                  \
      return old_val;                                                             \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##IntegerImpl<scalar, 4> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      uint32_t* address_as_ui = (uint32_t*)address;                               \
      uint32_t old = *address_as_ui;                                              \
      uint32_t assumed;                                                           \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old = atomicCAS(address_as_ui, assumed, OP(val, (scalar)old));            \
      } while (assumed != old);                                                   \
                                                                                  \
      return (scalar)assumed;                                                     \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##IntegerImpl<scalar, 8> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      unsigned long long* address_as_ull = (unsigned long long*)address;          \
      unsigned long long old = *address_as_ull;                                   \
      unsigned long long assumed;                                                 \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old = atomicCAS(address_as_ull, assumed, OP(val, (scalar)old));           \
      } while (assumed != old);                                                   \
                                                                                  \
      return (scalar)assumed;                                                     \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar, size_t size>                                         \
  struct Atomic##NAME##DecimalImpl;                                               \
                                                                                  \
  template <>                                                                     \
  struct Atomic##NAME##DecimalImpl<at::Half, 2> {                                 \
    inline __device__ at::Half operator()(at::Half* address, at::Half val) {      \
      unsigned int* address_as_ui =                                               \
          (unsigned int*)((char*)address - ((size_t)address & 2));                \
      unsigned int old = *address_as_ui;                                          \
      unsigned int assumed;                                                       \
      at::Half old_val;                                                           \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old_val.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);           \
        at::Half hsum = OP(old_val, val);                                         \
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)               \
                                  : (old & 0xffff0000) | hsum.x;                  \
        old = atomicCAS(address_as_ui, assumed, old);                             \
      } while (assumed != old);                                                   \
                                                                                  \
      return old_val;                                                             \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <>                                                                     \
  struct Atomic##NAME##DecimalImpl<at::BFloat16, 2> {                             \
    inline __device__ at::BFloat16 operator()(                                    \
        at::BFloat16* address,                                                    \
        at::BFloat16 val) {                                                       \
      unsigned int* address_as_ui =                                               \
          (unsigned int*)((char*)address - ((size_t)address & 2));                \
      unsigned int old = *address_as_ui;                                          \
      unsigned int assumed;                                                       \
      at::BFloat16 old_val;                                                       \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old_val.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);           \
        at::BFloat16 hsum = OP(old_val, val);                                     \
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)               \
                                  : (old & 0xffff0000) | hsum.x;                  \
        old = atomicCAS(address_as_ui, assumed, old);                             \
      } while (assumed != old);                                                   \
                                                                                  \
      return old_val;                                                             \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##DecimalImpl<scalar, 4> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      int* address_as_i = (int*)address;                                          \
      int old = *address_as_i;                                                    \
      int assumed;                                                                \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old = atomicCAS(                                                          \
            address_as_i,                                                         \
            assumed,                                                              \
            __float_as_int(OP(val, __int_as_float(assumed))));                    \
      } while (assumed != old);                                                   \
                                                                                  \
      return __int_as_float(assumed);                                             \
    }                                                                             \
  };                                                                              \
                                                                                  \
  template <typename scalar>                                                      \
  struct Atomic##NAME##DecimalImpl<scalar, 8> {                                   \
    inline __device__ scalar operator()(scalar* address, scalar val) {            \
      unsigned long long int* address_as_ull = (unsigned long long int*)address;  \
      unsigned long long int old = *address_as_ull;                               \
      unsigned long long int assumed;                                             \
                                                                                  \
      do {                                                                        \
        assumed = old;                                                            \
        old = atomicCAS(                                                          \
            address_as_ull,                                                       \
            assumed,                                                              \
            __double_as_longlong(OP(val, __longlong_as_double(assumed))));        \
      } while (assumed != old);                                                   \
                                                                                  \
      return __longlong_as_double(assumed);                                       \
    }                                                                             \
  };

#define OP(X, Y) Y + X
ATOMIC(Add)
#undef OP
static inline __device__ uint8_t atomAdd(uint8_t* address, uint8_t val) {
  return AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}
static inline __device__ int8_t atomAdd(int8_t* address, int8_t val) {
  return AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}
static inline __device__ int16_t atomAdd(int16_t* address, int16_t val) {
  return AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}
static inline __device__ int32_t atomAdd(int32_t* address, int32_t val) {
  return atomicAdd(address, val);
}
static inline __device__ int64_t atomAdd(int64_t* address, int64_t val) {
  return AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
#if defined(USE_ROCM) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || CUDA_VERSION < 10000))
static inline __device__ at::Half atomAdd(at::Half* address, at::Half val) {
  return AtomicAddDecimalImpl<at::Half, sizeof(at::Half)>()(address, val);
}
#else
static inline __device__ at::Half atomAdd(at::Half* address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}
#endif
static inline __device__ float atomAdd(float* address, float val) {
  return atomicAdd(address, val);
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
static inline __device__ double atomAdd(double* address, double val) {
  return AtomicAddDecimalImpl<double, sizeof(double)>()(address, val);
}
#else
static inline __device__ double atomAdd(double* address, double val) {
  return atomicAdd(address, val);
}
#endif
static inline __device__ at::BFloat16 atomAdd(
    at::BFloat16* address,
    at::BFloat16 val) {
  return AtomicAddDecimalImpl<at::BFloat16, sizeof(at::BFloat16)>()(address, val);
}

#define OP(X, Y) Y* X
ATOMIC(Mul)
#undef OP
static inline __device__ uint8_t atomMul(uint8_t* address, uint8_t val) {
  return AtomicMulIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}
static inline __device__ int8_t atomMul(int8_t* address, int8_t val) {
  return AtomicMulIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}
static inline __device__ int16_t atomMul(int16_t* address, int16_t val) {
  return AtomicMulIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}
static inline __device__ int32_t atomMul(int32_t* address, int32_t val) {
  return AtomicMulIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline __device__ int64_t atomMul(int64_t* address, int64_t val) {
  return AtomicMulIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ float atomMul(float* address, float val) {
  return AtomicMulDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ at::Half atomMul(at::Half* address, at::Half val) {
  return AtomicMulDecimalImpl<at::Half, sizeof(at::Half)>()(address, val);
}
static inline __device__ double atomMul(double* address, double val) {
  return AtomicMulDecimalImpl<double, sizeof(double)>()(address, val);
}
static inline __device__ at::BFloat16 atomMul(
    at::BFloat16* address,
    at::BFloat16 val) {
  return AtomicMulDecimalImpl<at::BFloat16, sizeof(at::BFloat16)>()(address, val);
}

#define OP(X, Y) Y / X
ATOMIC(Div)
#undef OP
static inline __device__ uint8_t atomDiv(uint8_t* address, uint8_t val) {
  return AtomicDivIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}
static inline __device__ int8_t atomDiv(int8_t* address, int8_t val) {
  return AtomicDivIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}
static inline __device__ int16_t atomDiv(int16_t* address, int16_t val) {
  return AtomicDivIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}
static inline __device__ int32_t atomDiv(int32_t* address, int32_t val) {
  return AtomicDivIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline __device__ int64_t atomDiv(int64_t* address, int64_t val) {
  return AtomicDivIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ at::Half atomDiv(at::Half* address, at::Half val) {
  return AtomicDivDecimalImpl<at::Half, sizeof(at::Half)>()(address, val);
}
static inline __device__ float atomDiv(float* address, float val) {
  return AtomicDivDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ double atomDiv(double* address, double val) {
  return AtomicDivDecimalImpl<double, sizeof(double)>()(address, val);
}
static inline __device__ at::BFloat16 atomDiv(
    at::BFloat16* address,
    at::BFloat16 val) {
  return AtomicDivDecimalImpl<at::BFloat16, sizeof(at::BFloat16)>()(address, val);
}

#define OP(X, Y) max(Y, X)
ATOMIC(Max)
#undef OP
static inline __device__ uint8_t atomMax(uint8_t* address, uint8_t val) {
  return AtomicMaxIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}
static inline __device__ int8_t atomMax(int8_t* address, int8_t val) {
  return AtomicMaxIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}
static inline __device__ int16_t atomMax(int16_t* address, int16_t val) {
  return AtomicMaxIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}
static inline __device__ int32_t atomMax(int32_t* address, int32_t val) {
  return atomicMax(address, val);
}
static inline __device__ int64_t atomMax(int64_t* address, int64_t val) {
  return AtomicMaxIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ at::Half atomMax(at::Half* address, at::Half val) {
  return AtomicMaxDecimalImpl<at::Half, sizeof(at::Half)>()(address, val);
}
static inline __device__ float atomMax(float* address, float val) {
  return AtomicMaxDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ double atomMax(double* address, double val) {
  return AtomicMaxDecimalImpl<double, sizeof(double)>()(address, val);
}
static inline __device__ at::BFloat16 atomMax(
    at::BFloat16* address,
    at::BFloat16 val) {
  return AtomicMaxDecimalImpl<at::BFloat16, sizeof(at::BFloat16)>()(address, val);
}

#define OP(X, Y) min(Y, X)
ATOMIC(Min)
#undef OP
static inline __device__ uint8_t atomMin(uint8_t* address, uint8_t val) {
  return AtomicMinIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}
static inline __device__ int8_t atomMin(int8_t* address, int8_t val) {
  return AtomicMinIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}
static inline __device__ int16_t atomMin(int16_t* address, int16_t val) {
  return AtomicMinIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}
static inline __device__ int32_t atomMin(int32_t* address, int32_t val) {
  return atomicMin(address, val);
}
static inline __device__ int64_t atomMin(int64_t* address, int64_t val) {
  return AtomicMinIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ at::Half atomMin(at::Half* address, at::Half val) {
  return AtomicMinDecimalImpl<at::Half, sizeof(at::Half)>()(address, val);
}
static inline __device__ float atomMin(float* address, float val) {
  return AtomicMinDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ double atomMin(double* address, double val) {
  return AtomicMinDecimalImpl<double, sizeof(double)>()(address, val);
}
static inline __device__ at::BFloat16 atomMin(
    at::BFloat16* address,
    at::BFloat16 val) {
  return AtomicMinDecimalImpl<at::BFloat16, sizeof(at::BFloat16)>()(address, val);
}
