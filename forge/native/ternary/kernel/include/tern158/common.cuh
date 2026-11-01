// common.cuh - internal helpers shared by all tern158 kernels.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "tern158.h"

#define T158_HD __host__ __device__ __forceinline__
#define T158_D __device__ __forceinline__

#define T158_CUDA_TRY(expr)                                                                          \
  do {                                                                                               \
    cudaError_t _e = (expr);                                                                         \
    if (_e != cudaSuccess) {                                                                         \
      fprintf(stderr, "[tern158] CUDA error %s at %s:%d : %s\n", cudaGetErrorName(_e), __FILE__,     \
              __LINE__, cudaGetErrorString(_e));                                                     \
      return T158_ERR_CUDA;                                                                          \
    }                                                                                                \
  } while (0)

#define T158_LAUNCH_CHECK()                                                                          \
  do {                                                                                               \
    cudaError_t _e = cudaGetLastError();                                                             \
    if (_e != cudaSuccess) {                                                                         \
      fprintf(stderr, "[tern158] launch error %s at %s:%d : %s\n", cudaGetErrorName(_e), __FILE__,   \
              __LINE__, cudaGetErrorString(_e));                                                     \
      return T158_ERR_CUDA;                                                                          \
    }                                                                                                \
  } while (0)

#define T158_REQUIRE(cond)                                                                           \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      fprintf(stderr, "[tern158] argument check failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__);   \
      return T158_ERR_ARG;                                                                           \
    }                                                                                                \
  } while (0)

namespace t158 {

constexpr int kWarp = 32;

inline int cdiv(int a, int b) { return (a + b - 1) / b; }
inline int64_t cdiv64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------------------------
T158_HD float to_f(float v) { return v; }
T158_HD float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
T158_HD float to_f(__half v) { return __half2float(v); }

template <typename T> T158_HD T from_f(float v);
template <> T158_HD float from_f<float>(float v) { return v; }
template <> T158_HD __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> T158_HD __half from_f<__half>(float v) { return __float2half(v); }

inline size_t dtype_size(t158DType d) { return d == T158_F32 ? 4 : 2; }

// Dispatch a lambda-less macro over the three element types. `T` is defined inside BODY.
#define T158_DISPATCH_DTYPE(dt, ...)                                                                \
  switch (dt) {                                                                                      \
    case T158_F32: { typedef float T; __VA_ARGS__; break; }                                                 \
    case T158_BF16: { typedef __nv_bfloat16 T; __VA_ARGS__; break; }                                        \
    case T158_F16: { typedef __half T; __VA_ARGS__; break; }                                                \
    default: return T158_ERR_ARG;                                                                    \
  }

// ---------------------------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------------------------
T158_D float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
T158_D float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
T158_D int warp_sum_i(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide sum. `sh` must hold >= 32 floats. All threads receive the result. Contains barriers.
T158_D float block_sum(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();  // protect `sh` against back-to-back use
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float r = 0.f;
  for (int i = 0; i < nw; ++i) r += sh[i];
  return r;
}
T158_D float block_max(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float r = sh[0];
  for (int i = 1; i < nw; ++i) r = fmaxf(r, sh[i]);
  return r;
}

// ---------------------------------------------------------------------------------------------
// ternary code helpers  (00 -> 0, 01 -> +1, 11 -> -1, 10 -> 0)
// ---------------------------------------------------------------------------------------------
T158_HD uint32_t enc_code(int t) { return t == 0 ? 0u : (t > 0 ? 1u : 3u); }
T158_HD int dec_code(uint32_t c) { return (c & 1u) ? ((c & 2u) ? -1 : 1) : 0; }

// 4 codes (low 8 bits of `b`) -> 4 packed int8 values (little endian: code0 in byte0).
T158_D uint32_t decode4_i8(uint32_t b) {
  uint32_t s = (b & 0x03u) | ((b & 0x0Cu) << 2) | ((b & 0x30u) << 4) | ((b & 0xC0u) << 6);
  // byte table: idx0 -> 0x00, idx1 -> 0x01, idx2 -> 0x00, idx3 -> 0xFF
  return __byte_perm(0xFF000100u, 0u, s);
}

// 16 codes -> 4 words of 4 int8 each.
T158_D void decode16_i8(uint32_t w, uint32_t out[4]) {
  out[0] = decode4_i8(w & 0xFFu);
  out[1] = decode4_i8((w >> 8) & 0xFFu);
  out[2] = decode4_i8((w >> 16) & 0xFFu);
  out[3] = decode4_i8((w >> 24) & 0xFFu);
}

// number of positions (of 16) where two code words differ
T158_D int code_diff_count(uint32_t a, uint32_t b) {
  uint32_t x = a ^ b;
  return __popc((x | (x >> 1)) & 0x55555555u);
}

// ---------------------------------------------------------------------------------------------
// RNG / stochastic rounding
// ---------------------------------------------------------------------------------------------
T158_D uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
T158_D uint32_t hash64to32(uint64_t seed, uint64_t idx) {
  uint64_t z = seed + idx * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return (uint32_t)(z >> 16) ^ (uint32_t)z;
}

// fp32 -> bf16 with stochastic rounding (unbiased): add random 16 low bits then truncate.
T158_D __nv_bfloat16 f2bf_sr(float f, uint32_t rnd) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return __float2bfloat16(f);  // inf / nan
  u += (rnd & 0xFFFFu);
  u &= 0xFFFF0000u;
  return __ushort_as_bfloat16((unsigned short)(u >> 16));
}

// ---------------------------------------------------------------------------------------------
// cp.async (sm_80+)
// ---------------------------------------------------------------------------------------------
T158_D void cp_async16(void* smem, const void* gmem, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = pred ? 16 : 0;  // src-size 0 -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(sz));
#else
  if (pred) *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
  else *reinterpret_cast<uint4*>(smem) = make_uint4(0, 0, 0, 0);
#endif
}
T158_D void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> T158_D void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// ---------------------------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------------------------
// round-to-nearest-even float -> int8 saturate
T158_D int8_t f2i8(float v) {
  int r = __float2int_rn(v);
  r = max(-127, min(127, r));
  return (int8_t)r;
}

}  // namespace t158
