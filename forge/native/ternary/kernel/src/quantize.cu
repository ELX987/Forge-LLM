// quantize.cu - weight quantization (absmean / TWN / Lloyd), packing, dequantization, code statistics.
#include "tern158/common.cuh"

using namespace t158;

namespace {

constexpr int kQThreads = 256;

// ---------------------------------------------------------------------------------------------
// vector helpers: load/store 16 consecutive elements of type T as floats
// ---------------------------------------------------------------------------------------------
template <typename T>
T158_D void load16(const T* src, float v[16]) {
  if constexpr (sizeof(T) == 4) {
    const float4* p = reinterpret_cast<const float4*>(src);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float4 f = p[i];
      v[4 * i + 0] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
    }
  } else {
    const uint4* p = reinterpret_cast<const uint4*>(src);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 u = p[i];
      const T* e = reinterpret_cast<const T*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) v[8 * i + j] = to_f(e[j]);
    }
  }
}

template <typename T>
T158_D void store16(T* dst, const float v[16]) {
  if constexpr (sizeof(T) == 4) {
    float4* p = reinterpret_cast<float4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
  } else {
    uint4* p = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 u;
      T* e = reinterpret_cast<T*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) e[j] = from_f<T>(v[8 * i + j]);
      p[i] = u;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// per-segment kernel: one block per (row n, group g) segment of G elements
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(kQThreads)
segment_quant_kernel(const T* __restrict__ w, int K, int G, int mode, int iters, uint32_t* codes,
                     float* scales, int64_t* stats, int count_flips) {
  __shared__ float sh[32];
  const int ngroups = K / G;
  const int seg = blockIdx.x;  // = n * ngroups + g
  const int n = seg / ngroups, g = seg - n * ngroups;
  const T* src = w + (int64_t)n * K + (int64_t)g * G;
  const int tid = threadIdx.x;

  // mean |w|
  float s = 0.f;
  for (int i = tid; i < G; i += blockDim.x) s += fabsf(to_f(src[i]));
  float mean = block_sum(s, sh) / (float)G;

  float alpha = mean, thr = 0.5f * mean;
  if (mode == T158_Q_TWN) {
    thr = 0.75f * mean;
    float hs = 0.f, hc = 0.f;
    for (int i = tid; i < G; i += blockDim.x) {
      float a = fabsf(to_f(src[i]));
      if (a > thr) { hs += a; hc += 1.f; }
    }
    hs = block_sum(hs, sh);
    hc = block_sum(hc, sh);
    alpha = hc > 0.f ? hs / hc : mean;
  } else if (mode == T158_Q_LLOYD) {
    for (int it = 0; it < iters; ++it) {
      float t = 0.5f * alpha, hs = 0.f, hc = 0.f;
      for (int i = tid; i < G; i += blockDim.x) {
        float a = fabsf(to_f(src[i]));
        if (a > t) { hs += a; hc += 1.f; }
      }
      hs = block_sum(hs, sh);
      hc = block_sum(hc, sh);
      alpha = hc > 0.f ? hs / hc : alpha;
    }
    thr = 0.5f * alpha;
  }
  if (!(alpha > 0.f)) { alpha = 0.f; thr = INFINITY; }  // all-zero segment -> all zero codes

  // quantize + pack: each thread handles whole 16-element words
  const int words = G / 16;
  uint32_t* dst = codes + (int64_t)n * (K / 16) + (int64_t)g * words;
  float zeros = 0.f, flips = 0.f;
  for (int j = tid; j < words; j += blockDim.x) {
    float v[16];
    load16<T>(src + (int64_t)j * 16, v);
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float a = fabsf(v[i]);
      uint32_t c = (a > thr) ? (v[i] > 0.f ? 1u : 3u) : 0u;
      word |= c << (2 * i);
    }
    if (count_flips) flips += (float)code_diff_count(dst[j], word);
    zeros += (float)(16 - __popc(word & 0x55555555u));
    dst[j] = word;
  }
  if (tid == 0) scales[seg] = alpha;
  if (stats) {
    zeros = block_sum(zeros, sh);
    flips = block_sum(flips, sh);
    if (tid == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[0]), (unsigned long long)flips);
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[1]), (unsigned long long)zeros);
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[2]), (unsigned long long)G);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// per-tensor path  (tmp as double[8]: 0=sum|w| 1=sum hi 2=count 3=thr 4=alpha 5=n)
// ---------------------------------------------------------------------------------------------
__global__ void tensor_reset_kernel(double* acc, double n, int full) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0;
    if (full) { acc[3] = 0.0; acc[4] = 0.0; acc[5] = n; }
  }
}

template <typename T>
__global__ void tensor_stats_kernel(const T* __restrict__ w, int64_t n, double* acc) {
  __shared__ float sh[32];
  const float thr = (float)acc[3];
  float s = 0.f, hs = 0.f, hc = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float a = fabsf(to_f(w[i]));
    s += a;
    if (a > thr) { hs += a; hc += 1.f; }
  }
  s = block_sum(s, sh);
  hs = block_sum(hs, sh);
  hc = block_sum(hc, sh);
  if (threadIdx.x == 0) {
    atomicAdd(&acc[0], (double)s);
    atomicAdd(&acc[1], (double)hs);
    atomicAdd(&acc[2], (double)hc);
  }
}

// phase 0: after the first pass (thr == 0): alpha = mean. phase 1: after a thresholded pass.
__global__ void tensor_update_kernel(double* acc, int mode, int phase) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  double n = acc[5];
  if (phase == 0) {
    double mean = acc[0] / n;
    acc[4] = mean;
    acc[3] = (mode == T158_Q_TWN) ? 0.75 * mean : 0.5 * mean;
  } else {
    double a = acc[2] > 0.0 ? acc[1] / acc[2] : acc[4];
    acc[4] = a;
    if (mode != T158_Q_TWN) acc[3] = 0.5 * a;
  }
}

template <typename T>
__global__ void tensor_quant_kernel(const T* __restrict__ w, int64_t n_words, const double* acc,
                                    uint32_t* codes, float* scales, int64_t* stats, int count_flips) {
  __shared__ float sh[32];
  const float thr = (float)acc[3];
  const float alpha = (float)acc[4];
  const bool dead = !(alpha > 0.f);
  float zeros = 0.f, flips = 0.f, cnt = 0.f;
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n_words; j += (int64_t)gridDim.x * blockDim.x) {
    float v[16];
    cnt += 16.f;
    load16<T>(w + j * 16, v);
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float a = fabsf(v[i]);
      uint32_t c = (!dead && a > thr) ? (v[i] > 0.f ? 1u : 3u) : 0u;
      word |= c << (2 * i);
    }
    if (count_flips) flips += (float)code_diff_count(codes[j], word);
    zeros += (float)(16 - __popc(word & 0x55555555u));
    codes[j] = word;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) scales[0] = dead ? 0.f : alpha;
  if (stats) {
    zeros = block_sum(zeros, sh);
    flips = block_sum(flips, sh);
    cnt = block_sum(cnt, sh);
    if (threadIdx.x == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[0]), (unsigned long long)flips);
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[1]), (unsigned long long)zeros);
      atomicAdd(reinterpret_cast<unsigned long long*>(&stats[2]), (unsigned long long)cnt);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// pack / unpack int8
// ---------------------------------------------------------------------------------------------
__global__ void pack_i8_kernel(const int8_t* __restrict__ t, int64_t n_words, uint32_t* codes) {
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n_words; j += (int64_t)gridDim.x * blockDim.x) {
    const int4* p = reinterpret_cast<const int4*>(t + j * 16);
    int4 q = *p;
    const int8_t* e = reinterpret_cast<const int8_t*>(&q);
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) word |= enc_code((int)e[i]) << (2 * i);
    codes[j] = word;
  }
}

__global__ void unpack_i8_kernel(const uint32_t* __restrict__ codes, int64_t n_words, int8_t* t) {
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n_words; j += (int64_t)gridDim.x * blockDim.x) {
    uint32_t o[4];
    decode16_i8(codes[j], o);
    reinterpret_cast<uint4*>(t + j * 16)[0] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// ---------------------------------------------------------------------------------------------
// dequantize (with optional blend toward a reference tensor)
// ---------------------------------------------------------------------------------------------
template <typename TO, typename TR>
__global__ void dequant_kernel(const uint32_t* __restrict__ codes, const float* __restrict__ scales, int K,
                               int G, int64_t n_words, TO* out, const TR* ref, float lam) {
  const int wpr = K / 16;  // words per row
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n_words; j += (int64_t)gridDim.x * blockDim.x) {
    int64_t n = j / wpr;
    int kw = (int)(j - n * wpr);
    float sc;
    if (G == 0) sc = scales[0];
    else sc = scales[n * (K / G) + (kw * 16) / G];
    uint32_t w = codes[j];
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = sc * (float)dec_code((w >> (2 * i)) & 3u);
    if (ref != nullptr) {
      float r[16];
      load16<TR>(ref + j * 16, r);
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = r[i] + lam * (v[i] - r[i]);
    }
    store16<TO>(out + j * 16, v);
  }
}

__global__ void code_stats_kernel(const uint32_t* __restrict__ codes, int64_t n_words, int64_t* stats) {
  __shared__ float sh[32];
  float zr = 0.f, pos = 0.f, neg = 0.f;
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < n_words; j += (int64_t)gridDim.x * blockDim.x) {
    uint32_t w = codes[j];
    uint32_t mag = w & 0x55555555u;
    uint32_t sgn = (w >> 1) & mag;
    zr += (float)(16 - __popc(mag));
    neg += (float)__popc(sgn);
    pos += (float)(__popc(mag) - __popc(sgn));
  }
  zr = block_sum(zr, sh);
  pos = block_sum(pos, sh);
  neg = block_sum(neg, sh);
  if (threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&stats[0]), (unsigned long long)zr);
    atomicAdd(reinterpret_cast<unsigned long long*>(&stats[1]), (unsigned long long)pos);
    atomicAdd(reinterpret_cast<unsigned long long*>(&stats[2]), (unsigned long long)neg);
  }
}

int grid_for(int64_t n, int threads, int sm_cap = 0) {
  int64_t b = (n + threads - 1) / threads;
  int cap = sm_cap > 0 ? sm_cap : 65535 * 4;
  return (int)(b < cap ? (b < 1 ? 1 : b) : cap);
}

}  // namespace

extern "C" {

t158Status t158_quantize_weights(const void* w, t158DType wdt, int N, int K, int group, t158QuantMode mode,
                                 int iters, uint32_t* codes, float* scales, float* tmp, int64_t* stats,
                                 int count_flips, cudaStream_t stream) {
  T158_REQUIRE(w && codes && scales);
  T158_REQUIRE(N > 0 && K > 0 && K % 16 == 0);
  T158_REQUIRE(group == 0 || (group % 16 == 0 && K % group == 0));
  T158_REQUIRE(iters >= 0 && iters <= 64);
  if (group != 0) {
    const int segs = N * (K / group);
    switch (wdt) {
      case T158_F32:
        segment_quant_kernel<float><<<segs, kQThreads, 0, stream>>>((const float*)w, K, group, mode, iters, codes, scales, stats, count_flips);
        break;
      case T158_BF16:
        segment_quant_kernel<__nv_bfloat16><<<segs, kQThreads, 0, stream>>>((const __nv_bfloat16*)w, K, group, mode, iters, codes, scales, stats, count_flips);
        break;
      case T158_F16:
        segment_quant_kernel<__half><<<segs, kQThreads, 0, stream>>>((const __half*)w, K, group, mode, iters, codes, scales, stats, count_flips);
        break;
      default: return T158_ERR_ARG;
    }
    T158_LAUNCH_CHECK();
    return T158_OK;
  }
  // per-tensor scale
  T158_REQUIRE(tmp != nullptr);
  double* acc = reinterpret_cast<double*>(tmp);
  const int64_t n = (int64_t)N * K, n_words = n / 16;
  const int grid = grid_for(n, kQThreads, 1024);
  const int gridq = grid_for(n_words, kQThreads, 1024);
  tensor_reset_kernel<<<1, 1, 0, stream>>>(acc, (double)n, 1);
#define T158_TENSOR_STATS()                                                                          \
  switch (wdt) {                                                                                     \
    case T158_F32: tensor_stats_kernel<float><<<grid, kQThreads, 0, stream>>>((const float*)w, n, acc); break; \
    case T158_BF16: tensor_stats_kernel<__nv_bfloat16><<<grid, kQThreads, 0, stream>>>((const __nv_bfloat16*)w, n, acc); break; \
    case T158_F16: tensor_stats_kernel<__half><<<grid, kQThreads, 0, stream>>>((const __half*)w, n, acc); break; \
    default: return T158_ERR_ARG;                                                                    \
  }
  T158_TENSOR_STATS();
  tensor_update_kernel<<<1, 1, 0, stream>>>(acc, (int)mode, 0);
  int passes = (mode == T158_Q_TWN) ? 1 : (mode == T158_Q_LLOYD ? iters : 0);
  for (int i = 0; i < passes; ++i) {
    tensor_reset_kernel<<<1, 1, 0, stream>>>(acc, (double)n, 0);
    T158_TENSOR_STATS();
    tensor_update_kernel<<<1, 1, 0, stream>>>(acc, (int)mode, 1);
  }
#undef T158_TENSOR_STATS
  switch (wdt) {
    case T158_F32: tensor_quant_kernel<float><<<gridq, kQThreads, 0, stream>>>((const float*)w, n_words, acc, codes, scales, stats, count_flips); break;
    case T158_BF16: tensor_quant_kernel<__nv_bfloat16><<<gridq, kQThreads, 0, stream>>>((const __nv_bfloat16*)w, n_words, acc, codes, scales, stats, count_flips); break;
    case T158_F16: tensor_quant_kernel<__half><<<gridq, kQThreads, 0, stream>>>((const __half*)w, n_words, acc, codes, scales, stats, count_flips); break;
    default: return T158_ERR_ARG;
  }
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_pack_i8(const int8_t* t, int N, int K, uint32_t* codes, cudaStream_t stream) {
  T158_REQUIRE(t && codes && K % 16 == 0);
  int64_t nw = (int64_t)N * K / 16;
  pack_i8_kernel<<<grid_for(nw, 256), 256, 0, stream>>>(t, nw, codes);
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_unpack_i8(const uint32_t* codes, int N, int K, int8_t* t, cudaStream_t stream) {
  T158_REQUIRE(t && codes && K % 16 == 0);
  int64_t nw = (int64_t)N * K / 16;
  unpack_i8_kernel<<<grid_for(nw, 256), 256, 0, stream>>>(codes, nw, t);
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_dequantize(const uint32_t* codes, const float* scales, int N, int K, int group, void* out,
                           t158DType odt, const void* ref, t158DType rdt, float lam, cudaStream_t stream) {
  T158_REQUIRE(codes && scales && out && K % 16 == 0);
  T158_REQUIRE(group == 0 || (group % 16 == 0 && K % group == 0));
  int64_t nw = (int64_t)N * K / 16;
  const int g = grid_for(nw, 256);
#define T158_DQ(TOUT, TREF, REFPTR)                                                                  \
  dequant_kernel<TOUT, TREF><<<g, 256, 0, stream>>>(codes, scales, K, group, nw, (TOUT*)out, REFPTR, lam)
  if (ref == nullptr) {
    switch (odt) {
      case T158_F32: T158_DQ(float, float, (const float*)nullptr); break;
      case T158_BF16: T158_DQ(__nv_bfloat16, float, (const float*)nullptr); break;
      case T158_F16: T158_DQ(__half, float, (const float*)nullptr); break;
      default: return T158_ERR_ARG;
    }
  } else {
    // blend path: reference is kept in rdt, output in odt (only the combinations used by the trainer)
    if (odt == T158_BF16 && rdt == T158_F32) { T158_DQ(__nv_bfloat16, float, (const float*)ref); }
    else if (odt == T158_BF16 && rdt == T158_BF16) { T158_DQ(__nv_bfloat16, __nv_bfloat16, (const __nv_bfloat16*)ref); }
    else if (odt == T158_F32 && rdt == T158_F32) { T158_DQ(float, float, (const float*)ref); }
    else if (odt == T158_F16 && rdt == T158_F32) { T158_DQ(__half, float, (const float*)ref); }
    else return T158_ERR_ARG;
  }
#undef T158_DQ
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_code_stats(const uint32_t* codes, int64_t n_words, int64_t* stats, cudaStream_t stream) {
  T158_REQUIRE(codes && stats);
  int grid = grid_for(n_words, 256, 1024);
  code_stats_kernel<<<grid, 256, 0, stream>>>(codes, n_words, stats);
  T158_LAUNCH_CHECK();
  return T158_OK;
}

}  // extern "C"
