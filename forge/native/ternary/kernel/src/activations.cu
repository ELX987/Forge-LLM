// activations.cu - per-token int8 quantization, RMSNorm (+fused quantization), Walsh-Hadamard, SwiGLU, RoPE.
#include "tern158/common.cuh"

using namespace t158;

namespace {

// ---- 8-element vector load/store -------------------------------------------------------------
template <typename T>
T158_D void load8(const T* src, float v[8]) {
  if constexpr (sizeof(T) == 4) {
    const float4* p = reinterpret_cast<const float4*>(src);
    float4 a = p[0], b = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    uint4 u = *reinterpret_cast<const uint4*>(src);
    const T* e = reinterpret_cast<const T*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = to_f(e[j]);
  }
}
template <typename T>
T158_D void store8(T* dst, const float v[8]) {
  if constexpr (sizeof(T) == 4) {
    float4* p = reinterpret_cast<float4*>(dst);
    p[0] = make_float4(v[0], v[1], v[2], v[3]);
    p[1] = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    uint4 u;
    T* e = reinterpret_cast<T*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) e[j] = from_f<T>(v[j]);
    *reinterpret_cast<uint4*>(dst) = u;
  }
}
T158_D void store8_i8(int8_t* dst, const float v[8], float inv) {
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    lo |= ((uint32_t)(uint8_t)f2i8(v[j] * inv)) << (8 * j);
    hi |= ((uint32_t)(uint8_t)f2i8(v[j + 4] * inv)) << (8 * j);
  }
  *reinterpret_cast<uint2*>(dst) = make_uint2(lo, hi);
}

// ---- per-token absmax int8 quantization -----------------------------------------------------
template <typename T>
__global__ void act_quant_kernel(const T* __restrict__ x, int K, int8_t* q, float* s) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  const T* xr = x + row * K;
  float mx = 0.f;
  for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
    float v[8];
    load8<T>(xr + i, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(v[j]));
  }
  mx = block_max(mx, sh);
  const float inv = mx > 0.f ? 127.f / mx : 0.f;
  for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
    float v[8];
    load8<T>(xr + i, v);
    store8_i8(q + row * K + i, v, inv);
  }
  if (threadIdx.x == 0) s[row] = mx / 127.f;
}

template <typename T>
__global__ void act_dequant_kernel(const int8_t* __restrict__ q, const float* __restrict__ s, int K, T* out) {
  const int64_t row = blockIdx.x;
  const float sc = s[row];
  for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
    uint2 u = *reinterpret_cast<const uint2*>(q + row * K + i);
    const int8_t* e = reinterpret_cast<const int8_t*>(&u);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (float)e[j] * sc;
    store8<T>(out + row * K + i, v);
  }
}

// ---- RMSNorm (+ optional int8 quantization of the output) -----------------------------------
template <typename T>
__global__ void rmsnorm_fwd_kernel(const T* __restrict__ x, const float* __restrict__ gamma, T* y, int8_t* q,
                                   float* s, float* rstd_out, int K, float eps) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  const T* xr = x + row * K;
  float ss = 0.f;
  for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
    float v[8];
    load8<T>(xr + i, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
  }
  ss = block_sum(ss, sh);
  const float rstd = rsqrtf(ss / (float)K + eps);
  if (threadIdx.x == 0 && rstd_out) rstd_out[row] = rstd;

  float mx = 0.f;
  for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
    float v[8];
    load8<T>(xr + i, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = v[j] * rstd * gamma[i + j];
      mx = fmaxf(mx, fabsf(v[j]));
    }
    if (y) store8<T>(y + row * K + i, v);
  }
  if (q) {
    mx = block_max(mx, sh);
    const float inv = mx > 0.f ? 127.f / mx : 0.f;
    for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
      float v[8];
      load8<T>(xr + i, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = v[j] * rstd * gamma[i + j];
      store8_i8(q + row * K + i, v, inv);
    }
    if (threadIdx.x == 0) s[row] = mx / 127.f;
  }
}

// dx = rstd * (dy*g - xhat * mean(dy*g*xhat)),  dgamma += dy * xhat
template <typename T>
__global__ void rmsnorm_bwd_kernel(const T* __restrict__ dy, const T* __restrict__ x, const float* __restrict__ gamma,
                                   const float* __restrict__ rstd, T* dx, float* dgp, int M, int K, int acc_dx) {
  __shared__ float sh[32];
  const int nb = gridDim.x;
  for (int64_t row = blockIdx.x; row < M; row += nb) {
    const float r = rstd[row];
    const T* dyr = dy + row * K;
    const T* xr = x + row * K;
    float dot = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
      float xh = to_f(xr[i]) * r;
      dot += to_f(dyr[i]) * gamma[i] * xh;
    }
    dot = block_sum(dot, sh) / (float)K;
    const bool first = (row == blockIdx.x);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
      float xh = to_f(xr[i]) * r;
      float d = to_f(dyr[i]);
      float g = r * (d * gamma[i] - xh * dot);
      if (acc_dx) g += to_f(dx[row * K + i]);
      dx[row * K + i] = from_f<T>(g);
      float dgv = d * xh;
      float* p = dgp + (int64_t)blockIdx.x * K + i;
      *p = first ? dgv : (*p + dgv);
    }
  }
}

// ---- Walsh-Hadamard on blocks of `blk` elements ----------------------------------------------
template <typename T>
__global__ void hadamard_kernel(const T* __restrict__ x, T* out, int K, int blk) {
  extern __shared__ float sm[];
  const int64_t base = (int64_t)blockIdx.x * K + (int64_t)blockIdx.y * blk;
  for (int i = threadIdx.x; i < blk; i += blockDim.x) sm[i] = to_f(x[base + i]);
  __syncthreads();
  for (int h = 1; h < blk; h <<= 1) {
    for (int p = threadIdx.x; p < blk / 2; p += blockDim.x) {
      int j = (p / h) * (2 * h) + (p % h);
      float a = sm[j], b = sm[j + h];
      sm[j] = a + b;
      sm[j + h] = a - b;
    }
    __syncthreads();
  }
  const float sc = rsqrtf((float)blk);
  for (int i = threadIdx.x; i < blk; i += blockDim.x) out[base + i] = from_f<T>(sm[i] * sc);
}

// ---- SwiGLU -----------------------------------------------------------------------------------
template <typename T>
__global__ void swiglu_fwd_kernel(const T* __restrict__ a, const T* __restrict__ b, T* out, int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float av = to_f(a[i]);
    float sg = 1.f / (1.f + __expf(-av));
    out[i] = from_f<T>(av * sg * to_f(b[i]));
  }
}
template <typename T>
__global__ void swiglu_bwd_kernel(const T* __restrict__ dout, const T* __restrict__ a, const T* __restrict__ b,
                                  T* da, T* db, int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float av = to_f(a[i]), bv = to_f(b[i]), d = to_f(dout[i]);
    float sg = 1.f / (1.f + __expf(-av));
    float silu = av * sg;
    da[i] = from_f<T>(d * bv * sg * (1.f + av * (1.f - sg)));
    db[i] = from_f<T>(d * silu);
  }
}

// ---- RoPE (rotate-half) ------------------------------------------------------------------------
template <typename T>
__global__ void rope_kernel(T* x, int T_, int H, int D, int seqlen, const float* cs, const float* sn, int inverse) {
  const int half = D / 2;
  const int64_t total = (int64_t)T_ * H * half;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
    int i = (int)(idx % half);
    int64_t th = idx / half;
    int h = (int)(th % H);
    int t = (int)(th / H);
    int pos = t % seqlen;
    float c = cs[pos * half + i], s = sn[pos * half + i];
    if (inverse) s = -s;
    T* p = x + ((int64_t)t * H + h) * D;
    float x1 = to_f(p[i]), x2 = to_f(p[i + half]);
    p[i] = from_f<T>(x1 * c - x2 * s);
    p[i + half] = from_f<T>(x2 * c + x1 * s);
  }
}

int grid1d(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b < 1) b = 1;
  if (b > 65535 * 8) b = 65535 * 8;
  return (int)b;
}
int row_threads(int K) {
  int t = K / 8;
  if (t >= 1024) return 1024;
  if (t >= 512) return 512;
  if (t >= 256) return 256;
  if (t >= 128) return 128;
  return 64;
}

}  // namespace

extern "C" {

t158Status t158_act_quant(const void* x, t158DType xdt, int M, int K, int8_t* q, float* s, cudaStream_t stream) {
  T158_REQUIRE(x && q && s && K % 8 == 0 && M > 0);
  const int th = row_threads(K);
  T158_DISPATCH_DTYPE(xdt, act_quant_kernel<T><<<M, th, 0, stream>>>((const T*)x, K, q, s));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_act_dequant(const int8_t* q, const float* s, int M, int K, void* out, t158DType odt,
                            cudaStream_t stream) {
  T158_REQUIRE(q && s && out && K % 8 == 0 && M > 0);
  const int th = row_threads(K);
  T158_DISPATCH_DTYPE(odt, act_dequant_kernel<T><<<M, th, 0, stream>>>(q, s, K, (T*)out));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_rmsnorm_fwd(const void* x, const float* gamma, void* y, int8_t* q, float* s, float* rstd, int M,
                            int K, float eps, t158DType dt, cudaStream_t stream) {
  T158_REQUIRE(x && gamma && K % 8 == 0 && M > 0);
  T158_REQUIRE((q == nullptr) == (s == nullptr));
  const int th = row_threads(K);
  T158_DISPATCH_DTYPE(dt, rmsnorm_fwd_kernel<T><<<M, th, 0, stream>>>((const T*)x, gamma, (T*)y, q, s, rstd, K, eps));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_rmsnorm_bwd(const void* dy, const void* x, const float* gamma, const float* rstd, void* dx,
                            float* dgamma_partial, int M, int K, int accumulate_dx, t158DType dt,
                            cudaStream_t stream) {
  T158_REQUIRE(dy && x && gamma && rstd && dx && dgamma_partial && M > 0 && K > 0);
  const int nb = M < 256 ? M : 256;
  int th = K >= 1024 ? 512 : 256;
  T158_DISPATCH_DTYPE(dt, rmsnorm_bwd_kernel<T><<<nb, th, 0, stream>>>((const T*)dy, (const T*)x, gamma, rstd, (T*)dx, dgamma_partial, M, K, accumulate_dx));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_hadamard(const void* x, void* out, int M, int K, int block, t158DType dt, cudaStream_t stream) {
  T158_REQUIRE(x && out && block >= 16 && block <= 8192 && (block & (block - 1)) == 0 && K % block == 0);
  dim3 grid(M, K / block);
  const int th = block / 2 < 1024 ? block / 2 : 1024;
  const size_t smem = (size_t)block * sizeof(float);
  T158_DISPATCH_DTYPE(dt, hadamard_kernel<T><<<grid, th, smem, stream>>>((const T*)x, (T*)out, K, block));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_swiglu_fwd(const void* a, const void* b, void* out, int64_t n, t158DType dt, cudaStream_t stream) {
  T158_REQUIRE(a && b && out && n > 0);
  T158_DISPATCH_DTYPE(dt, swiglu_fwd_kernel<T><<<grid1d(n, 256), 256, 0, stream>>>((const T*)a, (const T*)b, (T*)out, n));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_swiglu_bwd(const void* dout, const void* a, const void* b, void* da, void* db, int64_t n,
                           t158DType dt, cudaStream_t stream) {
  T158_REQUIRE(dout && a && b && da && db && n > 0);
  T158_DISPATCH_DTYPE(dt, swiglu_bwd_kernel<T><<<grid1d(n, 256), 256, 0, stream>>>((const T*)dout, (const T*)a, (const T*)b, (T*)da, (T*)db, n));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_rope(void* x, int T_, int H, int D, int seqlen, const float* cos_t, const float* sin_t, int inverse,
                     t158DType dt, cudaStream_t stream) {
  T158_REQUIRE(x && cos_t && sin_t && D % 2 == 0 && seqlen > 0);
  const int64_t total = (int64_t)T_ * H * (D / 2);
  T158_DISPATCH_DTYPE(dt, rope_kernel<T><<<grid1d(total, 256), 256, 0, stream>>>((T*)x, T_, H, D, seqlen, cos_t, sin_t, inverse));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

}  // extern "C"
