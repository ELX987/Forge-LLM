// optimizer.cu - AdamW latent-weight update with fp32 / bf16(stochastic rounding) master and moments,
// global-norm helpers.
#include "tern158/common.cuh"

using namespace t158;

namespace {

template <typename TS>
T158_D void store_state(TS* p, int64_t i, float v, uint32_t rnd) {
  if constexpr (sizeof(TS) == 4) p[i] = v;
  else p[i] = f2bf_sr(v, rnd);
}

// TW: master dtype (float | bf16), TS: moment dtype (float | bf16), TG: grad dtype
template <typename TW, typename TS, typename TG>
__global__ void adam_kernel(TW* __restrict__ w, const TG* __restrict__ g, TS* __restrict__ m, TS* __restrict__ v,
                            int64_t n, float lr, float b1, float b2, float eps, float wd, float gscale,
                            float bc1, float bc2, uint64_t seed) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const float gi = to_f(g[i]) * gscale;
    const float mi = b1 * to_f(m[i]) + (1.f - b1) * gi;
    const float vi = b2 * to_f(v[i]) + (1.f - b2) * gi * gi;
    const float mh = mi / bc1;
    const float vh = vi / bc2;
    float wi = to_f(w[i]);
    wi = wi * (1.f - lr * wd) - lr * mh / (sqrtf(vh) + eps);
    const uint32_t r = hash64to32(seed, (uint64_t)i);
    if constexpr (sizeof(TW) == 4) w[i] = wi;
    else w[i] = f2bf_sr(wi, r);
    store_state<TS>(m, i, mi, r * 2654435761u + 1u);
    store_state<TS>(v, i, vi, r * 40503u + 7u);
  }
}

template <typename TG>
__global__ void sumsq_kernel(const TG* __restrict__ g, int64_t n, float* acc) {
  __shared__ float sh[32];
  float s = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float v = to_f(g[i]);
    s += v * v;
  }
  s = block_sum(s, sh);
  if (threadIdx.x == 0) atomicAdd(acc, s);
}

template <typename TG>
__global__ void finite_kernel(const TG* __restrict__ g, int64_t n, int* flag) {
  int bad = 0;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    if (!isfinite(to_f(g[i]))) bad = 1;
  }
  if (bad) atomicOr(flag, 1);
}

int grid_n(int64_t n) {
  int64_t b = (n + 255) / 256;
  if (b < 1) b = 1;
  if (b > 8192) b = 8192;
  return (int)b;
}

template <typename TW, typename TS, typename TG>
void launch_adam(void* w, const void* g, void* m, void* v, int64_t n, const t158AdamParams* p, cudaStream_t st) {
  const float bc1 = 1.f - powf(p->beta1, (float)p->step);
  const float bc2 = 1.f - powf(p->beta2, (float)p->step);
  adam_kernel<TW, TS, TG><<<grid_n(n), 256, 0, st>>>((TW*)w, (const TG*)g, (TS*)m, (TS*)v, n, p->lr, p->beta1,
                                                      p->beta2, p->eps, p->weight_decay, p->grad_scale, bc1, bc2,
                                                      p->seed ^ ((uint64_t)p->step * 0x9E3779B97F4A7C15ULL));
}

template <typename TW, typename TS>
bool dispatch_g(t158DType gdt, void* w, const void* g, void* m, void* v, int64_t n, const t158AdamParams* p,
                cudaStream_t st) {
  switch (gdt) {
    case T158_F32: launch_adam<TW, TS, float>(w, g, m, v, n, p, st); return true;
    case T158_BF16: launch_adam<TW, TS, __nv_bfloat16>(w, g, m, v, n, p, st); return true;
    case T158_F16: launch_adam<TW, TS, __half>(w, g, m, v, n, p, st); return true;
  }
  return false;
}

}  // namespace

extern "C" {

t158Status t158_adam_update(void* w, t158DType wdt, const void* g, t158DType gdt, void* m, void* v, t158DType sdt,
                            int64_t n, const t158AdamParams* p, cudaStream_t stream) {
  T158_REQUIRE(w && g && m && v && p && n > 0 && p->step >= 1);
  T158_REQUIRE(wdt == T158_F32 || wdt == T158_BF16);
  T158_REQUIRE(sdt == T158_F32 || sdt == T158_BF16);
  bool ok = false;
  if (wdt == T158_F32 && sdt == T158_F32) ok = dispatch_g<float, float>(gdt, w, g, m, v, n, p, stream);
  else if (wdt == T158_F32 && sdt == T158_BF16) ok = dispatch_g<float, __nv_bfloat16>(gdt, w, g, m, v, n, p, stream);
  else if (wdt == T158_BF16 && sdt == T158_F32) ok = dispatch_g<__nv_bfloat16, float>(gdt, w, g, m, v, n, p, stream);
  else ok = dispatch_g<__nv_bfloat16, __nv_bfloat16>(gdt, w, g, m, v, n, p, stream);
  if (!ok) return T158_ERR_ARG;
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_sumsq(const void* g, t158DType dt, int64_t n, float* acc, cudaStream_t stream) {
  T158_REQUIRE(g && acc && n > 0);
  T158_DISPATCH_DTYPE(dt, sumsq_kernel<T><<<grid_n(n), 256, 0, stream>>>((const T*)g, n, acc));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_finite_check(const void* g, t158DType dt, int64_t n, int* flag, cudaStream_t stream) {
  T158_REQUIRE(g && flag && n > 0);
  T158_DISPATCH_DTYPE(dt, finite_kernel<T><<<grid_n(n), 256, 0, stream>>>((const T*)g, n, flag));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

}  // extern "C"
