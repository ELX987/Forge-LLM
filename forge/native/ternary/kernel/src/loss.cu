// loss.cu - fused cross entropy and distillation KL over [T, V] logits (chunk over T in the caller).
#include "tern158/common.cuh"

using namespace t158;

namespace {

template <typename T>
__global__ void ce_kernel(T* logits, const int64_t* __restrict__ labels, int V, int ignore, float gscale,
                          int compute_grad, float* loss_out) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  T* lr = logits + row * V;
  const int64_t lab = labels[row];
  if (lab == ignore || lab < 0 || lab >= V) {
    if (compute_grad)
      for (int i = threadIdx.x; i < V; i += blockDim.x) lr[i] = from_f<T>(0.f);
    if (threadIdx.x == 0) loss_out[row] = 0.f;
    return;
  }
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, to_f(lr[i]));
  mx = block_max(mx, sh);
  float se = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) se += __expf(to_f(lr[i]) - mx);
  se = block_sum(se, sh);
  const float lse = mx + logf(se);
  if (threadIdx.x == 0) loss_out[row] = lse - to_f(lr[lab]);
  if (compute_grad) {
    __syncthreads();  // everyone has read lr[lab] before it is overwritten
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
      float p = __expf(to_f(lr[i]) - lse);
      if (i == lab) p -= 1.f;
      lr[i] = from_f<T>(p * gscale);
    }
  }
}

template <typename T>
__global__ void kl_kernel(T* student, const T* __restrict__ teacher, int V, float temp, float gscale,
                          int compute_grad, float* loss_out) {
  __shared__ float sh[32];
  const int64_t row = blockIdx.x;
  T* s = student + row * V;
  const T* t = teacher + row * V;
  const float it = 1.f / temp;
  float ms = -INFINITY, mt = -INFINITY;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    ms = fmaxf(ms, to_f(s[i]) * it);
    mt = fmaxf(mt, to_f(t[i]) * it);
  }
  ms = block_max(ms, sh);
  mt = block_max(mt, sh);
  float ss = 0.f, st = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    ss += __expf(to_f(s[i]) * it - ms);
    st += __expf(to_f(t[i]) * it - mt);
  }
  ss = block_sum(ss, sh);
  st = block_sum(st, sh);
  const float lss = ms + logf(ss), lst = mt + logf(st);
  float kl = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float lps = to_f(s[i]) * it - lss;
    const float lpt = to_f(t[i]) * it - lst;
    const float pt = __expf(lpt);
    kl += pt * (lpt - lps);
  }
  kl = block_sum(kl, sh);
  if (threadIdx.x == 0) loss_out[row] = temp * temp * kl;
  if (compute_grad) {
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
      const float ps = __expf(to_f(s[i]) * it - lss);
      const float pt = __expf(to_f(t[i]) * it - lst);
      s[i] = from_f<T>(temp * (ps - pt) * gscale);
    }
  }
}

int loss_threads(int V) { return V >= 8192 ? 1024 : (V >= 2048 ? 512 : 256); }

}  // namespace

extern "C" {

t158Status t158_cross_entropy(void* logits, t158DType dt, const int64_t* labels, int T_, int V, int ignore_index,
                              float grad_scale, int compute_grad, float* loss_out, cudaStream_t stream) {
  T158_REQUIRE(logits && labels && loss_out && T_ > 0 && V > 0);
  T158_DISPATCH_DTYPE(dt, ce_kernel<T><<<T_, loss_threads(V), 0, stream>>>((T*)logits, labels, V, ignore_index, grad_scale, compute_grad, loss_out));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

t158Status t158_kl_div(void* student_logits, const void* teacher_logits, t158DType dt, int T_, int V,
                       float temperature, float grad_scale, int compute_grad, float* loss_out, cudaStream_t stream) {
  T158_REQUIRE(student_logits && teacher_logits && loss_out && T_ > 0 && V > 0 && temperature > 0.f);
  T158_DISPATCH_DTYPE(dt, kl_kernel<T><<<T_, loss_threads(V), 0, stream>>>((T*)student_logits, (const T*)teacher_logits, V, temperature, grad_scale, compute_grad, loss_out));
  T158_LAUNCH_CHECK();
  return T158_OK;
}

}  // extern "C"
