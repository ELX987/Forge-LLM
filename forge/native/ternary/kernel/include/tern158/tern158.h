/*
 * tern158.h - public C API of libtern158.
 *
 * Ternary (1.58-bit) training / conversion / fine-tuning kernels for NVIDIA GPUs
 * (sm_80 .. sm_121: A100/A30/A10/A40, RTX 30/40/50, L4/L40, H100/H200, B200/B300, GB200/GB300,
 *  RTX Ada / RTX PRO Blackwell).
 *
 * Conventions
 *   - Every function is asynchronous on the given stream and returns a t158Status.
 *   - Weight matrices are row-major W[N][K]  (N = out features, K = in features), K contiguous.
 *   - Ternary codes are packed 2 bits per weight, 16 weights per uint32, along K:
 *         word = codes[n * (K/16) + k/16],  code j (0..15) sits in bits [2j+1 : 2j]
 *         00 -> 0,  01 -> +1,  11 -> -1,  10 -> reserved (decodes to 0)
 *   - Scales are fp32, layout scales[n * (K/group) + k/group]  (group == K  -> per-row,
 *     group == 0 in the API means "one scale per tensor", stored in scales[0]).
 *   - Activations are quantized per token (row) with absmax/127 -> int8 + fp32 scale.
 *   - K must be a multiple of 16 (GEMM: multiple of 32 recommended, 128 for best speed).
 */
#ifndef TERN158_H_
#define TERN158_H_

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define T158_VERSION_MAJOR 0
#define T158_VERSION_MINOR 1
#define T158_VERSION_PATCH 0

typedef enum {
  T158_OK = 0,
  T158_ERR_ARG = 1,        /* invalid argument / unsupported shape */
  T158_ERR_ARCH = 2,       /* kernel not available for this GPU */
  T158_ERR_CUDA = 3,       /* a CUDA runtime call failed (see cudaGetLastError) */
  T158_ERR_WORKSPACE = 4,  /* workspace too small */
} t158Status;

typedef enum { T158_F32 = 0, T158_BF16 = 1, T158_F16 = 2 } t158DType;

/* weight quantizer rule */
typedef enum {
  T158_Q_ABSMEAN = 0, /* BitNet b1.58: alpha = mean|w|, thr = alpha/2                       */
  T158_Q_TWN = 1,     /* thr = 0.75*mean|w|, alpha = E[|w| : |w|>thr]                        */
  T158_Q_LLOYD = 2,   /* absmean start, `iters` Lloyd steps: alpha=E[|w|:|w|>alpha/2]        */
} t158QuantMode;

/* GEMM backend selection */
typedef enum {
  T158_GEMM_AUTO = 0,  /* pick the best backend that passed its self check for this GPU */
  T158_GEMM_DP4A = 1,  /* CUDA-core reference (any arch)                                */
  T158_GEMM_WMMA = 2,  /* int8 tensor cores through nvcuda::wmma                         */
  T158_GEMM_MMA = 3,   /* int8 tensor cores through mma.sync m16n8k32 + cp.async         */
  T158_GEMM_BF16 = 4,  /* dequantize to bf16 workspace + cuBLAS (for GPUs with slow INT8, e.g. sm_103) */
} t158GemmBackend;

/* ------------------------------------------------------------------------------------------ */
/* device / arch                                                                              */
/* ------------------------------------------------------------------------------------------ */
typedef struct {
  int device;
  int sm;                 /* major*10+minor, e.g. 86, 89, 90, 100, 103, 120          */
  int sm_count;
  int max_smem_optin;     /* bytes, per block opt-in                                 */
  int l2_bytes;
  int mem_clock_khz;
  int mem_bus_bits;
  size_t total_mem;
  int int8_tc_fast;       /* 1 if INT8 tensor cores run at full rate (0 on sm_103)    */
  char name[128];
} t158DeviceInfo;

t158Status t158_get_device_info(int device, t158DeviceInfo* out);
const char* t158_status_string(t158Status s);
int t158_version(void);
/* Best default backend for the device (before self-check), as a t158GemmBackend value. */
int t158_default_backend(int sm);
/* Mark a backend as unusable (e.g. self check failed) or force one. device-global state. */
void t158_set_backend(int device, int backend);
int t158_get_backend(int device);

/* L2 persistence window for hot, re-read buffers (packed weights). Ampere+; no-op otherwise. */
t158Status t158_set_l2_persist(cudaStream_t stream, const void* ptr, size_t bytes, float hit_ratio);
t158Status t158_clear_l2_persist(cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* weight quantization / packing                                                              */
/* ------------------------------------------------------------------------------------------ */

/* Quantize an fp32/bf16/fp16 weight matrix to packed ternary codes + fp32 scales.
 *   group  : 0 -> one scale per tensor; otherwise elements per scale along K (multiple of 16,
 *            K % group == 0); group == K gives per-row scales.
 *   iters  : number of Lloyd iterations (T158_Q_LLOYD only)
 *   tmp    : fp32 scratch, >= 16 floats (8-byte aligned), only needed for group == 0 (may be NULL otherwise)
 *   stats  : optional int64[4] accumulator: [0]=flips vs old codes, [1]=zeros, [2]=count, [3]=unused
 *            (flips are only counted if count_flips != 0, in which case `codes` must hold the old codes)
 */
t158Status t158_quantize_weights(const void* w, t158DType wdt, int N, int K, int group, t158QuantMode mode,
                                 int iters, uint32_t* codes, float* scales, float* tmp, int64_t* stats,
                                 int count_flips, cudaStream_t stream);

/* ternary int8 {-1,0,1} -> packed and back */
t158Status t158_pack_i8(const int8_t* t, int N, int K, uint32_t* codes, cudaStream_t stream);
t158Status t158_unpack_i8(const uint32_t* codes, int N, int K, int8_t* t, cudaStream_t stream);

/* codes*scales -> dense matrix (bf16/fp16/fp32); blend: out = (1-lam)*ref + lam*dequant if ref != NULL */
t158Status t158_dequantize(const uint32_t* codes, const float* scales, int N, int K, int group,
                           void* out, t158DType odt, const void* ref, t158DType rdt, float lam,
                           cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* activations                                                                                */
/* ------------------------------------------------------------------------------------------ */

/* per-token absmax int8:  q[M,K] int8, s[M] fp32 (x ~= q*s) */
t158Status t158_act_quant(const void* x, t158DType xdt, int M, int K, int8_t* q, float* s, cudaStream_t stream);
/* int8 -> bf16/fp16/fp32 : out = q*s[row] */
t158Status t158_act_dequant(const int8_t* q, const float* s, int M, int K, void* out, t158DType odt,
                            cudaStream_t stream);

/* y = rmsnorm(x)*gamma ; optionally also int8-quantizes y per token (q,s) and/or writes y itself.
 * Any of y / q+s / rstd may be NULL.  Used for BitLinear input norm ("SubLN" / pre-norm). */
t158Status t158_rmsnorm_fwd(const void* x, const float* gamma, void* y, int8_t* q, float* s, float* rstd,
                            int M, int K, float eps, t158DType dt, cudaStream_t stream);
/* dx (+= if accumulate) and dgamma_partial[nb][K] (fp32; nb = min(M,256) rows are written, the caller sums
 * over them). gamma is fp32. */
t158Status t158_rmsnorm_bwd(const void* dy, const void* x, const float* gamma, const float* rstd, void* dx,
                            float* dgamma_partial, int M, int K, int accumulate_dx, t158DType dt,
                            cudaStream_t stream);

/* Fast Walsh-Hadamard transform on contiguous blocks of `block` elements (power of two, 16..8192);
 * x[M,K] with K % block == 0, orthonormal (scaled by 1/sqrt(block)); in-place if out==x. Self-inverse. */
t158Status t158_hadamard(const void* x, void* out, int M, int K, int block, t158DType dt, cudaStream_t stream);

/* silu(a)*b ; a,b are the two halves (or separate tensors) of the SwiGLU hidden. */
t158Status t158_swiglu_fwd(const void* a, const void* b, void* out, int64_t n, t158DType dt, cudaStream_t stream);
t158Status t158_swiglu_bwd(const void* dout, const void* a, const void* b, void* da, void* db, int64_t n,
                           t158DType dt, cudaStream_t stream);

/* Rotary embedding, in place on x[T, H, D] (rotate-half convention), positions = pos0 + t % seqlen.
 * cos/sin tables [max_pos, D/2] fp32. inverse != 0 applies the transpose (for backward). */
t158Status t158_rope(void* x, int T, int H, int D, int seqlen, const float* cos_t, const float* sin_t, int inverse,
                     t158DType dt, cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* matrix multiplication                                                                      */
/* ------------------------------------------------------------------------------------------ */

/* Forward:  Y[M,N] = (Xq[M,K] . T[N,K]^T) * sx[m] * sw[n, k/group]     (int32 accumulation per group)
 *   Xq int8 per-token quantized activations, T packed ternary codes, group = elements per weight scale
 *   (0 = per tensor: sw is a single float). Y is written as bf16/fp16/fp32.
 */
t158Status t158_gemm_fwd(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y,
                         t158DType ydt, int M, int N, int K, int group, int backend, void* ws, size_t ws_bytes,
                         cudaStream_t stream);
/* workspace needed by the given backend for these shapes (0 for the int8 backends). */
size_t t158_gemm_workspace_bytes(int M, int N, int K, int backend, int device);

/* Backward w.r.t. input:  dX[M,K] = dY[M,N] . (T[N,K]*sw)
 * Implemented as dequantize-to-bf16 (workspace) + cuBLAS GEMM (peak bf16 tensor-core rate, fp32 accumulate).
 * dY bf16 or fp16, dX bf16/fp16 (dX == fp32 with accumulate != 0 adds into dX).                         */
t158Status t158_gemm_bwd_data(const void* dy, t158DType dydt, const uint32_t* codes, const float* sw, void* dx,
                              t158DType dxdt, int M, int N, int K, int group, int accumulate, void* ws,
                              size_t ws_bytes, cudaStream_t stream);
/* workspace for bwd_data: N*K*2 bytes (dequantized bf16 weight). */
size_t t158_bwd_data_workspace_bytes(int N, int K);

/* Small-M decode GEMV (M <= 16): memory bound path using dp4a, same math as gemm_fwd. */
t158Status t158_gemv_fwd(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y,
                         t158DType ydt, int M, int N, int K, int group, cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* optimizer (latent-weight update)                                                           */
/* ------------------------------------------------------------------------------------------ */

typedef struct {
  float lr;
  float beta1, beta2, eps;
  float weight_decay;     /* decoupled (AdamW) */
  float grad_scale;       /* grads are multiplied by this (1/accum, clip factor, loss scale) */
  int64_t step;           /* 1-based */
  uint64_t seed;          /* stochastic rounding seed */
} t158AdamParams;

/* dtype codes: w (master): F32 or BF16(stochastic rounding); m,v: F32 or BF16(stochastic rounding);
 * g: F32 / BF16 / F16. wd_mask: apply weight decay (1) or not (0) for the whole tensor via p->weight_decay. */
t158Status t158_adam_update(void* w, t158DType wdt, const void* g, t158DType gdt, void* m, void* v,
                            t158DType sdt, int64_t n, const t158AdamParams* p, cudaStream_t stream);

/* dense sum of squares of a tensor into *acc (fp32, zeroed by caller) for global-norm clipping */
t158Status t158_sumsq(const void* g, t158DType dt, int64_t n, float* acc, cudaStream_t stream);
/* fp32 zero + isfinite flag: *flag |= (value not finite) */
t158Status t158_finite_check(const void* g, t158DType dt, int64_t n, int* flag, cudaStream_t stream);

/* statistics over packed codes: stats[0]=zeros, stats[1]=pos, stats[2]=neg (int64, caller zeroes) */
t158Status t158_code_stats(const uint32_t* codes, int64_t n_words, int64_t* stats, cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* losses                                                                                     */
/* ------------------------------------------------------------------------------------------ */

/* Fused cross entropy over logits[T,V]; ignore_index labels give 0 loss/grad. Writes
 *   loss_out[T] (fp32 per token) and overwrites logits with d(sum loss)/d logits * grad_scale
 * when compute_grad != 0. Chunk over T in the caller to bound memory.                           */
t158Status t158_cross_entropy(void* logits, t158DType dt, const int64_t* labels, int T, int V, int ignore_index,
                              float grad_scale, int compute_grad, float* loss_out, cudaStream_t stream);

/* Distillation KL(teacher || student) with temperature, optional hard-label mix:
 *   loss = sum_t T^2 * KL(softmax(t/T) || softmax(s/T)), grads written over `student_logits` (scaled by
 *   grad_scale).  teacher_logits is read only.                                                    */
t158Status t158_kl_div(void* student_logits, const void* teacher_logits, t158DType dt, int T, int V,
                       float temperature, float grad_scale, int compute_grad, float* loss_out, cudaStream_t stream);

/* ------------------------------------------------------------------------------------------ */
/* self test hooks (used by tern158_selftest and by the python module at import time)         */
/* ------------------------------------------------------------------------------------------ */

/* Runs gemm_fwd / bwd_data for each backend on random data against a CPU/dp4a reference and
 * marks failing backends unusable. Returns bitmask of passing backends (bit b = backend b).
 * `log` receives a human-readable report (may be NULL). */
int t158_validate_backends(int device, char* log, int log_len);

#ifdef __cplusplus
}
#endif
#endif /* TERN158_H_ */
