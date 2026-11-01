// gemm.cu - ternary GEMM backends.
//   DP4A : CUDA-core reference / small-M GEMV (any sm >= 61)
//   WMMA : int8 tensor cores through nvcuda::wmma (layout agnostic safety net)
//   MMA  : int8 tensor cores through raw mma.sync.m16n8k32 + cp.async 3-stage pipeline (sm_80+)
//   BF16 : dequantize to bf16 workspace + cuBLAS  (sm_103 where INT8 tensor cores are crippled; also bwd_data)
//
// Math (all backends):  Y[m,n] = sx[m] * sum_g sw[n,g] * sum_{k in group g} xq[m,k] * T[n,k]
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <thread>

#include "tern158/common.cuh"

using namespace t158;
namespace wm = nvcuda::wmma;

namespace {

// =============================================================================================
// DP4A / GEMV
// =============================================================================================
template <typename OutT, int MT>
__global__ void __launch_bounds__(128)
gemv_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint32_t* __restrict__ codes,
            const float* __restrict__ sw, OutT* y, int M, int N, int K, int group) {
  const int warps_per_block = blockDim.x >> 5;
  const int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N) return;  // whole warp exits together
  const int lane = threadIdx.x & 31;
  const int W = K / 16;
  const int G = group == 0 ? K : group;
  const int ngr = K / G;
  const uint32_t* wr = codes + (int64_t)n * W;
  for (int m0 = blockIdx.y * MT; m0 < M; m0 += gridDim.y * MT) {
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  for (int w = lane; w < W; w += 32) {
    uint32_t d[4];
    decode16_i8(wr[w], d);
    const float scale = group == 0 ? sw[0] : sw[(int64_t)n * ngr + (w * 16) / G];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m0 + m < M) {
        const uint4 xv = *reinterpret_cast<const uint4*>(xq + (int64_t)(m0 + m) * K + (int64_t)w * 16);
        int s = __dp4a((int)xv.x, (int)d[0], 0);
        s = __dp4a((int)xv.y, (int)d[1], s);
        s = __dp4a((int)xv.z, (int)d[2], s);
        s = __dp4a((int)xv.w, (int)d[3], s);
        acc[m] += (float)s * scale;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0 && m0 + m < M) y[(int64_t)(m0 + m) * N + n] = from_f<OutT>(v * sx[m0 + m]);
  }
  }
}

// =============================================================================================
// WMMA int8 (64x64 tile, 4 warps of 32x32)
// =============================================================================================
template <typename OutT>
__global__ void __launch_bounds__(128)
gemm_wmma_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint32_t* __restrict__ codes,
                 const float* __restrict__ sw, OutT* y, int M, int N, int K, int group) {
  __shared__ __align__(128) int8_t As[2][64][16];
  __shared__ __align__(128) int8_t Bs[2][64][16];
  __shared__ __align__(128) int scratch[4][32 * 32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wmi = warp >> 1, wni = warp & 1;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  const int G = group == 0 ? K : group;
  const int ngr = K / G;
  const int tpg = G / 32;  // k-tiles (of 32) per group

  wm::fragment<wm::accumulator, 16, 16, 16, int> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wm::fill_fragment(acc[i][j], 0);
  float facc[32];
#pragma unroll
  for (int r = 0; r < 32; ++r) facc[r] = 0.f;

  const int lr = tid >> 1, lc = tid & 1;  // loader row / 16-byte chunk
  const int ktiles = K / 32;
  for (int kt = 0; kt < ktiles; ++kt) {
    const int k0 = kt * 32;
    {
      uint4 v = make_uint4(0, 0, 0, 0);
      if (m0 + lr < M) v = *reinterpret_cast<const uint4*>(xq + (int64_t)(m0 + lr) * K + k0 + lc * 16);
      *reinterpret_cast<uint4*>(&As[lc][lr][0]) = v;
      uint32_t cw = 0;
      if (n0 + lr < N) cw = codes[(int64_t)(n0 + lr) * (K / 16) + (k0 >> 4) + lc];
      uint32_t d[4];
      decode16_i8(cw, d);
      *reinterpret_cast<uint4*>(&Bs[lc][lr][0]) = make_uint4(d[0], d[1], d[2], d[3]);
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      wm::fragment<wm::matrix_a, 16, 16, 16, signed char, wm::row_major> a[2];
      wm::fragment<wm::matrix_b, 16, 16, 16, signed char, wm::col_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wm::load_matrix_sync(a[i], reinterpret_cast<const signed char*>(&As[ks][wmi * 32 + i * 16][0]), 16);
#pragma unroll
      for (int j = 0; j < 2; ++j) wm::load_matrix_sync(b[j], reinterpret_cast<const signed char*>(&Bs[ks][wni * 32 + j * 16][0]), 16);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wm::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
    if ((kt + 1) % tpg == 0) {
      const int g = kt / tpg;
      int* sc = scratch[warp];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wm::store_matrix_sync(sc + i * 16 * 32 + j * 16, acc[i][j], 32, wm::mem_row_major);
      __syncwarp();
      const int col = n0 + wni * 32 + lane;
      const float s = col < N ? (group == 0 ? sw[0] : sw[(int64_t)col * ngr + g]) : 0.f;
#pragma unroll
      for (int r = 0; r < 32; ++r) facc[r] += (float)sc[r * 32 + lane] * s;
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wm::fill_fragment(acc[i][j], 0);
    }
  }
  const int col = n0 + wni * 32 + lane;
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    const int row = m0 + wmi * 32 + r;
    if (row < M && col < N) y[(int64_t)row * N + col] = from_f<OutT>(facc[r] * sx[row]);
  }
}

// =============================================================================================
// MMA int8: mma.sync.m16n8k32, cp.async 3-stage pipeline.
//   tile BM x 128 x 64,   warp tile 64 x 32,   warps = (BM/64) x 4
// =============================================================================================
constexpr int kBN = 128, kBK = 64, kStages = 3;
constexpr int kAStride = 80;  // bytes per A smem row (64 data + 16 pad -> conflict free fragment loads)

T158_D void mma_s8(int& c0, int& c1, int& c2, int& c3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                   uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  __trap();
#endif
}

template <int WARPS_M, typename OutT>
__global__ void __launch_bounds__(WARPS_M * 4 * 32, 1)
gemm_mma_kernel(const int8_t* __restrict__ xq, const float* __restrict__ sx, const uint32_t* __restrict__ codes,
                const float* __restrict__ sw, OutT* y, int M, int N, int K, int group) {
  constexpr int BM = WARPS_M * 64;
  constexpr int NT = WARPS_M * 4 * 32;
  __shared__ __align__(16) uint8_t sA[kStages][BM * kAStride];
  __shared__ __align__(16) uint8_t sB[kStages][kBN * 16];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wmi = warp / 4, wni = warp % 4;
  const int gid = lane >> 2, tig = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * kBN;
  const int ktiles = K / kBK;
  const int G = group == 0 ? K : group;
  const int ngr = K / G;
  const int tpg = G / kBK;

  int acc[4][4][4];
  float facc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) { acc[a][b][e] = 0; facc[a][b][e] = 0.f; }

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * kBK;
    for (int c = tid; c < BM * 4; c += NT) {
      const int row = c >> 2, part = c & 3;
      const bool ok = (m0 + row) < M;
      const int8_t* src = xq + (int64_t)(ok ? (m0 + row) : 0) * K + k0 + part * 16;
      cp_async16(&sA[stage][row * kAStride + part * 16], src, ok);
    }
    for (int c = tid; c < kBN; c += NT) {
      const bool ok = (n0 + c) < N;
      const uint32_t* src = codes + (int64_t)(ok ? (n0 + c) : 0) * (K / 16) + (k0 >> 4);
      cp_async16(&sB[stage][c * 16], src, ok);
    }
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < ktiles) load_stage(s, s);
    cp_async_commit();
  }

  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nk = kt + kStages - 1;
      if (nk < ktiles) load_stage(nk % kStages, nk);
      cp_async_commit();
    }
    const uint8_t* A = sA[kt % kStages];
    const uint8_t* B = sB[kt % kStages];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t bf[4][2];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int nrow = wni * 32 + nt * 8 + gid;
        const uint32_t b0 = B[nrow * 16 + ks * 8 + tig];
        const uint32_t b1 = B[nrow * 16 + ks * 8 + 4 + tig];
        bf[nt][0] = decode4_i8(b0);
        bf[nt][1] = decode4_i8(b1);
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int r0 = wmi * 64 + mt * 16 + gid;
        const uint8_t* ap = A + r0 * kAStride + ks * 32 + tig * 4;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap + 8 * kAStride);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + 16);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap + 8 * kAStride + 16);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
          mma_s8(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3], a0, a1, a2, a3, bf[nt][0], bf[nt][1]);
      }
    }
    if ((kt + 1) % tpg == 0) {
      const int g = kt / tpg;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        float s0 = 0.f, s1 = 0.f;
        const int col = n0 + wni * 32 + nt * 8 + tig * 2;
        if (group == 0) {
          s0 = sw[0]; s1 = sw[0];
        } else {
          if (col < N) s0 = sw[(int64_t)col * ngr + g];
          if (col + 1 < N) s1 = sw[(int64_t)(col + 1) * ngr + g];
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          facc[mt][nt][0] += (float)acc[mt][nt][0] * s0;
          facc[mt][nt][1] += (float)acc[mt][nt][1] * s1;
          facc[mt][nt][2] += (float)acc[mt][nt][2] * s0;
          facc[mt][nt][3] += (float)acc[mt][nt][3] * s1;
          acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0;
        }
      }
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = m0 + wmi * 64 + mt * 16 + gid + (e >> 1) * 8;
        const int col = n0 + wni * 32 + nt * 8 + tig * 2 + (e & 1);
        if (row < M && col < N) y[(int64_t)row * N + col] = from_f<OutT>(facc[mt][nt][e] * sx[row]);
      }
    }
  }
}

// =============================================================================================
// cuBLAS helper
// =============================================================================================
struct BlasCache {
  std::mutex mu;
  std::map<std::pair<int, std::thread::id>, cublasHandle_t> h;
};
BlasCache& blas_cache() {
  static BlasCache c;
  return c;
}
cublasHandle_t get_handle() {
  int dev = 0;
  cudaGetDevice(&dev);
  BlasCache& c = blas_cache();
  std::lock_guard<std::mutex> lk(c.mu);
  auto key = std::make_pair(dev, std::this_thread::get_id());
  auto it = c.h.find(key);
  if (it != c.h.end()) return it->second;
  cublasHandle_t hd = nullptr;
  if (cublasCreate(&hd) != CUBLAS_STATUS_SUCCESS) return nullptr;
  c.h[key] = hd;
  return hd;
}

cudaDataType blas_type(t158DType d) {
  return d == T158_F32 ? CUDA_R_32F : (d == T158_BF16 ? CUDA_R_16BF : CUDA_R_16F);
}

// Row-major  C[M,Nn] = A[M,Kk] . op(B)   expressed with column-major cuBLAS
//   b_row_major_NK == true : B stored [Nn][Kk] row-major  (C = A . B^T)
//   b_row_major_NK == false: B stored [Kk][Nn] row-major  (C = A . B)
t158Status blas_gemm(const void* A, t158DType adt, const void* B, t158DType bdt, void* C, t158DType cdt, int M,
                     int Nn, int Kk, bool b_is_NK, float beta, cudaStream_t stream) {
  cublasHandle_t h = get_handle();
  if (!h) return T158_ERR_CUDA;
  if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS) return T158_ERR_CUDA;
  const float alpha = 1.f;
  cublasStatus_t st;
  if (b_is_NK) {
    // col-major: C^T[Nn,M] = B_c^T . A_c    with B_c = view [Kk x Nn] (ld Kk), A_c = view [Kk x M] (ld Kk)
    st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, Nn, M, Kk, &alpha, B, blas_type(bdt), Kk, A, blas_type(adt), Kk,
                      &beta, C, blas_type(cdt), Nn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else {
    // col-major: C^T[Nn,M] = B_c . A_c      with B_c = view [Nn x Kk] (ld Nn), A_c = view [Kk x M] (ld Kk)
    st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, Kk, &alpha, B, blas_type(bdt), Nn, A, blas_type(adt), Kk,
                      &beta, C, blas_type(cdt), Nn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[tern158] cublasGemmEx failed: %d\n", (int)st);
    return T158_ERR_CUDA;
  }
  return T158_OK;
}

template <typename OutT>
t158Status launch_gemv(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y, int M,
                       int N, int K, int group, cudaStream_t stream) {
  constexpr int MT = 8;
  dim3 grid((N + 3) / 4, std::min((M + MT - 1) / MT, 65535));
  gemv_kernel<OutT, MT><<<grid, 128, 0, stream>>>(xq, sx, codes, sw, (OutT*)y, M, N, K, group);
  T158_LAUNCH_CHECK();
  return T158_OK;
}

template <typename OutT>
t158Status launch_wmma(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y, int M,
                       int N, int K, int group, cudaStream_t stream) {
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  gemm_wmma_kernel<OutT><<<grid, 128, 0, stream>>>(xq, sx, codes, sw, (OutT*)y, M, N, K, group);
  T158_LAUNCH_CHECK();
  return T158_OK;
}

template <typename OutT>
t158Status launch_mma(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y, int M,
                      int N, int K, int group, cudaStream_t stream) {
  dim3 grid((N + kBN - 1) / kBN, 0);
  if (M <= 64) {
    grid.y = 1;
    gemm_mma_kernel<1, OutT><<<grid, 128, 0, stream>>>(xq, sx, codes, sw, (OutT*)y, M, N, K, group);
  } else {
    grid.y = (M + 127) / 128;
    gemm_mma_kernel<2, OutT><<<grid, 256, 0, stream>>>(xq, sx, codes, sw, (OutT*)y, M, N, K, group);
  }
  T158_LAUNCH_CHECK();
  return T158_OK;
}

bool mma_shape_ok(int K, int group) { return K % kBK == 0 && (group == 0 || (group % kBK == 0 && K % group == 0)); }
bool wmma_shape_ok(int K, int group) { return K % 32 == 0 && (group == 0 || (group % 32 == 0 && K % group == 0)); }

}  // namespace

extern "C" {

size_t t158_gemm_workspace_bytes(int M, int N, int K, int backend, int device) {
  (void)device;
  if (backend == T158_GEMM_BF16) return ((size_t)N * K + (size_t)M * K) * 2;
  return 0;
}

size_t t158_bwd_data_workspace_bytes(int N, int K) { return (size_t)N * K * 2; }

t158Status t158_gemv_fwd(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y,
                         t158DType ydt, int M, int N, int K, int group, cudaStream_t stream) {
  T158_REQUIRE(xq && sx && codes && sw && y && M > 0 && N > 0 && K % 16 == 0);
  T158_REQUIRE(group == 0 || (group % 16 == 0 && K % group == 0));
  switch (ydt) {
    case T158_F32: return launch_gemv<float>(xq, sx, codes, sw, y, M, N, K, group, stream);
    case T158_BF16: return launch_gemv<__nv_bfloat16>(xq, sx, codes, sw, y, M, N, K, group, stream);
    case T158_F16: return launch_gemv<__half>(xq, sx, codes, sw, y, M, N, K, group, stream);
  }
  return T158_ERR_ARG;
}

t158Status t158_gemm_fwd(const int8_t* xq, const float* sx, const uint32_t* codes, const float* sw, void* y,
                         t158DType ydt, int M, int N, int K, int group, int backend, void* ws, size_t ws_bytes,
                         cudaStream_t stream) {
  T158_REQUIRE(xq && sx && codes && sw && y && M > 0 && N > 0 && K > 0 && K % 16 == 0);
  T158_REQUIRE(group == 0 || (group % 16 == 0 && K % group == 0));
  int dev = 0;
  T158_CUDA_TRY(cudaGetDevice(&dev));
  if (backend == T158_GEMM_AUTO) backend = t158_get_backend(dev);

  // shape-based graceful degradation
  if (backend == T158_GEMM_MMA && !mma_shape_ok(K, group)) backend = T158_GEMM_WMMA;
  if (backend == T158_GEMM_WMMA && !wmma_shape_ok(K, group)) backend = T158_GEMM_DP4A;

  if (backend == T158_GEMM_BF16) {
    const size_t need = t158_gemm_workspace_bytes(M, N, K, backend, dev);
    if (ws == nullptr || ws_bytes < need) {
      // no workspace supplied: degrade to an int8 backend rather than fail
      backend = mma_shape_ok(K, group) ? T158_GEMM_MMA : (wmma_shape_ok(K, group) ? T158_GEMM_WMMA : T158_GEMM_DP4A);
    } else {
      // fp16 output needs fp16 operands (cuBLAS has no bf16-in / fp16-out); ints and ternary are exact in both
      const t158DType odt = (ydt == T158_F16) ? T158_F16 : T158_BF16;
      uint16_t* wd = reinterpret_cast<uint16_t*>(ws);
      uint16_t* xd = wd + (size_t)N * K;
      t158Status s = t158_dequantize(codes, sw, N, K, group, wd, odt, nullptr, T158_F32, 1.f, stream);
      if (s != T158_OK) return s;
      s = t158_act_dequant(xq, sx, M, K, xd, odt, stream);
      if (s != T158_OK) return s;
      return blas_gemm(xd, odt, wd, odt, y, ydt, M, N, K, true, 0.f, stream);
    }
  }
  if (backend == T158_GEMM_DP4A || (M <= 16 && backend != T158_GEMM_BF16)) {  // decode regime: bandwidth-bound GEMV wins (measured)
    if (M <= 16 || backend == T158_GEMM_DP4A) return t158_gemv_fwd(xq, sx, codes, sw, y, ydt, M, N, K, group, stream);
  }
  switch (backend) {
    case T158_GEMM_WMMA:
      switch (ydt) {
        case T158_F32: return launch_wmma<float>(xq, sx, codes, sw, y, M, N, K, group, stream);
        case T158_BF16: return launch_wmma<__nv_bfloat16>(xq, sx, codes, sw, y, M, N, K, group, stream);
        case T158_F16: return launch_wmma<__half>(xq, sx, codes, sw, y, M, N, K, group, stream);
      }
      break;
    case T158_GEMM_MMA:
      switch (ydt) {
        case T158_F32: return launch_mma<float>(xq, sx, codes, sw, y, M, N, K, group, stream);
        case T158_BF16: return launch_mma<__nv_bfloat16>(xq, sx, codes, sw, y, M, N, K, group, stream);
        case T158_F16: return launch_mma<__half>(xq, sx, codes, sw, y, M, N, K, group, stream);
      }
      break;
    default: break;
  }
  return T158_ERR_ARG;
}

t158Status t158_gemm_bwd_data(const void* dy, t158DType dydt, const uint32_t* codes, const float* sw, void* dx,
                              t158DType dxdt, int M, int N, int K, int group, int accumulate, void* ws,
                              size_t ws_bytes, cudaStream_t stream) {
  T158_REQUIRE(dy && codes && sw && dx && M > 0 && N > 0 && K > 0 && K % 16 == 0);
  T158_REQUIRE(dydt == T158_BF16 || dydt == T158_F16);
  T158_REQUIRE(ws != nullptr && ws_bytes >= t158_bwd_data_workspace_bytes(N, K));
  T158_REQUIRE(!accumulate || dxdt == T158_F32);
  T158_REQUIRE(dxdt == dydt || dxdt == T158_F32);
  // dequantize in the dtype of dY so both cuBLAS operands match
  t158Status s = t158_dequantize(codes, sw, N, K, group, ws, dydt, nullptr, T158_F32, 1.f, stream);
  if (s != T158_OK) return s;
  // dX[M,K] = dY[M,N] . Wd[N,K]   (B is [Kk=N][Nn=K] row-major)
  return blas_gemm(dy, dydt, ws, dydt, dx, dxdt, M, K, N, false, accumulate ? 1.f : 0.f, stream);
}

}  // extern "C"
