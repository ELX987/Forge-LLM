// validate.cu - runtime self-verification of the GEMM backends against a host reference.
// Run once per process/device at start-up; failing backends are never selected.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tern158/common.cuh"

using namespace t158;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 2685821657736338717ULL + 1442695040888963407ULL) {}
  uint32_t next() {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (uint32_t)((s * 2685821657736338717ULL) >> 32);
  }
  float uni() { return (next() >> 8) * (1.0f / 16777216.0f); }
};

struct Log {
  char* buf; int cap; int len;
  void put(const char* fmt, ...) {
    if (!buf || cap <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + len, (size_t)(cap - len), fmt, ap);
    va_end(ap);
    if (n > 0) len = (len + n < cap) ? len + n : cap - 1;
  }
};

struct Case { int M, N, K, group; };

template <typename P>
P* dalloc(size_t n) {
  P* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(P)) != cudaSuccess) return nullptr;
  return p;
}

float bf16_round(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  u &= 0xFFFF0000u;
  float r; memcpy(&r, &u, 4);
  return r;
}

// returns max relative error (vs max |ref|) of the forward GEMM for one backend, or -1 on launch failure
double run_fwd_case(const Case& c, int backend, Rng& rng) {
  const int M = c.M, N = c.N, K = c.K, G = c.group == 0 ? K : c.group, ngr = K / G;
  std::vector<int8_t> xq((size_t)M * K), tern((size_t)N * K);
  std::vector<float> sx(M), sw(c.group == 0 ? 1 : (size_t)N * ngr);
  std::vector<uint32_t> codes((size_t)N * K / 16, 0u);
  for (auto& v : xq) v = (int8_t)((int)(rng.next() % 255) - 127);
  for (auto& v : tern) v = (int8_t)((int)(rng.next() % 3) - 1);
  for (auto& v : sx) v = 0.002f + 0.01f * rng.uni();
  for (auto& v : sw) v = 0.01f + 0.05f * rng.uni();
  for (size_t j = 0; j < codes.size(); ++j) {
    uint32_t w = 0;
    for (int i = 0; i < 16; ++i) w |= enc_code(tern[j * 16 + i]) << (2 * i);
    codes[j] = w;
  }
  std::vector<double> ref((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double tot = 0;
      for (int g = 0; g < ngr; ++g) {
        long long s = 0;
        for (int k = g * G; k < (g + 1) * G; ++k) s += (long long)xq[(size_t)m * K + k] * tern[(size_t)n * K + k];
        tot += (double)s * (c.group == 0 ? sw[0] : sw[(size_t)n * ngr + g]);
      }
      ref[(size_t)m * N + n] = tot * sx[m];
    }
  int8_t* dx = dalloc<int8_t>(xq.size());
  float *dsx = dalloc<float>(sx.size()), *dsw = dalloc<float>(sw.size()), *dy = dalloc<float>((size_t)M * N);
  uint32_t* dc = dalloc<uint32_t>(codes.size());
  size_t wsb = t158_gemm_workspace_bytes(M, N, K, backend, 0);
  void* ws = wsb ? (void*)dalloc<uint8_t>(wsb) : nullptr;
  double err = -1;
  if (dx && dsx && dsw && dy && dc && (!wsb || ws)) {
    cudaMemcpy(dx, xq.data(), xq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsx, sx.data(), sx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsw, sw.data(), sw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0xFF, (size_t)M * N * 4);  // NaN fill: untouched outputs fail the check
    t158Status st = t158_gemm_fwd(dx, dsx, dc, dsw, dy, T158_F32, M, N, K, c.group, backend, ws, wsb, 0);
    if (st == T158_OK && cudaDeviceSynchronize() == cudaSuccess) {
      std::vector<float> y((size_t)M * N);
      cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
      double mx = 0, me = 0;
      for (size_t i = 0; i < y.size(); ++i) mx = fmax(mx, fabs(ref[i]));
      for (size_t i = 0; i < y.size(); ++i) {
        double e = fabs((double)y[i] - ref[i]);
        if (!(e == e)) e = 1e30;
        me = fmax(me, e);
      }
      err = me / (mx > 0 ? mx : 1.0);
    } else {
      cudaGetLastError();
    }
  }
  cudaFree(dx); cudaFree(dsx); cudaFree(dsw); cudaFree(dy); cudaFree(dc); if (ws) cudaFree(ws);
  return err;
}

double run_bwd_case(const Case& c, Rng& rng) {
  const int M = c.M, N = c.N, K = c.K, G = c.group == 0 ? K : c.group, ngr = K / G;
  std::vector<float> dyf((size_t)M * N), sw(c.group == 0 ? 1 : (size_t)N * ngr);
  std::vector<uint16_t> dyb((size_t)M * N);
  std::vector<int8_t> tern((size_t)N * K);
  std::vector<uint32_t> codes((size_t)N * K / 16, 0u);
  for (size_t i = 0; i < dyf.size(); ++i) {
    float f = bf16_round((rng.uni() - 0.5f) * 2.f);
    dyf[i] = f;
    uint32_t u; memcpy(&u, &f, 4);
    dyb[i] = (uint16_t)(u >> 16);
  }
  for (auto& v : tern) v = (int8_t)((int)(rng.next() % 3) - 1);
  for (auto& v : sw) v = 0.01f + 0.05f * rng.uni();
  for (size_t j = 0; j < codes.size(); ++j) {
    uint32_t w = 0;
    for (int i = 0; i < 16; ++i) w |= enc_code(tern[j * 16 + i]) << (2 * i);
    codes[j] = w;
  }
  std::vector<double> ref((size_t)M * K, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        double s = c.group == 0 ? sw[0] : sw[(size_t)n * ngr + k / G];
        ref[(size_t)m * K + k] += (double)dyf[(size_t)m * N + n] * tern[(size_t)n * K + k] * s;
      }
  void *ddy = dalloc<uint16_t>(dyb.size()), *ddx = dalloc<float>((size_t)M * K), *ws = dalloc<uint8_t>(t158_bwd_data_workspace_bytes(N, K));
  float* dsw = dalloc<float>(sw.size());
  uint32_t* dc = dalloc<uint32_t>(codes.size());
  double err = -1;
  if (ddy && ddx && ws && dsw && dc) {
    cudaMemcpy(ddy, dyb.data(), dyb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dsw, sw.data(), sw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(ddx, 0, (size_t)M * K * 4);
    t158Status st = t158_gemm_bwd_data(ddy, T158_BF16, dc, dsw, ddx, T158_F32, M, N, K, c.group, 0, ws,
                                       t158_bwd_data_workspace_bytes(N, K), 0);
    if (st == T158_OK && cudaDeviceSynchronize() == cudaSuccess) {
      std::vector<float> o((size_t)M * K);
      cudaMemcpy(o.data(), ddx, o.size() * 4, cudaMemcpyDeviceToHost);
      double mx = 0, me = 0;
      for (size_t i = 0; i < o.size(); ++i) mx = fmax(mx, fabs(ref[i]));
      for (size_t i = 0; i < o.size(); ++i) {
        double e = fabs((double)o[i] - ref[i]);
        if (!(e == e)) e = 1e30;
        me = fmax(me, e);
      }
      err = me / (mx > 0 ? mx : 1.0);
    } else {
      cudaGetLastError();
    }
  }
  cudaFree(ddy); cudaFree(ddx); cudaFree(ws); cudaFree(dsw); cudaFree(dc);
  return err;
}

}  // namespace

extern "C" int t158_validate_backends(int device, char* logbuf, int log_len) {
  Log L{logbuf, log_len, 0};
  if (logbuf && log_len > 0) logbuf[0] = 0;
  if (cudaSetDevice(device) != cudaSuccess) return 0;
  t158DeviceInfo di;
  if (t158_get_device_info(device, &di) != T158_OK) return 0;
  L.put("device %d: %s  sm_%d  %d SMs  int8_tc_fast=%d\n", device, di.name, di.sm, di.sm_count, di.int8_tc_fast);

  const Case cases[] = {{37, 70, 256, 0},   {130, 150, 512, 128}, {64, 128, 128, 128}, {200, 96, 384, 64},
                        {5, 40, 256, 256},  {257, 130, 192, 64},  {1, 64, 128, 128},   {160, 256, 256, 32}};
  struct B { int id; const char* name; double tol; int min_sm; };
  const B backs[] = {{T158_GEMM_DP4A, "dp4a", 2e-4, 61}, {T158_GEMM_WMMA, "wmma-int8", 2e-4, 80},
                     {T158_GEMM_MMA, "mma-int8", 2e-4, 80}, {T158_GEMM_BF16, "bf16-cublas", 2e-2, 80}};
  Rng rng(12345);
  int mask = 0;
  for (const B& b : backs) {
    if (di.sm < b.min_sm) { L.put("  %-12s skipped (needs sm_%d)\n", b.name, b.min_sm); continue; }
    bool ok = true;
    double worst = 0;
    int ran = 0;
    for (const Case& c : cases) {
      // the int8 backends degrade by design for unsupported shapes; only test what they claim to support
      if (b.id == T158_GEMM_MMA && !(c.K % 64 == 0 && (c.group == 0 || c.group % 64 == 0))) continue;
      if (b.id == T158_GEMM_WMMA && !(c.K % 32 == 0 && (c.group == 0 || c.group % 32 == 0))) continue;
      double e = run_fwd_case(c, b.id, rng);
      ++ran;
      if (e < 0 || e > b.tol) { ok = false; L.put("  %-12s FAIL M=%d N=%d K=%d G=%d err=%.3e\n", b.name, c.M, c.N, c.K, c.group, e); break; }
      worst = fmax(worst, e);
    }
    if (ok) { mask |= 1 << b.id; L.put("  %-12s ok   (%d cases, worst rel err %.2e)\n", b.name, ran, worst); }
  }
  {
    bool ok = true;
    double worst = 0;
    const Case bc[] = {{70, 96, 128, 0}, {33, 130, 256, 128}, {128, 64, 64, 64}};
    for (const Case& c : bc) {
      double e = run_bwd_case(c, rng);
      if (e < 0 || e > 2e-2) { ok = false; L.put("  bwd_data     FAIL M=%d N=%d K=%d G=%d err=%.3e\n", c.M, c.N, c.K, c.group, e); break; }
      worst = fmax(worst, e);
    }
    if (ok) L.put("  bwd_data     ok   (worst rel err %.2e)\n", worst);
    else mask |= (1 << 30);  // flag: backward failed
  }
  // choose the default backend: fastest passing one that suits the GPU
  int pick = T158_GEMM_DP4A;
  if (di.int8_tc_fast) {
    if (mask & (1 << T158_GEMM_MMA)) pick = T158_GEMM_MMA;
    else if (mask & (1 << T158_GEMM_WMMA)) pick = T158_GEMM_WMMA;
  } else {
    if (mask & (1 << T158_GEMM_BF16)) pick = T158_GEMM_BF16;
    else if (mask & (1 << T158_GEMM_MMA)) pick = T158_GEMM_MMA;
  }
  t158_set_backend(device, pick);
  const char* names[] = {"auto", "dp4a", "wmma-int8", "mma-int8", "bf16-cublas"};
  L.put("  -> default forward backend: %s\n", names[pick]);
  return mask;
}
