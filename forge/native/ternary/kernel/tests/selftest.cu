// selftest.cu - standalone validation + micro-benchmark for libtern158 (no PyTorch needed).
//   ./tern158_selftest [--device N] [--bench] [--json out.json] [--quick]
// Exit code 0 when every correctness test passes.  Paste the whole output into a bug report.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tern158/common.cuh"

using namespace t158;

static int g_fail = 0, g_pass = 0;
static std::vector<std::string> g_json;

static void report(const char* name, bool ok, const char* fmt = "", double a = 0, double b = 0) {
  char det[256];
  snprintf(det, sizeof(det), fmt, a, b);
  printf("[%s] %-34s %s\n", ok ? " OK " : "FAIL", name, det);
  char js[400];
  snprintf(js, sizeof(js), "{\"test\":\"%s\",\"ok\":%s}", name, ok ? "true" : "false");
  g_json.push_back(js);
  (ok ? g_pass : g_fail)++;
}

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } \
  } while (0)

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (uint32_t)((s * 2685821657736338717ULL) >> 32); }
  float uni() { return (next() >> 8) * (1.0f / 16777216.0f); }
  float normal() {
    float u1 = fmaxf(uni(), 1e-7f), u2 = uni();
    return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
  }
};

template <typename P> static P* dmalloc(size_t n) { P* p = nullptr; cudaMalloc(&p, n * sizeof(P)); return p; }
template <typename P> static P* to_dev(const std::vector<P>& v) {
  P* p = dmalloc<P>(v.size());
  cudaMemcpy(p, v.data(), v.size() * sizeof(P), cudaMemcpyHostToDevice);
  return p;
}
template <typename P> static std::vector<P> from_dev(const P* d, size_t n) {
  std::vector<P> v(n);
  cudaMemcpy(v.data(), d, n * sizeof(P), cudaMemcpyDeviceToHost);
  return v;
}

// ---------------------------------------------------------------------------------------------
static void host_quant_ref(const float* w, int G, int mode, int iters, float& alpha, float& thr) {
  double mean = 0;
  for (int i = 0; i < G; ++i) mean += fabs(w[i]);
  mean /= G;
  alpha = (float)mean; thr = 0.5f * alpha;
  auto cond = [&](float t) { double s = 0; int c = 0; for (int i = 0; i < G; ++i) { float a = fabsf(w[i]); if (a > t) { s += a; ++c; } } return c ? (float)(s / c) : -1.f; };
  if (mode == T158_Q_TWN) { thr = 0.75f * (float)mean; float a = cond(thr); alpha = a > 0 ? a : (float)mean; }
  if (mode == T158_Q_LLOYD) {
    for (int i = 0; i < iters; ++i) { float a = cond(0.5f * alpha); if (a > 0) alpha = a; }
    thr = 0.5f * alpha;
  }
}

static int test_quant(Rng& rng) {
  const int N = 48, K = 512;
  std::vector<float> w((size_t)N * K);
  for (auto& v : w) v = 0.02f * rng.normal();
  float* dw = to_dev(w);
  struct M { int mode; int iters; const char* nm; } modes[] = {{T158_Q_ABSMEAN, 0, "absmean"}, {T158_Q_TWN, 0, "twn"}, {T158_Q_LLOYD, 3, "lloyd3"}};
  for (int gsel = 0; gsel < 3; ++gsel) {
    int G = gsel == 0 ? 128 : (gsel == 1 ? K : 0);
    for (auto& m : modes) {
      int ngr = G ? K / G : 1;
      uint32_t* dc = dmalloc<uint32_t>((size_t)N * K / 16);
      float* ds = dmalloc<float>((size_t)N * ngr);
      float* tmp = dmalloc<float>(16);
      int64_t* st = dmalloc<int64_t>(4);
      cudaMemset(st, 0, 32);
      t158Status s = t158_quantize_weights(dw, T158_F32, N, K, G, (t158QuantMode)m.mode, m.iters, dc, ds, tmp, st, 0, 0);
      CK(cudaDeviceSynchronize());
      auto codes = from_dev(dc, (size_t)N * K / 16);
      auto sc = from_dev(ds, (size_t)N * ngr);
      long long mism = 0, total = (long long)N * K;
      double scale_err = 0;
      float ga = 0, gt = 0;
      if (G == 0) host_quant_ref(w.data(), N * K, m.mode, m.iters, ga, gt);
      for (int n = 0; n < N; ++n)
        for (int g = 0; g < ngr; ++g) {
          float alpha, thr;
          if (G) host_quant_ref(&w[(size_t)n * K + (size_t)g * G], G, m.mode, m.iters, alpha, thr);
          else { alpha = ga; thr = gt; }
          if (G == 0 && (n || g)) { /* per-tensor: a single scale lives in sc[0] */ }
          else scale_err = fmax(scale_err, fabs(alpha - sc[n * ngr + g]) / (alpha + 1e-12));
          for (int i = 0; i < (G ? G : K); ++i) {
            int k = G ? g * G + i : i;
            float v = w[(size_t)n * K + k];
            int t = fabsf(v) > thr ? (v > 0 ? 1 : -1) : 0;
            uint32_t c = (codes[((size_t)n * K + k) / 16] >> (2 * (k % 16))) & 3u;
            if (dec_code(c) != t) ++mism;
          }
        }
      char nm[64];
      snprintf(nm, sizeof(nm), "quantize %s G=%d", m.nm, G);
      report(nm, s == T158_OK && scale_err < 1e-3 && mism <= total / 5000, "scale_err=%.2e mismatches=%.0f", scale_err, (double)mism);
      cudaFree(dc); cudaFree(ds); cudaFree(tmp); cudaFree(st);
    }
  }
  cudaFree(dw);
  return 0;
}

static int test_pack_dequant(Rng& rng) {
  const int N = 20, K = 256;
  std::vector<int8_t> t((size_t)N * K);
  for (auto& v : t) v = (int8_t)((int)(rng.next() % 3) - 1);
  int8_t* dt = to_dev(t);
  uint32_t* dc = dmalloc<uint32_t>((size_t)N * K / 16);
  int8_t* dt2 = dmalloc<int8_t>(t.size());
  t158_pack_i8(dt, N, K, dc, 0);
  t158_unpack_i8(dc, N, K, dt2, 0);
  CK(cudaDeviceSynchronize());
  auto t2 = from_dev(dt2, t.size());
  report("pack/unpack roundtrip", t2 == t);
  std::vector<float> sw(N);
  for (auto& v : sw) v = 0.01f + rng.uni();
  float* dsw = to_dev(sw);
  float* dout = dmalloc<float>(t.size());
  t158_dequantize(dc, dsw, N, K, K, dout, T158_F32, nullptr, T158_F32, 1.f, 0);
  CK(cudaDeviceSynchronize());
  auto o = from_dev(dout, t.size());
  double e = 0;
  for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) e = fmax(e, fabs(o[(size_t)n * K + k] - sw[n] * t[(size_t)n * K + k]));
  report("dequantize per-row", e < 1e-6, "max err %.2e", e);
  // stats
  int64_t* dst = dmalloc<int64_t>(3);
  cudaMemset(dst, 0, 24);
  t158_code_stats(dc, (int64_t)N * K / 16, dst, 0);
  auto st = from_dev(dst, 3);
  long long z = 0, p = 0, ng = 0;
  for (auto v : t) { if (v == 0) ++z; else if (v > 0) ++p; else ++ng; }
  report("code_stats", st[0] == z && st[1] == p && st[2] == ng);
  cudaFree(dt); cudaFree(dc); cudaFree(dt2); cudaFree(dsw); cudaFree(dout); cudaFree(dst);
  return 0;
}

static int test_activations(Rng& rng) {
  const int M = 37, K = 1024;
  std::vector<float> x((size_t)M * K), gm(K);
  for (auto& v : x) v = rng.normal() * (1.f + 3.f * (rng.uni() > 0.99f));
  for (auto& v : gm) v = 0.5f + rng.uni();
  float* dx = to_dev(x);
  float* dg = to_dev(gm);
  int8_t* dq = dmalloc<int8_t>(x.size());
  float* ds = dmalloc<float>(M);
  t158_act_quant(dx, T158_F32, M, K, dq, ds, 0);
  CK(cudaDeviceSynchronize());
  auto q = from_dev(dq, x.size());
  auto s = from_dev(ds, M);
  int bad = 0;
  double serr = 0;
  for (int m = 0; m < M; ++m) {
    float mx = 0;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, fabsf(x[(size_t)m * K + k]));
    serr = fmax(serr, fabs(s[m] - mx / 127.f) / (mx / 127.f + 1e-12));
    for (int k = 0; k < K; ++k) {
      int r = (int)nearbyintf(x[(size_t)m * K + k] * (127.f / mx));
      if (abs(r - q[(size_t)m * K + k]) > 1) ++bad;
    }
  }
  report("act_quant per token", bad == 0 && serr < 1e-5, "scale err %.2e bad=%.0f", serr, bad);

  // rmsnorm fused
  float *dy = dmalloc<float>(x.size()), *dr = dmalloc<float>(M);
  int8_t* dq2 = dmalloc<int8_t>(x.size());
  float* ds2 = dmalloc<float>(M);
  t158_rmsnorm_fwd(dx, dg, dy, dq2, ds2, dr, M, K, 1e-6f, T158_F32, 0);
  CK(cudaDeviceSynchronize());
  auto y = from_dev(dy, x.size());
  double e = 0;
  std::vector<float> rs(M);
  for (int m = 0; m < M; ++m) {
    double ss = 0;
    for (int k = 0; k < K; ++k) ss += (double)x[(size_t)m * K + k] * x[(size_t)m * K + k];
    rs[m] = (float)(1.0 / sqrt(ss / K + 1e-6));
    for (int k = 0; k < K; ++k) e = fmax(e, fabs(y[(size_t)m * K + k] - x[(size_t)m * K + k] * rs[m] * gm[k]));
  }
  report("rmsnorm fwd (+fused quant)", e < 1e-4, "max err %.2e", e);

  // rmsnorm bwd vs central finite differences on a few rows
  std::vector<float> dyv((size_t)M * K);
  for (auto& v : dyv) v = rng.normal();
  float* ddy = to_dev(dyv);
  float *ddx = dmalloc<float>(x.size()), *dgp = dmalloc<float>((size_t)256 * K);
  t158_rmsnorm_bwd(ddy, dx, dg, dr, ddx, dgp, M, K, 0, T158_F32, 0);
  CK(cudaDeviceSynchronize());
  auto dxh = from_dev(ddx, x.size());
  double be = 0;
  for (int trial = 0; trial < 6; ++trial) {
    int m = trial * 5 % M, k = (trial * 131) % K;
    auto f = [&](float delta) {
      double ss = 0;
      for (int j = 0; j < K; ++j) { double v = x[(size_t)m * K + j] + (j == k ? delta : 0.f); ss += v * v; }
      double r = 1.0 / sqrt(ss / K + 1e-6), tot = 0;
      for (int j = 0; j < K; ++j) { double v = x[(size_t)m * K + j] + (j == k ? delta : 0.f); tot += v * r * gm[j] * dyv[(size_t)m * K + j]; }
      return tot;
    };
    double h = 1e-2, fd = (f((float)h) - f((float)-h)) / (2 * h);
    be = fmax(be, fabs(fd - dxh[(size_t)m * K + k]) / (fabs(fd) + 1e-2));
  }
  report("rmsnorm bwd (finite diff)", be < 2e-2, "rel err %.2e", be);

  // hadamard: involution + energy
  float* dh = dmalloc<float>(x.size());
  float* dh2 = dmalloc<float>(x.size());
  t158_hadamard(dx, dh, M, K, 256, T158_F32, 0);
  t158_hadamard(dh, dh2, M, K, 256, T158_F32, 0);
  CK(cudaDeviceSynchronize());
  auto h2 = from_dev(dh2, x.size());
  double he = 0;
  for (size_t i = 0; i < x.size(); ++i) he = fmax(he, fabs(h2[i] - x[i]));
  report("hadamard involution", he < 1e-4, "max err %.2e", he);

  // swiglu
  std::vector<float> a(4096), b(4096), da(4096), db(4096), g(4096);
  for (size_t i = 0; i < a.size(); ++i) { a[i] = rng.normal(); b[i] = rng.normal(); g[i] = rng.normal(); }
  float *dA = to_dev(a), *dB = to_dev(b), *dG = to_dev(g), *dO = dmalloc<float>(4096), *dDA = dmalloc<float>(4096), *dDB = dmalloc<float>(4096);
  t158_swiglu_fwd(dA, dB, dO, 4096, T158_F32, 0);
  t158_swiglu_bwd(dG, dA, dB, dDA, dDB, 4096, T158_F32, 0);
  CK(cudaDeviceSynchronize());
  auto o = from_dev(dO, 4096); auto oa = from_dev(dDA, 4096); auto ob = from_dev(dDB, 4096);
  double se = 0;
  for (int i = 0; i < 4096; ++i) {
    double sg = 1.0 / (1.0 + exp(-(double)a[i])), silu = a[i] * sg;
    se = fmax(se, fabs(o[i] - silu * b[i]));
    se = fmax(se, fabs(oa[i] - g[i] * b[i] * sg * (1 + a[i] * (1 - sg))));
    se = fmax(se, fabs(ob[i] - g[i] * silu));
  }
  report("swiglu fwd/bwd", se < 1e-4, "max err %.2e", se);
  cudaFree(dx); cudaFree(dg); cudaFree(dq); cudaFree(ds); cudaFree(dy); cudaFree(dr); cudaFree(dq2); cudaFree(ds2);
  cudaFree(ddy); cudaFree(ddx); cudaFree(dgp); cudaFree(dh); cudaFree(dh2);
  cudaFree(dA); cudaFree(dB); cudaFree(dG); cudaFree(dO); cudaFree(dDA); cudaFree(dDB);
  return 0;
}

static int test_loss_opt(Rng& rng) {
  const int T = 9, V = 1000;
  std::vector<float> lg((size_t)T * V), tl((size_t)T * V);
  std::vector<int64_t> lab(T);
  for (auto& v : lg) v = 2.f * rng.normal();
  for (auto& v : tl) v = 2.f * rng.normal();
  for (int t = 0; t < T; ++t) lab[t] = (t == 3) ? -100 : (int64_t)(rng.next() % V);
  float* dl = to_dev(lg);
  float* dt = to_dev(tl);
  int64_t* dlab = to_dev(lab);
  float* dloss = dmalloc<float>(T);
  float* dl2 = to_dev(lg);
  t158_cross_entropy(dl, T158_F32, dlab, T, V, -100, 1.f, 1, dloss, 0);
  CK(cudaDeviceSynchronize());
  auto loss = from_dev(dloss, T);
  auto grad = from_dev(dl, lg.size());
  double e = 0, ge = 0;
  for (int t = 0; t < T; ++t) {
    double mx = -1e30, se = 0;
    for (int v = 0; v < V; ++v) mx = fmax(mx, lg[(size_t)t * V + v]);
    for (int v = 0; v < V; ++v) se += exp(lg[(size_t)t * V + v] - mx);
    double lse = mx + log(se);
    double ref = lab[t] < 0 ? 0 : lse - lg[(size_t)t * V + lab[t]];
    e = fmax(e, fabs(ref - loss[t]));
    for (int v = 0; v < V; ++v) {
      double p = lab[t] < 0 ? 0 : exp(lg[(size_t)t * V + v] - lse) - (v == lab[t]);
      ge = fmax(ge, fabs(p - grad[(size_t)t * V + v]));
    }
  }
  report("cross_entropy loss+grad", e < 1e-4 && ge < 1e-5, "loss err %.2e grad err %.2e", e, ge);
  // KL: loss >= 0 and zero when student == teacher
  float* dz = dmalloc<float>(T);
  t158_kl_div(dl2, dt, T158_F32, T, V, 2.f, 1.f, 0, dz, 0);
  CK(cudaDeviceSynchronize());
  auto kl = from_dev(dz, T);
  bool pos = true;
  for (auto v : kl) pos = pos && v >= -1e-6f;
  float* dsame = to_dev(tl);
  t158_kl_div(dsame, dt, T158_F32, T, V, 2.f, 1.f, 0, dz, 0);
  CK(cudaDeviceSynchronize());
  auto kz = from_dev(dz, T);
  double kmax = 0;
  for (auto v : kz) kmax = fmax(kmax, fabs(v));
  report("kl_div (>=0, zero if equal)", pos && kmax < 1e-5, "max self-kl %.2e", kmax);

  // adam
  const int n = 1000;
  std::vector<float> w(n), g(n), m0(n, 0.f), v0(n, 0.f);
  for (int i = 0; i < n; ++i) { w[i] = rng.normal(); g[i] = rng.normal(); }
  float *dw = to_dev(w), *dg = to_dev(g), *dm = to_dev(m0), *dv = to_dev(v0);
  t158AdamParams p; memset(&p, 0, sizeof(p));
  p.lr = 1e-2f; p.beta1 = 0.9f; p.beta2 = 0.98f; p.eps = 1e-8f; p.weight_decay = 0.1f; p.grad_scale = 0.5f; p.step = 1; p.seed = 7;
  t158_adam_update(dw, T158_F32, dg, T158_F32, dm, dv, T158_F32, n, &p, 0);
  CK(cudaDeviceSynchronize());
  auto w1 = from_dev(dw, n);
  double ae = 0;
  for (int i = 0; i < n; ++i) {
    double gi = g[i] * 0.5, m = 0.1 * gi, v = 0.02 * gi * gi;
    double mh = m / 0.1, vh = v / 0.02;
    double ref = w[i] * (1 - 1e-2 * 0.1) - 1e-2 * mh / (sqrt(vh) + 1e-8);
    ae = fmax(ae, fabs(ref - w1[i]));
  }
  report("adam step (fp32)", ae < 1e-5, "max err %.2e", ae);
  // bf16 master with stochastic rounding must be unbiased on average
  std::vector<uint16_t> wb(n);
  for (int i = 0; i < n; ++i) { float f = 0.02f; uint32_t u; memcpy(&u, &f, 4); wb[i] = (uint16_t)(u >> 16); }
  uint16_t* dwb = to_dev(wb);
  float* dg2 = to_dev(std::vector<float>(n, 1.f));
  __nv_bfloat16* dmb = dmalloc<__nv_bfloat16>(n);
  __nv_bfloat16* dvb = dmalloc<__nv_bfloat16>(n);
  cudaMemset(dmb, 0, n * 2); cudaMemset(dvb, 0, n * 2);
  p.lr = 1e-5f; p.weight_decay = 0.f; p.grad_scale = 1.f;
  double drift = 0;
  for (int step = 1; step <= 50; ++step) { p.step = step; t158_adam_update(dwb, T158_BF16, dg2, T158_F32, dmb, dvb, T158_BF16, n, &p, 0); }
  CK(cudaDeviceSynchronize());
  auto wb2 = from_dev(dwb, n);
  for (int i = 0; i < n; ++i) { uint32_t u = (uint32_t)wb2[i] << 16; float f; memcpy(&f, &u, 4); drift += f; }
  drift = drift / n;  // expected ~ 0.02 - 50*1e-5 = 0.0195 ; round-to-nearest would stay at 0.02
  report("adam bf16 master + SR", fabs(drift - 0.0195) < 1.5e-4, "mean=%.5f (expect 0.0195) %.0f", drift, 0);
  cudaFree(dl); cudaFree(dt); cudaFree(dlab); cudaFree(dloss); cudaFree(dl2); cudaFree(dz); cudaFree(dsame);
  cudaFree(dw); cudaFree(dg); cudaFree(dm); cudaFree(dv); cudaFree(dwb); cudaFree(dg2); cudaFree(dmb); cudaFree(dvb);
  return 0;
}

// ---------------------------------------------------------------------------------------------
static int bench(int dev, bool quick) {
  t158DeviceInfo di;
  t158_get_device_info(dev, &di);
  printf("\n== benchmark (%s) ==\n", di.name);
  struct S { int M, N, K, G; };
  std::vector<S> shapes = {{1, 4096, 4096, 128}, {16, 4096, 4096, 128}, {512, 4096, 4096, 128}, {4096, 4096, 4096, 128}, {4096, 11008, 4096, 128}, {4096, 4096, 4096, 0}};
  if (quick) shapes = {{16, 4096, 4096, 128}, {2048, 4096, 4096, 128}};
  int backs[] = {T158_GEMM_DP4A, T158_GEMM_WMMA, T158_GEMM_MMA, T158_GEMM_BF16};
  const char* names[] = {"auto", "dp4a", "wmma", "mma", "bf16"};
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (auto& s : shapes) {
    size_t gw = s.G ? (size_t)s.N * (s.K / s.G) : 1;
    int8_t* xq = dmalloc<int8_t>((size_t)s.M * s.K);
    cudaMemset(xq, 1, (size_t)s.M * s.K);
    float *sx = dmalloc<float>(s.M), *sw = dmalloc<float>(gw), *y = dmalloc<float>((size_t)s.M * s.N);
    uint32_t* codes = dmalloc<uint32_t>((size_t)s.N * s.K / 16);
    cudaMemset(codes, 0x55, (size_t)s.N * s.K / 4);
    std::vector<float> ones(std::max<size_t>(gw, s.M), 1.f);
    cudaMemcpy(sx, ones.data(), s.M * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sw, ones.data(), gw * 4, cudaMemcpyHostToDevice);
    for (int b : backs) {
      if (b == T158_GEMM_DP4A && s.M > 64) continue;  // reference path, not meaningful for big M
      if (b != T158_GEMM_DP4A && di.sm < 80) continue;
      size_t wsb = t158_gemm_workspace_bytes(s.M, s.N, s.K, b, dev);
      void* ws = wsb ? (void*)dmalloc<uint8_t>(wsb) : nullptr;
      for (int i = 0; i < 3; ++i) t158_gemm_fwd(xq, sx, codes, sw, y, T158_F32, s.M, s.N, s.K, s.G, b, ws, wsb, 0);
      if (cudaDeviceSynchronize() != cudaSuccess) { printf("  backend %s crashed\n", names[b]); cudaGetLastError(); break; }
      int it = s.M >= 512 ? 10 : 50;
      cudaEventRecord(e0);
      for (int i = 0; i < it; ++i) t158_gemm_fwd(xq, sx, codes, sw, y, T158_F32, s.M, s.N, s.K, s.G, b, ws, wsb, 0);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      ms /= it;
      double tops = 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12;
      double gbs = ((double)s.N * s.K / 4.0 + (double)s.M * s.K) / (ms * 1e-3) / 1e9;
      printf("  M=%-5d N=%-6d K=%-5d G=%-4d %-6s %8.3f ms  %7.2f TOPS  %7.1f GB/s(weights+acts)\n", s.M, s.N, s.K, s.G, names[b], ms, tops, gbs);
      if (ws) cudaFree(ws);
    }
    cudaFree(xq); cudaFree(sx); cudaFree(sw); cudaFree(y); cudaFree(codes);
  }
  // multi-GPU topology
  int nd = 0;
  cudaGetDeviceCount(&nd);
  if (nd > 1) {
    printf("\n== peer access matrix (%d GPUs) ==\n", nd);
    for (int a = 0; a < nd; ++a) {
      for (int b = 0; b < nd; ++b) {
        int can = 0;
        if (a != b) cudaDeviceCanAccessPeer(&can, a, b);
        printf(" %d", a == b ? 2 : can);
      }
      printf("\n");
    }
    printf("(2 = self, 1 = P2P capable, 0 = via host)\n");
  }
  return 0;
}

int main(int argc, char** argv) {
  int dev = 0;
  bool do_bench = false, quick = false;
  const char* json = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bench")) do_bench = true;
    else if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
  }
  int nd = 0;
  if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) { printf("no CUDA device found\n"); return 2; }
  CK(cudaSetDevice(dev));
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("tern158 selftest  lib v%d  CUDA runtime %d driver %d  devices=%d\n", t158_version(), rt, drv, nd);
  char buf[8192];
  int mask = t158_validate_backends(dev, buf, sizeof(buf));
  printf("%s", buf);
  report("gemm backends (any int8 path)", (mask & ((1 << T158_GEMM_DP4A) | (1 << T158_GEMM_WMMA) | (1 << T158_GEMM_MMA) | (1 << T158_GEMM_BF16))) != 0);
  report("bwd_data (dequant+cuBLAS)", (mask & (1 << 30)) == 0);
  Rng rng(2026);
  if (test_quant(rng)) return 1;
  if (test_pack_dequant(rng)) return 1;
  if (test_activations(rng)) return 1;
  if (test_loss_opt(rng)) return 1;
  if (do_bench) bench(dev, quick);
  printf("\nsummary: %d passed, %d failed\n", g_pass, g_fail);
  if (json) {
    FILE* f = fopen(json, "w");
    if (f) {
      fprintf(f, "{\"passed\":%d,\"failed\":%d,\"backend_mask\":%d,\"tests\":[", g_pass, g_fail, mask);
      for (size_t i = 0; i < g_json.size(); ++i) fprintf(f, "%s%s", i ? "," : "", g_json[i].c_str());
      fprintf(f, "]}\n");
      fclose(f);
    }
  }
  return g_fail ? 1 : 0;
}
