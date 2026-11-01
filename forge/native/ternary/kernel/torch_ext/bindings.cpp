// bindings.cpp - PyTorch extension module `forge_ternary_C` over libtern158.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <string>
#include <vector>

#include "tern158/tern158.h"

namespace py = pybind11;

#define T_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define T_CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define T_CHECK_INPUT(x) \
  T_CHECK_CUDA(x);       \
  T_CHECK_CONTIG(x)
#define T_ALIGN16(x)   TORCH_CHECK(reinterpret_cast<uintptr_t>((x).data_ptr()) % 16 == 0, #x " must be 16-byte aligned (avoid odd slices)")
#define T_OK(call)                                                                                   \
  do {                                                                                               \
    t158Status _s = (call);                                                                          \
    TORCH_CHECK(_s == T158_OK, "tern158: ", #call, " failed: ", t158_status_string(_s));             \
  } while (0)

namespace {

cudaStream_t cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }

t158DType dt_of(const torch::Tensor& t) {
  switch (t.scalar_type()) {
    case torch::kFloat: return T158_F32;
    case torch::kBFloat16: return T158_BF16;
    case torch::kHalf: return T158_F16;
    default: TORCH_CHECK(false, "tern158: unsupported dtype ", t.scalar_type());
  }
  return T158_F32;
}
torch::ScalarType st_of(t158DType d) {
  return d == T158_F32 ? torch::kFloat : (d == T158_BF16 ? torch::kBFloat16 : torch::kHalf);
}
t158DType dt_from_torch(torch::ScalarType s) {
  if (s == torch::kFloat) return T158_F32;
  if (s == torch::kBFloat16) return T158_BF16;
  if (s == torch::kHalf) return T158_F16;
  TORCH_CHECK(false, "tern158: unsupported dtype");
  return T158_F32;
}

py::dict device_info(int dev) {
  t158DeviceInfo di;
  T_OK(t158_get_device_info(dev, &di));
  py::dict d;
  d["name"] = std::string(di.name);
  d["sm"] = di.sm;
  d["sm_count"] = di.sm_count;
  d["max_smem_optin"] = di.max_smem_optin;
  d["l2_bytes"] = di.l2_bytes;
  d["total_mem"] = (long long)di.total_mem;
  d["mem_clock_khz"] = di.mem_clock_khz;
  d["mem_bus_bits"] = di.mem_bus_bits;
  d["int8_tc_fast"] = di.int8_tc_fast;
  return d;
}

py::tuple validate(int dev) {
  std::vector<char> buf(16384, 0);
  int mask;
  {
    c10::cuda::CUDAGuard g(dev);
    mask = t158_validate_backends(dev, buf.data(), (int)buf.size());
  }
  return py::make_tuple(mask, std::string(buf.data()));
}

void quantize_weights_(const torch::Tensor& w, torch::Tensor codes, torch::Tensor scales,
                       c10::optional<torch::Tensor> tmp_o, c10::optional<torch::Tensor> stats_o, int group, int mode,
                       int iters, bool count_flips) {
  torch::Tensor tmp = tmp_o.has_value() ? *tmp_o : torch::Tensor();
  torch::Tensor stats = stats_o.has_value() ? *stats_o : torch::Tensor();
  T_CHECK_INPUT(w);
  T_CHECK_INPUT(codes);
  T_CHECK_INPUT(scales);
  T_ALIGN16(w);
  T_ALIGN16(codes);
  TORCH_CHECK(group != 0 || (tmp.defined() && tmp.numel() >= 16 && tmp.scalar_type() == torch::kFloat),
              "per-tensor quantization needs a float32 tmp tensor with >= 16 elements");
  TORCH_CHECK(!stats.defined() || stats.numel() == 0 || (stats.numel() >= 3 && stats.scalar_type() == torch::kInt64),
              "stats must be int64 with >= 3 elements");
  TORCH_CHECK(w.dim() == 2, "weight must be 2-D [N,K]");
  c10::cuda::CUDAGuard g(w.device());
  const int N = (int)w.size(0), K = (int)w.size(1);
  TORCH_CHECK(codes.scalar_type() == torch::kInt && codes.numel() == (int64_t)N * K / 16, "codes: int32 [N*K/16]");
  const int ngr = group == 0 ? 1 : K / group;
  TORCH_CHECK(scales.scalar_type() == torch::kFloat && scales.numel() == (group == 0 ? (int64_t)1 : (int64_t)N * ngr), "scales: float32 [N*K/group] (or 1 element for per-tensor)");
  float* tp = nullptr;
  if (tmp.defined() && tmp.numel() > 0) { T_CHECK_INPUT(tmp); tp = tmp.data_ptr<float>(); }
  int64_t* sp = nullptr;
  if (stats.defined() && stats.numel() > 0) { T_CHECK_INPUT(stats); sp = stats.data_ptr<int64_t>(); }
  T_OK(t158_quantize_weights(w.data_ptr(), dt_of(w), N, K, group, (t158QuantMode)mode, iters,
                             reinterpret_cast<uint32_t*>(codes.data_ptr<int>()), scales.data_ptr<float>(), tp, sp,
                             count_flips ? 1 : 0, cur_stream()));
}

torch::Tensor dequantize(const torch::Tensor& codes, const torch::Tensor& scales, int N, int K, int group,
                         torch::ScalarType out_dtype) {
  T_CHECK_INPUT(codes);
  T_CHECK_INPUT(scales);
  c10::cuda::CUDAGuard g(codes.device());
  auto out = torch::empty({N, K}, codes.options().dtype(out_dtype));
  T_OK(t158_dequantize(reinterpret_cast<const uint32_t*>(codes.data_ptr<int>()), scales.data_ptr<float>(), N, K,
                       group, out.data_ptr(), dt_from_torch(out_dtype), nullptr, T158_F32, 1.f, cur_stream()));
  return out;
}

// out = ref + lam*(dequant - ref)   (HF-style warm-up blend). ref is the fp32 master [N,K].
torch::Tensor dequantize_blend(const torch::Tensor& codes, const torch::Tensor& scales, const torch::Tensor& ref,
                               int group, double lam, torch::ScalarType out_dtype) {
  T_CHECK_INPUT(codes);
  T_CHECK_INPUT(scales);
  T_CHECK_INPUT(ref);
  c10::cuda::CUDAGuard g(codes.device());
  const int N = (int)ref.size(0), K = (int)ref.size(1);
  auto out = torch::empty({N, K}, codes.options().dtype(out_dtype));
  T_OK(t158_dequantize(reinterpret_cast<const uint32_t*>(codes.data_ptr<int>()), scales.data_ptr<float>(), N, K,
                       group, out.data_ptr(), dt_from_torch(out_dtype), ref.data_ptr(), dt_of(ref), (float)lam,
                       cur_stream()));
  return out;
}

std::vector<torch::Tensor> act_quant(const torch::Tensor& x) {
  T_CHECK_INPUT(x);
  T_ALIGN16(x);
  c10::cuda::CUDAGuard g(x.device());
  const int K = (int)x.size(-1);
  const int M = (int)(x.numel() / K);
  auto q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({M}, x.options().dtype(torch::kFloat));
  T_OK(t158_act_quant(x.data_ptr(), dt_of(x), M, K, q.data_ptr<int8_t>(), s.data_ptr<float>(), cur_stream()));
  return {q, s};
}

torch::Tensor act_dequant(const torch::Tensor& q, const torch::Tensor& s, torch::ScalarType odt) {
  T_CHECK_INPUT(q);
  T_CHECK_INPUT(s);
  c10::cuda::CUDAGuard g(q.device());
  const int M = (int)q.size(0), K = (int)q.size(1);
  auto out = torch::empty({M, K}, q.options().dtype(odt));
  T_OK(t158_act_dequant(q.data_ptr<int8_t>(), s.data_ptr<float>(), M, K, out.data_ptr(), dt_from_torch(odt), cur_stream()));
  return out;
}

// returns (y or empty, q or empty, s or empty, rstd)
std::vector<torch::Tensor> rmsnorm_fwd(const torch::Tensor& x, const torch::Tensor& gamma, double eps, bool want_y,
                                       bool want_q) {
  T_CHECK_INPUT(x);
  T_CHECK_INPUT(gamma);
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat, "gamma must be float32");
  c10::cuda::CUDAGuard g(x.device());
  const int K = (int)x.size(-1);
  const int M = (int)(x.numel() / K);
  torch::Tensor y, q, s;
  auto rstd = torch::empty({M}, x.options().dtype(torch::kFloat));
  if (want_y) y = torch::empty_like(x);
  if (want_q) {
    q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    s = torch::empty({M}, x.options().dtype(torch::kFloat));
  }
  T_OK(t158_rmsnorm_fwd(x.data_ptr(), gamma.data_ptr<float>(), want_y ? y.data_ptr() : nullptr,
                        want_q ? q.data_ptr<int8_t>() : nullptr, want_q ? s.data_ptr<float>() : nullptr,
                        rstd.data_ptr<float>(), M, K, (float)eps, dt_of(x), cur_stream()));
  return {y, q, s, rstd};
}

std::vector<torch::Tensor> rmsnorm_bwd(const torch::Tensor& dy, const torch::Tensor& x, const torch::Tensor& gamma,
                                       const torch::Tensor& rstd) {
  T_CHECK_INPUT(dy);
  T_CHECK_INPUT(x);
  T_CHECK_INPUT(gamma);
  T_CHECK_INPUT(rstd);
  TORCH_CHECK(dy.scalar_type() == x.scalar_type() && dy.sizes() == x.sizes(), "rmsnorm_bwd: dy/x mismatch");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat && rstd.scalar_type() == torch::kFloat, "gamma/rstd must be float32");
  c10::cuda::CUDAGuard g(x.device());
  const int K = (int)x.size(-1);
  const int M = (int)(x.numel() / K);
  TORCH_CHECK(gamma.numel() == K && rstd.numel() == M, "rmsnorm_bwd: size mismatch");
  const int nb = M < 256 ? M : 256;
  auto dx = torch::empty_like(x);
  auto part = torch::empty({nb, K}, x.options().dtype(torch::kFloat));
  T_OK(t158_rmsnorm_bwd(dy.data_ptr(), x.data_ptr(), gamma.data_ptr<float>(), rstd.data_ptr<float>(), dx.data_ptr(),
                        part.data_ptr<float>(), M, K, 0, dt_of(x), cur_stream()));
  return {dx, part.sum(0)};
}

torch::Tensor hadamard(const torch::Tensor& x, int block) {
  T_CHECK_INPUT(x);
  c10::cuda::CUDAGuard g(x.device());
  const int K = (int)x.size(-1);
  const int M = (int)(x.numel() / K);
  auto out = torch::empty_like(x);
  T_OK(t158_hadamard(x.data_ptr(), out.data_ptr(), M, K, block, dt_of(x), cur_stream()));
  return out;
}

torch::Tensor swiglu_fwd(const torch::Tensor& a, const torch::Tensor& b) {
  T_CHECK_INPUT(a);
  T_CHECK_INPUT(b);
  c10::cuda::CUDAGuard g(a.device());
  auto out = torch::empty_like(a);
  T_OK(t158_swiglu_fwd(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(), dt_of(a), cur_stream()));
  return out;
}
std::vector<torch::Tensor> swiglu_bwd(const torch::Tensor& dout, const torch::Tensor& a, const torch::Tensor& b) {
  T_CHECK_INPUT(dout);
  T_CHECK_INPUT(a);
  T_CHECK_INPUT(b);
  c10::cuda::CUDAGuard g(a.device());
  auto da = torch::empty_like(a), db = torch::empty_like(b);
  T_OK(t158_swiglu_bwd(dout.data_ptr(), a.data_ptr(), b.data_ptr(), da.data_ptr(), db.data_ptr(), a.numel(), dt_of(a),
                       cur_stream()));
  return {da, db};
}

// in place; x: [T, H, D]
void rope_(torch::Tensor x, const torch::Tensor& cos_t, const torch::Tensor& sin_t, int seqlen, bool inverse) {
  T_CHECK_INPUT(x);
  T_CHECK_INPUT(cos_t);
  T_CHECK_INPUT(sin_t);
  c10::cuda::CUDAGuard g(x.device());
  TORCH_CHECK(x.dim() == 3, "rope: x must be [T,H,D]");
  T_OK(t158_rope(x.data_ptr(), (int)x.size(0), (int)x.size(1), (int)x.size(2), seqlen, cos_t.data_ptr<float>(),
                 sin_t.data_ptr<float>(), inverse ? 1 : 0, dt_of(x), cur_stream()));
}

torch::Tensor gemm_fwd(const torch::Tensor& xq, const torch::Tensor& sx, const torch::Tensor& codes,
                       const torch::Tensor& sw, int N, int K, int group, torch::ScalarType out_dtype, int backend) {
  T_CHECK_INPUT(xq);
  T_CHECK_INPUT(sx);
  T_CHECK_INPUT(codes);
  T_CHECK_INPUT(sw);
  c10::cuda::CUDAGuard g(xq.device());
  const int M = (int)xq.size(0);
  TORCH_CHECK(xq.dim() == 2 && xq.size(1) == K && xq.scalar_type() == torch::kInt8, "gemm_fwd: bad xq");
  TORCH_CHECK(sx.numel() == M && sx.scalar_type() == torch::kFloat, "gemm_fwd: bad sx");
  TORCH_CHECK(codes.scalar_type() == torch::kInt && codes.numel() >= (int64_t)N * K / 16, "gemm_fwd: bad codes");
  TORCH_CHECK(sw.scalar_type() == torch::kFloat && sw.numel() >= (group == 0 ? 1 : (int64_t)N * (K / group)), "gemm_fwd: bad sw");
  T_ALIGN16(xq);
  T_ALIGN16(codes);
  auto y = torch::empty({M, N}, xq.options().dtype(out_dtype));
  int be = backend;
  if (be == T158_GEMM_AUTO) be = t158_get_backend(xq.get_device());
  torch::Tensor ws;
  size_t wsb = t158_gemm_workspace_bytes(M, N, K, be, xq.get_device());
  void* wp = nullptr;
  if (wsb) {
    ws = torch::empty({(int64_t)wsb}, xq.options().dtype(torch::kUInt8));
    wp = ws.data_ptr();
  }
  T_OK(t158_gemm_fwd(xq.data_ptr<int8_t>(), sx.data_ptr<float>(), reinterpret_cast<const uint32_t*>(codes.data_ptr<int>()),
                     sw.data_ptr<float>(), y.data_ptr(), dt_from_torch(out_dtype), M, N, K, group, be, wp, wsb, cur_stream()));
  return y;
}

torch::Tensor gemm_bwd_data(const torch::Tensor& dy, const torch::Tensor& codes, const torch::Tensor& sw, int N, int K,
                            int group, torch::ScalarType out_dtype) {
  T_CHECK_INPUT(dy);
  T_CHECK_INPUT(codes);
  T_CHECK_INPUT(sw);
  c10::cuda::CUDAGuard g(dy.device());
  const int M = (int)dy.size(0);
  TORCH_CHECK(dy.dim() == 2 && dy.size(1) == N, "gemm_bwd_data: dy must be [M,N]");
  TORCH_CHECK(dy.scalar_type() == torch::kBFloat16 || dy.scalar_type() == torch::kHalf, "dy must be bf16/fp16");
  TORCH_CHECK(out_dtype == dy.scalar_type() || out_dtype == torch::kFloat, "dx dtype must equal dy dtype or float32");
  TORCH_CHECK(codes.numel() >= (int64_t)N * K / 16, "gemm_bwd_data: bad codes");
  T_ALIGN16(dy);
  T_ALIGN16(codes);
  auto dx = torch::empty({M, K}, dy.options().dtype(out_dtype));
  auto ws = torch::empty({(int64_t)t158_bwd_data_workspace_bytes(N, K)}, dy.options().dtype(torch::kUInt8));
  T_OK(t158_gemm_bwd_data(dy.data_ptr(), dt_of(dy), reinterpret_cast<const uint32_t*>(codes.data_ptr<int>()),
                          sw.data_ptr<float>(), dx.data_ptr(), dt_from_torch(out_dtype), M, N, K, group, 0, ws.data_ptr(),
                          (size_t)ws.numel(), cur_stream()));
  return dx;
}

void adam_update_(torch::Tensor w, const torch::Tensor& g, torch::Tensor m, torch::Tensor v, double lr, double beta1,
                  double beta2, double eps, double wd, double grad_scale, int64_t step, int64_t seed) {
  T_CHECK_INPUT(w);
  T_CHECK_INPUT(g);
  T_CHECK_INPUT(m);
  T_CHECK_INPUT(v);
  TORCH_CHECK(w.numel() == g.numel() && w.numel() == m.numel() && w.numel() == v.numel(), "adam: size mismatch");
  TORCH_CHECK(m.scalar_type() == v.scalar_type(), "adam: m and v must share a dtype");
  c10::cuda::CUDAGuard gd(w.device());
  t158AdamParams p;
  p.lr = (float)lr; p.beta1 = (float)beta1; p.beta2 = (float)beta2; p.eps = (float)eps;
  p.weight_decay = (float)wd; p.grad_scale = (float)grad_scale; p.step = step; p.seed = (uint64_t)seed;
  T_OK(t158_adam_update(w.data_ptr(), dt_of(w), g.data_ptr(), dt_of(g), m.data_ptr(), v.data_ptr(), dt_of(m), w.numel(),
                        &p, cur_stream()));
}

// sum of squares over a list of tensors -> float32 scalar (device)
torch::Tensor sumsq(const std::vector<torch::Tensor>& ts) {
  TORCH_CHECK(!ts.empty(), "sumsq: empty list");
  c10::cuda::CUDAGuard g(ts[0].device());
  auto acc = torch::zeros({1}, ts[0].options().dtype(torch::kFloat));
  for (const auto& t : ts) {
    if (t.numel() == 0) continue;
    T_CHECK_INPUT(t);
    T_OK(t158_sumsq(t.data_ptr(), dt_of(t), t.numel(), acc.data_ptr<float>(), cur_stream()));
  }
  return acc;
}

torch::Tensor finite_check(const std::vector<torch::Tensor>& ts) {
  TORCH_CHECK(!ts.empty(), "finite_check: empty list");
  c10::cuda::CUDAGuard g(ts[0].device());
  auto flag = torch::zeros({1}, ts[0].options().dtype(torch::kInt));
  for (const auto& t : ts) {
    if (t.numel() == 0) continue;
    T_CHECK_INPUT(t);
    T_OK(t158_finite_check(t.data_ptr(), dt_of(t), t.numel(), flag.data_ptr<int>(), cur_stream()));
  }
  return flag;
}

torch::Tensor code_stats(const torch::Tensor& codes) {
  T_CHECK_INPUT(codes);
  c10::cuda::CUDAGuard g(codes.device());
  auto st = torch::zeros({3}, codes.options().dtype(torch::kInt64));
  T_OK(t158_code_stats(reinterpret_cast<const uint32_t*>(codes.data_ptr<int>()), codes.numel(), st.data_ptr<int64_t>(), cur_stream()));
  return st;  // zeros, pos, neg
}

// logits [T,V] modified in place to the gradient when compute_grad; returns per-token loss [T]
torch::Tensor cross_entropy_(torch::Tensor logits, const torch::Tensor& labels, int ignore_index, double grad_scale,
                             bool compute_grad) {
  T_CHECK_INPUT(logits);
  T_CHECK_INPUT(labels);
  TORCH_CHECK(labels.scalar_type() == torch::kLong, "labels must be int64");
  TORCH_CHECK(logits.dim() == 2 && labels.numel() >= logits.size(0), "cross_entropy: labels shorter than logits rows");
  c10::cuda::CUDAGuard g(logits.device());
  const int T = (int)logits.size(0), V = (int)logits.size(1);
  auto loss = torch::empty({T}, logits.options().dtype(torch::kFloat));
  T_OK(t158_cross_entropy(logits.data_ptr(), dt_of(logits), labels.data_ptr<int64_t>(), T, V, ignore_index,
                          (float)grad_scale, compute_grad ? 1 : 0, loss.data_ptr<float>(), cur_stream()));
  return loss;
}

torch::Tensor kl_div_(torch::Tensor student, const torch::Tensor& teacher, double temperature, double grad_scale,
                      bool compute_grad) {
  T_CHECK_INPUT(student);
  T_CHECK_INPUT(teacher);
  TORCH_CHECK(student.scalar_type() == teacher.scalar_type() && student.sizes() == teacher.sizes(), "kl: mismatch");
  c10::cuda::CUDAGuard g(student.device());
  const int T = (int)student.size(0), V = (int)student.size(1);
  auto loss = torch::empty({T}, student.options().dtype(torch::kFloat));
  T_OK(t158_kl_div(student.data_ptr(), teacher.data_ptr(), dt_of(student), T, V, (float)temperature, (float)grad_scale,
                   compute_grad ? 1 : 0, loss.data_ptr<float>(), cur_stream()));
  return loss;
}

void l2_persist(const torch::Tensor& t, double hit_ratio) {
  T_CHECK_INPUT(t);
  T_OK(t158_set_l2_persist(cur_stream(), t.data_ptr(), (size_t)t.nbytes(), (float)hit_ratio));
}
void l2_clear() { T_OK(t158_clear_l2_persist(cur_stream())); }

}  // namespace

PYBIND11_MODULE(forge_ternary_C, m) {
  m.doc() = "tern158 CUDA kernels (ternary 1.58-bit training/conversion/fine-tuning)";
  m.attr("VERSION") = t158_version();
  m.def("device_info", &device_info);
  m.def("validate", &validate, "run backend self-validation on a device -> (mask, log)");
  m.def("set_backend", [](int dev, int b) { t158_set_backend(dev, b); });
  m.def("get_backend", [](int dev) { return t158_get_backend(dev); });
  m.def("quantize_weights_", &quantize_weights_);
  m.def("dequantize", &dequantize);
  m.def("dequantize_blend", &dequantize_blend);
  m.def("act_quant", &act_quant);
  m.def("act_dequant", &act_dequant);
  m.def("rmsnorm_fwd", &rmsnorm_fwd);
  m.def("rmsnorm_bwd", &rmsnorm_bwd);
  m.def("hadamard", &hadamard);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("rope_", &rope_);
  m.def("gemm_fwd", &gemm_fwd);
  m.def("gemm_bwd_data", &gemm_bwd_data);
  m.def("adam_update_", &adam_update_);
  m.def("sumsq", &sumsq);
  m.def("finite_check", &finite_check);
  m.def("code_stats", &code_stats);
  m.def("cross_entropy_", &cross_entropy_);
  m.def("kl_div_", &kl_div_);
  m.def("l2_persist", &l2_persist);
  m.def("l2_clear", &l2_clear);
}
