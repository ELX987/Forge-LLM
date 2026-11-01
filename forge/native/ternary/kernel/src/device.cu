// device.cu - device query, backend selection state, L2 persistence helpers.
#include <cstring>

#include "tern158/common.cuh"

namespace {
constexpr int kMaxDev = 32;
int g_backend[kMaxDev];  // -1 = unset (use default)
bool g_init = false;
void init_once() {
  if (g_init) return;
  for (int i = 0; i < kMaxDev; ++i) g_backend[i] = -1;
  g_init = true;
}
}  // namespace

extern "C" {

int t158_version(void) { return T158_VERSION_MAJOR * 10000 + T158_VERSION_MINOR * 100 + T158_VERSION_PATCH; }

const char* t158_status_string(t158Status s) {
  switch (s) {
    case T158_OK: return "ok";
    case T158_ERR_ARG: return "invalid argument or unsupported shape";
    case T158_ERR_ARCH: return "kernel unavailable on this GPU";
    case T158_ERR_CUDA: return "CUDA runtime error";
    case T158_ERR_WORKSPACE: return "workspace too small";
  }
  return "unknown";
}

t158Status t158_get_device_info(int device, t158DeviceInfo* out) {
  T158_REQUIRE(out != nullptr);
  cudaDeviceProp p;
  T158_CUDA_TRY(cudaGetDeviceProperties(&p, device));
  memset(out, 0, sizeof(*out));
  out->device = device;
  out->sm = p.major * 10 + p.minor;
  out->sm_count = p.multiProcessorCount;
  out->max_smem_optin = (int)p.sharedMemPerBlockOptin;
  out->l2_bytes = p.l2CacheSize;
  int v = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION < 13000
  cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, device);
#endif
  out->mem_clock_khz = v;
  out->mem_bus_bits = p.memoryBusWidth;
  out->total_mem = p.totalGlobalMem;
  // Report (ch.13): Blackwell Ultra (sm_103, B300/GB300) runs INT8 tensor cores at a tiny fraction of
  // the FP8 rate, so the int8 offset trick is not portable there. Everything else is full rate.
  out->int8_tc_fast = (out->sm == 103) ? 0 : 1;
  strncpy(out->name, p.name, sizeof(out->name) - 1);
  return T158_OK;
}

int t158_default_backend(int sm) {
  if (sm >= 103 && sm < 110) return T158_GEMM_BF16;  // B300 / GB300 : slow int8
  if (sm >= 80) return T158_GEMM_MMA;
  return T158_GEMM_DP4A;
}

void t158_set_backend(int device, int backend) {
  init_once();
  if (device >= 0 && device < kMaxDev) g_backend[device] = backend;
}

int t158_get_backend(int device) {
  init_once();
  if (device < 0 || device >= kMaxDev) return T158_GEMM_DP4A;
  if (g_backend[device] >= 0) return g_backend[device];
  t158DeviceInfo di;
  if (t158_get_device_info(device, &di) != T158_OK) return T158_GEMM_DP4A;
  return t158_default_backend(di.sm);
}

t158Status t158_set_l2_persist(cudaStream_t stream, const void* ptr, size_t bytes, float hit_ratio) {
  int dev = 0;
  T158_CUDA_TRY(cudaGetDevice(&dev));
  cudaDeviceProp p;
  T158_CUDA_TRY(cudaGetDeviceProperties(&p, dev));
  if (p.major < 8 || p.persistingL2CacheMaxSize <= 0) return T158_OK;  // no-op where unsupported
  size_t lim = (size_t)p.persistingL2CacheMaxSize;
  T158_CUDA_TRY(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lim));
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
  size_t win = bytes < (size_t)p.accessPolicyMaxWindowSize ? bytes : (size_t)p.accessPolicyMaxWindowSize;
  attr.accessPolicyWindow.num_bytes = win;
  float hr = hit_ratio;
  if (win > lim) hr = hit_ratio * (float)lim / (float)win;  // do not thrash the persisting region
  attr.accessPolicyWindow.hitRatio = hr;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  T158_CUDA_TRY(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
  return T158_OK;
}

t158Status t158_clear_l2_persist(cudaStream_t stream) {
  int dev = 0;
  T158_CUDA_TRY(cudaGetDevice(&dev));
  cudaDeviceProp p;
  T158_CUDA_TRY(cudaGetDeviceProperties(&p, dev));
  if (p.major < 8) return T158_OK;
  cudaStreamAttrValue attr;
  memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.num_bytes = 0;
  T158_CUDA_TRY(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
  T158_CUDA_TRY(cudaCtxResetPersistingL2Cache());
  return T158_OK;
}

}  // extern "C"
