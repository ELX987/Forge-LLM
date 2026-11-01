// Minimal CUDA stub so g++ -fsyntax-only can type-check .cu sources on machines without the CUDA toolkit.
// NOT a real CUDA header.  Used only by tools/syntax_check.py.
#pragma once
#define CUDART_VERSION 12080
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
using std::min; using std::max; using std::isfinite;
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __align__(n) alignas(n)
#define __restrict__ __restrict
struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
extern uint3 threadIdx, blockIdx; extern dim3 blockDim, gridDim;
struct float4 { float x, y, z, w; }; struct uint4 { unsigned x, y, z, w; }; struct uint2 { unsigned x, y; }; struct int4 { int x, y, z, w; };
inline float4 make_float4(float a, float b, float c, float d) { return {a, b, c, d}; }
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return {a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return {a, b}; }
int __shfl_xor_sync(unsigned, int, int); float __shfl_xor_sync(unsigned, float, int);
int __popc(unsigned); unsigned __byte_perm(unsigned, unsigned, unsigned);
int __dp4a(int, int, int); int __float2int_rn(float); float __expf(float); float rsqrtf(float);
void __syncthreads(); void __syncwarp(); void __trap();
size_t __cvta_generic_to_shared(const void*);
unsigned __float_as_uint(float);
float atomicAdd(float*, float); double atomicAdd(double*, double); unsigned long long atomicAdd(unsigned long long*, unsigned long long);
int atomicOr(int*, int);
typedef int cudaError_t; enum { cudaSuccess = 0 };
typedef struct CUstream_st* cudaStream_t; typedef struct CUevent_st* cudaEvent_t;
const char* cudaGetErrorName(cudaError_t); const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError(); cudaError_t cudaDeviceSynchronize(); cudaError_t cudaGetDevice(int*); cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceCount(int*); cudaError_t cudaFree(void*); cudaError_t cudaMalloc(void**, size_t);
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind); cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaEventCreate(cudaEvent_t*); cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t); cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaRuntimeGetVersion(int*); cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaDeviceCanAccessPeer(int*, int, int);
struct cudaDeviceProp { char name[256]; int major, minor, multiProcessorCount; size_t sharedMemPerBlockOptin; int l2CacheSize, memoryBusWidth; size_t totalGlobalMem; int persistingL2CacheMaxSize, accessPolicyMaxWindowSize; };
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
enum cudaDeviceAttr { cudaDevAttrMemoryClockRate };
cudaError_t cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
enum cudaLimit { cudaLimitPersistingL2CacheSize };
cudaError_t cudaDeviceSetLimit(cudaLimit, size_t); cudaError_t cudaCtxResetPersistingL2Cache();
enum cudaAccessProperty { cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
struct cudaAccessPolicyWindow { void* base_ptr; size_t num_bytes; float hitRatio; cudaAccessProperty hitProp, missProp; };
union cudaStreamAttrValue { cudaAccessPolicyWindow accessPolicyWindow; };
enum cudaStreamAttrID { cudaStreamAttributeAccessPolicyWindow };
cudaError_t cudaStreamSetAttribute(cudaStream_t, cudaStreamAttrID, const cudaStreamAttrValue*);
