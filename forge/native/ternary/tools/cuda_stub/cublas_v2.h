#pragma once
#include "cuda_runtime.h"
typedef struct cublasContext* cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cudaDataType { CUDA_R_32F, CUDA_R_16BF, CUDA_R_16F };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT };
cublasStatus_t cublasCreate(cublasHandle_t*); cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
