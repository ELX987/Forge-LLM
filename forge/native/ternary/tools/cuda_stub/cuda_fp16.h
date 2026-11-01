#pragma once
#include "cuda_runtime.h"
struct __half { unsigned short x; };
float __half2float(__half); __half __float2half(float);
