#pragma once
#include "cuda_runtime.h"
struct __nv_bfloat16 { unsigned short x; };
float __bfloat162float(__nv_bfloat16); __nv_bfloat16 __float2bfloat16(float); __nv_bfloat16 __ushort_as_bfloat16(unsigned short);
