#ifndef EGG_GLOBALS_CUH
#define EGG_GLOBALS_CUH

#include "config.h"
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ int32_t d_EXP_LUT[SOFTMAX_LUT_SIZE];
__constant__ int8_t d_ACT_LUT[256];
__device__ int32_t d_ROPE_LUT[ROPE_LUT_SIZE];
__device__ unsigned long long d_total_updates;

#endif // EGG_GLOBALS_CUH
