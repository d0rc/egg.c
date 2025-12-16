#ifndef EGG_OPTIMIZER_SGD_CUH
#define EGG_OPTIMIZER_SGD_CUH

#include <cuda_runtime.h>
#include "../config.h"
#include "../model/definitions.h"

// Quantized Update Logic (Thresholding)
// Applies accumulated float update to integer weight
__device__ __forceinline__ void apply_quantized_update(WeightType &w, float &acc, int &change) {
    if (acc >= 1.0f) {
        if (w < MAX_VAL) { w++; change=1; acc -= 1.0f; }
        else acc = 1.0f; // Clamp accumulator
    } else if (acc <= -1.0f) {
        if (w > MIN_VAL) { w--; change=1; acc += 1.0f; }
        else acc = -1.0f; // Clamp accumulator
    }
}

#endif // EGG_OPTIMIZER_SGD_CUH
