#ifndef EGG_OPTIMIZER_BASE_H
#define EGG_OPTIMIZER_BASE_H

#include <stdint.h>
#include "../config.h"

// Adam State Structures
typedef struct {
    float m;
    float v;
    float acc; // Fractional accumulator for integer weights
} AdamParam;

#if USE_MUON == 1
typedef struct {
    float m;
    float acc;
} MuonParam;
#define HIDDEN_OPT_TYPE MuonParam
#else
#define HIDDEN_OPT_TYPE AdamParam
#endif

#endif // EGG_OPTIMIZER_BASE_H
