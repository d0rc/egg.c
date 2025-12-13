#ifndef EGG_MODEL_DEFINITIONS_H
#define EGG_MODEL_DEFINITIONS_H

#include <stdint.h>
#include "../config.h"
#include "../optimizer/base.h"

// --- TYPE ALIASES ---
#ifdef USE_TOKENIZER
using TokenType = uint32_t;
#else
using TokenType = uint8_t;
#endif

using WeightType    = int8_t;
using ActType       = int8_t;
using AccumType     = int32_t;   // 32-bit sufficient for ~6M max sum
using AttnAccumType = long long; // Need high precision for attention weighted sum
using VoteType      = int32_t;

typedef struct {
    WeightType embedding[VOCAB_SIZE * HIDDEN_DIM];
    WeightType emb_bias[HIDDEN_DIM];
    
#if NTT_MODE != 0
    // NTT coefficient embedding tables (for int32 decomposition: bytes 1, 2, 3)
    WeightType ntt_emb1[VOCAB_SIZE * HIDDEN_DIM];  // Byte 1 of NTT coefficient
    WeightType ntt_emb2[VOCAB_SIZE * HIDDEN_DIM];  // Byte 2 of NTT coefficient
    WeightType ntt_emb3[VOCAB_SIZE * HIDDEN_DIM];  // Byte 3 (sign byte) of NTT coefficient
#endif
    
    // Initial MLP Layer
    WeightType ln_init[HIDDEN_DIM];
    WeightType ln_init_bias[HIDDEN_DIM];
    WeightType w_emb_mlp_up[HIDDEN_DIM * (HIDDEN_DIM * 4)];
    WeightType mlp_emb_bias_up[HIDDEN_DIM * 4];
    WeightType w_emb_mlp_down[(HIDDEN_DIM * 4) * HIDDEN_DIM];
    WeightType mlp_emb_bias_down[HIDDEN_DIM];

    WeightType ln_1[N_LAYERS][HIDDEN_DIM];
    WeightType ln_1_bias[N_LAYERS][HIDDEN_DIM];
    WeightType w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType ln_2[N_LAYERS][HIDDEN_DIM];
    WeightType ln_2_bias[N_LAYERS][HIDDEN_DIM];
    WeightType w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    WeightType mlp_bias_up[N_LAYERS][HIDDEN_DIM * 4];
    WeightType w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    WeightType mlp_bias_down[N_LAYERS][HIDDEN_DIM];
    WeightType ln_f[HIDDEN_DIM];
    WeightType ln_f_bias[HIDDEN_DIM];
} TransformerModel;

typedef struct {
    AdamParam embedding[VOCAB_SIZE * HIDDEN_DIM];
    AdamParam emb_bias[HIDDEN_DIM];
    
#if NTT_MODE != 0
    // NTT coefficient embedding optimizer states
    AdamParam ntt_emb1[VOCAB_SIZE * HIDDEN_DIM];
    AdamParam ntt_emb2[VOCAB_SIZE * HIDDEN_DIM];
    AdamParam ntt_emb3[VOCAB_SIZE * HIDDEN_DIM];
#endif
    
    // Initial MLP Layer
    AdamParam ln_init[HIDDEN_DIM];
    AdamParam ln_init_bias[HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_emb_mlp_up[HIDDEN_DIM * (HIDDEN_DIM * 4)];
    AdamParam mlp_emb_bias_up[HIDDEN_DIM * 4];
    HIDDEN_OPT_TYPE w_emb_mlp_down[(HIDDEN_DIM * 4) * HIDDEN_DIM];
    AdamParam mlp_emb_bias_down[HIDDEN_DIM];

    AdamParam ln_1[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_1_bias[N_LAYERS][HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam ln_2[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_2_bias[N_LAYERS][HIDDEN_DIM];
    HIDDEN_OPT_TYPE w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    AdamParam mlp_bias_up[N_LAYERS][HIDDEN_DIM * 4];
    HIDDEN_OPT_TYPE w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    AdamParam mlp_bias_down[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_f[HIDDEN_DIM];
    AdamParam ln_f_bias[HIDDEN_DIM];
} AdamModel;

#endif // EGG_MODEL_DEFINITIONS_H
