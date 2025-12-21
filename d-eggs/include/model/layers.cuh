#ifndef EGG_MODEL_LAYERS_CUH
#define EGG_MODEL_LAYERS_CUH

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../config.h"
#include "../math/adaptive_norm.cuh"
#include "../utils/egg_math.h"
#include "definitions.h"
#include "../globals.cuh"

__device__ __forceinline__ float get_adaptive_factor(WeightType ov, uint32_t seed, int idx) {
#if ADAPTIVE_NOISE_MODE == 0
    return 1.0f;
#elif ADAPTIVE_NOISE_MODE == 1
    if (ov < 0) return 0.0f;
    if (ov < 64) return (float)ov / 64.0f;
    return 1.0f;
#elif ADAPTIVE_NOISE_MODE == 2
    //if (ov < 0) return 0.0f;
    //float p = (ov < 64) ? ((float)ov / 64.0f) : 1.0f;
    //if (p == 0.0f) return 0.0f;
    /*
    if (ov < -96) {
        return 0.0f;
    }
    if (ov > 96) {
        return 1.0f;
    }
    float p = ((float)ov + 127.0f) / 255.0f;

    uint32_t r = hash_rng(seed + SEED_OFF_GATE, idx);
    float r_norm = (float)r / 4294967295.0f;
    
    return (r_norm < p) ? 1.0f : 0.0f;
    */
    // Heavy-tailed noise (Pareto-like with alpha=2)

    // Scale derived from ov
    float scale = ((float)ov+127.0f) / 255.0f;

    uint32_t r = hash_rng(seed + SEED_OFF_GATE, idx);
    // Ensure u is in (0, 1]
    float u = ((float)r + 1.0f) / 4294967296.0f;

    // Pareto distribution: f(x) = 2/x^3, x >= 1. Mean = 2.
    // Multiplier = 1 / sqrt(u)
    // We normalize by 0.5 to bring mean of the stochastic part to 1.0
    return scale / sqrtf(u);
#else
    return 1.0f;
#endif
}

// Helper: Sum reduction + Broadcast
typedef cub::BlockReduce<AccumType, EGG_BLOCK_THREADS> BlockReduce;

__device__ __forceinline__ AccumType block_reduce_sum_broadcast(AccumType val, BlockReduce::TempStorage &storage, AccumType &shared_var) {
    AccumType total = BlockReduce(storage).Sum(val);
    if (threadIdx.x == 0) shared_var = total;
    __syncthreads();
    AccumType ret = shared_var;
    __syncthreads();
    return ret;
}

__device__ __forceinline__ int simd_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

__device__ __forceinline__ int32_t softmax_exp_lookup(int32_t diff) {
    int index = -diff;  // diff is negative or zero, so index is positive
    index = (index < 0) ? 0 : ((index > (SOFTMAX_LUT_SIZE - 1)) ? (SOFTMAX_LUT_SIZE - 1) : index);
    return d_EXP_LUT[index];
}

__device__ __forceinline__ AccumType compute_linear_projection(
    const int32_t * __restrict__ input_packed,
    const int32_t * __restrict__ weights_packed,
    int hid_dim_quads,
    int weight_stride,
    int tid_out,
    AccumType sb,
    uint32_t noise_seed,
    int ns,
    float scale_out = 1.0f
) {
    AccumType acc = 0;
    for(int k=0; k<hid_dim_quads; k++) {
        acc = simd_dp4a(input_packed[k], weights_packed[k * weight_stride + tid_out], acc);
    }
    
    if(ns != 0) {
        acc += ((sb * (AccumType)(noise_from_hash(noise_seed, tid_out) * scale_out)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
    }
    return acc;
}

__device__ __forceinline__ void compute_qkv_projection(
    const int32_t * __restrict__ input_packed,
    const int32_t * __restrict__ wq_packed,
    const int32_t * __restrict__ wk_packed,
    const int32_t * __restrict__ wv_packed,
    int hid_dim_quads,
    int weight_stride,
    int tid_out,
    AccumType &aq, AccumType &ak, AccumType &av,
    AccumType sbq, AccumType sbk, AccumType sbv,
    uint32_t seed_base,
    int ns,
    float scale_q = 1.0f, float scale_k = 1.0f, float scale_v = 1.0f
) {
    aq = 0; ak = 0; av = 0;
    for(int k=0; k<hid_dim_quads; k++) {
        int32_t v_pack = input_packed[k];
        int w_idx = k * weight_stride + tid_out;
        aq = simd_dp4a(v_pack, wq_packed[w_idx], aq);
        ak = simd_dp4a(v_pack, wk_packed[w_idx], ak);
        av = simd_dp4a(v_pack, wv_packed[w_idx], av);
    }
    
    if(ns != 0) {
        aq += ((sbq * (AccumType)(noise_from_hash(seed_base + SEED_OFF_Q_A, tid_out) * scale_q)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        ak += ((sbk * (AccumType)(noise_from_hash(seed_base + SEED_OFF_K_A, tid_out) * scale_k)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        av += ((sbv * (AccumType)(noise_from_hash(seed_base + SEED_OFF_V_A, tid_out) * scale_v)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
    }
}

__device__ __forceinline__ AccumType apply_rope_integer(AccumType val, int t, int tid) {

    int head_dim_idx = tid % HEAD_DIM;
    int pair_idx = head_dim_idx / 2;
    int is_odd = head_dim_idx % 2; 

    int lut_idx = t * HEAD_DIM + pair_idx * 2;
    int32_t c = d_ROPE_LUT[lut_idx];     // Cosine
    int32_t s = d_ROPE_LUT[lut_idx + 1]; // Sine

    AccumType neighbor_val = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    
    int64_t res;
    if (is_odd == 0) {
        res = ((int64_t)val * c - (int64_t)neighbor_val * s + (1 << (ROPE_SCALE_BIT - 1))) >> ROPE_SCALE_BIT;
    } else {
        res = ((int64_t)neighbor_val * s + (int64_t)val * c + (1 << (ROPE_SCALE_BIT - 1))) >> ROPE_SCALE_BIT;
    }

    return (AccumType)res;
}

__device__ __forceinline__ ActType apply_standard_norm(
    ActType val, 
    int tid, 
    BlockReduce::TempStorage &storage, 
    AccumType &shared_scalar,
    WeightType w, 
    WeightType b,
    uint32_t seed_base, 
    int off_w_a, int off_w_b,
    int off_b_a, int off_b_b,
    int ns,
    float scale_w = 1.0f,
    float scale_b = 1.0f
) {
    AccumType x = (AccumType)val;
    AccumType tot = block_reduce_sum_broadcast(abs(x), storage, shared_scalar);
    AccumType mn = tot / HIDDEN_DIM; 
    if(!mn) mn = 1;

    AccumType w_mod = w;
    AccumType b_mod = b;

    if (ns != 0) {
        // Universal Rank-1 Noise: Product of two independent samples
        int8_t wn1 = noise_from_hash(seed_base + off_w_a, tid);
        int8_t wn2 = noise_from_hash(seed_base + off_w_b, tid);
        w_mod += ((AccumType)(wn1 * wn2 * scale_w) * ns) >> SIGMA_SHIFT_VECTOR;
        
        int8_t bn1 = noise_from_hash(seed_base + off_b_a, tid);
        int8_t bn2 = noise_from_hash(seed_base + off_b_b, tid);
        b_mod += ((AccumType)(bn1 * bn2 * scale_b) * ns) >> SIGMA_SHIFT_VECTOR;
    }

    return clip( (x * w_mod) / mn + b_mod );
}

struct MlpConfig {
    int off_ln_a, off_ln_b; 
    int off_ln_bias_a, off_ln_bias_b;
    int off_up_a, off_up_b;
    int off_bias_up_a, off_bias_up_b;
    int off_dn_a, off_dn_b;
    int off_bias_dn_a, off_bias_dn_b;
    const char *n1, *n2, *n3;
};

__device__ const MlpConfig CFG_MLP_INIT = {
    SEED_OFF_LN_INIT_A, SEED_OFF_LN_INIT_B, 
    SEED_OFF_LN_INIT_BIAS_A, SEED_OFF_LN_INIT_BIAS_B,
    SEED_OFF_EMB_MLP_UP_A, SEED_OFF_EMB_MLP_UP_B, 
    SEED_OFF_EMB_MLP_BIAS_UP_A, SEED_OFF_EMB_MLP_BIAS_UP_B,
    SEED_OFF_EMB_MLP_DOWN_A, SEED_OFF_EMB_MLP_DOWN_B, 
    SEED_OFF_EMB_MLP_BIAS_DOWN_A, SEED_OFF_EMB_MLP_BIAS_DOWN_B,
    "LN_Init", "InitMLP_Exp", "InitMLP"
};
__device__ const MlpConfig CFG_MLP_LAYER = {
    SEED_OFF_LN_2_A, SEED_OFF_LN_2_B,
    SEED_OFF_LN_2_BIAS_A, SEED_OFF_LN_2_BIAS_B,
    SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, 
    SEED_OFF_MLP_BIAS_UP_A, SEED_OFF_MLP_BIAS_UP_B,
    SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, 
    SEED_OFF_MLP_BIAS_DOWN_A, SEED_OFF_MLP_BIAS_DOWN_B,
    "LN2", "MLP_Exp", "MLP"
};

__device__ void compute_mlp(
    int l, int t, int tid,
    ActType *s_x, ActType *s_mem,
    BlockReduce::TempStorage &temp_storage, AccumType &shared_scalar,
    // Weights
    const WeightType *ln_w, const WeightType *ln_b,
    const WeightType *up_w, const WeightType *up_b,
    const WeightType *dn_w, const WeightType *dn_b,
    // Config
    uint32_t seed_base, int ns, long step, int global_pop_offset,
    const MlpConfig &cfg,
    const AdaptiveScales * __restrict__ scales = nullptr
) {
    float s_ln = 1.0f, s_ln_b = 1.0f;
    if (scales) {
        if (l == 0 && cfg.off_ln_a == SEED_OFF_LN_INIT_A) { // Init MLP
            s_ln = get_adaptive_factor(scales->ln_init[tid], seed_base + cfg.off_ln_a, tid);
            s_ln_b = get_adaptive_factor(scales->ln_init_bias[tid], seed_base + cfg.off_ln_bias_a, tid);
        } else { // Layer MLP
            s_ln = get_adaptive_factor(scales->ln_2[l][tid], seed_base + cfg.off_ln_a, tid);
            s_ln_b = get_adaptive_factor(scales->ln_2_bias[l][tid], seed_base + cfg.off_ln_bias_a, tid);
        }
    }

    ActType *s_norm = &s_mem[HIDDEN_DIM];
    s_norm[tid] = apply_standard_norm(s_x[tid], tid, temp_storage, shared_scalar, ln_w[tid], ln_b[tid], seed_base, cfg.off_ln_a, cfg.off_ln_b, cfg.off_ln_bias_a, cfg.off_ln_bias_b, ns, s_ln, s_ln_b);
    __syncthreads();

    float s_up_in = 1.0f;
    if (scales) {
        if (l == 0 && cfg.off_ln_a == SEED_OFF_LN_INIT_A) s_up_in = get_adaptive_factor(scales->w_emb_mlp_up_row[tid], seed_base + cfg.off_up_b, tid);
        else s_up_in = get_adaptive_factor(scales->w_up_row[l][tid], seed_base + cfg.off_up_b, tid);
    }

    AccumType sb = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + cfg.off_up_b, tid) * s_up_in, temp_storage, shared_scalar);
    ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256];
    for(int sub=0; sub<4; sub++) {
        int oidx = tid + sub*HIDDEN_DIM;
        float s_up_out = 1.0f, s_up_b = 1.0f;
        if (scales) {
            if (l == 0 && cfg.off_ln_a == SEED_OFF_LN_INIT_A) {
                s_up_out = get_adaptive_factor(scales->w_emb_mlp_up_col[oidx], seed_base + cfg.off_up_a, oidx);
                s_up_b = get_adaptive_factor(scales->mlp_emb_bias_up[oidx], seed_base + cfg.off_bias_up_a, oidx);
            } else {
                s_up_out = get_adaptive_factor(scales->w_up_col[l][oidx], seed_base + cfg.off_up_a, oidx);
                s_up_b = get_adaptive_factor(scales->mlp_bias_up[l][oidx], seed_base + cfg.off_bias_up_a, oidx);
            }
        }

        AccumType a = compute_linear_projection((int32_t*)s_norm, (const int32_t*)up_w, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, sb, seed_base + cfg.off_up_a, ns, s_up_out);
        WeightType b = up_b[oidx];
        
        // Universal Rank-1 Noise for bias
        int8_t nb1 = noise_from_hash(seed_base + cfg.off_bias_up_a, oidx);
        int8_t nb2 = noise_from_hash(seed_base + cfg.off_bias_up_b, oidx);
        AccumType noise_val = ((AccumType)(nb1 * nb2 * s_up_b) * ns) >> SIGMA_SHIFT_VECTOR;
        
        s_mlp[oidx] = d_ACT_LUT[(uint8_t)clip((a>>SHIFT_MLP_UP) + b + noise_val)];
    }
    __syncthreads();

    float s_dn_in = 1.0f;
    AccumType pb = 0;
    for(int sub=0; sub<4; sub++) {
        if (scales) {
            if (l == 0 && cfg.off_ln_a == SEED_OFF_LN_INIT_A) s_dn_in = get_adaptive_factor(scales->w_emb_mlp_down_row[tid + sub*HIDDEN_DIM], seed_base + cfg.off_dn_b, tid + sub*HIDDEN_DIM);
            else s_dn_in = get_adaptive_factor(scales->w_down_row[l][tid + sub*HIDDEN_DIM], seed_base + cfg.off_dn_b, tid + sub*HIDDEN_DIM);
        }
        pb += (AccumType)s_mlp[tid + sub*HIDDEN_DIM] * noise_from_hash(seed_base + cfg.off_dn_b, tid + sub*HIDDEN_DIM) * s_dn_in;
    }
    sb = block_reduce_sum_broadcast(pb, temp_storage, shared_scalar);

    float s_dn_out = 1.0f, s_dn_b = 1.0f;
    if (scales) {
        if (l == 0 && cfg.off_ln_a == SEED_OFF_LN_INIT_A) {
            s_dn_out = get_adaptive_factor(scales->w_emb_mlp_down_col[tid], seed_base + cfg.off_dn_a, tid);
            s_dn_b = get_adaptive_factor(scales->mlp_emb_bias_down[tid], seed_base + cfg.off_bias_dn_a, tid);
        } else {
            s_dn_out = get_adaptive_factor(scales->w_down_col[l][tid], seed_base + cfg.off_dn_a, tid);
            s_dn_b = get_adaptive_factor(scales->mlp_bias_down[l][tid], seed_base + cfg.off_bias_dn_a, tid);
        }
    }

    AccumType adn = compute_linear_projection((int32_t*)s_mlp, (const int32_t*)dn_w, HIDDEN_DIM, HIDDEN_DIM, tid, sb, seed_base + cfg.off_dn_a, ns, s_dn_out);
    WeightType bdn = dn_b[tid];
    
    // Universal Rank-1 Noise for bias
    int8_t nbdn1 = noise_from_hash(seed_base + cfg.off_bias_dn_a, tid);
    int8_t nbdn2 = noise_from_hash(seed_base + cfg.off_bias_dn_b, tid);
    AccumType noise_val = ((AccumType)(nbdn1 * nbdn2 * s_dn_b) * ns) >> SIGMA_SHIFT_VECTOR;

    int32_t *w_max = (int32_t*)&s_mem[2*HIDDEN_DIM];
    
    s_x[tid] = adaptive_layer_normalize<EGG_BLOCK_THREADS/32>( (AccumType)s_x[tid] + (adn >> SHIFT_MLP_DOWN) + bdn + noise_val, tid, w_max); 
    __syncthreads();
}

__device__ void compute_attention(
    int l, int t, int tid,
    const TransformerModel * __restrict__ model,
    ActType *lkv_k, ActType *lkv_v,
    ActType *s_x, ActType *s_mem,
    BlockReduce::TempStorage &temp_storage, AccumType &shared_scalar,
    uint32_t seed_base, int ns, long step, int global_pop_offset,
    const AdaptiveScales * __restrict__ scales = nullptr
) {
    float s_ln = 1.0f, s_ln_b = 1.0f;
    if (scales) {
        s_ln = get_adaptive_factor(scales->ln_1[l][tid], seed_base + SEED_OFF_LN_1_A, tid);
        s_ln_b = get_adaptive_factor(scales->ln_1_bias[l][tid], seed_base + SEED_OFF_LN_1_BIAS_A, tid);
    }

    ActType *s_norm = &s_mem[HIDDEN_DIM];
    s_norm[tid] = apply_standard_norm(s_x[tid], tid, temp_storage, shared_scalar, model->ln_1[l][tid], model->ln_1_bias[l][tid], seed_base, SEED_OFF_LN_1_A, SEED_OFF_LN_1_B, SEED_OFF_LN_1_BIAS_A, SEED_OFF_LN_1_BIAS_B, ns, s_ln, s_ln_b);
    __syncthreads();

    float s_q_in = 1.0f, s_k_in = 1.0f, s_v_in = 1.0f;
    if (scales) {
        s_q_in = get_adaptive_factor(scales->w_q_row[l][tid], seed_base + SEED_OFF_Q_B, tid);
        s_k_in = get_adaptive_factor(scales->w_k_row[l][tid], seed_base + SEED_OFF_K_B, tid);
        s_v_in = get_adaptive_factor(scales->w_v_row[l][tid], seed_base + SEED_OFF_V_B, tid);
    }

    AccumType sbq = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_Q_B, tid) * s_q_in, temp_storage, shared_scalar);
    AccumType sbk = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_K_B, tid) * s_k_in, temp_storage, shared_scalar);
    AccumType sbv = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_V_B, tid) * s_v_in, temp_storage, shared_scalar);

    float s_q_out = 1.0f, s_k_out = 1.0f, s_v_out = 1.0f;
    if (scales) {
        s_q_out = get_adaptive_factor(scales->w_q_col[l][tid], seed_base + SEED_OFF_Q_A, tid);
        s_k_out = get_adaptive_factor(scales->w_k_col[l][tid], seed_base + SEED_OFF_K_A, tid);
        s_v_out = get_adaptive_factor(scales->w_v_col[l][tid], seed_base + SEED_OFF_V_A, tid);
    }

    AccumType aq, ak, av;
    compute_qkv_projection((int32_t*)s_norm, (const int32_t*)model->w_q[l], (const int32_t*)model->w_k[l], (const int32_t*)model->w_v[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, aq, ak, av, sbq, sbk, sbv, seed_base, ns, s_q_out, s_k_out, s_v_out);

    int32_t *w_max = (int32_t*)&s_mem[2*HIDDEN_DIM];
    ActType qv = adaptive_qkv_normalize<EGG_BLOCK_THREADS/32>(apply_rope_integer(aq, t, tid), tid, w_max);
    lkv_k[t*HIDDEN_DIM + tid] = adaptive_qkv_normalize<EGG_BLOCK_THREADS/32>(apply_rope_integer(ak, t, tid), tid, w_max);
    lkv_v[t*HIDDEN_DIM + tid] = adaptive_qkv_normalize<EGG_BLOCK_THREADS/32>(av, tid, w_max);
    __syncthreads();

    // Attention
    int32_t *s_attn = (int32_t*)&s_mem[2*HIDDEN_DIM];
    int32_t *s_h_max = (int32_t*)&s_mem[2*HIDDEN_DIM + N_HEADS*4];
    int h = tid / HEAD_DIM;
    if(tid < N_HEADS) { s_attn[tid] = 0; s_h_max[tid] = INT_MIN; }
    __syncthreads();

    // Pass 1
    for(int ctx=0; ctx <= t; ctx++) {
        AccumType df = (AccumType)qv * lkv_k[ctx*HIDDEN_DIM + tid];
        for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
        if ((tid % 32) == 0) atomicAdd(&s_attn[h], (int32_t)df);
        __syncthreads();
        if (tid < N_HEADS) { atomicMax(&s_h_max[tid], s_attn[tid]); s_attn[tid] = 0; }
        __syncthreads();
    }
    // Pass 2
    AttnAccumType w_v_sum = 0; uint64_t tot_sc = 0; int32_t my_h_max = s_h_max[h];
    if(tid < N_HEADS) s_attn[tid] = 0;
    __syncthreads();
    
    for(int ctx=0; ctx <= t; ctx++) {
        AccumType df = (AccumType)qv * lkv_k[ctx*HIDDEN_DIM + tid];
        for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
        if ((tid % 32) == 0) atomicAdd(&s_attn[h], (int32_t)df);
        __syncthreads();
        int32_t wt = softmax_exp_lookup((s_attn[h] >> SHIFT_ATTN) - (my_h_max >> SHIFT_ATTN));
        w_v_sum += (AttnAccumType)wt * lkv_v[ctx*HIDDEN_DIM + tid]; tot_sc += wt;
        __syncthreads();
        if(tid < N_HEADS) s_attn[tid] = 0;
        __syncthreads();
    }
    ActType ao = clip(w_v_sum / (tot_sc ? (int64_t)tot_sc : 1));
    s_norm[tid] = ao; __syncthreads();

    // Out Proj
    float s_o_in = 1.0f, s_o_out = 1.0f;
    if (scales) {
        s_o_in = get_adaptive_factor(scales->w_o_row[l][tid], seed_base + SEED_OFF_O_B, tid);
        s_o_out = get_adaptive_factor(scales->w_o_col[l][tid], seed_base + SEED_OFF_O_A, tid);
    }

    AccumType sb = block_reduce_sum_broadcast((AccumType)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid) * s_o_in, temp_storage, shared_scalar);
    AccumType aco = compute_linear_projection((int32_t*)s_norm, (const int32_t*)model->w_o[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, sb, seed_base + SEED_OFF_O_A, ns, s_o_out);
    s_x[tid] = adaptive_layer_normalize<EGG_BLOCK_THREADS/32>((AccumType)s_x[tid] + (aco >> SHIFT_OUT), tid, w_max);
    __syncthreads();
}

__device__ void compute_transformer_layer(
    int l, int t, int tid,
    const TransformerModel * __restrict__ model,
    ActType *lkv_k, ActType *lkv_v,
    ActType *s_x, ActType *s_mem,
    BlockReduce::TempStorage &temp_storage, AccumType &shared_scalar,
    uint32_t seed_base, int ns, long step, int global_pop_offset,
    const AdaptiveScales * __restrict__ scales = nullptr
) {
    compute_attention(l, t, tid, model, lkv_k, lkv_v, s_x, s_mem, temp_storage, shared_scalar, seed_base, ns, step, global_pop_offset, scales);
    compute_mlp(l, t, tid, s_x, s_mem, temp_storage, shared_scalar, model->ln_2[l], model->ln_2_bias[l], model->w_up[l], model->mlp_bias_up[l], model->w_down[l], model->mlp_bias_down[l], seed_base, ns, step, global_pop_offset, CFG_MLP_LAYER, scales);
}

#endif // EGG_MODEL_LAYERS_CUH
