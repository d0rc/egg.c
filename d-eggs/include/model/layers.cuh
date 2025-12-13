#ifndef EGG_MODEL_LAYERS_CUH
#define EGG_MODEL_LAYERS_CUH

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../config.h"
#include "../math/adaptive_norm.cuh"
#include "../utils/egg_math.h"
#include "definitions.h"
#include "../globals.cuh"

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
    index = (index < 0) ? 0 : ((index > 255) ? 255 : index);
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
    int ns
) {
    AccumType acc = 0;
    for(int k=0; k<hid_dim_quads; k++) {
        acc = simd_dp4a(input_packed[k], weights_packed[k * weight_stride + tid_out], acc);
    }
    
    if(ns != 0) {
        acc += ((sb * (AccumType)noise_from_hash(noise_seed, tid_out)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
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
    int ns
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
        aq += ((sbq * (AccumType)noise_from_hash(seed_base + SEED_OFF_Q_A, tid_out)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        ak += ((sbk * (AccumType)noise_from_hash(seed_base + SEED_OFF_K_A, tid_out)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        av += ((sbv * (AccumType)noise_from_hash(seed_base + SEED_OFF_V_A, tid_out)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
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
    int ns
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
        w_mod += ((AccumType)wn1 * wn2 * ns) >> SIGMA_SHIFT_VECTOR;
        
        int8_t bn1 = noise_from_hash(seed_base + off_b_a, tid);
        int8_t bn2 = noise_from_hash(seed_base + off_b_b, tid);
        b_mod += ((AccumType)bn1 * bn2 * ns) >> SIGMA_SHIFT_VECTOR;
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
    const MlpConfig &cfg
) {
    ActType *s_norm = &s_mem[HIDDEN_DIM];
    s_norm[tid] = apply_standard_norm(s_x[tid], tid, temp_storage, shared_scalar, ln_w[tid], ln_b[tid], seed_base, cfg.off_ln_a, cfg.off_ln_b, cfg.off_ln_bias_a, cfg.off_ln_bias_b, ns);
    __syncthreads();

    AccumType sb = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + cfg.off_up_b, tid), temp_storage, shared_scalar);
    ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256];
    for(int sub=0; sub<4; sub++) {
        int oidx = tid + sub*HIDDEN_DIM;
        AccumType a = compute_linear_projection((int32_t*)s_norm, (const int32_t*)up_w, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, sb, seed_base + cfg.off_up_a, ns);
        WeightType b = up_b[oidx];
        
        // Universal Rank-1 Noise for bias
        int8_t nb1 = noise_from_hash(seed_base + cfg.off_bias_up_a, oidx);
        int8_t nb2 = noise_from_hash(seed_base + cfg.off_bias_up_b, oidx);
        AccumType noise_val = ((AccumType)nb1 * nb2 * ns) >> SIGMA_SHIFT_VECTOR;
        
        s_mlp[oidx] = d_ACT_LUT[(uint8_t)clip((a>>SHIFT_MLP_UP) + b + noise_val)];
    }
    __syncthreads();

    AccumType pb = 0;
    for(int sub=0; sub<4; sub++) pb += (AccumType)s_mlp[tid + sub*HIDDEN_DIM] * noise_from_hash(seed_base + cfg.off_dn_b, tid + sub*HIDDEN_DIM);
    sb = block_reduce_sum_broadcast(pb, temp_storage, shared_scalar);

    AccumType adn = compute_linear_projection((int32_t*)s_mlp, (const int32_t*)dn_w, HIDDEN_DIM, HIDDEN_DIM, tid, sb, seed_base + cfg.off_dn_a, ns);
    WeightType bdn = dn_b[tid];
    
    // Universal Rank-1 Noise for bias
    int8_t nbdn1 = noise_from_hash(seed_base + cfg.off_bias_dn_a, tid);
    int8_t nbdn2 = noise_from_hash(seed_base + cfg.off_bias_dn_b, tid);
    AccumType noise_val = ((AccumType)nbdn1 * nbdn2 * ns) >> SIGMA_SHIFT_VECTOR;

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
    uint32_t seed_base, int ns, long step, int global_pop_offset
) {
    ActType *s_norm = &s_mem[HIDDEN_DIM];
    s_norm[tid] = apply_standard_norm(s_x[tid], tid, temp_storage, shared_scalar, model->ln_1[l][tid], model->ln_1_bias[l][tid], seed_base, SEED_OFF_LN_1_A, SEED_OFF_LN_1_B, SEED_OFF_LN_1_BIAS_A, SEED_OFF_LN_1_BIAS_B, ns);
    __syncthreads();

    AccumType sbq = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_Q_B, tid), temp_storage, shared_scalar);
    AccumType sbk = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_K_B, tid), temp_storage, shared_scalar);
    AccumType sbv = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_V_B, tid), temp_storage, shared_scalar);

    AccumType aq, ak, av;
    compute_qkv_projection((int32_t*)s_norm, (const int32_t*)model->w_q[l], (const int32_t*)model->w_k[l], (const int32_t*)model->w_v[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, aq, ak, av, sbq, sbk, sbv, seed_base, ns);

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
    AccumType sb = block_reduce_sum_broadcast((AccumType)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid), temp_storage, shared_scalar);
    AccumType aco = compute_linear_projection((int32_t*)s_norm, (const int32_t*)model->w_o[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, sb, seed_base + SEED_OFF_O_A, ns);
    s_x[tid] = adaptive_layer_normalize<EGG_BLOCK_THREADS/32>((AccumType)s_x[tid] + (aco >> SHIFT_OUT), tid, w_max);
    __syncthreads();
}

__device__ void compute_transformer_layer(
    int l, int t, int tid,
    const TransformerModel * __restrict__ model,
    ActType *lkv_k, ActType *lkv_v,
    ActType *s_x, ActType *s_mem,
    BlockReduce::TempStorage &temp_storage, AccumType &shared_scalar,
    uint32_t seed_base, int ns, long step, int global_pop_offset
) {
    compute_attention(l, t, tid, model, lkv_k, lkv_v, s_x, s_mem, temp_storage, shared_scalar, seed_base, ns, step, global_pop_offset);
    compute_mlp(l, t, tid, s_x, s_mem, temp_storage, shared_scalar, model->ln_2[l], model->ln_2_bias[l], model->w_up[l], model->mlp_bias_up[l], model->w_down[l], model->mlp_bias_down[l], seed_base, ns, step, global_pop_offset, CFG_MLP_LAYER);
}

#endif // EGG_MODEL_LAYERS_CUH
