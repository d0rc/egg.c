#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../include/config.h"
#include "../include/model/layers.cuh"
#include "../include/math/ntt.cuh"
#include "../include/utils/egg_math.h"
#include "../include/utils/debug.h"
#include "../include/globals.cuh"

// Global Definitions
// __constant__ int32_t d_EXP_LUT[SOFTMAX_LUT_SIZE];
// __constant__ int8_t d_ACT_LUT[256];
// __device__ int32_t d_ROPE_LUT[ROPE_LUT_SIZE];
// __device__ unsigned long long d_total_updates;

// Host-side LUTs for initialization
int32_t h_EXP_LUT[SOFTMAX_LUT_SIZE];
int8_t h_ACT_LUT[256];
int32_t h_ROPE_LUT[ROPE_LUT_SIZE];

void init_tables() {    
    for(int i=0; i<SOFTMAX_LUT_SIZE; i++) {
        double val = exp(-(double)i / SOFTMAX_EXP_SCALE) * SOFTMAX_SCALE;
        h_EXP_LUT[i] = (val >= 1.0) ? (int32_t)round(val) : 0;
    }

    for(int i=0; i<256; i++) {
        int8_t input = (int8_t)i;
        double x = (double)input / (1 << FIXED_POINT);
        // GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
        double y = 0.5 * x * (1.0 + erf(x / 1.41421356));
        // Convert back to fixed point
        int val = (int)round(y * (1 << FIXED_POINT));
        h_ACT_LUT[i] = (int8_t)((val > 127) ? 127 : ((val < -127) ? -127 : val));
    }

    for (int t = 0; t < SEQ_LEN; t++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            double theta = pow(10000.0, -2.0 * i / HEAD_DIM);
            double alpha = t * theta;
            double c = cos(alpha);
            double s = sin(alpha);
            
            // Scale by ROPE_SCALE (2^ROPE_SCALE_BIT, default 2^30)
            int32_t c_int = (int32_t)round(c * ROPE_SCALE);
            int32_t s_int = (int32_t)round(s * ROPE_SCALE);
            
            int base_idx = t * (HEAD_DIM) + i * 2;
            h_ROPE_LUT[base_idx] = c_int;
            h_ROPE_LUT[base_idx + 1] = s_int;
        }
    }
}

void copy_tables_to_device() {
    cudaMemcpyToSymbol(d_EXP_LUT, h_EXP_LUT, SOFTMAX_LUT_SIZE*sizeof(int32_t));
    cudaMemcpyToSymbol(d_ACT_LUT, h_ACT_LUT, 256*sizeof(int8_t));
    cudaMemcpyToSymbol(d_ROPE_LUT, h_ROPE_LUT, ROPE_LUT_SIZE*sizeof(int32_t));
}

__global__ void update_overlay_kernel(int *accum, WeightType *overlay, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int changes = accum[idx];
        accum[idx] = 0; // Reset
        
        int ov = overlay[idx];
        if (changes > 0) {
            ov += ADAPTIVE_NOISE_INC;
            if (ov > MAX_VAL) ov = MAX_VAL;
        } else {
            ov -= ADAPTIVE_NOISE_DEC;
            if (ov < MIN_VAL) ov = MIN_VAL;
        }
        overlay[idx] = (WeightType)ov;
    }
}

__device__ __forceinline__ WeightType get_embedding_byte(const WeightType *packed_embedding, int hidden_idx, int token_idx) {
    // Packed layout: [HIDDEN_DIM/4, VOCAB_SIZE] of int32 (4 bytes)
    // Map (hidden_idx, token_idx) to packed layout
    int chunk_idx = hidden_idx >> 2; 
    int byte_off = hidden_idx & 3;
    // Calculate linear byte index: (chunk * VOCAB + token) * 4 + offset
    // The packed_embedding pointer is technically WeightType* (int8_t*), so we treat it as byte array.
    long idx = ((long)chunk_idx * VOCAB_SIZE + token_idx) * 4 + byte_off;
    return packed_embedding[idx];
}

__global__ void __launch_bounds__(MAX_BLOCK_THREADS) train_sequence_kernel(
    const TokenType * __restrict__ dataset, long data_len, int start_idx,
    const TransformerModel * __restrict__ model,
    ActType * __restrict__ global_kv_cache,
    int32_t *accum_loss, uint32_t step_seed,
    int global_pop_offset, long step,
    const AdaptiveScales * __restrict__ scales = nullptr
) {
    int p_idx = global_pop_offset + blockIdx.x; 
    // if (p_idx >= POPULATION_SIZE) return; // Handled by caller logic usually, but good safety
    int tid = threadIdx.x;
    if (tid >= HIDDEN_DIM) return;

    extern __shared__ ActType s_mem[];
    ActType *s_x = s_mem; 
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ AccumType shared_scalar;

    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
#if USE_SAME_DATA == 1
    long stream_pos = start_idx % (data_len - SEQ_LEN);
#else
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
#endif
    int ns = (p_idx % 2 == 0) ? 1 : -1;
    size_t kv_layer_stride = 2ULL * SEQ_LEN * HIDDEN_DIM;
    size_t kv_ind_offset = (size_t)blockIdx.x * N_LAYERS * kv_layer_stride;

    long long my_loss = 0; // Loss accumulation needs high precision

#if NTT_MODE != 0
    // Pre-compute NTT transform of sequence (once at start)
    // Use end of s_mem for NTT buffer: after 2*HIDDEN_DIM + 512 + 4*HIDDEN_DIM
    int32_t *s_ntt = (int32_t*)&s_mem[2*HIDDEN_DIM + 512 + 4*HIDDEN_DIM];
    if (tid < SEQ_LEN) {
        s_ntt[tid] = (int32_t)dataset[stream_pos + tid];
    }
    __syncthreads();
    ntt_transform_sequence(s_ntt, SEQ_LEN, tid);
    __syncthreads();
#endif

    for (int t = 0; t < SEQ_LEN; t++) {
        
        // 1. Embedding
        TokenType input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        WeightType emb = get_embedding_byte(model->embedding, tid, input_token);
        WeightType ebias = model->emb_bias[tid];
        
        float s_emb_b = 1.0f;
        if (scales) s_emb_b = get_adaptive_factor(scales->emb_bias[tid], (step_seed + pair_idx) + SEED_OFF_EMB_BIAS_A, tid);

        int8_t emb_bias_n1 = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_BIAS_A, tid);
        int8_t emb_bias_n2 = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_BIAS_B, tid);
        AccumType emb_bias_chk = ((AccumType)(emb_bias_n1 * emb_bias_n2 * s_emb_b) * ns) >> SIGMA_SHIFT_VECTOR;
        
        float s_emb_row = 1.0f, s_emb_col = 1.0f;
        if (scales) {
            s_emb_row = get_adaptive_factor(scales->embedding_row[input_token], seed_emb, input_token);
            s_emb_col = get_adaptive_factor(scales->embedding_col[tid], seed_emb + HIDDEN_DIM, tid);
        }

        // RoPE: Absolute pos emb removed
        int8_t a_tok = noise_from_hash(seed_emb, input_token);
        int8_t b_dim = noise_from_hash(seed_emb + HIDDEN_DIM, tid);
        AccumType perturb = ((AccumType)(a_tok * b_dim * s_emb_row * s_emb_col) * ns) >> (FIXED_POINT + SIGMA_SHIFT);

#if NTT_MODE != 0
        // NTT coefficient embedding: decompose int32 NTT coeff into 4 bytes
        int32_t ntt_coeff = s_ntt[t];
        uint8_t nb1 = (uint8_t)((ntt_coeff >> 8) & 0xFF);
        uint8_t nb2 = (uint8_t)((ntt_coeff >> 16) & 0xFF);
        uint8_t nb3 = (uint8_t)((ntt_coeff >> 24) & 0xFF);
        
        WeightType ne1 = get_embedding_byte(model->ntt_emb1, tid, nb1);
        WeightType ne2 = get_embedding_byte(model->ntt_emb2, tid, nb2);
        WeightType ne3 = get_embedding_byte(model->ntt_emb3, tid, nb3);
        
        s_x[tid] = clip((AccumType)emb + ebias + emb_bias_chk + perturb + (ne1 >> 2) + (ne2 >> 2) + (ne3 >> 2));
#else
        s_x[tid] = clip((AccumType)emb + ebias + emb_bias_chk + perturb);
#endif
        __syncthreads();
        EGG_TRACE_STAT(step, t, -1, "Emb", s_x, HIDDEN_DIM);

        compute_mlp(0, t, tid, s_x, s_mem, temp_storage, shared_scalar, model->ln_init, model->ln_init_bias, model->w_emb_mlp_up, model->mlp_emb_bias_up, model->w_emb_mlp_down, model->mlp_emb_bias_down, step_seed + pair_idx, ns, step, global_pop_offset, CFG_MLP_INIT, scales);

        // 2. Stack
        for (int cycle = 0; cycle < NUMBER_OF_CYCLES; cycle++) {
            for (int l = 0; l < N_LAYERS; l++) {
                uint32_t seed_base = (step_seed + pair_idx) + (cycle * N_LAYERS * 1000) + (l * 1000);
                
                ActType* lkv_base = global_kv_cache + kv_ind_offset + (l * kv_layer_stride);
                ActType* lkv_k = lkv_base;
                ActType* lkv_v = lkv_base + SEQ_LEN*HIDDEN_DIM;

                compute_transformer_layer(
                    l, t, tid, model, 
                    lkv_k, lkv_v,
                    s_x, s_mem, 
                    temp_storage, shared_scalar,
                    seed_base, ns, step,
                    global_pop_offset,
                    scales
                );
            }
        }

        // 3. Final Head
        float s_ln_f = 1.0f, s_ln_f_b = 1.0f;
        if (scales) {
            s_ln_f = get_adaptive_factor(scales->ln_f[tid], (step_seed + pair_idx) + SEED_OFF_LN_F_A, tid);
            s_ln_f_b = get_adaptive_factor(scales->ln_f_bias[tid], (step_seed + pair_idx) + SEED_OFF_LN_F_BIAS_A, tid);
        }

        ActType nf = apply_standard_norm(
            s_x[tid], tid, temp_storage, shared_scalar,
            model->ln_f[tid], model->ln_f_bias[tid],
            step_seed + pair_idx, SEED_OFF_LN_F_A, SEED_OFF_LN_F_B, SEED_OFF_LN_F_BIAS_A, SEED_OFF_LN_F_BIAS_B, ns,
            s_ln_f, s_ln_f_b
        );
        
        // Reuse s_mem[HD] for normed
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();
        
        // Head Projection (using embedding weights transposed)
        // We don't have explicit head weights, we use embedding weights.
        // So we should use embedding scales?
        // Yes, embedding_col corresponds to output dim of embedding (HIDDEN_DIM).
        // Here we project FROM hidden TO vocab.
        // So input is HIDDEN_DIM. Output is VOCAB.
        // Weights are [HIDDEN, VOCAB] (transposed embedding).
        // So we use embedding_col for input scaling, embedding_row for output scaling.
        
        float s_head_in = 1.0f;
        if (scales) s_head_in = get_adaptive_factor(scales->embedding_col[tid], step_seed + pair_idx + SEED_OFF_EMB + HIDDEN_DIM, tid);

        AccumType sbh = block_reduce_sum_broadcast((AccumType)nf * noise_from_hash(step_seed + pair_idx + SEED_OFF_EMB + HIDDEN_DIM, tid) * s_head_in, temp_storage, shared_scalar);

        // --- Two-Pass Softmax (Loop-based for Large Vocab) ---
        
        // Pass 1: Find Max Logit
        int32_t local_max = INT_MIN;
        const int32_t *wh_p = (const int32_t*)model->embedding;
        int32_t *v_ptr_h = (int32_t*)s_norm;
        
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            float s_head_out = 1.0f;
            if (scales) s_head_out = get_adaptive_factor(scales->embedding_row[v], step_seed + pair_idx + SEED_OFF_EMB, v);
            
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, sbh, step_seed + pair_idx + SEED_OFF_EMB, ns, s_head_out);
            int32_t lgt = ah >> SHIFT_LOGIT;
            if (lgt > local_max) local_max = lgt;
        }
        
        // Reduce max across threads
        int32_t global_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
        __shared__ int32_t s_global_max;
        if (tid == 0) s_global_max = global_max;
        __syncthreads();
        global_max = s_global_max;

        // Pass 2: Sum Exp and Find Target Logit
        int64_t local_sum_ex = 0;
        __shared__ int32_t s_target_logit;
        if (tid == 0) s_target_logit = 0;
        __syncthreads();

        TokenType target_token = dataset[stream_pos + t + 1];
        
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            float s_head_out = 1.0f;
            if (scales) s_head_out = get_adaptive_factor(scales->embedding_row[v], step_seed + pair_idx + SEED_OFF_EMB, v);

            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, sbh, step_seed + pair_idx + SEED_OFF_EMB, ns, s_head_out);
            int32_t lgt = ah >> SHIFT_LOGIT;
            
            int32_t shifted = lgt - global_max;
            local_sum_ex += softmax_exp_lookup(shifted);
            
            if (v == (int)target_token) s_target_logit = lgt;
        }
        __syncthreads();
        
        // Reduce sum across threads (using 64-bit reduction)
        typedef cub::BlockReduce<long long, EGG_BLOCK_THREADS> BlockReduce64;
        __shared__ typename BlockReduce64::TempStorage temp_storage64;
        long long global_sum_ex = BlockReduce64(temp_storage64).Sum(local_sum_ex);

        // Loss Calc (Thread 0)
        if (tid == 0) {
            int64_t log_sum = 0;
            if (global_sum_ex > 0) {
                uint64_t x = (uint64_t)global_sum_ex; int pos = 0;
                while(x >= 256) { x >>= 8; pos += 8; }
                if(x >= 16) { x >>= 4; pos += 4; }
                if(x >= 4) { x >>= 2; pos += 2; }
                if(x >= 2) { pos += 1; }
                
                log_sum = (pos << 4) - (SOFTMAX_SCALE_BIT << 4);
            }
            my_loss += (log_sum >> 1) + global_max - s_target_logit;
        }
        __syncthreads(); 
    }

    if (tid == 0) accum_loss[blockIdx.x] = (int32_t)my_loss; // Store at block index (relative to chunk)
}

__global__ void compute_fitness_kernel(const int32_t *accum_loss, int32_t *fitnesses, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t p = accum_loss[2*idx]; int32_t n = accum_loss[2*idx+1];
    fitnesses[idx] = (p < n) ? 1 : ((n < p) ? -1 : 0);
}

__global__ void generate_sequence_kernel(
    TokenType * buffer, int seed_len, int gen_len,
    const TransformerModel * __restrict__ model,
    ActType * __restrict__ kv_cache,
    uint32_t seed,
    float temp, float min_p, float penalty
) {
    if (blockIdx.x > 0) return;
    int tid = threadIdx.x;
    if (tid >= HIDDEN_DIM) return;

    extern __shared__ ActType s_mem[];
    ActType *s_x = s_mem; 
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ AccumType shared_scalar;

    // Shared memory for generated tokens (for presence penalty)
    __shared__ TokenType s_gen_tokens[256]; 

    int total_len = seed_len + gen_len;
    size_t kv_layer_stride = 2ULL * total_len * HIDDEN_DIM;

    // Load initial seed tokens to shared memory
    if (tid < seed_len && tid < 256) {
        s_gen_tokens[tid] = buffer[tid];
    }
    __syncthreads();

    for (int t = 0; t < total_len - 1; t++) { 
        __syncthreads(); 

        TokenType input_token = buffer[t];
        
        // 1. Embedding
        WeightType emb = get_embedding_byte(model->embedding, tid, input_token);
        WeightType ebias = model->emb_bias[tid];
        
        s_x[tid] = clip((AccumType)emb + ebias);
        __syncthreads();

        compute_mlp(0, t, tid, s_x, s_mem, temp_storage, shared_scalar, model->ln_init, model->ln_init_bias, model->w_emb_mlp_up, model->mlp_emb_bias_up, model->w_emb_mlp_down, model->mlp_emb_bias_down, 0, 0, -1, 0, CFG_MLP_INIT);

        // 2. Layers
        for (int cycle = 0; cycle < NUMBER_OF_CYCLES; cycle++) {
            for (int l = 0; l < N_LAYERS; l++) {
                ActType* lkv_base = kv_cache + (l * kv_layer_stride);
                ActType* lkv_k = lkv_base;
                ActType* lkv_v = lkv_base + total_len*HIDDEN_DIM;

                compute_transformer_layer(
                    l, t, tid, model, 
                    lkv_k, lkv_v,
                    s_x, s_mem, 
                    temp_storage, shared_scalar,
                    0, 0, -1, 
                    0
                );
            }
        }

        // Final Head
        ActType nf = apply_standard_norm(
            s_x[tid], tid, temp_storage, shared_scalar,
            model->ln_f[tid], model->ln_f_bias[tid],
            0, 0, 0, 0, 0, 0
        );
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        // --- Three-Pass Sampling (Loop-based) ---
        
        // Precompute constants
        int32_t penalty_int = (int32_t)(penalty * (1 << SHIFT_LOGIT));
        int32_t min_p_limit = (int32_t)(logf(min_p) * SOFTMAX_EXP_SCALE);

        // Pass 1: Find Max Logit
        int32_t local_max = INT_MIN;
        const int32_t *wh_p = (const int32_t*)model->embedding;
        int32_t *v_ptr_h = (int32_t*)s_norm;
        
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, 0, 0, 0);
            int32_t lgt = ah >> SHIFT_LOGIT;
            
            // Presence Penalty
            for (int i = 0; i <= t; i++) {
                if (s_gen_tokens[i] == (TokenType)v) {
                    lgt -= penalty_int;
                }
            }

            if (lgt > local_max) local_max = lgt;
        }
        
        int32_t global_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
        __shared__ int32_t s_global_max;
        if (tid == 0) s_global_max = global_max;
        __syncthreads();
        global_max = s_global_max;

        // Pass 2: Sum Exp
        int64_t local_sum_ex = 0;
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, 0, 0, 0);
            int32_t lgt = ah >> SHIFT_LOGIT;
            
            // Presence Penalty
            for (int i = 0; i <= t; i++) {
                if (s_gen_tokens[i] == (TokenType)v) {
                    lgt -= penalty_int;
                }
            }

            int32_t shifted = lgt - global_max;
            
            // Temperature Scaling
            int32_t scaled_diff = (int32_t)(shifted / temp);

            // Min-p Filtering
            if (scaled_diff >= min_p_limit) {
                local_sum_ex += softmax_exp_lookup(scaled_diff);
            }
        }
        
        typedef cub::BlockReduce<long long, EGG_BLOCK_THREADS> BlockReduce64;
        __shared__ typename BlockReduce64::TempStorage temp_storage64;
        long long global_sum_ex = BlockReduce64(temp_storage64).Sum(local_sum_ex);
        
        __shared__ long long s_thresh;
        __shared__ int s_selected;
        if (tid == 0) {
            uint32_t s = seed + t * 555;
            uint32_t r = hash_rng(s, 0);
            s_thresh = (global_sum_ex > 0) ? (r % global_sum_ex) : 0;
            s_selected = 0; 
        }
        __syncthreads();

        // Pass 3: Select using BlockScan
        typedef cub::BlockScan<long long, EGG_BLOCK_THREADS> BlockScan64;
        __shared__ typename BlockScan64::TempStorage scan_storage;
        long long thread_prefix_sum;
        BlockScan64(scan_storage).ExclusiveSum(local_sum_ex, thread_prefix_sum);
        
        if (s_thresh >= thread_prefix_sum && s_thresh < thread_prefix_sum + local_sum_ex) {
            long long running = thread_prefix_sum;
            for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
                AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, 0, 0, 0);
                int32_t lgt = ah >> SHIFT_LOGIT;
                
                // Presence Penalty
                for (int i = 0; i <= t; i++) {
                    if (s_gen_tokens[i] == (TokenType)v) {
                        lgt -= penalty_int;
                    }
                }

                int32_t shifted = lgt - global_max;
                
                // Temperature Scaling
                int32_t scaled_diff = (int32_t)(shifted / temp);

                // Min-p Filtering
                int32_t ex = 0;
                if (scaled_diff >= min_p_limit) {
                    ex = softmax_exp_lookup(scaled_diff);
                }
                
                running += ex;
                if (running > s_thresh) {
                    s_selected = v;
                    break;
                }
            }
        }
        __syncthreads();

        if (t >= seed_len - 1) {
            if (tid == 0) {
                buffer[t + 1] = (TokenType)s_selected;
                if (t + 1 < 256) s_gen_tokens[t + 1] = (TokenType)s_selected;
            }
        }
        __syncthreads();
    }
}
