#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

// --- CONFIGURATION ---
#define HIDDEN_DIM 256
#define HEAD_DIM 64
#define N_LAYERS 2
#define SEQ_LEN 128     
#define VOCAB_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCK_THREADS 1024 
#define ALIGNED_DIM ((HIDDEN_DIM + 31) & ~31)
#define BLOCK_THREADS (ALIGNED_DIM > MAX_BLOCK_THREADS ? MAX_BLOCK_THREADS : ALIGNED_DIM)
#define N_HEADS (HIDDEN_DIM / HEAD_DIM)

// Population Sizing (Target 20GB)
#define POPULATION_BATCH_SIZE (8192 * 5)
#define POPULATION_SIZE (POPULATION_BATCH_SIZE * 1)

#define FIXED_POINT 4
#define SIGMA_SHIFT 0
#define SIGMA_SHIFT_VECTOR 2
#define MAX_VAL 127
#define MIN_VAL -127

// Softmax Configuration - Extended to 256 entries to match legacy range
#define SOFTMAX_SCALE_BIT 20
#define SOFTMAX_SCALE (1 << SOFTMAX_SCALE_BIT)  // 1,048,576
#define SOFTMAX_LUT_SIZE 256
// Scaling factor: legacy used 2^(i/16), we use exp(-i/K) where K = 16/ln(2) ≈ 23.08
// This makes exp(-i/23.08) ≈ 2^(-i/16), matching the legacy curve shape
#define SOFTMAX_EXP_SCALE 23.08

#define SEED_OFF_EMB 0
#define SEED_OFF_POS 1
#define SEED_OFF_LN_1 100
#define SEED_OFF_Q_A 200
#define SEED_OFF_Q_B 201
#define SEED_OFF_K_A 202
#define SEED_OFF_K_B 203
#define SEED_OFF_V_A 204
#define SEED_OFF_V_B 205
#define SEED_OFF_O_A 206
#define SEED_OFF_O_B 207
#define SEED_OFF_LN_2 300
#define SEED_OFF_MLP_UP_A 400
#define SEED_OFF_MLP_UP_B 401
#define SEED_OFF_MLP_DOWN_A 402
#define SEED_OFF_MLP_DOWN_B 403
#define SEED_OFF_LN_F 900
#define SEED_OFF_HEAD 999

// ADAM HYPERPARAMS
float get_learning_rate(long step) {
    if (step < 40) {
        return 0.9f;
    }
    if (step < 80) {
        return 0.7f;
    }
    if (step < 120) {
        return 0.5f;
    }
    return 0.1f;
}

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.98f
#define ADAM_EPS 1e-8f
#define ADAM_WEIGHT_DECAY 0.01f

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA Error: %s:%d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

// --- TYPE ALIASES ---
using WeightType    = int8_t;
using ActType       = int8_t;
using AccumType     = int32_t;   // 32-bit sufficient for ~6M max sum
using AttnAccumType = long long; // Need high precision for attention weighted sum
using VoteType      = int32_t;

typedef struct { uint8_t *data; long length; } Dataset;

typedef struct {
    WeightType embedding[VOCAB_SIZE * HIDDEN_DIM];
    WeightType pos_emb[SEQ_LEN * HIDDEN_DIM];
    WeightType ln_1[N_LAYERS][HIDDEN_DIM];
    WeightType w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    WeightType ln_2[N_LAYERS][HIDDEN_DIM];
    WeightType w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    WeightType w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    WeightType ln_f[HIDDEN_DIM];
    WeightType head[HIDDEN_DIM * VOCAB_SIZE];
} TransformerModel;

// Adam State Structures
typedef struct {
    float m;
    float v;
    float acc; // Fractional accumulator for integer weights
} AdamParam;

typedef struct {
    AdamParam embedding[VOCAB_SIZE * HIDDEN_DIM];
    AdamParam pos_emb[SEQ_LEN * HIDDEN_DIM];
    AdamParam ln_1[N_LAYERS][HIDDEN_DIM];
    AdamParam w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam ln_2[N_LAYERS][HIDDEN_DIM];
    AdamParam w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    AdamParam w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    AdamParam ln_f[HIDDEN_DIM];
    AdamParam head[HIDDEN_DIM * VOCAB_SIZE];
} AdamModel;

ActType *d_kv_cache = NULL;

// Legacy EXP2 table (still used for some operations)
__constant__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];

// New softmax LUT: exp(-i) * 2^20 for i in [0, 14]
__constant__ int32_t d_EXP_LUT[SOFTMAX_LUT_SIZE];
int32_t h_EXP_LUT[SOFTMAX_LUT_SIZE];
__device__ int32_t d_debug_updates[2];
__device__ unsigned long long d_total_updates;

// --- HOST HELPER ---
double get_time_diff_ms(struct timespec start, struct timespec end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1e6);
}

void init_tables() {
    // Legacy base-2 table
    for(int i=0; i<256; i++) h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
    
    // New scaled exponential LUT: exp(-i/K) * 2^20 where K ≈ 23.08
    // This matches legacy 2^(i/16) curve shape but with proper max-subtraction softmax
    for(int i=0; i<SOFTMAX_LUT_SIZE; i++) {
        double val = exp(-(double)i / SOFTMAX_EXP_SCALE) * SOFTMAX_SCALE;
        h_EXP_LUT[i] = (val >= 1.0) ? (int32_t)round(val) : 0;
    }
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *state = x; return x;
}
static inline int8_t gen_noise_host(uint32_t *rng) { return (int8_t)((xorshift32_host(rng) & 1 ? 1 : -1) * ((xorshift32_host(rng) >> 1) & 15)); }
void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) dst[c * rows + r] = src[r * cols + c];
}
void init_model(TransformerModel *model) {
    uint32_t rng = 42;
    TransformerModel *temp = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    if(!temp) exit(1);
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    transpose_matrix(model->embedding, temp->embedding, VOCAB_SIZE, HIDDEN_DIM);
    for(int i=0; i<SEQ_LEN*HIDDEN_DIM; i++) model->pos_emb[i] = gen_noise_host(&rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, HIDDEN_DIM, VOCAB_SIZE);
    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) { model->ln_1[l][i]=16; model->ln_2[l][i]=16; }
        int d2 = HIDDEN_DIM*HIDDEN_DIM;
        for(int i=0; i<d2; i++) temp->w_q[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_q[l], temp->w_q[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_k[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_k[l], temp->w_k[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_v[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_v[l], temp->w_v[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_o[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_o[l], temp->w_o[l], HIDDEN_DIM, HIDDEN_DIM);
        
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_up[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_up[l], temp->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM);
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_down[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_down[l], temp->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM);
    }
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_f[i]=16;
    free(temp);
}

// --- DEVICE KERNELS & HELPERS ---

typedef cub::BlockReduce<AccumType, BLOCK_THREADS> BlockReduce;

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16; return x;
}
__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx); return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}
__device__ __forceinline__ int8_t clip(AccumType a) { return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a); }

// Helper: Softmax exponential lookup - 256 entry version
// Input: diff = score - max_score (should be <= 0)
// Returns: exp(diff/K) * 2^20, clamped to [0, 255] index range
__device__ __forceinline__ int32_t softmax_exp_lookup(int32_t diff) {
    int index = -diff;  // diff is negative or zero, so index is positive
    index = (index < 0) ? 0 : ((index > 255) ? 255 : index);
    return d_EXP_LUT[index];
}

// Helper: Sum reduction + Broadcast
__device__ __forceinline__ AccumType block_reduce_sum_broadcast(AccumType val, BlockReduce::TempStorage &storage, AccumType &shared_var) {
    AccumType total = BlockReduce(storage).Sum(val);
    if (threadIdx.x == 0) shared_var = total;
    __syncthreads();
    AccumType ret = shared_var;
    __syncthreads();
    return ret;
}

__global__ void __launch_bounds__(MAX_BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset, long data_len, int start_idx,
    const TransformerModel * __restrict__ model,
    ActType * __restrict__ global_kv_cache,
    int32_t *accum_loss, uint32_t step_seed,
    int global_pop_offset
) {
    int p_idx = global_pop_offset + blockIdx.x; 
    if (p_idx >= POPULATION_SIZE) return;
    int tid = threadIdx.x;
    if (tid >= HIDDEN_DIM) return;

    extern __shared__ ActType s_mem[];
    ActType *s_x = s_mem; 
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ AccumType shared_scalar;

    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;
    size_t kv_layer_stride = 2ULL * SEQ_LEN * HIDDEN_DIM;
    size_t kv_ind_offset = (size_t)blockIdx.x * N_LAYERS * kv_layer_stride;

    long long my_loss = 0; // Loss accumulation needs high precision

    for (int t = 0; t < SEQ_LEN; t++) {
        
        // 1. Embedding
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        WeightType emb = model->embedding[tid * VOCAB_SIZE + input_token];
        WeightType pos = model->pos_emb[t * HIDDEN_DIM + tid];
        int8_t a_tok = noise_from_hash(seed_emb, input_token);
        int8_t b_dim = noise_from_hash(seed_emb + HIDDEN_DIM, tid);
        AccumType perturb = ((AccumType)a_tok * b_dim * ns) >> (FIXED_POINT + SIGMA_SHIFT);

        uint32_t seed_pos = (step_seed + pair_idx) + SEED_OFF_POS;
        int8_t a_pos = noise_from_hash(seed_pos, t);
        int8_t b_pos = noise_from_hash(seed_pos + SEQ_LEN, tid);
        AccumType perturb_pos = ((AccumType)a_pos * b_pos * ns) >> (FIXED_POINT + SIGMA_SHIFT);

        s_x[tid] = clip((AccumType)emb + pos + perturb + perturb_pos);
        __syncthreads();

        // 2. Stack
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed_base = (step_seed + pair_idx) + (l * 1000);
            ActType *s_norm = &s_mem[HIDDEN_DIM]; // Normalized buf

            // LN 1
            AccumType total_sum = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mean = total_sum / HIDDEN_DIM; if(!mean) mean=1;
            WeightType ln_w = model->ln_1[l][tid];
            int8_t ln_n = noise_from_hash(seed_base + SEED_OFF_LN_1, tid);
            ActType r_in = clip(((AccumType)s_x[tid] * (ln_w + (((AccumType)ln_n * ns) >> SIGMA_SHIFT_VECTOR))) / mean);
            s_norm[tid] = r_in; __syncthreads();

            // QKV Projection (Rank-1 Fast)
            AccumType sbq = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_Q_B, tid), temp_storage, shared_scalar);
            AccumType sbk = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_K_B, tid), temp_storage, shared_scalar);
            AccumType sbv = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_V_B, tid), temp_storage, shared_scalar);

            AccumType aq=0, ak=0, av=0;
            const WeightType *wq = &model->w_q[l][0];
            const WeightType *wk = &model->w_k[l][0];
            const WeightType *wv = &model->w_v[l][0];
            
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) {
                ActType v = s_norm[k];
                aq += (AccumType)v * wq[k*HIDDEN_DIM + tid];
                ak += (AccumType)v * wk[k*HIDDEN_DIM + tid];
                av += (AccumType)v * wv[k*HIDDEN_DIM + tid];
            }
            if(ns!=0) {
                aq += ((sbq * (AccumType)noise_from_hash(seed_base + SEED_OFF_Q_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                ak += ((sbk * (AccumType)noise_from_hash(seed_base + SEED_OFF_K_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                av += ((sbv * (AccumType)noise_from_hash(seed_base + SEED_OFF_V_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            }
            ActType qv = clip(aq>>8), kv = clip(ak>>8), vv = clip(av>>8);
            
            // Store KV
            ActType *lkv = global_kv_cache + kv_ind_offset + (l * kv_layer_stride);
            lkv[t*HIDDEN_DIM + tid] = kv;
            lkv[SEQ_LEN*HIDDEN_DIM + t*HIDDEN_DIM + tid] = vv;
            __syncthreads();

            // Attention (Per Head) - Two-Pass with atomicMax for stability
            int h = tid / HEAD_DIM;
            int32_t *s_attn_scores = (int32_t*)&s_mem[2*HIDDEN_DIM]; // Per-head scores accumulator
            int32_t *s_head_max = (int32_t*)&s_mem[2*HIDDEN_DIM + N_HEADS*4]; // Per-head max scores
            
            // Initialize per-head max to INT_MIN
            if(tid < N_HEADS) {
                s_attn_scores[tid] = 0;
                s_head_max[tid] = INT_MIN;
            }
            __syncthreads();

            // PASS 1: Compute all attention scores and find max per head using atomicMax
            for(int ctx=0; ctx <= t; ctx++) {
                ActType k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                AccumType df = (AccumType)qv * k_ctx;
                
                // Reduce across head (Head Size 64, Warp 32)
                for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                
                // Lane 0 of each warp has partial sum
                if ((tid % 32) == 0) atomicAdd(&s_attn_scores[h], (int32_t)df);
                __syncthreads();
                
                // Update max using atomicMax (CAS-based)
                if (tid < N_HEADS) {
                    atomicMax(&s_head_max[tid], s_attn_scores[tid]);
                    s_attn_scores[tid] = 0; // Reset for next context
                }
                __syncthreads();
            }
            
            // Now s_head_max[h] contains the max score for each head
            
            // PASS 2: Compute exp(score - max) and weighted sum
            AttnAccumType w_v_sum = 0;
            uint64_t tot_sc = 0;
            int32_t my_head_max = s_head_max[h];
            
            if(tid < N_HEADS) s_attn_scores[tid] = 0; // Reset for pass 2
            __syncthreads();
            
            for(int ctx=0; ctx <= t; ctx++) {
                ActType k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                AccumType df = (AccumType)qv * k_ctx;
                
                // Reduce across head
                for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                
                if ((tid % 32) == 0) atomicAdd(&s_attn_scores[h], (int32_t)df);
                __syncthreads();
                
                // Compute weight using new softmax LUT with max subtraction
                int32_t sc = s_attn_scores[h];
                int32_t shifted = (sc >> 3) - (my_head_max >> 3); // Normalize by max
                int32_t wt = softmax_exp_lookup(shifted);
                
                ActType v_ctx = lkv[SEQ_LEN*HIDDEN_DIM + ctx*HIDDEN_DIM + tid];
                w_v_sum += (AttnAccumType)wt * v_ctx;
                tot_sc += wt;
                
                __syncthreads();
                if(tid < N_HEADS) s_attn_scores[tid] = 0; // Reset
                __syncthreads();
            }
            if(tot_sc==0) tot_sc=1;
            // Scale down from 2^20 precision
            ActType ao = clip((w_v_sum / (int64_t)tot_sc));
            s_norm[tid] = ao; __syncthreads();

            // Output Proj
            AccumType sbo = block_reduce_sum_broadcast((AccumType)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid), temp_storage, shared_scalar);

            const WeightType *wo = &model->w_o[l][0];
            AccumType aco = 0;
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) aco += (AccumType)s_norm[k] * wo[k*HIDDEN_DIM + tid];
            if(ns!=0) aco += ((sbo * (AccumType)noise_from_hash(seed_base + SEED_OFF_O_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            s_x[tid] = clip((AccumType)s_x[tid] + (aco >> 8)); __syncthreads();

            // MLP Block
            // Norm 2
            AccumType mtot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mmn = mtot/HIDDEN_DIM; if(!mmn) mmn=1;
            WeightType l2w = model->ln_2[l][tid];
            int8_t l2n = noise_from_hash(seed_base + SEED_OFF_LN_2, tid);
            ActType n2x = clip(((AccumType)s_x[tid] * (l2w + (((AccumType)l2n*ns)>>SIGMA_SHIFT_VECTOR))) / mmn);
            s_norm[tid] = n2x; __syncthreads();

            // Expand
            AccumType sbup = block_reduce_sum_broadcast((AccumType)n2x * noise_from_hash(seed_base + SEED_OFF_MLP_UP_B, tid), temp_storage, shared_scalar);

            ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256]; 

            const WeightType *wup = &model->w_up[l][0];
            // Need to compute 4x output dims. Loop.
            for(int sub=0; sub<4; sub++) {
                int oidx = tid + sub*HIDDEN_DIM;
                AccumType aup = 0;
                #pragma unroll 8
                for(int k=0; k<HIDDEN_DIM; k++) aup += (AccumType)s_norm[k] * wup[k*(4*HIDDEN_DIM) + oidx];
                if(ns!=0) aup += ((sbup * (AccumType)noise_from_hash(seed_base + SEED_OFF_MLP_UP_A, oidx)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                ActType v = clip(aup>>8); if(v<0) v=0; // ReLU
                s_mlp[oidx] = v; 
            }
            __syncthreads();

            // Down Project
            // Input is s_mlp (4H).
            // Need scalar B projection of s_mlp.
            AccumType pbdn = 0;
            for(int sub=0; sub<4; sub++) pbdn += (AccumType)s_mlp[tid + sub*HIDDEN_DIM] * noise_from_hash(seed_base + SEED_OFF_MLP_DOWN_B, tid + sub*HIDDEN_DIM);
            AccumType sbdn = block_reduce_sum_broadcast(pbdn, temp_storage, shared_scalar);

            const WeightType *wdn = &model->w_down[l][0];
            AccumType adn = 0;
            #pragma unroll 8
            for(int k=0; k<(4*HIDDEN_DIM); k++) adn += (AccumType)s_mlp[k] * wdn[k*HIDDEN_DIM + tid];
            if(ns!=0) adn += ((sbdn * (AccumType)noise_from_hash(seed_base + SEED_OFF_MLP_DOWN_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            s_x[tid] = clip((AccumType)s_x[tid] + (adn >> 9)); __syncthreads();
        }

        // 3. Final Head
        AccumType ftot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType fmn = ftot/HIDDEN_DIM; if(!fmn) fmn=1;
        
        WeightType lfw = model->ln_f[tid];
        int8_t lfn = noise_from_hash(step_seed + pair_idx + SEED_OFF_LN_F, tid);
        ActType nf = clip(((AccumType)s_x[tid] * (lfw + (((AccumType)lfn*ns)>>SIGMA_SHIFT_VECTOR))) / fmn);
        
        // Reuse s_mem[HD] for normed
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        AccumType sbh = block_reduce_sum_broadcast((AccumType)nf * noise_from_hash(step_seed + pair_idx + SEED_OFF_HEAD + VOCAB_SIZE, tid), temp_storage, shared_scalar);

        // Compute Logits with new softmax (max-subtraction for stability)
        // Use shared memory for logits and max finding
        int32_t *s_logits = (int32_t*)&s_mem[2*HIDDEN_DIM];
        __shared__ int32_t s_logit_max;
        
        if(tid < VOCAB_SIZE) {
            AccumType ah = 0;
            const WeightType *wh = &model->head[0];
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) ah += (AccumType)s_norm[k] * wh[k*VOCAB_SIZE + tid];
            if(ns!=0) ah += ((sbh * (AccumType)noise_from_hash(step_seed + pair_idx + SEED_OFF_HEAD, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            int32_t lgt = ah >> 8;  // Keep as int32 for max comparison
            s_logits[tid] = lgt;
        }
        __syncthreads();
        
        // Find max logit using atomicMax (thread 0 initializes, all threads update)
        if (tid == 0) s_logit_max = INT_MIN;
        __syncthreads();
        if (tid < VOCAB_SIZE) atomicMax(&s_logit_max, s_logits[tid]);
        __syncthreads();
        
        // Compute exp(logit - max) using new LUT
        if(tid < VOCAB_SIZE) {
            int32_t shifted = s_logits[tid] - s_logit_max;
            int32_t ex = softmax_exp_lookup(shifted);
            ((int32_t*)s_x)[tid] = ex;
            ((int8_t*)s_x)[HIDDEN_DIM*4 + tid] = clip(s_logits[tid]); // Store clipped logit for loss
        }
        __syncthreads();

        // Loss Calc (Thread 0) - adjusted for new scale (2^20)
        if (tid == 0) {
            uint8_t target_token = dataset[stream_pos + t + 1];
            uint64_t sum_ex = 0;
            for(int i=0; i<VOCAB_SIZE; i++) sum_ex += ((int32_t*)s_x)[i];
            
            // Cross-entropy loss: -log(p_target) = -log(exp(x_t)/sum) = log(sum) - x_t
            // With new scale, sum is in 2^20 units
            // log2(sum) approximation for loss computation
            int64_t log_sum = 0;
            if (sum_ex > 0) {
                uint64_t x = sum_ex; int pos = 0;
                while(x >= 256) { x >>= 8; pos += 8; }
                if(x >= 16) { x >>= 4; pos += 4; }
                if(x >= 4) { x >>= 2; pos += 2; }
                if(x >= 2) { pos += 1; }
                
                // Scale factor: log2(2^20) = 20, so subtract 20 from pos for proper scaling
                log_sum = (pos << 4) - (20 << 4);
            }
            int8_t tgt_lgt = ((int8_t*)s_x)[HIDDEN_DIM*4 + target_token];
            int32_t tgt_val = (int32_t)tgt_lgt + 128;
            // Restore absolute scale: add max + 128 to the relative log_sum
            my_loss += (log_sum + s_logit_max + 128 - tgt_val);
        }
        __syncthreads(); 
    }

    if (tid == 0) accum_loss[p_idx] = (int32_t)my_loss;
}

__global__ void compute_fitness_kernel(const int32_t *accum_loss, int32_t *fitnesses, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t p = accum_loss[2*idx]; int32_t n = accum_loss[2*idx+1];
    fitnesses[idx] = (p < n) ? 1 : ((n < p) ? -1 : 0);
}

// ADAM W Implementation KERNELS

// update_matrix_adam_kernel
__global__ void update_matrix_adam_kernel(
    WeightType *W, 
    AdamParam *adam_state,
    int rows, int cols, 
    int off_A, int off_B, 
    int seed_base, 
    const int32_t *fitnesses, 
    uint32_t step_seed,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int change = 0;
    if (idx < rows * cols) {
        int r = idx / cols; 
        int c = idx % cols;
        
        // Approximate Gradient Computation
        VoteType vote = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
            int fit = fitnesses[p]; if(fit==0) continue;
            uint32_t s = step_seed + p + seed_base;
            vote += (VoteType)fit * noise_from_hash(s + off_A, c) * noise_from_hash(s + off_B, r);
        }
        
        // Gradient is negative of vote (since vote > 0 suggests moving in that direction improves fitness)
        // If fitness=1 (positive noise better), we want to go in +noise direction.
        // vote accumulates +noise. So vote is the step. 
        // Standard GD: w = w - lr * grad. 
        // Equiv to: w = w + lr * (-grad).
        // identifying "vote" as (-grad).
        
        float g = -(float)vote; 

        // Load Adam State
        AdamParam p = adam_state[idx];
        
        // Update Moments
        p.m = ADAM_BETA1 * p.m + (1.0f - ADAM_BETA1) * g;
        p.v = ADAM_BETA2 * p.v + (1.0f - ADAM_BETA2) * (g * g);
        
        // Bias Correction (Simplified: assume simplified or pre-bias-corrected for long steps)
        // For efficiency in chaotic integer training, often skipped or approximated, 
        // but let's do standard Bias Correction if we track steps, or just Raw Adam for simplicity 
        // given the massive noise. Let's stick to raw M/sqrt(V).
        
        float m_hat = p.m; // / (1 - beta1^t)
        float v_hat = p.v; // / (1 - beta2^t)
        
        // Update Step
        float step = - learning_rate * (m_hat / (sqrtf(v_hat) + ADAM_EPS) + ADAM_WEIGHT_DECAY * (float)W[idx]);
        
        // Accumulate
        p.acc += step;
        
        // Apply to Integer Weight
        int8_t w = W[idx];
        if (p.acc >= 1.0f) {
            if (w < MAX_VAL) { w++; change=1; p.acc -= 1.0f; }
            else p.acc = 1.0f; // Clamp accumulator
        } else if (p.acc <= -1.0f) {
             if (w > MIN_VAL) { w--; change=1; p.acc += 1.0f; }
             else p.acc = -1.0f; // Clamp accumulator
        }
        
        W[idx] = w;
        adam_state[idx] = p; // Write back state
    }

    if (change) atomicAdd(&s_count, 1);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) atomicAdd(&d_total_updates, (unsigned long long)s_count);
}

// update_vector_adam_kernel
__global__ void update_vector_adam_kernel(
    WeightType *V, 
    AdamParam *adam_state,
    int len, 
    int off_A, 
    int seed_base, 
    const int32_t *fitnesses, 
    uint32_t step_seed,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int change = 0;
    if (idx < len) {
        VoteType vote = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
            int fit = fitnesses[p]; if(fit==0) continue;
            vote += (VoteType)fit * noise_from_hash(step_seed + p + seed_base + off_A, idx);
        }
        
        float g = -(float)vote;
        AdamParam p = adam_state[idx];
        
        p.m = ADAM_BETA1 * p.m + (1.0f - ADAM_BETA1) * g;
        p.v = ADAM_BETA2 * p.v + (1.0f - ADAM_BETA2) * (g * g);
        
        float step = - learning_rate * (p.m / (sqrtf(p.v) + ADAM_EPS) + ADAM_WEIGHT_DECAY * (float)V[idx]);
        
        p.acc += step;
        
        int8_t v_val = V[idx];
        if (p.acc >= 1.0f) {
            if (v_val < MAX_VAL) { v_val++; change=1; p.acc -= 1.0f; }
            else p.acc = 1.0f;
        } else if (p.acc <= -1.0f) {
            if (v_val > MIN_VAL) { v_val--; change=1; p.acc += 1.0f; }
            else p.acc = -1.0f;
        }
        
        V[idx] = v_val;
        adam_state[idx] = p;
    }

    if (change) atomicAdd(&s_count, 1);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) atomicAdd(&d_total_updates, (unsigned long long)s_count);
}

__global__ void generate_sequence_kernel(
    uint8_t * buffer, int seed_len, int gen_len,
    const TransformerModel * __restrict__ model,
    ActType * __restrict__ kv_cache,
    uint32_t seed
) {
    if (blockIdx.x > 0) return;
    int tid = threadIdx.x;
    if (tid >= HIDDEN_DIM) return;

    extern __shared__ ActType s_mem[];
    ActType *s_x = s_mem; 
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ AccumType shared_scalar;
    __shared__ int32_t shared_logits[VOCAB_SIZE];

    int total_len = seed_len + gen_len;
    size_t kv_layer_stride = 2ULL * total_len * HIDDEN_DIM;

    for (int t = 0; t < total_len - 1; t++) { 
        __syncthreads(); 

        uint8_t input_token = buffer[t];
        
        // 1. Embedding
        WeightType emb = model->embedding[tid * VOCAB_SIZE + input_token];
        WeightType pos = model->pos_emb[(t % SEQ_LEN) * HIDDEN_DIM + tid]; 
        
        s_x[tid] = clip((AccumType)emb + pos);
        __syncthreads();

        // 2. Layers
        for (int l = 0; l < N_LAYERS; l++) {
            ActType *s_norm = &s_mem[HIDDEN_DIM];

            // LN 1
            AccumType total_sum = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mean = total_sum / HIDDEN_DIM; if(!mean) mean=1;
            WeightType ln_w = model->ln_1[l][tid];
            ActType r_in = clip(((AccumType)s_x[tid] * ln_w) / mean);
            s_norm[tid] = r_in; __syncthreads();

            // QKV
            const WeightType *wq = &model->w_q[l][0];
            const WeightType *wk = &model->w_k[l][0];
            const WeightType *wv = &model->w_v[l][0];
            
            AccumType aq=0, ak=0, av=0;
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) {
                ActType v = s_norm[k];
                aq += (AccumType)v * wq[k*HIDDEN_DIM + tid];
                ak += (AccumType)v * wk[k*HIDDEN_DIM + tid];
                av += (AccumType)v * wv[k*HIDDEN_DIM + tid];
            }
            ActType qv = clip(aq>>8), kv = clip(ak>>8), vv = clip(av>>8);

            // Store KV
            ActType *lkv = kv_cache + (l * kv_layer_stride);
            lkv[t*HIDDEN_DIM + tid] = kv;
            lkv[total_len*HIDDEN_DIM + t*HIDDEN_DIM + tid] = vv;
            __syncthreads();

            // Attention - Two-Pass with max-subtraction for stability
            int h = tid / HEAD_DIM;
            int32_t *s_attn_scores = (int32_t*)&s_mem[2*HIDDEN_DIM];
            int32_t *s_head_max = (int32_t*)&s_mem[2*HIDDEN_DIM + N_HEADS*4];
            
            if(tid < N_HEADS) {
                s_attn_scores[tid] = 0;
                s_head_max[tid] = INT_MIN;
            }
            __syncthreads();

            // PASS 1: Find max score per head
            for(int ctx=0; ctx <= t; ctx++) {
                ActType k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                AccumType df = (AccumType)qv * k_ctx;
                for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                if ((tid % 32) == 0) atomicAdd(&s_attn_scores[h], (int32_t)df);
                __syncthreads();
                
                if (tid < N_HEADS) {
                    atomicMax(&s_head_max[tid], s_attn_scores[tid]);
                    s_attn_scores[tid] = 0;
                }
                __syncthreads();
            }
            
            // PASS 2: Compute weighted sum with new softmax
            AttnAccumType w_v_sum = 0;
            uint64_t tot_sc = 0;
            int32_t my_head_max = s_head_max[h];
            
            if(tid < N_HEADS) s_attn_scores[tid] = 0;
            __syncthreads();

            for(int ctx=0; ctx <= t; ctx++) {
                ActType k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                AccumType df = (AccumType)qv * k_ctx;
                for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                if ((tid % 32) == 0) atomicAdd(&s_attn_scores[h], (int32_t)df);
                __syncthreads();

                int32_t sc = s_attn_scores[h];
                int32_t shifted = (sc >> 3) - (my_head_max >> 3);
                int32_t wt = softmax_exp_lookup(shifted);

                ActType v_ctx = lkv[total_len*HIDDEN_DIM + ctx*HIDDEN_DIM + tid];
                w_v_sum += (AttnAccumType)wt * v_ctx;
                tot_sc += wt;
                
                __syncthreads();
                if(tid < N_HEADS) s_attn_scores[tid] = 0;
                __syncthreads();
            }
            if(tot_sc==0) tot_sc=1;
            ActType ao = clip(w_v_sum / (int64_t)tot_sc);
            s_norm[tid] = ao; __syncthreads();

            // Output
            const WeightType *wo = &model->w_o[l][0];
            AccumType aco = 0;
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) aco += (AccumType)s_norm[k] * wo[k*HIDDEN_DIM + tid];
            s_x[tid] = clip((AccumType)s_x[tid] + (aco >> 8)); __syncthreads();

            // MLP
            AccumType mtot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mmn = mtot/HIDDEN_DIM; if(!mmn) mmn=1;
            WeightType l2w = model->ln_2[l][tid];
            ActType n2x = clip(((AccumType)s_x[tid] * l2w) / mmn);
            s_norm[tid] = n2x; __syncthreads();

            ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256];
            const WeightType *wup = &model->w_up[l][0];
            for(int sub=0; sub<4; sub++) {
                int oidx = tid + sub*HIDDEN_DIM;
                AccumType aup = 0;
                #pragma unroll 8
                for(int k=0; k<HIDDEN_DIM; k++) aup += (AccumType)s_norm[k] * wup[k*(4*HIDDEN_DIM) + oidx];
                ActType v = clip(aup>>8); if(v<0) v=0;
                s_mlp[oidx] = v;
            }
            __syncthreads();

            const WeightType *wdn = &model->w_down[l][0];
            AccumType adn = 0;
            #pragma unroll 8
            for(int k=0; k<(4*HIDDEN_DIM); k++) adn += (AccumType)s_mlp[k] * wdn[k*HIDDEN_DIM + tid];
            s_x[tid] = clip((AccumType)s_x[tid] + (adn >> 9)); __syncthreads();
        }

        // Final Head
        AccumType ftot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType fmn = ftot/HIDDEN_DIM; if(!fmn) fmn=1;
        WeightType lfw = model->ln_f[tid];
        ActType nf = clip(((AccumType)s_x[tid] * lfw) / fmn);
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        if(tid < VOCAB_SIZE) {
            AccumType ah = 0;
            const WeightType *wh = &model->head[0];
            #pragma unroll 8
            for(int k=0; k<HIDDEN_DIM; k++) ah += (AccumType)s_norm[k] * wh[k*VOCAB_SIZE + tid];
            shared_logits[tid] = (int32_t)d_EXP2_TABLE[(int32_t)clip(ah>>8) + 128];
        }
        __syncthreads();

        // Sample (Thread 0)
        if (t >= seed_len - 1) {
            if (tid == 0) {
                long long sum_exp = 0;
                for(int i=0; i<VOCAB_SIZE; i++) sum_exp += shared_logits[i];
                
                uint32_t s = seed + t * 555;
                uint32_t r = hash_rng(s, 0);
                long long thresh = (sum_exp > 0) ? (r % sum_exp) : 0;
                long long running = 0;
                int selected = 0;
                for(int i=0; i<VOCAB_SIZE; i++) {
                    running += shared_logits[i];
                    if(running > thresh) { selected = i; break; }
                }
                buffer[t + 1] = (uint8_t)selected;
            }
        }
        __syncthreads();
    }
}

void ensure_models_dir() {
    struct stat st = {0};
    if (stat("models", &st) == -1) {
        mkdir("models", 0700);
    }
}

void update_last_link(const char *target_filename, const char *link_name) {
    char link_path[256];
    sprintf(link_path, "models/%s", link_name);
    unlink(link_path);
    symlink(target_filename, link_path);
}

void save_model_info() {
    FILE *f = fopen("models/model.info", "w");
    if(!f) return;
    
    long params = sizeof(TransformerModel);
    long adam_size = sizeof(AdamModel);
    
    fprintf(f, "Model Configuration:\n");
    fprintf(f, "HIDDEN_DIM: %d\n", HIDDEN_DIM);
    fprintf(f, "HEAD_DIM: %d\n", HEAD_DIM);
    fprintf(f, "N_LAYERS: %d\n", N_LAYERS);
    fprintf(f, "SEQ_LEN: %d\n", SEQ_LEN);
    fprintf(f, "VOCAB_SIZE: %d\n", VOCAB_SIZE);
    fprintf(f, "N_HEADS: %d\n", N_HEADS);
    fprintf(f, "Model Parameters: %ld (%.2f MB)\n", params, params/(1024.0*1024.0));
    fprintf(f, "Adam State Size: %ld (%.2f MB)\n", adam_size, adam_size/(1024.0*1024.0));
    fprintf(f, "Population: %d\n", POPULATION_SIZE);
    fprintf(f, "Softmax Scale: 2^%d\n", SOFTMAX_SCALE_BIT);
    
    printf("\n--- Model Architecture ---\n");
    printf("Parameters: %ld (%.2f MB)\n", params, params/(1024.0*1024.0));
    printf("Optimizer State: %.2f MB\n", adam_size/(1024.0*1024.0));
    printf("Layers: %d, Hidden: %d, Heads: %d\n", N_LAYERS, HIDDEN_DIM, N_HEADS);
    printf("Sequence Length: %d, Vocab: %d\n", SEQ_LEN, VOCAB_SIZE);
    printf("--------------------------\n");
    
    fclose(f);
}

void save_checkpoint(TransformerModel *model, AdamModel *adam, long step) {
    char name_model[128], name_adam[128];
    sprintf(name_model, "egg_step-%08ld.model.bin", step);
    sprintf(name_adam, "egg_step-%08ld.adam.bin", step);
    
    char path_model[256], path_adam[256];
    sprintf(path_model, "models/%s", name_model);
    sprintf(path_adam, "models/%s", name_adam);
    
    FILE *f = fopen(path_model, "wb");
    if (f) {
        fwrite(model, sizeof(TransformerModel), 1, f);
        fclose(f);
        update_last_link(name_model, "egg_transformer_last.model.bin");
    }
    
    f = fopen(path_adam, "wb");
    if (f) {
        fwrite(adam, sizeof(AdamModel), 1, f);
        fclose(f);
        update_last_link(name_adam, "egg_transformer_last.adam.bin");
    }
}

int load_model(const char *filename, TransformerModel *model) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    printf("Loading model from %s...\n", filename);
    if (fread(model, sizeof(TransformerModel), 1, f) != 1) {
        printf("Error reading model from %s\n", filename);
        fclose(f);
        return 0;
    }
    fclose(f);
    printf("Model loaded successfully.\n");
    return 1;
}

int main() {
    signal(SIGINT, handle_sigint);
    init_tables();
    cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t));
    cudaMemcpyToSymbol(d_EXP_LUT, h_EXP_LUT, SOFTMAX_LUT_SIZE*sizeof(int32_t));

    printf("\n=== EGG TRANSFORMER ADAM ===\n");
    printf("AdamW Config: B1=%.3f B2=%.3f EPS=%.1e WD=%.4f (Dynamic LR)\n", 
           ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY);

    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("No input.txt\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET);
    ds.data=(uint8_t*)malloc(ds.length); fread(ds.data,1,ds.length,f); fclose(f);

    TransformerModel *h_model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    AdamModel *h_adam_state = (AdamModel*)calloc(1, sizeof(AdamModel));
    
    ensure_models_dir();
    save_model_info();
    
    if (load_model("models/egg_transformer_last.model.bin", h_model)) {
        printf("Resumed from models/egg_transformer_last.model.bin\n");
        // Try load adam
        FILE *fa = fopen("models/egg_transformer_last.adam.bin", "rb");
        if(fa) {
            fread(h_adam_state, sizeof(AdamModel), 1, fa);
            fclose(fa);
            printf("Resumed Adam state.\n");
        }
    } else if (load_model("models/egg_transformer_last.bin", h_model)) {
        // Fallback for legacy file
        printf("Resumed from legacy models/egg_transformer_last.bin\n");
    } else {
        init_model(h_model);
    }
    
    TransformerModel *d_model; CHECK_CUDA(cudaMalloc(&d_model, sizeof(TransformerModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(TransformerModel), cudaMemcpyHostToDevice));

    // Allocate Adam State
    AdamModel *d_adam_state; 
    CHECK_CUDA(cudaMalloc(&d_adam_state, sizeof(AdamModel)));
    CHECK_CUDA(cudaMemcpy(d_adam_state, h_adam_state, sizeof(AdamModel), cudaMemcpyHostToDevice));
    // Note: if fresh start, h_adam_state is 0-calloc'd, so this is equiv to Memset 0

    uint8_t *d_dataset; CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));

    int32_t *d_loss, *d_fit;
    CHECK_CUDA(cudaMalloc(&d_loss, POPULATION_SIZE * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_fit, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    size_t kv_size = (size_t)POPULATION_BATCH_SIZE * N_LAYERS * 2 * SEQ_LEN * HIDDEN_DIM;
    printf("Allocating KV Cache: %.2f GB\n", kv_size / (1024.0*1024*1024));
    CHECK_CUDA(cudaMalloc(&d_kv_cache, kv_size));

    unsigned long long *d_updates_ptr;
    CHECK_CUDA(cudaGetSymbolAddress((void**)&d_updates_ptr, d_total_updates));
    
    // Generation buffers
    int gen_seed_len = 32;
    int gen_output_len = 64; 
    int total_gen_len = gen_seed_len + gen_output_len;
    uint8_t *d_gen_buf; CHECK_CUDA(cudaMalloc(&d_gen_buf, total_gen_len));
    int8_t *d_gen_kv; CHECK_CUDA(cudaMalloc(&d_gen_kv, N_LAYERS * 2 * total_gen_len * HIDDEN_DIM));

    printf("Starting Training...\n");
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step<max_steps && keep_running; step++) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x12345678);
        
        size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM); 
        for (int offset = 0; offset < POPULATION_SIZE; offset += POPULATION_BATCH_SIZE) {
            train_sequence_kernel<<<POPULATION_BATCH_SIZE, BLOCK_THREADS, sm_size>>>(
                d_dataset, ds.length, step*SEQ_LEN, d_model, d_kv_cache, d_loss, seed, offset
            );
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        compute_fitness_kernel<<< (POPULATION_SIZE/2 + 255)/256, 256 >>>(d_loss, d_fit, POPULATION_SIZE/2);
        
        // Reset update counter
        CHECK_CUDA(cudaMemset(d_updates_ptr, 0, sizeof(unsigned long long)));

        // Adam Updates
        float current_lr = get_learning_rate(step);
        for(int l=0; l<N_LAYERS; l++) {
            int s_base = l * 1000;
            int d2 = HIDDEN_DIM*HIDDEN_DIM;
            
            update_matrix_adam_kernel<<< (d2+511)/512, 512 >>>( (WeightType*)d_model->w_q[l], (AdamParam*)d_adam_state->w_q[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_Q_A, SEED_OFF_Q_B, s_base, d_fit, seed, current_lr);
            update_matrix_adam_kernel<<< (d2+511)/512, 512 >>>( (WeightType*)d_model->w_k[l], (AdamParam*)d_adam_state->w_k[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_K_A, SEED_OFF_K_B, s_base, d_fit, seed, current_lr);
            update_matrix_adam_kernel<<< (d2+511)/512, 512 >>>( (WeightType*)d_model->w_v[l], (AdamParam*)d_adam_state->w_v[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_V_A, SEED_OFF_V_B, s_base, d_fit, seed, current_lr);
            update_matrix_adam_kernel<<< (d2+511)/512, 512 >>>( (WeightType*)d_model->w_o[l], (AdamParam*)d_adam_state->w_o[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_O_A, SEED_OFF_O_B, s_base, d_fit, seed, current_lr);
            
            update_matrix_adam_kernel<<< (HIDDEN_DIM*4*HIDDEN_DIM+511)/512, 512 >>>( (WeightType*)d_model->w_up[l], (AdamParam*)d_adam_state->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, s_base, d_fit, seed, current_lr);
            update_matrix_adam_kernel<<< (4*HIDDEN_DIM*HIDDEN_DIM+511)/512, 512 >>>( (WeightType*)d_model->w_down[l], (AdamParam*)d_adam_state->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, s_base, d_fit, seed, current_lr);
            
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_1[l], (AdamParam*)d_adam_state->ln_1[l], HIDDEN_DIM, SEED_OFF_LN_1, s_base, d_fit, seed, current_lr);
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_2[l], (AdamParam*)d_adam_state->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2, s_base, d_fit, seed, current_lr);
        }
        
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_f, (AdamParam*)d_adam_state->ln_f, HIDDEN_DIM, SEED_OFF_LN_F, 0, d_fit, seed, current_lr);
        update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512 >>>((WeightType*)d_model->head, (AdamParam*)d_adam_state->head, HIDDEN_DIM, VOCAB_SIZE, SEED_OFF_HEAD, SEED_OFF_HEAD+VOCAB_SIZE, 0, d_fit, seed, current_lr);
        update_matrix_adam_kernel<<< (VOCAB_SIZE*HIDDEN_DIM+511)/512, 512 >>>((WeightType*)d_model->embedding, (AdamParam*)d_adam_state->embedding, HIDDEN_DIM, VOCAB_SIZE, SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 0, d_fit, seed, current_lr);
        update_matrix_adam_kernel<<< (SEQ_LEN*HIDDEN_DIM+511)/512, 512 >>>((WeightType*)d_model->pos_emb, (AdamParam*)d_adam_state->pos_emb, SEQ_LEN, HIDDEN_DIM, SEED_OFF_POS+SEQ_LEN, SEED_OFF_POS, 0, d_fit, seed, current_lr);

        CHECK_CUDA(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        unsigned long long h_updates = 0;
        CHECK_CUDA(cudaMemcpy(&h_updates, d_updates_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        double step_ms = get_time_diff_ms(t0, t1);
        double tokens_per_sec = (double)(POPULATION_SIZE * SEQ_LEN) / (step_ms / 1000.0);

        thrust::device_ptr<int32_t> t_loss(d_loss);
        long long total_loss = thrust::reduce(t_loss, t_loss + POPULATION_SIZE, (long long)0);
        double avg_loss = (double)total_loss / (POPULATION_SIZE * SEQ_LEN * 16.0); 

        printf("Step %ld | Loss: %.4f | Time: %.2f ms | Updates: %llu | Speed: %.2f tok/s | LR: %.3f\n", 
            step, avg_loss, step_ms, h_updates, tokens_per_sec, current_lr);

        if (step % 5 == 0) {
            CHECK_CUDA(cudaMemcpy(d_gen_buf, d_dataset + (step*SEQ_LEN) % (ds.length-SEQ_LEN), gen_seed_len, cudaMemcpyDeviceToDevice));
            generate_sequence_kernel<<<1, BLOCK_THREADS, sm_size>>>(
                d_gen_buf, gen_seed_len, gen_output_len, d_model, d_gen_kv, seed+999
            );
            CHECK_CUDA(cudaDeviceSynchronize());
            uint8_t h_buf[256];
            CHECK_CUDA(cudaMemcpy(h_buf, d_gen_buf, total_gen_len, cudaMemcpyDeviceToHost));
            
            printf("\n--- GENERATION ---\n");
            printf("\033[32m"); 
            for(int i=0; i<gen_seed_len; i++) {
                char c = h_buf[i]; printf("%c", (c>=32 && c<=126) ? c : '.');
            }
            printf("\033[36m"); 
            for(int i=gen_seed_len; i<total_gen_len; i++) {
                char c = h_buf[i]; printf("%c", (c>=32 && c<=126) ? c : '.');
            }
            printf("\033[0m\n\n");

            CHECK_CUDA(cudaMemcpy(h_model, d_model, sizeof(TransformerModel), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_adam_state, d_adam_state, sizeof(AdamModel), cudaMemcpyDeviceToHost));
            save_checkpoint(h_model, h_adam_state, step);
        }
    }
    cudaFree(d_gen_buf); cudaFree(d_gen_kv);

    free(h_model); free(h_adam_state); free(ds.data);
    cudaFree(d_model); cudaFree(d_dataset); cudaFree(d_loss); cudaFree(d_fit); cudaFree(d_kv_cache);
    cudaFree(d_adam_state);
    return 0;
}
