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

#include "egg_debug_printer.h"
#include "egg_adaptive_normalize.h"

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

// --- CONFIGURATION ---
#define HIDDEN_DIM 512
#define HEAD_DIM 64
#define N_LAYERS 4
#define SEQ_LEN 128     
#define VOCAB_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCK_THREADS 1024 
#define ALIGNED_DIM ((HIDDEN_DIM + 31) & ~31)
#define BLOCK_THREADS (ALIGNED_DIM > MAX_BLOCK_THREADS ? MAX_BLOCK_THREADS : ALIGNED_DIM)
#define N_HEADS (HIDDEN_DIM / HEAD_DIM)

#define POPULATION_BATCH_SIZE (8192 * 5)
#define POPULATION_SIZE (POPULATION_BATCH_SIZE * 1)

#define FIXED_POINT 4
#define SIGMA_SHIFT 3
#define SIGMA_SHIFT_VECTOR 2
#define MAX_VAL 127
#define MIN_VAL -127

#define SOFTMAX_SCALE_BIT 18
#define SOFTMAX_SCALE (1 << SOFTMAX_SCALE_BIT)  
#define SOFTMAX_LUT_SIZE 256
#define SOFTMAX_EXP_SCALE 8.0 // was 11.54 

// RoPE Configuration
#define ROPE_SCALE_BIT 30
#define ROPE_SCALE (1 << ROPE_SCALE_BIT)
#define ROPE_LUT_SIZE (SEQ_LEN * (HEAD_DIM / 2) * 2)


#define SEED_OFF_EMB 0
// SEED_OFF_POS removed (RoPE)
#define SEED_OFF_EMB_BIAS 50
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
#define SEED_OFF_MLP_BIAS_UP 410
#define SEED_OFF_MLP_BIAS_DOWN 411
#define SEED_OFF_LN_F 900
#define SEED_OFF_LN_1_BIAS 110
#define SEED_OFF_LN_2_BIAS 310
#define SEED_OFF_LN_F_BIAS 910
#define SEED_OFF_HEAD 999

#define SEED_OFF_LN_INIT 500
#define SEED_OFF_LN_INIT_BIAS 501
#define SEED_OFF_EMB_MLP_UP_A 502
#define SEED_OFF_EMB_MLP_UP_B 503
#define SEED_OFF_EMB_MLP_DOWN_A 504
#define SEED_OFF_EMB_MLP_DOWN_B 505
#define SEED_OFF_EMB_MLP_BIAS_UP 506
#define SEED_OFF_EMB_MLP_BIAS_DOWN 507

#define SHIFT_ATTN 8
#define SHIFT_QKV 6
#define SHIFT_OUT 9
#define SHIFT_LOGIT 8
#define SHIFT_MLP_UP 8
#define SHIFT_MLP_DOWN 10

// ADAM HYPERPARAMS
float get_learning_rate(long step) {
    if (step > 1000) {
        return 0.01f;
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
    WeightType emb_bias[HIDDEN_DIM];
    // RoPE: pos_emb removed
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

// Adam State Structures
typedef struct {
    float m;
    float v;
    float acc; // Fractional accumulator for integer weights
} AdamParam;

typedef struct {
    AdamParam embedding[VOCAB_SIZE * HIDDEN_DIM];
    AdamParam emb_bias[HIDDEN_DIM];
    // RoPE: pos_emb removed
    
    // Initial MLP Layer
    AdamParam ln_init[HIDDEN_DIM];
    AdamParam ln_init_bias[HIDDEN_DIM];
    AdamParam w_emb_mlp_up[HIDDEN_DIM * (HIDDEN_DIM * 4)];
    AdamParam mlp_emb_bias_up[HIDDEN_DIM * 4];
    AdamParam w_emb_mlp_down[(HIDDEN_DIM * 4) * HIDDEN_DIM];
    AdamParam mlp_emb_bias_down[HIDDEN_DIM];

    AdamParam ln_1[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_1_bias[N_LAYERS][HIDDEN_DIM];
    AdamParam w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    AdamParam ln_2[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_2_bias[N_LAYERS][HIDDEN_DIM];
    AdamParam w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    AdamParam mlp_bias_up[N_LAYERS][HIDDEN_DIM * 4];
    AdamParam w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    AdamParam mlp_bias_down[N_LAYERS][HIDDEN_DIM];
    AdamParam ln_f[HIDDEN_DIM];
    AdamParam ln_f_bias[HIDDEN_DIM];
} AdamModel;

ActType *d_kv_cache = NULL;

// New softmax LUT: exp(-i) * 2^20 for i in [0, 14]
__constant__ int32_t d_EXP_LUT[SOFTMAX_LUT_SIZE];
int32_t h_EXP_LUT[SOFTMAX_LUT_SIZE];

// Activation LUT (GELU)
__constant__ int8_t d_ACT_LUT[256];
int8_t h_ACT_LUT[256];

// RoPE Look-Up Table: [SEQ_LEN][HEAD_DIM/2][2 (cos, sin)]
__constant__ int32_t d_ROPE_LUT[ROPE_LUT_SIZE];
int32_t h_ROPE_LUT[ROPE_LUT_SIZE];

__device__ int32_t d_debug_updates[2];
__device__ unsigned long long d_total_updates;

// --- HOST HELPER ---
double get_time_diff_ms(struct timespec start, struct timespec end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1e6);
}

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
            
            // Scale by ROPE_SCALE (2^16)
            int32_t c_int = (int32_t)round(c * ROPE_SCALE);
            int32_t s_int = (int32_t)round(s * ROPE_SCALE);
            
            int base_idx = t * (HEAD_DIM) + i * 2;
            h_ROPE_LUT[base_idx] = c_int;
            h_ROPE_LUT[base_idx + 1] = s_int;
        }
    }
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *state = x; return x;
}
static inline int8_t gen_noise_host(uint32_t *rng) { return (int8_t)((xorshift32_host(rng) & 1 ? 1 : -1) * ((xorshift32_host(rng) >> 1) & 15)); }

void repack_matrix(int8_t *dst, int8_t *src, int rows, int cols) {  
    int32_t *d32 = (int32_t*)dst;
    
    for(int k=0; k<rows; k+=4) {
        for(int tid=0; tid<cols; tid++) {
            // Pack 4 bytes
            uint32_t val = 0;
            for(int s=0; s<4; s++) {
                int8_t w = src[(k+s)*cols + tid]; // Src is [In][Out]
                val |= ((uint8_t)w) << (s*8);
            }
            // Store at chunk index
            int chunk_idx = (k/4) * cols + tid;
            d32[chunk_idx] = val;
        }
    }
}

void init_model(TransformerModel *model) {
    uint32_t rng = 42;
    TransformerModel *temp = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    if(!temp) exit(1);
    
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);

    repack_matrix(model->embedding, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
    
    for(int i=0; i<HIDDEN_DIM; i++) model->emb_bias[i] = 0;
    
    // Init Initial MLP
    for(int i=0; i<HIDDEN_DIM; i++) { model->ln_init[i]=16; model->ln_init_bias[i]=0; }
    // Note: Reusing w_up sized buffer from temp for host init if needed, or just casting
    // Safer: use the large w_up buffer in temp since size is same [HIDDEN_DIM * (4*HIDDEN_DIM)]
    for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_up[0][i] = gen_noise_host(&rng); 
    repack_matrix(model->w_emb_mlp_up, temp->w_up[0], HIDDEN_DIM, 4*HIDDEN_DIM);
    for(int i=0; i<4*HIDDEN_DIM; i++) model->mlp_emb_bias_up[i] = 0;

    for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_down[0][i] = gen_noise_host(&rng); 
    repack_matrix(model->w_emb_mlp_down, temp->w_down[0], 4*HIDDEN_DIM, HIDDEN_DIM);
    for(int i=0; i<HIDDEN_DIM; i++) model->mlp_emb_bias_down[i] = 0;


    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) { 
            model->ln_1[l][i]=16; model->ln_1_bias[l][i]=0; 
            model->ln_2[l][i]=16; model->ln_2_bias[l][i]=0; 
        }
        int d2 = HIDDEN_DIM*HIDDEN_DIM;
        // For standard weights: Input=HIDDEN, Output=HIDDEN.
        for(int i=0; i<d2; i++) temp->w_q[l][i] = gen_noise_host(&rng); repack_matrix(model->w_q[l], temp->w_q[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_k[l][i] = gen_noise_host(&rng); repack_matrix(model->w_k[l], temp->w_k[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_v[l][i] = gen_noise_host(&rng); repack_matrix(model->w_v[l], temp->w_v[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_o[l][i] = gen_noise_host(&rng); repack_matrix(model->w_o[l], temp->w_o[l], HIDDEN_DIM, HIDDEN_DIM);
        
        // MLP Up: In=HIDDEN, Out=4*HIDDEN.
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_up[l][i] = gen_noise_host(&rng); repack_matrix(model->w_up[l], temp->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM);
        for(int i=0; i<4*HIDDEN_DIM; i++) model->mlp_bias_up[l][i] = 0;
        
        // MLP Down: In=4*HIDDEN, Out=HIDDEN.
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_down[l][i] = gen_noise_host(&rng); repack_matrix(model->w_down[l], temp->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<HIDDEN_DIM; i++) model->mlp_bias_down[l][i] = 0;
    }
    for(int i=0; i<HIDDEN_DIM; i++) { model->ln_f[i]=16; model->ln_f_bias[i]=0; }
    free(temp);
}

// --- DEVICE KERNELS & HELPERS ---

typedef cub::BlockReduce<AccumType, BLOCK_THREADS> BlockReduce;

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16; return x;
}
__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx); return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 15));
}
__device__ __forceinline__ int8_t clip(AccumType a) { return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a); }

__device__ __forceinline__ int32_t softmax_exp_lookup(int32_t diff) {
    int index = -diff;  // diff is negative or zero, so index is positive
    index = (index < 0) ? 0 : ((index > 255) ? 255 : index);
    return d_EXP_LUT[index];
}

__device__ __forceinline__ int simd_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
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

// Helper: Sum reduction + Broadcast
__device__ __forceinline__ AccumType block_reduce_sum_broadcast(AccumType val, BlockReduce::TempStorage &storage, AccumType &shared_var) {
    AccumType total = BlockReduce(storage).Sum(val);
    if (threadIdx.x == 0) shared_var = total;
    __syncthreads();
    AccumType ret = shared_var;
    __syncthreads();
    return ret;
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

__global__ void __launch_bounds__(MAX_BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset, long data_len, int start_idx,
    const TransformerModel * __restrict__ model,
    ActType * __restrict__ global_kv_cache,
    int32_t *accum_loss, uint32_t step_seed,
    int global_pop_offset, long step
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
        
        if (step%5 == 0 && global_pop_offset == 0 && blockIdx.x == 0 && tid == 0 && t == 0) {
             printf("DEBUG: Dataset[0..10] at stream_pos=%ld: ", stream_pos);
             for(int i=0; i<10 && stream_pos+i < data_len; i++) printf("%d(%c) ", dataset[stream_pos+i], (dataset[stream_pos+i]>=32 && dataset[stream_pos+i]<=126) ? dataset[stream_pos+i] : '.');
             printf("\n");
        }

        // 1. Embedding
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        WeightType emb = get_embedding_byte(model->embedding, tid, input_token);
        WeightType ebias = model->emb_bias[tid];
        int8_t emb_bias_n = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_BIAS, tid);
        
        // RoPE: Absolute pos emb removed
        int8_t a_tok = noise_from_hash(seed_emb, input_token);
        int8_t b_dim = noise_from_hash(seed_emb + HIDDEN_DIM, tid);
        AccumType perturb = ((AccumType)a_tok * b_dim * ns) >> (FIXED_POINT + SIGMA_SHIFT);

        s_x[tid] = clip((AccumType)emb + ebias + (((AccumType)emb_bias_n * ns) >> SIGMA_SHIFT_VECTOR) + perturb);
        __syncthreads();
        EGG_TRACE_STAT(step, t, -1, "Emb", s_x, HIDDEN_DIM);

        // --- INITIAL MLP BLOCK ---
        // Norm Init (With Bias)
        AccumType mitot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType mimn = mitot/HIDDEN_DIM; if(!mimn) mimn=1;
        WeightType liw = model->ln_init[tid];
        int8_t lin = noise_from_hash((step_seed + pair_idx) + SEED_OFF_LN_INIT, tid);
        WeightType lib = model->ln_init_bias[tid];
        int8_t libn = noise_from_hash((step_seed + pair_idx) + SEED_OFF_LN_INIT_BIAS, tid);
        ActType nix = clip(((AccumType)s_x[tid] * (liw + (((AccumType)lin*ns)>>SIGMA_SHIFT_VECTOR))) / mimn
                         + lib + (((AccumType)libn * ns) >> SIGMA_SHIFT_VECTOR)); 
        // Reuse s_norm buffer (s_mem[HIDDEN_DIM])
        ActType *s_norm_init = &s_mem[HIDDEN_DIM];
        s_norm_init[tid] = nix; __syncthreads();
        EGG_TRACE_STAT(step, t, -1, "LN_Init", s_norm_init, HIDDEN_DIM);

        // Expand
        AccumType sbiup = block_reduce_sum_broadcast((AccumType)nix * noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_MLP_UP_B, tid), temp_storage, shared_scalar);
        
        // Reuse large scratchpad area for MLP acts
        ActType *s_mlp_init = &s_mem[2*HIDDEN_DIM + 256]; 

        const int32_t *wiup_p = (const int32_t*)model->w_emb_mlp_up;
        int32_t *v_ptr_iup = (int32_t*)s_norm_init;
        
        for(int sub=0; sub<4; sub++) {
            int oidx = tid + sub*HIDDEN_DIM;
            AccumType aup = compute_linear_projection(v_ptr_iup, wiup_p, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, sbiup, (step_seed + pair_idx) + SEED_OFF_EMB_MLP_UP_A, ns);
            
            WeightType b_up = model->mlp_emb_bias_up[oidx];
            int8_t n_b_up = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_MLP_BIAS_UP, oidx);
            
            ActType raw = clip((aup>>SHIFT_MLP_UP) + b_up + (((AccumType)n_b_up * ns) >> SIGMA_SHIFT_VECTOR)); 
            s_mlp_init[oidx] = d_ACT_LUT[(uint8_t)raw]; // GELU
        }
        __syncthreads();
        EGG_TRACE_STAT(step, t, -1, "InitMLP_Exp", s_mlp_init, 4*HIDDEN_DIM);

        AccumType pbidn = 0;
        for(int sub=0; sub<4; sub++) pbidn += (AccumType)s_mlp_init[tid + sub*HIDDEN_DIM] * noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_MLP_DOWN_B, tid + sub*HIDDEN_DIM);
        AccumType sbidn = block_reduce_sum_broadcast(pbidn, temp_storage, shared_scalar);

        const int32_t *widn_p = (const int32_t*)model->w_emb_mlp_down;
        int32_t *v_ptr_idn = (int32_t*)s_mlp_init;
        AccumType aidn = compute_linear_projection(v_ptr_idn, widn_p, HIDDEN_DIM, HIDDEN_DIM, tid, sbidn, (step_seed + pair_idx) + SEED_OFF_EMB_MLP_DOWN_A, ns);

        // Add MLP Bias Down
        WeightType b_idn = model->mlp_emb_bias_down[tid];
        int8_t n_b_idn = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_MLP_BIAS_DOWN, tid);
        
        // Adaptive Layer Norm for Initial MLP Residual
        // Reuse scratch/warp maxs buffer which is free here
        int32_t *warp_maxs_scratch_init = (int32_t*)&s_mem[2*HIDDEN_DIM];
        AccumType imlp_res = (AccumType)s_x[tid] + (aidn >> SHIFT_MLP_DOWN) + b_idn + (((AccumType)n_b_idn * ns) >> SIGMA_SHIFT_VECTOR);
        s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(imlp_res, tid, warp_maxs_scratch_init); __syncthreads();
        
        EGG_TRACE_STAT(step, t, -1, "InitMLP", s_x, HIDDEN_DIM);
        // ---------------------

        // 2. Stack
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed_base = (step_seed + pair_idx) + (l * 1000);
            ActType *s_norm = &s_mem[HIDDEN_DIM]; // Normalized buf

            // LN 1 (With Bias)
            AccumType total_sum = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mean = total_sum / HIDDEN_DIM; if(!mean) mean=1;
            WeightType ln_w = model->ln_1[l][tid];
            int8_t ln_n = noise_from_hash(seed_base + SEED_OFF_LN_1, tid);
            WeightType ln_b = model->ln_1_bias[l][tid];
            int8_t ln_bn = noise_from_hash(seed_base + SEED_OFF_LN_1_BIAS, tid);
            ActType r_in = clip(((AccumType)s_x[tid] * (ln_w + (((AccumType)ln_n * ns) >> SIGMA_SHIFT_VECTOR))) / mean 
                              + ln_b + (((AccumType)ln_bn * ns) >> SIGMA_SHIFT_VECTOR)); // Add bias
            s_norm[tid] = r_in; __syncthreads();
            EGG_TRACE_STAT(step, t, l, "LN1", s_norm, HIDDEN_DIM);

            // QKV Projection (Rank-1 Fast)
            AccumType sbq = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_Q_B, tid), temp_storage, shared_scalar);
            AccumType sbk = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_K_B, tid), temp_storage, shared_scalar);
            AccumType sbv = block_reduce_sum_broadcast((AccumType)r_in * noise_from_hash(seed_base + SEED_OFF_V_B, tid), temp_storage, shared_scalar);

            // SIMD MatMul
            int32_t *v_ptr = (int32_t*)s_norm;
            const int32_t *wq_p = (const int32_t*)&model->w_q[l][0];
            const int32_t *wk_p = (const int32_t*)&model->w_k[l][0];
            const int32_t *wv_p = (const int32_t*)&model->w_v[l][0];
            
            AccumType aq, ak, av;
            compute_qkv_projection(v_ptr, wq_p, wk_p, wv_p, HIDDEN_DIM/4, HIDDEN_DIM, tid, aq, ak, av, sbq, sbk, sbv, seed_base, ns);

            // Apply RoPE rotation to Q and K
            aq = apply_rope_integer(aq, t, tid);
            ak = apply_rope_integer(ak, t, tid);

            // Adaptive QKV Normalization
            // Reuse shared memory scratchpad at offset 2*HIDDEN_DIM (free at this point)
            int32_t *warp_maxs_scratch = (int32_t*)&s_mem[2*HIDDEN_DIM];
            ActType qv = adaptive_qkv_normalize<BLOCK_THREADS/32>(aq, tid, warp_maxs_scratch);
            ActType kv = adaptive_qkv_normalize<BLOCK_THREADS/32>(ak, tid, warp_maxs_scratch);
            ActType vv = adaptive_qkv_normalize<BLOCK_THREADS/32>(av, tid, warp_maxs_scratch);
            
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
            
            AttnAccumType w_v_sum = 0;
            uint64_t tot_sc = 0;
            int32_t my_head_max = s_head_max[h];
            
            if(tid < N_HEADS) s_attn_scores[tid] = 0; // Reset for pass 2
            __syncthreads();

            EGG_TRACE_ATTN_DECL(attn_dbg);
            EGG_TRACE_ATTN_INIT(attn_dbg);
            
            for(int ctx=0; ctx <= t; ctx++) {
                ActType k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                AccumType df = (AccumType)qv * k_ctx;
                
                // Reduce across head
                for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                
                if ((tid % 32) == 0) atomicAdd(&s_attn_scores[h], (int32_t)df);
                __syncthreads();
                
                // Compute weight using new softmax LUT with max subtraction
                int32_t sc = s_attn_scores[h];
                int32_t shifted = (sc >> SHIFT_ATTN) - (my_head_max >> SHIFT_ATTN); // Normalize by max. Scaled down.
                int32_t wt = softmax_exp_lookup(shifted);
                
                EGG_TRACE_ATTN_PROBE(step, l, h, t, ctx, sc, my_head_max, shifted, wt);
                EGG_TRACE_ATTN_ACCUM(attn_dbg, wt);

                ActType v_ctx = lkv[SEQ_LEN*HIDDEN_DIM + ctx*HIDDEN_DIM + tid];
                w_v_sum += (AttnAccumType)wt * v_ctx;
                tot_sc += wt;
                
                __syncthreads();
                if(tid < N_HEADS) s_attn_scores[tid] = 0; // Reset
                __syncthreads();
            }
            EGG_TRACE_ATTN_FINISH(attn_dbg, step, l, h, t);

            if(tot_sc==0) tot_sc=1;
            // Scale down from 2^20 precision
            ActType ao = clip((w_v_sum / (int64_t)tot_sc));
            s_norm[tid] = ao; __syncthreads();

            // Output Proj
            AccumType sbo = block_reduce_sum_broadcast((AccumType)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid), temp_storage, shared_scalar);

            const int32_t *wo_p = (const int32_t*)model->w_o[l];
            int32_t *v_ptr_o = (int32_t*)s_norm;
            AccumType aco = compute_linear_projection(v_ptr_o, wo_p, HIDDEN_DIM/4, HIDDEN_DIM, tid, sbo, seed_base + SEED_OFF_O_A, ns);
            // UPDATE: Adaptive Layer Norm for Attention Residual
            AccumType attn_res = (AccumType)s_x[tid] + (aco >> SHIFT_OUT);
            s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(attn_res, tid, warp_maxs_scratch); __syncthreads();
            
            EGG_TRACE_STAT(step, t, l, "Attn", s_x, HIDDEN_DIM);

            // MLP Block
            // Norm 2 (With Bias)
            AccumType mtot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mmn = mtot/HIDDEN_DIM; if(!mmn) mmn=1;
            WeightType l2w = model->ln_2[l][tid];
            int8_t l2n = noise_from_hash(seed_base + SEED_OFF_LN_2, tid);
            WeightType l2b = model->ln_2_bias[l][tid];
            int8_t l2bn = noise_from_hash(seed_base + SEED_OFF_LN_2_BIAS, tid);
            ActType n2x = clip(((AccumType)s_x[tid] * (l2w + (((AccumType)l2n*ns)>>SIGMA_SHIFT_VECTOR))) / mmn
                             + l2b + (((AccumType)l2bn * ns) >> SIGMA_SHIFT_VECTOR)); // Add bias
            s_norm[tid] = n2x; __syncthreads();
            EGG_TRACE_STAT(step, t, l, "LN2", s_norm, HIDDEN_DIM);

            // Expand
            AccumType sbup = block_reduce_sum_broadcast((AccumType)n2x * noise_from_hash(seed_base + SEED_OFF_MLP_UP_B, tid), temp_storage, shared_scalar);

            ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256]; 

            const int32_t *wup_p = (const int32_t*)model->w_up[l];
            int32_t *v_ptr_up = (int32_t*)s_norm;
            
            for(int sub=0; sub<4; sub++) {
                int oidx = tid + sub*HIDDEN_DIM;
                AccumType aup = compute_linear_projection(v_ptr_up, wup_p, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, sbup, seed_base + SEED_OFF_MLP_UP_A, ns);
                
                WeightType b_up = model->mlp_bias_up[l][oidx];
                int8_t n_b_up = noise_from_hash(seed_base + SEED_OFF_MLP_BIAS_UP, oidx);
                
                ActType raw = clip((aup>>SHIFT_MLP_UP) + b_up + (((AccumType)n_b_up * ns) >> SIGMA_SHIFT_VECTOR)); 
                s_mlp[oidx] = d_ACT_LUT[(uint8_t)raw]; // GELU
            }
            __syncthreads();
            EGG_TRACE_STAT(step, t, l, "MLP_Exp", s_mlp, 4*HIDDEN_DIM);

            AccumType pbdn = 0;
            for(int sub=0; sub<4; sub++) pbdn += (AccumType)s_mlp[tid + sub*HIDDEN_DIM] * noise_from_hash(seed_base + SEED_OFF_MLP_DOWN_B, tid + sub*HIDDEN_DIM);
            AccumType sbdn = block_reduce_sum_broadcast(pbdn, temp_storage, shared_scalar);

            const int32_t *wdn_p = (const int32_t*)model->w_down[l];
            int32_t *v_ptr_dn = (int32_t*)s_mlp;
            AccumType adn = compute_linear_projection(v_ptr_dn, wdn_p, HIDDEN_DIM, HIDDEN_DIM, tid, sbdn, seed_base + SEED_OFF_MLP_DOWN_A, ns);

            // Add MLP Bias Down
            WeightType b_dn = model->mlp_bias_down[l][tid];
            int8_t n_b_dn = noise_from_hash(seed_base + SEED_OFF_MLP_BIAS_DOWN, tid);
            
            // UPDATE: Adaptive Layer Norm for MLP Residual
            AccumType mlp_res = (AccumType)s_x[tid] + (adn >> SHIFT_MLP_DOWN) + b_dn + (((AccumType)n_b_dn * ns) >> SIGMA_SHIFT_VECTOR);
            s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(mlp_res, tid, warp_maxs_scratch); __syncthreads();
            
            EGG_TRACE_STAT(step, t, l, "MLP", s_x, HIDDEN_DIM);
        }

        // 3. Final Head
        AccumType ftot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType fmn = ftot/HIDDEN_DIM; if(!fmn) fmn=1;
        
        WeightType lfw = model->ln_f[tid];
        int8_t lfn = noise_from_hash(step_seed + pair_idx + SEED_OFF_LN_F, tid);
        WeightType lfb = model->ln_f_bias[tid];
        int8_t lfbn = noise_from_hash(step_seed + pair_idx + SEED_OFF_LN_F_BIAS, tid);
        ActType nf = clip(((AccumType)s_x[tid] * (lfw + (((AccumType)lfn*ns)>>SIGMA_SHIFT_VECTOR))) / fmn
                        + lfb + (((AccumType)lfbn * ns) >> SIGMA_SHIFT_VECTOR));
        
        // Reuse s_mem[HD] for normed
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();
        
        AccumType sbh = block_reduce_sum_broadcast((AccumType)nf * noise_from_hash(step_seed + pair_idx + SEED_OFF_EMB + HIDDEN_DIM, tid), temp_storage, shared_scalar);

        int32_t *s_logits = (int32_t*)&s_mem[2*HIDDEN_DIM];
        __shared__ int32_t s_logit_max;
        
        if(tid < VOCAB_SIZE) {
            const int32_t *wh_p = (const int32_t*)model->embedding;
            int32_t *v_ptr_h = (int32_t*)s_norm;
            
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, tid, sbh, step_seed + pair_idx + SEED_OFF_EMB, ns);
            
            int32_t lgt = ah >> SHIFT_LOGIT;  // Keep as int32 for max comparison
            s_logits[tid] = lgt;
        }
        __syncthreads();
        
        // Find max logit using atomicMax (thread 0 initializes, all threads update)
        if (tid == 0) s_logit_max = INT_MIN;
        __syncthreads();
        if (tid < VOCAB_SIZE) atomicMax(&s_logit_max, s_logits[tid]);
        __syncthreads();

        __shared__ int32_t s_target_logit;
        int32_t my_ex = 0;

        // Compute exp locally and identify target logit
        if(tid < VOCAB_SIZE) {
             int32_t lgt = s_logits[tid];
             int32_t shifted = lgt - s_logit_max;
             my_ex = softmax_exp_lookup(shifted);

             uint8_t target_token = dataset[stream_pos + t + 1];
             if (tid == (int)target_token) s_target_logit = lgt;
        }
        __syncthreads();
        
        // Sum exponentials using BlockReduce
        AccumType sum_ex = BlockReduce(temp_storage).Sum(my_ex);

        // Loss Calc (Thread 0)
        if (tid == 0) {
            int64_t log_sum = 0;
            if (sum_ex > 0) {
                uint64_t x = (uint64_t)sum_ex; int pos = 0;
                while(x >= 256) { x >>= 8; pos += 8; }
                if(x >= 16) { x >>= 4; pos += 4; }
                if(x >= 4) { x >>= 2; pos += 2; }
                if(x >= 2) { pos += 1; }
                
                // Scale factor: log2(2^20) = 20, so subtract 20 from pos for proper scaling
                log_sum = (pos << 4) - (SOFTMAX_SCALE_BIT << 4);
            }
            // Loss = log(sum_ex) + max_logit - target_logit
            my_loss += (log_sum >> 1) + s_logit_max - s_target_logit;
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
        int byte_off = idx & 3;
        int word_idx = idx >> 2;
        int tid = word_idx % cols; // Output Dim (c)
        int k_chunk = word_idx / cols; 
        int k = k_chunk * 4 + byte_off; // Input Dim (r)
        
        // Approximate Gradient Computation
        VoteType vote = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
            int fit = fitnesses[p]; if(fit==0) continue;
            uint32_t s = step_seed + p + seed_base;
            // Use k for input noise (off_B), tid for output noise (off_A)
            vote += (VoteType)fit * noise_from_hash(s + off_A, tid) * noise_from_hash(s + off_B, k);
        }
        
        float g = -(float)vote; 

        // Load Adam State
        AdamParam p = adam_state[idx];
        
        // Update Moments
        p.m = ADAM_BETA1 * p.m + (1.0f - ADAM_BETA1) * g;
        p.v = ADAM_BETA2 * p.v + (1.0f - ADAM_BETA2) * (g * g);
        
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
        WeightType emb = get_embedding_byte(model->embedding, tid, input_token);
        WeightType ebias = model->emb_bias[tid];
        
        s_x[tid] = clip((AccumType)emb + ebias);
        __syncthreads();

        // --- INITIAL MLP BLOCK (Gen) ---
        // Norm 
        AccumType mitot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType mimn = mitot/HIDDEN_DIM; if(!mimn) mimn=1;
        WeightType liw = model->ln_init[tid];
        WeightType lib = model->ln_init_bias[tid];
        ActType nix = clip(((AccumType)s_x[tid] * liw) / mimn + lib);
        ActType *s_norm_init = &s_mem[HIDDEN_DIM];
        s_norm_init[tid] = nix; __syncthreads();

        // Expand
        ActType *s_mlp_init = &s_mem[2*HIDDEN_DIM + 256]; 
        const int32_t *wiup_p = (const int32_t*)model->w_emb_mlp_up;
        int32_t *v_ptr_iup = (int32_t*)s_norm_init;
        for(int sub=0; sub<4; sub++) {
            int oidx = tid + sub*HIDDEN_DIM;
            AccumType aup = compute_linear_projection(v_ptr_iup, wiup_p, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, 0, 0, 0);
            WeightType b_up = model->mlp_emb_bias_up[oidx];
            ActType raw = clip((aup>>SHIFT_MLP_UP) + b_up); 
            s_mlp_init[oidx] = d_ACT_LUT[(uint8_t)raw];
        }
        __syncthreads();

        // Contract
        const int32_t *widn_p = (const int32_t*)model->w_emb_mlp_down;
        int32_t *v_ptr_idn = (int32_t*)s_mlp_init;
        AccumType aidn = compute_linear_projection(v_ptr_idn, widn_p, HIDDEN_DIM, HIDDEN_DIM, tid, 0, 0, 0);
        WeightType b_idn = model->mlp_emb_bias_down[tid];
        
        int32_t *warp_maxs_scratch_init = (int32_t*)&s_mem[2*HIDDEN_DIM];
        AccumType imlp_res = (AccumType)s_x[tid] + (aidn >> SHIFT_MLP_DOWN) + b_idn;
        s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(imlp_res, tid, warp_maxs_scratch_init); __syncthreads();
        // ------------------------------

        // 2. Layers
        for (int l = 0; l < N_LAYERS; l++) {
            ActType *s_norm = &s_mem[HIDDEN_DIM];

            // LN 1 (With Bias)
            AccumType total_sum = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mean = total_sum / HIDDEN_DIM; if(!mean) mean=1;
            WeightType ln_w = model->ln_1[l][tid];
            WeightType ln_b = model->ln_1_bias[l][tid];
            ActType r_in = clip(((AccumType)s_x[tid] * ln_w) / mean + ln_b);
            s_norm[tid] = r_in; __syncthreads();

            // QKV
            int32_t *v_ptr = (int32_t*)s_norm;
            const int32_t *wq_p = (const int32_t*)&model->w_q[l][0];
            const int32_t *wk_p = (const int32_t*)&model->w_k[l][0];
            const int32_t *wv_p = (const int32_t*)&model->w_v[l][0];
            
            AccumType aq, ak, av;
            compute_qkv_projection(v_ptr, wq_p, wk_p, wv_p, HIDDEN_DIM/4, HIDDEN_DIM, tid, aq, ak, av, 0, 0, 0, 0, 0);
            
            // Apply RoPE rotation to Q and K
            aq = apply_rope_integer(aq, t, tid);
            ak = apply_rope_integer(ak, t, tid);

            // Adaptive QKV Normalization
            // Reuse shared memory scratchpad 
            int32_t *warp_maxs_scratch = (int32_t*)&s_mem[2*HIDDEN_DIM];
            ActType qv = adaptive_qkv_normalize<BLOCK_THREADS/32>(aq, tid, warp_maxs_scratch);
            ActType kv = adaptive_qkv_normalize<BLOCK_THREADS/32>(ak, tid, warp_maxs_scratch);
            ActType vv = adaptive_qkv_normalize<BLOCK_THREADS/32>(av, tid, warp_maxs_scratch);

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
                int32_t shifted = (sc >> SHIFT_ATTN) - (my_head_max >> SHIFT_ATTN);
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
            const int32_t *wo_p = (const int32_t*)model->w_o[l];
            int32_t *v_ptr_o = (int32_t*)s_norm;
            AccumType aco = compute_linear_projection(v_ptr_o, wo_p, HIDDEN_DIM/4, HIDDEN_DIM, tid, 0, 0, 0);
            // UPDATE: Adaptive Layer Norm for Attn Residual (Gen)
            AccumType attn_res = (AccumType)s_x[tid] + (aco >> SHIFT_OUT);
            s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(attn_res, tid, warp_maxs_scratch); __syncthreads();

            // MLP
            AccumType mtot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
            AccumType mmn = mtot/HIDDEN_DIM; if(!mmn) mmn=1;
            WeightType l2w = model->ln_2[l][tid];
            WeightType l2b = model->ln_2_bias[l][tid];
            ActType n2x = clip(((AccumType)s_x[tid] * l2w) / mmn + l2b);
            s_norm[tid] = n2x; __syncthreads();

            ActType *s_mlp = &s_mem[2*HIDDEN_DIM + 256];
            const int32_t *wup_p = (const int32_t*)model->w_up[l];
            int32_t *v_ptr_up = (int32_t*)s_norm;
            for(int sub=0; sub<4; sub++) {
                int oidx = tid + sub*HIDDEN_DIM;
                AccumType aup = compute_linear_projection(v_ptr_up, wup_p, HIDDEN_DIM/4, 4*HIDDEN_DIM, oidx, 0, 0, 0);
                WeightType b_up = model->mlp_bias_up[l][oidx];
                ActType raw = clip((aup>>SHIFT_MLP_UP) + b_up); 
                s_mlp[oidx] = d_ACT_LUT[(uint8_t)raw];
            }
            __syncthreads();

            const int32_t *wdn_p = (const int32_t*)model->w_down[l];
            int32_t *v_ptr_dn = (int32_t*)s_mlp;
            AccumType adn = compute_linear_projection(v_ptr_dn, wdn_p, HIDDEN_DIM, HIDDEN_DIM, tid, 0, 0, 0);
            WeightType b_dn = model->mlp_bias_down[l][tid];
            // UPDATE: Adaptive Layer Norm for MLP Residual (Gen)
            AccumType mlp_res = (AccumType)s_x[tid] + (adn >> SHIFT_MLP_DOWN) + b_dn;
            s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>(mlp_res, tid, warp_maxs_scratch); __syncthreads();
        }

        // Final Head
        AccumType ftot = block_reduce_sum_broadcast(abs(s_x[tid]), temp_storage, shared_scalar);
        AccumType fmn = ftot/HIDDEN_DIM; if(!fmn) fmn=1;
        WeightType lfw = model->ln_f[tid];
        WeightType lfb = model->ln_f_bias[tid];
        ActType nf = clip(((AccumType)s_x[tid] * lfw) / fmn + lfb);
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        __shared__ int32_t s_gen_max;

        if(tid < VOCAB_SIZE) {
            const int32_t *wh_p = (const int32_t*)model->embedding;
            int32_t *v_ptr_h = (int32_t*)s_norm;
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, tid, 0, 0, 0);
            shared_logits[tid] = ah >> SHIFT_LOGIT;
        }
        __syncthreads();

        if (tid == 0) s_gen_max = INT_MIN;
        __syncthreads();
        if (tid < VOCAB_SIZE) atomicMax(&s_gen_max, shared_logits[tid]);
        __syncthreads();

        if(tid < VOCAB_SIZE) {
            int32_t shifted = shared_logits[tid] - s_gen_max;
            shared_logits[tid] = softmax_exp_lookup(shifted);
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

#include "egg_disk_utils.h"

int main() {
    signal(SIGINT, handle_sigint);
    init_tables();
    cudaMemcpyToSymbol(d_EXP_LUT, h_EXP_LUT, SOFTMAX_LUT_SIZE*sizeof(int32_t));
    cudaMemcpyToSymbol(d_ACT_LUT, h_ACT_LUT, 256*sizeof(int8_t));
    cudaMemcpyToSymbol(d_ROPE_LUT, h_ROPE_LUT, ROPE_LUT_SIZE*sizeof(int32_t));

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
        // Weights are already in Packed SIMD layout.

        // Try load adam
        FILE *fa = fopen("models/egg_transformer_last.adam.bin", "rb");
        if(fa) {
            printf("Loading Adam state...\n");
            if (fread(h_adam_state, sizeof(AdamModel), 1, fa) != 1) {
                printf("Error reading Adam state, resetting optimizer.\n");
                memset(h_adam_state, 0, sizeof(AdamModel));
            }
            fclose(fa);
        }
    } else {
        printf("No existing model found, initializing valid random model...\n");
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
                d_dataset, ds.length, step*SEQ_LEN, d_model, d_kv_cache, d_loss, seed, offset, step
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
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_1_bias[l], (AdamParam*)d_adam_state->ln_1_bias[l], HIDDEN_DIM, SEED_OFF_LN_1_BIAS, s_base, d_fit, seed, current_lr);
            
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_2[l], (AdamParam*)d_adam_state->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2, s_base, d_fit, seed, current_lr);
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_2_bias[l], (AdamParam*)d_adam_state->ln_2_bias[l], HIDDEN_DIM, SEED_OFF_LN_2_BIAS, s_base, d_fit, seed, current_lr);
            
            update_vector_adam_kernel<<< (4*HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->mlp_bias_up[l], (AdamParam*)d_adam_state->mlp_bias_up[l], 4*HIDDEN_DIM, SEED_OFF_MLP_BIAS_UP, s_base, d_fit, seed, current_lr);
            update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->mlp_bias_down[l], (AdamParam*)d_adam_state->mlp_bias_down[l], HIDDEN_DIM, SEED_OFF_MLP_BIAS_DOWN, s_base, d_fit, seed, current_lr);
        }
        
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_f, (AdamParam*)d_adam_state->ln_f, HIDDEN_DIM, SEED_OFF_LN_F, 0, d_fit, seed, current_lr);
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_f_bias, (AdamParam*)d_adam_state->ln_f_bias, HIDDEN_DIM, SEED_OFF_LN_F_BIAS, 0, d_fit, seed, current_lr);
        // Head update removed (shared with embedding)
        
        // Initial MLP Updates
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_init, (AdamParam*)d_adam_state->ln_init, HIDDEN_DIM, SEED_OFF_LN_INIT, 0, d_fit, seed, current_lr);
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->ln_init_bias, (AdamParam*)d_adam_state->ln_init_bias, HIDDEN_DIM, SEED_OFF_LN_INIT_BIAS, 0, d_fit, seed, current_lr);

        update_matrix_adam_kernel<<< (HIDDEN_DIM*4*HIDDEN_DIM+511)/512, 512 >>>( (WeightType*)d_model->w_emb_mlp_up, (AdamParam*)d_adam_state->w_emb_mlp_up, HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_UP_A, SEED_OFF_EMB_MLP_UP_B, 0, d_fit, seed, current_lr);
        update_vector_adam_kernel<<< (4*HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->mlp_emb_bias_up, (AdamParam*)d_adam_state->mlp_emb_bias_up, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_UP, 0, d_fit, seed, current_lr);

        update_matrix_adam_kernel<<< (4*HIDDEN_DIM*HIDDEN_DIM+511)/512, 512 >>>( (WeightType*)d_model->w_emb_mlp_down, (AdamParam*)d_adam_state->w_emb_mlp_down, 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_EMB_MLP_DOWN_A, SEED_OFF_EMB_MLP_DOWN_B, 0, d_fit, seed, current_lr);
        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->mlp_emb_bias_down, (AdamParam*)d_adam_state->mlp_emb_bias_down, HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_DOWN, 0, d_fit, seed, current_lr);


        update_vector_adam_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>((WeightType*)d_model->emb_bias, (AdamParam*)d_adam_state->emb_bias, HIDDEN_DIM, SEED_OFF_EMB_BIAS, 0, d_fit, seed, current_lr*0.1); 
        update_matrix_adam_kernel<<< (VOCAB_SIZE*HIDDEN_DIM+511)/512, 512 >>>((WeightType*)d_model->embedding, (AdamParam*)d_adam_state->embedding, HIDDEN_DIM, VOCAB_SIZE, SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 0, d_fit, seed, current_lr);
        // RoPE: Removed pos_emb update

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
