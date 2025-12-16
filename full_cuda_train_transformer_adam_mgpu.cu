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
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

#include "egg_debug_printer.h"
#include "egg_disk_log.h"
#include "egg_adaptive_normalize.h"

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    (void)write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

// --- CONFIGURATION (all overridable via -D flags) ---
#ifndef HIDDEN_DIM
#  define HIDDEN_DIM 256
#endif
#ifndef HEAD_DIM
#  define HEAD_DIM 64
#endif
#ifndef N_LAYERS
#  define N_LAYERS 4
#endif
#ifndef SEQ_LEN
#  define SEQ_LEN 32
#endif

#include "egg_ntt.cuh"

// NTT Mode: 0=disabled, 1=Walsh-Hadamard, 2=Fermat-257, 3=Fermat-65537
#ifndef NTT_MODE
#  define NTT_MODE 0
#endif
#ifndef VOCAB_SIZE
#  define VOCAB_SIZE 256
#endif

#if VOCAB_SIZE != 256
#  define USE_TOKENIZER 1
#endif

#ifdef USE_TOKENIZER
using TokenType = uint32_t;
#else
using TokenType = uint8_t;
#endif

#ifndef WARP_SIZE
#  define WARP_SIZE 32
#endif
#ifndef MAX_BLOCK_THREADS
#  define MAX_BLOCK_THREADS 1024
#endif
#define ALIGNED_DIM ((HIDDEN_DIM + 31) & ~31)
#define BLOCK_THREADS (ALIGNED_DIM > MAX_BLOCK_THREADS ? MAX_BLOCK_THREADS : ALIGNED_DIM)
#define N_HEADS (HIDDEN_DIM / HEAD_DIM)

#ifndef POPULATION_BATCH_SIZE
#   define POPULATION_BATCH_SIZE (8192 * 2)
#endif
#define POPULATION_SIZE (POPULATION_BATCH_SIZE * 4)

#ifndef FIXED_POINT
#  define FIXED_POINT 4
#endif
#ifndef SIGMA_SHIFT
#  define SIGMA_SHIFT 3
#endif
#ifndef SIGMA_SHIFT_VECTOR
#  define SIGMA_SHIFT_VECTOR 4
#endif
#define MAX_VAL 127
#define MIN_VAL -127

#ifndef SOFTMAX_SCALE_BIT
#  define SOFTMAX_SCALE_BIT 18
#endif
#define SOFTMAX_SCALE (1 << SOFTMAX_SCALE_BIT)
#ifndef SOFTMAX_LUT_SIZE
#  define SOFTMAX_LUT_SIZE 256
#endif
#ifndef SOFTMAX_EXP_SCALE
#  define SOFTMAX_EXP_SCALE 6.0
#endif

// RoPE Configuration
#ifndef ROPE_SCALE_BIT
#  define ROPE_SCALE_BIT 20
#endif
#define ROPE_SCALE (1 << ROPE_SCALE_BIT)
#define ROPE_LUT_SIZE (SEQ_LEN * (HEAD_DIM / 2) * 2)


#define SEED_OFF_EMB 0
// SEED_OFF_POS removed (RoPE)
#define SEED_OFF_EMB_BIAS_A 50
#define SEED_OFF_EMB_BIAS_B 51
#define SEED_OFF_LN_1_A 100
#define SEED_OFF_LN_1_B 101
#define SEED_OFF_LN_1_BIAS_A 110
#define SEED_OFF_LN_1_BIAS_B 111

#define SEED_OFF_Q_A 200
#define SEED_OFF_Q_B 201
#define SEED_OFF_K_A 202
#define SEED_OFF_K_B 203
#define SEED_OFF_V_A 204
#define SEED_OFF_V_B 205
#define SEED_OFF_O_A 206
#define SEED_OFF_O_B 207

#define SEED_OFF_LN_2_A 300
#define SEED_OFF_LN_2_B 301
#define SEED_OFF_LN_2_BIAS_A 310
#define SEED_OFF_LN_2_BIAS_B 311

#define SEED_OFF_MLP_UP_A 400
#define SEED_OFF_MLP_UP_B 401
#define SEED_OFF_MLP_DOWN_A 402
#define SEED_OFF_MLP_DOWN_B 403
#define SEED_OFF_MLP_BIAS_UP_A 410
#define SEED_OFF_MLP_BIAS_UP_B 411
#define SEED_OFF_MLP_BIAS_DOWN_A 412
#define SEED_OFF_MLP_BIAS_DOWN_B 413

#define SEED_OFF_LN_F_A 900
#define SEED_OFF_LN_F_B 901
#define SEED_OFF_LN_F_BIAS_A 910
#define SEED_OFF_LN_F_BIAS_B 911

#define SEED_OFF_HEAD 999

#define SEED_OFF_LN_INIT_A 500
#define SEED_OFF_LN_INIT_B 501
#define SEED_OFF_LN_INIT_BIAS_A 502
#define SEED_OFF_LN_INIT_BIAS_B 503

#define SEED_OFF_EMB_MLP_UP_A 510
#define SEED_OFF_EMB_MLP_UP_B 511
#define SEED_OFF_EMB_MLP_DOWN_A 512
#define SEED_OFF_EMB_MLP_DOWN_B 513
#define SEED_OFF_EMB_MLP_BIAS_UP_A 514
#define SEED_OFF_EMB_MLP_BIAS_UP_B 515
#define SEED_OFF_EMB_MLP_BIAS_DOWN_A 516
#define SEED_OFF_EMB_MLP_BIAS_DOWN_B 517

// NTT Embedding Seed Offsets (for int32 decomposition: bytes 1, 2, 3)
#define SEED_OFF_NTT_EMB1 600
#define SEED_OFF_NTT_EMB2 610
#define SEED_OFF_NTT_EMB3 620

#ifndef SHIFT_ATTN
#  define SHIFT_ATTN 8
#endif
#ifndef SHIFT_QKV
#  define SHIFT_QKV 6
#endif
#ifndef SHIFT_OUT
#  define SHIFT_OUT 9
#endif
#ifndef SHIFT_LOGIT
#  define SHIFT_LOGIT 8
#endif
#ifndef SHIFT_MLP_UP
#  define SHIFT_MLP_UP 8
#endif
#ifndef SHIFT_MLP_DOWN
#  define SHIFT_MLP_DOWN 10
#endif

// Generator settings
#ifndef HOST_GAUSSIAN
#  define HOST_GAUSSIAN false
#endif
#ifndef DEVICE_GAUSSIAN
#  define DEVICE_GAUSSIAN false
#endif
#ifndef HOST_MASK
#  define HOST_MASK 15
#endif
#ifndef DEVICE_MASK
#  define DEVICE_MASK 15
#endif

// TOPOLOGY THEORY MODE
// If 1, each GPU is initialized with a different seed (different weights).
// The training loop ignores this difference, applying identical updates to divergent models.
#ifndef TOPOLOGY_DIVERGENCE
#  define TOPOLOGY_DIVERGENCE 0
#endif

#ifndef USE_SAME_DATA
#  define USE_SAME_DATA 0
#endif

// ADAM HYPERPARAMS
float get_learning_rate(long step) {
    if (step < 300) {
        return 0.3f;
    }
    if (step < 600) {
        return 0.15f;
    }
    if (step < 1000) {
        return 0.075f;
    }
    return 0.05f;
    if (step < 200) {
        return 0.5f;
    }
    if (step < 300) {
        return 0.25f;
    }
    if (step < 400) {
        return 0.1f;
    }
    if (step < 500) {
        return 0.05f;
    }
    if (step < 600) {
        return 0.025f;
    }
    return 0.025f;
}

#ifndef ADAM_BETA1
#  define ADAM_BETA1 0.9f
#endif
#ifndef ADAM_BETA2
#  define ADAM_BETA2 0.95f
#endif
#ifndef ADAM_EPS
#  define ADAM_EPS 1e-8f
#endif
#ifndef ADAM_WEIGHT_DECAY
#  define ADAM_WEIGHT_DECAY 0.01f
#endif

#ifndef USE_MUON
#  define USE_MUON 0
#endif

#if USE_MUON
#  ifndef MUON_MOMENTUM
#    define MUON_MOMENTUM 0.85f
#  endif
#  ifndef MUON_LR_SCALE
#    define MUON_LR_SCALE 1.0f
#  endif
#endif

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA Error: %s:%d\n", cudaGetErrorString(err), __LINE__); exit(1); } }
#define CHECK_CUBLAS(call) { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error: %d at %s:%d\n", (int)err, __FILE__, __LINE__); exit(1); } }

// --- TYPE ALIASES ---
using WeightType    = int8_t;
using ActType       = int8_t;
using AccumType     = int32_t;   // 32-bit sufficient for ~6M max sum
using AttnAccumType = long long; // Need high precision for attention weighted sum
using VoteType      = int32_t;

typedef struct { void *data; long length; } Dataset;

typedef struct {
    WeightType embedding[VOCAB_SIZE * HIDDEN_DIM];
    WeightType emb_bias[HIDDEN_DIM];
    
#if NTT_MODE != 0
    // NTT coefficient embedding tables (for int32 decomposition: bytes 1, 2, 3)
    WeightType ntt_emb1[VOCAB_SIZE * HIDDEN_DIM];  // Byte 1 of NTT coefficient
    WeightType ntt_emb2[VOCAB_SIZE * HIDDEN_DIM];  // Byte 2 of NTT coefficient
    WeightType ntt_emb3[VOCAB_SIZE * HIDDEN_DIM];  // Byte 3 (sign byte) of NTT coefficient
#endif
    
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

#if USE_MUON == 1
typedef struct {
    float m;
    float acc;
} MuonParam;
#define HIDDEN_OPT_TYPE MuonParam
#else
#define HIDDEN_OPT_TYPE AdamParam
#endif

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

ActType *d_kv_cache = NULL;

// New softmax LUT: exp(-i) * 2^20 for i in [0, 14]
__constant__ int32_t d_EXP_LUT[SOFTMAX_LUT_SIZE];
int32_t h_EXP_LUT[SOFTMAX_LUT_SIZE];

// Activation LUT (GELU)
__constant__ int8_t d_ACT_LUT[256];
int8_t h_ACT_LUT[256];

// RoPE Look-Up Table: [SEQ_LEN][HEAD_DIM/2][2 (cos, sin)]
__device__ int32_t d_ROPE_LUT[ROPE_LUT_SIZE];
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
            
            // Scale by ROPE_SCALE (2^ROPE_SCALE_BIT, default 2^30)
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
static inline int8_t gen_noise_host(uint32_t *rng) { 
    uint32_t r = xorshift32_host(rng);
#if HOST_GAUSSIAN
    int8_t n1 = (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & HOST_MASK));
    int8_t n2 = (int8_t)((r & 32 ? 1 : -1) * ((r >> 6) & HOST_MASK));
    int8_t n3 = (int8_t)((r & 1024 ? 1 : -1) * ((r >> 11) & HOST_MASK));
    return n1 + n2 + n3;
#else 
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & HOST_MASK));
#endif
}

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

void init_model(TransformerModel *model, uint32_t seed) {
    uint32_t rng = seed;
    TransformerModel *temp = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    if(!temp) exit(1);
    
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);

    repack_matrix(model->embedding, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
    
    for(int i=0; i<HIDDEN_DIM; i++) model->emb_bias[i] = 0;
    
#if NTT_MODE != 0
    // Initialize NTT embedding tables
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    repack_matrix(model->ntt_emb1, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
    
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    repack_matrix(model->ntt_emb2, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
    
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    repack_matrix(model->ntt_emb3, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
#endif
    
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
    uint32_t r = hash_rng(s, idx); 
#if DEVICE_GAUSSIAN
    int8_t n1 = (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & DEVICE_MASK));
    int8_t n2 = (int8_t)((r & 32 ? 1 : -1) * ((r >> 6) & DEVICE_MASK));
    int8_t n3 = (int8_t)((r & 1024 ? 1 : -1) * ((r >> 11) & DEVICE_MASK));
    return n1 + n2 + n3;
#else 
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & DEVICE_MASK));
#endif
}
__device__ __forceinline__ int8_t clip(AccumType a) { return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a); }

#include "muon_internal.cuh"

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
    if(step != -1) EGG_TRACE_STAT(step, t, l, cfg.n1, s_norm, HIDDEN_DIM);

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
    if(step != -1) EGG_TRACE_STAT(step, t, l, cfg.n2, s_mlp, 4*HIDDEN_DIM);

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
    
    s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>( (AccumType)s_x[tid] + (adn >> SHIFT_MLP_DOWN) + bdn + noise_val, tid, w_max); 
    __syncthreads();
    if(step != -1) EGG_TRACE_STAT(step, t, l, cfg.n3, s_x, HIDDEN_DIM);
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
    if(step != -1) EGG_TRACE_STAT(step, t, l, "LN1", s_norm, HIDDEN_DIM);

    AccumType sbq = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_Q_B, tid), temp_storage, shared_scalar);
    AccumType sbk = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_K_B, tid), temp_storage, shared_scalar);
    AccumType sbv = block_reduce_sum_broadcast((AccumType)s_norm[tid] * noise_from_hash(seed_base + SEED_OFF_V_B, tid), temp_storage, shared_scalar);

    AccumType aq, ak, av;
    compute_qkv_projection((int32_t*)s_norm, (const int32_t*)model->w_q[l], (const int32_t*)model->w_k[l], (const int32_t*)model->w_v[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, aq, ak, av, sbq, sbk, sbv, seed_base, ns);

    int32_t *w_max = (int32_t*)&s_mem[2*HIDDEN_DIM];
    ActType qv = adaptive_qkv_normalize<BLOCK_THREADS/32>(apply_rope_integer(aq, t, tid), tid, w_max);
    lkv_k[t*HIDDEN_DIM + tid] = adaptive_qkv_normalize<BLOCK_THREADS/32>(apply_rope_integer(ak, t, tid), tid, w_max);
    lkv_v[t*HIDDEN_DIM + tid] = adaptive_qkv_normalize<BLOCK_THREADS/32>(av, tid, w_max);
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
    
    EGG_TRACE_ATTN_DECL(attn_dbg); EGG_TRACE_ATTN_INIT(attn_dbg);

    for(int ctx=0; ctx <= t; ctx++) {
        AccumType df = (AccumType)qv * lkv_k[ctx*HIDDEN_DIM + tid];
        for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
        if ((tid % 32) == 0) atomicAdd(&s_attn[h], (int32_t)df);
        __syncthreads();
        int32_t wt = softmax_exp_lookup((s_attn[h] >> SHIFT_ATTN) - (my_h_max >> SHIFT_ATTN));
        EGG_TRACE_ATTN_PROBE(step, l, h, t, ctx, s_attn[h], my_h_max, 0, wt);
        EGG_TRACE_ATTN_ACCUM(attn_dbg, wt);
        w_v_sum += (AttnAccumType)wt * lkv_v[ctx*HIDDEN_DIM + tid]; tot_sc += wt;
        __syncthreads();
        if(tid < N_HEADS) s_attn[tid] = 0;
        __syncthreads();
    }
    EGG_TRACE_ATTN_FINISH(attn_dbg, step, l, h, t);
    ActType ao = clip(w_v_sum / (tot_sc ? (int64_t)tot_sc : 1));
    s_norm[tid] = ao; __syncthreads();

    // Out Proj
    AccumType sb = block_reduce_sum_broadcast((AccumType)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid), temp_storage, shared_scalar);
    AccumType aco = compute_linear_projection((int32_t*)s_norm, (const int32_t*)model->w_o[l], HIDDEN_DIM/4, HIDDEN_DIM, tid, sb, seed_base + SEED_OFF_O_A, ns);
    s_x[tid] = adaptive_layer_normalize<BLOCK_THREADS/32>((AccumType)s_x[tid] + (aco >> SHIFT_OUT), tid, w_max);
    __syncthreads();
    if(step != -1) EGG_TRACE_STAT(step, t, l, "Attn", s_x, HIDDEN_DIM);
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

__global__ void __launch_bounds__(MAX_BLOCK_THREADS) train_sequence_kernel(
    const TokenType * __restrict__ dataset, long data_len, int start_idx,
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
        
        if (step%5 == 0 && global_pop_offset == 0 && blockIdx.x == 0 && tid == 0 && t == 0) {
             printf("DEBUG: Dataset[0..10] at stream_pos=%ld: ", stream_pos);
#ifdef USE_TOKENIZER
             for(int i=0; i<10 && stream_pos+i < data_len; i++) printf("%u ", dataset[stream_pos+i]);
#else
             for(int i=0; i<10 && stream_pos+i < data_len; i++) printf("%d(%c) ", dataset[stream_pos+i], (dataset[stream_pos+i]>=32 && dataset[stream_pos+i]<=126) ? dataset[stream_pos+i] : '.');
#endif
             printf("\n");
        }

        // 1. Embedding
        TokenType input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        WeightType emb = get_embedding_byte(model->embedding, tid, input_token);
        WeightType ebias = model->emb_bias[tid];
        
        int8_t emb_bias_n1 = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_BIAS_A, tid);
        int8_t emb_bias_n2 = noise_from_hash((step_seed + pair_idx) + SEED_OFF_EMB_BIAS_B, tid);
        AccumType emb_bias_chk = ((AccumType)emb_bias_n1 * emb_bias_n2 * ns) >> SIGMA_SHIFT_VECTOR;
        
        // RoPE: Absolute pos emb removed
        int8_t a_tok = noise_from_hash(seed_emb, input_token);
        int8_t b_dim = noise_from_hash(seed_emb + HIDDEN_DIM, tid);
        AccumType perturb = ((AccumType)a_tok * b_dim * ns) >> (FIXED_POINT + SIGMA_SHIFT);

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

        compute_mlp(0, t, tid, s_x, s_mem, temp_storage, shared_scalar, model->ln_init, model->ln_init_bias, model->w_emb_mlp_up, model->mlp_emb_bias_up, model->w_emb_mlp_down, model->mlp_emb_bias_down, step_seed + pair_idx, ns, step, global_pop_offset, CFG_MLP_INIT);

        // 2. Stack
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed_base = (step_seed + pair_idx) + (l * 1000);
            
            ActType* lkv_base = global_kv_cache + kv_ind_offset + (l * kv_layer_stride);
            ActType* lkv_k = lkv_base;
            ActType* lkv_v = lkv_base + SEQ_LEN*HIDDEN_DIM;

            compute_transformer_layer(
                l, t, tid, model, 
                lkv_k, lkv_v,
                s_x, s_mem, 
                temp_storage, shared_scalar,
                seed_base, ns, step,
                global_pop_offset
            );
        }

        // 3. Final Head
        ActType nf = apply_standard_norm(
            s_x[tid], tid, temp_storage, shared_scalar,
            model->ln_f[tid], model->ln_f_bias[tid],
            step_seed + pair_idx, SEED_OFF_LN_F_A, SEED_OFF_LN_F_B, SEED_OFF_LN_F_BIAS_A, SEED_OFF_LN_F_BIAS_B, ns
        );
        
        // Reuse s_mem[HD] for normed
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();
        
        AccumType sbh = block_reduce_sum_broadcast((AccumType)nf * noise_from_hash(step_seed + pair_idx + SEED_OFF_EMB + HIDDEN_DIM, tid), temp_storage, shared_scalar);

        // --- Two-Pass Softmax (Loop-based for Large Vocab) ---
        
        // Pass 1: Find Max Logit
        int32_t local_max = INT_MIN;
        const int32_t *wh_p = (const int32_t*)model->embedding;
        int32_t *v_ptr_h = (int32_t*)s_norm;
        
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, sbh, step_seed + pair_idx + SEED_OFF_EMB, ns);
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
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, sbh, step_seed + pair_idx + SEED_OFF_EMB, ns);
            int32_t lgt = ah >> SHIFT_LOGIT;
            
            int32_t shifted = lgt - global_max;
            local_sum_ex += softmax_exp_lookup(shifted);
            
            if (v == (int)target_token) s_target_logit = lgt;
        }
        __syncthreads();
        
        // Reduce sum across threads (using 64-bit reduction)
        typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce64;
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

    if (tid == 0) accum_loss[p_idx] = (int32_t)my_loss;
}

__global__ void compute_fitness_kernel(const int32_t *accum_loss, int32_t *fitnesses, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t p = accum_loss[2*idx]; int32_t n = accum_loss[2*idx+1];
    fitnesses[idx] = (p < n) ? 1 : ((n < p) ? -1 : 0);
}

// ADAM W Implementation KERNELS

__device__ __forceinline__ void apply_adam_update(
    AdamParam &p,
    WeightType &w,
    float g,
    float learning_rate,
    int &change
) {
    // Update Moments
    p.m = ADAM_BETA1 * p.m + (1.0f - ADAM_BETA1) * g;
    p.v = ADAM_BETA2 * p.v + (1.0f - ADAM_BETA2) * (g * g);
    
    float step = - learning_rate * (p.m / (sqrtf(p.v) + ADAM_EPS) + ADAM_WEIGHT_DECAY * (float)w);
    
    // Accumulate
    p.acc += step;
    
    // Apply to Integer Weight
    if (p.acc >= 1.0f) {
        if (w < MAX_VAL) { w++; change=1; p.acc -= 1.0f; }
        else p.acc = 1.0f; // Clamp accumulator
    } else if (p.acc <= -1.0f) {
            if (w > MIN_VAL) { w--; change=1; p.acc += 1.0f; }
            else p.acc = -1.0f; // Clamp accumulator
    }
}

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
        
        WeightType w = W[idx];
        AdamParam p = adam_state[idx];
        apply_adam_update(p, w, -(float)vote, learning_rate, change);
        W[idx] = w;
        adam_state[idx] = p;
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
    if (idx < len) {
        VoteType vote = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
            int fit = fitnesses[p]; if(fit==0) continue;
            // Universal Rank-1 Noise: Gradient is fit * (N1 * N2)
            VoteType n1 = (VoteType)noise_from_hash(step_seed + p + seed_base + off_A, idx);
            VoteType n2 = (VoteType)noise_from_hash(step_seed + p + seed_base + off_B, idx);
            vote += (VoteType)fit * n1 * n2;
        }
        
        WeightType v_val = V[idx];
        AdamParam p = adam_state[idx];
        apply_adam_update(p, v_val, -(float)vote, learning_rate, change);
        V[idx] = v_val;
        adam_state[idx] = p;
    }

    if (change) atomicAdd(&s_count, 1);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) atomicAdd(&d_total_updates, (unsigned long long)s_count);
}

__global__ void generate_sequence_kernel(
    TokenType * buffer, int seed_len, int gen_len,
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

    int total_len = seed_len + gen_len;
    size_t kv_layer_stride = 2ULL * total_len * HIDDEN_DIM;

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

        // Final Head
        ActType nf = apply_standard_norm(
            s_x[tid], tid, temp_storage, shared_scalar,
            model->ln_f[tid], model->ln_f_bias[tid],
            0, 0, 0, 0, 0, 0
        );
        ActType *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        // --- Three-Pass Sampling (Loop-based) ---
        
        // Pass 1: Find Max Logit
        int32_t local_max = INT_MIN;
        const int32_t *wh_p = (const int32_t*)model->embedding;
        int32_t *v_ptr_h = (int32_t*)s_norm;
        
        for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
            AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, 0, 0, 0);
            int32_t lgt = ah >> SHIFT_LOGIT;
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
            int32_t shifted = lgt - global_max;
            local_sum_ex += softmax_exp_lookup(shifted);
        }
        
        typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce64;
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
        typedef cub::BlockScan<long long, BLOCK_THREADS> BlockScan64;
        __shared__ typename BlockScan64::TempStorage scan_storage;
        long long thread_prefix_sum;
        BlockScan64(scan_storage).ExclusiveSum(local_sum_ex, thread_prefix_sum);
        
        if (s_thresh >= thread_prefix_sum && s_thresh < thread_prefix_sum + local_sum_ex) {
            long long running = thread_prefix_sum;
            for (int v = tid; v < VOCAB_SIZE; v += blockDim.x) {
                AccumType ah = compute_linear_projection(v_ptr_h, wh_p, HIDDEN_DIM/4, VOCAB_SIZE, v, 0, 0, 0);
                int32_t lgt = ah >> SHIFT_LOGIT;
                int32_t shifted = lgt - global_max;
                int32_t ex = softmax_exp_lookup(shifted);
                
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
            }
        }
        __syncthreads();
    }
}

#include "egg_disk_utils.h"

struct GPUContext {
    int id;
    cudaStream_t stream;
    TransformerModel *d_model;
    AdamModel *d_adam_state;
    TokenType *d_dataset;
    int32_t *d_loss;
    int32_t *d_fit;
    ActType *d_kv_cache;
    unsigned long long *d_updates_ptr;
    
#if USE_MUON == 1
    cublasHandle_t cublas_handle;
    MuonWorkspace muon_ws;
#endif
};


void compute_fitness_host(const int32_t *losses, int32_t *fitnesses, int count) {
    for(int i=0; i<count; i++) {
        int32_t p = losses[2*i];
        int32_t n = losses[2*i+1];
        fitnesses[i] = (p < n) ? 1 : ((n < p) ? -1 : 0);
    }
}

int main() {
    signal(SIGINT, handle_sigint);
    
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) { printf("No CUDA devices found!\n"); exit(1); }
    printf("Detected %d CUDA devices.\n", num_devices);

    char **vocab_table = NULL;
    uint32_t loaded_vocab_size = 0;

    printf("\n=== EGG TRANSFORMER ADAM MGPU ===\n");
    printf("AdamW Config: B1=%.3f B2=%.3f EPS=%.1e WD=%.4f (Dynamic LR)\n", 
           ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY);

    Dataset ds = {0,0};
#ifdef USE_TOKENIZER
    FILE *f = fopen("input.bin", "rb");
    if(!f) { printf("No input.bin\n"); exit(1); }
    fseek(f,0,SEEK_END); 
    long file_size = ftell(f);
    ds.length = file_size / sizeof(TokenType);
    fseek(f,0,SEEK_SET);
    ds.data = malloc(file_size);
    if(fread(ds.data, 1, file_size, f) != file_size) { printf("Error reading input.bin\n"); exit(1); }
    fclose(f);
    printf("Loaded input.bin: %ld tokens (size %ld bytes)\n", ds.length, file_size);

    FILE *fd = fopen("decoding.bin", "rb");
    if(fd) {
        if(fread(&loaded_vocab_size, sizeof(uint32_t), 1, fd) == 1) {
            vocab_table = (char**)malloc(loaded_vocab_size * sizeof(char*));
            for(uint32_t i=0; i<loaded_vocab_size; i++) {
                uint32_t len; 
                if(fread(&len, sizeof(uint32_t), 1, fd) != 1) break;
                vocab_table[i] = (char*)malloc(len+1);
                if(fread(vocab_table[i], 1, len, fd) != len) break;
                vocab_table[i][len] = 0;
            }
            printf("Loaded decoding.bin: %u tokens\n", loaded_vocab_size);
        }
        fclose(fd);
    }
#else
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("No input.txt\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET);
    ds.data=malloc(ds.length); 
    if(fread(ds.data, 1, ds.length, f) != ds.length) { printf("Error reading input.txt\n"); exit(1); }
    fclose(f);
    printf("Loaded input.txt: %ld bytes\n", ds.length);
#endif

    TransformerModel *h_model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    AdamModel *h_adam_state = (AdamModel*)calloc(1, sizeof(AdamModel));
    
    ensure_models_dir();
    save_model_info();
    
    // Initialize host model
    if (load_model("models/egg_transformer_last.model.bin", h_model)) {
        printf("Resumed from models/egg_transformer_last.model.bin\n");
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
        init_model(h_model, 42);
    }
    
    std::vector<GPUContext> gpus;
    size_t kv_size = (size_t)POPULATION_BATCH_SIZE * N_LAYERS * 2 * SEQ_LEN * HIDDEN_DIM;
    printf("Allocating KV Cache per GPU: %.2f GB\n", kv_size / (1024.0*1024*1024));

    // Init Logging
    size_t vram_per_gpu = sizeof(TransformerModel) + sizeof(AdamModel) + ds.length + 
                          POPULATION_SIZE*sizeof(int32_t) + (POPULATION_SIZE/2)*sizeof(int32_t) + kv_size;
    
    EggLogConfig log_cfg;
    memset(&log_cfg, 0, sizeof(log_cfg));
    log_cfg.num_gpus = num_devices;
    log_cfg.vram_per_gpu = vram_per_gpu;
    log_cfg.hidden_dim = HIDDEN_DIM;
    log_cfg.head_dim = HEAD_DIM;
    log_cfg.n_layers = N_LAYERS;
    log_cfg.seq_len = SEQ_LEN;
    log_cfg.vocab_size = VOCAB_SIZE;
    log_cfg.n_heads = N_HEADS;
    log_cfg.pop_size = POPULATION_SIZE;
    log_cfg.softmax_scale_bit = SOFTMAX_SCALE_BIT;
    log_cfg.host_gaussian = HOST_GAUSSIAN;
    log_cfg.device_gaussian = DEVICE_GAUSSIAN;
    log_cfg.host_mask = HOST_MASK;
    log_cfg.device_mask = DEVICE_MASK;
    
    log_cfg.fixed_point = FIXED_POINT;
    log_cfg.sigma_shift = SIGMA_SHIFT;
    log_cfg.sigma_shift_vector = SIGMA_SHIFT_VECTOR;
    
    log_cfg.shift_attn = SHIFT_ATTN;
    log_cfg.shift_qkv = SHIFT_QKV;
    log_cfg.shift_out = SHIFT_OUT;
    log_cfg.shift_logit = SHIFT_LOGIT;
    log_cfg.shift_mlp_up = SHIFT_MLP_UP;
    log_cfg.shift_mlp_down = SHIFT_MLP_DOWN;
    
    log_cfg.softmax_exp_scale = SOFTMAX_EXP_SCALE;
    
    log_cfg.adam_beta1 = ADAM_BETA1;
    log_cfg.adam_beta2 = ADAM_BETA2;
    log_cfg.adam_eps = ADAM_EPS;
    log_cfg.adam_weight_decay = ADAM_WEIGHT_DECAY;
    
#if USE_MUON == 1
    log_cfg.use_muon = 1;
    log_cfg.muon_momentum = MUON_MOMENTUM;
    log_cfg.muon_lr_scale = MUON_LR_SCALE;
#else
    log_cfg.use_muon = 0;
#endif

    EggLogState log_state = egg_log_init("models/training_log.csv", log_cfg);

    for(int i=0; i<num_devices; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        
        // Init tables per device
        init_tables();
        cudaMemcpyToSymbol(d_EXP_LUT, h_EXP_LUT, SOFTMAX_LUT_SIZE*sizeof(int32_t));
        cudaMemcpyToSymbol(d_ACT_LUT, h_ACT_LUT, 256*sizeof(int8_t));
        cudaMemcpyToSymbol(d_ROPE_LUT, h_ROPE_LUT, ROPE_LUT_SIZE*sizeof(int32_t));
        
        GPUContext ctx;
        ctx.id = i;
        CHECK_CUDA(cudaStreamCreate(&ctx.stream));
        CHECK_CUDA(cudaMalloc(&ctx.d_model, sizeof(TransformerModel)));

#if TOPOLOGY_DIVERGENCE
        printf("TOPOLOGY_DIVERGENCE: Re-initializing model for GPU %d with seed %d\n", i, 42 + i);
        init_model(h_model, 42 + i);
#endif

        CHECK_CUDA(cudaMemcpy(ctx.d_model, h_model, sizeof(TransformerModel), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMalloc(&ctx.d_adam_state, sizeof(AdamModel)));
        CHECK_CUDA(cudaMemcpy(ctx.d_adam_state, h_adam_state, sizeof(AdamModel), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMalloc(&ctx.d_dataset, ds.length * sizeof(TokenType)));
        CHECK_CUDA(cudaMemcpy(ctx.d_dataset, ds.data, ds.length * sizeof(TokenType), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMalloc(&ctx.d_loss, POPULATION_SIZE * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&ctx.d_fit, (POPULATION_SIZE/2) * sizeof(int32_t)));
        CHECK_CUDA(cudaMalloc(&ctx.d_kv_cache, kv_size));
        
        CHECK_CUDA(cudaGetSymbolAddress((void**)&ctx.d_updates_ptr, d_total_updates));
        
#if USE_MUON == 1
        CHECK_CUBLAS(cublasCreate(&ctx.cublas_handle));
        CHECK_CUBLAS(cublasSetStream(ctx.cublas_handle, ctx.stream));
        // Allocate workspace for max size (MLP: HIDDEN_DIM * 4*HIDDEN_DIM)
        size_t max_mx_elems = HIDDEN_DIM * (HIDDEN_DIM * 4);
        size_t gram_elems = HIDDEN_DIM * HIDDEN_DIM;
        CHECK_CUDA(cudaMalloc(&ctx.muon_ws.d_buf1, max_mx_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ctx.muon_ws.d_buf2, gram_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ctx.muon_ws.d_buf3, gram_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ctx.muon_ws.d_buf_swap, max_mx_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ctx.muon_ws.d_scalar, sizeof(float)));
#endif

        gpus.push_back(ctx);
    }
    
    // Host buffers for aggregation (Pinned Memory)
    int32_t *h_loss; CHECK_CUDA(cudaMallocHost(&h_loss, POPULATION_SIZE * sizeof(int32_t)));
    int32_t *h_fit; CHECK_CUDA(cudaMallocHost(&h_fit, (POPULATION_SIZE/2) * sizeof(int32_t)));

    // Generation buffers (on GPU 0)
    CHECK_CUDA(cudaSetDevice(0));
    int gen_seed_len = 32;
    int gen_output_len = 64; 
    int total_gen_len = gen_seed_len + gen_output_len;
    TokenType *d_gen_buf; CHECK_CUDA(cudaMalloc(&d_gen_buf, total_gen_len * sizeof(TokenType)));
    int8_t *d_gen_kv; CHECK_CUDA(cudaMalloc(&d_gen_kv, N_LAYERS * 2 * total_gen_len * HIDDEN_DIM));

    printf("Starting Training on %d GPUs...\n", num_devices);
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step<max_steps && keep_running; step++) {
#ifdef MAX_STEPS
        if (step >= MAX_STEPS) break;
#endif
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x12345678);
#if NTT_MODE != 0
        // Add SEQ_LEN*sizeof(int32_t) for NTT buffer
        size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM) + (SEQ_LEN * sizeof(int32_t)); 
#else
        size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM); 
#endif
        
        // 1. Distributed Evaluation
        int current_pop_offset = 0;
        
        // Queue parallel work
        while(current_pop_offset < POPULATION_SIZE) {
            for(int i=0; i<num_devices; i++) {
                if (current_pop_offset >= POPULATION_SIZE) break;
                
                CHECK_CUDA(cudaSetDevice(gpus[i].id));
                int batch = POPULATION_BATCH_SIZE;
                if (current_pop_offset + batch > POPULATION_SIZE) batch = POPULATION_SIZE - current_pop_offset;
                
                train_sequence_kernel<<<batch, BLOCK_THREADS, sm_size, gpus[i].stream>>>(
                    gpus[i].d_dataset, ds.length, step*SEQ_LEN, gpus[i].d_model, gpus[i].d_kv_cache, 
                    gpus[i].d_loss, seed, current_pop_offset, step
                );
                
                // Copy losses back to host asynchronously
                CHECK_CUDA(cudaMemcpyAsync(
                    h_loss + current_pop_offset, 
                    gpus[i].d_loss + current_pop_offset, 
                    batch * sizeof(int32_t), 
                    cudaMemcpyDeviceToHost,
                    gpus[i].stream
                ));
                
                current_pop_offset += batch;
            }
        }
        
        // Wait for all evaluation to complete
        for(int i=0; i<num_devices; i++) {
             CHECK_CUDA(cudaSetDevice(gpus[i].id));
             CHECK_CUDA(cudaStreamSynchronize(gpus[i].stream));
             cudaError_t err = cudaGetLastError();
             if (err != cudaSuccess) {
                 printf("ERROR after evaluation GPU %d: %s\n", i, cudaGetErrorString(err));
                 exit(1);
             }
        }
        
        // 2. Host Fitness Calculation
        compute_fitness_host(h_loss, h_fit, POPULATION_SIZE/2);
        
        // 3. Broadcast Fitness and Run Adam on all GPUs
        float current_lr = get_learning_rate(step);
        
        // Queue Updates
        for(int i=0; i<num_devices; i++) {
            CHECK_CUDA(cudaSetDevice(gpus[i].id));
            
            // Broadcast fitness
            CHECK_CUDA(cudaMemcpyAsync(gpus[i].d_fit, h_fit, (POPULATION_SIZE/2) * sizeof(int32_t), cudaMemcpyHostToDevice, gpus[i].stream));
            
            // Reset update counter
            CHECK_CUDA(cudaMemsetAsync(gpus[i].d_updates_ptr, 0, sizeof(unsigned long long), gpus[i].stream));

            // Optimizer Updates
    #define LAUNCH_ADAM_VECTOR(V_PTR, ADAM_PTR, LEN, SEED_A, SEED_B, BASE, LR_SCALE) \
        update_vector_adam_kernel<<< (LEN+255)/256, 256, 0, gpus[i].stream >>>( \
        (WeightType*)V_PTR, (AdamParam*)ADAM_PTR, LEN, SEED_A, SEED_B, \
        BASE, gpus[i].d_fit, seed, current_lr * LR_SCALE)

#if USE_MUON == 1
    // Muon update O has norm 1. Adam update g/sqrt(v) has magnitude ~1.
    // We scale Muon by a factor to match expected update magnitude for int8 weights.
    // heuristic: sqrt(HIDDEN_DIM) or similar might be needed, but start with 1.0 * LR

    #define LAUNCH_ADAM_MATRIX(M_PTR, OPT_PTR, ROWS, COLS, SEED_A, SEED_B, BASE) \
        do { \
             muon_momentum_update_kernel<<< (ROWS*COLS+511)/512, 512, 0, gpus[i].stream >>>( \
                (MuonParam*)OPT_PTR, ROWS, COLS, SEED_A, SEED_B, BASE, gpus[i].d_fit, seed, MUON_MOMENTUM); \
             muon_gather_m_kernel<<< (ROWS*COLS+511)/512, 512, 0, gpus[i].stream >>>((MuonParam*)OPT_PTR, gpus[i].muon_ws.d_buf1, ROWS, COLS); \
             perform_newton_schulz(gpus[i].cublas_handle, gpus[i].muon_ws, gpus[i].muon_ws.d_buf1, ROWS, COLS, gpus[i].stream); \
             muon_apply_update_kernel<<< (ROWS*COLS+511)/512, 512, 0, gpus[i].stream >>>( \
                (WeightType*)M_PTR, (MuonParam*)OPT_PTR, gpus[i].muon_ws.d_buf1, ROWS, COLS, current_lr * MUON_LR_SCALE, ADAM_WEIGHT_DECAY); \
        } while(0)
#else
    #define LAUNCH_ADAM_MATRIX(M_PTR, ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, BASE) \
        update_matrix_adam_kernel<<< (ROWS*COLS+511)/512, 512, 0, gpus[i].stream >>>( \
        (WeightType*)M_PTR, (AdamParam*)ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, \
        BASE, gpus[i].d_fit, seed, current_lr)
#endif

            for(int l=0; l<N_LAYERS; l++) {
                int s_base = l * 1000;
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_q[l], gpus[i].d_adam_state->w_q[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_Q_A, SEED_OFF_Q_B, s_base);
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_k[l], gpus[i].d_adam_state->w_k[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_K_A, SEED_OFF_K_B, s_base);
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_v[l], gpus[i].d_adam_state->w_v[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_V_A, SEED_OFF_V_B, s_base);
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_o[l], gpus[i].d_adam_state->w_o[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_O_A, SEED_OFF_O_B, s_base);
                
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_up[l], gpus[i].d_adam_state->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, s_base);
                LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_down[l], gpus[i].d_adam_state->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, s_base);
                
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_1[l], gpus[i].d_adam_state->ln_1[l], HIDDEN_DIM, SEED_OFF_LN_1_A, SEED_OFF_LN_1_B, s_base, 1.0f);
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_1_bias[l], gpus[i].d_adam_state->ln_1_bias[l], HIDDEN_DIM, SEED_OFF_LN_1_BIAS_A, SEED_OFF_LN_1_BIAS_B, s_base, 1.0f);
                
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_2[l], gpus[i].d_adam_state->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2_A, SEED_OFF_LN_2_B, s_base, 1.0f);
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_2_bias[l], gpus[i].d_adam_state->ln_2_bias[l], HIDDEN_DIM, SEED_OFF_LN_2_BIAS_A, SEED_OFF_LN_2_BIAS_B, s_base, 1.0f);
                
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->mlp_bias_up[l], gpus[i].d_adam_state->mlp_bias_up[l], 4*HIDDEN_DIM, SEED_OFF_MLP_BIAS_UP_A, SEED_OFF_MLP_BIAS_UP_B, s_base, 1.0f);
                LAUNCH_ADAM_VECTOR(gpus[i].d_model->mlp_bias_down[l], gpus[i].d_adam_state->mlp_bias_down[l], HIDDEN_DIM, SEED_OFF_MLP_BIAS_DOWN_A, SEED_OFF_MLP_BIAS_DOWN_B, s_base, 1.0f);
            }
            
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_f, gpus[i].d_adam_state->ln_f, HIDDEN_DIM, SEED_OFF_LN_F_A, SEED_OFF_LN_F_B, 0, 1.0f);
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_f_bias, gpus[i].d_adam_state->ln_f_bias, HIDDEN_DIM, SEED_OFF_LN_F_BIAS_A, SEED_OFF_LN_F_BIAS_B, 0, 1.0f);
            
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_init, gpus[i].d_adam_state->ln_init, HIDDEN_DIM, SEED_OFF_LN_INIT_A, SEED_OFF_LN_INIT_B, 0, 1.0f);
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->ln_init_bias, gpus[i].d_adam_state->ln_init_bias, HIDDEN_DIM, SEED_OFF_LN_INIT_BIAS_A, SEED_OFF_LN_INIT_BIAS_B, 0, 1.0f);

            LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_emb_mlp_up, gpus[i].d_adam_state->w_emb_mlp_up, HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_UP_A, SEED_OFF_EMB_MLP_UP_B, 0);
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->mlp_emb_bias_up, gpus[i].d_adam_state->mlp_emb_bias_up, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_UP_A, SEED_OFF_EMB_MLP_BIAS_UP_B, 0, 1.0f);

            LAUNCH_ADAM_MATRIX(gpus[i].d_model->w_emb_mlp_down, gpus[i].d_adam_state->w_emb_mlp_down, 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_EMB_MLP_DOWN_A, SEED_OFF_EMB_MLP_DOWN_B, 0);
            LAUNCH_ADAM_VECTOR(gpus[i].d_model->mlp_emb_bias_down, gpus[i].d_adam_state->mlp_emb_bias_down, HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_DOWN_A, SEED_OFF_EMB_MLP_BIAS_DOWN_B, 0, 1.0f);

            LAUNCH_ADAM_VECTOR(gpus[i].d_model->emb_bias, gpus[i].d_adam_state->emb_bias, HIDDEN_DIM, SEED_OFF_EMB_BIAS_A, SEED_OFF_EMB_BIAS_B, 0, 0.1f);
            // Explicitly use Adam kernel for embedding (never Muon) to match AdamParam struct type
            update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512, 0, gpus[i].stream >>>(
                (WeightType*)gpus[i].d_model->embedding, 
                (AdamParam*)gpus[i].d_adam_state->embedding, 
                HIDDEN_DIM, VOCAB_SIZE, 
                SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 
                0, gpus[i].d_fit, seed, current_lr
            );

#if NTT_MODE != 0
            // NTT embedding optimizer updates (use Adam, not Muon)
            update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512, 0, gpus[i].stream >>>(
                (WeightType*)gpus[i].d_model->ntt_emb1, 
                (AdamParam*)gpus[i].d_adam_state->ntt_emb1, 
                HIDDEN_DIM, VOCAB_SIZE, 
                SEED_OFF_NTT_EMB1, SEED_OFF_NTT_EMB1+HIDDEN_DIM, 
                0, gpus[i].d_fit, seed, current_lr * 0.5f
            );
            update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512, 0, gpus[i].stream >>>(
                (WeightType*)gpus[i].d_model->ntt_emb2, 
                (AdamParam*)gpus[i].d_adam_state->ntt_emb2, 
                HIDDEN_DIM, VOCAB_SIZE, 
                SEED_OFF_NTT_EMB2, SEED_OFF_NTT_EMB2+HIDDEN_DIM, 
                0, gpus[i].d_fit, seed, current_lr * 0.5f
            );
            update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512, 0, gpus[i].stream >>>(
                (WeightType*)gpus[i].d_model->ntt_emb3, 
                (AdamParam*)gpus[i].d_adam_state->ntt_emb3, 
                HIDDEN_DIM, VOCAB_SIZE, 
                SEED_OFF_NTT_EMB3, SEED_OFF_NTT_EMB3+HIDDEN_DIM, 
                0, gpus[i].d_fit, seed, current_lr * 0.5f
            );
#endif
            
    #undef LAUNCH_ADAM_MATRIX
    #undef LAUNCH_ADAM_VECTOR
        }
        
        // Wait for Adam Updates
        for(int i=0; i<num_devices; i++) {
             CHECK_CUDA(cudaSetDevice(gpus[i].id));
             CHECK_CUDA(cudaStreamSynchronize(gpus[i].stream));
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        // Updates count from GPU 0 (identically executed on all)
        CHECK_CUDA(cudaSetDevice(gpus[0].id));
        unsigned long long h_updates = 0;
        CHECK_CUDA(cudaMemcpy(&h_updates, gpus[0].d_updates_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        double step_ms = get_time_diff_ms(t0, t1);
        double tokens_per_sec = (double)(POPULATION_SIZE * SEQ_LEN) / (step_ms / 1000.0);

        long long total_loss = 0;
        for(int k=0; k<POPULATION_SIZE; k++) total_loss += h_loss[k];
        double avg_loss = (double)total_loss / (POPULATION_SIZE * SEQ_LEN * 16.0); 

        printf("Step %ld | Loss: %.4f | Time: %.2f ms | Updates: %llu | Speed: %.2f tok/s | LR: %.3f\n", 
            step, avg_loss, step_ms, h_updates, tokens_per_sec, current_lr);

        egg_log_record(&log_state, step, avg_loss, h_updates, current_lr);

        if (step % 5 == 0) {
            // Generation on GPU 0
            CHECK_CUDA(cudaSetDevice(gpus[0].id));
            CHECK_CUDA(cudaMemcpy(d_gen_buf, gpus[0].d_dataset + (step*SEQ_LEN) % (ds.length-SEQ_LEN), gen_seed_len * sizeof(TokenType), cudaMemcpyDeviceToDevice));
            generate_sequence_kernel<<<1, BLOCK_THREADS, sm_size, gpus[0].stream>>>(
                d_gen_buf, gen_seed_len, gen_output_len, gpus[0].d_model, d_gen_kv, seed+999
            );
            CHECK_CUDA(cudaStreamSynchronize(gpus[0].stream));
            TokenType h_buf[256];
            CHECK_CUDA(cudaMemcpy(h_buf, d_gen_buf, total_gen_len * sizeof(TokenType), cudaMemcpyDeviceToHost));
            
            printf("\n--- GENERATION ---\n");
            printf("\033[32m"); 
#ifdef USE_TOKENIZER
            for(int i=0; i<gen_seed_len; i++) {
                if(vocab_table && h_buf[i] < loaded_vocab_size) printf("%s", vocab_table[h_buf[i]]);
                else printf("%u ", h_buf[i]);
            }
            printf("\033[36m"); 
            for(int i=gen_seed_len; i<total_gen_len; i++) {
                if(vocab_table && h_buf[i] < loaded_vocab_size) printf("%s", vocab_table[h_buf[i]]);
                else printf("%u ", h_buf[i]);
            }
#else
            for(int i=0; i<gen_seed_len; i++) {
                char c = h_buf[i]; printf("%c", (c>=32 && c<=126) ? c : '.');
            }
            printf("\033[36m"); 
            for(int i=gen_seed_len; i<total_gen_len; i++) {
                char c = h_buf[i]; printf("%c", (c>=32 && c<=126) ? c : '.');
            }
#endif
            printf("\033[0m\n\n");

            // Save Model from GPU 0
            //CHECK_CUDA(cudaMemcpy(h_model, gpus[0].d_model, sizeof(TransformerModel), cudaMemcpyDeviceToHost));
            //CHECK_CUDA(cudaMemcpy(h_adam_state, gpus[0].d_adam_state, sizeof(AdamModel), cudaMemcpyDeviceToHost));
            //save_checkpoint(h_model, h_adam_state, step);
        }
    }
    
    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_gen_buf); cudaFree(d_gen_kv);
    cudaFreeHost(h_loss); cudaFreeHost(h_fit);
    egg_log_close(&log_state);

    for(auto &ctx : gpus) {
        CHECK_CUDA(cudaSetDevice(ctx.id));
        cudaFree(ctx.d_model); 
        cudaFree(ctx.d_dataset);
        cudaFree(ctx.d_loss);
        cudaFree(ctx.d_fit);
        cudaFree(ctx.d_kv_cache);
        cudaFree(ctx.d_adam_state);
#if USE_MUON == 1
        if(ctx.cublas_handle) cublasDestroy(ctx.cublas_handle);
        cudaFree(ctx.muon_ws.d_buf1);
        cudaFree(ctx.muon_ws.d_buf2);
        cudaFree(ctx.muon_ws.d_buf3);
        cudaFree(ctx.muon_ws.d_buf_swap);
        cudaFree(ctx.muon_ws.d_scalar);
#endif
        cudaStreamDestroy(ctx.stream);
    }

    free(h_model); free(h_adam_state); free(ds.data);
    return 0;
}
