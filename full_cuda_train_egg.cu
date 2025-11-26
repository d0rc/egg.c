#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Configuration ---
#define SM_CORES 128
#define VOCAB_SIZE 256
#define HIDDEN_DIM (SM_CORES * 1)
#define N_LAYERS 4
#define SEQ_LEN 256
#define BATCH 8 // Streaming batch size per block
#define POPULATION_SIZE SM_CORES * 16 * 4
#define SHARED_STRIDE (HIDDEN_DIM * 4)
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define UPDATE_THRESHOLD 870 * 40
#define MAX_VAL 127
#define MIN_VAL -127

#define MAX_STRIDE 8 

// --- Seed Offsets for RNG Consistency ---
#define SEED_OFF_GRU_M1 1
#define SEED_OFF_GRU_M2 2
#define SEED_OFF_GRU_M3 3
#define SEED_OFF_GRU_M4 4
#define SEED_OFF_MLP_EXP 5
#define SEED_OFF_MLP_PROJ 6
#define SEED_OFF_HEAD 999

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// --- Data Structures ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// Transposed Model for Coalesced Access.
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM]; 
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM]; 
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; 
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; 
    int8_t head[HIDDEN_DIM * VOCAB_SIZE]; 
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// Global Tables
__device__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];

__device__ int32_t d_debug_updates[2]; // 0: Inc, 1: Dec

// --- Helpers (Host) ---

void init_tables() {
    for(int i=0; i<256; i++) 
        h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_host(uint32_t *rng) {
    uint32_t r = xorshift32_host(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) {
        for(int c=0; c<cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void init_model(EggModel *model) {
    uint32_t rng = 42;
    EggModel *temp = (EggModel*)malloc(sizeof(EggModel)); 
    if (!temp) { printf("Failed to allocate temp model\n"); exit(1); }

    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_host(&rng);

    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, VOCAB_SIZE, HIDDEN_DIM);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) temp->gru_weights[0][0][i] = gen_noise_host(&rng);
            transpose_matrix(model->gru_weights[l][g], temp->gru_weights[0][0], HIDDEN_DIM, HIDDEN_DIM);
        }
        for(int m=0; m<2; m++) for(int i=0; i<HIDDEN_DIM; i++) model->gru_biases[l][m][i] = 0; 
    }
    
    for(int l=0; l<N_LAYERS; l++) {
        int dim_expand = HIDDEN_DIM * (HIDDEN_DIM * 4);
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][0], temp->mlp_weights[0][0], HIDDEN_DIM*4, HIDDEN_DIM);

        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][1], temp->mlp_weights[0][0], HIDDEN_DIM, HIDDEN_DIM*4);
        
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
    }
    
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    
    free(temp);
}


// --- DEVICE ---
#define KERNEL_LOOP(idx, limit) for(int idx = threadIdx.x; idx < (limit); idx += blockDim.x)

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; 
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__device__ __forceinline__ int8_t clip(long long a) {
    return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a);
}

__device__ __forceinline__ long long warpReduceSum64(long long val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int lo = __shfl_down_sync(0xFFFFFFFF, (int)val, offset);
        int hi = __shfl_down_sync(0xFFFFFFFF, (int)(val >> 32), offset);
        val += ((long long)hi << 32) | (unsigned int)lo;
    }
    return val;
}

__device__ __forceinline__ long long blockReduceSum64(long long val) {
    __syncthreads(); 
    static __shared__ long long shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum64(val);
    
    if (lane == 0) {
        if (wid < 32) shared[wid] = val; 
    }
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[threadIdx.x] : 0;
    
    if (wid == 0) val = warpReduceSum64(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    
    return shared[0];
}

extern __shared__ int8_t s_mem[]; // Shared memory: [BATCH][SHARED_STRIDE]

__global__ void generate_sequence_kernel(
    const EggModel * __restrict__ model,
    uint32_t seed,
    const uint8_t *seed_text,
    int seed_len,
    int gen_len,
    uint8_t *output
) {
    int8_t *s_ptr = s_mem;
    
    // Local hidden state
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    KERNEL_LOOP(i, HIDDEN_DIM) {
        for(int l=0; l<N_LAYERS; l++) h_local[l][(i - threadIdx.x)/blockDim.x] = 0;
    }

    int total_len = seed_len + gen_len;
    uint8_t current_token = 0; 
    
    for(int t=0; t < total_len; t++) {
        if (t < seed_len) current_token = seed_text[t];
        
        // 1. Embed
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = model->embedding[current_token * HIDDEN_DIM + i];
        __syncthreads();
        
        // 2. Layers
        for(int l=0; l<N_LAYERS; l++) {
             // LN 1
             long long local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             
             long long sum = blockReduceSum64(local_sum);
             if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][0][i]) / mean);
             __syncthreads();
             
             // Copy H to Shared (Buf 1) for MatMul
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[HIDDEN_DIM + i] = h_local[l][(i - threadIdx.x)/blockDim.x];
             __syncthreads();
             
             // GRU Phase 1
             const int8_t *w0 = &model->gru_weights[l][0][0];
             const int8_t *w1 = &model->gru_weights[l][1][0];
             
             // Computes W0 (Buf 3) and W1 (Buf 2) logic:
             // Read Buf 0 (X) and Buf 1 (H).
             // Buffer logic: W0 -> Buf 3 (Temp), W1 -> Buf 2 (Target for gating later)
             // But wait, W1 * H.
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w0[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 // Store temporarily in register/spare buf? 
                 // We use Buf 3 for Acc0, Buf 2 for Acc1.
                 s_ptr[3*HIDDEN_DIM + i] = acc0 >> 8;
                 s_ptr[2*HIDDEN_DIM + i] = acc1 >> 8;
             }
             __syncthreads();
             
             // Gates
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 int8_t b1 = s_ptr[3*HIDDEN_DIM + i];
                 int8_t b2 = s_ptr[2*HIDDEN_DIM + i];
                 int8_t ft = clip((long long)b1 + b2 + model->gru_biases[l][0][i]);
                 
                 int8_t h_val = s_ptr[HIDDEN_DIM + i]; // Buf 1 (H)
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 // We need to preserve FT for Update.
                 s_ptr[3*HIDDEN_DIM + i] = ft;    // Keep FT in Buf 3
                 s_ptr[2*HIDDEN_DIM + i] = gated; // Gated in Buf 2
             }
             __syncthreads();
             
             // GRU Phase 2: W2*X (Buf 0), W3*Gated (Buf 2)
             // Compute to Registers, Update H inline.
             const int8_t *w2 = &model->gru_weights[l][2][0];
             const int8_t *w3 = &model->gru_weights[l][3][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w2[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                 }
                 int8_t ft = s_ptr[3*HIDDEN_DIM + i];
                 int8_t ht = clip((acc0 >> 8) + (acc1 >> 8) + model->gru_biases[l][1][i]);
                 
                 int8_t h_curr = h_local[l][(i - threadIdx.x)/blockDim.x];
                 int32_t diff = ht - h_curr;
                 int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                 h_curr = clip(h_curr + update);
                 h_local[l][(i - threadIdx.x)/blockDim.x] = h_curr;
                 
                 // New X
                 s_ptr[i] = clip((long long)h_curr + s_ptr[i]);
             }
             __syncthreads();
             
             // Save MLP Residual (Buf 0 -> Buf 2?) No, register better for small array.
             int8_t mlp_resid[MAX_STRIDE];
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 mlp_resid[(i - threadIdx.x)/blockDim.x] = s_ptr[i];
             }
             
             // MLP LN (Buf 0)
             local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             sum = blockReduceSum64(local_sum);
             if(!sum) sum=1; mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][1][i]) / mean);
             __syncthreads();
             
             // MLP Expand (Buf 0 -> Buf 0..3)
             // Buffer in Registers first
             int8_t exp_res[MAX_STRIDE][4];
             const int8_t *w_exp = &model->mlp_weights[l][0][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     long long acc = 0;
                     for(int k=0; k<HIDDEN_DIM; k++) {
                         acc += (long long)s_ptr[k] * w_exp[k*(4*HIDDEN_DIM) + (i + sub*HIDDEN_DIM)];
                     }
                     exp_res[(i - threadIdx.x)/blockDim.x][sub] = clip(acc >> 8);
                 }
             }
             __syncthreads();
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     s_ptr[i + sub*HIDDEN_DIM] = exp_res[(i - threadIdx.x)/blockDim.x][sub];
                 }
             }
             __syncthreads();
             
             // MLP Project (Buf 0..3 -> Buf 0)
             const int8_t *w_proj = &model->mlp_weights[l][1][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc = 0;
                 for(int k=0; k<(4*HIDDEN_DIM); k++) {
                     acc += (long long)s_ptr[k] * w_proj[k*HIDDEN_DIM + i];
                 }
                 s_ptr[i] = clip((acc >> 9) + mlp_resid[(i - threadIdx.x)/blockDim.x]);
             }
             __syncthreads();
        }
        
        // 3. Head
        long long local_sum = 0;
        KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
        long long sum = blockReduceSum64(local_sum);
        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_out[i]) / mean);
        __syncthreads();
        
        // Logits
        const int8_t *w_head = &model->head[0];
        KERNEL_LOOP(i, VOCAB_SIZE) {
            long long acc = 0;
            for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)s_ptr[k] * w_head[k*VOCAB_SIZE + i]; 
            }
            s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
        }
        __syncthreads();
        
        // 4. Sample
        if (t >= seed_len) {
            long long val = 0;
            KERNEL_LOOP(i, VOCAB_SIZE) {
                int idx = (int32_t)s_ptr[HIDDEN_DIM + i] + 128;
                idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                val = d_EXP2_TABLE[idx];
            }
            long long sum_exp = blockReduceSum64(val); 
            
            uint32_t s = seed + t * 222;
            uint32_t r_val = hash_rng(s, 0);
            long long r_threshold = (sum_exp > 0) ? (r_val % sum_exp) : 0;
            
            __shared__ int selected_token;
            if(threadIdx.x == 0) selected_token = VOCAB_SIZE - 1;
            __syncthreads();
            
            // Parallel Scan/Selection is hard. Serial for now?
            if(threadIdx.x == 0) {
                long long running = 0;
                for(int i=0; i<VOCAB_SIZE; i++) {
                    int idx = (int32_t)s_ptr[HIDDEN_DIM + i] + 128;
                    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                    running += d_EXP2_TABLE[idx];
                    if (running > r_threshold) {
                        selected_token = i;
                        break;
                    }
                }
                current_token = (uint8_t)selected_token;
                output[t - seed_len] = current_token;
            }
            __syncthreads();
            current_token = (uint8_t)selected_token;
        }
    }
}

__global__ void __launch_bounds__(256) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ pop_states, 
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int block_p_idx = blockIdx.x; 
    int8_t *s_ptr = s_mem;
    
    // Load State
    int8_t h_local[BATCH][N_LAYERS][MAX_STRIDE];
    
    KERNEL_LOOP(i, HIDDEN_DIM) {
        for(int b=0; b<BATCH; b++) {
            int p_idx = block_p_idx * BATCH + b;
            for(int l=0; l<N_LAYERS; l++) {
                 h_local[b][l][(i - threadIdx.x)/blockDim.x] = pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
            }
        }
    }
    
    long long my_loss[BATCH]; 
    for(int b=0; b<BATCH; b++) my_loss[b] = 0;
    
    for (int t = 0; t < SEQ_LEN; t++) {
        __syncthreads(); 
        
        for(int b=0; b<BATCH; b++) {
            int p_idx = block_p_idx * BATCH + b;
            long pair_idx = p_idx / 2;
            long stride = data_len / (POPULATION_SIZE / 2);
            long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
            uint8_t input_token = dataset[stream_pos + t];
            
            KERNEL_LOOP(i, HIDDEN_DIM) {
                s_ptr[b * SHARED_STRIDE + i] = model->embedding[input_token * HIDDEN_DIM + i];
            }
        }
        __syncthreads();
        
        for (int l = 0; l < N_LAYERS; l++) {
            int8_t gru_resid[BATCH][MAX_STRIDE];

            // LN 1
            for(int b=0; b<BATCH; b++) {
                int8_t *sx_b = &s_ptr[b * SHARED_STRIDE];
                long long local_sum = 0;
                
                KERNEL_LOOP(i, HIDDEN_DIM) {
                    int8_t val = sx_b[i];
                    gru_resid[b][(i - threadIdx.x)/blockDim.x] = val;
                    local_sum += abs((long long)val);
                }
                long long sum = blockReduceSum64(local_sum);
                if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                
                KERNEL_LOOP(i, HIDDEN_DIM) {
                    sx_b[i] = clip(((long long)sx_b[i] * model->ln_weights[l][0][i]) / mean);
                }
            }
            __syncthreads();
            
            // Copy H to Shared (Buf 1)
            for(int b=0; b<BATCH; b++) {
                 KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + i] = h_local[b][l][(i - threadIdx.x)/blockDim.x];
            }
            __syncthreads();

            // Rank-1 Precalc (xB)
            long long xB_m1[BATCH], xB_m2[BATCH], xB_m3[BATCH], xB_m4[BATCH];
            
            for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                uint32_t seed = (step_seed + pair_idx) + (l * 100); 
                
                long long ls_1 = 0, ls_2 = 0;
                KERNEL_LOOP(i, HIDDEN_DIM) {
                    int8_t b1 = noise_from_hash(seed + SEED_OFF_GRU_M1 + HIDDEN_DIM, i);
                    ls_1 += (long long)s_ptr[b*SHARED_STRIDE+i] * b1;
                    
                    int8_t b2 = noise_from_hash(seed + SEED_OFF_GRU_M2 + HIDDEN_DIM, i);
                    ls_2 += (long long)s_ptr[b*SHARED_STRIDE+HIDDEN_DIM+i] * b2;
                }
                xB_m1[b] = blockReduceSum64(ls_1);
                xB_m2[b] = blockReduceSum64(ls_2);
            }

            // MatMul 1 & 2 + Gates
            const int8_t *w0 = &model->gru_weights[l][0][0];
            const int8_t *w1 = &model->gru_weights[l][1][0];
            
            KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long dot1[BATCH]; // In Registers
                 long long dot2[BATCH]; 
                 for(int b=0; b<BATCH; b++) { dot1[b]=0; dot2[b]=0; }

                 for(int k=0; k<HIDDEN_DIM; k++) {
                    int8_t v1 = w0[k*HIDDEN_DIM + i];
                    int8_t v2 = w1[k*HIDDEN_DIM + i];
                    for(int b=0; b<BATCH; b++) {
                        dot1[b] += (long long)s_ptr[b*SHARED_STRIDE + k] * v1;
                        dot2[b] += (long long)s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + k] * v2;
                    }
                }
                
                // Noise & Gates
                for(int b=0; b<BATCH; b++) {
                     int p_idx = block_p_idx * BATCH + b;
                     int ns = (p_idx % 2 == 0) ? 1 : -1;
                     int pair_idx = p_idx / 2;
                     uint32_t seed = (step_seed + pair_idx) + (l * 100);
                     
                     int8_t a1 = noise_from_hash(seed + SEED_OFF_GRU_M1, i);
                     if(ns!=0) dot1[b] += ((xB_m1[b] * (long long)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     
                     int8_t a2 = noise_from_hash(seed + SEED_OFF_GRU_M2, i);
                     if(ns!=0) dot2[b] += ((xB_m2[b] * (long long)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     
                     int8_t r1 = dot1[b] >> 8;
                     int8_t r2 = dot2[b] >> 8;
                     int8_t ft = clip((long long)r1 + r2 + model->gru_biases[l][0][i]);
                     int8_t h_val = s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + i];
                     int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                     
                     s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + i] = gated; // Buf 2
                     s_ptr[b*SHARED_STRIDE + 3*HIDDEN_DIM + i] = ft;    // Buf 3
                }
            }
            __syncthreads();
            
            // Rank-1 M3 & M4
            for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                uint32_t seed = (step_seed + pair_idx) + (l * 100);
                
                long long ls_3=0, ls_4=0;
                KERNEL_LOOP(i, HIDDEN_DIM) {
                    int8_t b3 = noise_from_hash(seed + SEED_OFF_GRU_M3 + HIDDEN_DIM, i);
                    ls_3 += (long long)s_ptr[b*SHARED_STRIDE+i] * b3;
                    
                    int8_t b4 = noise_from_hash(seed + SEED_OFF_GRU_M4 + HIDDEN_DIM, i);
                    ls_4 += (long long)s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + i] * b4;
                }
                xB_m3[b] = blockReduceSum64(ls_3);
                xB_m4[b] = blockReduceSum64(ls_4);
            }
            
            // Phase 2
            const int8_t *w2 = &model->gru_weights[l][2][0];
            const int8_t *w3 = &model->gru_weights[l][3][0];
            
            KERNEL_LOOP(i, HIDDEN_DIM) {
                long long dot1[BATCH], dot2[BATCH];  
                for(int b=0; b<BATCH; b++) { dot1[b]=0; dot2[b]=0; }
                
                for(int k=0; k<HIDDEN_DIM; k++) {
                    int8_t v3 = w2[k*HIDDEN_DIM + i]; 
                    int8_t v4 = w3[k*HIDDEN_DIM + i]; 
                    for(int b=0; b<BATCH; b++) {
                        dot1[b] += (long long)s_ptr[b*SHARED_STRIDE + k] * v3;
                        dot2[b] += (long long)s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + k] * v4;
                    }
                }
                
                for(int b=0; b<BATCH; b++) {
                     int p_idx = block_p_idx * BATCH + b;
                     int ns = (p_idx % 2 == 0) ? 1 : -1;
                     int pair_idx = p_idx / 2;
                     uint32_t seed = (step_seed + pair_idx) + (l * 100);
                     
                     int8_t a3 = noise_from_hash(seed + SEED_OFF_GRU_M3, i);
                     if(ns!=0) dot1[b] += ((xB_m3[b] * (long long)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     
                     int8_t a4 = noise_from_hash(seed + SEED_OFF_GRU_M4, i);
                     if(ns!=0) dot2[b] += ((xB_m4[b] * (long long)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     
                     int8_t ht = clip((dot1[b] >> 8) + (dot2[b] >> 8) + model->gru_biases[l][1][i]);
                     
                     int8_t ft = s_ptr[b*SHARED_STRIDE + 3*HIDDEN_DIM + i];
                     int8_t h_curr = h_local[b][l][(i - threadIdx.x)/blockDim.x];
                     int32_t diff = ht - h_curr;
                     int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                     h_curr = clip(h_curr + update);
                     h_local[b][l][(i - threadIdx.x)/blockDim.x] = h_curr;
                     
                     s_ptr[b*SHARED_STRIDE + i] = clip((long long)h_curr + gru_resid[b][(i - threadIdx.x)/blockDim.x]); 
                }
            }
            __syncthreads(); 
            
            // Pre-LN Resid (Use Register)
            int8_t mlp_resid[BATCH][MAX_STRIDE];
            KERNEL_LOOP(i, HIDDEN_DIM) {
                for(int b=0; b<BATCH; b++) mlp_resid[b][(i - threadIdx.x)/blockDim.x] = s_ptr[b*SHARED_STRIDE + i];
            }

            // MLP LN 2
            for(int b=0; b<BATCH; b++) {
                long long local_sum = 0;
                KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[b*SHARED_STRIDE + i]);
                long long sum = blockReduceSum64(local_sum);
                if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[b*SHARED_STRIDE + i] = clip(((long long)s_ptr[b*SHARED_STRIDE + i] * model->ln_weights[l][1][i]) / mean);
            }
            __syncthreads(); 
            
            // Expand
            // Buffer in Registers!
            int8_t mlp_res[BATCH][MAX_STRIDE][4];
            long long xB_mlp1[BATCH];
            
            // Rank-1 xB
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
                 long long ls = 0;
                 KERNEL_LOOP(i, HIDDEN_DIM) {
                     int8_t b_val = noise_from_hash(seed + SHARED_STRIDE, i); 
                     ls += (long long)s_ptr[b*SHARED_STRIDE+i] * b_val;
                 }
                 xB_mlp1[b] = blockReduceSum64(ls);
            }

            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
            
            KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                      int out_idx = i + sub * HIDDEN_DIM;
                      long long acc_mlp[BATCH]; for(int b=0; b<BATCH; b++) acc_mlp[b]=0;
                      
                      for(int k=0; k<HIDDEN_DIM; k++) {
                          int8_t w_val = w_mlp1[k*(4*HIDDEN_DIM) + out_idx];
                          for(int b=0; b<BATCH; b++) {
                              acc_mlp[b] += (long long)s_ptr[b*SHARED_STRIDE + k] * w_val;
                          }
                      }
                      
                      for(int b=0; b<BATCH; b++) {
                         int p_idx = block_p_idx * BATCH + b;
                         int ns = (p_idx % 2 == 0) ? 1 : -1;
                         int pair_idx = p_idx / 2;
                         uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
                         
                         int8_t a_val = noise_from_hash(seed, out_idx);
                         if(ns!=0) acc_mlp[b] += ((xB_mlp1[b] * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                         mlp_res[b][(i - threadIdx.x)/blockDim.x][sub] = clip(acc_mlp[b] >> 8);
                      }
                 }
             }
             __syncthreads();
             
             // Write Expand to Shared
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int b=0; b<BATCH; b++) {
                     for(int sub=0; sub<4; sub++) {
                         s_ptr[b*SHARED_STRIDE + i + sub*HIDDEN_DIM] = mlp_res[b][(i - threadIdx.x)/blockDim.x][sub];
                     }
                 }
             }
             __syncthreads();
             
             // xB for M2 (Input 4*HIDDEN)
             long long xB_mlp2[BATCH];
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
                 
                 long long local_sum = 0;
                 KERNEL_LOOP(i, HIDDEN_DIM) {
                     for(int sub=0; sub<4; sub++) {
                         int in_idx = i + sub*HIDDEN_DIM;
                         int8_t b_val = noise_from_hash(seed + HIDDEN_DIM, in_idx);
                         local_sum += (long long)s_ptr[b*SHARED_STRIDE + in_idx] * b_val;
                     }
                 }
                 xB_mlp2[b] = blockReduceSum64(local_sum);
             }
             
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc_proj[BATCH]; for(int b=0; b<BATCH; b++) acc_proj[b] = 0;
                 
                 for(int k=0; k<HIDDEN_DIM*4; k++) {
                     int8_t w_val = w_mlp2[k*HIDDEN_DIM + i];
                     for(int b=0; b<BATCH; b++) {
                         acc_proj[b] += (long long)s_ptr[b*SHARED_STRIDE + k] * w_val;
                     }
                 }
                 
                 for(int b=0; b<BATCH; b++) {
                     int p_idx = block_p_idx * BATCH + b;
                     int ns = (p_idx % 2 == 0) ? 1 : -1;
                     int pair_idx = p_idx / 2;
                     uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
                     
                     int8_t a_val = noise_from_hash(seed, i);
                     if(ns!=0) acc_proj[b] += ((xB_mlp2[b] * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     
                     int32_t res = acc_proj[b] >> 9; 
                     s_ptr[b*SHARED_STRIDE + i] = clip(res + mlp_resid[b][(i - threadIdx.x)/blockDim.x]);
                 }
             }
             __syncthreads();
        } 
        
        // 3. Head LN
        for(int b=0; b<BATCH; b++) {
            long long local_sum = 0;
            KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[b * SHARED_STRIDE + i]);
            long long sum = blockReduceSum64(local_sum);
            if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[b * SHARED_STRIDE + i] = clip(((long long)s_ptr[b * SHARED_STRIDE + i] * model->ln_out[i]) / mean);
        }
        __syncthreads();
        
        // 4. Head Dense & Loss
         long long xB_head[BATCH];
         for(int b=0; b<BATCH; b++) {
             int p_idx = block_p_idx * BATCH + b;
             int pair_idx = p_idx / 2;
             uint32_t seed = (step_seed+pair_idx) + SEED_OFF_HEAD;
             long long ls = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 int8_t b_val = noise_from_hash(seed + VOCAB_SIZE, i); 
                 ls += (long long)s_ptr[b*SHARED_STRIDE+i] * b_val;
             }
             xB_head[b] = blockReduceSum64(ls);
         }
         
         const int8_t *w_head = &model->head[0];
         
         // We need logits for all vocab.
         // KERNEL_LOOP over VOCAB_SIZE
         KERNEL_LOOP(i, VOCAB_SIZE) {
             long long acc[BATCH]; for(int b=0; b<BATCH; b++) acc[b]=0;
             for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t w = w_head[k*VOCAB_SIZE + i];
                for(int b=0; b<BATCH; b++) acc[b] += (long long)s_ptr[b*SHARED_STRIDE + k] * w;
            }
            
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + SEED_OFF_HEAD;
                 
                 int8_t a_val = noise_from_hash(seed, i);
                 if(ns!=0) acc[b] += ((xB_head[b] * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 // Store logit temporarily? or reuse shared?
                 // We can reuse shared offset HIDDEN_DIM + i?
                 // s_ptr is 4*HIDDEN.
                 // HEAD needs VOCAB.
                 // s_ptr[HIDDEN_DIM...] is free.
                 s_ptr[b * SHARED_STRIDE + HIDDEN_DIM + i] = clip(acc[b] >> 8); 
            }
         }
         __syncthreads();
         
          // 5. Softmax Loss
          uint8_t targets[BATCH];
          if(threadIdx.x==0) {
              for(int b=0; b<BATCH; b++) {
                  int p_idx = block_p_idx * BATCH + b;
                  long pair_idx = p_idx / 2;
                  long stride = data_len / (POPULATION_SIZE / 2);
                  long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
                  targets[b] = dataset[stream_pos + t + 1];
              }
          }
          __syncthreads();
          // Broadcast targets? No need, loop uses it.
          // But 'targets' is local array in thread 0.
          // Need shared? 
          __shared__ uint8_t s_targets[BATCH];
          if(threadIdx.x==0) { for(int b=0; b<BATCH; b++) s_targets[b] = targets[b]; }
          __syncthreads();
          
          for(int b=0; b<BATCH; b++) {
              long long local_exp = 0;
              KERNEL_LOOP(i, VOCAB_SIZE) {
                  int idx = (int32_t)s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + i] + 128;
                  idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                  local_exp += d_EXP2_TABLE[idx];
              }
              long long sum_exp = blockReduceSum64(local_exp);
              
              long long log_sum = 0;
              if(threadIdx.x==0) {
                 long long x = sum_exp;
                 if (x > 0) {
                     int pos = 0;
                     while(x >= 65536) { x >>= 16; pos += 16; }
                     if (x >= 256)  { x >>= 8;  pos += 8; }
                     if (x >= 16)   { x >>= 4;  pos += 4; }
                     if (x >= 4)    { x >>= 2;  pos += 2; }
                     if (x >= 2)    {           pos += 1; }
                     
                     long long fraction = (pos>=4) ? (sum_exp-(1LL<<pos))>>(pos-4) : (sum_exp-(1LL<<pos))<<(4-pos);
                     log_sum = (pos<<4) + fraction - 64;
                 }
                 
                 int32_t target_l = (int32_t)s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + s_targets[b]] + 128;
                 my_loss[b] += (log_sum - target_l);
              }
          }

    } // Seq Loop
    
    if (threadIdx.x == 0) {
        for(int b=0; b<BATCH; b++) accum_loss[block_p_idx * BATCH + b] = (int32_t)my_loss[b]; // Loss fits in int32 usually, but accumulator is LL inside
    }
    
    // Store State
    KERNEL_LOOP(i, HIDDEN_DIM) {
        for(int b=0; b<BATCH; b++) {
            int p_idx = block_p_idx * BATCH + b;
            for(int l=0; l<N_LAYERS; l++) {
                 pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = h_local[b][l][(i - threadIdx.x)/blockDim.x];
            }
        }
    }
}

__global__ void update_matrix_kernel(
    int8_t * __restrict__ W,
    int rows,
    int cols,
    int offset_A,
    int offset_B,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    
    int r = idx % rows; 
    int c = idx / rows; 
    
    long long vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        
        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + offset_A, r);
        int8_t b = noise_from_hash(s + offset_B, c);
        
        if (fit == 0) continue;
        
        vote += (long long)fit * (int)a * (int)b;
    }
    
    int8_t w_curr = W[c * rows + r]; 

    if(vote > UPDATE_THRESHOLD && w_curr < MAX_VAL) {
        w_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -UPDATE_THRESHOLD && w_curr > MIN_VAL) {
        w_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    W[c * rows + r] = w_curr; 
}

int main() {
    srand(time(NULL));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    init_tables();
    cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t));
    
    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("Error: input.txt not found!\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET); 
    ds.data=(uint8_t*)malloc(ds.length); 
    if(!fread(ds.data,1,ds.length,f)) { printf("Error reading input.txt\n"); exit(1); }
    fclose(f); 

    EggModel *h_model = (EggModel*)malloc(sizeof(EggModel));
    init_model(h_model); 
    
    EggModel *d_model;
    CHECK_CUDA(cudaMalloc(&d_model, sizeof(EggModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(EggModel), cudaMemcpyHostToDevice));
    
    uint8_t *d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));
    
    int32_t *d_accum_loss;
    CHECK_CUDA(cudaMalloc(&d_accum_loss, POPULATION_SIZE * sizeof(int32_t)));
    int32_t *h_accum_loss = (int32_t*)malloc(POPULATION_SIZE * sizeof(int32_t));
    
    int32_t *h_fitnesses = (int32_t*)malloc((POPULATION_SIZE/2) * sizeof(int32_t));
    int32_t *d_fitnesses;
    CHECK_CUDA(cudaMalloc(&d_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    int8_t *d_pop_states;
    size_t state_size = POPULATION_SIZE * N_LAYERS * HIDDEN_DIM;
    CHECK_CUDA(cudaMalloc(&d_pop_states, state_size));
    CHECK_CUDA(cudaMemset(d_pop_states, 0, state_size));
    
    printf("Starting EGGROLL CUDA Training (Batch=%d)...\n", BATCH);
    long max_steps = (ds.length - 1) / SEQ_LEN;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    unsigned long total_tokens = 0;

    for(long step=0; step<max_steps; step++) {
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        size_t shared_mem_size = BATCH * SHARED_STRIDE;
        train_sequence_kernel<<<POPULATION_SIZE / BATCH, 256, shared_mem_size>>>(
            d_dataset, ds.length, start_idx, d_model, d_pop_states, d_accum_loss, seed
        );
        cudaDeviceSynchronize();
        
        CHECK_CUDA(cudaMemcpy(h_accum_loss, d_accum_loss, POPULATION_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost));
        
        int fit_sum = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
             int32_t loss_pos = h_accum_loss[2*p];
             int32_t loss_neg = h_accum_loss[2*p+1];
             if (loss_pos < loss_neg) h_fitnesses[p] = 1;
             else if (loss_neg < loss_pos) h_fitnesses[p] = -1;
             else h_fitnesses[p] = 0;
             fit_sum += abs(h_fitnesses[p]);
        }
        CHECK_CUDA(cudaMemcpy(d_fitnesses, h_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t), cudaMemcpyHostToDevice));

        if (step % 10 == 0) {
            printf("[Debug] Step %ld Pair 0: Pos=%d Neg=%d Fit=%d | Total NonZero Fits: %d\n", 
                step, h_accum_loss[0], h_accum_loss[1], h_fitnesses[0], fit_sum);
        }
        
        int32_t zeros[2] = {0, 0};
        CHECK_CUDA(cudaMemcpyToSymbol(d_debug_updates, zeros, 2*sizeof(int32_t)));

        for(int l=0; l<N_LAYERS; l++) {
             int seed_base = l * 100;
             
             size_t gru_size = HIDDEN_DIM * HIDDEN_DIM * sizeof(int8_t);
             long off_g = offsetof(EggModel, gru_weights) + (l * 4 * HIDDEN_DIM * HIDDEN_DIM);
             
             for(int g=0; g<4; g++) {
                 int seed_off = SEED_OFF_GRU_M1 + g; 
                 update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                     (int8_t*)d_model + off_g + g*gru_size, 
                     HIDDEN_DIM, HIDDEN_DIM, 
                     seed_off, seed_off + HIDDEN_DIM, 
                     seed_base, d_fitnesses, seed
                 );
             }
             
             size_t mlp_exp_size = HIDDEN_DIM * (HIDDEN_DIM*4) * sizeof(int8_t);
             long off_m = offsetof(EggModel, mlp_weights) + (l * 2 * mlp_exp_size);
             
             int exp_rows = 4*HIDDEN_DIM;
             int exp_cols = HIDDEN_DIM;
             update_matrix_kernel<<< (exp_rows*exp_cols + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + 0, exp_rows, exp_cols, 
                 0, SHARED_STRIDE, 
                 seed_base + SEED_OFF_MLP_EXP, d_fitnesses, seed
             );
             
             update_matrix_kernel<<< (HIDDEN_DIM*(4*HIDDEN_DIM) + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + mlp_exp_size, HIDDEN_DIM, 4*HIDDEN_DIM, 
                 0, HIDDEN_DIM, 
                 seed_base + SEED_OFF_MLP_PROJ, d_fitnesses, seed
             );
        }
        
        long off_head = offsetof(EggModel, head);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head, VOCAB_SIZE, HIDDEN_DIM, 
            0, VOCAB_SIZE, 
            SEED_OFF_HEAD, d_fitnesses, seed
        );
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_tokens += POPULATION_SIZE * SEQ_LEN;
        
        if (step % 1 == 0) { 
            int32_t h_updates[2];
            CHECK_CUDA(cudaMemcpyFromSymbol(h_updates, d_debug_updates, 2*sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
            
            static uint8_t *d_output = NULL;
            static uint8_t *h_output = NULL;
            int seed_len = 30; 
            int gen_len = 50;
            if(!d_output) {
                CHECK_CUDA(cudaMalloc(&d_output, gen_len));
                h_output = (uint8_t*)malloc(gen_len);
            }
            
            struct timespec t_samp_start, t_samp_end;
            clock_gettime(CLOCK_MONOTONIC, &t_samp_start);
            
            generate_sequence_kernel<<<1, 256, 5 * HIDDEN_DIM * sizeof(int8_t)>>>(
                d_model, seed + 12345, d_dataset + start_idx, seed_len, gen_len, d_output
            );
            CHECK_CUDA(cudaDeviceSynchronize()); 
            CHECK_CUDA(cudaMemcpy(h_output, d_output, gen_len, cudaMemcpyDeviceToHost));
            
            clock_gettime(CLOCK_MONOTONIC, &t_samp_end);
            double sample_ms = ((t_samp_end.tv_sec - t_samp_start.tv_sec) * 1000.0) + 
                               ((t_samp_end.tv_nsec - t_samp_start.tv_nsec) / 1e6);
            
            printf("\033[32m");
            for(int i=0; i<seed_len; i++) {
                char c = ds.data[start_idx+i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[36m");
            for(int i=0; i<gen_len; i++) {
                char c = h_output[i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[0m\n");

            double avg_loss = 0;
            for(int i=0; i<POPULATION_SIZE; i++) avg_loss += h_accum_loss[i];
            avg_loss /= (double)(POPULATION_SIZE); 
            double loss_per_token = avg_loss / (SEQ_LEN * (1 << FIXED_POINT));
            
            double total_t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
            
            printf("\nStep %ld | Loss: %.4f | Up+: %d Up-: %d | GPU Sample: %.2f ms | Tok/s: %.2f\n", 
               step, loss_per_token, h_updates[0], h_updates[1], sample_ms,  total_tokens / total_t); 
        }
    }
    return 0;
}
