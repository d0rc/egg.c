#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Configuration ---
#define SM_CORES 128
#define WARP_SIZE 32
#define BLOCK_THREADS 256
#define BATCH WARP_SIZE
#define VOCAB_SIZE 256
#define HIDDEN_DIM (SM_CORES * 1)
#define N_LAYERS 4
#define SEQ_LEN 256
#define POPULATION_SIZE (SM_CORES * 128)
#define SHARED_STRIDE (HIDDEN_DIM * 4)
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define UPDATE_THRESHOLD 870 * 40
#define MAX_VAL 127
#define MIN_VAL -127

#define MAX_STRIDE 8 

// --- Seed Offsets for RNG Consistency ---
#define SEED_OFF_EMB 0
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

    // Embedding: transpose to column-major for coalesced GPU access
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    transpose_matrix(model->embedding, temp->embedding, VOCAB_SIZE, HIDDEN_DIM);

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

extern __shared__ int8_t s_mem[]; 

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
        
        // 1. Embed (column-major access: feature * VOCAB_SIZE + token)
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = model->embedding[i * VOCAB_SIZE + current_token];
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

__global__ void __launch_bounds__(BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ pop_states, 
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int p_idx = blockIdx.x * (BLOCK_THREADS / WARP_SIZE) + warp_id;
    
    if (p_idx >= POPULATION_SIZE) return;

    int8_t *my_s_ptr = &s_mem[warp_id * SHARED_STRIDE];
    
    // Load State: Single batch item per Warp
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE;
        for(int l=0; l<N_LAYERS; l++) {
             h_local[l][sub] = pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
        }
    }
    
    long long my_loss = 0;

    // Prepare Data Indices
    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;

    for (int t = 0; t < SEQ_LEN; t++) {
        __syncwarp();
        
        uint8_t input_token = dataset[stream_pos + t];
        
        // Embed with rank-1 perturbation (column-major access)
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        int8_t a_token = noise_from_hash(seed_emb, input_token);  // A[token] scalar
        
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int8_t base = model->embedding[i * VOCAB_SIZE + input_token];
            int8_t b_val = noise_from_hash(seed_emb + HIDDEN_DIM, i);
            
            long long perturb = ((long long)a_token * b_val * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            my_s_ptr[i] = clip((long long)base + perturb);
        }
        __syncwarp();
        
        for (int l = 0; l < N_LAYERS; l++) {
            // LN 1
            long long local_sum = 0;
            int8_t gru_resid[MAX_STRIDE];

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t val = my_s_ptr[i];
                gru_resid[i / WARP_SIZE] = val;
                local_sum += abs((long long)val);
            }
            long long sum = warpReduceSum64(local_sum);
            if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * model->ln_weights[l][0][i]) / mean);
            }
            __syncwarp();
            
            // Copy H to Shared (Buf 1)
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 my_s_ptr[HIDDEN_DIM + i] = h_local[l][i / WARP_SIZE];
            }
            __syncwarp();

            // Rank-1 Precalc (xB) for GRU Phase 1
            uint32_t seed = (step_seed + pair_idx) + (l * 100); 
            
            long long ls_1 = 0, ls_2 = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b1 = noise_from_hash(seed + SEED_OFF_GRU_M1 + HIDDEN_DIM, i);
                ls_1 += (long long)my_s_ptr[i] * b1;
                
                int8_t b2 = noise_from_hash(seed + SEED_OFF_GRU_M2 + HIDDEN_DIM, i);
                ls_2 += (long long)my_s_ptr[HIDDEN_DIM + i] * b2;
            }
            long long xB_m1 = warpReduceSum64(ls_1);
            long long xB_m2 = warpReduceSum64(ls_2);

            // MatMul 1 & 2 + Gates
            const int8_t *w0 = &model->gru_weights[l][0][0];
            const int8_t *w1 = &model->gru_weights[l][1][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long dot1 = 0;
                 long long dot2 = 0;
                 
                 for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w0[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 
                 // Noise
                 int8_t a1 = noise_from_hash(seed + SEED_OFF_GRU_M1, i);
                 if(ns!=0) dot1 += ((xB_m1 * (long long)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a2 = noise_from_hash(seed + SEED_OFF_GRU_M2, i);
                 if(ns!=0) dot2 += ((xB_m2 * (long long)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t ft = clip((dot1 >> 8) + (dot2 >> 8) + model->gru_biases[l][0][i]);
                 int8_t h_val = my_s_ptr[HIDDEN_DIM + i];
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 my_s_ptr[2*HIDDEN_DIM + i] = gated; // Buf 2
                 my_s_ptr[3*HIDDEN_DIM + i] = ft;    // Buf 3
            }
            __syncwarp();
            
            // Rank-1 M3 & M4
            long long ls_3=0, ls_4=0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b3 = noise_from_hash(seed + SEED_OFF_GRU_M3 + HIDDEN_DIM, i);
                ls_3 += (long long)my_s_ptr[i] * b3;
                
                int8_t b4 = noise_from_hash(seed + SEED_OFF_GRU_M4 + HIDDEN_DIM, i);
                ls_4 += (long long)my_s_ptr[2*HIDDEN_DIM + i] * b4;
            }
            long long xB_m3 = warpReduceSum64(ls_3);
            long long xB_m4 = warpReduceSum64(ls_4);
            
            // Phase 2
            const int8_t *w2 = &model->gru_weights[l][2][0];
            const int8_t *w3 = &model->gru_weights[l][3][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long dot1 = 0;
                long long dot2 = 0;
                
                for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w2[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                }
                
                int8_t a3 = noise_from_hash(seed + SEED_OFF_GRU_M3, i);
                if(ns!=0) dot1 += ((xB_m3 * (long long)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t a4 = noise_from_hash(seed + SEED_OFF_GRU_M4, i);
                if(ns!=0) dot2 += ((xB_m4 * (long long)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t ht = clip((dot1 >> 8) + (dot2 >> 8) + model->gru_biases[l][1][i]);
                
                int8_t ft = my_s_ptr[3*HIDDEN_DIM + i];
                int8_t h_curr = h_local[l][i / WARP_SIZE];
                int32_t diff = ht - h_curr;
                int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                h_curr = clip(h_curr + update);
                h_local[l][i / WARP_SIZE] = h_curr;
                
                my_s_ptr[i] = clip((long long)h_curr + gru_resid[i / WARP_SIZE]); 
            }
            __syncwarp(); 
            
            // Pre-LN Resid
            int8_t mlp_resid[MAX_STRIDE];
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                mlp_resid[i / WARP_SIZE] = my_s_ptr[i];
            }

            // MLP LN 2
            long long local_mlp_sum = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                local_mlp_sum += abs((long long)my_s_ptr[i]);
            }
            long long sum_mlp = warpReduceSum64(local_mlp_sum);
            if(!sum_mlp) sum_mlp=1; long long mean_mlp = sum_mlp/HIDDEN_DIM; if(!mean_mlp) mean_mlp=1;

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * model->ln_weights[l][1][i]) / mean_mlp);
            }
            __syncwarp(); 
            
            // Expand
            int8_t mlp_res[MAX_STRIDE][4];
            
            // Rank-1 xB
            uint32_t seed_exp = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
            long long ls_exp = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b_val = noise_from_hash(seed_exp + SHARED_STRIDE, i); 
                ls_exp += (long long)my_s_ptr[i] * b_val;
            }
            long long xB_mlp1 = warpReduceSum64(ls_exp);

            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                      int out_idx = i + sub * HIDDEN_DIM;
                      long long acc_mlp = 0;
                      
                      for(int k=0; k<HIDDEN_DIM; k++) {
                          acc_mlp += (long long)my_s_ptr[k] * w_mlp1[k*(4*HIDDEN_DIM) + out_idx];
                      }
                      
                      int8_t a_val = noise_from_hash(seed_exp, out_idx);
                      if(ns!=0) acc_mlp += ((xB_mlp1 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                      mlp_res[i / WARP_SIZE][sub] = clip(acc_mlp >> 8);
                 }
             }
             __syncwarp();
             
             // Write Expand to Shared
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     my_s_ptr[i + sub*HIDDEN_DIM] = mlp_res[i / WARP_SIZE][sub];
                 }
             }
             __syncwarp();
             
             // xB for M2 (Input 4*HIDDEN)
             uint32_t seed_proj = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
             long long ls_proj = 0;
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     int in_idx = i + sub*HIDDEN_DIM;
                     int8_t b_val = noise_from_hash(seed_proj + HIDDEN_DIM, in_idx);
                     ls_proj += (long long)my_s_ptr[in_idx] * b_val;
                 }
             }
             long long xB_mlp2 = warpReduceSum64(ls_proj);
             
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long acc_proj = 0;
                 
                 for(int k=0; k<HIDDEN_DIM*4; k++) {
                     acc_proj += (long long)my_s_ptr[k] * w_mlp2[k*HIDDEN_DIM + i];
                 }
                 
                 int8_t a_val = noise_from_hash(seed_proj, i);
                 if(ns!=0) acc_proj += ((xB_mlp2 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int32_t res = acc_proj >> 8; 
                 my_s_ptr[i] = clip(res + mlp_resid[i / WARP_SIZE]);
             }
             __syncwarp();
        } 
        
        // 3. Head LN
        long long local_sum = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            local_sum += abs((long long)my_s_ptr[i]);
        }
        long long sum = warpReduceSum64(local_sum);
        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             my_s_ptr[i] = clip(((long long)my_s_ptr[i] * model->ln_out[i]) / mean);
        }
        __syncwarp();
        
        // 4. Head Dense & Loss
         uint32_t seed_head = (step_seed+pair_idx) + SEED_OFF_HEAD;
         long long ls_head = 0;
         for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             int8_t b_val = noise_from_hash(seed_head + VOCAB_SIZE, i); 
             ls_head += (long long)my_s_ptr[i] * b_val;
         }
         long long xB_head = warpReduceSum64(ls_head);
         
         const int8_t *w_head = &model->head[0];
         
         // Logits: KERNEL_LOOP not needed -> Warp Loop over VOCAB_SIZE
         // Use lane_id similarly. VOCAB_SIZE = 256.
         // 256 / 32 = 8 iters per thread.
         for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
             long long acc = 0;
             for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)my_s_ptr[k] * w_head[k*VOCAB_SIZE + i];
             }
            
             int8_t a_val = noise_from_hash(seed_head, i);
             if(ns!=0) acc += ((xB_head * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
             
             // Store logit temporarily in shared mem (offset HIDDEN_DIM)
             my_s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
         }
         __syncwarp();
         
          // 5. Softmax Loss
          uint8_t target_token = dataset[stream_pos + t + 1];
          
          long long local_exp = 0;
          for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
              int idx = (int32_t)my_s_ptr[HIDDEN_DIM + i] + 128;
              idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
              local_exp += d_EXP2_TABLE[idx];
          }
          long long sum_exp = warpReduceSum64(local_exp);
          
          // Only lane 0 needs to compute final log_sum for this batch item
          if(lane_id == 0) {
             long long log_sum = 0;
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
             
             int32_t target_l = (int32_t)my_s_ptr[HIDDEN_DIM + target_token] + 128;
             my_loss += (log_sum - target_l);
          }
    } // Seq Loop
    
    if (lane_id == 0) {
        accum_loss[p_idx] = (int32_t)my_loss;
    }
    
    // Store State
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE; 
        for(int l=0; l<N_LAYERS; l++) {
             pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = h_local[l][sub];
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
        
        // New Grid Config
        int threads_per_block = BLOCK_THREADS;
        int warps_per_block = BLOCK_THREADS / WARP_SIZE;
        int blocks = POPULATION_SIZE / warps_per_block;
        size_t shared_mem_size = warps_per_block * SHARED_STRIDE;
        
        train_sequence_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
        
        // Update embedding (column-major: rows=HIDDEN_DIM, cols=VOCAB_SIZE)
        // Forward uses: A indexed by token (col), B indexed by feature (row)
        // So offset_A maps to col (token), offset_B maps to row (feature)
        long off_emb = offsetof(EggModel, embedding);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_emb, HIDDEN_DIM, VOCAB_SIZE, 
            HIDDEN_DIM, 0,  // Swapped: offset_A=HIDDEN_DIM (for B/feature), offset_B=0 (for A/token)
            SEED_OFF_EMB, d_fitnesses, seed
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
