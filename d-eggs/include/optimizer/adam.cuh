#ifndef EGG_OPTIMIZER_ADAM_CUH
#define EGG_OPTIMIZER_ADAM_CUH

#include <cuda_runtime.h>
#include "base.h"
#include "sgd.cuh"
#include "../utils/egg_math.h"
#include "../config.h"
#include "../globals.cuh"

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
    
    apply_quantized_update(w, p.acc, change);
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

#endif // EGG_OPTIMIZER_ADAM_CUH
