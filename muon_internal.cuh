#ifndef MUON_INTERNAL_CUH
#define MUON_INTERNAL_CUH

#ifdef USE_MUON

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// --- Muon Kernels ---

// 1. Momentum Update Kernel (Optimization Step 1)
// Calculates the gradient estimate (vote) and updates momentum 'm'.
// Does NOT update weights or other parameters.
__global__ void muon_momentum_update_kernel(
    MuonParam *muon_state,
    int rows, int cols, 
    int off_A, int off_B, 
    int seed_base, 
    const int32_t *fitnesses, 
    uint32_t step_seed,
    float momentum_factor // e.g. 0.95
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We don't need to track 'change' here as weights aren't modifying
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
            vote += (VoteType)fit * noise_from_hash(s + off_A, tid) * noise_from_hash(s + off_B, k);
        }
        
        // Update Momentum: m = mu * m + g
        // Note: Using standard SGD momentum accumulation
        MuonParam p = muon_state[idx];
        p.m = momentum_factor * p.m + (float)vote; // Accumulate vote (gradient proxy)
        muon_state[idx] = p;
    }
}

// 2. Gather Kernel
// Copies 'm' from "Blocked" MuonParam layout to "Row-Major" float buffer
// Buffer must be Row-Major for cuBLAS
__global__ void muon_gather_m_kernel(
    const MuonParam *muon_state,
    float *buffer,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        // Decode Blocked Layout
        int byte_off = idx & 3;
        int word_idx = idx >> 2;
        int tid = word_idx % cols; // Col
        int k_chunk = word_idx / cols; 
        int k = k_chunk * 4 + byte_off; // Row
        
        // Write to Standard Row-Major (k * cols + tid)
        if (k < rows && tid < cols) {
             buffer[k * cols + tid] = muon_state[idx].m;
        }
    }
}

// 3. Scale by Norm Kernel (Async Normalization)
// Reads norm from device memory and scales all elements by 1/norm
__global__ void muon_scale_by_norm_kernel(float *X, const float *norm_ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float norm = *norm_ptr;
        if (norm > 1e-6f) {
            X[idx] /= norm;
        }
    }
}

// 4. Newton-Schulz Term Kernel
// Computes B = 3I - A element-wise
// If we compute Gram matrix G = X^T X (Size KxK), we need (3I - G).
// K is the small dimension.
__global__ void muon_newton_schulz_term_kernel(
    float *B,
    const float *A,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K * K) {
        int row = idx / K;
        int col = idx % K;
        float val = A[idx];
        // 3I - A
        if (row == col) {
            B[idx] = 3.0f - val;
        } else {
            B[idx] = -val;
        }
    }
}

// 4. Apply Update Kernel (Optimization Step 2)
// Takes orthogonalized updates from Row-Major buffer, applies to Blocked weights.
__global__ void muon_apply_update_kernel(
    WeightType *W,
    MuonParam *muon_state,
    const float *update_buffer,
    int rows, int cols,
    float learning_rate,
    float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int change = 0;
    int size = rows * cols;
    if (idx < size) {
        // Decode Blocked Layout to find coordinate
        int byte_off = idx & 3;
        int word_idx = idx >> 2;
        int tid = word_idx % cols; // Col
        int k_chunk = word_idx / cols; 
        int k = k_chunk * 4 + byte_off; // Row

        WeightType w = W[idx];
        MuonParam p = muon_state[idx];
        
        // Read update from Row-Major Buffer
        float u = 0.0f;
        if (k < rows && tid < cols) {
            u = update_buffer[k * cols + tid];
        }

        // Muon Update: W -= lr * (O + wd * W)
        // We define 'step' as what we ADD to W.
        // step = -lr * (u + wd * w)
        float step = -learning_rate * (u + weight_decay * (float)w);

        // Accumulate and Quantize
        p.acc += step;
        
        // Apply to Integer Weight (Stochastic Rounding / Accumulator logic)
        if (p.acc >= 1.0f) {
            if (w < MAX_VAL) { w++; change=1; p.acc -= 1.0f; }
            else p.acc = 1.0f; 
        } else if (p.acc <= -1.0f) {
            if (w > MIN_VAL) { w--; change=1; p.acc += 1.0f; }
            else p.acc = -1.0f;
        }
        
        W[idx] = w;
        muon_state[idx] = p;
    }

    if (change) atomicAdd(&s_count, 1);
    __syncthreads();
    
    if (threadIdx.x == 0 && s_count > 0) atomicAdd(&d_total_updates, (unsigned long long)s_count);
}

// --- Muon Internals Class ---

struct MuonWorkspace {
    float *d_buf1;      // Size M*N (Momentum buffer)
    float *d_buf2;      // Size K*K (Gram matrix)
    float *d_buf3;      // Size K*K (Newton-Schulz temp)
    float *d_buf_swap;  // Size M*N (Temp for aliasing avoidance)
    float *d_scalar;    // 1 float for async norm result
};

void perform_newton_schulz(
    cublasHandle_t handle,
    MuonWorkspace &ws,
    float *d_X,     // Input/Output (Momentum buffer)
    int rows, int cols,
    cudaStream_t stream
) {
    // X is (rows x cols).
    // We want to orthogonalize the "short" dimension.
    // Usually Neural Networks weights: Out x In.
    // If Rows < Cols (Fat): Compute G = X * X^T (Size Rows x Rows).
    // If Rows > Cols (Tall): Compute G = X^T * X (Size Cols x Cols).
    
    // We treat X as matrix in column-major for cuBLAS?
    // C arrays are Row-Major. cuBLAS assumes Col-Major.
    // A (Rows x Cols) Row-Major matrix is equivalent to A^T (Cols x Rows) Col-Major.
    // Let's stick to logical dimensions.
    // If we have C-array X[Rows][Cols].
    
    // Case 1: Rows < Cols (e.g. 768 x 3072). Fat.
    // Logic: X * X^T gives (Rows x Rows).
    // In cuBLAS (Col-Major view): This is X_T * (X_T)^T.
    // Wait, confusion alert.
    // Let's interpret the pointer d_X as a flat buffer.
    // We want to approximate isometry.
    
    bool use_rows = (rows < cols);
    int K = use_rows ? rows : cols;
    int L = use_rows ? cols : rows;
    
    // 1. Async Frobenius Norm Scaling
    // Use device pointer mode to avoid implicit sync
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSnrm2(handle, rows * cols, d_X, 1, ws.d_scalar);  // Writes to device memory (async)
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    
    // Launch kernel to scale X by 1/norm (reads d_scalar on device)
    int n_elems = rows * cols;
    muon_scale_by_norm_kernel<<<(n_elems + 255) / 256, 256, 0, stream>>>(d_X, ws.d_scalar, n_elems);

    float alpha = 1.0f;
    float beta = 0.0f;

    // 5 Iterations
    for(int i=0; i<5; i++) {
        // Step A: Compute Gram Matrix
        // If use_rows (Flat, 768x3072): Compute X * X^T (768x768).
        // C-RowMajor: C = A * B.
        // cuBLAS: C^T = B^T * A^T.
        // We want C = X * X^T.
        // cuBLAS sees X^T (3072x768).
        // We want C^T (768x768) = (X * X^T)^T = X * X^T.
        // C_cublas = X_cublas^T * X_cublas.
        // No, let's step back.
        // We use cublasSgemm directly assuming Row Major via trick or just calculate correct op.
        // Easier: Use cublasSgemm with CUBLAS_OP_T/N logic.
        
        // Let's assume d_X is Row Major (Rows x Cols).
        // To cuBLAS it is Col Major (Cols x Rows). Let's call it Mc.
        // Mc has Cols rows, Rows cols.
        
        // Target: G = X * X^T (Rows x Rows).
        // In cuBLAS terms (Mc): Gc = Mc^T * Mc.
        // Mc^T is (Rows x Cols). Mc is (Cols x Rows).
        // Result Gc is (Rows x Rows).
        // Ops: A=Mc, B=Mc. OpA=T, OpB=N.
        // C = alpha * A^T * B.
        
        // Wait, if use_rows (Rows < Cols): K=Rows.
        // A=d_X (Cols x Rows col-major).
        // We need Rows x Rows result.
        // GEMM(Trans, NoTrans): (Rows x Cols) * (Cols x Rows) -> Rows x Rows.
        // Perfect.
        
        // If !use_rows (Cols < Rows): K=Cols.
        // Target: G = X^T * X (Cols x Cols).
        // In cuBLAS (Mc): Gc = Mc * Mc^T.
        // GEMM(NoTrans, Trans): (Cols x Rows) * (Rows x Cols) -> Cols x Cols.
        // Perfect.
        
        cublasOperation_t opA = use_rows ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = use_rows ? CUBLAS_OP_N : CUBLAS_OP_T;
        int m_gemm = K; // Result rows
        int n_gemm = K; // Result cols
        int k_gemm = L; // Inner dim
        
        // G = X * X^T or X^T * X
        alpha = 1.0f; beta = 0.0f;
        // LDA and LDB must both vary based on use_rows:
        // - When use_rows=true (rows < cols): LDA=L, LDB=L (both use col dimension)
        // - When use_rows=false (rows > cols): LDA=K, LDB=K (both use col dimension)
        // The leading dimension is always the number of columns in row-major = rows in col-major view
        int ld = use_rows ? L : K;
        
        cublasSgemm(handle, opA, opB, 
            m_gemm, n_gemm, k_gemm,
            &alpha, 
            d_X, ld,  // A with LDA
            d_X, ld,  // B with LDB (same buffer, same layout)
            &beta,
            ws.d_buf2, K // Result G
        );
        
        // Step B: B = 3I - G
        // Launch kernel on K x K
        int block = 256;
        int grid = (K*K + block - 1) / block;
        muon_newton_schulz_term_kernel<<<grid, block, 0, stream>>>(ws.d_buf3, ws.d_buf2, K);

        // Step C: X_new = 0.5 * X * B (or B * X)
        // If use_rows (G was Rows x Rows): B is Rows x Rows.
        // We need X_new (Rows x Cols) = B * X (Rows x Cols) ? No. 
        // It's X * B ? (Rows x Cols) * (Cols x Cols)? No, B is KxK (Rows x Rows).
        // So B * X involves (Rows x Rows) * (Rows x Cols).
        // Result: Rows x Cols.
        // In cuBLAS Logic (Mc):
        // X_new_c = X_c * B_c^T ?
        // B is symmetric (3I - XXT). B^T = B.
        // X_new_c (Cols x Rows) = X_c (Cols x Rows) * B_c (Rows x Rows).
        // GEMM(N, N): (Cols x Rows) * (Rows x Rows) -> Cols x Rows.
        // m=Cols(L), n=Rows(K), k=Rows(K).
        // Perfect.
        
        // If !use_rows: B is Cols x Cols.
        // X_new = X * B (RowMaj)? (Rows x Cols) * (Cols x Cols).
        // cuBLAS: X_new_c = B_c^T * X_c ?
        // B symmetric. B * X_c ?? No.
        // X_new_c (Cols x Rows) = B (Cols x Cols) * X_c (Cols x Rows).
        // GEMM(N, N): (Cols x Cols) * (Cols x Rows) -> Cols x Rows.
        // m=Cols(K), n=Rows(L), k=Cols(K).
        // Perfect.
        
        alpha = 0.5f; beta = 0.0f;
        if (use_rows) {
             // Mc * B
             // m=L, n=K, k=K
             cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 L, K, K,
                 &alpha,
                 d_X, L,
                 ws.d_buf3, K, // B
                 &beta,
                 ws.d_buf_swap, L // Out Temp (Use Swap to avoid aliasing d_X)
             );
        } else {
             // B * Mc
             // m=K, n=L, k=K
             cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 K, L, K,
                 &alpha,
                 ws.d_buf3, K, // B
                 d_X, K,
                 &beta,
                 ws.d_buf_swap, K // Out Temp (Use Swap to avoid aliasing d_X)
             );
        }

        // Copy back to X
        cudaMemcpyAsync(d_X, ws.d_buf_swap, rows*cols*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
}

#endif // USE_MUON

#endif // MUON_INTERNAL_CUH
