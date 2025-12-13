#ifndef EGG_MATH_NTT_CUH
#define EGG_MATH_NTT_CUH

/**
 * egg_ntt.cuh - Number-Theoretic Transform / Walsh-Hadamard Transform
 * 
 * LUT-free implementations using bit-shift arithmetic for modular reduction.
 * All transforms work in-place on shared memory arrays.
 */

#include <cuda_runtime.h>
#include "../config.h"

// ============================================================================
// Configuration
// ============================================================================

// Compile-time power-of-2 check for SEQ_LEN
#if NTT_MODE != 0
#  ifndef SEQ_LEN
#    error "SEQ_LEN must be defined before including egg_ntt.cuh"
#  endif
#  define NTT_IS_POW2(x) (((x) & ((x) - 1)) == 0)
#  if !NTT_IS_POW2(SEQ_LEN)
#    error "NTT_MODE requires SEQ_LEN to be a power of 2"
#  endif
#endif

// Mode-specific constants
#if NTT_MODE == 2
#  define NTT_PRIME 257
#  define NTT_ROOT 3          // Primitive 256th root of unity mod 257
#  define NTT_ROOT_INV 86     // Inverse of 3 mod 257
#  if SEQ_LEN > 256
#    error "Fermat-257 mode requires SEQ_LEN <= 256"
#  endif
#elif NTT_MODE == 3
#  define NTT_PRIME 65537
#  define NTT_ROOT 3          // Primitive 65536th root of unity mod 65537
#  define NTT_ROOT_INV 21846  // Inverse of 3 mod 65537
#  if SEQ_LEN > 65536
#    error "Fermat-65537 mode requires SEQ_LEN <= 65536"
#  endif
#endif

// ============================================================================
// Bit manipulation helpers
// ============================================================================

// Count trailing zeros (log2 for powers of 2)
__device__ __forceinline__ int ntt_ctz(int x) {
    return __ffs(x) - 1;  // CUDA intrinsic: find first set bit
}

// Bit-reverse an index for in-place NTT
__device__ __forceinline__ int ntt_bit_reverse(int x, int log2_n) {
    int result = 0;
    for (int i = 0; i < log2_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ============================================================================
// Walsh-Hadamard Transform (Mode 1)
// No modular arithmetic, just additions and subtractions
// ============================================================================

#if NTT_MODE == 1

__device__ __forceinline__ void wht_butterfly(int32_t &a, int32_t &b) {
    int32_t t = a;
    a = a + b;
    b = t - b;
}

/**
 * In-place Walsh-Hadamard Transform
 * Requires all threads in block to participate
 * @param data  Shared memory array of length `len`
 * @param len   Transform length (must be power of 2)
 * @param tid   Thread index (0 to len-1)
 */
__device__ void wht_transform_inplace(int32_t *data, int len, int tid) {
    // Iterative Cooley-Tukey style WHT
    for (int stride = 1; stride < len; stride <<= 1) {
        __syncthreads();
        
        // Each thread handles one butterfly per stage
        int pair_idx = tid / stride;
        int offset = tid % stride;
        int base = (pair_idx * 2) * stride + offset;
        
        if (base + stride < len) {
            wht_butterfly(data[base], data[base + stride]);
        }
    }
    __syncthreads();
}

/**
 * In-place Inverse Walsh-Hadamard Transform
 * WHT is its own inverse up to scaling: IWHT(x) = WHT(x) / N
 */
__device__ void wht_inverse_inplace(int32_t *data, int len, int tid) {
    wht_transform_inplace(data, len, tid);
    
    __syncthreads();
    
    // Scale by 1/N (using right shift for power-of-2)
    int log2_len = ntt_ctz(len);
    if (tid < len) {
        // Rounding: add half before shift
        data[tid] = (data[tid] + (1 << (log2_len - 1))) >> log2_len;
    }
}

#endif // NTT_MODE == 1

// ============================================================================
// Fermat Prime NTT: p = 2^k + 1 (Mode 2: p=257, Mode 3: p=65537)
// Uses bit-shift tricks for fast modular reduction
// ============================================================================

#if NTT_MODE == 2 || NTT_MODE == 3

#if NTT_MODE == 2
// Fermat-257: p = 2^8 + 1

/**
 * Fast modular reduction mod 257
 * x mod 257 = (x & 0xFF) - (x >> 8), with wraparound correction
 */
__device__ __forceinline__ int32_t mod257(int32_t x) {
    // Handle negative inputs
    while (x < 0) x += 257;
    
    // Reduce: x = lo - hi where x = hi * 256 + lo
    int32_t lo = x & 0xFF;
    int32_t hi = x >> 8;
    int32_t r = lo - hi;
    
    // If negative, add 257
    r += (r >> 31) & 257;
    
    // Final check for r == 257
    return (r >= 257) ? (r - 257) : r;
}

/**
 * Modular multiplication mod 257
 * Uses the fact that for p = 2^8 + 1:
 * (a * b) mod p can overflow int32, so we reduce carefully
 */
__device__ __forceinline__ int32_t mulmod257(int32_t a, int32_t b) {
    int32_t prod = a * b;  // Max: 256 * 256 = 65536, fits in int32
    return mod257(prod);
}

/**
 * Compute ω^k mod 257 where ω = 3 (primitive 256th root)
 * LUT-free: uses binary exponentiation
 */
__device__ int32_t pow_omega257(int k) {
    k = k & 255;  // k mod 256 (periodicity of roots)
    if (k == 0) return 1;
    
    int32_t result = 1;
    int32_t base = NTT_ROOT;  // ω = 3
    
    while (k > 0) {
        if (k & 1) {
            result = mulmod257(result, base);
        }
        base = mulmod257(base, base);
        k >>= 1;
    }
    return result;
}

/**
 * Compute ω^{-k} mod 257 for inverse NTT
 */
__device__ int32_t pow_omega_inv257(int k) {
    k = k & 255;
    if (k == 0) return 1;
    
    int32_t result = 1;
    int32_t base = NTT_ROOT_INV;  // ω^{-1} = 86
    
    while (k > 0) {
        if (k & 1) {
            result = mulmod257(result, base);
        }
        base = mulmod257(base, base);
        k >>= 1;
    }
    return result;
}

#define NTT_MOD mod257
#define NTT_MULMOD mulmod257
#define NTT_POW_OMEGA pow_omega257
#define NTT_POW_OMEGA_INV pow_omega_inv257
#define NTT_P 257
#define NTT_LOG2_MAX 8

#elif NTT_MODE == 3
// Fermat-65537: p = 2^16 + 1

/**
 * Fast modular reduction mod 65537
 * x mod 65537 = (x & 0xFFFF) - (x >> 16), with wraparound
 */
__device__ __forceinline__ int32_t mod65537(int32_t x) {
    // Handle negative inputs
    while (x < 0) x += 65537;
    
    int32_t lo = x & 0xFFFF;
    int32_t hi = x >> 16;
    int32_t r = lo - hi;
    
    r += (r >> 31) & 65537;
    return (r >= 65537) ? (r - 65537) : r;
}

/**
 * Modular multiplication mod 65537
 * Need 64-bit intermediate for large values
 */
__device__ __forceinline__ int32_t mulmod65537(int32_t a, int32_t b) {
    int64_t prod = (int64_t)a * b;
    
    // Reduce: prod mod 65537
    int32_t lo = (int32_t)(prod & 0xFFFF);
    int32_t hi = (int32_t)(prod >> 16) & 0xFFFF;
    int32_t top = (int32_t)(prod >> 32);
    
    // x mod (2^16+1): lo - hi + top (with wraparound)
    int32_t r = lo - hi + top;
    while (r < 0) r += 65537;
    while (r >= 65537) r -= 65537;
    return r;
}

/**
 * Compute ω^k mod 65537 where ω = 3
 */
__device__ int32_t pow_omega65537(int k) {
    k = k & 65535;  // k mod 65536
    if (k == 0) return 1;
    
    int32_t result = 1;
    int32_t base = NTT_ROOT;
    
    while (k > 0) {
        if (k & 1) {
            result = mulmod65537(result, base);
        }
        base = mulmod65537(base, base);
        k >>= 1;
    }
    return result;
}

__device__ int32_t pow_omega_inv65537(int k) {
    k = k & 65535;
    if (k == 0) return 1;
    
    int32_t result = 1;
    int32_t base = NTT_ROOT_INV;
    
    while (k > 0) {
        if (k & 1) {
            result = mulmod65537(result, base);
        }
        base = mulmod65537(base, base);
        k >>= 1;
    }
    return result;
}

#define NTT_MOD mod65537
#define NTT_MULMOD mulmod65537
#define NTT_POW_OMEGA pow_omega65537
#define NTT_POW_OMEGA_INV pow_omega_inv65537
#define NTT_P 65537
#define NTT_LOG2_MAX 16

#endif // NTT_MODE == 2 vs 3

/**
 * NTT Butterfly operation
 * (a, b) <- (a + ω^k * b, a - ω^k * b) mod p
 */
__device__ __forceinline__ void ntt_butterfly(int32_t &a, int32_t &b, int32_t w) {
    int32_t wb = NTT_MULMOD(w, b);
    int32_t a_new = NTT_MOD(a + wb);
    int32_t b_new = NTT_MOD(a - wb + NTT_P);  // Add p to avoid negative
    a = a_new;
    b = b_new;
}

/**
 * In-place Cooley-Tukey NTT (decimation-in-time)
 * Requires bit-reversal permutation first
 */
__device__ void ntt_dit_inplace(int32_t *data, int len, int tid, bool inverse) {
    int log2_len = ntt_ctz(len);
    
    // Bit-reversal permutation (all threads cooperate)
    __syncthreads();
    if (tid < len) {
        int rev = ntt_bit_reverse(tid, log2_len);
        if (tid < rev) {
            int32_t tmp = data[tid];
            data[tid] = data[rev];
            data[rev] = tmp;
        }
    }
    __syncthreads();
    
    // Butterfly passes
    for (int s = 1; s <= log2_len; s++) {
        int m = 1 << s;           // Butterfly group size
        int m2 = m >> 1;          // Half group size
        
        // Compute twiddle factor step: ω^{N/m} for forward, ω^{-N/m} for inverse
        int w_step = len >> s;
        
        __syncthreads();
        
        // Each thread handles butterflies in its range
        for (int k = tid; k < len / 2; k += blockDim.x) {
            int group = k / m2;
            int j = k % m2;
            int idx1 = group * m + j;
            int idx2 = idx1 + m2;
            
            int32_t w = inverse ? NTT_POW_OMEGA_INV(j * w_step) : NTT_POW_OMEGA(j * w_step);
            ntt_butterfly(data[idx1], data[idx2], w);
        }
    }
    __syncthreads();
    
    // For inverse NTT, divide by N
    if (inverse && tid < len) {
        // Compute N^{-1} mod p using Fermat's little theorem: N^{-1} = N^{p-2} mod p
        // For small N (power of 2), we can compute this
        int32_t n_inv;
        
        #if NTT_MODE == 2
        // Precomputed inverses for common sizes mod 257
        switch(len) {
            case 2:   n_inv = 129; break;  // 2^{-1} mod 257
            case 4:   n_inv = 193; break;  // 4^{-1} mod 257
            case 8:   n_inv = 225; break;  // 8^{-1} mod 257
            case 16:  n_inv = 241; break;  // 16^{-1} mod 257
            case 32:  n_inv = 249; break;  // 32^{-1} mod 257
            case 64:  n_inv = 253; break;  // 64^{-1} mod 257
            case 128: n_inv = 255; break;  // 128^{-1} mod 257
            case 256: n_inv = 256; break;  // 256^{-1} mod 257 = -1 mod 257 = 256
            default:  n_inv = 1; break;
        }
        #elif NTT_MODE == 3
        // For 65537, use similar precomputation or compute at runtime
        // N^{-1} = N^{65535} mod 65537 (expensive, but only done once)
        // For common power-of-2 sizes:
        switch(len) {
            case 2:     n_inv = 32769; break;
            case 4:     n_inv = 49153; break;
            case 8:     n_inv = 57345; break;
            case 16:    n_inv = 61441; break;
            case 32:    n_inv = 63489; break;
            case 64:    n_inv = 64513; break;
            case 128:   n_inv = 65025; break;
            case 256:   n_inv = 65281; break;
            case 512:   n_inv = 65409; break;
            case 1024:  n_inv = 65473; break;
            case 2048:  n_inv = 65505; break;
            case 4096:  n_inv = 65521; break;
            case 8192:  n_inv = 65529; break;
            case 16384: n_inv = 65533; break;
            case 32768: n_inv = 65535; break;
            case 65536: n_inv = 65536; break;
            default:    n_inv = 1; break;
        }
        #endif
        
        data[tid] = NTT_MULMOD(data[tid], n_inv);
    }
    __syncthreads();
}

/**
 * Forward Fermat NTT
 */
__device__ void fermat_ntt_inplace(int32_t *data, int len, int tid) {
    // Ensure all input values are in range [0, p-1]
    if (tid < len) {
        data[tid] = NTT_MOD(data[tid]);
    }
    __syncthreads();
    
    ntt_dit_inplace(data, len, tid, false);
}

/**
 * Inverse Fermat NTT
 */
__device__ void fermat_intt_inplace(int32_t *data, int len, int tid) {
    ntt_dit_inplace(data, len, tid, true);
}

#endif // NTT_MODE == 2 || NTT_MODE == 3

// ============================================================================
// Unified API (works for all modes)
// ============================================================================

/**
 * Apply forward transform (dispatches to appropriate implementation)
 */
__device__ void ntt_transform_sequence(int32_t *s_data, int len, int tid) {
#if NTT_MODE == 0
    // No-op: transform disabled
    (void)s_data; (void)len; (void)tid;
#elif NTT_MODE == 1
    wht_transform_inplace(s_data, len, tid);
#elif NTT_MODE == 2 || NTT_MODE == 3
    fermat_ntt_inplace(s_data, len, tid);
#endif
}

/**
 * Apply inverse transform
 */
__device__ void ntt_inverse_transform_sequence(int32_t *s_data, int len, int tid) {
#if NTT_MODE == 0
    (void)s_data; (void)len; (void)tid;
#elif NTT_MODE == 1
    wht_inverse_inplace(s_data, len, tid);
#elif NTT_MODE == 2 || NTT_MODE == 3
    fermat_intt_inplace(s_data, len, tid);
#endif
}

/**
 * Normalize a transform coefficient to int8 range [-127, 127]
 * Different transforms have different output ranges:
 *   - WHT: coefficients can be up to N * max_input = SEQ_LEN * 255
 *   - Fermat NTT: coefficients in [0, p-1]
 */
__device__ __forceinline__ int8_t ntt_normalize_coefficient(int32_t coeff, int len) {
#if NTT_MODE == 0
    (void)len;
    return (int8_t)coeff;
#elif NTT_MODE == 1
    // WHT: range is roughly [-N*128, N*128], normalize to [-127, 127]
    // Use shift for division by N (power of 2)
    int log2_len = ntt_ctz(len);
    int32_t scaled = coeff >> log2_len;  // Divide by N
    // Clamp to int8 range
    if (scaled > 127) return 127;
    if (scaled < -127) return -127;
    return (int8_t)scaled;
#elif NTT_MODE == 2
    // Fermat-257: range [0, 256], map to [-127, 128]
    (void)len;
    int32_t centered = coeff - 128;  // Center around 0
    if (centered > 127) return 127;
    if (centered < -127) return -127;
    return (int8_t)centered;
#elif NTT_MODE == 3
    // Fermat-65537: range [0, 65536], scale down
    (void)len;
    int32_t scaled = (coeff >> 8) - 128;  // Divide by 256, center
    if (scaled > 127) return 127;
    if (scaled < -127) return -127;
    return (int8_t)scaled;
#endif
}

/**
 * Get a single coefficient from a byte sequence (for windowed/on-demand use)
 * Computes only the requested coefficient without full transform
 */
__device__ int32_t ntt_get_single_coefficient(const uint8_t *sequence, int len, int coeff_idx) {
#if NTT_MODE == 0
    (void)len; (void)coeff_idx;
    return (int32_t)sequence[0];
#elif NTT_MODE == 1
    // WHT single coefficient: sum/difference pattern based on bit representation
    int32_t result = 0;
    for (int i = 0; i < len; i++) {
        // Sign is determined by popcount of (i AND coeff_idx)
        int bits = __popc(i & coeff_idx);
        int sign = (bits & 1) ? -1 : 1;
        result += sign * (int32_t)sequence[i];
    }
    return result;
#elif NTT_MODE == 2 || NTT_MODE == 3
    // NTT single coefficient: X[k] = sum_n x[n] * omega^{nk}
    int32_t result = 0;
    for (int n = 0; n < len; n++) {
        int32_t w = NTT_POW_OMEGA((n * coeff_idx) % len);
        result = NTT_MOD(result + NTT_MULMOD((int32_t)sequence[n], w));
    }
    return result;
#endif
}

/**
 * Check if NTT is enabled
 */
__device__ __host__ __forceinline__ bool ntt_is_enabled() {
    return NTT_MODE != 0;
}

/**
 * Get transform name for logging
 */
__host__ inline const char* ntt_get_mode_name() {
#if NTT_MODE == 0
    return "Disabled";
#elif NTT_MODE == 1
    return "Walsh-Hadamard";
#elif NTT_MODE == 2
    return "Fermat-257";
#elif NTT_MODE == 3
    return "Fermat-65537";
#else
    return "Unknown";
#endif
}

// ============================================================================
// Optional: Hybrid embedding helper
// Combines original byte embedding with NTT coefficient embedding
// ============================================================================

/**
 * Structure to hold both original and transformed data for hybrid embedding
 */
struct NttHybridInput {
    int32_t original;      // Original byte value
    int32_t transformed;   // NTT/WHT coefficient
};

/**
 * Prepare hybrid input at a position
 * Call after ntt_transform_sequence on full sequence
 */
__device__ __forceinline__ NttHybridInput ntt_prepare_hybrid(
    uint8_t original_byte,
    int32_t transformed_coeff,
    int len
) {
    NttHybridInput input;
    input.original = (int32_t)original_byte;
    input.transformed = ntt_normalize_coefficient(transformed_coeff, len);
    return input;
}

#endif // EGG_MATH_NTT_CUH
