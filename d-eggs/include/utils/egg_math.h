#ifndef EGG_UTILS_MATH_H
#define EGG_UTILS_MATH_H

#include <stdint.h>
#include "../config.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define EGG_HOST_DEVICE __host__ __device__
#define EGG_INLINE __forceinline__
#else
#define EGG_HOST_DEVICE
#define EGG_INLINE inline
#endif

EGG_HOST_DEVICE EGG_INLINE uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16; return x;
}

EGG_HOST_DEVICE EGG_INLINE int8_t noise_from_hash(uint32_t s, uint32_t idx) {
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

EGG_HOST_DEVICE EGG_INLINE int8_t clip(int32_t a) { return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a); }

// Host-side RNG for initialization
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

#endif // EGG_UTILS_MATH_H
