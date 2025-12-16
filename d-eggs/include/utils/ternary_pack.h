#ifndef EGG_TERNARY_PACK_H
#define EGG_TERNARY_PACK_H

#include <stdint.h>
#include <stddef.h>

// Base-3 Packing: 5 ternary values per byte
// Mapping: 0 -> 0, 1 -> 1, -1 -> 2
// Byte = v0 + v1*3 + v2*9 + v3*27 + v4*81
// Max value: 2 + 6 + 18 + 54 + 162 = 242 (fits in uint8_t)

static inline size_t ternary_pack_estimate_size(size_t count) {
    return (count + 4) / 5;
}

static inline void ternary_pack(const int32_t* src, size_t count, uint8_t* dst) {
    size_t i = 0;
    size_t out_idx = 0;
    
    // Process chunks of 5
    while (i + 5 <= count) {
        uint8_t val = 0;
        
        // Map values: -1->2, 0->0, 1->1
        // Formula: (v + 3) % 3
        // -1 -> 2
        // 0 -> 0
        // 1 -> 1
        
        uint8_t v0 = (src[i+0] + 3) % 3;
        uint8_t v1 = (src[i+1] + 3) % 3;
        uint8_t v2 = (src[i+2] + 3) % 3;
        uint8_t v3 = (src[i+3] + 3) % 3;
        uint8_t v4 = (src[i+4] + 3) % 3;
        
        val = v0 + v1*3 + v2*9 + v3*27 + v4*81;
        dst[out_idx++] = val;
        i += 5;
    }
    
    // Handle remainder
    if (i < count) {
        uint8_t val = 0;
        uint8_t mult = 1;
        for (; i < count; i++) {
            uint8_t v = (src[i] + 3) % 3;
            val += v * mult;
            mult *= 3;
        }
        dst[out_idx++] = val;
    }
}

static inline void ternary_unpack(const uint8_t* src, size_t count, int32_t* dst) {
    size_t i = 0;
    size_t in_idx = 0;
    
    // Process chunks of 5
    while (i + 5 <= count) {
        uint8_t val = src[in_idx++];
        
        uint8_t v0 = val % 3; val /= 3;
        uint8_t v1 = val % 3; val /= 3;
        uint8_t v2 = val % 3; val /= 3;
        uint8_t v3 = val % 3; val /= 3;
        uint8_t v4 = val; // Last one
        
        // Map back: 0->0, 1->1, 2->-1
        dst[i+0] = (v0 == 2) ? -1 : v0;
        dst[i+1] = (v1 == 2) ? -1 : v1;
        dst[i+2] = (v2 == 2) ? -1 : v2;
        dst[i+3] = (v3 == 2) ? -1 : v3;
        dst[i+4] = (v4 == 2) ? -1 : v4;
        
        i += 5;
    }
    
    // Handle remainder
    if (i < count) {
        uint8_t val = src[in_idx++];
        for (; i < count; i++) {
            uint8_t v = val % 3;
            val /= 3;
            dst[i] = (v == 2) ? -1 : v;
        }
    }
}

#endif // EGG_TERNARY_PACK_H
