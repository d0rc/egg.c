#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <random>
#include "include/utils/ternary_pack.h"

int main() {
    printf("Testing Ternary Pack...\n");
    
    // Test Case 1: Exact multiple of 5
    {
        int32_t src[] = {-1, 0, 1, -1, 0, 1, 1, 0, -1, 0};
        size_t count = 10;
        size_t size = ternary_pack_estimate_size(count);
        assert(size == 2);
        
        uint8_t* packed = (uint8_t*)malloc(size);
        ternary_pack(src, count, packed);
        
        int32_t dst[10];
        ternary_unpack(packed, count, dst);
        
        for(int i=0; i<count; i++) {
            if (src[i] != dst[i]) {
                printf("Mismatch at %d: %d != %d\n", i, src[i], dst[i]);
                return 1;
            }
        }
        free(packed);
        printf("Test 1 Passed\n");
    }
    
    // Test Case 2: Remainder
    {
        int32_t src[] = {-1, 0, 1, -1, 0, 1, 1};
        size_t count = 7;
        size_t size = ternary_pack_estimate_size(count);
        assert(size == 2);
        
        uint8_t* packed = (uint8_t*)malloc(size);
        ternary_pack(src, count, packed);
        
        int32_t dst[7];
        ternary_unpack(packed, count, dst);
        
        for(int i=0; i<count; i++) {
            if (src[i] != dst[i]) {
                printf("Mismatch at %d: %d != %d\n", i, src[i], dst[i]);
                return 1;
            }
        }
        free(packed);
        printf("Test 2 Passed\n");
    }
    
    // Benchmark
    printf("\nRunning Benchmark...\n");
    size_t count = 100 * 1000 * 1000; // 100M values
    printf("Generating %zu random ternary values...\n", count);
    
    std::vector<int32_t> src(count);
    // Use a simple LCG for speed instead of mt19937 for 100M values
    uint32_t state = 42;
    for(size_t i=0; i<count; i++) {
        state = state * 1664525 + 1013904223;
        int v = state % 3; // 0, 1, 2
        src[i] = (v == 2) ? -1 : v;
    }
    
    size_t packed_size = ternary_pack_estimate_size(count);
    std::vector<uint8_t> packed(packed_size);
    std::vector<int32_t> dst(count);
    
    // Measure Packing
    auto start = std::chrono::high_resolution_clock::now();
    ternary_pack(src.data(), count, packed.data());
    auto end = std::chrono::high_resolution_clock::now();
    double pack_time = std::chrono::duration<double>(end - start).count();
    
    printf("Packing Time:   %.4f s\n", pack_time);
    printf("Packing Speed:  %.2f M/s\n", (count / 1e6) / pack_time);
    
    // Measure Unpacking
    start = std::chrono::high_resolution_clock::now();
    ternary_unpack(packed.data(), count, dst.data());
    end = std::chrono::high_resolution_clock::now();
    double unpack_time = std::chrono::duration<double>(end - start).count();
    
    printf("Unpacking Time: %.4f s\n", unpack_time);
    printf("Unpacking Speed: %.2f M/s\n", (count / 1e6) / unpack_time);
    
    // Verify
    printf("Verifying...\n");
    for(size_t i=0; i<count; i++) {
        if (src[i] != dst[i]) {
            printf("Mismatch at %zu: %d != %d\n", i, src[i], dst[i]);
            return 1;
        }
    }
    printf("Verification Passed!\n");
    
    // Stats
    double bits_per_val = (double)packed_size * 8.0 / count;
    printf("\nStats:\n");
    printf("Original Size:  %.2f MB (assuming 32-bit)\n", (double)count * 4 / (1024*1024));
    printf("Packed Size:    %.2f MB\n", (double)packed_size / (1024*1024));
    printf("Efficiency:     %.4f bits/value\n", bits_per_val);
    printf("Theoretical:    1.5850 bits/value (log2(3))\n");
    
    return 0;
}
