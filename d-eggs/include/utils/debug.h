#ifndef EGG_UTILS_DEBUG_H
#define EGG_UTILS_DEBUG_H

#include <stdio.h>
#include <math.h>

#ifdef EGG_DEBUG

// --- ANSI Colors ---
#define C_RST  "\033[0m"
#define C_RED  "\033[31m"
#define C_GRN  "\033[32m"
#define C_YEL  "\033[33m"
#define C_BLU  "\033[34m"
#define C_MAG  "\033[35m"
#define C_CYN  "\033[36m"
#define C_WHT  "\033[37m"
#define C_GRA  "\033[90m"
#define C_BOLD "\033[1m"

// --- Activation Statistics ---
__device__ inline void debug_stat_impl(long step, int t, int l, const char* name, const int8_t* data, int len) {
    // Only print sample steps
    if (step != 0 && step % 10 != 0) return;

    float sum = 0, sum_sq = 0;
    int sat = 0, zero = 0;
    int min_v = 127, max_v = -127;
    int hist[256];
    for(int i=0; i<256; i++) hist[i] = 0;

    for(int i=0; i<len; i++) {
        int val = (int)data[i];
        sum += val;
        sum_sq += val * val;
        if (abs(val) >= 126) sat++;
        if (val == 0) zero++;
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
        hist[val + 128]++;
    }

    float mean = sum / len;
    float variance = (sum_sq / len) - (mean * mean);
    float std_dev = sqrtf(variance > 0 ? variance : 0);

    // Entropy
    float entropy = 0;
    for(int i=0; i<256; i++) {
        if (hist[i] > 0) {
            float p = (float)hist[i] / len;
            entropy -= p * log2f(p);
        }
    }
    
    // Calculate entropy as % of maximum (log2(256) = 8.0)
    float max_possible_entropy = 8.0f;
    float entropy_pct = (entropy / max_possible_entropy) * 100.0f;

    // Determine colors
    int sat_pct = (sat * 100) / len;
    const char* sat_col = (sat_pct > 10) ? C_RED : (sat_pct > 0) ? C_YEL : C_GRA;
    
    int zero_pct = (zero * 100) / len;
    const char* zero_col = (zero_pct > 50) ? C_BLU : (zero_pct > 10) ? C_CYN : C_GRA;
    
    const char* rng_col = (min_v <= -127 || max_v >= 127) ? C_RED : C_GRN;

    printf(C_GRA "[S%5ld L%2d T%3d] " C_WHT "%-14s" C_RST ": "
           "Mean:" C_BOLD "%5.1f" C_RST " Std:" C_BOLD "%4.1f" C_RST " "
           "Ent:" C_MAG "%4.2f" C_RST "(" C_MAG "%3.0f%%" C_RST ") "
           "%sSat:%3d%%%s %sZero:%3d%%%s "
           "%sRange:[%4d..%4d]" C_RST "\n",
           step, l, t, name, 
           mean, std_dev, 
           entropy, entropy_pct,
           sat_col, sat_pct, C_RST, 
           zero_col, zero_pct, C_RST, 
           rng_col, min_v, max_v);
}

// --- Attention Statistics ---
struct AttnDebugState {
    float sum_w_log_w;
    float sum_w;
    int32_t max_w;
    int count;
};

__device__ inline void debug_attn_init_impl(AttnDebugState* s) {
    s->sum_w_log_w = 0;
    s->sum_w = 0;
    s->max_w = 0;
    s->count = 0;
}

__device__ inline void debug_attn_accum_impl(AttnDebugState* s, int32_t wt) {
    if (wt > 0) {
        float wf = (float)wt;
        s->sum_w += wf;
        s->sum_w_log_w += wf * log2f(wf);
        if (wt > s->max_w) s->max_w = wt;
        s->count++;
    }
}

__device__ inline void debug_attn_finish_impl(AttnDebugState* s, long step, int l, int h, int t) {
    if (step != 0 && step % 10 != 0) return;
    
    // Entropy = log(S) - (Sum(w log w) / S)
    float entropy = 0;
    float entropy_pct = 0.0f;
    
    if (s->sum_w > 0) {
        entropy = log2f(s->sum_w) - (s->sum_w_log_w / s->sum_w);
    }
    
    // Max entropy is log2(N) where N is context length (approx count)
    if (s->count > 1) {
        float max_possible = log2f((float)s->count);
        entropy_pct = (entropy / max_possible) * 100.0f;
    }
    
    // Only print Head 0
    if (h == 0) { 
        printf(C_GRA "[S%5ld L%2d T%3d H%d] " C_YEL "Attn          " C_RST ": "
               "Ent:" C_MAG "%4.2f" C_RST "(" C_MAG "%3.0f%%" C_RST ") "
               "MaxW:" C_CYN "%6d" C_RST " SumW:" C_GRN "%8.0f" C_RST " Ctx:" C_BOLD "%4d" C_RST "\n",
               step, l, t, h, entropy, entropy_pct, s->max_w, s->sum_w, s->count);
    }
}

// --- Attention Probe (Raw Values) ---
__device__ inline void debug_attn_probe_impl(long step, int l, int h, int t, int ctx, int32_t sc, int32_t max, int32_t shifted, int32_t wt) {
    // Inspect Layer 0, Head 0, at Token 10, for the first 10 context positions
    // Reduced frequency to avoid spamming: step % 5 == 0
    if (step % 5 == 0 && l == 0 && h == 0 && t == 10 && ctx < 10) {
        printf(C_RED "ATTN_PROBE     " C_GRA "[S%5ld L%2d T%3d] " C_RST 
               "Ctx%3d: Raw:" C_CYN "%6d" C_RST " Max:" C_CYN "%6d" C_RST " -> Shft:" C_YEL "%6d" C_RST " -> Wt:" C_GRN "%6d" C_RST "\n", 
               step, l, t, ctx, sc, max, shifted, wt);
    }
}

// --- Macros ---
// Condition: Thread 0 only, Block 0 only, T=32 only (to sample a decently long sequence)
#define EGG_TRACE_COND (blockIdx.x == 0 && threadIdx.x == 0 && t == 32 && global_pop_offset == 0)
// For Attention, we need to init inside the head loop (tid loops over heads)
// But Thread 0 handles Head 0. Thread 1 handles Head 1.
// We only want to debug Head 0. So tid==0 is correct for Head 0.
#define EGG_ATTN_COND (blockIdx.x == 0 && threadIdx.x == 0 && t == 32 && global_pop_offset == 0)

#define EGG_TRACE_STAT(step, t, l, name, data, len) \
    if (EGG_TRACE_COND) { \
        debug_stat_impl(step, t, l, name, (const int8_t*)data, len); \
    }

#define EGG_TRACE_ATTN_DECL(name) AttnDebugState name
#define EGG_TRACE_ATTN_INIT(name) if (EGG_ATTN_COND) debug_attn_init_impl(&name)
#define EGG_TRACE_ATTN_ACCUM(name, wt) if (EGG_ATTN_COND) debug_attn_accum_impl(&name, wt)
#define EGG_TRACE_ATTN_FINISH(name, step, l, h, t) if (EGG_ATTN_COND) debug_attn_finish_impl(&name, step, l, h, t)
#define EGG_TRACE_ATTN_PROBE(step, l, h, t, ctx, sc, max, shifted, wt) \
    if (blockIdx.x == 0 && threadIdx.x == 0 && global_pop_offset == 0) { \
        debug_attn_probe_impl(step, l, h, t, ctx, sc, max, shifted, wt); \
    }

#else

#define EGG_TRACE_STAT(step, t, l, name, data, len)
#define EGG_TRACE_ATTN_DECL(name)
#define EGG_TRACE_ATTN_INIT(name)
#define EGG_TRACE_ATTN_ACCUM(name, wt)
#define EGG_TRACE_ATTN_FINISH(name, step, l, h, t)
#define EGG_TRACE_ATTN_PROBE(step, l, h, t, ctx, sc, max, shifted, wt)

#endif // EGG_DEBUG

#endif // EGG_UTILS_DEBUG_H
