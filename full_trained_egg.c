#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

// --- Configuration [cite: 275, 277, 288] ---
#define VOCAB_SIZE 256        // Byte-level tokenization
#define HIDDEN_DIM 128        // Model width
#define N_LAYERS 4            // Number of layers
#define SEQ_LEN 4096            // Sequence length for BPTT (truncated)
#define POPULATION_SIZE 64    // Number of perturbations per step
#define BATCH_SIZE 8          // Parallel streams
#define FIXED_POINT 4         // 4 bits for fractional part
#define SIGMA_SHIFT 4         // Noise scale (bitwise shift)
#define UPDATE_THRESHOLD 80  // Votes needed to flip a weight [cite: 1023]
#define MAX_VAL 127
#define MIN_VAL -127

// --- Lookup Tables [cite: 998-1000] ---
int32_t EXP2_TABLE[256];

// --- Data Structure ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// --- Recurrent State ---
typedef struct {
    int8_t h[N_LAYERS][HIDDEN_DIM];
} RecurrentState;

// --- Model Parameters Struct ---
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; // 0: bf, 1: bh
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; // 0: Expand, 1: Project
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM]; // 0: LN1, 1: LN2
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// --- Helper Functions ---

// Forward Declaration
int32_t compute_loss(int8_t *logits, uint8_t target);

void init_tables() {
    for(int i=0; i<256; i++) 
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// NEON-optimized saturated add for vectors
static inline int8x16_t vclipped_add_s8(int32x4_t a[4], int32x4_t b[4]) {
    // This function signature doesn't really make sense for element-wise int8 add.
    // Usually we just use vqadd_s8 for saturated addition of 8-bit vectors.
    // But the original clipped_add works on 32-bit intermediates.
    // If inputs are 32-bit, we can't easily pack back to 8-bit without checks.
    // However, for standard accumulation, our ranges might exceed int8 range significantly before clipping.
    // Let's stick to specific logic in matmul where we handle accumulation in int32.
    return vdupq_n_s8(0); 
}

// Simple RNG helpers for scalar usage
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_val(uint32_t *rng) {
    uint32_t r = xorshift32(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

// NEON Noise Generation
// Generates 16 int8 values [-15, 15] roughly following the distribution
// Replicates logic: sign * value
static inline void gen_noise_vector_neon(uint32_t *rng, int8_t *out, int count) {
    for (int i = 0; i < count; i += 16) {
        // Generate random bits
        uint32_t r0 = xorshift32(rng);
        uint32_t r1 = xorshift32(rng);
        uint32_t r2 = xorshift32(rng);
        uint32_t r3 = xorshift32(rng);
        
        // We need 16 nibbles + signs. 
        // Actually the original logic: (r&1 ? 1 : -1) * ((r>>1)&15) uses 5 bits per value.
        // We can pack this more efficiently but sticking to xorshift32 is fast enough.
        // Let's scalar generate to a buffer or vectorize the generation logic if bottlenecks.
        // On M4, scalar RNG might be slightly slow compared to vector ops, but matmul is dominant.
        // Let's do a simple block loop.
        for (int j = 0; j < 16; j++) {
             if(i+j < count) out[i+j] = gen_noise_val(rng);
        }
    }
}

// --- The Core "Rank-1 Perturbed Matrix Mul" (NEON Optimized) ---
void matmul_perturbed(
    const int8_t *in, const int8_t *w, int8_t *out, 
    int rows, int cols, 
    uint32_t layer_seed, int noise_sign,
    int shift
) {
    // 1. Generate Noise Vectors A and B
    int8_t A[rows];
    int8_t B[cols];
    
    uint32_t rng = layer_seed;
    gen_noise_vector_neon(&rng, A, rows);
    gen_noise_vector_neon(&rng, B, cols);

    // 2. Compute xB (Projection onto B)
    int32_t xB = 0;
    // Vectorize dot product in * B
    int32x4_t acc_v = vdupq_n_s32(0);
    int i = 0;
    for (; i <= cols - 16; i += 16) {
        int8x16_t in_v = vld1q_s8(&in[i]);
        int8x16_t b_v = vld1q_s8(&B[i]);
        // Multiply int8->int16, accumulate to int32
        int16x8_t mul_low = vmull_s8(vget_low_s8(in_v), vget_low_s8(b_v));
        int16x8_t mul_high = vmull_s8(vget_high_s8(in_v), vget_high_s8(b_v));
        acc_v = vpadalq_s16(acc_v, mul_low);
        acc_v = vpadalq_s16(acc_v, mul_high);
    }
    xB = vaddvq_s32(acc_v);
    // Remainder
    for (; i < cols; i++) {
        xB += (int32_t)in[i] * (int32_t)B[i];
    }

    // 3. Compute Result
    // xW + noise_sign * (xB * A)
    // We iterate over rows (out_dim), computing dot product of in * W[row]
    
    for(int r=0; r<rows; r++) {
        int32_t acc = 0;
        int32x4_t row_acc_v = vdupq_n_s32(0);
        
        const int8_t *w_row = &w[r * cols];
        
        int c = 0;
        for (; c <= cols - 16; c += 16) {
            int8x16_t in_v = vld1q_s8(&in[c]);
            int8x16_t w_v = vld1q_s8(&w_row[c]);
            
            int16x8_t mul_low = vmull_s8(vget_low_s8(in_v), vget_low_s8(w_v));
            int16x8_t mul_high = vmull_s8(vget_high_s8(in_v), vget_high_s8(w_v));
            row_acc_v = vpadalq_s16(row_acc_v, mul_low);
            row_acc_v = vpadalq_s16(row_acc_v, mul_high);
        }
        acc = vaddvq_s32(row_acc_v);
        for (; c < cols; c++) {
            acc += (int32_t)in[c] * (int32_t)w_row[c];
        }

        if (noise_sign != 0) {
            int32_t noise = (xB * (int32_t)A[r]) * noise_sign;
            acc += (noise >> (FIXED_POINT + SIGMA_SHIFT));
        }
        
        int32_t res = acc >> shift;
        if(res > MAX_VAL) out[r] = MAX_VAL;
        else if(res < MIN_VAL) out[r] = MIN_VAL;
        else out[r] = (int8_t)res;
    }
}

// --- Layer Norm [cite: 965] ---
void egg_ln(const int8_t *x, const int8_t *w, int8_t *out) {
    int32_t sum = 0;
    // NEON Sum Abs
    int32x4_t sum_v = vdupq_n_s32(0);
    int i = 0;
    for(; i <= HIDDEN_DIM - 16; i+=16) {
        int8x16_t xv = vld1q_s8(&x[i]);
        int8x16_t abs_xv = vabsq_s8(xv);
        // int8 -> uint16 -> accumulate to uint32/int32?
        // vpadal summation chain needed or simpler:
        // abs is positive, safe to treat as unsigned or signed positive.
        // vpaddl_u8 -> u16, vpaddl_u16 -> u32
        uint16x8_t s1 = vpaddlq_u8(vreinterpretq_u8_s8(abs_xv));
        uint32x4_t s2 = vpaddlq_u16(s1); // accumulates 16 bytes into 4 ints
        sum_v = vaddq_s32(sum_v, vreinterpretq_s32_u32(s2));
    }
    sum = vaddvq_s32(sum_v);
    for(; i<HIDDEN_DIM; i++) sum += abs(x[i]);

    if(sum == 0) sum = 1;
    int32_t mean = sum / HIDDEN_DIM;
    if(mean == 0) mean = 1;
    
    // Vectorize normalization
    for(i=0; i < HIDDEN_DIM; i++) {
        int32_t val = ((int32_t)x[i] * w[i]) / mean;
        if(val > MAX_VAL) out[i] = MAX_VAL;
        else if(val < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)val;
    }
}

// --- Forward Pass (With Noise Injection) [cite: 915] ---
void forward_pass(
    EggModel *model, 
    const uint8_t *inputs, 
    const uint8_t *targets,
    int seq_len, 
    int8_t *logits_out, 
    int32_t *accumulated_loss,
    uint32_t step_seed, 
    int noise_sign,
    RecurrentState *rnn_state
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM * 4], buf2[HIDDEN_DIM];
    
    // If no state provided, use a temporary zeroed one (stateless mode)
    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }
    
    if(accumulated_loss) *accumulated_loss = 0;

    for(int t=0; t<seq_len; t++) {
        // Embedding
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        // Layers
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            // -- GRU --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+1, noise_sign, 8);
            matmul_perturbed(rnn_state->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+2, noise_sign, 8);
            
            int8_t ft[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * rnn_state->h[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+3, noise_sign, 8);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+4, noise_sign, 8);
            
            int8_t ht[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            // State Update
            for(int i=0; i<HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - rnn_state->h[l][i])) >> 8;
                rnn_state->h[l][i] = clipped_add(rnn_state->h[l][i], update);
                x[i] = rnn_state->h[l][i]; // Output is new state
            }
            
            // Residual Add
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            // -- MLP --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);
            
            // Expand: Hidden -> 4*Hidden
            // Cols = HIDDEN_DIM (256). Sqrt(256)=16. Shift = 4+4=8.
            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, noise_sign, 8);
            
            // Project: 4*Hidden -> Hidden
            // Cols = HIDDEN_DIM*4 (1024). Sqrt(1024)=32. Shift = 4+5=9.
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, noise_sign, 9);

            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // Final Head
        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, noise_sign, 8);
        
        if(targets && accumulated_loss) {
            *accumulated_loss += compute_loss(logits_out, targets[t]);
        }
    }
}

// --- Helper for Loss Calculation ---
static inline int get_msb(uint32_t n) {
    int pos = 0;
    if (n >= 1<<16) { n >>= 16; pos += 16; }
    if (n >= 1<<8)  { n >>= 8;  pos += 8; }
    if (n >= 1<<4)  { n >>= 4;  pos += 4; }
    if (n >= 1<<2)  { n >>= 2;  pos += 2; }
    if (n >= 1<<1)  {           pos += 1; }
    return pos;
}

int32_t log2_fixed(int32_t x) {
    if (x <= 0) return 0;
    int k = get_msb(x);
    int32_t fraction;
    if (k >= 4) {
        fraction = (x - (1 << k)) >> (k - 4);
    } else {
        fraction = (x - (1 << k)) << (4 - k);
    }
    return (k << 4) + fraction - 64;
}

int32_t compute_loss(int8_t *logits, uint8_t target) {
    int32_t sum_exp = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        sum_exp += EXP2_TABLE[idx < 0 ? 0 : (idx > 255 ? 255 : idx)];
    }
    int32_t log_sum = log2_fixed(sum_exp);
    int32_t target_logit = (int32_t)logits[target] + 128;
    return log_sum - target_logit;
}

void update_matrix(
    int8_t *W, int rows, int cols, 
    uint32_t seed, 
    const int *fitnesses, 
    int pop_size
) {
    static int32_t votes[65536 * 4];
    memset(votes, 0, rows * cols * sizeof(int32_t));

    int8_t A[rows];
    int8_t B[cols];

    // Parallelize inner reconstruction loops? No, outer loop pop_size logic is easier.
    // But here we are outside the parallel block. We are aggregating.
    // Can we enable parallel "Vote" accumulation?
    // Yes, if we segment the population or the matrix.
    // Or just use simple loop, since updating is less frequent than forward pass logic?
    // Actually update is O(pop_size * weight_size). Forward is O(seq_len * layers * weight_size).
    // Seq_len=128, Pop=256.
    // Update happens ONCE per step.
    // Forward happens POP_SIZE times per step.
    // So Forward optimization is 256x more important. Update is relatively cheap.
    // We keep update single threaded for simplicity or use simple dispatch if needed.
    // Let's stick to serial update for now to avoid race conditions or complex reduction.

    for(int p=0; p<pop_size; p+=2) {
        uint32_t p_seed = seed + (p/2);
        uint32_t rng = p_seed;
        gen_noise_vector_neon(&rng, A, rows);
        gen_noise_vector_neon(&rng, B, cols);

        int f = fitnesses[p/2];
        if (f == 0) continue;

        // Vote = f * (AB^T)
        // Vectorize this outer product update?
        // W[i,j] += f * A[i] * B[j]
        for(int i=0; i<rows; i++) {
            int32_t val_A = A[i] * f; // Premultiply f
            int j = 0;
            int32x4_t val_A_v = vdupq_n_s32(val_A);
            for(; j <= cols - 4; j+=4) {
                // Load B[j..j+3]
                // int8 -> int32
                int8_t *b_ptr = &B[j];
                int32_t b0 = b_ptr[0];
                int32_t b1 = b_ptr[1];
                int32_t b2 = b_ptr[2];
                int32_t b3 = b_ptr[3];
                int32_t b_vals[4] = {b0, b1, b2, b3};
                int32x4_t b_v = vld1q_s32(b_vals);
                int32x4_t prod = vmulq_s32(val_A_v, b_v);
                
                int32_t *vote_ptr = &votes[i*cols + j];
                int32x4_t v_old = vld1q_s32(vote_ptr);
                vst1q_s32(vote_ptr, vaddq_s32(v_old, prod));
            }
            for(; j<cols; j++) {
                votes[i*cols + j] += val_A * B[j];
            }
        }
    }

    for(int i=0; i<rows*cols; i++) {
        if(votes[i] > UPDATE_THRESHOLD && W[i] < MAX_VAL) W[i]++;
        else if(votes[i] < -UPDATE_THRESHOLD && W[i] > MIN_VAL) W[i]--;
    }
}

// --- Sampling Helper ---
#define COLOR_GREEN "\033[32m"
#define COLOR_CYAN  "\033[36m"
#define COLOR_RESET "\033[0m"

int sample_logits(int8_t *logits) {
    int32_t probs[VOCAB_SIZE];
    int32_t sum = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        probs[i] = EXP2_TABLE[idx];
        sum += probs[i];
    }
    if(sum == 0) return 0;
    int32_t r = rand() % sum;
    int32_t acc = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        acc += probs[i];
        if(r < acc) return i;
    }
    return VOCAB_SIZE - 1;
}

void sample_model(EggModel *model, const uint8_t *seed_text, int seed_len, int gen_len) {
    int8_t logits[VOCAB_SIZE];
    RecurrentState state;
    memset(&state, 0, sizeof(state));

    printf(COLOR_GREEN);
    int input_token = 0;
    
    for(int t=0; t < seed_len + gen_len; t++) {
        if (t < seed_len) {
            input_token = seed_text[t];
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        } else {
            if (t == seed_len) printf(COLOR_CYAN);
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        }

        // Infer Only - No Noise
        forward_pass(model, (uint8_t*)&input_token, NULL, 1, logits, NULL, 0, 0, &state);
        // Note: forward_pass handles state update internally for seq_len=1
        
        if (t >= seed_len - 1) {
            input_token = sample_logits(logits);
        }
    }
    printf(COLOR_RESET "\n");
}

Dataset load_data(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if(!f) { printf("Error: Create 'input.txt' first.\n"); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t*)malloc(len);
    fread(data, 1, len, f);
    fclose(f);
    return (Dataset){data, len};
}

int main() {
    srand(time(NULL));
    init_tables();
    Dataset ds = load_data("input.txt");
    printf("Loaded dataset: %ld bytes\n", ds.length);

    EggModel *model;
    posix_memalign((void**)&model, 16, sizeof(EggModel));
    memset(model, 0, sizeof(EggModel));

    uint32_t init_rng = 42;
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_val(&init_rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) model->head[i] = gen_noise_val(&init_rng);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) {
                model->gru_weights[l][g][i] = gen_noise_val(&init_rng);
            }
        }
        for(int m=0; m<2; m++) {
             // Utilize the full allocated space: HIDDEN * (HIDDEN * 4)
             for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) {
                 model->mlp_weights[l][m][i] = gen_noise_val(&init_rng);
             }
        }
    }

    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    }

    int *pair_fitnesses = (int*)malloc((POPULATION_SIZE/2) * sizeof(int));
    int8_t *logits = (int8_t*)malloc(VOCAB_SIZE);
    
    // Persistent States
    RecurrentState *pop_states = (RecurrentState*)aligned_alloc(16, POPULATION_SIZE * sizeof(RecurrentState));
    RecurrentState main_state;
    memset(pop_states, 0, POPULATION_SIZE * sizeof(RecurrentState));
    memset(&main_state, 0, sizeof(RecurrentState));

    printf("Starting EGGROLL Training (Stateful + Optimized)...\n");
    
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    long total_tokens = 0;
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step < max_steps; step++) {
        // Use a more robust seed mixing to avoid correlations if time(NULL) doesn't change
        uint32_t step_seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        if(step % 10 == 0) {
            sample_model(model, &ds.data[start_idx], 30, 30);
            
            int32_t loss_val = 0;
            forward_pass(model, &ds.data[start_idx], &ds.data[start_idx+1], SEQ_LEN, logits, &loss_val, step_seed, 0, &main_state);
            
            struct timespec current_time;
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double elapsed_sec = (current_time.tv_sec - start_time.tv_sec) + 
                                 (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            double tps = (elapsed_sec > 0) ? (double)total_tokens / elapsed_sec : 0.0;
            // Loss is accumulated fixed point. Divide by SEQ_LEN to get average per token, then by 2^FIXED_POINT
            printf("Step %ld/%ld | Loss: %.4f | Tok/s: %.2f\n", step, max_steps, (double)loss_val / (SEQ_LEN * (1 << FIXED_POINT)), tps);
        }

    dispatch_apply(POPULATION_SIZE / 2, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t p_idx) {
        uint32_t p_seed = step_seed + (uint32_t)p_idx;
        int8_t local_logits[VOCAB_SIZE];
        int32_t loss_pos = 0, loss_neg = 0;
        
        // Multi-stream: Distribute pairs across the dataset
        long stride = ds.length / (POPULATION_SIZE / 2);
        long stream_idx = (start_idx + (p_idx * stride)) % (ds.length - SEQ_LEN);

        forward_pass(model, &ds.data[stream_idx], &ds.data[stream_idx+1], SEQ_LEN, local_logits, &loss_pos, p_seed, 1, &pop_states[p_idx*2]);
        forward_pass(model, &ds.data[stream_idx], &ds.data[stream_idx+1], SEQ_LEN, local_logits, &loss_neg, p_seed, -1, &pop_states[p_idx*2+1]);

        if (loss_pos < loss_neg) pair_fitnesses[p_idx] = 1;
            else if (loss_neg < loss_pos) pair_fitnesses[p_idx] = -1;
            else pair_fitnesses[p_idx] = 0;
        });

        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);
            update_matrix(model->gru_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed+1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed+2, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][2], HIDDEN_DIM, HIDDEN_DIM, l_seed+3, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][3], HIDDEN_DIM, HIDDEN_DIM, l_seed+4, pair_fitnesses, POPULATION_SIZE);
            
            update_matrix(model->mlp_weights[l][0], HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, pair_fitnesses, POPULATION_SIZE);
        }
        update_matrix(model->head, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, pair_fitnesses, POPULATION_SIZE);

        total_tokens += SEQ_LEN;
    }

    printf("Training Done.\n");
    free(ds.data); free(model); free(logits); free(pair_fitnesses);
    free(pop_states);
    return 0;
}
