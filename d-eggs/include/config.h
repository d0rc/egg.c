#ifndef EGG_CONFIG_H
#define EGG_CONFIG_H

// --- CONFIGURATION (all overridable via -D flags) ---
#ifndef HIDDEN_DIM
#  define HIDDEN_DIM 384
#endif
#ifndef HEAD_DIM
#  define HEAD_DIM 64
#endif
#ifndef N_LAYERS
#  define N_LAYERS 4
#endif
#ifndef NUMBER_OF_CYCLES
#  define NUMBER_OF_CYCLES 4
#endif
#ifndef SEQ_LEN
#  define SEQ_LEN 32
#endif

// NTT Mode: 0=disabled, 1=Walsh-Hadamard, 2=Fermat-257, 3=Fermat-65537
#ifndef NTT_MODE
#  define NTT_MODE 0
#endif
#ifndef VOCAB_SIZE
#  define VOCAB_SIZE 8192
#endif

#if VOCAB_SIZE != 256
#  define USE_TOKENIZER 1
#endif

#ifndef WARP_SIZE
#  define WARP_SIZE 32
#endif
#ifndef MAX_BLOCK_THREADS
#  define MAX_BLOCK_THREADS 1024
#endif
#define ALIGNED_DIM ((HIDDEN_DIM + 31) & ~31)
#define EGG_BLOCK_THREADS (ALIGNED_DIM > MAX_BLOCK_THREADS ? MAX_BLOCK_THREADS : ALIGNED_DIM)
#define N_HEADS (HIDDEN_DIM / HEAD_DIM)

#ifndef CHUNK_SIZE
#   define CHUNK_SIZE (8192)
#endif
#ifndef POPULATION_BATCH_SIZE
#   define POPULATION_BATCH_SIZE (CHUNK_SIZE * 1)
#endif
#define POPULATION_SIZE (POPULATION_BATCH_SIZE * 12)

#ifndef FIXED_POINT
#  define FIXED_POINT 4
#endif
#ifndef SIGMA_SHIFT
#  define SIGMA_SHIFT 3
#endif
#ifndef SIGMA_SHIFT_VECTOR
#  define SIGMA_SHIFT_VECTOR 4
#endif
#define MAX_VAL 127
#define MIN_VAL -127

#ifndef SOFTMAX_SCALE_BIT
#  define SOFTMAX_SCALE_BIT 18
#endif
#define SOFTMAX_SCALE (1 << SOFTMAX_SCALE_BIT)
#ifndef SOFTMAX_LUT_SIZE
#  define SOFTMAX_LUT_SIZE 256
#endif
#ifndef SOFTMAX_EXP_SCALE
#  define SOFTMAX_EXP_SCALE 6.0
#endif

// RoPE Configuration
#ifndef ROPE_SCALE_BIT
#  define ROPE_SCALE_BIT 20
#endif
#define ROPE_SCALE (1 << ROPE_SCALE_BIT)
#define ROPE_LUT_SIZE (SEQ_LEN * (HEAD_DIM / 2) * 2)

// Seed Offsets
#define SEED_OFF_EMB 0
#define SEED_OFF_EMB_BIAS_A 50
#define SEED_OFF_EMB_BIAS_B 51
#define SEED_OFF_LN_1_A 100
#define SEED_OFF_LN_1_B 101
#define SEED_OFF_LN_1_BIAS_A 110
#define SEED_OFF_LN_1_BIAS_B 111

#define SEED_OFF_Q_A 200
#define SEED_OFF_Q_B 201
#define SEED_OFF_K_A 202
#define SEED_OFF_K_B 203
#define SEED_OFF_V_A 204
#define SEED_OFF_V_B 205
#define SEED_OFF_O_A 206
#define SEED_OFF_O_B 207

#define SEED_OFF_LN_2_A 300
#define SEED_OFF_LN_2_B 301
#define SEED_OFF_LN_2_BIAS_A 310
#define SEED_OFF_LN_2_BIAS_B 311

#define SEED_OFF_MLP_UP_A 400
#define SEED_OFF_MLP_UP_B 401
#define SEED_OFF_MLP_DOWN_A 402
#define SEED_OFF_MLP_DOWN_B 403
#define SEED_OFF_MLP_BIAS_UP_A 410
#define SEED_OFF_MLP_BIAS_UP_B 411
#define SEED_OFF_MLP_BIAS_DOWN_A 412
#define SEED_OFF_MLP_BIAS_DOWN_B 413

#define SEED_OFF_LN_F_A 900
#define SEED_OFF_LN_F_B 901
#define SEED_OFF_LN_F_BIAS_A 910
#define SEED_OFF_LN_F_BIAS_B 911

#define SEED_OFF_HEAD 999

#define SEED_OFF_LN_INIT_A 500
#define SEED_OFF_LN_INIT_B 501
#define SEED_OFF_LN_INIT_BIAS_A 502
#define SEED_OFF_LN_INIT_BIAS_B 503

#define SEED_OFF_EMB_MLP_UP_A 510
#define SEED_OFF_EMB_MLP_UP_B 511
#define SEED_OFF_EMB_MLP_DOWN_A 512
#define SEED_OFF_EMB_MLP_DOWN_B 513
#define SEED_OFF_EMB_MLP_BIAS_UP_A 514
#define SEED_OFF_EMB_MLP_BIAS_UP_B 515
#define SEED_OFF_EMB_MLP_BIAS_DOWN_A 516
#define SEED_OFF_EMB_MLP_BIAS_DOWN_B 517

// NTT Embedding Seed Offsets
#define SEED_OFF_NTT_EMB1 600
#define SEED_OFF_NTT_EMB2 610
#define SEED_OFF_NTT_EMB3 620

#ifndef SHIFT_ATTN
#  define SHIFT_ATTN 8
#endif
#ifndef SHIFT_QKV
#  define SHIFT_QKV 6
#endif
#ifndef SHIFT_OUT
#  define SHIFT_OUT 9
#endif
#ifndef SHIFT_LOGIT
#  define SHIFT_LOGIT 8
#endif
#ifndef SHIFT_MLP_UP
#  define SHIFT_MLP_UP 8
#endif
#ifndef SHIFT_MLP_DOWN
#  define SHIFT_MLP_DOWN 10
#endif

// Generator settings
#ifndef HOST_GAUSSIAN
#  define HOST_GAUSSIAN false
#endif
#ifndef DEVICE_GAUSSIAN
#  define DEVICE_GAUSSIAN false
#endif
#ifndef HOST_MASK
#  define HOST_MASK 15
#endif
#ifndef DEVICE_MASK
#  define DEVICE_MASK 15
#endif

#ifndef USE_SAME_DATA
#  define USE_SAME_DATA 0
#endif

// Adam Hyperparams
#ifndef ADAM_BETA1
#  define ADAM_BETA1 0.9f
#endif
#ifndef ADAM_BETA2
#  define ADAM_BETA2 0.95f
#endif
#ifndef ADAM_EPS
#  define ADAM_EPS 1e-8f
#endif
#ifndef ADAM_WEIGHT_DECAY
#  define ADAM_WEIGHT_DECAY 0.01f
#endif

#ifndef USE_MUON
#  define USE_MUON 0
#endif

// Sampling Configuration
#ifndef SAMPLING_TEMP
#  define SAMPLING_TEMP 0.6f
#endif
#ifndef SAMPLING_MIN_P
#  define SAMPLING_MIN_P 0.08f
#endif
#ifndef SAMPLING_PRESENCE_PENALTY
#  define SAMPLING_PRESENCE_PENALTY 0.2f
#endif

#if USE_MUON
#  ifndef MUON_MOMENTUM
#    define MUON_MOMENTUM 0.85f
#  endif
#  ifndef MUON_LR_SCALE
#    define MUON_LR_SCALE 1.0f
#  endif
#endif

// Adaptive Thresholding for Winners Selection
#ifndef USE_ADAPTIVE_THRESHOLD
#  define USE_ADAPTIVE_THRESHOLD 1
#endif
#ifndef ADAPTIVE_THRESHOLD_ALPHA
#  define ADAPTIVE_THRESHOLD_ALPHA 0.1f
#endif

// Chunk Mean Filter: Favor perturbations that align with chunk trend
#ifndef CHUNK_MEAN_FILTER
#  define CHUNK_MEAN_FILTER 0
#endif
#ifndef CHUNK_MEAN_EXPONENT
#  define CHUNK_MEAN_EXPONENT 1.0
#endif

// Loss Haxxing (Retry chunks and pick best)
#ifndef LOSS_HAXXING
#  define LOSS_HAXXING 1
#endif
#ifndef LOSS_HAXXING_RETRIES
#  define LOSS_HAXXING_RETRIES 3
#endif
// Direction: 'min', 'max', or 'mixed'
#ifndef LOSS_HAXXING_DIRECTION
#  define LOSS_HAXXING_DIRECTION 'entmax'
#endif
// Ratio of 'min' chunks when direction='mixed' (0.0 to 1.0, default 0.5 = 50/50)
#ifndef LOSS_HAXXING_MIXED_RATIO
#  define LOSS_HAXXING_MIXED_RATIO 0.2f
#endif

// Adaptive Noise (Noise-Trained Layer)
// 0: Disabled (Scale 1.0), 1: Adaptive Scale, 2: Adaptive Gate
#ifndef ADAPTIVE_NOISE_MODE
#  define ADAPTIVE_NOISE_MODE 0
#endif
#define SEED_OFF_GATE 100000

#ifndef ADAPTIVE_NOISE_INC
#  define ADAPTIVE_NOISE_INC 5
#endif
#ifndef ADAPTIVE_NOISE_DEC
#  define ADAPTIVE_NOISE_DEC 1
#endif
#ifndef ADAPTIVE_NOISE_INIT
#  define ADAPTIVE_NOISE_INIT 64
#endif

#endif // EGG_CONFIG_H
