#ifndef EGG_DISK_LOG_H
#define EGG_DISK_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#define EGG_API_BASE "https://jlbhiestcyjduotrkmwl.supabase.co/functions/v1"
#define EGG_PROJECT_ID "f14854a9-9960-4272-b299-7076cf508480"

// Struct to hold logger state
typedef struct {
    char filename[256];
    struct timespec start_ts;
    char run_id[64];
    int tracking_enabled;
} EggLogState;

// Configuration struct for logging
typedef struct {
    int num_gpus;
    size_t vram_per_gpu;
    int hidden_dim;
    int head_dim;
    int n_layers;
    int seq_len;
    int vocab_size;
    int n_heads;
    int pop_size;
    int softmax_scale_bit;
    int host_gaussian;
    int device_gaussian;
    int host_mask;
    int device_mask;
    
    // Quantization / Sigmas
    int fixed_point;
    int sigma_shift;
    int sigma_shift_vector;
    
    // Shifts
    int shift_attn;
    int shift_qkv;
    int shift_out;
    int shift_logit;
    int shift_mlp_up;
    int shift_mlp_down;
    
    // Softmax
    float softmax_exp_scale;
    
    // Optimizer
    float adam_beta1;
    float adam_beta2;
    float adam_eps;
    float adam_weight_decay;
    
    // Muon
    int use_muon;
    float muon_momentum;
    float muon_lr_scale;
} EggLogConfig;

// Helper to extract value from simple JSON response
// Finds "id":"value" and copies value to buffer
static inline void _egg_parse_run_id(const char *json, char *buffer, size_t buf_size) {
    const char *key = "\"id\":\"";
    const char *start = strstr(json, key);
    if (start) {
        start += strlen(key);
        const char *end = strchr(start, '"');
        if (end) {
            size_t len = end - start;
            if (len < buf_size) {
                strncpy(buffer, start, len);
                buffer[len] = '\0';
            }
        }
    }
}

// Initialize the logger
// Opens file, writes header block + CSV header, closes file.
// Also initializes remote tracking run.
// Returns initialized state.
static inline EggLogState egg_log_init(
    const char *filename,
    EggLogConfig config
) {
    EggLogState state;
    memset(&state, 0, sizeof(state));
    strncpy(state.filename, filename, sizeof(state.filename) - 1);
    state.filename[sizeof(state.filename) - 1] = '\0';
    
    // Capture Start Time
    clock_gettime(CLOCK_MONOTONIC, &state.start_ts);
    
    time_t raw_time = time(NULL);
    struct tm *tm_info = localtime(&raw_time);
    char start_time_str[64];
    strftime(start_time_str, sizeof(start_time_str), "%Y-%m-%dT%H:%M:%S", tm_info);
    
    // Check file and write header if needed
    // Using "a" append mode.
    FILE *f = fopen(filename, "a");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        
        if (size > 0) fprintf(f, "\n"); // Spacing between runs if appending
        
        fprintf(f, "=== Training Run ===\n");
        fprintf(f, "Run Start:       %s\n", start_time_str);
        fprintf(f, "Hardware:        %d GPUs, %.2f MB VRAM/GPU\n", config.num_gpus, (double)config.vram_per_gpu / (1024.0 * 1024.0));
        fprintf(f, "Model Config:    Hidden=%d, Head=%d, Layers=%d, SeqLen=%d\n", 
                config.hidden_dim, config.head_dim, config.n_layers, config.seq_len);
        fprintf(f, "                 Vocab=%d, Heads=%d, PopSize=%d, SoftmaxScaleBit=%d\n", 
                config.vocab_size, config.n_heads, config.pop_size, config.softmax_scale_bit);
        fprintf(f, "Noise Config:    HostGaussian=%d, DeviceGaussian=%d, HostMask=%d, DeviceMask=%d\n",
                config.host_gaussian, config.device_gaussian, config.host_mask, config.device_mask);
        fprintf(f, "Quant Config:    FixedPoint=%d, SigmaShift=%d, SigmaShiftVec=%d\n",
                config.fixed_point, config.sigma_shift, config.sigma_shift_vector);
        fprintf(f, "Shift Config:    Attn=%d, QKV=%d, Out=%d, Logit=%d, MLP_Up=%d, MLP_Dn=%d\n",
                config.shift_attn, config.shift_qkv, config.shift_out, config.shift_logit, config.shift_mlp_up, config.shift_mlp_down);
        fprintf(f, "Optimizer:       Adam(B1=%.3f, B2=%.3f, Eps=%.1e, WD=%.4f)\n",
                config.adam_beta1, config.adam_beta2, config.adam_eps, config.adam_weight_decay);
        if (config.use_muon) {
            fprintf(f, "                 Muon(Mom=%.3f, Scale=%.3f)\n", config.muon_momentum, config.muon_lr_scale);
        }
        fprintf(f, "Misc:            SoftmaxExpScale=%.2f\n", config.softmax_exp_scale);
        fprintf(f, "----------------------------------------------------------------\n");
        fprintf(f, "Step,RelTime,Loss,Updates,LR\n");
        
        fclose(f);
    } else {
        perror("Failed to open log file");
    }

    // --- Tracking Integration (Create Run) ---
    // Construct simplified name and config (includes ALL EggLogConfig fields)
    char json_body[4096];
    snprintf(json_body, sizeof(json_body), 
        "{\"project_id\": \"%s\", \"name\": \"run-%s\", \"config\": {"
        "\"gpus\": %d, \"vram_mb\": %.2f, \"hidden\": %d, \"head_dim\": %d, "
        "\"layers\": %d, \"seq_len\": %d, "
        "\"pop_size\": %d, \"vocab\": %d, \"heads\": %d, "
        "\"host_gaussian\": %d, \"device_gaussian\": %d, "
        "\"host_mask\": %d, \"device_mask\": %d, "
        "\"fixed_point\": %d, \"sigma_shift\": %d, \"sigma_shift_vec\": %d, "
        "\"shift_attn\": %d, \"shift_qkv\": %d, \"shift_out\": %d, \"shift_logit\": %d, "
        "\"shift_mlp_up\": %d, \"shift_mlp_dn\": %d, "
        "\"softmax_scale_bit\": %d, \"softmax_exp_scale\": %.2f, "
        "\"adam_beta1\": %.3f, \"adam_beta2\": %.3f, \"adam_eps\": %.1e, \"adam_wd\": %.4f, "
        "\"use_muon\": %d, \"muon_mom\": %.3f, \"muon_scale\": %.3f"
        "}}", 
        EGG_PROJECT_ID, start_time_str,
        config.num_gpus, (double)config.vram_per_gpu / (1024.0 * 1024.0), 
        config.hidden_dim, config.head_dim, config.n_layers, config.seq_len,
        config.pop_size, config.vocab_size, config.n_heads,
        config.host_gaussian, config.device_gaussian,
        config.host_mask, config.device_mask,
        config.fixed_point, config.sigma_shift, config.sigma_shift_vector,
        config.shift_attn, config.shift_qkv, config.shift_out, config.shift_logit,
        config.shift_mlp_up, config.shift_mlp_down,
        config.softmax_scale_bit, config.softmax_exp_scale,
        config.adam_beta1, config.adam_beta2, config.adam_eps, config.adam_weight_decay,
        config.use_muon, config.muon_momentum, config.muon_lr_scale
    );

    // Call Create Run API (Blocking to get ID)
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), 
        "curl -s -X POST \"%s/runs\" "
        "-H \"Content-Type: application/json\" "
        "-d '%s'", 
        EGG_API_BASE, json_body);

    FILE *fp = popen(cmd, "r");
    if (fp) {
        char response[1024];
        if (fgets(response, sizeof(response), fp) != NULL) {
            _egg_parse_run_id(response, state.run_id, sizeof(state.run_id));
            if (strlen(state.run_id) > 0) {
                state.tracking_enabled = 1;
                printf("Tracking initialized: Run ID %s\n", state.run_id);
            }
        }
        pclose(fp);
    } else {
        // Fallback or error logging if desired, silent for now
    }
    
    return state;
}

// Record a training step
// Opens file, writes row, closes file.
// Also posts metrics to tracking API (detached)
static inline void egg_log_record(
    EggLogState *state,
    long step,
    double loss,
    unsigned long long updates,
    float lr
) {
    struct timespec current_ts;
    clock_gettime(CLOCK_MONOTONIC, &current_ts);
    
    double rel_time = ((double)(current_ts.tv_sec - state->start_ts.tv_sec)) + 
                      ((double)(current_ts.tv_nsec - state->start_ts.tv_nsec) / 1e9);
                      
    FILE *f = fopen(state->filename, "a");
    if (f) {
        // Write: Step, RelTime, Loss, Updates, LR
        fprintf(f, "%ld,%.4f,%.4f,%llu,%.6f\n", 
                step, rel_time, loss, updates, lr);
        fclose(f);
    }

    // --- Tracking Integration (Log Metrics) ---
    if (state->tracking_enabled) {
        char json_body[512];
        // Matches Python: "metrics": [{"name": "loss", "value": loss, "step": step}, ...]
        snprintf(json_body, sizeof(json_body), 
            "{\"metrics\": ["
            "{\"name\": \"loss\", \"value\": %.4f, \"step\": %ld}, "
            "{\"name\": \"updates\", \"value\": %llu, \"step\": %ld}, "
            "{\"name\": \"lr\", \"value\": %.6f, \"step\": %ld}]}", 
            loss, step, updates, step, lr, step);

        char cmd[1024];
        // Use background execution (&) and silence output
        snprintf(cmd, sizeof(cmd), 
            "curl -s -X POST \"%s/runs/%s/metrics\" "
            "-H \"Content-Type: application/json\" "
            "-d '%s' > /dev/null 2>&1 &", 
            EGG_API_BASE, state->run_id, json_body);

        system(cmd);
    }
}

static inline void egg_log_close(EggLogState *state) {
    // --- Tracking Integration (Mark Completed) ---
    if (state->tracking_enabled) {
        // Blocking call to ensure it finishes before exit
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), 
            "curl -s -X PATCH \"%s/runs/%s\" "
            "-H \"Content-Type: application/json\" "
            "-d '{\"status\": \"completed\"}' > /dev/null 2>&1", 
            EGG_API_BASE, state->run_id);
        
        system(cmd);
        printf("Tracking completed for Run ID %s\n", state->run_id);
    }
}

#endif // EGG_DISK_LOG_H
