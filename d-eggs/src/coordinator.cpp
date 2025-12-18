#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cinttypes>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <chrono>
#include <csignal>

#include "protocol.h"
#include "config.h"
#include "utils/egg_math.h"
#include "utils/training.h"
#include "utils/log.h"
#include "utils/ternary_pack.h"
#include "utils/checkpoint.h"

// Loss Haxxing - Direction selection helpers
// Determines whether to use 'min' or 'max' selection for each step
// For 'mixed' mode: LOSS_HAXXING_MIXED_RATIO fraction of steps use 'min'

inline bool is_min_mode_for_step([[maybe_unused]] uint64_t step) {
#if LOSS_HAXXING_DIRECTION == 'min'
    return true;
#elif LOSS_HAXXING_DIRECTION == 'mixed'
    int period = (LOSS_HAXXING_MIXED_RATIO > 0) ? (int)(1.0f / LOSS_HAXXING_MIXED_RATIO) : 1;
    if (period < 1) period = 1;
    return (step % period) == 0;
#else  // 'max' or default
    return false;
#endif
}

inline uint64_t get_initial_best_loss_for_step(uint64_t step) {
    return is_min_mode_for_step(step) ? UINT64_MAX : 0;
}

// Constants
#define PORT 12345
// CHUNK_SIZE is defined in config.h
#define MAX_HISTORY 5000
#define STRAGGLER_TIMEOUT_MS 90000  // 90 seconds before re-assigning a chunk

// Network Stats (global atomics)
std::atomic<uint64_t> g_bytes_sent(0);
std::atomic<uint64_t> g_bytes_received(0);

// Global Configuration
std::string g_save_dir = "";

// Global Scheduler
ReduceLROnPlateau g_scheduler;

// Global State
struct GlobalState {
    uint64_t current_step = 0;
    uint64_t current_seed = 42; // Initial seed
    float current_lr = 0.0f;    // Current Learning Rate
    
    // History of fitness vectors for catch-up
    // Map step -> {vector (packed), lr}
    std::map<uint64_t, std::pair<std::vector<uint8_t>, float>> fitness_history;
    
    // Current Step State
    std::vector<int32_t> current_fitness; // Size: POPULATION_SIZE / 2
    int64_t step_sum_loss = 0;            // Aggregated loss for current step
    std::vector<bool> chunk_completed;    // Size: POPULATION_SIZE / CHUNK_SIZE
    std::vector<bool> chunk_in_progress;  // Size: POPULATION_SIZE / CHUNK_SIZE
    std::vector<int> chunk_attempts;      // Number of successful attempts per chunk
    std::vector<uint32_t> chunk_shifts;   // Shift used for the best result of each chunk
    std::vector<uint64_t> chunk_best_loss;// Best loss seen for each chunk
    std::vector<std::chrono::steady_clock::time_point> chunk_assign_time; // When each chunk was last assigned
    std::deque<int> chunk_queue;          // Queue of chunks to assign
    int chunks_remaining = 0;
    uint64_t step_total_updates = 0;      // Aggregated updates for current step
    uint64_t step_min_updates = UINT64_MAX;
    uint64_t step_max_updates = 0;
    uint64_t step_transmissions = 0;
    
    double prev_loss = 0.0;
    uint64_t prev_max_updates = 0;

    std::chrono::steady_clock::time_point step_start_time;

    std::mutex mutex;
};

GlobalState g_state;

// Global Logger State
EggLogState g_log_state;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    egg_log_close(&g_log_state);
    exit(0);
}

// Helper: Humanize bytes
std::string humanize_bytes(uint64_t bytes) {
    char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / (1024.0 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", (double)bytes / (1024.0 * 1024));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", (double)bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%" PRIu64 " B", bytes);
    }
    return std::string(buf);
}

// Helper: Humanize tokens
std::string humanize_tokens(uint64_t tokens) {
    char buf[64];
    if (tokens >= 1000ULL * 1000 * 1000) {
        snprintf(buf, sizeof(buf), "%.2f B", (double)tokens / 1e9);
    } else if (tokens >= 1000ULL * 1000) {
        snprintf(buf, sizeof(buf), "%.2f M", (double)tokens / 1e6);
    } else if (tokens >= 1000) {
        snprintf(buf, sizeof(buf), "%.2f K", (double)tokens / 1e3);
    } else {
        snprintf(buf, sizeof(buf), "%" PRIu64, tokens);
    }
    return std::string(buf);
}

// Helper: Send Packet
bool send_packet(int sock, uint8_t opcode, const void* payload, uint32_t len) {
    uint8_t header[EGG_HEADER_SIZE];
    egg_write_header(header, opcode, len);
    
    if (send(sock, header, EGG_HEADER_SIZE, 0) != EGG_HEADER_SIZE) return false;
    g_bytes_sent += EGG_HEADER_SIZE;
    if (len > 0) {
        if (send(sock, payload, len, 0) != len) return false;
        g_bytes_sent += len;
    }
    return true;
}

// Helper: Recv Packet
bool recv_exact(int sock, void* buf, uint32_t len) {
    uint32_t total = 0;
    uint8_t* p = (uint8_t*)buf;
    while (total < len) {
        ssize_t n = recv(sock, p + total, len - total, 0);
        if (n <= 0) return false;
        total += n;
    }
    g_bytes_received += len;
    return true;
}

void handle_client(int sock) {
    std::cout << "Client connected: " << sock << std::endl;
    
    while (true) {
        uint8_t header[EGG_HEADER_SIZE];
        if (!recv_exact(sock, header, EGG_HEADER_SIZE)) break;
        
        uint8_t opcode;
        uint32_t payload_len;
        if (egg_parse_header(header, &opcode, &payload_len) != 0) {
            std::cerr << "Invalid header" << std::endl;
            break;
        }
        
        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0) {
            if (!recv_exact(sock, payload.data(), payload_len)) break;
        }
        
        if (opcode == OP_JOB_REQUEST) {
            EggJobRequest req;
            if (payload_len < sizeof(EggJobRequest)) break;
            egg_deserialize_job_request(payload.data(), &req);
            
            std::lock_guard<std::mutex> lock(g_state.mutex);
            
            // Check if client needs update
            if (req.last_step < g_state.current_step) {
                // Send update for next step (req.last_step)
                auto it = g_state.fitness_history.find(req.last_step);
                if (it != g_state.fitness_history.end()) {
                    const auto& entry = it->second;
                    const auto& packed_fit = entry.first;
                    float history_lr = entry.second;
                    
                    EggJobResponseHeader resp;
                    resp.seed = g_state.current_seed; // Not used for update, but keep consistent
                    resp.last_step = req.last_step + 1; // Target step
                    resp.data_position = 0;
                    resp.model_size = packed_fit.size();
                    resp.learning_rate = history_lr; // Send LR from history
                    
                    // Serialize header
                    uint8_t resp_buf[32]; // Updated size
                    egg_serialize_job_response_header(resp_buf, &resp);
                    
                    // Send Header + HeaderPayload + ModelData
                    uint8_t packet_header[EGG_HEADER_SIZE];
                    egg_write_header(packet_header, OP_JOB_RESPONSE, 32 + resp.model_size);
                    
                    send(sock, packet_header, EGG_HEADER_SIZE, 0);
                    send(sock, resp_buf, 32, 0);
                    send(sock, packed_fit.data(), resp.model_size, 0);
                    
                    continue; // Loop to let client apply update
                } else {
                    // Too old or future?
                    std::cerr << "Client too old: " << req.last_step << " vs " << g_state.current_step << std::endl;
                    // Send empty wait
                    EggJobResponseHeader resp = {0, 0, 0, 0, 0.0f};
                    resp.data_position = (uint64_t)-1;
                    uint8_t resp_buf[32];
                    egg_serialize_job_response_header(resp_buf, &resp);
                    send_packet(sock, OP_JOB_RESPONSE, resp_buf, 32);
                    continue;
                }
            }
            
            // Client is up to date (req.last_step == g_state.current_step)
            // Assign Chunk
            int chunk_idx = -1;
            
            int queue_len = g_state.chunk_queue.size();
            for (int attempt = 0; attempt < queue_len && chunk_idx == -1; attempt++) {
                if (g_state.chunk_queue.empty()) break;
                
                int candidate = g_state.chunk_queue.front();
                g_state.chunk_queue.pop_front();
                
                if (g_state.chunk_completed[candidate]) {
                    continue; // Already done, discard entirely
                }
                
                // Check if currently in-progress
                if (g_state.chunk_in_progress[candidate]) {
                    auto elapsed = std::chrono::steady_clock::now() - g_state.chunk_assign_time[candidate];
                    if (elapsed < std::chrono::milliseconds(STRAGGLER_TIMEOUT_MS)) {
                        // Still being worked on, re-queue at back and try next
                        g_state.chunk_queue.push_back(candidate);
                        continue;
                    }
                    // Straggler timeout - allow re-assignment
                    std::cerr << "Chunk " << candidate << " timed out after " 
                              << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() 
                              << "s, re-assigning" << std::endl;
                }
                
                // Assign this chunk
                chunk_idx = candidate;
                g_state.chunk_in_progress[candidate] = true;
                g_state.chunk_assign_time[candidate] = std::chrono::steady_clock::now();
                g_state.chunk_queue.push_back(candidate); // Re-queue for straggler handling
                break;
            }
            
            if (chunk_idx != -1) {
                // Found a chunk
                EggJobResponseHeader resp;
                // Apply shift for this attempt
                uint32_t shift = g_state.chunk_attempts[chunk_idx];
                resp.seed = g_state.current_seed + shift;
                resp.last_step = g_state.current_step;
                resp.data_position = chunk_idx * CHUNK_SIZE; // Start index
                resp.model_size = 0; // No update
                resp.learning_rate = g_state.current_lr;
                
                uint8_t resp_buf[32];
                egg_serialize_job_response_header(resp_buf, &resp);
                send_packet(sock, OP_JOB_RESPONSE, resp_buf, 32);
            } else {
                // No chunks left, wait for next step
                EggJobResponseHeader resp = {0, 0, 0, 0, 0.0f};
                resp.data_position = (uint64_t)-1;
                
                uint8_t resp_buf[32];
                egg_serialize_job_response_header(resp_buf, &resp);
                send_packet(sock, OP_JOB_RESPONSE, resp_buf, 32);
            }
            
        } else if (opcode == OP_LOG_MESSAGE) {
            // Print log message from worker (e.g., generation output)
            std::string msg((char*)payload.data(), payload_len);
            std::cout << msg << std::flush;
            
        } else if (opcode == OP_RESULT) {
            EggResultHeader res;
            if (payload_len < 44) break;  // Updated from 36 to 44
            egg_deserialize_result_header(payload.data(), &res);
            
            int chunk_idx = res.data_position / CHUNK_SIZE;
            
            // Unpack fitness
            int num_fitness = CHUNK_SIZE / 2;
            std::vector<int32_t> h_fit(num_fitness);
            ternary_unpack(payload.data() + 44, num_fitness, h_fit.data());
            
            std::lock_guard<std::mutex> lock(g_state.mutex);
            
            if (res.last_step == g_state.current_step) {
                // Track updates stats (count every transmission)
                g_state.step_transmissions++;
                g_state.step_total_updates += res.updates_count;
                if (res.updates_count < g_state.step_min_updates) g_state.step_min_updates = res.updates_count;
                if (res.updates_count > g_state.step_max_updates) g_state.step_max_updates = res.updates_count;

                // Clear in-progress flag (even if already completed by another worker)
                g_state.chunk_in_progress[chunk_idx] = false;
                
                if (!g_state.chunk_completed[chunk_idx]) {
                    // The shift used for this result is the current chunk_attempts value
                    // (before incrementing, since that's what was sent during assignment)
                    uint32_t used_shift = g_state.chunk_attempts[chunk_idx];
                    g_state.chunk_attempts[chunk_idx]++;
                    
                    bool is_best = false;
                    bool use_min = is_min_mode_for_step(g_state.current_step);
                    if ((use_min && res.sum_loss < g_state.chunk_best_loss[chunk_idx]) ||
                        (!use_min && res.sum_loss > g_state.chunk_best_loss[chunk_idx])) {
                        g_state.chunk_best_loss[chunk_idx] = res.sum_loss;
                        // Store the shift that produced this best result
                        g_state.chunk_shifts[chunk_idx] = used_shift;
                        is_best = true;
                        
                        // Copy fitness (only if best)
                        int start_fit_idx = res.data_position / 2;
                        for(int i=0; i<num_fitness; i++) {
                            if (start_fit_idx + i < g_state.current_fitness.size()) {
                                g_state.current_fitness[start_fit_idx + i] = h_fit[i];
                            }
                        }
                    }

                    if (LOSS_HAXXING) {
                        printf("[Haxxing] Chunk %d Attempt %d/%d | Loss: %" PRIu64 " | Best: %" PRIu64 "%s\n", 
                               chunk_idx, g_state.chunk_attempts[chunk_idx], LOSS_HAXXING_RETRIES, 
                               res.sum_loss, g_state.chunk_best_loss[chunk_idx], is_best ? " *" : "");
                    }

                    if (g_state.chunk_attempts[chunk_idx] < LOSS_HAXXING_RETRIES) {
                        // Retry
                        g_state.chunk_queue.push_back(chunk_idx);
                    } else {
                        // Finalize
                        g_state.chunk_completed[chunk_idx] = true;
                        g_state.chunks_remaining--;
                        
                        // Accumulate Loss (Use the BEST loss)
                        g_state.step_sum_loss += g_state.chunk_best_loss[chunk_idx];
                        
                        if (LOSS_HAXXING) {
                            printf("[Haxxing] Chunk %d Finalized. Selected Loss: %" PRIu64 "\n", 
                                   chunk_idx, g_state.chunk_best_loss[chunk_idx]);
                        }
                    
                        if (g_state.chunks_remaining == 0) {
                        // Step Complete
                        // Calculate Avg Loss
                        double avg_loss = (double)g_state.step_sum_loss / (POPULATION_SIZE * SEQ_LEN * 16.0);
                        
                        // Calculate Time and Speed
                        auto now = std::chrono::steady_clock::now();
                        double step_ms = std::chrono::duration<double, std::milli>(now - g_state.step_start_time).count();
                        double tokens_per_sec = (double)(POPULATION_SIZE * SEQ_LEN) / (step_ms / 1000.0);
                        
                        // Update Learning Rate using Scheduler
                        float current_lr = (float)get_learning_rate_adaptive(&g_scheduler, g_state.current_step, avg_loss);
                        g_state.current_lr = current_lr;
                        
                        // Network Stats
                        uint64_t sent = g_bytes_sent.load();
                        uint64_t recv = g_bytes_received.load();
                        double net_mbps = (double)(sent + recv) / (step_ms / 1000.0) / (1024.0 * 1024.0);

                        // Determine colors
                        const char* loss_color = "";
                        if (g_state.current_step > 0) {
                            if (avg_loss < g_state.prev_loss) loss_color = "\033[92m"; // Bright Green
                            else if (avg_loss > g_state.prev_loss) loss_color = "\033[91m"; // Bright Red
                        }
                        
                        const char* updates_color = "";
                        if (g_state.current_step > 0) {
                            if (g_state.step_max_updates > g_state.prev_max_updates) updates_color = "\033[36m"; // Cyan
                            else if (g_state.step_max_updates < g_state.prev_max_updates) updates_color = "\033[34m"; // Blue
                        }
                        const char* reset_color = "\033[0m";

                        // Construct Updates String
                        char updates_detail[128];
                        if (g_state.step_min_updates == 0 || g_state.step_min_updates == g_state.step_max_updates) {
                            snprintf(updates_detail, sizeof(updates_detail), "(n=%" PRIu64 ", max=%s%" PRIu64 "%s)", 
                                     g_state.step_transmissions, updates_color, g_state.step_max_updates, reset_color);
                        } else {
                            snprintf(updates_detail, sizeof(updates_detail), "(n=%" PRIu64 ", min=%" PRIu64 ", max=%s%" PRIu64 "%s)", 
                                     g_state.step_transmissions, g_state.step_min_updates, updates_color, g_state.step_max_updates, reset_color);
                        }

                        uint64_t total_tokens = g_state.current_step * POPULATION_SIZE * SEQ_LEN;

                        // Print Log
                        printf("Step %" PRIu64 " | Tokens: %s | Loss: %s%.4f%s | Time: %.2f ms | Updates: %" PRIu64 " %s | Speed: %.2f tok/s | LR: %.6f | Net: %.2f MB/s (Tx: %s, Rx: %s)\n", 
                               g_state.current_step, 
                               humanize_tokens(total_tokens).c_str(),
                               loss_color, avg_loss, reset_color,
                               step_ms, g_state.step_total_updates, updates_detail,
                               tokens_per_sec, current_lr, net_mbps,
                               humanize_bytes(sent).c_str(), humanize_bytes(recv).c_str());

                        // Update previous values
                        g_state.prev_loss = avg_loss;
                        g_state.prev_max_updates = g_state.step_max_updates;

                        // Remote logging
                        egg_log_record(&g_log_state, g_state.current_step, avg_loss, 
                                       g_state.step_total_updates, current_lr);

                        // Store history (packed)
                        size_t packed_size = ternary_pack_estimate_size(g_state.current_fitness.size());
                        std::vector<uint8_t> packed_fit(packed_size);
                        ternary_pack(g_state.current_fitness.data(), g_state.current_fitness.size(), packed_fit.data());
                        
                        // Append chunk_shifts to packed_fit
                        size_t shifts_size = g_state.chunk_shifts.size() * sizeof(uint32_t);
                        size_t original_size = packed_fit.size();
                        packed_fit.resize(original_size + shifts_size);
                        memcpy(packed_fit.data() + original_size, g_state.chunk_shifts.data(), shifts_size);

                        g_state.fitness_history[g_state.current_step] = {packed_fit, g_state.current_lr};

                        if (g_state.fitness_history.size() > MAX_HISTORY) {
                            g_state.fitness_history.erase(g_state.fitness_history.begin());
                        }
                        
                        // Advance
                        g_state.current_step++;
                        g_state.current_seed = hash_rng(g_state.current_seed, g_state.current_step); // Simple evolution

                        // Save Checkpoint (with scheduler)
                        trigger_save_checkpoint(g_save_dir, g_state.current_step, g_state.current_seed, packed_fit, g_scheduler);
                        
                        // Reset for next step
                        int total_chunks = POPULATION_SIZE / CHUNK_SIZE;
                        g_state.chunk_completed.assign(total_chunks, false);
                        g_state.chunk_in_progress.assign(total_chunks, false);
                        g_state.chunk_attempts.assign(total_chunks, 0);
                        g_state.chunk_shifts.assign(total_chunks, 0);
                        // All chunks in same step use same initial value based on step mode
                        uint64_t init_loss = get_initial_best_loss_for_step(g_state.current_step);
                        g_state.chunk_best_loss.assign(total_chunks, init_loss);
                        g_state.chunks_remaining = total_chunks;
                        g_state.chunk_queue.clear();
                        for(int i=0; i<total_chunks; i++) g_state.chunk_queue.push_back(i);
                        
                        std::fill(g_state.current_fitness.begin(), g_state.current_fitness.end(), 0);
                        g_state.step_sum_loss = 0;
                        g_state.step_total_updates = 0;  // Reset updates counter
                        g_state.step_min_updates = UINT64_MAX;
                        g_state.step_max_updates = 0;
                        g_state.step_transmissions = 0;
                        g_state.step_start_time = std::chrono::steady_clock::now();
                        
                        // Reset network counters for next step
                        g_bytes_sent = 0;
                        g_bytes_received = 0;
                    }
                }
            }
        }
        }
    }
    
    close(sock);
    std::cout << "Client disconnected: " << sock << std::endl;
}

int main(int argc, char** argv) {
    // Parse Arguments
    std::string load_dir = "";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--save-dir" && i + 1 < argc) {
            g_save_dir = argv[++i];
        } else if (arg == "--load-dir" && i + 1 < argc) {
            load_dir = argv[++i];
        }
    }

    // Register signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Init Scheduler
    g_scheduler = init_scheduler_default(0.5);
    g_state.current_lr = g_scheduler.current_lr;
    
    // Init State
    g_state.current_fitness.resize(POPULATION_SIZE / 2);
    int total_chunks = POPULATION_SIZE / CHUNK_SIZE;
    g_state.chunk_completed.resize(total_chunks, false);
    g_state.chunk_in_progress.resize(total_chunks, false);
    g_state.chunk_attempts.resize(total_chunks, 0);
    g_state.chunk_shifts.resize(total_chunks, 0);
    // All chunks in same step use same initial value based on step mode
    uint64_t init_loss = get_initial_best_loss_for_step(g_state.current_step);
    g_state.chunk_best_loss.assign(total_chunks, init_loss);
    g_state.chunk_assign_time.resize(total_chunks);
    g_state.chunks_remaining = total_chunks;
    for(int i=0; i<total_chunks; i++) g_state.chunk_queue.push_back(i);
    
    // Load Checkpoint if requested
    if (!load_dir.empty()) {
        std::vector<uint8_t> loaded_fit;
        if (load_checkpoint(load_dir, g_state.current_step, g_state.current_seed, loaded_fit, g_scheduler, &g_state.fitness_history, MAX_HISTORY)) {
            // Restore LR from scheduler
            g_state.current_lr = g_scheduler.current_lr;
        }
    }

    // Initialize remote logging
    EggLogConfig log_config = {};
    log_config.num_gpus = 0;  // Coordinator doesn't know GPU count
    log_config.vram_per_gpu = 0;
    log_config.hidden_dim = HIDDEN_DIM;
    log_config.head_dim = HEAD_DIM;
    log_config.n_layers = N_LAYERS;
    log_config.seq_len = SEQ_LEN;
    log_config.vocab_size = VOCAB_SIZE;
    log_config.n_heads = N_HEADS;
    log_config.pop_size = POPULATION_SIZE;
    log_config.softmax_scale_bit = SOFTMAX_SCALE_BIT;
    log_config.host_gaussian = HOST_GAUSSIAN;
    log_config.device_gaussian = DEVICE_GAUSSIAN;
    log_config.host_mask = HOST_MASK;
    log_config.device_mask = DEVICE_MASK;
    log_config.fixed_point = FIXED_POINT;
    log_config.sigma_shift = SIGMA_SHIFT;
    log_config.sigma_shift_vector = SIGMA_SHIFT_VECTOR;
    log_config.shift_attn = SHIFT_ATTN;
    log_config.shift_qkv = SHIFT_QKV;
    log_config.shift_out = SHIFT_OUT;
    log_config.shift_logit = SHIFT_LOGIT;
    log_config.shift_mlp_up = SHIFT_MLP_UP;
    log_config.shift_mlp_down = SHIFT_MLP_DOWN;
    log_config.softmax_exp_scale = SOFTMAX_EXP_SCALE;
    log_config.adam_beta1 = ADAM_BETA1;
    log_config.adam_beta2 = ADAM_BETA2;
    log_config.adam_eps = ADAM_EPS;
    log_config.adam_weight_decay = ADAM_WEIGHT_DECAY;
    log_config.use_muon = USE_MUON;
#if USE_MUON
    log_config.muon_momentum = MUON_MOMENTUM;
    log_config.muon_lr_scale = MUON_LR_SCALE;
#else
    log_config.muon_momentum = 0.0f;
    log_config.muon_lr_scale = 0.0f;
#endif
    g_log_state = egg_log_init("coordinator.log", log_config);
    
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) { perror("socket failed"); exit(1); }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) { perror("bind failed"); exit(1); }
    if (listen(server_fd, 3) < 0) { perror("listen failed"); exit(1); }
    
    std::cout << "Coordinator listening on port " << PORT << std::endl;
    
    printf("\n=== EGG DISTRIBUTED COORDINATOR ===\n");
    printf("Model Config:\n");
    printf("  Hidden Dim: %d\n", HIDDEN_DIM);
    printf("  Layers:     %d\n", N_LAYERS);
    printf("  Heads:      %d\n", N_HEADS);
    printf("  Seq Len:    %d\n", SEQ_LEN);
    printf("  Vocab Size: %d\n", VOCAB_SIZE);
    printf("Training Config:\n");
    printf("  Population: %d\n", POPULATION_SIZE);
    printf("  Chunk Mean Filter: %d (Exp: %.2f)\n", CHUNK_MEAN_FILTER, (double)CHUNK_MEAN_EXPONENT);
    printf("  Chunk Size: %d\n", CHUNK_SIZE);
    printf("  Chunks:     %d\n", total_chunks);
    printf("===================================\n\n");

    g_state.step_start_time = std::chrono::steady_clock::now();

    while (true) {
        int new_socket = accept(server_fd, NULL, NULL);
        if (new_socket < 0) { perror("accept failed"); continue; }
        
        std::thread(handle_client, new_socket).detach();
    }
    
    return 0;
}
