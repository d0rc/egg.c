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
#include "utils/terminal.h"

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

inline uint64_t get_initial_best_metric(uint64_t step) {
#if LOSS_HAXXING_DIRECTION == 'entmax'
    return 0; // Maximize entropy
#elif LOSS_HAXXING_DIRECTION == 'min'
    return UINT64_MAX;
#elif LOSS_HAXXING_DIRECTION == 'mixed'
    return is_min_mode_for_step(step) ? UINT64_MAX : 0;
#else // 'max'
    return 0;
#endif
}

// Constants
#define PORT 12345
// CHUNK_SIZE is defined in config.h
#define MAX_HISTORY 5000
#define STRAGGLER_TIMEOUT_MS 90000  // 90 seconds before re-assigning a chunk

// Job Status
enum JobStatus {
    JOB_IDLE = 0,
    JOB_IN_PROGRESS = 1,
    JOB_COMPLETED = 2
};

struct Job {
    int chunk_idx;
    int attempt_idx;
};

struct JobResultData {
    uint64_t sum_loss;
    uint64_t sum_entropy;
    std::vector<int32_t> fitness;
    bool received = false;
};

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
    int64_t step_sum_entropy = 0;         // Aggregated entropy for current step
    
    std::vector<bool> chunk_completed;    // Size: POPULATION_SIZE / CHUNK_SIZE
    
    // Job Tracking
    std::vector<std::vector<int>> job_status; // [chunk_idx][attempt_idx] -> JobStatus
    std::vector<std::vector<std::chrono::steady_clock::time_point>> job_assign_time; // [chunk_idx][attempt_idx]
    std::vector<int> chunk_finished_attempts; // Number of completed attempts per chunk
    
    std::vector<std::vector<JobResultData>> step_results; // [chunk_idx][attempt_idx]

    std::vector<uint32_t> chunk_shifts;   // Shift used for the best result of each chunk
    
    std::deque<Job> idle_queue;           // Queue of IDLE jobs to assign
    std::chrono::steady_clock::time_point last_straggler_check;
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

void finalize_step() {
    // 1. Scan all results to find Overall Min/Max and Select Best
    uint64_t overall_min_loss = UINT64_MAX;
    uint64_t overall_max_loss = 0;
    uint64_t overall_min_entropy = UINT64_MAX;
    uint64_t overall_max_entropy = 0;

    int total_chunks = g_state.step_results.size();
    
    // Reset step accumulators
    g_state.step_sum_loss = 0;
    g_state.step_sum_entropy = 0;

    // Selected stats
    uint64_t selected_min_loss = UINT64_MAX;
    uint64_t selected_max_loss = 0;
    uint64_t selected_min_entropy = UINT64_MAX;
    uint64_t selected_max_entropy = 0;

    for (int c = 0; c < total_chunks; c++) {
        int best_attempt = -1;
        uint64_t best_metric_val = get_initial_best_metric(g_state.current_step);
        
        // Determine mode for this step
        bool use_min = false;
#if LOSS_HAXXING_DIRECTION == 'min'
        use_min = true;
#elif LOSS_HAXXING_DIRECTION == 'mixed'
        use_min = is_min_mode_for_step(g_state.current_step);
#endif
        // 'entmax' and 'max' use max (use_min = false)

        for (int a = 0; a < LOSS_HAXXING_RETRIES; a++) {
            const auto& res = g_state.step_results[c][a];
            if (!res.received) continue;

            // Overall stats
            if (res.sum_loss < overall_min_loss) overall_min_loss = res.sum_loss;
            if (res.sum_loss > overall_max_loss) overall_max_loss = res.sum_loss;
            if (res.sum_entropy < overall_min_entropy) overall_min_entropy = res.sum_entropy;
            if (res.sum_entropy > overall_max_entropy) overall_max_entropy = res.sum_entropy;

            // Selection Logic
            uint64_t current_metric;
#if LOSS_HAXXING_DIRECTION == 'entmax'
            current_metric = res.sum_entropy;
#else
            current_metric = res.sum_loss;
#endif

            bool better = false;
            if (use_min) {
                if (current_metric < best_metric_val) better = true;
            } else {
                if (current_metric > best_metric_val) better = true;
            }

            if (better || best_attempt == -1) {
                best_metric_val = current_metric;
                best_attempt = a;
            }
        }

        // Apply Best Attempt
        if (best_attempt != -1) {
            const auto& best_res = g_state.step_results[c][best_attempt];
            
            // Update Fitness
            int num_fitness = CHUNK_SIZE / 2;
            int start_fit_idx = c * num_fitness;
            for(int i=0; i<num_fitness; i++) {
                if (start_fit_idx + i < g_state.current_fitness.size()) {
                    g_state.current_fitness[start_fit_idx + i] = best_res.fitness[i];
                }
            }
            
            // Update Shifts
            g_state.chunk_shifts[c] = best_attempt;
            
            // Accumulate Step Stats
            g_state.step_sum_loss += best_res.sum_loss;
            g_state.step_sum_entropy += best_res.sum_entropy;

            // Selected Stats
            if (best_res.sum_loss < selected_min_loss) selected_min_loss = best_res.sum_loss;
            if (best_res.sum_loss > selected_max_loss) selected_max_loss = best_res.sum_loss;
            if (best_res.sum_entropy < selected_min_entropy) selected_min_entropy = best_res.sum_entropy;
            if (best_res.sum_entropy > selected_max_entropy) selected_max_entropy = best_res.sum_entropy;
        }
    }

    // Print Report
    printf("\n[Step %" PRIu64 " Report]\n", g_state.current_step);
    printf("Overall:  Loss [%" PRIu64 ", %" PRIu64 "] | Entropy [%" PRIu64 ", %" PRIu64 "]\n", 
           overall_min_loss, overall_max_loss, overall_min_entropy, overall_max_entropy);
    printf("Selected: Loss [\033[92m%" PRIu64 "\033[0m, \033[91m%" PRIu64 "\033[0m] | Entropy [\033[36m%" PRIu64 "\033[0m, \033[34m%" PRIu64 "\033[0m]\n", 
           selected_min_loss, selected_max_loss, selected_min_entropy, selected_max_entropy);

    // Calculate Avg Loss & Entropy
    double avg_loss = (double)g_state.step_sum_loss / (POPULATION_SIZE * SEQ_LEN * 16.0);
    double avg_entropy = (double)g_state.step_sum_entropy / (POPULATION_SIZE * SEQ_LEN * 16.0);
    
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
    printf("Step %" PRIu64 " | Tokens: %s | Loss: %s%.4f%s | Entropy: %.4f | Time: %.2f ms | Updates: %" PRIu64 " %s | Speed: %.2f tok/s | LR: %.6f | Net: %.2f MB/s (Tx: %s, Rx: %s)\n", 
           g_state.current_step, 
           humanize_tokens(total_tokens).c_str(),
           loss_color, avg_loss, reset_color,
           avg_entropy,
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
    g_state.chunk_completed.assign(total_chunks, false);
    g_state.chunk_finished_attempts.assign(total_chunks, 0);
    g_state.chunk_shifts.assign(total_chunks, 0);
    
    // Reset Job Status
    g_state.job_status.assign(total_chunks, std::vector<int>(LOSS_HAXXING_RETRIES, JOB_IDLE));
    g_state.job_assign_time.assign(total_chunks, std::vector<std::chrono::steady_clock::time_point>(LOSS_HAXXING_RETRIES));
    
    // Reset Results
    g_state.step_results.assign(total_chunks, std::vector<JobResultData>(LOSS_HAXXING_RETRIES));

    // All chunks in same step use same initial value based on step mode
    g_state.chunks_remaining = total_chunks;
    
    // Pre-fill Queue
    g_state.idle_queue.clear();
    for(int c=0; c<total_chunks; c++) {
        for(int a=0; a<LOSS_HAXXING_RETRIES; a++) {
            g_state.idle_queue.push_back({c, a});
        }
    }
    
    std::fill(g_state.current_fitness.begin(), g_state.current_fitness.end(), 0);
    g_state.step_sum_loss = 0;
    g_state.step_sum_entropy = 0;
    g_state.step_total_updates = 0;  // Reset updates counter
    g_state.step_min_updates = UINT64_MAX;
    g_state.step_max_updates = 0;
    g_state.step_transmissions = 0;
    g_state.step_start_time = std::chrono::steady_clock::now();
    
    // Reset network counters for next step
    g_bytes_sent = 0;
    g_bytes_received = 0;
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
            // Assign Job
            Job assigned_job = {-1, -1};
            
            // 1. Try Idle Queue (O(1))
            while (!g_state.idle_queue.empty()) {
                Job candidate = g_state.idle_queue.front();
                g_state.idle_queue.pop_front();
                
                if (g_state.chunk_completed[candidate.chunk_idx]) continue;
                if (g_state.job_status[candidate.chunk_idx][candidate.attempt_idx] == JOB_COMPLETED) continue;
                
                // Found valid idle job
                assigned_job = candidate;
                break;
            }
            
            // 2. If no idle job, check for stragglers (Rate Limited)
            if (assigned_job.chunk_idx == -1) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_check = now - g_state.last_straggler_check;
                if (elapsed_check > std::chrono::milliseconds(1000)) {
                    g_state.last_straggler_check = now;
                    
                    // Scan for stragglers
                    int total_chunks = g_state.job_status.size();
                    for(int c=0; c<total_chunks && assigned_job.chunk_idx == -1; c++) {
                        if (g_state.chunk_completed[c]) continue;
                        
                        for(int a=0; a<LOSS_HAXXING_RETRIES; a++) {
                            if (g_state.job_status[c][a] == JOB_IN_PROGRESS) {
                                auto job_elapsed = now - g_state.job_assign_time[c][a];
                                if (job_elapsed > std::chrono::milliseconds(STRAGGLER_TIMEOUT_MS)) {
                                    // Found straggler
                                    std::cerr << "Chunk " << c << " Attempt " << a 
                                              << " timed out after " 
                                              << std::chrono::duration_cast<std::chrono::seconds>(job_elapsed).count() 
                                              << "s, re-assigning" << std::endl;
                                    assigned_job = {c, a};
                                    break; 
                                }
                            }
                        }
                    }
                }
            }
            
            if (assigned_job.chunk_idx != -1) {
                g_state.job_status[assigned_job.chunk_idx][assigned_job.attempt_idx] = JOB_IN_PROGRESS;
                g_state.job_assign_time[assigned_job.chunk_idx][assigned_job.attempt_idx] = std::chrono::steady_clock::now();
                
                // Found a job
                EggJobResponseHeader resp;
                // Apply shift for this attempt
                uint32_t shift = assigned_job.attempt_idx;
                resp.seed = g_state.current_seed + shift;
                resp.last_step = g_state.current_step;
                resp.data_position = assigned_job.chunk_idx * CHUNK_SIZE; // Start index
                resp.model_size = 0; // No update
                resp.learning_rate = g_state.current_lr;
                
                uint8_t resp_buf[32];
                egg_serialize_job_response_header(resp_buf, &resp);
                send_packet(sock, OP_JOB_RESPONSE, resp_buf, 32);
            } else {
                // No jobs left, wait for next step
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
            if (payload_len < 52) break;
            egg_deserialize_result_header(payload.data(), &res);
            
            int chunk_idx = res.data_position / CHUNK_SIZE;
            
            // Unpack fitness
            int num_fitness = CHUNK_SIZE / 2;
            std::vector<int32_t> h_fit(num_fitness);
            ternary_unpack(payload.data() + 52, num_fitness, h_fit.data());
            
            std::lock_guard<std::mutex> lock(g_state.mutex);
            
            if (res.last_step == g_state.current_step) {
                // Track updates stats (count every transmission)
                g_state.step_transmissions++;
                g_state.step_total_updates += res.updates_count;
                if (res.updates_count < g_state.step_min_updates) g_state.step_min_updates = res.updates_count;
                if (res.updates_count > g_state.step_max_updates) g_state.step_max_updates = res.updates_count;

                // Infer attempt_idx from seed
                // seed = current_seed + attempt_idx
                // attempt_idx = seed - current_seed
                int attempt_idx = (int)(res.seed - g_state.current_seed);
                
                // Validate attempt_idx
                if (attempt_idx < 0 || attempt_idx >= LOSS_HAXXING_RETRIES) {
                    std::cerr << "Invalid attempt_idx " << attempt_idx << " for chunk " << chunk_idx << std::endl;
                    continue;
                }

                // Store Result
                if (!g_state.step_results[chunk_idx][attempt_idx].received) {
                    g_state.step_results[chunk_idx][attempt_idx].sum_loss = res.sum_loss;
                    g_state.step_results[chunk_idx][attempt_idx].sum_entropy = res.sum_entropy;
                    g_state.step_results[chunk_idx][attempt_idx].fitness = h_fit;
                    g_state.step_results[chunk_idx][attempt_idx].received = true;
                }

                // Mark attempt as completed
                if (g_state.job_status[chunk_idx][attempt_idx] != JOB_COMPLETED) {
                    g_state.job_status[chunk_idx][attempt_idx] = JOB_COMPLETED;
                    g_state.chunk_finished_attempts[chunk_idx]++;
                    
                    // Log Progress
                    print_progress_overwrite("[Progress] Chunk %d Attempt %d/%d received. Remaining chunks: %d", 
                           chunk_idx, attempt_idx, LOSS_HAXXING_RETRIES, g_state.chunks_remaining);

                    // Check if chunk is fully complete
                    if (!g_state.chunk_completed[chunk_idx] && g_state.chunk_finished_attempts[chunk_idx] >= LOSS_HAXXING_RETRIES) {
                        g_state.chunk_completed[chunk_idx] = true;
                        g_state.chunks_remaining--;
                        
                        if (g_state.chunks_remaining == 0) {
                            finalize_step();
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
    g_state.chunk_finished_attempts.resize(total_chunks, 0);
    g_state.chunk_shifts.resize(total_chunks, 0);
    
    // Init Job Status
    g_state.job_status.resize(total_chunks, std::vector<int>(LOSS_HAXXING_RETRIES, JOB_IDLE));
    g_state.job_assign_time.resize(total_chunks, std::vector<std::chrono::steady_clock::time_point>(LOSS_HAXXING_RETRIES));
    
    // Init Results
    g_state.step_results.resize(total_chunks, std::vector<JobResultData>(LOSS_HAXXING_RETRIES));

    // All chunks in same step use same initial value based on step mode
    g_state.chunks_remaining = total_chunks;
    
    // Pre-fill Queue
    for(int c=0; c<total_chunks; c++) {
        for(int a=0; a<LOSS_HAXXING_RETRIES; a++) {
            g_state.idle_queue.push_back({c, a});
        }
    }
    
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
    printf("  Haxxing Retries: %d\n", LOSS_HAXXING_RETRIES);
    printf("===================================\n\n");

    g_state.step_start_time = std::chrono::steady_clock::now();

    while (true) {
        int new_socket = accept(server_fd, NULL, NULL);
        if (new_socket < 0) { perror("accept failed"); continue; }
        
        std::thread(handle_client, new_socket).detach();
    }
    
    return 0;
}
