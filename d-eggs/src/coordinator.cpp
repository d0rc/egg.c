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

// Constants
#define PORT 12345
// CHUNK_SIZE is defined in config.h
#define MAX_HISTORY 5000
#define STRAGGLER_TIMEOUT_MS 90000  // 90 seconds before re-assigning a chunk

// Network Stats (global atomics)
std::atomic<uint64_t> g_bytes_sent(0);
std::atomic<uint64_t> g_bytes_received(0);

// Global State
struct GlobalState {
    uint64_t current_step = 0;
    uint64_t current_seed = 42; // Initial seed
    
    // History of fitness vectors for catch-up
    // Map step -> vector (packed)
    std::map<uint64_t, std::vector<uint8_t>> fitness_history;
    
    // Current Step State
    std::vector<int32_t> current_fitness; // Size: POPULATION_SIZE / 2
    int64_t step_sum_loss = 0;            // Aggregated loss for current step
    std::vector<bool> chunk_completed;    // Size: POPULATION_SIZE / CHUNK_SIZE
    std::vector<bool> chunk_in_progress;  // Size: POPULATION_SIZE / CHUNK_SIZE
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

// Helper: Compute Fitness
void compute_fitness(const std::vector<int32_t>& losses, std::vector<int32_t>& fitnesses) {
    int count = losses.size() / 2;
    fitnesses.resize(count);
    for(int i=0; i<count; i++) {
        int32_t p = losses[2*i];
        int32_t n = losses[2*i+1];
        fitnesses[i] = (p < n) ? 1 : ((n < p) ? -1 : 0);
    }
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
                // We need fitness for step `req.last_step` to advance to `req.last_step + 1`
                // Wait, logic check:
                // Client is at S. Wants to go to S+1. Needs Fitness(S).
                // Server has History[S] -> Fitness(S).
                
                auto it = g_state.fitness_history.find(req.last_step);
                if (it != g_state.fitness_history.end()) {
                    const auto& packed_fit = it->second;
                    
                    EggJobResponseHeader resp;
                    resp.seed = g_state.current_seed; // Not used for update, but keep consistent
                    resp.last_step = req.last_step + 1; // Target step
                    resp.data_position = 0;
                    resp.model_size = packed_fit.size();
                    
                    // Serialize header
                    uint8_t resp_buf[28];
                    egg_serialize_job_response_header(resp_buf, &resp);
                    
                    // Send Header + HeaderPayload + ModelData
                    // Protocol says: Header(10) + Payload(28 + ModelSize)
                    
                    uint8_t packet_header[EGG_HEADER_SIZE];
                    egg_write_header(packet_header, OP_JOB_RESPONSE, 28 + resp.model_size);
                    
                    send(sock, packet_header, EGG_HEADER_SIZE, 0);
                    send(sock, resp_buf, 28, 0);
                    send(sock, packed_fit.data(), resp.model_size, 0);
                    
                    continue; // Loop to let client apply update
                } else {
                    // Too old or future?
                    // If too old, we should send SYNC (not implemented yet, just disconnect or wait)
                    std::cerr << "Client too old: " << req.last_step << " vs " << g_state.current_step << std::endl;
                    // Send empty wait
                    EggJobResponseHeader resp = {0, 0, 0, 0};
                    uint8_t resp_buf[28];
                    egg_serialize_job_response_header(resp_buf, &resp);
                    send_packet(sock, OP_JOB_RESPONSE, resp_buf, 28);
                    continue;
                }
            }
            
            // Client is up to date (req.last_step == g_state.current_step)
            // Assign Chunk
            int chunk_idx = -1;
            
            // Queue-based assignment with straggler handling
            // We take from front, check if available (not in-progress or timed out)
            // Re-queue at back only for straggler handling
            
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
                resp.seed = g_state.current_seed;
                resp.last_step = g_state.current_step;
                resp.data_position = chunk_idx * CHUNK_SIZE; // Start index
                resp.model_size = 0; // No update
                
                uint8_t resp_buf[28];
                egg_serialize_job_response_header(resp_buf, &resp);
                send_packet(sock, OP_JOB_RESPONSE, resp_buf, 28);
            } else {
                // No chunks left, wait for next step
                EggJobResponseHeader resp = {0, 0, 0, 0}; // data_pos=0 means wait? No, protocol says data_pos is start.
                // How to signal WAIT?
                // Protocol note: "If model_size == 0, no model update... Compute nodes skip processing."
                // But we need to signal "No Job".
                // I'll use `data_position = 0xFFFFFFFFFFFFFFFF` (-1) as WAIT signal.
                resp.data_position = (uint64_t)-1;
                
                uint8_t resp_buf[28];
                egg_serialize_job_response_header(resp_buf, &resp);
                send_packet(sock, OP_JOB_RESPONSE, resp_buf, 28);
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
                    g_state.chunk_completed[chunk_idx] = true;
                    g_state.chunks_remaining--;
                    
                    // Accumulate Loss
                    g_state.step_sum_loss += res.sum_loss;
                    
                    // Copy fitness
                    int start_fit_idx = res.data_position / 2;
                    for(int i=0; i<num_fitness; i++) {
                        if (start_fit_idx + i < g_state.current_fitness.size()) {
                            g_state.current_fitness[start_fit_idx + i] = h_fit[i];
                        }
                    }
                    
                    // std::cout << "Step " << g_state.current_step << ": Chunk " << chunk_idx << " done. Remaining: " << g_state.chunks_remaining << std::endl;
                    
                    if (g_state.chunks_remaining == 0) {
                        // Step Complete
                        // Calculate Avg Loss
                        double avg_loss = (double)g_state.step_sum_loss / (POPULATION_SIZE * SEQ_LEN * 16.0);
                        
                        // Calculate Time and Speed
                        auto now = std::chrono::steady_clock::now();
                        double step_ms = std::chrono::duration<double, std::milli>(now - g_state.step_start_time).count();
                        double tokens_per_sec = (double)(POPULATION_SIZE * SEQ_LEN) / (step_ms / 1000.0);
                        float current_lr = get_learning_rate(g_state.current_step);
                        
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
                        printf("Step %" PRIu64 " | Tokens: %s | Loss: %s%.4f%s | Time: %.2f ms | Updates: %" PRIu64 " %s | Speed: %.2f tok/s | LR: %.3f | Net: %.2f MB/s (Tx: %s, Rx: %s)\n", 
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
                        g_state.fitness_history[g_state.current_step] = packed_fit;

                        if (g_state.fitness_history.size() > MAX_HISTORY) {
                            g_state.fitness_history.erase(g_state.fitness_history.begin());
                        }
                        
                        // Advance
                        g_state.current_step++;
                        g_state.current_seed = hash_rng(g_state.current_seed, g_state.current_step); // Simple evolution
                        
                        // Reset for next step
                        int total_chunks = POPULATION_SIZE / CHUNK_SIZE;
                        g_state.chunk_completed.assign(total_chunks, false);
                        g_state.chunk_in_progress.assign(total_chunks, false);
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
    
    close(sock);
    std::cout << "Client disconnected: " << sock << std::endl;
}

int main(int argc, char** argv) {
    // Register signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Init State
    g_state.current_fitness.resize(POPULATION_SIZE / 2);
    int total_chunks = POPULATION_SIZE / CHUNK_SIZE;
    g_state.chunk_completed.resize(total_chunks, false);
    g_state.chunk_in_progress.resize(total_chunks, false);
    g_state.chunk_assign_time.resize(total_chunks);
    g_state.chunks_remaining = total_chunks;
    for(int i=0; i<total_chunks; i++) g_state.chunk_queue.push_back(i);
    
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
