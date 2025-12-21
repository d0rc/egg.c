#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <csignal>
#include <cmath>

#include "../include/protocol.h"
#include "../include/config.h"
#include "../include/model/definitions.h"
#include "../include/utils/io.h"
#include "../include/utils/egg_math.h"
#include "../include/utils/training.h"
#include "../include/utils/ternary_pack.h"
#include "../include/optimizer/adam.cuh"
#include "kernels.cu"
#include "../include/utils/concurrent_queue.h"
#include "../include/worker_job_types.h"

// Global Queues
ConcurrentQueue<JobItem> job_queue;
ConcurrentQueue<ResultItem> result_queue;

// Network Stats
std::atomic<uint64_t> bytes_sent(0);
std::atomic<uint64_t> bytes_received(0);
volatile bool g_running = true;

void sig_handler(int) { g_running = false; }

// --- Vocabulary for Detokenization (when USE_TOKENIZER is defined) ---
#if USE_TOKENIZER
std::vector<std::string> g_vocab;
bool g_vocab_loaded = false;

bool load_vocabulary(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        std::cerr << "Warning: Could not open vocabulary file: " << path << std::endl;
        return false;
    }
    
    uint32_t vocab_size;
    if (fread(&vocab_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    printf("Loading vocabulary with %u tokens...\n", vocab_size);
    g_vocab.resize(vocab_size);
    
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t len;
        if (fread(&len, 4, 1, f) != 1) {
            fclose(f);
            return false;
        }
        g_vocab[i].resize(len);
        if (len > 0 && fread(&g_vocab[i][0], 1, len, f) != len) {
            fclose(f);
            return false;
        }
    }
    
    fclose(f);
    printf("Vocabulary loaded successfully (%u tokens)\n", vocab_size);
    return true;
}

std::string detokenize(const TokenType* tokens, int count) {
    std::string result;
    for (int i = 0; i < count; i++) {
        if (tokens[i] < g_vocab.size()) {
            result += g_vocab[tokens[i]];
        } else {
            result += "<" + std::to_string(tokens[i]) + ">";
        }
    }
    return result;
}
#endif

// Helper: Repack Matrix
void repack_matrix(int8_t *dst, int8_t *src, int rows, int cols) {  
    int32_t *d32 = (int32_t*)dst;
    for(int k=0; k<rows; k+=4) {
        for(int tid=0; tid<cols; tid++) {
            uint32_t val = 0;
            for(int s=0; s<4; s++) {
                int8_t w = src[(k+s)*cols + tid]; 
                val |= ((uint8_t)w) << (s*8);
            }
            int chunk_idx = (k/4) * cols + tid;
            d32[chunk_idx] = val;
        }
    }
}

// Helper: Init Model
void init_model(TransformerModel *model, uint32_t seed) {
    uint32_t rng = seed;
    TransformerModel *temp = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    repack_matrix(model->embedding, (int8_t*)temp->embedding, HIDDEN_DIM, VOCAB_SIZE);
    
    for(int i=0; i<HIDDEN_DIM; i++) model->emb_bias[i] = 0;
    
    // Init Initial MLP
    for(int i=0; i<HIDDEN_DIM; i++) { model->ln_init[i]=16; model->ln_init_bias[i]=0; }
    for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_emb_mlp_up[i] = gen_noise_host(&rng); 
    repack_matrix(model->w_emb_mlp_up, temp->w_emb_mlp_up, HIDDEN_DIM, 4*HIDDEN_DIM);
    for(int i=0; i<4*HIDDEN_DIM; i++) model->mlp_emb_bias_up[i] = 0;

    for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_emb_mlp_down[i] = gen_noise_host(&rng); 
    repack_matrix(model->w_emb_mlp_down, temp->w_emb_mlp_down, 4*HIDDEN_DIM, HIDDEN_DIM);
    for(int i=0; i<HIDDEN_DIM; i++) model->mlp_emb_bias_down[i] = 0;

    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) { 
            model->ln_1[l][i]=16; model->ln_1_bias[l][i]=0; 
            model->ln_2[l][i]=16; model->ln_2_bias[l][i]=0; 
        }
        int d2 = HIDDEN_DIM*HIDDEN_DIM;
        for(int i=0; i<d2; i++) temp->w_q[l][i] = gen_noise_host(&rng); repack_matrix(model->w_q[l], temp->w_q[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_k[l][i] = gen_noise_host(&rng); repack_matrix(model->w_k[l], temp->w_k[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_v[l][i] = gen_noise_host(&rng); repack_matrix(model->w_v[l], temp->w_v[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_o[l][i] = gen_noise_host(&rng); repack_matrix(model->w_o[l], temp->w_o[l], HIDDEN_DIM, HIDDEN_DIM);
        
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_up[l][i] = gen_noise_host(&rng); repack_matrix(model->w_up[l], temp->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM);
        for(int i=0; i<4*HIDDEN_DIM; i++) model->mlp_bias_up[l][i] = 0;
        
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_down[l][i] = gen_noise_host(&rng); repack_matrix(model->w_down[l], temp->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<HIDDEN_DIM; i++) model->mlp_bias_down[l][i] = 0;
    }
    for(int i=0; i<HIDDEN_DIM; i++) { model->ln_f[i]=16; model->ln_f_bias[i]=0; }
    free(temp);
}

// Helper: Send Packet
bool send_packet(int sock, uint8_t opcode, const void* payload, uint32_t len) {
    uint8_t header[EGG_HEADER_SIZE];
    egg_write_header(header, opcode, len);
    if (send(sock, header, EGG_HEADER_SIZE, 0) != EGG_HEADER_SIZE) return false;
    bytes_sent += EGG_HEADER_SIZE;
    if (len > 0) {
        if (send(sock, payload, len, 0) != len) return false;
        bytes_sent += len;
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
    bytes_received += len;
    return true;
}

// Host-side struct to hold device pointers (avoids touching d_model pointer on host)
// Helper: Check Pointer Attributes
void check_pointer(void* ptr, const char* name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        printf("[Memory] %s: Invalid pointer (Error: %s)\n", name, cudaGetErrorString(err));
    } else {
        printf("[Memory] %s: Type=%d (2=Device, 1=Host, 3=Managed), Device=%d, HostPtr=%p\n", 
               name, attr.type, attr.device, attr.hostPointer);
    }
}

struct TransformerModelPointers {
    WeightType* embedding;
    WeightType* emb_bias;
    
#if NTT_MODE != 0
    WeightType* ntt_emb1;
    WeightType* ntt_emb2;
    WeightType* ntt_emb3;
#endif

    WeightType* ln_init;
    WeightType* ln_init_bias;
    WeightType* w_emb_mlp_up;
    WeightType* mlp_emb_bias_up;
    WeightType* w_emb_mlp_down;
    WeightType* mlp_emb_bias_down;

    WeightType* ln_1[N_LAYERS];
    WeightType* ln_1_bias[N_LAYERS];
    WeightType* w_q[N_LAYERS];
    WeightType* w_k[N_LAYERS];
    WeightType* w_v[N_LAYERS];
    WeightType* w_o[N_LAYERS];
    WeightType* ln_2[N_LAYERS];
    WeightType* ln_2_bias[N_LAYERS];
    WeightType* w_up[N_LAYERS];
    WeightType* mlp_bias_up[N_LAYERS];
    WeightType* w_down[N_LAYERS];
    WeightType* mlp_bias_down[N_LAYERS];
    
    WeightType* ln_f;
    WeightType* ln_f_bias;
};

// Helper: Allocate Managed Memory with Residency
void allocate_managed(void** ptr, size_t size, int device_id) {
    cudaMallocManaged(ptr, size);
    cudaMemAdvise(*ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(*ptr, size, cudaMemAdviseSetAccessedBy, device_id);
    cudaMemPrefetchAsync(*ptr, size, device_id, 0);
}

void network_thread_func(std::string server_ip, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        g_running = false;
        return;
    }
    std::cout << "Connected to " << server_ip << " (Network Thread)" << std::endl;

    uint64_t net_step = 0;
    uint64_t net_seed = 42;
    bool should_delay = false;
    auto last_wait_time = std::chrono::steady_clock::now();

    while (g_running) {
        // 1. Send Results
        ResultItem res;
        while (result_queue.try_pop(res)) {
            if (res.type == ResultType::COMPUTE) {
                uint8_t res_buf[52];
                egg_serialize_result_header(res_buf, &res.header);
                
                uint8_t packet_header[EGG_HEADER_SIZE];
                egg_write_header(packet_header, OP_RESULT, 52 + res.header.result_size);
                
                send(sock, packet_header, EGG_HEADER_SIZE, 0);
                send(sock, res_buf, 52, 0);
                if (res.header.result_size > 0) {
                    send(sock, res.payload.data(), res.header.result_size, 0);
                }
                bytes_sent += EGG_HEADER_SIZE + 52 + res.header.result_size;
            } else if (res.type == ResultType::LOG_MESSAGE) {
                send_packet(sock, OP_LOG_MESSAGE, res.payload.data(), res.payload.size());
            }
        }

        // 2. Prefetch
        if (job_queue.size() < 2) {
            if (should_delay) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_wait_time).count() < 100) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                should_delay = false;
            }

            EggJobRequest req;
            req.seed = net_seed;
            req.last_step = net_step;
            req.data_position = 0;
            
            uint8_t req_buf[24];
            egg_serialize_job_request(req_buf, &req);
            if (!send_packet(sock, OP_JOB_REQUEST, req_buf, 24)) {
                g_running = false;
                break;
            }
            
            uint8_t header[EGG_HEADER_SIZE];
            if (!recv_exact(sock, header, EGG_HEADER_SIZE)) {
                g_running = false;
                break;
            }
            
            uint8_t opcode;
            uint32_t payload_len;
            egg_parse_header(header, &opcode, &payload_len);
            
            if (opcode == OP_JOB_RESPONSE) {
                std::vector<uint8_t> payload(payload_len);
                if (!recv_exact(sock, payload.data(), payload_len)) {
                    g_running = false;
                    break;
                }
                
                EggJobResponseHeader resp;
                egg_deserialize_job_response_header(payload.data(), &resp);
                
                if (resp.model_size > 0) {
                    net_step = resp.last_step;
                    net_seed = hash_rng(net_seed, net_step);
                } else if (resp.data_position == (uint64_t)-1) {
                    should_delay = true;
                    last_wait_time = std::chrono::steady_clock::now();
                    continue;
                }
                
                JobItem item;
                item.header = resp;
                item.payload = std::move(payload);
                job_queue.push(std::move(item));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    job_queue.cancel();
    close(sock);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_ip>" << std::endl;
        return 1;
    }
    const char* server_ip = argv[1];
    int port = 12345;

    printf("\n=== EGG DISTRIBUTED WORKER ===\n");
    printf("Model Config:\n");
    printf("  Hidden Dim: %d\n", HIDDEN_DIM);
    printf("  Layers:     %d\n", N_LAYERS);
    printf("  Heads:      %d\n", N_HEADS);
    printf("  Seq Len:    %d\n", SEQ_LEN);
    printf("  Vocab Size: %d\n", VOCAB_SIZE);
    printf("  Chunk Mean Filter: %d (Exp: %.2f)\n", CHUNK_MEAN_FILTER, (double)CHUNK_MEAN_EXPONENT);
    printf("Connecting to Coordinator at %s:%d...\n", server_ip, port);
    printf("==============================\n\n");

    // 1. Init CUDA & Model
    int device_id = 0;
    cudaGetDevice(&device_id);

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    
    init_tables();
    copy_tables_to_device();
    
    TransformerModel *h_model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    AdamModel *h_adam_state = (AdamModel*)calloc(1, sizeof(AdamModel));
    init_model(h_model, 42); // Fixed seed for now
    
    TransformerModel *d_model;
    AdamModel *d_adam_state;
    allocate_managed((void**)&d_model, sizeof(TransformerModel), device_id);
    allocate_managed((void**)&d_adam_state, sizeof(AdamModel), device_id);
    cudaMemcpy(d_model, h_model, sizeof(TransformerModel), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adam_state, h_adam_state, sizeof(AdamModel), cudaMemcpyHostToDevice);
    
    free(h_model);
    free(h_adam_state);

    check_pointer(d_model, "d_model");
    check_pointer(d_adam_state, "d_adam_state");

    // Cache device pointers to avoid touching d_model on host
    TransformerModelPointers d_ptrs;
    d_ptrs.embedding = d_model->embedding;
    d_ptrs.emb_bias = d_model->emb_bias;
#if NTT_MODE != 0
    d_ptrs.ntt_emb1 = d_model->ntt_emb1;
    d_ptrs.ntt_emb2 = d_model->ntt_emb2;
    d_ptrs.ntt_emb3 = d_model->ntt_emb3;
#endif
    d_ptrs.ln_init = d_model->ln_init;
    d_ptrs.ln_init_bias = d_model->ln_init_bias;
    d_ptrs.w_emb_mlp_up = d_model->w_emb_mlp_up;
    d_ptrs.mlp_emb_bias_up = d_model->mlp_emb_bias_up;
    d_ptrs.w_emb_mlp_down = d_model->w_emb_mlp_down;
    d_ptrs.mlp_emb_bias_down = d_model->mlp_emb_bias_down;
    
    for(int l=0; l<N_LAYERS; l++) {
        d_ptrs.ln_1[l] = d_model->ln_1[l];
        d_ptrs.ln_1_bias[l] = d_model->ln_1_bias[l];
        d_ptrs.w_q[l] = d_model->w_q[l];
        d_ptrs.w_k[l] = d_model->w_k[l];
        d_ptrs.w_v[l] = d_model->w_v[l];
        d_ptrs.w_o[l] = d_model->w_o[l];
        d_ptrs.ln_2[l] = d_model->ln_2[l];
        d_ptrs.ln_2_bias[l] = d_model->ln_2_bias[l];
        d_ptrs.w_up[l] = d_model->w_up[l];
        d_ptrs.mlp_bias_up[l] = d_model->mlp_bias_up[l];
        d_ptrs.w_down[l] = d_model->w_down[l];
        d_ptrs.mlp_bias_down[l] = d_model->mlp_bias_down[l];
    }
    d_ptrs.ln_f = d_model->ln_f;
    d_ptrs.ln_f_bias = d_model->ln_f_bias;

    // Load Dataset
    const char* dataset_path = "input.bin";
#ifndef USE_TOKENIZER
    if (access("input.txt", F_OK) != -1) {
        dataset_path = "input.txt";
        printf("Byte-level mode: Loading %s...\n", dataset_path);
    }
#endif

    FILE *f = fopen(dataset_path, "rb");
    if(!f) { std::cerr << "No input file: " << dataset_path << std::endl; return 1; }
    fseek(f,0,SEEK_END); long file_size = ftell(f); fseek(f,0,SEEK_SET);
    long ds_len = file_size / sizeof(TokenType);
    TokenType *h_ds = (TokenType*)malloc(file_size);
    if (fread(h_ds, 1, file_size, f) != file_size) {
        std::cerr << "Error reading " << dataset_path << std::endl;
    }
    fclose(f);
    
    TokenType *d_dataset;
    allocate_managed((void**)&d_dataset, file_size, device_id);
    cudaMemcpy(d_dataset, h_ds, file_size, cudaMemcpyHostToDevice);
    free(h_ds);

    check_pointer(d_dataset, "d_dataset");
    
#if USE_TOKENIZER
    // Load vocabulary for detokenization
    g_vocab_loaded = load_vocabulary("decoding.bin");
    if (!g_vocab_loaded) {
        std::cerr << "Warning: Generation output will show token IDs instead of text" << std::endl;
    }
#endif
    
    // Buffers
    long long *d_loss; allocate_managed((void**)&d_loss, POPULATION_SIZE * sizeof(long long), device_id); // Max size
    long long *d_entropy; allocate_managed((void**)&d_entropy, POPULATION_SIZE * sizeof(long long), device_id);
    int32_t *d_fit; allocate_managed((void**)&d_fit, (POPULATION_SIZE/2) * sizeof(int32_t), device_id);
    uint32_t *d_chunk_shifts; allocate_managed((void**)&d_chunk_shifts, (POPULATION_SIZE/CHUNK_SIZE) * sizeof(uint32_t), device_id);

    // Adaptive Noise Scales
    AdaptiveScales *d_scales;
    allocate_managed((void**)&d_scales, sizeof(AdaptiveScales), device_id);
    cudaMemset(d_scales, ADAPTIVE_NOISE_INIT, sizeof(AdaptiveScales)); // Init to 64

    // Accumulators for Adaptive Noise
    // Ensure buffers are large enough for the largest dimensions (Vocab or 4*Hidden)
    size_t max_dim = (VOCAB_SIZE > 4*HIDDEN_DIM) ? VOCAB_SIZE : 4*HIDDEN_DIM;
    int *d_row_accum; allocate_managed((void**)&d_row_accum, max_dim * sizeof(int), device_id);
    int *d_col_accum; allocate_managed((void**)&d_col_accum, max_dim * sizeof(int), device_id);

    // Graph Update Params
    uint32_t *d_update_seed; 
    float *d_update_lr;
    float *d_update_lr_bias;
    cudaMalloc(&d_update_seed, sizeof(uint32_t));
    cudaMalloc(&d_update_lr, sizeof(float));
    cudaMalloc(&d_update_lr_bias, sizeof(float));
    
    void* ptr_total_updates;
    cudaGetSymbolAddress(&ptr_total_updates, d_total_updates);
    
    cudaStream_t update_stream;
    cudaStreamCreate(&update_stream);
    
    cudaGraph_t update_graph;
    cudaGraphExec_t update_instance = NULL;
    bool graph_created = false;

    ActType *d_kv_cache; 
    size_t kv_size = (size_t)POPULATION_BATCH_SIZE * N_LAYERS * 2 * SEQ_LEN * HIDDEN_DIM; // Approx
    allocate_managed((void**)&d_kv_cache, kv_size, device_id); // Need to ensure enough for chunk size
    
    check_pointer(d_kv_cache, "d_kv_cache");
    printf("[Memory] d_kv_cache size: %.2f GB\n", (double)kv_size / (1024.0*1024.0*1024.0));

    // Generation buffers
    int gen_seed_len = 32;
    int gen_output_len = 64; 
    int total_gen_len = gen_seed_len + gen_output_len;
    TokenType *d_gen_buf; allocate_managed((void**)&d_gen_buf, total_gen_len * sizeof(TokenType), device_id);
    ActType *d_gen_kv; allocate_managed((void**)&d_gen_kv, N_LAYERS * 2 * total_gen_len * HIDDEN_DIM, device_id);

    uint64_t current_step = 0;
    uint64_t current_seed = 42;
    uint64_t last_updates_count = 0;

    // 2. Start Network Thread
    std::thread net_thread(network_thread_func, std::string(server_ip), port);

    // 3. Loop
    while (g_running) {
        JobItem job;
        if (!job_queue.pop(job)) { // Blocking, returns false if cancelled
            continue;
        }

        if (job.header.model_size > 0) {
            // Update Model
            printf("[Worker] Updating model to Step %lu...\n", job.header.last_step);
            
            // Unpack fitness
            size_t num_fitness = POPULATION_SIZE / 2;
            std::vector<int32_t> h_fit(num_fitness);
            ternary_unpack(job.payload.data() + 32, num_fitness, h_fit.data());
            
            cudaMemcpy(d_fit, h_fit.data(), num_fitness * sizeof(int32_t), cudaMemcpyHostToDevice);

            // Extract Chunk Shifts
            size_t packed_size = ternary_pack_estimate_size(num_fitness);
            size_t shifts_offset = 32 + packed_size;
            size_t total_chunks = POPULATION_SIZE / CHUNK_SIZE;
            size_t shifts_size = total_chunks * sizeof(uint32_t);
            
            if (job.payload.size() >= shifts_offset + shifts_size) {
                cudaMemcpy(d_chunk_shifts, job.payload.data() + shifts_offset, shifts_size, cudaMemcpyHostToDevice);
            } else {
                cudaMemset(d_chunk_shifts, 0, shifts_size);
            }
            
            float lr = job.header.learning_rate;
            float lr_bias = lr * 0.1f;
            
            // Update seed/lr on device
            cudaMemcpyAsync(d_update_seed, &current_seed, sizeof(uint32_t), cudaMemcpyHostToDevice, update_stream);
            cudaMemcpyAsync(d_update_lr, &lr, sizeof(float), cudaMemcpyHostToDevice, update_stream);
            cudaMemcpyAsync(d_update_lr_bias, &lr_bias, sizeof(float), cudaMemcpyHostToDevice, update_stream);

            if (!graph_created) {
                printf("[Worker] Capturing CUDA Graph for Optimizer Step...\n");
                auto start_cap = std::chrono::high_resolution_clock::now();
                
                cudaStreamBeginCapture(update_stream, cudaStreamCaptureModeGlobal);
                
                // Reset updates counter
                cudaMemsetAsync(ptr_total_updates, 0, sizeof(unsigned long long), update_stream);

                // Run Updates
                #define UPDATE_OVERLAY(ACCUM, OVERLAY, COUNT) \
                    if (ADAPTIVE_NOISE_MODE > 0) { \
                        update_overlay_kernel<<< (COUNT+255)/256, 256, 0, update_stream >>>(ACCUM, OVERLAY, COUNT); \
                    }

                #define LAUNCH_ADAM_MATRIX(M_PTR, ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, BASE, ROW_OV, COL_OV) \
                    cudaMemsetAsync(d_row_accum, 0, ROWS * sizeof(int), update_stream); \
                    cudaMemsetAsync(d_col_accum, 0, COLS * sizeof(int), update_stream); \
                    update_matrix_adam_kernel<<< (ROWS*COLS+511)/512, 512, 0, update_stream >>>( \
                    (WeightType*)M_PTR, (AdamParam*)ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, \
                    BASE, d_fit, d_update_seed, d_update_lr, d_chunk_shifts, d_row_accum, d_col_accum); \
                    UPDATE_OVERLAY(d_row_accum, ROW_OV, ROWS); \
                    UPDATE_OVERLAY(d_col_accum, COL_OV, COLS);

                #define LAUNCH_ADAM_VECTOR(V_PTR, ADAM_PTR, LEN, SEED_A, SEED_B, BASE, LR_PTR, OV) \
                    cudaMemsetAsync(d_row_accum, 0, LEN * sizeof(int), update_stream); \
                    update_vector_adam_kernel<<< (LEN+255)/256, 256, 0, update_stream >>>( \
                    (WeightType*)V_PTR, (AdamParam*)ADAM_PTR, LEN, SEED_A, SEED_B, \
                    BASE, d_fit, d_update_seed, LR_PTR, d_chunk_shifts, d_row_accum); \
                    UPDATE_OVERLAY(d_row_accum, OV, LEN);

                for(int l=0; l<N_LAYERS; l++) {
                    int s_base = l * 1000;
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_q[l], d_adam_state->w_q[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_Q_A, SEED_OFF_Q_B, s_base, d_scales->w_q_row[l], d_scales->w_q_col[l]);
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_k[l], d_adam_state->w_k[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_K_A, SEED_OFF_K_B, s_base, d_scales->w_k_row[l], d_scales->w_k_col[l]);
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_v[l], d_adam_state->w_v[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_V_A, SEED_OFF_V_B, s_base, d_scales->w_v_row[l], d_scales->w_v_col[l]);
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_o[l], d_adam_state->w_o[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_O_A, SEED_OFF_O_B, s_base, d_scales->w_o_row[l], d_scales->w_o_col[l]);
                    
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_up[l], d_adam_state->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, s_base, d_scales->w_up_row[l], d_scales->w_up_col[l]);
                    LAUNCH_ADAM_MATRIX(d_ptrs.w_down[l], d_adam_state->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, s_base, d_scales->w_down_row[l], d_scales->w_down_col[l]);
                    
                    LAUNCH_ADAM_VECTOR(d_ptrs.ln_1[l], d_adam_state->ln_1[l], HIDDEN_DIM, SEED_OFF_LN_1_A, SEED_OFF_LN_1_B, s_base, d_update_lr, d_scales->ln_1[l]);
                    LAUNCH_ADAM_VECTOR(d_ptrs.ln_1_bias[l], d_adam_state->ln_1_bias[l], HIDDEN_DIM, SEED_OFF_LN_1_BIAS_A, SEED_OFF_LN_1_BIAS_B, s_base, d_update_lr, d_scales->ln_1_bias[l]);
                    
                    LAUNCH_ADAM_VECTOR(d_ptrs.ln_2[l], d_adam_state->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2_A, SEED_OFF_LN_2_B, s_base, d_update_lr, d_scales->ln_2[l]);
                    LAUNCH_ADAM_VECTOR(d_ptrs.ln_2_bias[l], d_adam_state->ln_2_bias[l], HIDDEN_DIM, SEED_OFF_LN_2_BIAS_A, SEED_OFF_LN_2_BIAS_B, s_base, d_update_lr, d_scales->ln_2_bias[l]);
                    
                    LAUNCH_ADAM_VECTOR(d_ptrs.mlp_bias_up[l], d_adam_state->mlp_bias_up[l], 4*HIDDEN_DIM, SEED_OFF_MLP_BIAS_UP_A, SEED_OFF_MLP_BIAS_UP_B, s_base, d_update_lr, d_scales->mlp_bias_up[l]);
                    LAUNCH_ADAM_VECTOR(d_ptrs.mlp_bias_down[l], d_adam_state->mlp_bias_down[l], HIDDEN_DIM, SEED_OFF_MLP_BIAS_DOWN_A, SEED_OFF_MLP_BIAS_DOWN_B, s_base, d_update_lr, d_scales->mlp_bias_down[l]);
                }
                
                LAUNCH_ADAM_VECTOR(d_ptrs.ln_f, d_adam_state->ln_f, HIDDEN_DIM, SEED_OFF_LN_F_A, SEED_OFF_LN_F_B, 0, d_update_lr, d_scales->ln_f);
                LAUNCH_ADAM_VECTOR(d_ptrs.ln_f_bias, d_adam_state->ln_f_bias, HIDDEN_DIM, SEED_OFF_LN_F_BIAS_A, SEED_OFF_LN_F_BIAS_B, 0, d_update_lr, d_scales->ln_f_bias);
                
                LAUNCH_ADAM_VECTOR(d_ptrs.ln_init, d_adam_state->ln_init, HIDDEN_DIM, SEED_OFF_LN_INIT_A, SEED_OFF_LN_INIT_B, 0, d_update_lr, d_scales->ln_init);
                LAUNCH_ADAM_VECTOR(d_ptrs.ln_init_bias, d_adam_state->ln_init_bias, HIDDEN_DIM, SEED_OFF_LN_INIT_BIAS_A, SEED_OFF_LN_INIT_BIAS_B, 0, d_update_lr, d_scales->ln_init_bias);

                LAUNCH_ADAM_MATRIX(d_ptrs.w_emb_mlp_up, d_adam_state->w_emb_mlp_up, HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_UP_A, SEED_OFF_EMB_MLP_UP_B, 0, d_scales->w_emb_mlp_up_row, d_scales->w_emb_mlp_up_col);
                LAUNCH_ADAM_VECTOR(d_ptrs.mlp_emb_bias_up, d_adam_state->mlp_emb_bias_up, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_UP_A, SEED_OFF_EMB_MLP_BIAS_UP_B, 0, d_update_lr, d_scales->mlp_emb_bias_up);

                LAUNCH_ADAM_MATRIX(d_ptrs.w_emb_mlp_down, d_adam_state->w_emb_mlp_down, 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_EMB_MLP_DOWN_A, SEED_OFF_EMB_MLP_DOWN_B, 0, d_scales->w_emb_mlp_down_row, d_scales->w_emb_mlp_down_col);
                LAUNCH_ADAM_VECTOR(d_ptrs.mlp_emb_bias_down, d_adam_state->mlp_emb_bias_down, HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_DOWN_A, SEED_OFF_EMB_MLP_BIAS_DOWN_B, 0, d_update_lr, d_scales->mlp_emb_bias_down);

                LAUNCH_ADAM_VECTOR(d_ptrs.emb_bias, d_adam_state->emb_bias, HIDDEN_DIM, SEED_OFF_EMB_BIAS_A, SEED_OFF_EMB_BIAS_B, 0, d_update_lr_bias, d_scales->emb_bias);
                
                cudaMemsetAsync(d_row_accum, 0, VOCAB_SIZE * sizeof(int), update_stream);
                cudaMemsetAsync(d_col_accum, 0, HIDDEN_DIM * sizeof(int), update_stream);
                update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512, 0, update_stream >>>(
                    (WeightType*)d_ptrs.embedding, 
                    (AdamParam*)d_adam_state->embedding, 
                    HIDDEN_DIM, VOCAB_SIZE, 
                    SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 
                    0, d_fit, d_update_seed, d_update_lr, d_chunk_shifts,
                    d_col_accum, d_row_accum
                );
                UPDATE_OVERLAY(d_col_accum, d_scales->embedding_col, HIDDEN_DIM);
                UPDATE_OVERLAY(d_row_accum, d_scales->embedding_row, VOCAB_SIZE);
                
                cudaStreamEndCapture(update_stream, &update_graph);
                cudaGraphInstantiate(&update_instance, update_graph, NULL, NULL, 0);
                graph_created = true;
                
                auto end_cap = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> ms = end_cap - start_cap;
                
                size_t num_nodes = 0;
                cudaGraphGetNodes(update_graph, NULL, &num_nodes);
                printf("[Worker] Graph captured: %zu nodes in %.2f ms\n", num_nodes, ms.count());
            }
            
            cudaGraphLaunch(update_instance, update_stream);
            
            // Read updates count
            unsigned long long updates = 0;
            cudaMemcpyAsync(&updates, ptr_total_updates, sizeof(unsigned long long), cudaMemcpyDeviceToHost, update_stream);
            cudaStreamSynchronize(update_stream);
            last_updates_count = updates;

            current_step = job.header.last_step;
            current_seed = hash_rng(current_seed, current_step);
            
            continue;
        }
        
        // Task Assigned
        int chunk_idx = job.header.data_position / CHUNK_SIZE;
        int count = CHUNK_SIZE;
        uint32_t task_seed = job.header.seed;

        printf("[Worker] Processing Step %lu, Chunk %d\n", current_step, chunk_idx);

        // Generation (Chunk 0 only, every 5 steps)
        if (chunk_idx == 0 && current_step % 5 == 0) {
            // Copy seed data
            cudaMemcpy(d_gen_buf, d_dataset + (current_step*SEQ_LEN) % (ds_len-SEQ_LEN), gen_seed_len * sizeof(TokenType), cudaMemcpyDeviceToDevice);
            
            size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM);
            generate_sequence_kernel<<<1, EGG_BLOCK_THREADS, sm_size>>>(
                d_gen_buf, gen_seed_len, gen_output_len, d_model, d_gen_kv, task_seed+999,
                SAMPLING_TEMP, SAMPLING_MIN_P, SAMPLING_PRESENCE_PENALTY
            );
            cudaDeviceSynchronize();
            
            TokenType h_buf[256];
            cudaMemcpy(h_buf, d_gen_buf, total_gen_len * sizeof(TokenType), cudaMemcpyDeviceToHost);
            
            // Format String with detokenization
            std::string gen_str = "\n--- GENERATION ---\n\033[32m";
#if USE_TOKENIZER
            if (g_vocab_loaded) {
                gen_str += detokenize(h_buf, gen_seed_len);
                gen_str += "\033[36m";
                gen_str += detokenize(h_buf + gen_seed_len, gen_output_len);
            } else {
                gen_str += "[";
                for(int i=0; i<gen_seed_len; i++) {
                    if (i > 0) gen_str += ",";
                    gen_str += std::to_string(h_buf[i]);
                }
                gen_str += "]\033[36m[";
                for(int i=gen_seed_len; i<total_gen_len; i++) {
                    if (i > gen_seed_len) gen_str += ",";
                    gen_str += std::to_string(h_buf[i]);
                }
                gen_str += "]";
            }
#else
            for(int i=0; i<gen_seed_len; i++) {
                char c = h_buf[i]; gen_str += (c>=32 && c<=126) ? c : '.';
            }
            gen_str += "\033[36m";
            for(int i=gen_seed_len; i<total_gen_len; i++) {
                char c = h_buf[i]; gen_str += (c>=32 && c<=126) ? c : '.';
            }
#endif
            gen_str += "\033[0m\n\n";
            
            // Send Log
            ResultItem log_item;
            log_item.type = ResultType::LOG_MESSAGE;
            log_item.payload.assign(gen_str.begin(), gen_str.end());
            result_queue.push(std::move(log_item));
        }

        // Run Kernel
        int global_pop_offset = job.header.data_position;
        size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM); 
        
        train_sequence_kernel<<<count, EGG_BLOCK_THREADS, sm_size>>>(
            d_dataset, ds_len, current_step*SEQ_LEN, d_model, d_kv_cache, 
            d_loss, d_entropy, task_seed, global_pop_offset, current_step, d_scales
        );
        
        // Copy Result
        std::vector<long long> h_loss(count);
        std::vector<long long> h_entropy(count);
        cudaMemcpy(h_loss.data(), d_loss, count * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_entropy.data(), d_entropy, count * sizeof(long long), cudaMemcpyDeviceToHost);
        
        // Compute Fitness
        int64_t sum_loss = 0;
        int64_t sum_entropy = 0;
        int num_fitness = count / 2;
        std::vector<int32_t> diffs(num_fitness);
        
        double sum_diff_val = 0.0;
        for(int i=0; i<num_fitness; i++) {
            long long p = h_loss[2*i];
            long long n = h_loss[2*i+1];
            sum_loss += p + n;
            sum_entropy += h_entropy[2*i] + h_entropy[2*i+1];
            diffs[i] = (int32_t)(n - p); // Diff should fit in int32 usually, or we clamp?
            // If n-p overflows int32, we should clamp it for fitness calculation
            long long diff_ll = n - p;
            if (diff_ll > INT_MAX) diffs[i] = INT_MAX;
            else if (diff_ll < INT_MIN) diffs[i] = INT_MIN;
            else diffs[i] = (int32_t)diff_ll;

#if LOSS_HAXXING_DIRECTION == 'entmax' && ENTMAX_IGNORE_ZERO_ENT_DATA
            if (h_entropy[2*i] == 0 || h_entropy[2*i+1] == 0) {
                diffs[i] = 0;
            }
#endif

            sum_diff_val += diffs[i];
        }

#if CHUNK_MEAN_FILTER
        double mean_diff = sum_diff_val / num_fitness;
        if (mean_diff != 0.0) {
            double sign = (mean_diff > 0) ? 1.0 : -1.0;
            mean_diff = sign * std::pow(std::abs(mean_diff), (double)CHUNK_MEAN_EXPONENT);
        }
        for(int i=0; i<num_fitness; i++) {
            diffs[i] += (int32_t)mean_diff;
        }
#endif

        std::vector<int32_t> h_fit(num_fitness);

#if USE_ADAPTIVE_THRESHOLD
        double sum_abs_diff = 0.0;
        for(int i=0; i<num_fitness; i++) {
            sum_abs_diff += std::abs((double)diffs[i]);
        }
        double mad = sum_abs_diff / num_fitness;
        double threshold = mad * ADAPTIVE_THRESHOLD_ALPHA;
        for(int i=0; i<num_fitness; i++) {
            if (std::abs((double)diffs[i]) < threshold) {
                h_fit[i] = 0;
            } else {
                h_fit[i] = (diffs[i] > 0) ? 1 : -1;
            }
        }
#else
        for(int i=0; i<num_fitness; i++) {
            h_fit[i] = (diffs[i] > 0) ? 1 : ((diffs[i] < 0) ? -1 : 0);
        }
#endif
        
        // Pack
        size_t packed_size = ternary_pack_estimate_size(num_fitness);
        std::vector<uint8_t> packed_fit(packed_size);
        ternary_pack(h_fit.data(), num_fitness, packed_fit.data());
        
        // Send Result
        ResultItem res_item;
        res_item.type = ResultType::COMPUTE;
        res_item.header.seed = task_seed;
        res_item.header.last_step = current_step;
        res_item.header.data_position = job.header.data_position;
        res_item.header.updates_count = last_updates_count;
        res_item.header.sum_loss = sum_loss;
        res_item.header.sum_entropy = sum_entropy;
        res_item.header.result_size = packed_size;
        res_item.payload = std::move(packed_fit);
        
        last_updates_count = 0;
        
        result_queue.push(std::move(res_item));
    }
    
    g_running = false;
    net_thread.join();
    
    printf("\n[Worker] Goodbye!\n");
    return 0;
}
