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

#include "../include/protocol.h"
#include "../include/config.h"
#include "../include/model/definitions.h"
#include "../include/utils/io.h"
#include "../include/utils/egg_math.h"
#include "../include/utils/training.h"
#include "../include/utils/ternary_pack.h"
#include "../include/optimizer/adam.cuh"
#include "kernels.cu"

// Network Stats
std::atomic<uint64_t> bytes_sent(0);
std::atomic<uint64_t> bytes_received(0);

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
    printf("Connecting to Coordinator at %s:%d...\n", server_ip, port);
    printf("==============================\n\n");

    // 1. Init CUDA & Model
    init_tables();
    copy_tables_to_device();
    
    TransformerModel *h_model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    AdamModel *h_adam_state = (AdamModel*)calloc(1, sizeof(AdamModel));
    init_model(h_model, 42); // Fixed seed for now
    
    TransformerModel *d_model;
    AdamModel *d_adam_state;
    cudaMalloc(&d_model, sizeof(TransformerModel));
    cudaMalloc(&d_adam_state, sizeof(AdamModel));
    cudaMemcpy(d_model, h_model, sizeof(TransformerModel), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adam_state, h_adam_state, sizeof(AdamModel), cudaMemcpyHostToDevice);
    
    // Load Dataset
    FILE *f = fopen("input.bin", "rb"); // Assume input.bin exists
    if(!f) { std::cerr << "No input.bin" << std::endl; return 1; }
    fseek(f,0,SEEK_END); long file_size = ftell(f); fseek(f,0,SEEK_SET);
    long ds_len = file_size / sizeof(TokenType);
    TokenType *h_ds = (TokenType*)malloc(file_size);
    if (fread(h_ds, 1, file_size, f) != file_size) {
        std::cerr << "Error reading input.bin" << std::endl;
    }
    fclose(f);
    
    TokenType *d_dataset;
    cudaMalloc(&d_dataset, file_size);
    cudaMemcpy(d_dataset, h_ds, file_size, cudaMemcpyHostToDevice);
    
#if USE_TOKENIZER
    // Load vocabulary for detokenization
    g_vocab_loaded = load_vocabulary("decoding.bin");
    if (!g_vocab_loaded) {
        std::cerr << "Warning: Generation output will show token IDs instead of text" << std::endl;
    }
#endif
    
    // Buffers
    int32_t *d_loss; cudaMalloc(&d_loss, POPULATION_SIZE * sizeof(int32_t)); // Max size
    int32_t *d_fit; cudaMalloc(&d_fit, (POPULATION_SIZE/2) * sizeof(int32_t));
    ActType *d_kv_cache; 
    size_t kv_size = (size_t)POPULATION_BATCH_SIZE * N_LAYERS * 2 * SEQ_LEN * HIDDEN_DIM; // Approx
    cudaMalloc(&d_kv_cache, kv_size); // Need to ensure enough for chunk size
    
    // Generation buffers
    int gen_seed_len = 32;
    int gen_output_len = 64; 
    int total_gen_len = gen_seed_len + gen_output_len;
    TokenType *d_gen_buf; cudaMalloc(&d_gen_buf, total_gen_len * sizeof(TokenType));
    ActType *d_gen_kv; cudaMalloc(&d_gen_kv, N_LAYERS * 2 * total_gen_len * HIDDEN_DIM);

    uint64_t current_step = 0;
    uint64_t current_seed = 42;
    uint64_t last_updates_count = 0;

    // 2. Connect
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &serv_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed"); return 1;
    }
    std::cout << "Connected to " << server_ip << std::endl;

    // 3. Loop
    while (true) {
        // Send JOB_REQUEST
        EggJobRequest req;
        req.seed = current_seed;
        req.last_step = current_step;
        req.data_position = 0; // Not used in request
        
        uint8_t req_buf[24];
        egg_serialize_job_request(req_buf, &req);
        send_packet(sock, OP_JOB_REQUEST, req_buf, 24);
        
        // Recv Header
        uint8_t header[EGG_HEADER_SIZE];
        if (!recv_exact(sock, header, EGG_HEADER_SIZE)) break;
        
        uint8_t opcode;
        uint32_t payload_len;
        egg_parse_header(header, &opcode, &payload_len);
        
        if (opcode == OP_JOB_RESPONSE) {
            std::vector<uint8_t> payload(payload_len);
            recv_exact(sock, payload.data(), payload_len);
            
            EggJobResponseHeader resp;
            egg_deserialize_job_response_header(payload.data(), &resp);
            
            if (resp.model_size > 0) {
                // Update Model
                printf("[Worker] Updating model to Step %lu...\n", resp.last_step);
                
                // Unpack fitness
                size_t num_fitness = POPULATION_SIZE / 2;
                std::vector<int32_t> h_fit(num_fitness);
                ternary_unpack(payload.data() + 28, num_fitness, h_fit.data());
                
                cudaMemcpy(d_fit, h_fit.data(), num_fitness * sizeof(int32_t), cudaMemcpyHostToDevice);
                
                float lr = get_learning_rate(current_step);
                
                // Reset updates counter
                unsigned long long zero = 0;
                cudaMemcpyToSymbol(d_total_updates, &zero, sizeof(unsigned long long));

                // Run Updates
                #define LAUNCH_ADAM_MATRIX(M_PTR, ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, BASE) \
                    update_matrix_adam_kernel<<< (ROWS*COLS+511)/512, 512 >>>( \
                    (WeightType*)M_PTR, (AdamParam*)ADAM_PTR, ROWS, COLS, SEED_A, SEED_B, \
                    BASE, d_fit, current_seed, lr)

                #define LAUNCH_ADAM_VECTOR(V_PTR, ADAM_PTR, LEN, SEED_A, SEED_B, BASE, LR_SCALE) \
                    update_vector_adam_kernel<<< (LEN+255)/256, 256 >>>( \
                    (WeightType*)V_PTR, (AdamParam*)ADAM_PTR, LEN, SEED_A, SEED_B, \
                    BASE, d_fit, current_seed, lr * LR_SCALE)

                for(int l=0; l<N_LAYERS; l++) {
                    int s_base = l * 1000;
                    LAUNCH_ADAM_MATRIX(d_model->w_q[l], d_adam_state->w_q[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_Q_A, SEED_OFF_Q_B, s_base);
                    LAUNCH_ADAM_MATRIX(d_model->w_k[l], d_adam_state->w_k[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_K_A, SEED_OFF_K_B, s_base);
                    LAUNCH_ADAM_MATRIX(d_model->w_v[l], d_adam_state->w_v[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_V_A, SEED_OFF_V_B, s_base);
                    LAUNCH_ADAM_MATRIX(d_model->w_o[l], d_adam_state->w_o[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_O_A, SEED_OFF_O_B, s_base);
                    
                    LAUNCH_ADAM_MATRIX(d_model->w_up[l], d_adam_state->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, s_base);
                    LAUNCH_ADAM_MATRIX(d_model->w_down[l], d_adam_state->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, s_base);
                    
                    LAUNCH_ADAM_VECTOR(d_model->ln_1[l], d_adam_state->ln_1[l], HIDDEN_DIM, SEED_OFF_LN_1_A, SEED_OFF_LN_1_B, s_base, 1.0f);
                    LAUNCH_ADAM_VECTOR(d_model->ln_1_bias[l], d_adam_state->ln_1_bias[l], HIDDEN_DIM, SEED_OFF_LN_1_BIAS_A, SEED_OFF_LN_1_BIAS_B, s_base, 1.0f);
                    
                    LAUNCH_ADAM_VECTOR(d_model->ln_2[l], d_adam_state->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2_A, SEED_OFF_LN_2_B, s_base, 1.0f);
                    LAUNCH_ADAM_VECTOR(d_model->ln_2_bias[l], d_adam_state->ln_2_bias[l], HIDDEN_DIM, SEED_OFF_LN_2_BIAS_A, SEED_OFF_LN_2_BIAS_B, s_base, 1.0f);
                    
                    LAUNCH_ADAM_VECTOR(d_model->mlp_bias_up[l], d_adam_state->mlp_bias_up[l], 4*HIDDEN_DIM, SEED_OFF_MLP_BIAS_UP_A, SEED_OFF_MLP_BIAS_UP_B, s_base, 1.0f);
                    LAUNCH_ADAM_VECTOR(d_model->mlp_bias_down[l], d_adam_state->mlp_bias_down[l], HIDDEN_DIM, SEED_OFF_MLP_BIAS_DOWN_A, SEED_OFF_MLP_BIAS_DOWN_B, s_base, 1.0f);
                }
                
                LAUNCH_ADAM_VECTOR(d_model->ln_f, d_adam_state->ln_f, HIDDEN_DIM, SEED_OFF_LN_F_A, SEED_OFF_LN_F_B, 0, 1.0f);
                LAUNCH_ADAM_VECTOR(d_model->ln_f_bias, d_adam_state->ln_f_bias, HIDDEN_DIM, SEED_OFF_LN_F_BIAS_A, SEED_OFF_LN_F_BIAS_B, 0, 1.0f);
                
                LAUNCH_ADAM_VECTOR(d_model->ln_init, d_adam_state->ln_init, HIDDEN_DIM, SEED_OFF_LN_INIT_A, SEED_OFF_LN_INIT_B, 0, 1.0f);
                LAUNCH_ADAM_VECTOR(d_model->ln_init_bias, d_adam_state->ln_init_bias, HIDDEN_DIM, SEED_OFF_LN_INIT_BIAS_A, SEED_OFF_LN_INIT_BIAS_B, 0, 1.0f);

                LAUNCH_ADAM_MATRIX(d_model->w_emb_mlp_up, d_adam_state->w_emb_mlp_up, HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_UP_A, SEED_OFF_EMB_MLP_UP_B, 0);
                LAUNCH_ADAM_VECTOR(d_model->mlp_emb_bias_up, d_adam_state->mlp_emb_bias_up, 4*HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_UP_A, SEED_OFF_EMB_MLP_BIAS_UP_B, 0, 1.0f);

                LAUNCH_ADAM_MATRIX(d_model->w_emb_mlp_down, d_adam_state->w_emb_mlp_down, 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_EMB_MLP_DOWN_A, SEED_OFF_EMB_MLP_DOWN_B, 0);
                LAUNCH_ADAM_VECTOR(d_model->mlp_emb_bias_down, d_adam_state->mlp_emb_bias_down, HIDDEN_DIM, SEED_OFF_EMB_MLP_BIAS_DOWN_A, SEED_OFF_EMB_MLP_BIAS_DOWN_B, 0, 1.0f);

                LAUNCH_ADAM_VECTOR(d_model->emb_bias, d_adam_state->emb_bias, HIDDEN_DIM, SEED_OFF_EMB_BIAS_A, SEED_OFF_EMB_BIAS_B, 0, 0.1f);
                
                update_matrix_adam_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512 >>>(
                    (WeightType*)d_model->embedding, 
                    (AdamParam*)d_adam_state->embedding, 
                    HIDDEN_DIM, VOCAB_SIZE, 
                    SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 
                    0, d_fit, current_seed, lr
                );
                
                cudaDeviceSynchronize();
                
                // Read updates count
                unsigned long long updates = 0;
                cudaMemcpyFromSymbol(&updates, d_total_updates, sizeof(unsigned long long));
                last_updates_count = updates;

                current_step = resp.last_step; // Should be current_step + 1
                current_seed = hash_rng(current_seed, current_step); // Evolve seed
                
                // std::cout << "Updated to Step " << current_step << std::endl;
                
                // Loop again to get task
                continue;
            }
            
            if (resp.data_position == (uint64_t)-1) {
                // Wait
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Task Assigned
            int chunk_idx = resp.data_position / CHUNK_SIZE;
            int count = CHUNK_SIZE;
            
            printf("[Worker] Processing Step %lu, Chunk %d\n", current_step, chunk_idx);

            // Generation (Chunk 0 only, every 5 steps)
            if (chunk_idx == 0 && current_step % 5 == 0) {
                // Copy seed data
                cudaMemcpy(d_gen_buf, d_dataset + (current_step*SEQ_LEN) % (ds_len-SEQ_LEN), gen_seed_len * sizeof(TokenType), cudaMemcpyDeviceToDevice);
                
                size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM);
                generate_sequence_kernel<<<1, EGG_BLOCK_THREADS, sm_size>>>(
                    d_gen_buf, gen_seed_len, gen_output_len, d_model, d_gen_kv, current_seed+999,
                    SAMPLING_TEMP, SAMPLING_MIN_P, SAMPLING_PRESENCE_PENALTY
                );
                cudaDeviceSynchronize();
                
                TokenType h_buf[256];
                cudaMemcpy(h_buf, d_gen_buf, total_gen_len * sizeof(TokenType), cudaMemcpyDeviceToHost);
                
                // Format String with detokenization
                std::string gen_str = "\n--- GENERATION ---\n\033[32m";
#if USE_TOKENIZER
                if (g_vocab_loaded) {
                    // Detokenize using vocabulary
                    gen_str += detokenize(h_buf, gen_seed_len);
                    gen_str += "\033[36m";
                    gen_str += detokenize(h_buf + gen_seed_len, gen_output_len);
                } else {
                    // Fallback: show token IDs
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
                // Byte-level vocab: display characters directly
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
                send_packet(sock, OP_LOG_MESSAGE, gen_str.c_str(), gen_str.size());
            }

            // Run Kernel
            // Need to map chunk_idx to global_pop_offset
            int global_pop_offset = resp.data_position;
            
            // We need to run `count` threads/blocks.
            // `train_sequence_kernel` uses `blockIdx.x` as index into population chunk.
            // So launch `count` blocks.
            
            // sm_size calculation
            size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM); 
            
            train_sequence_kernel<<<count, EGG_BLOCK_THREADS, sm_size>>>(
                d_dataset, ds_len, current_step*SEQ_LEN, d_model, d_kv_cache, 
                d_loss, current_seed, global_pop_offset, current_step
            );
            
            // Copy Result
            std::vector<int32_t> h_loss(count);
            cudaMemcpy(h_loss.data(), d_loss, count * sizeof(int32_t), cudaMemcpyDeviceToHost);
            
            // Compute Fitness & Sum Loss
            int64_t sum_loss = 0;
            int num_fitness = count / 2;
            std::vector<int32_t> h_fit(num_fitness);
            for(int i=0; i<num_fitness; i++) {
                int32_t p = h_loss[2*i];
                int32_t n = h_loss[2*i+1];
                sum_loss += p + n;
                h_fit[i] = (p < n) ? 1 : ((n < p) ? -1 : 0);
            }
            
            // Pack
            size_t packed_size = ternary_pack_estimate_size(num_fitness);
            std::vector<uint8_t> packed_fit(packed_size);
            ternary_pack(h_fit.data(), num_fitness, packed_fit.data());
            
            // Send Result
            EggResultHeader res;
            res.seed = current_seed;
            res.last_step = current_step;
            res.data_position = resp.data_position;
            res.updates_count = last_updates_count;
            res.sum_loss = sum_loss;
            res.result_size = packed_size;
            
            last_updates_count = 0; // Reset so we don't double count if we process multiple chunks
            
            uint8_t res_buf[44];
            egg_serialize_result_header(res_buf, &res);
            
            uint8_t packet_header[EGG_HEADER_SIZE];
            egg_write_header(packet_header, OP_RESULT, 44 + res.result_size);
            
            send(sock, packet_header, EGG_HEADER_SIZE, 0);
            send(sock, res_buf, 44, 0);
            send(sock, packed_fit.data(), res.result_size, 0);
            
            // std::cout << "Computed Chunk " << chunk_idx << " for Step " << current_step << std::endl;
        }
    }
    
    return 0;
}
