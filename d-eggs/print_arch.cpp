#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "include/config.h"
#include "include/model/definitions.h"

// ANSI Colors
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

void print_header() {
    printf(BOLD WHITE "Network Architecture\n" RESET);
    printf("================================================================================\n");
    printf(BOLD "%-25s %-20s %-15s %-10s\n" RESET, "Component", "Dimensions", "Params", "Size (MB)");
    printf("--------------------------------------------------------------------------------\n");
}

void print_row(const char* name, std::string dims, size_t params, const char* color) {
    double size_mb = (double)params / (1024.0 * 1024.0); // Assuming 1 byte per param (int8_t)
    printf("%s%-25s" RESET " %-20s %-15zu %.4f\n", color, name, dims.c_str(), params, size_mb);
}

int main() {
    size_t total_params = 0;
    
    printf(BOLD CYAN "Configuration:\n" RESET);
    printf("  Hidden Dim: %d\n", HIDDEN_DIM);
    printf("  Heads:      %d (Head Dim: %d)\n", N_HEADS, HEAD_DIM);
    printf("  Layers:     %d (Cycles: %d, Total: %d)\n", N_LAYERS, NUMBER_OF_CYCLES, N_LAYERS * NUMBER_OF_CYCLES);
    printf("  Vocab Size: %d\n", VOCAB_SIZE);
    printf("  Seq Len:    %d\n", SEQ_LEN);
    printf("  NTT Mode:   %d\n", NTT_MODE);
    printf("\n");

    print_header();

    // Embeddings
    size_t emb_params = (size_t)VOCAB_SIZE * HIDDEN_DIM;
    print_row("Embedding", "[" + std::to_string(VOCAB_SIZE) + ", " + std::to_string(HIDDEN_DIM) + "]", emb_params, GREEN);
    total_params += emb_params;

    size_t emb_bias_params = HIDDEN_DIM;
    print_row("Embedding Bias", "[" + std::to_string(HIDDEN_DIM) + "]", emb_bias_params, GREEN);
    total_params += emb_bias_params;

#if NTT_MODE != 0
    size_t ntt_params = (size_t)VOCAB_SIZE * HIDDEN_DIM;
    print_row("NTT Emb 1", "[" + std::to_string(VOCAB_SIZE) + ", " + std::to_string(HIDDEN_DIM) + "]", ntt_params, GREEN);
    print_row("NTT Emb 2", "[" + std::to_string(VOCAB_SIZE) + ", " + std::to_string(HIDDEN_DIM) + "]", ntt_params, GREEN);
    print_row("NTT Emb 3", "[" + std::to_string(VOCAB_SIZE) + ", " + std::to_string(HIDDEN_DIM) + "]", ntt_params, GREEN);
    total_params += ntt_params * 3;
#endif

    // Initial MLP
    size_t ln_init_params = HIDDEN_DIM;
    print_row("LN Init", "[" + std::to_string(HIDDEN_DIM) + "]", ln_init_params, YELLOW);
    total_params += ln_init_params;
    
    size_t ln_init_bias_params = HIDDEN_DIM;
    print_row("LN Init Bias", "[" + std::to_string(HIDDEN_DIM) + "]", ln_init_bias_params, YELLOW);
    total_params += ln_init_bias_params;

    size_t w_emb_mlp_up_params = (size_t)HIDDEN_DIM * (HIDDEN_DIM * 4);
    print_row("Emb MLP Up", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM * 4) + "]", w_emb_mlp_up_params, YELLOW);
    total_params += w_emb_mlp_up_params;

    size_t mlp_emb_bias_up_params = HIDDEN_DIM * 4;
    print_row("Emb MLP Bias Up", "[" + std::to_string(HIDDEN_DIM * 4) + "]", mlp_emb_bias_up_params, YELLOW);
    total_params += mlp_emb_bias_up_params;

    size_t w_emb_mlp_down_params = (size_t)(HIDDEN_DIM * 4) * HIDDEN_DIM;
    print_row("Emb MLP Down", "[" + std::to_string(HIDDEN_DIM * 4) + ", " + std::to_string(HIDDEN_DIM) + "]", w_emb_mlp_down_params, YELLOW);
    total_params += w_emb_mlp_down_params;

    size_t mlp_emb_bias_down_params = HIDDEN_DIM;
    print_row("Emb MLP Bias Down", "[" + std::to_string(HIDDEN_DIM) + "]", mlp_emb_bias_down_params, YELLOW);
    total_params += mlp_emb_bias_down_params;

    // Layers
    printf("--------------------------------------------------------------------------------\n");
    printf(BOLD BLUE "Transformer Layers (x%d, Cycles: %d)\n" RESET, N_LAYERS, NUMBER_OF_CYCLES);
    
    size_t layer_params = 0;
    
    // Attention
    size_t ln1_params = HIDDEN_DIM;
    print_row("  LN 1", "[" + std::to_string(HIDDEN_DIM) + "]", ln1_params, BLUE);
    layer_params += ln1_params;
    
    size_t ln1_bias_params = HIDDEN_DIM;
    print_row("  LN 1 Bias", "[" + std::to_string(HIDDEN_DIM) + "]", ln1_bias_params, BLUE);
    layer_params += ln1_bias_params;

    size_t w_q_params = (size_t)HIDDEN_DIM * HIDDEN_DIM;
    print_row("  W_Q", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM) + "]", w_q_params, BLUE);
    layer_params += w_q_params;

    size_t w_k_params = (size_t)HIDDEN_DIM * HIDDEN_DIM;
    print_row("  W_K", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM) + "]", w_k_params, BLUE);
    layer_params += w_k_params;

    size_t w_v_params = (size_t)HIDDEN_DIM * HIDDEN_DIM;
    print_row("  W_V", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM) + "]", w_v_params, BLUE);
    layer_params += w_v_params;

    size_t w_o_params = (size_t)HIDDEN_DIM * HIDDEN_DIM;
    print_row("  W_O", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM) + "]", w_o_params, BLUE);
    layer_params += w_o_params;

    // MLP
    size_t ln2_params = HIDDEN_DIM;
    print_row("  LN 2", "[" + std::to_string(HIDDEN_DIM) + "]", ln2_params, MAGENTA);
    layer_params += ln2_params;

    size_t ln2_bias_params = HIDDEN_DIM;
    print_row("  LN 2 Bias", "[" + std::to_string(HIDDEN_DIM) + "]", ln2_bias_params, MAGENTA);
    layer_params += ln2_bias_params;

    size_t w_up_params = (size_t)HIDDEN_DIM * (HIDDEN_DIM * 4);
    print_row("  W_Up", "[" + std::to_string(HIDDEN_DIM) + ", " + std::to_string(HIDDEN_DIM * 4) + "]", w_up_params, MAGENTA);
    layer_params += w_up_params;

    size_t mlp_bias_up_params = HIDDEN_DIM * 4;
    print_row("  MLP Bias Up", "[" + std::to_string(HIDDEN_DIM * 4) + "]", mlp_bias_up_params, MAGENTA);
    layer_params += mlp_bias_up_params;

    size_t w_down_params = (size_t)(HIDDEN_DIM * 4) * HIDDEN_DIM;
    print_row("  W_Down", "[" + std::to_string(HIDDEN_DIM * 4) + ", " + std::to_string(HIDDEN_DIM) + "]", w_down_params, MAGENTA);
    layer_params += w_down_params;

    size_t mlp_bias_down_params = HIDDEN_DIM;
    print_row("  MLP Bias Down", "[" + std::to_string(HIDDEN_DIM) + "]", mlp_bias_down_params, MAGENTA);
    layer_params += mlp_bias_down_params;

    total_params += layer_params * N_LAYERS;

    // Final LN
    printf("--------------------------------------------------------------------------------\n");
    size_t ln_f_params = HIDDEN_DIM;
    print_row("LN Final", "[" + std::to_string(HIDDEN_DIM) + "]", ln_f_params, RED);
    total_params += ln_f_params;

    size_t ln_f_bias_params = HIDDEN_DIM;
    print_row("LN Final Bias", "[" + std::to_string(HIDDEN_DIM) + "]", ln_f_bias_params, RED);
    total_params += ln_f_bias_params;

    printf("================================================================================\n");
    printf(BOLD "Total Parameters: %zu\n" RESET, total_params);
    printf(BOLD "Total Size:       %.2f MB\n" RESET, (double)total_params / (1024.0 * 1024.0));
    
    return 0;
}
