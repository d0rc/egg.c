#ifndef EGG_UTILS_IO_H
#define EGG_UTILS_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../model/definitions.h"

static inline void ensure_models_dir() {
    struct stat st = {0};
    if (stat("models", &st) == -1) {
        mkdir("models", 0700);
    }
}

static inline void update_last_link(const char *target_filename, const char *link_name) {
    char link_path[256];
    sprintf(link_path, "models/%s", link_name);
    unlink(link_path);
    (void)symlink(target_filename, link_path);
}

static inline void save_model_info() {
    FILE *f = fopen("models/model.info", "w");
    if(!f) return;
    
    long params = sizeof(TransformerModel);
    long adam_size = sizeof(AdamModel);
    
    fprintf(f, "Model Configuration:\n");
    fprintf(f, "HIDDEN_DIM: %d\n", HIDDEN_DIM);
    fprintf(f, "HEAD_DIM: %d\n", HEAD_DIM);
    fprintf(f, "N_LAYERS: %d\n", N_LAYERS);
    fprintf(f, "SEQ_LEN: %d\n", SEQ_LEN);
    fprintf(f, "VOCAB_SIZE: %d\n", VOCAB_SIZE);
    fprintf(f, "N_HEADS: %d\n", N_HEADS);
    fprintf(f, "Weight Sharing: Yes\n");
    fprintf(f, "Biases: Embedding(Yes) Output(No)\n");
    fprintf(f, "Model Parameters: %ld (%.2f MB)\n", params, params/(1024.0*1024.0));
    fprintf(f, "Adam State Size: %ld (%.2f MB)\n", adam_size, adam_size/(1024.0*1024.0));
    fprintf(f, "Population: %d\n", POPULATION_SIZE);
    fprintf(f, "Softmax Scale: 2^%d\n", SOFTMAX_SCALE_BIT);
    
    printf("\n--- Model Architecture ---\n");
    printf("Parameters: %ld (%.2f MB)\n", params, params/(1024.0*1024.0));
    printf("Optimizer State: %.2f MB\n", adam_size/(1024.0*1024.0));
    printf("Layers: %d, Hidden: %d, Heads: %d\n", N_LAYERS, HIDDEN_DIM, N_HEADS);
    printf("Sequence Length: %d, Vocab: %d\n", SEQ_LEN, VOCAB_SIZE);
    printf("--------------------------\n");
    
    fclose(f);
}

static inline void save_checkpoint(TransformerModel *model, AdamModel *adam, long step) {
    char name_model[128], name_adam[128];
    sprintf(name_model, "egg_step-%08ld.model.bin", step);
    sprintf(name_adam, "egg_step-%08ld.adam.bin", step);
    
    char path_model[256], path_adam[256];
    sprintf(path_model, "models/%s", name_model);
    sprintf(path_adam, "models/%s", name_adam);
    
    FILE *f = fopen(path_model, "wb");
    if (f) {
        (void)fwrite(model, sizeof(TransformerModel), 1, f);
        fclose(f);
        update_last_link(name_model, "egg_transformer_last.model.bin");
    }
    
    f = fopen(path_adam, "wb");
    if (f) {
        (void)fwrite(adam, sizeof(AdamModel), 1, f);
        fclose(f);
        update_last_link(name_adam, "egg_transformer_last.adam.bin");
    }
}

static inline int load_model(const char *filename, TransformerModel *model) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    printf("Loading model from %s...\n", filename);
    if (fread(model, sizeof(TransformerModel), 1, f) != 1) {
        printf("Error reading model from %s\n", filename);
        fclose(f);
        return 0;
    }
    fclose(f);
    printf("Model loaded successfully.\n");
    return 1;
}

#endif // EGG_UTILS_IO_H
