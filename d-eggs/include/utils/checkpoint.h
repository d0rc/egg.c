#ifndef EGG_UTILS_CHECKPOINT_H
#define EGG_UTILS_CHECKPOINT_H

#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cstring>
#include <cinttypes>

// Magic: EGGC (Egg Checkpoint)
#define CHECKPOINT_MAGIC 0x45474743
#define CHECKPOINT_VERSION 1

struct CheckpointHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t step;
    uint64_t seed;
    uint64_t data_size;
};

static inline void ensure_directory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0755);
    }
}

static inline void save_checkpoint_worker(std::string dir, uint64_t step, uint64_t seed, std::vector<uint8_t> data) {
    ensure_directory(dir);
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/checkpoint_%08" PRIu64 ".bin", dir.c_str(), step);
    
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open checkpoint file for writing: " << filename << std::endl;
        return;
    }
    
    CheckpointHeader header;
    header.magic = CHECKPOINT_MAGIC;
    header.version = CHECKPOINT_VERSION;
    header.step = step;
    header.seed = seed;
    header.data_size = data.size();
    
    outfile.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!data.empty()) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    
    outfile.close();
    // std::cout << "Saved checkpoint: " << filename << std::endl;
}

static inline void trigger_save_checkpoint(const std::string& dir, uint64_t step, uint64_t seed, const std::vector<uint8_t>& data) {
    if (dir.empty()) return;
    // Launch detached thread with copied data
    std::thread(save_checkpoint_worker, dir, step, seed, data).detach();
}

static inline bool load_checkpoint(const std::string& dir, uint64_t& step, uint64_t& seed, std::vector<uint8_t>& data) {
    if (dir.empty()) return false;
    
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    
    struct dirent* ent;
    uint64_t max_step = 0;
    bool found = false;
    std::string best_file;
    
    while ((ent = readdir(d)) != NULL) {
        std::string name = ent->d_name;
        if (name.rfind("checkpoint_", 0) == 0 && name.find(".bin") != std::string::npos) {
            // Extract step
            try {
                size_t start = 11; // len("checkpoint_")
                size_t end = name.find(".bin");
                std::string num_str = name.substr(start, end - start);
                uint64_t s = std::stoull(num_str);
                
                if (!found || s > max_step) {
                    max_step = s;
                    best_file = name;
                    found = true;
                }
            } catch (...) {
                continue;
            }
        }
    }
    closedir(d);
    
    if (!found) return false;
    
    std::string path = dir + "/" + best_file;
    std::ifstream infile(path, std::ios::binary);
    if (!infile.is_open()) return false;
    
    CheckpointHeader header;
    infile.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != CHECKPOINT_MAGIC) {
        std::cerr << "Invalid checkpoint magic in " << path << std::endl;
        return false;
    }
    
    step = header.step;
    seed = header.seed;
    
    if (header.data_size > 0) {
        data.resize(header.data_size);
        infile.read(reinterpret_cast<char*>(data.data()), header.data_size);
    } else {
        data.clear();
    }
    
    std::cout << "Loaded checkpoint from " << path << " (Step " << step << ")" << std::endl;
    return true;
}

#endif // EGG_UTILS_CHECKPOINT_H
