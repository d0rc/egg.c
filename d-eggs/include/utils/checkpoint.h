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
#include <map>
#include <utility>

#include "utils/training.h"

// Magic: EGGC (Egg Checkpoint)
#define CHECKPOINT_MAGIC 0x45474743
#define CHECKPOINT_VERSION 2

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

static inline void save_checkpoint_worker(std::string dir, uint64_t step, uint64_t seed, std::vector<uint8_t> data, ReduceLROnPlateau scheduler) {
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
    
    // Write Scheduler State (V2)
    std::vector<uint8_t> sched_buf;
    serialize_scheduler(sched_buf, scheduler);
    outfile.write(reinterpret_cast<const char*>(sched_buf.data()), sched_buf.size());
    
    if (!data.empty()) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    
    outfile.close();
    // std::cout << "Saved checkpoint: " << filename << std::endl;
}

static inline void trigger_save_checkpoint(const std::string& dir, uint64_t step, uint64_t seed, const std::vector<uint8_t>& data, const ReduceLROnPlateau& scheduler) {
    if (dir.empty()) return;
    // Launch detached thread with copied data and scheduler
    std::thread(save_checkpoint_worker, dir, step, seed, data, scheduler).detach();
}

// Helper to load a single checkpoint file
static inline bool load_checkpoint_file(const std::string& path, uint64_t& step, uint64_t& seed, std::vector<uint8_t>& data, ReduceLROnPlateau* scheduler = nullptr) {
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
    
    // Read Scheduler State if V2
    if (header.version >= 2 && scheduler) {
        std::vector<uint8_t> sched_buf(sizeof(ReduceLROnPlateau));
        infile.read(reinterpret_cast<char*>(sched_buf.data()), sched_buf.size());
        deserialize_scheduler(sched_buf.data(), *scheduler);
    } else if (header.version >= 2) {
        // Skip scheduler data if not requested
        infile.seekg(sizeof(ReduceLROnPlateau), std::ios::cur);
    }
    
    if (header.data_size > 0) {
        data.resize(header.data_size);
        infile.read(reinterpret_cast<char*>(data.data()), header.data_size);
    } else {
        data.clear();
    }
    return true;
}

static inline bool load_checkpoint(const std::string& dir, uint64_t& step, uint64_t& seed, std::vector<uint8_t>& data, ReduceLROnPlateau& scheduler, std::map<uint64_t, std::pair<std::vector<uint8_t>, float>>* history = nullptr, size_t max_history = 0) {
    if (dir.empty()) return false;
    
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    
    struct dirent* ent;
    std::vector<std::pair<uint64_t, std::string>> checkpoints;
    
    while ((ent = readdir(d)) != NULL) {
        std::string name = ent->d_name;
        if (name.rfind("checkpoint_", 0) == 0 && name.find(".bin") != std::string::npos) {
            try {
                size_t start = 11; // len("checkpoint_")
                size_t end = name.find(".bin");
                std::string num_str = name.substr(start, end - start);
                uint64_t s = std::stoull(num_str);
                checkpoints.push_back({s, name});
            } catch (...) {
                continue;
            }
        }
    }
    closedir(d);
    
    if (checkpoints.empty()) return false;
    
    // Sort by step
    std::sort(checkpoints.begin(), checkpoints.end());
    
    // Load latest
    const auto& latest = checkpoints.back();
    std::string path = dir + "/" + latest.second;
    
    if (!load_checkpoint_file(path, step, seed, data, &scheduler)) return false;
    
    std::cout << "Loaded checkpoint from " << path << " (Step " << step << ")" << std::endl;
    
    // Load history if requested
    if (history) {
        history->clear();
        
        // Determine range: [start_idx, end_idx)
        size_t count = checkpoints.size();
        size_t start_idx = 0;
        if (max_history > 0 && count > max_history) {
            start_idx = count - max_history;
        }
        
        for (size_t i = start_idx; i < count; i++) {
            const auto& cp = checkpoints[i];
            
            if (cp.first == step) {
                // It's the latest one. We have data and scheduler.
                if (step > 0) {
                    history->insert({step - 1, {data, (float)scheduler.current_lr}});
                }
            } else {
                // Load from disk
                uint64_t s, sd;
                std::vector<uint8_t> d;
                ReduceLROnPlateau temp_sched;
                // Init temp_sched with default just in case
                temp_sched = init_scheduler_default(0.0);
                
                std::string p = dir + "/" + cp.second;
                if (load_checkpoint_file(p, s, sd, d, &temp_sched)) {
                    if (s > 0) {
                        history->insert({s - 1, {d, (float)temp_sched.current_lr}});
                    }
                }
            }
        }
        std::cout << "Loaded " << history->size() << " history entries." << std::endl;
    }
    
    return true;
}

#endif // EGG_UTILS_CHECKPOINT_H
