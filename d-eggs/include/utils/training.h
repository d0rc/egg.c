#ifndef EGG_TRAINING_H
#define EGG_TRAINING_H

#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

// Structure to hold scheduler state
typedef struct {
    double base_lr;          // Initial learning rate
    double current_lr;       // Current learning rate
    double factor;           // Factor to multiply LR by when reducing (e.g., 0.1)
    int patience;            // Number of epochs to wait for improvement
    int patience_counter;    // Counter for patience
    double min_lr;           // Minimum learning rate
    double best_loss;        // Best loss observed so far
    double threshold;        // Minimum change to qualify as improvement
    int cooldown;            // Cooldown counter after LR reduction
    int cooldown_counter;    // Current cooldown counter
    bool verbose;            // Print messages when LR changes
    int mode;                // 0: 'min' (loss should decrease), 1: 'max' (metric should increase)
    int threshold_mode;      // 0: 'rel' (relative), 1: 'abs' (absolute)
    bool initialized;        // Whether best_loss has been initialized
} ReduceLROnPlateau;

// Initialize the scheduler
inline ReduceLROnPlateau init_scheduler(double base_lr, double factor, int patience, 
                                 double min_lr, double threshold, int cooldown,
                                 bool verbose, int mode, int threshold_mode) {
    ReduceLROnPlateau scheduler;
    scheduler.base_lr = base_lr;
    scheduler.current_lr = base_lr;
    scheduler.factor = factor;
    scheduler.patience = patience;
    scheduler.patience_counter = 0;
    scheduler.min_lr = min_lr;
    scheduler.best_loss = (mode == 0) ? DBL_MAX : -DBL_MAX;
    scheduler.threshold = threshold;
    scheduler.cooldown = cooldown;
    scheduler.cooldown_counter = 0;
    scheduler.verbose = verbose;
    scheduler.mode = mode;
    scheduler.threshold_mode = threshold_mode;
    scheduler.initialized = false;
    return scheduler;
}

// Default initialization with typical values
inline ReduceLROnPlateau init_scheduler_default(double base_lr) {
    return init_scheduler(
        base_lr,     // base_lr
        0.90,         // factor (aggressive decay)
        30,         // patience (steps)
        5e-3,        // min_lr
        1e-2,        // threshold
        100,          // cooldown
        true,        // verbose
        0,           // mode: 'min'
        0            // threshold_mode: 'rel'
    );
}

// Check if new value represents improvement over best value
inline bool is_better(double new_val, double best_val, int mode, int threshold_mode, double threshold) {
    if (mode == 0) { // 'min' mode - loss should decrease
        if (threshold_mode == 0) { // 'rel' mode
            return new_val < best_val * (1.0 - threshold);
        } else { // 'abs' mode
            return new_val < best_val - threshold;
        }
    } else { // 'max' mode - metric should increase
        if (threshold_mode == 0) { // 'rel' mode
            return new_val > best_val * (1.0 + threshold);
        } else { // 'abs' mode
            return new_val > best_val + threshold;
        }
    }
}

// Main function to get learning rate
inline double get_learning_rate_adaptive(ReduceLROnPlateau* scheduler, long step, double loss_or_metric) {
    // On first call, initialize best loss/metric
    if (!scheduler->initialized) {
        scheduler->best_loss = loss_or_metric;
        scheduler->initialized = true;
        return scheduler->current_lr;
    }
    
    // Check if we're in cooldown
    if (scheduler->cooldown_counter > 0) {
        scheduler->cooldown_counter--;
        return scheduler->current_lr;
    }
    
    // Check if current value is better than best
    if (is_better(loss_or_metric, scheduler->best_loss, 
                  scheduler->mode, scheduler->threshold_mode, scheduler->threshold)) {
        scheduler->best_loss = loss_or_metric;
        scheduler->patience_counter = 0;
    } else {
        scheduler->patience_counter++;
        
        // Check if we've exceeded patience
        if (scheduler->patience_counter >= scheduler->patience) {
            // Reduce learning rate
            double new_lr = scheduler->current_lr * scheduler->factor;
            
            // Ensure we don't go below min_lr
            if (new_lr < scheduler->min_lr) {
                new_lr = scheduler->min_lr;
            }
            
            // Only reduce if we're actually changing the LR
            if (new_lr < scheduler->current_lr) {
                if (scheduler->verbose) {
                    printf("Step %ld: Reducing learning rate from %.6f to %.6f\n", 
                           step, scheduler->current_lr, new_lr);
                }
                scheduler->current_lr = new_lr;
                scheduler->patience_counter = 0;
                scheduler->cooldown_counter = scheduler->cooldown;
            }
        }
    }
    
    return scheduler->current_lr;
}

// Serialization Helpers
inline void serialize_scheduler(std::vector<uint8_t>& buffer, const ReduceLROnPlateau& scheduler) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&scheduler);
    buffer.insert(buffer.end(), ptr, ptr + sizeof(ReduceLROnPlateau));
}

inline void deserialize_scheduler(const uint8_t* buffer, ReduceLROnPlateau& scheduler) {
    std::memcpy(&scheduler, buffer, sizeof(ReduceLROnPlateau));
}

// Legacy static schedule (kept for reference or fallback)
inline float get_learning_rate(long step) {
    if (step < 100) return 0.5f;
    if (step < 200) return 0.25f;
    return 0.125f;
}

#endif // EGG_TRAINING_H
