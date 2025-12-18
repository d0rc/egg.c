## **Implementation Proposal**

```c
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

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
ReduceLROnPlateau init_scheduler(double base_lr, double factor, int patience, 
                                 double min_lr, double threshold, int cooldown,
                                 bool verbose, int mode, int threshold_mode) {
    ReduceLROnPlateau scheduler = {
        .base_lr = base_lr,
        .current_lr = base_lr,
        .factor = factor,
        .patience = patience,
        .patience_counter = 0,
        .min_lr = min_lr,
        .best_loss = (mode == 0) ? DBL_MAX : -DBL_MAX, // 'min' mode: initialize to max, 'max' mode: initialize to min
        .threshold = threshold,
        .cooldown = cooldown,
        .cooldown_counter = 0,
        .verbose = verbose,
        .mode = mode, // 0: 'min', 1: 'max'
        .threshold_mode = threshold_mode, // 0: 'rel', 1: 'abs'
        .initialized = false
    };
    return scheduler;
}

// Default initialization with typical values
ReduceLROnPlateau init_scheduler_default(double base_lr) {
    return init_scheduler(
        base_lr,     // base_lr
        0.1,         // factor
        10,          // patience
        1e-6,        // min_lr
        1e-4,        // threshold
        0,           // cooldown
        true,        // verbose
        0,           // mode: 'min'
        0            // threshold_mode: 'rel'
    );
}

// Check if new value represents improvement over best value
static bool is_better(double new_val, double best_val, int mode, int threshold_mode, double threshold) {
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
double get_learning_rate(ReduceLROnPlateau* scheduler, int step, double loss_or_metric) {
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
                    printf("Step %d: Reducing learning rate from %.6f to %.6f\n", 
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

// Reset scheduler to initial state (useful for multiple training runs)
void reset_scheduler(ReduceLROnPlateau* scheduler) {
    scheduler->current_lr = scheduler->base_lr;
    scheduler->patience_counter = 0;
    scheduler->best_loss = (scheduler->mode == 0) ? DBL_MAX : -DBL_MAX;
    scheduler->cooldown_counter = 0;
    scheduler->initialized = false;
}

// Get current learning rate without updating
double get_current_lr(ReduceLROnPlateau* scheduler) {
    return scheduler->current_lr;
}
```

## **Usage Example**

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Example training simulation
int main() {
    // Initialize scheduler
    ReduceLROnPlateau scheduler = init_scheduler_default(0.001);
    
    // Simulate training
    srand(time(NULL));
    double loss = 0.5;
    
    for (int epoch = 0; epoch < 100; epoch++) {
        // Simulate training and get validation loss
        // In reality, this would come from your training loop
        
        // Simulate loss that improves slowly then plateaus
        if (epoch < 20) {
            loss -= 0.01 + ((double)rand() / RAND_MAX) * 0.02;
        } else if (epoch < 40) {
            loss -= 0.001 + ((double)rand() / RAND_MAX) * 0.002;
        } else {
            // Plateau - small random fluctuations
            loss += ((double)rand() / RAND_MAX - 0.5) * 0.005;
        }
        
        // Ensure loss doesn't go negative
        if (loss < 0.01) loss = 0.01;
        
        // Get learning rate for this epoch
        double lr = get_learning_rate(&scheduler, epoch, loss);
        
        printf("Epoch %3d: Loss = %.4f, LR = %.6f\n", epoch, loss, lr);
        
        // Use the learning rate in your optimizer
        // optimizer_lr = lr;
    }
    
    // Example with custom parameters
    printf("\n--- Custom Scheduler Example ---\n");
    ReduceLROnPlateau custom_scheduler = init_scheduler(
        0.01,    // base_lr
        0.5,     // factor (reduce by half)
        5,       // patience (5 epochs)
        1e-5,    // min_lr
        0.001,   // threshold
        2,       // cooldown (2 epochs)
        true,    // verbose
        0,       // mode: 'min'
        1        // threshold_mode: 'abs'
    );
    
    // Simulate a few steps
    double test_loss = 0.5;
    for (int i = 0; i < 20; i++) {
        test_loss += ((double)rand() / RAND_MAX - 0.5) * 0.1;
        double lr = get_learning_rate(&custom_scheduler, i, test_loss);
        printf("Step %2d: Loss = %.4f, LR = %.6f\n", i, test_loss, lr);
    }
    
    return 0;
}
```

## **Key Features**

1. **Pure ANSI C**: No dependencies, works with any C compiler
2. **Configurable modes**:
    - `mode = 0`: 'min' (for loss minimization)
    - `mode = 1`: 'max' (for metric maximization)
3. **Threshold modes**:
    - `threshold_mode = 0`: Relative improvement
    - `threshold_mode = 1`: Absolute improvement
4. **Memory efficient**: No dynamic allocations
5. **Thread-safe**: As long as each thread has its own scheduler instance

## **Advanced Usage**

```c
// Monitor accuracy instead of loss (mode = 1 for 'max')
ReduceLROnPlateau acc_scheduler = init_scheduler(
    0.001,  // base_lr
    0.1,    // factor
    5,      // patience
    1e-6,   // min_lr
    0.001,  // threshold
    0,      // cooldown
    true,   // verbose
    1,      // mode: 'max' (for accuracy)
    0       // threshold_mode: 'rel'
);

// Use in training loop
double accuracy = compute_accuracy();
double lr = get_learning_rate(&acc_scheduler, epoch, accuracy);
```