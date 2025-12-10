#!/bin/bash
#
# Hyperparameter Sweep Script for EGG Transformer
# Runs 500 steps per configuration with fresh model each time
#

set -e  # Exit on error

# Configuration
MAX_STEPS=500
SOURCE_FILE="full_cuda_train_transformer_adam_mgpu.cu"
OUTPUT_DIR="sweep_results_$(date +%Y%m%d_%H%M%S)"
BINARY_NAME="train_sweep"

# CUDA compilation flags - adjust arch for your GPU
NVCC_BASE_FLAGS="-O3 -std=c++17 -lcublas"
# Detect GPU architecture if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [ -n "$GPU_ARCH" ]; then
        NVCC_BASE_FLAGS="$NVCC_BASE_FLAGS -arch=sm_$GPU_ARCH"
        echo "Detected GPU arch: sm_$GPU_ARCH"
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
TOTAL_RUNS=0
CURRENT_RUN=0
FAILED_RUNS=0
PASSED_RUNS=0

# Time tracking
SWEEP_START_TIME=0
declare -a RUN_DURATIONS

# Results array
declare -a RESULTS

# Trap Ctrl+C
trap ctrl_c INT
function ctrl_c() {
    echo -e "\n${YELLOW}[INTERRUPTED] Sweep stopped by user${NC}"
    generate_summary
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Helper function to format seconds as human-readable time
format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    elif [ $seconds -lt 3600 ]; then
        local mins=$((seconds / 60))
        local secs=$((seconds % 60))
        echo "${mins}m ${secs}s"
    else
        local hours=$((seconds / 3600))
        local mins=$(( (seconds % 3600) / 60 ))
        echo "${hours}h ${mins}m"
    fi
}

# Function to calculate and display progress
show_progress() {
    local remaining=$((TOTAL_RUNS - CURRENT_RUN))
    local percent=0
    if [ $TOTAL_RUNS -gt 0 ]; then
        percent=$((CURRENT_RUN * 100 / TOTAL_RUNS))
    fi
    
    # Calculate average time per run
    local avg_time=0
    local eta_seconds=0
    if [ ${#RUN_DURATIONS[@]} -gt 0 ]; then
        local total_duration=0
        for d in "${RUN_DURATIONS[@]}"; do
            total_duration=$((total_duration + d))
        done
        avg_time=$((total_duration / ${#RUN_DURATIONS[@]}))
        eta_seconds=$((avg_time * remaining))
    fi
    
    # Calculate elapsed time
    local now=$(date +%s)
    local elapsed=$((now - SWEEP_START_TIME))
    
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo -e " Progress: ${GREEN}$PASSED_RUNS passed${NC} | ${RED}$FAILED_RUNS failed${NC} | ${YELLOW}$remaining remaining${NC} (${percent}% complete)"
    if [ $avg_time -gt 0 ]; then
        echo -e " Avg time/run: $(format_time $avg_time) | Elapsed: $(format_time $elapsed) | ETA: ~$(format_time $eta_seconds)"
    else
        echo -e " Elapsed: $(format_time $elapsed) | ETA: calculating..."
    fi
    echo "══════════════════════════════════════════════════════════════"
}

# Function to run a single experiment
run_experiment() {
    local name=$1
    shift
    local defines="$@"
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    # Show progress header
    show_progress
    
    echo -e "\n${CYAN}▶ [$CURRENT_RUN/$TOTAL_RUNS] Running: $name${NC}"
    echo "  Defines: $defines"
    
    # Clean previous model to start fresh
    rm -rf models/
    mkdir -p models/
    
    # Compile with specific defines
    echo "Compiling..."
    if ! nvcc $NVCC_BASE_FLAGS -DMAX_STEPS=$MAX_STEPS $defines \
         "$SOURCE_FILE" -o "$BINARY_NAME" 2>"$OUTPUT_DIR/${name}_compile.log"; then
        echo -e "${RED}[FAILED] Compilation failed for $name${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        RESULTS+=("$name,COMPILE_FAILED,N/A")
        return 0  # Return 0 to not trigger set -e, we track failures ourselves
    fi
    
    # Run and capture output
    echo "Training for $MAX_STEPS steps..."
    local start_time=$(date +%s)
    
    if ! ./"$BINARY_NAME" 2>&1 | tee "$OUTPUT_DIR/${name}.log"; then
        echo -e "${RED}[FAILED] Training failed for $name${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        RESULTS+=("$name,TRAIN_FAILED,N/A")
        return 0  # Return 0 to not trigger set -e
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Copy training log if it exists
    if [ -f "models/training_log.csv" ]; then
        cp models/training_log.csv "$OUTPUT_DIR/${name}_training.csv"
    fi
    
    # Extract final loss from log
    local final_loss=$(grep "Step $((MAX_STEPS-1))\|Step $MAX_STEPS" "$OUTPUT_DIR/${name}.log" | tail -1 | grep -oP 'Loss: \K[0-9.]+' || echo "N/A")
    
    echo -e "${GREEN}✓ [DONE] $name completed in ${duration}s, Final Loss: $final_loss${NC}"
    RESULTS+=("$name,SUCCESS,$final_loss")
    
    # Track duration for ETA calculation
    RUN_DURATIONS+=($duration)
    PASSED_RUNS=$((PASSED_RUNS + 1))
    
    return 0
}

# Function to generate summary
generate_summary() {
    local now=$(date +%s)
    local total_elapsed=$((now - SWEEP_START_TIME))
    local success_rate=0
    if [ $CURRENT_RUN -gt 0 ]; then
        success_rate=$((PASSED_RUNS * 100 / CURRENT_RUN))
    fi
    
    # Calculate average time per run
    local avg_time=0
    if [ ${#RUN_DURATIONS[@]} -gt 0 ]; then
        local total_duration=0
        for d in "${RUN_DURATIONS[@]}"; do
            total_duration=$((total_duration + d))
        done
        avg_time=$((total_duration / ${#RUN_DURATIONS[@]}))
    fi
    
    echo -e "\n${CYAN}=== Generating Summary ===${NC}"
    
    local summary_file="$OUTPUT_DIR/sweep_summary.csv"
    echo "experiment,status,final_loss" > "$summary_file"
    
    for result in "${RESULTS[@]}"; do
        echo "$result" >> "$summary_file"
    done
    
    echo -e "${GREEN}Summary saved to: $summary_file${NC}"
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           SWEEP RESULTS SUMMARY                               ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║ Completed: $CURRENT_RUN / $TOTAL_RUNS experiments"
    echo "║ Passed: $PASSED_RUNS | Failed: $FAILED_RUNS | Success rate: ${success_rate}%"
    echo "║ Total time: $(format_time $total_elapsed) | Avg per run: $(format_time $avg_time)"
    echo "║ Finished at: $(date)"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║ TOP 10 BY LOSS (lower is better):                             ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    tail -n +2 "$summary_file" | grep "SUCCESS" | sort -t',' -k3 -n | head -10 | while read line; do
        echo "║  $line"
    done
    echo "╚═══════════════════════════════════════════════════════════════╝"
}

# Count total experiments first
count_experiments() {
    TOTAL_RUNS=0
    
    # Baseline: 1
    # RoPE scaling: 5
    # Adam beta1: 2, Adam beta2: 2
    # Weight decay: 2
    # Muon toggle: 1, Muon momentum: 2
    # Sigma shift: 2, Sigma shift vector: 2
    # Device mask: 2
    # Softmax exp scale: 3
    # Shift attention: 2
    # Gaussian noise: 6
    # Total: 1+5+2+2+2+1+2+2+2+2+3+2+6 = 32
    
    TOTAL_RUNS=32
    
    echo "Total experiments to run: $TOTAL_RUNS"
}

# Main sweep execution
main() {
    # Initialize start time
    SWEEP_START_TIME=$(date +%s)
    
    echo "========================================="
    echo "   EGG Transformer Hyperparameter Sweep"
    echo "========================================="
    echo "Source: $SOURCE_FILE"
    echo "Steps per run: $MAX_STEPS"
    echo "Started at: $(date)"
    echo ""
    
    count_experiments
    echo ""
    
    # ============================================
    # BASELINE
    # ============================================
    run_experiment "baseline" ""
    
    # ============================================
    # RoPE SCALING (Priority)
    # ============================================
    run_experiment "rope_scale_16" "-DROPE_SCALE_BIT=16"
    run_experiment "rope_scale_18" "-DROPE_SCALE_BIT=18"
    run_experiment "rope_scale_20" "-DROPE_SCALE_BIT=20"
    run_experiment "rope_scale_24" "-DROPE_SCALE_BIT=24"
    run_experiment "rope_scale_30" "-DROPE_SCALE_BIT=30"
    
    # ============================================
    # ADAM OPTIMIZER
    # ============================================
    # Beta1 variations
    run_experiment "adam_beta1_0.85" "-DADAM_BETA1=0.85f"
    run_experiment "adam_beta1_0.95" "-DADAM_BETA1=0.95f"
    
    # Beta2 variations
    run_experiment "adam_beta2_0.95" "-DADAM_BETA2=0.95f"
    run_experiment "adam_beta2_0.99" "-DADAM_BETA2=0.99f"
    
    # Weight decay variations
    run_experiment "wd_0.001" "-DADAM_WEIGHT_DECAY=0.001f"
    run_experiment "wd_0.1" "-DADAM_WEIGHT_DECAY=0.1f"
    
    # ============================================
    # MUON VS ADAM
    # ============================================
    # Test without Muon (pure Adam)
    run_experiment "no_muon" "-UUSE_MUON"
    
    # Muon momentum variations (only if Muon enabled by default)
    run_experiment "muon_momentum_0.8" "-DMUON_MOMENTUM=0.8f"
    run_experiment "muon_momentum_0.9" "-DMUON_MOMENTUM=0.9f"
    
    # ============================================
    # NOISE PARAMETERS
    # ============================================
    # Sigma shift (controls noise magnitude in matmuls)
    run_experiment "sigma_shift_3" "-DSIGMA_SHIFT=3"
    run_experiment "sigma_shift_5" "-DSIGMA_SHIFT=5"
    
    # Sigma shift vector (controls noise in vectors)
    run_experiment "sigma_shift_vec_2" "-DSIGMA_SHIFT_VECTOR=2"
    run_experiment "sigma_shift_vec_4" "-DSIGMA_SHIFT_VECTOR=4"
    
    # Device mask (noise range: 2^bits - 1)
    run_experiment "device_mask_7" "-DDEVICE_MASK=7"
    run_experiment "device_mask_31" "-DDEVICE_MASK=31"
    
    # ============================================
    # SOFTMAX / ATTENTION SCALING
    # ============================================
    # Softmax exp scale (temperature)
    run_experiment "softmax_scale_6" "-DSOFTMAX_EXP_SCALE=6.0"
    run_experiment "softmax_scale_10" "-DSOFTMAX_EXP_SCALE=10.0"
    run_experiment "softmax_scale_12" "-DSOFTMAX_EXP_SCALE=12.0"
    
    # Attention shift
    run_experiment "shift_attn_6" "-DSHIFT_ATTN=6"
    run_experiment "shift_attn_10" "-DSHIFT_ATTN=10"
    
    # ============================================
    # GAUSSIAN NOISE (Experimental)
    # Note: Gaussian sums 3 noise samples, giving ~3x larger magnitude!
    # May need sigma shift compensation for stability.
    # ============================================
    
    # Device Gaussian only (with extra sigma shift to compensate for 3x magnitude)
    run_experiment "device_gaussian" "-DDEVICE_GAUSSIAN=true"
    run_experiment "device_gaussian_sigma6" "-DDEVICE_GAUSSIAN=true -DSIGMA_SHIFT=6"
    
    # Host Gaussian only
    run_experiment "host_gaussian" "-DHOST_GAUSSIAN=true"
    run_experiment "host_gaussian_sigma6" "-DHOST_GAUSSIAN=true -DSIGMA_SHIFT=6"
    
    # Both Gaussian (likely unstable without compensation)
    run_experiment "both_gaussian" "-DHOST_GAUSSIAN=true -DDEVICE_GAUSSIAN=true"
    run_experiment "both_gaussian_sigma6" "-DHOST_GAUSSIAN=true -DDEVICE_GAUSSIAN=true -DSIGMA_SHIFT=6"
    
    # ============================================
    # SUMMARY
    # ============================================
    generate_summary
    
    # Cleanup
    rm -f "$BINARY_NAME"
    
    echo -e "\n${GREEN}Sweep complete!${NC}"
    echo "Results directory: $OUTPUT_DIR"
}

# Run main
main "$@"
