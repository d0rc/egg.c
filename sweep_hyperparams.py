import itertools
import subprocess
import time
import sys
import os
import random

# --- CONFIGURATION ---
# Define your search space here.
# Keys must match the macro names in config.h (or LEARNING_RATE/MAX_STEPS).
SEARCH_SPACE = {
    "LEARNING_RATE": [0.1, 0.5],
    "NUMBER_OF_CYCLES": [1, 2, 4],
    "SOFTMAX_EXP_SCALE": [8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    "SOFTMAX_LUT_SIZE": [512, 1024, 2048, 4096, 8192],
    "SOFTMAX_SCALE_BIT": [16, 18, 20, 22],
    "SIGMA_SHIFT": [3, 4, 5],
    "SIGMA_SHIFT_VECTOR": [3, 4, 5],
    "ADAM_BETA1": [0.95],
    "ADAM_BETA2": [0.95, 0.99],
    "ADAM_EPS": [1e-8],
    "ADAM_WEIGHT_DECAY": [0.2, 0.3, 0.4],
    "ROPE_SCALE_BIT": [14, 16, 20, 24],
    "NTT_MODE": [0],
    "CHUNK_MEAN_FILTER": [1],
    "CHUNK_MEAN_EXPONENT": [2.0],
    "USE_ADAPTIVE_THRESHOLD": [1],
}

# Fixed parameters for all runs
FIXED_PARAMS = {
    "MAX_STEPS": 1000,
}

# Path to d-eggs directory
BASE_DIR = "d-eggs"

# Experiments that have already been run and should be skipped
COMPLETED_EXPERIMENTS = set()
COMPLETED_LOG_FILE = "completed_experiments.txt"

# Load completed experiments from file
if os.path.exists(COMPLETED_LOG_FILE):
    try:
        with open(COMPLETED_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    COMPLETED_EXPERIMENTS.add(line)
        print(f"Loaded {len(COMPLETED_EXPERIMENTS)} completed experiments from {COMPLETED_LOG_FILE}")
    except Exception as e:
        print(f"Error loading completed experiments: {e}")

def run_command(cmd, cwd=None):
    print(f"Executing: {cmd}")
    ret = subprocess.call(cmd, shell=True, cwd=cwd)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def sanitize_value(val):
    """Convert Python values to C macro compatible strings."""
    if isinstance(val, bool):
        return "1" if val else "0"
    return str(val)

def main():
    # Parse arguments for distributed sweeping
    node_id = 1
    total_nodes = 1
    if len(sys.argv) >= 3:
        try:
            node_id = int(sys.argv[1])
            total_nodes = int(sys.argv[2])
            print(f"Running as Node {node_id} of {total_nodes}")
        except ValueError:
            print("Usage: python3 sweep_hyperparams.py [node_id] [total_nodes]")
            sys.exit(1)

    # Generate all combinations
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*values))
    
    # Fix seed for deterministic shuffling across nodes
    random.seed(42)
    random.shuffle(combinations)

    print(f"Found {len(combinations)} configurations to sweep.")
    
    for i, combo in enumerate(combinations):
        # Distributed check
        if i % total_nodes != (node_id - 1):
            continue

        # Build configuration dictionary
        config = dict(zip(keys, combo))
        config.update(FIXED_PARAMS)
        
        # Generate Experiment Name
        # e.g. sweep_LR0.5_SCALE64.0
        exp_name_parts = ["sweep"]
        for k, v in config.items():
            if k == "MAX_STEPS": continue # Don't include max steps in name
            
            # Shorten keys for readability
            short_key = k
            if k == "LEARNING_RATE": short_key = "LR"
            elif k == "SOFTMAX_EXP_SCALE": short_key = "SCALE"
            elif k == "NUMBER_OF_CYCLES": short_key = "CYC"
            elif k == "SOFTMAX_LUT_SIZE": short_key = "LUT"
            elif k == "SOFTMAX_SCALE_BIT": short_key = "SBIT"
            elif k == "SIGMA_SHIFT": short_key = "SIG"
            elif k == "SIGMA_SHIFT_VECTOR": short_key = "SIGV"
            elif k == "ADAM_BETA1": short_key = "B1"
            elif k == "ADAM_BETA2": short_key = "B2"
            elif k == "ADAM_EPS": short_key = "EPS"
            elif k == "ADAM_WEIGHT_DECAY": short_key = "WD"
            elif k == "ROPE_SCALE_BIT": short_key = "ROPE"
            elif k == "NTT_MODE": short_key = "NTT"
            elif k == "CHUNK_MEAN_FILTER": short_key = "CMF"
            elif k == "CHUNK_MEAN_EXPONENT": short_key = "CMFE"
            elif k == "USE_ADAPTIVE_THRESHOLD": short_key = "ATH"
            elif k == "LOSS_HAXXING": short_key = "HAX"
            
            exp_name_parts.append(f"{short_key}{sanitize_value(v)}")
        
        experiment_name = "_".join(exp_name_parts)
        
        if experiment_name in COMPLETED_EXPERIMENTS:
            print(f"Skipping completed experiment: {experiment_name}")
            continue

        print(f"\n=== Starting Experiment {i+1}/{len(combinations)} ===")
        print(f"Experiment Name: {experiment_name}")
        print("Configuration:", config)
        
        # Construct EXTRA_FLAGS
        flags = []
        flags.append(f'-DEXPERIMENT_NAME=\\"{experiment_name}\\"')
        for k, v in config.items():
            flags.append(f"-D{k}={sanitize_value(v)}")
        
        extra_flags_str = " ".join(flags)
        
        # Compile
        print("Compiling...")
        run_command("make clean", cwd=BASE_DIR)
        # We pass EXTRA_FLAGS to make. Note: We need to escape quotes if needed, 
        # but here we just pass the string.
        # Using 'make coordinator worker' to compile both.
        run_command(f"make coordinator worker EXTRA_FLAGS='{extra_flags_str}'", cwd=BASE_DIR)
        
        # Distribute Workers
        print("Distributing Workers...")
        run_command("./distribute_workers.sh")

        # Run Coordinator
        print("Running Coordinator...")
        # Launch coordinator in background
        coord_proc = subprocess.Popen("./coordinator", shell=True, cwd=BASE_DIR)
        
        # Wait 10 seconds
        print("Waiting 10 seconds before checking coordinator status...")
        time.sleep(10)
        
        # Check if still running
        if coord_proc.poll() is None:
            print("Coordinator is still running. Launching ready.sh...")
            run_command("./ready.sh")
            
            # Wait for coordinator to finish
            print("Waiting for coordinator to finish...")
            coord_proc.wait()
        else:
            print("Coordinator exited prematurely with code:", coord_proc.returncode)
        
        print(f"Experiment {experiment_name} completed.")
        
        # Save to completed log
        try:
            with open(COMPLETED_LOG_FILE, "a") as f:
                f.write(experiment_name + "\n")
            COMPLETED_EXPERIMENTS.add(experiment_name)
        except Exception as e:
            print(f"Error saving completed experiment: {e}")

        # Optional: Sleep briefly between runs
        time.sleep(2)

if __name__ == "__main__":
    main()
