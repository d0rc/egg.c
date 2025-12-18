#!/bin/bash

# Detect available NVIDIA GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No NVIDIA GPUs detected!"
    exit 1
fi

echo "Detected $GPU_COUNT GPU(s), launching workers..."

# Launch a worker for each GPU
for GPU_ID in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    echo "Starting worker on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID ./worker 127.0.0.1 &
done

echo "All workers started."
wait
