# EGGROLL in C

A minimalist, dependency-free implementation of the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) algorithm family.

**Mission**: To get the most capable pairs of models / platforms cross-implementations of EGGROLL family of algorithms as possible and squeeze every possible bit of performance from the equipment, implementing everything required from the scratch in hardware-optimized fashion.

This project demonstrates **integer-only training** of a language model, completely bypassing the need for standard floating-point arithmetic or heavy ML frameworks like PyTorch or JAX.

<a id="key-features"></a>
## Key Features

*   **Pure C / Bare Metal**: Old-school fashioned and goal-oriented. Zero external dependencies on the CPU side, keeping it close to the metal.
*   **Apple Silicon Optimized**: Vectorized operations using ARM NEON intrinsics and parallelized via Grand Central Dispatch (GCD).
*   **NVIDIA CUDA Optimized**: Custom GPU kernels utilizing Warp-level primitives, Shared Memory, and CUB/Thrust for maximum throughput.
*   **Integer Only**: Operates primarily on `int8` weights/activations with `int32` (CPU) or `int64` (GPU) accumulation—sticking to integer math as long as it yields the best performance for the hardware.
*   **Gradient Free**: Uses Evolution Strategies (ES) with low-rank perturbations instead of backpropagation. It's both wisdom and freedom!

<a id="quick-start"></a>
## Quick Start

<a id="prepare-data"></a>
### 1. Prepare Data
Ensure you have a text dataset named `input.txt` in the current directory.

<a id="compile-and-run"></a>
### 2. Compile & Run

#### Apple Silicon / CPU
```bash
clang -O3 full_trained_egg.c -o egg
./egg
```

#### NVIDIA GPU (CUDA)
```bash
nvcc -O3 full_cuda_train_egg.cu -o egg_cuda
./egg_cuda
```

![Training Output](_imgs_/egg_train.jpeg)

<a id="advanced-implementations"></a>
## Advanced Implementations

<a id="int8nativeformer"></a>
### Int8NativeFormer (`full_cuda_train_transformer_adam_mgpu.cu`)

An `int8` model.

*   **Native `int8` Architecture**: Operates on raw bytes with a compact `N`-layer, `H`-dim topology.
*   **Quantized Sigmoid Self-Attention**: An `int32/int64` accumulation scheme and quantized weighting.
*   **Auto-Norm & Entropy Monitoring**: Adaptive normalization layers.
*   **EGG DEBUG**: debug-printing "tool" to monitor entropy flow through the network and weights distribution and saturation.
*   **Information-Regulated Optimizer**: A hybrid **ES-AdamW** approach where the optimizer (`float32`) regulates the amount of updates applied to the integer weights, ensuring stable learning.
*   **Performance**: Achieves **~300k tokens/second** with a population of 40,000+ (8192×5) on a single 4090 GPU setup, reaching loss rates (~1.45 bits/byte).

<a id="multi-gpu-strategy"></a>

#### Multi-GPU Strategy
The system employs a **Synchronous Replicated Model** with **Sharded Evaluation**:
*   **Sharded Evaluation**: The population is split across GPUs, with each evaluating a subset of perturbations in parallel.
*   **Implicit Synchronization**: Instead of exchanging gradients (All-Reduce), GPUs receive only **fitness scores**. Since noise is deterministic, each GPU independently reconstructs the update, keeping replicas synchronized with negligible bandwidth.

![image-20251206052311833](_imgs_/trained-transformer-debugger.png)

#### Compile & Run (Multi-GPU)
```bash
nvcc -O3 -arch=native full_cuda_train_transformer_adam_mgpu.cu -o egg_transformer_mgpu
./egg_transformer_mgpu
```

<a id="debugging"></a>
### Debugging (`egg_debug_printer.h`)

A lightweight header-only tool for monitoring integer model stability and detecting saturation or mode collapse.

*   **Metrics**: Tracks Mean, StdDev, bit-level Entropy (0.00-8.00), and Saturation percentages per layer.
*   **Usage**: Define `EGG_DEBUG` during compilation to enable ANSI-colored logs for activations and attention scores.

<a id="configuration"></a>
## Configuration

![Configuration](_imgs_/egg_config.jpeg)

<a id="community"></a>
## Community & Contributing

We are friendly and welcome all sorts of contributions!

*   **Testers**: Open issues with a description of your available compute, join existing issues if you can platforms described there.
*   **Moderators**: To keep all this under control.
*   **Creatives**: Even if you have nice creative IDEA on README design - you're welcome.

<a id="references"></a>
## References

*   **Original JAX Implementation**: [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg)
*   **Original Paper & Project**: [EGGROLL Website](https://eshyperscale.github.io/)
