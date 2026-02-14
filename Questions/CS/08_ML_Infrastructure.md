# ML Infrastructure - Interview Q&A

Deep dive into GPU programming, MLOps, model optimization, and deployment infrastructure.

---

## Table of Contents

- [[#Part 1: GPU Programming & CUDA]]
  - [[#Explain the GPU Memory Hierarchy (Global, Shared, Local, Constant)]]
  - [[#What is a Kernel Launch? How do Grids and Blocks work?]]
  - [[#What is the difference between CPU and GPU execution models?]]
  - [[#What are Tensor Cores and when are they used?]]
- [[#Part 2: Model Optimization]]
  - [[#What is Quantization (INT8 vs FP16 vs FP32)?]]
  - [[#What is Pruning? (Structured vs Unstructured)]]
  - [[#What is Knowledge Distillation?]]
  - [[#How does Mixed Precision Training work?]]
- [[#Part 3: MLOps & Infrastructure]]
  - [[#What is Experiment Tracking (e.g., MLflow)?]]
  - [[#What is a Model Registry?]]
  - [[#What is Data Versioning (DVC) and why is it important?]]
  - [[#How do you handle dependency management (Docker vs Conda)?]]
- [[#Part 4: Serving Frameworks]]
  - [[#TensorRT vs ONNX Runtime vs TorchServe?]]
  - [[#What is Triton Inference Server?]]
  - [[#What is Model Batching and how does it improve throughput?]]

---

## Part 1: GPU Programming & CUDA

### Explain the GPU Memory Hierarchy (Global, Shared, Local, Constant)

*   **Global Memory**: Large (e.g., 24GB+), accessible by all threads, slow latency (hundreds of cycles).
*   **Shared Memory**: Small (e.g., 48KB-164KB per block), accessible by threads in the *same block*, extremely fast (register speed). User-managed cache.
*   **Local Memory**: Private to each thread, stored in global memory (slow!) unless it spills to cache. Used for register spills.
*   **Constant Memory**: Read-only, cached. Good for broadcast constants.
*   **Registers**: Fastest, private to thread. Limited quantity per thread.

**Optimization Tip**: Minimize Global Memory access. Load data into Shared Memory or Registers for computation.

### What is a Kernel Launch? How do Grids and Blocks work?

*   **Kernel**: A function executed on the GPU.
*   **Grid**: The collection of all threads launched for a kernel.
*   **Block**: A group of threads that execute on the same **Streaming Multiprocessor (SM)**. Threads in a block can share memory and synchronize.
*   **Thread**: The fundamental unit of execution.

**Hierarchy**: Grid -> Blocks -> Warps (32 threads) -> Threads.

### What is the difference between CPU and GPU execution models?

*   **CPU**: Latency-oriented. Few powerful cores (ALUs). Complex control logic (branch prediction, out-of-order execution). Good for sequential logic.
*   **GPU**: Throughput-oriented. Thousands of simpler cores. Massive parallelism to hide memory latency. **SIMT** (Single Instruction, Multiple Threads) architecture – all threads in a Warp execute the same instruction.

### What are Tensor Cores and when are they used?

Specialized hardware units on NVIDIA GPUs (Volta+) designed for **Matrix Multiplication** (GEMM).
*   They perform `D = A * B + C` on 4x4 or larger matrices in a single clock cycle.
*   **Speedup**: Up to 8x-16x faster than standard CUDA cores for mixed-precision math (FP16/BF16/TF32).
*   Used heavily in Deep Learning training and inference.

---

## Part 2: Model Optimization

### What is Quantization (INT8 vs FP16 vs FP32)?

Reducing the precision of model weights and activations to save memory and compute.
*   **FP32 (Single Precision)**: Standard training. High memory usage.
*   **FP16 / BF16 (Half Precision)**: Standard on modern GPUs. 2x speedup, 0.5x memory.
*   **INT8**: Inference only. 4x speedup, 0.25x memory. Requires calibration to handle dynamic range.

**Post-Training Quantization (PTQ)** vs **Quantization-Aware Training (QAT)**.

### What is Pruning? (Structured vs Unstructured)

Removing weights (setting them to zero) to sparsify the model.
*   **Unstructured Pruning**: Zero out individual weights. Good compression ratio, but sparse matrices are hard to accelerate on standard GPUs.
*   **Structured Pruning**: Remove entire channels, filters, or layers. Lower compression ratio, but resulting model is structurally smaller and runs faster on standard hardware.

### What is Knowledge Distillation?

Training a small **Student** model to mimic the behavior of a large **Teacher** model.
*   **Mechanism**: Student learns from "Soft Labels" (Teacher's output probabilities) + Hard Labels (Ground truth).
*   **Benefit**: Student is much smaller and faster but retains most of Teacher's accuracy.

### How does Mixed Precision Training work?

Combining FP16 and FP32 during training to get speed of FP16 and stability of FP32.
1.  Keep a **Master Copy** of weights in FP32.
2.  Perform forward/backward pass in **FP16**.
3.  **Loss Scaling**: Multiply loss by a factor (e.g., 2^10) to prevent small gradients from underflowing to zero in FP16.
4.  Update Master Copy (FP32) with gradients.

---

## Part 3: MLOps & Infrastructure

### What is Experiment Tracking (e.g., MLflow)?

System to log parameters (LR, Batch Size), code versions (Git commit), and metrics (Loss, Accuracy) for every training run.
**Why**: Reproducibility. Comparison of runs to find best model.

### What is a Model Registry?

Central repository for managing model lifecycle.
*   Versioning (v1, v2).
*   Stage Transitions (Staging -> Production -> Archived).
*   Metadata (Who trained it? What dataset? Metrics?).

### What is Data Versioning (DVC) and why is it important?

**Data Version Control (DVC)** brings Git-like versioning to large datasets.
*   **Problem**: Git can't handle 100GB files.
*   **Solution**: DVC stores light metadata in Git (pointers), and actual data in S3/GCS.
*   **Benefit**: `git checkout v1.0` restores both code *and* the exact dataset used to train v1.0.

### How do you handle dependency management (Docker vs Conda)?

*   **Conda/Pip**: Manages Python libraries. Good for local dev. **Fragile** across environments (system libraries differ).
*   **Docker**: Encapsulates entire OS, system libraries (CUDA, cuDNN), and Python env. **Guarantees** reproducibility. Standard for production deployment.

---

## Part 4: Serving Frameworks

### TensorRT vs ONNX Runtime vs TorchServe?

*   **TorchServe / TFServing**: Standard, flexible. Good for Python logic. Slower than compiled runtimes.
*   **ONNX Runtime**: Cross-platform standard. Optimizes computation graph. Faster than TorchServe. Run models from any framework (PyTorch -> ONNX).
*   **TensorRT (NVIDIA)**: Maximum performance on NVIDIA GPUs. Compiles model to specific GPU architecture. Performs layer fusion, kernel auto-tuning. Hardest to use, fastest inference.

### What is Triton Inference Server?

NVIDIA's open-source serving software.
*   **Multi-framework**: Runs TensorRT, PyTorch, ONNX, TensorFlow models simultaneously.
*   **Dynamic Batching**: Groups incoming requests server-side to maximize GPU throughput.
*   **Concurrent Execution**: Runs multiple models (or copies of same model) in parallel on same GPU.

### What is Model Batching and how does it improve throughput?

Processing multiple inference requests together as a single tensor.
*   **Why**: GPUs are bandwidth-limited. Processing 1 image takes almost same time as 8 images.
*   **Dynamic Batching**: Server waits a small window (e.g., 5ms) to collect requests, forms a batch, executes, then splits results back to users.
*   **Trade-off**: Increases Latency (wait time) to improve Throughput (QPS).
