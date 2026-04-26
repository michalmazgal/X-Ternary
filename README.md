# 🚀 X-Ternary: Native 2-bit Sparse Quantization

X-Ternary is a hardware-native 2-bit quantization scheme for Large Language Models (LLMs). While most research focuses on the mathematical complexity of 1.58-bit (BitNet), X-Ternary prioritizes **hardware efficiency** and **structural sparsity**.

## 🧠 The 4-State System
Every weight is stored using exactly 2 bits, representing four distinct states:
- `01` ➔ **+1** (Positive)
- `10` ➔ **-1** (Negative)
- `00` ➔ **0** (Neutral)
- `11` ➔ **x** (**The Turbo State** - Structural Sparsity)



## ⚡ Why X-Ternary?
- **Zero Packing Overhead**: Perfectly aligns with 8-bit bytes (4 weights per byte), unlike 1.58-bit schemes which require complex bit-shifting.
- **Native NVIDIA 2:4 Support**: The `x` state is designed to map directly to NVIDIA's Structured Sparsity, enabling immediate 2x throughput on Ampere and Hopper GPUs.
- **Energy-Efficiency**: The `x` state (11) can act as a physical gate to "power off" computation at the transistor level, drastically reducing TCO and heat.

## 🛠 Project Roadmap
- [ ] Simulation scripts for X-Ternary quantization.
- [ ] Benchmarks on DeepSeek-V3/R1.
- [ ] Custom CUDA kernels for native 2-bit inference.
- [ ] 
