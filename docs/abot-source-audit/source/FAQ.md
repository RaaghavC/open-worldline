# Frequently Asked Questions (FAQ)

This document addresses common questions regarding hardware requirements, software environment, and runtime compatibility for **ABot-World**.

---

## Hardware Compatibility

### Can ABot-World run on NVIDIA GeForce RTX 3090 / 4090?

Peak GPU memory usage is approximately **19 GB**, with at least **64 GB** of system RAM recommended. Under these constraints, GPUs such as the RTX 3090 (24 GB) and RTX 4090 (24 GB) are in principle capable of running the released causal student model.

We have verified inference on **RTX 4090**. RTX 3090 has not yet been evaluated on our side, though we expect it to be feasible given the memory footprint. Please ensure a CUDA environment consistent with the [Setup](README.md#-setup) instructions.

### Is multi-GPU inference supported?

Not at present. The current release targets **single-GPU** desktop inference. Multi-GPU parallelism could, in principle, improve throughput, but has not been systematically validated. We recommend single-GPU deployment until multi-GPU support is officially released.

---

## Software Environment

### Which CUDA versions are supported?

We have tested **CUDA 12.8** and **CUDA 12.9** (`cu128` / `cu129`). Other CUDA versions have not undergone full verification. When in doubt, please align with the environment described in the [Setup](README.md#-setup) section.

### Is Windows supported? Which operating system is recommended?

We strongly recommend **Linux**. Support for certain components under Windows is limited or fragile. For reproducible installation and inference, please prefer Linux (e.g., Ubuntu 22.04 as in our reference setup).

---

## Gradio Demo

### Which browser should I use for the Gradio interface?

We recommend **Google Chrome** for online Gradio usage, where we have observed the most consistent behavior.

---

## Related Resources

- Installation and environment setup: [README.md](README.md#-setup)
- FlashAttention glibc issues: [Dao-AILab/flash-attention#1708](https://github.com/Dao-AILab/flash-attention/issues/1708)
- SageAttention source build: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/main)
