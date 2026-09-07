# The MPS gradient difference comes from three near-zero L1 residuals

September 7, 2026. The separate diagnostic completed on the same actual training pair and memory checkpoint as the failed MPS batching check. The original L1 input-gradient check still fails its original limits. Those limits and the original result are unchanged.

This run retains both prediction arrays, targets, ordinary L1 gradients, residual signs and gradients from an identical supplied output gradient. Three of 1,597,440 prediction elements change between exact zero error and a residual of ±5.96046448e-8. None changes between opposite nonzero signs. The absolute-error loss has different derivatives at zero and at a nonzero residual, even when the pixel values differ by very little.

| Comparison | Measured result |
|---|---:|
| Scalar L1 loss difference | 0 |
| Largest prediction difference | 4.17233e-7 |
| Prediction RMS difference | 3.34671e-8 |
| Ordinary RGB-gradient maximum difference | 3.13001e-7 |
| RGB-gradient maximum difference with identical supplied gradient | 2.04636e-12 |
| Relative L2 difference with identical supplied gradient | 2.57168e-7 |

The shared gradient is obtained directly from the reference loss with automatic differentiation and detached before reuse. Both linear comparison objectives include the same target term, preserving the fact that an observed RGB tensor participates in both histories and loss targets. All trainable-parameter and RGB-gradient comparisons under this common gradient meet the unchanged original bounds. The [report](diagnostic.json) decomposes the original difference into the change in the loss derivative and the much smaller difference in backpropagation.

This supports using the batched path as a numerically approximate training implementation for this measured case. It does not establish identical optimizer trajectories, a stable speedup, learned memory quality or equality for every possible input. A separate complete training profile is still required. The diagnostic used one carry-mode pair, performed no optimizer update and opened no reserved scene.

The exact executed sources and original reports are copied without changes. To keep the public artifact smaller, the original tensor file is stored as [diagnostic-tensors.pt.gz](diagnostic-tensors.pt.gz). Decompression was verified against the raw byte count and hash recorded by the executed diagnostic. Reconstruct it with:

```sh
gzip -dk diagnostic-tensors.pt.gz
```

Run that command from this directory and use a new output filename if the raw file already exists. [Publication checksums](publication.json) distinguish the compressed file from its exact reconstructed tensor file. Code and original gradient artifacts use Apache-2.0; source RGB is from the separately released original CC0 capture.
