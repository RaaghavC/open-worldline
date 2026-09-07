# One MPS pooling operation and gradient check

The original observation-pooling helper passed one actual MPS check on an FP32 tensor shaped `[1,48,18,32]`, producing `[1,48,4,8]`. It used the real `observation_pool2d` dispatch. Calling the unsupported native adaptive-pooling kernel was made an error during this check, and automatic CPU fallback was disabled.

The CPU reference used PyTorch 2.5.1 `F.adaptive_avg_pool2d` on the same input and upstream gradient. The MPS output's maximum absolute difference was **1.1920928955078125e-7**, and its input gradient matched the CPU reference exactly. Both were finite and met the previously chosen absolute tolerance 2e-7 and relative tolerance 3e-6. This measures the explicit overlapping-bin implementation, not interpolation or a changed pooling grid.

The synchronized forward and backward took 0.225271 seconds. The parent process completed in 1.483037 seconds, with exit code zero under a 60-second timeout. This single operation is not a foundation training latency or memory measurement. No model, foundation parameter, optimizer or generated image was used. The tensors alone occupied less than one megabyte.

[report.json](report.json) contains the measured comparisons and individual tensor hashes. [tensors.safetensors](tensors.safetensors) retains the original input, shared upstream gradient, CPU/MPS outputs and CPU/MPS input gradients. [terminal.json](terminal.json) and [worker.log](worker.log) preserve the completed subprocess record. The exact executed helper, model dispatch source and check script are retained. The check script is an executed-source record with its original workspace layout, not a new portable launcher.

[cpu-report.json](cpu-report.json) is a verified copy of the preceding 45-check CPU report, added during publication. It is identical to the report hash read before MPS execution. Its 15 new pooling checks include doubly nondivisible shapes, overlapping bins, gradients, batch isolation and preservation of the existing CPU dispatch; the other 30 checks cover the prior adapter tests.

[publication.json](publication.json) binds every delivered file. Measured report values, tensor files and executed source bytes were not changed. The code is Apache-2.0 and implements the same floor/ceil bin boundaries described in [POOLING.md](../../POOLING.md). The old failed native5B probe remains failed. This operation check does not establish full-model training feasibility or image quality.
