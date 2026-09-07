# Actual native5B pair on CPU

The complete pretrained model ran one positive/negative prediction pair on the CPU using the same selective BF16/FP32 policy, inputs, observation, text and source as the earlier [MPS pair](../pair-v1/README.md). All 825 loaded tensor records match exactly. No GPU allocation, training, action input or full video generation occurred.

Loading took 82.97 seconds. The positive and negative calls took 174.08 and 175.88 seconds, respectively. Total execution was 432.98 seconds. Peak sampled process RSS was 10,327,605,248 bytes. Both outputs were finite and the known first latent was preserved after the one CPU UniPC update.

The [measured report](metrics.json), [parent completion](terminal.json), inputs, velocities, one-step latent and exact source are retained. The [device comparison](../device-comparison-v1/README.md) describes numerical differences from the MPS result. CPU execution with the same mixed precision is a second backend, not a full-FP32 or official CUDA reference.
