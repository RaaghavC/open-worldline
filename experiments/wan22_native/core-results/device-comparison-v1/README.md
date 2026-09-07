# Same inputs on CPU and MPS

The native5B CPU and MPS profiles used bit-exact initial noise, noisy latents, observed prefix and token times. All 825 loaded weight records, genuine text, source, observation and sampling identities match. This read-only comparison checks their saved outputs without repeating inference.

| Saved future-latent output | RMS difference divided by CPU RMS | Maximum absolute difference |
| --- | ---: | ---: |
| Positive prediction | 0.38379% | 0.20906 |
| Negative prediction | 0.42265% | 0.21825 |
| Guided prediction | 1.91576% | 0.66612 |
| Latent after the first solver update | 0.01050% | 0.00271 |

These values summarize 110,592 future-latent elements per output. The [complete report](report.json) also gives all-latent, observed-latent and per-latent-frame differences, cosine similarities and exact equality counts. The known prefix in the one-step latent is bit-exact on both devices.

Guidance magnifies some prediction differences. This initial-step comparison does not establish agreement through all 50 generation steps and cannot identify the cause of the [failed clip](../../sample-results/clip50-v1/README.md) by itself. Both devices use the declared mixed precision. The CPU output is not a full-FP32 or CUDA ground truth, and there is no pass threshold or image-quality claim in this descriptive comparison.

Original [CPU](../cpu-pair-v1/README.md) and [MPS](../pair-v1/README.md) artifacts, comparison source and all output hashes are retained. The separate [complete real-video reconstruction](../../action_data/README.md#measured-17-frame-reconstruction) retained the room layout and door motion at 32.26 dB PSNR across all 17 frames.

An [independent NumPy audit](independent-audit.json) recomputed 256 numeric values from the saved tensors. Its largest reduction-order difference was 3.87e-14. It also checked the input tensors, all 825 weight metadata records, source identities and exact float32 guidance arithmetic on both devices. This validates the comparison arithmetic; it adds no image-quality threshold. The [audit source](audit-wan22-device-comparison.py) accepts the CPU run, MPS run and comparison report as explicit paths and imports no model code.
