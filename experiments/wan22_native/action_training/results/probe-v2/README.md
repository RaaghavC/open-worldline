# Two numerical optimizer updates completed

The fresh second probe completed two paired updates of the original 947,712-parameter FP32 action adapter. Both updates had finite gradients. The first GRU gradient was zero, as expected from the zero output projection; the second was nonzero. All 825 recorded foundation parameter values matched before and after training.

This establishes that the prescribed training computation can run on this machine. **The visual foundation remains failed.** The trainer generated no images, measured no action-control score, and did not admit the fixed16 pilot. No image-quality, generalization or novelty conclusion follows from this probe.

| Measurement | Update 1 | Update 2 |
|---|---:|---:|
| Window start | 0 | 8 |
| Shared integer timestep | 506 | 628 |
| Paired future velocity MSE | 0.2454436421 | 0.1776974425 |
| Global gradient L2 before clipping | 0.0094944797 | 0.0095356749 |
| Command GRU gradient L2 | 0 | 0.0000436382470 |
| Synchronized paired-update time | 6.235 seconds | 4.913 seconds |

The two loss values use different windows, timesteps and noise. They cannot establish a learning trend. Each update uses two sequential batch-one forwards, closed then open, with half of each future-latent MSE accumulated before one optimizer step. Both gradient norms were below the clipping threshold of 1.

Worker elapsed time was **129.239487 seconds**. Verifying and loading the external foundation took 87.266 seconds, before-training parameter hashing 16.086 seconds, and after-training hashing 13.810 seconds. Those times include neither image generation nor a decoder. The sampled peak process RSS was 0.778061 GiB, peak MPS active memory 9.647139 GiB, and peak MPS driver memory 10.351608 GiB. Minimum sampled available system memory was 3.284973 GiB. Samples occurred about every half second and can miss brief peaks; the limits remained 900 seconds, 18 GiB and 2 GiB available memory.

The [artifact audit](artifact-audit.json) verifies all 84 parent/worker source snapshots, the 825 before/after parameter records against the loader, exact prescribed noise/RNG draws, prompt identity, and all three checkpoint bundles. Checkpoints [0](result/checkpoint-0000/manifest.json), [1](result/checkpoint-0001/manifest.json) and [2](result/checkpoint-0002/manifest.json) retain the original adapter values, optimizer state, CPU and private-generator RNG state, and file/tensor hashes. Checkpoint 0 has no optimizer moments; checkpoints 1 and 2 contain finite moments for all 18 adapter tensors with the corresponding exact step counts. AdamW weight decay can change upstream weights even when their first-update data gradient is zero; the reported second-update GRU norm is the direct gradient check.

The [first failed probe](../probe-failure-v1/README.md) remains unchanged. This run started from the original seed with a fresh adapter and optimizer, after the separately tested [MPS pooling fix](../../../action_adapter/POOLING.md). It did not resume the failure. The two branches at start 0 share the canonical derived observation; only that pair has different outgoing commands. The second pair has equal commands and different observations. Both are overlapping windows from one Atrium layout.

[result/metrics.json](result/metrics.json), [terminal.json](terminal.json), checkpoints, noise/context tensors, logs, memory samples and measured source snapshots preserve their executed bytes. [publication.json](publication.json) binds original and published files and enumerates operational path changes in `launch.json`. The raw work directory was not modified. External foundation weights and the local raw RGB capture are explicit launch placeholders and are not bundled. The complete original cache provenance remains linked through hashes and the public cache files.

[admission/executed.json](admission/executed.json) preserves the exact parent decision, SHA256 `98f87f6b2754b6ce49df930aa2a9a6a99d69079114c84eaecac55b338f2b9462`. Its scope is optimizer-feasibility-only and visual status is failed. [admission/portable.json](admission/portable.json) changes only the three evidence image paths and has a separately recorded hash. It is not the exact file passed to the worker. Neither copy admits fixed16 or improved visual quality.

The original adapter, training code and recovery files use Apache-2.0. The external foundation attribution is retained in the native [NOTICE](../../../NOTICE). Genuine cached text attribution is in the [text-cache record](../../../../wan_adapter/text_cache/native-results/README.md), and original RGB attribution is in the [Atrium data license](../../../../atrium_data/DATA-LICENSE). No external foundation checkpoint is included.
