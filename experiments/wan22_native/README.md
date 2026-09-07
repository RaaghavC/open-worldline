# Wan2.2 TI2V-5B execution experiment

This package contains an explicit CPU/MPS execution port of the **external pretrained Wan2.2 TI2V-5B model**. It preserves the official architecture, parameter names and tensor shapes. It is not an original Worldline model, an action-controlled model, or evidence of improved image quality.

Official code is pinned to commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`; weights are pinned to `Wan-AI/Wan2.2-TI2V-5B` revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. The model, attention and shared configuration source copies are byte-exact. See `NOTICE`, `LICENSE-APACHE-2.0.txt` and `provenance.json` for attribution, URLs and hashes. No external weights are included in the repository.

## Precision and loading

The native transformer has 4,999,787,712 parameters in 825 tensors. The declared storage policy retains 68,573,376 parameters in FP32 and stores 4,931,214,336 in BF16, requiring 10,136,722,176 bytes, or 9.44056 GiB, for parameters. That count comes from a meta model and source headers. It excludes activations, temporary conversions, allocator caches, the VAE, text encoding and other applications.

Time embedding/projection, head, norm parameters and block modulation stay FP32. Residual, modulation, guidance and solver arithmetic stay FP32. Other Linear operations and patch Conv3d explicitly cast their inputs to the BF16 weight dtype. There is no MPS BF16 autocast. Original norm equations are retained. Global, noncausal SDPA uses BF16 Q/K/V with key masking and restores the original query dtype. Text is padded to 512 tokens before projection with no added text mask. Tested real-valued RoPE and CPU-double timestep functions are imported from the earlier attributed native-control port.

The full model can only be constructed on `meta`. `load_weights.load_core(directory, device=..., check=...)` verifies the pinned configuration, index and three shard hashes, then reads one source tensor at a time. It records original FP32 and loaded tensor hashes and checks each converted value against `source.to(declared_dtype)`. A CPU conversion reference, destination parameter and temporary verification copy may coexist with that source tensor. CPU FP32 parameters receive owned copies, so they do not retain source-file mappings. There is no full state dictionary, optimizer or master FP32 model copy.

## Native input and text contract

The proposed reduced diagnostic is 17 frames at 512×288 pixels. The **new 48-channel Wan2.2 VAE** produces latents `[48,5,18,32]`, corresponding to 720 patch tokens. The first 144 tokens represent the observed initial latent frame and receive time zero; future and padding tokens receive the solver time. The old 16-channel Wan2.1 cache is incompatible.

The first RGB observation must be encoded alone in a separate new-VAE process. The core process reads that verified cache, genuine text and reproducible noise. It receives no future RGB, actions, teacher labels, camera pose or renderer data. It restores the clean first latent after the CPU solver step. The separately owned codec implementation and profiler are `codec.py` and `codec_profile.py`.

`text-reuse.json` verifies exact equality of the Wan2.2 configured negative prompt with the existing cached 403-byte prompt, all four tokenizer files, and the UMT5 encoder hash. The positive text remains the same Atrium description. The reused cached contexts were computed in FP32; they are not claimed bitwise equal to an official BF16 encoder run. The text encoder is not loaded into the core process.

## CPU gates and one-pair profiler

The implementation tests use tiny synthetic weights and a nonzero output head to compare the full literal official model with the FP32 and selective-BF16 port. They also check production parameter counts on meta, prefix/token order, key masking, text padding, explicit dtypes, exact streamed conversion, source-file ownership and malformed inputs. The independent tests cover production 128-wide attention heads, batch isolation, mapping cleanup and nonfinite-source rejection. CPU SDPA parity does not establish CUDA FlashAttention or actual MPS equivalence.

```sh
python -m pytest experiments/wan22_native/test_core.py \
  experiments/wan22_native/test_independent.py \
  experiments/wan22_native/test_profile_independent.py -q
```

The [retained CPU evidence](core-results/README.md) includes exact source snapshots, both independent reports, earlier failures and the weight-download manifest. Current implementation checks passed all 16 tests; the independent core and profiler checks passed seven and four tests respectively. The latest affected-source rerun passed 20 checks in total. No actual 5B forward was run for these records.

The core runtime/test versions are listed in `requirements-core.txt`. No package installation was needed for the local tests: the existing pinned Wan environment was used, with its base Python's installed pytest appended after the environment's package paths. Execution reports record that command and the resulting versions.

Only after the new codec cache and source-bound CPU reports pass, an explicitly scheduled process may run:

```sh
python -m experiments.wan22_native.profile_core \
  --weights /path/to/wan22-ti2v5b-weights \
  --observation-cache /path/to/completed-new-vae-initial-observation \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --cpu-report experiments/wan22_native/core-results/cpu-v3/report.json \
  --independent-report experiments/wan22_native/core-results/independent-core/report.json \
  --output /path/to/new-native5b-pair-profile --device mps
```

The parent enforces 900 seconds, 18 GiB process RSS and at least 2 GiB available system memory. The worker applies an MPS allocator cap and samples Metal allocation during loading and inference. The core records its own five inherited MPS allocator/runtime environment values and the device recommended-memory bytes in `metrics.json`; `launch.json` also records the exact variables passed to the child process. Unset values remain JSON null. Codec settings are recorded separately and are not applied to the core automatically. Both source copies and imported helper sources are snapshotted. Failures and partial outputs remain in the new run directory.

The profiler runs exactly one positive/negative forward pair at the initial native timestep and one official CPU UniPC step. It retains the initial noise, clean-prefix latent, token times, velocities and resulting latent. Defaults are the native 50 steps, shift 5 and guidance 5. The linear 50-pair timing estimate excludes decode and later-step variation and does not automatically authorize a full clip. There is no full generation loop in this profiler. Full-video GPU memory, speed and image quality require their own measurements; none follows from the parameter count or CPU tests.

## Actual local core measurement

The [first actual 5B core profile](core-results/pair-v1/README.md) completed on the M4 Pro: 96.110 seconds to verify and stream weights, 5.765 seconds for the positive forward and 3.263 seconds for the negative forward. All outputs were finite and the known prefix was preserved after one CPU UniPC step. The complete measured inputs, outputs, sources and load records are retained. This is one pair, not a video.

The native codec reconstructed the original first image successfully. Two full 17-frame decode-cost attempts subsequently hit the fixed 2 GiB free-system-memory floor, including an attempt with an earlier allocator cleanup threshold. Those failures remain retained. The [third full decoder measurement](CODEC.md#bounded-reconstruction-profile) completed in 86.92 seconds, including 79.99 seconds decoding, with synchronization and unused-buffer cleanup after each causal convolution. Its first reconstructed frame exactly matches the earlier saved RGB pixels. This used synthetic future latents and measured execution cost, not generated image quality.

## Complete clip and failed visual inspection

The [full native 50-step run](sample-results/clip50-v1/README.md) completed in 527.57 seconds. All 100 positive/negative calls and 50 prefix restorations passed their execution checks. The separate FP32 decoder produced one starting reconstruction and 16 generated future frames. All original frames, raw decoded pixels, final latents, source and measurements are retained.

The future frames contain colored, warped surfaces around the doorway. The visual result failed inspection. The 17-frame, 512 × 288 result cannot establish the advertised 720P quality, useful action control or frontier-world-model performance. The [sampler documentation](SAMPLE.md) describes exact inputs, process separation and source-bound timing admission.

The [source-contract audit](core-results/contract-audit-v1/README.md) found no concrete mismatch in the checked sampling and normalization equations. A [same-input CPU/MPS comparison](core-results/device-comparison-v1/README.md) measured differences in the initial predictions; it does not establish whole-video agreement or identify the cause of the visual failure. The [complete real-video reconstruction](action_data/README.md#measured-17-frame-reconstruction) retained the room layout and door motion, with 32.26 dB PSNR across all 17 frames. All reconstruction pixels and the independent scoring audit are included.

The separate original [action adapter](action_adapter/README.md) has 947,712 trainable parameters and a frozen 5B foundation. Its CPU checks cover ordered commands, the observed-image input, gradient flow and unchanged base parameters. The [training-data preparation](action_data/README.md) uses eight fixed original Atrium windows and independently encoded starting observations. After correcting an unsupported MPS pooling operation, a [two-update numerical probe](action_training/results/probe-v2/README.md) completed. It verified nonzero second-update GRU gradients and all 825 unchanged foundation parameter records. These components have not demonstrated native-5B action control. The longer pilot remains closed because the visual foundation still fails.

The [fixed shift-3 comparison](shift3/results/clip50-v1/README.md) completed in 477.50 seconds with every original input retained. Severe distortion remained, so the changed shift did not resolve the visual failure. A separate [CPU portability diagnostic](codec-cpu-diagnostics/v1/README.md) investigates one Linux-only tiny-fixture equality failure; the actual Mac cache remains unchanged.

The separate [TAEHV comparison](tiny_decoder/results/README.md) measures a much smaller external approximate decoder on the same saved latents. The first real-video reconstruction decode took 0.869 seconds at 27.23 dB PSNR; the full VAE reconstructed at 32.26 dB in 61.37 seconds. Both generated controls remain visibly distorted. All 51 output frames, raw pixels, scores and independently checked identities are retained.

The [independent official-equation CPU reference](official_cpu/README.md) completed the same initial positive/negative prediction pair in 507.00 seconds of worker time. It streamed original FP32 weights through the upstream equations, with CPU attention and precision-context substitutions. All 825 original tensor hashes matched the earlier run. Peak combined process memory was 1.48 GB. Against this reference, the earlier CPU future predictions differed by 0.361% positive, 0.373% negative and 2.270% after guidance, measured as RMSE divided by reference RMS. The corresponding Mac differences were 0.342%, 0.423% and 1.989%. This initial-pair result establishes neither CUDA equivalence nor a cause for the failed generated video.
