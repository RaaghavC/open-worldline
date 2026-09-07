# Two-size native Wan2.2 diagnostic

This package prepares a controlled resolution experiment after the [native A100 diagnostic](../../../docs/cloud-gpu-diagnostic-results-2026-09-07.md) also produced severely distorted future frames. It does not contain a new trained model or establish a quality improvement. GPU results for this package have not been produced.

The retained model is the external Apache-2.0 Wan2.2 TI2V-5B at the exact revisions and parameter hashes documented in [the frozen CUDA reference](../cuda_reference/README.md). This package imports that unchanged parameter loader, FlashAttention 2 implementation, VAE and upstream CPU UniPC solver. Its own code handles two shapes, independent image encoding, measured admission, resource limits and retained evidence.

## The controlled comparison

| Condition | RGB frames | Native latent C,F,H,W | Tokens | Clean observed tokens |
|---|---|---|---:|---:|
| Baseline | 17 × 512 × 288 | 48,5,18,32 | 720 | 144 |
| Spatial | 17 × 1248 × 704 | 48,5,44,78 | 4,290 | 858 |

Both use the same original room image, cached positive and native-negative text, 50 UniPC steps, shift 5 and guidance 5. The first latent frame stays clean in both classifier-free guidance branches and after each of the 50 solver updates. There are exactly 100 denoiser predictions per generated clip.

The original image has 512 × 288 pixels. The pinned upstream size calculation selects 1248 × 704 under its 1280 × 704 area budget. The spatial input is resized to 1252 × 704 with Lanczos, then cropped two pixels from each horizontal edge. This adds no source-image detail. The native VAE encodes each processed RGB image independently; cached latents are never resized.

The baseline keeps the exact prior saved noise. The larger CPU noise draw is generated once, saved and selected by the complete manifest hash. A seed does not establish equal noise between different shapes or processor architectures. This is one fixed scene and noise draw per shape, not a statistically controlled multi-seed benchmark.

The [complete prepared packet](prepared-inputs-v1/manifest.json) contains 10 files totaling 20,817,899 bytes. Its manifest SHA256 is `765b0863f6b0ce8d9bb0e4edf4fdbcc64fa9e96b997eb5e61c010eb0c2f1ccdf`. It contains original task-created room imagery and synthetic noise, the already published external text embeddings and the retained baseline tensors. It contains no external model weights. Preparation executed on CPU and did not run an observation encoder.

The 17-frame duration remains shorter than the upstream 121-frame default. Cached text remains the earlier FP32 encoding, and the solver remains on CPU. Therefore this experiment tests spatial size while retaining those other choices. It does not reproduce every upstream default.

A larger clip also changes the observation encoder from the earlier retained MPS encoding to a fresh CUDA encoding. The fresh baseline pair measures the resulting initial-velocity difference; it cannot exclude later amplification. A matching fresh-encoding baseline full clip is necessary before attributing any visual improvement solely to resolution. The runner supports that additional baseline clip, subject to its own measured admission and remaining rental time.

## Three separate execution modes

1. **Codec:** Load the original FP32 VAE, independently encode both inputs, and compare the fresh baseline encoding with the retained MPS observation. Repeat each encoded observation five times to measure a full 17-frame decode at that size. These repeated-latent outputs are explicitly labeled codec timing proxies. They are not generated futures. All raw FP32 RGB values and the reconstructed first-frame PNG are retained.
2. **Pair:** After validating completed codec evidence, load the original 825 FP32 core tensors and measure one positive/negative prediction pair at one chosen size. Retain both velocities and their guided result.
3. **Clip:** After validating the completed matching pair and codec runs, execute the fixed 50-step schedule and decode the result in a separate process. Retain all 50 solver latents, all 17 FP32 RGB frames, all PNGs and a contact sheet. A GIF is only a palette-quantized preview at 120 ms per frame, about 8.33 fps playback. It does not measure generation speed.

Every mode is a CPU-only plan unless `--execute` is provided. Each execution requires an exact input manifest hash and a passing CPU report bound to the current runtime sources and tests. Pair and clip admission also requires matching measured hardware, environment, input and source identities. A changed artifact, incomplete count, nonzero worker exit, watchdog stop or cleanup failure prevents admission.

## Time and memory limits

| Mode | Whole-mode deadline |
|---|---:|
| Codec, both sizes | 600 seconds |
| Pair, one size | 900 seconds |
| Clip, core and decode combined | 1,800 seconds |

All modes retain the prior 48 GiB combined host RSS and 60 GiB CUDA-reserved limits, with at least 8 GiB available host and GPU memory. The required device has at least 70 GiB total memory and the exact caller-declared GPU name. The reviewed runtime is Torch 2.5.1, CUDA 12.4 and FlashAttention 2.7.4.post1, with no FlashAttention 3 override.

The clip admission estimate is:

```
core load + 50 × measured pair × 2.0
          + 1.2 × (codec load + measured 17-frame decode) + 30 seconds
```

It must be strictly below 1,800 seconds. The prior small A100 clip took about 1.78 times the sampling duration predicted by simply multiplying its first pair by 50. The factor 2.0 provides a larger allowance for that observed difference. It is a planning heuristic and does not guarantee completion at a larger size. The 30-second allowance covers artifact work approximately; the actual parent deadline applies throughout execution and evidence writing.

This package does not create or delete cloud resources. Any cloud run also requires an independent instance-deletion controller and enough remaining instance time for the admitted work and artifact recovery. Process deadlines are not a provider-enforced spending cap.

## CPU validation and use

Use the same dependencies as the frozen CUDA reference. From the repository root:

The [retained CPU review](reviews/cpu-v2.json) passed all 91 checks in 5.966 seconds, with 29 runtime/config source identities and seven test files unchanged throughout. CUDA remained uninitialized and no model weight values were loaded. The [actual codec plan](reviews/codec-plan-v1.json) then passed its input, source and review checks without model execution. A prior development suite run is retained locally; two fixtures were incomplete while the stricter artifact validator was being added, and the final suite corrects those fixtures.

```sh
python -m experiments.wan22_native.spatial_reference.test_protocol \
  --output work/spatial-cpu-review.json

python -m experiments.wan22_native.spatial_reference.run \
  --mode codec --profile both \
  --expected-gpu NVIDIA\ A100-SXM4-80GB \
  --weights /absolute/path/to/original/weights \
  --inputs /absolute/path/to/prepared-input-packet \
  --input-manifest-sha256 765b0863f6b0ce8d9bb0e4edf4fdbcc64fa9e96b997eb5e61c010eb0c2f1ccdf \
  --cpu-report work/spatial-cpu-review.json \
  --output work/spatial-codec-plan
```

Use a fresh output directory for every plan or execution. On the admitted CUDA host, add `--execute` to a new invocation. Pair additionally needs `--codec-result` and either `--profile baseline` or `--profile spatial`. Clip additionally needs the completed `--pair-result` for that profile. Preserve all earlier result directories with the clip so the admission can be audited.

The CPU suite tests the real schedule using injected predictors, input integrity and preprocessing, native-call argument routing, lossless RGB frame retention, memory and deadline guards, evidence corruption, and failure cleanup. It does not load original weight values or execute CUDA. Passing it establishes software checks, not model quality.

No action adapter is loaded here. The original action-training pilot remains closed while the pretrained model's future frames fail visual review. Genie 3 parity and three novel, high-impact advances remain unproven.
