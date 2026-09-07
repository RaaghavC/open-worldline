# Decoder buffer cleanup result

The full pretrained FP32 decoder completed all 17 frames with low watermark 0.6, per-chunk cleanup and synchronization plus allocator cleanup after each native causal convolution. The measured run took 86.92 seconds, including 4.16 seconds to verify/load weights and 79.99 seconds to decode. Sampled Metal driver memory peaked at 10.30 GiB; the minimum sampled available system memory was 3.54 GiB. All unchanged resource limits passed. [Measured run](codec-results/full-decode-v3/metrics.json), [artifact and pixel verification](codec-results/full-decode-v3/artifact-audit.json)

The actual model registered 62 causal convolution modules. Decoding invoked 35 of them, producing 169 cleanup callbacks, all completed. The encoder modules were not called. The event log matches the report exactly, and all added hooks and causal caches were cleared. The earlier tiny CPU fixture's 129 calls reflected its smaller block count; the full model's actual count is 169.

The saved first reconstruction is pixel-identical to the successful standalone first-image reconstruction after removing only the diagnostic frame's 32-pixel header: zero changed RGB channel values. This comparison establishes saved 8-bit pixel identity. It does not assert FP32 internal tensor identity. The remaining 16 frames decode retained Gaussian noise. No denoiser ran, and these are not generated world-model futures or evidence of image quality.

Two full 17-frame decode attempts stopped because available system memory fell below the unchanged 2 GiB floor. Setting the allocator low watermark to 0.6 did not complete the second attempt. Both failures retain their original inputs, source, measurements and watchdog records. [First stop](codec-results/full-decode-stopped-v1/watchdog-stop.json), [second stop](codec-results/full-decode-stopped-v2/watchdog-stop.json)

The successful configuration adds synchronization and `empty_cache` immediately after each native `CausalConv3d` forward call. The new `codec_memory.py` context manager installs removable forward hooks. It returns no replacement output and changes no convolution, normalization, latent scaling, precision, tiling or temporal cache equation. The successful first-image wrapper and reader remain byte-identical, so the existing independently encoded starting observation stays valid.

| Attempt | Cleanup policy | Result | Measured seconds | Peak sampled driver memory | Minimum sampled available memory |
| --- | --- | --- | ---: | ---: | ---: |
| v1 | Per chunk, default low watermark | Stopped | 39.40 | 15.20 GiB | 1.57 GiB |
| v2 | Per chunk, low watermark 0.6 | Stopped | 37.19 | 13.90 GiB | 1.86 GiB |
| v3 | Per chunk and causal convolution, low watermark 0.6 | Passed | 86.92 | 10.30 GiB | 3.54 GiB |

The stopped durations are partial runs and cannot be compared as complete decode speeds. The v3 callback durations sum to 74.66 seconds, including waits for queued GPU work. That sum is not isolated allocator overhead.

Four CPU tests used the native 48-channel codec with reduced layer widths and 17 complete output frames. The resulting pixels were exactly equal with and without the new hooks, including the existing per-chunk cleanup. The fixture registered 46 causal convolution modules and observed 129 native convolution calls during decoding. Each call received one cleanup callback in the same order as independent native-call instrumentation. Registration failure, a failing callback and a native interruption removed every new hook and cleared caches. Mocked MPS calls confirmed synchronization occurs before allocator cleanup. No actual GPU operation ran in these tests. [CPU results and exact source](codec-results/memory-cpu-v1/tests.json)

The full decoder profiler requires those source-matching equality checks when `--per-convolution-cleanup` is used. It saves per-layer callback counts and time, and flushes a JSONL record after each callback so a stopped run retains completed cleanup events. Callback timing includes waits for previously queued GPU work and the allocator call. It does not measure isolated allocator overhead. Actual total decode time and sampled memory must establish whether the policy helps on the full model.

The following records the completed v3 launch from the repository root. Its output directory now exists and must remain unchanged. A reproduction must use a new output directory:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
  ../../work/wan-adapter-env/bin/python experiments/wan22_native/codec_decode_profile.py \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --observation-run ../../work/wan22-native-codec-image-v2 \
  --cpu-report experiments/wan22_native/codec-results/decode-cpu-v4/tests.json \
  --codec-cpu-report experiments/wan22_native/codec-results/cpu-v3/tests.json \
  --memory-cpu-report experiments/wan22_native/codec-results/memory-cpu-v1/tests.json \
  --device mps --allocator-cleanup --per-convolution-cleanup \
  --output ../../work/wan22-native-codec-decode17-v3
```

The 900-second, 18 GiB and 2 GiB minimum-available-memory limits are unchanged. This completed synthetic decoder measurement can supply the decoder cost for a separately gated native clip. Its exact source, weights, input, runtime environment and cleanup policy must match that clip's decoder. It establishes no image-quality result.
