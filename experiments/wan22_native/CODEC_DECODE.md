# Full 17-frame decoder cost

The full pretrained FP32 decoder completed the prescribed 17-frame synthetic test in 86.92 seconds. Weight verification/loading took 4.16 seconds and decoding took 79.99 seconds. Sampled Metal driver memory peaked at 10.30 GiB and available system memory remained at least 3.54 GiB in the samples. The run preserved all 17 annotated images, exact latent inputs, source, runtime settings, 169 convolution-cleanup events and memory records. [Complete measured run](codec-results/full-decode-v3/metrics.json)

The first reconstructed image has exactly the same saved RGB bytes as the standalone first-image result, excluding its diagnostic header. All five native temporal chunks completed with output lengths 1, 4, 4, 4 and 4. Source, input, output, event-log and memory checks are recorded in the [artifact audit](codec-results/full-decode-v3/artifact-audit.json). This was a decoder-cost test using one known first latent and four noise latents. No world-model denoiser ran.

The first full-shape pretrained decode stopped after 39.40 seconds when available system memory fell below 2 GiB. The last recorded available memory was 1,680,408,576 bytes; Metal driver memory was 15,940,239,360 bytes. A second attempt with low watermark 0.6 also stopped, after 37.19 seconds, with 1,997,963,264 bytes available. Neither produced a completed decode or output images. These incomplete timings cannot admit a full clip run. [First stop](codec-results/full-decode-stopped-v1/watchdog-stop.json), [second stop](codec-results/full-decode-stopped-v2/watchdog-stop.json)

The successful third configuration added optional synchronization and unused-buffer cleanup after each native causal convolution. Its exact CPU output and hook checks, preserved arithmetic, actual counts and reproduction command are described in [Decoder buffer cleanup result](CODEC_MEMORY.md).

The profiler passed nine CPU input/record/gate tests, including the allocator environment record and the separate per-convolution cleanup gate. Four additional cleanup tests establish exact tiny CPU output equality and hook removal. [Current decoder CPU evidence](codec-results/decode-cpu-v4/provenance.json), [cleanup CPU evidence](codec-results/memory-cpu-v1/tests.json), [independent cleanup review](codec-results/memory-independent-v1/report.json)

The new input has shape `[1,48,5,18,32]` in native normalized latent coordinates. Its first latent comes from a completed, verified Wan2.2 first-image codec run. The remaining four latents come from a retained FP32 CPU Gaussian noise tensor. The profiler reads only `observation.safetensors` and the first-image run's JSON record. It reads no RGB file, action command, old Wan2.1 latent or future training target, and runs no denoiser.

[Canonical noise provenance](codec-inputs/manifest.json) records the generator, seed 20260918, software version, source and tensor/file hashes. The retained file SHA-256 is `aa4725c2d1ada01de94b46aa43110a8d16f24ee596e7cee96b2035a72368bcc1`. Those exact bytes define the input on every platform. Recreating a tensor from the seed is not used as a cross-platform identity check. Both the original noise and the combined observation/noise latent are saved with the run.

The official decoder expands the five latent chunks into 1, 4, 4, 4 and 4 output frames. Hooks record each chunk's time, shape, dtype and first-chunk flag. The complete decode measurement also includes normalization, latent transfer, convolution before the chunk loop, unpatching and output clamp. The optional per-chunk allocator cleanup is the same one whose CPU output equality was verified for this codec. Both timing and cleanup hooks are removed on errors.

The profile uses FP32 in its own process, with the existing 900-second limit, 18 GiB sampled RSS/Metal allocation ceiling and 2 GiB minimum available system memory. The updated codec explicitly disables an ambient autocast context. The 5B transformer and UMT5 encoder must have exited before this codec process starts.

The following records the first full-shape launch from `outputs/open-worldline`. That output directory now contains the retained stopped run and cannot be reused. Any retry requires a separately reviewed configuration and a new output directory:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../../work/wan-adapter-env/bin/python experiments/wan22_native/codec_decode_profile.py \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --observation-run ../../work/wan22-native-codec-image-v2 \
  --cpu-report experiments/wan22_native/codec-results/decode-cpu-v2/tests.json \
  --codec-cpu-report experiments/wan22_native/codec-results/cpu-v3/tests.json \
  --device mps --allocator-cleanup \
  --output ../../work/wan22-native-codec-decode17-v1
```

Use `--device cpu` for explicit CPU execution. Omitting `--allocator-cleanup` is a separately declared allocator policy; it does not change decoder arithmetic. Each output directory must be new. The profiler rejects incomplete first-image records, changed source/weight identities, changed observations and stale CPU checks.

The second attempt changed only the allocator low watermark to `0.6`, set before importing PyTorch in the new process. It still stopped at the available-memory floor. Native codec equations, FP32 computation, chunk cleanup and all resource limits remained unchanged. The report records low/high watermark, automatic fallback, fast-math and preferred Metal-kernel environment values, plus recommended device memory. PyTorch uses the low watermark to trigger cache collection and adaptive command-buffer commits; it is not the 18 GiB hard cap. This command is retained as the v2 failure record. [Pinned PyTorch allocator source](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/mps/MPSAllocator.mm)

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
  ../../work/wan-adapter-env/bin/python experiments/wan22_native/codec_decode_profile.py \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --observation-run ../../work/wan22-native-codec-image-v2 \
  --cpu-report experiments/wan22_native/codec-results/decode-cpu-v3/tests.json \
  --codec-cpu-report experiments/wan22_native/codec-results/cpu-v3/tests.json \
  --device mps --allocator-cleanup \
  --output ../../work/wan22-native-codec-decode17-v2
```

All 17 images are retained with a visible **Synthetic decode cost diagnostic** header outside their unchanged 512 by 288 pixel region. The first frame is reconstructed from the known initial latent; the other 16 are decodes of synthetic noise. No image-quality score is reported, and none of the frames count as world-model predictions. Preview playback is 10 frames per second, separate from measured decoder speed.

The saved report includes exact source, input, weight and output hashes; first-image provenance; codec load time; total decode time; all five chunk times; finite-output and cleared-cache checks; and sampled memory. Use the measured codec load plus full decode and artifact overhead to estimate a later native clip's total time. A synthetic decode tests execution cost at this shape. It does not establish generated image quality or numerical equality with CUDA.

The CPU suite covers canonical noise without RNG regeneration, prefix/suffix assembly without caller mutation, observation-only materialization, rejection of failed or stale first-image records, all five native decoder chunks and unchanged output under measurement/cleanup hooks, hook removal on partial registration or interruption, and failure recording. Previous source and CPU evidence remain preserved. Current checks include the explicit FP32 autocast guard and the corrected first-image reader for the pinned fully opaque RGBA image.
