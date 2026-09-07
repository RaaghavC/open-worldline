# Full 17-frame decoder cost

The separate full-decode profiler passed seven CPU tests. It has not decoded the real pretrained model at this shape yet. Its purpose is to measure the decoder time and memory needed after a future 17-frame transformer run. The first-image reconstruction profile cannot supply that measurement because it decodes only the first temporal chunk. [CPU evidence and exact source](codec-results/decode-cpu/provenance.json)

The new input has shape `[1,48,5,18,32]` in native normalized latent coordinates. Its first latent comes from a completed, verified Wan2.2 first-image codec run. The remaining four latents come from a retained FP32 CPU Gaussian noise tensor. The profiler reads only `observation.safetensors` and the first-image run's JSON record. It reads no RGB file, action command, old Wan2.1 latent or future training target, and runs no denoiser.

[Canonical noise provenance](codec-inputs/manifest.json) records the generator, seed 20260918, software version, source and tensor/file hashes. The retained file SHA-256 is `aa4725c2d1ada01de94b46aa43110a8d16f24ee596e7cee96b2035a72368bcc1`. Those exact bytes define the input on every platform. Recreating a tensor from the seed is not used as a cross-platform identity check. Both the original noise and the combined observation/noise latent are saved with the run.

The official decoder expands the five latent chunks into 1, 4, 4, 4 and 4 output frames. Hooks record each chunk's time, shape, dtype and first-chunk flag. The complete decode measurement also includes normalization, latent transfer, convolution before the chunk loop, unpatching and output clamp. The optional per-chunk allocator cleanup is the same one whose CPU output equality was verified for this codec. Both timing and cleanup hooks are removed on errors.

The profile uses FP32 in its own process, with the existing 900-second limit, 18 GiB sampled RSS/Metal allocation ceiling and 2 GiB minimum available system memory. The updated codec explicitly disables an ambient autocast context. The 5B transformer and UMT5 encoder must have exited before this codec process starts.

For this workspace, run from `outputs/open-worldline` after the first-image codec process has completed and the new run is scheduled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../../work/wan-adapter-env/bin/python experiments/wan22_native/codec_decode_profile.py \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --observation-run ../../work/wan22-native-codec-image-v1 \
  --cpu-report experiments/wan22_native/codec-results/decode-cpu/tests.json \
  --codec-cpu-report experiments/wan22_native/codec-results/cpu-v2/tests.json \
  --device mps --allocator-cleanup \
  --output ../../work/wan22-native-codec-decode17-v1
```

Use `--device cpu` for explicit CPU execution. Omitting `--allocator-cleanup` is a separately declared allocator policy; it does not change decoder arithmetic. Each output directory must be new. The profiler rejects incomplete first-image records, changed source/weight identities, changed observations and stale CPU checks.

All 17 images are retained with a visible **Synthetic decode cost diagnostic** header outside their unchanged 512 by 288 pixel region. The first frame is reconstructed from the known initial latent; the other 16 are decodes of synthetic noise. No image-quality score is reported, and none of the frames count as world-model predictions. Preview playback is 10 frames per second, separate from measured decoder speed.

The saved report includes exact source, input, weight and output hashes; first-image provenance; codec load time; total decode time; all five chunk times; finite-output and cleared-cache checks; and sampled memory. Use the measured codec load plus full decode and artifact overhead to estimate a later native clip's total time. A synthetic decode tests execution cost at this shape. It does not establish generated image quality or numerical equality with CUDA.

The CPU suite covers canonical noise without RNG regeneration, prefix/suffix assembly without caller mutation, observation-only materialization, rejection of failed or stale first-image records, all five native decoder chunks and unchanged output under measurement/cleanup hooks, hook removal on partial registration or interruption, and failure recording. The prior first-image codec files and their original CPU/meta evidence remain preserved; only the independently found ambient-autocast bug was fixed before measurements.
