# Wan2.2 first-image codec

The separate FP32 codec wrapper passed ten small CPU tests. The official checkpoint's metadata matches all 196 architecture tensors and 704,688,668 parameters. No real checkpoint tensor values were loaded for computation, and no GPU encode/decode has run. [Current CPU tests and exact checked source](codec-results/cpu-v2/provenance.json)

Independent review found that a caller's ambient autocast could reduce internal precision despite the original wrapper returning FP32 tensors. Explicit autocast-disabled contexts now surround native encode/decode. A regression checks every Conv3d output and exact equality inside and outside CPU BF16 autocast. The previous nine-test package and its metadata/source inspection remain [preserved](codec-results/cpu/provenance.json); the loader and architecture were unchanged by this fix.

This is Alibaba Wan's pretrained codec with an original local execution wrapper. It is not an original Worldline model or a trained adaptation. [Official source at commit 42bf4cf](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/modules/vae2_2.py) is copied unchanged in `vendor/vae2_2.py`, under Apache-2.0. Its SHA-256 is `eab5ce4aa1ce03af2978f2a8e8364c419f5fbb8535d265ac86b0b02ddcf0c1f6`. [Source and normalization record](codec-source.json)

The pinned `Wan2.2_VAE.pth` file is 2,818,839,170 bytes, SHA-256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`, from [official weight revision 921dbaf](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e). The metadata inspection used `torch.load(weights_only=True, mmap=True, map_location='meta')`, materialized no weight values, and found no missing or unexpected keys. Its 6.14-second interval includes a complete streamed file hash. RSS after inspection was 218,791,936 bytes, not a measured peak for actual inference.

## Input and output

`Wan22Codec.encode` accepts finite FP32 RGB in `[-1,1]`, shaped `[B,3,1+4n,H,W]`, with spatial dimensions divisible by 16. The output is normalized `[B,48,1+n,H/16,W/16]`. One 512 by 288 initial image produces `[1,48,1,18,32]`. This differs from the old 16-channel Wan2.1 cache, which cannot be reused.

The wrapper retains the official 2-by-2 spatial patch order, `[False, True, True]` temporal downsampling, all 48 FP32 means and standard deviations, the causal caches, and `first_chunk=True` for the first decoded latent. Encoding subtracts the native mean and multiplies by reciprocal standard deviation. Decoding applies the inverse transformation, then the official output clamp. Native network equations are unchanged.

Every public encode/decode call clears caches before execution and in `finally`, including validation or neural-operation errors. Errors propagate to the caller. There is no CUDA autocast context or catch-and-return-None behavior. Checkpoint loading uses safe tensor-only deserialization, mmap, a meta architecture, exact key/shape/dtype checks and FP32 parameters. The full codec should run in a separate process from the 5B transformer and text encoder.

## What the CPU tests establish

The fixture keeps 48 latent channels, the real spatial/temporal stages and native layer equations while reducing encoder/decoder widths to 4. It checks:

- Exact native normalization and encode/decode equation equality against direct calls to the unchanged model.
- Patch order against an independently written reshape/permute expression.
- Repeated A/B/A encode and decode, unchanged caller inputs, and cleared caches.
- Encoding the first image alone versus the first latent of a five-frame clip, plus a perturbation of all future images. Both maximum first-latent differences were 0.0 in this CPU test.
- Native first-chunk decoder flags and output shapes.
- Optional per-chunk cleanup callbacks with exact CPU output equality, correct chunk counts and hook removal.
- Cache/hook cleanup after exceptions, rejection of invalid/nonfinite inputs, safe mmap loader arguments and rejection of stale CPU reports.

The optional MPS cleanup callback only releases unused allocator buffers after each encoder/decoder chunk returns. Its CPU hook test establishes unchanged arithmetic and hook behavior. It does not establish an MPS memory reduction, MPS/CPU numerical equality or actual-weight quality.

## Bounded reconstruction profile

The profiler accepts only the original Atrium `open/0000.png` image, SHA-256 `7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780`. It requires the original 512 by 288 RGB pixels and performs no resizing or cropping. It reads no old latent, actions, future image or training target.

For this workspace, run from `outputs/open-worldline` after the GPU run has been scheduled. The first-image output directory must not already exist:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../../work/wan-adapter-env/bin/python experiments/wan22_native/codec_profile.py \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --image ../../work/atrium-pilot/dense-pair/open/0000.png \
  --cpu-report experiments/wan22_native/codec-results/cpu-v2/tests.json \
  --device mps --allocator-cleanup \
  --output ../../work/wan22-native-codec-image-v1
```

Use `--device cpu` for explicit CPU execution. Add `--allocator-cleanup` only when choosing the already CPU-tested per-chunk allocator callback; the flag is retained in the report. Each process has a 900-second wall-clock limit after startup, an 18 GiB sampled RSS/Metal allocation ceiling and a 2 GiB minimum available-memory stop. Automatic MPS CPU fallback must be disabled. A watchdog stop retains its own terminal record and cannot qualify as a successful profile.

The run saves the separately encoded `observation.safetensors` tensor, original and reconstructed images, side-by-side comparison, file/source/weight hashes, PSNR and pixel MAE, cache checks, stage timings and sampled memory. The saved latent key is `observation`, shape `[1,48,1,18,32]`. These are first-image codec reconstruction measurements, not generated future frames or a world-model quality result. No weight download is performed by the helper or profiler.

After this process exits successfully, the separate [full 17-frame decoder cost profile](CODEC_DECODE.md) reads its saved observation. The [evidence index](codec-results/index.json) includes the original CPU/meta records, the autocast fix, current checked source and independent review. For reproduction outside this workspace, replace only the environment, weight, original-image and new-output paths with their local equivalents.
