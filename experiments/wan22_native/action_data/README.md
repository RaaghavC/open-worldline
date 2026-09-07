# Original Atrium RGB and command cache for Wan2.2

This package prepares eight development clips for the attributed native Wan2.2 codec. It contains no trained video model or new model weights. CPU preparation passed 12 implementation checks and five independent checks. The separately scheduled full 17-frame reconstruction also completed; its measured evidence is retained below. The eight-window cache also completed in a separate run, with its tensors and measured evidence retained below.

The completed reconstruction uses the original 17 RGB frames from `open-0000`. It encodes the first image alone, encodes the full clip, verifies that both produce exactly the same first latent, and decodes the full target latent. The decoder receives only that latent. Original RGB is used afterward to calculate MSE, MAE and PSNR for each frame, all 17 frames together, the first frame, and the later 16 frames. All output frames are reconstructions of supplied images, not generated futures.

The full cache has eight windows in this fixed order: `closed-0000`, `open-0000`, `closed-0008`, `open-0008`, `closed-0032`, `open-0032`, `closed-0049`, `open-0049`. Each contains 17 original RGB frames and 16 commands. Commands come from the destination frame's `action_from_previous` field. They retain requested meters, radians and the interaction pulse without normalization.

## Measured 17-frame reconstruction

The original room, doorway and window structure are preserved in the reconstructed clip. Thin plant and window details are blurred in some later frames. The severe prismatic distortion seen in the separate generated clip is absent from this known-RGB reconstruction. This result shows that the codec can reconstruct this supplied clip; it does not identify the cause of the failed generation or establish generation quality.

| Scored original RGB frames | MSE on RGB `[0, 1]` | MAE on RGB `[0, 1]` | PSNR |
| --- | --- | --- | --- |
| All 17 frames | 0.000593911 | 0.00856443 | 32.2628 dB |
| Initial frame | 0.0000239286 | 0.00300176 | 46.2108 dB |
| Later 16 frames | 0.000629535 | 0.00891210 | 32.0098 dB |

These scores use the retained FP32 reconstruction before uint8 rounding. A separate artifact audit recomputed MSE and MAE exactly, verified all 17 truth/reconstruction PNG pairs, and checked every recorded output and source hash. The full-target first latent equals the independently encoded observation exactly. That observation is also byte-identical to the earlier standalone first-image cache. No target prefix was replaced after encoding.

The guarded worker interval was 81.7488 seconds; the parent launch and worker interval was 83.3416 seconds. Measured stages were 5.3952 seconds to verify/load the codec, 1.2554 seconds to encode the initial RGB alone, 11.5342 seconds to encode all 17 original frames, and 61.3685 seconds to decode them. These are offline codec timings, not interactive display speed.

Peak sampled MPS driver allocation was 10,456,088,576 bytes. Minimum sampled available system memory was 2,217,132,032 bytes, above the retained 2 GiB floor. All 323 per-convolution cleanup calls completed across 62 invoked layers. There were six encoder chunks and five decoder chunks. Cleanup hooks were removed and native caches were clear when the worker completed.

- [Original/reconstruction contact sheet](results/roundtrip-v1/result/comparison.png)
- [All-frame reconstruction preview](results/roundtrip-v1/result/reconstruction-preview.gif), played at 10 frames per second
- [Raw measured results](results/roundtrip-v1/result/metrics.json) and [terminal status](results/roundtrip-v1/terminal.json)
- [Independent artifact audit](results/roundtrip-v1/artifact-audit.json)
- [Publication file identities](results/roundtrip-v1/publication.json)

The bundle contains all 85 original run files. Eighty-four remain byte-exact; `request.json` replaces four absolute local paths with repository-relative equivalents. Its original hash and the exact changed fields are disclosed in `publication.json`; the original request remains unchanged in the work run. Model tensors, images, scores, logs, source snapshots and failure records are not altered.

## Completed eight-window cache

The cache completed in a 130.3589-second guarded worker interval, with a 131.6998-second parent launch and worker interval. It contains eight training windows and seven independently encoded starting observations. All 11 recorded causal checks passed with exactly zero difference; no target prefix was replaced after encoding. No transformer or optimizer ran during cache creation.

A separate audit independently read 118 original PNGs, verified their hashes and RGB arrays, reconstructed the six-channel commands from the destination capture records, and checked every saved tensor shape, finite value, file hash and tensor hash. It verified the 18 one-level channel changes in only the derived closed start-0 image. The two start-0 observations are identical, and their command tensors differ only at the first interaction pulse. The other three branch pairs have different observations and matching commands.

The saved open start-0 target and observation match the earlier real reconstruction tensors exactly. The audit rechecked each saved target prefix directly; it checked the two recorded future-image perturbations and the recorded initial-image repeat against their retained results, without performing another encode.

There were 58 encoder chunks and no decoder chunks. All 1,490 cleanup calls completed across 27 invoked encoder convolution layers. Peak sampled MPS driver allocation was 6,420,021,248 bytes; minimum sampled available system memory was 4,200,841,216 bytes. Cleanup hooks were removed and native caches were clear at completion.

Use `results/cache-v1/result/` as the directory passed to `load_training_window` or `load_condition`. Both readers successfully loaded all eight packaged windows, and the current source, codec and prior reconstruction gates passed. The manifest SHA256 is `450d2c5d07881a0f42bb027bba666105921c50ad7c9259bf914883eaeeb3e39e`.

- [Cache manifest and tensor hashes](results/cache-v1/result/manifest.json)
- [Raw cache metrics](results/cache-v1/result/metrics.json) and [terminal status](results/cache-v1/terminal.json)
- [Independent original-data and tensor audit](results/cache-v1/artifact-audit.json)
- [Publication file identities](results/cache-v1/publication.json)

The cache bundle contains all 62 original run files. Sixty-one remain byte-exact; the same four local path fields in `request.json` are converted to portable repository-relative paths, with original and published hashes disclosed. All tensors, source snapshots, metrics and logs remain byte-exact. The raw work directory is unchanged.

This is a development cache from one layout, not a held-out generalization result. Passing its checks establishes the recorded data and conditioning boundaries; it does not establish trained action control or generated video quality.

## Exact initial-image correction

The raw closed and open images at frame 0 differ in 18 RGB channel values, each by one uint8 level. The cache builder copies the open frame into **only the derived closed clip's first RGB frame**, before encoding that entire clip. It records all 18 coordinates and old/new values. It does not change either original PNG, reuse old latents or replace a target prefix after encoding.

One independently encoded canonical initial image is shared by both start-0 branches. The other six starting observations are encoded separately from their actual first images. Those later branch images differ, while their outgoing command sequences match. These later pairs do not isolate an action effect. All clips come from one development layout, with camera yaw and a programmed remote door toggle.

## Cache interface

Each window file contains exactly these FP32 tensors:

| Key | Shape | Use |
| --- | --- | --- |
| `target` | `[1, 48, 5, 18, 32]` | Full original 17-frame codec encoding for training |
| `observation` | `[1, 48, 1, 18, 32]` | First RGB encoded alone |
| `commands` | `[1, 16, 6]` | Right/up/forward meters, left yaw/up pitch radians, interaction pulse |

The manifest schema is `worldline-wan22-atrium-action-cache-v1`. It records file/tensor/source hashes, original PNG hashes, derived RGB hashes, correction coordinates, exact codec identity and process environment. The worker must complete eight first-latent checks, two future-image perturbation checks, and an initial-image repeat after other encoding work. Failed checks retain evidence and stop; they do not alter the target latent.

`cache.load_training_window(result_directory, window_id)` returns all three tensors plus provenance. `cache.load_condition(...)` reads only `observation` and `commands` through `safe_open`. A CPU test supplies a poisoned target and rejects any attempt to materialize it on that conditioning path. Both readers reject incomplete worker results and stopped caches.

## CPU preparation

From the repository root, using the existing isolated environment:

```sh
../../work/wan-adapter-env/bin/python -m experiments.wan22_native.action_data.test_codec_path \
  --output ../../work/wan22-action-data-cpu-v2/tests.json

../../work/wan-adapter-env/bin/python -m experiments.wan22_native.action_data.roundtrip \
  --capture ../../work/atrium-pilot/dense-pair \
  --output ../../work/wan22-action-roundtrip-plan-v2

../../work/wan-adapter-env/bin/python -m experiments.wan22_native.action_data.build \
  --capture ../../work/atrium-pilot/dense-pair \
  --output ../../work/wan22-action-cache-plan-v2
```

Both data commands default to planning. They verify the original files without loading VAE weights or executing a GPU operation. Outputs must be new. The retained CPU report covers exact 17-frame encode and decode equality with both cleanup hooks, first-latent causality, exception cleanup, RGB correction, command units, scoring and the inference tensor boundary. It does not measure actual-weight reconstruction quality.

## Scheduled reconstruction

The command below launches one codec child process. Run it only when the model slot is free. No download occurs.

```sh
../../work/wan-adapter-env/bin/python -m experiments.wan22_native.action_data.roundtrip \
  --capture ../../work/atrium-pilot/dense-pair \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --cpu-report experiments/wan22_native/action_data/cpu-v2/tests.json \
  --decoder-run experiments/wan22_native/codec-results/full-decode-v3 \
  --device mps --execute \
  --output ../../work/wan22-action-roundtrip-v1
```

The parent and worker keep the 900-second limit; the worker stops above 18 GiB of process or MPS driver memory, or below 2 GiB of available system RAM. The process reproduces the completed decoder's environment: low allocator watermark `0.6`, automatic CPU fallback `0`, and unset high watermark, fast-math and prefer-Metal overrides. It applies the unchanged native FP32 codec, existing temporal-chunk cleanup and existing per-convolution synchronization/cache cleanup. The original 48 normalization values, patchification, native causal cache and decoder arithmetic remain unchanged.

Results are under the output's `result/` directory. `encoded.safetensors` retains target and independent observation. `rgb-input.safetensors` retains the exact 17 original uint8 RGB frames. `reconstruction.safetensors` retains FP32 reconstructed pixels before uint8 rounding. `truth/` and `reconstruction/` contain all 17 native-size PNGs; the side-by-side GIF and contact sheet have labels outside the image region. Preview playback is 10 frames per second, separate from measured encode/decode duration. Raw metrics, sampled memory, cleanup events, source snapshots, exceptions and terminal status are retained.

A completed roundtrip does not by itself authorize training or show that generated video is correct. Its purpose is to determine whether the full temporal codec reconstructs this original clip before interpreting failures from generated latents.

## Cache encoding after reconstruction review

```sh
../../work/wan-adapter-env/bin/python -m experiments.wan22_native.action_data.build \
  --capture ../../work/atrium-pilot/dense-pair \
  --weights ../../work/wan22-ti2v5b-weights/Wan2.2_VAE.pth \
  --cpu-report experiments/wan22_native/action_data/cpu-v2/tests.json \
  --decoder-run experiments/wan22_native/codec-results/full-decode-v3 \
  --roundtrip-run ../../work/wan22-action-roundtrip-v1 \
  --device mps --execute \
  --output ../../work/wan22-action-cache-v1
```

This command additionally requires completed, source-matching real roundtrip evidence. It has the same resource limits and does not execute a transformer or train parameters.

## Provenance and licenses

The copied initial PNGs in `source-images/` retain their exact original file bytes and the original Atrium dataset's CC0-1.0 dedication. The original capture manifest is pinned to SHA256 `942eaf38badb1c2de5489fac59b7727e5c2a3d5699ec44aa415b7862c8440ef0`. The frozen design is retained as `source-plan.md.txt`, SHA256 `52fead080ba299f89c7bc1aad1f58c1679c93c8ec735c177aa99c5b3916e6a62`.

This new data and execution code is Apache-2.0. The native codec is attributed upstream Wan2.2 code and official external weights; its existing package NOTICE, license and source record apply. No external weight file is bundled here. The old 16-channel Wan1.3B cache is not accepted.

CPU revision 1 is retained unchanged. Revision 2 adds exact manifest-to-worker hash binding, all 11 named causal checks and all seven independent observation records in the cache reader. The reconstruction operations and codec worker arithmetic are unchanged. Both revisions passed 12 CPU tests.

The separate independent CPU review passed five checks. It traced the complete mocked roundtrip worker, verified first-image storage isolation and original RGB bytes, retained a failed prefix check before decoder execution, and checked child termination after the total time limit. Its evidence is retained in `independent-cpu-v1/`. No real VAE weights or GPU execution were used for that review.
