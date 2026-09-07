# Proposed spatial-size test after the reduced CUDA clip failed

Read-only recommendation, September 7, 2026. The original note was drafted while the reduced CUDA clip was running. Its completed frames subsequently showed severe prismatic distortions, and this publication copy updates that timing context. The proposed larger-shape test remains unimplemented and unexecuted. The original note is retained alongside this copy. Preparing and updating the note made no cloud call, credential access, model run, download or runner change.

**The next direct test should change spatial size alone: keep 17 frames, but use the official image preprocessing at its approximately 720p area, yielding 1248 × 704 for the same Atrium image.** This would test whether the very small spatial grid contributes to the failure. It would not settle the separate 17-versus-121-frame question. The current frozen CUDA runner cannot run this shape, and the available timing evidence does not admit it under the existing 900-second limit. The completed paid Pod has been deleted. A future test would require separate preparation, measured admission and a fresh resource plan.

## What the official sources establish

The pinned [TI2V-5B configuration](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/configs/wan_ti2v_5B.py) sets **121 frames, 24 fps, 50 steps, shift 5 and guidance 5**. The [official README](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/README.md#run-text-image-to-video-generation) documents the model's 720p size option as `1280*704` or `704*1280`. These are intended configuration and demonstration settings, not a published guarantee for every input.

The [image function](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/textimage2video.py#L411) permits frame counts of `4n+1`, so 17 is structurally valid. Spatial dimensions must match its VAE and patch grid. There is no reviewed assertion that 17 frames at 512 × 288 are a validated quality setting. One source nuance matters: `generate()` itself has an 81-frame function default, while `i2v()` and the TI2V model configuration have 121; the configured recipe is therefore the appropriate 121-frame reference. Likewise, a low-level `i2v()` default of 40 steps does not override the explicitly chosen 50-step configuration used here.

The official image preprocessing calls `best_output_size`, then Lanczos resize and center crop. Re-evaluating only the retained scalar helper gives:

```
best_output_size(512, 288, 32, 32, 1280 * 704) = (1248, 704)
```

For this source image, the resize is 1252 × 704 followed by a centered 1248 × 704 crop. The latent is `[48, 5, 44, 78]` for 17 frames, with **4,290 patch tokens** and **858 observed-prefix tokens**. Do not resize the existing cached latent; the resized RGB must be independently encoded with the native 48-channel VAE. [Preprocessing](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/textimage2video.py#L457), [sizing helper](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/utils/utils.py#L202).

| Shape | Latent frames | Patch tokens | Ratio to the current token count |
| --- | ---: | ---: | ---: |
| Current: 17 × 512 × 288 | 5 | 720 | 1 |
| Proposed spatial test: 17 × 1248 × 704 | 5 | 4,290 | 5.9583 |
| Duration-only alternative: 121 × 512 × 288 | 31 | 4,464 | 6.2 |
| Both configured dimensions: 121 × 1248 × 704 | 31 | 26,598 | 36.9417 |

Both spatial and temporal differences are substantial. The current spatial area is about 16.8% of the source-specific intended area; its 5 latent frames are about 16.1% of the configured 31. The conditioned first latent occupies 20% of the current temporal grid versus about 3.2% at 121 frames. These changes could affect learned behavior, but no source or completed comparison here proves that they cause the prismatic surfaces. Changing both dimensions together could test the full shape setting but would not identify which dimension mattered.

## Fixed comparison and remaining differences

Use the exact original input PNG, SHA256 `7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780`, from the original open-0000 capture. Preserve raw bytes and record the derived resized/cropped RGB and newly encoded observation. Keep this positive prompt verbatim:

> A sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details, and green plants.

Keep the existing `atrium` and `native_negative` contexts, source/weight revisions, original FP32 parameter storage, CUDA BF16 autocast, FlashAttention 2, CPU UniPC50, shift 5 and CFG 5. Keep seed **20260908** and explicitly save the new shape-specific noise. A changed tensor shape requires different noise data; equal seeds do not make the two latent tensors identical or establish a pixel-paired comparison. If the new observation encoder runs on CUDA instead of the cache's original MPS path, first compare its original-size encoding with that cache; a material encoding difference must not be folded silently into a resolution explanation. This is a single-seed diagnostic, not a general quality estimate.

The already completed [source-contract audit](../../../../core-results/contract-audit-v1/audit.md) settled the reviewed mask, negative-branch image conditioning, schedule and normalization equations. Both text branches get the same clean image prefix and zero token time there; the prefix is restored after every solver step. Those equations should stay unchanged. The completed real-video VAE reconstruction and two tiny-decoder comparisons also make another unchanged decoder swap a poor next test.

The completed CUDA clip also showed severe prismatic room distortions. This demonstrates that Mac-specific execution is not necessary for this failure category in this reduced-shape test. It does not prove pixel identity, equivalence of every prediction, the cause of the distortion or reproduction of the official end-to-end example: the CUDA control still shares the reduced shape, cached initial observation, FP32-produced UMT5 contexts and CPU solver. The cache was produced at FP32 while native T5 configuration requests BF16. That remaining numerical difference is declared, not newly identified as the cause. The proposed spatial test keeps that text cache fixed rather than changing several factors together.

If the spatial test preserves room/door structure and removes the severe colored distortions, that supports spatial-size sensitivity for this image, prompt and seed. If it also fails, duration, shared conditioning and broader input/checkpoint behavior remain unresolved. Neither outcome establishes general action learning, photorealism or Genie 3 parity.

## Why it was not admitted during the paid session

The recovered actual CUDA pair reports **232.9607 seconds loading** and **2.22367 seconds for its two predictions**, with parent completion in 243.1861 seconds. Holding load constant and scaling only prediction time by token count gives:

| Proposed shape | Linear-only pair estimate | Existing clip admission formula |
| --- | ---: | ---: |
| 17 × 1248 × 704 | 13.2494 s | 1,224.5 s |
| 121 × 512 × 288 | 13.7868 s | 1,256.8 s |
| 121 × 1248 × 704 | 82.1461 s | 5,358.3 s |

Formula: `1.2 * (load + 50 * pair) + 120 + 30`. These are planning calculations, not runtime predictions or mathematical lower bounds. Attention work scales differently, FlashAttention avoids storing a full score matrix, device utilization can change, and loading may benefit from a cache. The larger decoder's cost is unmeasured. Even this simple projection exceeds the current 900-second gate for either single-dimension test.

The frozen files explicitly reject new shapes: `cuda_reference/sampling.py` fixes `[48,5,18,32]`, 720 tokens and 144 prefix tokens; `official_cpu/inputs.py` fixes the saved-input shapes/hashes; `cuda_reference/decode.py` fixes the 17 × 288 × 512 output. Its CLI has no size or frame-count option. The minimal transfer also lacks the original PNG and an arbitrary-size observation encoder. Running the upstream `generate.py` directly is not an immediate substitute: it changes the preparation path and requires sources/dependencies plus an uncached 11 GB text encoder not included in this transfer.

The smallest runtime preparation for this proposed experiment would be a separately tested shape-capable input path and one actual 4,290-token positive/negative pair, plus a measured native-size observation encode/decode. Only those measurements could support a new complete-clip admission. No such preparation was implemented during this session, so **no additional shape experiment was admitted**. The reduced clip was recovered and reviewed, and exact-ID Pod deletion was confirmed. A future shape test needs its own implementation, measurements and resource decision. These linear estimates do not authorize extending a paid window or weakening the guards.

## Retained evidence read

- `outputs/open-worldline/experiments/wan22_native/core-results/contract-audit-v1/official-source/wan/textimage2video.py`, SHA256 `228f2fabf23014ed41ec6b8cd713d5be7371bb558501c71a62a0258b491a877b`.
- The adjacent `wan/configs/wan_ti2v_5B.py`, SHA256 `26d9e7b9c555eb0900b13751267a596556f869e7aee2d1572afc2a0a4a76f4c7`.
- `work/i2v-next-path-audit/wan22/README.md`, the pinned official README; `work/wan22-native-contract-audit/wan_utils_utils.py`, containing the scalar helper.
- `work/wan22-next-reference-audit/RECOMMENDATION.md`, prior intended-size and numerical-path assessment. Its old Mac timing estimates were not reused for CUDA.
- `work/runpod-launch-v1/recovered-pair-v1/results/pair-run-v1/core/result/metrics.json`, the actual completed CUDA timing source; the shared prompt/input identities are in its parent `metrics.json`.

Preparation of the original recommendation evaluated only the scalar size helper and integer/token timing arithmetic. Neither that analysis nor this publication edit loaded inference code or model parameter values. The separately recorded completed CUDA clip is evidence from its own execution, not from this planning note.
