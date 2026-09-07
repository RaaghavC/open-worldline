# Local neural baseline study

Measured on September 7, 2026, on an Apple M4 Pro with 24 GB shared memory. These experiments ran released neural models locally. They used no inference service. They help identify models that fit this computer and reveal the cost of generating actual image pixels.

DIAMOND generated action-conditioned game frames. WorldFM generated images conditioned on a reference and a rendered guide. Neither run is a matched comparison with Worldline or Genie 3. Their tasks, resolutions, histories, sampling methods and measurement scopes differ.

## DIAMOND CS:GO

The [authors' source](https://github.com/eloialonso/diamond/tree/851cefb497733d27f1b85c804104638765860fca) was pinned to commit `851cefb497733d27f1b85c804104638765860fca`. The public [author checkpoint](https://huggingface.co/eloialonso/diamond/tree/5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f) was downloaded separately. Its exact revision and all asset hashes are recorded in the [asset manifest](../experiments/baselines/diamond-assets-manifest.json). Checkpoint SHA-256: `9a56a599cec69863717001660871418af1ac3598762167a5cbda73076951bcb6`.

The original dynamics model has 330,489,219 parameters; its neural upsampler has 51,149,187. Inference used PyTorch 2.5.1, MPS and float32. Each arm began with the same four real frames from released spawn 0, then generated 12 frames. Subsequent history contained generated frames. The output resolution was 280 by 150 pixels. Each action arm reset the CPU and MPS random seeds to 20260907.

| Authors' setting | Action | Median per-frame inference | First frame | Number of generated frames |
| --- | --- | ---: | ---: | ---: |
| Fast: 1 dynamics step, 1 upsampler step | Idle | 289 ms | 6,984 ms | 12 |
| Fast | Forward | 277 ms | 143 ms | 12 |
| Fast | Turn left | 293 ms | 144 ms | 12 |
| Higher quality: 3 dynamics steps, 10 upsampler steps | Idle | 1,531 ms | 2,977 ms | 12 |
| Higher quality | Forward | 2,251 ms | 2,560 ms | 12 |

These are synchronized model-plus-upsampler timings. They exclude history updates, image copies, file writes and display. The first idle call includes cold execution; later arms reuse loaded models. The computer was in active use, so these small samples are not an isolated hardware benchmark or a stable frame-rate guarantee. The setting name "higher quality" comes from the authors' configuration, not a measured preference result here.

Changing the control changed the output. Under the fast setting, paired pixel MAE against idle was 21.51 for forward and 40.30 for left, on the 0 to 255 scale. This measures sensitivity only. The supplied spawn bundle contains future actions but no corresponding future true images, so it cannot establish correct movement, prediction accuracy, FVD or LPIPS. Identical-arm repeated-run determinism was not separately tested.

At the end of the fast run, allocated MPS tensors occupied 1,530,700,032 bytes and MPS driver allocations occupied 2,297,282,560 bytes. These are final allocations, not peak memory; the categories overlap. Detailed samples and frame hashes: [fast results](../experiments/baselines/results/diamond-fast.json), [higher-quality results](../experiments/baselines/results/diamond-quality.json).

The source is MIT licensed. A separate checkpoint redistribution grant was not verified, and dataset terms differ across the repository and model-host metadata. This repository includes the measurement code and metrics, with a separate user-initiated download. It does not include the external checkpoint or game images. The [provenance audit](diamond-provenance.md) records that distinction.

## WorldFM core on Apple Silicon

The [official source](https://github.com/inspatio/worldfm/tree/d51ada211079d185285076b56fb803069a571838) was pinned to `d51ada211079d185285076b56fb803069a571838`. The [official weights](https://huggingface.co/inspatio/worldfm/tree/48d206b813a6ff3ddc1977b6cce8b6769c3bbad6) were pinned independently. The main model has 612,192,416 parameters and the VAE has 83,653,863. Total downloaded model files were 2,793,220,939 bytes, plus a 631-byte configuration.

The released core accepts a reference RGB image and a target-view RGB guide. Its default preparation pipeline has additional models and restrictions; those tools were bypassed for this direct core test. The test supplied Worldline's original alien-scene render as both reference and guide, center cropped to 512 by 512 pixels. It tested execution on one out-of-distribution rendered input. It did not test new camera views or learned physical transitions.

A [compatibility patch](../experiments/baselines/worldfm-core-mps.patch) made xformers optional, moved unsupported bicubic resizing to CPU, transferred position embeddings with the correct device and dtype, and added synchronized timing and checkpoint diagnostics. It preserved the original sampling equations. Only the intentionally recreated position embedding was absent when loading; no unexpected keys were reported.

| Measurement | Actual result |
| --- | ---: |
| Device and precision | MPS, float16 |
| Output | 512 by 512, two sampling steps |
| Model load | 9.29 seconds |
| First generated frame | 8.24 seconds |
| Next two generated frames | 4.94 and 4.88 seconds |
| Mean of those two later frames | 4.91 seconds, or 0.204 generated frames/second |
| Sampled peak process RSS | 4,235,460,608 bytes |
| Sampled peak MPS tensor allocation | 3,275,160,064 bytes |
| Sampled peak MPS driver allocation | 5,870,108,672 bytes |

Memory categories overlap and must not be summed. All three frames contained finite values. Visual inspection showed that the model retained the rendered input's stylized appearance. It did not demonstrate a photorealistic improvement. Three same-view samples are insufficient to assess scene consistency or general quality. Raw timing and hash evidence: [WorldFM results](../experiments/baselines/results/worldfm-mps.json).

The source license and weight-card metadata declare Apache-2.0. The inspected card does not explain the VAE's earlier training provenance. External weights are downloaded separately and are not part of Worldline's original models.

## Local appearance editing and a failed transfer test

A separate appearance experiment used [Runpod's 4-bit conversion](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit/tree/7ee1b3aa8178a1240050490072196a57da2bf2a9) of Black Forest Labs' FLUX.2 klein 4B through unmodified MFLUX 0.19.1. One 768 by 432 reference-conditioned edit took 51.95 seconds after 3.18 seconds of loading. Peak MLX array allocation was 6.45 GiB; maximum sampled active-plus-cache allocation was 7.12 GiB. These overlapping memory values must not be summed.

The image gained visible rock fractures, water reflections and translucent crystal facets. It also changed individual crystal shapes and sizes. The major mountains and water channel remained recognizable by inspection; exact geometry preservation was not measured. This was one external-model image edit, with no new training or inference service.

In a subsequent single WorldFM test, the edited image supplied the reference and the original render supplied the guide. **The improved materials did not transfer.** The output retained the original smooth purple terrain and opaque triangular crystals. That first and only inference took 5.88 seconds after 8.41 seconds of loading. The test does not establish a working appearance pipeline for new camera views, physical transitions or interactive use.

The [appearance experiment](../experiments/baselines/flux4b/README.md) includes input/output images, exact model attribution, pinned downloads, timings, memory evidence and reproduction commands. External weights remain separate from Worldline's original models. The quantized conversion names the upstream model but does not pin the original revision that it converted; this provenance limit is recorded.

## Development decisions

DIAMOND provides a runnable example of actual action-conditioned neural video on this computer, at low resolution. WorldFM's released frame model fits local memory, but this test provides no reason to describe it as an appearance upgrade or a real-time replacement for the editor's renderer.

Worldline's separate [original RGB experiment](room-rgb-experiment.md) compares two newly trained architectures on the same original room data. Its errors cannot be ranked against the above systems because the data and tasks differ. None of these results establishes frontier visual quality, general physical accuracy or a new scientific contribution.

Reproduction commands, pinned downloads and third-party source licenses are in [experiments/baselines](../experiments/baselines/README.md).
