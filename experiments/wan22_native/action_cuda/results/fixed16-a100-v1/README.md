# Completed fresh 16-update CUDA training pilot

**The original 947,712-parameter adapter completed all 16 prescribed paired updates on an A100.** It started from the saved fresh initialization and a new optimizer, without continuing the earlier probe. All 825 external foundation parameter values remained unchanged. The [independent saved-artifact audit](audit/report.json) passed. **This run trained an adapter; it generated no video and measured no rendered command response, graphics improvement or generalization.**

The foundation remains external pretrained Wan2.2 TI2V-5B, with original FP32 weights, native CUDA BF16 autocast, FlashAttention 2 and complex RoPE. The original adapter sits after the final transformer block and trains through its frozen native head. Each training window has 17 RGB frames, 16 recorded commands and a spatial latent `[1,48,5,44,78]`, representing 1,248 × 704 after preprocessing. Original Atrium images are 512 × 288; enlarging them adds no original detail.

## What completed

| Measurement | Result |
| --- | --- |
| Paired optimizer updates | 16, with 32 sequential B=1 training forwards |
| Window schedule | Starts 0, 8, 32, 49 repeated four times; closed then open |
| Fresh initialization seed | 20260907; actual saved initialization/noise bytes are retained |
| Pretraining native/zero-adapter comparison | Both start-0 comparisons bit-exact |
| GRU gradient L2 | First update 0; second `6.2454055e-5`; final `2.4329461e-4` |
| Foundation values | All 825 before/after hashes match |
| Sum of 16 paired-update intervals | 14.442762 seconds; median 0.900044 seconds |
| Foundation loading | 122.522927 seconds |
| Inner training routine | 114.347286 seconds, including value checks and artifact writing |
| Complete worker / parent | 250.294527 / 260.465302 seconds |
| Sampled peak CUDA reserved memory | 21,583,888,384 bytes, about 20.10 GiB |

Intervals overlap and must not be added together. Update timing covers two branch forwards/backwards and one optimizer step; it is not video-generation time. Different windows and noise/timesteps produce different loss values, so the 16 losses are not a comparable learning curve. Every branch loss, input identity, gradient norm and timing remains in the exact [training record](metrics/training.json), with [worker](metrics/worker.json), [parent](metrics/parent.json) and [terminal](metrics/terminal.json) records alongside it. [summary.json](summary.json) provides a concise hash-bound view.

Each pair shares its saved noise and integer timestep. Training uses half each branch's future-only latent flow MSE, excludes the initial latent, and preserves the independently encoded initial observation. The AdamW settings remain learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and gradient clipping at 1. The final checkpoint was prescribed in advance; no validation-based checkpoint selection or continuation is claimed.

## Independent evidence and scope

The [actual audit](audit/report.json) checked all 17 checkpoints, 32 retained velocity predictions, 16 gradient bundles containing 288 arrays, all original foundation identities, inputs and sampled resource records. It independently recomputed future-only losses and checked saved AdamW states. CPU float64 equation residuals are reported descriptively; the audit does not replay GPU backward operations or select new numerical tolerances. The [audit source](audit/audit.py) and [input inventory](audit/inventory.json) are retained unchanged.

There are eight overlapping windows from one original layout. Only start 0 has a shared observation with different outgoing commands. Its derived first image uses the established 18-channel-value correction before encoding; raw images remain unchanged. Starts 8, 32 and 49 have different starting images and identical commands. These data cannot establish unseen-scene behavior, long-term visual memory, learned physical interaction, novelty or Genie 3 parity. An action-controlled rendered comparison must be measured separately.

## Final checkpoint and exact source

The compact payload includes only [checkpoint 16](checkpoint-0016/adapter.safetensors), SHA256 `d15994f0cb1bdf54e264e75ecddd9fd126f3e32df04a6e913703d18b95e0db3d`, and its original [manifest](checkpoint-0016/manifest.json). The manifest also names the optimizer recovery file in the full archive. The complete archive retains checkpoints 0 through 16, all optimizer/RNG states, raw gradients/predictions, saved noises, cache inputs and earlier probe evidence. An experimental checkpoint is not a validated action controller.

The four exact executed additions are [prototype.py](source/local/prototype.py), [runner.py](source/local/runner.py), [test_prototype.py](source/local/test_prototype.py) and [test_runner.py](source/local/test_runner.py). Their [CPU report](preflight/runner-cpu-report-v2.json) passed nine checks before execution. An [earlier fixture failure](preflight/earlier-cpu-fixture-failure.json) occurred because four mocked tests used a symlinked Mac temporary root; resolving that fixture path preserved the production no-symlink rule. No numerical tolerance changed.

The foundation and existing local dependencies come from repository commit `099eeae6268da852b0acb580258e4c168f900520`. The [publication map](code-provenance.json) binds all four additions and 61 repository source identities. Sixty unchanged dependency files are shared with the earlier [two-update payload](../a100-v1/README.md), while the remaining source is included here. The full archive contains all executed sources. Source snapshots preserve executed bytes, including historical plan fields; the actual terminal and audit records establish completion.

## Separate execution limit

The exact [admission](plans/admission.json) permitted fresh fixed16 training only, with image generation and resume both false. A stricter external process-group limit was configured to send SIGINT after 720 seconds, with a 10-second kill grace, preserving at least 300 seconds for recovery. The unchanged internal worker limit remained 900 seconds. The run completed in 260.465 seconds without a timeout; the deadline was not extended.

The [public remaining-time record](execution/remaining-time-public.json) preserves every timing, reserve and limit value from the parent's check. Only the provider resource identifier was removed. Its original SHA256, published SHA256 and exact removed field are recorded in [code-provenance.json](code-provenance.json). This is separate operational provenance, not a modification to the immutable recovered run. All scientific measurements, checkpoints and copied source bytes are unchanged. A worker timeout alone does not cap provider billing.

## Recovery and licensing

The verified full recovery contains **485 files totaling 616,066,326 bytes**, compressed into **six parts totaling 464,908,340 bytes**. The [index](recovery/index.json) SHA256 is `09ac05c19f18d881ddde0d2242fdb828e1262ae3361e178e220ea1a8b799ef2a`; the compressed stream SHA256 is `25994887e79c2e80d3911b92469e306faa1d9d20ceb6c321f2a854c54b16ed10`. [Local recovery verification](recovery/recovery-verified.json) passed. The [public release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-fixed16-a100-v1) contains all six parts. Its [published record](release/published-release.json) and [remote asset check](release/remote-asset-verification.json) bind all 17 release assets. The [local downloader check](recovery/local-downloader-verification.json) independently recovered every original file; a [fresh public HTTPS download](recovery/public-download-verification.json) also recovered all 485 files with the same hashes. The release target commit records publication context, while `099eeae6268da852b0acb580258e4c168f900520` and the four exact additions identify executed code. See [DOWNLOADER.md](DOWNLOADER.md).

Original adapter, training code and checkpoint use [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt). External Wan source/model/VAE keep their [license](licenses/WAN-APACHE-2.0.txt) and [NOTICE](licenses/WAN-NOTICE.txt). Original Atrium imagery and metadata use [CC0](licenses/ATRIUM-CC0.txt). No external foundation checkpoint, credential or account-lifecycle log is included in this small payload. [LICENSE-NOTES.md](LICENSE-NOTES.md) records the component boundaries.
